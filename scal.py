"""
SCAL Procedure in PyTorch.

Reference:
[Hwang et al. 2022] Uncertainty-based Selective Clustering for Active Learning (https://ieeexplore.ieee.org/abstract/document/9925155)
"""


import os
import pickle
import random
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.cluster import AgglomerativeClustering

from tqdm import tqdm

from config import *
from train_utils import frozen, free
from models.resnet import ResNet18
from models.lossnet import LossNet
from data.sampler import SubsetSequentialSampler
from data.data_transform import get_data


def loss_pred_loss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2]
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def ft_epoch(models, criterion, optimizers, dataloaders):
    models['backbone'].eval()
    models['module'].train()

    free(models['module'])
    frozen(models['backbone'])

    for data in tqdm(dataloaders['ft'], leave=False, total=len(dataloaders['ft'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['ft'].zero_grad()

        scores, features, _ = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        features[0] = features[0].detach()
        features[1] = features[1].detach()
        features[2] = features[2].detach()
        features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        loss = loss_pred_loss(pred_loss, target_loss, margin=MARGIN)

        loss.backward()
        optimizers['ft'].step()


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    free(models['module'])
    free(models['backbone'])
    
    models['module'].train()
    models['backbone'].train()

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features, _ = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        loss = torch.sum(target_loss) / target_loss.size(0)

        if epoch > epoch_loss:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_module_loss = loss_pred_loss(pred_loss, target_loss, margin=MARGIN)
        loss += WEIGHT * m_module_loss

        loss.backward()
        
        optimizers['module'].step()
        optimizers['backbone'].step()


def test(models, dataloaders, mode='val'):
    models['module'].eval()
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')

    for epoch in range(num_epochs):
        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['module'].step()
        schedulers['backbone'].step()

    for epoch in range(30):
        ft_epoch(models, criterion, optimizers, dataloaders)

    print('>> Finished.')
    
    
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            scores, features, _ = models['backbone'](inputs)
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()


def get_real_loss(models, data_loader, criterion):
    models['backbone'].eval()
    models['module'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, features, _ = models['backbone'](inputs)
            target_loss = criterion(scores, labels)

            uncertainty = torch.cat((uncertainty, target_loss), 0)

    return uncertainty.cpu()


def clustering(model, cluster_size, data_loader):
    model.eval()

    features = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.cuda()

            _, _, feature = model(inputs)

            features = torch.cat((features, feature), 0)
    features = features.cpu().numpy()

    return AgglomerativeClustering(n_clusters=cluster_size, linkage='complete').fit_predict(features)


def sampling(cluster_dict):
    sampled = []

    for key in cluster_dict:
        sampled.append(cluster_dict[key][-1])

    return sampled


if __name__ == '__main__':
    for trial in range(1, TRIALS+1):
        random.seed(76240 + trial)
        np.random.seed(76240 + trial)
        torch.manual_seed(76240 + trial)
        torch.cuda.manual_seed(76240 + trial)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        data_idx = [i for i in range(50000)]
        random.shuffle(data_idx)
        labeled_set, unlabeled_set = data_idx[:INIT_SIZE], data_idx[INIT_SIZE:]

        resnet18 = ResNet18(num_classes=CLS_CNT, channel_size=CHANNEL_SIZE).cuda()
        loss_module = LossNet().cuda()
        models = {'backbone': resnet18, 'module': loss_module}
        
        random.shuffle(labeled_set)
        removal_size = len(labeled_set) // 100
        removal_size = removal_size - 1 if removal_size % 2 else removal_size
        train_transform_data, test_transform_data, evaluate_transform_data = get_data('./data', DATASET)

        
        train_loader = DataLoader(train_transform_data, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set[:-removal_size]), pin_memory=True)
        test_loader = DataLoader(test_transform_data, batch_size=BATCH)
        ft_loader = DataLoader(train_transform_data, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set),
                               pin_memory=True)
        dataloaders = {'train': train_loader, 'test': test_loader, 'ft': ft_loader}

        fp = open(f'record_{trial}.txt', 'w')
        for cycle in range(CYCLES):
            print(f'cycle {cycle + 1} start -  labeled data size: {len(labeled_set)} / unlabeled data size: {len(unlabeled_set)}')
            criterion = nn.CrossEntropyLoss(reduction='none').cuda()

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            optim_ft = optim.SGD(models['module'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            optimizers = {'backbone': optim_backbone, 'ft': optim_ft, 'module': optim_module}

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            print(f'labeled: {len(labeled_set)} / unlabeled: {len(unlabeled_set)}')
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL)
            acc = test(models, dataloaders, mode='test')

            fp.write(f'{acc}\n')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial, TRIALS, cycle + 1,
                                                                                        CYCLES, len(set(labeled_set)), acc))

            if cycle < CYCLES - 1:
                unlabeled_loader = DataLoader(evaluate_transform_data, batch_size=BATCH,
                                  sampler=SubsetSequentialSampler(unlabeled_set),
                                  pin_memory=True)

                uncertainty = get_uncertainty(models, unlabeled_loader)

                arg = np.argsort(uncertainty)
                subset = list(torch.tensor(unlabeled_set)[arg][-SUBSET:].numpy())
                subset_label = clustering(models['backbone'], ADDENDUM,
                                          DataLoader(evaluate_transform_data, batch_size=BATCH,
                                                     sampler=SubsetSequentialSampler(subset),
                                                     pin_memory=True))

                subset_cluster = {}
                for i, idx in enumerate(subset):
                    if subset_label[i] not in subset_cluster:
                        subset_cluster[subset_label[i]] = [idx]
                    else:
                        subset_cluster[subset_label[i]].append(idx)

                sampled_data = sampling(subset_cluster)
                sampled_loader = DataLoader(evaluate_transform_data, batch_size=BATCH,
                                            sampler=SubsetSequentialSampler(sampled_data),
                                            pin_memory=True)
                sampled_real_loss = get_real_loss(models, sampled_loader, criterion)
                sampled_arg = np.argsort(sampled_real_loss)
                sampled_data = list(torch.tensor(sampled_data)[sampled_arg].numpy())[::-1]

                labeled_loader = DataLoader(evaluate_transform_data, batch_size=BATCH,
                                            sampler=SubsetSequentialSampler(labeled_set),
                                            pin_memory=True)
                labeled_real_loss = get_real_loss(models, labeled_loader, criterion)
                labeled_arg = np.argsort(labeled_real_loss)
                labeled_set = list(torch.tensor(labeled_set)[labeled_arg].numpy())

                labeled_set += sampled_data
                unlabeled_set = list(set(unlabeled_set) - set(labeled_set))

            if cycle == CYCLES - 2:
                dataloaders['train'] = DataLoader(train_transform_data, batch_size=BATCH,
                                                    sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
            else:
                _size = removal_size // 2
                dataloaders['train'] = DataLoader(train_transform_data, batch_size=BATCH,
                                                    sampler=SubsetRandomSampler(labeled_set[_size:-_size]),
                                                    pin_memory=True)

            dataloaders['ft'] = DataLoader(train_transform_data, batch_size=BATCH,
                                           sampler=SubsetRandomSampler(labeled_set), pin_memory=True)

        fp.close()
