import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST
from data.custom_dataset import MyDataset

NORM = {'cifar10': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
        'cifar10_imbalanced': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
        'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
        'caltech': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        'fashion': [[0.5], [0.5]],
        }


def get_data(path, data, *args):
    if 'caltech' in data:
        train_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=224, padding=4),
            T.ToTensor(),
            T.Normalize(NORM['caltech'][0], NORM['caltech'][1]),
        ])

        test_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(NORM['caltech'][0], NORM['caltech'][1]),
        ])
    else:
        _size = 28 if data == 'fashion' else 32
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=_size, padding=_size // 8),
            T.ToTensor(),
            T.Normalize(NORM[data][0], NORM[data][1]),
        ])

        test_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(NORM[data][0], NORM[data][1]),
        ])

    if data == 'cifar10' or data == 'cifar10_imbalanced':
        train_transform_data = CIFAR10(path, train=True, download=True, transform=train_transform)
        evaluate_transform_data = CIFAR10(path, train=True, download=False, transform=test_transform)
        test_transform_data = CIFAR10(path, train=False, download=True, transform=test_transform)
    elif data == 'cifar100':
        train_transform_data = CIFAR100(path, train=True, download=True, transform=train_transform)
        evaluate_transform_data = CIFAR100(path, train=True, download=False, transform=test_transform)
        test_transform_data = CIFAR100(path, train=False, download=True, transform=test_transform)
    elif data == 'fashion':
        train_transform_data = FashionMNIST(path, train=True, download=True, transform=train_transform)
        evaluate_transform_data = FashionMNIST(path, train=True, download=False, transform=test_transform)
        test_transform_data = FashionMNIST(path, train=False, download=True, transform=test_transform)
    elif 'caltech' in data:
        train_inputs, train_labels, val_inputs, val_labels = args
        train_transform_data = MyDataset(train_inputs, train_labels, transform=train_transform)
        evaluate_transform_data = MyDataset(train_inputs, train_labels, transform=test_transform)
        test_transform_data = MyDataset(val_inputs, val_labels, transform=test_transform)
    else:
        raise FileNotFoundError

    return train_transform_data, test_transform_data, evaluate_transform_data
