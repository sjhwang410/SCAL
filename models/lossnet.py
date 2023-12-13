'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

import torch
import torch.nn as nn 


class LossNet(nn.Module):
    def __init__(self, num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features):
        out1 = self.GAP(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = self.relu(self.FC1(out1))

        out2 = self.GAP(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = self.relu(self.FC2(out2))

        out3 = self.GAP(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = self.relu(self.FC3(out3))

        out4 = self.GAP(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = self.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))

        return out
