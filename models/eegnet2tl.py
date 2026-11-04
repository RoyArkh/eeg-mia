"""EEGNet2TL, Cobilean et al.'s own model (EEGNet2TL in github.com/CobV/MIA_BCI).

EEGNet with a second temporal convolution (half the kernel length) before the spatial
one. Settings as in Cobilean et al.: 16 filters, D=2, kernel 32, separable kernel 128,
second pooling 32.
"""

import torch
import torch.nn as nn


def build(n_channels, n_times, sampling_rate, dropout):
    return EEGNet2TL(n_channels, n_times, dropout)


class EEGNet2TL(nn.Module):
    def __init__(self, n_channels, n_times, dropout, filters=16, D=2, kernel_length=32,
                 separable_kernel=128, pool2=32):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, filters, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(filters),
            nn.Dropout(dropout),
        )
        self.temporal2 = nn.Sequential(
            nn.Conv2d(filters, filters, (1, kernel_length // 2), padding="same", bias=False),
            nn.BatchNorm2d(filters),
            nn.Dropout(dropout),
        )
        self.spatial = nn.Sequential(       # depthwise: D spatial filters per temporal filter
            nn.Conv2d(filters, filters * D, (n_channels, 1), groups=filters, bias=False),
            nn.BatchNorm2d(filters * D),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.pool1 = nn.AvgPool2d((1, 5), stride=(1, 5))
        self.separable = nn.Sequential(
            nn.Conv2d(filters * D, filters * D, (1, separable_kernel), padding="same",
                      groups=filters * D, bias=False),
            nn.Conv2d(filters * D, filters * D, (1, 1), bias=False),
            nn.BatchNorm2d(filters * D),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.pool2 = nn.AvgPool2d((1, pool2), stride=(1, pool2))
        with torch.no_grad():
            n_features = self.features(torch.zeros(1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(n_features, 1)

    def features(self, x):                 # x: [batch, channels, time]
        x = x.unsqueeze(1)                 # -> [batch, 1, channels, time]
        x = self.temporal2(self.temporal(x))
        x = self.pool1(self.spatial(x))
        x = self.pool2(self.separable(x))
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self.features(x))
