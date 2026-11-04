"""Conv1DModel, Cobilean et al.'s own model (Conv1DModel in github.com/CobV/MIA_BCI).

Three 1-D conv blocks (kernels 64, 32, 16), global average pooling, linear output.
The first block is a grouped convolution with one group per EEG channel, so the number
of channels must divide 32.
"""

import torch.nn as nn


def build(n_channels, n_times, sampling_rate, dropout):
    return Conv1DModel(n_channels, dropout)


class Conv1DModel(nn.Module):
    def __init__(self, n_channels, dropout):
        super().__init__()
        self.blocks = nn.Sequential(
            conv_block(n_channels, 32, kernel_size=64, groups=n_channels, dropout=dropout),
            conv_block(32, 32, kernel_size=32, groups=32, dropout=dropout),
            conv_block(32, 14, kernel_size=16, groups=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(14, 1)

    def forward(self, x):                  # x: [batch, channels, time]
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def conv_block(in_channels, out_channels, kernel_size, groups, dropout):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(dropout),
    )
