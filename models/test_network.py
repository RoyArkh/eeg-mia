"""TestNetwork: the size-controllable CNN of Cobilean et al. Experiment 4 (HEALTHINF §3.3).

A stack of `n_layers` temporal convolutions with `filters` filters each, then one
depthwise spatial convolution, average pooling and a linear layer. The paper varies
n_layers 1-5 and filters 20/40/60/80, which gives about 1.9K to 830K parameters.

Same network as mia_bci/exp4_model.py with filter_mode="per_layer" (the mode used for
the paper: every layer gets `filters` filters).
"""

import torch
import torch.nn as nn


def build(n_channels, n_times, sampling_rate, dropout, filters=40, n_layers=2):
    return TestNetwork(n_channels, n_times, dropout, filters, n_layers)


class TestNetwork(nn.Module):
    __test__ = False   # not a pytest test class, despite the name

    def __init__(self, n_channels, n_times, dropout, filters=40, n_layers=2, kernel_length=32):
        super().__init__()
        layers = []
        in_filters = 1
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(in_filters, filters, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
                nn.BatchNorm2d(filters),
                nn.ELU(),
                nn.Dropout(dropout),
            ]
            in_filters = filters
        self.temporal = nn.Sequential(*layers)
        self.spatial = nn.Sequential(
            nn.Conv2d(filters, filters, (n_channels, 1), groups=filters, bias=False),
            nn.BatchNorm2d(filters),
            nn.ELU(),
            nn.AvgPool2d((1, 5), stride=(1, 5)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_features = self.features(torch.zeros(1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(n_features, 1)

    def features(self, x):                 # x: [batch, channels, time]
        x = x.unsqueeze(1)                 # -> [batch, 1, channels, time]
        return self.spatial(self.temporal(x)).flatten(1)

    def forward(self, x):
        return self.classifier(self.features(x))
