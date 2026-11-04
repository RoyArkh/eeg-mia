"""The target models. Each maps [batch, channels, time] to one logit per window."""

import torch
import torch.nn as nn
from braindecode.models import EEGNet, ShallowFBCSPNet

import config


def build_model(name, n_channels, n_times, dropout):
    """Hyperparameters are the ones that took effect in mia_bci/exp1_models.py."""
    if name == "EEGNet":
        return EEGNet(
            n_chans=n_channels, n_outputs=1, n_times=n_times,
            F1=32, D=3, F2=64, kernel_length=64, pool_mode="mean",
            drop_prob=dropout,
        )
    if name == "ShallowConvNet":
        return ShallowFBCSPNet(
            n_chans=n_channels, n_outputs=1, n_times=n_times,
            n_filters_time=48, filter_time_length=25, n_filters_spat=40,
            pool_time_length=25, pool_time_stride=5, pool_mode="mean",
            conv_nonlin=nn.ReLU, activation_pool_nonlin=nn.ReLU,
            split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
            drop_prob=dropout,
        )
    if name == "Conv1DModel":
        return Conv1DModel(n_channels, dropout)
    raise ValueError(f"Unknown model: {name}")


def forward_logits(model, x):
    """One logit per window, shape [batch]. Fails if a model returns anything else."""
    return model(x).reshape(len(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Conv1DModel(nn.Module):
    """Three 1-D conv blocks (kernels 64, 32, 16), global average pooling, linear output.

    Cobilean et al. (2025); same layers as Conv1DModel in mia_bci/exp1_models.py.
    """

    def __init__(self, n_channels, dropout):
        super().__init__()
        self.blocks = nn.Sequential(
            conv_block(n_channels, 32, kernel_size=64, groups=n_channels, dropout=dropout),
            conv_block(32, 32, kernel_size=32, groups=32, dropout=dropout),
            conv_block(32, 14, kernel_size=16, groups=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(14, 1)

    def forward(self, x):
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


if __name__ == "__main__":
    x = torch.randn(8, 4, config.WINDOW)
    for name in config.MODELS:
        model = build_model(name, 4, config.WINDOW, config.DROPOUT).eval()
        print(f"{name:<15} {count_params(model):>7,} parameters, output shape {tuple(forward_logits(model, x).shape)}")
