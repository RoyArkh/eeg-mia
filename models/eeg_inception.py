"""EEG-Inception (Santamaria-Vazquez et al. 2020), braindecode's EEGInceptionERP.

Settings from Cobilean et al. The kernel lengths are given in seconds
(scales_samples_s), so the model needs the dataset's sampling rate.
"""

import torch.nn as nn
from braindecode.models import EEGInceptionERP


def build(n_channels, n_times, sampling_rate, dropout):
    return EEGInceptionERP(
        n_chans=n_channels, n_outputs=1, n_times=n_times, sfreq=sampling_rate,
        scales_samples_s=(0.4, 0.2, 0.1), n_filters=8, activation=nn.ELU,
        batch_norm_alpha=0.01, depth_multiplier=2, pooling_sizes=(4, 2, 2, 2),
        drop_prob=dropout,
    )
