"""ShallowConvNet (Schirrmeister et al. 2017), braindecode's ShallowFBCSPNet.

Cobilean et al. also set pool_nonlin=relu, but braindecode does not accept it, so it never
took effect in the original runs (mia_bci/exp1_models.py dropped it silently). It is
left out here.
"""

import torch.nn as nn
from braindecode.models import ShallowFBCSPNet


def build(n_channels, n_times, sampling_rate, dropout):
    return ShallowFBCSPNet(
        n_chans=n_channels, n_outputs=1, n_times=n_times,
        n_filters_time=48, filter_time_length=25, n_filters_spat=40,
        pool_time_length=25, pool_time_stride=5, pool_mode="mean",
        conv_nonlin=nn.ReLU, activation_pool_nonlin=nn.ReLU,
        split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
        drop_prob=dropout,
    )
