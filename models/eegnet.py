"""EEGNet (Lawhern et al. 2018), from braindecode.

Cobilean et al. also set third_kernel_size=(8, 4), but braindecode does not accept it, so
it never took effect in the original runs (mia_bci/exp1_models.py dropped it silently).
It is left out here.
"""

from braindecode.models import EEGNet


def build(n_channels, n_times, sampling_rate, dropout):
    return EEGNet(
        n_chans=n_channels, n_outputs=1, n_times=n_times,
        F1=32, D=3, F2=64, kernel_length=64, pool_mode="mean",
        drop_prob=dropout,
    )
