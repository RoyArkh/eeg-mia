"""TIDNet (Kostas & Rudzicz 2020), from braindecode. Settings from Cobilean et al."""

from braindecode.models import TIDNet


def build(n_channels, n_times, sampling_rate, dropout):
    return TIDNet(
        n_chans=n_channels, n_outputs=1, n_times=n_times,
        s_growth=16, t_filters=12, pooling=15, temp_layers=2, spat_layers=1,
        temp_span=0.05, bottleneck=3, summary=-1,
        drop_prob=dropout,
    )
