"""Datasets. Each file in this folder holds one dataset.

A dataset is a class. It is created with `(data_dir=None, stride=None, participants=None)`
(None means the dataset's default) and provides:

    name, n_channels, n_times, sampling_rate, participants
    splits(participant_ids, rng)  -> {"train": (X, y), "val": (X, y), "test": (X, y)}
                                     X: float32 [windows, channels, time], y: float32 [windows] (0/1)
    settings()                    -> dict of its settings, saved with every result
    label                         -> short text for the results file name, e.g. "mobi-stride275"
    default_counts (optional)     -> participant counts for run_participant_sweep.py

MoBI reads the preprocessed files from mia_bci. The other datasets read the .npz files
that prepare_data.py writes, through datasets/segments.py.

To add a dataset, write datasets/<name>.py and add it to DATASETS below.
"""

from datasets.eegmmidb import EEGMMIDB
from datasets.mobi import MoBI
from datasets.sleepedf import SleepEDF
from datasets.tuab import TUAB

DATASETS = {
    "mobi": MoBI,
    "sleepedf": SleepEDF,
    "eegmmidb": EEGMMIDB,
    "tuab": TUAB,
}


def load_dataset(name, **options):
    """Load dataset `name`. Options left at None use the dataset's defaults."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}")
    options = {key: value for key, value in options.items() if value is not None}
    return DATASETS[name](**options)
