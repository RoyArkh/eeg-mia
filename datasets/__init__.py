"""Datasets. Each file in this folder holds one dataset.

A dataset is a class. It is created with `(data_dir=None, stride=None, participants=None)`
(None means the dataset's default) and provides:

    name, n_channels, n_times, sampling_rate, participants
    splits(participant_ids, rng)  -> {"train": (X, y), "val": (X, y), "test": (X, y)}
                                     X: float32 [windows, channels, time], y: float32 [windows] (0/1)
    settings()                    -> dict of its settings, saved with every result
    label                         -> short text for the results file name, e.g. "mobi-stride275"

To add a dataset, write datasets/<name>.py and add it to DATASETS below.
"""

from datasets.mobi import MoBI

DATASETS = {
    "mobi": MoBI,
}


def load_dataset(name, **options):
    """Load dataset `name`. Options left at None use the dataset's defaults."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}")
    options = {key: value for key, value in options.items() if value is not None}
    return DATASETS[name](**options)
