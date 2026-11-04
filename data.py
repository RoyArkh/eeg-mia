"""Load the preprocessed MoBI recordings and cut them into labelled EEG windows."""

import os
from dataclasses import dataclass

import numpy as np

import config

REST, GAME = 0, 1
SPLITS = ["train", "val", "test"]


@dataclass
class Subject:
    id: int
    signals: dict   # label -> EEG array [channels, time]
    starts: dict    # (split, label) -> array of window start indices


def load_subject(data_dir, subject_id, stride, window=config.WINDOW):
    """Load one participant and split each recording into train/val/test windows."""
    npz = np.load(os.path.join(data_dir, f"Subject{subject_id}.npz"))
    if int(npz["fs"]) != config.SAMPLING_RATE:
        raise ValueError(f"Subject{subject_id}: expected {config.SAMPLING_RATE} Hz, got {npz['fs']}")

    recordings = {
        REST: (npz["baseline"], npz["baseline_bad"]),
        GAME: (npz["task"], npz["task_bad"]),
    }
    signals = {}
    starts = {}
    for label, (signal, bad) in recordings.items():
        signals[label] = signal.astype(np.float32)
        good_starts = good_window_starts(bad.astype(bool), window, stride)
        train, val, test = split_by_time(good_starts, window, stride)
        starts[("train", label)] = train
        starts[("val", label)] = val
        starts[("test", label)] = test
    return Subject(subject_id, signals, starts)


def good_window_starts(bad, window, stride, max_bad_fraction=config.MAX_BAD_FRACTION):
    """Start indices of windows with at most `max_bad_fraction` poor-contact samples."""
    starts = []
    for start in range(0, len(bad) - window + 1, stride):
        if bad[start:start + window].mean() <= max_bad_fraction:
            starts.append(start)
    return np.array(starts, dtype=int)


def split_by_time(starts, window, stride):
    """Split one recording's windows in time order: [train][gap][val][gap][test].

    The gap is long enough that no train window shares a sample with a val or test
    window. Same rule as split_by_count() in mia_bci/exp1_run.ipynb.
    """
    n = len(starts)
    if n < config.MIN_WINDOWS_TO_SPLIT:
        return starts, starts[:0], starts[:0]

    n_val = max(1, int(n * config.VAL_FRACTION))
    n_test = max(1, int(n * (1.0 - config.TRAIN_FRACTION - config.VAL_FRACTION)))
    gap = max(1, window // stride)
    # a very short recording gets a smaller gap so that train keeps >= 40% of its windows
    while gap > 0 and n - n_val - n_test - 2 * gap < max(n_val, int(0.4 * n)):
        gap -= 1

    n_train = n - n_val - n_test - 2 * gap
    train = starts[:n_train]
    val = starts[n_train + gap:n_train + gap + n_val]
    test = starts[n_train + gap + n_val + gap:]
    return train, val, test


def balanced_windows(subject, split, rng):
    """Equal numbers of rest and game windows for one subject, as (subject_id, label, start)."""
    rest = subject.starts[(split, REST)]
    game = subject.starts[(split, GAME)]
    k = min(len(rest), len(game))
    if k == 0:
        return []
    windows = []
    for label, starts in [(REST, rest), (GAME, game)]:
        for start in rng.choice(starts, k, replace=False):
            windows.append((subject.id, label, int(start)))
    return windows


def build_splits(subjects, subject_ids, rng):
    """Balanced train/val/test window lists for the chosen participants."""
    splits = {}
    for split in SPLITS:
        windows = []
        for subject_id in subject_ids:
            windows += balanced_windows(subjects[subject_id], split, rng)
        splits[split] = windows
    return splits


def windows_to_arrays(windows, subjects, window=config.WINDOW):
    """Cut the windows out of the recordings. Returns X [n, channels, time] and y [n].

    Each channel of each window is z-scored, then clipped to +-Z_CLIP.
    """
    X = np.stack([subjects[sid].signals[label][:, start:start + window] for sid, label, start in windows])
    y = np.array([label for _, label, _ in windows], dtype=np.float32)
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-7
    X = np.clip((X - mean) / std, -config.Z_CLIP, config.Z_CLIP)
    return X, y
