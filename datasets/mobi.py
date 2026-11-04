"""MoBI: rest vs. video-game EEG from 11 children (Muse headband, 4 channels, 110 Hz).

The files are the preprocessed recordings from mia_bci/preprocessed/: one SubjectN.npz
per participant with the continuous rest ("baseline") and game ("task") EEG, a
poor-contact flag per sample, and the sampling rate.

A model sees 2.5 s windows (275 samples) taken every `stride` samples. Each recording is
split in time order into train | gap | val | gap | test, so that no train window shares
a sample with a val or test window. Same pipeline as mia_bci/exp1_run.ipynb.
"""

import os
from dataclasses import dataclass

import numpy as np

import config

NAME = "mobi"
DATA_DIR = os.path.join(config.DATA_ROOT, "mobi")
PARTICIPANTS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13]   # the 11 with both rest and game recordings
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
SAMPLING_RATE = 110   # Hz
WINDOW = 275          # samples (2.5 s)
STRIDE = 275          # 275 = no overlap, 1 = 99.6% overlap (the paper's main MoBI setting)

MAX_BAD_FRACTION = 0.30    # drop windows where >30% of samples have poor electrode contact
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15        # the remaining 15% is test
MIN_WINDOWS_TO_SPLIT = 8   # recordings with fewer windows go entirely to train
Z_CLIP = 4.0               # z-scored windows are clipped to +-4

REST, GAME = 0, 1
SPLITS = ["train", "val", "test"]


class MoBI:
    """All MoBI participants, cut into windows at one stride."""

    name = NAME
    channels = CHANNELS
    n_channels = len(CHANNELS)
    n_times = WINDOW
    sampling_rate = SAMPLING_RATE

    def __init__(self, data_dir=DATA_DIR, stride=STRIDE, participants=PARTICIPANTS):
        self.data_dir = data_dir
        self.stride = stride
        self.participants = list(participants)
        self.subjects = {}
        for subject_id in self.participants:
            self.subjects[subject_id] = load_subject(data_dir, subject_id, stride)

    def splits(self, participant_ids, rng):
        """Balanced train/val/test windows of `participant_ids`, as {split: (X, y)}."""
        windows = build_splits(self.subjects, participant_ids, rng)
        return {split: windows_to_arrays(windows[split], self.subjects) for split in SPLITS}

    def settings(self):
        return {"dataset": NAME, "stride": self.stride, "window": WINDOW}

    @property
    def label(self):
        return f"{NAME}-stride{self.stride}"


@dataclass
class Subject:
    id: int
    signals: dict   # label -> EEG array [channels, time]
    starts: dict    # (split, label) -> array of window start indices


def load_subject(data_dir, subject_id, stride, window=WINDOW):
    """Load one participant and split each recording into train/val/test windows."""
    npz = np.load(os.path.join(data_dir, f"Subject{subject_id}.npz"))
    if int(npz["fs"]) != SAMPLING_RATE:
        raise ValueError(f"Subject{subject_id}: expected {SAMPLING_RATE} Hz, got {npz['fs']}")

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


def good_window_starts(bad, window, stride, max_bad_fraction=MAX_BAD_FRACTION):
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
    if n < MIN_WINDOWS_TO_SPLIT:
        return starts, starts[:0], starts[:0]

    n_val = max(1, int(n * VAL_FRACTION))
    n_test = max(1, int(n * (1.0 - TRAIN_FRACTION - VAL_FRACTION)))
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


def windows_to_arrays(windows, subjects, window=WINDOW):
    """Cut the windows out of the recordings. Returns X [n, channels, time] and y [n].

    Each channel of each window is z-scored, then clipped to +-Z_CLIP.
    """
    if not windows:
        return np.zeros((0, len(CHANNELS), window), dtype=np.float32), np.zeros(0, dtype=np.float32)
    X = np.stack([subjects[sid].signals[label][:, start:start + window] for sid, label, start in windows])
    y = np.array([label for _, label, _ in windows], dtype=np.float32)
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-7
    X = np.clip((X - mean) / std, -Z_CLIP, Z_CLIP)
    return X, y
