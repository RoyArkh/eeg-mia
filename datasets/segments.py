"""Windowing, splitting and loading shared by the datasets that prepare_data.py writes.

prepare_data.py writes one .npz file per recording into data/<dataset>/, holding:

    signal     float32 [channels, time], filtered and normalized: each channel of the recording
               minus its median, divided by its IQR-based standard deviation, clipped to +-CLIP
    labels     int8 [time], one label per sample in the dataset's own coding; -1 = not usable
    subject    participant id
    fs         sampling rate in Hz
    channels   channel names
    center_uv, scale_uv   per channel, the median and scale (microvolts) that were removed

A dataset maps those labels to 0/1 (LABEL_MAP; anything else becomes unusable). A window
is `window` samples that all have the same usable label. Windows start where a run of equal
labels starts and then every `stride` samples, so a 30 s window lines up with the 30 s
sleep epochs and a 4 s window with the start of a motor-imagery trial.

Three ways to split into train/val/test:
- "time": each recording is split in time order, [train][gap][val][gap][test], 70/15/15 by
  number of windows, like MoBI. With time_split_unit = "participant", a participant's
  recordings are first put one after the other (for short runs, such as EEGMMIDB's).
- "blocks": each recording is cut into blocks of `block` samples (e.g. 10 minutes), and
  the blocks go to train, val and test (70/15/15) in a random order that is fixed per
  recording. Every split then covers the whole recording, which matters when the labels
  depend on the time (sleep: deep sleep early in the night, REM late).
- "subject": whole participants go to train, val or test (70/15/15, keeping the share of
  each label). Members and non-members are different people.
In "time" and "blocks", windows that would share a sample with a window of another split
are dropped (the gap), and members and non-members come from the same people. Each split
then gets equally many windows of label 0 and 1 (per participant for "time" and "blocks",
per split for "subject").
"""

import glob
import os
import zlib
from dataclasses import dataclass

import numpy as np

SPLITS = ["train", "val", "test"]
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15          # the remaining 15% is test
MIN_WINDOWS_TO_SPLIT = 8     # recordings with fewer windows go entirely to train
CLIP = 10.0                  # normalized signals are clipped to +-CLIP (robust standard deviations)


@dataclass
class Recording:
    subject: object          # participant id (int, or str for TUAB)
    name: str                # file name without .npz
    signal: np.ndarray       # [channels, time]
    starts: np.ndarray       # start of every usable window
    labels: np.ndarray       # 0/1 label of every usable window


def window_starts(labels, window, stride):
    """Start and label of every window whose samples all have the same usable (>= 0) label."""
    labels = np.asarray(labels)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(labels)) + 1, [len(labels)]])
    starts, window_labels = [], []
    for begin, end in zip(bounds[:-1], bounds[1:]):
        if labels[begin] < 0:
            continue
        for start in range(begin, end - window + 1, stride):
            starts.append(start)
            window_labels.append(labels[begin])
    return np.array(starts, dtype=int), np.array(window_labels, dtype=int)


def split_by_time(starts, window, min_windows=MIN_WINDOWS_TO_SPLIT):
    """Indices (into `starts`) of the train, val and test windows of one recording.

    Time order [train][gap][val][gap][test]: the last 15% of windows are test, the 15%
    before them val, the rest train; windows that would share a sample with the next
    block are dropped from the earlier block.
    """
    order = np.argsort(starts, kind="stable")
    n = len(order)
    if n < min_windows:
        return order, order[:0], order[:0]
    n_val = max(1, int(n * VAL_FRACTION))
    n_test = max(1, int(n * (1.0 - TRAIN_FRACTION - VAL_FRACTION)))

    test = order[n - n_test:]
    before = order[:n - n_test]
    before = before[starts[before] + window <= starts[test].min()]
    val = before[len(before) - n_val:] if len(before) > n_val else before[:0]
    train = before[:len(before) - len(val)]
    if len(val):
        train = train[starts[train] + window <= starts[val].min()]
    return train, val, test


def split_by_blocks(starts, labels, window, block, seed, min_windows=MIN_WINDOWS_TO_SPLIT,
                    n_candidates=200):
    """Indices of the train, val and test windows of one recording, split in blocks of time.

    Blocks of `block` samples (by window start) go to train/val/test, 70/15/15. Of
    `n_candidates` random assignments (fixed by `seed`), the one is used whose splits are
    closest to the recording's share of label 1: so that, for example, REM sleep is not
    missing from the test blocks of a night by chance. Where windows of two splits
    overlap, the train window is dropped (or the val window, if val and test overlap),
    so no sample is in two splits.
    """
    n = len(starts)
    if n < min_windows:
        return np.arange(n), np.arange(0), np.arange(0)
    block_of = np.searchsorted(np.unique(starts // block), starts // block)
    n_blocks = block_of.max() + 1
    n_val = max(1, int(round(n_blocks * VAL_FRACTION)))
    n_test = max(1, int(round(n_blocks * (1.0 - TRAIN_FRACTION - VAL_FRACTION))))
    windows_in_block = np.bincount(block_of, minlength=n_blocks)
    label1_in_block = np.bincount(block_of, weights=labels == 1, minlength=n_blocks)
    share = label1_in_block.sum() / n

    rng = np.random.default_rng(seed)
    best, best_error = None, np.inf
    for _ in range(n_candidates):
        order = rng.permutation(n_blocks)
        split_of_block = np.zeros(n_blocks, dtype=int)          # 0 train, 1 val, 2 test
        split_of_block[order[:n_val]] = 1
        split_of_block[order[n_val:n_val + n_test]] = 2
        error = 0.0
        for k in range(3):
            chosen = split_of_block == k
            count = windows_in_block[chosen].sum()
            error += abs(label1_in_block[chosen].sum() / count - share) if count else 1.0
        if error < best_error:
            best, best_error = split_of_block, error
    split = best[block_of]

    keep = np.ones(n, dtype=bool)
    order = np.argsort(starts, kind="stable")
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            if starts[j] >= starts[i] + window:
                break
            if split[i] != split[j]:
                keep[i if split[i] < split[j] else j] = False   # keep the test (then val) window
    return tuple(np.flatnonzero(keep & (split == k)) for k in range(3))


def split_subjects(subject_labels, rng):
    """Assign participants to train/val/test, 70/15/15 within each label. Returns {split: [ids]}."""
    splits = {split: [] for split in SPLITS}
    for label in sorted(set(subject_labels.values())):
        ids = [sid for sid, lab in subject_labels.items() if lab == label]
        ids = [ids[i] for i in rng.permutation(len(ids))]
        n = len(ids)
        n_val = int(round(n * VAL_FRACTION))
        n_test = int(round(n * (1.0 - TRAIN_FRACTION - VAL_FRACTION)))
        if n >= 3:
            n_val, n_test = max(1, n_val), max(1, n_test)
        n_train = n - n_val - n_test
        splits["train"] += ids[:n_train]
        splits["val"] += ids[n_train:n_train + n_val]
        splits["test"] += ids[n_train + n_val:]
    return splits


def balance(windows, rng):
    """Equally many windows of label 0 and 1, picked at random. `windows`: list of (rec, start, label)."""
    by_label = {0: [w for w in windows if w[2] == 0], 1: [w for w in windows if w[2] == 1]}
    k = min(len(by_label[0]), len(by_label[1]))
    picked = []
    for label in (0, 1):
        for i in sorted(rng.choice(len(by_label[label]), k, replace=False)):
            picked.append(by_label[label][i])
    return picked


class PreparedDataset:
    """A dataset of .npz recordings written by prepare_data.py. Subclasses set the class attributes."""

    name = None
    window = None            # samples
    default_stride = None    # samples; default: no overlap
    label_map = None         # stored label -> 0/1
    split_mode = "time"      # "time", "blocks" or "subject"
    time_split_unit = "recording"   # "recording" or "participant" (see the module docstring)
    block = None             # samples per block, for split_mode "blocks"
    default_counts = None    # participant counts for the participant sweep
    excluded = ()            # participants left out (e.g. after the data check)

    def __init__(self, data_dir=None, stride=None, participants=None):
        self.data_dir = data_dir or default_data_dir(self.name)
        self.stride = stride or self.default_stride or self.window
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {self.data_dir}. "
                                    f"Run: uv run python prepare_data.py {self.name}")
        wanted = None if participants is None else {str(p) for p in participants}
        self.recordings = {}       # subject -> [Recording]
        for path in files:
            with np.load(path) as npz:
                subject = npz["subject"].item()
                if subject in self.excluded or (wanted is not None and str(subject) not in wanted):
                    continue
                fs = float(npz["fs"])
                self.channels = [str(c) for c in npz["channels"]]
                labels = map_labels(npz["labels"], self.label_map)
                starts, window_labels = window_starts(labels, self.window, self.stride)
                if len(starts) == 0:
                    continue
                recording = Recording(subject, os.path.basename(path)[:-4],
                                      npz["signal"].astype(np.float32), starts, window_labels)
            self.recordings.setdefault(subject, []).append(recording)
            self.sampling_rate = fs
        if not self.recordings:
            raise ValueError(f"No usable recordings for participants {participants} in {self.data_dir}")
        self.participants = sorted(self.recordings, key=sort_key)
        self.n_channels = len(self.channels)
        self.n_times = self.window

    def splits(self, participant_ids, rng):
        """Balanced train/val/test windows of `participant_ids`, as {split: (X, y)}."""
        windows = {split: [] for split in SPLITS}
        if self.split_mode == "subject":
            subject_labels = {sid: self.subject_label(sid) for sid in participant_ids}
            for split, ids in split_subjects(subject_labels, rng).items():
                for sid in ids:
                    for rec in self.recordings[sid]:
                        windows[split] += [(rec, s, lab) for s, lab in zip(rec.starts, rec.labels)]
                windows[split] = balance(windows[split], rng)
        else:
            for sid in participant_ids:
                per_subject = self.time_split(sid)
                for split in SPLITS:
                    windows[split] += balance(per_subject[split], rng)
        return {split: self.to_arrays(windows[split]) for split in SPLITS}

    def time_split(self, subject_id):
        """{split: [(recording, start, label)]} of one participant, before balancing ("time", "blocks")."""
        per_subject = {split: [] for split in SPLITS}
        for group in self.time_split_groups(subject_id):
            # positions on one time line; recordings are a window apart, so they never overlap
            items, positions, offset = [], [], 0
            for rec in group:
                items += [(rec, s, lab) for s, lab in zip(rec.starts, rec.labels)]
                positions.append(rec.starts + offset)
                offset += rec.signal.shape[1] + self.window
            positions = np.concatenate(positions)
            if self.split_mode == "blocks":
                seed = zlib.crc32("+".join(rec.name for rec in group).encode())
                labels = np.array([lab for _, _, lab in items])
                index_per_split = split_by_blocks(positions, labels, self.window, self.block, seed)
            else:
                index_per_split = split_by_time(positions, self.window)
            for split, index in zip(SPLITS, index_per_split):
                per_subject[split] += [items[i] for i in index]
        return per_subject

    def time_split_groups(self, subject_id):
        """Recordings that are split together in time order: each alone, or all of a participant."""
        recordings = sorted(self.recordings[subject_id], key=lambda rec: rec.name)
        if self.time_split_unit == "participant":
            return [recordings]
        return [[rec] for rec in recordings]

    def subject_label(self, subject_id):
        """The most common window label of a participant (for "subject" splits)."""
        labels = np.concatenate([rec.labels for rec in self.recordings[subject_id]])
        return int(np.bincount(labels, minlength=2).argmax())

    def to_arrays(self, windows):
        if not windows:
            return (np.zeros((0, self.n_channels, self.window), dtype=np.float32),
                    np.zeros(0, dtype=np.float32))
        X = np.stack([rec.signal[:, start:start + self.window] for rec, start, _ in windows])
        y = np.array([label for _, _, label in windows], dtype=np.float32)
        return X.astype(np.float32), y

    def settings(self):
        return {"dataset": self.name, "stride": self.stride, "window": self.window,
                "split_mode": self.split_mode, "block": self.block}

    @property
    def label(self):
        return f"{self.name}-stride{self.stride}"


def map_labels(stored, label_map):
    """Stored labels -> 0/1, everything not in `label_map` -> -1."""
    labels = np.full(len(stored), -1, dtype=np.int8)
    for stored_label, label in label_map.items():
        labels[stored == stored_label] = label
    return labels


def default_data_dir(name):
    import config
    return os.path.join(config.DATA_ROOT, name)


def sort_key(subject):
    return (0, subject, "") if isinstance(subject, (int, np.integer)) else (1, 0, str(subject))


def robust_normalize(signal, clip=CLIP):
    """Each channel minus its median, divided by IQR / 1.349 (= the std for Gaussian data), clipped.

    Returns the normalized float32 signal and the per-channel center and scale that were removed.
    """
    center = np.median(signal, axis=1, keepdims=True)
    q75, q25 = np.percentile(signal, [75, 25], axis=1, keepdims=True)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale > 0, scale, 1.0)
    normalized = np.clip((signal - center) / scale, -clip, clip).astype(np.float32)
    return normalized, center.ravel(), scale.ravel()
