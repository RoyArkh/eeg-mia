"""Checks for the datasets. Run with: uv run pytest"""

import os

import numpy as np
import pytest

import datasets
from datasets import mobi

needs_mobi = pytest.mark.skipif(
    not os.path.exists(os.path.join(mobi.DATA_DIR, "Subject1.npz")),
    reason=f"MoBI .npz files not found in {mobi.DATA_DIR}",
)

# Balanced (train, val, test) window counts per participant, as printed by the
# original mia_bci/exp1_run.ipynb. Matching them means the windows are the same.
ORIGINAL_COUNTS = {
    275: {1: (38, 8, 8), 2: (32, 6, 6), 3: (24, 4, 4), 4: (14, 2, 2), 5: (32, 6, 6), 6: (16, 4, 4),
          7: (14, 2, 2), 8: (22, 4, 4), 10: (32, 6, 6), 11: (8, 2, 2), 13: (18, 2, 2)},
    206: {1: (52, 10, 10), 2: (42, 8, 8), 3: (30, 6, 6), 4: (18, 2, 2), 5: (42, 8, 8), 6: (22, 4, 4),
          7: (18, 4, 4), 8: (28, 6, 6), 10: (42, 8, 8), 11: (12, 2, 2), 13: (20, 4, 4)},
    138: {1: (76, 16, 16), 2: (64, 12, 12), 3: (44, 10, 10), 4: (26, 4, 4), 5: (64, 12, 12), 6: (34, 6, 6),
          7: (28, 6, 6), 8: (44, 8, 8), 10: (64, 12, 12), 11: (16, 4, 4), 13: (34, 6, 6)},
    69: {1: (150, 32, 32), 2: (120, 26, 26), 3: (84, 20, 20), 4: (44, 10, 10), 5: (120, 26, 26), 6: (60, 14, 14),
         7: (50, 12, 12), 8: (80, 18, 18), 10: (120, 26, 26), 11: (28, 8, 8), 13: (58, 14, 14)},
    1: {1: (9794, 2334, 2334), 2: (7758, 1898, 1898), 3: (5450, 1402, 1402), 4: (2620, 796, 796),
        5: (7758, 1898, 1898), 6: (3718, 1030, 1030), 7: (2986, 874, 874), 8: (5082, 1322, 1322),
        10: (7758, 1898, 1898), 11: (1598, 578, 578), 13: (3476, 980, 980)},
}


def test_unknown_dataset_is_an_error():
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load_dataset("no_such_dataset")


def test_good_window_starts_drops_bad_windows():
    bad = np.zeros(1000, dtype=bool)
    bad[300:600] = True   # windows starting at 250 and 500 are more than 30% bad
    starts = mobi.good_window_starts(bad, window=250, stride=250, max_bad_fraction=0.3)
    assert list(starts) == [0, 750]


def test_split_by_time_leaves_gaps_between_blocks():
    starts = np.arange(0, 10000, 100)
    train, val, test = mobi.split_by_time(starts, window=250, stride=100)
    assert (len(train), len(val), len(test)) == (66, 15, 15)
    assert train.max() + 250 <= val.min()
    assert val.max() + 250 <= test.min()


@needs_mobi
@pytest.mark.parametrize("stride", ORIGINAL_COUNTS)
def test_window_counts_match_original_code(stride):
    for subject_id, expected in ORIGINAL_COUNTS[stride].items():
        subject = mobi.load_subject(mobi.DATA_DIR, subject_id, stride)
        counts = []
        for split in mobi.SPLITS:
            n_rest = len(subject.starts[(split, mobi.REST)])
            n_game = len(subject.starts[(split, mobi.GAME)])
            counts.append(2 * min(n_rest, n_game))
        assert tuple(counts) == expected, f"participant {subject_id}"


@needs_mobi
@pytest.mark.parametrize("stride", ORIGINAL_COUNTS)
def test_no_shared_samples_between_splits(stride):
    for subject_id in mobi.PARTICIPANTS:
        subject = mobi.load_subject(mobi.DATA_DIR, subject_id, stride)
        for label in [mobi.REST, mobi.GAME]:
            blocks = [subject.starts[(split, label)] for split in mobi.SPLITS]
            blocks = [block for block in blocks if len(block) > 0]
            for earlier, later in zip(blocks, blocks[1:]):
                assert earlier.max() + mobi.WINDOW <= later.min(), f"participant {subject_id}"


@needs_mobi
def test_windows_are_balanced_and_normalized():
    dataset = datasets.load_dataset("mobi", stride=275)
    splits = dataset.splits(dataset.participants, np.random.default_rng(0))
    assert [len(y) for _, y in splits.values()] == [250, 46, 46]
    for X, y in splits.values():
        assert X.shape == (len(y), dataset.n_channels, dataset.n_times)
        assert X.dtype == np.float32
        assert y.sum() == len(y) / 2
        assert np.abs(X).max() <= mobi.Z_CLIP


@needs_mobi
def test_empty_split_gives_empty_arrays():
    # with no participants every split is empty; this must not crash
    dataset = datasets.load_dataset("mobi", stride=275, participants=[11])
    splits = dataset.splits([], np.random.default_rng(0))
    for X, y in splits.values():
        assert X.shape == (0, dataset.n_channels, dataset.n_times)
        assert len(y) == 0
