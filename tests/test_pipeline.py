"""Checks for the data pipeline, models and attack. Run with: uv run pytest"""

import os

import numpy as np
import pytest
import torch

import attacks
import config
import data
import models
import pipeline

needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(config.DATA_DIR, "Subject1.npz")),
    reason=f"MoBI .npz files not found in {config.DATA_DIR}",
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


def test_good_window_starts_drops_bad_windows():
    bad = np.zeros(1000, dtype=bool)
    bad[300:600] = True   # windows starting at 250 and 500 are more than 30% bad
    starts = data.good_window_starts(bad, window=250, stride=250, max_bad_fraction=0.3)
    assert list(starts) == [0, 750]


def test_split_by_time_leaves_gaps_between_blocks():
    starts = np.arange(0, 10000, 100)
    train, val, test = data.split_by_time(starts, window=250, stride=100)
    assert (len(train), len(val), len(test)) == (66, 15, 15)
    assert train.max() + 250 <= val.min()
    assert val.max() + 250 <= test.min()


@needs_data
@pytest.mark.parametrize("stride", ORIGINAL_COUNTS)
def test_window_counts_match_original_code(stride):
    for subject_id, expected in ORIGINAL_COUNTS[stride].items():
        subject = data.load_subject(config.DATA_DIR, subject_id, stride)
        counts = []
        for split in data.SPLITS:
            n_rest = len(subject.starts[(split, data.REST)])
            n_game = len(subject.starts[(split, data.GAME)])
            counts.append(2 * min(n_rest, n_game))
        assert tuple(counts) == expected, f"participant {subject_id}"


@needs_data
@pytest.mark.parametrize("stride", ORIGINAL_COUNTS)
def test_no_shared_samples_between_splits(stride):
    for subject_id in config.PARTICIPANTS:
        subject = data.load_subject(config.DATA_DIR, subject_id, stride)
        for label in [data.REST, data.GAME]:
            blocks = [subject.starts[(split, label)] for split in data.SPLITS]
            blocks = [block for block in blocks if len(block) > 0]
            for earlier, later in zip(blocks, blocks[1:]):
                assert earlier.max() + config.WINDOW <= later.min(), f"participant {subject_id}"


@needs_data
def test_windows_are_balanced_and_normalized():
    subjects = {}
    for subject_id in config.PARTICIPANTS:
        subjects[subject_id] = data.load_subject(config.DATA_DIR, subject_id, stride=275)
    splits = data.build_splits(subjects, config.PARTICIPANTS, np.random.default_rng(0))
    for windows in splits.values():
        X, y = data.windows_to_arrays(windows, subjects)
        assert X.shape == (len(windows), 4, config.WINDOW)
        assert X.dtype == np.float32
        assert y.sum() == len(y) / 2
        assert np.abs(X).max() <= config.Z_CLIP


@pytest.mark.parametrize("name", config.MODELS)
def test_models_give_one_logit_per_window(name):
    model = models.build_model(name, n_channels=4, n_times=config.WINDOW, dropout=0.2).eval()
    logits = models.forward_logits(model, torch.randn(8, 4, config.WINDOW))
    assert logits.shape == (8,)
    assert 1_000 < models.count_params(model) < 100_000   # paper: on the order of 10^4


def test_conv1d_model_matches_original_size():
    model = models.Conv1DModel(n_channels=4, dropout=0.2)
    assert models.count_params(model) == 10_489   # value from mia_bci/exp1_models.py


def test_loss_attack_perfect_separation():
    rng = np.random.default_rng(0)
    member_losses = rng.uniform(0.0, 0.4, 500)
    nonmember_losses = rng.uniform(0.6, 1.0, 500)
    result = attacks.loss_attack(member_losses, nonmember_losses, rng)
    assert result["attack_auc"] == 1.0
    assert result["attack_tpr"] == 1.0


def test_loss_attack_without_leak_is_near_chance():
    rng = np.random.default_rng(0)
    losses = rng.random(20_000)
    result = attacks.loss_attack(losses[:10_000], losses[10_000:], rng)
    assert abs(result["attack_auc"] - 0.5) < 0.02


@needs_data
def test_short_run_end_to_end():
    subjects = {}
    for subject_id in config.PARTICIPANTS:
        subjects[subject_id] = data.load_subject(config.DATA_DIR, subject_id, stride=275)
    rng = np.random.default_rng(0)
    result = pipeline.run_one("Conv1DModel", config.PARTICIPANTS, subjects, rng,
                              seed=0, device=torch.device("cpu"), epochs=2)
    assert result["n_train_windows"] == 250
    assert result["n_test_windows"] == 46
    assert 0.0 <= result["attack_auc"] <= 1.0
