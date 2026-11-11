"""Checks for the prepared datasets (Sleep-EDF, EEGMMIDB, TUAB) on made-up recordings.

No download needed. Run with: uv run pytest
"""

import mne
import numpy as np
import pytest
import torch

import attacks
import datasets
import pipeline
import prepare_data
from datasets import segments, sleepedf, tuab
from datasets.segments import SPLITS


def test_windows_have_one_label_and_start_at_label_changes():
    labels = np.array([-1] * 5 + [0] * 30 + [1] * 25 + [-1] * 10 + [0] * 9)
    starts, window_labels = segments.window_starts(labels, window=10, stride=10)
    assert list(starts) == [5, 15, 25, 35, 45]
    assert list(window_labels) == [0, 0, 0, 1, 1]
    for start, label in zip(starts, window_labels):
        assert (labels[start:start + 10] == label).all()


def test_half_overlap_windows():
    starts, _ = segments.window_starts(np.zeros(40, dtype=int), window=10, stride=5)
    assert list(starts) == [0, 5, 10, 15, 20, 25, 30]


@pytest.mark.parametrize("stride", [10, 5])
def test_time_split_has_no_shared_samples(stride):
    starts = np.arange(0, 1000, stride)
    train, val, test = segments.split_by_time(starts, window=10)
    assert len(train) > len(val) > 0 and len(test) > 0
    assert starts[train].max() + 10 <= starts[val].min()
    assert starts[val].max() + 10 <= starts[test].min()


def test_subject_split_keeps_labels_and_people_apart():
    labels = {f"p{i}": i % 2 for i in range(40)}
    split = segments.split_subjects(labels, np.random.default_rng(0))
    assert sorted(sum(split.values(), [])) == sorted(labels)
    assert not set(split["train"]) & (set(split["val"]) | set(split["test"]))
    for ids in split.values():
        assert 0 < sum(labels[i] for i in ids) < len(ids)   # both labels in every split


def write_fake_recordings(folder, n_subjects=4, nights=2, epochs=120):
    """Fake Sleep-EDF nights: alternating blocks of N2 and REM epochs, a bit of wake."""
    stages = sleepedf.STAGES
    rng = np.random.default_rng(1)
    for subject in range(n_subjects):
        for night in range(1, nights + 1):
            epoch_stages = [stages["W"]] * 4 + [stages["N2"], stages["N2"], stages["REM"]] * (epochs // 3)
            labels = np.repeat(epoch_stages, 3000)
            signal = rng.normal(size=(2, len(labels))) * 30
            signal[:, labels == stages["REM"]] *= 0.5
            prepare_data.save(str(folder), f"S{subject:02d}_N{night}", signal, labels, subject, 100.0,
                              ["Fpz-Cz", "Pz-Oz"])


def test_sleepedf_splits_are_balanced_and_normalized(tmp_path):
    write_fake_recordings(tmp_path)
    for stride in [3000, 1500]:
        dataset = datasets.load_dataset("sleepedf", data_dir=str(tmp_path), stride=stride)
        assert dataset.participants == [0, 1, 2, 3]
        assert (dataset.n_channels, dataset.n_times, dataset.sampling_rate) == (2, 3000, 100.0)
        splits = dataset.splits([0, 1, 2], np.random.default_rng(0))
        for X, y in splits.values():
            assert len(y) > 0 and y.sum() == len(y) / 2
            assert X.shape == (len(y), 2, 3000) and X.dtype == np.float32
            assert np.abs(X).max() <= segments.CLIP


def test_tuab_split_puts_each_patient_in_one_split(tmp_path):
    rng = np.random.default_rng(2)
    for i in range(30):
        signal = rng.normal(size=(16, 30000)) * 20
        labels = np.full(30000, i % 2)
        prepare_data.save(str(tmp_path), f"pat{i:02d}", signal, labels, f"pat{i:02d}", 100.0,
                          [f"{a}-{b}" for a, b in tuab.BIPOLAR])
    dataset = datasets.load_dataset("tuab", data_dir=str(tmp_path))
    assert len(dataset.participants) == 30 and dataset.n_channels == 16
    rng = np.random.default_rng(0)
    windows = {}
    for split, ids in segments.split_subjects({p: dataset.subject_label(p) for p in dataset.participants},
                                              np.random.default_rng(0)).items():
        windows[split] = set(ids)
    assert not windows["train"] & windows["test"]
    splits = dataset.splits(dataset.participants, rng)
    for X, y in splits.values():
        assert y.sum() == len(y) / 2 and X.shape[1:] == (16, 1000)


def test_sleep_stage_labels_from_annotations():
    annotations = mne.Annotations(onset=[0, 30, 60, 90], duration=[30, 30, 30, 30],
                                  description=["Sleep stage W", "Sleep stage 4", "Sleep stage R",
                                               "Sleep stage ?"])
    labels = prepare_data.sleep_stage_labels(annotations, n_samples=12000, fs=100)
    assert list(labels[::3000]) == [0, 3, 4, -1]


def test_trim_keeps_margin_around_sleep():
    labels = np.zeros(10000, dtype=np.int8)
    labels[4000:6000] = 2
    assert prepare_data.trim_to_sleep(labels, fs=1, margin_seconds=1000) == (3000, 7000)


def test_tuab_bipolar_montage():
    electrodes = sorted({e for pair in tuab.BIPOLAR for e in pair}) + ["A1"]
    info = mne.create_info([f"EEG {e}-REF" for e in electrodes], 250.0, "eeg")
    data = np.arange(len(electrodes))[:, None] * np.ones((1, 500)) * 1e-6
    bipolar = prepare_data.tuab_bipolar(mne.io.RawArray(data, info))
    index = {e: i for i, e in enumerate(electrodes)}
    expected = [index[a] - index[b] for a, b in tuab.BIPOLAR]
    assert bipolar.shape == (16, 500)
    assert np.allclose(bipolar[:, 0], expected)


def test_short_run_on_fake_sleep_data(tmp_path):
    write_fake_recordings(tmp_path, n_subjects=3)
    dataset = datasets.load_dataset("sleepedf", data_dir=str(tmp_path))
    result = pipeline.run_one("EEGNet2TL", dataset.participants, dataset, ["loss", "max_confidence"],
                              np.random.default_rng(0), seed=0, device=torch.device("cpu"), epochs=2)
    assert result["n_train_windows"] > result["n_test_windows"] > 0
    assert 0.0 <= result["loss_auc"] <= 1.0
    assert set(SPLITS) <= {k.split("_")[0] for k in result if k.endswith("_acc")}
    assert attacks.ATTACKS


@pytest.mark.parametrize("stride", [3000, 1500])
def test_block_split_spreads_splits_and_shares_no_samples(stride):
    starts = np.arange(0, 3_000_000 - 3000 + 1, stride)        # ~8 h at 100 Hz
    train, val, test = segments.split_by_blocks(starts, window=3000, block=60000, seed=7)
    for earlier, later in [(train, val), (train, test), (val, test)]:
        a, b = starts[earlier][:, None], starts[later][None, :]
        assert ((a + 3000 <= b) | (b + 3000 <= a)).all()          # no shared samples
    for index in (val, test):
        assert 0.10 < len(index) / len(starts) < 0.20
        assert starts[index].min() < 1_000_000 and starts[index].max() > 2_000_000   # all night
    again = segments.split_by_blocks(starts, window=3000, block=60000, seed=7)
    assert all((x == y).all() for x, y in zip((train, val, test), again))
