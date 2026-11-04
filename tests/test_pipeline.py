"""End-to-end checks: training, attacks and results files. Run with: uv run pytest"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

import attacks
import datasets
import pipeline
import recording
from datasets import mobi

needs_mobi = pytest.mark.skipif(
    not os.path.exists(os.path.join(mobi.DATA_DIR, "Subject1.npz")),
    reason=f"MoBI .npz files not found in {mobi.DATA_DIR}",
)


@needs_mobi
def test_short_run_with_every_attack():
    dataset = datasets.load_dataset("mobi", stride=275)
    rng = np.random.default_rng(0)
    result = pipeline.run_one("Conv1DModel", dataset.participants, dataset, list(attacks.ATTACKS), rng,
                              seed=0, device=torch.device("cpu"), epochs=2,
                              attack_options={"lira": {"n_shadows": 6, "verbose": False}})
    assert result["n_train_windows"] == 250
    assert result["n_test_windows"] == 46
    assert result["n_attack"] == 92
    for name in attacks.ATTACKS:
        assert 0.0 <= result[f"{name}_auc"] <= 1.0
        assert 0.0 <= result[f"{name}_tpr"] <= 1.0


class FakeDataset:
    label = "fake-stride5"

    def settings(self):
        return {"dataset": "fake", "stride": 5}


def test_results_file_name_and_contents(tmp_path):
    path = recording.results_path(["EEGNet", "TIDNet"], ["loss", "lira"], FakeDataset(), 50,
                                  results_dir=str(tmp_path))
    name = os.path.basename(path)
    assert name.endswith("_EEGNet+TIDNet_loss+lira_fake-stride5-epochs50.csv")
    assert name[:19].replace("-", "").replace("_", "").isdigit()   # YYYY-MM-DD_HH-MM-SS

    info = recording.run_info("single", FakeDataset(), torch.device("cpu"), 50)
    recording.save([{**info, "model": "EEGNet", "loss_auc": 0.6}], path)
    saved = pd.read_csv(path)
    assert list(saved["loss_auc"]) == [0.6]
    assert saved.loc[0, "dataset"] == "fake" and saved.loc[0, "epochs"] == 50

    # a second run started in the same second gets its own file
    second = recording.results_path(["EEGNet", "TIDNet"], ["loss", "lira"], FakeDataset(), 50,
                                    results_dir=str(tmp_path))
    assert second != path
