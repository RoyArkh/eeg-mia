"""Checks for the models. Run with: uv run pytest"""

import pytest
import torch

import models

# Parameter counts of the original models (mia_bci/exp1_models.py and exp4_model.py with
# filter_mode="per_layer", built with the bundled braindecode) for 4 channels x 275 samples.
ORIGINAL_SIZES = {
    "EEGNet": 11_009,
    "ShallowConvNet": 10_849,
    "Conv1DModel": 10_489,
    "EEGNet2TL": 10_081,
    "TIDNet": 10_709,
    "EEGInception": 10_613,
    "TestNetwork": 55_081,   # 40 filters, 2 layers (the defaults)
}


@pytest.mark.parametrize("name", models.MODELS)
def test_models_give_one_logit_per_window(name):
    model = models.build_model(name, n_channels=4, n_times=275, sampling_rate=110, dropout=0.2).eval()
    logits = models.forward_logits(model, torch.randn(8, 4, 275))
    assert logits.shape == (8,)


@pytest.mark.parametrize("name", models.MODELS)
def test_models_match_original_size(name):
    model = models.build_model(name, n_channels=4, n_times=275, sampling_rate=110, dropout=0.2)
    assert models.count_params(model) == ORIGINAL_SIZES[name]


@pytest.mark.parametrize("filters, n_layers, size", [(20, 1, 1_901), (60, 3, 236_341), (80, 5, 827_521)])
def test_test_network_sizes_match_original(filters, n_layers, size):
    model = models.build_model("TestNetwork", 4, 275, 110, 0.2, filters=filters, n_layers=n_layers)
    assert models.count_params(model) == size


def test_unknown_model_is_an_error():
    with pytest.raises(ValueError, match="Unknown model"):
        models.build_model("NoSuchNet", 4, 275, 110, 0.2)
