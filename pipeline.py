"""One experiment run: choose windows, train a model, attack it."""

import platform
import time

import braindecode
import numpy as np
import torch

import attacks
import config
import data
import models
import train


def run_one(model_name, subject_ids, subjects, rng, seed, device, epochs=config.EPOCHS):
    """Train `model_name` on `subject_ids` and run the LOSS attack.

    Members are training windows, non-members are test windows (val windows were
    seen by early stopping, so they are left out). Returns a dict of results, or
    None if a split has fewer than MIN_SPLIT_SIZE windows.
    """
    start_time = time.time()
    splits = data.build_splits(subjects, subject_ids, rng)
    if min(len(windows) for windows in splits.values()) < config.MIN_SPLIT_SIZE:
        return None

    tensors = {}
    for split, windows in splits.items():
        X, y = data.windows_to_arrays(windows, subjects)
        tensors[split] = (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))

    torch.manual_seed(seed)
    n_channels, n_times = tensors["train"][0].shape[1:]
    model = models.build_model(model_name, n_channels, n_times, config.DROPOUT).to(device)
    epochs_run = train.train_model(model, tensors["train"], tensors["val"], epochs, seed)

    result = {
        "model": model_name,
        "n_participants": len(subject_ids),
        "participants": "-".join(str(s) for s in subject_ids),
        "seed": seed,
        "n_params": models.count_params(model),
        "epochs_run": epochs_run,
    }
    for split in data.SPLITS:
        result[f"n_{split}_windows"] = len(splits[split])

    losses = {}
    for split in data.SPLITS:
        losses[split], accuracy = train.evaluate(model, *tensors[split])
        result[f"{split}_loss"] = float(losses[split].mean())
        result[f"{split}_acc"] = accuracy
    result["loss_gap"] = result["test_loss"] - result["train_loss"]
    result["acc_gap"] = result["train_acc"] - result["test_acc"]

    result.update(attacks.loss_attack(losses["train"], losses["test"], rng))
    result["seconds"] = round(time.time() - start_time, 1)
    return result


def print_environment(device):
    print(f"python {platform.python_version()}, torch {torch.__version__}, "
          f"braindecode {braindecode.__version__}, numpy {np.__version__}, device: {device}")
