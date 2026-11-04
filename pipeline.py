"""One experiment run: get the windows, train a target model, attack it."""

import time
from functools import partial

import numpy as np
import torch

import attacks
import config
import models
import train

SPLITS = ["train", "val", "test"]


def run_one(model_name, participant_ids, dataset, attack_names, rng, seed, device,
            epochs=config.EPOCHS, attack_options=None):
    """Train `model_name` on `participant_ids` of `dataset` and run the attacks in `attack_names`.

    Members are training windows, non-members are test windows (val windows were seen by
    early stopping, so they are left out). Returns a dict of results, or None if a split
    has fewer than MIN_SPLIT_SIZE windows.
    """
    start_time = time.time()
    splits = dataset.splits(participant_ids, rng)
    if min(len(y) for _, y in splits.values()) < config.MIN_SPLIT_SIZE:
        return None

    tensors = {}
    for split, (X, y) in splits.items():
        tensors[split] = (torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))

    torch.manual_seed(seed)
    model = build(model_name, dataset, device)
    epochs_run = train.train_model(model, tensors["train"], tensors["val"], epochs, seed)

    result = {
        "model": model_name,
        "n_participants": len(participant_ids),
        "participants": "-".join(str(s) for s in participant_ids),
        "seed": seed,
        "n_params": models.count_params(model),
        "epochs_run": epochs_run,
    }
    for split in SPLITS:
        result[f"n_{split}_windows"] = len(splits[split][1])

    logits = {}
    for split in SPLITS:
        logits[split], losses, accuracy = train.evaluate(model, *tensors[split])
        result[f"{split}_loss"] = float(losses.mean())
        result[f"{split}_acc"] = accuracy
    result["loss_gap"] = result["test_loss"] - result["train_loss"]
    result["acc_gap"] = result["train_acc"] - result["test_acc"]

    # attack windows: equally many members (train) and non-members (test)
    y_train, y_test = splits["train"][1], splits["test"][1]
    members, nonmembers = attacks.pick_balanced(len(y_train), len(y_test), rng)
    attack_data = attacks.AttackData(
        logits=np.concatenate([logits["train"][members], logits["test"][nonmembers]]),
        y=np.concatenate([y_train[members], y_test[nonmembers]]),
        is_member=np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))]),
        seed=seed,
        # shadow models (LiRA) may train on the train windows followed by the test windows
        pool_size=len(y_train) + len(y_test),
        pool_index=np.concatenate([members, len(y_train) + nonmembers]),
        train_shadow=partial(train_shadow, model_name, dataset, tensors, members, nonmembers,
                             device, epochs),
    )
    result.update(attacks.run_attacks(attack_names, attack_data, attack_options))
    result["seconds"] = round(time.time() - start_time, 1)
    return result


def build(model_name, dataset, device):
    return models.build_model(model_name, dataset.n_channels, dataset.n_times,
                              dataset.sampling_rate, config.DROPOUT).to(device)


def train_shadow(model_name, dataset, tensors, members, nonmembers, device, epochs, in_pool, seed):
    """Train a shadow model exactly like the target, on the pool windows where `in_pool` is True.

    The pool is the train windows followed by the test windows. Returns the shadow model's
    logits on the attack windows (members, then non-members).
    """
    X_train, y_train = tensors["train"]
    X_test, y_test = tensors["test"]
    n_train = len(y_train)
    pick_train = torch.from_numpy(np.flatnonzero(in_pool[:n_train])).to(device)
    pick_test = torch.from_numpy(np.flatnonzero(in_pool[n_train:])).to(device)
    X = torch.cat([X_train[pick_train], X_test[pick_test]])
    y = torch.cat([y_train[pick_train], y_test[pick_test]])

    torch.manual_seed(seed)
    shadow = build(model_name, dataset, device)
    train.train_model(shadow, (X, y), tensors["val"], epochs, seed)

    attack_X = torch.cat([X_train[torch.from_numpy(members).to(device)],
                          X_test[torch.from_numpy(nonmembers).to(device)]])
    return train.predict_logits(shadow, attack_X).cpu().numpy()
