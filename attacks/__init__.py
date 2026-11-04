"""Membership inference attacks. Each file in this folder holds one attack.

Every attack file has a function

    scores(data, **options) -> one membership score per attack window

where `data` is an AttackData (below) and a HIGHER score means "more likely a training
member". run_attacks() scores all chosen attacks on the same windows of the same
target model and turns each attack's scores into AUC and TPR at 1% FPR, so the attacks
are directly comparable.

To add an attack, write attacks/<name>.py with a scores() function and add it to ATTACKS
below.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import config
from attacks import confidence, lira, loss, max_confidence, modified_entropy

ATTACKS = {
    "loss": loss,
    "confidence": confidence,
    "max_confidence": max_confidence,
    "modified_entropy": modified_entropy,
    "lira": lira,
}


@dataclass
class AttackData:
    """Everything an attack may use. Members come first, then non-members."""

    logits: np.ndarray       # the target model's logit for each attack window
    y: np.ndarray            # true label of each attack window (0 or 1)
    is_member: np.ndarray    # 1 = the target trained on this window, 0 = it did not
    seed: int                # for attacks that need their own random numbers

    # Only for attacks that train shadow models (LiRA). The shadow models may train on a
    # "pool" of windows that contains every attack window.
    pool_size: int = 0
    pool_index: np.ndarray = None    # position of each attack window in the pool
    train_shadow: Callable = None    # train_shadow(in_pool, seed) trains a model like the target
                                     # on the pool windows where in_pool is True, and returns
                                     # its logits on the attack windows


def pick_balanced(n_members, n_nonmembers, rng):
    """Indices of equally many members and non-members, chosen at random."""
    k = min(n_members, n_nonmembers)
    return rng.choice(n_members, k, replace=False), rng.choice(n_nonmembers, k, replace=False)


def run_attacks(names, data, options=None):
    """Run each attack in `names` on `data`.

    `options` maps an attack name to keyword arguments for its scores(), for example
    {"lira": {"n_shadows": 16}}. Returns {"n_attack": ..., "<name>_auc": ..., "<name>_tpr": ...}.
    If an attack cannot score some windows (NaN), they are left out and
    "<name>_n_scored" says how many were scored.
    """
    options = options or {}
    result = {"n_attack": len(data.is_member)}
    for name in names:
        scores = np.asarray(ATTACKS[name].scores(data, **options.get(name, {})), dtype=np.float64)
        scored = np.isfinite(scores)
        is_member = data.is_member[scored]
        result[f"{name}_auc"] = float(roc_auc_score(is_member, scores[scored]))
        result[f"{name}_tpr"] = tpr_at_fpr(is_member, scores[scored])
        if not scored.all():
            result[f"{name}_n_scored"] = int(scored.sum())
    return result


def tpr_at_fpr(is_member, scores, target_fpr=config.TARGET_FPR):
    """True positive rate at a fixed false positive rate, read off the ROC curve."""
    fpr, tpr, _ = roc_curve(is_member, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def resolve_names(names):
    """Check attack names from the command line. "all" means every attack."""
    if "all" in names:
        return list(ATTACKS)
    unknown = [name for name in names if name not in ATTACKS]
    if unknown:
        raise ValueError(f"Unknown attack(s) {unknown}. Available: {', '.join(ATTACKS)}, all")
    return list(names)
