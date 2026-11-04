"""Membership inference attacks."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(is_member, scores, target_fpr=0.01):
    """True positive rate at a fixed false positive rate, read off the ROC curve."""
    fpr, tpr, _ = roc_curve(is_member, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def loss_attack(member_losses, nonmember_losses, rng):
    """LOSS attack (Yeom et al. 2018): a lower loss means "member".

    Uses equally many members and non-members (randomly subsampled).
    """
    k = min(len(member_losses), len(nonmember_losses))
    members = rng.choice(member_losses, k, replace=False)
    nonmembers = rng.choice(nonmember_losses, k, replace=False)

    scores = -np.concatenate([members, nonmembers])
    is_member = np.concatenate([np.ones(k), np.zeros(k)])
    return {
        "attack_auc": float(roc_auc_score(is_member, scores)),
        "attack_tpr": tpr_at_fpr(is_member, scores),
        "n_attack": 2 * k,
    }
