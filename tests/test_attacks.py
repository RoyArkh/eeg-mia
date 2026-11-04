"""Checks for the attacks, on made-up logits where the right answer is known.

Run with: uv run pytest
"""

import numpy as np
import pytest

import attacks

SIMPLE_ATTACKS = [name for name in attacks.ATTACKS if name != "lira"]


def make_data(true_class_logit, y, n_members):
    """AttackData from true-class logits (members first)."""
    logits = np.where(y == 1, true_class_logit, -true_class_logit)
    is_member = np.r_[np.ones(n_members), np.zeros(len(y) - n_members)]
    return attacks.AttackData(logits=logits, y=y, is_member=is_member, seed=0)


def test_unknown_attack_is_an_error():
    with pytest.raises(ValueError, match="Unknown attack"):
        attacks.resolve_names(["loss", "no_such_attack"])


def test_all_means_every_attack():
    assert attacks.resolve_names(["all"]) == list(attacks.ATTACKS)


def test_pick_balanced_takes_equally_many():
    members, nonmembers = attacks.pick_balanced(250, 46, np.random.default_rng(0))
    assert len(members) == len(nonmembers) == 46
    assert len(set(members)) == 46 and members.max() < 250


@pytest.mark.parametrize("name", SIMPLE_ATTACKS)
def test_perfect_separation(name):
    # members: confidently right; non-members: unsure (|logit| <= 1)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000).astype(float)
    true_class_logit = np.r_[rng.uniform(3, 5, 500), rng.uniform(-1, 1, 500)]
    result = attacks.run_attacks([name], make_data(true_class_logit, y, 500))
    assert result[f"{name}_auc"] == 1.0
    assert result[f"{name}_tpr"] == 1.0


@pytest.mark.parametrize("name", SIMPLE_ATTACKS)
def test_no_leak_is_near_chance(name):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000).astype(float)
    true_class_logit = rng.normal(0, 2, 20_000)   # members and non-members look the same
    result = attacks.run_attacks([name], make_data(true_class_logit, y, 10_000))
    assert abs(result[f"{name}_auc"] - 0.5) < 0.02


def test_loss_confidence_and_modified_entropy_agree_on_a_binary_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000).astype(float)
    true_class_logit = np.r_[rng.normal(1, 2, 1000), rng.normal(0, 2, 1000)]
    result = attacks.run_attacks(SIMPLE_ATTACKS, make_data(true_class_logit, y, 1000))
    assert result["loss_auc"] == result["confidence_auc"] == result["modified_entropy_auc"]
    assert result["loss_tpr"] == result["confidence_tpr"] == result["modified_entropy_tpr"]
    assert result["max_confidence_auc"] != result["loss_auc"]   # ignores the label, so differs


def make_lira_data(n_windows=400, member_boost=1.0, noise=0.3, difficulty_spread=3.0):
    """Made-up target and shadow models where LiRA should clearly beat LOSS.

    Each window has its own difficulty (its usual true-class logit). Training on a window
    raises its logit by `member_boost`. The difficulty varies much more between windows
    than the boost, so the raw loss says little about membership, but comparing each
    window with its own OUT logits does.
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n_windows).astype(float)
    sign = np.where(y == 1, 1.0, -1.0)
    difficulty = rng.normal(0, difficulty_spread, n_windows)
    is_member = np.r_[np.ones(n_windows // 2), np.zeros(n_windows - n_windows // 2)]

    def train_shadow(in_pool, seed):
        shadow_rng = np.random.default_rng(seed)
        return sign * (difficulty + member_boost * in_pool + shadow_rng.normal(0, noise, n_windows))

    target_logits = sign * (difficulty + member_boost * is_member + rng.normal(0, noise, n_windows))
    return attacks.AttackData(logits=target_logits, y=y, is_member=is_member, seed=0,
                              pool_size=n_windows, pool_index=np.arange(n_windows),
                              train_shadow=train_shadow)


@pytest.mark.parametrize("statistic", ["logit", "loss"])
def test_lira_calibrates_per_window(statistic):
    data = make_lira_data()
    options = {"lira": {"n_shadows": 16, "statistic": statistic, "verbose": False}}
    result = attacks.run_attacks(["loss", "lira"], data, options)
    assert result["loss_auc"] < 0.7
    assert result["lira_auc"] > 0.9


def test_lira_skips_windows_with_too_few_out_models():
    data = make_lira_data()
    result = attacks.run_attacks(["lira"], data, {"lira": {"n_shadows": 2, "verbose": False}})
    assert 0 < result["lira_n_scored"] < result["n_attack"]


def test_lira_without_shadow_models_is_an_error():
    data = make_data(np.zeros(10), np.zeros(10), 5)
    with pytest.raises(ValueError, match="shadow"):
        attacks.run_attacks(["lira"], data)
