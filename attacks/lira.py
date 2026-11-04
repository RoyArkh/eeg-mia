"""LiRA, offline version (Carlini et al. 2022; HEALTHINF §3.4, Eq. 6).

Idea: compare the target model's output on a window with the outputs of models that
did NOT train on that window. A window that every model finds easy then no longer looks
like a member just because its loss is low. This per-window calibration is why LiRA can
differ from the LOSS attack, while the confidence and entropy attacks cannot.

1. Train `n_shadows` shadow models. Each one trains exactly like the target model (same
   model, epochs and early stopping on the val windows) on a random half of the pool.
   The pool is the target's train windows plus the test windows, so it holds every
   attack window, and each attack window is left out of about half the shadow models.
2. For each attack window, take its statistic from the shadow models that did not train
   on it ("OUT" models) and fit a Gaussian: mean and standard deviation.
3. score = how many standard deviations the target model's statistic lies on the
   "member" side of that mean.

Windows left out of fewer than 2 shadow models get no score (NaN) and are not counted.

Statistic (`statistic=`):
- "logit" (default): the logit-scaled confidence log(p / (1 - p)) of the true class, as
  in Carlini et al. and HEALTHINF Eq. 6. For a one-logit model this is simply the logit,
  with the sign flipped for rest windows. It is close to Gaussian, which LiRA assumes.
- "loss": the BCE loss, which is what mia_bci/LiRA.ipynb used.
The score is the z-score. The paper applies the Gaussian CDF to it, which keeps the
order of the windows, so AUC and TPR are the same.

Differences from mia_bci/LiRA.ipynb (cell 11):
- There, a separate pool of 2,400 windows from participants 1-6 was built, the target
  was trained on a random half of it for 30 epochs without early stopping, and the
  members were that half. Here the target and the attack windows are the same as for
  the other attacks, so LiRA can be compared with them directly.
- At stride < 275 neighbouring windows overlap, and the notebook's random halves put
  overlapping windows on both sides of the member/non-member line. Here the target's
  members (train) and non-members (test) are separated by the time gap, as for every
  attack.

Known limitation: the shadow models' random halves are still drawn window by window.
At stride < 275 an "OUT" shadow model has usually trained on overlapping neighbours of
the window, so it is not fully OUT. This weakens the calibration at high overlap.
Drawing the halves in blocks of neighbouring windows would fix it.

Cost: n_shadows extra trainings per target model.
"""

import time

import numpy as np

N_SHADOWS = 64       # the paper's setting
STATISTIC = "logit"


def scores(data, n_shadows=N_SHADOWS, statistic=STATISTIC, verbose=True):
    if data.train_shadow is None:
        raise ValueError("LiRA needs shadow models, but AttackData.train_shadow is not set")
    if statistic not in ("logit", "loss"):
        raise ValueError(f"statistic must be 'logit' or 'loss', not {statistic!r}")

    # out_values[s, i]: statistic of shadow model s on attack window i, NaN if s trained on i
    out_values = np.full((n_shadows, len(data.logits)), np.nan)
    start_time = time.time()
    for s in range(n_shadows):
        shadow_seed = data.seed + 1 + s
        in_pool = np.random.default_rng(shadow_seed).random(data.pool_size) < 0.5
        shadow_logits = data.train_shadow(in_pool, shadow_seed)
        is_out = ~in_pool[data.pool_index]
        out_values[s, is_out] = window_statistic(shadow_logits, data.y, statistic)[is_out]
        if verbose and ((s + 1) % 8 == 0 or s + 1 == n_shadows):
            print(f"    LiRA: {s + 1}/{n_shadows} shadow models trained "
                  f"({time.time() - start_time:.0f} s)", flush=True)

    n_out = np.sum(~np.isnan(out_values), axis=0)
    usable = n_out >= 2
    mean = np.full(len(data.logits), np.nan)
    std = np.full(len(data.logits), np.nan)
    mean[usable] = np.nanmean(out_values[:, usable], axis=0)
    std[usable] = np.nanstd(out_values[:, usable], axis=0) + 1e-6

    z = (window_statistic(data.logits, data.y, statistic) - mean) / std
    # members have a HIGHER logit statistic but a LOWER loss than OUT models
    return z if statistic == "logit" else -z


def window_statistic(logits, y, statistic):
    true_class_logit = np.where(y == 1, logits, -logits).astype(np.float64)
    if statistic == "logit":
        return true_class_logit
    return np.logaddexp(0.0, -true_class_logit)   # BCE loss
