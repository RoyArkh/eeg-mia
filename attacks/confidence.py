"""Confidence attack, true-class version: a high probability of the TRUE class means "member".

score = p(true class)

This is what mia_bci/attacks.py computed. The loss is -log p(true class), so this ranks
windows like the LOSS attack and gives the same AUC and TPR, except for rounding: when the
true-class logit is above about 37, p rounds to exactly 1.0 and those windows tie. In the
1,526 runs of 2026-11-04 this changed the AUC by at most 0.0003. The HEALTHINF paper's
Eq. 4 uses the top-class probability instead: see max_confidence.py.
"""

import numpy as np
from scipy.special import expit


def scores(data):
    true_class_logit = np.where(data.y == 1, data.logits, -data.logits).astype(np.float64)
    return expit(true_class_logit)
