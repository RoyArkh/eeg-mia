"""Confidence attack, true-class version: a high probability of the TRUE class means "member".

score = p(true class)

This is what mia_bci/attacks.py computed. The loss is -log p(true class), so this ranks
windows exactly like the LOSS attack and gives the same AUC and TPR. The HEALTHINF
paper's Eq. 4 uses the top-class probability instead: see max_confidence.py.
"""

import numpy as np
from scipy.special import expit


def scores(data):
    true_class_logit = np.where(data.y == 1, data.logits, -data.logits).astype(np.float64)
    return expit(true_class_logit)
