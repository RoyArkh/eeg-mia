"""Confidence attack, top-class version (HEALTHINF Eq. 4): a high top-class probability means "member".

score = max(p(rest), p(game))

Unlike the other simple attacks, this ignores the true label: a window the model
classifies confidently but WRONGLY also scores high. So it can rank windows differently
from the LOSS attack.
"""

import numpy as np
from scipy.special import expit


def scores(data):
    return expit(np.abs(data.logits.astype(np.float64)))
