"""LOSS attack (Yeom et al. 2018): a low loss on a window means "member".

score = -(binary cross-entropy of the target model on the window)
"""

import numpy as np


def scores(data):
    # log-odds of the true class: the logit for game windows, minus the logit for rest windows
    true_class_logit = np.where(data.y == 1, data.logits, -data.logits).astype(np.float64)
    loss = np.logaddexp(0.0, -true_class_logit)   # = -log p(true class), computed stably
    return -loss
