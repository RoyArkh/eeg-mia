"""Modified entropy attack (Song & Mittal 2021): a low modified entropy means "member".

With p_y the probability of the true class and p_k the other classes' probabilities:

    Mentr = -(1 - p_y) log(p_y) - sum over k != y of p_k log(1 - p_k)

With two classes the other class has p_k = 1 - p_y, so the two terms are equal:

    Mentr = -2 (1 - p_y) log(p_y)

score = -Mentr. Mentr falls steadily as p_y rises, so on a binary model this ranks
windows like the LOSS attack and gives the same AUC and TPR.
"""

import numpy as np
from scipy.special import expit


def scores(data):
    true_class_logit = np.where(data.y == 1, data.logits, -data.logits).astype(np.float64)
    p_wrong = expit(-true_class_logit)                   # 1 - p_y
    log_p_true = -np.logaddexp(0.0, -true_class_logit)   # log(p_y), computed stably
    return 2.0 * p_wrong * log_p_true                    # = -Mentr
