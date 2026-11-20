# expect: WA
# Computes softplus as np.log(1 + np.exp(.)) without log1p; log p_t loses precision once s*z > ~15 and is 0 past 37.
import numpy as np


def focal_loss_and_grad(z, y, alpha=0.25, gamma=2.0):
    z = np.asarray(z, float); y = np.asarray(y, float)
    s = 2 * y - 1
    x = s * z
    log_pt = -np.log(1 + np.exp(-x))
    q = 1 / (1 + np.exp(x))
    pt = 1 / (1 + np.exp(-x))
    a_t = np.where(y == 1, alpha, 1 - alpha)
    m = q ** gamma
    n = len(z)
    return float(np.mean(-a_t * m * log_pt)), a_t * s * m * (gamma * pt * log_pt - q) / n
