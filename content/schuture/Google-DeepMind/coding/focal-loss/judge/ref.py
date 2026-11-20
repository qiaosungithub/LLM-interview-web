"""Expected values for focal_loss_and_grad, computed with mpmath at 40 significant digits straight from the
statement's definitions (p_t = sigma(s z), FL = -alpha_t (1 - p_t)^gamma log p_t) and Part 1's closed-form
derivative; rounding to float64 happens only at the end, so every expected value is correctly rounded."""
import mpmath
import numpy as np

mpmath.mp.dps = 40


def terms(z, y, alpha, gamma):
    """Per-example FL and dFL/dz as mpmath numbers."""
    z, s = mpmath.mpf(float(z)), (1 if y == 1.0 else -1)
    x = s * z
    log_pt = -mpmath.log1p(mpmath.exp(-x))
    p_t = mpmath.exp(log_pt)
    one_minus = mpmath.exp(-mpmath.log1p(mpmath.exp(x)))        # sigma(-x) = 1 - p_t
    a_t = mpmath.mpf(alpha) if y == 1.0 else 1 - mpmath.mpf(alpha)
    g = mpmath.mpf(gamma)
    mod = one_minus ** g
    return -a_t * mod * log_pt, a_t * s * mod * (g * p_t * log_pt - one_minus)


def focal(z, y, alpha=0.25, gamma=2.0):
    """(mean loss, gradient of the mean loss) as float64, like the statement's focal_loss_and_grad."""
    pairs = [terms(a, b, alpha, gamma) for a, b in zip(np.asarray(z, dtype=float), np.asarray(y, dtype=float))]
    n = len(pairs)
    return float(mpmath.fsum(p[0] for p in pairs) / n), np.array([float(p[1] / n) for p in pairs])
