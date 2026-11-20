# expect: WA
# Uses (1 - p_t)^(gamma - 1) in front of the bracket, mixing in the CE-ratio formula.
import numpy as np


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    # NOTE: log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x)) = -logaddexp(0, -x); this never
    # forms exp of a large positive number, unlike np.log(1 / (1 + np.exp(-x))) computed directly.
    return -np.logaddexp(0.0, -x)


def _focal_terms(z: np.ndarray, y: np.ndarray, alpha: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-example FL and per-example d(FL)/dz, stable for every finite z. Shared by Part 2's
    focal_loss_and_grad and Part 3's population- and dataset-level measurements."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s = 2.0 * y - 1.0                                        # +1 where y == 1, -1 where y == 0
    log_pt = _log_sigmoid(s * z)                               # log(p_t), stable for every finite z
    p_t = np.exp(log_pt)                                        # NOTE: from the same stable log -- never "1 - (1 - p_t)"
    one_minus_pt = np.exp(_log_sigmoid(-s * z))                   # 1 - p_t, its own stable computation
    alpha_t = np.where(y == 1.0, alpha, 1.0 - alpha)
    loss = -alpha_t * one_minus_pt ** gamma * log_pt
    # NOTE: Part 1's closed form, built only from log_pt / p_t / one_minus_pt above -- never
    # recomputes p_t or 1 - p_t by subtracting one from the other.
    grad = alpha_t * s * one_minus_pt ** (gamma - 1) * (gamma * p_t * log_pt - one_minus_pt)
    return loss, grad


def focal_loss_and_grad(z: np.ndarray, y: np.ndarray, alpha: float = 0.25,
                         gamma: float = 2.0) -> tuple[float, np.ndarray]:
    loss, grad = _focal_terms(z, y, alpha, gamma)
    return float(loss.mean()), grad / len(grad)          # NOTE: mean loss's gradient -- divide by N, not the sum's
