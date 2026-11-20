"""Independent restatement of the problem, used by tests.py for expected values and inputs.

Every quantity is computed position by position in float64, straight from the statement's definitions:
log softmax(z)_v = z_v - (max z + log sum_k exp(z_k - max z)), and the target distribution
p = (1 - eps) one_hot(y) + eps / V. Masked positions are skipped without looking at their target.
"""
import math

import numpy as np


def log_softmax_row(z):
    m = max(z)
    lse = m + math.log(math.fsum(math.exp(v - m) for v in z))
    return [v - lse for v in z]


def target_row(y, V, eps):
    p = [eps / V] * V
    p[y] += 1.0 - eps
    return p


def position_losses(logits, targets, mask, eps):
    """(sum of the unmasked per-position losses, number of unmasked positions)."""
    B, T, V = logits.shape
    total, count = [], 0
    for b in range(B):
        for t in range(T):
            if not mask[b, t]:
                continue
            logq = log_softmax_row([float(v) for v in logits[b, t]])
            p = target_row(int(targets[b, t]), V, eps)
            total.append(-math.fsum(pv * lv for pv, lv in zip(p, logq)))
            count += 1
    return math.fsum(total), count


def smoothed(logits, targets, mask, eps):
    total, count = position_losses(logits, targets, mask, eps)
    return total / count if count else 0.0


def masked(logits, targets, mask):
    return smoothed(logits, targets, mask, 0.0)


def plain(logits, targets):
    return masked(logits, targets, np.ones(targets.shape, dtype=bool))


def grad(logits, targets, mask, eps):
    """d(mean loss)/d(logits) = (softmax(z) - p) / N at each unmasked position, zero elsewhere."""
    B, T, V = logits.shape
    count = int(np.count_nonzero(mask))
    out = np.zeros((B, T, V))
    if count == 0:
        return out
    for b in range(B):
        for t in range(T):
            if not mask[b, t]:
                continue
            q = [math.exp(lv) for lv in log_softmax_row([float(v) for v in logits[b, t]])]
            p = target_row(int(targets[b, t]), V, eps)
            out[b, t] = [(qv - pv) / count for qv, pv in zip(q, p)]
    return out


# ---------------------------------------------------------------- inputs

def random_batch(gen, B, T, V, scale=4.0, p_keep=0.6, bool_mask=False, junk=True):
    """Logits rounded to 2 decimals (so a failing call stays short enough to paste), targets in [0, V), a mask
    with at least one kept position, and targets at masked positions replaced by out-of-range junk."""
    logits = np.round(gen.normal(scale=scale, size=(B, T, V)), 2) + 0.0
    targets = gen.integers(0, V, size=(B, T))
    mask = (gen.random(size=(B, T)) < p_keep).astype(np.int64)
    if not mask.any():
        mask[gen.integers(0, B), gen.integers(0, T)] = 1
    if junk:
        bad = gen.choice(np.array([-1, -100, -999, V, V + 5, 10 ** 9]), size=(B, T))
        targets = np.where(mask == 1, targets, bad)
    return logits, targets, (mask.astype(bool) if bool_mask else mask)


def clean_targets(targets, mask):
    return np.where(np.asarray(mask).astype(bool), targets, 0)
