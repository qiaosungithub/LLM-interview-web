# expect: WA
# Wrong: label smoothing gives the target class 1 - eps and spreads eps over the other V - 1 classes only.
import numpy as np


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits, targets):
    x = np.asarray(logits, dtype=np.float64)
    logq = log_softmax(x)
    picked = np.take_along_axis(logq, np.asarray(targets)[..., None], axis=-1)[..., 0]
    return float(-picked.mean())


def cross_entropy_masked(logits, targets, mask):
    return cross_entropy_smoothed(logits, targets, mask, 0.0)


def target_dist(targets, keep, V, eps):
    safe = np.where(keep, targets, 0)
    p = np.full(safe.shape + (V,), eps / max(V - 1, 1))
    np.put_along_axis(p, safe[..., None], 1.0 - eps, axis=-1)
    return p


def denominator(keep):
    return keep.sum()


def cross_entropy_smoothed(logits, targets, mask, label_smoothing):
    x = np.asarray(logits, dtype=np.float64)
    keep = np.asarray(mask).astype(bool)
    p = target_dist(np.asarray(targets), keep, x.shape[-1], label_smoothing)
    nll = -(p * log_softmax(x)).sum(axis=-1) * keep
    count = denominator(keep)
    if keep.sum() == 0:
        return 0.0
    return float(nll.sum() / count)


def cross_entropy_grad(logits, targets, mask, label_smoothing):
    x = np.asarray(logits, dtype=np.float64)
    keep = np.asarray(mask).astype(bool)
    if keep.sum() == 0:
        return np.zeros_like(x)
    p = target_dist(np.asarray(targets), keep, x.shape[-1], label_smoothing)
    grad = (softmax(x) - p) / denominator(keep)
    return grad * keep[..., None]
