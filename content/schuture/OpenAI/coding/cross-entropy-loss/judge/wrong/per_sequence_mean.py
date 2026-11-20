# expect: WA
# Wrong: averages each sequence over its own unmasked positions, then averages the sequences, instead of one mean over all unmasked positions.
import numpy as np


def cross_entropy(logits, targets):
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets)
    m = x.max(axis=-1, keepdims=True)                        # NOTE: keepdims=True, else `x - m` below
    shifted = x - m                                           #       broadcasts along the wrong axis when T == V
    log_z = np.log(np.exp(shifted).sum(axis=-1)) + m[..., 0]  # (B, T), always finite: shifted <= 0 everywhere
    picked = np.take_along_axis(x, y[..., None], axis=-1)[..., 0]   # (B, T), logits at the target class
    nll = log_z - picked   # NOTE: this is -[(z_y - m) - log sum_v exp(z_v - m)]; never softmax(x) followed by log(.)
    return float(nll.mean())


def cross_entropy_masked(logits, targets, mask):
    x = np.asarray(logits, dtype=np.float64)
    keep = np.asarray(mask).astype(bool)          # NOTE: mask may arrive as 0/1 ints; x[mask] would then do
                                                    #       FANCY indexing, not boolean masking
    safe_targets = np.where(keep, targets, 0)      # NOTE: dummy class 0 at masked slots, so the gather below
                                                     #       never reads an out-of-range target
    m = x.max(axis=-1, keepdims=True)
    shifted = x - m
    log_z = np.log(np.exp(shifted).sum(axis=-1)) + m[..., 0]
    picked = np.take_along_axis(x, safe_targets[..., None], axis=-1)[..., 0]
    nll = (log_z - picked) * keep                   # zero out masked positions before summing
    count = keep.sum()
    if count == 0:
        return 0.0
    per_seq = keep.sum(axis=1)
    return float((nll.sum(axis=1)[per_seq > 0] / per_seq[per_seq > 0]).mean())  # NOTE: "/" not "//" -- floor division would silently
                                                       #       truncate the loss to an integer


def cross_entropy_smoothed(logits, targets, mask, label_smoothing):
    x = np.asarray(logits, dtype=np.float64)
    V = x.shape[-1]
    keep = np.asarray(mask).astype(bool)
    safe_targets = np.where(keep, targets, 0)

    m = x.max(axis=-1, keepdims=True)
    shifted = x - m
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1))         # (B, T)
    log_z = log_sum_exp + m[..., 0]
    log_q_y = np.take_along_axis(x, safe_targets[..., None], axis=-1)[..., 0] - log_z   # log q at the target
    mean_log_q = shifted.mean(axis=-1) - log_sum_exp           # mean over v of log q_v; no (B, T, V) target tensor built
    eps = label_smoothing
    nll = -(1.0 - eps) * log_q_y - eps * mean_log_q
    nll = nll * keep
    count = keep.sum()
    if count == 0:
        return 0.0
    return float(nll.sum() / count)


def cross_entropy_grad(logits, targets, mask, label_smoothing):
    x = np.asarray(logits, dtype=np.float64)
    V = x.shape[-1]
    keep = np.asarray(mask).astype(bool)
    safe_targets = np.where(keep, targets, 0)
    count = keep.sum()
    if count == 0:
        return np.zeros_like(x)

    m = x.max(axis=-1, keepdims=True)
    q = np.exp(x - m)
    q /= q.sum(axis=-1, keepdims=True)                          # q = softmax(x), shape (B, T, V)

    eps = label_smoothing
    p = np.full_like(x, eps / V)                                # p = target distribution
    np.put_along_axis(p, safe_targets[..., None], (1.0 - eps) + eps / V, axis=-1)
    grad = (q - p) / count                                      # NOTE: divide by count, matching the mean reduction
    grad *= keep[..., None]                                     # zero the gradient row of every masked position
    return grad
