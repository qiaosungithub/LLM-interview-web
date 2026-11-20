# expect: WA
# Converts the whole input to float64 first, a length-N copy whenever the logits are float32.
import numpy as np


def softmax_entropy(logits):
    x = np.asarray(logits, dtype=np.float64)
    exp_x = np.exp(x)                       # NOTE: np.exp(1000.0) is inf, with a RuntimeWarning
                                             #       "overflow encountered in exp"
    p = exp_x / np.sum(exp_x)               # NOTE: once exp_x holds an inf, p is inf / inf = nan
                                             #       ("invalid value encountered in divide")
    return float(-np.sum(p * np.log(p)))    # NOTE: even without overflow, a p_i that underflows to
                                             #       exactly 0.0 makes log(p_i) = -inf ("divide by zero
                                             #       encountered in log"), and 0.0 * -inf is nan too
                                             #       ("invalid value encountered in multiply")


def log_softmax_stable(logits):
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x)
    shifted = x - m                          # NOTE: shifted <= 0 everywhere, so np.exp below never overflows
    total = np.exp(shifted).sum()
    return shifted - np.log(total)           # NOTE: never calls log on a value that could be 0


def softmax_entropy_stable(logits):
    log_p = log_softmax_stable(logits)
    p = np.exp(log_p)                        # NOTE: p can still underflow to exactly 0.0 here, but log_p
                                              #       stays finite, so p * log_p below is 0.0 * finite, not nan
    return float(-np.sum(p * log_p))


def _chunks(x, block_size):
    for start in range(0, x.shape[0], block_size):
        # NOTE: convert to float64 one slice at a time. Converting the whole array up front would build a
        #       length-N copy whenever logits is not float64 already (float32 logits, for instance)
        yield np.asarray(x[start:start + block_size], dtype=np.float64)   # at most block_size elements live
                                                                            # at once; no size-N array is built


def softmax_entropy_blockwise(logits, block_size=2):
    x = np.asarray(logits, dtype=np.float64)

    m = -np.inf
    for chunk in _chunks(x, block_size):          # pass 1: the global maximum
        m = max(m, float(np.max(chunk)))

    s = u = 0.0
    for chunk in _chunks(x, block_size):           # pass 2: s and u together, given m
        shifted = chunk - m
        w = np.exp(shifted)
        s += float(np.sum(w))
        u += float(np.sum(w * shifted))

    return float(np.log(s) - u / s)


def _centered_stats(x):
    m = float(np.max(x))
    shifted = x - m
    w = np.exp(shifted)
    return m, float(np.sum(w)), float(np.sum(w * shifted))


def _fold(acc, chunk):
    m0, s0, u0 = acc
    m1, s1, u1 = chunk
    if s0 == 0.0:                 # NOTE: acc is still the empty accumulator (m0 = -inf). Without this guard,
        return chunk               #       exp(m0 - m) is 0.0 and s0 * (m0 - m) is 0.0 * -inf = nan below
    m = max(m0, m1)
    a, b = np.exp(m0 - m), np.exp(m1 - m)
    s = s0 * a + s1 * b
    u = (u0 + s0 * (m0 - m)) * a + (u1 + s1 * (m1 - m)) * b
    return m, s, u


def softmax_entropy_online(blocks):
    acc = (-np.inf, 0.0, 0.0)
    for block in blocks:
        acc = _fold(acc, _centered_stats(np.asarray(block, dtype=np.float64)))
    m, s, u = acc
    return float(np.log(s) - u / s)
