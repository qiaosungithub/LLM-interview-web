# expect: WA
# Subtracts the max but takes log(p + 1e-12) instead of the centred log p_i, so every term is off by about 1e-12.
import math

import numpy as np


def softmax_entropy(logits):
    e = np.exp(np.asarray(logits, dtype=np.float64))
    p = e / e.sum()
    return float(-(p * np.log(p)).sum())


def softmax_entropy_stable(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max())
    p = e / e.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))    # no nan when p underflows to 0, but biased by about N * 1e-12


def softmax_entropy_blockwise(logits, block_size=2):
    x = np.asarray(logits)
    m = -math.inf
    for a in range(0, len(x), block_size):
        m = max(m, float(x[a:a + block_size].max()))
    s = u = 0.0
    for a in range(0, len(x), block_size):
        z = x[a:a + block_size].astype(np.float64) - m
        e = np.exp(z)
        s += float(e.sum())
        u += float(e @ z)
    return math.log(s) - u / s


def softmax_entropy_online(blocks):
    m, s, u = None, 0.0, 0.0
    for chunk in blocks:
        c = np.asarray(chunk, dtype=np.float64)
        cm = float(c.max())
        if m is None or cm > m:
            if m is not None:
                d = m - cm
                f = math.exp(d)
                u, s = f * (u + d * s), f * s
            m = cm
        z = c - m
        e = np.exp(z)
        s += float(e.sum())
        u += float(e @ z)
    return math.log(s) - u / s
