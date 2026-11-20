"""Judge-side references for tests.py: the statement's formulas with explicit loops over heads, positions and keys,
a max-shifted softmax and exactly rounded sums (math.fsum)."""
import math

import numpy as np


def softmax_avg(scores, values):
    """sum_j softmax(scores)_j * values[j] for one row of scores; values is a list of equal-length lists."""
    m = max(scores)
    w = [math.exp(s - m) for s in scores]
    total = math.fsum(w)
    return [math.fsum(wj * v[c] for wj, v in zip(w, values)) / total for c in range(len(values[0]))]


def causal_mha_slow(x, Wq, Wk, Wv, Wo, n_heads):
    T, d = x.shape
    d_h = d // n_heads
    Q, K, V = (x @ Wq).tolist(), (x @ Wk).tolist(), (x @ Wv).tolist()
    merged = [[0.0] * d for _ in range(T)]
    for h in range(n_heads):
        cols = range(h * d_h, (h + 1) * d_h)
        for t in range(T):
            scores = [math.fsum(Q[t][c] * K[s][c] for c in cols) / math.sqrt(d_h) for s in range(t + 1)]
            row = softmax_avg(scores, [[V[s][c] for c in cols] for s in range(t + 1)])
            for c, value in zip(cols, row):
                merged[t][c] = value
    return np.array(merged) @ Wo


def attention_slow(Q, K, V):
    d_h = Q.shape[1]
    Kl, Vl = K.tolist(), V.tolist()
    out = []
    for q in Q.tolist():
        scores = [math.fsum(a * b for a, b in zip(q, k)) / math.sqrt(d_h) for k in Kl]
        out.append(softmax_avg(scores, Vl))
    return np.array(out)

