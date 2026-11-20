"""Helpers for tests.py: the statement's formulas as explicit index loops (no broadcasting), and the variants that
the tests use to say what a wrong answer seems to do."""
import math

import numpy as np


def softmax_rows(scores):
    """Stable softmax of a list of finite floats."""
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    total = sum(e)
    return [x / total for x in e]


# ---------------------------------------------------------------- Part 1
def attention(Q, K, V, mask=None):
    """Part 1: a fully masked query gets the zero vector."""
    B, Tq, d = Q.shape
    Tk, dv = K.shape[1], V.shape[2]
    out = np.zeros((B, Tq, dv))
    for b in range(B):
        for i in range(Tq):
            visible = [j for j in range(Tk) if mask is None or mask[b, i, j]]
            if not visible:
                continue
            w = softmax_rows([float(np.dot(Q[b, i], K[b, j])) / math.sqrt(d) for j in visible])
            for wj, j in zip(w, visible):
                out[b, i] += wj * V[b, j]
    return out


# ---------------------------------------------------------------- Part 2
def multi_head(X, W_q, W_k, W_v, W_o, n_heads, strided=False, scale_dim=None, causal=True):
    """Part 2. Head h uses columns h * d_head + e. Variants: strided=True gives head h the columns e * H + h,
    scale_dim divides the scores by sqrt(scale_dim) instead of sqrt(d_head), causal=False drops the mask."""
    B, T, dm = X.shape
    dh = dm // n_heads
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    concat = np.zeros((B, T, dm))
    for b in range(B):
        for h in range(n_heads):
            cols = [e * n_heads + h if strided else h * dh + e for e in range(dh)]
            for i in range(T):
                keys = range(i + 1) if causal else range(T)
                w = softmax_rows([float(np.dot(Q[b, i, cols], K[b, j, cols])) / math.sqrt(scale_dim or dh)
                                  for j in keys])
                for wj, j in zip(w, keys):
                    concat[b, i, cols] += wj * V[b, j, cols]
    return concat @ W_o


# ---------------------------------------------------------------- Part 3
def layer_norm(x, gamma, beta, eps=1e-5):
    B, T, d = x.shape
    out = np.zeros_like(x)
    for b in range(B):
        for t in range(T):
            row = [float(v) for v in x[b, t]]
            mu = sum(row) / d
            var = sum((v - mu) ** 2 for v in row) / d
            for k in range(d):
                out[b, t, k] = (row[k] - mu) / math.sqrt(var + eps) * gamma[k] + beta[k]
    return out


def causal_attention(x, W_q, W_k, W_v):
    """Part 2 with H = 1."""
    return multi_head(x, W_q, W_k, W_v, np.eye(x.shape[2]), 1)


def block(x, gamma, beta, W_q, W_k, W_v):
    return x + causal_attention(layer_norm(x, gamma, beta), W_q, W_k, W_v)


# The statement's buggy lines, one at a time, to recognise an answer that kept one of them.
def layer_norm_batch_mean(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def causal_attention_buggy(x, W_q, W_k, W_v, floor=False, max_axis=-1):
    T, d = x.shape[1], x.shape[2]
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    raw = Q @ np.swapaxes(K, -1, -2)
    scores = raw // np.sqrt(d) if floor else raw / np.sqrt(d)
    scores = np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -np.inf)
    exp_scores = np.exp(scores - scores.max(axis=max_axis, keepdims=True))
    return (exp_scores / exp_scores.sum(axis=-1, keepdims=True)) @ V
