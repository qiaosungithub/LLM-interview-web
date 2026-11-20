# expect: WA
# Masks with -1e9 instead of -inf and drops the guards: a fully masked query gets uniform weights.
import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = np.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d)            # (B, T_q, T_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    m = np.max(scores, axis=-1, keepdims=True)                       # (B, T_q, 1)
    exp_scores = np.exp(scores - m)                                  # (B, T_q, T_k)
    denom = np.sum(exp_scores, axis=-1, keepdims=True)                # (B, T_q, 1)
    weights = exp_scores / denom
    return np.einsum('bqk,bkd->bqd', weights, V)


def multi_head_self_attention(X, W_q, W_k, W_v, W_o, n_heads):
    B, T, d_model = X.shape
    d_head = d_model // n_heads

    Q = np.einsum('btd,de->bte', X, W_q)                              # (B, T, d_model)
    K = np.einsum('btd,de->bte', X, W_k)                              # (B, T, d_model)
    V = np.einsum('btd,de->bte', X, W_v)                              # (B, T, d_model)

    Qh = np.einsum('bthe->bhte', Q.reshape(B, T, n_heads, d_head))    # (B, H, T, d_head)
    Kh = np.einsum('bthe->bhte', K.reshape(B, T, n_heads, d_head))    # (B, H, T, d_head)
    Vh = np.einsum('bthe->bhte', V.reshape(B, T, n_heads, d_head))    # (B, H, T, d_head)

    scores = np.einsum('bhqe,bhke->bhqk', Qh, Kh) / np.sqrt(d_head)   # (B, H, T, T)
    visible = np.tril(np.ones((T, T), dtype=bool))                    # (T, T); visible[i, j] = query i sees key j
    scores = np.where(visible, scores, -np.inf)

    m = np.max(scores, axis=-1, keepdims=True)                        # (B, H, T, 1); never -inf: i sees itself
    exp_scores = np.exp(scores - m)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (B, H, T, T)

    head_out = np.einsum('bhqk,bhke->bhqe', weights, Vh)               # (B, H, T, d_head)
    merged = np.einsum('bhte->bthe', head_out).reshape(B, T, d_model)  # (B, T, d_model)
    return np.einsum('btd,de->bte', merged, W_o)                       # (B, T, d_model)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)                             # NOTE: was axis=0
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)               # NOTE: was missing keepdims=True
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def causal_self_attention(x, W_q, W_k, W_v):
    T, d = x.shape[1], x.shape[2]
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d)                 # NOTE: was // np.sqrt(d)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-1, keepdims=True)                             # NOTE: was axis=-2
    exp_scores = np.exp(scores - m)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return probs @ V


def transformer_block_forward(x, gamma, beta, W_q, W_k, W_v):
    return x + causal_self_attention(layer_norm(x, gamma, beta), W_q, W_k, W_v)
