import numpy as np


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                  mask: np.ndarray | None = None) -> np.ndarray:
    """Q: (B, T_q, d), K: (B, T_k, d), V: (B, T_k, d_v). mask: (B, T_q, T_k) bool or None,
    True where the query may attend to the key. Returns (B, T_q, d_v)."""
    raise NotImplementedError


def multi_head_self_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                               W_o: np.ndarray, n_heads: int) -> np.ndarray:
    """X: (B, T, d_model). W_q, W_k, W_v, W_o: (d_model, d_model). d_model must be divisible by
    n_heads; d_head = d_model // n_heads. Applies a causal mask. Returns (B, T, d_model)."""
    raise NotImplementedError


# Part 3: the code below has four bugs; find and fix them.
def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """x: (B, T, d). gamma, beta: (d,). Normalizes each (batch, time) row over its d features."""
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def causal_self_attention(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> np.ndarray:
    """x: (B, T, d). W_q, W_k, W_v: (d, d). Single-head causal self-attention. Returns (B, T, d)."""
    T, d = x.shape[1], x.shape[2]
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = (Q @ np.swapaxes(K, -1, -2)) // np.sqrt(d)
    visible = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(visible, scores, -np.inf)
    m = scores.max(axis=-2, keepdims=True)
    exp_scores = np.exp(scores - m)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    return probs @ V


def transformer_block_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                               W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> np.ndarray:
    """x: (B, T, d). Pre-norm residual block: x + causal_self_attention(layer_norm(x))."""
    return x + causal_self_attention(layer_norm(x, gamma, beta), W_q, W_k, W_v)
