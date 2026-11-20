import numpy as np


def causal_mha(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
               n_heads: int) -> np.ndarray:
    """x: (T, d). Wq, Wk, Wv, Wo: (d, d). d must be divisible by n_heads; raises ValueError
    otherwise. Returns (T, d)."""
    raise NotImplementedError


class KVCache:
    def __init__(self, n_heads: int, d_head: int) -> None:
        """An empty cache of keys and values for n_heads heads of size d_head."""
        raise NotImplementedError

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        """k, v: (n_heads, d_head), one new position's keys and values. Raises ValueError
        if either does not have that shape."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of positions appended so far."""
        raise NotImplementedError


def decode_step(x_t: np.ndarray, cache: KVCache, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                Wo: np.ndarray, n_heads: int) -> np.ndarray:
    """x_t: (d,), one new position's input. Appends this position's keys and values to cache,
    then returns the (d,) output row for this position, attending over every position cache
    now holds (including this one)."""
    raise NotImplementedError


def kv_cache_bytes(n_layers: int, n_kv_heads: int, d_head: int, seq_len: int, batch: int,
                   bytes_per_elem: int) -> int:
    """Total bytes of every layer's KV cache for `batch` sequences of length seq_len, with
    n_kv_heads key/value heads of size d_head per layer and bytes_per_elem bytes per stored
    number."""
    raise NotImplementedError


def attention_blockwise(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block: int) -> np.ndarray:
    """Q: (T_q, d_h). K, V: (T_k, d_h). Returns softmax(Q @ K.T / sqrt(d_h)) @ V, computed
    `block` positions of K/V at a time, for any block >= 1."""
    raise NotImplementedError
