# expect: WA
# append validates the shape of k but not of v.
import numpy as np


def causal_mha(x, Wq, Wk, Wv, Wo, n_heads):
    T, d = x.shape
    if d % n_heads != 0:
        raise ValueError(f"d={d} is not divisible by n_heads={n_heads}")
    d_h = d // n_heads

    Q, K, V = x @ Wq, x @ Wk, x @ Wv                                 # each (T, d)
    Qh = Q.reshape(T, n_heads, d_h).transpose(1, 0, 2)                # (h, T, d_h)
    Kh = K.reshape(T, n_heads, d_h).transpose(1, 0, 2)                # (h, T, d_h)
    Vh = V.reshape(T, n_heads, d_h).transpose(1, 0, 2)                # (h, T, d_h)

    # NOTE: scale by sqrt(d_h), the per-head dimension -- not sqrt(d), which equals d_h only when n_heads == 1.
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(d_h)                 # (h, T, T)
    visible = np.tril(np.ones((T, T), dtype=bool))                     # visible[t, s]: position t may see key s
    # NOTE: mask before the softmax max/exp, not after normalising -- zeroing weights post hoc would
    # leave the visible weights summing to less than 1 instead of exactly 1.
    scores = np.where(visible, scores, -np.inf)

    m = scores.max(axis=-1, keepdims=True)                             # (h, T, 1); never -inf, t always sees itself
    weights = np.exp(scores - m)
    weights /= weights.sum(axis=-1, keepdims=True)

    head_out = weights @ Vh                                             # (h, T, d_h)
    merged = head_out.transpose(1, 0, 2).reshape(T, d)                  # (T, d)
    return merged @ Wo


class KVCache:
    def __init__(self, n_heads, d_head):
        self.n_heads = n_heads
        self.d_head = d_head
        self._keys = []            # one (n_heads, d_head) entry per position appended so far
        self._values = []

    def append(self, k, v):
        shape = (self.n_heads, self.d_head)
        if k.shape != shape:
            raise ValueError(f"expected shape {shape}, got k={k.shape}, v={v.shape}")
        self._keys.append(k)
        self._values.append(v)

    def __len__(self):
        return len(self._keys)

    def arrays(self):
        # NOTE: stacking here costs O(len(self)); a production cache pre-allocates a fixed-size
        # buffer and writes into it instead, so a single step costs O(d_head), not O(t) (Follow-ups).
        K = np.stack(self._keys, axis=1)          # (n_heads, t, d_head)
        V = np.stack(self._values, axis=1)        # (n_heads, t, d_head)
        return K, V


def decode_step(x_t, cache, Wq, Wk, Wv, Wo, n_heads):
    d = x_t.shape[0]
    d_head = d // n_heads
    q_t = (x_t @ Wq).reshape(n_heads, d_head)
    k_t = (x_t @ Wk).reshape(n_heads, d_head)
    v_t = (x_t @ Wv).reshape(n_heads, d_head)
    cache.append(k_t, v_t)                          # NOTE: append before attending, so position t attends to itself

    K, V = cache.arrays()                             # (n_heads, t_so_far, d_head)
    scores = (q_t[:, None, :] * K).sum(axis=-1) / np.sqrt(d_head)   # (n_heads, t_so_far)
    m = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - m)
    weights /= weights.sum(axis=-1, keepdims=True)
    head_out = (weights[:, :, None] * V).sum(axis=1)    # (n_heads, d_head)
    return head_out.reshape(d) @ Wo


def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, batch, bytes_per_elem):
    return n_layers * 2 * n_kv_heads * d_head * seq_len * batch * bytes_per_elem


def attention_blockwise(Q, K, V, block):
    T_q, d_h = Q.shape
    T_k = K.shape[0]
    scale = np.sqrt(d_h)

    m = np.full(T_q, -np.inf)                # running max score, per query
    l = np.zeros(T_q)                         # running normaliser, per query
    o = np.zeros((T_q, d_h))                  # running unnormalised output, per query

    for start in range(0, T_k, block):
        end = min(start + block, T_k)                          # NOTE: the last block may be shorter than `block`
        s = Q @ K[start:end].T / scale                          # (T_q, end - start); one block of scores at a time
        m_new = np.maximum(m, s.max(axis=1))
        # NOTE: m starts at -inf, but m_new = max(m, block_max) is finite already at the first block (a
        # block is never empty), so m - m_new is -inf - finite = -inf, not -inf - (-inf) = nan -- exp(-inf)
        # is a clean 0, so no separate guard is needed here.
        correction = np.exp(m - m_new)                           # rescales the accumulators to the new max
        p = np.exp(s - m_new[:, None])
        l = correction * l + p.sum(axis=1)
        o = correction[:, None] * o + p @ V[start:end]
        m = m_new

    return o / l[:, None]
