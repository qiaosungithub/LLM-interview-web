# expect: WA
# Uses each block's own max as the new reference instead of max(running m, block max); exp(m_old - m_new) overflows when a later block scores far lower.
import numpy as np


def _split_heads(M, h):
    T, d = M.shape
    return M.reshape(T, h, d // h).transpose(1, 0, 2)          # (h, T, d_h)


def causal_mha(x, Wq, Wk, Wv, Wo, n_heads):
    T, d = x.shape
    if n_heads <= 0 or d % n_heads:
        raise ValueError("d must be divisible by n_heads")
    dh = d // n_heads
    Q, K, V = (_split_heads(x @ W, n_heads) for W in (Wq, Wk, Wv))
    S = np.einsum("htc,hsc->hts", Q, K) / np.sqrt(dh)
    S = np.where(np.tril(np.ones((T, T), bool)), S, -np.inf)
    S -= S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=-1, keepdims=True)
    out = np.einsum("hts,hsc->htc", P, V)                     # (h, T, dh)
    return out.transpose(1, 0, 2).reshape(T, d) @ Wo


class KVCache:
    def __init__(self, n_heads, d_head):
        self.n_heads, self.d_head = int(n_heads), int(d_head)
        self._k, self._v = [], []

    def append(self, k, v):
        k, v = np.asarray(k, dtype=float), np.asarray(v, dtype=float)
        want = (self.n_heads, self.d_head)
        if k.shape != want or v.shape != want:
            raise ValueError(f"expected {want}, got {k.shape} and {v.shape}")
        self._k.append(k.copy()); self._v.append(v.copy())

    def __len__(self):
        return len(self._k)

    def keys(self):
        return np.stack(self._k, axis=1)     # (h, t, dh)

    def values(self):
        return np.stack(self._v, axis=1)


def decode_step(x_t, cache, Wq, Wk, Wv, Wo, n_heads):
    d = x_t.shape[0]
    dh = d // n_heads
    q = (x_t @ Wq).reshape(n_heads, dh)
    cache.append((x_t @ Wk).reshape(n_heads, dh), (x_t @ Wv).reshape(n_heads, dh))
    K, V = cache.keys(), cache.values()
    s = np.einsum("hc,htc->ht", q, K) / np.sqrt(dh)
    w = np.exp(s - s.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    o = np.einsum("ht,htc->hc", w, V).reshape(d)
    return o @ Wo


def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, batch, bytes_per_elem):
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * bytes_per_elem


def attention_blockwise(Q, K, V, block):
    Tq, dh = Q.shape
    Tk = K.shape[0]
    scale = 1.0 / np.sqrt(dh)
    m = np.full(Tq, -np.inf)
    l = np.zeros(Tq)
    o = np.zeros((Tq, V.shape[1]))
    for a in range(0, Tk, block):
        Kb, Vb = K[a:a + block], V[a:a + block]
        S = np.einsum("qc,kc->qk", Q, Kb) * scale
        m_new = S.max(axis=1)
        corr = np.exp(m - m_new)          # exp(-inf) = 0 on the first block
        P = np.exp(S - m_new[:, None])
        l = l * corr + P.sum(axis=1)
        o = o * corr[:, None] + P @ Vb
        m = m_new
    return o / l[:, None]
