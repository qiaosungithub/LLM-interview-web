"""Helpers for tests.py: brute-force answers straight from the statement (no tree at all), plus input generators and
the text that shows an array as pasteable code.

Every test input uses integers, half-integers or multiples of 2**-20 in [0, 1), with d <= 5. For those, each
squared difference and every partial sum of them is exactly representable in float64, so the squared distances
below are exact and ties are real ties, whatever order an answer adds the terms in."""
import numpy as np


def sq_dists(P, q):
    return ((np.asarray(P, dtype=np.float64) - np.asarray(q, dtype=np.float64)) ** 2).sum(axis=1)


def nearest(P, q):
    """argmin returns the first, so the smallest, index among exact ties."""
    return int(np.argmin(sq_dists(P, q)))


def knn(P, q, k):
    d2 = sq_dists(P, q)
    return [int(i) for i in np.lexsort((np.arange(len(d2)), d2))[:k]]     # by distance, then index


def count_in_box(P, lo, hi):
    return sum(1 for p in np.asarray(P).tolist() if all(lo[i] <= p[i] <= hi[i] for i in range(len(lo))))


def nearest_many(P, Q, chunk=200):
    out = []
    for s in range(0, len(Q), chunk):
        d2 = ((Q[s:s + chunk, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        out.extend(int(i) for i in d2.argmin(axis=1))
    return out


# ---------------------------------------------------------------- generators

def grid_cloud(r, n, d, lo=-3, hi=3):
    """n integer points in [lo, hi]^d (many equal coordinates, repeated points and distance ties), with the code
    that rebuilds them."""
    seed = r.randrange(10 ** 6)
    text = f"np.random.default_rng({seed}).integers({lo}, {hi + 1}, size=({n}, {d})).astype(float)"
    return np.random.default_rng(seed).integers(lo, hi + 1, size=(n, d)).astype(float), text


def dyadic_cloud(r, n, d):
    """n points in [0, 1)^d on the grid of multiples of 2**-20, with the code that rebuilds them."""
    seed = r.randrange(10 ** 6)
    text = f"np.random.default_rng({seed}).integers(0, 2**20, size=({n}, {d})) / 2**20"
    return np.random.default_rng(seed).integers(0, 2 ** 20, size=(n, d)) / 2 ** 20, text


def grid_query(r, d, lo=-4, hi=4):
    """Integer or half-integer coordinates, so a query often sits exactly between two split values."""
    return np.array([r.randint(2 * lo, 2 * hi) / 2 for _ in range(d)], dtype=np.float64)


def dyadic_query(r, d, bits=20):
    """A point in [0, 1)^d on the grid of multiples of 2**-bits."""
    m = 1 << bits
    return np.array([r.randrange(m) / m for _ in range(d)], dtype=np.float64)


# ---------------------------------------------------------------- display

def _code(v):
    if isinstance(v, list):
        return "[" + ", ".join(_code(e) for e in v) + "]"
    return "np.inf" if v == np.inf else "-np.inf" if v == -np.inf else repr(v)


def arr_text(a):
    """An array as code that rebuilds it exactly, e.g. np.array([[2.0, 3.0], [5.0, 4.0]])."""
    return f"np.array({_code(np.asarray(a, dtype=np.float64).tolist())})"
