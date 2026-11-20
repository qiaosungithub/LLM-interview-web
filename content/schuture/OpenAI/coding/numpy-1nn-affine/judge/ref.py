"""Independent restatement of the problem, used by tests.py for inputs and for the set of acceptable answers.

The statement fixes nn(q) as the smallest index at the minimum squared distance, computed exactly, and accepts any
index whose distance is within floating-point rounding error of the minimum, except that with small integer
coordinates (where float64 arithmetic is exact) a tie must go to the smallest index. nearest_sets() returns, for every
query, the indices a correct answer may pick under that rule. Distances are taken from the definition
sum_k (q_k - x_ik)^2; the expanded formula is used only to find the few candidates near each row's minimum.

Rounding bound. A length-d dot product computed in float64 is off by at most d * u * sum_k |a_k b_k| (u = 2^-53), in
any summation order. The expansion |q|^2 + |x|^2 - 2 q.x, the definition, and the logits 2 q.x - |x|^2 all consist
of such dot products plus two or three more roundings, so each computed distance is off by at most
(d + 3) * u * s(q, x) with s(q, x) = sum_k (|q_k| + |x_ik|)^2. Comparing two distances doubles that. The tolerance
below is 8 (d + 3) u max_i s(q, x_i), four times the worst case, plus 8u for the softmax: exp(t) rounds to exactly 1
for |t| below about 1.1e-16, so two logits that close have equal probabilities.

Inputs come from np.random.default_rng(seed) by a fixed recipe, and each recipe has a one-line text form, so that a
failing call can be pasted into 自定义运行 as it is.
"""
import numpy as np

U = 2.0 ** -53


# ---------------------------------------------------------------- inputs

def inputs(kind, seed, n, d, m, c, a):
    """X_train (n, d), y_train (n,) with labels in 0..c-1, X_query (m, d).
    kind "normal": standard normal times a; "uniform": uniform in [-a, a]; "grid": integers in [-a, a];
    "decimal": multiples of 0.1 from 0 to a / 10."""
    g = np.random.default_rng(seed)
    draw = {
        "normal": lambda shape: g.standard_normal(shape) * a,
        "uniform": lambda shape: g.uniform(-a, a, shape),
        "grid": lambda shape: g.integers(-a, a + 1, shape).astype(float),
        "decimal": lambda shape: g.integers(0, a + 1, shape) / 10,
    }[kind]
    X_train = draw((n, d))
    y_train = g.integers(0, c, n)
    X_query = draw((m, d))
    return X_train, y_train, X_query


def recipe(kind, seed, n, d, m, c, a):
    draw = {
        "normal": lambda shape: f"g.standard_normal({shape})" + ("" if a == 1 else f" * {a}"),
        "uniform": lambda shape: f"g.uniform(-{a}, {a}, {shape})",
        "grid": lambda shape: f"g.integers({-a}, {a + 1}, {shape}).astype(float)",
        "decimal": lambda shape: f"g.integers(0, {a + 1}, {shape}) / 10",
    }[kind]
    return (f"g = np.random.default_rng({seed}); X_train = {draw((n, d))}; y_train = g.integers(0, {c}, {n}); "
            f"X_query = {draw((m, d))}")


def literal(X_train, y_train, X_query):
    """Fixed small inputs as one line of code."""
    return (f"X_train = np.array({X_train.tolist()}); y_train = np.array({y_train.tolist()}); "
            f"X_query = np.array({X_query.tolist()})")


# ---------------------------------------------------------------- distances and the acceptable answers

def row_distances(X_train, q):
    """Squared distances from one query to every training point, by the definition."""
    return ((X_train - q) ** 2).sum(axis=1)


def tolerance(X_train, X_query):
    """Per query: how far apart two computed distances may be and still count as a tie (see the module docstring)."""
    d = X_train.shape[1]
    Q, X = np.abs(X_query), np.abs(X_train)
    s = (Q ** 2).sum(axis=1)[:, None] + (X ** 2).sum(axis=1) + 2.0 * Q @ X.T        # (m, n)
    return 8 * (d + 3) * U * s.max(axis=1) + 8 * U                                  # (m,)


def nearest_sets(X_train, X_query, exact):
    """For each query, the sorted array of training indices a correct answer may return.
    exact=True (small integer coordinates): only nn(q), the smallest index at the minimum.
    exact=False: every index whose distance is within tolerance() of the minimum."""
    tol = tolerance(X_train, X_query)
    fast = (X_query ** 2).sum(axis=1)[:, None] + (X_train ** 2).sum(axis=1) - 2.0 * X_query @ X_train.T
    # Each fast entry is within tol / 8 of the exact distance, so every index within tol of the minimum is a candidate.
    rows, cols = np.nonzero(fast <= fast.min(axis=1, keepdims=True) + 2 * tol[:, None])
    dist = ((X_query[rows] - X_train[cols]) ** 2).sum(axis=1)                      # the definition, candidates only
    starts = np.searchsorted(rows, np.arange(len(X_query) + 1))
    sets = []
    for j in range(len(X_query)):
        c, dj = cols[starts[j]:starts[j + 1]], dist[starts[j]:starts[j + 1]]
        if exact:
            sets.append(c[dj == dj.min()][:1])                                      # cols come sorted
        else:
            sets.append(c[dj <= dj.min() + tol[j]])
    return sets


# ---------------------------------------------------------------- Part 2

def softmax(Z):
    """Row-wise softmax, shifted by the row maximum so that large logits stay finite."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def probs_tolerance(X_query, W, b):
    """Absolute tolerance between two float64 computations of softmax(X_query @ W + b).
    Every logit is a length-d dot product plus a bias, off by at most (d + 1) u (|Q| @ |W| + |b|) in either
    computation. A change dz in the logits changes p_i by at most 2 p_i max|dz|, and exp and the division add a few
    u of relative error, so with p_i <= 1 the bound is 4 (d + 1) u scale + 4u; this is four times that."""
    d = X_query.shape[1]
    scale = float((np.abs(X_query) @ np.abs(W) + np.abs(b)).max())
    return 16 * (d + 1) * U * scale + 16 * U


def row_sum_tolerance(n):
    """|sum of n computed probabilities - 1|: each is off by a few u relative, and the sum adds n roundings."""
    return 8 * (n + 2) * U
