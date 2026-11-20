"""Brute forces written from the statement alone, random inputs, and the statistics behind the frequency tests.

- ref_iter(items): Part 1's rules applied to a finite list.
- random_items(r): a random Part 1 input mixing ints, None and invalid elements.
- random_weights(r, kind, n): weight vectors for Parts 2 and 3.
- Frequency checks. For a correct sampler the count c of a value with probability p in n draws is Binomial(n, p),
  and the Chernoff bound gives P(c >= n q) <= exp(-n KL(q || p)) for q >= p (and the same for c <= n q, q <= p),
  with KL the Bernoulli relative entropy. A check over k cells accepts when n KL(c / n || p) <= log(2 k / ALPHA)
  for every cell, so by the union bound a correct sampler fails it with probability at most ALPHA, for any seed.
  Rounding moves a correct sampler's probabilities by about 1e-16, far too little to matter at these n."""
import math

ALPHA = 1e-10


def ref_iter(items):
    """(values yielded, index of the first invalid element or None, positions), where positions[k] is the index
    of the element that produced the k-th value."""
    values, positions = [], []
    for i, x in enumerate(items):
        if x is None:
            continue
        if isinstance(x, bool) or not isinstance(x, int):
            return values, i, positions
        values.append(x)
        positions.append(i)
    return values, None, positions


INVALID = [True, False, 3.0, "3", "x", 2.5, b"1", [1], 1 + 0j]


def random_items(r):
    items = []
    for _ in range(r.randint(0, 14)):
        u = r.random()
        if u < 0.2:
            items.append(None)
        elif u < 0.85:
            items.append(r.randint(-5, 5))
        elif u < 0.9:
            items.append(r.choice([-1, 1]) * r.randint(10 ** 18, 10 ** 30))
        else:
            items.append(r.choice(INVALID))
    return items


KINDS = ["uniform", "integers", "zeros", "equal", "skewed"]


def random_weights(r, kind, n):
    if kind == "uniform":
        ws = [round(r.uniform(0.05, 5.0), 3) for _ in range(n)]
    elif kind == "integers":
        ws = [float(r.randint(1, 9)) for _ in range(n)]
    elif kind == "zeros":
        ws = [0.0 if r.random() < 0.4 else round(r.uniform(0.1, 5.0), 3) for _ in range(n)]
        if not any(ws):
            ws[r.randrange(n)] = 1.0
    elif kind == "equal":
        ws = [1.0] * n
    else:
        ws = [round(r.uniform(0.01, 0.5), 3) for _ in range(n)]
        ws[r.randrange(n)] = 20.0
    return ws


def probabilities(values, weights):
    """{value: weight / sum(weights)} for distinct values, the sum taken exactly."""
    total = math.fsum(weights)
    return {v: w / total for v, w in zip(values, weights)}


def kl(q, p):
    """Relative entropy of Bernoulli(q) from Bernoulli(p), 0 < p < 1."""
    out = 0.0
    if q > 0:
        out += q * math.log(q / p)
    if q < 1:
        out += (1 - q) * math.log((1 - q) / (1 - p))
    return out


def limit(cells):
    return math.log(2 * cells / ALPHA)


def accepts(c, n, p, t):
    return n * kl(c / n, p) <= t


def count_range(n, p, t):
    """The counts a check with threshold t accepts: an interval around n p (n KL is convex in c)."""
    mid = min(n, max(0, round(n * p)))
    a, b = 0, mid                      # smallest accepted count
    while a < b:
        m = (a + b) // 2
        if accepts(m, n, p, t):
            b = m
        else:
            a = m + 1
    lo = a
    a, b = mid, n                      # largest accepted count
    while a < b:
        m = (a + b + 1) // 2
        if accepts(m, n, p, t):
            a = m
        else:
            b = m - 1
    return lo, a
