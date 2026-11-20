"""Helpers for tests.py: a brute force over plans that checks the rules as stated, an independent O(n) DP for
inputs too big for the brute force, and random input generators."""
from collections import deque

MOD = 1_000_000_007
INF = float("inf")


def brute(c, G, H):
    """(minimum cost, exact number of plans with that cost), trying every subset of the inner frames.
    Part 1 is the case H = G (no gap can then be long)."""
    n = len(c)
    if n == 0:
        return 0, 1                     # the empty plan
    if n == 1:
        return c[0], 1                  # frame 0 is both the first and the last frame
    best, ways = None, 0
    inner = n - 2
    for mask in range(1 << inner):
        keys = [0] + [k + 1 for k in range(inner) if mask >> k & 1] + [n - 1]
        gaps = [b - a for a, b in zip(keys, keys[1:])]
        if max(gaps) > G:
            continue
        if any(x > H and y > H for x, y in zip(gaps, gaps[1:])):
            continue
        cost = sum(c[k] for k in keys)
        if best is None or cost < best:
            best, ways = cost, 1
        elif cost == best:
            ways += 1
    return best, ways


def _merge(a, b):
    if a[0] != b[0]:
        return a if a[0] < b[0] else b
    return a[0], (a[1] + b[1]) % MOD


class _Window:
    """Indices arrive in increasing order with a (cost, ways) pair. query(lo) returns the smallest cost among the
    indices >= lo and the total ways of the indices that reach it, or (INF, 0). Amortised O(1) per call."""

    def __init__(self):
        self.groups = deque()           # [cost, deque of (index, ways), sum of those ways]; cost rises front to back

    def push(self, idx, cost, ways):
        if cost == INF:
            return
        while self.groups and self.groups[-1][0] > cost:
            self.groups.pop()           # older and dearer: never the minimum again
        if self.groups and self.groups[-1][0] == cost:
            group = self.groups[-1]
            group[1].append((idx, ways))
            group[2] += ways
        else:
            self.groups.append([cost, deque([(idx, ways)]), ways])

    def query(self, lo):
        while self.groups:
            group = self.groups[0]
            while group[1] and group[1][0][0] < lo:
                group[2] -= group[1].popleft()[1]
            if group[1]:
                return group[0], group[2] % MOD
            self.groups.popleft()
        return INF, 0


def fast(c, G, H):
    """Same result as brute() with the count taken modulo MOD, in O(n): sliding-window minima with tie counts."""
    n = len(c)
    if n == 0:
        return 0, 1
    short = [(INF, 0)] * n              # keyframe at i, the last gap is not long (or there is no gap)
    long_ = [(INF, 0)] * n              # keyframe at i, the last gap is long
    short[0] = (c[0], 1)
    near, far = _Window(), _Window()    # near: j in [i-H, i-1], either state; far: j in [i-G, i-H-1], short only
    for i in range(1, n):
        near.push(i - 1, *_merge(short[i - 1], long_[i - 1]))
        if i - H - 1 >= 0:
            far.push(i - H - 1, *short[i - H - 1])
        cost, ways = near.query(i - H)
        short[i] = (c[i] + cost, ways)
        cost, ways = far.query(i - G)
        if cost != INF:
            long_[i] = (c[i] + cost, ways)
    return _merge(short[n - 1], long_[n - 1])


def fibonacci_mod(m):
    """F(m) mod MOD with F(0) = 0, F(1) = 1: the number of ways to write m - 1 as an ordered sum of 1s and 2s."""
    a, b = 0, 1
    for _ in range(m):
        a, b = b, (a + b) % MOD
    return a


def ones_and_twos_no_adjacent_twos_mod(m):
    """Ordered sums of 1s and 2s equal to m with no two 2s next to each other, mod MOD:
    t[m] = t[m-1] (ends with 1) + t[m-3] (ends with 1, 2), t[0] = t[1] = 1, t[2] = 2."""
    t = [1, 1, 2]
    while len(t) <= m:
        t.append((t[-1] + t[-3]) % MOD)
    return t[m]


def random_input(r, n_lo, n_hi, top=None):
    """(c, G, H) with n in [n_lo, n_hi]; G up to n + 1, H in [1, G]; costs from all-zero (many ties) to 10**9."""
    n = r.randint(n_lo, n_hi)
    G = r.randint(1, n + 1)
    H = r.randint(1, G)
    top = r.choice([0, 1, 3, 9, 10 ** 9]) if top is None else top
    return [r.randint(0, top) for _ in range(n)], G, H
