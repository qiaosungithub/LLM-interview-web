# expect: WA
# Wrong for Part 3: reduces the costs modulo 10**9 + 7 together with the counts (the page: costs must never be reduced).
INF = float("inf")
MOD = 1_000_000_007


def min_cost(c, G):
    n = len(c)
    if n == 0:
        return 0
    best = [0] * n
    best[0] = c[0]
    for i in range(1, n):
        best[i] = c[i] + min(best[max(0, i - G):i])
    return best[n - 1]


def min_cost_limited(c, G, H):
    n = len(c)
    if n == 0:
        return 0
    short = [INF] * n
    long_ = [INF] * n
    short[0] = c[0]
    for i in range(1, n):
        near, far = max(0, i - H), max(0, i - G)
        short[i] = c[i] + min(min(short[near:i]), min(long_[near:i]))
        if far < near:
            long_[i] = c[i] + min(short[far:near])
    return min(short[n - 1], long_[n - 1])


def _best(cost, ways, lo, hi):
    best, total = INF, 0
    for j in range(lo, hi):
        if cost[j] < best:
            best, total = cost[j], ways[j]
        elif cost[j] == best:
            total += ways[j]
    return best, total % MOD


def _merge(a, b):
    if a[0] != b[0]:
        return a if a[0] < b[0] else b
    return a[0], (a[1] + b[1]) % MOD


def count_optimal(c, G, H):
    n = len(c)
    if n == 0:
        return 0, 1
    s_cost, s_ways = [INF] * n, [0] * n
    l_cost, l_ways = [INF] * n, [0] * n
    s_cost[0], s_ways[0] = c[0] % MOD, 1
    for i in range(1, n):
        near, far = max(0, i - H), max(0, i - G)
        cost, ways = _merge(_best(s_cost, s_ways, near, i), _best(l_cost, l_ways, near, i))
        s_cost[i], s_ways[i] = (c[i] + cost) % MOD, ways              # the cost is reduced too
        cost, ways = _best(s_cost, s_ways, far, near)
        if cost != INF:
            l_cost[i], l_ways[i] = (c[i] + cost) % MOD, ways
    return _merge((s_cost[n - 1], s_ways[n - 1]), (l_cost[n - 1], l_ways[n - 1]))
