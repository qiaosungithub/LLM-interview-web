# expect: WA
# Wrong for Parts 1 and 2: greedy -- from each keyframe jump to the cheapest frame in reach, or to the end once it is in reach.
INF = float("inf")
MOD = 1_000_000_007


def min_cost(c, G):
    n = len(c)
    if n == 0:
        return 0
    pos, total = 0, c[0]
    while pos != n - 1:
        if n - 1 - pos <= G:
            pos = n - 1
        else:
            pos = min(range(pos + 1, pos + G + 1), key=lambda j: (c[j], -j))    # cheapest, farthest on ties
        total += c[pos]
    return total


def min_cost_limited(c, G, H):
    n = len(c)
    if n == 0:
        return 0
    pos, last_long, total = 0, False, c[0]
    while pos != n - 1:
        reach = H if last_long else G       # after a long gap the next gap must be short
        if n - 1 - pos <= reach:
            nxt = n - 1
        else:
            nxt = min(range(pos + 1, pos + reach + 1), key=lambda j: (c[j], -j))
        last_long = nxt - pos > H
        total += c[nxt]
        pos = nxt
    return total


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
    s_cost[0], s_ways[0] = c[0], 1
    for i in range(1, n):
        near, far = max(0, i - H), max(0, i - G)
        cost, ways = _merge(_best(s_cost, s_ways, near, i), _best(l_cost, l_ways, near, i))
        s_cost[i], s_ways[i] = c[i] + cost, ways
        cost, ways = _best(s_cost, s_ways, far, near)
        l_cost[i], l_ways[i] = c[i] + cost, ways
    return _merge((s_cost[n - 1], s_ways[n - 1]), (l_cost[n - 1], l_ways[n - 1]))
