def min_cost(c, G):
    n = len(c)
    if n == 0:
        return 0
    best = [0] * n            # best[i]: cheapest valid choice on frames 0..i that ends with a keyframe at i
    best[0] = c[0]
    for i in range(1, n):
        best[i] = c[i] + min(best[max(0, i - G):i])      # previous keyframe j with i - G <= j < i
    return best[n - 1]


INF = float("inf")


def min_cost_limited(c, G, H):
    n = len(c)
    if n == 0:
        return 0
    short = [INF] * n         # the last gap is not long (or there is no gap yet)
    long_ = [INF] * n         # the last gap is long
    short[0] = c[0]
    for i in range(1, n):
        near = max(0, i - H)  # j in [near, i) makes a short gap, j in [far, near) a long one
        far = max(0, i - G)
        short[i] = c[i] + min(min(short[near:i]), min(long_[near:i]))
        if far < near:
            long_[i] = c[i] + min(short[far:near])       # NOTE: short[j] only, a long gap may not follow a long gap
    return min(short[n - 1], long_[n - 1])


MOD = 1_000_000_007


def _best(cost, ways, lo, hi):
    """Smallest cost at positions lo..hi-1 and how many choices reach it; (INF, 0) for an empty range."""
    best, total = INF, 0
    for j in range(lo, hi):
        if cost[j] < best:
            best, total = cost[j], ways[j]
        elif cost[j] == best:
            total += ways[j]
    return best, total % MOD


def _merge(a, b):
    """Combine two (cost, ways) alternatives that describe disjoint sets of choices."""
    if a[0] != b[0]:
        return a if a[0] < b[0] else b
    return a[0], (a[1] + b[1]) % MOD


def count_optimal(c, G, H):
    n = len(c)
    if n == 0:
        return 0, 1           # the empty plan
    s_cost, s_ways = [INF] * n, [0] * n       # same meaning as short[] in Part 2, plus a count
    l_cost, l_ways = [INF] * n, [0] * n       # same meaning as long_[]
    s_cost[0], s_ways[0] = c[0], 1
    for i in range(1, n):
        near = max(0, i - H)
        far = max(0, i - G)
        cost, ways = _merge(_best(s_cost, s_ways, near, i), _best(l_cost, l_ways, near, i))
        s_cost[i], s_ways[i] = c[i] + cost, ways
        cost, ways = _best(s_cost, s_ways, far, near)
        l_cost[i], l_ways[i] = c[i] + cost, ways         # stays (INF, 0) when no long gap can end here
    return _merge((s_cost[n - 1], s_ways[n - 1]), (l_cost[n - 1], l_ways[n - 1]))
