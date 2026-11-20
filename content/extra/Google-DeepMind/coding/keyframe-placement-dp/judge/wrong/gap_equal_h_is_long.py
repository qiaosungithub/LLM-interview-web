# expect: WA
# Wrong for Parts 2 and 3: treats a gap of exactly H as long (the statement: long means greater than H).
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
        for j in range(max(0, i - G), i):
            if i - j >= H:                                  # off by one: should be i - j > H
                long_[i] = min(long_[i], c[i] + short[j])
            else:
                short[i] = min(short[i], c[i] + min(short[j], long_[j]))
    return min(short[n - 1], long_[n - 1])


def _add(acc, cost, ways):
    if cost < acc[0]:
        return [cost, ways]
    if cost == acc[0]:
        return [cost, (acc[1] + ways) % MOD]
    return acc


def count_optimal(c, G, H):
    n = len(c)
    if n == 0:
        return 0, 1
    short = [[INF, 0] for _ in range(n)]
    long_ = [[INF, 0] for _ in range(n)]
    short[0] = [c[0], 1]
    for i in range(1, n):
        for j in range(max(0, i - G), i):
            if i - j >= H:                                  # off by one: should be i - j > H
                long_[i] = _add(long_[i], c[i] + short[j][0], short[j][1])
            else:
                short[i] = _add(short[i], c[i] + short[j][0], short[j][1])
                short[i] = _add(short[i], c[i] + long_[j][0], long_[j][1])
    best = _add(list(short[n - 1]), *long_[n - 1])
    return best[0], best[1]
