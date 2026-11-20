# expect: TLE
# Correct but exponential: enumerates every valid plan by depth-first search, which the statement says cannot work.
INF = float("inf")
MOD = 1_000_000_007


def _search(c, G, H):
    n = len(c)
    if n == 0:
        return 0, 1
    if n == 1:
        return c[0], 1
    best = [INF, 0]

    def go(last, last_long, cost):
        if last == n - 1:
            if cost < best[0]:
                best[0], best[1] = cost, 1
            elif cost == best[0]:
                best[1] += 1
            return
        for nxt in range(last + 1, min(n - 1, last + G) + 1):
            is_long = nxt - last > H
            if not (is_long and last_long):
                go(nxt, is_long, cost + c[nxt])

    go(0, False, c[0])
    return best[0], best[1] % MOD


def min_cost(c, G):
    return _search(c, G, G)[0]


def min_cost_limited(c, G, H):
    return _search(c, G, H)[0]


def count_optimal(c, G, H):
    return _search(c, G, H)
