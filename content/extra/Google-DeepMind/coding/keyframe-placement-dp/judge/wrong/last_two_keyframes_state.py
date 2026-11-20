# expect: TLE
# Correct but too slow for Parts 2 and 3: the state is the last two keyframes (j, i), O(n * G^2) time instead of O(n * G).
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


def _pairs(c, G, H):
    """end[i][j] = (cost, ways) over valid plans on frames 0..i whose last two keyframes are j < i."""
    n = len(c)
    end = [{} for _ in range(n)]
    for i in range(1, n):
        for j in range(max(0, i - G), i):
            if j == 0:
                end[i][j] = (c[0] + c[i], 1)
                continue
            cost, ways = INF, 0
            for k, (pc, pw) in end[j].items():
                if i - j > H and j - k > H:          # two long gaps in a row
                    continue
                if pc < cost:
                    cost, ways = pc, pw
                elif pc == cost:
                    ways += pw
            if cost != INF:
                end[i][j] = (cost + c[i], ways % MOD)
    return end


def count_optimal(c, G, H):
    n = len(c)
    if n == 0:
        return 0, 1
    if n == 1:
        return c[0], 1
    best, total = INF, 0
    for cost, ways in _pairs(c, G, H)[n - 1].values():
        if cost < best:
            best, total = cost, ways
        elif cost == best:
            total += ways
    return best, total % MOD


def min_cost_limited(c, G, H):
    return count_optimal(c, G, H)[0]
