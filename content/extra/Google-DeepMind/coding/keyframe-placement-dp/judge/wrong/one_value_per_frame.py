# expect: WA
# Wrong for Parts 2 and 3: one value per frame plus a flag for how that one cheapest plan ended (the page's pitfall).
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
    best = [INF] * n
    was_long = [False] * n          # did the cheapest plan found for frame i end with a long gap?
    best[0] = c[0]
    for i in range(1, n):
        for j in range(max(0, i - G), i):
            is_long = i - j > H
            if is_long and was_long[j]:
                continue
            cand = best[j] + c[i]
            if cand < best[i] or (cand == best[i] and was_long[i] and not is_long):
                best[i], was_long[i] = cand, is_long
    return best[n - 1]


def count_optimal(c, G, H):
    n = len(c)
    if n == 0:
        return 0, 1
    best = [INF] * n
    ways = [0] * n
    was_long = [False] * n
    best[0], ways[0] = c[0], 1
    for i in range(1, n):
        for j in range(max(0, i - G), i):
            is_long = i - j > H
            if is_long and was_long[j]:
                continue
            cand = best[j] + c[i]
            if cand < best[i]:
                best[i], ways[i], was_long[i] = cand, ways[j], is_long
            elif cand == best[i]:
                ways[i] = (ways[i] + ways[j]) % MOD
                was_long[i] = was_long[i] and is_long     # prefer to remember a short last gap
    return best[n - 1], ways[n - 1]
