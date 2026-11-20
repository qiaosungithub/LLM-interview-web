# expect: WA
# Part 4 forward DP keeps only the best score per (row, column, moves used) and the value before it, so a lower-scoring arrival that would earn a later Y bonus is thrown away.
import sys
from functools import lru_cache

sys.setrecursionlimit(10000)
MOD = 10 ** 9 + 7


def _moves(i, j, k, N, M):
    for nj in (j - 1, j, j + 1):
        if 0 <= nj < M:
            yield i + 1, nj, k
    if k > 0 and i + 2 < N:
        yield i + 2, j, k - 1


def _tables(board, K):
    N, M = len(board), len(board[0])

    @lru_cache(maxsize=None)
    def best(i, j, k):
        if i == N - 1:
            return board[i][j], 1
        top, ways = None, 0
        for a, b, c in _moves(i, j, k, N, M):
            s, w = best(a, b, c)
            if top is None or s > top:
                top, ways = s, w
            elif s == top:
                ways = (ways + w) % MOD
        return board[i][j] + top, ways

    return best


def max_score(board, p, K):
    return _tables(board, K)(0, p, K)[0]


def count_optimal_paths(board, p, K):
    return _tables(board, K)(0, p, K)[1] % MOD


def optimal_path(board, p, K):
    N, M = len(board), len(board[0])
    best = _tables(board, K)
    i, j, k = 0, p, K
    out = [(0, p)]
    while i < N - 1:
        need = best(i, j, k)[0] - board[i][j]
        for a, b, c in _moves(i, j, k, N, M):
            if best(a, b, c)[0] == need:
                i, j, k = a, b, c
                break
        out.append((i, j))
    return out


def max_score_with_bonuses(board, p, K, X, Y):
    N, M = len(board), len(board[0])
    # best[(i, j, used)] = (score, value of the cell visited before (i, j)); only the best score is kept
    best = {(0, p, 0): (board[0][p], None)}
    for i in range(N - 1):
        for j in range(M):
            for used in range(K + 1):
                if (i, j, used) not in best:
                    continue
                s, before = best[(i, j, used)]
                here = board[i][j]
                for a, b, c in _moves(i, j, K - used, N, M):
                    there = board[a][b]
                    v = s + there + (X if here == there else 0)
                    if before is not None and before < here < there:
                        v += Y
                    key = (a, b, K - c)
                    if key not in best or v > best[key][0]:
                        best[key] = (v, here)
    return max(s for (i, j, u), (s, _) in best.items() if i == N - 1)
