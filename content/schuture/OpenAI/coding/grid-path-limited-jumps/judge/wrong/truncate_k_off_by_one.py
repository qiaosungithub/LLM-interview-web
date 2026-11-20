# expect: WA
# Caps K at (N - 2) // 2 instead of (N - 1) // 2, one special move too few when N is odd.
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
    K = min(K, max(0, (N - 2) // 2))

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
    K = min(K, max(0, (len(board) - 2) // 2))
    return _tables(board, K)(0, p, K)[0]


def count_optimal_paths(board, p, K):
    K = min(K, max(0, (len(board) - 2) // 2))
    return _tables(board, K)(0, p, K)[1] % MOD


def optimal_path(board, p, K):
    N, M = len(board), len(board[0])
    K = min(K, max(0, (N - 2) // 2))
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
    K = min(K, max(0, (N - 2) // 2))

    @lru_cache(maxsize=None)
    def g(i, j, k, rose):
        here = board[i][j]
        if i == N - 1:
            return here
        res = None
        for a, b, c in _moves(i, j, k, N, M):
            there = board[a][b]
            up = here < there
            v = g(a, b, c, up) + (X if here == there else 0) + (Y if rose and up else 0)
            if res is None or v > res:
                res = v
        return here + res

    return g(0, p, K, False)
