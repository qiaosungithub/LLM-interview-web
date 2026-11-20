# expect: TLE
# Enumerates every path: correct, but the number of paths grows like 3^N.
MOD = 10 ** 9 + 7


def _all_paths(board, p, K):
    N, M = len(board), len(board[0])
    out = []

    def go(i, j, k, path):
        if i == N - 1:
            out.append(list(path))
            return
        for nj in (j - 1, j, j + 1):
            if 0 <= nj < M:
                go(i + 1, nj, k, path + [(i + 1, nj)])
        if k > 0 and i + 2 <= N - 1:
            go(i + 2, j, k - 1, path + [(i + 2, j)])

    go(0, p, K, [(0, p)])
    return out


def _score(board, path):
    return sum(board[r][c] for r, c in path)


def max_score(board, p, K):
    return max(_score(board, q) for q in _all_paths(board, p, K))


def optimal_path(board, p, K):
    paths = _all_paths(board, p, K)
    best = max(_score(board, q) for q in paths)
    return min(q for q in paths if _score(board, q) == best)


def count_optimal_paths(board, p, K):
    paths = _all_paths(board, p, K)
    best = max(_score(board, q) for q in paths)
    return sum(1 for q in paths if _score(board, q) == best) % MOD


def max_score_with_bonuses(board, p, K, X, Y):
    best = None
    for q in _all_paths(board, p, K):
        v = [board[r][c] for r, c in q]
        s = sum(v) + X * sum(v[t - 1] == v[t] for t in range(1, len(v)))
        s += Y * sum(v[t - 2] < v[t - 1] < v[t] for t in range(2, len(v)))
        best = s if best is None else max(best, s)
    return best
