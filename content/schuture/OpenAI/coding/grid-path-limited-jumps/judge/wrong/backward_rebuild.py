# expect: WA
# optimal_path rebuilds the path backwards from the best end cell, taking the smallest column at each step: that minimises the path read from the end, not from the start.
def _best_scores(board, K):
    """dp[i][j][k]: max base score of a path from (i, j) to row N - 1, using at
    most k more special moves."""
    N, M = len(board), len(board[0])
    dp = [[[0] * (K + 1) for _ in range(M)] for _ in range(N)]
    for j in range(M):
        for k in range(K + 1):
            dp[N - 1][j][k] = board[N - 1][j]
    for i in range(N - 2, -1, -1):
        for j in range(M):
            for k in range(K + 1):
                candidates = [dp[i + 1][j + dj][k] for dj in (-1, 0, 1) if 0 <= j + dj < M]
                if k > 0 and i + 2 <= N - 1:              # NOTE: excludes i == N-2, where i+2 == N is outside the grid
                    candidates.append(dp[i + 2][j][k - 1])
                dp[i][j][k] = board[i][j] + max(candidates)   # NOTE: dj = 0 is always in range, candidates is never empty
    return dp


def max_score(board, p, K):
    return _best_scores(board, K)[0][p][K]

def optimal_path(board, p, K):
    # forward DP over prefixes, then walk back from the best end cell
    N, M = len(board), len(board[0])
    NEG = float("-inf")
    f = [[[NEG] * (K + 1) for _ in range(M)] for _ in range(N)]
    f[0][p][0] = board[0][p]
    for i in range(N - 1):
        for j in range(M):
            for u in range(K + 1):
                if f[i][j][u] == NEG:
                    continue
                for nj in (j - 1, j, j + 1):
                    if 0 <= nj < M:
                        f[i + 1][nj][u] = max(f[i + 1][nj][u], f[i][j][u] + board[i + 1][nj])
                if u < K and i + 2 <= N - 1:
                    f[i + 2][j][u + 1] = max(f[i + 2][j][u + 1], f[i][j][u] + board[i + 2][j])
    best = max(f[N - 1][j][u] for j in range(M) for u in range(K + 1))
    i, j, u = next((N - 1, j, u) for j in range(M) for u in range(K + 1) if f[N - 1][j][u] == best)
    path = [(i, j)]
    while i > 0:
        before = [(i - 1, j + dj, u) for dj in (-1, 0, 1) if 0 <= j + dj < M]
        if u > 0 and i >= 2:
            before.append((i - 2, j, u - 1))
        for (a, b, w) in before:            # the smallest column first
            if f[a][b][w] != NEG and f[a][b][w] + board[i][j] == f[i][j][u]:
                i, j, u = a, b, w
                break
        path.append((i, j))
    return path[::-1]


MOD = 10 ** 9 + 7


def count_optimal_paths(board, p, K):
    N, M = len(board), len(board[0])
    dp = [[[0] * (K + 1) for _ in range(M)] for _ in range(N)]
    cnt = [[[0] * (K + 1) for _ in range(M)] for _ in range(N)]
    for j in range(M):
        for k in range(K + 1):
            dp[N - 1][j][k] = board[N - 1][j]
            cnt[N - 1][j][k] = 1
    for i in range(N - 2, -1, -1):
        for j in range(M):
            for k in range(K + 1):
                moves = [(i + 1, j + dj, k) for dj in (-1, 0, 1) if 0 <= j + dj < M]
                if k > 0 and i + 2 <= N - 1:
                    moves.append((i + 2, j, k - 1))
                best, total = None, 0
                for (ni, nj, nk) in moves:
                    val = dp[ni][nj][nk]
                    if best is None or val > best:       # NOTE: strict '>' — a new best discards the old tie count
                        best, total = val, cnt[ni][nj][nk]
                    elif val == best:                     # NOTE: compare the real (possibly negative) scores, never
                        total = (total + cnt[ni][nj][nk]) % MOD  #      `% MOD`-reduced ones; only `total` is reduced
                dp[i][j][k] = board[i][j] + best
                cnt[i][j][k] = total
    return cnt[0][p][K]

def max_score_with_bonuses(board, p, K, X, Y):
    """g[i][j][k][u]: max base score plus bonuses of a path from (i, j) to row N - 1,
    using at most k more special moves, where u = 1 iff the cell visited just before
    (i, j) has a smaller value than (i, j)."""
    N, M = len(board), len(board[0])
    g = [[[[0, 0] for _ in range(K + 1)] for _ in range(M)] for _ in range(N)]
    for j in range(M):
        for k in range(K + 1):
            g[N - 1][j][k] = [board[N - 1][j], board[N - 1][j]]
    for i in range(N - 2, -1, -1):
        for j in range(M):
            cur = board[i][j]
            for k in range(K + 1):
                moves = [(i + 1, j + dj, k) for dj in (-1, 0, 1) if 0 <= j + dj < M]
                if k > 0 and i + 2 <= N - 1:
                    moves.append((i + 2, j, k - 1))
                for u in (0, 1):
                    best = None
                    for (ni, nj, nk) in moves:
                        nxt = board[ni][nj]
                        up = 1 if cur < nxt else 0     # NOTE: this is the next cell's u
                        cand = g[ni][nj][nk][up] + (X if cur == nxt else 0) + (Y if u and up else 0)
                        if best is None or cand > best:
                            best = cand
                    g[i][j][k][u] = cur + best
    return g[0][p][K][0]   # NOTE: u = 0 at the start: no earlier cell, so no triple can end at t = 1
