"""Helpers for tests.py: every path enumerated for small boards, an independent DP for big ones, a path checker
and random boards.

A path is a list of cells (row, column) starting at (0, p). A special move goes from (i, j) to (i + 2, j).
"""
MOD = 10 ** 9 + 7


# ---------------------------------------------------------------- brute force: every path

def paths(board, p, K):
    """Every path from (0, p) to row N - 1 using at most K special moves."""
    N, M = len(board), len(board[0])
    out = []

    def go(i, j, k, path):
        if i == N - 1:
            out.append(list(path))
            return
        for nj in (j - 1, j, j + 1):
            if 0 <= nj < M:
                path.append((i + 1, nj))
                go(i + 1, nj, k, path)
                path.pop()
        if k > 0 and i + 2 <= N - 1:
            path.append((i + 2, j))
            go(i + 2, j, k - 1, path)
            path.pop()

    go(0, p, K, [(0, p)])
    return out


def base(board, path):
    return sum(board[r][c] for r, c in path)


def total(board, path, X, Y):
    """Base score plus both bonuses, straight from the definitions in the statement."""
    v = [board[r][c] for r, c in path]
    equal = sum(1 for t in range(1, len(v)) if v[t - 1] == v[t])
    rising = sum(1 for t in range(2, len(v)) if v[t - 2] < v[t - 1] < v[t])
    return sum(v) + X * equal + Y * rising


def brute(board, p, K, X=0, Y=0):
    """(max base score, smallest optimal path, number of optimal paths mod MOD, max score with bonuses)."""
    every = paths(board, p, K)
    best = max(base(board, q) for q in every)
    optimal = [q for q in every if base(board, q) == best]
    return best, min(optimal), len(optimal) % MOD, max(total(board, q, X, Y) for q in every)


# ---------------------------------------------------------------- an independent DP for big boards

def forward(board, p, K):
    """(max base score, number of optimal paths mod MOD) by a forward DP over (row, column, special moves used),
    filled from the start cell downwards. Checked against brute() on random boards while writing the judge."""
    N, M = len(board), len(board[0])
    K = min(K, (N - 1) // 2)
    best = [[[None] * (K + 1) for _ in range(M)] for _ in range(N)]
    ways = [[[0] * (K + 1) for _ in range(M)] for _ in range(N)]
    best[0][p][0], ways[0][p][0] = board[0][p], 1

    def push(i, j, u, score, count):
        if best[i][j][u] is None or score > best[i][j][u]:
            best[i][j][u], ways[i][j][u] = score, count
        elif score == best[i][j][u]:
            ways[i][j][u] = (ways[i][j][u] + count) % MOD

    for i in range(N - 1):
        for j in range(M):
            for u in range(K + 1):
                s = best[i][j][u]
                if s is None:
                    continue
                c = ways[i][j][u]
                for nj in (j - 1, j, j + 1):
                    if 0 <= nj < M:
                        push(i + 1, nj, u, s + board[i + 1][nj], c)
                if u < K and i + 2 <= N - 1:
                    push(i + 2, j, u + 1, s + board[i + 2][j], c)
    ends = [(best[N - 1][j][u], ways[N - 1][j][u]) for j in range(M) for u in range(K + 1)
            if best[N - 1][j][u] is not None]
    top = max(s for s, _ in ends)
    return top, sum(c for s, c in ends if s == top) % MOD


def forward_bonus(board, p, K, X, Y):
    """Max base score plus bonuses by a forward DP over (row, column, special moves used, whether the cell
    visited before this one has a smaller value)."""
    N, M = len(board), len(board[0])
    K = min(K, (N - 1) // 2)
    best = [[[[None, None] for _ in range(K + 1)] for _ in range(M)] for _ in range(N)]
    best[0][p][0][0] = board[0][p]
    for i in range(N - 1):
        for j in range(M):
            here = board[i][j]
            for u in range(K + 1):
                for rose in (0, 1):
                    s = best[i][j][u][rose]
                    if s is None:
                        continue
                    targets = [(i + 1, nj, u) for nj in (j - 1, j, j + 1) if 0 <= nj < M]
                    if u < K and i + 2 <= N - 1:
                        targets.append((i + 2, j, u + 1))
                    for a, b, w in targets:
                        there = board[a][b]
                        up = int(here < there)
                        score = s + there + (X if here == there else 0) + (Y if rose and up else 0)
                        cell = best[a][b][w]
                        if cell[up] is None or score > cell[up]:
                            cell[up] = score
    return max(s for j in range(M) for u in range(K + 1) for s in best[N - 1][j][u] if s is not None)


def smallest_path(board, p, K):
    """The lexicographically smallest optimal path. exact[i][j][r] is the best score from (i, j) to row N - 1
    using exactly r special moves; the walk from (0, p) takes, at each cell, the smallest next cell from which
    the rest can still reach the optimum with the special moves left."""
    N, M = len(board), len(board[0])
    K = min(K, (N - 1) // 2)
    exact = [[[None] * (K + 1) for _ in range(M)] for _ in range(N)]
    for j in range(M):
        exact[N - 1][j][0] = board[N - 1][j]
    for i in range(N - 2, -1, -1):
        for j in range(M):
            for r in range(K + 1):
                options = [exact[i + 1][nj][r] for nj in (j - 1, j, j + 1) if 0 <= nj < M]
                if r > 0 and i + 2 <= N - 1:
                    options.append(exact[i + 2][j][r - 1])
                options = [x for x in options if x is not None]
                if options:
                    exact[i][j][r] = board[i][j] + max(options)

    def rest(i, j, left):
        return max(x for x in exact[i][j][: left + 1] if x is not None)

    path, i, j, left = [(0, p)], 0, p, K
    while i < N - 1:
        goal = rest(i, j, left) - board[i][j]
        nxt = [(i + 1, nj, left) for nj in (j - 1, j, j + 1) if 0 <= nj < M]
        if left > 0 and i + 2 <= N - 1:
            nxt.append((i + 2, j, left - 1))
        i, j, left = next(s for s in nxt if any(x is not None for x in exact[s[0]][s[1]][: s[2] + 1])
                          and rest(*s) == goal)
        path.append((i, j))
    return path


# ---------------------------------------------------------------- checking a returned path

def path_problem(board, p, K, path):
    """None if path is a legal path for (board, p, K), else a Chinese sentence saying what is wrong."""
    N, M = len(board), len(board[0])
    if not path:
        return "返回的路径是空的"
    if path[0] != (0, p):
        return f"路径应当从起点 (0, {p}) 出发，返回的第一个格子是 {path[0]}"
    jumps = 0
    for t in range(1, len(path)):
        (i, j), (a, b) = path[t - 1], path[t]
        if i == N - 1:
            return f"路径第一次到达最后一行（第 {N - 1} 行）就停止了，{path[t - 1]} 之后不应当还有格子"
        if a == i + 1 and abs(b - j) <= 1 and 0 <= b < M:
            continue
        if a == i + 2 and b == j and a <= N - 1:
            jumps += 1
            continue
        return f"从 {path[t - 1]} 到 {path[t]} 既不是普通移动，也不是合法的特殊移动"
    if path[-1][0] != N - 1:
        return f"路径应当走到最后一行（第 {N - 1} 行），返回的路径停在 {path[-1]}"
    if jumps > K:
        return f"路径用了 {jumps} 次特殊移动，超过了 K = {K}"
    return None


# ---------------------------------------------------------------- random boards

POOLS = [[-3, -2, -1, 0, 0, 1, 1, 2, 3], [0, 0, 1], [4, 4, 4, -4], list(range(-9, 10)), [-1, 0, 1, 2]]


def random_board(r, N, M, wide=False):
    """Small value pools make tied paths and equal neighbours common; wide draws from the full range."""
    if wide:
        return [[r.randint(-10 ** 9, 10 ** 9) for _ in range(M)] for _ in range(N)]
    pool = r.choice(POOLS)
    return [[r.choice(pool) for _ in range(M)] for _ in range(N)]


def random_small(r):
    """(board, p, K) small enough to enumerate every path."""
    N, M = r.randint(1, 8), r.randint(1, 4)
    board = random_board(r, N, M, wide=r.random() < 0.15)
    return board, r.randrange(M), min(N, r.choice([0, 0, 1, 1, 2, 3, r.randint(0, N)]))   # the statement allows K <= N
