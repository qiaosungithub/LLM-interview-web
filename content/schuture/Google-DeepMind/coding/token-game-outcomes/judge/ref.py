"""References written from the statement alone, and random graph generators.

- fixpoint_outcomes: the definition of W / L / D iterated literally. W_t (L_t) is the set of positions where the
  mover can force a win (the opponent can force a win) within t moves; W_{t+1} = {v : some successor in L_t},
  L_{t+1} = {v : every successor in W_t}. Both only grow, so they stop changing after at most n + 1 rounds;
  whatever is in neither is a draw. Quadratic, used for small graphs.
- linear_outcomes: the same sets computed with counters in O(n + m), used for the big graphs (cross-checked
  against fixpoint_outcomes while writing this judge).
- token_game: plain minimax over the multiset of token positions, for small Part 3 cases.
- grundy: mex over successors, nodes taken once all their successors are done; used for big Part 3 cases."""
import random
from collections import deque


def succ_lists(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    return adj


def fixpoint_outcomes(n, edges):
    adj = succ_lists(n, edges)
    win, lose = set(), {v for v in range(n) if not adj[v]}
    while True:
        new_win = {v for v in range(n) if any(w in lose for w in adj[v])}
        new_lose = {v for v in range(n) if all(w in win for w in adj[v])}
        if new_win == win and new_lose == lose:
            return ["W" if v in win else "L" if v in lose else "D" for v in range(n)]
        win, lose = new_win, new_lose


def linear_outcomes(n, edges):
    adj = succ_lists(n, edges)
    preds = [[] for _ in range(n)]
    for u, v in edges:
        preds[v].append(u)
    left = [len(a) for a in adj]                     # successors not yet known to be W
    out = ["D"] * n
    todo = deque()
    for v in range(n):
        if not adj[v]:
            out[v] = "L"
            todo.append(v)
    while todo:
        x = todo.popleft()
        for p in preds[x]:
            if out[p] != "D":
                continue
            if out[x] == "L":
                out[p] = "W"
                todo.append(p)
            else:
                left[p] -= 1
                if left[p] == 0:
                    out[p] = "L"
                    todo.append(p)
    return out


def winning_move(n, edges, v, outcome):
    return min((w for u, w in edges if u == v and outcome[w] == "L"), default=None)


def token_game(n, edges, tokens):
    adj = succ_lists(n, edges)
    memo = {}

    def wins(pos):                                   # pos: sorted tuple of token positions
        if pos not in memo:
            memo[pos] = any(not wins(tuple(sorted(pos[:i] + (w,) + pos[i + 1:])))
                            for i, v in enumerate(pos) for w in adj[v])
        return memo[pos]
    return wins(tuple(sorted(tokens)))


def grundy(n, edges):
    adj = succ_lists(n, edges)
    preds = [[] for _ in range(n)]
    for u, v in edges:
        preds[v].append(u)
    waiting = [len(a) for a in adj]
    g = [0] * n
    ready = [v for v in range(n) if not adj[v]]
    while ready:
        v = ready.pop()
        seen = {g[w] for w in adj[v]}
        g[v] = next(x for x in range(len(seen) + 1) if x not in seen)
        for p in preds[v]:
            waiting[p] -= 1
            if waiting[p] == 0:
                ready.append(p)
    return g


def random_dag(r, n, p):
    """Edges between random pairs, always from the earlier to the later node of a hidden random order, so
    node numbers are not a topological order."""
    order = list(range(n))
    r.shuffle(order)
    edges = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n) if r.random() < p]
    r.shuffle(edges)
    return edges


def random_digraph(r, n, p):
    edges = [(u, v) for u in range(n) for v in range(n) if u != v and r.random() < p]
    r.shuffle(edges)
    return edges


def dag_text(n, mult, steps):
    """A big DAG as a Python expression the user can paste: node i of a hidden chain is labelled i * mult % n,
    with edges from i to i + d for each d in steps."""
    return f"[(i * {mult} % {n}, (i + d) * {mult} % {n}) for i in range({n}) for d in {steps!r} if i + d < {n}]"


def dag_edges(n, mult, steps):
    return [(i * mult % n, (i + d) * mult % n) for i in range(n) for d in steps if i + d < n]
