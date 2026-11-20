# expect: RE
# Wrong: evaluates outcomes and Grundy numbers by memoised recursion along the edges; the call stack grows with the
# path length, which the statement rules out for n up to 10^5 (RecursionError on a long chain).
from collections import deque


def _topo_order(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Kahn's algorithm: returns the nodes of an acyclic graph in a topological order (every edge u -> v
    has u appear before v). Iterative, so the call stack never grows with the size of the graph."""
    adj: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(u for u in range(n) if indeg[u] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    assert len(order) == n     # NOTE: fewer than n nodes emitted would mean a cycle; ruled out by the statement
    return order


def outcomes_dag(n: int, edges: list[tuple[int, int]]) -> list[str]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    outcome = [""] * n

    def solve(v):
        if not outcome[v]:
            win = False
            for w in adj[v]:
                if solve(w) == "L":
                    win = True
            outcome[v] = "W" if win else "L"
        return outcome[v]
    for v in range(n):
        solve(v)
    return outcome


def winning_move(n: int, edges: list[tuple[int, int]], v: int) -> int | None:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, w in edges:
        adj[u].append(w)
    outcome = outcomes_dag(n, edges)    # NOTE: this is O(n + m) on its own -- call it once and reuse the
    for w in sorted(adj[v]):            # array when checking many nodes, instead of recomputing it per call
        if outcome[w] == "L":
            return w
    return None

def outcomes(n: int, edges: list[tuple[int, int]]) -> list[str]:
    adj: list[list[int]] = [[] for _ in range(n)]
    pred: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        pred[v].append(u)
    unresolved = [len(succs) for succs in adj]   # successors of v not yet known to be "W"
    label = [None] * n
    q = deque(v for v in range(n) if not adj[v])    # sinks: the mover there has no move at all
    for v in q:
        label[v] = "L"
    while q:
        x = q.popleft()
        for p in pred[x]:
            if label[p] is not None:
                continue        # NOTE: a label, once assigned, never changes -- skip an already-decided p
            if label[x] == "L":
                label[p] = "W"       # p can move to x and hand the opponent a loss
                q.append(p)
            else:
                unresolved[p] -= 1
                if unresolved[p] == 0:
                    label[p] = "L"   # every successor of p now leads to a win for whoever moves there
                    q.append(p)
    return [label[v] if label[v] is not None else "D" for v in range(n)]

def first_player_wins(n: int, edges: list[tuple[int, int]], tokens: list[int]) -> bool:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    grundy = [None] * n

    def mex_of(v):
        if grundy[v] is None:
            seen = {mex_of(w) for w in adj[v]}
            g = 0
            while g in seen:
                g += 1
            grundy[v] = g
        return grundy[v]
    for v in range(n):
        mex_of(v)
    xor_sum = 0
    for t in tokens:
        xor_sum ^= grundy[t]
    return xor_sum != 0
