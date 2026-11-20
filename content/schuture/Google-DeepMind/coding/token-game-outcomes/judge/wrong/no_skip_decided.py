# expect: WA
# Wrong: Part 2 drops the page's "a label, once assigned, never changes" check; a "W" node reached from two "L"
# successors is queued twice and lowers its predecessors' counters twice.
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
    # NOTE: reverse topological order -- sinks first -- so every successor of v is already filled in by
    # the time v itself is evaluated; an explicit order rather than recursion keeps the call stack at O(1)
    for v in reversed(_topo_order(n, edges)):
        outcome[v] = "W" if any(outcome[w] == "L" for w in adj[v]) else "L"
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
    grundy = [0] * n
    for v in reversed(_topo_order(n, edges)):
        seen = {grundy[w] for w in adj[v]}
        g = 0
        while g in seen:        # NOTE: mex(seen) takes at most len(seen) + 1 steps -- by pigeonhole, len(seen)
            g += 1               # distinct non-negative integers cannot cover every value in 0 .. len(seen)
        grundy[v] = g
    xor_sum = 0
    for t in tokens:
        xor_sum ^= grundy[t]
    return xor_sum != 0
