# expect: WA
# Wrong: Part 2 by memoised DFS that answers "D" for a node still on the stack and caches whatever follows from
# that, so a node on a cycle can be stored as "D" although the cycle has an exit that decides it.
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
    for u, v in edges:
        adj[u].append(v)
    state: list[str | None] = [None] * n      # None: not visited, "?": on the stack, else the outcome

    def solve(v: int) -> str:
        if state[v] == "?":
            return "D"                         # back to a node on the stack: the game can go on forever
        if state[v] is None:
            state[v] = "?"
            results = [solve(w) for w in adj[v]]
            if "L" in results:
                state[v] = "W"
            elif all(r == "W" for r in results):
                state[v] = "L"
            else:
                state[v] = "D"
        return state[v]

    for v in range(n):
        solve(v)
    return state


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
