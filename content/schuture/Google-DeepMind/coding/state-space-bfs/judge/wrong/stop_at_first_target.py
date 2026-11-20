# expect: WA
# Wrong for Part 2: stops at the first T state the search dequeues and returns its count, missing T reached at the same distance with other keys.
from collections import deque

def _parse(grid: list[str]) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    R, C = len(grid), len(grid[0])
    start = target = None
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'T':
                target = (r, c)
    return R, C, start, target


def _transitions(grid, R, C, r, c, mask):
    """Yields (nr, nc, nmask) for every legal one-cell move out of state (r, c, mask): edge-adjacent,
    inside the grid, not a wall, and not a locked door (an uppercase letter whose key bit is not yet set
    in mask). Stepping onto a key cell sets its bit in nmask; every other cell leaves mask unchanged."""
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if not (0 <= nr < R and 0 <= nc < C):
            continue
        ch = grid[nr][nc]
        if ch == '#':
            continue
        if ch.isupper() and ch not in ('S', 'T'):
            if not (mask >> (ord(ch) - ord('A'))) & 1:
                continue        # NOTE: door's key bit not held yet -- this move is not legal
            nmask = mask
        elif ch.islower():
            nmask = mask | (1 << (ord(ch) - ord('a')))
        else:
            nmask = mask
        yield nr, nc, nmask


def _bfs(grid):
    """BFS over states (row, col, mask of keys held). Returns (target_cell, dist, ways): dist maps every
    reached state to its distance from S, and ways maps it to the number of distinct minimal-length
    state-sequences reaching it (Part 2). T is terminal -- once a state's cell is T it is never expanded."""
    R, C, start, target = _parse(grid)
    s0 = (start[0], start[1], 0)
    dist = {s0: 0}
    ways = {s0: 1}
    q = deque([s0])
    while q:
        r, c, mask = q.popleft()
        if (r, c) == target:
            continue            # NOTE: T is terminal -- reaching it ends the walk, so it is never expanded
        d = dist[(r, c, mask)]
        for nr, nc, nmask in _transitions(grid, R, C, r, c, mask):
            ns = (nr, nc, nmask)
            if ns not in dist:
                dist[ns] = d + 1
                ways[ns] = ways[(r, c, mask)]
                q.append(ns)
            elif dist[ns] == d + 1:
                ways[ns] += ways[(r, c, mask)]     # NOTE: another shortest way into ns, from a different state
    return target, dist, ways


def shortest_path(grid: list[str]) -> int:
    target, dist, _ways = _bfs(grid)
    lengths = [d for (r, c, _m), d in dist.items() if (r, c) == target]
    return min(lengths) if lengths else -1

def count_shortest_paths(grid: list[str], mod: int = 1_000_000_007) -> int:
    target, dist, ways = _bfs(grid)
    at_target = [(d, ways[s]) for s, d in dist.items() if s[0] == target[0] and s[1] == target[1]]
    if not at_target:
        return 0
    best = min(d for d, _ in at_target)
    return sum(w for d, w in at_target if d == best) % mod    # NOTE: only the states at the minimum distance

import heapq

def _step_cost(ch: str) -> int:
    return 2 if ch == '~' else 1


def _dijkstra(grid):
    R, C, start, target = _parse(grid)
    s0 = (start[0], start[1], 0)
    dist = {s0: 0}
    ways = {s0: 1}
    settled = set()
    heap = [(0, s0)]
    while heap:
        d, state = heapq.heappop(heap)
        if state in settled:
            continue            # NOTE: a stale heap entry -- this state's cost was already finalised
        settled.add(state)      # NOTE: ways[state] is only trustworthy from here, once popped at its least cost
        r, c, mask = state
        if (r, c) == target:
            continue            # NOTE: T is terminal -- never expanded, even once popped
        for nr, nc, nmask in _transitions(grid, R, C, r, c, mask):
            step = _step_cost(grid[nr][nc])
            ns = (nr, nc, nmask)
            nd = d + step
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                ways[ns] = ways[state]
                heapq.heappush(heap, (nd, ns))
            elif nd == dist[ns]:
                ways[ns] += ways[state]    # NOTE: only a tight edge (nd == the current best) may add to ways
    return target, dist, ways


def shortest_path_weighted(grid: list[str]) -> int:
    target, dist, _ways = _dijkstra(grid)
    lengths = [d for (r, c, _m), d in dist.items() if (r, c) == target]
    return min(lengths) if lengths else -1


def count_shortest_paths_weighted(grid: list[str], mod: int = 1_000_000_007) -> int:
    target, dist, ways = _dijkstra(grid)
    at_target = [(d, ways[s]) for s, d in dist.items() if s[0] == target[0] and s[1] == target[1]]
    if not at_target:
        return 0
    best = min(d for d, _ in at_target)
    return sum(w for d, w in at_target if d == best) % mod


def count_shortest_paths(grid: list[str], mod: int = 1_000_000_007) -> int:
    R, C, start, target = _parse(grid)
    s0 = (start[0], start[1], 0)
    dist = {s0: 0}
    ways = {s0: 1}
    q = deque([s0])
    while q:
        state = q.popleft()
        r, c, mask = state
        if (r, c) == target:
            return ways[state] % mod
        d = dist[state]
        for nr, nc, nmask in _transitions(grid, R, C, r, c, mask):
            ns = (nr, nc, nmask)
            if ns not in dist:
                dist[ns] = d + 1
                ways[ns] = ways[state]
                q.append(ns)
            elif dist[ns] == d + 1:
                ways[ns] += ways[state]
    return 0
