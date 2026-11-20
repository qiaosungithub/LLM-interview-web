# expect: WA
# Wrong for Parts 2 and 3: reduces modulo mod only when two ways into a state are added, so the start's count of 1 and the sum over T's key sets are returned unreduced.
import heapq

KEYS = "abcdef"
DOORS = "ABCDEF"


def _parse(grid):
    R, C = len(grid), len(grid[0])
    flat = "".join(grid)
    return R, C, flat, flat.index("S"), flat.index("T")


def _neighbours(R, C, flat, pos, mask):
    r, c = divmod(pos, C)
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < R and 0 <= nc < C:
            q = nr * C + nc
            ch = flat[q]
            if ch == "#":
                continue
            k = DOORS.find(ch)
            if k >= 0 and not mask >> k & 1:
                continue
            k = KEYS.find(ch)
            yield q, (mask | 1 << k) if k >= 0 else mask, 2 if ch == "~" else 1


def _layers(grid, mod):
    """Frontier-by-frontier search over (cell, mask) states: each frontier is a dict state -> number of ways."""
    R, C, flat, s, t = _parse(grid)
    seen = {(s, 0)}
    frontier = {(s, 0): 1}
    steps = 0
    while frontier:
        hits = [w for (p, m), w in frontier.items() if p == t]
        if hits:
            return steps, sum(hits)
        nxt = {}
        for (p, m), w in frontier.items():
            for q, nm, _ in _neighbours(R, C, flat, p, m):
                if (q, nm) in seen:
                    continue
                nxt[(q, nm)] = (nxt[(q, nm)] + w) % mod if (q, nm) in nxt else w
        seen.update(nxt)
        frontier = nxt
        steps += 1
    return -1, 0


def shortest_path(grid):
    return _layers(grid, 1 << 62)[0]


def count_shortest_paths(grid, mod=1_000_000_007):
    return _layers(grid, mod)[1]


def _dijkstra(grid, mod):
    R, C, flat, s, t = _parse(grid)
    best = {(s, 0): 0}
    ways = {(s, 0): 1}
    done = set()
    heap = [(0, s, 0)]
    answer, total = -1, 0
    while heap:
        d, p, m = heapq.heappop(heap)
        if (p, m) in done:
            continue
        if answer >= 0 and d > answer:
            break
        done.add((p, m))
        if p == t:
            answer = d
            total = total + ways[(p, m)]
            continue
        for q, nm, w in _neighbours(R, C, flat, p, m):
            nd = d + w
            old = best.get((q, nm))
            if old is None or nd < old:
                best[(q, nm)] = nd
                ways[(q, nm)] = ways[(p, m)]
                heapq.heappush(heap, (nd, q, nm))
            elif nd == old:
                ways[(q, nm)] = (ways[(q, nm)] + ways[(p, m)]) % mod
    return answer, total


def shortest_path_weighted(grid):
    return _dijkstra(grid, 1 << 62)[0]


def count_shortest_paths_weighted(grid, mod=1_000_000_007):
    return _dijkstra(grid, mod)[1]
