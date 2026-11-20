"""Helpers for tests.py: brute-force references, random inputs, and the big workloads (built at import time)."""
import heapq
import math


# ---------------------------------------------------------------- Part 3 references

def bellman_ford(adj, source):
    """Shortest distances by repeated relaxation of every edge: slow, but obviously right."""
    n = len(adj)
    dist = [math.inf] * n
    dist[source] = 0
    for _ in range(n):
        changed = False
        for u in range(n):
            if dist[u] < math.inf:
                for v, w in adj[u]:
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        changed = True
        if not changed:
            break
    return dist


def dijkstra_heapq(adj, source):
    """A fast reference for the big graph (heapq is fine here: this is the judge, not the answer)."""
    dist = [math.inf] * len(adj)
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def random_graph(r, n, m, low, high):
    """n nodes, m random edges (self-loops and parallel edges included), integer weights in [low, high]."""
    adj = [[] for _ in range(n)]
    for _ in range(m):
        adj[r.randrange(n)].append((r.randrange(n), r.randint(low, high)))
    return adj


def copy_graph(adj):
    return [list(edges) for edges in adj]


# ---------------------------------------------------------------- key names for the operation sequences

def key_names(count):
    """count distinct short key names: a..z, then k0, k1, ..."""
    if count <= 26:
        return [chr(ord("a") + i) for i in range(count)]
    return [f"k{i}" for i in range(count)]


# ---------------------------------------------------------------- big workloads

def heap_workload(r, n):
    """Part 1: n random items to build from, then n rounds of push + pop, then pop to empty. Returns
    (init, pushes, wanted pops) where the pops are what any correct heap returns (ties are equal ints)."""
    init = [r.randrange(10**9) for _ in range(n)]
    pushes = [r.randrange(10**9) for _ in range(n)]
    h = list(init)
    heapq.heapify(h)
    pops = []
    for x in pushes:
        heapq.heappush(h, x)
        pops.append(heapq.heappop(h))
    while h:
        pops.append(heapq.heappop(h))
    return init, pushes, pops


def pq_workload(r, n):
    """Part 2: push n keys, then n rounds of pop / push a new key / update a random key (and every 4th round
    remove a random key), then pop to empty. All priorities are distinct, so every result is determined.
    Returns a list of (op, want) with op = ("push", k, p) | ("pop",) | ("update", k, p) | ("remove", k)."""
    prios = iter(r.sample(range(10**9), 3 * n + 10))
    model, heap = {}, []
    live, where = [], {}                       # live keys, for an O(1) random choice and removal

    def add_live(k):
        where[k] = len(live)
        live.append(k)

    def drop_live(k):
        i = where.pop(k)
        last = live.pop()
        if i < len(live):
            live[i] = last
            where[last] = i

    def smallest():
        while True:
            p, k = heap[0]
            if model.get(k) == p:
                return k, p
            heapq.heappop(heap)

    out = []

    def push(k):
        p = next(prios)
        model[k] = p
        heapq.heappush(heap, (p, k))
        add_live(k)
        out.append((("push", k, p), None))

    def pop():
        k, p = smallest()
        heapq.heappop(heap)
        del model[k]
        drop_live(k)
        out.append((("pop",), (k, p)))

    for k in range(n):
        push(k)
    nxt = n
    for rnd in range(n):
        pop()
        push(nxt)
        nxt += 1
        k, p = live[r.randrange(len(live))], next(prios)
        model[k] = p
        heapq.heappush(heap, (p, k))
        out.append((("update", k, p), None))
        if rnd % 4 == 0:
            k = live[r.randrange(len(live))]
            out.append((("remove", k), model.pop(k)))
            drop_live(k)
    while model:
        pop()
    return out
