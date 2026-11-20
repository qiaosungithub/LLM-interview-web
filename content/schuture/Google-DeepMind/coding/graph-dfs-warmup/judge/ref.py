"""Helpers for tests.py: each answer restated straight from the statement in a different way from the page
(brute-force counting, a sweep instead of bisect, a DFS that keeps the current path and steps to the smallest
unvisited neighbour, union-find for components, E > V - C for cycles), plus random input generators."""


# ---------------------------------------------------------------- Part 1

def above_threshold(arr, threshold):
    idx = []
    for i in range(len(arr)):
        if arr[i] > threshold:
            idx.append(i)
    return len(idx), idx


def count_above_brute(arr, thresholds):
    return [sum(1 for x in arr if x > t) for t in thresholds]


def count_above_sweep(arr, thresholds):
    """Thresholds visited from the largest down while a pointer walks the sorted array from the top."""
    values = sorted(arr)
    out = [0] * len(thresholds)
    k = len(values)                     # values[k:] are the elements > the current threshold
    above = 0
    for j in sorted(range(len(thresholds)), key=lambda j: thresholds[j], reverse=True):
        while k > 0 and values[k - 1] > thresholds[j]:
            k -= 1
            above += 1
        out[j] = above
    return out


# ---------------------------------------------------------------- Part 2

def graph(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, [])
        adj.setdefault(v, [])
        if u != v:
            if v not in adj[u]:
                adj[u].append(v)
            if u not in adj[v]:
                adj[v].append(u)
    return {node: sorted(ns) for node, ns in adj.items()}


def graph_fast(edges):
    """Same as graph(), with sets; for big inputs."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set())
        adj.setdefault(v, set())
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return {node: sorted(ns) for node, ns in adj.items()}


def preorder(adj, start, visited=None):
    """DFS as the statement words it: from the node on top of the current path, step to its smallest unvisited
    neighbour; when it has none, backtrack. Iterative, so any depth works."""
    visited = set() if visited is None else visited
    visited.add(start)
    order = [start]
    path = [(start, 0)]                  # (node, index of the next neighbour to try)
    while path:
        node, k = path[-1]
        ns = adj.get(node, [])
        while k < len(ns) and ns[k] in visited:
            k += 1
        if k == len(ns):
            path.pop()
            continue
        path[-1] = (node, k + 1)
        nxt = ns[k]
        visited.add(nxt)
        order.append(nxt)
        path.append((nxt, 0))
    return order


def dfs_order(edges, start):
    return preorder(graph_fast(edges), start)


# ---------------------------------------------------------------- Part 3

def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def groups(edges):
    """Union-find over the raw edges: the node sets of the components."""
    parent = {}
    for u, v in edges:
        parent.setdefault(u, u)
        parent.setdefault(v, v)
        ru, rv = _find(parent, u), _find(parent, v)
        if ru != rv:
            parent[ru] = rv
    out = {}
    for x in parent:
        out.setdefault(_find(parent, x), []).append(x)
    return list(out.values())


def components(edges):
    adj = graph_fast(edges)
    return [preorder(adj, min(g)) for g in sorted(groups(edges), key=min)]


def has_cycle(edges):
    """A forest with V nodes and C components has exactly V - C edges; any more and some component has a cycle."""
    distinct = {frozenset((u, v)) for u, v in edges if u != v}
    nodes = {x for e in edges for x in e}
    return len(distinct) > len(nodes) - len(groups(edges))


# ---------------------------------------------------------------- generators

def random_edges(r, n_nodes, n_edges, names=None):
    """Random undirected edges with repeats, reversed repeats and self-loops."""
    names = names or [chr(ord("A") + i) for i in range(n_nodes)]
    nodes = r.sample(names, n_nodes)
    edges = []
    for _ in range(n_edges):
        u, v = r.choice(nodes), r.choice(nodes)
        edges.append((u, v))
        if r.random() < 0.2:
            edges.append((v, u))
        if r.random() < 0.1:
            edges.append((u, v))
    r.shuffle(edges)
    return edges


def random_forest(r, nodes, extra=0):
    """A random forest over nodes (some nodes start new trees), plus `extra` edges that close cycles."""
    order = list(nodes)
    r.shuffle(order)
    edges = []
    for i in range(1, len(order)):
        if r.random() < 0.85:
            edges.append((order[i], order[r.randrange(i)]))
    pairs = {frozenset(e) for e in edges}
    comp = groups(edges + [(x, x) for x in order])
    big = [g for g in comp if len(g) >= 3]
    for _ in range(extra):
        if not big:
            break
        g = r.choice(big)
        u, v = r.sample(g, 2)
        if frozenset((u, v)) not in pairs:
            edges.append((u, v))
            pairs.add(frozenset((u, v)))
    edges = [(v, u) if r.random() < 0.5 else (u, v) for u, v in edges]
    edges += [(x, x) for x in order if r.random() < 0.05]
    r.shuffle(edges)
    return edges
