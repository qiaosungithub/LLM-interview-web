"""Helpers for tests.py: answers that only follow single parent pointers, tree generators, and the big inputs."""
import bisect


# ---------------------------------------------------------------- brute force

def is_tree(parent):
    """True when parent describes a rooted tree: one -1, every other entry a node id, every node reaches it."""
    n = len(parent)
    if sum(1 for p in parent if p == -1) != 1:
        return False
    if any(p != -1 and not 0 <= p < n for p in parent):
        return False
    for v in range(n):
        x, steps = v, 0
        while parent[x] != -1:
            x = parent[x]
            steps += 1
            if steps > n:
                return False
    return True


def up_chain(parent, v):
    """v, its parent, ..., the root."""
    out = [v]
    while parent[out[-1]] != -1:
        out.append(parent[out[-1]])
    return out


def kth(parent, v, k):
    chain = up_chain(parent, v)
    return chain[k] if k < len(chain) else -1


def lca(parent, u, v):
    above_u = set(up_chain(parent, u))
    return next(a for a in up_chain(parent, v) if a in above_u)


def path_weights(parent, weight, u, v):
    """The weights of the edges on the path u - v (each edge is named by its lower node)."""
    a = lca(parent, u, v)
    out = []
    for x in (u, v):
        while x != a:
            out.append(weight[x])
            x = parent[x]
    return out


def distance(parent, u, v):
    return len(path_weights(parent, [0] * len(parent), u, v))


def path_max(parent, weight, u, v):
    ws = path_weights(parent, weight, u, v)
    return max(ws) if ws else None


# ---------------------------------------------------------------- generators

SHAPES = ["random", "path", "star", "binary", "caterpillar", "thin"]


def relabel(par, r):
    """par is written over positions 0..n-1 with position 0 the root; returns it with shuffled labels."""
    n = len(par)
    label = list(range(n))
    r.shuffle(label)
    parent = [0] * n
    for i in range(n):
        parent[label[i]] = -1 if par[i] == -1 else label[par[i]]
    return parent


def random_tree(r, n, shape):
    """A parent array of the given shape with shuffled labels, so a parent's id is often larger."""
    if shape == "random":
        par = [-1] + [r.randrange(i) for i in range(1, n)]
    elif shape == "path":
        par = [-1] + [i - 1 for i in range(1, n)]
    elif shape == "star":
        par = [-1] + [0] * (n - 1)
    elif shape == "binary":
        par = [-1] + [(i - 1) // 2 for i in range(1, n)]
    elif shape == "caterpillar":                  # a spine 0, 2, 4, ... with one leaf hanging off each
        par = [-1] + [i - 1 if i % 2 else i - 2 for i in range(1, n)]
        par[1:2] = [0] * min(1, n - 1)
    else:                                         # "thin": long and narrow, parents 1..3 positions back
        par = [-1] + [max(0, i - 1 - r.randrange(3)) for i in range(1, n)]
    return relabel(par, r)


def comb(r, spine, teeth):
    """A spine of `spine` nodes from the root, each with a tooth (a path) of 0..teeth nodes hanging off it."""
    par = [-1] + [i - 1 for i in range(1, spine)]
    for s in range(spine):
        top = s
        for _ in range(r.randint(0, teeth)):
            par.append(top)
            top = len(par) - 1
    return relabel(par, r)


# ---------------------------------------------------------------- the big inputs (built at import time)

N = 200_000
STEP = 7919                                       # coprime to N, so i -> i * STEP % N is a relabelling
POS = [i * STEP % N for i in range(N)]            # POS[i] is the label of position i


def chain_parent():
    """A path of N nodes; the node at depth i is POS[i]."""
    parent = [0] * N
    for i in range(N):
        parent[POS[i]] = POS[i - 1] if i else -1
    return parent


def arms_parent():
    """Two arms of N / 2 nodes meeting at the root: position i hangs under position max(0, i - 2), so odd
    positions form one arm and even positions the other. Position i has depth (i + 1) // 2."""
    parent = [0] * N
    for i in range(N):
        parent[POS[i]] = POS[max(0, i - 2)] if i else -1
    return parent


def range_max_offline(values, queries):
    """For each (i, j) with i < j: max(values[i + 1 .. j]). One pass with a decreasing stack."""
    by_right = {}
    for q, (i, j) in enumerate(queries):
        by_right.setdefault(j, []).append((i, q))
    out = [None] * len(queries)
    stack = []                                    # positions whose value beats everything after them so far
    for x, val in enumerate(values):
        while stack and values[stack[-1]] <= val:
            stack.pop()
        stack.append(x)
        for i, q in by_right.get(x, ()):
            out[q] = values[stack[bisect.bisect_right(stack, i)]]
    return out
