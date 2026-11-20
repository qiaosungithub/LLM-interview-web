"""Helpers for tests.py: independent ways to get each answer, written straight from the statement, plus random
input generators. None of them uses the page's recursion over range minima with every tied plank removed."""


# ---------------------------------------------------------------- Part 1

def row_runs(h):
    """[(y, start, end)] for every maximal run of existing cells in row y (end exclusive), row by row."""
    runs = []
    n = len(h)
    for y in range(max(h, default=0)):
        i = 0
        while i < n:
            if h[i] <= y:
                i += 1
                continue
            j = i
            while j < n and h[j] > y:
                j += 1
            runs.append((y, i, j))
            i = j
    return runs


def horizontal_by_rows(h):
    """Part 1 by counting the runs of every row: one stroke per run is needed and enough. Small heights only."""
    return len(row_runs(h))


def horizontal_by_ends(h):
    """Part 1 for any heights: a run ends at plank i in rows h[i + 1] <= y < h[i] (h[n] = 0), so count run ends."""
    return sum(max(0, a - b) for a, b in zip(h, list(h[1:]) + [0]))


# ---------------------------------------------------------------- Part 3

def brute_min_strokes(h):
    """Tries every set of planks painted vertically; then each row run holding an unpainted plank needs one
    horizontal stroke, and one is enough. Exponential: n <= 10."""
    n = len(h)
    spans = [sum(1 << i for i in range(s, e)) for _, s, e in row_runs(h)]
    best = None
    for mask in range(1 << n):
        cost = bin(mask).count("1") + sum(1 for span in spans if span & ~mask)
        best = cost if best is None else min(best, cost)
    return best


def matching_min_strokes(h):
    """Every cell lies in exactly one row run (its only horizontal stroke worth using) and one plank (its vertical
    stroke), so the answer is a minimum vertex cover of the bipartite graph planks x runs with one edge per cell,
    which by Konig's theorem is the size of a maximum matching. Small heights only (one vertex per row run)."""
    runs = row_runs(h)
    adj = [[] for _ in h]                   # plank -> the runs holding its cells
    for k, (_, s, e) in enumerate(runs):
        for i in range(s, e):
            adj[i].append(k)
    owner = [-1] * len(runs)                # run -> matched plank

    def augment(p, seen):
        for k in adj[p]:
            if not seen[k]:
                seen[k] = True
                if owner[k] == -1 or augment(owner[k], seen):
                    owner[k] = p
                    return True
        return False

    return sum(1 for p in range(len(h)) if augment(p, [False] * len(runs)))


def split_min_strokes(h):
    """Divide and conquer on a single minimum per range (the leftmost one, p), with an explicit stack. The cost
    of the cells of [l, r] at rows >= b is min(r - l + 1, (m - b) + cost([l, p - 1], m) + cost([p + 1, r], m))
    with m = h[p]; when m <= b, plank p has no cell left and the cost is just that of the two sides. Any
    heights; O(n^2) at worst."""
    if not h:
        return 0
    cost = {}                               # (l, r) -> cost; non-empty ranges never repeat in this split tree
    stack = [(0, len(h) - 1, 0, False)]
    while stack:
        l, r, b, done = stack.pop()
        m = min(h[l:r + 1])
        p = h.index(m, l, r + 1)
        sides = [(sl, sr) for sl, sr in ((l, p - 1), (p + 1, r)) if sl <= sr]
        if not done:
            stack.append((l, r, b, True))
            stack.extend((sl, sr, max(m, b), False) for sl, sr in sides)
            continue
        inner = sum(cost.pop(s) for s in sides)
        cost[(l, r)] = inner if m <= b else min(r - l + 1, (m - b) + inner)
    return cost[(0, len(h) - 1)]


# ---------------------------------------------------------------- generators

def random_heights(r, n_max, h_max, zeros=0.15):
    n = r.randint(0, n_max)
    kind = r.randrange(4)
    if kind == 0:                                           # anything
        hs = [r.randint(0, h_max) for _ in range(n)]
    elif kind == 1:                                         # few distinct heights: plateaus and ties
        levels = [r.randint(0, h_max) for _ in range(r.randint(1, 3))]
        hs = [r.choice(levels) for _ in range(n)]
    elif kind == 2:                                         # a hill or a valley
        hs = sorted(r.randint(0, h_max) for _ in range(n))
        k = r.randint(0, n)
        hs = hs[:k] + sorted(hs[k:], reverse=True)
        if r.random() < 0.5:
            hs = [h_max - x for x in hs]
    else:                                                   # thin spikes on a low base
        hs = [r.randint(0, 2) if r.random() < 0.6 else r.randint(0, h_max) for _ in range(n)]
    return [0 if r.random() < zeros else x for x in hs]
