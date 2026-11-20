"""A brute force written from the statement alone, used to compute the expected answers of the random tests,
plus random grid generators.

brute(grid, weighted) counts walks directly: it keeps, for every exact cost c, how many cell sequences of cost c
end in each (cell, keys held) pair, and stops at the first cost at which some sequence reaches T. A walk that
passes T earlier would have stopped earlier, so every sequence counted there is a path in the statement's sense.
No visited set, no tie handling: the count is a plain sum over sequences."""

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def find(grid, ch):
    for r, row in enumerate(grid):
        c = row.find(ch)
        if c >= 0:
            return r, c
    raise AssertionError(ch)


def moves(grid, r, c, keys):
    """Every legal move out of cell (r, c) holding `keys` (a frozenset of letters): (nr, nc, new keys, cost)."""
    R, C = len(grid), len(grid[0])
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < R and 0 <= nc < C):
            continue
        ch = grid[nr][nc]
        if ch == "#":
            continue
        if ch in "ABCDEF" and ch.lower() not in keys:
            continue
        yield nr, nc, (keys | {ch} if ch in "abcdef" else keys), (2 if ch == "~" else 1)


def brute(grid, weighted=False):
    """(least cost, exact number of least-cost paths), or (-1, 0) when T cannot be reached."""
    start, target = find(grid, "S"), find(grid, "T")
    s0 = (start[0], start[1], frozenset())
    seen, stack = {s0}, [s0]                     # plain reachability first, so an unreachable T terminates
    while stack:
        r, c, keys = stack.pop()
        if (r, c) == target:
            continue
        for nr, nc, nk, _ in moves(grid, r, c, keys):
            if (nr, nc, nk) not in seen:
                seen.add((nr, nc, nk))
                stack.append((nr, nc, nk))
    if not any((r, c) == target for r, c, _ in seen):
        return -1, 0
    layers = {0: {s0: 1}}                        # cost -> {(r, c, keys): number of sequences}
    cost = 0
    while True:
        layer = layers.pop(cost, {})
        hit = sum(n for (r, c, _), n in layer.items() if (r, c) == target)
        if hit:
            return cost, hit
        for (r, c, keys), n in layer.items():
            for nr, nc, nk, step in moves(grid, r, c, keys):
                nxt = layers.setdefault(cost + (step if weighted else 1), {})
                nxt[(nr, nc, nk)] = nxt.get((nr, nc, nk), 0) + n
        cost += 1


def random_grid(r, rows, cols, letters, wall=0.25, mud=0.0, extra=0):
    """A rows x cols grid with S, T, one key and one door for each of the first `letters` letters, `extra` more
    random keys or doors (repeats and doors without keys allowed), walls, and mud on a share of the floor."""
    cells = [(i, j) for i in range(rows) for j in range(cols)]
    r.shuffle(cells)
    g = [["#" if r.random() < wall else "." for _ in range(cols)] for _ in range(rows)]
    marks = ["S", "T"] + [ch for k in range(letters) for ch in ("abcdef"[k], "ABCDEF"[k])]
    marks += [r.choice("abcdef"[:letters] + "ABCDEF"[:letters + 1]) for _ in range(extra)]
    for (i, j), ch in zip(cells, marks[:len(cells)]):
        g[i][j] = ch
    if mud:
        for i in range(rows):
            for j in range(cols):
                if g[i][j] == "." and r.random() < mud:
                    g[i][j] = "~"
    return ["".join(row) for row in g]
