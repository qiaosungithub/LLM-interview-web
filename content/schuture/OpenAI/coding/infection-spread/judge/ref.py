"""Models of the statement for tests.py, written from its rules and sharing no code with the page's answer.

The plain versions rewrite the whole grid every day, exactly as the statement describes the synchronous update.
The *_fast versions follow only the infected cells and are used for the big inputs; tests.py checks the two agree.

The grid builders make the big inputs of the tests, so a failing big case can be rebuilt in 自定义运行:
    from ref import snake
    print(days_until_all_infected(snake(31, 31)))
"""
import random

HEALTHY, INFECTED, IMMUNE, DEAD = 0, 1, 2, 3


def _size(grid):
    return len(grid), (len(grid[0]) if grid else 0)


def _neighbours(rows, cols, r, c):
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


# ---------------------------------------------------------------- Parts 1-2

def spread_days(grid):
    """The first day with no healthy cell, or -1: one whole-grid rewrite per day."""
    rows, cols = _size(grid)
    state = [list(row) for row in grid]
    day = 0
    while any(v == HEALTHY for row in state for v in row):
        new = [row[:] for row in state]
        for r in range(rows):
            for c in range(cols):
                if state[r][c] == HEALTHY and any(state[nr][nc] == INFECTED
                                                  for nr, nc in _neighbours(rows, cols, r, c)):
                    new[r][c] = INFECTED
        if new == state:
            return -1
        state, day = new, day + 1
    return day


def spread_days_fast(grid):
    """Same answer, day by day over the cells infected the day before."""
    rows, cols = _size(grid)
    state = [list(row) for row in grid]
    healthy = sum(v == HEALTHY for row in state for v in row)
    fresh = [(r, c) for r in range(rows) for c in range(cols) if state[r][c] == INFECTED]
    day = 0
    while healthy:
        new = []
        for r, c in fresh:
            for nr, nc in _neighbours(rows, cols, r, c):
                if state[nr][nc] == HEALTHY:
                    state[nr][nc] = INFECTED
                    new.append((nr, nc))
        if not new:
            return -1
        healthy -= len(new)
        fresh, day = new, day + 1
    return day


# ---------------------------------------------------------------- Parts 3-4

def outbreak(grid, D, spread=1, death=None):
    """(first day with no infected cell, number of cells that died): one whole-grid pass per phase per day."""
    rows, cols = _size(grid)
    state = [list(row) for row in grid]
    since = [[0 if v == INFECTED else None for v in row] for row in state]
    doomed = [[False] * cols for _ in range(rows)]
    day = deaths = 0
    while any(v == INFECTED for row in state for v in row):
        day += 1
        for r in range(rows):                   # phase 1: recovery
            for c in range(cols):
                if state[r][c] == INFECTED and day - since[r][c] >= D:
                    state[r][c] = DEAD if doomed[r][c] else IMMUNE
                    deaths += doomed[r][c]
        before = [row[:] for row in state]      # phase 2: spread, counted on the grid after phase 1
        for r in range(rows):
            for c in range(cols):
                if before[r][c] != HEALTHY:
                    continue
                n = sum(before[nr][nc] == INFECTED for nr, nc in _neighbours(rows, cols, r, c))
                if n >= spread:
                    state[r][c], since[r][c] = INFECTED, day
                    doomed[r][c] = death is not None and n >= death
    return day, deaths


def outbreak_fast(grid, D, spread=1, death=None):
    """Same answer, looking only at the infected cells and their neighbours."""
    rows, cols = _size(grid)
    state = [list(row) for row in grid]
    infected = {(r, c): (0, False) for r in range(rows) for c in range(cols) if state[r][c] == INFECTED}
    day = deaths = 0
    while infected:
        day += 1
        for cell, (t0, dies) in list(infected.items()):
            if day - t0 >= D:
                del infected[cell]
                state[cell[0]][cell[1]] = DEAD if dies else IMMUNE
                deaths += dies
        count = {}
        for r, c in infected:
            for nb in _neighbours(rows, cols, r, c):
                if state[nb[0]][nb[1]] == HEALTHY:
                    count[nb] = count.get(nb, 0) + 1
        for (r, c), n in count.items():
            if n >= spread:
                state[r][c] = INFECTED
                infected[(r, c)] = (day, death is not None and n >= death)
    return day, deaths


# ---------------------------------------------------------------- Part 5

def burn(grid, line, index):
    """A copy of grid with row or column `index` (line is "row" or "col") set to dead."""
    return [[DEAD if (r if line == "row" else c) == index else v for c, v in enumerate(row)]
            for r, row in enumerate(grid)]


def burn_options(grid):
    """(description, burnt grid, cells burnt) for burning nothing, each row and each column."""
    rows, cols = _size(grid)
    yield "什么都不烧", [list(row) for row in grid], 0
    for r in range(rows):
        yield f"烧掉第 {r} 行", burn(grid, "row", r), cols
    for c in range(cols):
        yield f"烧掉第 {c} 列", burn(grid, "col", c), rows


def fewest_deaths(grid, D, death, spread=1, fast=False):
    """(smallest total deaths, the choice that reaches it) over the R + C + 1 choices."""
    run = outbreak_fast if fast else outbreak
    return min((cost + run(g, D, spread, death)[1], what) for what, g, cost in burn_options(grid))


# ---------------------------------------------------------------- grids

def random_grid(rows, cols, seed, infected=0.1, immune=0.0, sources=None):
    """Each cell immune with probability `immune`, else infected with probability `infected`; with sources=k,
    exactly k cells (not immune where possible) are infected instead."""
    r = random.Random(seed)
    grid = [[IMMUNE if r.random() < immune else HEALTHY for _ in range(cols)] for _ in range(rows)]
    if sources is None:
        for row in grid:
            for c, v in enumerate(row):
                if v == HEALTHY and r.random() < infected:
                    row[c] = INFECTED
    else:
        cells = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == HEALTHY]
        for i, j in r.sample(cells, min(sources, len(cells))):
            grid[i][j] = INFECTED
    return grid


def snake(rows, cols):
    """A corridor walled by immune cells that winds through the whole grid, infected at its start (0, 0): every
    odd row is immune except one gap, at the right end and the left end in turn."""
    grid = [[HEALTHY] * cols for _ in range(rows)]
    for r in range(1, rows, 2):
        grid[r] = [IMMUNE] * cols
        grid[r][cols - 1 if r % 4 == 1 else 0] = HEALTHY
    grid[0][0] = INFECTED
    return grid
