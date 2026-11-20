# expect: WA
# Version A keeps each healthy cell's count of infected neighbours across days instead of counting afresh each day.
from collections import Counter, deque

HEALTHY, INFECTED, IMMUNE, DEAD = 0, 1, 2, 3
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))      # NOTE: add the four diagonals for the 8-neighbour variant


def days_until_all_infected(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    reached = [[cell != HEALTHY for cell in row] for row in grid]      # NOTE: the input grid is not modified
    frontier = deque((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == INFECTED)
    healthy = sum(cell == HEALTHY for row in grid for cell in row)

    days = 0
    while frontier and healthy:
        days += 1
        for _ in range(len(frontier)):          # NOTE: exactly one BFS level per day
            r, c = frontier.popleft()
            for dr, dc in STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not reached[nr][nc]:
                    reached[nr][nc] = True      # immune cells start as reached, so they are never entered
                    healthy -= 1
                    frontier.append((nr, nc))
    return days if healthy == 0 else -1         # NOTE: healthy cells left over are walled off -> -1

def simulate(grid, recover_after, spread_threshold=1, death_threshold=None):
    """Returns (day on which the outbreak ends, number of dead cells)."""
    if not grid or not grid[0]:
        return 0, 0
    rows, cols = len(grid), len(grid[0])
    state = [row[:] for row in grid]
    # NOTE: the cells infected in the input carry a timer too, with t0 = 0; they recover on day D
    infected_on = {(r, c): 0 for r in range(rows) for c in range(cols) if state[r][c] == INFECTED}
    doomed = set()
    day = deaths = 0

    pressure = Counter()                        # healthy cell -> infected neighbours counted so far
    while infected_on:
        day += 1
        # Phase 1: recovery.  NOTE: it comes BEFORE the spread of the same day, and the test is >=, not >
        for cell in [cell for cell, since in infected_on.items() if day - since >= recover_after]:
            del infected_on[cell]
            r, c = cell
            state[r][c] = DEAD if cell in doomed else IMMUNE
            deaths += cell in doomed

        # Phase 2: spread.  NOTE: count first, apply afterwards. Writing into `state` while counting
        # would let a cell infected today infect its neighbours today.
        for r, c in infected_on:                # NOTE: only infected cells count; immune and dead ones do not
            for dr, dc in STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and state[nr][nc] == HEALTHY:
                    pressure[(nr, nc)] += 1
        for (r, c), count in pressure.items():
            if count >= spread_threshold and state[r][c] == HEALTHY:
                state[r][c] = INFECTED
                infected_on[(r, c)] = day
                if death_threshold is not None and count >= death_threshold:
                    doomed.add((r, c))          # decided now, carried out when the D days are over
    return day, deaths


def days_until_outbreak_ends(grid, D):          # Part 3 is the engine with threshold 1 and no deaths
    return simulate(grid, D)[0]

def min_deaths(grid, recover_after, death_threshold, spread_threshold=1):
    """Part 5: the best of burning nothing and burning one whole row or column on day 0."""
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    lines = [[(r, c) for c in range(cols)] for r in range(rows)]
    lines += [[(r, c) for r in range(rows)] for c in range(cols)]
    best = simulate(grid, recover_after, spread_threshold, death_threshold)[1]   # burning nothing
    for line in lines:
        burnt = [row[:] for row in grid]
        for r, c in line:
            burnt[r][c] = DEAD                  # NOTE: a burnt cell counts as a death whatever it was
        rest = simulate(burnt, recover_after, spread_threshold, death_threshold)[1]
        best = min(best, len(line) + rest)      # NOTE: the line itself already costs len(line) deaths
    return best
