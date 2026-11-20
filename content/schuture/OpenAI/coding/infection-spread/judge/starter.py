def days_until_all_infected(grid: list[list[int]]) -> int:
    """grid[r][c] is 0 (healthy) or 1 (infected); from Part 2 on also 2 (immune). Returns the day, or -1."""
    raise NotImplementedError


def days_until_outbreak_ends(grid: list[list[int]], D: int) -> int:
    """grid holds 0, 1 or 2, and D >= 1. Returns the first day on which no cell is infected."""
    raise NotImplementedError


def simulate(grid: list[list[int]], recover_after: int, spread_threshold: int = 1,
             death_threshold: int | None = None) -> tuple[int, int]:
    """recover_after is D. spread_threshold is the K of version A. death_threshold is the K of
    version B, or None when nobody dies. Returns (day on which the outbreak ends, number of dead cells)."""
    raise NotImplementedError


def min_deaths(grid: list[list[int]], recover_after: int, death_threshold: int,
               spread_threshold: int = 1) -> int:
    """Returns the smallest number of dead cells over the R + C + 1 choices."""
    raise NotImplementedError
