def min_cost(c: list[int], G: int) -> int:
    """Part 1: the minimum cost of a valid plan (frames 0 and n - 1 are keyframes, every gap is at most G)."""
    raise NotImplementedError


def min_cost_limited(c: list[int], G: int, H: int) -> int:
    """Part 2: the minimum cost when, in addition, two adjacent gaps may not both be long (greater than H)."""
    raise NotImplementedError


def count_optimal(c: list[int], G: int, H: int) -> tuple[int, int]:
    """Part 3: (minimum cost, number of plans with that cost modulo 1_000_000_007) under the Part 2 rules."""
    raise NotImplementedError
