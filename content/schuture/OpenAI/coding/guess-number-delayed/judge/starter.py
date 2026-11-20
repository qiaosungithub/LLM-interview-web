def find_secret(n: int, check) -> int:
    """Part 1: returns secret, calling check at most 2 * ceil(log2(n)) + 1 times."""
    raise NotImplementedError


def find_secret_batched(n: int, check_batch) -> int:
    """Part 2: returns secret, calling check_batch at most 2 * ceil(log(n, 3)) + 1 times."""
    raise NotImplementedError


def solve_games(bounds: dict[str, int], check_round) -> dict[str, int]:
    """Part 3: returns every game's secret, calling check_round at most 2 * ceil(log2(max(bounds.values()))) + 1 times."""
    raise NotImplementedError
