def shortest_path(grid: list[str]) -> int:
    """Part 1: the number of moves on a shortest path from S to T, or -1 if T is unreachable."""
    raise NotImplementedError


def count_shortest_paths(grid: list[str], mod: int = 1_000_000_007) -> int:
    """Part 2: the number of distinct shortest paths from S to T, modulo mod; 0 if T is unreachable."""
    raise NotImplementedError


def shortest_path_weighted(grid: list[str]) -> int:
    """Part 3: the least total cost from S to T when entering mud (~) costs 2, or -1 if T is unreachable."""
    raise NotImplementedError


def count_shortest_paths_weighted(grid: list[str], mod: int = 1_000_000_007) -> int:
    """Part 3: the number of distinct least-cost paths from S to T, modulo mod."""
    raise NotImplementedError
