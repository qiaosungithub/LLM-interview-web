def outcomes_dag(n: int, edges: list[tuple[int, int]]) -> list[str]:
    """Part 1: the outcome ("W" or "L") of a token on each node of an acyclic graph, without recursion."""
    raise NotImplementedError


def winning_move(n: int, edges: list[tuple[int, int]], v: int) -> int | None:
    """Part 1: the smallest successor w of v whose outcome is "L", or None if v itself is "L"."""
    raise NotImplementedError


def outcomes(n: int, edges: list[tuple[int, int]]) -> list[str]:
    """Part 2: the outcome ("W", "L" or "D") of a token on each node of a graph that may have cycles."""
    raise NotImplementedError


def first_player_wins(n: int, edges: list[tuple[int, int]], tokens: list[int]) -> bool:
    """Part 3: whether the first player can force a win with tokens on the given nodes of an acyclic graph."""
    raise NotImplementedError
