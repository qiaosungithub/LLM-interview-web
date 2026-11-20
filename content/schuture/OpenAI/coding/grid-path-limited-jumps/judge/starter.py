def max_score(board: list[list[int]], p: int, K: int) -> int:
    """Part 1: the largest base score of a path from (0, p) to row N - 1 using at most K special moves."""
    raise NotImplementedError


def optimal_path(board: list[list[int]], p: int, K: int) -> list[tuple[int, int]]:
    """Part 2: the lexicographically smallest cell list among the paths that attain the maximum base score."""
    raise NotImplementedError


def count_optimal_paths(board: list[list[int]], p: int, K: int) -> int:
    """Part 3: the number of distinct maximum-base-score paths, modulo 10**9 + 7."""
    raise NotImplementedError


def max_score_with_bonuses(board: list[list[int]], p: int, K: int, X: int, Y: int) -> int:
    """Part 4: the largest base score plus equal-neighbour (X) and rising-triple (Y) bonuses over all paths."""
    raise NotImplementedError
