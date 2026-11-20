def above_threshold(arr: list[int], threshold: int) -> tuple[int, list[int]]:
    """Return how many elements are strictly greater than threshold, and their indices in increasing order."""
    raise NotImplementedError


def count_above_many(arr: list[int], thresholds: list[int]) -> list[int]:
    """For each threshold, in order, count the elements strictly greater than it, in O((n + q) log n) total."""
    raise NotImplementedError


def build_graph(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Return every node of the undirected graph with its neighbours sorted in increasing order."""
    raise NotImplementedError


def dfs_order(edges: list[tuple[str, str]], start: str) -> list[str]:
    """Return the DFS preorder from start, smallest unvisited neighbour first, without deep recursion."""
    raise NotImplementedError


def components(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Return each connected component as its DFS order from its smallest node, ordered by that node."""
    raise NotImplementedError


def has_cycle(edges: list[tuple[str, str]]) -> bool:
    """Return whether the graph contains a cycle of three or more distinct nodes."""
    raise NotImplementedError
