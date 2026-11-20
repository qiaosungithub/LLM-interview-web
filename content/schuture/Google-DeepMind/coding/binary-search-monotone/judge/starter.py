from collections.abc import Callable


def bitonic_min_max(a: list[int]) -> tuple[int, int]:
    """Part 1: (min(a), max(a)) of a bitonic list: the minimum in O(1), the maximum in O(log n) comparisons."""
    raise NotImplementedError


def inverse(f: Callable[[int], int], y: int) -> int | None:
    """Part 2: the positive integer x with f(x) == y, or None if no positive integer maps to y."""
    raise NotImplementedError


def inverse_floor(f: Callable[[int], int], y: int) -> int | None:
    """Part 2: the largest positive integer x with f(x) <= y, or None if even f(1) > y."""
    raise NotImplementedError


def bitonic_max_with_plateaus(a: list[int]) -> int:
    """Part 3(a): the maximum of a non-strictly bitonic list (plateaus allowed on either slope)."""
    raise NotImplementedError


def bisect_root(g: Callable[[float], float], y: float, lo: float, hi: float, eps: float) -> float:
    """Part 3(b): an x with |x - x*| <= eps, where g(x*) == y; must terminate in floating point even when eps == 0."""
    raise NotImplementedError
