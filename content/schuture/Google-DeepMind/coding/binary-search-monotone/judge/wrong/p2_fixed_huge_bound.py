# expect: TLE
# Part 2: binary search on the fixed range [1, 2**100] instead of doubling: calls 2 ** x near x = 2**99 (hangs) and misses answers above 2**100.
from collections.abc import Callable


def bitonic_min_max(a: list[int]) -> tuple[int, int]:
    # ceil-mid backward comparison: lo is always a candidate peak
    lo, hi = 0, len(a) - 1
    while lo != hi:
        mid = -(-(lo + hi) // 2)
        if a[mid - 1] < a[mid]:
            lo = mid
        else:
            hi = mid - 1
    first, last = a[0], a[len(a) - 1]
    return (first if first < last else last, a[lo])


def _floor(f, y):
    one = f(1)
    if one > y:
        return None, None
    # gallop: step grows, lo always has f(lo) <= y
    lo, flo = 1, one
    hi = 1 << 100  # "big enough" fixed bound
    while hi - lo > 1:
        mid = (lo + hi) // 2
        fm = f(mid)
        if fm <= y:
            lo, flo = mid, fm
        else:
            hi = mid
    return lo, flo


def inverse(f: Callable[[int], int], y: int) -> int | None:
    x, fx = _floor(f, y)
    if x is None or fx != y:
        return None
    return x


def inverse_floor(f: Callable[[int], int], y: int) -> int | None:
    return _floor(f, y)[0]


def bitonic_max_with_plateaus(a: list[int]) -> int:
    best = a[0]
    for v in a:
        if v > best:
            best = v
    return best


def bisect_root(g: Callable[[float], float], y: float, lo: float, hi: float, eps: float) -> float:
    lo, hi = float(lo), float(hi)
    for _ in range(5000):
        if hi - lo <= eps:
            break
        mid = 0.5 * lo + 0.5 * hi
        if not (lo < mid < hi):
            break
        if g(mid) >= y:
            hi = mid
        else:
            lo = mid
    return 0.5 * lo + 0.5 * hi if eps > 0 else (lo if abs(g(lo) - y) <= abs(g(hi) - y) else hi)
