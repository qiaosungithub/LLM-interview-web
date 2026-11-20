from collections.abc import Callable


def bitonic_min_max(a: list[int]) -> tuple[int, int]:
    n = len(a)
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < a[mid + 1]:   # NOTE: floor-rounded mid < hi, so a[mid + 1] exists; comparing backward
            lo = mid + 1          #       (a[mid - 1], lo = mid) can loop forever once hi == lo + 1
        else:
            hi = mid
    return min(a[0], a[-1]), a[lo]


def _locate(f: Callable[[int], int], y: int) -> tuple[int, int] | None:
    """Returns (m, f(m)) for m = the largest positive integer with f(m) <= y, or None if even
    f(1) > y. f is strictly increasing. Every value of f computed here is cached and reused, so
    no argument is ever evaluated twice."""
    lo, f_lo = 1, f(1)
    if f_lo > y:
        return None
    hi, f_hi = lo, f_lo
    while f_hi < y:                 # NOTE: doubles with no assumed bound on m -- unlike a [0, y] search,
        lo, f_lo = hi, f_hi         #       this never assumes f(x) >= x (see the discussion below)
        hi *= 2                     # NOTE: a fixed-width language must guard hi and f(hi) against
        f_hi = f(hi)                #       overflow here; Python integers grow as needed
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f_mid = f(mid)
        if f_mid <= y:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return (hi, f_hi) if f_hi <= y else (lo, f_lo)   # NOTE: f_hi is already cached -- no new call to f


def inverse_floor(f: Callable[[int], int], y: int) -> int | None:
    located = _locate(f, y)
    return located[0] if located is not None else None


def inverse(f: Callable[[int], int], y: int) -> int | None:
    located = _locate(f, y)
    if located is None:
        return None
    x, f_x = located
    return x if f_x == y else None


def bitonic_max_with_plateaus(a: list[int]) -> int:
    return max(a)   # NOTE: no shortcut is possible in the worst case -- see the adversary argument above


def bisect_root(g: Callable[[float], float], y: float, lo: float, hi: float, eps: float) -> float:
    while hi - lo > eps:
        mid = lo + (hi - lo) / 2
        if mid == lo or mid == hi:   # NOTE: the bracket cannot shrink further in floating point -- this
            break                    #       is what makes the loop terminate even when eps == 0
        if g(mid) < y:
            lo = mid
        else:
            hi = mid
    return lo
