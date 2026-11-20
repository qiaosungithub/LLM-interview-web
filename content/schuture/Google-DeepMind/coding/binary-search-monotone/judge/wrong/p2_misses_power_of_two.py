# expect: WA
# Part 2: doubles while f(hi) < y, then searches only [hi // 2, hi), so it misses the answer when f(hi) == y at a power of 2.
def bitonic_min_max(a):
    lo, hi = 0, len(a) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < a[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return min(a[0], a[-1]), a[lo]


def inverse_floor(f, y):
    if f(1) > y:
        return None
    hi = 1
    while f(hi) < y:
        hi *= 2
    if hi == 1:
        return 1
    lo, hi = hi // 2, hi - 1           # largest x in [lo, hi] with f(x) <= y; f(lo) < y
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if f(mid) <= y:
            lo = mid
        else:
            hi = mid - 1
    return lo


def inverse(f, y):
    x = inverse_floor(f, y)
    return x if x is not None and f(x) == y else None


def bitonic_max_with_plateaus(a):
    return max(a)


def bisect_root(g, y, lo, hi, eps):
    while hi - lo > eps:
        mid = lo + (hi - lo) / 2
        if mid == lo or mid == hi:
            break
        if g(mid) < y:
            lo = mid
        else:
            hi = mid
    return lo
