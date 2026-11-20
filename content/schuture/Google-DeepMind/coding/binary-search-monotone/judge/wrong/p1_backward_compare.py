# expect: TLE
# Part 1 (the page's pitfall): floor-rounded mid compared backward with a[mid - 1]; lo = mid never moves once hi == lo + 1.
def bitonic_min_max(a):
    lo, hi = 0, len(a) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if mid == 0 or a[mid] > a[mid - 1]:
            lo = mid
        else:
            hi = mid - 1
    return min(a[0], a[-1]), a[lo]


def inverse_floor(f, y):
    if f(1) > y:
        return None
    lo, hi = 1, 1
    while f(hi) <= y:
        lo, hi = hi, hi * 2
    while hi - lo > 1:                 # f(lo) <= y < f(hi)
        mid = (lo + hi) // 2
        if f(mid) <= y:
            lo = mid
        else:
            hi = mid
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
