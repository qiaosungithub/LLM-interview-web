# expect: TLE
# Part 2: correct, but tries x = 1, 2, 3, ... one at a time: O(m) calls to f instead of O(log m).
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
    x = 1
    while f(x + 1) <= y:
        x += 1
    return x


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
