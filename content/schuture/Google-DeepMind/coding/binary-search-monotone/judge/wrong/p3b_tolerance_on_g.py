# expect: WA
# Part 3(b): stops once |g(mid) - y| <= eps, a tolerance on g rather than on x; wrong when g is flat.
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
    while True:
        mid = lo + (hi - lo) / 2
        g_mid = g(mid)
        if abs(g_mid - y) <= eps or mid == lo or mid == hi:
            return mid
        if g_mid < y:
            lo = mid
        else:
            hi = mid
