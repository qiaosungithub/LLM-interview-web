# expect: WA
# Wrong for Part 2: when the right run is used up first, the left elements still waiting are copied without
# adding the whole right run to their counts.
from bisect import bisect_left


def count_smaller_brute(nums):
    return [sum(1 for y in nums[i + 1:] if y < x) for i, x in enumerate(nums)]


def count_smaller_merge(nums):
    # bottom-up iterative merge sort over indices, new lists per level
    n = len(nums)
    counts = [0] * n
    runs = [[i] for i in range(n)]
    while len(runs) > 1:
        nxt = []
        for k in range(0, len(runs) - 1, 2):
            a, b = runs[k], runs[k + 1]
            out = []
            i = j = 0
            la, lb = len(a), len(b)
            while i < la and j < lb:
                if nums[b[j]] < nums[a[i]]:
                    out.append(b[j])
                    j += 1
                else:
                    counts[a[i]] += j
                    out.append(a[i])
                    i += 1
            while i < la:
                out.append(a[i])            # BUG: forgets counts[a[i]] += lb
                i += 1
            out.extend(b[j:])
            nxt.append(out)
        if len(runs) % 2:
            nxt.append(runs[-1])
        runs = nxt
    return counts


def count_smaller_fenwick(nums):
    vals = sorted(set(nums))
    m = len(vals)
    tree = [0] * (m + 1)
    res = [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        r = bisect_left(vals, nums[i])  # ranks 0..r-1 are strictly smaller
        s, p = 0, r
        while p > 0:
            s += tree[p]
            p -= p & -p
        res[i] = s
        p = r + 1
        while p <= m:
            tree[p] += 1
            p += p & -p
    return res


class OnlineSmallerCounter:
    """Iterative (bottom-up) segment tree over [lo, hi]."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        size = 1
        while size < hi - lo + 1:
            size *= 2
        self.size = size
        self.t = [0] * (2 * size)

    def add(self, x):
        if x < self.lo or x > self.hi:
            raise ValueError("out of range")
        t = self.t
        pos = x - self.lo
        # sum over [0, pos)
        s = 0
        l, r = self.size, self.size + pos
        while l < r:
            if l & 1:
                s += t[l]
                l += 1
            if r & 1:
                r -= 1
                s += t[r]
            l >>= 1
            r >>= 1
        p = self.size + pos
        while p:
            t[p] += 1
            p >>= 1
        return s
