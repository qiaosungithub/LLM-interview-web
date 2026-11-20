# expect: WA
# Wrong for Part 3 (b): keeps the values in a mutable default argument, so every OnlineSmallerCounter shares one
# list and a new counter still sees the values added to earlier ones.
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
                counts[a[i]] += j
                out.append(a[i])
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
    def __init__(self, lo, hi, seen=[]):            # BUG: one list shared by every counter
        self.lo, self.hi, self.seen = lo, hi, seen

    def add(self, x):
        if not self.lo <= x <= self.hi:
            raise ValueError(x)
        k = bisect_left(self.seen, x)
        self.seen.insert(k, x)
        return k
