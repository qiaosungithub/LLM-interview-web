# expect: RE
# Wrong for Part 3 (a): no coordinate compression; shifts by the minimum and indexes the tree by value, so a value
# range of 10**18 needs a tree of that size.
def count_smaller_brute(nums):
    return [sum(1 for y in nums[i + 1:] if y < x) for i, x in enumerate(nums)]


def count_smaller_merge(nums):
    n = len(nums)
    counts, order, buf = [0] * n, list(range(n)), [0] * n

    def sort(lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        sort(lo, mid)
        sort(mid, hi)
        i, j, k = lo, mid, lo
        while i < mid:
            while j < hi and nums[order[j]] < nums[order[i]]:
                buf[k] = order[j]
                j += 1
                k += 1
            counts[order[i]] += j - mid
            buf[k] = order[i]
            i += 1
            k += 1
        order[lo:k] = buf[lo:k]

    sort(0, n)
    return counts


class Fenwick:
    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def add(self, pos):
        pos += 1
        while pos < len(self.tree):
            self.tree[pos] += 1
            pos += pos & -pos

    def prefix_sum(self, count):
        total = 0
        while count > 0:
            total += self.tree[count]
            count -= count & -count
        return total


def count_smaller_fenwick(nums):
    if not nums:
        return []
    low = min(nums)
    seen, counts = Fenwick(max(nums) - low + 1), [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        counts[i] = seen.prefix_sum(nums[i] - low)
        seen.add(nums[i] - low)
    return counts


class OnlineSmallerCounter:
    def __init__(self, lo, hi):
        self.lo, self.hi, self.seen = lo, hi, Fenwick(hi - lo + 1)

    def add(self, x):
        if not self.lo <= x <= self.hi:
            raise ValueError(x)
        smaller = self.seen.prefix_sum(x - self.lo)
        self.seen.add(x - self.lo)
        return smaller
