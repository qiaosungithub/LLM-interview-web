# expect: WA
# Wrong for Part 2: the merge moves a right-half element first when it is <= the left one, so equal elements are counted too.
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
            while j < hi and nums[order[j]] <= nums[order[i]]:     # bug: <= also moves equal elements first
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
    rank = {v: r for r, v in enumerate(sorted(set(nums)))}
    seen, counts = Fenwick(len(rank)), [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        counts[i] = seen.prefix_sum(rank[nums[i]])
        seen.add(rank[nums[i]])
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
