# expect: TLE
# Correct but too slow for Part 3: a plain count array instead of a Fenwick tree, so every query sums a prefix of
# it, O(m) per element in (a) and O(hi - lo) per add in (b) -- the page's "why not just a count array" follow-up.
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
    rank = {v: r for r, v in enumerate(sorted(set(nums)))}
    cnt, counts = [0] * len(rank), [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        r = rank[nums[i]]
        counts[i] = sum(cnt[:r])
        cnt[r] += 1
    return counts


class OnlineSmallerCounter:
    def __init__(self, lo, hi):
        self.lo, self.hi, self.cnt = lo, hi, [0] * (hi - lo + 1)

    def add(self, x):
        if not self.lo <= x <= self.hi:
            raise ValueError(x)
        pos = x - self.lo
        smaller = sum(self.cnt[:pos])
        self.cnt[pos] += 1
        return smaller
