def count_smaller_brute(nums: list[int]) -> list[int]:
    n = len(nums)
    counts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if nums[j] < nums[i]:               # strictly smaller: equal elements are not counted
                counts[i] += 1
    return counts


def count_smaller_merge(nums: list[int]) -> list[int]:
    n = len(nums)
    counts = [0] * n
    order = list(range(n))                      # indices; sorted by nums[index] piece by piece
    buf = [0] * n                               # scratch space for merging

    def sort(lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        sort(lo, mid)
        sort(mid, hi)
        i, j, k = lo, mid, lo
        while i < mid:
            while j < hi and nums[order[j]] < nums[order[i]]:   # NOTE: strict; ties take the left one
                buf[k] = order[j]
                j += 1
                k += 1
            counts[order[i]] += j - mid         # right-half elements strictly smaller than this one
            buf[k] = order[i]
            i += 1
            k += 1
        order[lo:k] = buf[lo:k]                 # order[k:hi] is already in its final place

    sort(0, n)
    return counts


class Fenwick:
    """Counts over positions 0..size-1: point add, and sum over a prefix."""

    def __init__(self, size: int) -> None:
        self.tree = [0] * (size + 1)            # 1-based internally

    def add(self, pos: int, delta: int = 1) -> None:
        pos += 1
        tree = self.tree
        while pos < len(tree):
            tree[pos] += delta
            pos += pos & -pos                   # next node whose range covers this position

    def prefix_sum(self, count: int) -> int:
        """Sum over positions 0..count-1 (0 if count == 0)."""
        total = 0
        tree = self.tree
        while count > 0:
            total += tree[count]
            count -= count & -count             # strip the lowest set bit
        return total


def count_smaller_fenwick(nums: list[int]) -> list[int]:
    values = sorted(set(nums))                  # coordinate compression
    rank = {v: r for r, v in enumerate(values)}
    seen = Fenwick(len(values))                 # seen[r] = how many elements of rank r lie to the right
    counts = [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        r = rank[nums[i]]
        counts[i] = seen.prefix_sum(r)          # NOTE: ranks 0..r-1 only, so equal values are excluded
        seen.add(r)
    return counts


class OnlineSmallerCounter:
    def __init__(self, lo: int, hi: int) -> None:
        if lo > hi:
            raise ValueError("empty value range")
        self._lo, self._hi = lo, hi
        self._seen = Fenwick(hi - lo + 1)

    def add(self, x: int) -> int:
        if not self._lo <= x <= self._hi:
            raise ValueError(f"value out of range: {x}")
        pos = x - self._lo
        smaller = self._seen.prefix_sum(pos)    # values in [lo, x - 1] seen so far
        self._seen.add(pos)
        return smaller
