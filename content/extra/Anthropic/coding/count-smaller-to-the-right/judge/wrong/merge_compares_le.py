# expect: RE
# Wrong for Part 2: the textbook merge (take the left element when left <= right) gets the counts right, but compares
# elements with <=, which the statement forbids (only < is allowed), so it breaks on elements that only support <.
def count_smaller_brute(nums):
    return [sum(1 for y in nums[i + 1:] if y < x) for i, x in enumerate(nums)]


def count_smaller_merge(nums):
    n = len(nums)
    counts = [0] * n
    order = list(range(n))

    def sort(lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        sort(lo, mid)
        sort(mid, hi)
        merged, i, j = [], lo, mid
        while i < mid and j < hi:
            if nums[order[i]] <= nums[order[j]]:
                counts[order[i]] += j - mid
                merged.append(order[i])
                i += 1
            else:
                merged.append(order[j])
                j += 1
        while i < mid:
            counts[order[i]] += j - mid
            merged.append(order[i])
            i += 1
        merged.extend(order[j:hi])
        order[lo:hi] = merged

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
