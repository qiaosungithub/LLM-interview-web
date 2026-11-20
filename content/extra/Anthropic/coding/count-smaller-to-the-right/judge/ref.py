"""Helpers for tests.py: the definition as a brute force, an independent O(n log n) method for big inputs, and
random input generators."""
import bisect


def brute(nums):
    """counts[i] = number of j > i with nums[j] < nums[i], straight from the definition."""
    return [sum(1 for y in nums[i + 1:] if y < x) for i, x in enumerate(nums)]


def fast(nums):
    """Same result as brute(), in O(n log n): walk from the right, keep a binary indexed tree over the sorted
    distinct values, and count the values already seen that are strictly smaller."""
    values = sorted(set(nums))
    tree = [0] * (len(values) + 1)
    out = [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        k = bisect.bisect_left(values, nums[i])       # number of distinct values below nums[i]
        total, p = 0, k
        while p:
            total += tree[p]
            p &= p - 1
        out[i] = total
        p = k + 1
        while p < len(tree):
            tree[p] += 1
            p += p & -p
    return out


def online_brute(lo, hi, ops):
    """What OnlineSmallerCounter(lo, hi).add(x) returns for each x of ops: a count, or ValueError for a value
    outside [lo, hi] (which is then not recorded)."""
    seen, out = [], []
    for x in ops:
        if not lo <= x <= hi:
            out.append(ValueError)
        else:
            out.append(sum(1 for y in seen if y < x))
            seen.append(x)
    return out


SPANS = [0, 1, 3, 10, 1000, 10 ** 18]


def random_nums(r, n_lo, n_hi, span=None):
    """A list of n_lo..n_hi integers in [-span, span]; small spans give many duplicates."""
    span = r.choice(SPANS) if span is None else span
    return [r.randint(-span, span) for _ in range(r.randint(n_lo, n_hi))]
