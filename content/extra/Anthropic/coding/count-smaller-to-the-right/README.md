# Count Smaller to the Right

## Problem

You are given a list of integers `nums` of length $n$. Return a list `counts` of the same length, where `counts[i]` is the number of indices `j` with
`j > i` and `nums[j] < nums[i]`, that is, how many elements to the right of `nums[i]` are smaller than it.

Conventions:

- “Smaller” means **strictly less than**. Elements equal to `nums[i]` do not count.
- $0 \le n \le 2 \times 10^5$. Elements are arbitrary Python integers; they can be negative, can have absolute value up to $10^{18}$, and can repeat.
- Do not modify the `nums` you are given. If `nums` is empty, return `[]`.

```text
[5, 2, 6, 1]        -> [2, 1, 1, 0]
# to the right of 5 are 2 and 1; to the right of 2 is 1; to the right of 6 is 1; nothing is to the right of 1

[3, 3, 1, 3, 1, 2]  -> [3, 3, 0, 2, 0, 0]
# to the right of the first 3 are 3, 1, 3, 1, 2: only 1, 1, 2 are strictly smaller; the two 3s don't count

[4, 4, 4, 2, 4]     -> [1, 1, 1, 0, 0]
[2, 0, 1]           -> [2, 0, 0]
[-1, -1]            -> [0, 0]
[7]                 -> [0]
[]                  -> []
```

### Part 1 — The brute-force solution and its cost

Start with the most direct solution.

```py
def count_smaller_brute(nums: list[int]) -> list[int]:
    """counts[i] = number of j > i with nums[j] < nums[i]."""
```

Then answer: exactly how many element comparisons does it make (as an exact expression in $n$)? How many is that at $n = 2 \times 10^5$, and roughly how long would it take in pure
Python?

### Part 2 — Merge sort

Use the idea behind merge sort to write a solution that runs in $O(n \log n)$ in the worst case. Elements may only be compared with `<`,
so that the same code also works on floats or strings.

```py
def count_smaller_merge(nums: list[int]) -> list[int]:
    """Same result as count_smaller_brute, in O(n log n) time. Compares elements with < only."""
```

You should be able to explain two things: at which step each pair `(i, j)` gets counted, and why duplicate elements are not overcounted.

### Part 3 — Fenwick tree, and how to choose between the two solutions

(a) Write another $O(n \log n)$ solution with a Fenwick tree (also called a binary indexed tree). Elements can be very large
and can be negative, so you cannot use values directly as indices; do coordinate compression first.

```py
def count_smaller_fenwick(nums: list[int]) -> list[int]:
    """Same result as count_smaller_brute, with a Fenwick tree over the compressed values."""
```

(b) Now the data arrives one number at a time. As each number arrives, you must immediately answer “how many of the numbers that arrived earlier are strictly smaller than it?”. The value range is known
in advance: every number lies in `[lo, hi]`, with `lo <= hi` and `hi - lo < 10**6`.

```py
class OnlineSmallerCounter:
    def __init__(self, lo: int, hi: int) -> None:
        """All values passed to add() lie in [lo, hi]."""

    def add(self, x: int) -> int:
        """Returns how many of the values added before this call are strictly smaller than x,
        then records x. Raises ValueError (and records nothing) if x is outside [lo, hi].
        O(log(hi - lo + 1)) per call."""
```

```text
c = OnlineSmallerCounter(-2, 5)
c.add(3)   -> 0
c.add(-2)  -> 0
c.add(3)   -> 1      # the earlier arrivals are 3, -2: only -2 is strictly smaller than 3
c.add(5)   -> 3
c.add(0)   -> 1
c.add(6)   -> ValueError
```

Feeding it `nums` from right to left gives exactly this problem's `counts` (in reverse order).

(c) Answer verbally: when is each of the two solutions, merge sort and Fenwick tree, the better fit? At a minimum, cover whether the input has to be available all at once,
which line of code is responsible for duplicate elements, and how much extra space each needs.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Before writing any code, confirm two things with the interviewer: whether equal elements count (here they don't), and the range of values (here they can be large and negative,
so Part 3 needs coordinate compression). Each of these changes one line of the code.

### Part 1

Two nested loops: for each `i`, scan the elements to its right.

```python
def count_smaller_brute(nums: list[int]) -> list[int]:
    n = len(nums)
    counts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if nums[j] < nums[i]:               # strictly smaller: equal elements are not counted
                counts[i] += 1
    return counts
```

Each pair `i < j` is compared exactly once, so the number of comparisons is exactly $n(n-1)/2$, regardless of the input; time is $O(n^2)$, and the only extra space
is the output. At $n = 2 \times 10^5$ that is $19\,999\,900\,000$ comparisons, about $2 \times 10^{10}$. A loop like this in pure Python
does roughly $10^7$ comparisons per second (an order of magnitude only; it varies by machine), so it would take around half an hour, which is unacceptable.

The brute force is useful as the “answer key”: both of the later solutions are cross-checked against it on small random arrays.

### Part 2

When merge sort merges two halves, every element of the left half was originally to the **left** of every element of the right half. So the counting can be done during the merge:
when an element of the left half is placed into the output, the right-half elements that have already been placed are exactly the right-half elements smaller than it.

Concretely, we sort indices rather than values: `order` holds indices, sorted by `nums[index]`, so that each count can be added back to
`counts[original index]`. When merging `order[lo:mid]` and `order[mid:hi]`, with left pointer `i` and right pointer `j`:

- as long as the current right element is **strictly smaller** than the current left element, place the right one first and advance `j`;
- otherwise place the left element. At this point `j - mid` elements of the right half have been placed, and they are exactly all the right-half elements strictly smaller than it;
  add that to its `counts`.

Answers to the two questions:

- **At which step each pair is counted.** For any `i < j`, exactly one merge in the recursion splits them into the left and right halves (the level where they are first
  separated). Before that they are in the same segment and have not yet been compared against each other; after that they are in the same merged segment
  and never end up on opposite sides again. So each pair is considered exactly once.
- **Why duplicates are not overcounted.** On a tie, the inner `<` is false, so the left element is placed first. The equal element on the right has not been placed yet,
  so it is not included in `j - mid`. In other words, the word “strictly” lives entirely in this one `<`; this is also what makes the merge stable (left first on ties).
  Write it as `<=` and you get “the number of elements to the right that are less than or equal”.

```python
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
```

The loop ends once the left half has been placed: the remaining right-half elements are all no smaller than the maximum of the left half and are already in the correct position, so they don't need to be moved.

Complexity: in one merge, every comparison advances `i` or `j` by one step, so the number of comparisons is less than the segment length `hi - lo`. The recursion has
$\lceil \log_2 n \rceil$ levels, and the segment lengths on each level add up to at most $n$, so the total number of comparisons is at most
$n \lceil \log_2 n \rceil$, for any input. Time is $O(n \log n)$, extra space is $O(n)$ (`order`, `buf`),
and the recursion depth is $\lceil \log_2 n \rceil$, which is 18 levels at $n = 2 \times 10^5$.

### Part 3

**(a) Fenwick tree.** Think of it from the other direction: scan from right to left, maintaining the set of “elements already scanned”. When the scan reaches `nums[i]`, the question is
how many values in the set are strictly smaller than it; after answering, add it to the set. This needs a structure that supports two operations: increment the count of a value by one;
and query the total count of values smaller than a given value. Once values are laid out in order, the latter is a prefix sum, and a Fenwick tree does both operations in $O(\log m)$.

Values can be very large or negative, so compress first: sort the distinct values that appear; the rank of a value `v` is the number of distinct values smaller than it,
in the range `0..m-1`, where $m$ is the number of distinct values. Equal values have the same rank.

The Fenwick tree uses a 1-based array `tree`, where `tree[p]` stores the sum of the segment that ends at position `p` and has length `p & -p` (the lowest set bit
of `p`). A prefix-sum query repeatedly strips the lowest set bit; a point update repeatedly adds the lowest set bit.

```python
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
```

Duplicate elements are handled by the `prefix_sum(r)` line: it only sums ranks `0..r-1`, not `r` itself. Write `prefix_sum(r + 1)`
and it becomes “less than or equal”.

Complexity: one `prefix_sum` or `add` visits at most $\lfloor \log_2 m \rfloor + 1$ nodes, and there are $2n$ operations in total,
so the work on the tree is $O(n \log m)$. Compression requires a sort, $O(m \log m)$. Altogether, the worst case is $O(n \log n)$. The `set`
and the `rank` dict are hash structures with expected $O(1)$ lookups; if you want to avoid hashing entirely, you can binary search in `values` instead,
at $O(\log m)$ per lookup, without changing the overall complexity. Extra space is $O(m)$.

**(b) Online version.** When the value range is known, no compression is needed; use `x - lo` directly as the position.

```python
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
```

Each `add` is $O(\log R)$, where $R = hi - lo + 1$; space is $O(R)$, independent of how many numbers have arrived.

**(c) How to choose.**

| | Merge sort | Fenwick tree |
| --- | --- | --- |
| Must the input be available all at once? | Yes. There is no answer until the whole array is sorted | Not when the value range is known: answer each number as it arrives. When the range is unknown, you must first collect all values to compress them, which makes it offline too |
| Requirement on elements | Only that they can be compared with `<` | They must map to integer positions `0..m-1` |
| What handles duplicates | The `<` in the merge (ties place the left one first, i.e. stability) | The query boundary `prefix_sum(r)` |
| Extra space | $O(n)$ | $O(m)$ or $O(R)$, less when there are many duplicates |
| Worst-case time | $O(n \log n)$ | $O(n \log n)$ |

For an array given all at once, either works; in an interview, merge sort doesn't require writing a separate data structure, so it is easier to get right on the first try. If the data is streaming, or
you later need to support “remove a number” or “how many are less than an arbitrary threshold”, use the Fenwick tree: removal is just `add(pos, -1)`,
which merge sort cannot do. In both versions there is exactly one place where duplicates can go wrong, so after writing the code, test it specifically on examples with duplicates.

### Follow-ups

**Q: How do you compute the total number of inversions in the array?** An inversion is a pair with `i < j` and `nums[i] > nums[j]`, so the total is exactly
`sum(counts)`. If you only need the total, the merge doesn't need to carry indices: sort the values directly and accumulate `j - mid`.

**Q: What if the condition becomes `nums[i] > 2 * nums[j]`?** Now the “condition used for counting” and the “order used for sorting” are no longer the same comparison,
so you can't count while merging. The fix is to split each merge into two passes: with both halves sorted, first count with two pointers (for each element of the left half in increasing order,
the right pointer only moves right), then do an ordinary merge. The complexity is still $O(n \log n)$. With a Fenwick tree, the query position becomes
“how many distinct values are strictly smaller than `nums[i] / 2`”, found by binary search in the sorted `values`.

**Q: What if data streams in, but the value range is not known in advance?** A Fenwick tree needs the range of positions up front, so it won't work. Options:
a balanced search tree with subtree sizes, or a skip list, at $O(\log n)$ per operation; a dynamically allocated segment tree that creates nodes on demand, at $O(\log R)$ per operation;
or, simplest of all, a sorted list where you find the position with `bisect` and then insert. That last one finds the position in $O(\log n)$, but insertion has to shift
elements, $O(n)$ per insert in the worst case and $O(n^2)$ in total; the constant factor is very small, though, so it is good enough when the data is not large.

**Q: Why not just allocate an array of counts indexed by value?** Two reasons. The values go up to $10^{18}$, so such an array won't fit; hence compression.
After compression, with only a plain count array, each query has to sum all the counts before it, which is $O(m)$, $O(nm)$ in total; with few duplicates
that is as slow as brute force. The Fenwick tree brings that sum down to $O(\log m)$.

**Q: Could the recursion get too deep?** No. Each level splits a segment in half, so the depth is $\lceil \log_2 n \rceil$; $2 \times 10^5$
elements need only 18 levels. The approaches that really can recurse too deeply are those that partition by value (such as a degenerate quicksort); merge sort doesn't have this problem.

<details>
<summary>Checks (runnable)</summary>

```python
import bisect
import math
import random
import time

# ---- every example in the statement ----
EXAMPLES = [
    ([5, 2, 6, 1], [2, 1, 1, 0]),
    ([3, 3, 1, 3, 1, 2], [3, 3, 0, 2, 0, 0]),
    ([4, 4, 4, 2, 4], [1, 1, 1, 0, 0]),
    ([2, 0, 1], [2, 0, 0]),
    ([-1, -1], [0, 0]),
    ([7], [0]),
    ([], []),
]
SOLVERS = (count_smaller_brute, count_smaller_merge, count_smaller_fenwick)


def reference(nums):
    """An independent brute force: shares no code with the solutions above."""
    return [sum(1 for y in nums[i + 1:] if y < x) for i, x in enumerate(nums)]


for nums, want in EXAMPLES:
    assert reference(nums) == want, nums
    for solve in SOLVERS:
        before = list(nums)
        assert solve(nums) == want, (solve.__name__, nums)
        assert nums == before                       # the input is not modified

c = OnlineSmallerCounter(-2, 5)
assert [c.add(x) for x in (3, -2, 3, 5, 0)] == [0, 0, 1, 3, 1]
for bad in (6, -3):
    try:
        c.add(bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
assert c.add(5) == 4                                # the rejected values were not recorded


# ---- random arrays against the independent brute force ----
def online(nums, lo, hi):
    counter = OnlineSmallerCounter(lo, hi)
    return [counter.add(x) for x in reversed(nums)][::-1]


rng = random.Random(20261117)
checked = 0
for trial in range(4000):
    n = rng.choice([0, 1, 2, 3, 5, 8, 13, 40])
    span = rng.choice([0, 1, 3, 10, 1000, 10**18])   # small spans force many duplicates
    nums = [rng.randint(-span, span) for _ in range(n)]
    want = reference(nums)
    for solve in SOLVERS:
        assert solve(nums) == want, (solve.__name__, nums)
    if span <= 1000:
        assert online(nums, -span, span) == want
    checked += 1
assert checked == 4000


# ---- operation counts instead of wall-clock time ----
class Counted:
    """Wraps a value and counts every < comparison made on it."""
    comparisons = 0
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        Counted.comparisons += 1
        return self.value < other.value


def comparisons(solve, values):
    wrapped = [Counted(v) for v in values]
    Counted.comparisons = 0
    result = solve(wrapped)
    return Counted.comparisons, result


for n in (0, 1, 2, 7, 300, 600):
    values = [rng.randint(-50, 50) for _ in range(n)]
    used, result = comparisons(count_smaller_brute, values)
    assert used == n * (n - 1) // 2                 # exactly n(n-1)/2, whatever the input
    assert result == reference(values)
assert 200_000 * 199_999 // 2 == 19_999_900_000     # the figure quoted in Part 1

merge_counts = {}
for n in (1, 2, 3, 1000, 4096, 8192, 16384):
    for shape in ("random", "sorted", "reversed", "constant", "few values"):
        values = {
            "random": [rng.randint(-10**9, 10**9) for _ in range(n)],
            "sorted": list(range(n)),
            "reversed": list(range(n, 0, -1)),
            "constant": [5] * n,
            "few values": [rng.randint(0, 3) for _ in range(n)],
        }[shape]
        used, result = comparisons(count_smaller_merge, values)
        bound = n * math.ceil(math.log2(n)) if n > 1 else 0
        assert used <= bound, (n, shape, used)
        assert result == count_smaller_fenwick(values)
        merge_counts[n, shape] = used
# doubling n a bit more than doubles the work: n log n growth, nowhere near the 4x of the brute force
for n in (4096, 8192):
    ratio = merge_counts[2 * n, "random"] / merge_counts[n, "random"]
    assert 2.0 < ratio < 2.3, ratio
assert merge_counts[16384, "random"] > 0.8 * 16384 * 14              # and it really is about n log2 n


class CountingList(list):
    touches = 0

    def __getitem__(self, index):
        CountingList.touches += 1
        return list.__getitem__(self, index)


class CountingFenwick(Fenwick):
    def __init__(self, size):
        super().__init__(size)
        self.tree = CountingList(self.tree)


PlainFenwick = Fenwick
Fenwick = CountingFenwick                           # count_smaller_fenwick looks the name up when called
try:
    for n, distinct in ((1, 1), (1000, 1), (1000, 7), (4096, 4096), (20000, 300), (20000, 10**9)):
        values = [rng.randrange(distinct) for _ in range(n)]
        m = len(set(values))
        CountingList.touches = 0
        result = count_smaller_fenwick(values)
        # one prefix_sum and one add per element, each visiting at most floor(log2 m) + 1 nodes
        assert CountingList.touches <= 2 * n * (int(math.log2(m)) + 1), (n, m, CountingList.touches)
        assert result == count_smaller_merge(values)
finally:
    Fenwick = PlainFenwick


# ---- the tie rule is exactly where "strictly" lives ----
def count_not_larger_merge(nums):
    """count_smaller_merge with <= in the merge: counts j > i with nums[j] <= nums[i]."""
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
            while j < hi and nums[order[j]] <= nums[order[i]]:
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


differs = 0
for trial in range(500):
    nums = [rng.randint(0, 4) for _ in range(rng.randint(0, 30))]
    loose = [sum(1 for y in nums[i + 1:] if y <= x) for i, x in enumerate(nums)]
    assert count_not_larger_merge(nums) == loose
    differs += loose != reference(nums)
assert differs > 400                                # with duplicates the two answers really differ
assert count_not_larger_merge([3, 3, 1, 3, 1, 2]) == [5, 4, 1, 2, 0, 0]


# ---- follow-ups ----
def inversions(nums):
    return sum(1 for i in range(len(nums)) for j in range(i + 1, len(nums)) if nums[i] > nums[j])


def count_double_smaller(nums):
    """counts[i] = number of j > i with nums[i] > 2 * nums[j]: count first, then merge."""
    n = len(nums)
    counts, order = [0] * n, list(range(n))

    def sort(lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        sort(lo, mid)
        sort(mid, hi)
        j = mid
        for i in range(lo, mid):                    # both halves are sorted, so j only moves right
            while j < hi and 2 * nums[order[j]] < nums[order[i]]:
                j += 1
            counts[order[i]] += j - mid
        order[lo:hi] = sorted(order[lo:hi], key=nums.__getitem__)   # stands in for a plain merge

    sort(0, n)
    return counts


def count_smaller_insort(nums):
    """Sorted list + bisect: the simple answer when the value range is not known in advance."""
    seen, counts = [], [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        counts[i] = bisect.bisect_left(seen, nums[i])               # elements strictly smaller
        bisect.insort(seen, nums[i])
    return counts


for trial in range(1000):
    nums = [rng.randint(-20, 20) for _ in range(rng.randint(0, 40))]
    assert sum(count_smaller_merge(nums)) == inversions(nums)
    assert count_double_smaller(nums) == [
        sum(1 for y in nums[i + 1:] if x > 2 * y) for i, x in enumerate(nums)]
    assert count_smaller_insort(nums) == reference(nums)

# the merge works on anything comparable with <
assert count_smaller_merge(["b", "a", "c", "a"]) == [2, 0, 1, 0]
assert count_smaller_merge([2.5, -1.0, 2.5, 0.0]) == [2, 0, 1, 0]
assert math.ceil(math.log2(200_000)) == 18          # the recursion depth quoted above


# ---- the largest allowed input: the fast versions agree with each other and with a third method ----
big = [rng.randint(-10**18, 10**18) for _ in range(150_000)] + [rng.randint(-50, 50) for _ in range(50_000)]
rng.shuffle(big)
start = time.perf_counter()
by_merge = count_smaller_merge(big)
middle = time.perf_counter()
by_fenwick = count_smaller_fenwick(big)
end = time.perf_counter()
assert by_merge == by_fenwick and len(by_merge) == 200_000
# counts[i] depends only on nums[i:], so a suffix can be checked on its own by the third method
assert by_merge[-50_000:] == count_smaller_insort(big[-50_000:])
dense = [rng.randint(0, 999) for _ in range(200_000)]
assert online(dense, 0, 999) == count_smaller_merge(dense)
# informational only: wall-clock time is not asserted on a shared machine
print(f"n = 200000: merge {middle - start:.2f}s, Fenwick {end - middle:.2f}s")
print("all checks passed")
```

</details>

</details>
