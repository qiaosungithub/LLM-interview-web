# 右侧更小元素的个数

## 题目

给定一个整数列表 `nums`，长度记为 $n$。返回一个同样长度的列表 `counts`，其中 `counts[i]` 是满足
`j > i` 且 `nums[j] < nums[i]` 的下标 `j` 的个数，也就是 `nums[i]` 右边有多少个元素比它小。

约定：

- “更小”指**严格小于**。和 `nums[i]` 相等的元素不算。
- $0 \le n \le 2 \times 10^5$。元素是任意 Python 整数，可以为负，绝对值可达 $10^{18}$，可以重复。
- 不能修改传入的 `nums`。`nums` 为空时返回 `[]`。

```text
[5, 2, 6, 1]        -> [2, 1, 1, 0]
# 5 的右边有 2 和 1；2 的右边有 1；6 的右边有 1；1 的右边没有元素

[3, 3, 1, 3, 1, 2]  -> [3, 3, 0, 2, 0, 0]
# 第一个 3 的右边是 3, 1, 3, 1, 2：只有 1, 1, 2 严格小于它，两个 3 不算

[4, 4, 4, 2, 4]     -> [1, 1, 1, 0, 0]
[2, 0, 1]           -> [2, 0, 0]
[-1, -1]            -> [0, 0]
[7]                 -> [0]
[]                  -> []
```

### Part 1 —— 暴力解法和它的代价

先写最直接的解法。

```py
def count_smaller_brute(nums: list[int]) -> list[int]:
    """counts[i] = number of j > i with nums[j] < nums[i]."""
```

然后回答：它一共做多少次元素比较（写成 $n$ 的精确式子）？$n = 2 \times 10^5$ 时是多少次，纯 Python
大概要跑多久？

### Part 2 —— 归并排序

用归并排序（merge sort）的思路写一个最坏情况下 $O(n \log n)$ 的解法。元素之间只允许用 `<` 比较，
这样同一份代码也能用在浮点数或字符串上。

```py
def count_smaller_merge(nums: list[int]) -> list[int]:
    """Same result as count_smaller_brute, in O(n log n) time. Compares elements with < only."""
```

要能说清两件事：每一对 `(i, j)` 是在哪一步被数到的；重复元素为什么不会被多数。

### Part 3 —— 树状数组，以及两种解法怎么选

(a) 用树状数组（Fenwick tree，也叫 binary indexed tree）再写一个 $O(n \log n)$ 的解法。元素可以很大，
也可以是负数，所以不能直接拿值当下标，要先做坐标压缩（coordinate compression）。

```py
def count_smaller_fenwick(nums: list[int]) -> list[int]:
    """Same result as count_smaller_brute, with a Fenwick tree over the compressed values."""
```

(b) 现在数据是一个一个到达的。每到一个数，要立刻回答“之前到达的数里，有多少个严格小于它”。值域事先
知道：所有的数都在 `[lo, hi]` 里，`lo <= hi`，且 `hi - lo < 10**6`。

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
c.add(3)   -> 1      # 之前到达的是 3, -2：只有 -2 严格小于 3
c.add(5)   -> 3
c.add(0)   -> 1
c.add(6)   -> ValueError
```

把 `nums` 从右往左依次喂给它，得到的正好是本题的 `counts`（顺序反过来）。

(c) 口头回答：归并和树状数组这两种解法，分别在什么情况下更合适？至少要说到：输入是不是必须一次给全、
重复元素由哪一行代码负责、各需要多少额外空间。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前先向面试官确认两点：相等的元素算不算（这里不算），以及值的范围（这里可以很大、可以为负，
所以 Part 3 需要坐标压缩）。这两点都会改变代码里的某一行。

### Part 1

两层循环，对每个 `i` 扫一遍它右边的元素。

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

每一对 `i < j` 恰好比较一次，所以比较次数正好是 $n(n-1)/2$，和输入内容无关；时间 $O(n^2)$，额外空间
只有输出。$n = 2 \times 10^5$ 时是 $19\,999\,900\,000$ 次，约 $2 \times 10^{10}$。纯 Python 的这种循环
每秒大约做 $10^7$ 次比较（只是量级，随机器变化），所以要跑半小时左右，不可接受。

暴力解的用处是当“标准答案”：后面两个解法都拿它在随机小数组上对拍。

### Part 2

归并排序在合并两段时，左半段的每个元素原本都在右半段每个元素的**左边**。所以合并的时候顺手就能数：
把左半段的某个元素放进结果时，右半段里已经先被放进去的那些元素，正好是右半段中比它小的元素。

具体做法是不排值，而是排下标：`order` 里存下标，按 `nums[下标]` 排序，这样数出来的个数能加回
`counts[原下标]`。合并 `order[lo:mid]` 和 `order[mid:hi]` 时，左指针 `i`，右指针 `j`：

- 只要右边的当前元素**严格小于**左边的当前元素，就先放右边的，`j` 前进；
- 否则放左边的元素，此时右半段已经放了 `j - mid` 个，它们就是右半段里严格小于它的全部元素，
  加到它的 `counts` 上。

两个问题的答案：

- **每一对在哪一步被数到。** 对任意 `i < j`，递归过程中恰好有一次合并把它们分在左右两半（它们第一次
  被切开的那一层）。在那之前它们在同一段里，还没被比较归属；在那之后它们在同一个已合并的段里，
  再也不会分居左右。所以每一对恰好被考虑一次。
- **重复元素为什么不会多数。** 相等时内层的 `<` 为假，先放左边的元素。这时右边那个相等的元素还没被放，
  不计入 `j - mid`。也就是说，“严格”二字全在这一个 `<` 上；这同时也是稳定归并（相等时左边在前）。
  把它写成 `<=`，算出来的就变成“右侧小于等于的个数”。

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

左半段放完后循环就结束：右半段剩下的元素都不小于左半段的最大值，本来就在正确的位置上，不用搬。

复杂度：一次合并里，每做一次比较，`i` 或 `j` 就前进一步，所以比较次数小于段长 `hi - lo`。递归有
$\lceil \log_2 n \rceil$ 层，每层各段的长度加起来不超过 $n$，所以总比较次数不超过
$n \lceil \log_2 n \rceil$，对任何输入都成立。时间 $O(n \log n)$，额外空间 $O(n)$（`order`、`buf`），
递归深度 $\lceil \log_2 n \rceil$，$n = 2 \times 10^5$ 时是 18 层。

### Part 3

**(a) 树状数组。** 换一个方向想：从右往左扫，维护“已经扫过的元素”的集合。扫到 `nums[i]` 时，要问的是
集合里有多少个值严格小于它，问完再把它放进集合。这需要一个支持两种操作的结构：某个值的计数加一；
查询小于某个值的计数之和。按值排开之后，后者是一个前缀和，树状数组两种操作都是 $O(\log m)$。

值可能很大也可能为负，所以先压缩：把出现过的不同的值排序，值 `v` 的名次（rank）是比它小的不同值
的个数，范围是 `0..m-1`，$m$ 是不同值的个数。相等的值名次相同。

树状数组用下标从 1 开始的数组 `tree`，`tree[p]` 存的是以位置 `p` 结尾、长度为 `p & -p`（`p` 的最低位
的 1）的那一段的和。查前缀和时不断去掉最低位的 1，单点加时不断加上最低位的 1。

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

重复元素由 `prefix_sum(r)` 这一行负责：它只加名次 `0..r-1`，不含 `r` 本身。写成 `prefix_sum(r + 1)`
就变成“小于等于”。

复杂度：一次 `prefix_sum` 或 `add` 最多访问 $\lfloor \log_2 m \rfloor + 1$ 个节点，一共 $2n$ 次操作，
所以树上的工作量是 $O(n \log m)$。压缩要排序，$O(m \log m)$。合起来最坏 $O(n \log n)$。其中 `set`
和 `rank` 字典是哈希结构，查找是期望 $O(1)$；想要完全不依赖哈希，可以改成在 `values` 上二分，
每次 $O(\log m)$，总复杂度不变。额外空间 $O(m)$。

**(b) 在线版本。** 值域已知时不需要压缩，直接用 `x - lo` 当位置。

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

每次 `add` 是 $O(\log R)$，$R = hi - lo + 1$；空间 $O(R)$，和到达了多少个数无关。

**(c) 怎么选。**

| | 归并 | 树状数组 |
| --- | --- | --- |
| 输入要不要一次给全 | 要。整个数组排完才有答案 | 值域已知时不用，来一个答一个；值域未知时要先收集所有值做压缩，也就成了离线的 |
| 对元素的要求 | 只要能用 `<` 比较 | 要能映射成 `0..m-1` 的整数位置 |
| 重复元素由谁负责 | 合并时的那个 `<`（相等先放左边，即稳定） | 查询的边界 `prefix_sum(r)` |
| 额外空间 | $O(n)$ | $O(m)$ 或 $O(R)$，重复很多时更省 |
| 最坏时间 | $O(n \log n)$ | $O(n \log n)$ |

一次性给全的数组，两种都行；面试时归并不需要另外写一个数据结构，更容易一次写对。数据是流式的、或者
之后还要支持“删掉一个数”“问小于任意阈值的有多少个”，就用树状数组：删除只是 `add(pos, -1)`，
而归并做不了。两种写法里，重复元素出错的地方都只有一处，写完要用带重复的例子专门测。

### 追问

**问：怎么求整个数组的逆序对总数？** 逆序对是满足 `i < j` 且 `nums[i] > nums[j]` 的对数，正好是
`sum(counts)`。只要总数的话，归并时不用带下标，直接排值、累加 `j - mid` 就行。

**问：条件改成 `nums[i] > 2 * nums[j]` 呢？** 这时“计数用的条件”和“排序用的顺序”不再是同一个比较，
不能在合并的同时数。做法是把一次合并拆成两趟：两半各自有序，先用双指针数（对左半段从小到大的每个
元素，右指针只会向右走），再做普通的合并。复杂度还是 $O(n \log n)$。用树状数组的话，查询位置换成
“严格小于 `nums[i] / 2` 的不同值有多少个”，在排好序的 `values` 上二分得到。

**问：流式到达，但值域事先不知道，怎么办？** 树状数组需要预先知道位置的范围，用不了。可选的做法：
带子树大小的平衡搜索树或跳表，每次 $O(\log n)$；动态开点的线段树，按需创建节点，每次 $O(\log R)$；
或者最简单的，维护一个有序列表，用 `bisect` 找位置再插入。最后这种查找是 $O(\log n)$，但插入要搬动
元素，最坏每次 $O(n)$，总共 $O(n^2)$，只是常数很小，数据量不大时够用。

**问：为什么不直接开一个按值计数的数组？** 两个原因。值域到 $10^{18}$，数组开不下，所以要压缩。
压缩之后如果只用普通计数数组，每次查询要把前面的计数加一遍，是 $O(m)$，总共 $O(nm)$，重复少的时候
和暴力一样慢。树状数组把这次求和降到 $O(\log m)$。

**问：递归会不会太深？** 不会。每层把段切成两半，深度是 $\lceil \log_2 n \rceil$，$2 \times 10^5$
个元素只有 18 层。真正会递归过深的是按值划分的写法（比如退化的快排），归并没有这个问题。

<details>
<summary>验证代码（可运行）</summary>

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
