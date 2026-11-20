# 关键帧放置：间隔受限的最小代价与最优方案计数

## 题目

一段视频有 `n` 帧，编号 `0` 到 `n - 1`。编码器要选出一部分帧作为关键帧（keyframe，可以单独解码的
帧），其余的帧只保存和前面帧的差异。把第 `i` 帧做成关键帧要多花 `c[i]` 个字节。选法要满足两条规则：

- 第 `0` 帧和第 `n - 1` 帧必须是关键帧：开头没有可以参考的帧；结尾是关键帧，才能和下一段视频直接
  拼接。
- 相邻两个关键帧的编号之差叫做间隔（gap）。每个间隔都不能超过 `G`，否则拖动进度条时要连续解码的帧
  太多。

一个方案（plan）就是关键帧编号的集合，它的费用是这些帧的 `c[i]` 之和。

三个部分的输入范围相同：`0 <= n <= 2000`，`c[i]` 是整数且 `0 <= c[i] <= 10**9`，`1 <= G <= 2000`；
Part 2 和 Part 3 另有 `1 <= H <= G`。输入保证在这个范围内，不需要校验。`n = 0`（没有帧）时唯一的
方案是空集，费用为 `0`；`n = 1` 时第 `0` 帧既是首帧又是尾帧，费用只算一次。`G` 可以比 `n` 大，这时
间隔上限不起作用。

### Part 1 —— 最小费用

```py
def min_cost(c: list[int], G: int) -> int: ...
```

返回合法方案的最小费用。因为 `G >= 1`，把每一帧都选上总是合法的，所以答案一定存在。逐个枚举方案
行不通：中间的 `n - 2` 帧每一帧都可选可不选，共 $2^{n-2}$ 种。要求 `n = 2000` 时一秒内算完。

```text
c = [3, 1, 4, 1, 5], G = 2   ->  10    关键帧 {0, 1, 3, 4}：3 + 1 + 1 + 5
c = [3, 1, 4, 1, 5], G = 4   ->  8     关键帧 {0, 4}，间隔正好是 4
c = [3, 1, 4, 1, 5], G = 1   ->  14    每一帧都得选
c = [7],             G = 3   ->  7
c = [],              G = 1   ->  0
```

### Part 2 —— 长间隔不能连着出现

播放器的缓冲区能消化偶尔一次长间隔，但不能连续两次。现在多给一个整数 `H`（`1 <= H <= G`）：大于
`H` 的间隔叫做长间隔（它仍然不能超过 `G`）。新增一条规则：**相邻的两个间隔不能都是长间隔。** 也就
是说，对任意连续的三个关键帧 `a < b < d`，`b - a` 和 `d - b` 不能同时大于 `H`。

```py
def min_cost_limited(c: list[int], G: int, H: int) -> int: ...
```

返回新规则下的最小费用。间隔为 `1` 永远不是长间隔，所以答案仍然一定存在。

```text
c = [0, 2, 1, 3, 0], G = 2, H = 1   ->  3     {0, 1, 2, 4}；{0, 2, 4} 只要 1，但它的两个间隔都是 2
c = [3, 1, 4, 1, 5], G = 3, H = 2   ->  9     {0, 3, 4} 或 {0, 1, 4}
c = [3, 1, 4, 1, 5], G = 4, H = 2   ->  8     {0, 4}：只有一个间隔，单独一个长间隔是允许的
c = [3, 1, 4, 1, 5], G = 2, H = 2   ->  10    H = G 时不存在长间隔，和 Part 1 相同
```

### Part 3 —— 最优方案有多少个

码率控制模块想知道编码器有多大的选择余地。在 Part 2 的规则下，求最小费用，以及费用恰好等于最小费用
的方案有多少个。两个方案只要关键帧集合不同，就算不同的方案。方案数可能非常大，把它对
$10^9 + 7$ 取模；费用不取模。

```py
def count_optimal(c: list[int], G: int, H: int) -> tuple[int, int]: ...
# returns (minimum cost, number of plans with that cost modulo 1_000_000_007)
```

要求在 Part 2 的代码上改，不要另起一套；`n = 2000` 时几秒内算完。

```text
c = [3, 1, 4, 1, 5], G = 3, H = 2   ->  (9, 2)    {0, 3, 4} 和 {0, 1, 4}
c = [0, 2, 1, 3, 0], G = 2, H = 1   ->  (3, 1)    只有 {0, 1, 2, 4}
c = [0, 0, 0, 0],    G = 3, H = 3   ->  (0, 4)    中间两帧各自可选可不选
c = [0, 0, 0, 0],    G = 2, H = 1   ->  (0, 3)    {0, 1, 3}、{0, 2, 3}、{0, 1, 2, 3}
c = [],              G = 1, H = 1   ->  (0, 1)    只有空方案
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认。第一，首尾两帧是不是都必须选、`n = 1` 怎么算：都必须选，`n = 1` 时它们是
同一帧，费用只算一次。第二，费用能不能是 `0`：可以，所以费用相同的方案可能很多，Part 3 不能假设
最优方案只有一个。

### Part 1

**找状态。** 从左到右决定哪些帧是关键帧。假设已经定下第 `i` 帧是关键帧，那么后面的帧怎么选才合法、
还要花多少钱，只和 `i` 有关：下一个关键帧必须落在 `i + 1` 到 `i + G` 之间，这和 `i` 之前选了哪些帧
没有关系。所以每个 `i` 只需要记一个数：

- `best[i]`：只看第 `0` 到第 `i` 帧，并且第 `i` 帧是关键帧时，合法选法的最小费用。

**递推。** 第 `i` 帧（`i >= 1`）前面的那个关键帧 `j` 必须满足 `i - G <= j <= i - 1`，枚举它：

$$
\mathrm{best}[0] = c[0], \qquad
\mathrm{best}[i] = c[i] + \min_{\max(0,\, i-G) \,\le\, j \,<\, i} \mathrm{best}[j].
$$

尾帧必须是关键帧，所以答案就是 `best[n - 1]`。

**为什么对。** 对 `i` 做归纳。任何一个以 `i` 结尾的合法选法，去掉 `i` 之后是一个以某个 `j` 结尾的
合法选法，并且 `j` 在窗口 `[i - G, i - 1]` 里，所以它的费用不低于 `c[i] + best[j]`。反过来，把 `i`
接在任何一个以窗口内的 `j` 结尾的合法选法后面，新增的间隔不超过 `G`，结果仍然合法。两个方向合起来
就是上面的等式。

**复杂度。** 一共 `n` 个状态，每个状态最多看 `G` 个前驱：时间 $O(nG)$，空间 $O(n)$。`n = G = 2000`
时上限是四百万次比较（前面的帧前驱不满 `G` 个，实际约两百万次）。窗口里的最小值还可以用单调队列降到总共 $O(n)$，见追问。

```python
def min_cost(c, G):
    n = len(c)
    if n == 0:
        return 0
    best = [0] * n            # best[i]: cheapest valid choice on frames 0..i that ends with a keyframe at i
    best[0] = c[0]
    for i in range(1, n):
        best[i] = c[i] + min(best[max(0, i - G):i])      # previous keyframe j with i - G <= j < i
    return best[n - 1]
```

### Part 2

新规则让“后面怎么选才合法”不再只取决于最后一个关键帧的位置：如果最后一个间隔是长间隔，下一个间隔
就必须是短的。所以状态里要多带一位信息，即最后一个间隔是不是长间隔：

- `short[i]`：第 `i` 帧是关键帧，并且最后一个间隔不是长间隔时的最小费用。`i = 0` 时还没有间隔，也
  归在这里。
- `long[i]`：第 `i` 帧是关键帧，并且最后一个间隔是长间隔时的最小费用；没有这样的选法时记为无穷大。

设前一个关键帧是 `j`。`i - j <= H` 时这一步是短间隔，`j` 处是哪种状态都可以；`H < i - j <= G` 时这
一步是长间隔，`j` 处就只能是 `short`：

$$
\mathrm{short}[i] = c[i] + \min_{i-H \,\le\, j \,<\, i} \min(\mathrm{short}[j],\, \mathrm{long}[j]),
\qquad
\mathrm{long}[i] = c[i] + \min_{i-G \,\le\, j \,<\, i-H} \mathrm{short}[j].
$$

（下标小于 `0` 的 `j` 不存在，直接略去。）答案是 `min(short[n - 1], long[n - 1])`。正确性的论证和
Part 1 一样，只是“以 `i` 结尾的合法选法”按最后一个间隔的类型分成了两类。

**一个常见的错误**是仍然给每一帧只记一个最小费用，再顺手记下“取得这个最小费用的选法，最后一步是
不是长间隔”。看题目里的例子 `c = [0, 2, 1, 3, 0], G = 2, H = 1`：到第 `2` 帧最便宜的走法是
`{0, 2}`（费用 `1`，最后一步是长间隔），于是 `{0, 1, 2}`（费用 `3`，最后一步是短间隔）被丢掉了。
可是从第 `2` 帧跳到第 `4` 帧又是一个长间隔，只有被丢掉的那个选法才接得上。只记一个值的写法最后得到
`4`，正确答案是 `3`。费用更高但后面限制更少的选法不能丢，这就是必须把状态拆成两个的原因。

**复杂度。** 状态数变成两倍；对每个 `i`，两个窗口合起来仍然不超过 `G` 个前驱。时间 $O(nG)$，空间
$O(n)$。

```python
INF = float("inf")


def min_cost_limited(c, G, H):
    n = len(c)
    if n == 0:
        return 0
    short = [INF] * n         # the last gap is not long (or there is no gap yet)
    long_ = [INF] * n         # the last gap is long
    short[0] = c[0]
    for i in range(1, n):
        near = max(0, i - H)  # j in [near, i) makes a short gap, j in [far, near) a long one
        far = max(0, i - G)
        short[i] = c[i] + min(min(short[near:i]), min(long_[near:i]))
        if far < near:
            long_[i] = c[i] + min(short[far:near])       # NOTE: short[j] only, a long gap may not follow a long gap
    return min(short[n - 1], long_[n - 1])
```

### Part 3

把每个状态存的东西从“最小费用”换成一对数“（最小费用，达到这个费用的选法个数）”。递推的形状完全
不变，只需要把两种操作换掉：

- 在几个来源里取最小值时：费用更小的一方胜出，并带上它的个数；费用相等时，个数相加。
- 接上第 `i` 帧时：费用加上 `c[i]`，个数不变。

**为什么不重不漏。** 一个以 `i` 结尾的选法，可以唯一地拆成“前一个关键帧 `j`、它在 `j` 处属于
`short` 还是 `long`”再加上 `i`。不同的 `j`，或者同一个 `j` 的两种状态，对应的选法集合互不相交。所以
把所有取到最小费用的来源的个数加起来，每个最优选法恰好被数一次。`short[i]` 和 `long[i]` 两类也互不
相交（最后一个间隔要么是长间隔，要么不是），最后合并它们时同样是“小的胜出，相等相加”。

两个细节。个数每次相加之后取模，**费用绝不能取模**，因为要拿它比大小。费用最大约为
$2000 \times 10^9 = 2 \times 10^{12}$，Python 的整数没有问题，换成 C++ 或 Java 要用 64 位整数。

**复杂度。** 和 Part 2 相同：时间 $O(nG)$，空间 $O(n)$。

```python
MOD = 1_000_000_007


def _best(cost, ways, lo, hi):
    """Smallest cost at positions lo..hi-1 and how many choices reach it; (INF, 0) for an empty range."""
    best, total = INF, 0
    for j in range(lo, hi):
        if cost[j] < best:
            best, total = cost[j], ways[j]
        elif cost[j] == best:
            total += ways[j]
    return best, total % MOD


def _merge(a, b):
    """Combine two (cost, ways) alternatives that describe disjoint sets of choices."""
    if a[0] != b[0]:
        return a if a[0] < b[0] else b
    return a[0], (a[1] + b[1]) % MOD


def count_optimal(c, G, H):
    n = len(c)
    if n == 0:
        return 0, 1           # the empty plan
    s_cost, s_ways = [INF] * n, [0] * n       # same meaning as short[] in Part 2, plus a count
    l_cost, l_ways = [INF] * n, [0] * n       # same meaning as long_[]
    s_cost[0], s_ways[0] = c[0], 1
    for i in range(1, n):
        near = max(0, i - H)
        far = max(0, i - G)
        cost, ways = _merge(_best(s_cost, s_ways, near, i), _best(l_cost, l_ways, near, i))
        s_cost[i], s_ways[i] = c[i] + cost, ways
        cost, ways = _best(s_cost, s_ways, far, near)
        l_cost[i], l_ways[i] = c[i] + cost, ways         # stays (INF, 0) when no long gap can end here
    return _merge((s_cost[n - 1], s_ways[n - 1]), (l_cost[n - 1], l_ways[n - 1]))
```

### 追问

- **`n` 和 `G` 都到 $2 \times 10^5$ 时，$O(nG)$ 不够用，怎么办？** 递推里要的是一个随 `i` 向右滑动
  的窗口里的最小值。用单调队列（monotonic deque）：队列里的值从队首到队尾递增，新值入队前把队尾
  不比它小的值弹掉，队首过期就弹出，队首永远是窗口最小值。每个下标进出队列各一次，Part 1 总共
  $O(n)$。Part 2 的两个窗口 `[i - H, i - 1]` 和 `[i - G, i - H - 1]` 也都是随 `i` 右移的，各用一个
  队列即可。验证代码里有这两个版本，并在 `n = 200000` 上运行。
- **规则改成“长间隔最多连续出现 `R` 次”，状态怎么变？** 状态里那一位信息换成一个小计数器：`f[i][r]`
  表示第 `i` 帧是关键帧、末尾恰好连着 `r` 个长间隔时的最小费用，`r` 从 `0` 到 `R`。短间隔把 `r`
  清零，长间隔把 `r` 加一且要求加之前 `r < R`。时间 $O(nGR)$。Part 2 就是 `R = 1`。设计状态的原则
  始终是同一条：过去的信息里，只保留会影响后面是否合法的那一部分。验证代码里实现了这个一般版本。
- **不只数最优方案，想数所有合法方案，要重写吗？** 不用。把所有费用都换成 `0` 再调用
  `count_optimal`：这时每个合法方案的费用都是 `0`，全都是最优的，返回的个数就是合法方案的总数。
  例如 `n >= 2` 且 `H = G >= n - 1` 时，中间每一帧可选可不选，结果是 $2^{n-2}$。
- **这段代码你会怎么测？** 小规模输入上和“枚举中间帧的所有子集、按题目规则逐条检查”的暴力解比对，
  同时比对费用和个数；`n` 不超过 `10` 时暴力解足够快。边界情况单独写：`n = 0, 1, 2`，`G = 1`（只能
  全选），`H = G`（应当和 Part 1 一致），费用全为 `0`（个数最大）。最后在 `n = 2000` 上运行一次，
  确认时间，并用能手算的输入核对取模后的个数（比如全 `0` 费用、`G = H = 2` 时个数是斐波那契数）。

<details>
<summary>验证代码（可运行）</summary>

```python
import itertools
import random
import time
from collections import deque

# ---- every example in the statement ----
assert min_cost([3, 1, 4, 1, 5], 2) == 10
assert min_cost([3, 1, 4, 1, 5], 4) == 8
assert min_cost([3, 1, 4, 1, 5], 1) == 14
assert min_cost([7], 3) == 7
assert min_cost([], 1) == 0

assert min_cost_limited([0, 2, 1, 3, 0], 2, 1) == 3
assert min_cost_limited([3, 1, 4, 1, 5], 3, 2) == 9
assert min_cost_limited([3, 1, 4, 1, 5], 4, 2) == 8
assert min_cost_limited([3, 1, 4, 1, 5], 2, 2) == 10

assert count_optimal([3, 1, 4, 1, 5], 3, 2) == (9, 2)
assert count_optimal([0, 2, 1, 3, 0], 2, 1) == (3, 1)
assert count_optimal([0, 0, 0, 0], 3, 3) == (0, 4)
assert count_optimal([0, 0, 0, 0], 2, 1) == (0, 3)
assert count_optimal([], 1, 1) == (0, 1)


# ---- independent brute force: every subset of the inner frames, checked against the rules as stated ----
def brute(c, G, H, R=1):
    """(min cost, number of plans with that cost); at most R long gaps in a row (the statement has R = 1)."""
    n = len(c)
    if n == 0:
        return 0, 1
    best, ways = None, 0
    inner = range(1, n - 1)
    for size in range(len(inner) + 1):
        for chosen in itertools.combinations(inner, size):
            keys = sorted({0, n - 1, *chosen})
            gaps = [b - a for a, b in zip(keys, keys[1:])]
            run = longest_run = 0
            for g in gaps:
                run = run + 1 if g > H else 0
                longest_run = max(longest_run, run)
            if any(g > G for g in gaps) or longest_run > R:
                continue
            cost = sum(c[k] for k in keys)
            if best is None or cost < best:
                best, ways = cost, 1
            elif cost == best:
                ways += 1
    return best, ways


def one_value_shortcut(c, G, H):
    """The mistake described in Part 2: one cost per frame plus a flag for how that one plan ended."""
    n = len(c)
    best = [INF] * n
    was_long = [False] * n
    best[0] = c[0]
    for i in range(1, n):
        for j in range(max(0, i - G), i):
            is_long = i - j > H
            if is_long and was_long[j]:
                continue
            cand = best[j] + c[i]
            if cand < best[i] or (cand == best[i] and was_long[i] and not is_long):
                best[i], was_long[i] = cand, is_long
    return best[n - 1]


# ---- follow-up 1: the same recurrences with monotonic deques, O(n) ----
class MinWindow:
    """Minimum of the pushed values whose index is still >= lo; indices arrive in increasing order."""

    def __init__(self):
        self.q = deque()

    def push(self, idx, val):
        while self.q and self.q[-1][1] >= val:
            self.q.pop()
        self.q.append((idx, val))

    def min_from(self, lo):
        while self.q and self.q[0][0] < lo:
            self.q.popleft()
        return self.q[0][1] if self.q else INF


def min_cost_fast(c, G):
    if not c:
        return 0
    win = MinWindow()
    best = c[0]
    win.push(0, best)
    for i in range(1, len(c)):
        best = c[i] + win.min_from(i - G)
        win.push(i, best)
    return best


def min_cost_limited_fast(c, G, H):
    n = len(c)
    if n == 0:
        return 0
    short, long_ = [INF] * n, [INF] * n
    short[0] = c[0]
    near, far = MinWindow(), MinWindow()      # near: min(short, long) on [i-H, i-1]; far: short on [i-G, i-H-1]
    for i in range(1, n):
        near.push(i - 1, min(short[i - 1], long_[i - 1]))
        if i - H - 1 >= 0:
            far.push(i - H - 1, short[i - H - 1])
        short[i] = c[i] + near.min_from(i - H)
        long_[i] = c[i] + far.min_from(i - G)
    return min(short[n - 1], long_[n - 1])


# ---- follow-up 2: at most R long gaps in a row ----
def min_cost_runs(c, G, H, R):
    n = len(c)
    if n == 0:
        return 0
    f = [[INF] * (R + 1) for _ in range(n)]   # f[i][r]: keyframe at i, exactly r long gaps at the end
    f[0][0] = c[0]
    for i in range(1, n):
        for j in range(max(0, i - G), i):
            if i - j <= H:
                f[i][0] = min(f[i][0], c[i] + min(f[j]))
            else:
                for r in range(R):
                    f[i][r + 1] = min(f[i][r + 1], c[i] + f[j][r])
    return min(f[n - 1])


# ---- the mistake from Part 2 really is a mistake on the statement's example ----
assert one_value_shortcut([0, 2, 1, 3, 0], 2, 1) == 4 and brute([0, 2, 1, 3, 0], 2, 1) == (3, 1)

# ---- edge cases ----
for G in (1, 2, 5):
    for H in range(1, G + 1):
        assert count_optimal([], G, H) == (0, 1) and min_cost_limited([], G, H) == 0
        assert count_optimal([9], G, H) == (9, 1) and min_cost_limited([9], G, H) == 9
        assert count_optimal([4, 6], G, H) == (10, 1) and min_cost([4, 6], G) == 10      # both ends are forced
assert min_cost([5] * 7, 1) == 35 and count_optimal([5] * 7, 1, 1) == (35, 1)             # G = 1: every frame
assert count_optimal([5] * 7, 3, 3) == (15, 1)        # all equal: fewest keyframes, {0, 3, 6} is the only way
assert count_optimal([5] * 7, 3, 2) == (20, 7)        # {0, 3, 6} is now forbidden; four keyframes: the gaps are
assert brute([5] * 7, 3, 2) == (20, 7)                # 2+2+2 or an ordering of 1+2+3, seven ways
assert count_optimal([5] * 6, 2000, 2000) == (10, 1)  # G larger than n: only the two ends
assert min_cost([0, 10 ** 9, 10 ** 9, 0], 2) == 10 ** 9

# ---- random comparison with the brute force ----
rng = random.Random(20261117)
saw_tie = saw_rule_binding = saw_shortcut_wrong = saw_long_used = 0
for _ in range(4000):
    n = rng.randint(0, 10)
    G = rng.randint(1, max(1, n + 1))
    H = rng.randint(1, G)
    top = rng.choice([0, 1, 3, 9, 10 ** 9])
    c = [rng.randint(0, top) for _ in range(n)]
    want_cost, want_ways = brute(c, G, H)
    assert min_cost_limited(c, G, H) == want_cost, (c, G, H)
    assert count_optimal(c, G, H) == (want_cost, want_ways % MOD), (c, G, H)
    assert min_cost(c, G) == brute(c, G, G)[0] == min_cost_limited(c, G, G), (c, G)
    assert min_cost_fast(c, G) == min_cost(c, G)
    assert min_cost_limited_fast(c, G, H) == want_cost, (c, G, H)
    R = rng.randint(1, 3)
    assert min_cost_runs(c, G, H, R) == brute(c, G, H, R)[0], (c, G, H, R)
    assert count_optimal([0] * n, G, H) == (0, brute([0] * n, G, H)[1])     # follow-up 3: all valid plans
    saw_tie += want_ways > 1
    saw_rule_binding += want_cost > min_cost(c, G)
    saw_long_used += want_cost < min_cost(c, H)          # cheaper than every plan without a long gap
    saw_shortcut_wrong += n >= 1 and one_value_shortcut(c, G, H) != want_cost
assert min(saw_tie, saw_rule_binding, saw_long_used, saw_shortcut_wrong) >= 20, \
    (saw_tie, saw_rule_binding, saw_long_used, saw_shortcut_wrong)

# ---- the largest size in the stated range, with answers that can be worked out by hand ----
N = 2000
big = [rng.randint(0, 10 ** 9) for _ in range(N)]
t0 = time.perf_counter()
a1 = min_cost(big, 2000)
a2 = min_cost_limited(big, 2000, 1000)
a3 = count_optimal(big, 2000, 1000)
zeros_free = count_optimal([0] * N, 2000, 2000)
zeros_fib = count_optimal([0] * N, 2, 2)
zeros_no_two = count_optimal([0] * N, 2, 1)
print(f"n = 2000, six calls: {time.perf_counter() - t0:.2f} s")      # printed, not asserted
assert a3[0] == a2 == min_cost_limited_fast(big, 2000, 1000) and a1 == min_cost_fast(big, 2000) and a1 <= a2
assert zeros_free == (0, pow(2, N - 2, MOD))                 # every inner frame is free

fib = [0, 1]                                  # gaps of 1 or 2 that add up to N - 1: Fibonacci number F(N)
while len(fib) <= N:
    fib.append(fib[-1] + fib[-2])
assert zeros_fib == (0, fib[N] % MOD)

t = [1, 1, 2]                                 # the same, but no two gaps of 2 next to each other
while len(t) <= N - 1:                        # t[m] = t[m-1] (ends with 1) + t[m-3] (ends with 1, 2)
    t.append(t[-1] + t[-3])
assert zeros_no_two == (0, t[N - 1] % MOD)
assert count_optimal([5] * 1999, 2, 2) == (5000, 1)          # 1998 = 999 gaps of 2, only one way
assert 2000 * 10 ** 9 > 2 ** 31 and 2000 * 10 ** 9 < 2 ** 63  # costs need 64 bits, and 64 bits are enough

# ---- follow-up 1 at n = 200000 ----
huge = [rng.randint(0, 10 ** 9) for _ in range(200_000)]
t0 = time.perf_counter()
h1 = min_cost_fast(huge, 150_000)
h2 = min_cost_limited_fast(huge, 150_000, 70_000)
print(f"n = 200000, deque versions: {time.perf_counter() - t0:.2f} s")
assert h1 <= h2 <= min_cost_fast(huge, 70_000)               # more freedom can only lower the cost
print("all checks passed")
```

</details>

</details>
