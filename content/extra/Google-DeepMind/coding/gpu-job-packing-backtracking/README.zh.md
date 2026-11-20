# 把训练作业分到 GPU 上：回溯的搜索顺序、剪枝与分支限界

## 题目

一台机器上有 `k` 块完全相同的 GPU，每块显存都是 `cap`（单位 GiB）。有 `n` 个作业要放上去，第 `i` 个
作业需要 `jobs[i]` GiB 显存。一个作业必须整个放在一块 GPU 上，不能拆开；一块 GPU 可以放多个作业，只要
它们的显存加起来不超过 `cap`。分配方案用一个长度为 `n` 的列表 `assign` 表示，`assign[i]` 是作业 `i`
所在 GPU 的编号（`0` 到 `k - 1`）。

输入范围：`0 <= n <= 14`，`1 <= k <= 14`，`1 <= jobs[i] <= 10**6`，`1 <= cap <= 10**7`，都是整数。输入
保证在范围内，不需要校验。`n = 0` 时空列表 `[]` 就是合法方案。

**节点计数。** 面试官会数你的搜索访问了多少个节点（node）：递归函数每被调用一次算一个节点，包括最开始
那次调用和放完全部作业时的那次调用。每个函数都接受一个可选的字典 `stats`，返回前把节点数写进
`stats["nodes"]`。这个数只取决于代码怎么搜索，和机器快慢无关，所以下面用它来比较剪枝的效果，而不是
用运行时间。

### Part 1 —— 找一个分配方案

```py
def pack(jobs: list[int], k: int, cap: int, stats: dict | None = None) -> list[int] | None: ...
```

返回字典序最小的合法 `assign`（先比 `assign[0]`，相同再比 `assign[1]`，依此类推）；没有合法方案时返回
`None`。这一部分只要求正确，面试官只用 `n <= 8` 的输入测试。

```text
jobs = [3, 5, 2, 4], k = 2, cap = 7  ->  [0, 1, 1, 0]    GPU 0 放 3 + 4 = 7，GPU 1 放 5 + 2 = 7
jobs = [3, 5, 2, 4], k = 3, cap = 5  ->  [0, 1, 0, 2]    3 + 2 = 5，5，4
jobs = [3, 5, 2, 4], k = 2, cap = 6  ->  None            总量 14 > 2 × 6
jobs = [8],          k = 2, cap = 7  ->  None            单个作业就放不下
jobs = [],           k = 3, cap = 1  ->  []
```

### Part 2 —— 剪枝

现在 `n` 可以到 `14`。返回任意一个合法方案即可（不再要求字典序最小，面试官只检查它是否合法），没有就
返回 `None`。

```py
def pack_fast(jobs: list[int], k: int, cap: int, stats: dict | None = None) -> list[int] | None: ...
```

下面两个输入各有 12 个作业、4 块 GPU。按 Part 1 的参考写法（作业按输入顺序处理，GPU 按编号从小到大
试，放之前先检查装不装得下，找到第一个方案就停）计数，访问的节点数如下：

```text
A: jobs = [6, 9, 13, 15, 8, 9, 10, 15, 3, 3, 23, 30], k = 4, cap = 36
   有解：总量 144 = 4 × 36，例如 {30, 6}、{23, 13}、{15, 15, 3, 3}、{10, 9, 9, 8}
   Part 1 的写法访问 66005 个节点

B: jobs = [31, 7, 13, 6, 30, 9, 9, 13, 31, 24, 40, 31], k = 4, cap = 61
   无解：总量 244 = 4 × 61，每块都得正好装满；但能和 40 放在一起的作业只有 13、13、9、9、7、6，
   它们凑不出 40 那块还差的 21
   Part 1 的写法访问 350105 个节点
```

要求：在 A 和 B 上访问的节点数都不超过上面的百分之一；`n = 14` 的输入，包括作业大小都差不多、刚好
装不下这种难的输入，要在一秒左右跑完。说清楚每一种剪枝为什么不会漏掉解，以及它为什么能省掉这么多
节点。

### Part 3 —— 让最忙的那块 GPU 尽量空

不再给 `cap`。一个方案的峰值（peak）是各块 GPU 显存占用中的最大值。求峰值最小的方案。

```py
def min_peak(jobs: list[int], k: int, stats: dict | None = None) -> tuple[int, list[int]]: ...
# returns (smallest possible peak, one assignment that reaches it); any such assignment is accepted
```

`n = 0` 时返回 `(0, [])`。`n = 14` 时要在一秒左右跑完，尽量用上 Part 2 的代码。

```text
jobs = [3, 3, 2, 2, 2], k = 2  ->  峰值 6    {3, 3} 和 {2, 2, 2}
jobs = [3, 5, 2, 4],    k = 3  ->  峰值 5    {5}、{3, 2}、{4}
A, k = 4                       ->  峰值 36
B, k = 4                       ->  峰值 62   总量除以 4 是 61，但 Part 2 说明了 61 装不下
jobs = [],              k = 2  ->  峰值 0
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前有两点值得先确认。第一，作业能不能拆到两块 GPU 上：不能，否则问题就变成简单的除法。第二，
GPU 是不是完全相同：是。这一点决定了 Part 2 最有效的那种剪枝能不能用，题目一旦改成“每块 GPU 显存
不同”，它就要跟着改。

### Part 1

**思路。** 一个作业一个作业地决定放在哪块 GPU 上。对作业 `i`，按编号从小到大试每一块装得下它的 GPU：
放上去，递归处理作业 `i + 1`；如果后面失败了，就把它拿下来（撤销），再试下一块。所有 GPU 都试过还不行，
就向上一层报告失败。

**不变量。** 调用 `place(i)` 时，`loads[g]` 正好是作业 `0..i-1` 中放在 GPU `g` 上的那些作业的显存之和，
并且每个 `loads[g]` 都不超过 `cap`。撤销那一行 `loads[g] -= jobs[i]` 就是为了让这个不变量在回到上一层
时仍然成立。

**为什么返回的是字典序最小的方案。** 搜索按字典序访问部分方案：先固定 `assign[0]` 的最小可能值，在
它下面再固定 `assign[1]` 的最小可能值，依此类推。被“装不下”剪掉的分支已经有一块 GPU 超了 `cap`，而
显存只会越加越多，所以这些分支下面不存在合法方案。因此第一个走到底的完整方案，就是字典序最小的合法
方案。

**复杂度。** 搜索树第 `d` 层最多有 $k^d$ 个节点，每个节点要试 `k` 块 GPU，最坏时间是
$O(k^{n+1})$；空间是递归深度加上两个数组，$O(n + k)$。`n <= 8` 时没有问题，`n = 14`、`k = 4` 时
$4^{14} \approx 2.7 \times 10^8$，就不行了。

```python
def pack(jobs, k, cap, stats=None):
    n = len(jobs)
    loads = [0] * k           # loads[g]: memory used on GPU g by jobs 0..i-1
    assign = [0] * n
    nodes = 0

    def place(i):
        nonlocal nodes
        nodes += 1
        if i == n:
            return True
        for g in range(k):    # smaller GPU numbers first, so the first full assignment is the smallest one
            if loads[g] + jobs[i] <= cap:
                loads[g] += jobs[i]
                assign[i] = g
                if place(i + 1):
                    return True
                loads[g] -= jobs[i]          # undo before trying the next GPU
        return False

    found = place(0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None
```

### Part 2

Part 1 慢，不是因为每个节点做的事多，而是因为它把大量时间花在注定失败的子树上。两种剪枝分别对付两个
原因。

**剪枝一：先放大的作业。** 把作业按显存从大到小排序后再搜索。大作业能去的 GPU 最少，先放它们，冲突会
在靠近树根的地方暴露出来；在那里失败一次，剪掉的是一整棵大子树。按输入顺序搜索时，B 里最大的 40 排在
第 11 个，前 10 个作业的所有摆法都要先排一遍，每一种都要到第 11 层才发现 40 放不进去。排序只改变搜索
顺序，不改变搜索的范围，所以不会漏解。这个思路一般叫“先处理最受约束的变量”（fail-first）。

**剪枝二：GPU 之间的对称性。** GPU 完全相同。如果两块 GPU 当前的占用相同，把当前作业放进哪一块，后面
的两棵子树互为镜像：把这两块 GPU 的编号对调，一棵子树里的每个方案就变成另一棵里的一个方案，而且都
合法。所以同一个占用值只需要试一块 GPU。最常见的情形是空 GPU：所有空 GPU 都等价，只试一块。没有这条，
第一个作业就有 `k` 个等价的选择，第二个作业又有 `k - 1` 个……整棵树被白白复制了很多倍。

**正确性。** 假设某个合法方案把作业 `i` 放在 GPU `g2` 上，而另一块 GPU `g1` 在放作业 `i` 之前的占用和
`g2` 一样。把作业 `i` 及之后所有作业的 `g1`、`g2` 对调：两块 GPU 最终的占用正好互换，仍然都不超过
`cap`，作业 `i` 之前的部分不变。所以跳过 `g2` 不会丢掉“有解”这个结论。

另外，搜索之前先做两个 $O(n)$ 的检查：总量不能超过 `k * cap`，最大的作业不能超过 `cap`。不满足就直接
返回 `None`，节点数记为 `0`。

**效果。** 下表是各个版本在题目的 A、B 上访问的节点数（验证代码里逐项断言）：

| 版本 | A（有解） | B（无解） |
| --- | ---: | ---: |
| Part 1 原样 | 66005 | 350105 |
| 只加对称剪枝 | 15491 | 13356 |
| 只按大小降序 | 13 | 1577 |
| 两个都加 | 13 | 16 |

A 排序之后第一次往下走就成功了，`13 = n + 1` 个节点，一次回溯都没有。B 要证明无解，两种剪枝单独用都
不够，合起来才降到 16 个节点：排好序后前四个作业 40、31、31、31 两两相加都超过 61，各占一块 GPU，
对称剪枝让这一步只有一种走法；接下来 30 只能和某个 31 放在一起，24 只能放进剩下的某个 31，几步之内就
走到死路。

**能省多少，取决于输入。** 在验证代码的随机输入上（`n` 为 9 到 11，`k` 为 3 或 4，作业大小 5 到 40），
`cap` 比最优峰值小 1 的无解输入，节点数的中位数缩小了几百倍，而且没有一个输入变差；`cap` 正好等于最优
峰值的有解输入，中位数只缩小几倍，个别输入反而变多，因为原来的顺序碰巧很早就撞上了一个解。剪枝的大头
收益在“必须把整棵树搜完”的时候：证明无解，或者像 Part 3 那样证明没有更好的解。

**复杂度。** 最坏情况仍然是指数级的：`k = 2` 时这个问题就是划分问题（partition problem），是 NP 完全
的，不要指望多项式算法。可以给出的上界是：有了对称剪枝，第 `d` 层的不同节点对应把前 `d` 个作业分成
至多 `k` 个无标号组的不同分法（两条路径第一次分岔时，当前作业进了占用不同、因而作业集合也不同的两块
GPU），所以第 `d` 层最多有 $\sum_{j \le k} S(d, j)$ 个节点，$S$ 是第二类斯特林数（Stirling numbers of
the second kind），比 $k^d$ 小得多；`cap` 的限制再剪掉其中的大部分。每个节点的工作量是 $O(k)$，空间是
$O(n + k)$。

```python
def pack_fast(jobs, k, cap, stats=None, sort_desc=True, skip_equal=True):
    """Any valid assignment or None. The two switches exist only so that the checks can measure each pruning."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i]) if sort_desc else list(range(n))
    loads = [0] * k
    assign = [0] * n
    nodes = 0

    def place(pos):
        nonlocal nodes
        nodes += 1
        if pos == n:
            return True
        i = order[pos]                       # the pos-th largest job
        tried = set()                        # loads already tried for this job at this node
        for g in range(k):
            if loads[g] + jobs[i] > cap:
                continue
            if skip_equal:
                if loads[g] in tried:        # NOTE: same load as a GPU tried before, so the subtree is a mirror image
                    continue
                tried.add(loads[g])
            loads[g] += jobs[i]
            assign[i] = g
            if place(pos + 1):
                return True
            loads[g] -= jobs[i]
        return False

    found = sum(jobs) <= k * cap and max(jobs, default=0) <= cap and place(0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None
```

### Part 3

这是一个优化问题。搜索同一棵树，但不再找到一个解就停，而是一边搜一边记住目前最好的解（incumbent），
并用它剪枝。这种做法叫分支限界（branch and bound）。

1. **先用贪心拿一个不错的解。** 作业从大到小，每个放到当前最空的 GPU 上（这个贪心叫 LPT，longest
   processing time first）。它的峰值作为初始的 `best`。
2. **算一个下界。** 任何方案的峰值都不小于最大的作业，也不小于 $\lceil \text{总量} / k \rceil$。记两者
   中较大的为 `lower`。如果贪心已经达到 `lower`，直接返回，不用搜索。
3. **搜索时剪枝。** 沿用 Part 2 的排序和对称剪枝。把作业放上去之后，如果这块 GPU 的占用已经 `>= best`，
   这个分支下面的方案峰值都 `>= best`，不可能严格更好，跳过。这相当于每一刻都在用 `cap = best - 1`
   跑 Part 2 的搜索，而且 `best` 越来越小，剪得越来越狠。
4. **走到叶子就更新。** 能走到叶子，说明路上每一步都 `< best`，这是一个严格更好的解，记下来。一旦
   `best == lower`，不可能再好了，立刻结束整个搜索。

**正确性。** 被剪掉的分支峰值都 `>= best`，不会是严格更好的解；对称剪枝的论证和 Part 2 相同（对调两块
GPU 的编号不改变峰值）。所以搜索结束时，`best` 就是最小峰值。

**效果。** 不用“`>= best` 就剪”这条规则（其余都保留，包括贪心初值和达到下界就停）时，A 要访问
700662 个节点，B 要 624928 个；用了之后分别是 13 和 16。验证代码里还有一个贪心给出 69、下界是 61、
最优是 62 的输入，分支限界访问 4424 个节点，不用这条规则要 2412066 个。

**复杂度。** 最坏情况同样是指数级的。搜到的每个节点上各块 GPU 的占用都小于贪心峰值，所以节点数不超过
Part 2 在 `cap = 贪心峰值 - 1` 下、找到解也不停、把整棵树搜完时的节点数。排序
$O(n \log n)$，贪心 $O(nk)$，每个节点 $O(k)$，空间 $O(n + k)$。验证代码在随机生成的 `n = 14` 输入里
挑出了节点最多的几个，打印它们的运行时间。

```python
def min_peak(jobs, k, stats=None, use_bound=True):
    """(smallest peak, an assignment reaching it). use_bound=False only exists so the checks can measure the bound."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i])
    loads = [0] * k
    assign = [0] * n
    for i in order:                          # greedy LPT: largest job onto the least loaded GPU
        g = loads.index(min(loads))
        loads[g] += jobs[i]
        assign[i] = g
    best, best_assign = max(loads), assign[:]
    lower = max(max(jobs, default=0), -(-sum(jobs) // k))
    loads = [0] * k
    nodes = 0

    def place(pos, peak):
        nonlocal best, best_assign, nodes
        nodes += 1
        if pos == n:
            if peak < best:                  # always true here: every step on the way stayed below best
                best, best_assign = peak, assign[:]
            return
        i = order[pos]
        tried = set()
        for g in range(k):
            new = loads[g] + jobs[i]
            if loads[g] in tried or (use_bound and new >= best):     # NOTE: >= because only a strictly better plan helps
                continue
            tried.add(loads[g])
            loads[g] = new
            assign[i] = g
            place(pos + 1, max(peak, new))
            loads[g] = new - jobs[i]
            if best == lower:                # nothing can beat the lower bound: unwind the whole search
                return

    if best > lower:
        place(0, 0)
    if stats is not None:
        stats["nodes"] = nodes               # search nodes only; the greedy pass is not counted
    return best, best_assign
```

### 追问

- **为什么不直接用贪心？** LPT 不一定最优。题目里的 `[3, 3, 2, 2, 2]`、`k = 2`：贪心先把两个 3 分开，
  三个 2 再轮流放，得到 `{3, 2, 2}` 和 `{3, 2}`，峰值 7，而最优是 6。Graham 证明过 LPT 的峰值不超过
  最优值的 $\frac{4}{3} - \frac{1}{3k}$ 倍，验证代码在随机输入上检查了这个界。贪心适合当初值和兜底；
  需要精确最优时才搜索。
- **能不能二分峰值，每次调用 Part 2？** 可以：峰值的答案落在 `[lower, 贪心峰值]` 里，对它二分，每次用
  `pack_fast` 判断能不能装下，调用次数是对数级的。缺点是每次“装不下”的判断都要把整棵树搜完，而这
  恰恰是最贵的那一种搜索；分支限界在一次搜索里不断收紧 `best`，前面搜过的部分不会重来。验证代码用二分
  的版本和 `min_peak` 互相核对。
- **什么时候回溯会变成动态规划？** 当同一个状态被反复搜到的时候。排好序后，放完前 `pos` 个作业，剩下
  的问题只取决于各块 GPU 占用的多重集合，和哪个作业在哪块 GPU 上无关。所以可以把失败过的
  `(pos, 排序后的占用)` 记在一个集合里，再遇到就直接返回失败，这就是带记忆化的搜索，也就是动态规划。
  它有没有用，要看状态重不重复：14 个大小都是 5 的作业、`k = 4`、`cap = 19` 时节点数从 2035 降到 61；
  作业大小各不相同时，占用的组合几乎不会重复，记忆化只会多占内存。另一种写法是对作业子集做动态规划：
  `f[j][mask]` 表示用 `j` 块 GPU 放下 `mask` 里的作业时的最小峰值，枚举最后一块 GPU 放的子集，时间
  $O(k \cdot 3^n)$。它的运行时间不看运气，验证代码把它当作独立的暴力解。
- **如果有些作业不能放在同一块 GPU 上（比如两个作业都要独占某个设备），哪些剪枝还成立？** 先放大的
  作业仍然成立，它只是换了搜索顺序；冲突多的作业也应该提前放。对称剪枝要收紧：两块 GPU 占用相同，但
  装的作业不同，对后面作业的冲突就不同，不再互为镜像。只有完全空着的 GPU 之间仍然等价，所以只能保留
  “空 GPU 只试一块”这一条。
- **`n` 再大一点，比如 16 块大小差不多的作业，会怎样？** 会慢。验证代码里有一个 16 个作业、大小都在
  1005 到 1049 之间、`k = 5` 的输入，`cap` 比最优峰值小 1 时，`pack_fast` 要访问 3904649 个节点才能
  证明无解。可以再加一条“浪费空间”的剪枝：一块 GPU 剩下的空间如果比最小的作业还小，这部分空间就永远
  用不上了；浪费的总量超过 `k * cap - 总量` 时剪掉。在这个输入上，节点数降到 1043003，只省了大约四分
  之三，仍然是指数级。到了这个规模，更现实的做法是换成整数规划求解器，或者接受贪心的近似解。

<details>
<summary>验证代码（可运行）</summary>

```python
import itertools
import random
import time

A = [6, 9, 13, 15, 8, 9, 10, 15, 3, 3, 23, 30]
B = [31, 7, 13, 6, 30, 9, 9, 13, 31, 24, 40, 31]
C = [19, 19, 27, 27, 17, 16, 32, 34, 36, 28, 17, 29]


def is_valid(jobs, k, cap, assign):
    if assign is None or len(assign) != len(jobs) or any(not 0 <= g < k for g in assign):
        return False
    loads = [0] * k
    for size, g in zip(jobs, assign):
        loads[g] += size
    return max(loads, default=0) <= cap


def peak_of(jobs, k, assign):
    loads = [0] * k
    for size, g in zip(jobs, assign):
        loads[g] += size
    return max(loads, default=0)


def run(fn, *args, **kw):
    stats = {}
    out = fn(*args, stats=stats, **kw)
    return out, stats["nodes"]


# ---- independent brute forces ----
def brute_first(jobs, k, cap):
    """itertools.product yields assignments in lexicographic order, so the first valid one is the smallest."""
    for assign in itertools.product(range(k), repeat=len(jobs)):
        if is_valid(jobs, k, cap, list(assign)):
            return list(assign)
    return None


def brute_peak(jobs, k):
    return min(peak_of(jobs, k, a) for a in itertools.product(range(k), repeat=len(jobs)))


def subset_dp_peak(jobs, k):
    """f[mask] after j rounds: smallest peak placing the jobs in mask on j GPUs. O(k * 3^n)."""
    n = len(jobs)
    total = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        total[mask] = total[mask ^ low] + jobs[low.bit_length() - 1]
    f = total[:]
    for _ in range(k - 1):
        g = f[:]
        for mask in range(1 << n):
            sub = mask
            while sub:
                cand = max(f[mask ^ sub], total[sub])
                if cand < g[mask]:
                    g[mask] = cand
                sub = (sub - 1) & mask
        f = g
    return f[(1 << n) - 1]


# ---- every example in the statement ----
assert pack([3, 5, 2, 4], 2, 7) == [0, 1, 1, 0]
assert pack([3, 5, 2, 4], 3, 5) == [0, 1, 0, 2]
assert pack([3, 5, 2, 4], 2, 6) is None
assert pack([8], 2, 7) is None
assert pack([], 3, 1) == []
for jobs, k, cap in (([3, 5, 2, 4], 2, 7), ([3, 5, 2, 4], 3, 5), ([], 3, 1)):
    assert is_valid(jobs, k, cap, pack_fast(jobs, k, cap))
assert pack_fast([3, 5, 2, 4], 2, 6) is None and pack_fast([8], 2, 7) is None

assert run(pack, A, 4, 36)[1] == 66005 and is_valid(A, 4, 36, pack(A, 4, 36))
assert run(pack, B, 4, 61) == (None, 350105)
assert sum(A) == 4 * 36 and sum(B) == 4 * 61
A_groups = [[30, 6], [23, 13], [15, 15, 3, 3], [10, 9, 9, 8]]       # the solution quoted for A
assert sorted(A) == sorted(x for grp in A_groups for x in grp) and all(sum(grp) == 36 for grp in A_groups)
can_join_40 = sorted(x for x in B if x + 40 <= 61)
assert can_join_40 == [6, 7, 9, 9, 13, 13]
assert all(sum(c) != 21 for r in range(len(can_join_40) + 1) for c in itertools.combinations(can_join_40, r))
assert subset_dp_peak(B, 4) == 62                    # so cap = 61 really is impossible

# the table in Part 2: each pruning alone and both together
assert [run(pack_fast, A, 4, 36, sort_desc=s, skip_equal=e)[1] for s, e in ((0, 0), (0, 1), (1, 0), (1, 1))] == [66005, 15491, 13, 13]
assert [run(pack_fast, B, 4, 61, sort_desc=s, skip_equal=e)[1] for s, e in ((0, 0), (0, 1), (1, 0), (1, 1))] == [350105, 13356, 1577, 16]
assert is_valid(A, 4, 36, run(pack_fast, A, 4, 36)[0]) and run(pack_fast, B, 4, 61)[0] is None
assert run(pack_fast, A, 4, 36)[1] * 100 <= 66005 and run(pack_fast, B, 4, 61)[1] * 100 <= 350105

for jobs, k, want in (([3, 3, 2, 2, 2], 2, 6), ([3, 5, 2, 4], 3, 5), (A, 4, 36), (B, 4, 62), ([], 2, 0)):
    p, assign = min_peak(jobs, k)
    assert p == want and len(assign) == len(jobs) and peak_of(jobs, k, assign) == p, (jobs, k, p)
assert min_peak([], 2) == (0, [])

# the numbers quoted in Part 3
assert run(min_peak, A, 4)[1] == 13 and run(min_peak, A, 4, use_bound=False)[1] == 700662
assert run(min_peak, B, 4)[1] == 16 and run(min_peak, B, 4, use_bound=False)[1] == 624928
(c_peak, _), c_nodes = run(min_peak, C, 5)
(c_peak2, _), c_nodes_unbounded = run(min_peak, C, 5, use_bound=False)


def lpt(jobs, k):
    loads = [0] * k
    for size in sorted(jobs, reverse=True):
        loads[loads.index(min(loads))] += size
    return max(loads, default=0)


assert lpt(C, 5) == 69 and max(max(C), -(-sum(C) // 5)) == 61 and c_peak == c_peak2 == 62 == subset_dp_peak(C, 5)
assert c_nodes == 4424 and c_nodes_unbounded == 2412066
assert lpt([3, 3, 2, 2, 2], 2) == 7                  # the greedy is not optimal (follow-up 1)

# ---- edge cases ----
assert pack_fast([], 5, 3) == [] and run(pack_fast, [], 5, 3)[1] == 1
assert pack([7], 1, 7) == [0] and pack_fast([7], 1, 7) == [0] and min_peak([7], 1) == (7, [0])
assert min_peak([7], 4)[0] == 7 and min_peak([4, 4], 1)[0] == 8 and pack([4, 4], 1, 7) is None
assert min_peak([2, 9, 3], 14)[0] == 9               # more GPUs than jobs
assert run(pack_fast, [5] * 14, 4, 20)[0] is not None    # all equal: three or four jobs per GPU
assert min_peak([5] * 14, 4)[0] == 20
assert run(pack_fast, [5] * 14, 4, 19) == (None, 2035)   # at most 3 per GPU, 12 < 14; total 70 <= 76 passes the quick check
assert run(pack_fast, [10 ** 6] * 3, 2, 10 ** 7 - 1)[0] is not None and min_peak([10 ** 6] * 14, 14)[0] == 10 ** 6
assert run(pack_fast, [3, 3, 3], 2, 4) == (None, 0)      # the quick total check: 9 > 8, no search at all

# ---- random comparison with the brute forces ----
rng = random.Random(20261118)
saw_none = saw_some = saw_lex_differs = 0
for _ in range(3000):
    n = rng.randint(0, 6)
    k = rng.randint(1, 3)
    jobs = [rng.randint(1, rng.choice([2, 5, 12, 10 ** 6])) for _ in range(n)]
    best = brute_peak(jobs, k) if n else 0
    cap = max(1, best + rng.choice([-3, -1, 0, 0, 1, 4]))
    want = brute_first(jobs, k, cap)
    assert pack(jobs, k, cap) == want, (jobs, k, cap)
    fast = pack_fast(jobs, k, cap)
    assert (fast is None) == (want is None) and (fast is None or is_valid(jobs, k, cap, fast)), (jobs, k, cap)
    p, assign = min_peak(jobs, k)
    assert p == best and peak_of(jobs, k, assign) == p and len(assign) == n, (jobs, k)
    saw_none += want is None
    saw_some += want is not None
    saw_lex_differs += fast is not None and fast != want
assert min(saw_none, saw_some, saw_lex_differs) >= 100, (saw_none, saw_some, saw_lex_differs)

for _ in range(300):                                 # larger n against the subset DP
    n = rng.randint(7, 10)
    k = rng.randint(2, 5)
    jobs = [rng.randint(1, rng.choice([10, 50, 1000])) for _ in range(n)]
    best = subset_dp_peak(jobs, k)
    p, assign = min_peak(jobs, k)
    assert p == best and peak_of(jobs, k, assign) == best
    assert is_valid(jobs, k, best, pack_fast(jobs, k, best)) and pack_fast(jobs, k, best - 1) is None
    k_lpt = lpt(jobs, k)
    assert k_lpt * 3 * k <= best * (4 * k - 1)       # Graham's bound: LPT <= (4/3 - 1/(3k)) * optimum


# ---- node counts: how much the pruning saves, on random instances (no timing involved) ----
def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


rng = random.Random(8)
ratios = {True: [], False: []}
totals = {True: [0, 0], False: [0, 0]}
for _ in range(150):
    n = rng.randint(9, 11)
    k = rng.randint(3, 4)
    jobs = [rng.randint(5, 40) for _ in range(n)]
    best = min_peak(jobs, k)[0]
    for cap in (best - 1, best):
        (_, plain), (_, fast) = run(pack, jobs, k, cap), run(pack_fast, jobs, k, cap)
        ratios[cap == best].append(plain / max(fast, 1))     # fast is 0 when the quick total check already says no
        totals[cap == best][0] += plain
        totals[cap == best][1] += fast
infeasible, feasible = ratios[False], ratios[True]
print(f"infeasible (cap = best - 1): median {median(infeasible):.0f}x fewer nodes, total {totals[False][0]} -> {totals[False][1]}")
print(f"feasible   (cap = best):     median {median(feasible):.1f}x fewer nodes, total {totals[True][0]} -> {totals[True][1]}")
assert median(infeasible) >= 100 and min(infeasible) >= 1          # "a few hundred times", never worse
assert 2 <= median(feasible) <= 30 and min(feasible) < 1             # "a few times", and some instances get worse


# the Stirling-number bound on the nodes per level of the pruned search
def stirling_rows(n, k):
    S = [[0] * (k + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            S[i][j] = S[i - 1][j - 1] + j * S[i - 1][j]
    return [sum(row) for row in S]


assert sum(stirling_rows(12, 4)) == 934120 and sum(4 ** d for d in range(13)) == 22369621
for _ in range(200):
    n = rng.randint(6, 11)
    k = rng.randint(2, 5)
    jobs = [rng.randint(1, 30) for _ in range(n)]
    best = min_peak(jobs, k)[0]
    bound = sum(stirling_rows(n, k))
    assert run(pack_fast, jobs, k, best - 1)[1] <= bound and run(pack_fast, jobs, k, 10 ** 7)[1] <= bound


def full_tree_nodes(jobs, k, cap):
    """Nodes of the Part 2 tree (sorted order, one GPU per load value) when the search never stops early."""
    order = sorted(jobs, reverse=True)
    loads = [0] * k

    def count(pos):
        total = 1
        if pos < len(order):
            tried = set()
            for g in range(k):
                if loads[g] + order[pos] <= cap and loads[g] not in tried:
                    tried.add(loads[g])
                    loads[g] += order[pos]
                    total += count(pos + 1)
                    loads[g] -= order[pos]
        return total

    return count(0)


for _ in range(300):                                 # the bound on min_peak's nodes stated in Part 3
    n = rng.randint(1, 10)
    k = rng.randint(1, 5)
    jobs = [rng.randint(1, rng.choice([5, 30, 1000])) for _ in range(n)]
    assert run(min_peak, jobs, k)[1] <= full_tree_nodes(jobs, k, lpt(jobs, k) - 1)


# ---- follow-ups ----
def min_peak_bisect(jobs, k):
    lo, hi = max(max(jobs, default=0), -(-sum(jobs) // k)), lpt(jobs, k)
    while lo < hi:                                   # invariant: the answer lies in [lo, hi]
        mid = (lo + hi) // 2
        if pack_fast(jobs, k, mid) is not None:
            hi = mid
        else:
            lo = mid + 1
    return lo


def pack_memo(jobs, k, cap, stats=None):
    """pack_fast plus a set of dead states (pos, sorted loads): backtracking that remembers is a DP."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i])
    loads, assign, dead, nodes = [0] * k, [0] * n, set(), 0

    def place(pos):
        nonlocal nodes
        nodes += 1
        if pos == n:
            return True
        key = (pos, tuple(sorted(loads)))
        if key in dead:
            return False
        i, tried = order[pos], set()
        for g in range(k):
            if loads[g] + jobs[i] > cap or loads[g] in tried:
                continue
            tried.add(loads[g])
            loads[g] += jobs[i]
            assign[i] = g
            if place(pos + 1):
                return True
            loads[g] -= jobs[i]
        dead.add(key)
        return False

    found = sum(jobs) <= k * cap and max(jobs, default=0) <= cap and place(0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None


def pack_waste(jobs, k, cap, stats=None):
    """pack_fast plus the wasted-space bound from the last follow-up."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i])
    loads, assign, nodes = [0] * k, [0] * n, 0
    smallest, allowed = min(jobs, default=0), k * cap - sum(jobs)

    def place(pos, waste):
        nonlocal nodes
        nodes += 1
        if pos == n:
            return True
        i, tried = order[pos], set()
        for g in range(k):
            new = loads[g] + jobs[i]
            if new > cap or loads[g] in tried:
                continue
            tried.add(loads[g])
            lost = cap - new if cap - new < smallest else 0      # no remaining job fits in what is left
            if waste + lost > allowed:
                continue
            loads[g] = new
            assign[i] = g
            if place(pos + 1, waste + lost):
                return True
            loads[g] = new - jobs[i]
        return False

    found = allowed >= 0 and max(jobs, default=0) <= cap and place(0, 0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None


for _ in range(1500):
    n = rng.randint(0, 10)
    k = rng.randint(1, 4)
    jobs = [rng.randint(1, rng.choice([2, 3, 10, 40])) for _ in range(n)]
    cap = rng.randint(1, 45)
    want_none = pack_fast(jobs, k, cap) is None
    for fn in (pack_memo, pack_waste):
        out, nodes = run(fn, jobs, k, cap)
        assert (out is None) == want_none and (out is None or is_valid(jobs, k, cap, out)), (fn.__name__, jobs, k, cap)
        assert nodes <= run(pack_fast, jobs, k, cap)[1]
    if n:
        assert min_peak_bisect(jobs, k) == min_peak(jobs, k)[0]

assert run(pack_memo, [5] * 14, 4, 19) == (None, 61)         # follow-up 3: 2035 -> 61 when states repeat

HARD = [1006, 1012, 1016, 1027, 1022, 1033, 1016, 1031, 1040, 1045, 1041, 1005, 1007, 1044, 1049, 1007]
hard_best = 4025
assert min(HARD) == 1005 and max(HARD) == 1049
assert is_valid(HARD, 5, hard_best, pack_fast(HARD, 5, hard_best))  # 4025 fits; 4024 does not (below)
t0 = time.perf_counter()
hard_out, hard_nodes = run(pack_fast, HARD, 5, hard_best - 1)
t1 = time.perf_counter()
waste_out, waste_nodes = run(pack_waste, HARD, 5, hard_best - 1)
t2 = time.perf_counter()
print(f"n = 16 near-equal, cap = best - 1: pack_fast {hard_nodes} nodes ({t1 - t0:.1f} s), "
      f"with the wasted-space bound {waste_nodes} nodes ({t2 - t1:.1f} s)")
assert hard_out is None and waste_out is None
assert hard_nodes == 3904649 and waste_nodes == 1043003

# ---- the largest size in the stated range (n = 14): the hardest inputs a random search found ----
WORST_PACK = ([130, 129, 126, 100, 127, 100, 128, 109, 149, 131, 123, 148, 125, 137], 4)
WORST_BB = ([66, 32, 48, 35, 26, 47, 64, 70, 67, 52, 39, 68, 36, 54], 6)
big_random = ([rng.randint(1, 10 ** 6) for _ in range(14)], 14)
for jobs, k in (WORST_PACK, WORST_BB, big_random, ([5] * 14, 4)):
    t0 = time.perf_counter()
    (best, assign), bb_nodes = run(min_peak, jobs, k)
    (yes, yes_nodes), (no, no_nodes) = run(pack_fast, jobs, k, best), run(pack_fast, jobs, k, best - 1)
    print(f"n = 14, k = {k}: peak {best}, min_peak {bb_nodes} nodes, pack_fast {yes_nodes} / {no_nodes} nodes, "
          f"{time.perf_counter() - t0:.2f} s")
    assert peak_of(jobs, k, assign) == best and is_valid(jobs, k, best, yes) and no is None
print("all checks passed")
```

</details>

</details>
