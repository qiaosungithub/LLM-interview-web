# 跳表：有序多重集合与按名次查询

## 题目

实现一个*有序多重集合*（sorted multiset）：里面存整数，允许重复，可以随时插入、删除，并且能按从小到大的
顺序读出来。要求用*跳表*（skip list）实现。不能用“Python 列表加 `bisect`”，也不能用现成的有序容器。

跳表的结构如下。最底下的第 0 层是一条单向链表，按键从小到大串起所有节点。每个节点在插入时随机得到一个
*高度*（height）`h`：先取 `h = 1`，然后反复抛硬币，每次以概率 `p`（默认 `1/2`）把 `h` 加 1，第一次没抛中
就停，另外 `h` 不超过上限 `MAX_LEVEL = 32`。高度为 `h` 的节点同时出现在第 `0, 1, ..., h-1` 层，每一层都是
一条按键有序的链表。最左边有一个不存键的头节点 `head`，它出现在每一层。

```text
键 3, 7, 12, 19, 25, 31，高度依次是 1, 3, 1, 2, 1, 1：

第 2 层   head ----------> 7 -----------------------------------> None
第 1 层   head ----------> 7 -----------> 19 -------------------> None
第 0 层   head ---> 3 ---> 7 ---> 12 ---> 19 ---> 25 ---> 31 ---> None
```

查找从 `head` 的最高层出发：只要右边节点的键比目标小，就往右走；否则下降一层。在第 0 层走不动时，右边那
个节点就是第一个键不小于目标的节点。上图里找 25：第 2 层从 `head` 走到 7，第 1 层从 7 走到 19，第 0 层不
动，19 的右边就是 25。一共往右走了 2 步；只沿第 0 层走要 4 步。

规模和要求：

- 键是 `[-10**9, 10**9]` 内的整数，操作总数至多 `2 * 10**5`。
- 设 `n` 是当前元素个数。每个操作的*期望*（expected）时间要求是 $O(\log n)$。这里的期望是对结构自己抛的
  硬币取的，对任何输入顺序都要成立，包括按升序插入。
- 构造函数的 `seed` 用来初始化结构自己的随机数发生器 `random.Random(seed)`。它只影响内部各节点的高度，
  **不能影响任何操作的返回值**。`p` 满足 `0 < p < 1`。

### Part 1 —— 插入与查找

```py
class SkipList:
    def __init__(self, seed: int | None = None, p: float = 0.5) -> None: ...
    def add(self, key: int) -> None: ...            # duplicates are kept
    def __contains__(self, key: int) -> bool: ...   # `key in sl`
    def __len__(self) -> int: ...                   # number of stored keys, duplicates counted
    def __iter__(self): ...                         # keys in ascending order, duplicates included
```

重复的键全部保留：`add(10)` 两次，集合里就有两个 10。空集合的长度是 0，遍历得到空序列，任何键都不在里面。

```text
sl = SkipList(seed=7)
依次 add 30, 10, 20, 10

list(sl) == [10, 10, 20, 30]
len(sl) == 4
(20 in sl) is True
(25 in sl) is False
```

### Part 2 —— 删除与范围查询

```py
    def remove(self, key: int) -> bool: ...                      # remove ONE occurrence
    def keys_between(self, lo: int, hi: int) -> list[int]: ...   # every key with lo <= key <= hi
```

`remove(key)` 删掉一个等于 `key` 的元素并返回 `True`；有多个相等的元素时只删一个。集合里没有这个键时返回
`False`，集合不变。

`keys_between(lo, hi)` 按升序返回所有满足 `lo <= key <= hi` 的键，两端都包含，重复的键出现几次就返回几
次。`lo > hi` 时返回空列表。设返回 `m` 个键，期望时间要求是 $O(\log n + m)$。

```text
接着 Part 1 的例子，现在是 [10, 10, 20, 30]

sl.remove(10) is True            # 现在是 [10, 20, 30]
sl.remove(15) is False           # 没有 15，集合不变
sl.keys_between(10, 25) == [10, 20]
sl.keys_between(20, 20) == [20]
sl.keys_between(31, 99) == []
sl.keys_between(25, 10) == []    # lo > hi
```

### Part 3 —— 按名次查询

```py
class IndexedSkipList(SkipList):
    def kth(self, i: int) -> int: ...       # the i-th smallest key, counting from 0
    def rank(self, key: int) -> int: ...    # how many stored keys are strictly smaller than key
```

给前两部分的结构加上两个操作（可以直接改原来的类，也可以像上面这样写成子类）。

`kth(i)` 返回从小到大排在第 `i` 位的键，从 0 数起，相当于 `list(sl)[i]`。`i` 不在 `[0, len)` 内时抛出
`IndexError`；负数下标也算越界，不做 Python 那种倒着数。

`rank(key)` 返回集合里严格小于 `key` 的元素个数，重复的键各算一次。`key` 不必在集合里。

两个操作的期望时间都要求是 $O(\log n)$，`add` 和 `remove` 也要保持期望 $O(\log n)$。沿第 0 层一个一个数过
去是 $O(n)$，不满足要求。

```text
isl = IndexedSkipList(seed=7)
依次 add 30, 10, 20, 10, 20, 5        # 现在是 [5, 10, 10, 20, 20, 30]
isl.remove(10) is True                # 现在是 [5, 10, 20, 20, 30]

isl.kth(0) == 5      isl.kth(1) == 10      isl.kth(3) == 20      isl.kth(4) == 30
isl.kth(5)  -> IndexError
isl.kth(-1) -> IndexError

isl.rank(20) == 2        # 比 20 小的是 5 和 10
isl.rank(21) == 4
isl.rank(5) == 0
isl.rank(1) == 0
isl.rank(99) == 5
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前先问清楚三件事。重复的键怎么处理（全部保留，`remove` 只删一个：这决定了删除时到底该摘哪个节
点）。复杂度是最坏情况还是期望（期望，而且是对结构自己的随机数取的，不是对输入取的）。`seed` 管什么（只管
内部形状，返回值不能随它变：这是可以直接测试的性质）。

45 分钟里的合理节奏是：Part 1 和 Part 2 写完并跑通，Part 3 先讲清楚要多存什么、更新时怎么改，再写代码。

### Part 1

**数据布局。** 一个节点存一个键和一个长度等于高度的数组 `next`，`next[i]` 是它在第 `i` 层的后继。`head`
的高度是 `MAX_LEVEL`。另外记 `_level`：当前真正用到的层数，空表时是 1。

**不变式。** 每个操作结束时下面三条都成立，验证代码在每次操作之后从头检查它们：

1. 第 0 层按键从小到大（相等的挨在一起）串起全部节点。
2. 第 `i` 层恰好是高度大于 `i` 的那些节点，前后顺序和第 0 层一致。
3. `_level` 等于最高节点的高度（空表时为 1），`head` 在第 `_level` 层及以上的指针都是 `None`。

**核心子程序。** `_predecessors(key)` 对每一层 `i` 求出 `update[i]`：第 `i` 层里最后一个键**严格小于**
`key` 的节点，没有就是 `head`。做法就是题目里的查找：从最高层开始，能往右就往右，走不动就把当前节点记为
这一层的答案，然后下降一层，从同一个节点继续。可以从同一个节点继续，是因为它的键小于 `key`，而且它也在下
一层里，所以下一层的答案不会在它左边。走完之后 `update[0].next[0]` 就是第一个键不小于 `key` 的节点，查
找、插入、删除、范围查询都从这里开始。

**插入。** 先求 `update`，再抽高度 `h`，然后在第 `0 .. h-1` 层把新节点接在 `update[i]` 后面，这是普通的
链表插入。`h` 超过 `_level` 时，新启用的那些层的前驱是 `head`，`update` 数组的初值就是它。因为比较用的是
严格小于，新节点会排在已有的相同键的最左边。对多重集合来说放在哪个位置都对，但 Part 2 的删除要靠“我们操
作的总是相同键里最左边的那个”。

**代价为什么是期望 $O(\log n)$。** 记 $L = \log_{1/p} n$。

- 指针数：一个节点高度至少为 $k$ 的概率是 $p^{k-1}$，所以期望高度是 $1 + p + p^2 + \dots = 1/(1-p)$。
  $p = 1/2$ 时平均每个节点 2 个指针。
- 层数：高度超过 $L + c$ 的节点，期望个数是 $n \cdot p^{L+c} = p^c$。所以层数超过 $L + c$ 的概率不超过
  $p^c$，期望层数是 $L + O(1)$。
- 查找步数：把查找路径倒过来看。从终点往回走，站在第 $i$ 层的某个节点上时，如果它的高度大于 $i + 1$，路
  径一定是从上一层下来的，往回就是“上升一层”；否则路径是从左边过来的，往回就是“左移一步”。它的高度是否大
  于 $i+1$ 由一次独立的硬币决定，概率是 $p$。所以每上升一层平均要 $1/p$ 步，升到第 $L$ 层平均要 $L/p$
  步；第 $L$ 层以上的节点期望只有常数个。Pugh 的原始论文给出的结论是：路径长度的期望不超过
  $L/p + 1/(1-p)$，比较次数再加 1。

$p = 1/2$、$n = 8192$ 时这个上界是 $2 \times 13 + 2 + 1 = 29$ 次比较。验证代码按升序插入 8192 个键（这是
让不平衡的二叉搜索树退化成链表的顺序），对 8 个种子各做 2000 次查找：比较次数的平均值约 27，单个种子的平
均值在 24 到 29 之间。注意上界说的是期望，某一次运气不好的抽签可以超过它。

`MAX_LEVEL = 32` 对这道题的规模绰绰有余：$p = 1/2$ 时一个节点抽到 32 的概率是 $2^{-31}$，插入
$2 \times 10^5$ 次，有节点碰到上限的概率不超过 $2 \times 10^5 \cdot 2^{-31} < 10^{-4}$。就算碰到了，被截断
的只是高度，结构仍然正确。

```python
import random

MAX_LEVEL = 32


class _Node:
    __slots__ = ("key", "next")

    def __init__(self, key, height: int) -> None:
        self.key = key
        self.next = [None] * height       # next[i]: the following node on level i


class SkipList:
    """A sorted multiset. Level 0 links every node in key order; level i links the nodes taller than i."""

    def __init__(self, seed: int | None = None, p: float = 0.5) -> None:
        self._rng = random.Random(seed)   # the structure's own coins; they never change an answer
        self._p = p
        self._head = _Node(None, MAX_LEVEL)
        self._level = 1                   # number of levels in use
        self._n = 0

    def _random_height(self) -> int:
        h = 1
        while h < MAX_LEVEL and self._rng.random() < self._p:
            h += 1
        return h

    def _predecessors(self, key) -> list:
        """update[i] = the last node on level i whose key is < key (the head if there is none)."""
        update = [self._head] * MAX_LEVEL
        x = self._head
        for i in range(self._level - 1, -1, -1):
            nxt = x.next[i]
            while nxt is not None and nxt.key < key:     # NOTE: strict <, so we stop left of equal keys
                x = nxt
                nxt = x.next[i]
            update[i] = x                 # the next level continues from the same node
        return update

    def __len__(self) -> int:
        return self._n

    def __contains__(self, key) -> bool:
        x = self._predecessors(key)[0].next[0]           # the first node with key >= key, if any
        return x is not None and x.key == key

    def __iter__(self):
        x = self._head.next[0]
        while x is not None:
            yield x.key
            x = x.next[0]

    def add(self, key) -> None:
        update = self._predecessors(key)
        h = self._random_height()
        if h > self._level:
            self._level = h               # update[i] is already the head on the newly used levels
        node = _Node(key, h)
        for i in range(h):
            node.next[i] = update[i].next[i]
            update[i].next[i] = node
        self._n += 1
```

### Part 2

**删除。** 求出 `update`，令 `target = update[0].next[0]`。如果它不存在或者键不等于 `key`，集合里没有这个
键，返回 `False`。否则要把 `target` 从它所在的每一层摘掉。

关键的一点：对每个 `i < target 的高度`，都有 `update[i].next[i] is target`。理由是，`target` 是第 0 层顺
序里第一个键不小于 `key` 的节点；`update[i]` 是第 `i` 层里最后一个键小于 `key` 的节点，所以它在第 `i` 层
的后继是“第 `i` 层里第一个键不小于 `key` 的节点”；`target` 在第 `i` 层里，又排在所有这样的节点的最左
边，所以这个后继就是 `target`。于是每一层的摘除都是一句 `update[i].next[i] = target.next[i]`。

这也是有重复键时最容易写错的地方。如果查找时在高层一看到相等的键就停下，拿到的可能是一个比较高、但不是最
左边的相等节点；这时低层的 `update[i]` 指向的是它左边另一个相等的节点，照上面那句去改指针，会把两者之间
的节点一起从链表里丢掉。验证代码里有一个故意这样写的版本，随机测试能抓到它。

摘完之后，如果最高层空了，就把 `_level` 往下降，保持不变式 3。

**范围查询。** `_predecessors(lo)[0].next[0]` 是第一个键不小于 `lo` 的节点，从它开始沿第 0 层往右走，直
到键超过 `hi`。定位是期望 $O(\log n)$，之后每输出一个键走一步，合计期望 $O(\log n + m)$。`lo > hi` 时第
一个节点的键已经大于 `hi`，循环一次都不执行，返回空列表，不需要单独判断。

```python
def remove(self, key) -> bool:
    update = self._predecessors(key)
    target = update[0].next[0]            # the LEFTMOST node with this key, if the key is present
    if target is None or target.key != key:
        return False
    for i in range(len(target.next)):
        update[i].next[i] = target.next[i]               # NOTE: update[i].next[i] is target on these levels
    while self._level > 1 and self._head.next[self._level - 1] is None:
        self._level -= 1                  # the tallest tower may just have gone
    self._n -= 1
    return True


def keys_between(self, lo, hi) -> list:
    out = []
    x = self._predecessors(lo)[0].next[0]
    while x is not None and x.key <= hi:
        out.append(x.key)
        x = x.next[0]
    return out


SkipList.remove = remove
SkipList.keys_between = keys_between
```

### Part 3

**要多存什么。** 给每个指针配一个*跨度*（span）：这个指针一步跨过了第 0 层的多少个位置。位置这样编号：
`head` 是 0，第 `i` 小的键（从 0 数起）是 `i + 1`，末尾的 `None` 是 `n + 1`。新增的不变式是：

4. 对每个节点 `x`（包括 `head`）和它在使用中的每一层 `i`，`x.span[i]` 等于 `x.next[i]` 的位置减去 `x` 的
   位置。

把 `None` 也算作一个位置（`n + 1`），是为了让指向末尾的指针也有确定的跨度，这样插入、删除、查询都不需要
为“后面没有节点了”写特殊情况。

```text
位置      0         1      2      3       4       5       6       7
第 2 层   head ----------> 7 -----------------------------------> None      跨度 2, 5
第 1 层   head ----------> 7 -----------> 19 -------------------> None      跨度 2, 2, 3
第 0 层   head ---> 3 ---> 7 ---> 12 ---> 19 ---> 25 ---> 31 ---> None      跨度全是 1
```

**查询。** `kth(i)` 要走到位置 `want = i + 1`。从最高层开始，记当前位置为 `at`；只要 `at + span <= want`
就往右走并把跨度加到 `at` 上，否则下降一层。因为 `None` 的位置 `n + 1` 大于 `want`，循环不会走出链表。第 0
层的跨度都是 1，所以结束时 `at == want`。上图里 `kth(4)`：`want = 5`，第 2 层走到 7（`at = 2`），第 1 层走到
19（`at = 4`），第 0 层走到 25（`at = 5`），返回 25。

`rank(key)` 就是普通的查找，一路把走过的跨度加起来。停下时所在的节点是最后一个键小于 `key` 的节点，它的位
置正好是“严格小于 `key` 的键的个数”。

**更新。** 查找时顺便记下每层前驱的位置 `pos[i]`。新节点的位置是 `at = pos[0] + 1`。

- 第 `i < h` 层：指针 `update[i] → 后继` 被新节点切成两段。`update[i]` 到新节点的跨度是 `at - pos[i]`。
  原来的后继因为前面多了一个节点，位置变成 `pos[i] + 旧跨度 + 1`，所以新节点到它的跨度是
  `pos[i] + 旧跨度 + 1 - at`。
- 第 `h <= i < _level` 层：新节点不在这一层，但它落在 `update[i]` 那个指针的下面，这个指针的跨度加 1。**这
  一步最容易忘。**
- 新启用的层：先把 `head` 在这些层的跨度设成 `n + 1`（指向 `None`），再按第一条处理。

删除是反过来的：`target` 所在的层，前驱的跨度变成“两段之和减 1”；`target` 上方的层，跨度减 1。

**代价。** `kth` 和 `rank` 走的路径和普通查找同一种形状，期望 $O(\log n)$。`add` 和 `remove` 现在要改所有
使用中的层，而不只是新节点所在的 `h` 层；层数的期望是 $O(\log n)$，所以仍然是期望 $O(\log n)$。内存上每
个指针多一个整数。

```python
class _SpanNode(_Node):
    __slots__ = ("span",)

    def __init__(self, key, height: int) -> None:
        super().__init__(key, height)
        self.span = [1] * height          # span[i]: how many level-0 positions next[i] moves forward


class IndexedSkipList(SkipList):
    """Positions: the head is 0, the i-th smallest key is i + 1, the None at the end is n + 1.
    Invariant: x.span[i] == position(x.next[i]) - position(x) on every level in use."""

    def __init__(self, seed: int | None = None, p: float = 0.5) -> None:
        super().__init__(seed, p)
        self._head = _SpanNode(None, MAX_LEVEL)

    def _predecessors_with_pos(self, key) -> tuple[list, list[int]]:
        """As _predecessors, plus pos[i] = the position of update[i]."""
        update = [self._head] * MAX_LEVEL
        pos = [0] * MAX_LEVEL
        x, at = self._head, 0
        for i in range(self._level - 1, -1, -1):
            nxt = x.next[i]
            while nxt is not None and nxt.key < key:
                at += x.span[i]
                x = nxt
                nxt = x.next[i]
            update[i], pos[i] = x, at
        return update, pos

    def add(self, key) -> None:
        update, pos = self._predecessors_with_pos(key)
        h = self._random_height()
        if h > self._level:
            for i in range(self._level, h):
                self._head.span[i] = self._n + 1         # a fresh level: head -> None
            self._level = h
        node = _SpanNode(key, h)
        at = pos[0] + 1                   # the new node's position
        for i in range(self._level):
            pred = update[i]
            if i < h:
                node.next[i] = pred.next[i]
                pred.next[i] = node
                node.span[i] = pos[i] + pred.span[i] + 1 - at    # the old target moved one place right
                pred.span[i] = at - pos[i]
            else:
                pred.span[i] += 1         # NOTE: the pointer passes over the new node -- easy to forget
        self._n += 1

    def remove(self, key) -> bool:
        update, _ = self._predecessors_with_pos(key)
        target = update[0].next[0]
        if target is None or target.key != key:
            return False
        h = len(target.next)
        for i in range(self._level):
            pred = update[i]
            if i < h:
                pred.span[i] += target.span[i] - 1       # two pointers merge, one position disappears
                pred.next[i] = target.next[i]
            else:
                pred.span[i] -= 1
        while self._level > 1 and self._head.next[self._level - 1] is None:
            self._level -= 1
        self._n -= 1
        return True

    def kth(self, i: int):
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range for {self._n} keys")
        x, at, want = self._head, 0, i + 1
        for lv in range(self._level - 1, -1, -1):
            while at + x.span[lv] <= want:               # NOTE: None sits at n + 1 > want, so x never
                at += x.span[lv]                         #       runs off the end
                x = x.next[lv]
        return x.key

    def rank(self, key) -> int:
        return self._predecessors_with_pos(key)[1][0]    # position of the last node with a smaller key
```

### 追问

- **什么时候“排好序的数组加二分”就够了？** 比想象的更经常。如果集合建好之后不再变，排序数组全面占优：
  `kth` 是直接下标，`rank` 是一次二分，内存连续。有插入删除时，`bisect.insort` 定位是 $O(\log n)$，挪动元
  素是 $O(n)$，但挪动是一次 `memmove`，常数极小。在这台机器的 CPython 上测过一次（换机器数字会变）：插
  入 2 万个随机键，`insort` 比上面的跳表快大约一倍；到 6 万个左右两者持平；带跨度的版本更慢，到本题的上
  限 20 万个键时仍然不比 `insort` 快。所以在 Python 里、在这个规模下，面试时应该先说“我会用排序数组”，再
  说明什么情况下它不够：$n$ 更大而且更新频繁（`insort` 全部插入的总代价是 $O(n^2)$），换成指针操作便宜的
  语言，或者需要多线程同时修改。
- **“期望 $O(\log n)$”的期望是对什么取的？最坏情况呢？** 是对结构自己的硬币取的，对输入没有任何假设，所
  以按升序插入不会像不平衡的二叉搜索树那样退化。最坏情况仍然是 $O(n)$：如果所有节点的高度都抽到 1，它就
  是一条链表，只是概率极小。层数超过 $\log_2 n + c$ 的概率不超过 $2^{-c}$；验证代码对 400 个种子数了
  $n = 1024$、$c = 4$ 的情形，约 6% 的种子超过，上界是 6.25%。真正的弱点是随机数被对手知道：能看到种子或者
  能观察到各节点高度的人，可以专门删掉高的节点，让结构退化。这种场合要用确定性的平衡树。
- **`p` 怎么选？** 这是内存和查找时间之间的旋钮。每个节点的期望指针数是 $1/(1-p)$：$p = 1/2$ 时是 2，
  $p = 1/4$ 时是 1.33。查找步数的主项是 $(1/p)\log_{1/p} n$：$p = 1/2$ 时是 $2\log_2 n$，$p = 1/4$ 时是
  $4 \log_4 n = 2\log_2 n$，两者相同。所以 $p = 1/4$ 省掉三分之一的指针，查找并不变慢（验证代码实测
  $n = 8192$ 时两者的平均比较次数是 27 和 24.5 左右），代价是步数的波动更大。带跨度时每个指针还要多一个整
  数，小一点的 `p` 更划算。Redis 的有序集合用的就是带跨度的跳表，`p` 取 1/4。
- **加了跨度之后，更新付出了什么？** 不带跨度时，一次插入只改新节点所在的 `h` 层，平均 2 层；带跨度后要
  改所有使用中的层，大约 $\log_{1/p} n$ 层，内存也从每个键约 120 字节涨到约 200 字节（CPython，
  $p = 1/2$；排序数组里每个键只占一个 8 字节的指针）。实测插入慢了将近一倍。所以只在确实需要名次时才
  加。有了 `rank`，“区间里有多少个键”也不必把键列出来：对整数键，答案是
  `max(0, rank(hi + 1) - rank(lo))`，期望 $O(\log n)$，和区间里有多少个键无关。如果所有可能出现的键事先已
  知（离线，或者值域很小），在键的值域上建一个树状数组（Fenwick tree）也能回答名次，而且更短。
- **和平衡二叉搜索树比，为什么有人选跳表？** 两者的操作集合和量级相同，区别在保证的性质和实现的难度。红黑
  树、AVL 树给的是最坏情况 $O(\log n)$，跳表给的是期望。跳表没有旋转，插入和删除只改前驱的几个指针，代码
  短得多，45 分钟里写得完；第 0 层本身就是有序链表，范围扫描不需要中序遍历。因为修改是局部的，它也比较容
  易做成并发结构（Java 的 `ConcurrentSkipListMap` 就是）。代价是指针多、节点分散，缓存命中不如数组和
  B 树。

<details>
<summary>验证代码（可运行）</summary>

```python
import bisect
import math
import sys
import time


def _raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


def _filled(cls, keys, seed=7, p=0.5):
    sl = cls(seed=seed, p=p)
    for key in keys:
        sl.add(key)
    return sl


# --- the statement's examples for Parts 1 and 2, on both classes and under several seeds ---
assert len(SkipList()) == 0 and list(SkipList()) == [] and (5 in SkipList()) is False      # default arguments
for cls in (SkipList, IndexedSkipList):
    for seed in (0, 7, 12345):
        sl = _filled(cls, (30, 10, 20, 10), seed)
        assert list(sl) == [10, 10, 20, 30] and len(sl) == 4
        assert (20 in sl) is True and (25 in sl) is False
        assert sl.remove(10) is True and list(sl) == [10, 20, 30] and len(sl) == 3
        assert sl.remove(15) is False and list(sl) == [10, 20, 30] and len(sl) == 3
        assert sl.keys_between(10, 25) == [10, 20]
        assert sl.keys_between(20, 20) == [20]
        assert sl.keys_between(31, 99) == []
        assert sl.keys_between(25, 10) == []
        empty = cls(seed)
        assert len(empty) == 0 and list(empty) == [] and (5 in empty) is False
        assert empty.remove(5) is False and empty.keys_between(-10**9, 10**9) == []

# --- the statement's example for Part 3 ---
for seed in (0, 7, 12345):
    isl = _filled(IndexedSkipList, (30, 10, 20, 10, 20, 5), seed)
    assert list(isl) == [5, 10, 10, 20, 20, 30]
    assert isl.remove(10) is True and list(isl) == [5, 10, 20, 20, 30]
    assert [isl.kth(i) for i in range(5)] == [5, 10, 20, 20, 30]
    assert _raises(IndexError, isl.kth, 5) and _raises(IndexError, isl.kth, -1)
    assert (isl.rank(20), isl.rank(21), isl.rank(5), isl.rank(1), isl.rank(99)) == (2, 4, 0, 0, 5)
    empty = IndexedSkipList(seed)
    assert empty.rank(3) == 0 and _raises(IndexError, empty.kth, 0)


# --- the two pictures on this page, rebuilt with the heights they show ---
def _with_heights(cls, keys, heights):
    sl = cls(seed=0)
    it = iter(heights)
    sl._random_height = lambda: next(it)             # choose the heights instead of flipping coins
    for key in keys:
        sl.add(key)
    return sl


def _level_nodes(sl, i):
    out = []
    x = sl._head.next[i]
    while x is not None:
        out.append(x)
        x = x.next[i]
    return out


def _comparisons(sl, key):
    """The walk of _predecessors, counted: (moves to the right, loop tests). Every loop test either moves
    right or ends a level, so tests = moves + levels."""
    moves = 0
    x = sl._head
    for i in range(sl._level - 1, -1, -1):
        while x.next[i] is not None and x.next[i].key < key:
            x = x.next[i]
            moves += 1
    return moves, moves + sl._level


PICTURE_KEYS = (3, 7, 12, 19, 25, 31)
pic = _with_heights(IndexedSkipList, PICTURE_KEYS, (1, 3, 1, 2, 1, 1))
assert pic._level == 3
assert [[x.key for x in _level_nodes(pic, i)] for i in (2, 1, 0)] == [[7], [7, 19], list(PICTURE_KEYS)]
assert [u.key for u in pic._predecessors(25)[:3]] == [19, 19, 7]     # levels 0, 1, 2
assert _comparisons(pic, 25)[0] == 2                                 # "2 steps to the right"
flat = _with_heights(SkipList, PICTURE_KEYS, (1,) * 6)               # level 0 only: a plain linked list
assert _comparisons(flat, 25)[0] == 4                                # "4 steps along level 0"
head = pic._head
assert [x.span[2] for x in [head] + _level_nodes(pic, 2)] == [2, 5]
assert [x.span[1] for x in [head] + _level_nodes(pic, 1)] == [2, 2, 3]
assert [x.span[0] for x in [head] + _level_nodes(pic, 0)] == [1] * 7
assert pic.kth(4) == 25 and pic.rank(25) == 4


# --- invariants 1-4 of the explanation, checked from scratch against a sorted Python list ---
def _check_invariants(sl, model):
    nodes = _level_nodes(sl, 0)
    assert [x.key for x in nodes] == model and len(sl) == len(model)          # 1: level 0 holds every key
    tallest = max((len(x.next) for x in nodes), default=1)
    assert sl._level == tallest <= MAX_LEVEL                                  # 3: _level is the tallest tower
    assert all(sl._head.next[i] is None for i in range(sl._level, MAX_LEVEL))
    position = {id(x): i + 1 for i, x in enumerate(nodes)}
    position[id(sl._head)] = 0
    indexed = isinstance(sl, IndexedSkipList)
    for i in range(sl._level):
        want = [x for x in nodes if len(x.next) > i]                          # 2: level i = towers taller than i
        got = _level_nodes(sl, i)
        assert len(got) == len(want) and all(a is b for a, b in zip(got, want))
        if indexed:                                                           # 4: spans are position differences
            for x, nxt in zip([sl._head] + got, got + [None]):
                target = len(nodes) + 1 if nxt is None else position[id(nxt)]
                assert x.span[i] == target - position[id(x)]


def _run_random(cls, seed, n_ops, key_range, p=0.5, check_every=1):
    """A random operation sequence against the brute-force model: a sorted list scanned linearly."""
    rng = random.Random(seed)
    sl = cls(seed=seed * 7 + 1, p=p)
    model = []
    indexed = issubclass(cls, IndexedSkipList)
    for step in range(n_ops):
        r = rng.random()
        key = rng.randint(-key_range, key_range)
        if r < 0.40:
            sl.add(key)
            bisect.insort(model, key)
        elif r < 0.65:
            if model and rng.random() < 0.5:
                key = rng.choice(model)                  # half of the removals are guaranteed to hit
            present = key in model
            assert sl.remove(key) is present
            if present:
                model.remove(key)
        elif r < 0.75:
            assert (key in sl) is (key in model)
        elif r < 0.87:
            hi = key + rng.randint(-2, 12) * max(1, key_range // 8)
            want = [k for k in model if key <= k <= hi]
            assert sl.keys_between(key, hi) == want
            if indexed:                                  # the counting formula from the follow-ups
                assert max(0, sl.rank(hi + 1) - sl.rank(key)) == len(want)
        elif indexed and r < 0.94:
            assert sl.rank(key) == sum(k < key for k in model)
        elif indexed:
            i = rng.randint(-2, len(model) + 1)
            if 0 <= i < len(model):
                assert sl.kth(i) == model[i]
            else:
                assert _raises(IndexError, sl.kth, i)
        else:
            assert len(sl) == len(model) and (not model or (model[0] in sl and model[-1] in sl))
        if step % check_every == 0:
            _check_invariants(sl, model)
    _check_invariants(sl, model)
    assert list(sl) == model
    return len(model)


for cls in (SkipList, IndexedSkipList):
    for seed in range(40):
        _run_random(cls, seed, 400, 12)                  # few distinct keys: duplicates everywhere
    for seed in range(10):
        _run_random(cls, 100 + seed, 400, 10**9)         # the full key range
    for p in (0.25, 0.9, 1.0):                           # p = 1.0 is outside the contract: every tower hits
        for seed in range(5):                            # MAX_LEVEL, which exercises the cap
            _run_random(cls, 200 + seed, 300, 12, p=p)
    assert _run_random(cls, 300, 4000, 50) > 300         # one long run, invariants after EVERY operation
    assert _run_random(cls, 301, 30000, 10**9, check_every=500) > 3000
    assert _run_random(cls, 302, 30000, 400, check_every=500) > 3000


# --- negative controls: the two mistakes the explanation warns about must be caught by the tests above ---
class _ForgetsToWiden(IndexedSkipList):
    def add(self, key):
        update, pos = self._predecessors_with_pos(key)
        h = self._random_height()
        if h > self._level:
            for i in range(self._level, h):
                self._head.span[i] = self._n + 1
            self._level = h
        node = _SpanNode(key, h)
        at = pos[0] + 1
        for i in range(h):                               # BUG: the levels above h never get their += 1
            pred = update[i]
            node.next[i] = pred.next[i]
            pred.next[i] = node
            node.span[i] = pos[i] + pred.span[i] + 1 - at
            pred.span[i] = at - pos[i]
        self._n += 1


class _RemovesAnyEqualNode(SkipList):
    def remove(self, key):
        update = [self._head] * MAX_LEVEL
        x, target = self._head, None
        for i in range(self._level - 1, -1, -1):
            while x.next[i] is not None and x.next[i].key < key:
                x = x.next[i]
            update[i] = x
            if target is None and x.next[i] is not None and x.next[i].key == key:
                target = x.next[i]                       # BUG: the first equal node seen from above is the
        if target is None:                               #      tallest-leftmost, not the leftmost
            return False
        for i in range(len(target.next)):
            update[i].next[i] = target.next[i]
        while self._level > 1 and self._head.next[self._level - 1] is None:
            self._level -= 1
        self._n -= 1
        return True


for broken in (_ForgetsToWiden, _RemovesAnyEqualNode):
    assert all(_raises(AssertionError, _run_random, broken, seed, 400, 12) for seed in range(5)), broken


# --- the seed changes the shape, never an answer ---
def _transcript(seed, p):
    rng = random.Random(99)                              # the same operations every time
    sl = IndexedSkipList(seed=seed, p=p)
    out = []
    for _ in range(1500):
        r, key = rng.random(), rng.randint(-30, 30)
        if r < 0.45:
            sl.add(key)
        elif r < 0.70:
            out.append(sl.remove(key))
        elif r < 0.80:
            out.append(key in sl)
        elif r < 0.90:
            out.append(sl.keys_between(key, key + 9))
        else:
            out.append((len(sl), sl.rank(key), sl.kth(len(sl) // 2) if len(sl) else None))
    out.append(list(sl))
    return out, sl._level


base, base_level = _transcript(0, 0.5)
levels_seen = {base_level}
for seed in range(1, 8):
    out, level = _transcript(seed, 0.5)
    assert out == base
    levels_seen.add(level)
assert len(levels_seen) > 1                              # ... while the shapes really did differ
for p in (0.01, 0.25, 0.9, 1.0):
    assert _transcript(3, p)[0] == base

# --- adversarial input 1: every key equal ---
for cls in (SkipList, IndexedSkipList):
    sl, model = cls(seed=5), []
    for i in range(3000):
        sl.add(7)
        model.append(7)
        if i % 300 == 0:
            _check_invariants(sl, model)
    _check_invariants(sl, model)
    assert 7 in sl and 6 not in sl and 8 not in sl
    assert sl.keys_between(7, 7) == model and sl.keys_between(8, 9) == [] and sl.keys_between(-5, 6) == []
    assert _comparisons(sl, 8)[1] < 150                  # passing 3000 equal keys does not take 3000 steps
    if cls is IndexedSkipList:
        assert sl.rank(7) == 0 and sl.rank(8) == 3000 and sl.kth(0) == sl.kth(2999) == 7
    for i in range(3000):
        assert sl.remove(7) is True
        model.pop()
        if i % 300 == 0:
            _check_invariants(sl, model)
    assert sl.remove(7) is False and len(sl) == 0 and sl._level == 1
    _check_invariants(sl, [])

# --- adversarial input 2: the largest sequence the statement allows, in ascending order ---
N_MAX = 200_000
big = IndexedSkipList(seed=11)
for key in range(-N_MAX // 2, N_MAX // 2):
    big.add(key)
assert len(big) == N_MAX
_check_invariants(big, list(range(-N_MAX // 2, N_MAX // 2)))
rng = random.Random(4)
total_tests = 0
for _ in range(2000):
    i = rng.randrange(N_MAX)
    key = i - N_MAX // 2
    assert big.kth(i) == key and big.rank(key) == i and key in big
    total_tests += _comparisons(big, key)[1]
assert total_tests / 2000 < 60                           # about 2 * log2(200000) = 35, nowhere near 200000
assert big.keys_between(-3, 3) == [-3, -2, -1, 0, 1, 2, 3] and big.rank(10**9) == N_MAX
down = _filled(SkipList, range(20_000, 0, -1), seed=3)   # and descending
assert list(down) == list(range(1, 20_001))
assert sum(_comparisons(down, rng.randrange(1, 20_001))[1] for _ in range(1000)) / 1000 < 60

# --- the cost claims of Part 1, across 8 seeds: pointers per node and comparisons per search ---
N_COST = 8192
measured = {}
for p in (0.5, 0.25):
    levels_needed = math.log(N_COST, 1 / p)
    bound = levels_needed / p + 1 / (1 - p) + 1          # Pugh's bound on the expected number of comparisons
    means = []
    for seed in range(8):
        sl = _filled(SkipList, range(N_COST), seed, p)   # ascending: the order that ruins a plain BST
        pointers = sum(len(x.next) for x in _level_nodes(sl, 0)) / N_COST
        assert abs(pointers - 1 / (1 - p)) < 0.03 / (1 - p)          # within 3% of 1 / (1 - p)
        rng = random.Random(seed)
        mean = sum(_comparisons(sl, rng.randrange(N_COST))[1] for _ in range(2000)) / 2000
        assert mean < 1.2 * bound                        # one unlucky seed may exceed an EXPECTATION a little
        means.append(mean)
    measured[p] = (bound, sum(means) / len(means), min(means), max(means))
    assert measured[p][1] <= bound                       # ... but the average over seeds does not
bound, average, low, high = measured[0.5]
assert bound == 29 and 26.5 < average < 27.5 and 24 < low and high < 29      # the numbers quoted in Part 1
assert 24 < measured[0.25][1] < 25                       # "about 24.5" for p = 1/4

# --- the tail bound: P(more than log2(n) + c levels) <= 2**-c ---
too_tall = 0
for seed in range(400):
    coins = SkipList(seed=seed)
    too_tall += max(coins._random_height() for _ in range(1024)) > 10 + 4
assert 0.04 < too_tall / 400 <= 2 ** -4                  # "about 6%" against the bound 6.25%

# --- memory per key in CPython (follow-ups) ---
plain = _filled(SkipList, range(20_000), 0)
spans = _filled(IndexedSkipList, range(20_000), 0)
plain_bytes = sum(sys.getsizeof(x) + sys.getsizeof(x.next) for x in _level_nodes(plain, 0)) / 20_000
span_bytes = sum(sys.getsizeof(x) + sys.getsizeof(x.next) + sys.getsizeof(x.span)
                 for x in _level_nodes(spans, 0)) / 20_000
assert 112 < plain_bytes < 128 and 190 < span_bytes < 210
assert sys.getsizeof(list(range(20_000))) / 20_000 < 8.1


# --- timing (follow-ups): printed, not asserted, because wall-clock time on a shared machine is noisy ---
def _best_of_three(fn):
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


rng = random.Random(0)
timing_keys = [rng.randrange(10**9) for _ in range(20_000)]


def _insort_all():
    arr = []
    for key in timing_keys:
        bisect.insort(arr, key)


t_skip = _best_of_three(lambda: _filled(SkipList, timing_keys, 0))
t_span = _best_of_three(lambda: _filled(IndexedSkipList, timing_keys, 0))
t_insort = _best_of_three(_insort_all)
# The text quotes one measurement: insort about twice as fast as the skip list at 20000 keys, spans costing
# almost another factor of two. Compare with what this run prints.
print(f"insert 20000 keys: insort {t_insort:.3f}s, skip list {t_skip:.3f}s, skip list with spans {t_span:.3f}s")

print("all checks passed")
```

</details>

</details>
