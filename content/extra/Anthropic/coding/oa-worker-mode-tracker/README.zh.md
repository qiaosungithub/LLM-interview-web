# 在线测评：多 worker 众数统计

## 题目

很多 worker 各自上报整数（可以把每个 worker 理解成一个数据分片，shard）。实现一个 `ModeTracker` 类，把
它们的数据汇总起来，随时回答“全局出现次数最多的值是哪个”。还要能把某个 worker 整个移除：它上报过的
所有数据都要从统计里抹掉。

题目分四关，按顺序解锁。每一关都在同一个 `ModeTracker` 类上增加方法，前面各关的方法继续有效。

对四关都适用的约定：

- worker 的 id 是非空字符串，区分大小写。
- 上报的值是整数，$-10^9 \le \text{value} \le 10^9$。
- 一次测试里，方法调用总数不超过 $10^3$，所有 `report` 上报的值加起来不超过 $10^4$ 个。
- 失败的调用不改变任何状态。
- 下面四关的例子是连着的：每一关都接着上一关结束时的状态往下做。

### 第 1 关 —— 收集各个 worker 的数据

```py
class ModeTracker:
    def add_worker(self, worker_id: str) -> bool:
        """Registers a worker. False if a worker with this id is registered already."""

    def report(self, worker_id: str, values: list[int]) -> bool:
        """The worker reports these values (1 to 1000 of them). Every list element counts once,
        so a value that appears twice in the list is counted twice.
        False, and nothing is recorded, if the worker is not registered."""

    def get_count(self, value: int) -> int:
        """How many times `value` has been reported, summed over all registered workers.
        0 if nobody reported it."""

    def get_worker_count(self, worker_id: str, value: int) -> int | None:
        """How many times this worker has reported `value` (possibly 0).
        None if the worker is not registered."""
```

```text
t = ModeTracker()
t.add_worker("w1")                -> True
t.add_worker("w1")                -> False     # 已经注册过
t.add_worker("w2")                -> True
t.report("w1", [5, 3, 5])         -> True
t.report("w2", [3, 3, 9])         -> True
t.report("w9", [1])               -> False     # w9 没有注册，这次上报不记录
t.get_count(3)                    -> 3         # w1 报了 1 次，w2 报了 2 次
t.get_count(5)                    -> 2
t.get_count(1)                    -> 0
t.get_worker_count("w1", 5)       -> 2
t.get_worker_count("w2", 5)       -> 0         # 注册了，但没报过 5
t.get_worker_count("w9", 5)       -> None      # 没有注册
```

### 第 2 关 —— 全局众数

```py
    def get_mode(self) -> int | None:
        """The value with the highest global count. If several values share the highest count,
        the smallest of them. None if there is no data at all."""

    def top_values(self, k: int) -> list[str]:
        """The k values with the highest global counts, each formatted as "value(count)",
        ordered by count descending, then by value ascending. Fewer than k values: all of them."""
```

- 并列时比较的是数值，不是字符串：`2` 排在 `10` 前面，`-4` 排在 `2` 前面。
- 全局计数为 0 的值不算数据，不会出现在任何结果里。
- `k >= 1`。没有数据时 `top_values` 返回 `[]`。
- 格式就是 Python 的 `f"{value}({count})"`，负数写成 `"-4(2)"`。

接着第 1 关的状态（全局计数：3 出现 3 次，5 出现 2 次，9 出现 1 次）：

```text
t.get_mode()                      -> 3
t.top_values(2)                   -> ["3(3)", "5(2)"]
t.top_values(10)                  -> ["3(3)", "5(2)", "9(1)"]
t.report("w1", [5])               -> True
t.get_mode()                      -> 3         # 3 和 5 都是 3 次，取较小的值
t.top_values(1)                   -> ["3(3)"]
```

### 第 3 关 —— 移除 worker

```py
    def remove_worker(self, worker_id: str) -> int | None:
        """Unregisters the worker and erases everything it ever reported from all counts.
        Returns how many reported values were erased (0 for a worker that never reported).
        None if the worker is not registered."""
```

- 移除之后，所有查询的结果都和“这个 worker 从来没有存在过”一样。
- 这个 id 以后可以再次 `add_worker`，得到的是一个全新的、没有任何数据的 worker。
- 注意 `0` 和 `None` 不是一回事：注册了但没报过数据的 worker，移除时返回 `0`。

接着上面的状态（w1 报过 5、3、5、5；w2 报过 3、3、9）：

```text
t.remove_worker("w2")             -> 3         # 抹掉 w2 的 3 个值：3、3、9
t.get_count(3)                    -> 1
t.get_count(9)                    -> 0
t.get_mode()                      -> 5
t.top_values(10)                  -> ["5(3)", "3(1)"]     # 9 的计数回到 0，不再出现
t.remove_worker("w2")             -> None      # 已经不在了
t.report("w2", [9])               -> False
t.add_worker("w2")                -> True      # 重新注册，是一个全新的 worker
t.get_worker_count("w2", 3)       -> 0
```

### 第 4 关 —— 合并 worker

（这一关是我们补的：来源只有前三关对应的要点。这类测评的第 4 关常见的形式之一是“合并”。）

```py
    def merge_workers(self, target_id: str, source_id: str) -> bool:
        """Everything `source_id` has reported now counts as reported by `target_id`,
        and `source_id` is unregistered. False if either worker is not registered
        or the two ids are equal."""
```

- 合并不改变任何全局计数，`get_mode` 和 `top_values` 的结果不变。
- 合并之后 `remove_worker(target_id)` 会把并进来的数据一起抹掉。
- `source_id` 以后可以重新注册，得到全新的 worker。

接着上面的状态（w1 报过 5、3、5、5；w2 刚重新注册，没有数据）：

```text
t.report("w2", [7, 7, 5])         -> True
t.add_worker("w3")                -> True
t.report("w3", [7])               -> True
t.merge_workers("w1", "w2")       -> True
t.get_worker_count("w1", 7)       -> 2         # 从 w2 并进来的
t.get_worker_count("w1", 5)       -> 4         # 自己的 3 次 + 并进来的 1 次
t.get_worker_count("w2", 7)       -> None      # w2 不再是注册的 worker
t.get_count(7)                    -> 3         # 全局计数不受合并影响
t.get_mode()                      -> 5         # 5 出现 4 次，7 出现 3 次
t.merge_workers("w1", "w2")       -> False     # w2 不存在
t.merge_workers("w1", "w1")       -> False     # 两个 id 相同
t.remove_worker("w1")             -> 7         # 自己的 4 个值 + 并进来的 3 个值
t.get_mode()                      -> 7         # 只剩 w3 报的那一个 7
t.top_values(5)                   -> ["7(1)"]
```

## 参考解答

<details>
<summary>展开参考解答</summary>

整道题的关键在第 3 关：要能移除一个 worker，就必须记得**每个 worker 各自**报过什么。只存全局计数是不够
的，到时候不知道每个值该减多少。所以从第 1 关起就存两层：

- `by_worker[worker_id]`：这个 worker 的计数表，值 → 次数。
- `total`：全局计数表，值 → 次数。

要维持的不变量：对每个值，`total[value]` 等于所有已注册 worker 的 `by_worker[w][value]` 之和；计数为 0 的
值不留在 `total` 里。全局计数只在 `_add` 这一个函数里修改，后面每一关都通过它来改。

下面用子类一关一关地加功能，方便分段展示。真实测评里直接在同一个类上加方法就行。

### Level 1

`report` 对列表里的每个值，把 worker 自己的计数加 1，全局计数加 1。`get_worker_count` 直接查
`Counter`：`Counter` 对不存在的键返回 0，而且不会因为查询而多出一个条目。

复杂度：`report` 是 $O(m)$，$m$ 是这次上报的值的个数；其余操作平均 $O(1)$。

```python
from collections import Counter


class TrackerL1:
    def __init__(self):
        self.by_worker = {}                  # worker id -> Counter: value -> how often this worker reported it
        self.total = {}                      # value -> global count; values with count 0 are not kept

    def add_worker(self, worker_id):
        if worker_id in self.by_worker:
            return False
        self.by_worker[worker_id] = Counter()
        return True

    def report(self, worker_id, values):
        mine = self.by_worker.get(worker_id)
        if mine is None:
            return False
        for value in values:
            mine[value] += 1
            self._add(value, 1)
        return True

    def get_count(self, value):
        return self.total.get(value, 0)

    def get_worker_count(self, worker_id, value):
        mine = self.by_worker.get(worker_id)
        return None if mine is None else mine[value]         # a Counter answers 0 for a missing key

    def _add(self, value, delta):
        """The only place that changes a global count."""
        count = self.total.get(value, 0) + delta
        if count:
            self.total[value] = count
        else:
            del self.total[value]                            # back to 0: the value is gone
```

### Level 2

直接在 `total` 上算。众数是“次数最大，其次值最小”，写成对键 `(-次数, 值)` 取最小。`top_values` 用同一个
键排序。因为计数为 0 的值不在 `total` 里，这两个方法不用再过滤。

设当前有 $V$ 个不同的值：`get_mode` 扫一遍是 $O(V)$，`top_values` 排序是 $O(V \log V)$。按题面的规模
（最多 $10^3$ 次调用，最多 $10^4$ 个不同的值），全部调用加起来最坏是 $10^7$ 到 $10^8$ 次基本操作（`top_values` 的排序还要乘上 $\log V$），够用，而且这种写法
最不容易错。数据量再大就不行了，更快的写法见后面的追问。

```python
class TrackerL2(TrackerL1):
    def get_mode(self):
        if not self.total:
            return None
        return min(self.total, key=lambda value: (-self.total[value], value))    # highest count, then smallest value

    def top_values(self, k):
        ranked = sorted(self.total.items(), key=lambda item: (-item[1], item[0]))
        return [f"{value}({count})" for value, count in ranked[:k]]
```

### Level 3

把这个 worker 的计数表整个取出来，对里面每个值，从全局计数里减去它贡献的次数。减到 0 的值由 `_add`
从 `total` 里删掉，所以它不会再出现在 `top_values` 里。返回值是这张表里所有次数之和。

因为 `get_mode` 每次都从 `total` 现算，移除之后众数自动是对的。如果在第 2 关缓存了“当前众数”，这一关就
要小心：移除可能让原来的众数掉下去，而缓存里没有“第二名”的信息。

复杂度：$O(d)$，$d$ 是这个 worker 报过的不同值的个数。

```python
class TrackerL3(TrackerL2):
    def remove_worker(self, worker_id):
        mine = self.by_worker.pop(worker_id, None)
        if mine is None:
            return None
        for value, count in mine.items():
            self._add(value, -count)                         # take back exactly what this worker put in
        return sum(mine.values())
```

### Level 4

把 `source` 的计数表加到 `target` 的计数表上，再把 `source` 从注册表里去掉。全局计数不用动：这些数据
原来算在全局里，现在还算。`Counter.update` 做的是“次数相加”；普通 `dict` 的 `update` 是“覆盖”，用错了
会丢数据。

复杂度：$O(d)$，$d$ 是 `source` 报过的不同值的个数。

```python
class TrackerL4(TrackerL3):
    def merge_workers(self, target_id, source_id):
        if target_id == source_id or target_id not in self.by_worker or source_id not in self.by_worker:
            return False
        self.by_worker[target_id].update(self.by_worker.pop(source_id))      # Counter.update adds the counts
        return True                                                          # global counts do not change


ModeTracker = TrackerL4
```

### 追问

**数据量大，而且每次上报之后都要问一次众数，`get_mode` 不能每次扫全表。怎么办？**

只有上报的话很简单：记住当前众数，每次加 1 之后和它比一下就行。难点在移除：一次移除可以让很多值的计数
同时下降，原来的众数可能不再是第一，这时需要知道“下一个是谁”。下面的验证代码里有一个只记“当前最好”的
版本，移除之后它就答错了。

一个写起来很短的办法是**带懒删除的堆**。堆里放 `(-次数, 值)`。每当某个值的计数变了（不管是增是减），
就把它的新计数压进堆；旧的条目不去找、不去删，留在堆里作废。查询时看堆顶：如果堆顶记录的次数和这个值
现在的真实次数不一样，说明它已经作废，弹掉再看下一个；第一个没作废的堆顶就是答案。

为什么是对的：每个还有数据的值，它最近一次变化时都压过一条“当前次数”的记录，所以每个值都有一条有效
记录在堆里。有效记录里键最小的那条，就是次数最大、值最小的那个。作废的记录只会在到达堆顶时被弹掉，
不影响结果。

还要防一件事：作废的记录如果一直沉在堆的下面，堆会越来越大。所以当堆的大小超过“有效值个数的两倍再加
一点余量”时，直接用 `total` 重建一个只含有效记录的堆。

复杂度（$V$ 是当前不同值的个数）：堆的大小始终是 $O(V)$，所以每次计数变化压一条记录是 $O(\log V)$。
每条记录最多被弹出一次，弹出的代价可以记在压入的账上。重建一次是 $O(V)$，而两次重建之间至少发生了
$V$ 次计数变化，平摊下来每次变化 $O(1)$。所以 `get_mode` 是平摊 $O(\log V)$，`report` 是
$O(m \log V)$，`remove_worker` 是 $O(d \log V)$。`top_values` 仍然是排序。

```python
import heapq


class FastModeTracker(TrackerL4):
    """get_mode in amortised O(log V): a max-heap of (count, value) with lazy deletion."""

    SLACK = 64                               # rebuild once the heap holds more than 2 * live values + SLACK entries

    def __init__(self):
        super().__init__()
        self.heap = []                       # (-count, value); some entries may be out of date

    def _add(self, value, delta):
        super()._add(value, delta)
        if value in self.total:
            heapq.heappush(self.heap, (-self.total[value], value))           # after every change, up or down
        if len(self.heap) > 2 * len(self.total) + self.SLACK:                # mostly stale: rebuild from scratch
            self.heap = [(-count, v) for v, count in self.total.items()]
            heapq.heapify(self.heap)

    def get_mode(self):
        while self.heap and self.total.get(self.heap[0][1], 0) != -self.heap[0][0]:
            heapq.heappop(self.heap)                                         # this count is no longer true
        return self.heap[0][1] if self.heap else None
```

**用“次数 → 值的集合”这样的桶行不行？** 行。再记一个“当前最大次数”：上报让它最多加 1；移除之后如果
最大的桶空了，就一格一格往下找，往下走的格数不会超过之前上报把它抬高的格数，平摊下来是 $O(1)$。麻烦在
并列时要取桶里最小的值：桶得是有序的，或者每个桶再配一个堆。Python 标准库里没有有序集合，所以这里懒删除
堆的写法更短。

**内存能省吗？** 精确统计的话省不了多少：为了能移除，每个 worker 的计数表都得留着，内存是各个 worker
“不同值个数”之和。如果只需要近似的次数，可以让每个 worker 维护一个 Count-Min sketch。它是一张固定
大小的计数表，用同一组哈希函数的两张表可以按格相加，也可以按格相减，所以“移除一个 worker”就是把它那张
表从总表里减掉。不过它只能回答“某个值大约出现了多少次”，候选的高频值是哪些还得另外记。Misra–Gries 这类
只保留若干候选值的摘要可以合并，但标准形式不支持减法，用不到这里。

**真实的分布式环境里，上报可能重发，怎么保证不重复计数？** “加 1”这种增量不是幂等的，同一批数据重发
一次就多算一次。两种常见做法：一是每个 worker 给自己的每一批数据编上递增的序号，汇总方记住每个 worker
已经处理到哪个序号，见到处理过的序号就丢弃；二是 worker 每次上报自己的**累计**计数表而不是增量，汇总方
用新表替换这个 worker 的旧表（按差值更新全局计数），这样重发多少次结果都一样，代价是消息更大。本题里
“每个 worker 单独存一张表”的结构，正好支持第二种做法。另外，如果数据留在各个分片上、只能靠消息传递来求
全局众数，那是另一道题，见 [分布式众数、中位数与排序](#/anthropic/coding/distributed-mode-median)。

**`top_values(k)` 怎么更快？** 一次调用的话，`heapq.nsmallest(k, total.items(), key=...)` 是
$O(V \log k)$，比整体排序省。如果调用很频繁，就需要一个按 `(-次数, 值)` 排好序、支持增删的结构（平衡树，
或第三方的 `sortedcontainers.SortedList`），每次计数变化更新一次，取前 $k$ 个就不用再排序。

<details>
<summary>验证代码（可运行）</summary>

```python
import random

# --- 1. the statement's examples, level by level, on one tracker (both implementations) ---
for cls in (ModeTracker, FastModeTracker):
    t = cls()
    assert t.get_mode() is None and t.top_values(3) == [] and t.get_count(5) == 0
    # Level 1
    assert t.add_worker("w1") is True
    assert t.add_worker("w1") is False
    assert t.add_worker("w2") is True
    assert t.report("w1", [5, 3, 5]) is True
    assert t.report("w2", [3, 3, 9]) is True
    assert t.report("w9", [1]) is False
    assert (t.get_count(3), t.get_count(5), t.get_count(1)) == (3, 2, 0)
    assert t.get_worker_count("w1", 5) == 2
    assert t.get_worker_count("w2", 5) == 0
    assert t.get_worker_count("w9", 5) is None
    # Level 2
    assert t.get_mode() == 3
    assert t.top_values(2) == ["3(3)", "5(2)"]
    assert t.top_values(10) == ["3(3)", "5(2)", "9(1)"]
    assert t.report("w1", [5]) is True
    assert t.get_mode() == 3                                 # 3 and 5 both have count 3: the smaller value wins
    assert t.top_values(1) == ["3(3)"]
    # Level 3
    assert t.remove_worker("w2") == 3
    assert (t.get_count(3), t.get_count(9)) == (1, 0)
    assert t.get_mode() == 5
    assert t.top_values(10) == ["5(3)", "3(1)"]
    assert t.remove_worker("w2") is None
    assert t.report("w2", [9]) is False
    assert t.add_worker("w2") is True
    assert t.get_worker_count("w2", 3) == 0
    assert t.remove_worker("w2") == 0                        # registered but silent: 0, not None
    assert t.add_worker("w2") is True
    # Level 4
    assert t.report("w2", [7, 7, 5]) is True
    assert t.add_worker("w3") is True
    assert t.report("w3", [7]) is True
    assert t.merge_workers("w1", "w2") is True
    assert (t.get_worker_count("w1", 7), t.get_worker_count("w1", 5)) == (2, 4)
    assert t.get_worker_count("w2", 7) is None
    assert t.get_count(7) == 3
    assert t.get_mode() == 5
    assert t.merge_workers("w1", "w2") is False              # w2 is gone
    assert t.merge_workers("w1", "w1") is False
    assert t.merge_workers("w9", "w1") is False
    assert t.remove_worker("w1") == 7                        # 4 reports of its own + 3 merged in from w2
    assert t.get_mode() == 7
    assert t.top_values(5) == ["7(1)"]
    assert t.remove_worker("w3") == 1
    assert t.get_mode() is None and t.top_values(5) == []

# --- 2. details of the stated rules ---
for cls in (ModeTracker, FastModeTracker):
    t = cls()
    t.add_worker("a")
    t.report("a", [-4, -4, 10, 2, 10])
    assert t.get_mode() == -4                                # ties are broken by the number, not by its text
    assert t.top_values(3) == ["-4(2)", "10(2)", "2(1)"]     # as text "10" < "2", as numbers 2 < 10
    assert t.get_worker_count("a", 777) == 0
    assert 777 not in t.by_worker["a"]                       # asking did not create an entry
    t.add_worker("b")
    t.report("b", [2, 2])                                    # 2 now has count 3
    assert t.get_mode() == 2
    assert t.merge_workers("b", "a") is True                 # merging changes no global count
    assert t.top_values(9) == ["2(3)", "-4(2)", "10(2)"]
    assert t.add_worker("a") is True                         # the id is free again and starts empty
    assert t.get_worker_count("a", 2) == 0 and t.get_worker_count("b", 2) == 3


# --- 3. random sequences against a model that keeps the raw reports and recomputes everything ---
class LogTracker:
    def __init__(self):
        self.workers = []                # registered ids
        self.log = []                    # [worker id, value], one item per reported value

    def add_worker(self, w):
        if w in self.workers:
            return False
        self.workers.append(w)
        return True

    def report(self, w, values):
        if w not in self.workers:
            return False
        self.log.extend([w, v] for v in values)
        return True

    def get_count(self, value):
        return len([1 for _, v in self.log if v == value])

    def get_worker_count(self, w, value):
        if w not in self.workers:
            return None
        return len([1 for who, v in self.log if who == w and v == value])

    def ranking(self):
        values = sorted({v for _, v in self.log})
        values.sort(key=self.get_count, reverse=True)        # stable: equal counts keep ascending values
        return values

    def get_mode(self):
        ranking = self.ranking()
        return ranking[0] if ranking else None

    def top_values(self, k):
        return ["%d(%d)" % (v, self.get_count(v)) for v in self.ranking()[:k]]

    def remove_worker(self, w):
        if w not in self.workers:
            return None
        erased = len([1 for who, _ in self.log if who == w])
        self.log = [item for item in self.log if item[0] != w]
        self.workers.remove(w)
        return erased

    def merge_workers(self, target, source):
        if target == source or target not in self.workers or source not in self.workers:
            return False
        for item in self.log:
            if item[0] == source:
                item[0] = target
        self.workers.remove(source)
        return True


class EagerRebuildTracker(FastModeTracker):
    SLACK = 0                            # rebuild as often as possible, to exercise that path


WORKERS = ["w1", "w2", "w3", "w4"]
OPS = (["add_worker"] * 3 + ["report"] * 8 + ["get_count"] + ["get_worker_count"] + ["get_mode"] * 4
       + ["top_values"] * 2 + ["remove_worker"] * 2 + ["merge_workers"])


def random_call(rng):
    op = rng.choice(OPS)
    if op == "report":
        return op, (rng.choice(WORKERS), [rng.randint(-3, 5) for _ in range(rng.randint(1, 6))])
    if op == "get_count":
        return op, (rng.randint(-3, 5),)
    if op == "get_worker_count":
        return op, (rng.choice(WORKERS), rng.randint(-3, 5))
    if op == "get_mode":
        return op, ()
    if op == "top_values":
        return op, (rng.randint(1, 10),)
    if op == "merge_workers":
        return op, (rng.choice(WORKERS), rng.choice(WORKERS))
    return op, (rng.choice(WORKERS),)


rng = random.Random(71)
seen = {}
mode_changed_by_removal = 0
for _ in range(600):
    trackers = [LogTracker(), ModeTracker(), FastModeTracker(), EagerRebuildTracker()]
    for _ in range(rng.randint(1, 100)):
        op, args = random_call(rng)
        mode_before = trackers[0].get_mode()
        results = [getattr(t, op)(*args) for t in trackers]
        assert results[1:] == results[:1] * 3, (op, args, results)
        seen[op, type(results[0]).__name__, bool(results[0])] = True
        if op == "remove_worker" and results[0] and trackers[0].get_mode() != mode_before:
            mode_changed_by_removal += 1
        modes = [t.get_mode() for t in trackers]
        assert modes[1:] == modes[:1] * 3, (op, args, modes)
        for fast in trackers[2:]:
            assert len(fast.heap) <= 2 * len(fast.total) + fast.SLACK + 1    # the heap stays close to the live values
            assert all(count > 0 for count in fast.total.values())
for case in [("add_worker", "bool", True), ("add_worker", "bool", False), ("report", "bool", True),
             ("report", "bool", False), ("get_worker_count", "NoneType", False), ("get_worker_count", "int", True),
             ("get_mode", "NoneType", False), ("remove_worker", "NoneType", False), ("remove_worker", "int", True),
             ("remove_worker", "int", False), ("merge_workers", "bool", True), ("merge_workers", "bool", False)]:
    assert case in seen, case                                # every kind of answer really occurred
assert mode_changed_by_removal >= 100, mode_changed_by_removal


# --- 4. negative control: "just remember the current best" survives reports but not removals ---
class RememberBestTracker(TrackerL4):
    def __init__(self):
        super().__init__()
        self.best = None

    def _add(self, value, delta):
        super()._add(value, delta)
        if value in self.total and (self.best is None or
                                    (-self.total[value], value) < (-self.total.get(self.best, 0), self.best)):
            self.best = value

    def get_mode(self):
        return self.best


for cls, after_reports, after_removal in ((RememberBestTracker, 1, 1), (ModeTracker, 1, 2), (FastModeTracker, 1, 2)):
    t = cls()
    t.add_worker("a"), t.add_worker("b")
    t.report("a", [1, 1, 1]), t.report("b", [2, 2])
    assert t.get_mode() == after_reports
    t.remove_worker("a")                                     # the true mode is now 2
    assert t.get_mode() == after_removal, cls.__name__

# --- 5. the lazy heap: a count that goes down right after a rebuild must still be found ---
t = FastModeTracker()
for name in "abcd":
    t.add_worker(name)
t.report("a", [1, 1, 1]), t.report("b", [1, 1]), t.report("d", [2, 2, 2, 2])
heap_before = t.heap
t.report("c", [3] * 200)
t.remove_worker("c")
assert t.heap is not heap_before                             # a rebuild replaced the list: only current counts are in it
assert t.get_mode() == 1                                     # 1 has count 5, 2 has count 4
t.remove_worker("a")                                         # the count of 1 drops from 5 to 2
assert t.get_mode() == 2
t.remove_worker("d")                                         # 2 disappears
assert t.get_mode() == 1 and t.top_values(3) == ["1(2)"]     # needs the entry pushed when the count went down

# --- 6. the lazy heap under a long run: stale entries on top, counts that return to an old value, rebuilds ---
t, slow = FastModeTracker(), ModeTracker()
for tracker in (t, slow):
    tracker.add_worker("keep")
    tracker.report("keep", [0])
rebuilds = 0
for round_no in range(300):
    name = f"tmp{round_no % 7}"
    batch = [rng.randint(1, 40) for _ in range(rng.randint(1, 30))]
    heap_before = t.heap
    for tracker in (t, slow):
        tracker.add_worker(name)
        tracker.report(name, batch)
    assert t.get_mode() == slow.get_mode()
    if round_no % 3 == 2:
        for tracker in (t, slow):
            for k in range(7):
                tracker.remove_worker(f"tmp{k}")
        assert t.get_mode() == slow.get_mode() == 0          # only the single report of "keep" is left
        assert len(t.heap) <= 2 * len(t.total) + 65
    rebuilds += t.heap is not heap_before                    # a rebuild replaces the list object
assert rebuilds >= 50, rebuilds
print("all checks passed")
```

</details>

</details>
