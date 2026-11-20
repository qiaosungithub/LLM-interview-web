# 从零实现堆和带索引的优先队列

## 题目

你刚用 `heapq` 写完一个最短路，面试官接着说：现在不许用 `heapq`，自己把堆写出来；写完之后，
再让它支持“按键修改优先级”。

整道题的约定：

- 不能用 `heapq`，也不能用 `sorted`、`list.sort`、`min` 来代替堆的操作。
- 元素（或优先级）之间只用 `<` 比较。
- 有多个最小元素时，先弹出哪一个都可以。

### Part 1 —— 二叉堆

```py
class MinHeap:
    def __init__(self, items=()) -> None:
        """Builds a heap holding all of items, in O(n)."""

    def push(self, item) -> None: ...

    def pop(self):
        """Removes and returns a smallest item. Raises IndexError if the heap is empty."""

    def peek(self):
        """Returns a smallest item without removing it. Raises IndexError if the heap is empty."""

    def __len__(self) -> int: ...
```

要求 `push` 和 `pop` 是 $O(\log n)$，`peek` 是 $O(1)$，构造函数是 $O(n)$。

```text
h = MinHeap([5, 3, 8, 1])
h.peek()                       -> 1
len(h)                         -> 4
h.push(0); h.push(4)
[h.pop() for _ in range(6)]    -> [0, 1, 3, 4, 5, 8]
h.pop()                        -> IndexError
```

同时回答：`push`、`pop` 最多各做多少次比较？为什么一次建堆是 $O(n)$，而逐个 `push` 是 $O(n \log n)$？

### Part 2 —— 带索引的优先队列

现在每个元素是一个键（key）加一个优先级（priority）。键可哈希、互不相同；键之间不一定能比较大小。
除了取出优先级最小的元素，还要能按键修改优先级、按键删除。记录“每个键在堆里的位置”可以用 `dict`。

```py
class IndexedPQ:
    def __init__(self) -> None: ...

    def push(self, key, priority) -> None:
        """Adds key. Raises ValueError if key is already in the queue."""

    def pop(self):
        """Removes and returns (key, priority) with the smallest priority. IndexError if empty."""

    def peek(self):
        """(key, priority) with the smallest priority, without removing it. IndexError if empty."""

    def update(self, key, priority) -> None:
        """Changes the priority of key; it may go up or down. KeyError if key is not in the queue."""

    def remove(self, key):
        """Removes key and returns its priority. KeyError if key is not in the queue."""

    def priority(self, key):
        """The current priority of key. KeyError if key is not in the queue."""

    def __contains__(self, key) -> bool: ...

    def __len__(self) -> int: ...
```

要求 `push`、`pop`、`update`、`remove` 都只做 $O(\log n)$ 次优先级比较和 $O(\log n)$ 次 `dict` 操作；
其余方法是常数次 `dict` 或列表操作。

```text
q = IndexedPQ()
for key, p in [("a", 1), ("b", 12), ("c", 4), ("d", 18), ("e", 14), ("f", 13), ("g", 7)]:
    q.push(key, p)
q.remove("d")        -> 18
一直 pop 到空         -> ("a", 1), ("c", 4), ("g", 7), ("b", 12), ("f", 13), ("e", 14)

q = IndexedPQ()
q.push("x", 5); q.push("y", 3); q.push("z", 9)
q.update("z", 1)                         # 调小
q.update("y", 7)                         # 调大
q.peek()             -> ("z", 1)
q.priority("y")      -> 7
"x" in q             -> True
"w" in q             -> False
q.push("x", 0)       -> ValueError
q.update("w", 1)     -> KeyError
q.remove("w")        -> KeyError
一直 pop 到空         -> ("z", 1), ("x", 5), ("y", 7)
q.pop()              -> IndexError
```

同时回答：`remove` 把别的元素挪到被删的位置之后，这个元素应该往上调整还是往下调整？

### Part 3 —— 用在 Dijkstra 上

有向图有 $V$ 个节点，编号 `0..V-1`，$V \ge 1$。`adj[u]` 是从 `u` 出发的边的列表，每条边是 `(v, w)`，
`w` 是非负整数。可以有重边和自环。求从 `source` 到每个节点的最短距离；到不了的节点距离是 `math.inf`。

用两种方式各写一遍：

```py
def dijkstra_indexed(adj: list[list[tuple[int, int]]], source: int) -> list:
    """Shortest distances from source. Uses IndexedPQ, with update() as decrease-key,
    so the queue never holds more than one entry per node."""


def dijkstra_lazy(adj: list[list[tuple[int, int]]], source: int) -> list:
    """Same result with only the MinHeap of Part 1: push a new entry whenever a distance
    improves, and skip entries that are out of date when they are popped."""
```

```text
adj = [[(1, 7), (2, 2)], [(3, 1)], [(1, 3), (3, 8)], [], [(0, 1)]]
dijkstra_indexed(adj, 0)  -> [0, 5, 2, 6, inf]
dijkstra_lazy(adj, 0)     -> [0, 5, 2, 6, inf]
# 0 -> 2 -> 1 的长度 2 + 3 = 5 比直接走 0 -> 1 的 7 短；节点 4 到不了
```

然后比较这两种写法：堆里最多同时有多少个元素？时间复杂度各是多少？各在什么情况下更合适？

## 参考解答

<details>
<summary>展开参考解答</summary>

### Part 1

二叉堆是一棵存在数组里的完全二叉树：下标 `i` 的两个孩子是 `2i + 1` 和 `2i + 2`，父节点是
`(i - 1) // 2`。不变量只有一条：**任何元素都不小于它的父节点**。所以最小元素在下标 0。

两个基本动作：

- 上浮（sift up）：元素比父节点小，就和父节点交换，一直到不小于父节点或到达根。
- 下沉（sift down）：在两个孩子里挑较小的那个，如果它比当前元素小就交换，一直到两个孩子都不更小
  或没有孩子。必须挑**较小的**孩子，否则换上来的孩子会比它的兄弟大，不变量又坏了。

`push` 是把新元素放到末尾再上浮。`pop` 是取走根，把末尾的元素放到根上再下沉；末尾元素拿走不会在
树中间留洞。

```python
class MinHeap:
    """Binary min-heap in a list: the children of index i are 2i + 1 and 2i + 2."""

    def __init__(self, items=()) -> None:
        self._a = list(items)
        for i in range(len(self._a) // 2 - 1, -1, -1):   # last internal node first, root last
            self._sift_down(i)

    def __len__(self) -> int:
        return len(self._a)

    def peek(self):
        if not self._a:
            raise IndexError("peek from an empty heap")
        return self._a[0]

    def push(self, item) -> None:
        self._a.append(item)
        self._sift_up(len(self._a) - 1)

    def pop(self):
        a = self._a
        if not a:
            raise IndexError("pop from an empty heap")
        last = a.pop()
        if not a:
            return last                         # it was the only item
        top, a[0] = a[0], last
        self._sift_down(0)
        return top

    def _sift_up(self, i: int) -> None:
        a = self._a
        while i > 0:
            parent = (i - 1) // 2
            if not a[i] < a[parent]:
                break
            a[i], a[parent] = a[parent], a[i]
            i = parent

    def _sift_down(self, i: int) -> None:
        a = self._a
        n = len(a)
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            if child + 1 < n and a[child + 1] < a[child]:
                child += 1                      # NOTE: the smaller of the two children
            if not a[child] < a[i]:
                break
            a[i], a[child] = a[child], a[i]
            i = child
```

**比较次数。** 有 $n$ 个元素的堆，最深的节点深度是 $\lfloor \log_2 n \rfloor$。

- `push`：上浮每层比较一次，最多 $\lfloor \log_2 n \rfloor$ 次（$n$ 是插入后的大小）。
- `pop`：下沉每层比较两次（先比两个孩子，再比较小的孩子和当前元素），最多
  $2 \lfloor \log_2 n \rfloor$ 次。

**建堆为什么是 $O(n)$。** 构造函数从最后一个有孩子的节点开始，倒着对每个节点做一次下沉。处理到某个
节点时，它的两棵子树已经是堆，下沉之后以它为根的子树也是堆。一个节点下沉的代价和它的**高度**
（到最底层的距离）成正比，而不是和深度成正比。树里绝大多数节点都靠近底部：高度为 $h$ 的节点大约
有 $n / 2^{h+1}$ 个，每个最多比较 $2h$ 次，总数大约是

$$\sum_{h \ge 1} 2h \cdot \frac{n}{2^{h+1}} = n \sum_{h \ge 1} \frac{h}{2^h} = 2n .$$

所以建堆是 $O(n)$（下面验证代码里试过的输入，比较次数都不超过 $2n$）。逐个 `push` 则相反：代价和
**深度**成正比，而一半的节点在最底层，深度约为 $\log_2 n$。输入是降序时每个新元素都要一路浮到根，
第 $i$ 个元素（从 1 数起）正好比较 $\lfloor \log_2 i \rfloor$ 次，总共约 $n \log_2 n$ 次。

### Part 2

堆只能快速找到最小的元素，找不到“键 `k` 现在在数组的哪个下标”。所以再维护一张位置表 `_pos`：
键 → 它在堆数组里的下标。有了下标，修改和删除都只是一次局部调整。

实现的关键是**让位置表永远和数组一致**。做法是把“移动元素”收口到一个函数 `_swap` 里：堆里的元素
只通过 `_swap` 换位置，`_swap` 同时更新两个键的位置。上浮和下沉都调用它，就不会漏。

- `update`：改掉优先级之后，新优先级可能比父节点小（要上浮），也可能比孩子大（要下沉）。
  先上浮再下沉，两个里面至多有一个真的会移动。
- `remove`：把要删的元素和**末尾**的元素交换，弹出末尾，再调整换过来的那个元素。
- `pop`：就是 `remove` 根上的那个键。

**`remove` 之后往哪个方向调整：两个方向都可能。** `pop` 时换到根上的元素只可能下沉，容易让人以为
`remove` 也一样。但末尾的元素和被删的位置不一定在同一条从根到叶的路径上，它可能比被删位置的父节点
还小。题目的第一个例子就是这种情况：依次插入之后数组是 `[1, 12, 4, 18, 14, 13, 7]`，删除 18
（下标 3）时，末尾的 7 被换到下标 3，而它的父节点是 12。只下沉的话，7 留在 12 下面，之后会先弹出 12
再弹出 7。

堆里存 `(priority, key)`，比较时**只比优先级** `entry[0]`，不直接比较整个元组：元组比较在优先级
相等时会接着去比键，而键不一定能比较。

```python
class IndexedPQ:
    """Min-priority queue over distinct hashable keys, with update and remove by key."""

    def __init__(self) -> None:
        self._heap = []          # (priority, key) entries, heap-ordered by priority alone
        self._pos = {}           # key -> index of its entry in self._heap

    def __len__(self) -> int:
        return len(self._heap)

    def __contains__(self, key) -> bool:
        return key in self._pos

    def priority(self, key):
        return self._heap[self._pos[key]][0]    # KeyError if key is absent

    def peek(self):
        if not self._heap:
            raise IndexError("peek from an empty queue")
        priority, key = self._heap[0]
        return key, priority

    def push(self, key, priority) -> None:
        if key in self._pos:
            raise ValueError(f"key already queued: {key!r}")
        self._heap.append((priority, key))
        self._pos[key] = len(self._heap) - 1
        self._sift_up(len(self._heap) - 1)

    def update(self, key, priority) -> None:
        i = self._pos[key]                      # KeyError if key is absent
        self._heap[i] = (priority, key)
        self._sift_down(self._sift_up(i))       # NOTE: the new priority may send it either way

    def remove(self, key):
        i = self._pos[key]                      # KeyError if key is absent
        last = len(self._heap) - 1
        self._swap(i, last)                     # harmless when i == last
        priority, _ = self._heap.pop()
        del self._pos[key]
        if i < last:                            # another entry now sits in slot i
            self._sift_down(self._sift_up(i))   # NOTE: it came from elsewhere in the tree: either way
        return priority

    def pop(self):
        if not self._heap:
            raise IndexError("pop from an empty queue")
        key = self._heap[0][1]
        return key, self.remove(key)

    def _swap(self, i: int, j: int) -> None:
        heap, pos = self._heap, self._pos
        heap[i], heap[j] = heap[j], heap[i]
        pos[heap[i][1]] = i                     # NOTE: entries move only here, so this is the only
        pos[heap[j][1]] = j                     # place where the position map has to be kept in step

    def _sift_up(self, i: int) -> int:
        """Moves the entry at i up as far as it must go; returns its final index."""
        heap = self._heap
        while i > 0:
            parent = (i - 1) // 2
            if not heap[i][0] < heap[parent][0]:
                break
            self._swap(i, parent)
            i = parent
        return i

    def _sift_down(self, i: int) -> int:
        """Moves the entry at i down as far as it must go; returns its final index."""
        heap = self._heap
        n = len(heap)
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            if child + 1 < n and heap[child + 1][0] < heap[child][0]:
                child += 1
            if not heap[child][0] < heap[i][0]:
                break
            self._swap(i, child)
            i = child
        return i
```

**复杂度。** `push` 最多 $\lfloor \log_2 n \rfloor$ 次优先级比较（$n$ 是插入后的大小）；`pop`、`update`、
`remove` 最多 $2 \lfloor \log_2 n \rfloor + 2$ 次（$n$ 是操作前的大小）。每次交换另有两次 `dict` 写入，
每个公开方法开头有一次 `dict` 查找。`dict` 操作是期望 $O(1)$，所以准确的说法是：比较次数最坏
$O(\log n)$，总时间期望 $O(\log n)$。如果键本来就是 `0..n-1` 的整数，位置表可以换成列表，
这时总时间就是最坏 $O(\log n)$。

### Part 3

Dijkstra 每一步取出当前距离最小、还没确定的节点 `u`，它的距离此时就是最终值（边权非负保证了这一点），
然后用 `u` 的出边去松弛（relax）邻居：如果 `dist[u] + w < dist[v]`，就把 `dist[v]` 改小。问题在于
`v` 可能已经在队列里了，带着旧的、更大的距离。两种处理方式：

**带索引的写法**：`v` 已在队列里就 `update` 成新距离（这个操作叫 decrease-key），不在就 `push`。
队列里每个节点至多一条记录。

```python
import math


def dijkstra_indexed(adj: list[list[tuple[int, int]]], source: int) -> list:
    dist = [math.inf] * len(adj)
    dist[source] = 0
    pq = IndexedPQ()
    pq.push(source, 0)
    while pq:
        u, d = pq.pop()                         # d == dist[u], and it is final
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                if v in pq:
                    pq.update(v, nd)            # decrease-key
                else:
                    pq.push(v, nd)
    return dist
```

**懒删除的写法**：不改旧记录，直接再 `push` 一条 `(新距离, v)`。旧记录留在堆里，等它被弹出时，
发现它的距离比 `dist[v]` 大，说明已经过期，跳过。

```python
def dijkstra_lazy(adj: list[list[tuple[int, int]]], source: int) -> list:
    dist = [math.inf] * len(adj)
    dist[source] = 0
    heap = MinHeap([(0, source)])
    while heap:
        d, u = heap.pop()
        if d > dist[u]:
            continue                            # NOTE: stale entry -- u was already settled with less
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heap.push((nd, v))              # the old entry for v stays in the heap
    return dist
```

**比较。** 记节点数 $V$、边数 $E$。

| | 带索引（decrease-key） | 懒删除 |
| --- | --- | --- |
| 堆里最多的元素数 | $V$ | 不超过 $E$（每次松弛成功都多一条；起点那一条在第一次松弛前已经弹出） |
| 堆操作次数 | 至多 $V$ 次 `push`、$V$ 次 `pop`、$E$ 次 `update` | 至多 $E + 1$ 次 `push`、$E + 1$ 次 `pop` |
| 时间 | $O((V + E) \log V)$ | $O(E \log E)$ |
| 额外代码 | 位置表，每次交换都要维护 | 一行过期判断 |

要点有三个：

- **渐近时间其实一样。** 没有重边时 $E \le V^2$，所以 $\log E \le 2 \log V$，$O(E \log E)$ 就是
  $O(E \log V)$。用二叉堆做 decrease-key 本身也要 $O(\log V)$，并没有比“再 `push` 一条”便宜。教科书里
  $O(E + V \log V)$ 的界需要 decrease-key 均摊 $O(1)$ 的堆（比如斐波那契堆），二叉堆做不到。
- **真正的差别在空间和过期元素。** 带索引的写法堆里至多 $V$ 条；懒删除一共要入堆 $E + 1$ 次，同一时刻最多 $E$ 条。最坏情况是
  这样一种图：对每一对 $u < v$ 都有一条边 `u -> v`，权重 $(v - u)^2$。最短路是一步一步走，
  `dist[v] = v`；而按 `u` 从小到大处理时，每条边都会把 `dist[v]` 再改小一次，所以懒删除要入堆
  $E + 1$ 次。不过在权重随机的图上，成功的松弛远少于 $E$，过期元素并不多。
- **怎么选。** 只是跑一次最短路、内存够用，懒删除更简单，也可以直接用现成的堆。图很稠密又在意内存，
  或者同一个队列还需要任意的 `update` / `remove`（比如带取消功能的任务调度、定时器），就用带索引的版本。

两种写法都依赖边权非负：只有这样，弹出的节点距离才是最终值，每个节点才只会被处理一次。

### 追问

**问：优先级相同的元素，要求按进入队列的先后弹出，怎么改？** 二叉堆本身不稳定。给每条记录加一个
递增的序号，比较时比 `(priority, seq)`：优先级相同就比序号，序号互不相同，所以永远不会比到键。
`update` 时是沿用旧序号还是发新序号，取决于“修改优先级算不算重新排队”，要和面试官确认。

**问：位置表一定要用 `dict` 吗？** 不一定。键是 `0..n-1` 的整数时（比如 Dijkstra 里的节点编号），
用一个长度为 $n$ 的列表，`-1` 表示不在队列里。这样没有哈希，所有操作都是最坏情况的界，常数也更小。

**问：换成 d 叉堆有什么好处？** 每个节点 $d$ 个孩子，树高是 $\log_d n$。上浮每层比较一次，变快了；
下沉每层要在 $d$ 个孩子里找最小的，变慢了。Dijkstra 里 `update`（上浮）最多 $E$ 次，`pop`（下沉）
只有 $V$ 次，稠密图上前者多得多，所以取 $d \approx E / V$ 可以把总时间降到
$O(E \log_{E/V} V)$。

**问：边权有负数时，这两段代码会怎样？** “弹出的节点距离已是最终值”不再成立。上面两段代码都允许
一个节点在弹出之后因为距离又变小而重新入队，所以只要没有负环，最后算出的距离仍然是对的（下面的
验证代码在带负权的无环图上核对了这一点）；但同一个节点可能被反复处理，前面的复杂度分析全部作废，
已知最坏情况下操作次数是指数级的。有负环且从起点可达时，循环不会结束。有负权就该用 Bellman–Ford。

<details>
<summary>验证代码（可运行）</summary>

```python
import random

rng = random.Random(20261117)


def raises(exc, fn, *args) -> bool:
    try:
        fn(*args)
    except exc:
        return True
    return False


# ---- Part 1: the example ----
h = MinHeap([5, 3, 8, 1])
assert h.peek() == 1 and len(h) == 4
h.push(0); h.push(4)
assert [h.pop() for _ in range(6)] == [0, 1, 3, 4, 5, 8]
assert raises(IndexError, h.pop) and raises(IndexError, h.peek) and len(h) == 0
assert len(MinHeap()) == 0 and MinHeap([7]).pop() == 7


def is_heap(a) -> bool:
    return all(not a[i] < a[(i - 1) // 2] for i in range(1, len(a)))


# ---- Part 1: random operations against a sorted list, the heap invariant after every operation ----
for trial in range(1500):
    items = [rng.randint(-5, 5) for _ in range(rng.randint(0, 30))]
    h, model = MinHeap(items), sorted(items)
    assert is_heap(h._a)
    for _ in range(40):
        if rng.random() < 0.5:
            x = rng.randint(-5, 5)
            h.push(x)
            model.append(x)
            model.sort()
        elif model:
            assert h.peek() == model[0]
            assert h.pop() == model.pop(0)
        else:
            assert raises(IndexError, h.pop)
        assert is_heap(h._a) and len(h) == len(model)


# ---- Part 1: comparison counts ----
class Counted:
    """Wraps a value and counts every < comparison made on it."""
    comparisons = 0
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        Counted.comparisons += 1
        return self.value < other.value


def counted(values):
    Counted.comparisons = 0
    return [Counted(v) for v in values]


def log2_floor(n: int) -> int:
    return n.bit_length() - 1


for n in list(range(0, 130)) + [1000, 4096, 5000, 20000]:
    shapes = (list(range(n, 0, -1)), list(range(n)), [rng.random() for _ in range(n)], [0] * n)
    for values in shapes:
        h = MinHeap(counted(values))
        assert Counted.comparisons <= 2 * n          # building is linear
        assert is_heap(h._a)
    # n pushes of a descending sequence: every item climbs to the root
    h = MinHeap()
    for item in counted(range(n, 0, -1)):
        h.push(item)
    assert Counted.comparisons == sum(log2_floor(i) for i in range(1, n + 1))
    # ... and of an ascending sequence: one failed comparison each
    h = MinHeap()
    for item in counted(range(n)):
        h.push(item)
    assert Counted.comparisons == max(n - 1, 0)

n = 20000
descending_pushes = sum(log2_floor(i) for i in range(1, n + 1))
assert descending_pushes > 0.9 * n * (math.log2(n) - 2)      # about n log2 n, versus at most 2n to build
h = MinHeap(counted(rng.random() for _ in range(n)))
build = Counted.comparisons
for size in range(n, 0, -1):                                 # every single operation is within its bound
    Counted.comparisons = 0
    h.pop()
    assert Counted.comparisons <= 2 * log2_floor(size)
for size in range(1, 2000):
    Counted.comparisons = 0
    h.push(Counted(rng.random()))
    assert Counted.comparisons <= log2_floor(size)
print(f"n = {n}: build {build} comparisons (2n = {2 * n}), descending pushes {descending_pushes}")


# ---- Part 2: the examples ----
def drain(q) -> list:
    out = []
    while q:
        out.append(q.pop())
    return out


EXAMPLE = [("a", 1), ("b", 12), ("c", 4), ("d", 18), ("e", 14), ("f", 13), ("g", 7)]
q = IndexedPQ()
for key, p in EXAMPLE:
    q.push(key, p)
assert [p for p, _ in q._heap] == [1, 12, 4, 18, 14, 13, 7]   # the array quoted in the explanation
assert q.remove("d") == 18
assert drain(q) == [("a", 1), ("c", 4), ("g", 7), ("b", 12), ("f", 13), ("e", 14)]

q = IndexedPQ()
q.push("x", 5); q.push("y", 3); q.push("z", 9)
q.update("z", 1)
q.update("y", 7)
assert q.peek() == ("z", 1) and q.priority("y") == 7 and len(q) == 3
assert "x" in q and "w" not in q
assert raises(ValueError, q.push, "x", 0)
assert raises(KeyError, q.update, "w", 1) and raises(KeyError, q.remove, "w")
assert raises(KeyError, q.priority, "w")
assert q.priority("x") == 5 and len(q) == 3                  # the failed calls changed nothing
assert drain(q) == [("z", 1), ("x", 5), ("y", 7)]
assert raises(IndexError, q.pop) and raises(IndexError, q.peek)


# removal that only sifts down: the example comes out in the wrong order
class SiftDownOnlyPQ(IndexedPQ):
    def remove(self, key):
        i = self._pos[key]
        last = len(self._heap) - 1
        self._swap(i, last)
        priority, _ = self._heap.pop()
        del self._pos[key]
        if i < last:
            self._sift_down(i)
        return priority


q = SiftDownOnlyPQ()
for key, p in EXAMPLE:
    q.push(key, p)
q.remove("d")
assert [p for p, _ in q._heap] == [1, 12, 4, 7, 14, 13]      # 7 is left below 12
assert [k for k, _ in drain(q)] == ["a", "c", "b", "g", "f", "e"]   # 12 comes out before 7

# keys are never compared, even when priorities tie
keys = [object(), "s", 3, (1, 2), None, 2.5, frozenset()]
q = IndexedPQ()
for key in keys:
    q.push(key, 1)
q.update("s", 1)
out = drain(q)
assert len(out) == len(keys) and all(p == 1 for _, p in out)
assert all(any(k is key for k, _ in out) for key in keys)


# ---- Part 2: random operations against a brute-force model, all invariants after every operation ----
def consistent(pq, model) -> bool:
    """The array, the position map and the model agree, and the array is in heap order."""
    heap, pos = pq._heap, pq._pos
    if not len(heap) == len(pos) == len(model) == len(pq):
        return False
    for i, (priority, key) in enumerate(heap):
        if pos.get(key) != i:                        # the position map points at the entry
            return False
        if key not in model or model[key] != priority:
            return False
        if i and priority < heap[(i - 1) // 2][0]:   # heap order
            return False
    return True


def random_ops(cls, steps, key_range, priority_range, inspect=True):
    """Runs random operations against a dict model. Returns None if nothing went wrong, "answer" at
    the first wrong return value, or (only with inspect) "invariant" at the first broken invariant."""
    pq, model = cls(), {}
    for _ in range(steps):
        op, key = rng.random(), rng.randrange(key_range)
        if op < 0.35:
            p = rng.randrange(priority_range)
            if key in model:
                assert raises(ValueError, pq.push, key, p)
            else:
                pq.push(key, p)
                model[key] = p
        elif op < 0.60:
            p = rng.randrange(priority_range)
            if key in model:
                pq.update(key, p)
                model[key] = p
            else:
                assert raises(KeyError, pq.update, key, p)
        elif op < 0.75:
            if key in model:
                if pq.remove(key) != model.pop(key):
                    return "answer"
            else:
                assert raises(KeyError, pq.remove, key)
        elif model:
            smallest = min(model.values())           # brute force: scan everything
            k, p = pq.peek()
            if p != smallest or model.get(k) != p or pq.pop() != (k, p):
                return "answer"
            del model[k]
        else:
            assert raises(IndexError, pq.pop) and raises(IndexError, pq.peek)
        if (key in pq) != (key in model) or len(pq) != len(model):
            return "answer"
        if key in model and pq.priority(key) != model[key]:
            return "answer"
        if inspect and not consistent(pq, model):
            return "invariant"
    return None


for trial in range(1500):
    key_range, priority_range = rng.choice([3, 13, 60]), rng.choice([2, 10, 1000])
    assert random_ops(IndexedPQ, 80, key_range, priority_range) is None

def remove_heavy(cls, size, inspect=True):
    """Fills the queue with size keys, then empties it by remove (half the time) or pop.
    Same return convention as random_ops."""
    pq, model = cls(), {}
    for key in range(size):
        p = rng.randrange(1000)
        pq.push(key, p)
        model[key] = p
    while model:
        if rng.random() < 0.5:
            key = rng.choice(list(model))
            if pq.remove(key) != model.pop(key):
                return "answer"
        else:
            k, p = pq.pop()
            if p != min(model.values()) or model.pop(k, None) != p:
                return "answer"
        if inspect and not consistent(pq, model):
            return "invariant"
    return None


assert all(remove_heavy(IndexedPQ, 200) is None for _ in range(30))
# The sift-down-only removal: asserting the invariant after every operation finds it in almost every
# run; looking only at what the operations return finds it far less often, because a misplaced entry
# is often removed or repaired before it would come out of pop in the wrong order.
RUNS = 100
by_invariant = sum(remove_heavy(SiftDownOnlyPQ, 200) is not None for _ in range(RUNS))
by_answers = sum(remove_heavy(SiftDownOnlyPQ, 200, inspect=False) is not None for _ in range(RUNS))
assert by_invariant >= 90 and by_answers < by_invariant, (by_invariant, by_answers)
mixed = sum(random_ops(SiftDownOnlyPQ, 200, 40, 1000) is not None for _ in range(RUNS))
print(f"sift-down-only remove, {RUNS} runs each: remove-heavy caught by invariants {by_invariant}, "
      f"by return values alone {by_answers}; the mixed workload above caught it in {mixed}")

class CountingDict(dict):
    """A dict that counts lookups, membership tests, writes and deletions made through [] and in."""
    ops = 0

    def __getitem__(self, key):
        CountingDict.ops += 1
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        CountingDict.ops += 1
        super().__setitem__(key, value)

    def __delitem__(self, key):
        CountingDict.ops += 1
        super().__delitem__(key)

    def __contains__(self, key):
        CountingDict.ops += 1
        return super().__contains__(key)


# every operation stays within its comparison bound, and its dict operations are O(log n) too:
# two writes per swap, at most one swap per comparison plus the swap with the last entry in remove
for trial in range(100):
    pq, keys = IndexedPQ(), []
    pq._pos = CountingDict()
    for step in range(400):
        n = len(pq)
        op = rng.random()
        Counted.comparisons = CountingDict.ops = 0
        if op < 0.4 or n == 0:
            pq.push(step, Counted(rng.random()))
            keys.append(step)
            assert Counted.comparisons <= log2_floor(n + 1)
        elif op < 0.7:
            pq.update(rng.choice(keys), Counted(rng.choice([rng.random(), -1.0, 2.0])))
        elif op < 0.85:
            pq.remove(keys.pop(rng.randrange(len(keys))))
        else:
            keys.remove(pq.pop()[0])
        assert Counted.comparisons <= 2 * log2_floor(max(n, 1)) + 2
        assert CountingDict.ops <= 2 * (Counted.comparisons + 1) + 3


# ---- Part 3 ----
EXAMPLE_ADJ = [[(1, 7), (2, 2)], [(3, 1)], [(1, 3), (3, 8)], [], [(0, 1)]]
assert dijkstra_indexed(EXAMPLE_ADJ, 0) == [0, 5, 2, 6, math.inf]
assert dijkstra_lazy(EXAMPLE_ADJ, 0) == [0, 5, 2, 6, math.inf]
assert dijkstra_indexed([[]], 0) == dijkstra_lazy([[]], 0) == [0]
assert dijkstra_indexed([[(0, 3)], []], 0) == dijkstra_lazy([[(0, 3)], []], 0) == [0, math.inf]


def dijkstra_quadratic(adj, source):
    """The O(V^2) textbook version: no heap at all, a linear scan for the closest open node."""
    n = len(adj)
    dist = [math.inf] * n
    dist[source] = 0
    done = [False] * n
    for _ in range(n):
        u = -1
        for x in range(n):
            if not done[x] and dist[x] < math.inf and (u < 0 or dist[x] < dist[u]):
                u = x
        if u < 0:
            break
        done[u] = True
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist


def bellman_ford(adj, source):
    """A different algorithm altogether; also valid with negative weights (no negative cycles)."""
    n = len(adj)
    dist = [math.inf] * n
    dist[source] = 0
    for _ in range(n):
        changed = False
        for u in range(n):
            if dist[u] < math.inf:
                for v, w in adj[u]:
                    if dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        changed = True
        if not changed:
            break
    return dist


def random_graph(n, m, low, high, forward_only=False):
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u, v = rng.randrange(n), rng.randrange(n)    # self-loops and parallel edges included
        if forward_only and u >= v:
            continue
        adj[u].append((v, rng.randint(low, high)))
    return adj


assert dijkstra_quadratic(EXAMPLE_ADJ, 0) == bellman_ford(EXAMPLE_ADJ, 0) == [0, 5, 2, 6, math.inf]
for trial in range(1500):
    n = rng.randint(1, 12)
    adj = random_graph(n, rng.randint(0, 4 * n), 0, rng.choice([0, 1, 9]))   # zero weights included
    source = rng.randrange(n)
    want = dijkstra_quadratic(adj, source)
    assert bellman_ford(adj, source) == want
    assert dijkstra_indexed(adj, source) == want and dijkstra_lazy(adj, source) == want
for trial in range(20):
    adj = random_graph(300, 1500, 0, 50)
    want = dijkstra_quadratic(adj, 0)
    assert dijkstra_indexed(adj, 0) == want and dijkstra_lazy(adj, 0) == want

# negative weights without negative cycles (edges only go from lower to higher numbers):
# both versions still end with the right distances, as the follow-up says
for trial in range(1000):
    n = rng.randint(1, 10)
    adj = random_graph(n, 4 * n, -5, 10, forward_only=True)
    want = bellman_ford(adj, 0)
    assert dijkstra_indexed(adj, 0) == want and dijkstra_lazy(adj, 0) == want


# ---- Part 3: how large the queue gets (operation counts, not wall-clock time) ----
class TrackedHeap(MinHeap):
    entries = peak = 0

    def __init__(self, items=()):
        super().__init__(items)
        TrackedHeap.entries += len(self)
        TrackedHeap.peak = max(TrackedHeap.peak, len(self))

    def push(self, item):
        super().push(item)
        TrackedHeap.entries += 1
        TrackedHeap.peak = max(TrackedHeap.peak, len(self))


class TrackedPQ(IndexedPQ):
    pushes = updates = peak = 0

    def push(self, key, priority):
        super().push(key, priority)
        TrackedPQ.pushes += 1
        TrackedPQ.peak = max(TrackedPQ.peak, len(self))

    def update(self, key, priority):
        super().update(key, priority)
        TrackedPQ.updates += 1


def run_tracked(adj, source=0):
    """Runs both versions with instrumented queues (the functions look the class names up when called)."""
    global MinHeap, IndexedPQ
    plain = MinHeap, IndexedPQ
    TrackedHeap.entries = TrackedHeap.peak = 0
    TrackedPQ.pushes = TrackedPQ.updates = TrackedPQ.peak = 0
    MinHeap, IndexedPQ = TrackedHeap, TrackedPQ
    try:
        a, b = dijkstra_indexed(adj, source), dijkstra_lazy(adj, source)
    finally:
        MinHeap, IndexedPQ = plain
    assert a == b
    return a


# the worst case for lazy deletion: an edge u -> v of weight (v - u)^2 for every u < v
V = 150
adj = [[(v, (v - u) ** 2) for v in range(u + 1, V)] for u in range(V)]
E = V * (V - 1) // 2
assert run_tracked(adj) == list(range(V)) == dijkstra_quadratic(adj, 0)
assert TrackedHeap.entries == E + 1                  # every single edge pushed a new entry
assert TrackedHeap.peak > E // 2                     # and most of them were in the heap at once
assert TrackedPQ.pushes == V and TrackedPQ.updates == E - (V - 1)
assert TrackedPQ.peak <= V
print(f"dense graph V={V}, E={E}: lazy heap peak {TrackedHeap.peak}, indexed queue peak {TrackedPQ.peak}")

# a sparse graph with random weights: few stale entries
V, E = 5000, 25000
adj = random_graph(V, E, 1, 100)
dist = run_tracked(adj)
reached = sum(d < math.inf for d in dist)
assert TrackedPQ.pushes == reached and TrackedPQ.peak <= V
assert TrackedPQ.updates <= E and TrackedHeap.entries <= E + 1
assert TrackedHeap.entries < 2 * reached             # far fewer than E + 1 entries
print(f"sparse graph V={V}, E={E}: reached {reached}, lazy pushed {TrackedHeap.entries} entries "
      f"(peak {TrackedHeap.peak}), indexed queue peak {TrackedPQ.peak}")
print("all checks passed")
```

</details>

</details>
