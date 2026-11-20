# 有根树上的祖先、最近公共祖先与路径最大值

## 题目

一棵*有根树*（rooted tree）有 `n` 个节点，编号 `0 .. n-1`，用父节点数组 `parent` 给出：`parent[v]` 是节点
`v` 的父节点，根节点的 `parent` 是 `-1`。编号没有任何顺序保证，父节点的编号可以比子节点大。节点 `v` 的
*深度*（depth）是从 `v` 走到根要经过的边数，根的深度是 `0`。从 `v` 沿父指针往上走 `k` 条边到达的节点，叫
做 `v` 的*第 k 个祖先*（k-th ancestor）；本题约定 `k = 0` 时就是 `v` 自己，也就是说一个节点是它自己的祖先。

规模：`1 <= n <= 2 * 10**5`，查询总数至多 `2 * 10**5`。树可以非常深，一条 `2 * 10**5` 个节点的链是合法输
入。所以“每次查询沿父指针一步一步往上走”不够：链上一次查询就要走 $O(n)$ 步，全部查询加起来是
$4 \times 10^{10}$ 步。要求是：构造 $O(n \log n)$，每次查询 $O(\log n)$。

### Part 1 —— 第 k 个祖先

```py
class RootedTree:
    def __init__(self, parent: list[int]) -> None: ...
    def kth_ancestor(self, v: int, k: int) -> int: ...   # the node k edges above v, or -1 if there is none
```

构造函数要检查 `parent` 确实描述了一棵有根树，否则抛出 `ValueError`。合法的条件是：恰好有一个位置的值是
`-1`（它就是根）；其余每个位置的值都是 `[0, n)` 里的节点编号；每个节点沿父指针一直往上走，最终都能到达
根。空列表、两个根、某个节点的父节点是它自己、几个节点的父指针绕成一圈，都属于不合法。

`kth_ancestor(v, k)`：`k = 0` 返回 `v`；`k` 不超过 `v` 的深度时返回对应的祖先；`k` 大于 `v` 的深度时返回
`-1`。`k` 可以很大（比如 `10**18`）。`v` 不在 `[0, n)` 内抛出 `IndexError`；`k < 0` 抛出 `ValueError`。

下面三个部分都用同一棵树举例：

```text
parent = [3, 0, 0, -1, 3, 4, 4, 6, 1]          # n = 9，根是节点 3

                3                深度 0
              /   \
             0     4             深度 1
            / \   / \
           1   2 5   6           深度 2
           |         |
           8         7           深度 3

kth_ancestor(8, 0) == 8
kth_ancestor(8, 1) == 1
kth_ancestor(8, 2) == 0
kth_ancestor(8, 3) == 3
kth_ancestor(8, 4) == -1        # 节点 8 的深度是 3，没有第 4 个祖先
kth_ancestor(5, 2) == 3
kth_ancestor(3, 0) == 3
kth_ancestor(3, 1) == -1        # 根没有父节点
```

### Part 2 —— 最近公共祖先与距离

```py
    def lca(self, u: int, v: int) -> int: ...        # lowest common ancestor of u and v
    def distance(self, u: int, v: int) -> int: ...   # number of edges on the path between u and v
```

`u` 和 `v` 的*最近公共祖先*（lowest common ancestor，LCA）是同时为两者祖先的节点里最深的那一个。因为节
点是它自己的祖先，所以 `lca(u, u) == u`；如果 `u` 本身是 `v` 的祖先，`lca(u, v) == u`。树上两个节点之间只
有一条不重复经过节点的路径：从 `u` 往上走到 LCA，再往下走到 `v`。`distance(u, v)` 是这条路径上的边数，
`distance(u, u) == 0`。编号不在 `[0, n)` 内抛出 `IndexError`。两个函数都要求每次调用 $O(\log n)$。

```text
lca(8, 2) == 0        distance(8, 2) == 3        # 路径 8 - 1 - 0 - 2
lca(8, 7) == 3        distance(8, 7) == 6        # 路径 8 - 1 - 0 - 3 - 4 - 6 - 7
lca(5, 7) == 4        distance(5, 7) == 3        # 路径 5 - 4 - 6 - 7
lca(6, 7) == 6        distance(6, 7) == 1        # 6 本身就是 7 的祖先
lca(3, 8) == 3        distance(3, 8) == 3        # 3 是根
lca(7, 7) == 7        distance(7, 7) == 0
```

### Part 3 —— 路径上的最大边权

```py
    def __init__(self, parent: list[int], weight: list[int] | None = None) -> None: ...
    def path_max(self, u: int, v: int) -> int | None: ...
```

现在每条边有一个整数权重：`weight[v]` 是 `v` 与 `parent[v]` 之间那条边的权重，取值在 `[-10**9, 10**9]`
内，可以是负数，也可以重复。`weight[根]` 不对应任何边，它的值被忽略。`weight` 的长度不等于 `n` 时构造函
数抛出 `ValueError`。

`path_max(u, v)` 返回 `u` 到 `v` 的路径上最大的边权。`u == v` 时路径上没有边，返回 `None`。编号不在
`[0, n)` 内抛出 `IndexError`；构造时没有给 `weight` 却调用 `path_max`，抛出 `ValueError`。每次调用同样要求
$O(\log n)$，额外的预处理不超过 $O(n \log n)$。

```text
weight = [4, 7, 1, 0, 2, 9, 3, 5, 6]            # weight[3] 是根的位置，被忽略

                    3
                4 /   \ 2
                 /     \
                0       4
             7 / \ 1 9 / \ 3
              1   2   5   6
            6 |           | 5
              8           7

path_max(8, 2) == 7          # 边权依次是 6, 7, 1
path_max(8, 7) == 7          # 6, 7, 4, 2, 3, 5
path_max(5, 7) == 9          # 9, 3, 5
path_max(2, 7) == 5          # 1, 4, 2, 3, 5
path_max(2, 4) == 4          # 1, 4, 2
path_max(6, 7) == 5          # 只有一条边
path_max(7, 7) is None       # 没有边
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手之前先问清楚四件事。编号有没有顺序（没有，父节点编号可以更大，所以不能假设“按编号从小到大处理时父节
点已经处理过”）。树有多深（可以是一条链，所以深度的计算不能用递归，查询也不能一步一步走）。一个节点算不
算自己的祖先（算，这决定了 `lca(u, u)` 和“`u` 是 `v` 的祖先”两种情况的答案）。输入是否保证合法（不保证，
构造函数要检查）。

三个部分用的是同一个想法：*倍增*（binary lifting）。预先为每个节点存下“往上跳 $1, 2, 4, 8, \dots$ 条边
到哪里”，任何长度的上跳都能拆成至多 $\log_2 n$ 次这样的跳。

### Part 1

**先算深度，顺便检查输入。** 从 `parent` 建出每个节点的子节点列表，然后从根开始做广度优先搜索（BFS），
用一个列表当队列。不用递归：Python 默认的递归上限是 1000 层，一条 `2 * 10**5` 个节点的链会直接抛出
`RecursionError`（验证代码里演示了这一点）。BFS 结束后，如果访问到的节点数不等于 `n`，说明有节点走不到
根。此时每个非根节点都有一个合法的父节点，从一个走不到根的节点出发一直往上走不会停，只能是进了一个环，
所以这一条检查同时覆盖了“父节点是自己”和“几个节点绕成一圈”。

**倍增表。** `up[j][v]` 是 `v` 往上跳 $2^j$ 条边到达的节点；如果还没跳够就已经到了根，就停在根上。为了让
“停在根上”不需要任何特殊判断，在表里把根的父节点写成根自己。这样有递推

$$\texttt{up}[j][v] = \texttt{up}[j-1]\big[\,\texttt{up}[j-1][v]\,\big],$$

也就是跳 $2^j$ 条边等于连续跳两次 $2^{j-1}$ 条边。它在根附近也成立：如果第一次跳已经停在根上，第二次从根
出发仍然停在根上。

需要多少层：深度最大是 $n-1$，所以取最小的 `LOG >= 1` 使 $2^{\texttt{LOG}} > n-1$，也就是
`max(1, (n - 1).bit_length())`。$n = 2 \times 10^5$ 时 `LOG = 18`。

**查询。** 先比较 `k` 和 `depth[v]`：`k` 更大就直接返回 `-1`。否则 `k <= depth[v] <= n - 1 < 2**LOG`，把
`k` 写成二进制，第 $j$ 位是 1 就跳一次 $2^j$。这些跳的总长度正好是 `k`，先跳哪一位结果都一样。先比较深度
这一步有两个作用：`k = 10**18` 这样的输入不会去访问不存在的层；表里“停在根上”的值不会被误当成答案返回。

代价：建表是 `LOG` 次长度为 `n` 的列表推导，时间和内存都是 $O(n \log n)$；一次查询至多跳 `LOG` 次，
$O(\log n)$。

```python
class RootedTree:
    """A rooted tree given by its parent array. up[j][v] is the node 2**j edges above v, clamped at the
    root: inside the table the root is its own parent, so a jump that would leave the tree stays put."""

    def __init__(self, parent: list[int]) -> None:
        n = len(parent)
        roots = [v for v in range(n) if parent[v] == -1]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root, found {len(roots)}")
        root = roots[0]
        children: list[list[int]] = [[] for _ in range(n)]
        for v, p in enumerate(parent):
            if v == root:
                continue
            if not 0 <= p < n:
                raise ValueError(f"parent[{v}] = {p} is not a node")
            children[p].append(v)
        depth = [0] * n
        order = [root]                    # BFS order; the list is its own queue
        for u in order:                   # NOTE: no recursion -- a path of 2 * 10**5 nodes is a legal input
            for c in children[u]:
                depth[c] = depth[u] + 1
                order.append(c)
        if len(order) != n:               # an unreached node can only be walking around a cycle
            raise ValueError("some node does not reach the root")
        self.n, self.root, self.depth, self.order = n, root, depth, order
        self.log = max(1, (n - 1).bit_length())      # smallest LOG >= 1 with 2**LOG > n - 1 >= every depth
        first = list(parent)
        first[root] = root                # NOTE: the root points at itself, so every jump clamps for free
        self.up = [first]
        for _ in range(1, self.log):
            prev = self.up[-1]
            self.up.append([prev[a] for a in prev])  # 2**j edges = 2**(j-1) edges, twice

    def _check_node(self, v: int) -> None:
        if not 0 <= v < self.n:
            raise IndexError(f"node {v} out of range [0, {self.n})")

    def kth_ancestor(self, v: int, k: int) -> int:
        self._check_node(v)
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > self.depth[v]:
            return -1                     # NOTE: decided by depth alone, which also guarantees k < 2**LOG
        j = 0
        while k:
            if k & 1:
                v = self.up[j][v]
            k >>= 1
            j += 1
        return v
```

### Part 2

分两步。

**第一步：拉到同一深度。** 把较深的那个节点往上跳 `depth[u] - depth[v]` 条边，这就是一次 Part 1 的查询。
跳完之后如果两个节点重合，说明原来较浅的那个就是另一个的祖先，它就是答案。

**第二步：两个节点一起往上跳。** 现在 `u != v`，深度相同。设它们各自还要往上走 $t$ 条边才到 LCA，
$t \ge 1$ 但不知道是多少。关键的性质是单调性：一起往上走 $s$ 条边之后，$s < t$ 时两个节点不同，$s \ge t$
时两个节点相同（两条路一旦汇合就不再分开，表里停在根上的值也满足这一点）。我们想恰好走 $t - 1$ 条边，停
在 LCA 的正下方。做法是从最高位到最低位逐位试：如果一起跳 $2^j$ 之后两个节点**仍然不同**，就跳，否则不
跳。

不变式：处理第 $j$ 位之前，记 $r$ 为“还要再走多少条边才到 LCA 的正下方”，则 $0 \le r < 2^{j+1}$。开始时
$r = t - 1 \le \texttt{depth} - 1 < 2^{\texttt{LOG}}$，成立。“跳 $2^j$ 之后仍然不同”等价于
$2^j < r + 1$，即 $2^j \le r$：这时跳过去，$r$ 减去 $2^j$ 后小于 $2^j$；否则 $r$ 本来就小于 $2^j$。两种情
况下不变式对 $j - 1$ 都成立。处理完第 0 位后 $r < 1$，即 $r = 0$：两个节点正好在 LCA 的两个不同子节点上，
答案是 `up[0][u]`。

如果改成“对 $s$ 做二分，每次调用 `kth_ancestor`”，结果也对，但每次查询是 $O(\log^2 n)$；逐位试只扫一遍
各层，是 $O(\log n)$。

距离直接由深度得到：`u` 到 LCA 有 `depth[u] - depth[a]` 条边，`v` 到 LCA 有 `depth[v] - depth[a]` 条边。

```python
def lca(self, u: int, v: int) -> int:
    self._check_node(u)
    self._check_node(v)
    depth, up = self.depth, self.up
    if depth[u] < depth[v]:
        u, v = v, u
    u = self.kth_ancestor(u, depth[u] - depth[v])    # both are now at the same depth
    if u == v:
        return u                                     # one of them was an ancestor of the other
    for j in range(self.log - 1, -1, -1):
        if up[j][u] != up[j][v]:                     # NOTE: jump only while the two stay different,
            u, v = up[j][u], up[j][v]                #       so the loop ends one edge below the answer
    return up[0][u]


def distance(self, u: int, v: int) -> int:
    a = self.lca(u, v)
    return self.depth[u] + self.depth[v] - 2 * self.depth[a]


RootedTree.lca = lca
RootedTree.distance = distance
```

### Part 3

路径 `u → v` 由两段“往上走”组成：`u` 到 LCA，`v` 到 LCA。两段的长度都能从深度算出来，所以问题变成：从
`x` 往上走 `k` 条边，这些边里最大的权重是多少。

这和 Part 1 是同一个拆法，只是每跳一次要顺带知道这一跳经过的边里的最大值。再建一张表：`mx[j][v]` 是从
`v` 往上的 $2^j$ 条边里最大的权重，递推是

$$\texttt{mx}[j][v] = \max\big(\texttt{mx}[j-1][v],\ \texttt{mx}[j-1][\,\texttt{up}[j-1][v]\,]\big),$$

前半段 $2^{j-1}$ 条边和后半段 $2^{j-1}$ 条边各取最大值。根的位置填 $-\infty$：根往上没有边，不够 $2^j$ 条
边就到了根的时候，多出来的部分取到的都是 $-\infty$，不影响最大值。查询时把 `k` 按二进制拆开，每跳一次用
对应的 `mx[j][v]` 更新答案；这些块互不重叠，合起来正好是那 `k` 条边。

`u == v` 时两段长度都是 0，没有边，按题意返回 `None`；其他情况下路径上至少有一条边，结果一定是某条边的权
重，不会是 $-\infty$。一次 `path_max` 是一次 `lca` 加两次上跳，$O(\log n)$；多出来的内存是第二张
$n \times \texttt{LOG}$ 的表。

```python
NEG_INF = float("-inf")


class RootedTree(RootedTree):
    """Adds edge weights. mx[j][v] is the largest weight among the 2**j edges going up from v (fewer if
    the root comes first: the root's own slot holds -inf, which never wins a max)."""

    def __init__(self, parent: list[int], weight: list[int] | None = None) -> None:
        super().__init__(parent)
        self.mx = None
        if weight is None:
            return
        if len(weight) != self.n:
            raise ValueError("weight must have one entry per node")
        first = list(weight)
        first[self.root] = NEG_INF        # NOTE: weight[root] belongs to no edge
        self.mx = [first]
        for j in range(1, self.log):
            below, half = self.mx[-1], self.up[j - 1]
            self.mx.append([max(m, below[a]) for m, a in zip(below, half)])

    def _max_up(self, v: int, k: int):
        """Largest weight among the k edges going up from v (-inf when k == 0). Needs k <= depth[v]."""
        best = NEG_INF
        j = 0
        while k:
            if k & 1:
                best = max(best, self.mx[j][v])   # NOTE: read the block's maximum BEFORE leaving v
                v = self.up[j][v]
            k >>= 1
            j += 1
        return best

    def path_max(self, u: int, v: int) -> int | None:
        if self.mx is None:
            raise ValueError("the tree was built without weights")
        a = self.lca(u, v)                # also validates u and v
        if u == v:
            return None                   # the path has no edge
        return max(self._max_up(u, self.depth[u] - self.depth[a]),
                   self._max_up(v, self.depth[v] - self.depth[a]))
```

### 追问

- **什么时候不需要这张表？** 树不深的时候。沿父指针一步一步走，一次查询的代价是 $O(h)$，$h$ 是树的高度。
  如果树是平衡的（$h = O(\log n)$），朴素做法的查询代价和倍增一样，内存只要 $O(n)$，代码也短得多。查询很
  少的时候也不值得建表：建表是 $n \log n$ 次操作，查询总共只走 $q \cdot h$ 步。只有“树可能很深，而且查询很
  多”时倍增才划算，所以面试里应该先问这两个数。
- **这张表占多少内存，能不能省？** 表有 $n \times \texttt{LOG}$ 项，$n = 2 \times 10^5$ 时是 $3.6 \times
  10^6$ 项。在 CPython 里每项是一个 8 字节的指针，`up` 这一张表的各行加起来约 29 MB，带权重时 `mx` 再占同
  样多。换成 NumPy 的 `int32` 数组是 14.4 MB，而且建表可以向量化成一行 `up[j] = up[j-1][up[j-1]]`。再省就
  要换结构：只为每个节点存一个“往上 $B = \lceil\sqrt{n}\rceil$ 条边”的指针，查询时先大步跳、剩下不足
  $B$ 条边再一步一步走。内存降到 $O(n)$，一次查询变成 $O(\sqrt{n})$：$n = 2 \times 10^5$ 时 $B = 448$，
  至多约 900 步，而倍增是 18 步。这是内存和查询时间之间的直接交换。
- **如果聚合不是最大值而是求和？** 不需要第二张表。按 BFS 顺序算出 `pre[v]`（从 `v` 到根的边权之和），
  `u` 到 `v` 的路径和就是 `pre[u] + pre[v] - 2 * pre[lca(u, v)]`，额外内存 $O(n)$。能这样做是因为加法有逆
  运算，可以把 LCA 以上被多算的那一段减掉；最大值没有逆运算，“根到 `u` 的最大值”里分不出哪一部分属于
  LCA 以上，所以才需要按块存的表。
- **树会变化怎么办？** 看是哪种变化。*加一个叶子*很便宜：新节点在每一层的那一项只依赖它的祖先，祖先的
  项都已经在了，每层补一项即可。只有节点数涨到 $2^{\texttt{LOG}}$ 以上时才要给所有节点多建一层，这件事在
  节点数每翻一倍时发生一次，所以*摊还*（amortised）下来每次加叶子是 $O(\log n)$。*改一条边的权重*就不行
  了：`mx` 里所有经过这条边的项都要改，最坏是整棵子树乘以层数。求和的情形可以把树按 DFS 进出时间
  压平成数组，再用树状数组（Fenwick tree）做“子树整体加、单点查”；最大值的情形要用*重链剖分*（heavy-light
  decomposition）加线段树（segment tree），查询变成 $O(\log^2 n)$。*给节点换父节点*则要用 link-cut tree
  这一类专门的动态树结构。
- **只要 LCA，还有别的做法吗？** 有两种常见的。一种是把树的欧拉序（Euler tour）写成数组，LCA 变成数组上的
  区间最小值，用稀疏表（sparse table）预处理 $O(n \log n)$、查询 $O(1)$；它比倍增查询更快，但回答不了“第
  k 个祖先”和“路径最大值”。另一种是所有查询事先已知时用 Tarjan 的离线算法：一次 DFS 加并查集
  （union-find），总代价接近线性。倍增的优点是一张表同时支持三种查询，而且是在线的。

下面的代码对应“求和”和“加叶子”两条追问，验证代码会检查它们。

```python
def root_prefix_sums(tree: RootedTree, weight: list[int]) -> list[int]:
    """pre[v] = total weight of the edges from v up to the root (0 for the root itself)."""
    pre = [0] * tree.n
    for v in tree.order[1:]:              # BFS order: a parent is always filled in before its children
        pre[v] = pre[tree.up[0][v]] + weight[v]
    return pre


def path_sum(tree: RootedTree, pre: list[int], u: int, v: int) -> int:
    return pre[u] + pre[v] - 2 * pre[tree.lca(u, v)]


class GrowingTree(RootedTree):
    """Adds add_leaf(): a new node's rows depend only on its ancestors' rows, which already exist."""

    def add_leaf(self, p: int, w: int = 0) -> int:
        self._check_node(p)
        v = self.n
        if self.n >= 1 << self.log:                   # keep 2**LOG > n - 1: one more level, for everyone
            top = self.up[-1]
            if self.mx is not None:
                self.mx.append([max(m, self.mx[-1][a]) for m, a in zip(self.mx[-1], top)])
            self.up.append([top[a] for a in top])
            self.log += 1
        self.n += 1
        self.depth.append(self.depth[p] + 1)
        self.order.append(v)                          # appending keeps "parents before children"
        self.up[0].append(p)
        for j in range(1, self.log):
            self.up[j].append(self.up[j - 1][self.up[j - 1][v]])
        if self.mx is not None:
            self.mx[0].append(w)
            for j in range(1, self.log):
                self.mx[j].append(max(self.mx[j - 1][v], self.mx[j - 1][self.up[j - 1][v]]))
        return v
```

<details>
<summary>验证代码（可运行）</summary>

```python
import math
import random
import sys

import numpy as np

# --- the statement's tree, all three parts ---
PARENT = [3, 0, 0, -1, 3, 4, 4, 6, 1]
WEIGHT = [4, 7, 1, 0, 2, 9, 3, 5, 6]
t = RootedTree(PARENT, WEIGHT)
assert t.root == 3 and t.depth == [1, 2, 2, 0, 1, 2, 2, 3, 3]

assert [t.kth_ancestor(8, k) for k in range(5)] == [8, 1, 0, 3, -1]
assert t.kth_ancestor(5, 2) == 3
assert t.kth_ancestor(3, 0) == 3 and t.kth_ancestor(3, 1) == -1
assert t.kth_ancestor(8, 10**18) == -1

for u, v, want_lca, want_dist in [(8, 2, 0, 3), (8, 7, 3, 6), (5, 7, 4, 3), (6, 7, 6, 1), (3, 8, 3, 3),
                                  (7, 7, 7, 0)]:
    assert t.lca(u, v) == t.lca(v, u) == want_lca, (u, v)
    assert t.distance(u, v) == t.distance(v, u) == want_dist, (u, v)

for u, v, want in [(8, 2, 7), (8, 7, 7), (5, 7, 9), (2, 7, 5), (2, 4, 4), (6, 7, 5), (7, 7, None)]:
    assert t.path_max(u, v) == t.path_max(v, u) == want, (u, v)
assert RootedTree(PARENT, [4, 7, 1, 10**9, 2, 9, 3, 5, 6]).path_max(8, 7) == 7   # weight[root] is ignored


# --- error behaviour ---
def _raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


for bad in ([], [0], [1, 0], [-1, -1], [-1, 5], [-1, -2], [-1, 0, 2], [-1, 2, 1], [1, 2, 0, -1],
            [-1, 0, 3, 4, 2]):
    assert _raises(ValueError, RootedTree, bad), bad      # no root / two roots / bad id / self-loop / cycles
assert RootedTree([-1]).kth_ancestor(0, 0) == 0 and RootedTree([-1]).lca(0, 0) == 0     # a single node
assert RootedTree([-1]).kth_ancestor(0, 1) == -1 and RootedTree([-1], [5]).path_max(0, 0) is None
assert _raises(ValueError, RootedTree, PARENT, WEIGHT[:-1])
assert _raises(ValueError, t.kth_ancestor, 8, -1)
for bad_node in (-1, 9):
    assert _raises(IndexError, t.kth_ancestor, bad_node, 0)
    assert _raises(IndexError, t.lca, bad_node, 0) and _raises(IndexError, t.lca, 0, bad_node)
    assert _raises(IndexError, t.distance, bad_node, 0)
    assert _raises(IndexError, t.path_max, bad_node, 0) and _raises(IndexError, t.path_max, bad_node, bad_node)
assert _raises(ValueError, RootedTree(PARENT).path_max, 8, 2)        # built without weights


# --- an independent brute force: nothing but single parent-pointer steps ---
def _chain(parent, v):
    """v, its parent, ..., the root."""
    out = [v]
    while parent[out[-1]] != -1:
        out.append(parent[out[-1]])
    return out


def _brute_kth(parent, v, k):
    chain = _chain(parent, v)
    return chain[k] if k < len(chain) else -1


def _brute_lca(parent, u, v):
    above_u = set(_chain(parent, u))
    return next(a for a in _chain(parent, v) if a in above_u)


def _brute_path_weights(parent, weight, u, v):
    a = _brute_lca(parent, u, v)
    out = []
    for x in (u, v):
        while x != a:
            out.append(weight[x])
            x = parent[x]
    return out


def _random_tree(rng, n, shape):
    """A parent array of the given shape with shuffled labels (so a parent's id is not smaller than its
    child's). `par` is written over positions 0..n-1, position 0 being the root."""
    if shape == "random":
        par = [-1] + [rng.randrange(i) for i in range(1, n)]
    elif shape == "path":
        par = [-1] + [i - 1 for i in range(1, n)]
    elif shape == "star":
        par = [-1] + [0] * (n - 1)
    elif shape == "binary":
        par = [-1] + [(i - 1) // 2 for i in range(1, n)]
    elif shape == "caterpillar":                      # a spine 0, 2, 4, ... with one leaf hanging off each
        par = [-1] + [i - 1 if i % 2 else i - 2 for i in range(1, n)]
        par[1:2] = [0] * min(1, n - 1)
    else:                                             # "thin": long and narrow, parents 1..3 positions back
        par = [-1] + [max(0, i - 1 - rng.randrange(3)) for i in range(1, n)]
    label = list(range(n))
    rng.shuffle(label)
    parent = [0] * n
    for i in range(n):
        parent[label[i]] = -1 if par[i] == -1 else label[par[i]]
    return parent


def _compare_with_brute(cls, parent, weight):
    """Every (u, v) pair and every k against the brute force; returns the number of mismatches."""
    n = len(parent)
    tree = cls(parent, weight)
    bad = 0
    for u in range(n):
        chain = _chain(parent, u)
        assert tree.depth[u] == len(chain) - 1
        for k in range(n + 2):
            bad += tree.kth_ancestor(u, k) != _brute_kth(parent, u, k)
        for v in range(n):
            a = _brute_lca(parent, u, v)
            ws = _brute_path_weights(parent, weight, u, v)
            bad += tree.lca(u, v) != a
            bad += tree.distance(u, v) != len(ws)
            bad += tree.path_max(u, v) != (max(ws) if ws else None)
    return bad


def _check_tables(parent, weight):
    """The invariant behind every query: each table entry equals its definition, clamped at the root."""
    tree = RootedTree(parent, weight)
    assert tree.log == max(1, (tree.n - 1).bit_length()) and 2 ** tree.log > max(tree.depth)
    assert len(tree.up) == len(tree.mx) == tree.log
    for v in range(tree.n):
        chain = _chain(parent, v)
        for j in range(tree.log):
            steps = min(2 ** j, len(chain) - 1)
            assert tree.up[j][v] == chain[steps]
            want = max((weight[x] for x in chain[:steps]), default=NEG_INF)
            assert tree.mx[j][v] == want


SHAPES = ["random", "path", "star", "binary", "caterpillar", "thin"]
pairs_checked = 0
for shape in SHAPES:
    for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 33):
        for seed in range(3):
            rng = random.Random(f"{shape}-{n}-{seed}")
            parent = _random_tree(rng, n, shape)
            # small weight ranges force ties; all-equal and all-negative weights are included
            lo, hi = [(-5, 5), (7, 7), (-10**9, -10**9 + 3)][seed]
            weight = [rng.randint(lo, hi) for _ in range(n)]
            assert _compare_with_brute(RootedTree, parent, weight) == 0, (shape, n, seed)
            _check_tables(parent, weight)
            pairs_checked += n * n
assert pairs_checked > 30_000


# --- a negative control: the comparison above must be able to fail ---
class _SkipsLastBit(RootedTree):
    def lca(self, u, v):
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        u = self.kth_ancestor(u, self.depth[u] - self.depth[v])
        if u == v:
            return u
        for j in range(self.log - 1, 0, -1):          # BUG: never tries the 2**0 jump
            if self.up[j][u] != self.up[j][v]:
                u, v = self.up[j][u], self.up[j][v]
        return self.up[0][u]


rng = random.Random(5)
control_parent = _random_tree(rng, 33, "binary")
assert _compare_with_brute(_SkipsLastBit, control_parent, [1] * 33) > 0


# --- the largest path the statement allows: 2 * 10**5 nodes in one chain, labels shuffled ---
N = 200_000
rng = random.Random(2026)
label = list(range(N))
rng.shuffle(label)                        # label[i] is the node at depth i
path_parent = [0] * N
path_parent[label[0]] = -1
for i in range(1, N):
    path_parent[label[i]] = label[i - 1]
path_weight = [rng.randint(-10**9, 10**9) for _ in range(N)]
big = RootedTree(path_parent, path_weight)           # must not raise RecursionError
assert big.log == 18 and big.depth[label[N - 1]] == N - 1


def _depth_recursive(parent, v):
    return 0 if parent[v] == -1 else 1 + _depth_recursive(parent, parent[v])


assert _depth_recursive(path_parent, label[500]) == 500
assert _raises(RecursionError, _depth_recursive, path_parent, label[N - 1])   # why the reference uses BFS

prefix_max = [NEG_INF] * N                # prefix_max[i]: largest weight on the edges from depth i up to the root
for i in range(1, N):
    prefix_max[i] = max(prefix_max[i - 1], path_weight[label[i]])
for _ in range(20_000):
    i, j = rng.randrange(N), rng.randrange(N)
    k = rng.randrange(N + 5)
    assert big.kth_ancestor(label[i], k) == (label[i - k] if k <= i else -1)
    assert big.lca(label[i], label[j]) == label[min(i, j)]
    assert big.distance(label[i], label[j]) == abs(i - j)
    assert big.path_max(label[0], label[i]) == (prefix_max[i] if i else None)
    lo = rng.randrange(N - 300)
    hi = lo + rng.randrange(300)
    want = max((path_weight[label[x]] for x in range(lo + 1, hi + 1)), default=None)
    assert big.path_max(label[lo], label[hi]) == big.path_max(label[hi], label[lo]) == want
assert big.kth_ancestor(label[N - 1], N - 1) == label[0] and big.kth_ancestor(label[N - 1], N) == -1

# the table's size, as quoted in the follow-ups
assert big.log * N == 3_600_000
up_megabytes = sum(sys.getsizeof(row) for row in big.up) / 1e6
assert 28 < up_megabytes < 31, up_megabytes        # "about 29 MB" in the follow-ups
up_np = np.empty((big.log, N), dtype=np.int32)
up_np[0] = big.up[0]
for j in range(1, big.log):
    up_np[j] = up_np[j - 1][up_np[j - 1]]             # the one-line vectorised build
assert up_np.nbytes == 14_400_000 and all((up_np[j] == big.up[j]).all() for j in range(big.log))

# --- follow-up: one pointer per node, ceil(sqrt(n)) edges up -- O(n) memory, O(sqrt(n)) per query ---
def _sqrt_jumps(parent, block):
    """jump[v] = the node `block` edges above v, or -1. One DFS with an explicit stack; `chain` holds the
    root-to-v path of the node being visited."""
    children = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p != -1:
            children[p].append(v)
    jump, chain, stack = [-1] * len(parent), [], [(parent.index(-1), 0)]
    while stack:
        v, d = stack.pop()
        del chain[d:]                     # back to the chain of v's parent
        chain.append(v)
        if d >= block:
            jump[v] = chain[d - block]
        stack.extend((c, d + 1) for c in children[v])
    return jump


def _kth_by_sqrt_jumps(parent, depth, jump, block, v, k):
    """(answer, number of pointer steps taken)."""
    if k > depth[v]:
        return -1, 0
    steps = 0
    while k >= block:
        v, k, steps = jump[v], k - block, steps + 1
    while k:
        v, k, steps = parent[v], k - 1, steps + 1
    return v, steps


for seed in range(30):
    rng2 = random.Random(3000 + seed)
    n = rng2.randint(1, 40)
    parent = _random_tree(rng2, n, rng2.choice(SHAPES))
    small = RootedTree(parent)
    for block in (1, 2, 3, math.isqrt(n - 1) + 1, n + 1):
        jump = _sqrt_jumps(parent, block)
        for v in range(n):
            for k in range(n + 2):
                assert _kth_by_sqrt_jumps(parent, small.depth, jump, block, v, k)[0] == _brute_kth(parent, v, k)

BLOCK = math.isqrt(N - 1) + 1             # ceil(sqrt(N))
assert BLOCK == 448 and (BLOCK - 1) ** 2 < N <= BLOCK ** 2
jump = _sqrt_jumps(path_parent, BLOCK)
worst_steps = 0
for _ in range(2_000):
    v, k = label[rng.randrange(N)], rng.randrange(N)
    got, steps = _kth_by_sqrt_jumps(path_parent, big.depth, jump, BLOCK, v, k)
    assert got == big.kth_ancestor(v, k)
    worst_steps = max(worst_steps, steps)
assert 600 < worst_steps <= N // BLOCK + BLOCK - 1 < 900     # "at most about 900 steps", against 18 levels

# --- two more shapes of the same size: two long arms meeting at the root, and a star ---
half = N // 2
arm_parent = [-1] + [max(0, i - 2) for i in range(1, N)]     # node 0 is the root; odd nodes form one arm,
arm_weight = [0] + [rng.randint(-50, 50) for _ in range(1, N)]   # even nodes the other
arms = RootedTree(arm_parent, arm_weight)
arm_prefix = [NEG_INF] * N                # largest weight between node x and the root
for x in range(1, N):
    arm_prefix[x] = max(arm_weight[x], arm_prefix[x - 2] if x > 2 else NEG_INF)
for _ in range(10_000):
    a, b = 2 * rng.randrange(half) + 1, 2 * rng.randrange(1, half)        # one node on each arm
    assert arms.lca(a, b) == 0
    assert arms.distance(a, b) == (a + 1) // 2 + b // 2
    assert arms.path_max(a, b) == max(arm_prefix[a], arm_prefix[b])
    c = 2 * rng.randrange(half) + 1                                        # two nodes on the same arm
    assert arms.lca(a, c) == min(a, c) and arms.distance(a, c) == abs(a - c) // 2

star = RootedTree([-1] + [0] * (N - 1), list(range(N)))
for _ in range(2_000):
    a, b = rng.randrange(1, N), rng.randrange(1, N)
    assert star.lca(a, b) == (a if a == b else 0)
    assert star.path_max(a, b) == (None if a == b else max(a, b))
    assert star.kth_ancestor(a, 1) == 0 and star.kth_ancestor(a, 2) == -1

# --- follow-up: path sums from root prefix sums ---
for seed in range(40):
    rng = random.Random(1000 + seed)
    n = rng.randint(1, 40)
    parent = _random_tree(rng, n, rng.choice(SHAPES))
    weight = [rng.randint(-9, 9) for _ in range(n)]
    tree = RootedTree(parent, weight)
    pre = root_prefix_sums(tree, weight)
    for u in range(n):
        for v in range(n):
            assert path_sum(tree, pre, u, v) == sum(_brute_path_weights(parent, weight, u, v))

# --- follow-up: growing a tree leaf by leaf gives exactly the tables of a fresh build ---
for seed in range(30):
    rng = random.Random(2000 + seed)
    weighted = seed % 3 != 0                          # every third run grows a tree without weights
    grown = GrowingTree([-1], [0] if weighted else None)
    parent, weight = [-1], [0]
    for _ in range(rng.choice([1, 2, 7, 40, 70])):
        # mostly extend the newest node (so the tree gets deep), sometimes branch
        p = len(parent) - 1 if rng.random() < 0.8 else rng.randrange(len(parent))
        w = rng.randint(-9, 9)
        assert grown.add_leaf(p, w) == len(parent)
        parent.append(p)
        weight.append(w)
        fresh = RootedTree(parent, weight if weighted else None)
        assert grown.n == fresh.n and grown.log == fresh.log and grown.depth == fresh.depth
        assert grown.up == fresh.up and grown.mx == fresh.mx
    if weighted:
        assert _compare_with_brute(lambda *_: grown, parent, weight) == 0
    else:
        assert all(grown.lca(u, v) == _brute_lca(parent, u, v) for u in range(grown.n) for v in range(grown.n))

print("all checks passed")
```

</details>

</details>
