# Ancestors, Lowest Common Ancestors and Path Maxima on a Rooted Tree

## Problem

A *rooted tree* has `n` nodes, numbered `0 .. n-1`, given by a parent array `parent`: `parent[v]` is the
parent of node `v`, and the root's `parent` is `-1`. The numbering comes with no ordering guarantee; a parent
may have a larger id than its child. The *depth* of node `v` is the number of edges on the way from `v` to the
root; the root has depth `0`. The node reached by walking `k` edges up from `v` along parent pointers is
called the *k-th ancestor* of `v`; in this problem, `k = 0` means `v` itself, i.e. a node is its own ancestor.

Size: `1 <= n <= 2 * 10**5`, and at most `2 * 10**5` queries in total. The tree can be very deep: a chain of
`2 * 10**5` nodes is a valid input. So "walk up the parent pointers one step at a time for each query" is not
good enough: on a chain a single query takes $O(n)$ steps, and all the queries together take
$4 \times 10^{10}$ steps. Requirement: $O(n \log n)$ construction and $O(\log n)$ per query.

### Part 1 — The k-th ancestor

```py
class RootedTree:
    def __init__(self, parent: list[int]) -> None: ...
    def kth_ancestor(self, v: int, k: int) -> int: ...   # the node k edges above v, or -1 if there is none
```

The constructor must check that `parent` really describes a rooted tree and raise `ValueError` otherwise.
It is valid when: exactly one position holds `-1` (that is the root); every other position holds a node id in
`[0, n)`; and every node eventually reaches the root by following parent pointers. An empty list, two roots, a
node that is its own parent, and several nodes whose parent pointers form a cycle are all invalid.

`kth_ancestor(v, k)`: for `k = 0`, return `v`; when `k` is at most the depth of `v`, return the corresponding
ancestor; when `k` is greater than the depth of `v`, return `-1`. `k` can be very large (say `10**18`). Raise
`IndexError` if `v` is not in `[0, n)`; raise `ValueError` if `k < 0`.

All three parts use the same tree as the example:

```text
parent = [3, 0, 0, -1, 3, 4, 4, 6, 1]          # n = 9, the root is node 3

                3                depth 0
              /   \
             0     4             depth 1
            / \   / \
           1   2 5   6           depth 2
           |         |
           8         7           depth 3

kth_ancestor(8, 0) == 8
kth_ancestor(8, 1) == 1
kth_ancestor(8, 2) == 0
kth_ancestor(8, 3) == 3
kth_ancestor(8, 4) == -1        # node 8 has depth 3, so it has no 4th ancestor
kth_ancestor(5, 2) == 3
kth_ancestor(3, 0) == 3
kth_ancestor(3, 1) == -1        # the root has no parent
```

### Part 2 — Lowest common ancestor and distance

```py
    def lca(self, u: int, v: int) -> int: ...        # lowest common ancestor of u and v
    def distance(self, u: int, v: int) -> int: ...   # number of edges on the path between u and v
```

The *lowest common ancestor* (LCA) of `u` and `v` is the deepest node that is an ancestor of both. Since a
node is its own ancestor, `lca(u, u) == u`; if `u` itself is an ancestor of `v`, then `lca(u, v) == u`. Between
two nodes of a tree there is exactly one path that does not repeat a node: go up from `u` to the LCA, then down
to `v`. `distance(u, v)` is the number of edges on this path, and `distance(u, u) == 0`. Raise `IndexError` for
ids outside `[0, n)`. Both functions must be $O(\log n)$ per call.

```text
lca(8, 2) == 0        distance(8, 2) == 3        # path 8 - 1 - 0 - 2
lca(8, 7) == 3        distance(8, 7) == 6        # path 8 - 1 - 0 - 3 - 4 - 6 - 7
lca(5, 7) == 4        distance(5, 7) == 3        # path 5 - 4 - 6 - 7
lca(6, 7) == 6        distance(6, 7) == 1        # 6 itself is an ancestor of 7
lca(3, 8) == 3        distance(3, 8) == 3        # 3 is the root
lca(7, 7) == 7        distance(7, 7) == 0
```

### Part 3 — The maximum edge weight on a path

```py
    def __init__(self, parent: list[int], weight: list[int] | None = None) -> None: ...
    def path_max(self, u: int, v: int) -> int | None: ...
```

Now every edge has an integer weight: `weight[v]` is the weight of the edge between `v` and `parent[v]`, in
`[-10**9, 10**9]`; weights may be negative and may repeat. `weight[root]` belongs to no edge and its value is
ignored. If the length of `weight` is not `n`, the constructor raises `ValueError`.

`path_max(u, v)` returns the largest edge weight on the path from `u` to `v`. When `u == v` the path has no
edges; return `None`. Raise `IndexError` for ids outside `[0, n)`; calling `path_max` when no `weight` was
given at construction raises `ValueError`. Each call must again be $O(\log n)$, with no more than
$O(n \log n)$ extra preprocessing.

```text
weight = [4, 7, 1, 0, 2, 9, 3, 5, 6]            # weight[3] is the root's slot and is ignored

                    3
                4 /   \ 2
                 /     \
                0       4
             7 / \ 1 9 / \ 3
              1   2   5   6
            6 |           | 5
              8           7

path_max(8, 2) == 7          # edge weights in order: 6, 7, 1
path_max(8, 7) == 7          # 6, 7, 4, 2, 3, 5
path_max(5, 7) == 9          # 9, 3, 5
path_max(2, 7) == 5          # 1, 4, 2, 3, 5
path_max(2, 4) == 4          # 1, 4, 2
path_max(6, 7) == 5          # only one edge
path_max(7, 7) is None       # no edges
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Before writing any code, clarify four things. Is the numbering ordered? (No: a parent's id can be larger, so
you can't assume that "processing nodes in increasing id order means the parent was already processed".) How
deep can the tree be? (It can be a chain, so depths can't be computed recursively, and queries can't walk one
step at a time.) Does a node count as its own ancestor? (Yes; this determines the answers for `lca(u, u)` and
for the case where "`u` is an ancestor of `v`".) Is the input guaranteed to be valid? (No; the constructor has
to check.)

All three parts use the same idea: *binary lifting*. Precompute, for every node, "where do I land after jumping
up $1, 2, 4, 8, \dots$ edges"; any upward jump of any length can then be broken into at most $\log_2 n$ such
jumps.

### Part 1

**Compute depths first, checking the input along the way.** Build each node's list of children from `parent`,
then run a breadth-first search (BFS) from the root, using a list as the queue. No recursion: Python's default
recursion limit is 1000 frames, so a chain of `2 * 10**5` nodes would raise `RecursionError` right away (the
verification code demonstrates this). After the BFS, if the number of visited nodes is not `n`, some node can't
reach the root. At that point every non-root node has a valid parent, so walking up from a node that never
reaches the root never stops, which means it must be going around a cycle; this one check therefore covers both
"a node is its own parent" and "several nodes form a cycle".

**The lifting table.** `up[j][v]` is the node reached by jumping $2^j$ edges up from `v`; if the root is reached
before the jump is complete, it stops at the root. So that "stop at the root" needs no special-casing, the table
records the root's parent as the root itself. Then we have the recurrence

$$\texttt{up}[j][v] = \texttt{up}[j-1]\big[\,\texttt{up}[j-1][v]\,\big],$$

i.e. jumping $2^j$ edges is the same as jumping $2^{j-1}$ edges twice in a row. It holds near the root too: if
the first jump already stops at the root, the second jump starts from the root and still stops at the root.

How many levels: the maximum depth is $n-1$, so take the smallest `LOG >= 1` with $2^{\texttt{LOG}} > n-1$,
i.e. `max(1, (n - 1).bit_length())`. For $n = 2 \times 10^5$, `LOG = 18`.

**Queries.** First compare `k` with `depth[v]`: if `k` is larger, return `-1` immediately. Otherwise
`k <= depth[v] <= n - 1 < 2**LOG`; write `k` in binary and, for every bit $j$ that is 1, jump $2^j$ once. The
jumps add up to exactly `k`, and the order in which the bits are taken doesn't matter. Comparing with the depth
first does two things: an input like `k = 10**18` never touches a level that doesn't exist; and the
"stopped at the root" values in the table are never mistakenly returned as answers.

Cost: building the table takes `LOG` list comprehensions of length `n`, so $O(n \log n)$ time and memory; a
query takes at most `LOG` jumps, $O(\log n)$.

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

Two steps.

**Step 1: bring both to the same depth.** Jump the deeper node up by `depth[u] - depth[v]` edges, which is just
one Part 1 query. If the two nodes now coincide, the originally shallower one is an ancestor of the other, and
it is the answer.

**Step 2: jump both nodes up together.** Now `u != v` and they have the same depth. Say each still has to walk
up $t$ edges to reach the LCA, with $t \ge 1$ but unknown. The key property is monotonicity: after walking up
$s$ edges together, the two nodes are different when $s < t$ and the same when $s \ge t$ (once two paths merge
they never separate again, and the stopped-at-the-root values in the table satisfy this too). We want to walk
exactly $t - 1$ edges and stop right below the LCA. The method is to try bit by bit from the highest to the
lowest: if after jumping $2^j$ together the two nodes are **still different**, jump; otherwise don't.

Invariant: before processing bit $j$, let $r$ be "how many more edges until right below the LCA"; then
$0 \le r < 2^{j+1}$. Initially $r = t - 1 \le \texttt{depth} - 1 < 2^{\texttt{LOG}}$, so it holds. "Still
different after jumping $2^j$" is equivalent to $2^j < r + 1$, i.e. $2^j \le r$: in that case we jump, and
after subtracting $2^j$, $r$ is less than $2^j$; otherwise $r$ was already less than $2^j$. Either way the
invariant holds for $j - 1$. After processing bit 0, $r < 1$, i.e. $r = 0$: the two nodes sit exactly on two
different children of the LCA, and the answer is `up[0][u]`.

If you instead "binary search on $s$, calling `kth_ancestor` each time", the result is also correct, but each
query is $O(\log^2 n)$; trying bit by bit scans the levels only once, $O(\log n)$.

The distance follows directly from depths: `u` is `depth[u] - depth[a]` edges from the LCA, and `v` is
`depth[v] - depth[a]` edges from it.

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

The path `u → v` consists of two "upward" walks: `u` to the LCA and `v` to the LCA. Both lengths can be
computed from depths, so the problem becomes: walking `k` edges up from `x`, what is the largest weight among
those edges?

This is the same decomposition as Part 1, except that each jump also needs to know the maximum over the edges
it covers. Build a second table: `mx[j][v]` is the largest weight among the $2^j$ edges going up from `v`, with
the recurrence

$$\texttt{mx}[j][v] = \max\big(\texttt{mx}[j-1][v],\ \texttt{mx}[j-1][\,\texttt{up}[j-1][v]\,]\big),$$

taking the maximum of the first $2^{j-1}$ edges and of the second $2^{j-1}$ edges. The root's slot is filled
with $-\infty$: there are no edges above the root, so when the root is reached before $2^j$ edges, the
remaining part only picks up $-\infty$, which doesn't affect the maximum. At query time, split `k` into binary
and update the answer with the corresponding `mx[j][v]` on every jump; these blocks don't overlap and together
make up exactly those `k` edges.

When `u == v` both walks have length 0 and there are no edges, so per the statement return `None`; in every
other case the path has at least one edge, so the result is always some edge's weight, never $-\infty$. One
`path_max` is one `lca` plus two upward walks, $O(\log n)$; the extra memory is a second
$n \times \texttt{LOG}$ table.

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

### Follow-ups

- **When don't you need the table?** When the tree is shallow. Walking up the parent pointers one step at a
  time costs $O(h)$ per query, where $h$ is the height of the tree. If the tree is balanced ($h = O(\log n)$),
  the naive approach has the same query cost as binary lifting, needs only $O(n)$ memory, and the code is much
  shorter. It also isn't worth building the table when there are few queries: building it takes $n \log n$
  operations, while all queries together only walk $q \cdot h$ steps. Binary lifting pays off only when "the
  tree may be deep and there are many queries", so in the interview you should ask for these two numbers first.
- **How much memory does the table use, and can you save some?** The table has $n \times \texttt{LOG}$
  entries: $3.6 \times 10^6$ for $n = 2 \times 10^5$. In CPython each entry is an 8-byte pointer, so the rows
  of the `up` table alone add up to about 29 MB, and with weights `mx` takes as much again. As NumPy `int32`
  arrays it is 14.4 MB, and the build can be vectorized to one line, `up[j] = up[j-1][up[j-1]]`. Saving more
  means changing the structure: store for each node only one pointer "$B = \lceil\sqrt{n}\rceil$ edges up";
  at query time take big jumps first, then walk the remaining fewer-than-$B$ edges one step at a time. Memory
  drops to $O(n)$ and a query becomes $O(\sqrt{n})$: for $n = 2 \times 10^5$, $B = 448$, at most about 900
  steps, versus 18 for binary lifting. That is a direct trade between memory and query time.
- **What if the aggregate is a sum instead of a maximum?** You don't need a second table. Compute `pre[v]`
  (the sum of edge weights from `v` to the root) in BFS order; the path sum from `u` to `v` is then
  `pre[u] + pre[v] - 2 * pre[lca(u, v)]`, with $O(n)$ extra memory. This works because addition has an
  inverse, so you can subtract the part above the LCA that was counted twice; maximum has no inverse, and
  "the maximum from the root to `u`" can't tell you which part lies above the LCA, which is why you need the
  table stored in blocks.
- **What if the tree changes?** It depends on the change. *Adding a leaf* is cheap: the new node's entry at each
  level depends only on its ancestors, whose entries already exist, so you add one entry per level. Only when
  the node count grows past $2^{\texttt{LOG}}$ do you have to build one more level for all nodes, which happens
  once each time the node count doubles, so *amortised* each leaf addition is $O(\log n)$. *Changing an edge's
  weight* doesn't work this way: every entry in `mx` that covers this edge has to change, in the worst case the
  whole subtree times the number of levels. For sums, you can flatten the tree into an array by DFS entry/exit
  times and use a Fenwick tree for "add to a whole subtree, query a single point"; for maxima you need
  *heavy-light decomposition* plus a segment tree, and queries become $O(\log^2 n)$. *Re-parenting a node*
  requires a dedicated dynamic-tree structure such as a link-cut tree.
- **If you only need the LCA, are there other approaches?** Two common ones. One writes the tree's Euler tour
  as an array, turning LCA into a range minimum on the array, which a sparse table handles with $O(n \log n)$
  preprocessing and $O(1)$ queries; its queries are faster than binary lifting, but it can't answer "k-th
  ancestor" or "path maximum". The other, when all queries are known in advance, is Tarjan's offline
  algorithm: one DFS plus union-find, with nearly linear total cost. The strength of binary lifting is that a
  single table supports all three kinds of queries, and online.

The code below corresponds to the "sum" and "adding a leaf" follow-ups; the verification code checks both.

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
<summary>Verification code (runnable)</summary>

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
