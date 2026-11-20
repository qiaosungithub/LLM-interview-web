# Indexed Priority Queue from Scratch

## Problem

You have just written a shortest-path solution with `heapq`, and the interviewer follows up: now you're not allowed to use `heapq`; write the heap yourself. Once that's done,
make it support “change the priority of a key”.

Conventions for the whole problem:

- You may not use `heapq`, nor `sorted`, `list.sort` or `min` as a substitute for heap operations.
- Elements (or priorities) are compared with `<` only.
- When there are several smallest elements, any of them may be popped first.

### Part 1 — Binary heap

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

`push` and `pop` must be $O(\log n)$, `peek` must be $O(1)$, and the constructor must be $O(n)$.

```text
h = MinHeap([5, 3, 8, 1])
h.peek()                       -> 1
len(h)                         -> 4
h.push(0); h.push(4)
[h.pop() for _ in range(6)]    -> [0, 1, 3, 4, 5, 8]
h.pop()                        -> IndexError
```

Also answer: at most how many comparisons do `push` and `pop` each make? Why is building the heap in one go $O(n)$, while pushing the elements one by one is $O(n \log n)$?

### Part 2 — Indexed priority queue

Now each element is a key plus a priority. Keys are hashable and distinct; keys are not necessarily comparable with each other.
Besides removing the element with the smallest priority, you must also be able to change a priority by key and remove by key. You may use a `dict` to record “where each key is in the heap”.

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

`push`, `pop`, `update` and `remove` must each make only $O(\log n)$ priority comparisons and $O(\log n)$ `dict` operations;
the other methods use a constant number of `dict` or list operations.

```text
q = IndexedPQ()
for key, p in [("a", 1), ("b", 12), ("c", 4), ("d", 18), ("e", 14), ("f", 13), ("g", 7)]:
    q.push(key, p)
q.remove("d")        -> 18
pop until empty      -> ("a", 1), ("c", 4), ("g", 7), ("b", 12), ("f", 13), ("e", 14)

q = IndexedPQ()
q.push("x", 5); q.push("y", 3); q.push("z", 9)
q.update("z", 1)                         # decrease
q.update("y", 7)                         # increase
q.peek()             -> ("z", 1)
q.priority("y")      -> 7
"x" in q             -> True
"w" in q             -> False
q.push("x", 0)       -> ValueError
q.update("w", 1)     -> KeyError
q.remove("w")        -> KeyError
pop until empty      -> ("z", 1), ("x", 5), ("y", 7)
q.pop()              -> IndexError
```

Also answer: after `remove` moves another element into the vacated position, should that element be sifted up or sifted down?

### Part 3 — Using it in Dijkstra

A directed graph has $V$ nodes numbered `0..V-1`, with $V \ge 1$. `adj[u]` is the list of edges leaving `u`; each edge is `(v, w)`,
where `w` is a non-negative integer. Parallel edges and self-loops are allowed. Compute the shortest distance from `source` to every node; the distance to an unreachable node is `math.inf`.

Write it two ways:

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
# 0 -> 2 -> 1 has length 2 + 3 = 5, shorter than going 0 -> 1 directly (7); node 4 is unreachable
```

Then compare the two approaches: at most how many elements are in the heap at the same time? What is the time complexity of each? When is each the better fit?

## Reference solution

<details>
<summary>Show the reference solution</summary>

### Part 1

A binary heap is a complete binary tree stored in an array: the two children of index `i` are `2i + 1` and `2i + 2`, and its parent is
`(i - 1) // 2`. There is only one invariant: **no element is smaller than its parent**. So the smallest element is at index 0.

Two basic moves:

- Sift up: if an element is smaller than its parent, swap it with the parent, until it is no smaller than its parent or reaches the root.
- Sift down: pick the smaller of the two children, and if it is smaller than the current element, swap; repeat until neither child is smaller
  or there are no children. You must pick the **smaller** child; otherwise the child that moves up would be larger than its sibling, breaking the invariant again.

`push` puts the new element at the end and sifts it up. `pop` takes the root, moves the last element to the root and sifts it down; removing the last element does not leave
a hole in the middle of the tree.

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

**Number of comparisons.** In a heap with $n$ elements, the deepest node has depth $\lfloor \log_2 n \rfloor$.

- `push`: sifting up makes one comparison per level, at most $\lfloor \log_2 n \rfloor$ (where $n$ is the size after insertion).
- `pop`: sifting down makes two comparisons per level (first between the two children, then between the smaller child and the current element), at most
  $2 \lfloor \log_2 n \rfloor$.

**Why building the heap is $O(n)$.** The constructor starts from the last node that has children and sifts down each node, going backwards. When it reaches a
node, both of its subtrees are already heaps, and after sifting down, the subtree rooted at that node is a heap too. The cost of sifting a node down is proportional to its **height**
(its distance to the bottom level), not its depth. The vast majority of nodes in the tree are near the bottom: there are roughly $n / 2^{h+1}$ nodes of height $h$,
each making at most $2h$ comparisons, so the total is roughly

$$\sum_{h \ge 1} 2h \cdot \frac{n}{2^{h+1}} = n \sum_{h \ge 1} \frac{h}{2^h} = 2n .$$

So building the heap is $O(n)$ (on every input tried in the checks below, the number of comparisons never exceeds $2n$). Pushing one at a time is the opposite: the cost is proportional to
**depth**, and half the nodes are on the bottom level, at depth about $\log_2 n$. When the input is in descending order, every new element has to float all the way up to the root;
the $i$-th element (counting from 1) makes exactly $\lfloor \log_2 i \rfloor$ comparisons, about $n \log_2 n$ in total.

### Part 2

A heap can only find the smallest element quickly; it cannot find “at which index of the array key `k` currently sits”. So we also maintain a position table `_pos`:
key → its index in the heap array. With the index in hand, changing or removing an element is just one local adjustment.

The key to the implementation is **keeping the position table consistent with the array at all times**. The way to do that is to funnel every “element move” through one function, `_swap`: elements in the heap
change position only through `_swap`, and `_swap` updates the positions of both keys at the same time. Sift up and sift down both call it, so nothing gets missed.

- `update`: after changing the priority, the new priority may be smaller than the parent's (needs sifting up) or larger than a child's (needs sifting down).
  Sift up, then sift down; at most one of the two will actually move anything.
- `remove`: swap the element to be removed with the **last** element, pop the last element, then adjust the element that was swapped in.
- `pop`: just `remove` the key at the root.

**Which direction to adjust after `remove`: either is possible.** During `pop`, the element swapped onto the root can only sift down, which makes it tempting to assume
`remove` works the same way. But the last element and the vacated position are not necessarily on the same root-to-leaf path, so it may be smaller than the parent of the vacated position.
The first example in the statement is exactly this case: after the pushes, the array is `[1, 12, 4, 18, 14, 13, 7]`; when 18
(index 3) is removed, the last element, 7, is swapped into index 3, whose parent is 12. If you only sift down, 7 stays below 12, and later 12 is popped
before 7.

The heap stores `(priority, key)` and compares **only the priority** `entry[0]`, never the whole tuple: tuple comparison, on equal priorities, would go on
to compare the keys, and keys are not necessarily comparable.

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

**Complexity.** `push` makes at most $\lfloor \log_2 n \rfloor$ priority comparisons (where $n$ is the size after insertion); `pop`, `update` and
`remove` make at most $2 \lfloor \log_2 n \rfloor + 2$ (where $n$ is the size before the operation). Each swap also does two `dict` writes,
and each public method starts with one `dict` lookup. `dict` operations are expected $O(1)$, so the precise statement is: worst-case number of comparisons
$O(\log n)$, expected total time $O(\log n)$. If the keys are integers `0..n-1` to begin with, the position table can be replaced by a list,
and then the total time is worst-case $O(\log n)$.

### Part 3

At each step, Dijkstra removes the unsettled node `u` with the smallest current distance; its distance is final at that point (non-negative edge weights guarantee this).
It then relaxes `u`'s neighbors using its outgoing edges: if `dist[u] + w < dist[v]`, lower `dist[v]`. The catch is that
`v` may already be in the queue, carrying an old, larger distance. There are two ways to handle this:

**The indexed version**: if `v` is already in the queue, `update` it to the new distance (this operation is called decrease-key); otherwise `push` it.
The queue holds at most one entry per node.

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

**The lazy-deletion version**: leave the old entry alone and simply `push` another `(new distance, v)`. The old entry stays in the heap; when it is eventually popped,
its distance turns out to be larger than `dist[v]`, which means it is stale, and it is skipped.

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

**Comparison.** Let $V$ be the number of nodes and $E$ the number of edges.

| | Indexed (decrease-key) | Lazy deletion |
| --- | --- | --- |
| Maximum number of elements in the heap | $V$ | At most $E$ (each successful relaxation adds one; the source's entry is popped before the first relaxation) |
| Number of heap operations | At most $V$ `push`es, $V$ `pop`s, $E$ `update`s | At most $E + 1$ `push`es, $E + 1$ `pop`s |
| Time | $O((V + E) \log V)$ | $O(E \log E)$ |
| Extra code | A position table, maintained on every swap | One staleness check |

Three key points:

- **The asymptotic time is actually the same.** Without parallel edges, $E \le V^2$, so $\log E \le 2 \log V$, and $O(E \log E)$ is just
  $O(E \log V)$. Decrease-key on a binary heap itself costs $O(\log V)$, so it is no cheaper than “pushing another entry”. The textbook bound of
  $O(E + V \log V)$ requires a heap with amortized $O(1)$ decrease-key (such as a Fibonacci heap), which a binary heap cannot provide.
- **The real difference is space and stale entries.** The indexed version holds at most $V$ entries in the heap; lazy deletion makes $E + 1$ pushes in total, with at most $E$ entries at any one time. The worst case is
  a graph like this: for every pair $u < v$ there is an edge `u -> v` with weight $(v - u)^2$. The shortest path goes one step at a time,
  `dist[v] = v`; but when the nodes `u` are processed in increasing order, every edge lowers `dist[v]` once more, so lazy deletion makes
  $E + 1$ pushes. On graphs with random weights, however, successful relaxations are far fewer than $E$, and there are not many stale entries.
- **How to choose.** If you are just running shortest paths once and memory is not a concern, lazy deletion is simpler and can use an off-the-shelf heap directly. If the graph is dense and memory matters,
  or the same queue also needs arbitrary `update` / `remove` (for example, task scheduling with cancellation, or timers), use the indexed version.

Both versions depend on non-negative edge weights: only then is a popped node's distance final, and only then is each node processed exactly once.

### Follow-ups

**Q: How would you change it if elements with equal priority must be popped in the order they entered the queue?** A binary heap is not stable on its own. Give each entry an
increasing sequence number and compare `(priority, seq)`: equal priorities fall back to the sequence number, and sequence numbers are all distinct, so the keys are never compared.
Whether `update` keeps the old sequence number or issues a new one depends on whether “changing the priority counts as re-queueing”; confirm that with the interviewer.

**Q: Does the position table have to be a `dict`?** No. When the keys are integers `0..n-1` (such as node numbers in Dijkstra),
use a list of length $n$, with `-1` meaning “not in the queue”. That avoids hashing, makes every bound a worst-case bound, and has smaller constants too.

**Q: What do you gain by switching to a d-ary heap?** Each node has $d$ children, and the tree height is $\log_d n$. Sift up makes one comparison per level, so it gets faster;
sift down has to find the smallest of $d$ children at each level, so it gets slower. In Dijkstra there are up to $E$ `update`s (sift up) but
only $V$ `pop`s (sift down); on dense graphs the former far outnumber the latter, so choosing $d \approx E / V$ brings the total time down to
$O(E \log_{E/V} V)$.

**Q: What happens to these two pieces of code when edge weights can be negative?** “A popped node's distance is already final” no longer holds. Both pieces of code above allow
a node to re-enter the queue after being popped if its distance drops again, so as long as there is no negative cycle, the final distances are still correct (the
checks below verify this on acyclic graphs with negative weights); but the same node may be processed over and over, all the complexity analysis above goes out the window,
and the number of operations is known to be exponential in the worst case. If a negative cycle is reachable from the source, the loop never terminates. With negative weights you should use Bellman–Ford.

<details>
<summary>Checks (runnable)</summary>

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
