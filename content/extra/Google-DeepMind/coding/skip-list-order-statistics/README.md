# Skip List: a Sorted Multiset with Order Statistics

## Problem

Implement a *sorted multiset*: it stores integers, allows duplicates, supports insertion and deletion at any
time, and can be read out in ascending order. It must be implemented as a *skip list*. You may not use "a
Python list plus `bisect`", and you may not use a ready-made sorted container.

A skip list is structured as follows. At the bottom, level 0 is a singly linked list that strings all the
nodes together in ascending key order. Each node gets a random *height* `h` when it is inserted: start with
`h = 1`, then flip a coin repeatedly, each time increasing `h` by 1 with probability `p` (default `1/2`), and
stop at the first miss; `h` is also capped at `MAX_LEVEL = 32`. A node of height `h` appears on levels
`0, 1, ..., h-1`, and every level is a linked list sorted by key. On the far left there is a head node `head`
that stores no key and appears on every level.

```text
Keys 3, 7, 12, 19, 25, 31 with heights 1, 3, 1, 2, 1, 1:

Level 2   head ----------> 7 -----------------------------------> None
Level 1   head ----------> 7 -----------> 19 -------------------> None
Level 0   head ---> 3 ---> 7 ---> 12 ---> 19 ---> 25 ---> 31 ---> None
```

A search starts at the top level of `head`: as long as the key of the node to the right is smaller than the
target, move right; otherwise drop down a level. When you can't move any further on level 0, the node to the
right is the first node whose key is not less than the target. To find 25 in the picture above: on level 2,
go from `head` to 7; on level 1, go from 7 to 19; on level 0, don't move; the node to the right of 19 is 25.
That is 2 steps to the right in total; walking along level 0 alone takes 4 steps.

Sizes and requirements:

- Keys are integers in `[-10**9, 10**9]`, and there are at most `2 * 10**5` operations in total.
- Let `n` be the current number of elements. Every operation must run in *expected* $O(\log n)$ time. The
  expectation is over the structure's own coin flips, and it must hold for every input order, including
  inserting in ascending order.
- The constructor's `seed` initializes the structure's own random number generator `random.Random(seed)`. It
  only affects the heights of the internal nodes and **must not affect the return value of any operation**.
  `p` satisfies `0 < p < 1`.

### Part 1 — Insertion and lookup

```py
class SkipList:
    def __init__(self, seed: int | None = None, p: float = 0.5) -> None: ...
    def add(self, key: int) -> None: ...            # duplicates are kept
    def __contains__(self, key: int) -> bool: ...   # `key in sl`
    def __len__(self) -> int: ...                   # number of stored keys, duplicates counted
    def __iter__(self): ...                         # keys in ascending order, duplicates included
```

All duplicate keys are kept: after `add(10)` twice, the set contains two 10s. An empty set has length 0,
iterating over it yields an empty sequence, and no key is in it.

```text
sl = SkipList(seed=7)
add 30, 10, 20, 10 in that order

list(sl) == [10, 10, 20, 30]
len(sl) == 4
(20 in sl) is True
(25 in sl) is False
```

### Part 2 — Deletion and range queries

```py
    def remove(self, key: int) -> bool: ...                      # remove ONE occurrence
    def keys_between(self, lo: int, hi: int) -> list[int]: ...   # every key with lo <= key <= hi
```

`remove(key)` removes one element equal to `key` and returns `True`; when there are several equal elements,
it removes only one. If the key isn't in the set, it returns `False` and leaves the set unchanged.

`keys_between(lo, hi)` returns, in ascending order, every key with `lo <= key <= hi`, both ends inclusive; a
duplicated key is returned as many times as it occurs. When `lo > hi` it returns an empty list. If it returns
`m` keys, the expected time must be $O(\log n + m)$.

```text
Continuing the example from Part 1, the set is now [10, 10, 20, 30]

sl.remove(10) is True            # now [10, 20, 30]
sl.remove(15) is False           # there is no 15, the set is unchanged
sl.keys_between(10, 25) == [10, 20]
sl.keys_between(20, 20) == [20]
sl.keys_between(31, 99) == []
sl.keys_between(25, 10) == []    # lo > hi
```

### Part 3 — Order-statistic queries

```py
class IndexedSkipList(SkipList):
    def kth(self, i: int) -> int: ...       # the i-th smallest key, counting from 0
    def rank(self, key: int) -> int: ...    # how many stored keys are strictly smaller than key
```

Add two operations to the structure from the first two parts (you can modify the original class directly, or
write a subclass as above).

`kth(i)` returns the key at position `i` in ascending order, counting from 0, i.e. `list(sl)[i]`. If `i` is
not in `[0, len)`, raise `IndexError`; negative indices are also out of range, with no Python-style counting
from the end.

`rank(key)` returns the number of elements in the set strictly smaller than `key`, counting each duplicate
separately. `key` need not be in the set.

Both operations must run in expected $O(\log n)$ time, and `add` and `remove` must stay expected
$O(\log n)$ as well. Counting nodes one by one along level 0 is $O(n)$ and does not meet the requirement.

```text
isl = IndexedSkipList(seed=7)
add 30, 10, 20, 10, 20, 5 in that order   # now [5, 10, 10, 20, 20, 30]
isl.remove(10) is True                    # now [5, 10, 20, 20, 30]

isl.kth(0) == 5      isl.kth(1) == 10      isl.kth(3) == 20      isl.kth(4) == 30
isl.kth(5)  -> IndexError
isl.kth(-1) -> IndexError

isl.rank(20) == 2        # the keys smaller than 20 are 5 and 10
isl.rank(21) == 4
isl.rank(5) == 0
isl.rank(1) == 0
isl.rank(99) == 5
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Before you start, clarify three things. How duplicate keys are handled (all are kept, and `remove` deletes
only one: this decides which node a deletion should unlink). Whether the complexity is worst-case or expected
(expected, and over the structure's own random numbers, not over the input). What `seed` controls (only the
internal shape; return values must not change with it: this is a property you can test directly).

A reasonable pace for 45 minutes: get Parts 1 and 2 written and running, and for Part 3 first explain what
extra data to store and how updates change it, then write the code.

### Part 1

**Data layout.** A node stores one key and an array `next` whose length equals its height; `next[i]` is its
successor on level `i`. `head` has height `MAX_LEVEL`. We also keep `_level`: the number of levels actually in
use, which is 1 for an empty list.

**Invariants.** The following three hold at the end of every operation; the check code verifies them from
scratch after every operation:

1. Level 0 strings all the nodes together in ascending key order (equal keys next to each other).
2. Level `i` consists of exactly the nodes taller than `i`, in the same order as on level 0.
3. `_level` equals the height of the tallest node (1 for an empty list), and `head`'s pointers on level
   `_level` and above are all `None`.

**Core subroutine.** `_predecessors(key)` finds `update[i]` for each level `i`: the last node on level `i`
whose key is **strictly less than** `key`, or `head` if there is none. The method is exactly the search from
the statement: start at the top level, move right whenever you can, and when you can't, record the current
node as this level's answer, drop down a level, and continue from the same node. Continuing from the same
node works because its key is less than `key` and it is also on the next level down, so the answer on that
level can't be to its left. When the walk ends, `update[0].next[0]` is the first node whose key is not less
than `key`; lookup, insertion, deletion and range queries all start from there.

**Insertion.** Compute `update`, draw a height `h`, then on levels `0 .. h-1` link the new node in after
`update[i]`, which is ordinary linked-list insertion. When `h` exceeds `_level`, the predecessor on the newly
used levels is `head`, which is what the `update` array is initialized to. Because the comparison is strict
less-than, the new node is placed to the left of all existing equal keys. For a multiset any position would be
correct, but deletion in Part 2 relies on "the node we operate on is always the leftmost of the equal keys".

**Why the cost is expected $O(\log n)$.** Let $L = \log_{1/p} n$.

- Number of pointers: a node has height at least $k$ with probability $p^{k-1}$, so the expected height is
  $1 + p + p^2 + \dots = 1/(1-p)$. With $p = 1/2$ that is 2 pointers per node on average.
- Number of levels: the expected number of nodes taller than $L + c$ is $n \cdot p^{L+c} = p^c$. So the
  probability of more than $L + c$ levels is at most $p^c$, and the expected number of levels is
  $L + O(1)$.
- Search steps: look at the search path backwards. Walking back from the end, when you are standing on some
  node on level $i$, if its height is greater than $i + 1$, the path must have come down from the level above,
  so going backwards means "climb one level"; otherwise the path came from the left, so going backwards means
  "step one to the left". Whether its height exceeds $i+1$ is decided by one independent coin, with probability
  $p$. So each level climbed takes $1/p$ steps on average, and reaching level $L$ takes $L/p$ steps on
  average; above level $L$ there are only a constant number of nodes in expectation. Pugh's original paper
  gives the result: the expected path length is at most $L/p + 1/(1-p)$, plus 1 for the number of
  comparisons.

With $p = 1/2$ and $n = 8192$ this bound is $2 \times 13 + 2 + 1 = 29$ comparisons. The check code inserts
8192 keys in ascending order (the order that turns an unbalanced binary search tree into a linked list) and
does 2000 searches for each of 8 seeds: the mean number of comparisons is about 27, and the per-seed means
range from 24 to 29. Note that the bound is on the expectation; one unlucky draw can exceed it.

`MAX_LEVEL = 32` is more than enough for this problem's sizes: with $p = 1/2$ a node draws 32 with
probability $2^{-31}$, so over $2 \times 10^5$ insertions the probability that any node hits the cap is at
most $2 \times 10^5 \cdot 2^{-31} < 10^{-4}$. Even if one does, only the height is truncated, and the structure
is still correct.

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

**Deletion.** Compute `update` and let `target = update[0].next[0]`. If it doesn't exist or its key isn't
`key`, the key is not in the set; return `False`. Otherwise `target` has to be unlinked from every level it
is on.

The key point: for every `i < height of target`, `update[i].next[i] is target`. The reason: `target` is the
first node in level-0 order whose key is not less than `key`; `update[i]` is the last node on level `i` whose
key is less than `key`, so its successor on level `i` is "the first node on level `i` whose key is not less
than `key`"; `target` is on level `i` and comes before all such nodes, so that successor is `target`. So
unlinking on each level is the single statement `update[i].next[i] = target.next[i]`.

This is also the easiest place to go wrong when there are duplicate keys. If the search stops as soon as it
sees an equal key on a high level, it may get a tall equal node that is not the leftmost one; the `update[i]`
on lower levels then points to another equal node to its left, and rewiring the pointers with the statement
above drops the nodes between them from the list as well. The check code has a deliberately broken version
written this way, and the random tests catch it.

After unlinking, if the top level has become empty, lower `_level` to keep invariant 3.

**Range query.** `_predecessors(lo)[0].next[0]` is the first node whose key is not less than `lo`; from there,
walk right along level 0 until the key exceeds `hi`. Locating the start is expected $O(\log n)$, and after that
each output key costs one step, for expected $O(\log n + m)$ in total. When `lo > hi`, the first node's key is
already greater than `hi`, the loop doesn't run at all, and an empty list is returned, with no special case
needed.

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

**What extra data to store.** Give every pointer a *span*: how many level-0 positions that pointer jumps over
in one step. Positions are numbered like this: `head` is 0, the `i`-th smallest key (counting from 0) is
`i + 1`, and the `None` at the end is `n + 1`. The new invariant is:

4. For every node `x` (including `head`) and every level `i` it uses, `x.span[i]` equals the position of
   `x.next[i]` minus the position of `x`.

Counting `None` as a position (`n + 1`) gives pointers to the end a well-defined span too, so insertion,
deletion and queries need no special case for "there are no more nodes".

```text
Position  0         1      2      3       4       5       6       7
Level 2   head ----------> 7 -----------------------------------> None      spans 2, 5
Level 1   head ----------> 7 -----------> 19 -------------------> None      spans 2, 2, 3
Level 0   head ---> 3 ---> 7 ---> 12 ---> 19 ---> 25 ---> 31 ---> None      all spans are 1
```

**Queries.** `kth(i)` has to reach position `want = i + 1`. Start at the top level and keep the current
position in `at`; as long as `at + span <= want`, move right and add the span to `at`; otherwise drop down a
level. Since the position of `None`, `n + 1`, is greater than `want`, the loop never runs off the end of the
list. All spans on level 0 are 1, so at the end `at == want`. For `kth(4)` in the picture above: `want = 5`;
level 2 moves to 7 (`at = 2`), level 1 moves to 19 (`at = 4`), level 0 moves to 25 (`at = 5`); return 25.

`rank(key)` is an ordinary search that adds up the spans it crosses along the way. The node it stops at is the
last node whose key is less than `key`, and its position is exactly "the number of keys strictly less than
`key`".

**Updates.** During the search, also record the position `pos[i]` of each level's predecessor. The new node's
position is `at = pos[0] + 1`.

- On levels `i < h`: the pointer `update[i] → successor` is cut into two by the new node. The span from
  `update[i]` to the new node is `at - pos[i]`. The old successor now has one more node in front of it, so its
  position becomes `pos[i] + old span + 1`, and the span from the new node to it is
  `pos[i] + old span + 1 - at`.
- On levels `h <= i < _level`: the new node is not on this level, but it falls under the pointer from
  `update[i]`, so that pointer's span increases by 1. **This step is the easiest one to forget.**
- On newly used levels: first set `head`'s span on these levels to `n + 1` (pointing to `None`), then handle
  them as in the first case.

Deletion is the reverse: on the levels where `target` appears, the predecessor's span becomes "the sum of the
two pieces minus 1"; on the levels above `target`, the span decreases by 1.

**Cost.** `kth` and `rank` follow a path of the same shape as an ordinary search: expected $O(\log n)$. `add`
and `remove` now have to update every level in use, not just the `h` levels the new node is on; the expected
number of levels is $O(\log n)$, so they are still expected $O(\log n)$. Memory-wise, each pointer gets one
extra integer.

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

### Follow-ups

- **When is "a sorted array plus binary search" good enough?** More often than you'd think. If the set never
  changes after it is built, a sorted array wins across the board: `kth` is a direct index, `rank` is one
  binary search, and memory is contiguous. With insertions and deletions, `bisect.insort` locates the spot in
  $O(\log n)$ and shifts elements in $O(n)$, but the shift is a single `memmove` with a tiny constant. One
  measurement on CPython on this machine (the numbers will differ on other machines): inserting 20,000 random
  keys, `insort` is about twice as fast as the skip list above; the two break even at around 60,000; the
  version with spans is slower still, and at this problem's limit of 200,000 keys it is still no faster than
  `insort`. So in Python and at this scale, in an interview you should first say "I would use a sorted
  array", and then explain when it stops being enough: larger $n$ with frequent updates (the total cost of
  all `insort` insertions is $O(n^2)$), a language where pointer operations are cheap, or multiple threads
  modifying it concurrently.
- **What is the expectation in "expected $O(\log n)$" taken over? What about the worst case?** It is over the
  structure's own coins, with no assumption about the input, so inserting in ascending order does not
  degenerate the way an unbalanced binary search tree does. The worst case is still $O(n)$: if every node draws
  height 1, it is a linked list, just with tiny probability. The probability of more than $\log_2 n + c$
  levels is at most $2^{-c}$; the check code counts the case $n = 1024$, $c = 4$ over 400 seeds, and about 6%
  of seeds exceed it, against a bound of 6.25%. The real weakness is an adversary who knows the random numbers:
  someone who can see the seed or observe the node heights can deliberately delete the tall nodes and make
  the structure degenerate. In that setting, use a deterministic balanced tree.
- **How do you choose `p`?** It is a knob between memory and search time. The expected number of pointers per
  node is $1/(1-p)$: 2 for $p = 1/2$ and 1.33 for $p = 1/4$. The leading term of the search steps is
  $(1/p)\log_{1/p} n$: $2\log_2 n$ for $p = 1/2$ and $4 \log_4 n = 2\log_2 n$ for $p = 1/4$, the same. So
  $p = 1/4$ saves a third of the pointers without slowing searches down (the check code measures mean
  comparison counts of about 27 and 24.5 at $n = 8192$), at the cost of more variance in the number of steps.
  With spans, every pointer also carries an extra integer, which makes a smaller `p` even more worthwhile.
  Redis's sorted sets use a skip list with spans, with `p` set to 1/4.
- **What do updates pay for the spans?** Without spans, an insertion only touches the `h` levels the new node
  is on, 2 on average; with spans it has to update every level in use, about $\log_{1/p} n$ of them, and
  memory goes from about 120 bytes per key to about 200 (CPython, $p = 1/2$; in a sorted array each key takes
  only one 8-byte pointer). Measured insertion is almost twice as slow. So only add spans when you actually
  need ranks. With `rank`, "how many keys are in the range" doesn't require listing the keys either: for
  integer keys the answer is `max(0, rank(hi + 1) - rank(lo))`, expected $O(\log n)$ regardless of how many keys
  are in the range. If every possible key is known in advance (offline, or a small value range), a Fenwick tree
  over the key range can also answer rank queries, and is shorter.
- **Compared with a balanced binary search tree, why would anyone choose a skip list?** Both support the same
  set of operations at the same order of cost; the differences are in the guarantees and how hard they are to
  implement. Red-black trees and AVL trees give worst-case $O(\log n)$; a skip list gives an expectation. A skip
  list has no rotations; insertion and deletion only change a few predecessor pointers, so the code is much
  shorter and can be written in 45 minutes; level 0 is itself a sorted linked list, so range scans need no
  in-order traversal. Because modifications are local, it is also relatively easy to make concurrent (Java's
  `ConcurrentSkipListMap` is one). The cost is more pointers and scattered nodes, so its cache behavior is
  worse than arrays and B-trees.

<details>
<summary>Checks (runnable)</summary>

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
