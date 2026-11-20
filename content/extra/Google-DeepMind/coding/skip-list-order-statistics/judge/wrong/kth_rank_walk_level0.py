# expect: TLE
# Right answers, but kth and rank count along level 0 (the O(n) method the statement rules out).
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


# Part 2

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


# Part 3

class IndexedSkipList(SkipList):
    def kth(self, i: int):
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range for {self._n} keys")
        x = self._head.next[0]
        for _ in range(i):                # BUG: counts along level 0, O(n)
            x = x.next[0]
        return x.key

    def rank(self, key) -> int:
        count = 0
        x = self._head.next[0]
        while x is not None and x.key < key:     # BUG: counts along level 0, O(n)
            count += 1
            x = x.next[0]
        return count
