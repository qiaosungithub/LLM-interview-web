# expect: RE
# The height loop has no MAX_LEVEL cap while the head has exactly 32 levels; with a large p a node gets taller than the head.
import random

MAX_LEVEL = 32


class Node:
    __slots__ = ("key", "forward", "span")

    def __init__(self, key, h):
        self.key = key
        self.forward = [None] * h
        self.span = [0] * h


class SkipList:
    def __init__(self, seed=None, p=0.5):
        self.rng = random.Random(seed)
        self.p = p
        self.head = Node(None, MAX_LEVEL)
        self.level = 1
        self.n = 0

    def _rand_level(self):
        lvl = 1
        while self.rng.random() < self.p:
            lvl += 1
        return lvl

    def _search(self, key):
        update = [self.head] * MAX_LEVEL
        rank = [0] * MAX_LEVEL
        x = self.head
        r = 0
        for i in reversed(range(self.level)):
            while x.forward[i] is not None and x.forward[i].key < key:
                r += x.span[i]
                x = x.forward[i]
            update[i] = x
            rank[i] = r
        return update, rank

    def add(self, key):
        update, rank = self._search(key)
        lvl = self._rand_level()
        if lvl > self.level:
            for i in range(self.level, lvl):
                update[i] = self.head
                rank[i] = 0
                self.head.span[i] = self.n + 1
            self.level = lvl
        x = Node(key, lvl)
        for i in range(lvl):
            x.forward[i] = update[i].forward[i]
            update[i].forward[i] = x
            x.span[i] = update[i].span[i] - (rank[0] - rank[i])
            update[i].span[i] = rank[0] - rank[i] + 1
        for i in range(lvl, self.level):
            update[i].span[i] += 1
        self.n += 1

    def __contains__(self, key):
        update, _ = self._search(key)
        x = update[0].forward[0]
        return x is not None and x.key == key

    def __len__(self):
        return self.n

    def __iter__(self):
        x = self.head.forward[0]
        while x:
            yield x.key
            x = x.forward[0]

    def remove(self, key):
        update, _ = self._search(key)
        x = update[0].forward[0]
        if x is None or x.key != key:
            return False
        for i in range(self.level):
            if update[i].forward[i] is x:
                update[i].span[i] += x.span[i] - 1
                update[i].forward[i] = x.forward[i]
            else:
                update[i].span[i] -= 1
        while self.level > 1 and self.head.forward[self.level - 1] is None:
            self.level -= 1
        self.n -= 1
        return True

    def keys_between(self, lo, hi):
        res = []
        if lo > hi:
            return res
        update, _ = self._search(lo)
        x = update[0].forward[0]
        while x is not None and x.key <= hi:
            res.append(x.key)
            x = x.forward[0]
        return res


class IndexedSkipList(SkipList):
    def kth(self, i):
        if i < 0 or i >= self.n:
            raise IndexError("kth out of range")
        want = i + 1
        x, at = self.head, 0
        for lv in reversed(range(self.level)):
            while x.forward[lv] is not None and at + x.span[lv] <= want:
                at += x.span[lv]
                x = x.forward[lv]
        return x.key

    def rank(self, key):
        return self._search(key)[1][0]
