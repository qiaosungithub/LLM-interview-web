# expect: TLE
# Part 2 keeps a (size, start) index and takes the lowest start among all blocks big enough: O(n) per allocate.
import random


class MemoryAllocator:
    """Part 1: free blocks in two dicts (by start and by end); first fit scans the sorted starts."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.by_start = {0: capacity}
        self.by_end = {capacity: 0}
        self.used = {}

    def allocate(self, size: int) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        for s in sorted(self.by_start):
            n = self.by_start[s]
            if n >= size:
                del self.by_start[s]
                del self.by_end[s + n]
                if n > size:
                    self.by_start[s + size] = n - size
                    self.by_end[s + n] = s + size
                self.used[s] = size
                return s
        raise MemoryError("no block large enough")

    def free(self, address: int, size: int) -> None:
        if size <= 0 or self.used.get(address) != size:
            raise ValueError("bad free")
        del self.used[address]
        lo, hi = address, address + size
        if lo in self.by_end:
            lo = self.by_end.pop(lo)
            del self.by_start[lo]
        if hi in self.by_start:
            n = self.by_start.pop(hi)
            del self.by_end[hi + n]
            hi = hi + n
        self.by_start[lo] = hi - lo
        self.by_end[hi] = lo



import bisect


class MemoryAllocatorLog:
    """Free blocks indexed by start (sorted list) and by (size, start); first fit = lowest start among sizes >= s."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError
        self.starts = [0]
        self.size_of = {0: capacity}
        self.by_size = [(capacity, 0)]
        self.used = {}

    def _add(self, s, n):
        bisect.insort(self.starts, s)
        self.size_of[s] = n
        bisect.insort(self.by_size, (n, s))

    def _remove(self, s):
        n = self.size_of.pop(s)
        self.starts.pop(bisect.bisect_left(self.starts, s))
        self.by_size.pop(bisect.bisect_left(self.by_size, (n, s)))
        return n

    def allocate(self, size):
        if size <= 0:
            raise ValueError
        i = bisect.bisect_left(self.by_size, (size, -1))
        if i == len(self.by_size):
            raise MemoryError
        s = min(st for _n, st in self.by_size[i:])
        n = self._remove(s)
        if n > size:
            self._add(s + size, n - size)
        self.used[s] = size
        return s

    def free(self, address, size):
        if size <= 0 or self.used.get(address) != size:
            raise ValueError
        del self.used[address]
        lo, hi = address, address + size
        i = bisect.bisect_left(self.starts, address)
        if i > 0:
            p = self.starts[i - 1]
            if p + self.size_of[p] == lo:
                lo = p
                self._remove(p)
        if hi in self.size_of:
            hi += self._remove(hi)
        self._add(lo, hi - lo)
