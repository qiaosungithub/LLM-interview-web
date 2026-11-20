# expect: WA
# remove only sifts the moved entry down, but it may have to go up.
import math


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
            self._sift_down(i)
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
