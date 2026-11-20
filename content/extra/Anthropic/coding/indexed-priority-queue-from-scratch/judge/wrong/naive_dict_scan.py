# expect: TLE
# No heap: a dict whose pop scans every key, O(n) per pop.
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
    """A dict from key to priority: pop and peek scan every key."""

    def __init__(self) -> None:
        self._p = {}

    def __len__(self) -> int:
        return len(self._p)

    def __contains__(self, key) -> bool:
        return key in self._p

    def priority(self, key):
        return self._p[key]

    def _best(self):
        if not self._p:
            raise IndexError("empty queue")
        best = None
        for key, p in self._p.items():
            if best is None or p < best[1]:
                best = (key, p)
        return best

    def peek(self):
        return self._best()

    def push(self, key, priority) -> None:
        if key in self._p:
            raise ValueError(key)
        self._p[key] = priority

    def update(self, key, priority) -> None:
        if key not in self._p:
            raise KeyError(key)
        self._p[key] = priority

    def remove(self, key):
        return self._p.pop(key)

    def pop(self):
        key, p = self._best()
        del self._p[key]
        return key, p


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
