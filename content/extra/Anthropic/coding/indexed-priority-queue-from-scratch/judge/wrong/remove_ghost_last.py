# expect: WA
# remove deletes the key from the position map before pointing the moved last entry at the hole;
# when the removed key was itself the last entry, that puts it back, so it still looks queued.
import math


class MinHeap:
    """Hole-based sifting; pop uses Floyd's bottom-up descent (go to a leaf, then bubble back up)."""

    def __init__(self, items=()):
        self.a = list(items)
        for i in reversed(range(len(self.a) // 2)):
            self._down_floyd(i)

    def __len__(self):
        return len(self.a)

    def peek(self):
        if not self.a:
            raise IndexError("empty")
        return self.a[0]

    def push(self, item):
        a = self.a
        a.append(item)
        self._up(len(a) - 1, 0)

    def pop(self):
        a = self.a
        if not a:
            raise IndexError("empty")
        last = a.pop()
        if not a:
            return last
        top = a[0]
        a[0] = last
        self._down_floyd(0)
        return top

    def _up(self, i, stop):
        a = self.a
        x = a[i]
        while i > stop:
            p = (i - 1) >> 1
            if x < a[p]:
                a[i] = a[p]
                i = p
            else:
                break
        a[i] = x

    def _down_floyd(self, i):
        a = self.a
        n = len(a)
        start = i
        x = a[i]
        c = 2 * i + 1
        while c < n:
            if c + 1 < n and a[c + 1] < a[c]:
                c += 1
            a[i] = a[c]
            i = c
            c = 2 * i + 1
        a[i] = x
        self._up(i, start)


class IndexedPQ:
    def __init__(self):
        self._heap = []          # [priority, key]
        self._pos = {}

    def __len__(self):
        return len(self._heap)

    def __contains__(self, key):
        return key in self._pos

    def priority(self, key):
        return self._heap[self._pos[key]][0]

    def peek(self):
        if not self._heap:
            raise IndexError("empty")
        p, k = self._heap[0]
        return k, p

    def push(self, key, priority):
        if key in self._pos:
            raise ValueError("duplicate key")
        self._heap.append([priority, key])
        self._pos[key] = len(self._heap) - 1
        self._up(len(self._heap) - 1)

    def update(self, key, priority):
        i = self._pos[key]
        self._heap[i][0] = priority
        self._down(self._up(i))

    def remove(self, key):
        i = self._pos[key]
        prio = self._heap[i][0]
        del self._pos[key]
        last = self._heap[-1]
        self._heap[i] = last              # fill the hole with the last entry
        self._pos[last[1]] = i            # (when key itself was last, this puts it back)
        self._heap.pop()
        if i < len(self._heap):
            self._down(self._up(i))
        return prio

    def pop(self):
        if not self._heap:
            raise IndexError("empty")
        key = self._heap[0][1]
        return key, self.remove(key)

    def _move(self, i, j):
        h = self._heap
        h[i], h[j] = h[j], h[i]
        self._pos[h[i][1]] = i
        self._pos[h[j][1]] = j

    def _up(self, i):
        h = self._heap
        while i > 0 and h[i][0] < h[(i - 1) // 2][0]:
            self._move(i, (i - 1) // 2)
            i = (i - 1) // 2
        return i

    def _down(self, i):
        h = self._heap
        n = len(h)
        while True:
            c = 2 * i + 1
            if c >= n:
                return i
            if c + 1 < n and h[c + 1][0] < h[c][0]:
                c += 1
            if not h[c][0] < h[i][0]:
                return i
            self._move(i, c)
            i = c


def dijkstra_indexed(adj, source):
    n = len(adj)
    dist = [math.inf] * n
    dist[source] = 0
    pq = IndexedPQ()
    for v in range(n):                 # every node queued once up front
        pq.push(v, dist[v])
    while len(pq):
        u, d = pq.pop()
        if d == math.inf:
            break                      # the rest are unreachable
        for v, w in adj[u]:
            if v in pq and d + w < dist[v]:
                dist[v] = d + w
                pq.update(v, d + w)
    return dist


def dijkstra_lazy(adj, source):
    n = len(adj)
    dist = [math.inf] * n
    dist[source] = 0
    done = [False] * n
    h = MinHeap([(0, source)])
    while len(h):
        d, u = h.pop()
        if done[u]:
            continue
        done[u] = True
        for v, w in adj[u]:
            if not done[v] and d + w < dist[v]:
                dist[v] = d + w
                h.push((d + w, v))
    return dist
