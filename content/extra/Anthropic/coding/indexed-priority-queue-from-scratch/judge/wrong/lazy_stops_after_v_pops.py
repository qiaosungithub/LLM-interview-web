# expect: WA
# dijkstra_lazy stops after V pops, but stale pops count too, so it can stop before every node is settled.
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
    """Entries are [priority, key, index] lists; the dict maps key -> entry, so moving an entry
    never touches the dict."""

    def __init__(self):
        self.h = []
        self.at = {}

    def __len__(self):
        return len(self.h)

    def __contains__(self, key):
        return key in self.at

    def priority(self, key):
        return self.at[key][0]

    def peek(self):
        if not self.h:
            raise IndexError("empty")
        e = self.h[0]
        return (e[1], e[0])

    def push(self, key, priority):
        if key in self.at:
            raise ValueError("duplicate key")
        e = [priority, key, len(self.h)]
        self.h.append(e)
        self.at[key] = e
        self._up(e)

    def update(self, key, priority):
        e = self.at[key]
        old = e[0]
        e[0] = priority
        if priority < old:
            self._up(e)
        else:
            self._down(e)

    def remove(self, key):
        e = self.at.pop(key)
        h = self.h
        last = h.pop()
        if last is not e:
            last[2] = e[2]
            h[e[2]] = last
            i = last[2]
            if i > 0 and last[0] < h[(i - 1) >> 1][0]:
                self._up(last)
            else:
                self._down(last)
        return e[0]

    def pop(self):
        if not self.h:
            raise IndexError("empty")
        e = self.h[0]
        self.remove(e[1])
        return (e[1], e[0])

    def _up(self, e):
        h = self.h
        i = e[2]
        while i:
            p = (i - 1) >> 1
            q = h[p]
            if e[0] < q[0]:
                h[i] = q
                q[2] = i
                i = p
            else:
                break
        h[i] = e
        e[2] = i

    def _down(self, e):
        h = self.h
        n = len(h)
        i = e[2]
        while True:
            c = 2 * i + 1
            if c >= n:
                break
            if c + 1 < n and h[c + 1][0] < h[c][0]:
                c += 1
            q = h[c]
            if q[0] < e[0]:
                h[i] = q
                q[2] = i
                i = c
            else:
                break
        h[i] = e
        e[2] = i


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
    popped = 0
    while len(h) and popped < n:       # every node is settled after n pops
        d, u = h.pop()
        popped += 1
        if done[u]:
            continue
        done[u] = True
        for v, w in adj[u]:
            if not done[v] and d + w < dist[v]:
                dist[v] = d + w
                h.push((d + w, v))
    return dist
