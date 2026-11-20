# expect: WA
# Wrong for Part 1: lists each distinct timestamp once (dict.fromkeys), but a message seen twice at the same
# timestamp occurred twice, so both timestamps belong in the list. Passes the sample.
import heapq
from collections import deque


def most_frequent(events):
    if len(events) == 0:
        raise ValueError("empty")
    stamps = {}                      # insertion order == first-occurrence order
    for msg, ts in events:
        if msg not in stamps:
            stamps[msg] = []
        stamps[msg].append(ts)
    best = None
    for msg, ts_list in stamps.items():
        if best is None or len(ts_list) > len(stamps[best]):
            best = msg
    return (best, list(dict.fromkeys(stamps[best])))   # drops repeated timestamps


class _Bucket:
    __slots__ = ("head", "tail", "size")

    def __init__(self):
        self.head = self.tail = None
        self.size = 0


class WindowTopMessage:
    """Per-count doubly linked lists kept in dicts (prev/next pointers by message)."""

    def __init__(self, n):
        self.n = n
        self.q = deque()
        self.cnt = {}
        self.prv = {}
        self.nxt = {}
        self.buckets = {}
        self.mx = 0

    def _unlink(self, msg, c):
        b = self.buckets[c]
        p, q = self.prv.pop(msg), self.nxt.pop(msg)
        if p is None:
            b.head = q
        else:
            self.nxt[p] = q
        if q is None:
            b.tail = p
        else:
            self.prv[q] = p
        b.size -= 1
        if b.size == 0:
            del self.buckets[c]

    def _append(self, msg, c):
        b = self.buckets.get(c)
        if b is None:
            b = self.buckets[c] = _Bucket()
        self.prv[msg] = b.tail
        self.nxt[msg] = None
        if b.tail is None:
            b.head = msg
        else:
            self.nxt[b.tail] = msg
        b.tail = msg
        b.size += 1

    def add(self, message, timestamp):
        if len(self.q) >= self.n:
            old = self.q.popleft()
            c = self.cnt[old]
            self._unlink(old, c)
            if c == 1:
                del self.cnt[old]
            else:
                self.cnt[old] = c - 1
                self._append(old, c - 1)
            if c == self.mx and c not in self.buckets:
                self.mx = c - 1
        self.q.append(message)
        c = self.cnt.get(message, 0)
        if c:
            self._unlink(message, c)
        self.cnt[message] = c + 1
        self._append(message, c + 1)
        if c + 1 > self.mx:
            self.mx = c + 1
        return (self.buckets[self.mx].tail, self.mx)


class SpaceSaving:
    """Heap of (count, stamp, message) with lazy deletion; rebuilt when it gets too stale."""

    def __init__(self, k):
        self.k = k
        self.cnt = {}
        self.stamp = {}
        self.t = 0
        self.heap = []

    def _set(self, msg, c):
        self.t += 1
        self.cnt[msg] = c
        self.stamp[msg] = self.t
        heapq.heappush(self.heap, (c, self.t, msg))
        if len(self.heap) > 4 * self.k + 16:
            self.heap = [(self.cnt[m], self.stamp[m], m) for m in self.cnt]
            heapq.heapify(self.heap)

    def add(self, message):
        if message in self.cnt:
            self._set(message, self.cnt[message] + 1)
            return
        if len(self.cnt) < self.k:
            self._set(message, 1)
            return
        while True:
            c, s, m = heapq.heappop(self.heap)
            if self.cnt.get(m) == c and self.stamp[m] == s:
                break
        del self.cnt[m], self.stamp[m]
        self._set(message, c + 1)

    def top(self, m):
        items = list(self.cnt.items())
        items.sort(key=lambda kv: kv[0])
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[:m]
