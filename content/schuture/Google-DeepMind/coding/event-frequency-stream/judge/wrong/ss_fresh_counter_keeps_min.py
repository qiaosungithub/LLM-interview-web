# expect: WA
# Wrong for Part 3: a fresh counter at 1 does not lower the cached minimum once an earlier counter has grown,
# so the next eviction removes a higher counter than the lowest. Passes the sample.
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
    return (best, list(stamps[best]))


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


from collections import OrderedDict


class SpaceSaving:
    """Buckets by counter value, each an OrderedDict (oldest at that value first), plus the minimum value."""

    def __init__(self, k):
        self.k = k
        self.cnt = {}
        self.bucket = {}
        self.low = 0

    def _move(self, msg, new):
        old = self.cnt.get(msg)
        if old is not None:
            del self.bucket[old][msg]
            if not self.bucket[old]:
                del self.bucket[old]
        self.cnt[msg] = new
        self.bucket.setdefault(new, OrderedDict())[msg] = None

    def add(self, message):
        if message in self.cnt:
            old = self.cnt[message]
            self._move(message, old + 1)
            if old == self.low and old not in self.bucket:
                self.low = old + 1
            return
        if len(self.cnt) < self.k:
            if not self.cnt:
                self.low = 1        # only the very first counter sets the minimum; later fresh counters at 1 do not
            self._move(message, 1)
            return
        m = self.low
        victim = next(iter(self.bucket[m]))
        del self.cnt[victim]
        del self.bucket[m][victim]
        if not self.bucket[m]:
            del self.bucket[m]
            self.low = m + 1
        self._move(message, m + 1)

    def top(self, m):
        return sorted(self.cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:m]
