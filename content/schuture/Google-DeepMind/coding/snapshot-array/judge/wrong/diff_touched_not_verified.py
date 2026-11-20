# expect: WA
# Wrong for Part 3: reports every index touched between a and b, even one written back to the value it had at a.
from bisect import bisect_right


class SnapshotArray:
    def __init__(self, length):
        self._length = length
        self._next_id = 0
        self._history = {}              # index -> [(snap_id, value), ...]
        self._released = -1
        self._sweep_pos = 0
        self._journal = []              # journal[s]: indices set in the interval sealed as s
        self._pending_touched = set()

    def _check_index(self, index):
        if not (0 <= index < self._length):
            raise IndexError(index)

    def _check_taken(self, snap_id):
        if not (0 <= snap_id < self._next_id):
            raise IndexError(snap_id)

    def _check_live(self, snap_id):
        self._check_taken(snap_id)
        if snap_id <= self._released:
            raise ValueError(snap_id)

    def _compact(self, index):
        h = self._history.get(index)
        if not h or self._released < 0:
            return
        pos = bisect_right(h, self._released, key=lambda e: e[0]) - 1
        if pos > 0:
            del h[:pos]

    def set(self, index, val):
        self._check_index(index)
        self._compact(index)
        h = self._history.setdefault(index, [])
        if h and h[-1][0] == self._next_id:
            h[-1] = (self._next_id, val)
        else:
            h.append((self._next_id, val))
        self._pending_touched.add(index)

    def snap(self):
        self._journal.append(self._pending_touched)
        self._pending_touched = set()
        self._next_id += 1
        return self._next_id - 1

    def get(self, index, snap_id):
        self._check_index(index)
        self._check_live(snap_id)
        h = self._history.get(index)
        if not h:
            return 0
        pos = bisect_right(h, snap_id, key=lambda e: e[0]) - 1
        return h[pos][1] if pos >= 0 else 0

    def release(self, snap_id):
        self._check_taken(snap_id)
        if snap_id > self._released:
            self._released = snap_id
        for _ in range(min(4, self._length)):
            self._compact(self._sweep_pos)
            self._sweep_pos = (self._sweep_pos + 1) % self._length

    def stored_entries(self):
        return sum(len(h) for h in self._history.values())

    def diff(self, a, b):
        self._check_live(a)
        self._check_live(b)
        if not a < b:
            raise ValueError((a, b))
        touched = set()
        for s in range(a + 1, b + 1):
            touched |= self._journal[s]
        out = {}
        for index in touched:
            out[index] = (self.get(index, a), self.get(index, b))
        return out
