# expect: TLE
# Correct but too slow for Part 3: diff compares every index that has ever been written, not only those written between a and b.
from bisect import bisect_right


class SnapshotArray:
    """Per written index: parallel lists ids[] / vals[]; the entry for the pending id (= number of snaps taken)
    is overwritten in place. A per-id log of touched indices drives diff. Release compacts eagerly."""

    def __init__(self, length):
        self.n = length
        self.ids = {}
        self.vals = {}
        self.taken = 0
        self.dead = -1
        self.log = {}           # id -> set of indices written while that id was pending

    def _idx(self, i):
        if not (0 <= i < self.n):
            raise IndexError("index out of range")

    def _sid(self, s):
        if not (0 <= s < self.taken):
            raise IndexError("snapshot id not taken")
        if s <= self.dead:
            raise ValueError("snapshot released")

    def set(self, index, val):
        self._idx(index)
        ids = self.ids.setdefault(index, [])
        vals = self.vals.setdefault(index, [])
        if ids and ids[-1] == self.taken:
            vals[-1] = val
        else:
            ids.append(self.taken)
            vals.append(val)
        self.log.setdefault(self.taken, set()).add(index)

    def snap(self):
        self.taken += 1
        return self.taken - 1

    def _value(self, i, s):
        ids = self.ids.get(i)
        if not ids:
            return 0
        k = bisect_right(ids, s)
        return self.vals[i][k - 1] if k else 0

    def get(self, index, snap_id):
        self._idx(index)
        self._sid(snap_id)
        return self._value(index, snap_id)

    def release(self, snap_id):
        if not (0 <= snap_id < self.taken):
            raise IndexError("snapshot id not taken")
        if snap_id <= self.dead:
            return
        self.dead = snap_id
        for i in list(self.ids):
            ids = self.ids[i]
            k = bisect_right(ids, snap_id)
            if k > 1:
                del ids[:k - 1]
                del self.vals[i][:k - 1]
        for s in [s for s in self.log if s <= snap_id + 1]:
            del self.log[s]

    def stored_entries(self):
        return sum(len(v) for v in self.ids.values())

    def diff(self, a, b):
        self._sid(a)
        self._sid(b)
        if not a < b:
            raise ValueError("need a < b")
        touched = [i for i, ids in self.ids.items() if ids]
        out = {}
        for i in touched:
            va, vb = self._value(i, a), self._value(i, b)
            if va != vb:
                out[i] = (va, vb)
        return out
