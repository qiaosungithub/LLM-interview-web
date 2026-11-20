# expect: TLE
# Correct but too slow: each snapshot keeps a dict of what changed since the previous one, and get walks back snapshot by snapshot (the page's second design: O(k) per get, not O(log s)).
class SnapshotArray:
    def __init__(self, length):
        self._length = length
        self._pending = {}
        self._snaps = []            # snaps[s]: {index: value} for the indices set in the interval sealed as s
        self._released = -1

    def _check_index(self, index):
        if not (0 <= index < self._length):
            raise IndexError(index)

    def _check_live(self, snap_id):
        if not (0 <= snap_id < len(self._snaps)):
            raise IndexError(snap_id)
        if snap_id <= self._released:
            raise ValueError(snap_id)

    def set(self, index, val):
        self._check_index(index)
        self._pending[index] = val

    def snap(self):
        self._snaps.append(self._pending)
        self._pending = {}
        return len(self._snaps) - 1

    def _value(self, index, snap_id):
        for s in range(snap_id, -1, -1):
            if index in self._snaps[s]:
                return self._snaps[s][index]
        return 0

    def get(self, index, snap_id):
        self._check_index(index)
        self._check_live(snap_id)
        return self._value(index, snap_id)

    def release(self, snap_id):
        if not (0 <= snap_id < len(self._snaps)):
            raise IndexError(snap_id)
        if snap_id <= self._released:
            return
        merged = {}                 # fold every released interval into the boundary snapshot
        for s in range(max(self._released, 0), snap_id + 1):
            merged.update(self._snaps[s])
            self._snaps[s] = {}
        self._snaps[snap_id] = merged
        self._released = snap_id

    def stored_entries(self):
        return sum(len(d) for d in self._snaps) + len(self._pending)

    def diff(self, a, b):
        self._check_live(a)
        self._check_live(b)
        if not a < b:
            raise ValueError((a, b))
        touched = set()
        for s in range(a + 1, b + 1):
            touched |= self._snaps[s].keys()
        out = {}
        for index in touched:
            va, vb = self._value(index, a), self._value(index, b)
            if va != vb:
                out[index] = (va, vb)
        return out
