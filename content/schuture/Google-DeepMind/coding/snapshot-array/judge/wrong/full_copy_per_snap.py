# expect: TLE
# Correct but too slow: snap copies the whole array (the page's first design: O(length) per snap and O(length) memory per snapshot).
class SnapshotArray:
    def __init__(self, length):
        self._length = length
        self._cur = [0] * length
        self._snaps = []
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
        self._cur[index] = val

    def snap(self):
        self._snaps.append(list(self._cur))
        return len(self._snaps) - 1

    def get(self, index, snap_id):
        self._check_index(index)
        self._check_live(snap_id)
        return self._snaps[snap_id][index]

    def release(self, snap_id):
        if not (0 <= snap_id < len(self._snaps)):
            raise IndexError(snap_id)
        for s in range(self._released + 1, snap_id + 1):
            self._snaps[s] = None   # a released copy can be dropped
        self._released = max(self._released, snap_id)

    def stored_entries(self):
        return sum(len(c) for c in self._snaps if c is not None)

    def diff(self, a, b):
        self._check_live(a)
        self._check_live(b)
        if not a < b:
            raise ValueError((a, b))
        sa, sb = self._snaps[a], self._snaps[b]
        return {i: (sa[i], sb[i]) for i in range(self._length) if sa[i] != sb[i]}
