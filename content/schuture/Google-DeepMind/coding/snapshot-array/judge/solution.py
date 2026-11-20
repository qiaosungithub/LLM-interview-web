from bisect import bisect_right


class SnapshotArray:
    """array[i] is 0 until set. get(i, s) reads what array[i] held when snapshot s was taken; a write
    made after s, or not yet covered by any snap at all, is never visible to it."""

    def __init__(self, length: int) -> None:
        self._length = length
        self._next_id = 0                                     # id whichever snap() call is next will return
        self._history: dict[int, list[tuple[int, int]]] = {}  # index -> [(snap_id, value), ...], lazy

    def _check_index(self, index: int) -> None:
        if not (0 <= index < self._length):
            raise IndexError(f"index {index} out of range [0, {self._length})")

    def _check_taken(self, snap_id: int) -> None:
        if not (0 <= snap_id < self._next_id):
            raise IndexError(f"snapshot {snap_id} was never taken")

    def set(self, index: int, val: int) -> None:
        self._check_index(index)
        h = self._history.setdefault(index, [])
        if h and h[-1][0] == self._next_id:
            h[-1] = (self._next_id, val)    # NOTE: overwrite -- an earlier set in this same pending
        else:                               #      interval can never be read by any snapshot
            h.append((self._next_id, val))

    def snap(self) -> int:
        snap_id = self._next_id
        self._next_id += 1
        return snap_id

    def get(self, index: int, snap_id: int) -> int:
        self._check_index(index)
        self._check_taken(snap_id)
        h = self._history.get(index)
        if not h:
            return 0
        pos = bisect_right(h, snap_id, key=lambda e: e[0]) - 1  # NOTE: -1 -- bisect_right lands one past
        return h[pos][1] if pos >= 0 else 0                     #      every id <= snap_id; back up one


class SnapshotArray(SnapshotArray):
    """Adds release(): the caller promises never to read a snapshot <= released again."""

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._released = -1                                   # -1: nothing released yet

    def _compact(self, index: int) -> None:
        """Drops every entry of index's history that release() has made permanently unreachable."""
        h = self._history.get(index)
        if not h or self._released < 0:
            return
        pos = bisect_right(h, self._released, key=lambda e: e[0]) - 1
        if pos > 0:          # NOTE: pos itself is the floor entry to KEEP -- only what precedes it is dead
            del h[:pos]

    def _check_live(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id <= self._released:
            raise ValueError(f"snapshot {snap_id} was released (released up to {self._released})")

    def get(self, index: int, snap_id: int) -> int:
        self._check_index(index)
        self._check_live(snap_id)
        return super().get(index, snap_id)

    def stored_entries(self) -> int:
        return sum(len(h) for h in self._history.values())


class SnapshotArrayEagerRelease(SnapshotArray):
    """release() compacts every index immediately."""

    def release(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id > self._released:
            self._released = snap_id
        for index in range(self._length):
            self._compact(index)


class SnapshotArrayLazyRelease(SnapshotArray):
    """release() only raises the boundary and nudges a small, fixed-size batch of a round-robin sweep
    forward; the rest of an index's compaction happens the next time that index is set."""

    _SWEEP_BATCH = 4          # arbitrary and small -- illustrates the mechanism, not a tuned constant

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._sweep_pos = 0

    def set(self, index: int, val: int) -> None:
        self._check_index(index)
        self._compact(index)     # NOTE: on-touch -- the cheapest moment to drop this index's dead entries
        super().set(index, val)

    def release(self, snap_id: int) -> None:
        self._check_taken(snap_id)
        if snap_id > self._released:
            self._released = snap_id
        batch = min(self._SWEEP_BATCH, self._length)   # NOTE: min() -- also keeps this at 0 when length == 0
        for _ in range(batch):
            self._compact(self._sweep_pos)
            self._sweep_pos = (self._sweep_pos + 1) % self._length


SnapshotArray = SnapshotArrayLazyRelease   # the fuller answer: cheap release(), memory still bounded


class SnapshotArray(SnapshotArray):
    """Adds diff(): every index whose value changed between two live snapshots."""

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self._journal: list[set[int]] = []       # journal[s]: indices set during the interval sealed as s
        self._pending_touched: set[int] = set()

    def set(self, index: int, val: int) -> None:
        super().set(index, val)             # validates index, applies the inherited on-touch compaction
        self._pending_touched.add(index)     # NOTE: only after a successful set -- a rejected one changed nothing

    def snap(self) -> int:
        self._journal.append(self._pending_touched)
        self._pending_touched = set()
        return super().snap()

    def diff(self, a: int, b: int) -> dict[int, tuple[int, int]]:
        self._check_live(a)
        self._check_live(b)
        if not a < b:
            raise ValueError(f"diff requires a < b, got a={a}, b={b}")
        touched: set[int] = set()
        for s in range(a + 1, b + 1):
            touched |= self._journal[s]
        result: dict[int, tuple[int, int]] = {}
        for index in touched:
            va, vb = self.get(index, a), self.get(index, b)
            if va != vb:
                result[index] = (va, vb)
        return result
