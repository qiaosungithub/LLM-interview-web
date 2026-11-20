class SnapshotArray:
    def __init__(self, length: int) -> None:
        """Part 1: an array of length integers, every index starting at 0."""

    def set(self, index: int, val: int) -> None:
        """Part 1: writes val at index; IndexError if index is outside [0, length)."""
        raise NotImplementedError

    def snap(self) -> int:
        """Part 1: takes a snapshot and returns its id: 0, 1, 2, ..."""
        raise NotImplementedError

    def get(self, index: int, snap_id: int) -> int:
        """Part 1: the value at index when snap_id was taken; IndexError for a bad index or a snap_id not yet returned."""
        raise NotImplementedError

    def release(self, snap_id: int) -> None:
        """Part 2: the caller will never again read a snapshot id <= snap_id; a later get on one raises ValueError."""
        raise NotImplementedError

    def stored_entries(self) -> int:
        """Part 2: the total number of (snap_id, value) entries currently held across every index."""
        raise NotImplementedError

    def diff(self, a: int, b: int) -> dict[int, tuple[int, int]]:
        """Part 3: every index whose value differs between live snapshots a < b, mapped to (value_at_a, value_at_b)."""
        raise NotImplementedError
