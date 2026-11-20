def pack(jobs: list[int], k: int, cap: int, stats: dict | None = None) -> list[int] | None:
    """The lexicographically smallest valid assignment, or None if there is none; writes the node count to stats["nodes"]."""
    raise NotImplementedError


def pack_fast(jobs: list[int], k: int, cap: int, stats: dict | None = None) -> list[int] | None:
    """Any valid assignment, or None if there is none, for n up to 14; writes the node count to stats["nodes"]."""
    raise NotImplementedError


def min_peak(jobs: list[int], k: int, stats: dict | None = None) -> tuple[int, list[int]]:
    """(smallest possible peak, one assignment that reaches it); writes the node count to stats["nodes"]."""
    raise NotImplementedError
