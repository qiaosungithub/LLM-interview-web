class ModeTracker:
    def __init__(self) -> None:
        """Creates a tracker with no workers and no data."""

    # Level 1
    def add_worker(self, worker_id: str) -> bool:
        """Registers a worker. False if a worker with this id is registered already."""
        raise NotImplementedError

    def report(self, worker_id: str, values: list[int]) -> bool:
        """The worker reports these values, each list element counted once; False if it is not registered."""
        raise NotImplementedError

    def get_count(self, value: int) -> int:
        """How many times value has been reported, summed over all registered workers."""
        raise NotImplementedError

    def get_worker_count(self, worker_id: str, value: int) -> int | None:
        """How many times this worker has reported value (possibly 0); None if it is not registered."""
        raise NotImplementedError

    # Level 2
    def get_mode(self) -> int | None:
        """The value with the highest global count, the smallest one on a tie; None if there is no data."""
        raise NotImplementedError

    def top_values(self, k: int) -> list[str]:
        """The k values with the highest counts as "value(count)", count descending, then value ascending."""
        raise NotImplementedError

    # Level 3
    def remove_worker(self, worker_id: str) -> int | None:
        """Unregisters the worker, erases all it reported and returns how many values; None if not registered."""
        raise NotImplementedError

    # Level 4
    def merge_workers(self, target_id: str, source_id: str) -> bool:
        """Moves everything source_id reported to target_id and unregisters source_id; False if invalid."""
        raise NotImplementedError
