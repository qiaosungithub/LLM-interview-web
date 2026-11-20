def most_frequent(events: list[tuple[str, int]]) -> tuple[str, list[int]]:
    """Part 1: the message that occurs most often and its timestamps in input order; ties go to the message whose
    first occurrence comes earliest; raises ValueError on []."""
    raise NotImplementedError


class WindowTopMessage:
    def __init__(self, n: int):
        """Part 2: holds only the last n events seen (the oldest held event is evicted before a new one is inserted)."""

    def add(self, message: str, timestamp: int) -> tuple[str, int]:
        """(message, count) with the highest count after the arrival; ties go to the most recent count change."""
        raise NotImplementedError


class SpaceSaving:
    def __init__(self, k: int):
        """Part 3: Space-Saving with counters for at most k distinct messages."""

    def add(self, message: str) -> None:
        """Processes one message of the unbounded stream."""
        raise NotImplementedError

    def top(self, m: int) -> list[tuple[str, int]]:
        """Monitored messages by reported count, descending; ties by message."""
        raise NotImplementedError
