from typing import Any


class ManualClock:
    """The statement's manual clock, given as is (the tests use their own copy of it)."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        self._now += dt   # dt may be negative: this simulates the clock moving backward


class TimeMap:
    def __init__(self, clock: Any = None) -> None:
        """Parts 1-3: an empty store; `clock` is any object with now() -> float, by default one calling time.time()."""

    def set(self, key: str, value: Any, timestamp: float | None = None) -> float:
        """Records that, as of `timestamp` (clock.now() when omitted), `key` holds `value`; from Part 3 on, keeps each
        key's recorded timestamps strictly increasing and returns the timestamp actually recorded."""
        raise NotImplementedError

    def get(self, key: str, timestamp: float | None = None) -> Any | None:
        """The value with the largest recorded timestamp <= `timestamp` (clock.now() when omitted), or None."""
        raise NotImplementedError


class ThreadSafeTimeMap:
    def __init__(self, clock: Any = None) -> None:
        """Part 4: Part 3's TimeMap, safe to call from many threads at the same time."""

    def set(self, key: str, value: Any, timestamp: float | None = None) -> float:
        """Part 3's set, behaving as if it took effect at a single instant."""
        raise NotImplementedError

    def get(self, key: str, timestamp: float | None = None) -> Any | None:
        """Part 3's get, behaving as if it took effect at a single instant."""
        raise NotImplementedError
