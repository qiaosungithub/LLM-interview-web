# expect: WA
# The default clock uses time.monotonic(), whose readings are not the time.time() the statement asks for.
import bisect
import threading
import time
from typing import Any


class ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        self._now += dt   # dt may be negative: this simulates the clock moving backward


# ---------------------------------------------------------------- Part 1

class TimeMap:
    def __init__(self) -> None:
        self._timestamps: dict[str, list[float]] = {}
        self._values: dict[str, list[Any]] = {}

    def set(self, key: str, value: Any, timestamp: float) -> None:
        timestamps = self._timestamps.setdefault(key, [])
        values = self._values.setdefault(key, [])
        i = bisect.bisect_left(timestamps, timestamp)
        if i < len(timestamps) and timestamps[i] == timestamp:
            values[i] = value               # NOTE: same timestamp as an existing entry -- the later call wins
        else:
            timestamps.insert(i, timestamp)
            values.insert(i, value)

    def get(self, key: str, timestamp: float) -> Any | None:
        timestamps = self._timestamps.get(key)
        if not timestamps:
            return None
        i = bisect.bisect_right(timestamps, timestamp)  # NOTE: not bisect_left -- an exact match still counts
        return self._values[key][i - 1] if i > 0 else None


# ---------------------------------------------------------------- Part 2

class RealClock:
    def now(self) -> float:
        return time.monotonic()


class TimeMap(TimeMap):
    def __init__(self, clock: Any = None) -> None:
        super().__init__()
        self._clock = clock if clock is not None else RealClock()

    def set(self, key: str, value: Any, timestamp: float | None = None) -> None:
        super().set(key, value, self._clock.now() if timestamp is None else timestamp)

    def get(self, key: str, timestamp: float | None = None) -> Any | None:
        return super().get(key, self._clock.now() if timestamp is None else timestamp)


# ---------------------------------------------------------------- Part 3

class TimeMap(TimeMap):
    def __init__(self, clock: Any = None) -> None:
        super().__init__(clock)
        self._last: dict[str, float] = {}

    def set(self, key: str, value: Any, timestamp: float | None = None) -> float:
        if timestamp is None:
            timestamp = self._clock.now()
        last = self._last.get(key)
        if last is not None and timestamp <= last:  # NOTE: a backward clock and a stale explicit timestamp
            timestamp = last + 1                     #      both land here
        self._timestamps.setdefault(key, []).append(timestamp)  # NOTE: append -- timestamp now exceeds all earlier
        self._values.setdefault(key, []).append(value)
        self._last[key] = timestamp
        return timestamp


# ---------------------------------------------------------------- Part 4

class PerKeyLockTimeMap(TimeMap):
    def __init__(self, clock: Any = None) -> None:
        super().__init__(clock)
        self._key_locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._registry_lock:  # NOTE: guards only the creation of a key's lock, not the operation under it
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
        return lock

    def set(self, key: str, value: Any, timestamp: float | None = None) -> float:
        with self._lock_for(key):  # NOTE: a missing timestamp is read from the clock in here, by Part 3's set
            return super().set(key, value, timestamp)

    def get(self, key: str, timestamp: float | None = None) -> Any | None:
        with self._lock_for(key):  # NOTE: get needs it too -- set appends the timestamp before the value
            return super().get(key, timestamp)


ThreadSafeTimeMap = PerKeyLockTimeMap


class GlobalLockTimeMap(TimeMap):
    def __init__(self, clock: Any = None) -> None:
        super().__init__(clock)
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, timestamp: float | None = None) -> float:
        with self._lock:
            return super().set(key, value, timestamp)

    def get(self, key: str, timestamp: float | None = None) -> Any | None:
        with self._lock:
            return super().get(key, timestamp)
