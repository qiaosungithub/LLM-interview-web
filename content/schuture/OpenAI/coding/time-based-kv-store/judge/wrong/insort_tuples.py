# expect: WA
# Parts 1-2 only; keeps (timestamp, value) tuples with insort, so equal timestamps are ordered by value, not by call.
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
        self._entries: dict[str, list[tuple[float, Any]]] = {}

    def set(self, key: str, value: Any, timestamp: float) -> None:
        bisect.insort(self._entries.setdefault(key, []), (timestamp, value))

    def get(self, key: str, timestamp: float) -> Any | None:
        entries = self._entries.get(key, [])
        i = bisect.bisect_right(entries, timestamp, key=lambda e: e[0])
        return entries[i - 1][1] if i > 0 else None


# ---------------------------------------------------------------- Part 2

class RealClock:
    def now(self) -> float:
        return time.time()


class TimeMap(TimeMap):
    def __init__(self, clock: Any = None) -> None:
        super().__init__()
        self._clock = clock if clock is not None else RealClock()

    def set(self, key: str, value: Any, timestamp: float | None = None) -> None:
        super().set(key, value, self._clock.now() if timestamp is None else timestamp)

    def get(self, key: str, timestamp: float | None = None) -> Any | None:
        return super().get(key, self._clock.now() if timestamp is None else timestamp)


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
