# expect: WA
# Records under the lock but reads the key's last timestamp to return after releasing it, so it may return another call's.
import bisect
import threading
import time
from typing import Any


class _WallClock:
    def now(self) -> float:
        return time.time()


class TimeMap:
    """Part 3 rules. Per key: a list of (timestamp, value) pairs in increasing timestamp order."""

    def __init__(self, clock: Any = None) -> None:
        self._clock = _WallClock() if clock is None else clock
        self._hist: dict = {}

    def _when(self, timestamp):
        return self._clock.now() if timestamp is None else timestamp

    def set(self, key, value, timestamp=None):
        when = self._when(timestamp)
        hist = self._hist.get(key)
        if hist is None:
            self._hist[key] = [(when, value)]
            return when
        top = hist[-1][0]
        if not when > top:
            when = top + 1
        hist.append((when, value))
        return when

    def get(self, key, timestamp=None):
        when = self._when(timestamp)
        hist = self._hist.get(key)
        if not hist:
            return None
        i = bisect.bisect_right(hist, when, key=lambda p: p[0])
        return hist[i - 1][1] if i else None


class ThreadSafeTimeMap(TimeMap):
    def __init__(self, clock=None):
        super().__init__(clock)
        self._mu = threading.Lock()

    def set(self, key, value, timestamp=None):
        with self._mu:
            super().set(key, value, timestamp)
        return self._hist[key][-1][0]

    def get(self, key, timestamp=None):
        with self._mu:
            return super().get(key, timestamp)
