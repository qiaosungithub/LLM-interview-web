# expect: TLE
# Wrong for Part 2: right answers, but scans every held message's count after each add: O(distinct) per add,
# not the O(1) amortised the statement requires.
from collections import OrderedDict, deque


def most_frequent(events: list[tuple[str, int]]) -> tuple[str, list[int]]:
    if not events:
        raise ValueError("events must be non-empty")
    counts: dict[str, int] = {}
    timestamps: dict[str, list[int]] = {}
    first_index: dict[str, int] = {}
    for i, (message, ts) in enumerate(events):
        counts[message] = counts.get(message, 0) + 1
        timestamps.setdefault(message, []).append(ts)
        first_index.setdefault(message, i)   # NOTE: setdefault -- keeps the first occurrence, never overwrites it
    best = max(counts, key=lambda m: (counts[m], -first_index[m]))
    return best, timestamps[best]


class WindowTopMessage:
    def __init__(self, n: int) -> None:
        self._n = n
        self._window: deque[str] = deque()
        self._count: dict[str, int] = {}
        self._last_change: dict[str, int] = {}
        self._changes = 0

    def _change(self, message: str, delta: int) -> None:
        self._changes += 1
        self._count[message] = self._count.get(message, 0) + delta
        self._last_change[message] = self._changes
        if not self._count[message]:
            del self._count[message]

    def add(self, message: str, timestamp: int) -> tuple[str, int]:
        if len(self._window) == self._n:
            self._change(self._window.popleft(), -1)
        self._window.append(message)
        self._change(message, +1)
        best = max(self._count, key=lambda m: (self._count[m], self._last_change[m]))
        return best, self._count[best]


class SpaceSaving:
    """Space-Saving with a stream-summary structure: buckets keyed by counter value, each an ordered
    set of the messages currently holding that value, plus O(1) access to the minimum bucket."""

    def __init__(self, k: int) -> None:
        self._k = k
        self._count: dict[str, int] = {}
        self._bucket: dict[int, OrderedDict[str, None]] = {}
        self._min_count = 0

    def _reinsert(self, message: str, new_count: int) -> None:
        old_count = self._count.get(message)
        if old_count is not None:
            del self._bucket[old_count][message]
            if not self._bucket[old_count]:
                del self._bucket[old_count]
        self._count[message] = new_count
        self._bucket.setdefault(new_count, OrderedDict())[message] = None

    def add(self, message: str) -> None:
        if message in self._count:
            old = self._count[message]
            self._reinsert(message, old + 1)
            if old == self._min_count and not self._bucket.get(old):
                self._min_count = old + 1   # NOTE: this message just left the sole occupant of the minimum bucket
            return
        if len(self._count) < self._k:
            self._reinsert(message, 1)
            self._min_count = 1             # NOTE: a fresh counter starts at 1, the least any counter can be
            return
        m = self._min_count
        evicted = next(iter(self._bucket[m]))   # NOTE: longest-held at the minimum -- the statement's tie-break
        del self._count[evicted]
        del self._bucket[m][evicted]
        if not self._bucket[m]:
            del self._bucket[m]
            self._min_count = m + 1         # NOTE: derived in the text -- the arriving message witnesses m + 1
        self._reinsert(message, m + 1)

    def top(self, m: int) -> list[tuple[str, int]]:
        ranked = sorted(self._count.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:m]
