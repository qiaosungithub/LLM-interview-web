# expect: TLE
# Wrong for Part 2: the page's code before its audit fix, with each count's group in a plain dict. Deleting keys
# leaves slots that next(reversed(d)) must skip, so a read is not O(1) once a group loses its last keys.
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
    """Reports the most frequent message among the last n held events after every arrival."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._window: deque[tuple[str, int]] = deque()   # held events, oldest first
        self._count: dict[str, int] = {}
        self._bucket: dict[int, dict[str, None]] = {}   # count -> ordered set of messages at that count
        self._max_count = 0

    def _increment(self, message: str) -> None:
        old = self._count.get(message, 0)
        new = old + 1
        if old:
            del self._bucket[old][message]
            if not self._bucket[old]:
                del self._bucket[old]
        self._count[message] = new
        self._bucket.setdefault(new, {})[message] = None   # NOTE: appended last -- most recently changed
        if new > self._max_count:
            self._max_count = new

    def _decrement(self, message: str) -> None:
        old = self._count[message]
        new = old - 1
        del self._bucket[old][message]
        emptied = not self._bucket[old]
        if emptied:
            del self._bucket[old]        # NOTE: drop the empty bucket, or a later self._max_count could
                                          # point at a bucket that no longer holds any message
        if new:
            self._count[message] = new
            self._bucket.setdefault(new, {})[message] = None
        else:
            del self._count[message]
        if old == self._max_count and emptied:
            self._max_count -= 1         # NOTE: derived in the text -- the new maximum is exactly old - 1

    def add(self, message: str, timestamp: int) -> tuple[str, int]:
        if len(self._window) == self._n:
            evicted_message, _ = self._window.popleft()   # NOTE: eviction is numbered before insertion
            self._decrement(evicted_message)
        self._window.append((message, timestamp))
        self._increment(message)
        top = next(reversed(self._bucket[self._max_count]))
        return top, self._max_count


class SpaceSaving:
    """Space-Saving with a stream-summary structure: buckets keyed by counter value, each an ordered
    set of the messages currently holding that value, plus O(1) access to the minimum bucket."""

    def __init__(self, k: int) -> None:
        self._k = k
        self._count: dict[str, int] = {}
        self._bucket: dict[int, dict[str, None]] = {}
        self._min_count = 0

    def _reinsert(self, message: str, new_count: int) -> None:
        old_count = self._count.get(message)
        if old_count is not None:
            del self._bucket[old_count][message]
            if not self._bucket[old_count]:
                del self._bucket[old_count]
        self._count[message] = new_count
        self._bucket.setdefault(new_count, {})[message] = None

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
