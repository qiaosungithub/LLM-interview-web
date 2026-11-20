# expect: WA
# Wrong for Part 2: splits the largest range of the hashed key space on each add -- moves only what must move, but who owns what depends on the add order.
import bisect
import hashlib
import heapq


class Shard:
    def __init__(self, id, start, end):
        self.id, self.start, self.end = id, start, end

    def __repr__(self):
        return f"Shard({self.id!r}, {self.start}, {self.end})"


def rebalance(limit, shards):
    if not shards:
        return []
    order = sorted(shards, key=lambda sh: (sh.start, sh.end, sh.id))
    free_times = []
    kept = []
    for sh in order:
        t_min = float("-inf") if len(free_times) < limit else heapq.heappop(free_times)
        new_start = sh.start if t_min <= sh.start else t_min
        if new_start > sh.end:
            if t_min != float("-inf"):
                heapq.heappush(free_times, t_min)
            continue
        kept.append(Shard(sh.id, new_start, sh.end))
        heapq.heappush(free_times, sh.end + 1)
    kept.sort(key=lambda sh: (sh.start, sh.end, sh.id))
    frontier, owner = kept[0].start - 1, None
    for sh in kept:
        if sh.start > frontier + 1:
            owner.end = sh.start - 1
        if sh.end > frontier:
            frontier, owner = sh.end, sh
    kept.sort(key=lambda sh: (sh.start, sh.end, sh.id))
    return kept


SPACE = 1 << 32


class ShardRouter:
    """Each shard owns a range of the hashed key space: a new shard takes the upper half of the largest range,
    a removed shard's range goes to the shard just below it (or above it, for the lowest range)."""

    def __init__(self):
        self._starts, self._owners = [], []      # sorted range starts; _owners[i] owns [_starts[i], _starts[i + 1])

    def _size(self, i):
        end = self._starts[i + 1] if i + 1 < len(self._starts) else SPACE
        return end - self._starts[i]

    def add_shard(self, shard_id):
        if shard_id in self._owners:
            raise ValueError(shard_id)
        if not self._owners:
            self._starts, self._owners = [0], [shard_id]
            return
        i = max(range(len(self._starts)), key=lambda j: (self._size(j), -j))
        mid = self._starts[i] + self._size(i) // 2
        self._starts.insert(i + 1, mid)
        self._owners.insert(i + 1, shard_id)

    def remove_shard(self, shard_id):
        if shard_id not in self._owners:
            raise ValueError(shard_id)
        i = self._owners.index(shard_id)
        if i == 0 and len(self._owners) > 1:
            self._starts.pop(1)                  # the range above takes over from 0
            self._owners.pop(0)
        else:
            self._starts.pop(i)
            self._owners.pop(i)

    def locate(self, key):
        if not self._owners:
            raise LookupError("no shards")
        h = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % SPACE
        return self._owners[bisect.bisect_right(self._starts, h) - 1]
