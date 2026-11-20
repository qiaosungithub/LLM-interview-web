# expect: WA
# Wrong for Part 2: hash(key) mod N over the sorted ids -- even and deterministic, but adding or removing a shard moves most keys.
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


class ShardRouter:
    def __init__(self):
        self._shards = set()

    def add_shard(self, shard_id):
        if shard_id in self._shards:
            raise ValueError(shard_id)
        self._shards.add(shard_id)

    def remove_shard(self, shard_id):
        if shard_id not in self._shards:
            raise ValueError(shard_id)
        self._shards.remove(shard_id)

    def locate(self, key):
        if not self._shards:
            raise LookupError("no shards")
        ids = sorted(self._shards)       # sorted, so the add order does not matter
        return ids[int(hashlib.md5(str(key).encode()).hexdigest(), 16) % len(ids)]
