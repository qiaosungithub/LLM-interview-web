# expect: WA
# Wrong for Part 1: the gap filler updates its owner on 'sh.end >= frontier', so a tie goes to the last shard in (start, end, id) order, not the smallest start then id.
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
        if sh.end >= frontier:
            frontier, owner = sh.end, sh
    kept.sort(key=lambda sh: (sh.start, sh.end, sh.id))
    return kept


def _ring_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


class ShardRouter:
    def __init__(self, num_vnodes=150):
        self.num_vnodes = num_vnodes
        self._hashes, self._owners, self._shards = [], [], set()

    def add_shard(self, shard_id):
        if shard_id in self._shards:
            raise ValueError(shard_id)
        self._shards.add(shard_id)
        for v in range(self.num_vnodes):
            h = _ring_hash(f"{shard_id}#{v}")
            i = bisect.bisect_left(self._hashes, h)
            self._hashes.insert(i, h)
            self._owners.insert(i, shard_id)

    def remove_shard(self, shard_id):
        if shard_id not in self._shards:
            raise ValueError(shard_id)
        self._shards.discard(shard_id)
        keep = [(h, o) for h, o in zip(self._hashes, self._owners) if o != shard_id]
        self._hashes = [h for h, _ in keep]
        self._owners = [o for _, o in keep]

    def locate(self, key):
        if not self._hashes:
            raise LookupError("no shards")
        i = bisect.bisect_left(self._hashes, _ring_hash(str(key)))
        return self._owners[i % len(self._owners)]
