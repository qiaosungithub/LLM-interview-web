import bisect
import hashlib
import heapq

NEG_INF = float('-inf')


class Shard:
    def __init__(self, id, start, end):
        self.id, self.start, self.end = id, start, end

    def __repr__(self):
        return f"Shard({self.id!r}, {self.start}, {self.end})"


def rebalance(limit, shards):
    if not shards:
        return []
    order = sorted(shards, key=lambda sh: (sh.start, sh.end, sh.id))

    free_times = []    # min-heap of "next free key" for machines in use; size <= limit
    kept = []
    for sh in order:
        if len(free_times) < limit:
            t_min = NEG_INF                       # an unused machine: always free
        else:
            t_min = heapq.heappop(free_times)
        new_start = sh.start if t_min <= sh.start else t_min
        if new_start > sh.end:
            if t_min != NEG_INF:
                heapq.heappush(free_times, t_min)  # NOTE: hand the machine back unused, don't drop it
            continue
        kept.append(Shard(sh.id, new_start, sh.end))
        heapq.heappush(free_times, sh.end + 1)

    if not kept:
        return []

    # Gap fill: scan the kept shards in (start, end, id) order of their step-4 ranges.
    kept.sort(key=lambda sh: (sh.start, sh.end, sh.id))
    frontier = kept[0].start - 1        # NOTE: the very first PROCESSED shard is never shifted (the
    frontier_owner = None               #       heap starts empty), so kept[0].start == min(all starts)
    for sh in kept:
        if sh.start > frontier + 1:
            frontier_owner.end = sh.start - 1     # NOTE: only stretches over keys with coverage 0
        if sh.end > frontier:           # NOTE: strict '>': ties go to the smallest (start, id)
            frontier, frontier_owner = sh.end, sh
    # NOTE: no hole after the last kept shard -- the largest input end is always covered

    kept.sort(key=lambda sh: (sh.start, sh.end, sh.id))
    return kept


def _ring_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


class ShardRouter:
    def __init__(self, num_vnodes=150):
        self.num_vnodes = num_vnodes
        self._ring_hashes = []     # sorted
        self._ring_owner = []      # _ring_owner[i] owns _ring_hashes[i]
        self._shards = set()

    def add_shard(self, shard_id):
        if shard_id in self._shards:
            raise ValueError(f"shard already present: {shard_id}")
        self._shards.add(shard_id)
        for v in range(self.num_vnodes):
            h = _ring_hash(f"{shard_id}#{v}")
            i = bisect.bisect_left(self._ring_hashes, h)
            self._ring_hashes.insert(i, h)          # NOTE: O(T) per insertion -- a plain sorted list
            self._ring_owner.insert(i, shard_id)

    def remove_shard(self, shard_id):
        if shard_id not in self._shards:
            raise ValueError(f"no such shard: {shard_id}")
        self._shards.discard(shard_id)
        keep = [(h, o) for h, o in zip(self._ring_hashes, self._ring_owner) if o != shard_id]
        self._ring_hashes = [h for h, _ in keep]
        self._ring_owner = [o for _, o in keep]

    def locate(self, key):
        if not self._ring_hashes:
            raise LookupError("locate() called with no shards")
        h = _ring_hash(str(key))
        i = bisect.bisect_left(self._ring_hashes, h)
        if i == len(self._ring_hashes):
            i = 0                                   # wrap past the largest hash
        return self._ring_owner[i]
