# expect: WA
# Wrong for Part 1: a machine's next free key is the end of its last shard rather than end + 1, so a shifted shard starts on a key that is already at the limit.
import bisect
import hashlib
import heapq
from dataclasses import dataclass


@dataclass
class Shard:
    id: str
    start: int
    end: int


def rebalance(limit, shards):
    order = sorted(shards, key=lambda s: (s.start, s.end, s.id))
    free = []
    kept = []
    for s in order:
        t = heapq.heappop(free) if len(free) == limit else None
        new_start = s.start if t is None or t <= s.start else t
        if new_start > s.end:
            if t is not None:
                heapq.heappush(free, t)
            continue
        kept.append(Shard(s.id, new_start, s.end))
        heapq.heappush(free, s.end)
    if not kept:
        return []
    kept.sort(key=lambda s: (s.start, s.end, s.id))
    reach, owner = kept[0].start - 1, None
    for s in kept:
        if s.start > reach + 1:
            owner.end = s.start - 1
        if s.end > reach:
            reach, owner = s.end, s
    kept.sort(key=lambda s: (s.start, s.end, s.id))
    return kept


def _h(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


class ShardRouter:
    VNODES = 100

    def __init__(self):
        self._ids = set()
        self._points = []          # sorted (hash, id)

    def add_shard(self, shard_id):
        if shard_id in self._ids:
            raise ValueError(shard_id)
        self._ids.add(shard_id)
        for p in self._vnodes(shard_id):
            bisect.insort(self._points, (p, shard_id))

    def remove_shard(self, shard_id):
        if shard_id not in self._ids:
            raise ValueError(shard_id)
        self._ids.remove(shard_id)
        self._points = [pt for pt in self._points if pt[1] != shard_id]

    def _vnodes(self, shard_id):
        return [_h(f"{shard_id}#{v}") for v in range(self.VNODES)]

    def locate(self, key):
        if not self._points:
            raise LookupError("no shards")
        h = _h(str(key))
        i = bisect.bisect_left(self._points, (h, ""))
        return self._points[i % len(self._points)][1]
