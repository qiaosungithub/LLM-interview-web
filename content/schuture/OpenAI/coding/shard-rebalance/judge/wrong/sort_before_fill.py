# expect: WA
# Wrong for Part 1: sorts the survivors by their step-4 (start, end, id) before closing the holes, so an extended shard keeps its old place in the output.
import bisect
import hashlib
from dataclasses import dataclass


@dataclass
class Shard:
    id: str
    start: int
    end: int


def rebalance(limit, shards):
    order = sorted(((s.start, s.end, s.id) for s in shards))
    starts, ends = [], []          # sorted step-4 starts / ends of the kept shards
    kept = []                      # [id, start, end]
    for start, end, sid in order:
        def cov(k):
            return bisect.bisect_right(starts, k) - bisect.bisect_left(ends, k)
        chosen = None
        if cov(start) < limit:
            chosen = start
        else:
            i = bisect.bisect_left(ends, start)          # kept ends >= start; coverage can drop at end + 1
            for e in ends[i:]:
                k = e + 1
                if k > end:
                    break
                if cov(k) < limit:
                    chosen = k
                    break
        if chosen is None:
            continue
        kept.append([sid, chosen, end])
        bisect.insort(starts, chosen)
        bisect.insort(ends, end)
    if not kept:
        return []
    lo = min(s.start for s in shards)
    hi = max(s.end for s in shards)
    best_at_end = {}
    for g in kept:
        cur = best_at_end.get(g[2])
        if cur is None or (g[1], g[0]) < (cur[1], cur[0]):
            best_at_end[g[2]] = g
    step4 = sorted((g[1], g[2]) for g in kept)
    holes = []
    reach = lo - 1
    for s, e in step4:
        if s > reach + 1:
            holes.append((reach + 1, s - 1))
        reach = max(reach, e)
    if reach < hi:
        holes.append((reach + 1, hi))
    kept.sort(key=lambda g: (g[1], g[2], g[0]))
    for first, last in holes:
        best_at_end[first - 1][2] = last
    out = [Shard(i, s, e) for i, s, e in kept]
    return out


def _score(key, sid):
    return hashlib.sha256(f"{key}\x00{sid}".encode()).digest()


class ShardRouter:
    def __init__(self):
        self._ids = set()

    def add_shard(self, shard_id):
        if shard_id in self._ids:
            raise ValueError(shard_id)
        self._ids.add(shard_id)

    def remove_shard(self, shard_id):
        if shard_id not in self._ids:
            raise ValueError(shard_id)
        self._ids.remove(shard_id)

    def locate(self, key):
        if not self._ids:
            raise LookupError("no shards")
        return max(self._ids, key=lambda sid: (_score(key, sid), sid))
