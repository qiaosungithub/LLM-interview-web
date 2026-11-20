# expect: TLE
# Correct but walks the key space one key at a time (coverage per key, holes per key); keys reach 1e9, so too slow.
import hashlib


class Shard:
    def __init__(self, id, start, end):
        self.id, self.start, self.end = id, start, end


def rebalance(limit, shards):
    if not shards:
        return []
    order = sorted(shards, key=lambda sh: (sh.start, sh.end, sh.id))
    kept = []
    for sh in order:
        for k in range(sh.start, sh.end + 1):
            if sum(1 for g in kept if g.start <= k <= g.end) < limit:
                kept.append(Shard(sh.id, k, sh.end))
                break
    step4 = [(g.start, g.end) for g in kept]
    lo, hi = min(sh.start for sh in shards), max(sh.end for sh in shards)
    k = lo
    while k <= hi:
        if any(s <= k <= e for s, e in step4):
            k += 1
            continue
        last = k
        while last + 1 <= hi and not any(s <= last + 1 <= e for s, e in step4):
            last += 1
        owner = min((g for g in kept if (g.start, g.end) in step4 and g.end == k - 1), key=lambda g: (g.start, g.id))
        owner.end = last
        k = last + 1
    return sorted(kept, key=lambda g: (g.start, g.end, g.id))


def _score(shard_id, key):
    return hashlib.sha256(f"{shard_id}|{key}".encode()).digest()


class ShardRouter:
    """Rendezvous (highest random weight) hashing: no ring, O(N) per locate."""

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
        return max(self._shards, key=lambda s: _score(s, key))
