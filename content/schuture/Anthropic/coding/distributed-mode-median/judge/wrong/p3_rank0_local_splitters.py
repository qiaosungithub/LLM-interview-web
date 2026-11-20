# expect: WA
# Rank 0 picks the splitters from its own shard only; when shards hold different value ranges almost everything lands on one rank.
import bisect
import heapq
from collections import Counter

P = 10
MASK = (1 << 64) - 1


def _owner(v: int) -> int:
    # splitmix64 finaliser over the value (folded to 64 bits), so arithmetic patterns in the values do not matter
    x = (v ^ (v >> 64) ^ 0x5DEECE66D) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    x ^= x >> 31
    return x % P


def global_mode(rank, shard, send, recv, barrier):
    parts = [{} for _ in range(P)]
    for v, c in Counter(shard).items():
        parts[_owner(v)][v] = c
    for dst in range(P):
        send(dst, parts[dst])            # every rank sends to every rank, empty or not: the receiver recvs all P
    total = Counter()
    for src in range(P):
        total.update(recv(src))
    best = None
    if total:
        top = max(total.values())
        best = (top, min(v for v, c in total.items() if c == top))
    send(0, best)
    if rank != 0:
        return None
    cands = [b for b in (recv(src) for src in range(P)) if b is not None]
    top = max(c for c, _ in cands)
    return min(v for c, v in cands if c == top)


def global_median(rank, shard, send, recv, barrier):
    mine = sorted(shard)
    send(0, (mine[0], mine[-1], len(mine)) if mine else None)
    if rank == 0:
        info = [x for x in (recv(src) for src in range(P)) if x is not None]
        lo, hi = min(x[0] for x in info), max(x[1] for x in info)
        need = (sum(x[2] for x in info) - 1) // 2 + 1        # smallest value with at least k+1 elements <= it
        while lo < hi:
            mid = (lo + hi) // 2
            for dst in range(1, P):
                send(dst, mid)
            cnt = bisect.bisect_right(mine, mid) + sum(recv(src) for src in range(1, P))
            if cnt >= need:
                hi = mid
            else:
                lo = mid + 1
        for dst in range(1, P):
            send(dst, None)
        return lo
    while True:
        mid = recv(0)
        if mid is None:
            return None
        send(0, bisect.bisect_right(mine, mid))


STEP = 48


def sample_sort(rank, shard, send, recv, barrier):
    mine = sorted(shard)
    if rank == 0:
        samples = mine[STEP // 2::STEP] or mine
        m = len(samples)
        splitters = [samples[(i * m) // P] for i in range(1, P)] if m else []
        for dst in range(P):
            send(dst, splitters)
    splitters = recv(0)
    cuts = [0] + [bisect.bisect_right(mine, s) for s in splitters] + [len(mine)]
    if not splitters:
        cuts = [0] * P + [len(mine)]
    for dst in range(P):
        send(dst, mine[cuts[dst]:cuts[dst + 1]])
    runs = [recv(src) for src in range(P)]
    return list(heapq.merge(*runs))
