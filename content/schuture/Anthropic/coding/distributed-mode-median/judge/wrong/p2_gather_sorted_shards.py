# expect: WA
# Every rank sends its sorted shard to rank 0, which merges them: bytes grow with N.
import bisect
import hashlib
import random

NUM_RANKS = 10


def _local_histogram(shard: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for v in shard:
        counts[v] = counts.get(v, 0) + 1
    return counts


HASH_SEED = b"global-mode-shuffle"   # any constant all ranks agree on, picked without looking at the data


def _owner(value: int) -> int:
    # NOTE: a seeded hash shared by every rank. Not `value % NUM_RANKS`: values sharing one residue would
    # all land on one owner. Not the built-in hash(): it is the identity on small ints.
    digest = hashlib.blake2b(str(value).encode(), digest_size=8, key=HASH_SEED).digest()
    return int.from_bytes(digest, "big") % NUM_RANKS


def global_mode(rank, shard, send, recv, barrier):
    local = _local_histogram(shard)
    outgoing = [[] for _ in range(NUM_RANKS)]
    for value, count in local.items():
        outgoing[_owner(value)].append((value, count))
    for dst in range(NUM_RANKS):
        if dst != rank:
            send(dst, outgoing[dst])    # NOTE: send even [] -- recv(src) below waits specifically for dst

    owned = dict(outgoing[rank])
    for src in range(NUM_RANKS):
        if src != rank:
            for value, count in recv(src):
                owned[value] = owned.get(value, 0) + count

    if owned:
        best_count = max(owned.values())
        winner = min(v for v, c in owned.items() if c == best_count)
        candidate = (winner, owned[winner])
    else:
        candidate = None                # NOTE: this owner's slice of the key space is empty

    if rank != 0:
        send(0, candidate)
        return None
    candidates = [candidate] + [recv(src) for src in range(1, NUM_RANKS)]
    best_value, _ = min((c for c in candidates if c is not None), key=lambda vc: (-vc[1], vc[0]))
    return best_value


def global_median(rank, shard, send, recv, barrier):
    import heapq
    local_sorted = sorted(shard)
    if rank != 0:
        send(0, local_sorted)
        return None
    runs = [local_sorted] + [recv(src) for src in range(1, NUM_RANKS)]
    total_n = sum(len(run) for run in runs)
    k = (total_n - 1) // 2
    for i, v in enumerate(heapq.merge(*runs)):
        if i == k:
            return v


SAMPLE_SIZE = 8


def _bucket_of(value: int, splitters: list[int]) -> int:
    return bisect.bisect_right(splitters, value)   # NOTE: ties go to the bucket starting at that splitter


def sample_sort(rank, shard, send, recv, barrier):
    rng = random.Random(rank)     # NOTE: seeded by rank alone, so the sample is reproducible regardless
    sample = rng.sample(shard, min(len(shard), SAMPLE_SIZE))   #      of thread-scheduling order

    if rank != 0:
        send(0, sample)
    else:
        combined = sorted(sample + [v for src in range(1, NUM_RANKS) for v in recv(src)])
        splitters = [combined[i * len(combined) // NUM_RANKS] for i in range(1, NUM_RANKS)] if combined else []
        for dst in range(1, NUM_RANKS):
            send(dst, splitters)

    if rank != 0:
        splitters = recv(0)
    barrier()                     # everyone now has the same splitters before the exchange begins

    outgoing = [[] for _ in range(NUM_RANKS)]
    for v in shard:
        outgoing[_bucket_of(v, splitters)].append(v)
    for dst in range(NUM_RANKS):
        if dst != rank:
            send(dst, outgoing[dst])   # NOTE: send even [] -- every rank recv's from every other rank once
    mine = outgoing[rank]
    for src in range(NUM_RANKS):
        if src != rank:
            mine.extend(recv(src))
    mine.sort()
    return mine
