# expect: WA
# Hash-partitions the raw elements instead of local counts: traffic grows with N, not D.
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
    for value in shard:
        outgoing[_owner(value)].append(value)
    for dst in range(NUM_RANKS):
        if dst != rank:
            send(dst, outgoing[dst])    # NOTE: send even [] -- recv(src) below waits specifically for dst

    owned = _local_histogram(outgoing[rank])
    for src in range(NUM_RANKS):
        if src != rank:
            for value in recv(src):
                owned[value] = owned.get(value, 0) + 1

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
    local_sorted = sorted(shard)
    local_min = local_sorted[0] if local_sorted else None
    local_max = local_sorted[-1] if local_sorted else None

    if rank != 0:
        send(0, (local_min, local_max, len(shard)))
    else:
        lo, hi, total_n = local_min, local_max, len(shard)
        for src in range(1, NUM_RANKS):
            r_min, r_max, r_len = recv(src)
            total_n += r_len
            if r_len:                      # NOTE: an empty shard contributes no bound on lo/hi
                lo = r_min if lo is None else min(lo, r_min)
                hi = r_max if hi is None else max(hi, r_max)
    barrier()                              # every rank now has (or, on rank 0, knows) the global range

    if rank != 0:
        while True:
            guess = recv(0)
            if guess is None:              # NOTE: rank 0's signal that the answer is already found
                return None
            less = sum(1 for v in local_sorted if v < guess)
            equal = sum(1 for v in local_sorted if v == guess)
            send(0, (less, equal))

    k = (total_n - 1) // 2
    answer = None
    while answer is None:
        guess = (lo + hi) // 2
        for dst in range(1, NUM_RANKS):
            send(dst, guess)
        total_less = sum(1 for v in local_sorted if v < guess)
        total_equal = sum(1 for v in local_sorted if v == guess)
        for src in range(1, NUM_RANKS):
            r_less, r_equal = recv(src)
            total_less += r_less
            total_equal += r_equal
        if total_less > k:
            hi = guess - 1
        elif total_less + total_equal > k:
            answer = guess
        else:
            lo = guess + 1
    for dst in range(1, NUM_RANKS):
        send(dst, None)                    # NOTE: every worker is blocked in recv(0) -- release them all
    return answer


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
