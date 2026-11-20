"""Helpers for tests.py: brute forces that are obviously right, and checks of a returned assignment."""
import itertools


def loads_of(jobs, k, assign):
    loads = [0] * k
    for size, g in zip(jobs, assign):
        loads[g] += size
    return loads


def brute_first(jobs, k, cap):
    """The lexicographically smallest valid assignment, or None: itertools.product yields assignments in
    lexicographic order, so the first valid one is the answer. Only for small k ** n."""
    for assign in itertools.product(range(k), repeat=len(jobs)):
        if max(loads_of(jobs, k, assign), default=0) <= cap:
            return list(assign)
    return None


def brute_peak(jobs, k):
    """Smallest peak over every assignment. Only for small k ** n."""
    return min(max(loads_of(jobs, k, a), default=0) for a in itertools.product(range(k), repeat=len(jobs)))


def gpus_needed(jobs, cap):
    """Fewest GPUs of size cap that hold all jobs (every job must be <= cap). Bitmask DP: best[mask] is the smallest
    (GPUs used, load of the last GPU) over every order of filling GPUs one after another with the jobs in mask;
    the smaller pair is never worse for the jobs still to come. O(2^n * n)."""
    n = len(jobs)
    best = [(n + 2, 0)] * (1 << n)
    best[0] = (1, 0)
    for mask in range(1 << n):
        used, load = best[mask]
        for i in range(n):
            bit = 1 << i
            if mask & bit:
                continue
            cand = (used, load + jobs[i]) if load + jobs[i] <= cap else (used + 1, jobs[i])
            if cand < best[mask | bit]:
                best[mask | bit] = cand
    return best[-1][0]


def feasible(jobs, k, cap):
    return max(jobs, default=0) <= cap and gpus_needed(jobs, cap) <= k


def dp_peak(jobs, k):
    """Smallest peak: binary search on cap with the DP above."""
    lo = max(max(jobs, default=0), -(-sum(jobs) // k))
    hi = max(lo, sum(jobs))
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(jobs, k, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def lpt_peak(jobs, k):
    """The greedy of the page: largest job first, onto the least loaded GPU."""
    loads = [0] * k
    for size in sorted(jobs, reverse=True):
        loads[loads.index(min(loads))] += size
    return max(loads, default=0)


def assign_problem(jobs, k, got, cap=None):
    """None when got is an assignment of the n jobs to GPUs 0..k-1 (whose loads stay <= cap if cap is given),
    else a Chinese sentence saying what is wrong."""
    if not isinstance(got, (list, tuple)):
        return f"应当返回一个长度为 n = {len(jobs)} 的列表，得到的是 {type(got).__name__}"
    if len(got) != len(jobs):
        return f"assign 的长度应当是作业数 n = {len(jobs)}，得到 {len(got)}"
    for i, g in enumerate(got):
        if not isinstance(g, int) or not 0 <= g < k:
            return f"assign[{i}] = {g!r}，不是 0 到 {k - 1} 之间的 GPU 编号"
    if cap is not None:
        loads = loads_of(jobs, k, got)
        for g, load in enumerate(loads):
            if load > cap:
                members = [i for i, x in enumerate(got) if x == g]
                sizes = " + ".join(str(jobs[i]) for i in members)
                return f"GPU {g} 上放了作业 {members}，显存 {sizes} = {load}，超过了 cap = {cap}"
    return None
