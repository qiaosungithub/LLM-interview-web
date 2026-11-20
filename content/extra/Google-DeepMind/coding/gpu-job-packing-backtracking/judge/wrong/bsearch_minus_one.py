# expect: WA
# Part 3 binary searches the peak but sets hi = mid - 1 when mid is feasible, so it can step
# below the answer and return an infeasible peak with a plan for a larger one.
def pack(jobs, k, cap, stats=None):
    n = len(jobs)
    loads = [0] * k           # loads[g]: memory used on GPU g by jobs 0..i-1
    assign = [0] * n
    nodes = 0

    def place(i):
        nonlocal nodes
        nodes += 1
        if i == n:
            return True
        for g in range(k):    # smaller GPU numbers first, so the first full assignment is the smallest one
            if loads[g] + jobs[i] <= cap:
                loads[g] += jobs[i]
                assign[i] = g
                if place(i + 1):
                    return True
                loads[g] -= jobs[i]          # undo before trying the next GPU
        return False

    found = place(0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None


def pack_fast(jobs, k, cap, stats=None, sort_desc=True, skip_equal=True):
    """Any valid assignment or None. The two switches exist only so that the checks can measure each pruning."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i]) if sort_desc else list(range(n))
    loads = [0] * k
    assign = [0] * n
    nodes = 0

    def place(pos):
        nonlocal nodes
        nodes += 1
        if pos == n:
            return True
        i = order[pos]                       # the pos-th largest job
        tried = set()                        # loads already tried for this job at this node
        for g in range(k):
            if loads[g] + jobs[i] > cap:
                continue
            if skip_equal:
                if loads[g] in tried:        # NOTE: same load as a GPU tried before, so the subtree is a mirror image
                    continue
                tried.add(loads[g])
            loads[g] += jobs[i]
            assign[i] = g
            if place(pos + 1):
                return True
            loads[g] -= jobs[i]
        return False

    found = sum(jobs) <= k * cap and max(jobs, default=0) <= cap and place(0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None


def min_peak(jobs, k, stats=None):
    """Binary search on the peak between the lower bound and the total, deciding each value with pack_fast."""
    lo = max(max(jobs, default=0), -(-sum(jobs) // k))
    hi = max(lo, sum(jobs))
    best_assign = [0] * len(jobs)
    nodes = 0
    while lo < hi:
        mid = (lo + hi) // 2
        st = {}
        got = pack_fast(jobs, k, mid, st)
        nodes += st["nodes"]
        if got is not None:
            hi = mid - 1                     # mid works, look for something smaller
            best_assign = got
        else:
            lo = mid + 1
    if stats is not None:
        stats["nodes"] = nodes
    return lo, best_assign
