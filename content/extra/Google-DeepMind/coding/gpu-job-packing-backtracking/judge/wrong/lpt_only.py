# expect: WA
# Part 3 returns the greedy LPT plan without searching.
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


def min_peak(jobs, k, stats=None, use_bound=True):
    """(smallest peak, an assignment reaching it). use_bound=False only exists so the checks can measure the bound."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i])
    loads = [0] * k
    assign = [0] * n
    for i in order:                          # greedy LPT: largest job onto the least loaded GPU
        g = loads.index(min(loads))
        loads[g] += jobs[i]
        assign[i] = g
    best, best_assign = max(loads), assign[:]
    lower = max(max(jobs, default=0), -(-sum(jobs) // k))
    loads = [0] * k
    nodes = 0

    def place(pos, peak):
        nonlocal best, best_assign, nodes
        nodes += 1
        if pos == n:
            if peak < best:                  # always true here: every step on the way stayed below best
                best, best_assign = peak, assign[:]
            return
        i = order[pos]
        tried = set()
        for g in range(k):
            new = loads[g] + jobs[i]
            if loads[g] in tried or (use_bound and new >= best):     # NOTE: >= because only a strictly better plan helps
                continue
            tried.add(loads[g])
            loads[g] = new
            assign[i] = g
            place(pos + 1, max(peak, new))
            loads[g] = new - jobs[i]
            if best == lower:                # nothing can beat the lower bound: unwind the whole search
                return

    if stats is not None:
        stats["nodes"] = nodes               # search nodes only; the greedy pass is not counted
    return best, best_assign
