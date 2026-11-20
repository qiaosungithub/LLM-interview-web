# expect: WA
# The real probe is always (lo + hi) // 2, even when an earlier filler already used it.
def find_secret(n, check):
    if n == 1:
        return 1                        # only one candidate: nothing to compare
    lo, hi = 1, n
    used = set()
    right_scan, left_scan = n, 1        # frontiers for filler values outside [lo, hi]

    def real_probe():
        return (lo + hi) // 2

    def filler():
        nonlocal right_scan, left_scan
        while right_scan > hi:                       # values above hi are provably irrelevant
            cand, right_scan = right_scan, right_scan - 1
            if cand not in used:
                return cand
        while left_scan < lo:                        # then values below lo
            cand, left_scan = left_scan, left_scan + 1
            if cand not in used:
                return cand
        for x in range(lo, hi + 1):      # NOTE: the first filler always lands here (nothing is
            if x not in used:            # outside [lo, hi] yet), and so do fillers once [1, n] runs dry
                return x

    pending = real_probe()
    used.add(pending)
    check(pending)
    is_real = True                      # role of `pending`: was it this round's bisection probe?

    while True:
        if lo == hi:
            return lo
        nxt = filler() if is_real else real_probe()
        used.add(nxt)
        r = check(nxt)
        # NOTE: narrow on ANY revealed comparison, real or filler -- a filler forced inside
        # [lo, hi] (only for tiny n) is still honest information about secret.
        if r == 0:
            return pending
        elif r == -1:
            lo = max(lo, pending + 1)
        elif r == 1:
            hi = min(hi, pending - 1)
        pending = nxt
        is_real = not is_real
        if lo == hi:
            return lo

def find_secret_batched(n, check_batch):
    if n == 1:
        return 1
    lo, hi = 1, n
    used = set()
    right_scan, left_scan = n, 1

    def nudge(target, avoid):
        for delta in range(hi - lo + 1):
            for cand in (target - delta, target + delta):
                if lo <= cand <= hi and cand not in used and cand not in avoid:
                    return cand
        return None

    def real_points():
        if hi - lo + 1 <= 2:                  # NOTE: one point already settles two candidates,
            p = nudge((lo + hi) // 2, set())  # and for tiny n it leaves the filler a value to use
            return [p] if p is not None else []
        t1, t2 = lo + (hi - lo) // 3, lo + 2 * (hi - lo) // 3
        p1 = nudge(t1, set())
        p2 = nudge(t2, {p1} if p1 is not None else set())
        return [p for p in (p1, p2) if p is not None]

    def filler():
        nonlocal right_scan, left_scan
        while right_scan > hi:
            cand, right_scan = right_scan, right_scan - 1
            if cand not in used:
                return cand
        while left_scan < lo:
            cand, left_scan = left_scan, left_scan + 1
            if cand not in used:
                return cand
        for x in range(lo, hi + 1):
            if x not in used:
                return x

    def apply(xs, rs):
        nonlocal lo, hi
        found = None
        for x, r in zip(xs, rs):
            if r == -1: lo = max(lo, x + 1)
            elif r == 1: hi = min(hi, x - 1)
            else: found = x
        return found

    pending = real_points()
    used.update(pending)
    check_batch(pending)
    is_real = True

    while True:
        if lo == hi:
            return lo
        nxt = [filler()] if is_real else real_points()
        used.update(nxt)
        rs = check_batch(nxt)
        found = apply(pending, rs)
        if found is not None:
            return found
        pending = nxt
        is_real = not is_real
        if lo == hi:
            return lo

def solve_games(bounds, check_round):
    lo = {g: 1 for g in bounds}
    hi = dict(bounds)
    solved = {g: lo[g] == hi[g] for g in bounds}     # n[g] == 1: nothing to ask

    def apply(batch, results):
        if not results:
            return
        for g, r in results.items():
            x = batch[g]
            if r == -1: lo[g] = max(lo[g], x + 1)
            elif r == 1: hi[g] = min(hi[g], x - 1)
            else: lo[g] = hi[g] = x
            solved[g] = lo[g] == hi[g]

    pending, is_real = {}, True
    while not all(solved.values()):
        # NOTE: recomputed fresh every real round, so a game that just finished simply drops out
        guesses = {g: (lo[g] + hi[g]) // 2 for g in bounds if not solved[g]} if is_real else {}
        results = check_round(guesses)
        apply(pending, results)
        pending, is_real = guesses, not is_real
    return {g: lo[g] for g in bounds}
