"""Independent restatements of Part 1's rule, used to compute expected answers. A shard is a tuple (id, start, end)."""


class Shard:
    """The tests' own shard: what is passed in when the answer has no usable Shard class, and how shards are shown."""

    def __init__(self, id, start, end):
        self.id, self.start, self.end = id, start, end

    def __repr__(self):
        return f"Shard({self.id!r}, {self.start}, {self.end})"

    def __eq__(self, other):
        return isinstance(other, Shard) and (self.id, self.start, self.end) == (other.id, other.start, other.end)

    __hash__ = None


def _order(specs):
    return sorted(specs, key=lambda s: (s[1], s[2], s[0]))


def _fill_holes(specs, kept):
    """Step 5, given the step-4 ranges in `kept` (list of [id, start, end]); returns the final sorted tuples."""
    if not kept:
        return []
    lo, hi = min(s[1] for s in specs), max(s[2] for s in specs)
    holes, reach = [], lo - 1
    for _, ks, ke in sorted(kept, key=lambda g: g[1]):
        if ks > reach + 1:
            holes.append((reach + 1, ks - 1))
        reach = max(reach, ke)
    if reach < hi:
        holes.append((reach + 1, hi))
    step4 = [tuple(g) for g in kept]
    final = {g[0]: [g[1], g[2]] for g in kept}
    for first, last in holes:
        owner = min((g for g in step4 if g[2] == first - 1), key=lambda g: (g[1], g[0]))
        final[owner[0]][1] = last
    return sorted(((i, s, e) for i, (s, e) in final.items()), key=lambda t: (t[1], t[2], t[0]))


def brute(limit, specs):
    """Steps 2-4 key by key: the first key of [start, end] whose coverage over the kept shards is below limit.
    Only for small coordinates."""
    kept = []
    for sid, start, end in _order(specs):
        for k in range(start, end + 1):
            if sum(1 for _, ks, ke in kept if ks <= k <= ke) < limit:
                kept.append([sid, k, end])
                break
    if not kept:
        return []
    # step 5 key by key as well
    lo, hi = min(s[1] for s in specs), max(s[2] for s in specs)
    step4 = [tuple(g) for g in kept]
    covered = [any(ks <= k <= ke for _, ks, ke in step4) for k in range(lo, hi + 1)]
    final = {g[0]: [g[1], g[2]] for g in kept}
    k = lo
    while k <= hi:
        if covered[k - lo]:
            k += 1
            continue
        last = k
        while last + 1 <= hi and not covered[last + 1 - lo]:
            last += 1
        owner = min((g for g in step4 if g[2] == k - 1), key=lambda g: (g[1], g[0]))
        final[owner[0]][1] = last
        k = last + 1
    return sorted(((i, s, e) for i, (s, e) in final.items()), key=lambda t: (t[1], t[2], t[0]))


def sweep(limit, specs):
    """Same rule for any coordinates. Coverage over the kept shards only drops just after a kept end, so the key
    step 2 looks for is either start itself or some kept end + 1 inside (start, end]."""
    kept = []
    for sid, start, end in _order(specs):
        candidates = sorted({start} | {ke + 1 for _, _, ke in kept if start < ke + 1 <= end})
        for c in candidates:
            if sum(1 for _, ks, ke in kept if ks <= c <= ke) < limit:
                kept.append([sid, c, end])
                break
    return _fill_holes(specs, kept)


def random_specs(r, n, lo, hi, max_len=None):
    ids = r.sample([chr(ord("a") + i) for i in range(26)], n) if n <= 26 else [f"s{i}" for i in range(n)]
    out = []
    for sid in ids:
        a = r.randint(lo, hi)
        b = r.randint(lo, hi) if max_len is None else min(hi, a + r.randint(0, max_len))
        out.append((sid, min(a, b), max(a, b)))
    return out
