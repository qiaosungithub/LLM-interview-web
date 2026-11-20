# expect: WA
# Part 4 never undoes a choice when it backtracks, so packages from an abandoned branch stay chosen and are listed.
from collections import defaultdict, deque


def parse_version(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def _first_true(items, test):
    """Index of the first item for which test is True, assuming False...False True...True; len(items) if none."""
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if test(items[mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo


def find_earliest_supported_monotone(versions, is_supported):
    ordered = sorted(versions, key=parse_version)
    i = _first_true(ordered, is_supported)
    return ordered[i] if i < len(ordered) else None


def find_earliest_supported(versions, is_supported):
    for v in sorted(versions, key=parse_version):
        if is_supported(v):
            return v
    return None


def find_earliest_supported_hierarchical(versions, is_supported):
    memo = {}

    def probe(v):
        if v not in memo:
            memo[v] = bool(is_supported(v))
        return memo[v]

    tree = defaultdict(lambda: defaultdict(list))
    for v in versions:
        a, b, _ = parse_version(v)
        tree[a][b].append(v)
    majors = sorted(tree)
    if not majors:
        return None

    def major_rep(m):
        return max((v for group in tree[m].values() for v in group), key=parse_version)

    mi = _first_true(majors, lambda m: probe(major_rep(m)))
    if mi == len(majors):
        return None
    groups = tree[majors[mi]]
    minors = sorted(groups)
    ni = _first_true(minors, lambda n: probe(max(groups[n], key=parse_version)))
    if ni == len(minors):
        return None
    patches = sorted(groups[minors[ni]], key=parse_version)
    pi = _first_true(patches, probe)
    return patches[pi] if pi < len(patches) else None


_OPS = {
    "==": lambda a, b: a == b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


def _holds(version, constraint):
    for op in ("==", ">=", "<=", ">", "<"):
        if constraint.startswith(op):
            return _OPS[op](parse_version(version), parse_version(constraint[len(op):].strip()))
    raise ValueError(constraint)


def resolve(registry, root):
    chosen = {}

    def compatible(name, version):
        for other, ov in chosen.items():
            c = registry.requires(other, ov).get(name)
            if c is not None and not _holds(version, c):
                return False
        for dep, c in registry.requires(name, version).items():
            if dep in chosen and not _holds(chosen[dep], c):
                return False
        return True

    def order():
        indeg = {n: 0 for n in chosen}
        users = defaultdict(list)
        for n, v in chosen.items():
            for dep in registry.requires(n, v):
                indeg[n] += 1
                users[dep].append(n)
        queue = deque(sorted(n for n, d in indeg.items() if d == 0))
        out = []
        while queue:
            n = queue.popleft()
            out.append(f"{n}@{chosen[n]}")
            for u in users[n]:
                indeg[u] -= 1
                if indeg[u] == 0:
                    queue.append(u)
        return out if len(out) == len(chosen) else None

    def search(todo):
        todo = [n for n in todo if n not in chosen]
        if not todo:
            return order()
        name, rest = todo[0], todo[1:]
        for version in sorted(registry.versions(name), key=parse_version, reverse=True):
            if compatible(name, version):
                chosen[name] = version
                found = search(rest + list(registry.requires(name, version)))
                if found is not None:
                    return found
        return None

    return search([root])
