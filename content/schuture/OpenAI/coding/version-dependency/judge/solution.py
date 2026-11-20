import operator
from collections import deque


class CountingProbe:
    """Wraps is_supported and counts how many times it is actually called."""
    def __init__(self, is_supported):
        self._is_supported = is_supported
        self.calls = 0

    def __call__(self, version: str) -> bool:
        self.calls += 1
        return self._is_supported(version)


class Registry:
    """The available packages: name -> version -> requires."""
    def __init__(self):
        self._versions = {}

    def add_version(self, name: str, version: str, requires: dict[str, str] | None = None) -> None:
        """Registers one available (name, version) and its direct dependencies."""
        self._versions.setdefault(name, {})[version] = dict(requires or {})

    def versions(self, name: str) -> list[str]:
        """All versions registered for name (empty if the name is unknown)."""
        return list(self._versions.get(name, {}))

    def requires(self, name: str, version: str) -> dict[str, str]:
        """The constraints declared by exactly this (name, version)."""
        return self._versions[name][version]


# ---------------------------------------------------------------- Part 1

def parse_version(version):
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)   # NOTE: int() drops leading zeros; never compare the raw strings


def first_true_index(n, predicate):
    """Leftmost index in range(n) with predicate(i) True, assuming predicate is monotone; else None."""
    left, right, result = 0, n - 1, None
    while left <= right:
        mid = (left + right) // 2
        if predicate(mid):
            result = mid
            right = mid - 1          # an earlier True may still exist -- keep searching left
        else:
            left = mid + 1
    return result


def find_earliest_supported_monotone(versions, is_supported):
    ordered = sorted(versions, key=parse_version)   # NOTE: numeric order; "1.10.0" < "1.9.0" as plain strings
    idx = first_true_index(len(ordered), lambda i: is_supported(ordered[i]))
    return ordered[idx] if idx is not None else None


# ---------------------------------------------------------------- Part 2

def find_earliest_supported(versions, is_supported):
    for v in sorted(versions, key=parse_version):
        if is_supported(v):        # sorted, so this first True is already the earliest
            return v
    return None


# ---------------------------------------------------------------- Part 3

def find_earliest_supported_hierarchical(versions, is_supported):
    cache = {}
    def probe(v):
        if v not in cache:                     # NOTE: never call is_supported twice for one version
            cache[v] = is_supported(v)
        return cache[v]

    tree = {}                                   # major -> minor -> patch-sorted versions
    for v in versions:
        major, minor, _ = parse_version(v)
        tree.setdefault(major, {}).setdefault(minor, []).append(v)
    for minors in tree.values():
        for patches in minors.values():
            patches.sort(key=parse_version)

    majors = sorted(tree)
    rep = lambda i: max((p[-1] for p in tree[majors[i]].values()), key=parse_version)   # NOTE: latest = rep
    mi = first_true_index(len(majors), lambda i: probe(rep(i)))
    if mi is None:
        return None

    minors = sorted(tree[majors[mi]])
    ni = first_true_index(len(minors), lambda i: probe(tree[majors[mi]][minors[i]][-1]))
    patches = tree[majors[mi]][minors[ni]]
    pi = first_true_index(len(patches), lambda i: probe(patches[i]))
    return patches[pi]


# ---------------------------------------------------------------- Part 4

_OPS = {"==": operator.eq, ">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


def satisfies(version, constraint):
    for op in ("==", ">=", "<=", ">", "<"):        # NOTE: two-character operators first, or ">=" is read as ">"
        if constraint.startswith(op):
            return _OPS[op](parse_version(version), parse_version(constraint[len(op):]))
    raise ValueError(f"bad constraint: {constraint!r}")


def install_order(registry, chosen):
    """Kahn's algorithm over the chosen versions; None if their dependencies form a cycle."""
    indegree = {name: 0 for name in chosen}
    dependents = {name: [] for name in chosen}
    for name, version in chosen.items():
        for dep in registry.requires(name, version):
            dependents[dep].append(name)
            indegree[name] += 1
    queue = deque(sorted(name for name in chosen if indegree[name] == 0))
    order = []
    while queue:
        name = queue.popleft()
        order.append(f"{name}@{chosen[name]}")
        for other in sorted(dependents[name]):
            indegree[other] -= 1
            if indegree[other] == 0:
                queue.append(other)
    return order if len(order) == len(chosen) else None      # NOTE: a shorter order means a cycle


def resolve(registry, root):
    def search(chosen, pending):
        """chosen: name -> version so far; pending: declared (name, constraint) pairs not yet enforced."""
        if not pending:
            return install_order(registry, chosen)            # a cyclic choice fails here and the search goes on
        (name, constraint), rest = pending[0], pending[1:]
        if name in chosen:                                     # version already fixed: the constraint is only checked
            return search(chosen, rest) if satisfies(chosen[name], constraint) else None
        for version in sorted(registry.versions(name), key=parse_version, reverse=True):   # newest first
            if satisfies(version, constraint):
                found = search({**chosen, name: version}, rest + list(registry.requires(name, version).items()))
                if found is not None:
                    return found
        return None                                            # NOTE: the caller then tries its own next version

    return search({}, [(root, ">=0.0.0")])
