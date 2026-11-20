"""Helpers for tests.py: an independent restatement of the rules (numeric order, earliest supported version,
constraint checks, a brute force over every version combination, a validity checker for an install order) and
random input generators."""
import itertools
import re


def parse(version):
    a, b, c = version.split(".")
    return int(a), int(b), int(c)


class Registry:
    """The statement's Registry, verbatim; tests build registries with it and pass them to the answer."""
    def __init__(self):
        self._versions = {}

    def add_version(self, name, version, requires=None):
        self._versions.setdefault(name, {})[version] = dict(requires or {})

    def versions(self, name):
        return list(self._versions.get(name, {}))

    def requires(self, name, version):
        return self._versions[name][version]


# ---------------------------------------------------------------- Parts 1-3

class Probe:
    """is_supported for a fixed table; records every call. A version outside the table answers False and is
    reported by the test afterwards."""

    def __init__(self, support):
        self.support = dict(support)
        self.calls = []

    def __call__(self, version):
        self.calls.append(version)
        return bool(self.support.get(version, False))

    def unknown(self):
        return [v for v in self.calls if v not in self.support]

    def repeated(self):
        seen, out = set(), []
        for v in self.calls:
            if v in seen and v not in out:
                out.append(v)
            seen.add(v)
        return out


def earliest(support):
    """Numerically smallest version whose value is True, or None (a plain scan)."""
    hits = [v for v, ok in support.items() if ok]
    return min(hits, key=parse) if hits else None


def fmt(t, r=None, zeros=0.0):
    """A version string for the tuple t; each part gets leading zeros with probability `zeros`."""
    parts = []
    for x in t:
        s = str(x)
        if r is not None and r.random() < zeros:
            s = "0" * r.randint(1, 2) + s
        parts.append(s)
    return ".".join(parts)


def random_tuples(r, n, top=(3, 12, 12)):
    """n distinct (major, minor, patch) tuples with parts drawn from 0..top."""
    n = min(n, (top[0] + 1) * (top[1] + 1) * (top[2] + 1))
    out = set()
    while len(out) < n:
        out.add((r.randint(0, top[0]), r.randint(0, top[1]), r.randint(0, top[2])))
    return sorted(out)


def monotone_support(r, tuples, zeros=0.3):
    """{version: bool} for the given tuples (in random order), False up to a random cut and True after it."""
    cut = r.randint(0, len(tuples))
    pairs = [(fmt(t, r, zeros), i >= cut) for i, t in enumerate(sorted(tuples))]
    r.shuffle(pairs)
    return dict(pairs)


def any_support(r, tuples, zeros=0.3, p_true=0.4):
    pairs = [(fmt(t, r, zeros), r.random() < p_true) for t in sorted(tuples)]
    r.shuffle(pairs)
    return dict(pairs)


def tree_tuples(r, n_major, max_minor, max_patch, top=None):
    """A ragged tree of distinct versions: n_major majors, each with 1..max_minor minors, each with
    1..max_patch patches. Numbers are drawn sparsely (not 0, 1, 2, ...) so that string order differs."""
    top = top or max(n_major, max_minor, max_patch) * 2 + 3
    out = []
    for major in sorted(r.sample(range(top), n_major)):
        for minor in sorted(r.sample(range(top), r.randint(1, max_minor))):
            for patch in sorted(r.sample(range(top), r.randint(1, max_patch))):
                out.append((major, minor, patch))
    return out


def a1_a3_support(r, tuples, zeros=0.3, cuts=None):
    """{version: bool} satisfying A1-A3 for the given tuples. Every such pattern is reachable: a False major
    representative forces its whole major False, a False minor representative its whole group.
    `cuts` = (major index, minor index, patch index) of the first True, instead of drawing it at random;
    the index of the first True major may equal the number of majors (nothing supported)."""
    tree = {}
    for t in sorted(tuples):
        tree.setdefault(t[0], {}).setdefault(t[1], []).append(t)
    majors = sorted(tree)
    cm = r.randint(0, len(majors)) if cuts is None else cuts[0]
    value = {}
    for i, major in enumerate(majors):
        minors = sorted(tree[major])
        if i < cm:
            cn = len(minors)
        elif cuts is not None and i == cm:
            cn = cuts[1] % len(minors)
        else:
            cn = r.randint(0, len(minors) - 1)
        for j, minor in enumerate(minors):
            patches = tree[major][minor]
            if j < cn:
                cp = len(patches)
            elif cuts is not None and i == cm and j == cn:
                cp = cuts[2] % len(patches)
            else:
                cp = r.randint(0, len(patches) - 1)
            for k, t in enumerate(patches):
                value[t] = k >= cp
    pairs = [(fmt(t, r, zeros), value[t]) for t in sorted(tuples)]
    r.shuffle(pairs)
    return dict(pairs)


def shape(versions):
    """(M, N_minor, P) as the statement defines them."""
    tree = {}
    for v in versions:
        a, b, _ = parse(v)
        tree.setdefault(a, {}).setdefault(b, 0)
        tree[a][b] += 1
    M = len(tree)
    N = max((len(ms) for ms in tree.values()), default=0)
    P = max((c for ms in tree.values() for c in ms.values()), default=0)
    return M, N, P


def call_budget(versions):
    """A generous concrete reading of O(log M + log N_minor + log P): twice the probes of a plain bisection
    at each level (bit_length(k) = floor(log2 k) + 1 probes for k candidates), plus 2."""
    M, N, P = shape(versions)
    return 2 * (M.bit_length() + N.bit_length() + P.bit_length()) + 2, (M, N, P)


# ---------------------------------------------------------------- Part 4

_CONSTRAINT = re.compile(r"^(==|>=|<=|>|<)(\d+\.\d+\.\d+)$")
_CMP = {"==": lambda a, b: a == b, ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def satisfies(version, constraint):
    m = _CONSTRAINT.match(constraint)
    assert m, constraint
    return _CMP[m.group(1)](parse(version), parse(m.group(2)))


def _closure_ok(registry, root, chosen):
    """True if every package reachable from root under `chosen` has a version there whose constraints hold
    and the reachable packages have no dependency cycle. Returns (ok, reachable set)."""
    reach, stack = set(), [root]
    while stack:
        name = stack.pop()
        if name in reach:
            continue
        if name not in chosen:
            return False, reach
        reach.add(name)
        for dep, constraint in registry.requires(name, chosen[name]).items():
            if dep not in chosen or not satisfies(chosen[dep], constraint):
                return False, reach
            stack.append(dep)
    # cycle check by repeatedly removing packages whose dependencies are all removed
    left = set(reach)
    while left:
        free = [n for n in left if not any(d in left for d in registry.requires(n, chosen[n]))]
        if not free:
            return False, reach
        left.difference_update(free)
    return True, reach


def brute_choice(registry, root):
    """Tries every combination of one version per registered name. Returns one valid answer (an install order
    of the packages reachable from root) or None if there is none."""
    names = list(registry._versions)
    if root not in registry._versions:
        return None
    for combo in itertools.product(*[registry.versions(n) for n in names]):
        chosen = dict(zip(names, combo))
        ok, reach = _closure_ok(registry, root, chosen)
        if ok:
            order, done = [], set()
            while len(order) < len(reach):
                for n in sorted(reach - done):
                    if all(d in done for d in registry.requires(n, chosen[n])):
                        order.append(f"{n}@{chosen[n]}")
                        done.add(n)
            return order
    return None


def check_order(registry, root, got):
    """None if `got` is a valid answer for a satisfiable instance, else a sentence saying what is wrong."""
    if not isinstance(got, (list, tuple)):
        return f"应当返回 list[str]，得到的是 {type(got).__name__}"
    chosen, position = {}, {}
    for i, entry in enumerate(got):
        if not isinstance(entry, str) or entry.count("@") != 1:
            return f"第 {i} 项 {entry!r} 不是 \"name@version\" 形式的字符串"
        name, version = entry.split("@")
        if name in chosen:
            return f"包 {name} 出现了不止一次（每个包只能选一个版本）"
        if version not in registry.versions(name):
            return f"{entry} 不是注册表里的版本"
        chosen[name], position[name] = version, i
    if root not in chosen:
        return f"结果里没有根包 {root}"
    reach, stack = set(), [root]
    while stack:
        name = stack.pop()
        if name in reach:
            continue
        reach.add(name)
        version = chosen[name]
        for dep, constraint in registry.requires(name, version).items():
            if dep not in chosen:
                return f"{name}@{version} 依赖 {dep}{constraint}，但结果里没有 {dep}"
            if not satisfies(chosen[dep], constraint):
                return f"{name}@{version} 要求 {dep}{constraint}，但选的是 {dep}@{chosen[dep]}"
            if position[dep] >= position[name]:
                if dep == name:
                    return f"{name}@{version} 依赖它自己，没有安装顺序，这个选择不合法"
                return f"安装顺序不对：{name}@{version} 依赖 {dep}，{dep} 却没有排在它前面"
            stack.append(dep)
    extra = [e for e in got if e.split("@")[0] not in reach]
    if extra:
        return f"列出了所选版本并不需要的包：{', '.join(extra)}"
    return None


def registry_code(registry, root, limit=40):
    """Code that rebuilds the registry and calls resolve, to paste into 自定义运行."""
    lines = ["reg = Registry()"]
    for name, table in registry._versions.items():
        for version, req in table.items():
            lines.append(f"reg.add_version({name!r}, {version!r}, {req!r})" if req
                         else f"reg.add_version({name!r}, {version!r})")
    if len(lines) > limit + 1:
        return f"resolve(<{len(lines) - 1} 个版本的注册表>, {root!r})"
    return "\n".join(lines + [f"resolve(reg, {root!r})"])


def build(rows):
    """rows: (name, version, requires) triples -> Registry."""
    reg = Registry()
    for name, version, req in rows:
        reg.add_version(name, version, req)
    return reg


def random_registry(r, names, max_versions=3, max_deps=2, zeros=0.15, p_dep=0.6, p_self=0.03, p_unknown=0.03):
    reg = Registry()
    for name in names:
        tuples = sorted({(r.randint(1, 2), r.randint(0, 2), r.randint(0, 2)) for _ in range(r.randint(1, max_versions))})
        others = [n for n in names if n != name]
        for t in tuples:
            req = {}
            for dep in r.sample(others, k=min(max_deps, len(others))):
                if r.random() < p_dep:
                    req[dep] = r.choice([">=", "<=", ">", "<", "=="]) + fmt(
                        (r.randint(1, 2), r.randint(0, 2), r.randint(0, 2)), r, zeros)
            if r.random() < p_self:
                req[name] = ">=0.0.0"
            if r.random() < p_unknown:
                req["ghost"] = ">=1.0.0"
            reg.add_version(name, fmt(t, r, zeros), req)
    return reg
