"""Brute forces written from the statement alone, and random input generators.

Random facts come from hidden unit sizes: unit u is worth size[u] of some base unit, so a consistent fact is
(a, b, size[a] / size[b]) and the true factor from src to dst is size[src] / size[dst]. A contradicting fact
multiplies that ratio by a factor far from 1 (at least 1e-4 away), so rounding (about 1e-15 per step) can never
move a decision across the rel_tol = 1e-9 line.

- closure(facts): every factor implied by the facts, by repeatedly multiplying known factors (a Floyd-Warshall
  style closure, as in the page's own check); used for Part 1 and Part 2 on small inputs.
- first_contradiction(facts, rel_tol): rebuilds the closure of facts[:i] before every fact i.
- Dsu: plain connectivity, used to know which pairs are connected and which units are known.
- Model: Part 3 straight from the statement, keeping the factor between every two connected units."""
import math


def closure(facts):
    units = sorted({u for a, b, _ in facts for u in (a, b)})
    f = {u: {u: 1.0} for u in units}
    for a, b, r in facts:
        f[a].setdefault(b, r)
        f[b].setdefault(a, 1.0 / r)
    for k in units:
        for i in units:
            if k in f[i]:
                for j, x in list(f[k].items()):
                    f[i].setdefault(j, f[i][k] * x)
    return f


def convert(facts, value, src, dst):
    f = closure(facts)
    if src not in f or dst not in f[src]:
        return None
    return value * f[src][dst]


def first_contradiction(facts, rel_tol=1e-9):
    for i, (a, b, r) in enumerate(facts):
        f = closure(facts[:i])
        if a in f and b in f[a] and not math.isclose(f[a][b], r, rel_tol=rel_tol):
            return i
    return -1


class Dsu:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def known(self, x):
        return x in self.parent

    def connected(self, x, y):
        return self.known(x) and self.known(y) and self.find(x) == self.find(y)


def sizes(r, units, lo=0.2, hi=5.0):
    return {u: r.uniform(lo, hi) for u in units}


def far_factor(r):
    """A multiplier that makes a fact contradict. A product or ratio of two of these is either 1 (up to rounding) or
    at least 1e-8 away from 1, ten times rel_tol, so even two perturbed facts meeting on one cycle are judged the
    same way by every correct answer."""
    return r.choice([2.0, 0.5, 3.0, 1 / 3, 10.0, 1.0001, 0.9999, 1.01])


def consistent_facts(r, size, groups, extra):
    """Spanning trees over `groups` (lists of units), plus `extra` redundant facts inside the groups, some of them
    repeats or stated backwards; shuffled."""
    facts = []
    for g in groups:
        order = list(g)
        r.shuffle(order)
        for i in range(1, len(order)):
            a, b = order[i], order[r.randrange(i)]
            if r.random() < 0.5:
                a, b = b, a
            facts.append((a, b, size[a] / size[b]))
    big = [g for g in groups if len(g) >= 2]
    for _ in range(extra if big else 0):
        g = r.choice(big)
        a, b = r.sample(g, 2)
        facts.append((a, b, size[a] / size[b]))
    if facts and r.random() < 0.3:
        a, b, x = r.choice(facts)
        facts.append(r.choice([(a, b, x), (b, a, 1.0 / x)]))
    r.shuffle(facts)
    return facts


def split_units(r, units, parts):
    units = list(units)
    r.shuffle(units)
    parts = min(parts, len(units))
    cuts = sorted(r.sample(range(1, len(units)), parts - 1)) if parts > 1 else []
    out, prev = [], 0
    for c in cuts + [len(units)]:
        out.append(units[prev:c])
        prev = c
    return [g for g in out if g]


class Model:
    """Part 3 from the statement: f[x][y] is the factor with 1 x = f[x][y] y, kept for every pair of connected
    units. A fact between two connected units is checked against that table; any other fact is accepted and the
    table is filled in for every pair it newly connects."""

    def __init__(self):
        self.f = {}

    def add_fact(self, a, b, r):
        f = self.f
        if a in f and b in f[a]:
            return math.isclose(f[a][b], r, rel_tol=1e-9)
        for u in (a, b):
            f.setdefault(u, {u: 1.0})
        left, right = list(f[a]), list(f[b])
        for x in left:
            for y in right:
                v = f[x][a] * r * f[b][y]
                f[x][y], f[y][x] = v, 1.0 / v
        return True

    def query(self, src, dst):
        return self.f.get(src, {}).get(dst)


def random_converter_ops(r, units, size, n_ops, p_add=0.55, p_bad=0.4):
    """A random add_fact / query sequence over `units`, as (method, args). Facts follow the hidden sizes; a fact
    between two units that are already connected is perturbed with probability p_bad, so it must be rejected."""
    dsu = Dsu()
    ops = []
    for _ in range(n_ops):
        if r.random() < p_add:
            a, b = r.sample(units, 2)
            ratio = size[a] / size[b]
            if dsu.connected(a, b) and r.random() < p_bad:
                ops.append(("add_fact", (a, b, ratio * far_factor(r))))
            else:
                ops.append(("add_fact", (a, b, ratio)))
                dsu.union(a, b)
        else:
            ops.append(("query", (r.choice(units), r.choice(units))))
    return ops
