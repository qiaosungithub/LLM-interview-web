# expect: WA
# The reverse edge b -> a gets weight r instead of 1 / r, so any conversion against a fact's direction is wrong.
from collections import defaultdict
import math


def _graph(facts):
    g = defaultdict(list)
    for a, b, r in facts:
        g[a].append((b, r))
        g[b].append((a, r))
    return g


def _factor(g, src, dst):
    if src not in g or dst not in g:
        return None
    seen, stack = {src}, [(src, 1.0)]
    while stack:
        u, f = stack.pop()
        if u == dst:
            return f
        for v, w in g[u]:
            if v not in seen:
                seen.add(v)
                stack.append((v, f * w))
    return None


def convert(facts, value, src, dst):
    f = _factor(_graph(facts), src, dst)
    return None if f is None else value * f


def first_contradiction(facts, rel_tol=1e-9):
    g = defaultdict(list)
    for i, (a, b, r) in enumerate(facts):
        f = _factor(g, a, b)
        if f is not None and not math.isclose(f, r, rel_tol=rel_tol):
            return i
        g[a].append((b, r))
        g[b].append((a, r))
    return -1


class Converter:
    def __init__(self):
        self.g = defaultdict(list)

    def add_fact(self, a, b, r):
        f = _factor(self.g, a, b)
        if f is not None and not math.isclose(f, r, rel_tol=1e-9):
            return False
        self.g[a].append((b, r))
        self.g[b].append((a, r))
        return True

    def query(self, src, dst):
        return _factor(self.g, src, dst)
