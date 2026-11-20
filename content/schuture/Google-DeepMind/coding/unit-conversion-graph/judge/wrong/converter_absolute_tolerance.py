# expect: WA
# Converter.add_fact compares the implied factor with r using an absolute tolerance of 1e-9, so tiny factors never contradict and huge ones always do.
import math
from collections import deque


def _adj(facts):
    g = {}
    for a, b, r in facts:
        g.setdefault(a, []).append((b, r))
        g.setdefault(b, []).append((a, 1.0 / r))
    return g


def convert(facts, value, src, dst):
    g = _adj(facts)
    if src not in g or dst not in g:
        return None
    seen = {src: 1.0}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            return value * seen[u]
        for v, w in g[u]:
            if v not in seen:
                seen[v] = seen[u] * w
                q.append(v)
    return None


class _Dsu:
    """pot[x]: 1 x = pot[x] * parent[x]. Iterative two-pass find, union by rank."""

    def __init__(self):
        self.par = {}
        self.pot = {}
        self.rank = {}

    def has(self, x):
        return x in self.par

    def add(self, x):
        if x not in self.par:
            self.par[x] = x
            self.pot[x] = 1.0
            self.rank[x] = 0

    def find(self, x):
        path = []
        while self.par[x] != x:
            path.append(x)
            x = self.par[x]
        root = x
        # from the node nearest the root outward
        acc = 1.0
        for y in reversed(path):
            acc = self.pot[y] * acc
            self.pot[y] = acc
            self.par[y] = root
        return root

    def factor(self, x):
        """root, f with 1 x = f root"""
        r = self.find(x)
        return r, (1.0 if x == r else self.pot[x])

    def link(self, a, b, r):
        """1 a = r b; returns None if joined, else implied factor a->b"""
        ra, fa = self.factor(a)
        rb, fb = self.factor(b)
        if ra == rb:
            return fa / fb
        # 1 a = fa ra, 1 b = fb rb, 1 a = r b = r fb rb  => 1 ra = r fb / fa rb
        k = r * fb / fa
        if self.rank[ra] < self.rank[rb]:
            self.par[ra] = rb
            self.pot[ra] = k
        else:
            self.par[rb] = ra
            self.pot[rb] = 1.0 / k
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
        return None


def first_contradiction(facts, rel_tol=1e-9):
    d = _Dsu()
    for i, (a, b, r) in enumerate(facts):
        if d.has(a) and d.has(b):
            ra, fa = d.factor(a)
            rb, fb = d.factor(b)
            if ra == rb:
                if not math.isclose(fa / fb, r, rel_tol=rel_tol):
                    return i
                continue
        d.add(a)
        d.add(b)
        d.link(a, b, r)
    return -1


class Converter:
    def __init__(self):
        self._d = _Dsu()

    def add_fact(self, a, b, r):
        d = self._d
        if d.has(a) and d.has(b):
            ra, fa = d.factor(a)
            rb, fb = d.factor(b)
            if ra == rb:
                return abs(fa / fb - r) <= 1e-9
        d.add(a)
        d.add(b)
        d.link(a, b, r)
        return True

    def query(self, src, dst):
        d = self._d
        if not (d.has(src) and d.has(dst)):
            return None
        ra, fa = d.factor(src)
        rb, fb = d.factor(dst)
        if ra != rb:
            return None
        return fa / fb
