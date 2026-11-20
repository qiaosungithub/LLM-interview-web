# expect: WA
# add_fact compares w_b / w_a (the factor from b to a) with r, so restating a known fact is rejected.
from collections import defaultdict


def _add_edges(graph: dict[str, list[tuple[str, float]]], a: str, b: str, r: float) -> None:
    """One fact contributes two directed edges: a -> b weighted r, and its inverse."""
    graph[a].append((b, r))
    graph[b].append((a, 1.0 / r))    # NOTE: the reverse edge divides -- it is 1 / r, never r again


def _build_graph(facts: list[tuple[str, str, float]]) -> dict[str, list[tuple[str, float]]]:
    graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for a, b, r in facts:
        _add_edges(graph, a, b, r)
    return graph


def _path_factor(graph: dict[str, list[tuple[str, float]]], src: str, dst: str) -> float | None:
    """Product of edge weights along some path from src to dst (None if none exists). Assumes graph
    was built from consistent facts, so it never matters which path a search happens to find."""
    if src not in graph or dst not in graph:
        return None
    if src == dst:
        return 1.0
    visited = {src}
    stack = [(src, 1.0)]             # NOTE: an explicit stack -- iterative, so no recursion-depth limit
    while stack:
        unit, acc = stack.pop()
        for neighbour, weight in graph[unit]:
            product = acc * weight   # NOTE: repeated multiplication -- floating-point error can accumulate
            if neighbour == dst:
                return product
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append((neighbour, product))
    return None


def convert(facts: list[tuple[str, str, float]], value: float, src: str, dst: str) -> float | None:
    factor = _path_factor(_build_graph(facts), src, dst)
    return None if factor is None else value * factor

import math


def first_contradiction(facts: list[tuple[str, str, float]], rel_tol: float = 1e-9) -> int:
    graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for i, (a, b, r) in enumerate(facts):
        implied = _path_factor(graph, a, b)
        if implied is not None and not math.isclose(implied, r, rel_tol=rel_tol):
            return i                          # NOTE: only an already-connected pair can have an implied factor
        _add_edges(graph, a, b, r)
    return -1

REL_TOL = 1e-9    # NOTE: shared with first_contradiction's default, so Part 2 and Part 3 agree on every fact


class Converter:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._weight: dict[str, float] = {}    # 1 unit = weight[unit] * parent[unit]
        self._size: dict[str, int] = {}

    def _find(self, x: str) -> tuple[str, float]:
        """Path-compressing find. Returns (root, w) with 1 x = w * root."""
        if self._parent[x] == x:
            return x, 1.0
        root, w = self._find(self._parent[x])
        self._weight[x] *= w             # NOTE: compose with the OLD weight[x] first, then overwrite it
        self._parent[x] = root           # path compression -- x now points directly at the root
        return root, self._weight[x]

    def add_fact(self, a: str, b: str, r: float) -> bool:
        for u in (a, b):
            if u not in self._parent:
                self._parent[u] = u
                self._weight[u] = 1.0
                self._size[u] = 1
        root_a, w_a = self._find(a)               # 1 a = w_a * root_a
        root_b, w_b = self._find(b)                # 1 b = w_b * root_b
        if root_a == root_b:
            return math.isclose(w_b / w_a, r, rel_tol=REL_TOL)   # NOTE: no mutation on either outcome here
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
            w_a, w_b = w_b, w_a
            r = 1.0 / r                            # NOTE: swapping which side is "a" inverts the stated factor
        self._parent[root_b] = root_a
        self._weight[root_b] = w_a / (r * w_b)     # derived above: 1 root_b = (w_a / (r * w_b)) * root_a
        self._size[root_a] += self._size[root_b]
        return True

    def query(self, src: str, dst: str) -> float | None:
        if src not in self._parent or dst not in self._parent:
            return None
        root_src, w_src = self._find(src)
        root_dst, w_dst = self._find(dst)
        if root_src != root_dst:
            return None
        return w_src / w_dst                       # same w_x / w_y reasoning as the case above
