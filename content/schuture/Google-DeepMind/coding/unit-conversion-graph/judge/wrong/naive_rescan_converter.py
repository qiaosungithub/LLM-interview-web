# expect: TLE
# Converter searches the whole graph of facts on every call, which the statement rules out.
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

REL_TOL = 1e-9


class Converter:
    """Keeps the graph of all facts and searches it on every call, as Part 1 does."""

    def __init__(self) -> None:
        self._graph: dict[str, list[tuple[str, float]]] = defaultdict(list)

    def add_fact(self, a: str, b: str, r: float) -> bool:
        implied = _path_factor(self._graph, a, b)
        if implied is not None and not math.isclose(implied, r, rel_tol=REL_TOL):
            return False
        _add_edges(self._graph, a, b, r)
        return True

    def query(self, src: str, dst: str) -> float | None:
        return _path_factor(self._graph, src, dst)
