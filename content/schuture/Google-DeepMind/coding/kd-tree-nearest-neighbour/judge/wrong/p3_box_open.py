# expect: WA
# Part 3 count_in_box tests lo < p < hi, so points on an edge of the box are not counted.

import numpy as np


class KDTree:
    """A k-d tree over the rows of P (shape (n, d)). Node t stores one point's row index in
    self.point[t], its split axis in self.axis[t], and its children's node ids in self.left[t] /
    self.right[t] (-1 when absent) -- flat arrays indexed by node id, not a chain of Python objects."""

    def __init__(self, P: np.ndarray) -> None:
        P = np.asarray(P, dtype=np.float64)
        if P.ndim != 2 or P.shape[0] < 1:
            raise ValueError("P must have shape (n, d) with n >= 1")
        self.P = P
        self.n, self.d = P.shape
        self.point = np.empty(self.n, dtype=np.int64)
        self.axis = np.empty(self.n, dtype=np.int64)
        self.left = np.full(self.n, -1, dtype=np.int64)
        self.right = np.full(self.n, -1, dtype=np.int64)
        self._next_node = 0
        self.root = self._build(list(range(self.n)), depth=0)

    def _build(self, indices: list[int], depth: int) -> int:
        if not indices:
            return -1
        axis = depth % self.d
        indices.sort(key=lambda i: (self.P[i, axis], i))    # NOTE: index breaks ties -- otherwise which
                                                              #       duplicate becomes this node is unspecified
        mid = len(indices) // 2
        node = self._next_node
        self._next_node += 1
        self.point[node] = indices[mid]
        self.axis[node] = axis
        self.left[node] = self._build(indices[:mid], depth + 1)
        self.right[node] = self._build(indices[mid + 1:], depth + 1)
        return node


def nearest(self, q: np.ndarray) -> int:
    return self._nearest_visited(q)[0]


def _nearest_visited(self, q: np.ndarray) -> tuple[int, int]:
    """Same answer as nearest(q), plus the number of nodes visited -- nearest() uses this directly,
    and the checks below measure the count to quantify how much pruning actually saves."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self.d,):
        raise ValueError(f"q must have shape ({self.d},)")
    best_d2 = np.inf
    best_idx = self.n          # NOTE: sentinel larger than every real index 0..n-1, so any point beats it once
    visited = 0

    def recurse(node: int) -> None:
        nonlocal best_d2, best_idx, visited
        if node == -1:
            return
        visited += 1
        pi = self.point[node]
        diff_vec = self.P[pi] - q
        d2 = float(diff_vec @ diff_vec)             # NOTE: squared distance -- sqrt is monotone, so wasted work
        if (d2, pi) < (best_d2, best_idx):           # NOTE: tuple compare enforces "ties -> smallest index"
            best_d2, best_idx = d2, pi
        a = self.axis[node]
        diff = q[a] - self.P[pi, a]
        near, far = (self.left[node], self.right[node]) if diff <= 0 else (self.right[node], self.left[node])
        recurse(near)
        if diff * diff <= best_d2:                  # NOTE: <= not < -- the far side can still hold a tie at
            recurse(far)                             #       the exact same distance with a smaller index

    recurse(self.root)
    return int(best_idx), visited


KDTree.nearest = nearest
KDTree._nearest_visited = _nearest_visited


import heapq


def knn(self, q: np.ndarray, k: int) -> list[int]:
    return self._knn_visited(q, k)[0]


def _knn_visited(self, q: np.ndarray, k: int) -> tuple[list[int], int]:
    """Same answer as knn(q, k), plus the number of nodes visited -- knn() uses this directly, and the
    checks below use the count to confirm the worst case visits every node, exactly as Part 2's does."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (self.d,):
        raise ValueError(f"q must have shape ({self.d},)")
    if not (1 <= k <= self.n):
        raise ValueError(f"k must satisfy 1 <= k <= {self.n}")
    heap: list[tuple[float, int]] = []        # max-heap via negation; heap[0] is the current worst of the top k
    visited = 0

    def recurse(node: int) -> None:
        nonlocal visited
        if node == -1:
            return
        visited += 1
        pi = self.point[node]
        diff_vec = self.P[pi] - q
        d2 = float(diff_vec @ diff_vec)
        key = (-d2, -int(pi))
        if len(heap) < k:
            heapq.heappush(heap, key)
        elif key > heap[0]:                    # NOTE: strictly better than the current worst -- the same
            heapq.heapreplace(heap, key)        #       (distance, index) ordering as nearest's tie rule
        a = self.axis[node]
        diff = q[a] - self.P[pi, a]
        near, far = (self.left[node], self.right[node]) if diff <= 0 else (self.right[node], self.left[node])
        recurse(near)
        worst_d2 = -heap[0][0] if len(heap) == k else np.inf   # NOTE: nothing may be pruned before k are found
        if diff * diff <= worst_d2:
            recurse(far)

    recurse(self.root)
    ordered = sorted((-neg_d2, -neg_idx) for neg_d2, neg_idx in heap)
    return [idx for _d2, idx in ordered], visited


KDTree.knn = knn
KDTree._knn_visited = _knn_visited


def _fill_sizes(self, node: int, size: np.ndarray) -> int:
    if node == -1:
        return 0
    total = 1 + self._fill_sizes(self.left[node], size) + self._fill_sizes(self.right[node], size)
    size[node] = total
    return total


def _sizes(self) -> np.ndarray:
    cached = getattr(self, "_size_cache", None)
    if cached is None:
        cached = np.zeros(self.n, dtype=np.int64)
        self._fill_sizes(self.root, cached)
        self._size_cache = cached           # NOTE: cached once -- reused by every later count_in_box call
    return cached


def count_in_box(self, lo: np.ndarray, hi: np.ndarray) -> int:
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != (self.d,) or hi.shape != (self.d,):
        raise ValueError(f"lo and hi must have shape ({self.d},)")
    if np.any(lo > hi):
        return 0                             # NOTE: an empty box holds no points -- not an error
    cell_lo = np.full(self.d, -np.inf)
    cell_hi = np.full(self.d, np.inf)
    return self._count(self.root, cell_lo, cell_hi, lo, hi, self._sizes())


def _count(self, node: int, cell_lo: np.ndarray, cell_hi: np.ndarray,
           lo: np.ndarray, hi: np.ndarray, size: np.ndarray) -> int:
    if node == -1:
        return 0
    if np.any(cell_hi < lo) or np.any(cell_lo > hi):
        return 0                                            # NOTE: this node's whole region misses the box
    if np.all(cell_lo >= lo) and np.all(cell_hi <= hi):
        return int(size[node])                                # NOTE: whole region inside -- O(1) via the size
    pi = self.point[node]
    p = self.P[pi]
    total = 1 if np.all((lo < p) & (p < hi)) else 0
    a = self.axis[node]
    s = p[a]
    left_hi = cell_hi.copy()
    left_hi[a] = min(left_hi[a], s)
    right_lo = cell_lo.copy()
    right_lo[a] = max(right_lo[a], s)
    total += self._count(self.left[node], cell_lo, left_hi, lo, hi, size)
    total += self._count(self.right[node], right_lo, cell_hi, lo, hi, size)
    return total


KDTree._fill_sizes = _fill_sizes
KDTree._sizes = _sizes
KDTree.count_in_box = count_in_box
KDTree._count = _count
