# expect: WA
# Part 2: stores the points as float32, so close distances round and the wrong point can win.
import heapq

import numpy as np


class KDTree:
    def __init__(self, P: np.ndarray) -> None:
        P = np.asarray(P)
        if P.ndim != 2 or P.shape[0] == 0:
            raise ValueError("P must have shape (n, d) with n >= 1")
        self.n, self.d = P.shape
        self.pts = P.astype(np.float32).tolist()   # half the memory
        # node arrays: point index, left child, right child (-1 = none)
        self.idx, self.lc, self.rc = [], [], []
        d, pts = self.d, self.pts
        # iterative build: (ordered index list, depth, parent node, which side)
        self.root = -1
        work = [(list(range(self.n)), 0, -1, 0)]
        while work:
            items, depth, parent, side = work.pop()
            a = depth % d
            items.sort(key=lambda i: (pts[i][a], i))
            mid = len(items) // 2
            node = len(self.idx)
            self.idx.append(items[mid])
            self.lc.append(-1)
            self.rc.append(-1)
            if parent == -1:
                self.root = node
            elif side == 0:
                self.lc[parent] = node
            else:
                self.rc[parent] = node
            if mid > 0:
                work.append((items[:mid], depth + 1, node, 0))
            if mid + 1 < len(items):
                work.append((items[mid + 1:], depth + 1, node, 1))

    def _query(self, q):
        q = np.asarray(q)
        if q.shape != (self.d,):
            raise ValueError("q must have shape (d,)")
        return [float(x) for x in q.tolist()]

    def _d2(self, i, q):
        return sum((x - y) * (x - y) for x, y in zip(self.pts[i], q))

    def nearest(self, q: np.ndarray) -> int:
        q = self._query(q)
        best = [float("inf"), -1]

        def walk(node, depth):
            if node == -1:
                return
            i = self.idx[node]
            dd = self._d2(i, q)
            if (dd, i) < (best[0], best[1]) or best[1] == -1:
                best[0], best[1] = dd, i
            a = depth % self.d
            diff = q[a] - self.pts[i][a]
            near, far = (self.lc[node], self.rc[node]) if diff < 0 else (self.rc[node], self.lc[node])
            walk(near, depth + 1)
            if diff * diff <= best[0]:
                walk(far, depth + 1)

        walk(self.root, 0)
        return best[1]

    def knn(self, q: np.ndarray, k: int) -> list[int]:
        q = self._query(q)
        if not 1 <= k <= self.n:
            raise ValueError("need 1 <= k <= n")
        heap = []  # max-heap on (d2, index) via negation: (-d2, -index)

        def walk(node, depth):
            if node == -1:
                return
            i = self.idx[node]
            dd = self._d2(i, q)
            if len(heap) < k:
                heapq.heappush(heap, (-dd, -i))
            elif (dd, i) < (-heap[0][0], -heap[0][1]):
                heapq.heapreplace(heap, (-dd, -i))
            a = depth % self.d
            diff = q[a] - self.pts[i][a]
            near, far = (self.lc[node], self.rc[node]) if diff < 0 else (self.rc[node], self.lc[node])
            walk(near, depth + 1)
            if len(heap) < k or diff * diff <= -heap[0][0]:
                walk(far, depth + 1)

        walk(self.root, 0)
        return [-ni for _, ni in sorted(heap, key=lambda t: (-t[0], -t[1]))]

    def count_in_box(self, lo: np.ndarray, hi: np.ndarray) -> int:
        lo, hi = np.asarray(lo), np.asarray(hi)
        if lo.shape != (self.d,) or hi.shape != (self.d,):
            raise ValueError("lo and hi must have shape (d,)")
        lo, hi = lo.tolist(), hi.tolist()
        if any(l > h for l, h in zip(lo, hi)):
            return 0
        count = 0
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            if node == -1:
                continue
            p = self.pts[self.idx[node]]
            if all(l <= x <= h for l, x, h in zip(lo, p, hi)):
                count += 1
            a = depth % self.d
            s = p[a]
            # left subtree holds coordinates <= s on axis a, right subtree >= s
            if lo[a] <= s:
                stack.append((self.lc[node], depth + 1))
            if hi[a] >= s:
                stack.append((self.rc[node], depth + 1))
        return count
