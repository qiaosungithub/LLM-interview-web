# expect: TLE
# Every query walks up one parent pointer at a time: O(depth) per query instead of O(log n).
class RootedTree:
    def __init__(self, parent: list[int], weight: list[int] | None = None) -> None:
        n = len(parent)
        roots = [v for v in range(n) if parent[v] == -1]
        if len(roots) != 1:
            raise ValueError("expected exactly one root")
        root = roots[0]
        children = [[] for _ in range(n)]
        for v, p in enumerate(parent):
            if v == root:
                continue
            if not 0 <= p < n:
                raise ValueError(f"parent[{v}] = {p} is not a node")
            children[p].append(v)
        depth = [0] * n
        order = [root]
        for u in order:
            for c in children[u]:
                depth[c] = depth[u] + 1
                order.append(c)
        if len(order) != n:
            raise ValueError("some node does not reach the root")
        if weight is not None and len(weight) != n:
            raise ValueError("weight must have one entry per node")
        self.n, self.parent, self.depth = n, list(parent), depth
        self.weight = None if weight is None else list(weight)

    def _check_node(self, v: int) -> None:
        if not 0 <= v < self.n:
            raise IndexError(f"node {v} out of range [0, {self.n})")

    def kth_ancestor(self, v: int, k: int) -> int:
        self._check_node(v)
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > self.depth[v]:
            return -1
        for _ in range(k):
            v = self.parent[v]
        return v

    def lca(self, u: int, v: int) -> int:
        self._check_node(u)
        self._check_node(v)
        while self.depth[u] > self.depth[v]:
            u = self.parent[u]
        while self.depth[v] > self.depth[u]:
            v = self.parent[v]
        while u != v:
            u, v = self.parent[u], self.parent[v]
        return u

    def distance(self, u: int, v: int) -> int:
        a = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[a]

    def path_max(self, u: int, v: int) -> int | None:
        if self.weight is None:
            raise ValueError("the tree was built without weights")
        a = self.lca(u, v)
        if u == v:
            return None
        best = None
        for x in (u, v):
            while x != a:
                if best is None or self.weight[x] > best:
                    best = self.weight[x]
                x = self.parent[x]
        return best
