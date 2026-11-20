# expect: WA
# LOG = int(log2(n)) is one level short when n = 2**m + 1, and kth_ancestor only looks at the low LOG bits of k.
import math

NEG_INF = float("-inf")


class RootedTree:
    """A rooted tree given by its parent array. up[j][v] is the node 2**j edges above v, clamped at the
    root: inside the table the root is its own parent, so a jump that would leave the tree stays put.
    mx[j][v] is the largest weight among the 2**j edges going up from v (fewer if the root comes first:
    the root's own slot holds -inf, which never wins a max)."""

    def __init__(self, parent: list[int], weight: list[int] | None = None) -> None:
        n = len(parent)
        roots = [v for v in range(n) if parent[v] == -1]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root, found {len(roots)}")
        root = roots[0]
        children: list[list[int]] = [[] for _ in range(n)]
        for v, p in enumerate(parent):
            if v == root:
                continue
            if not 0 <= p < n:
                raise ValueError(f"parent[{v}] = {p} is not a node")
            children[p].append(v)
        depth = [0] * n
        order = [root]                    # BFS order; the list is its own queue
        for u in order:                   # NOTE: no recursion -- a path of 2 * 10**5 nodes is a legal input
            for c in children[u]:
                depth[c] = depth[u] + 1
                order.append(c)
        if len(order) != n:               # an unreached node can only be walking around a cycle
            raise ValueError("some node does not reach the root")
        self.n, self.root, self.depth, self.order = n, root, depth, order
        self.log = max(1, int(math.log2(n)))          # levels 2**0 .. 2**(LOG - 1)
        first = list(parent)
        first[root] = root                # NOTE: the root points at itself, so every jump clamps for free
        self.up = [first]
        for _ in range(1, self.log):
            prev = self.up[-1]
            self.up.append([prev[a] for a in prev])  # 2**j edges = 2**(j-1) edges, twice

        # Part 3: the second table, only when weights are given
        self.mx = None
        if weight is None:
            return
        if len(weight) != self.n:
            raise ValueError("weight must have one entry per node")
        first = list(weight)
        first[self.root] = NEG_INF        # NOTE: weight[root] belongs to no edge
        self.mx = [first]
        for j in range(1, self.log):
            below, half = self.mx[-1], self.up[j - 1]
            self.mx.append([max(m, below[a]) for m, a in zip(below, half)])

    def _check_node(self, v: int) -> None:
        if not 0 <= v < self.n:
            raise IndexError(f"node {v} out of range [0, {self.n})")

    # ------------------------------------------------------------ Part 1

    def kth_ancestor(self, v: int, k: int) -> int:
        self._check_node(v)
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > self.depth[v]:
            return -1                     # NOTE: decided by depth alone, which also guarantees k < 2**LOG
        for j in range(self.log):
            if k >> j & 1:
                v = self.up[j][v]
        return v

    # ------------------------------------------------------------ Part 2

    def lca(self, u: int, v: int) -> int:
        self._check_node(u)
        self._check_node(v)
        depth, up = self.depth, self.up
        if depth[u] < depth[v]:
            u, v = v, u
        u = self.kth_ancestor(u, depth[u] - depth[v])    # both are now at the same depth
        if u == v:
            return u                                     # one of them was an ancestor of the other
        for j in range(self.log - 1, -1, -1):
            if up[j][u] != up[j][v]:                     # NOTE: jump only while the two stay different,
                u, v = up[j][u], up[j][v]                #       so the loop ends one edge below the answer
        return up[0][u]

    def distance(self, u: int, v: int) -> int:
        a = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[a]

    # ------------------------------------------------------------ Part 3

    def _max_up(self, v: int, k: int):
        """Largest weight among the k edges going up from v (-inf when k == 0). Needs k <= depth[v]."""
        best = NEG_INF
        j = 0
        while k:
            if k & 1:
                best = max(best, self.mx[j][v])   # NOTE: read the block's maximum BEFORE leaving v
                v = self.up[j][v]
            k >>= 1
            j += 1
        return best

    def path_max(self, u: int, v: int) -> int | None:
        if self.mx is None:
            raise ValueError("the tree was built without weights")
        a = self.lca(u, v)                # also validates u and v
        if u == v:
            return None                   # the path has no edge
        return max(self._max_up(u, self.depth[u] - self.depth[a]),
                   self._max_up(v, self.depth[v] - self.depth[a]))
