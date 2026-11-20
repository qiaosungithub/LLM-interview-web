# expect: WA
# An Euler-order answer: lca takes the shallowest node over preorder positions i..j instead of
# i+1..j and returns its parent, which is wrong when u is an ancestor of v.
from bisect import bisect_right


class RootedTree:
    """Heavy-light decomposition. One preorder (heavy child first) drives everything:
    - kth_ancestor: binary search among the nodes of the target depth by preorder index;
    - lca: range minimum over the preorder (sparse table) of (depth, parent);
    - path_max: O(log n) heavy chains, each a range maximum from a sparse table."""

    def __init__(self, parent, weight=None):
        n = len(parent)
        roots = [v for v in range(n) if parent[v] == -1]
        if len(roots) != 1:
            raise ValueError("need exactly one root")
        for p in parent:
            if p != -1 and not (0 <= p < n):
                raise ValueError("parent id out of range")
        if weight is not None and len(weight) != n:
            raise ValueError("weight must have length n")
        root = roots[0]
        kids = [[] for _ in range(n)]
        for v in range(n):
            if v != root:
                kids[parent[v]].append(v)
        bfs = [root]
        depth = [0] * n
        for v in bfs:
            for c in kids[v]:
                depth[c] = depth[v] + 1
                bfs.append(c)
        if len(bfs) != n:
            raise ValueError("some nodes do not reach the root")
        size = [1] * n
        for v in reversed(bfs):
            if v != root:
                size[parent[v]] += size[v]
        for v in range(n):
            if len(kids[v]) > 1:
                kids[v].sort(key=size.__getitem__)       # heaviest last, popped first
        pre = [0] * n
        order = []
        head = [0] * n
        stack = [root]
        head[root] = root
        while stack:
            v = stack.pop()
            pre[v] = len(order)
            order.append(v)
            ch = kids[v]
            if ch:
                for c in ch[:-1]:
                    head[c] = c
                    stack.append(c)
                head[ch[-1]] = head[v]
                stack.append(ch[-1])
        self.n, self.parent, self.depth, self.pre, self.order, self.head = n, parent, depth, pre, order, head
        # nodes of each depth in preorder
        by_depth = [[] for _ in range(max(depth) + 1)]
        for v in order:
            by_depth[depth[v]].append(v)
        self.by_depth = by_depth
        self.pre_at_depth = [[pre[x] for x in row] for row in by_depth]
        # LCA: for preorder positions i < j, the lca of order[i] and order[j] is the parent of the shallowest
        # node in positions i+1..j (or order[i] itself when it is an ancestor). Key = depth * n + node.
        base = [depth[v] * n + v for v in order]
        self.rmq = self._table(base, min)
        if weight is not None:
            w = [weight[v] for v in order]           # weight of the edge above order[i]
            self.wmax = self._table(w, max)
        else:
            self.wmax = None

    @staticmethod
    def _table(a, fn):
        levels = [a]
        step = 1
        while 2 * step <= len(a):
            prev = levels[-1]
            if fn is min:
                levels.append([x if x < y else y for x, y in zip(prev, prev[step:])])
            else:
                levels.append([x if x > y else y for x, y in zip(prev, prev[step:])])
            step *= 2
        return levels

    @staticmethod
    def _query(levels, lo, hi, big):
        """fn over a[lo..hi] inclusive, lo <= hi."""
        j = (hi - lo + 1).bit_length() - 1
        row = levels[j]
        x, y = row[lo], row[hi - (1 << j) + 1]
        return (x if x > y else y) if big else (x if x < y else y)

    def _check(self, v):
        if not (0 <= v < self.n):
            raise IndexError(v)

    def kth_ancestor(self, v, k):
        self._check(v)
        if k < 0:
            raise ValueError("k must be >= 0")
        t = self.depth[v] - k
        if t < 0:
            return -1
        row = self.pre_at_depth[t]
        return self.by_depth[t][bisect_right(row, self.pre[v]) - 1]

    def lca(self, u, v):
        self._check(u)
        self._check(v)
        if u == v:
            return u
        i, j = self.pre[u], self.pre[v]
        if i > j:
            i, j = j, i
        key = self._query(self.rmq, i, j, False)
        return self.parent[key % self.n]

    def distance(self, u, v):
        a = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[a]

    def path_max(self, u, v):
        self._check(u)
        self._check(v)
        if self.wmax is None:
            raise ValueError("no weights")
        head, pre, depth, par, t = self.head, self.pre, self.depth, self.parent, self.wmax
        best = None
        while head[u] != head[v]:
            if depth[head[u]] < depth[head[v]]:
                u, v = v, u
            h = head[u]
            m = self._query(t, pre[h], pre[u], True)
            if best is None or m > best:
                best = m
            u = par[h]
        if u != v:
            if depth[u] > depth[v]:
                u, v = v, u
            m = self._query(t, pre[u] + 1, pre[v], True)
            if best is None or m > best:
                best = m
        return best
