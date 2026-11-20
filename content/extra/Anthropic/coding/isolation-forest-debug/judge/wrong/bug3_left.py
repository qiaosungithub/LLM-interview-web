# expect: WA
# Bug 3 left in: size_ is recorded after the first return, so leaves keep size 0.
"""Isolation Forest (Liu, Ting & Zhou, 2008) for anomaly detection.

The docstrings in this file are the specification.

Training.  The forest has n_trees isolation trees.  Each tree is grown on its
own subsample of psi = min(max_samples, n) distinct rows of X, drawn uniformly
at random without replacement and independently of the other trees.  The
height limit of every tree is l = ceil(log2(psi)).

A node that holds the rows S at depth e (the root has depth 0) is a leaf if

  (a) e >= l, or
  (b) |S| <= 1, or
  (c) every feature is constant in S.

Otherwise pick a feature q uniformly at random among the features that are
not constant in S, and a split value p uniformly from [min_S q, max_S q); a
draw equal to min_S q is replaced by max_S q.  Rows with x[q] < p go to the
left child, all other rows go to the right child.

Storage.  All trees live in three arrays of shape (n_trees, 2 ** (l + 1) - 1)
in heap layout: node 0 is the root, and the children of node i are 2 * i + 1
(left) and 2 * i + 2 (right).

  feature_[t, i]    split feature of node i of tree t; -1 if that node is a
                    leaf or does not exist
  threshold_[t, i]  split value; 0.0 if the node has no split
  size_[t, i]       number of subsample rows that reached the node; 0 if the
                    node does not exist

Scoring.  The path length of x in one tree is h(x) = e + c(size), where e is
the number of edges between the root and the leaf that x falls into, and size
is the size of that leaf.  c(n) is the average path length of an unsuccessful
search in a binary search tree built from n points:

  c(n) = 2 * H(n - 1) - 2 * (n - 1) / n   for n > 2,
  c(2) = 1,   c(n) = 0 for n <= 1,        with H(i) = ln(i) + 0.5772156649.

The anomaly score is s(x) = 2 ** (-E[h(x)] / c(psi)), where E[h(x)] is the
mean of h(x) over the trees.  It lies in (0, 1]; anomalies score close to 1,
and a point whose mean path length equals c(psi) scores exactly 0.5.
"""
import numpy as np

EULER_GAMMA = 0.5772156649


def c_factor(n):
    """c(n) from the module docstring, elementwise.

    n: an integer or an array of integers of any shape.
    Returns an array of floats of the same shape.
    """
    n = np.asarray(n)
    out = np.zeros(n.shape)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + EULER_GAMMA) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


class IsolationForest:
    """n_trees isolation trees; see the module docstring.

    n_trees:     number of trees (>= 1).
    max_samples: requested subsample size (>= 2); the size actually used is
                 max_samples_ = min(max_samples, n), set by fit.
    seed:        anything np.random.default_rng accepts.
    """

    def __init__(self, n_trees=100, max_samples=256, seed=None):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.seed = seed

    def fit(self, X):
        """Grow the forest on X of shape (n, d), n >= 2.  Returns self.

        Sets max_samples_, height_limit_ (l), samples_ of shape
        (n_trees, max_samples_) -- the row indices of each tree's subsample --
        and the arrays feature_, threshold_, size_.
        """
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        self.max_samples_ = min(self.max_samples, len(X))
        self.height_limit_ = int(np.ceil(np.log2(self.max_samples_)))
        order = np.argsort(rng.random((self.n_trees, len(X))), axis=1)
        self.samples_ = order[:, :self.max_samples_]
        n_nodes = 2 ** (self.height_limit_ + 1) - 1
        self.feature_ = np.full((self.n_trees, n_nodes), -1)
        self.threshold_ = np.zeros((self.n_trees, n_nodes))
        self.size_ = np.zeros((self.n_trees, n_nodes), dtype=int)
        for t in range(self.n_trees):
            self._grow(t, X[self.samples_[t]], 0, 0, rng)
        return self

    def _grow(self, t, S, node, depth, rng):
        """Fill in node `node` of tree t; it holds the rows S and is at `depth`."""
        if depth >= self.height_limit_ or len(S) <= 1:
            return
        self.size_[t, node] = len(S)
        lo, hi = S.min(axis=0), S.max(axis=0)
        candidates = np.flatnonzero(lo < hi)
        if candidates.size == 0:
            return
        q = rng.choice(candidates)
        p = rng.uniform(lo[q], hi[q])
        if p == lo[q]:
            p = hi[q]
        self.feature_[t, node] = q
        self.threshold_[t, node] = p
        go_left = S[:, q] < p
        self._grow(t, S[go_left], 2 * node + 1, depth + 1, rng)
        self._grow(t, S[~go_left], 2 * node + 2, depth + 1, rng)

    def path_length(self, X):
        """h(x) of every row of X in every tree, shape (n_trees, n)."""
        X = np.asarray(X, dtype=float)
        tree = np.arange(self.n_trees)[:, None]
        cols = np.arange(len(X))
        node = np.zeros((self.n_trees, len(X)), dtype=int)
        edges = np.zeros((self.n_trees, len(X)))
        for _ in range(self.height_limit_):
            q = self.feature_[tree, node]
            internal = q >= 0
            go_right = X[cols, q] >= self.threshold_[tree, node]
            node = np.where(internal, 2 * node + 1 + go_right, node)
            edges += internal
        return edges + c_factor(self.size_[tree, node])

    def score(self, X):
        """Anomaly score s(x) of every row of X, shape (n,)."""
        mean_path = self.path_length(X).mean(axis=0)
        norm = c_factor(self.max_samples_)
        return 2.0 ** (-mean_path / norm)


# Part 2
def expected_depth(n_max):
    depth = np.zeros(n_max)
    weighted = 0.0  # sum of k * D(k) for k < n
    for n in range(2, n_max + 1):
        weighted += (n - 1) * depth[n - 2]
        depth[n - 1] = 1.0 + 2.0 * weighted / (n * (n - 1))
    return depth
