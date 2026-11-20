# Debug a vectorised Isolation Forest implementation

## Problem

This is a companion exercise to [Debug a vectorised Extra-Trees implementation](#/anthropic/coding/extratrees-debug). Nobody in the evidence reports being asked about Isolation Forest. We wrote a separate one in the same format and under the same rules, to give you another round of "read the spec, find the bugs".

The `iforest.py` below implements Isolation Forest (Liu, Ting, Zhou, 2008), an unsupervised anomaly detection method, in vectorised NumPy. Each tree splits a subsample drawn without replacement at random until points are isolated or a height limit is reached. The shorter a point's average path length across the trees, the easier it is to isolate, and the higher its anomaly score. All trees are stored in a few arrays of shape `(n_trees, n_nodes)`, and at scoring time all trees and all points descend together. The code mostly works (most of the time it ranks the outliers first), but it has several bugs. `test_iforest.py` is the test suite you are given.

The rules are the same as in the Extra-Trees problem:

- Only change `iforest.py`. Don't change the tests or the docstrings. They are correct, and the docstrings are the specification.
- Keep it vectorised: don't rewrite NumPy vector operations as nested Python loops.
- No scikit-learn or any other existing implementation, and no AI. You may consult the NumPy / Python documentation.
- Time limit: 60 minutes. Every test is worth the same.

To practise locally: save the two code blocks below as `iforest.py` and `test_iforest.py` in the same directory and run `python test_iforest.py` (it needs only NumPy, not pytest). `python test_iforest.py test_few_points` runs a single test and prints the traceback.

### Part 1 — Make all the tests pass

`iforest.py`:

```py
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
    out = np.zeros_like(n)
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
        order = np.argsort(rng.random((self.n_trees, len(X))), axis=0)
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
            node = 2 * node + 1 + go_right
            edges += internal
        return edges + c_factor(self.size_[tree, node])

    def score(self, X):
        """Anomaly score s(x) of every row of X, shape (n,)."""
        mean_path = self.path_length(X).mean(axis=0)
        norm = c_factor(self.max_samples)
        return 2.0 ** -mean_path / norm
```

`test_iforest.py`:

```py
"""Tests for iforest.py.

Run all of them:   python test_iforest.py
Run one of them:   python test_iforest.py test_few_points     (prints the traceback)
"""
import numpy as np

from iforest import IsolationForest, c_factor


def test_c_factor():
    c = c_factor(np.array([0, 1, 2, 3, 4, 256]))
    assert np.allclose(c, [0.0, 0.0, 1.0, 1.2073923576, 1.8516559071, 10.2447709201])
    assert c_factor(np.array([[2, 5], [9, 1]])).shape == (2, 2)
    assert np.isclose(float(c_factor(100)), 8.3646710301)
    assert np.all(np.diff(c_factor(np.arange(2, 300))) > 0)


def test_subsamples():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    forest = IsolationForest(n_trees=40, max_samples=64, seed=0).fit(X)
    assert forest.max_samples_ == 64 and forest.height_limit_ == 6
    assert forest.samples_.shape == (40, 64)
    assert forest.samples_.min() >= 0 and forest.samples_.max() < 300
    assert all(len(np.unique(row)) == 64 for row in forest.samples_)
    assert len(np.unique(forest.samples_)) > 270
    assert len({tuple(row) for row in forest.samples_.tolist()}) == 40


def test_tree_structure():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    forest = IsolationForest(n_trees=20, max_samples=100, seed=1).fit(X)
    limit = forest.height_limit_
    assert limit == 7
    assert forest.feature_.shape == forest.threshold_.shape == forest.size_.shape == (20, 255)
    assert np.all(forest.size_[:, 0] == 100)
    internal = forest.feature_ >= 0
    assert not internal[:, 2 ** limit - 1:].any()
    t, i = np.nonzero(internal)
    left, right = forest.size_[t, 2 * i + 1], forest.size_[t, 2 * i + 2]
    assert np.all(left >= 1) and np.all(right >= 1)
    assert np.array_equal(forest.size_[t, i], left + right)
    is_leaf = ~internal & (forest.size_ > 0)
    assert np.all((forest.size_ * is_leaf).sum(axis=1) == 100)
    assert np.all(forest.threshold_[~internal] == 0.0)


def test_path_length_manual():
    forest = IsolationForest(n_trees=2, max_samples=4)
    forest.max_samples_ = 4
    forest.height_limit_ = 2
    forest.feature_ = np.array([[0, -1, 1, -1, -1, -1, -1],
                                [-1, -1, -1, -1, -1, -1, -1]])
    forest.threshold_ = np.array([[0.5, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    forest.size_ = np.array([[4, 1, 3, 0, 0, 2, 1],
                             [4, 0, 0, 0, 0, 0, 0]])
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 3.0], [0.5, 2.0]])
    c4 = 1.8516559071
    paths = forest.path_length(X)
    assert paths.shape == (2, 4)
    assert np.allclose(paths, [[1.0, 3.0, 2.0, 2.0], [c4, c4, c4, c4]])
    assert np.allclose(forest.score(X), [0.5864059, 0.4032969, 0.4863082, 0.4863082])


def test_constant_data():
    X = np.full((40, 3), 2.5)
    forest = IsolationForest(n_trees=10, max_samples=16, seed=0).fit(X)
    assert np.all(forest.feature_ == -1)
    assert np.all(forest.size_[:, 0] == 16)
    assert np.allclose(forest.score(X), 0.5)
    assert np.allclose(forest.score(np.array([[0.0, 9.0, -3.0]])), 0.5)
    forest = IsolationForest(n_trees=10, seed=0).fit(X)
    assert forest.max_samples_ == 40
    assert np.allclose(forest.score(X), 0.5)


def test_few_points():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [50.0]])
    forest = IsolationForest(n_trees=200, seed=0).fit(X)
    assert forest.max_samples_ == 8 and forest.height_limit_ == 3
    assert forest.samples_.shape == (200, 8)
    scores = forest.score(X)
    assert scores.shape == (8,)
    assert np.all((scores > 0) & (scores <= 1))
    assert scores.argmax() == 7
    assert scores[7] > 0.7
    assert scores[:7].max() < 0.65


def make_blob_with_outliers():
    rng = np.random.default_rng(3)
    inliers = rng.normal(size=(400, 2))
    outliers = rng.uniform(5.0, 8.0, size=(10, 2)) * rng.choice([-1.0, 1.0], size=(10, 2))
    return np.concatenate([inliers, outliers])


def test_ranks_outliers_first():
    X = make_blob_with_outliers()
    scores = IsolationForest(n_trees=60, max_samples=64, seed=0).fit(X).score(X)
    assert scores.shape == (410,)
    assert np.mean(scores[:400] < scores[400:].min()) > 0.97


def test_score_scale():
    X = make_blob_with_outliers()
    scores = IsolationForest(n_trees=60, max_samples=64, seed=0).fit(X).score(X)
    assert np.all((scores > 0) & (scores <= 1))
    assert scores[400:].min() > 0.6
    assert 0.35 < np.median(scores[:400]) < 0.53


def test_reproducible():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 4))
    a = IsolationForest(n_trees=8, max_samples=32, seed=5).fit(X)
    b = IsolationForest(n_trees=8, max_samples=32, seed=5).fit(X)
    c = IsolationForest(n_trees=8, max_samples=32, seed=6).fit(X)
    assert np.array_equal(a.samples_, b.samples_)
    assert np.array_equal(a.feature_, b.feature_)
    assert np.array_equal(a.threshold_, b.threshold_)
    assert not np.array_equal(a.threshold_, c.threshold_)
    assert np.array_equal(a.score(X), b.score(X))


if __name__ == "__main__":
    import sys
    import traceback

    tests = {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}
    wanted = sys.argv[1:] or list(tests)
    failed = 0
    for name in wanted:
        try:
            tests[name]()
            print("PASS", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, "-", type(exc).__name__)
            if sys.argv[1:]:
                traceback.print_exc()
    print(f"{len(wanted) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
```

Without any changes, `python test_iforest.py` prints:

```text
FAIL test_c_factor - AssertionError
FAIL test_subsamples - AssertionError
FAIL test_tree_structure - AssertionError
FAIL test_path_length_manual - AssertionError
FAIL test_constant_data - AssertionError
FAIL test_few_points - IndexError
PASS test_ranks_outliers_first
FAIL test_score_scale - AssertionError
PASS test_reproducible
2 passed, 7 failed
```

Task: fix `iforest.py` so that all 9 tests pass. The fixed code must meet the docstring specification, not just turn the tests green. Explain which places you changed and what was wrong in each.

### Part 2 — Where c(n) comes from

Consider $n$ equally spaced one-dimensional points (say $0, 1, \dots, n-1$). With no height limit, keep splitting according to the docstring's split rule until every leaf holds a single point. Let $D(n)$ be the expected depth of the leaf containing a randomly chosen point, with the expectation taken over both the randomness of the tree and the choice of point. $D(1) = 0$.

1. Write down a recurrence for $D(n)$ and implement `expected_depth(n_max)`: it returns a float array of length `n_max` whose `i`-th element is $D(i + 1)$, in $O(n_{max})$ time.
2. Prove that $D(n) = D(n-1) + 2/n$, and hence $D(n) = 2(H_n - 1)$, where $H_n = 1 + 1/2 + \dots + 1/n$ is the exact harmonic number. How does it relate to $c(n)$ in the docstring? The docstring approximates $H(i)$ by $\ln i + 0.5772156649$; how far off is $c(n)$ for $n = 3$ and for $n = 256$? Why is $c(2) = 1$ specified separately?

```py
def expected_depth(n_max):
    """[D(1), D(2), ..., D(n_max)] as a float array; n_max >= 1."""
```

```text
expected_depth(4) = [0, 1, 5/3, 13/6] ≈ [0.0, 1.0, 1.6667, 2.1667]
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

### Part 1

**Approach.** Fix the crash first: the `IndexError` in `test_few_points` points to the indices in `samples_`. Then use the docstring as a checklist and go through it item by item: how the subsamples are drawn, the height limit, the leaf rules, what `size_` records, how `h(x) = e + c(size)` is computed, the three-case definition of $c(n)$, and the score formula. `test_path_length_manual` is a small tree worked out by hand, which makes it the best reference for the scoring part: it catches three bugs at once. Note that `test_ranks_outliers_first` passes from the start: several bugs change only the scale of the scores, not the ranking, so you can't find them just by checking "does it pick out the anomalies".

There are 6 bugs. The last column of the table lists the tests that fail when "only this bug remains and the other 5 are fixed" (the check code runs each case):

| # | Location | What's wrong | Tests that fail with only this bug |
|---|---|---|---|
| 1 | `c_factor` | `np.zeros_like(n)` inherits the integer dtype, so float results are truncated | `test_c_factor`, `test_path_length_manual` |
| 2 | Subsampling in `fit` | `argsort(..., axis=0)` sorts along the wrong axis | `test_subsamples`, `test_few_points` (IndexError) |
| 3 | `_grow` | Returns before recording `size_`, so leaf sizes stay 0 | `test_tree_structure`, `test_score_scale` |
| 4 | `path_length` | Points that have reached a leaf keep descending | `test_path_length_manual`, `test_constant_data` |
| 5 | `score` | Uses `max_samples` instead of the actual `max_samples_` | `test_constant_data`, `test_few_points` |
| 6 | `score` | Operator precedence: `2.0 ** -mean_path / norm` | `test_path_length_manual`, `test_constant_data`, `test_few_points`, `test_score_scale` |

**Bug 1: `np.zeros_like` inherits the integer dtype.** The input to `c_factor` is always integer: the leaf sizes `size_` are an integer array, and `max_samples_` is a Python integer. `np.zeros_like(n)` creates an integer output array too, and the floats assigned into it later are silently truncated: $c(3) = 1.207$ becomes 1, and $c(256) = 10.245$ becomes 10. The docstring says it returns a float array. `test_c_factor` checks the values directly; in `test_path_length_manual`, $c(4) = 1.85$ is truncated to 1. Change it to `np.zeros(n.shape)` (float64 by default).

**Bug 2: `argsort` sorts along the wrong axis.** What we want is one random permutation of $0..n-1$ per tree (per row), taking the first $\psi$ as the subsample. `axis=0` instead sorts along the tree direction, so each column is a permutation of $0..n_{trees}-1$, and `samples_` ends up holding tree numbers, not row numbers. With fewer trees than samples (`test_subsamples`: 40 trees, 300 rows), only the first 40 rows are ever used; with more trees than samples (`test_few_points`: 200 trees, 8 rows), the indices go out of bounds and raise `IndexError`. Change it to `axis=1`.

**Bug 3: returning before recording.** `self.size_[t, node] = len(S)` has been placed after the first `return`. Nodes that become leaves because they reached the height limit or have fewer than 2 points never get their size recorded, and it stays 0. The docstring says `size_` is the number of samples that reached the node. At scoring time, the $c(\text{size})$ of these leaves becomes $c(0) = 0$. Leaves at the height limit that should still hold several points lose "the depth of the part that wasn't grown", so normal points get shorter paths and higher scores: in `test_score_scale` the median inlier score rises from about 0.44 to about 0.58. `test_tree_structure` catches the child nodes of size 0. Move this assignment back to the first line of the function.

**Bug 4: points that have already reached a leaf keep descending.** `path_length` pushes all points down together for a fixed $l$ rounds. A point that has reached a leaf should stay where it is, i.e. `np.where(internal, 2 * node + 1 + go_right, node)`; the code instead updates `node` unconditionally, letting those points walk on into descendant nodes that don't exist. The edge count isn't wrong (`edges += internal` adds 1 only at internal nodes), but the final lookup hits `size_ = 0` of a nonexistent node, so $c(\text{size})$ is lost. When a leaf holds only 1 point, $c(1) = 0$ and you can't tell the difference; on continuous data, every leaf that stops before the height limit holds just 1 point. Only leaves that "stop before the limit yet hold several points" expose this bug, namely leaves of identical points: in `test_constant_data` the root is a leaf and the score should be 0.5, but it comes out as 1; the same happens with the second tree in the hand-built example, which has only a root.

**Bug 5: using the requested value instead of the actual one.** The score should be normalised by the actual subsample size $\psi$ = `max_samples_` = `min(max_samples, n)`, but the code uses `max_samples` (default 256). When the data has fewer than 256 rows, $c(256) \approx 10.2$ is much larger than $c(\psi)$, and every score is pushed toward 1. `test_few_points` has only 8 points, and the highest inlier score goes from about 0.5 to above 0.8; in the second part of `test_constant_data` (40 rows, default `max_samples`) the scores are no longer 0.5.

**Bug 6: operator precedence.** By Python's precedence rules, `2.0 ** -mean_path / norm` is `(2.0 ** -mean_path) / norm`, not $2^{-E[h]/c(\psi)}$. The score still decreases monotonically with the mean path length, so the ranking is unchanged and `test_ranks_outliers_first` still passes; but the scale is completely wrong: a point whose mean path length equals $c(\psi)$ should score 0.5, but with $\psi = 64$ it gets $2^{-c(64)}/c(64) \approx 0.00075$. Add the parentheses.

**Comparison with the paper.** The fixed code matches Liu et al.'s iForest: each tree uses $\psi$ points drawn without replacement, with height limit $l = \lceil \log_2 \psi \rceil$; at the current node, pick a random feature and a split value uniformly between its minimum and maximum, with `x[q] < p` going left; the path length is the number of edges plus the $c(\cdot)$ correction for the leaf size; the score is $s(x) = 2^{-E[h(x)]/c(\psi)}$. The two special cases $c(2) = 1$ and $c(n \le 1) = 0$ match scikit-learn's implementation (the check code compares the values for $n = 0..1000$). The paper's defaults are $\psi = 256$ and 100 trees.

The fixed `iforest.py`:

```python
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
        self.size_[t, node] = len(S)
        if depth >= self.height_limit_ or len(S) <= 1:
            return
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
```

### Part 2

**Recurrence.** The points are distinct and equally spaced, so when the root's split value $p$ is uniform between $x_1$ and $x_n$ (the docstring's rule amounts to drawing uniformly on $(x_1, x_n]$), it falls into each of the $n - 1$ gaps with probability $1/(n-1)$. If it falls into the $i$-th gap, there are $i$ points on the left and $n - i$ on the right; both sides are still equally spaced points under the same split rule, so the depth distribution within each subtree is the same as for "growing a tree on $i$ points from scratch". The randomly chosen point is on the left with probability $i/n$:

$$D(n) = 1 + \frac{1}{n-1}\sum_{i=1}^{n-1}\left(\frac{i}{n}D(i) + \frac{n-i}{n}D(n-i)\right) = 1 + \frac{2}{n(n-1)}\sum_{k=1}^{n-1} k\,D(k).$$

Keeping a running prefix sum of $\sum k\,D(k)$ makes each $n$ cost $O(1)$:

```python
def expected_depth(n_max):
    depth = np.zeros(n_max)
    weighted = 0.0  # sum of k * D(k) for k < n
    for n in range(2, n_max + 1):
        weighted += (n - 1) * depth[n - 2]
        depth[n - 1] = 1.0 + 2.0 * weighted / (n * (n - 1))
    return depth
```

**Closed form.** Let $W(m) = \sum_{k=1}^{m} k\,D(k)$. From the recurrence, $n(n-1)(D(n) - 1) = 2W(n-1)$ and $(n-1)(n-2)(D(n-1) - 1) = 2W(n-2)$. Subtracting the two, the right-hand side is $2(n-1)D(n-1)$; dividing both sides by $n - 1$ and rearranging gives $n\,D(n) = n\,D(n-1) + 2$, i.e. $D(n) = D(n-1) + 2/n$ (for $n = 2$ it also holds by direct substitution). Summing from $D(1) = 0$: $D(n) = 2(1/2 + \dots + 1/n) = 2(H_n - 1)$.

**Relation to $c(n)$.** With exact harmonic numbers, $2H_{n-1} - 2(n-1)/n = 2H_{n-1} + 2/n - 2 = 2(H_n - 1)$, which is exactly $D(n)$. So $c(n)$ is "the average depth of a point when the tree is grown all the way on $n$ points". The paper describes it as the average path length of an unsuccessful search in a binary search tree, which is the same quantity. Adding $c(m)$ for a leaf that hit the height limit with $m$ points left is an estimate of the average depth of the subtree that wasn't grown. (For general data the gaps have different lengths, so they are no longer equally likely to be split, and $c(n)$ is only an approximation.)

The docstring approximates $H(i)$ by $\ln i + \gamma$. For $n = 3$ the exact value is $5/3 \approx 1.6667$ and the approximation is 1.2074, off by $-0.459$; for $n = 256$ the exact value is about 10.2487 and the approximation is 10.2448, off by only $-0.0039$. The smaller $i$ is, the further $H(i) - \ln i$ is from $\gamma$ ($H(1) - \ln 1 = 1$, while $\gamma \approx 0.577$). For $n = 2$ the approximate formula gives $2(\ln 1 + \gamma) - 1 = 2\gamma - 1 \approx 0.154$, while the true value is 1, which is too far off, so $c(2) = 1$ is specified separately; for $n = 1$ the formula contains $\ln 0$, which is undefined, and a one-point leaf needs no further splitting, so $c(1) = 0$.

### Follow-ups

- **Why subsample? How large should $\psi$ be?** The paper says subsampling mitigates two problems: swamping (normal points too close to anomalies get misclassified) and masking (anomalies form clusters and cover for each other). $\psi = 256$ with 100 trees is basically enough in the paper's experiments, and increasing them further gains little. A small $\psi$ also keeps every tree small, so the training cost is independent of $n$ (apart from the sampling itself).
- **Why stop at the height limit? What does adding $c(\text{size})$ do?** Anomalies are isolated at shallow depths; the deep parts only concern normal points, and their exact depth doesn't matter for the ranking, so there's no need to grow the tree all the way. A leaf that stops at the limit still has $m$ points, and adding $c(m)$ estimates the remaining depth, making truncated paths comparable with fully grown trees. Without this term (bugs 3 and 4), normal points get paths that are too short and scores that are too high.
- **Why normalise by $c(\psi)$ rather than $c(n)$?** The trees are grown on $\psi$ points, so the expected path length of an ordinary point is about $c(\psi)$, independent of the total sample size $n$. After normalising by $c(\psi)$, a point with $E[h] = c(\psi)$ scores 0.5; the closer a score is to 1, the more anomalous, and points well below 0.5 can roughly be regarded as normal. Use the wrong number (bug 5) and the whole scale is off.
- **How do scikit-learn's `score_samples`, `decision_function` and `predict` relate to $s(x)$ here?** `score_samples` returns $-s(x)$ (lower is more anomalous). With `contamination='auto'`, `offset_ = -0.5` and `decision_function = score_samples - offset_` $= 0.5 - s(x)$; `predict` gives $-1$ (anomaly) when that is below 0 (i.e. $s(x) > 0.5$) and 1 otherwise. When `contamination` is set to a fraction $r$, `offset_` becomes the $100r$-th percentile of `score_samples` on the training set. (The check code verifies these against scikit-learn.)
- **Complexity and storage?** Each tree has at most $l$ levels, all the nodes on a level together hold at most $\psi$ points, and computing the minima and maxima costs $O(\psi d)$, so training one tree is $O(\psi d \log \psi)$ in the worst case (the height limit makes this a worst-case bound, not just an expectation). Scoring is $l$ rounds of vectorised descent at $O(n_{trees} \cdot n)$ each, $O(n_{trees}\, n \log \psi)$ in total. The heap layout stores no child pointers, since the children of node $i$ are simply $2i+1$ and $2i+2$, which is what lets all trees and all points descend at once; the cost is wasted slots in unbalanced trees. Since $2^l < 2\psi$, each tree's array has length $2^{l+1} - 1 < 4\psi$, so the waste is bounded.

<details>
<summary>Checks (runnable)</summary>

```python
import ast
import contextlib
import io
import sys
import time
import types
import warnings
from fractions import Fraction
from functools import lru_cache

import numpy as np

T0 = time.perf_counter()

# The statement's two files, byte for byte.
BUGGY_SOURCE = r'''"""Isolation Forest (Liu, Ting & Zhou, 2008) for anomaly detection.

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
    out = np.zeros_like(n)
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
        order = np.argsort(rng.random((self.n_trees, len(X))), axis=0)
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
            node = 2 * node + 1 + go_right
            edges += internal
        return edges + c_factor(self.size_[tree, node])

    def score(self, X):
        """Anomaly score s(x) of every row of X, shape (n,)."""
        mean_path = self.path_length(X).mean(axis=0)
        norm = c_factor(self.max_samples)
        return 2.0 ** -mean_path / norm
'''

TEST_SOURCE = r'''"""Tests for iforest.py.

Run all of them:   python test_iforest.py
Run one of them:   python test_iforest.py test_few_points     (prints the traceback)
"""
import numpy as np

from iforest import IsolationForest, c_factor


def test_c_factor():
    c = c_factor(np.array([0, 1, 2, 3, 4, 256]))
    assert np.allclose(c, [0.0, 0.0, 1.0, 1.2073923576, 1.8516559071, 10.2447709201])
    assert c_factor(np.array([[2, 5], [9, 1]])).shape == (2, 2)
    assert np.isclose(float(c_factor(100)), 8.3646710301)
    assert np.all(np.diff(c_factor(np.arange(2, 300))) > 0)


def test_subsamples():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    forest = IsolationForest(n_trees=40, max_samples=64, seed=0).fit(X)
    assert forest.max_samples_ == 64 and forest.height_limit_ == 6
    assert forest.samples_.shape == (40, 64)
    assert forest.samples_.min() >= 0 and forest.samples_.max() < 300
    assert all(len(np.unique(row)) == 64 for row in forest.samples_)
    assert len(np.unique(forest.samples_)) > 270
    assert len({tuple(row) for row in forest.samples_.tolist()}) == 40


def test_tree_structure():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    forest = IsolationForest(n_trees=20, max_samples=100, seed=1).fit(X)
    limit = forest.height_limit_
    assert limit == 7
    assert forest.feature_.shape == forest.threshold_.shape == forest.size_.shape == (20, 255)
    assert np.all(forest.size_[:, 0] == 100)
    internal = forest.feature_ >= 0
    assert not internal[:, 2 ** limit - 1:].any()
    t, i = np.nonzero(internal)
    left, right = forest.size_[t, 2 * i + 1], forest.size_[t, 2 * i + 2]
    assert np.all(left >= 1) and np.all(right >= 1)
    assert np.array_equal(forest.size_[t, i], left + right)
    is_leaf = ~internal & (forest.size_ > 0)
    assert np.all((forest.size_ * is_leaf).sum(axis=1) == 100)
    assert np.all(forest.threshold_[~internal] == 0.0)


def test_path_length_manual():
    forest = IsolationForest(n_trees=2, max_samples=4)
    forest.max_samples_ = 4
    forest.height_limit_ = 2
    forest.feature_ = np.array([[0, -1, 1, -1, -1, -1, -1],
                                [-1, -1, -1, -1, -1, -1, -1]])
    forest.threshold_ = np.array([[0.5, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    forest.size_ = np.array([[4, 1, 3, 0, 0, 2, 1],
                             [4, 0, 0, 0, 0, 0, 0]])
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 3.0], [0.5, 2.0]])
    c4 = 1.8516559071
    paths = forest.path_length(X)
    assert paths.shape == (2, 4)
    assert np.allclose(paths, [[1.0, 3.0, 2.0, 2.0], [c4, c4, c4, c4]])
    assert np.allclose(forest.score(X), [0.5864059, 0.4032969, 0.4863082, 0.4863082])


def test_constant_data():
    X = np.full((40, 3), 2.5)
    forest = IsolationForest(n_trees=10, max_samples=16, seed=0).fit(X)
    assert np.all(forest.feature_ == -1)
    assert np.all(forest.size_[:, 0] == 16)
    assert np.allclose(forest.score(X), 0.5)
    assert np.allclose(forest.score(np.array([[0.0, 9.0, -3.0]])), 0.5)
    forest = IsolationForest(n_trees=10, seed=0).fit(X)
    assert forest.max_samples_ == 40
    assert np.allclose(forest.score(X), 0.5)


def test_few_points():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [50.0]])
    forest = IsolationForest(n_trees=200, seed=0).fit(X)
    assert forest.max_samples_ == 8 and forest.height_limit_ == 3
    assert forest.samples_.shape == (200, 8)
    scores = forest.score(X)
    assert scores.shape == (8,)
    assert np.all((scores > 0) & (scores <= 1))
    assert scores.argmax() == 7
    assert scores[7] > 0.7
    assert scores[:7].max() < 0.65


def make_blob_with_outliers():
    rng = np.random.default_rng(3)
    inliers = rng.normal(size=(400, 2))
    outliers = rng.uniform(5.0, 8.0, size=(10, 2)) * rng.choice([-1.0, 1.0], size=(10, 2))
    return np.concatenate([inliers, outliers])


def test_ranks_outliers_first():
    X = make_blob_with_outliers()
    scores = IsolationForest(n_trees=60, max_samples=64, seed=0).fit(X).score(X)
    assert scores.shape == (410,)
    assert np.mean(scores[:400] < scores[400:].min()) > 0.97


def test_score_scale():
    X = make_blob_with_outliers()
    scores = IsolationForest(n_trees=60, max_samples=64, seed=0).fit(X).score(X)
    assert np.all((scores > 0) & (scores <= 1))
    assert scores[400:].min() > 0.6
    assert 0.35 < np.median(scores[:400]) < 0.53


def test_reproducible():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 4))
    a = IsolationForest(n_trees=8, max_samples=32, seed=5).fit(X)
    b = IsolationForest(n_trees=8, max_samples=32, seed=5).fit(X)
    c = IsolationForest(n_trees=8, max_samples=32, seed=6).fit(X)
    assert np.array_equal(a.samples_, b.samples_)
    assert np.array_equal(a.feature_, b.feature_)
    assert np.array_equal(a.threshold_, b.threshold_)
    assert not np.array_equal(a.threshold_, c.threshold_)
    assert np.array_equal(a.score(X), b.score(X))


if __name__ == "__main__":
    import sys
    import traceback

    tests = {k: v for k, v in globals().items() if k.startswith("test_") and callable(v)}
    wanted = sys.argv[1:] or list(tests)
    failed = 0
    for name in wanted:
        try:
            tests[name]()
            print("PASS", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, "-", type(exc).__name__)
            if sys.argv[1:]:
                traceback.print_exc()
    print(f"{len(wanted) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
'''

EXPECTED_OUTPUT = """FAIL test_c_factor - AssertionError
FAIL test_subsamples - AssertionError
FAIL test_tree_structure - AssertionError
FAIL test_path_length_manual - AssertionError
FAIL test_constant_data - AssertionError
FAIL test_few_points - IndexError
PASS test_ranks_outliers_first
FAIL test_score_scale - AssertionError
PASS test_reproducible
2 passed, 7 failed
"""

# (bug, buggy text, fixed text): every fix is a one-place replacement.
FIXES = [
    ("bug 1", "out = np.zeros_like(n)", "out = np.zeros(n.shape)"),
    ("bug 2", "order = np.argsort(rng.random((self.n_trees, len(X))), axis=0)",
              "order = np.argsort(rng.random((self.n_trees, len(X))), axis=1)"),
    ("bug 3", "        if depth >= self.height_limit_ or len(S) <= 1:\n"
              "            return\n"
              "        self.size_[t, node] = len(S)\n",
              "        self.size_[t, node] = len(S)\n"
              "        if depth >= self.height_limit_ or len(S) <= 1:\n"
              "            return\n"),
    ("bug 4", "node = 2 * node + 1 + go_right",
              "node = np.where(internal, 2 * node + 1 + go_right, node)"),
    ("bug 5", "norm = c_factor(self.max_samples)", "norm = c_factor(self.max_samples_)"),
    ("bug 6", "return 2.0 ** -mean_path / norm", "return 2.0 ** (-mean_path / norm)"),
]
ALL = [name for name, _, _ in FIXES]


def build_source(fixed):
    src = BUGGY_SOURCE
    for name, old, new in FIXES:
        assert src.count(old) == 1 and new not in src, name
        if name in fixed:
            src = src.replace(old, new)
    return src


def load(src):
    mod = types.ModuleType("iforest")
    exec(compile(src, "iforest.py", "exec"), mod.__dict__)
    return mod


@contextlib.contextmanager
def quiet():
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield


def exec_tests(mod, name):
    saved = sys.modules.get("iforest")
    sys.modules["iforest"] = mod
    try:
        ns = {"__name__": name}
        exec(compile(TEST_SOURCE, "test_iforest.py", "exec"), ns)
    finally:
        if saved is None:
            del sys.modules["iforest"]
        else:
            sys.modules["iforest"] = saved
    return ns


def run_tests(mod):
    """{test name: 'ok' or the exception type} for every test in test_iforest.py."""
    ns = exec_tests(mod, "test_iforest")
    result = {}
    for name, fn in ns.items():
        if name.startswith("test_") and callable(fn):
            try:
                with quiet():
                    fn()
                result[name] = "ok"
            except Exception as exc:  # noqa: BLE001
                result[name] = type(exc).__name__
    return result


def runner_output(mod):
    """What `python test_iforest.py` prints when iforest.py is `mod`."""
    out, saved_argv = io.StringIO(), sys.argv
    sys.argv = ["test_iforest.py"]
    try:
        with contextlib.redirect_stdout(out), quiet():
            try:
                exec_tests(mod, "__main__")
            except SystemExit:
                pass
    finally:
        sys.argv = saved_argv
    return out.getvalue()


def failures(mod):
    return {k: v for k, v in run_tests(mod).items() if v != "ok"}


# 1. The buggy file prints exactly the listing shown in the statement.
assert runner_output(load(BUGGY_SOURCE)) == EXPECTED_OUTPUT

# 2. With all six fixes, all 9 tests pass.
fixed_mod = load(build_source(ALL))
result = run_tests(fixed_mod)
assert len(result) == 9 and set(result.values()) == {"ok"}, result

# 3. Each bug matters on its own: fix the other five, and exactly these tests fail.
A = "AssertionError"
ONLY_THIS_BUG_LEFT = {
    "bug 1": {"test_c_factor": A, "test_path_length_manual": A},
    "bug 2": {"test_subsamples": A, "test_few_points": "IndexError"},
    "bug 3": {"test_tree_structure": A, "test_score_scale": A},
    "bug 4": {"test_path_length_manual": A, "test_constant_data": A},
    "bug 5": {"test_constant_data": A, "test_few_points": A},
    "bug 6": {"test_path_length_manual": A, "test_constant_data": A, "test_few_points": A,
              "test_score_scale": A},
}
only = {}
for bug, expected in ONLY_THIS_BUG_LEFT.items():
    only[bug] = load(build_source([b for b in ALL if b != bug]))
    got = failures(only[bug])
    assert got == expected, (bug, got)

# 4. The symptoms quoted in the bug paragraphs, each with only that bug left.
assert only["bug 1"].c_factor(np.array([3, 4, 256])).tolist() == [1, 1, 10]
assert float(only["bug 1"].c_factor(64)) == 7.0
X300 = np.random.default_rng(0).normal(size=(300, 2))
assert only["bug 2"].IsolationForest(40, 64, seed=0).fit(X300).samples_.max() < 40
blob = exec_tests(fixed_mod, "test_iforest")["make_blob_with_outliers"]()


def inlier_median(mod):
    return np.median(mod.IsolationForest(60, 64, seed=0).fit(blob).score(blob)[:400])


medians = inlier_median(fixed_mod), inlier_median(only["bug 3"])
print("inlier median, fixed vs bug 3 only: %.3f, %.3f" % medians)
assert 0.42 < medians[0] < 0.46 and 0.55 < medians[1] < 0.61
X = np.random.default_rng(1).normal(size=(500, 3))
f3 = only["bug 3"].IsolationForest(20, 100, seed=1).fit(X)
t, i = np.nonzero(f3.feature_ >= 0)
assert np.any(f3.size_[t, 2 * i + 1] == 0)
assert np.allclose(only["bug 4"].IsolationForest(10, 16, seed=0).fit(np.full((40, 3), 2.5))
                   .score(np.full((5, 3), 2.5)), 1.0)
few = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [50.0]])
top = IsolationForest(200, seed=0).fit(few).score(few)[:7].max()
print("few points, highest inlier score when fixed: %.3f" % top)
assert 0.45 < top < 0.55
top5 = only["bug 5"].IsolationForest(200, seed=0).fit(few).score(few)[:7].max()
print("few points, highest inlier score with bug 5 only: %.3f" % top5)
assert top5 > 0.8
c64 = float(c_factor(64))
assert abs(2.0 ** -c64 / c64 - 0.00075) < 0.00005


# 5. The docstrings (the specification) are untouched, and the buggy file has no comments.
def docstrings(src):
    tree = ast.parse(src)
    return [ast.get_docstring(node) for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))]


assert docstrings(BUGGY_SOURCE) == docstrings(build_source(ALL))
assert "#" not in BUGGY_SOURCE

# 6. The fixed code printed above (already run in this namespace) passes all tests and
#    builds exactly the same forests as the replacement-built module.
page = types.ModuleType("iforest")
page.__dict__.update(EULER_GAMMA=EULER_GAMMA, c_factor=c_factor, IsolationForest=IsolationForest)
assert set(run_tests(page).values()) == {"ok"}
rng = np.random.default_rng(10)
for trial in range(4):
    X = rng.integers(0, 4, size=(90, 3)).astype(float)
    a = IsolationForest(7, 32, seed=trial).fit(X)
    b = fixed_mod.IsolationForest(7, 32, seed=trial).fit(X)
    for attr in ("samples_", "feature_", "threshold_", "size_"):
        assert np.array_equal(getattr(a, attr), getattr(b, attr))
    assert np.array_equal(a.score(X), b.score(X))

# 7. c(n): agrees with scikit-learn's implementation, and with the exact 2 (H_n - 1)
#    except for the ln-approximation error quoted in Part 2.
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.ensemble._iforest import _average_path_length

ns = np.arange(0, 1001)
assert np.allclose(c_factor(ns), _average_path_length(ns), rtol=0, atol=1e-9)
harmonic = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, 1001))])
exact = np.where(ns >= 2, 2 * (harmonic[np.maximum(ns, 1)] - 1), 0.0)
assert np.isclose(exact[3], 5 / 3) and np.isclose(exact[2], 1.0)
assert round(float(c_factor(3) - exact[3]), 4) == -0.4593
assert round(float(c_factor(256) - exact[256]), 4) == -0.0039
assert round(2 * EULER_GAMMA - 1, 3) == 0.154
assert round(float(c_factor(3)), 4) == 1.2074 and round(float(c_factor(256)), 4) == 10.2448
assert round(float(exact[256]), 4) == 10.2487


# 8. Every tree obeys the module docstring: route each subsample in plain Python.
def check_forest(forest, X):
    psi, limit = forest.max_samples_, forest.height_limit_
    assert psi == min(forest.max_samples, len(X)) and limit == int(np.ceil(np.log2(psi)))
    n_nodes = 2 ** (limit + 1) - 1
    for t in range(forest.n_trees):
        assert len(set(forest.samples_[t].tolist())) == psi
        visited = set()
        stack = [(0, 0, forest.samples_[t].tolist())]
        while stack:
            node, depth, rows = stack.pop()
            visited.add(node)
            assert forest.size_[t, node] == len(rows)
            sub = X[rows]
            movable = [f for f in range(X.shape[1]) if len(rows) and sub[:, f].min() < sub[:, f].max()]
            q = forest.feature_[t, node]
            if depth >= limit or len(rows) <= 1 or not movable:
                assert q == -1 and forest.threshold_[t, node] == 0.0
                continue
            p = forest.threshold_[t, node]
            assert q in movable and sub[:, q].min() < p <= sub[:, q].max()
            left = [r for r in rows if X[r, q] < p]
            right = [r for r in rows if not X[r, q] < p]
            assert left and right
            stack += [(2 * node + 1, depth + 1, left), (2 * node + 2, depth + 1, right)]
        rest = [i for i in range(n_nodes) if i not in visited]
        assert np.all(forest.size_[t, rest] == 0) and np.all(forest.feature_[t, rest] == -1)
        assert np.all(forest.threshold_[t, rest] == 0.0)


def path_slow(forest, t, x):
    node, edges = 0, 0
    while forest.feature_[t, node] >= 0:
        q, p = forest.feature_[t, node], forest.threshold_[t, node]
        node = 2 * node + 1 if x[q] < p else 2 * node + 2
        edges += 1
    return edges + float(c_factor(forest.size_[t, node]))


for trial in range(6):
    n, d = int(rng.integers(2, 150)), int(rng.integers(1, 5))
    X = rng.integers(0, 3, size=(n, d)).astype(float) if trial % 2 else rng.normal(size=(n, d))
    forest = IsolationForest(int(rng.integers(1, 12)), int(rng.integers(2, 80)), seed=trial).fit(X)
    check_forest(forest, X)
    Q = np.concatenate([X[:20], rng.normal(size=(10, d)) * 3])
    slow = [[path_slow(forest, t, q) for q in Q] for t in range(forest.n_trees)]
    assert np.allclose(forest.path_length(Q), slow)
    assert np.allclose(forest.score(Q), 2.0 ** (-np.mean(slow, axis=0) / c_factor(forest.max_samples_)))


# 9. On n evenly spaced points, the mean path length over many trees matches an exact
#    dynamic program for the height-limited expectation (same recurrence as Part 2).
def expected_path(n, limit):
    @lru_cache(None)
    def T(k, e):
        if e >= limit or k <= 1:
            return float(c_factor(k))
        return 1.0 + 2.0 / (k * (k - 1)) * sum(j * T(j, e + 1) for j in range(1, k))
    return T(n, 0)


for n in (11, 16):
    x = np.arange(n, dtype=float)[:, None]
    forest = IsolationForest(n_trees=1500, seed=n).fit(x)
    per_tree = forest.path_length(x).mean(axis=1)
    err = abs(per_tree.mean() - expected_path(n, forest.height_limit_))
    assert err < 5 * per_tree.std() / np.sqrt(len(per_tree)), (n, err)

# 10. Statistical agreement with scikit-learn on a two-cluster data set with outliers.
X = np.concatenate([blob, np.random.default_rng(7).normal(size=(300, 2)) * [3.0, 0.5] + [2.0, 0.0]])
ours = IsolationForest(n_trees=300, max_samples=128, seed=1).fit(X).score(X)
sk = SklearnIsolationForest(n_estimators=300, max_samples=128, bootstrap=False, random_state=1).fit(X)
theirs = -sk.score_samples(X)
print(f"vs scikit-learn: corr {np.corrcoef(ours, theirs)[0, 1]:.4f}, "
      f"mean |diff| {np.abs(ours - theirs).mean():.4f}")
assert np.corrcoef(ours, theirs)[0, 1] > 0.99 and np.abs(ours - theirs).mean() < 0.015

# 11. Follow-up 4: how scikit-learn turns s(x) into decisions.
assert sk.offset_ == -0.5
assert np.allclose(sk.decision_function(X), sk.score_samples(X) + 0.5)
assert np.array_equal(sk.predict(X), np.where(sk.decision_function(X) < 0, -1, 1))
sk10 = SklearnIsolationForest(n_estimators=50, contamination=0.1, random_state=0).fit(X)
assert np.isclose(sk10.offset_, np.percentile(sk10.score_samples(X), 10))

# 12. Follow-up 5: the heap arrays have fewer than 4 * psi slots per tree.
for psi in range(2, 2000):
    assert 2 ** (int(np.ceil(np.log2(psi))) + 1) - 1 < 4 * psi

# 13. Part 2: the example, exact rationals, the closed form, and the link to c(n).
assert np.allclose(expected_depth(4), [0.0, 1.0, 5 / 3, 13 / 6])
D = [Fraction(0)]
for n in range(2, 41):
    D.append(1 + Fraction(2, n * (n - 1)) * sum(k * D[k - 1] for k in range(1, n)))
for n in range(1, 41):
    H_n = sum(Fraction(1, i) for i in range(1, n + 1))
    assert D[n - 1] == 2 * (H_n - 1)
    if n >= 2:
        assert D[n - 1] == D[n - 2] + Fraction(2, n)
big = expected_depth(1000)
assert np.allclose(big, [float(d) for d in D] + list(exact[41:1001]))
assert np.allclose(big[1:], exact[2:1001])

print(f"checks took {time.perf_counter() - T0:.1f} s")
print("all checks passed")
```

</details>

</details>
