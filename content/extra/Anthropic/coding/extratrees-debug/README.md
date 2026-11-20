# Debug a vectorised Extra-Trees implementation

## Problem

The `trees.py` below is a vectorized NumPy implementation of an Extremely Randomized Trees classifier (Extra-Trees; Geurts, Ernst and Wehenkel, 2006). A single tree, `ExtraTree`, stores its nodes in a few flat arrays, and batch prediction walks all samples down the tree together; `ExtraTreesClassifier` is an ensemble of trees that predicts by majority vote. The code mostly works, but it has several bugs: some inputs make it crash, others make the results wrong or the accuracy worse. `test_trees.py` is the test file you are given.

Several interview-prep sites say the second-round online assessment (OA) for Anthropic Fellows is this kind of problem: fix `trees.py` within 60 minutes so that all the tests pass. The real code has not been made public. The code, bugs and tests below are our own, written to the original rules:

- Change only `trees.py`. Do not change the tests or the docstrings. They are all correct; the docstrings are the spec.
- Keep it vectorized: do not rewrite NumPy vector operations as nested Python loops.
- No scikit-learn or any other ready-made tree implementation, and no AI. You may consult the NumPy / Python docs. Accounts differ on whether web search is allowed; prepare as if it is not.
- Every test is worth the same. Tests have a 30-second time limit and 4 GB of memory. An infinite loop will hang the platform. A hint from the original: `test_small_features` and `test_ensemble` are the hardest and can be left for last.
- The original platform uses pytest; you can add `breakpoint()` to the code and debug with `pytest -k <test name> --pdb`.

Practicing locally: save the two code blocks below as `trees.py` and `test_trees.py` in the same directory and run `python test_trees.py` (only NumPy is needed, not pytest). `python test_trees.py test_ensemble` runs a single test and prints the traceback. Time yourself for 60 minutes.

### Part 1 — Make all the tests pass

`trees.py`:

```py
"""Extremely Randomized Trees (Geurts, Ernst & Wehenkel, 2006) for classification.

The docstrings in this file are the specification.

Growing a tree.  Every tree is grown on the whole training set (no bootstrap).
A node that holds the sample subset S becomes a leaf if any of these is true:

  (a) |S| < min_samples_split,
  (b) all labels in S are equal,
  (c) every feature is constant in S.

Otherwise the node is split:

  1. Let A be the set of features that are not constant in S.  Draw
     k = min(max_features, |A|) distinct features from A uniformly at random.
  2. For each drawn feature f, draw one cut-point uniformly from
     [min_S f, max_S f).  A draw equal to min_S f is replaced by max_S f.
  3. Keep the candidate (f, cut) with the highest score (see split_scores);
     on ties, the one drawn first.
  4. Samples with x[f] < cut go to the left child, all other samples go to the
     right child.  By construction both children receive at least one sample.

Storage.  Nodes are numbered in pre-order (the root is 0, and the left subtree
of a node is numbered before its right subtree) and kept in flat arrays of
length n_nodes:

  feature_[i]    split feature of node i, -1 for a leaf
  threshold_[i]  cut-point of node i, nan for a leaf
  left_[i]       id of the left child, -1 for a leaf
  right_[i]      id of the right child, -1 for a leaf
  value_[i]      shape (n_classes,): how many training samples of each class
                 reached node i
"""
import numpy as np


def entropy(counts):
    """Shannon entropy, in bits, of each row of `counts`.

    counts: array of shape (..., m) holding non-negative counts.
    Returns an array of shape (...).  0 * log(0) counts as 0, and a row that
    is all zeros has entropy 0.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=-1)


def split_scores(go_left, onehot):
    """Score k candidate splits of the same n samples.

    go_left: bool array (n, k); go_left[i, j] is True if sample i goes to the
             left child under candidate j.
    onehot:  array (n, n_classes), the one-hot labels of the samples.

    Returns an array of shape (k,) with

        score = 2 * I / (H_split + H_class)

    where H_class is the entropy of the labels, H_split is the entropy of the
    left/right partition and I is their mutual information, i.e. H_class minus
    the size-weighted mean of the label entropies of the two children.
    The score lies in [0, 1].  It is 0 when the denominator is 0.
    """
    n = len(onehot)
    total = onehot.sum(axis=0)
    left = go_left.T.astype(float) @ onehot
    right = total - left
    n_left = left.sum(axis=1)
    n_right = n - n_left
    h_class = entropy(total)
    h_split = entropy(np.stack([n_left, n_right], axis=1))
    gain = h_class - (n_left * entropy(left) + n_right * entropy(right)) / n
    denom = h_split + h_class
    return np.divide(2.0 * gain, denom, out=np.zeros_like(gain), where=denom > 0)


class ExtraTree:
    """One extremely randomized classification tree.

    max_features:      number of candidate features drawn at each node (K >= 1).
    min_samples_split: a node with fewer samples is never split (n_min >= 2).
    seed:              anything np.random.default_rng accepts.
    """

    def __init__(self, max_features, min_samples_split=2, seed=None):
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        """Grow the tree on X (n, d) and labels y (n,), n >= 1.  Returns self.

        Sets classes_ (the sorted distinct labels) and the node arrays that
        the module docstring describes.
        """
        X = np.asarray(X, dtype=float)
        self.classes_, codes = np.unique(y, return_inverse=True)
        onehot = np.eye(len(self.classes_))[codes]
        self._rng = np.random.default_rng(self.seed)
        self._nodes = []
        self._grow(X, onehot, np.arange(len(X)))
        feature, threshold, left, right, value = zip(*self._nodes)
        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.left_ = np.array(left)
        self.right_ = np.array(right)
        self.value_ = np.array(value)
        return self

    def _grow(self, X, onehot, idx):
        """Append the subtree for the samples `idx`; return the id of its root."""
        node = len(self._nodes)
        counts = onehot[idx].sum(axis=0)
        self._nodes.append([-1, np.nan, -1, -1, counts])
        split = self._pick_split(X, onehot, idx, counts)
        if split is not None:
            feature, cut = split
            go_left = X[idx, feature] < cut
            left = self._grow(X, onehot, idx[go_left])
            right = self._grow(X, onehot, idx[~go_left])
            self._nodes[node][:4] = [feature, cut, left, right]
        return node

    def _pick_split(self, X, onehot, idx, counts):
        """Return (feature, cut) for the node holding `idx`, or None for a leaf."""
        if len(idx) <= self.min_samples_split or counts.max() == counts.sum():
            return None
        lo, hi = X.min(axis=0), X.max(axis=0)
        if np.any(lo == hi):
            return None
        candidates = np.flatnonzero(lo < hi)
        feats = self._rng.choice(candidates, size=self.max_features, replace=False)
        cuts = self._rng.uniform(lo[feats], hi[feats])
        cuts = np.where(cuts > lo[feats], cuts, hi[feats])
        go_left = X[np.ix_(idx, feats)] < cuts
        best = np.argmax(split_scores(go_left, onehot[idx]))
        return int(feats[best]), float(cuts[best])

    def apply(self, X):
        """Id of the leaf that each row of X ends up in, shape (n,).

        Every sample starts at the root.  At an internal node i it moves to
        left_[i] if x[feature_[i]] < threshold_[i], otherwise to right_[i].
        """
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        while True:
            rows = np.flatnonzero(self.feature_[node] >= 0)
            if rows.size == 0:
                return node
            at = node[rows]
            go_left = X[rows, self.feature_[at]] <= self.threshold_[at]
            node[rows] = np.where(go_left, self.left_[at], self.right_[at])

    def predict_proba(self, X):
        """Class frequencies in the leaf of each sample, shape (n, n_classes)."""
        counts = self.value_[self.apply(X)]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Most frequent class in the leaf of each sample; ties go to the
        smallest class."""
        return self.classes_[self.value_[self.apply(X)].argmax(axis=1)]


class ExtraTreesClassifier:
    """An ensemble of ExtraTree, all grown on the same training set.

    n_estimators:      number of trees (M >= 1).
    max_features:      K of every tree; None means round(sqrt(n_features)).
    min_samples_split: n_min of every tree.
    seed:              the trees get independent random streams derived from it.
    """

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        """Grow n_estimators trees on (X, y).  Returns self."""
        X = np.asarray(X, dtype=float)
        k = self.max_features
        if k is None:
            k = max(1, round(np.sqrt(X.shape[1])))
        streams = np.random.SeedSequence(self.seed).spawn(self.n_estimators)
        self.trees_ = [ExtraTree(k, self.min_samples_split, s).fit(X, y) for s in streams]
        self.classes_ = self.trees_[0].classes_
        return self

    def predict_proba(self, X):
        """Fraction of the trees that vote for each class, shape (n, n_classes).

        Every tree casts exactly one vote per sample, for the class that
        ExtraTree.predict returns.  Each row therefore sums to 1.
        """
        X = np.asarray(X, dtype=float)
        votes = np.stack([t.value_[t.apply(X)].argmax(axis=1) for t in self.trees_])
        counts = np.zeros((len(X), len(self.classes_)))
        counts[np.arange(len(X)), votes] += 1
        return counts / len(self.trees_)

    def predict(self, X):
        """Majority vote of the trees; ties go to the smallest class."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
```

`test_trees.py`:

```py
"""Tests for trees.py.

Run all of them:   python test_trees.py
Run one of them:   python test_trees.py test_ensemble     (prints the traceback)
"""
import numpy as np

from trees import ExtraTree, ExtraTreesClassifier, entropy, split_scores


def test_split_scores():
    h = entropy([[2, 2], [4, 0], [0, 0], [1, 3]])
    assert np.allclose(h, [1.0, 0.0, 0.0, 0.8112781244591328])
    onehot = np.eye(2)[[0, 0, 0, 1, 1, 1]]
    go_left = np.array([[1, 1, 1, 1, 0],
                        [1, 1, 0, 1, 0],
                        [1, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0]], dtype=bool)
    scores = split_scores(go_left, onehot)
    assert scores.shape == (5,)
    assert np.allclose(scores, [1.0, 0.4787039714, 0.0817041659, 0.0, 0.0])


def test_apply_manual_tree():
    tree = ExtraTree(max_features=1)
    tree.classes_ = np.array(["a", "b", "c"])
    tree.feature_ = np.array([0, -1, 1, -1, -1])
    tree.threshold_ = np.array([2.0, np.nan, 5.0, np.nan, np.nan])
    tree.left_ = np.array([1, -1, 3, -1, -1])
    tree.right_ = np.array([2, -1, 4, -1, -1])
    tree.value_ = np.array([[3.0, 3.0, 4.0],
                            [3.0, 0.0, 0.0],
                            [0.0, 3.0, 4.0],
                            [0.0, 3.0, 1.0],
                            [0.0, 0.0, 3.0]])
    X = np.array([[1.0, 9.0], [2.0, 4.0], [3.0, 5.0], [2.5, 4.9], [-1.0, 0.0], [7.0, 6.0]])
    assert tree.apply(X).tolist() == [1, 3, 4, 3, 1, 4]
    assert tree.predict(X).tolist() == ["a", "b", "c", "b", "a", "c"]
    assert np.allclose(tree.predict_proba(X)[:2], [[1.0, 0.0, 0.0], [0.0, 0.75, 0.25]])


def test_noisy_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 3, size=200)
    for n_min in (2, 5, 20):
        tree = ExtraTree(max_features=2, min_samples_split=n_min, seed=n_min).fit(X, y)
        is_leaf = tree.feature_ < 0
        size = tree.value_.sum(axis=1)
        is_pure = (tree.value_ > 0).sum(axis=1) == 1
        assert np.all(size[~is_leaf] >= n_min)
        assert np.all(is_pure[is_leaf] | (size[is_leaf] < n_min))
    assert len(ExtraTree(max_features=2, min_samples_split=201, seed=0).fit(X, y).feature_) == 1
    assert len(ExtraTree(max_features=2, min_samples_split=200, seed=0).fit(X, y).feature_) >= 3
    full = ExtraTree(max_features=2, seed=1).fit(X, y)
    assert np.array_equal(full.predict(X), y)


def test_tree_structure():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    tree = ExtraTree(max_features=3, seed=0).fit(X, y)
    n_nodes = len(tree.feature_)
    is_leaf = tree.feature_ < 0
    size = tree.value_.sum(axis=1)
    assert tree.value_.shape == (n_nodes, 2)
    assert tree.value_[0].tolist() == np.bincount(y).tolist()
    assert np.all(size >= 1)
    assert n_nodes == 2 * is_leaf.sum() - 1
    parents = np.flatnonzero(~is_leaf)
    assert np.array_equal(tree.left_[parents], parents + 1)
    assert np.array_equal(tree.value_[parents],
                          tree.value_[tree.left_[parents]] + tree.value_[tree.right_[parents]])
    assert np.all(np.isnan(tree.threshold_) == is_leaf)
    reached = np.bincount(tree.apply(X), minlength=n_nodes)
    assert np.array_equal(reached[is_leaf], size[is_leaf])
    assert np.all(reached[~is_leaf] == 0)


def test_duplicate_points():
    X = np.array([[0.0, 1.0]] * 5 + [[1.0, 0.0]] * 3 + [[1.0, 1.0]] * 2)
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 2])
    for seed in range(4):
        tree = ExtraTree(max_features=1, seed=seed).fit(X, y)
        assert (tree.feature_ < 0).sum() == 3
        proba = tree.predict_proba(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
        assert np.allclose(proba, [[0.4, 0.6, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
        assert tree.predict(np.array([[1.0, 1.0]])).tolist() == [1]


def test_small_features():
    x = np.linspace(0.0, 1.0, 40)[:, None]
    bands = np.floor(x[:, 0] * 4).astype(int) % 2
    tree = ExtraTree(max_features=1, seed=0).fit(x, bands)
    assert np.array_equal(tree.predict(x), bands)

    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
    parity = corners.sum(axis=1).astype(int) % 2
    for seed in range(10):
        tree = ExtraTree(max_features=3, seed=seed).fit(corners, parity)
        assert np.array_equal(tree.predict(corners), parity)
        assert len(tree.feature_) == 15


def make_grid_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, 6)).astype(float)
    y = 3 + 4 * ((X[:, 0] + X[:, 1] >= 3).astype(int) + (X[:, 2] >= 2).astype(int))
    return X, y


def test_ensemble():
    X, y = make_grid_data(250, seed=0)
    X_test, y_test = make_grid_data(400, seed=1)
    forest = ExtraTreesClassifier(n_estimators=25, seed=0).fit(X, y)
    assert forest.classes_.tolist() == [3, 7, 11]
    assert len(forest.trees_) == 25
    proba = forest.predict_proba(X_test)
    assert proba.shape == (400, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    votes = np.stack([tree.predict(X_test) for tree in forest.trees_])
    assert np.allclose(proba[:, 2], (votes == 11).mean(axis=0))
    assert np.array_equal(forest.predict(X), y)
    assert np.mean(forest.predict(X_test) == y_test) > 0.85


def test_reproducible():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] > 0).astype(int)
    a = ExtraTreesClassifier(n_estimators=4, seed=7).fit(X, y)
    b = ExtraTreesClassifier(n_estimators=4, seed=7).fit(X, y)
    for s, t in zip(a.trees_, b.trees_):
        assert np.array_equal(s.feature_, t.feature_)
        assert np.array_equal(s.threshold_, t.threshold_, equal_nan=True)
    assert len({float(t.threshold_[0]) for t in a.trees_}) == 4
    assert a.trees_[0].max_features == 2


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

Without any changes, `python test_trees.py` prints:

```text
PASS test_split_scores
FAIL test_apply_manual_tree - AssertionError
FAIL test_noisy_labels - AssertionError
FAIL test_tree_structure - AssertionError
FAIL test_duplicate_points - RecursionError
FAIL test_small_features - AssertionError
FAIL test_ensemble - AssertionError
PASS test_reproducible
2 passed, 6 failed
```

Task: fix `trees.py` so that all 8 tests pass. The fixed code must meet the docstring spec, not just turn the tests green. Explain which places you changed and what was wrong in each.

### Part 2 — A scoring function for regression

The original problem has only Part 1. This part is an extension we added, to check whether you really understood this code.

In the paper, regression trees score candidate splits by relative variance reduction. Following the interface of `split_scores`, write a vectorized `variance_scores`:

```py
def variance_scores(go_left, y):
    """Score k candidate splits of the same n samples for regression.

    go_left: bool array (n, k), as in split_scores.  n >= 1.
    y:       float array (n,), the targets.

    Returns an array of shape (k,) with

        score = (var(y) - n_l / n * var(y_l) - n_r / n * var(y_r)) / var(y)

    where var is the population variance (divide by the count), y_l and y_r
    are the targets that go left and right, and n_l, n_r are their counts.
    A child with no samples contributes 0.  If all targets are equal
    (var(y) == 0), every score is 0.
    """
```

No Python loops (neither over samples nor over candidate splits). The mean of `y` may be many orders of magnitude larger than its spread (for example `1e8` plus noise with variance about 1), and even then the scores must be accurate to within `1e-5`. Example:

```text
y       = [1, 2, 3, 10]
the 3 columns of go_left: [T, T, F, F], [T, T, T, F], [T, T, T, T]
result  = [0.5, 0.96, 0.0]
```

One more question: to turn `ExtraTree` into a regression tree, what else needs to change besides the scoring function?

## Reference solution

<details>
<summary>Show the reference solution</summary>

### Part 1

**Approach.** First run all the tests and look at the kinds of failure. `RecursionError` and `ValueError` are crashes, and they are best fixed first: a crash hides the assertions behind it, and once it is fixed the other failures become visible. Then use the docstrings as a checklist and compare the code against them line by line: the three stopping rules (a), (b), (c), the 4 steps of a split, the routing rule in `apply`, and how the ensemble votes. After each fix, rerun all the tests, because one bug can mask another (see follow-up 1).

There are 6 bugs. The last column of the table is the tests that fail when only that bug is left and the other 5 are fixed (each one is run in the verification code):

| # | Location | What is wrong | Tests failing when only this bug is left |
|---|---|---|---|
| 1 | First line of `_pick_split` | `<=`: a node whose sample count equals `min_samples_split` is no longer split | `test_noisy_labels`, `test_small_features` |
| 2 | `lo, hi` in `_pick_split` | Uses the min and max of the whole `X`, not of the current node's `X[idx]` | `test_tree_structure`, `test_duplicate_points` (RecursionError), `test_small_features` |
| 3 | Constant check in `_pick_split` | `np.any`: stops as soon as any one feature is constant | `test_duplicate_points`, `test_small_features`, `test_ensemble` |
| 4 | Feature sampling in `_pick_split` | Always draws `max_features` features instead of `min(max_features, len(candidates))` | `test_small_features`, `test_ensemble` (both ValueError) |
| 5 | `apply` | `<=`: a point exactly equal to the threshold goes left | `test_apply_manual_tree` |
| 6 | `ExtraTreesClassifier.predict_proba` | `counts[rows, votes] += 1` adds only once for repeated indices | `test_ensemble` |

**Bug 1: an off-by-one in the stopping condition.** Rule (a) makes a node a leaf only when $|S| <$ `min_samples_split`, so a node whose sample count equals it must still be split. The code says `<=`. With the default `min_samples_split = 2`, even a node with "two samples, two labels" is not split, so the tree cannot memorize the training set: in `test_small_features` both the 40-point stripe data and the 3-dimensional parity data come out wrong. In `test_noisy_labels`, `min_samples_split=200` with exactly 200 samples, so the root should be split but becomes a leaf; with `n_min` of 2 and 5, the tree also contains impure leaves whose sample count equals exactly `n_min`, which violates the spec.

```py
if len(idx) < self.min_samples_split or counts.max() == counts.sum():
```

**Bug 2: the cut-point range uses the global data.** Step 2 of the spec says the cut point is drawn from $[\min_S f, \max_S f)$, where $S$ is the current node's samples. The code instead takes the min and max over the whole `X`. The cut point can then fall outside the current node's range, so all samples go to the same child and the other child is empty, which violates "both children get at least one sample" (`test_tree_structure` checks that every node has at least one sample). Worse, if a node's samples are identical but their labels differ (in `test_duplicate_points` there are two rows `[1, 1]` with labels 1 and 2), every split hands the same samples unchanged to a child, recursing forever into a `RecursionError` (on the original platform this may well just hang). The fix:

```py
lo, hi = X[idx].min(axis=0), X[idx].max(axis=0)
```

**Bug 3: the constant check uses `any`.** Rule (c) stops only when "every feature is constant in $S$", but the code says `np.any(lo == hi)`, which stops as soon as any one feature is constant. With integer features, after a few levels some feature is almost always constant within the node, so the tree stops early and cannot even fit the training set. The grid data in `test_ensemble` (6 integer features taking values 0 to 3) is mispredicted on the training set itself; for the 3-dimensional parity data, after the first split the dimension just split on is constant, and the tree stops at depth 1. Change it to `np.all`.

**Bug 4: no minimum when there are too few candidate features.** Step 1 draws $k = \min(K, |A|)$ features without replacement from the set $A$ of non-constant features. The code always draws `max_features`, so when there are fewer than $K$ non-constant features, `rng.choice(..., replace=False)` raises `ValueError`. The parity data uses $K = 3$, and after one split only 2 non-constant features remain; the grid data uses the default $K = \mathrm{round}(\sqrt 6) = 2$, and deep nodes often have only 1 left. The fix:

```py
k = min(self.max_features, len(candidates))
feats = self._rng.choice(candidates, size=k, replace=False)
```

**Bug 5: the comparison goes the wrong way at prediction time.** Both the `apply` docstring and step 4 of training send `x[f] < cut` left, but the code says `<=`. Training data almost never lands exactly on a cut point (cut points are drawn uniformly from a continuous range), so none of the other tests can see it; only in the hand-built tree of `test_apply_manual_tree` do `[2.0, 4.0]` and `[3.0, 5.0]` sit exactly on the thresholds. Change it back to `<`.

**Bug 6: `+=` with fancy indexing does not accumulate.** `votes` has shape `(n_trees, n)`, and `(np.arange(n), votes)` broadcasts to `(n_trees, n)` (row, class) index pairs, many of them repeated. `a[idx] += 1` is equivalent to `a[idx] = a[idx] + 1`: it reads the values first and then writes them all back, so a repeated index counts only once. As a result each row records "how many classes got any vote" instead of the vote counts, the rows do not sum to 1, and the class picked by argmax is wrong. Use the unbuffered `np.add.at` instead:

```py
np.add.at(counts, (np.arange(len(X)), votes), 1)
```

**Comparison with the paper.** The fixed code matches the algorithm in the Geurts et al. paper: stop when the node has fewer than $n_{min}$ samples, all features are constant, or all labels are the same; otherwise draw $K$ of the features that are non-constant in the current node, without replacement, draw one cut point uniformly on each one's $[\min, \max]$ within the node, and take the highest-scoring split; the classification score is the normalized information gain $2 I / (H_s + H_c)$; every tree uses all training samples (no bootstrap), and the ensemble takes a majority vote. The only difference is that we use a half-open interval plus "replace the cut point with the max if it comes out equal to the min" to avoid empty children (see follow-up 4).

The fixed `trees.py`:

```python
"""Extremely Randomized Trees (Geurts, Ernst & Wehenkel, 2006) for classification.

The docstrings in this file are the specification.

Growing a tree.  Every tree is grown on the whole training set (no bootstrap).
A node that holds the sample subset S becomes a leaf if any of these is true:

  (a) |S| < min_samples_split,
  (b) all labels in S are equal,
  (c) every feature is constant in S.

Otherwise the node is split:

  1. Let A be the set of features that are not constant in S.  Draw
     k = min(max_features, |A|) distinct features from A uniformly at random.
  2. For each drawn feature f, draw one cut-point uniformly from
     [min_S f, max_S f).  A draw equal to min_S f is replaced by max_S f.
  3. Keep the candidate (f, cut) with the highest score (see split_scores);
     on ties, the one drawn first.
  4. Samples with x[f] < cut go to the left child, all other samples go to the
     right child.  By construction both children receive at least one sample.

Storage.  Nodes are numbered in pre-order (the root is 0, and the left subtree
of a node is numbered before its right subtree) and kept in flat arrays of
length n_nodes:

  feature_[i]    split feature of node i, -1 for a leaf
  threshold_[i]  cut-point of node i, nan for a leaf
  left_[i]       id of the left child, -1 for a leaf
  right_[i]      id of the right child, -1 for a leaf
  value_[i]      shape (n_classes,): how many training samples of each class
                 reached node i
"""
import numpy as np


def entropy(counts):
    """Shannon entropy, in bits, of each row of `counts`.

    counts: array of shape (..., m) holding non-negative counts.
    Returns an array of shape (...).  0 * log(0) counts as 0, and a row that
    is all zeros has entropy 0.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=-1)


def split_scores(go_left, onehot):
    """Score k candidate splits of the same n samples.

    go_left: bool array (n, k); go_left[i, j] is True if sample i goes to the
             left child under candidate j.
    onehot:  array (n, n_classes), the one-hot labels of the samples.

    Returns an array of shape (k,) with

        score = 2 * I / (H_split + H_class)

    where H_class is the entropy of the labels, H_split is the entropy of the
    left/right partition and I is their mutual information, i.e. H_class minus
    the size-weighted mean of the label entropies of the two children.
    The score lies in [0, 1].  It is 0 when the denominator is 0.
    """
    n = len(onehot)
    total = onehot.sum(axis=0)
    left = go_left.T.astype(float) @ onehot
    right = total - left
    n_left = left.sum(axis=1)
    n_right = n - n_left
    h_class = entropy(total)
    h_split = entropy(np.stack([n_left, n_right], axis=1))
    gain = h_class - (n_left * entropy(left) + n_right * entropy(right)) / n
    denom = h_split + h_class
    return np.divide(2.0 * gain, denom, out=np.zeros_like(gain), where=denom > 0)


class ExtraTree:
    """One extremely randomized classification tree.

    max_features:      number of candidate features drawn at each node (K >= 1).
    min_samples_split: a node with fewer samples is never split (n_min >= 2).
    seed:              anything np.random.default_rng accepts.
    """

    def __init__(self, max_features, min_samples_split=2, seed=None):
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        """Grow the tree on X (n, d) and labels y (n,), n >= 1.  Returns self.

        Sets classes_ (the sorted distinct labels) and the node arrays that
        the module docstring describes.
        """
        X = np.asarray(X, dtype=float)
        self.classes_, codes = np.unique(y, return_inverse=True)
        onehot = np.eye(len(self.classes_))[codes]
        self._rng = np.random.default_rng(self.seed)
        self._nodes = []
        self._grow(X, onehot, np.arange(len(X)))
        feature, threshold, left, right, value = zip(*self._nodes)
        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.left_ = np.array(left)
        self.right_ = np.array(right)
        self.value_ = np.array(value)
        return self

    def _grow(self, X, onehot, idx):
        """Append the subtree for the samples `idx`; return the id of its root."""
        node = len(self._nodes)
        counts = onehot[idx].sum(axis=0)
        self._nodes.append([-1, np.nan, -1, -1, counts])
        split = self._pick_split(X, onehot, idx, counts)
        if split is not None:
            feature, cut = split
            go_left = X[idx, feature] < cut
            left = self._grow(X, onehot, idx[go_left])
            right = self._grow(X, onehot, idx[~go_left])
            self._nodes[node][:4] = [feature, cut, left, right]
        return node

    def _pick_split(self, X, onehot, idx, counts):
        """Return (feature, cut) for the node holding `idx`, or None for a leaf."""
        if len(idx) < self.min_samples_split or counts.max() == counts.sum():
            return None
        lo, hi = X[idx].min(axis=0), X[idx].max(axis=0)
        if np.all(lo == hi):
            return None
        candidates = np.flatnonzero(lo < hi)
        k = min(self.max_features, len(candidates))
        feats = self._rng.choice(candidates, size=k, replace=False)
        cuts = self._rng.uniform(lo[feats], hi[feats])
        cuts = np.where(cuts > lo[feats], cuts, hi[feats])
        go_left = X[np.ix_(idx, feats)] < cuts
        best = np.argmax(split_scores(go_left, onehot[idx]))
        return int(feats[best]), float(cuts[best])

    def apply(self, X):
        """Id of the leaf that each row of X ends up in, shape (n,).

        Every sample starts at the root.  At an internal node i it moves to
        left_[i] if x[feature_[i]] < threshold_[i], otherwise to right_[i].
        """
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        while True:
            rows = np.flatnonzero(self.feature_[node] >= 0)
            if rows.size == 0:
                return node
            at = node[rows]
            go_left = X[rows, self.feature_[at]] < self.threshold_[at]
            node[rows] = np.where(go_left, self.left_[at], self.right_[at])

    def predict_proba(self, X):
        """Class frequencies in the leaf of each sample, shape (n, n_classes)."""
        counts = self.value_[self.apply(X)]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Most frequent class in the leaf of each sample; ties go to the
        smallest class."""
        return self.classes_[self.value_[self.apply(X)].argmax(axis=1)]


class ExtraTreesClassifier:
    """An ensemble of ExtraTree, all grown on the same training set.

    n_estimators:      number of trees (M >= 1).
    max_features:      K of every tree; None means round(sqrt(n_features)).
    min_samples_split: n_min of every tree.
    seed:              the trees get independent random streams derived from it.
    """

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        """Grow n_estimators trees on (X, y).  Returns self."""
        X = np.asarray(X, dtype=float)
        k = self.max_features
        if k is None:
            k = max(1, round(np.sqrt(X.shape[1])))
        streams = np.random.SeedSequence(self.seed).spawn(self.n_estimators)
        self.trees_ = [ExtraTree(k, self.min_samples_split, s).fit(X, y) for s in streams]
        self.classes_ = self.trees_[0].classes_
        return self

    def predict_proba(self, X):
        """Fraction of the trees that vote for each class, shape (n, n_classes).

        Every tree casts exactly one vote per sample, for the class that
        ExtraTree.predict returns.  Each row therefore sums to 1.
        """
        X = np.asarray(X, dtype=float)
        votes = np.stack([t.value_[t.apply(X)].argmax(axis=1) for t in self.trees_])
        counts = np.zeros((len(X), len(self.classes_)))
        np.add.at(counts, (np.arange(len(X)), votes), 1)
        return counts / len(self.trees_)

    def predict(self, X):
        """Majority vote of the trees; ties go to the smallest class."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
```

### Part 2

**Approach.** First center $y$: $z = y - \bar y$, with total sum of squares $T = \sum z_i^2 = n\,\mathrm{var}(y)$. The left child's sum of squares is $\sum_{L} z_i^2 - s_l^2 / n_l$, where $s_l = \sum_L z_i$; likewise on the right, with $s_r = -s_l$ (since $\sum z_i = 0$). Substituting into the formula,

$$\text{score} = \frac{T - \mathrm{SS}_l - \mathrm{SS}_r}{T} = \frac{s_l^2 / n_l + s_l^2 / n_r}{T}.$$

$s_l$ is computed for all candidates at once: `z @ go_left`, of shape `(k,)`; `n_l = go_left.sum(axis=0)`. For an empty side $s_l = 0$, and `np.divide(..., where=n > 0)` makes it contribute 0. Centering first avoids the cancellation error of "mean of the squares minus square of the mean" when $y$ is large. The complexity is $O(nk)$, the same as `split_scores`.

```python
def variance_scores(go_left, y):
    y = np.asarray(y, dtype=float)
    go_left = np.asarray(go_left, dtype=bool)
    k = go_left.shape[1]
    if np.all(y == y[0]):
        return np.zeros(k)
    z = y - y.mean()
    s_left = z @ go_left
    n_left = go_left.sum(axis=0)
    n_right = len(y) - n_left
    between = (np.divide(s_left ** 2, n_left, out=np.zeros(k), where=n_left > 0)
               + np.divide(s_left ** 2, n_right, out=np.zeros(k), where=n_right > 0))
    return between / (z @ z)
```

The example: $\bar y = 4$, $T = 9 + 4 + 1 + 36 = 50$. Column 1 has $s_l = -3 - 2 = -5$, score $(25/2 + 25/2)/50 = 0.5$; column 2 has $s_l = -6$, score $(36/3 + 36/1)/50 = 0.96$; column 3 has an empty right side, $s_l = 0$, score 0.

**What else changes for a regression tree.** Leaves store the mean of the targets (or their sum and count) instead of class counts; stopping rule (b) becomes "all target values are equal"; a single tree's prediction returns the leaf mean, and the ensemble averages the trees' predictions instead of voting. The paper's default parameters also differ: regression uses $K = d$ (all features are candidates) and $n_{min} = 5$; classification uses $K = \sqrt d$ and $n_{min} = 2$.

### Follow-up questions

- **Does the order in which you fix the bugs matter?** Yes, the bugs mask one another. With bug 2 using the global range, a feature that is constant within a node is never constant globally, so neither bug 3 (stopping on `any`) nor bug 4 (`ValueError`) is triggered. If you fix bug 6 first, `test_ensemble` passes right away; then fix bug 2 and it fails again, because bugs 3 and 4 are exposed. And bug 3 makes the tree stop as soon as a constant feature appears, so it masks bug 4 as well. That is why you rerun all the tests after every fix, not just the one that was failing.
- **Besides `np.add.at`, how else can you count votes?** You can encode (row, class) as one integer and use `bincount`: `np.bincount((np.arange(n) * n_classes + votes).ravel(), minlength=n * n_classes).reshape(n, n_classes)`; or compare into a boolean tensor and sum: `(votes[:, :, None] == np.arange(n_classes)).sum(axis=0)`, which needs $O(MnC)$ memory. All three give the same result (checked in the verification code).
- **Should the ensemble use a majority vote, or average the trees' probabilities?** The paper uses a majority vote; scikit-learn averages `predict_proba` (soft voting). With `min_samples_split = 2`, a leaf keeps splitting until it is pure as long as some feature can still be split, so leaves are almost all pure and the two methods give the same predictions (checked on the grid data in the verification code). They differ only when leaves are impure (identical features, different labels) or $n_{min}$ is large, and then soft voting keeps more information.
- **Why replace the cut point with the max when it comes out equal to the min?** `rng.uniform(lo, hi)` returns a value in $[lo, hi)$, but floating-point rounding can return exactly `lo`, in which case no sample satisfies `x < lo` and the left child is empty. On ordinary data this is vanishingly unlikely (on the order of $2^{-53}$); but if there are only a few representable floats between `lo` and `hi`, for example `lo = 1e16, hi = 1e16 + 2`, about half the draws return `lo` (measured in the verification code). With `hi` instead, `x < hi` sends every sample except those at the max to the left, so neither side is empty.
- **What is the complexity? What is the risk of recursion?** One node costs $O(|S| \cdot (d + K C))$: the min and max of every feature are $O(|S| d)$, and scoring $K$ candidates is $O(|S| K C)$. The nodes at one level hold at most $n$ samples in total, so one tree costs $O(n \cdot \text{depth} \cdot (d + KC))$. With random cut points the depth on typical data is $O(\log n)$ (for $n$ equally spaced points the average leaf depth is exactly $2(H_n - 1) \approx 2 \ln n$, checked by simulation in the verification code), and $O(n)$ in the worst case. At prediction time each round of `apply` moves every sample still at an internal node down one level; the number of rounds equals the maximum depth, and each round is $O(n)$. `_grow` is recursive and fails once the depth exceeds Python's default limit of 1000; to be safe, rewrite it with an explicit stack.

<details>
<summary>Verification code (runnable)</summary>

```python
import ast
import contextlib
import io
import math
import sys
import time
import types
import warnings

import numpy as np

T0 = time.perf_counter()

# The statement's two files, byte for byte.
BUGGY_SOURCE = r'''"""Extremely Randomized Trees (Geurts, Ernst & Wehenkel, 2006) for classification.

The docstrings in this file are the specification.

Growing a tree.  Every tree is grown on the whole training set (no bootstrap).
A node that holds the sample subset S becomes a leaf if any of these is true:

  (a) |S| < min_samples_split,
  (b) all labels in S are equal,
  (c) every feature is constant in S.

Otherwise the node is split:

  1. Let A be the set of features that are not constant in S.  Draw
     k = min(max_features, |A|) distinct features from A uniformly at random.
  2. For each drawn feature f, draw one cut-point uniformly from
     [min_S f, max_S f).  A draw equal to min_S f is replaced by max_S f.
  3. Keep the candidate (f, cut) with the highest score (see split_scores);
     on ties, the one drawn first.
  4. Samples with x[f] < cut go to the left child, all other samples go to the
     right child.  By construction both children receive at least one sample.

Storage.  Nodes are numbered in pre-order (the root is 0, and the left subtree
of a node is numbered before its right subtree) and kept in flat arrays of
length n_nodes:

  feature_[i]    split feature of node i, -1 for a leaf
  threshold_[i]  cut-point of node i, nan for a leaf
  left_[i]       id of the left child, -1 for a leaf
  right_[i]      id of the right child, -1 for a leaf
  value_[i]      shape (n_classes,): how many training samples of each class
                 reached node i
"""
import numpy as np


def entropy(counts):
    """Shannon entropy, in bits, of each row of `counts`.

    counts: array of shape (..., m) holding non-negative counts.
    Returns an array of shape (...).  0 * log(0) counts as 0, and a row that
    is all zeros has entropy 0.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=-1)


def split_scores(go_left, onehot):
    """Score k candidate splits of the same n samples.

    go_left: bool array (n, k); go_left[i, j] is True if sample i goes to the
             left child under candidate j.
    onehot:  array (n, n_classes), the one-hot labels of the samples.

    Returns an array of shape (k,) with

        score = 2 * I / (H_split + H_class)

    where H_class is the entropy of the labels, H_split is the entropy of the
    left/right partition and I is their mutual information, i.e. H_class minus
    the size-weighted mean of the label entropies of the two children.
    The score lies in [0, 1].  It is 0 when the denominator is 0.
    """
    n = len(onehot)
    total = onehot.sum(axis=0)
    left = go_left.T.astype(float) @ onehot
    right = total - left
    n_left = left.sum(axis=1)
    n_right = n - n_left
    h_class = entropy(total)
    h_split = entropy(np.stack([n_left, n_right], axis=1))
    gain = h_class - (n_left * entropy(left) + n_right * entropy(right)) / n
    denom = h_split + h_class
    return np.divide(2.0 * gain, denom, out=np.zeros_like(gain), where=denom > 0)


class ExtraTree:
    """One extremely randomized classification tree.

    max_features:      number of candidate features drawn at each node (K >= 1).
    min_samples_split: a node with fewer samples is never split (n_min >= 2).
    seed:              anything np.random.default_rng accepts.
    """

    def __init__(self, max_features, min_samples_split=2, seed=None):
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        """Grow the tree on X (n, d) and labels y (n,), n >= 1.  Returns self.

        Sets classes_ (the sorted distinct labels) and the node arrays that
        the module docstring describes.
        """
        X = np.asarray(X, dtype=float)
        self.classes_, codes = np.unique(y, return_inverse=True)
        onehot = np.eye(len(self.classes_))[codes]
        self._rng = np.random.default_rng(self.seed)
        self._nodes = []
        self._grow(X, onehot, np.arange(len(X)))
        feature, threshold, left, right, value = zip(*self._nodes)
        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.left_ = np.array(left)
        self.right_ = np.array(right)
        self.value_ = np.array(value)
        return self

    def _grow(self, X, onehot, idx):
        """Append the subtree for the samples `idx`; return the id of its root."""
        node = len(self._nodes)
        counts = onehot[idx].sum(axis=0)
        self._nodes.append([-1, np.nan, -1, -1, counts])
        split = self._pick_split(X, onehot, idx, counts)
        if split is not None:
            feature, cut = split
            go_left = X[idx, feature] < cut
            left = self._grow(X, onehot, idx[go_left])
            right = self._grow(X, onehot, idx[~go_left])
            self._nodes[node][:4] = [feature, cut, left, right]
        return node

    def _pick_split(self, X, onehot, idx, counts):
        """Return (feature, cut) for the node holding `idx`, or None for a leaf."""
        if len(idx) <= self.min_samples_split or counts.max() == counts.sum():
            return None
        lo, hi = X.min(axis=0), X.max(axis=0)
        if np.any(lo == hi):
            return None
        candidates = np.flatnonzero(lo < hi)
        feats = self._rng.choice(candidates, size=self.max_features, replace=False)
        cuts = self._rng.uniform(lo[feats], hi[feats])
        cuts = np.where(cuts > lo[feats], cuts, hi[feats])
        go_left = X[np.ix_(idx, feats)] < cuts
        best = np.argmax(split_scores(go_left, onehot[idx]))
        return int(feats[best]), float(cuts[best])

    def apply(self, X):
        """Id of the leaf that each row of X ends up in, shape (n,).

        Every sample starts at the root.  At an internal node i it moves to
        left_[i] if x[feature_[i]] < threshold_[i], otherwise to right_[i].
        """
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        while True:
            rows = np.flatnonzero(self.feature_[node] >= 0)
            if rows.size == 0:
                return node
            at = node[rows]
            go_left = X[rows, self.feature_[at]] <= self.threshold_[at]
            node[rows] = np.where(go_left, self.left_[at], self.right_[at])

    def predict_proba(self, X):
        """Class frequencies in the leaf of each sample, shape (n, n_classes)."""
        counts = self.value_[self.apply(X)]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Most frequent class in the leaf of each sample; ties go to the
        smallest class."""
        return self.classes_[self.value_[self.apply(X)].argmax(axis=1)]


class ExtraTreesClassifier:
    """An ensemble of ExtraTree, all grown on the same training set.

    n_estimators:      number of trees (M >= 1).
    max_features:      K of every tree; None means round(sqrt(n_features)).
    min_samples_split: n_min of every tree.
    seed:              the trees get independent random streams derived from it.
    """

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        """Grow n_estimators trees on (X, y).  Returns self."""
        X = np.asarray(X, dtype=float)
        k = self.max_features
        if k is None:
            k = max(1, round(np.sqrt(X.shape[1])))
        streams = np.random.SeedSequence(self.seed).spawn(self.n_estimators)
        self.trees_ = [ExtraTree(k, self.min_samples_split, s).fit(X, y) for s in streams]
        self.classes_ = self.trees_[0].classes_
        return self

    def predict_proba(self, X):
        """Fraction of the trees that vote for each class, shape (n, n_classes).

        Every tree casts exactly one vote per sample, for the class that
        ExtraTree.predict returns.  Each row therefore sums to 1.
        """
        X = np.asarray(X, dtype=float)
        votes = np.stack([t.value_[t.apply(X)].argmax(axis=1) for t in self.trees_])
        counts = np.zeros((len(X), len(self.classes_)))
        counts[np.arange(len(X)), votes] += 1
        return counts / len(self.trees_)

    def predict(self, X):
        """Majority vote of the trees; ties go to the smallest class."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
'''

TEST_SOURCE = r'''"""Tests for trees.py.

Run all of them:   python test_trees.py
Run one of them:   python test_trees.py test_ensemble     (prints the traceback)
"""
import numpy as np

from trees import ExtraTree, ExtraTreesClassifier, entropy, split_scores


def test_split_scores():
    h = entropy([[2, 2], [4, 0], [0, 0], [1, 3]])
    assert np.allclose(h, [1.0, 0.0, 0.0, 0.8112781244591328])
    onehot = np.eye(2)[[0, 0, 0, 1, 1, 1]]
    go_left = np.array([[1, 1, 1, 1, 0],
                        [1, 1, 0, 1, 0],
                        [1, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0]], dtype=bool)
    scores = split_scores(go_left, onehot)
    assert scores.shape == (5,)
    assert np.allclose(scores, [1.0, 0.4787039714, 0.0817041659, 0.0, 0.0])


def test_apply_manual_tree():
    tree = ExtraTree(max_features=1)
    tree.classes_ = np.array(["a", "b", "c"])
    tree.feature_ = np.array([0, -1, 1, -1, -1])
    tree.threshold_ = np.array([2.0, np.nan, 5.0, np.nan, np.nan])
    tree.left_ = np.array([1, -1, 3, -1, -1])
    tree.right_ = np.array([2, -1, 4, -1, -1])
    tree.value_ = np.array([[3.0, 3.0, 4.0],
                            [3.0, 0.0, 0.0],
                            [0.0, 3.0, 4.0],
                            [0.0, 3.0, 1.0],
                            [0.0, 0.0, 3.0]])
    X = np.array([[1.0, 9.0], [2.0, 4.0], [3.0, 5.0], [2.5, 4.9], [-1.0, 0.0], [7.0, 6.0]])
    assert tree.apply(X).tolist() == [1, 3, 4, 3, 1, 4]
    assert tree.predict(X).tolist() == ["a", "b", "c", "b", "a", "c"]
    assert np.allclose(tree.predict_proba(X)[:2], [[1.0, 0.0, 0.0], [0.0, 0.75, 0.25]])


def test_noisy_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = rng.integers(0, 3, size=200)
    for n_min in (2, 5, 20):
        tree = ExtraTree(max_features=2, min_samples_split=n_min, seed=n_min).fit(X, y)
        is_leaf = tree.feature_ < 0
        size = tree.value_.sum(axis=1)
        is_pure = (tree.value_ > 0).sum(axis=1) == 1
        assert np.all(size[~is_leaf] >= n_min)
        assert np.all(is_pure[is_leaf] | (size[is_leaf] < n_min))
    assert len(ExtraTree(max_features=2, min_samples_split=201, seed=0).fit(X, y).feature_) == 1
    assert len(ExtraTree(max_features=2, min_samples_split=200, seed=0).fit(X, y).feature_) >= 3
    full = ExtraTree(max_features=2, seed=1).fit(X, y)
    assert np.array_equal(full.predict(X), y)


def test_tree_structure():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    tree = ExtraTree(max_features=3, seed=0).fit(X, y)
    n_nodes = len(tree.feature_)
    is_leaf = tree.feature_ < 0
    size = tree.value_.sum(axis=1)
    assert tree.value_.shape == (n_nodes, 2)
    assert tree.value_[0].tolist() == np.bincount(y).tolist()
    assert np.all(size >= 1)
    assert n_nodes == 2 * is_leaf.sum() - 1
    parents = np.flatnonzero(~is_leaf)
    assert np.array_equal(tree.left_[parents], parents + 1)
    assert np.array_equal(tree.value_[parents],
                          tree.value_[tree.left_[parents]] + tree.value_[tree.right_[parents]])
    assert np.all(np.isnan(tree.threshold_) == is_leaf)
    reached = np.bincount(tree.apply(X), minlength=n_nodes)
    assert np.array_equal(reached[is_leaf], size[is_leaf])
    assert np.all(reached[~is_leaf] == 0)


def test_duplicate_points():
    X = np.array([[0.0, 1.0]] * 5 + [[1.0, 0.0]] * 3 + [[1.0, 1.0]] * 2)
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 2])
    for seed in range(4):
        tree = ExtraTree(max_features=1, seed=seed).fit(X, y)
        assert (tree.feature_ < 0).sum() == 3
        proba = tree.predict_proba(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
        assert np.allclose(proba, [[0.4, 0.6, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
        assert tree.predict(np.array([[1.0, 1.0]])).tolist() == [1]


def test_small_features():
    x = np.linspace(0.0, 1.0, 40)[:, None]
    bands = np.floor(x[:, 0] * 4).astype(int) % 2
    tree = ExtraTree(max_features=1, seed=0).fit(x, bands)
    assert np.array_equal(tree.predict(x), bands)

    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
    parity = corners.sum(axis=1).astype(int) % 2
    for seed in range(10):
        tree = ExtraTree(max_features=3, seed=seed).fit(corners, parity)
        assert np.array_equal(tree.predict(corners), parity)
        assert len(tree.feature_) == 15


def make_grid_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, 6)).astype(float)
    y = 3 + 4 * ((X[:, 0] + X[:, 1] >= 3).astype(int) + (X[:, 2] >= 2).astype(int))
    return X, y


def test_ensemble():
    X, y = make_grid_data(250, seed=0)
    X_test, y_test = make_grid_data(400, seed=1)
    forest = ExtraTreesClassifier(n_estimators=25, seed=0).fit(X, y)
    assert forest.classes_.tolist() == [3, 7, 11]
    assert len(forest.trees_) == 25
    proba = forest.predict_proba(X_test)
    assert proba.shape == (400, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    votes = np.stack([tree.predict(X_test) for tree in forest.trees_])
    assert np.allclose(proba[:, 2], (votes == 11).mean(axis=0))
    assert np.array_equal(forest.predict(X), y)
    assert np.mean(forest.predict(X_test) == y_test) > 0.85


def test_reproducible():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] > 0).astype(int)
    a = ExtraTreesClassifier(n_estimators=4, seed=7).fit(X, y)
    b = ExtraTreesClassifier(n_estimators=4, seed=7).fit(X, y)
    for s, t in zip(a.trees_, b.trees_):
        assert np.array_equal(s.feature_, t.feature_)
        assert np.array_equal(s.threshold_, t.threshold_, equal_nan=True)
    assert len({float(t.threshold_[0]) for t in a.trees_}) == 4
    assert a.trees_[0].max_features == 2


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

EXPECTED_OUTPUT = """PASS test_split_scores
FAIL test_apply_manual_tree - AssertionError
FAIL test_noisy_labels - AssertionError
FAIL test_tree_structure - AssertionError
FAIL test_duplicate_points - RecursionError
FAIL test_small_features - AssertionError
FAIL test_ensemble - AssertionError
PASS test_reproducible
2 passed, 6 failed
"""

# (bug, buggy text, fixed text): every fix is a one-place replacement.
FIXES = [
    ("bug 1", "if len(idx) <= self.min_samples_split or",
              "if len(idx) < self.min_samples_split or"),
    ("bug 2", "lo, hi = X.min(axis=0), X.max(axis=0)",
              "lo, hi = X[idx].min(axis=0), X[idx].max(axis=0)"),
    ("bug 3", "if np.any(lo == hi):", "if np.all(lo == hi):"),
    ("bug 4", "        feats = self._rng.choice(candidates, size=self.max_features, replace=False)\n",
              "        k = min(self.max_features, len(candidates))\n"
              "        feats = self._rng.choice(candidates, size=k, replace=False)\n"),
    ("bug 5", "go_left = X[rows, self.feature_[at]] <= self.threshold_[at]",
              "go_left = X[rows, self.feature_[at]] < self.threshold_[at]"),
    ("bug 6", "counts[np.arange(len(X)), votes] += 1",
              "np.add.at(counts, (np.arange(len(X)), votes), 1)"),
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
    mod = types.ModuleType("trees")
    exec(compile(src, "trees.py", "exec"), mod.__dict__)
    return mod


@contextlib.contextmanager
def quiet():
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield


def exec_tests(mod, name):
    saved = sys.modules.get("trees")
    sys.modules["trees"] = mod
    try:
        ns = {"__name__": name}
        exec(compile(TEST_SOURCE, "test_trees.py", "exec"), ns)
    finally:
        if saved is None:
            del sys.modules["trees"]
        else:
            sys.modules["trees"] = saved
    return ns


def run_tests(mod):
    """{test name: 'ok' or the exception type} for every test in test_trees.py."""
    ns = exec_tests(mod, "test_trees")
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
    """What `python test_trees.py` prints when trees.py is `mod`."""
    out, saved_argv = io.StringIO(), sys.argv
    sys.argv = ["test_trees.py"]
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

# 2. With all six fixes, all 8 tests pass.
fixed_mod = load(build_source(ALL))
result = run_tests(fixed_mod)
assert len(result) == 8 and set(result.values()) == {"ok"}, result

# 3. Each bug matters on its own: fix the other five, and exactly these tests fail.
ONLY_THIS_BUG_LEFT = {
    "bug 1": {"test_noisy_labels": "AssertionError", "test_small_features": "AssertionError"},
    "bug 2": {"test_tree_structure": "AssertionError", "test_duplicate_points": "RecursionError",
              "test_small_features": "AssertionError"},
    "bug 3": {"test_duplicate_points": "AssertionError", "test_small_features": "AssertionError",
              "test_ensemble": "AssertionError"},
    "bug 4": {"test_small_features": "ValueError", "test_ensemble": "ValueError"},
    "bug 5": {"test_apply_manual_tree": "AssertionError"},
    "bug 6": {"test_ensemble": "AssertionError"},
}
for bug, expected in ONLY_THIS_BUG_LEFT.items():
    got = failures(load(build_source([b for b in ALL if b != bug])))
    assert got == expected, (bug, got)

# 4. Masking (follow-up 1): with only bug 6 fixed, test_ensemble passes; fixing bug 2
#    as well makes it fail again; with only bugs 3 and 4 left, bug 4's ValueError is hidden.
assert "test_ensemble" not in failures(load(build_source(["bug 6"])))
assert failures(load(build_source(["bug 2", "bug 6"])))["test_ensemble"] == "AssertionError"
got = failures(load(build_source(["bug 1", "bug 2", "bug 5", "bug 6"])))
assert got["test_ensemble"] == got["test_small_features"] == "AssertionError"


# 4b. The symptoms quoted in the bug paragraphs, each with only that bug left.
only = {bug: load(build_source([b for b in ALL if b != bug])) for bug in ("bug 1", "bug 2", "bug 3")}
make_grid_data = exec_tests(fixed_mod, "test_trees")["make_grid_data"]
rng = np.random.default_rng(0)  # the data of test_noisy_labels
X, y = rng.normal(size=(200, 4)), rng.integers(0, 3, size=200)
assert len(only["bug 1"].ExtraTree(2, min_samples_split=200, seed=0).fit(X, y).feature_) == 1
for n_min in (2, 5):
    t = only["bug 1"].ExtraTree(2, min_samples_split=n_min, seed=n_min).fit(X, y)
    size, pure = t.value_.sum(axis=1), (t.value_ > 0).sum(axis=1) == 1
    assert np.any((t.feature_ < 0) & ~pure & (size == n_min))
rng = np.random.default_rng(1)  # the data of test_tree_structure
X = rng.normal(size=(150, 3))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
assert np.any(only["bug 2"].ExtraTree(3, seed=0).fit(X, y).value_.sum(axis=1) == 0)
corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
parity = corners.sum(axis=1).astype(int) % 2
assert all(len(only["bug 3"].ExtraTree(3, seed=s).fit(corners, parity).feature_) == 3
           for s in range(10))
X, y = make_grid_data(250, seed=0)
assert np.mean(only["bug 3"].ExtraTreesClassifier(25, seed=0).fit(X, y).predict(X) == y) < 1

# 5. The docstrings (the specification) are untouched, and the buggy file has no comments.
def docstrings(src):
    tree = ast.parse(src)
    return [ast.get_docstring(node) for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))]


assert docstrings(BUGGY_SOURCE) == docstrings(build_source(ALL))
assert "#" not in BUGGY_SOURCE

# 6. The fixed code printed above (already run in this namespace) passes all tests and
#    grows exactly the same trees as the replacement-built module.
page = types.ModuleType("trees")
page.__dict__.update(entropy=entropy, split_scores=split_scores,
                     ExtraTree=ExtraTree, ExtraTreesClassifier=ExtraTreesClassifier)
assert set(run_tests(page).values()) == {"ok"}
rng = np.random.default_rng(10)
for trial in range(4):
    X = rng.integers(0, 3, size=(80, 4)).astype(float)
    y = rng.integers(0, 3, size=80)
    a = ExtraTreesClassifier(n_estimators=5, seed=trial).fit(X, y)
    b = fixed_mod.ExtraTreesClassifier(n_estimators=5, seed=trial).fit(X, y)
    for s, t in zip(a.trees_, b.trees_):
        for attr in ("feature_", "threshold_", "left_", "right_", "value_"):
            assert np.array_equal(getattr(s, attr), getattr(t, attr), equal_nan=True)


# 7. split_scores against a plain-Python mutual-information computation.
def H(counts):
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c > 0) if total else 0.0


def score_slow(labels, left_mask, n_classes):
    n = len(labels)
    left = [l for l, m in zip(labels, left_mask) if m]
    right = [l for l, m in zip(labels, left_mask) if not m]
    count = lambda part: [part.count(c) for c in range(n_classes)]
    h_class = H(count(labels))
    h_split = H([len(left), len(right)])
    info = h_class - (len(left) * H(count(left)) + len(right) * H(count(right))) / n
    return 2 * info / (h_split + h_class) if h_split + h_class > 0 else 0.0


for _ in range(300):
    n, k, m = rng.integers(1, 13), rng.integers(1, 6), rng.integers(1, 5)
    labels = rng.integers(0, m, size=n)
    go_left = rng.random((n, k)) < 0.5
    got = split_scores(go_left, np.eye(m)[labels])
    want = [score_slow(labels.tolist(), go_left[:, j].tolist(), m) for j in range(k)]
    assert np.allclose(got, want) and np.all((got >= -1e-12) & (got <= 1 + 1e-12))


# 8. Every node obeys the module docstring: route the training samples in plain Python.
def check_tree(tree, X, y, n_min):
    codes = np.searchsorted(tree.classes_, y)
    stack, seen = [(0, list(range(len(X))))], 0
    while stack:
        node, rows = stack.pop()
        seen += 1
        assert rows
        counts = np.bincount(codes[rows], minlength=len(tree.classes_))
        assert np.array_equal(tree.value_[node], counts)
        sub = X[rows]
        constant = [f for f in range(X.shape[1]) if sub[:, f].min() == sub[:, f].max()]
        pure = (counts > 0).sum() == 1
        f = tree.feature_[node]
        if f < 0:
            assert len(rows) < n_min or pure or len(constant) == X.shape[1]
            assert np.isnan(tree.threshold_[node]) and tree.left_[node] == tree.right_[node] == -1
            continue
        assert len(rows) >= n_min and not pure and f not in constant
        cut = tree.threshold_[node]
        assert sub[:, f].min() < cut <= sub[:, f].max()
        left = [r for r in rows if X[r, f] < cut]
        right = [r for r in rows if not X[r, f] < cut]
        assert left and right and tree.left_[node] == node + 1
        stack += [(tree.right_[node], right), (tree.left_[node], left)]
    assert seen == len(tree.feature_)


for trial in range(12):
    n, d = rng.integers(1, 120), rng.integers(1, 6)
    if trial % 2:
        X = rng.integers(0, 3, size=(n, d)).astype(float)
    else:
        X = rng.normal(size=(n, d)).round(1)
    y = rng.integers(0, 3, size=n) * 10 - 5
    k, n_min = int(rng.integers(1, d + 1)), int(rng.choice([2, 3, 7]))
    tree = ExtraTree(max_features=k, min_samples_split=n_min, seed=trial).fit(X, y)
    check_tree(tree, X, y, n_min)

    # apply() against a plain traversal, including points that sit exactly on a cut.
    def apply_slow(x):
        node = 0
        while tree.feature_[node] >= 0:
            go_left = x[tree.feature_[node]] < tree.threshold_[node]
            node = tree.left_[node] if go_left else tree.right_[node]
        return node

    Q = rng.normal(size=(50, d))
    for i, node in enumerate(np.flatnonzero(tree.feature_ >= 0)[:50]):
        Q[i, tree.feature_[node]] = tree.threshold_[node]
    Q = np.concatenate([Q, X])
    assert tree.apply(Q).tolist() == [apply_slow(q) for q in Q]

# 9. With K = d and one feature equal to the label, the root takes the perfect split.
X = np.column_stack([rng.normal(size=(100, 4)), np.repeat([0.0, 1.0], 50)])
y = X[:, 4].astype(int)
for seed in range(20):
    tree = ExtraTree(max_features=5, seed=seed).fit(X, y)
    assert tree.feature_[0] == 4 and len(tree.feature_) == 3

# 10. Ensemble probabilities are vote fractions (counted with plain loops), and with pure
#     leaves the hard vote equals averaging the trees' predict_proba (follow-up 3).
X, y = rng.normal(size=(60, 3)), rng.integers(0, 4, size=60)
Q = rng.normal(size=(40, 3))
forest = ExtraTreesClassifier(n_estimators=7, seed=3).fit(X, y)
slow = np.zeros((40, 4))
for t in forest.trees_:
    for i, label in enumerate(t.predict(Q)):
        slow[i, forest.classes_.tolist().index(label)] += 1 / 7
assert np.allclose(forest.predict_proba(Q), slow)
X, y = make_grid_data(250, seed=0)
X_test, y_test = make_grid_data(400, seed=1)
forest = ExtraTreesClassifier(n_estimators=50, seed=0).fit(X, y)
soft = np.mean([t.predict_proba(X_test) for t in forest.trees_], axis=0)
assert np.array_equal(forest.classes_[soft.argmax(axis=1)], forest.predict(X_test))

# 11. Follow-up 2: fancy-index += counts a repeated (row, class) pair once; np.add.at does not.
votes = np.array([[0, 1], [0, 1], [2, 1]])
a = np.zeros((2, 3))
a[np.arange(2), votes] += 1
b = np.zeros((2, 3))
np.add.at(b, (np.arange(2), votes), 1)
assert a.tolist() == [[1, 0, 1], [0, 1, 0]] and b.tolist() == [[2, 0, 1], [0, 3, 0]]
n, n_classes = votes.shape[1], 3
c = np.bincount((np.arange(n) * n_classes + votes).ravel(), minlength=n * n_classes).reshape(n, n_classes)
e = (votes[:, :, None] == np.arange(n_classes)).sum(axis=0)
assert np.array_equal(c, b) and np.array_equal(e, b)

# 12. Follow-up 4: uniform(lo, hi) can return lo exactly; the guard then uses hi.
draws = np.random.default_rng(0).uniform(1e16, 1e16 + 2, size=1000)
assert 0.3 < np.mean(draws == 1e16) < 0.7
for seed in range(30):
    tree = ExtraTree(max_features=1, seed=seed).fit(np.array([[1e16], [1e16 + 2]]), [0, 1])
    assert tree.threshold_[0] == 1e16 + 2 and len(tree.feature_) == 3

# 13. Follow-up 5: on n evenly spaced points with distinct labels, the mean leaf depth is
#     2 (H_n - 1) (every gap is equally likely to be cut; see the Isolation Forest page).
n = 16
x = np.arange(n, dtype=float)[:, None]
mean_depths = []
for seed in range(400):
    tree = ExtraTree(max_features=1, seed=seed).fit(x, np.arange(n))
    depth = np.zeros(len(tree.feature_), dtype=int)
    for i in np.flatnonzero(tree.feature_ >= 0):  # pre-order: parents come first
        depth[tree.left_[i]] = depth[tree.right_[i]] = depth[i] + 1
    mean_depths.append(depth[tree.apply(x)].mean())
want = 2 * (sum(1 / i for i in range(1, n + 1)) - 1)
assert abs(np.mean(mean_depths) - want) < 5 * np.std(mean_depths) / np.sqrt(400)

# 14. Sanity check against scikit-learn (used here as a reference only, never in trees.py).
from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTrees

ours = np.mean(forest.predict(X_test) == y_test)
theirs = np.mean(SklearnExtraTrees(n_estimators=50, max_features=2, bootstrap=False,
                                   random_state=0).fit(X, y).predict(X_test) == y_test)
print(f"test accuracy: ours {ours:.4f}, scikit-learn {theirs:.4f}")
assert ours > 0.95 and theirs > 0.95 and abs(ours - theirs) < 0.03

# 15. Part 2: the example, a plain-Python brute force, and the edge cases.
go_left = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)
assert np.allclose(variance_scores(go_left, [1.0, 2.0, 3.0, 10.0]), [0.5, 0.96, 0.0])


def variance_score_slow(y, mask):
    var = lambda v: sum((t - sum(v) / len(v)) ** 2 for t in v) / len(v) if v else 0.0
    left = [t for t, m in zip(y, mask) if m]
    right = [t for t, m in zip(y, mask) if not m]
    total = var(y)
    if total == 0:
        return 0.0
    return (total - len(left) / len(y) * var(left) - len(right) / len(y) * var(right)) / total


for _ in range(300):
    n, k = rng.integers(2, 15), rng.integers(1, 5)
    y = rng.normal(size=n) * 100 + 1e3
    go_left = rng.random((n, k)) < 0.5
    want = [variance_score_slow(y.tolist(), go_left[:, j].tolist()) for j in range(k)]
    assert np.allclose(variance_scores(go_left, y), want)
assert np.array_equal(variance_scores(np.ones((3, 2), dtype=bool), [0.1] * 3), [0.0, 0.0])
assert np.array_equal(variance_scores(np.array([[True, False]]), [5.0]), [0.0, 0.0])

print(f"checks took {time.perf_counter() - T0:.1f} s")
print("all checks passed")
```

</details>

</details>
