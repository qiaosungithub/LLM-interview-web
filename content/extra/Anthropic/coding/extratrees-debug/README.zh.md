# 调试向量化的 Extra-Trees 实现

## 题目

下面的 `trees.py` 用向量化 NumPy 实现了极端随机树（Extremely Randomized Trees，简称 Extra-Trees；Geurts、Ernst、Wehenkel，2006）分类器。单棵树 `ExtraTree` 把节点存在几个平铺数组里，批量预测时所有样本一起往下走；`ExtraTreesClassifier` 是多棵树的集成，按多数票预测。代码大体能用，但有几个 bug：有些输入会让它崩溃，有些会让结果出错或准确率变差。`test_trees.py` 是给你的测试。

几家备考网站说，Anthropic Fellows 第二轮在线测评（OA）就是这种题：60 分钟内修好 `trees.py`，让测试全部通过。真题代码没有公开。下面的代码、bug 和测试是我们自己写的，规则照原题：

- 只改 `trees.py`。不要改测试，也不要改 docstring。它们都是对的，docstring 就是规格。
- 保持向量化：不要把 NumPy 的向量运算改写成 Python 嵌套循环。
- 不能用 scikit-learn 或任何现成的树实现，不能用 AI。可以查 NumPy / Python 文档。能不能上网搜索，各家说法不一，按不能准备。
- 每个测试分值相同。测试运行时间上限 30 秒，内存 4 GB。死循环会让平台卡住。原题的提示：`test_small_features` 和 `test_ensemble` 最难，可以留到最后。
- 原平台用 pytest，可以在代码里加 `breakpoint()`，再用 `pytest -k <测试名> --pdb` 调试。

本地练习：把下面两个代码块分别存成 `trees.py` 和 `test_trees.py`，放在同一个目录，运行 `python test_trees.py`（只需要 NumPy，不需要 pytest）。`python test_trees.py test_ensemble` 只跑一个测试，并打印 traceback。给自己计时 60 分钟。

### Part 1 —— 让测试全部通过

`trees.py`：

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

`test_trees.py`：

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

不做任何修改时，`python test_trees.py` 的输出是：

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

任务：修好 `trees.py`，让 8 个测试全部通过。修完的代码要符合 docstring 的规格，而不只是让测试变绿。说明你改了哪几处，每处错在哪里。

### Part 2 —— 回归版本的打分函数

原题只有 Part 1。这一部分是我们加的延伸，用来检查你是否真的读懂了这份代码。

论文里，回归树用相对方差减少（relative variance reduction）给候选切分打分。照 `split_scores` 的接口，写出向量化的 `variance_scores`：

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

不能写 Python 循环（不管是对样本还是对候选切分）。`y` 的均值可能比它的波动大很多个数量级（例如 `1e8` 加上方差约 1 的噪声），这时得分的误差也不能超过 `1e-5`。例子：

```text
y       = [1, 2, 3, 10]
go_left 的 3 列：[T, T, F, F]、[T, T, T, F]、[T, T, T, T]
结果    = [0.5, 0.96, 0.0]
```

再回答一个问题：如果要把 `ExtraTree` 改成回归树，除了换掉打分函数，还有哪些地方要改？

## 参考解答

<details>
<summary>展开参考解答</summary>

### Part 1

**思路。** 先把全部测试跑一遍，看失败的种类。`RecursionError` 和 `ValueError` 是崩溃，最好先修：崩溃会挡住后面的断言，修好它们，别的失败才看得清。然后拿 docstring 当清单，逐行对照代码：三条停止规则（a）（b）（c），分裂的 4 个步骤，`apply` 的走向规则，集成的投票方式。每修一处，就把全部测试重跑一遍，因为一个 bug 可能掩盖另一个（见追问 1）。

共有 6 个 bug。下表最后一列是"只剩这一个 bug、其他 5 个都修好"时失败的测试（验证代码里逐个跑过）：

| # | 位置 | 错在哪 | 只剩它时失败的测试 |
|---|---|---|---|
| 1 | `_pick_split` 第一行 | `<=`：样本数正好等于 `min_samples_split` 的节点不再分裂 | `test_noisy_labels`、`test_small_features` |
| 2 | `_pick_split` 求 `lo, hi` | 用了整个 `X` 的最小值和最大值，不是当前节点的 `X[idx]` | `test_tree_structure`、`test_duplicate_points`（RecursionError）、`test_small_features` |
| 3 | `_pick_split` 常量检查 | `np.any`：只要有一个特征是常量就停 | `test_duplicate_points`、`test_small_features`、`test_ensemble` |
| 4 | `_pick_split` 抽特征 | 总是抽 `max_features` 个，没有取 `min(max_features, len(candidates))` | `test_small_features`、`test_ensemble`（都是 ValueError） |
| 5 | `apply` | `<=`：正好等于阈值的点走了左边 | `test_apply_manual_tree` |
| 6 | `ExtraTreesClassifier.predict_proba` | `counts[rows, votes] += 1` 遇到重复的下标只加一次 | `test_ensemble` |

**Bug 1：停止条件差一（off-by-one）。** 规则（a）是 $|S| <$ `min_samples_split` 时才是叶子，所以样本数正好等于它的节点还要继续分。代码写成了 `<=`。默认 `min_samples_split = 2`，于是"两个样本、两种标签"的节点也不分，树没法把训练集记住：`test_small_features` 里 40 个点的条带数据和 3 维奇偶数据都会出错。`test_noisy_labels` 里 `min_samples_split=200` 而样本正好 200 个，根节点本该分裂，却成了叶子；`n_min` 取 2 和 5 时，树里还会出现"样本数正好等于 `n_min`、标签不纯"的叶子，违反规格。

```py
if len(idx) < self.min_samples_split or counts.max() == counts.sum():
```

**Bug 2：切点范围用了全局数据。** 规格第 2 步说切点从 $[\min_S f, \max_S f)$ 里抽，$S$ 是当前节点的样本。代码却对整个 `X` 求最小、最大值。这样切点可能落在当前节点的取值范围之外，所有样本进同一个子节点，另一个子节点是空的，违反"两个子节点都至少有一个样本"（`test_tree_structure` 检查每个节点至少一个样本）。更糟的是，如果一个节点里的样本完全相同、标签却不同（`test_duplicate_points` 里两行 `[1, 1]`，标签是 1 和 2），每次切分都把同一批样本原样传给子节点，无限递归，得到 `RecursionError`（在原平台上可能就是卡死）。改法：

```py
lo, hi = X[idx].min(axis=0), X[idx].max(axis=0)
```

**Bug 3：常量检查用了 `any`。** 规则（c）是"每个特征在 $S$ 里都是常量"才停，代码写成 `np.any(lo == hi)`，只要有一个特征是常量就停。整数特征分几层之后，几乎总有某个特征在节点里变成常量，于是树早早停下，连训练集都拟合不了。`test_ensemble` 的网格数据（取值 0 到 3 的 6 个整数特征）在训练集上就预测错；3 维奇偶数据第一次切分后，切过的那一维就是常量，树停在第 1 层。改成 `np.all`。

**Bug 4：候选特征不够时没有取最小值。** 第 1 步是从非常量特征集合 $A$ 里不放回地抽 $k = \min(K, |A|)$ 个。代码总是抽 `max_features` 个，非常量特征比 $K$ 少时，`rng.choice(..., replace=False)` 抛 `ValueError`。奇偶数据用 $K = 3$，切过一次后只剩 2 个非常量特征；网格数据默认 $K = \mathrm{round}(\sqrt 6) = 2$，深处的节点常常只剩 1 个。改法：

```py
k = min(self.max_features, len(candidates))
feats = self._rng.choice(candidates, size=k, replace=False)
```

**Bug 5：预测时比较方向错了。** `apply` 的 docstring 和训练时的第 4 步都是 `x[f] < cut` 走左边，代码写成了 `<=`。训练数据几乎从不正好落在切点上（切点是连续均匀抽的），所以别的测试都看不出来；只有手工树 `test_apply_manual_tree` 里的 `[2.0, 4.0]` 和 `[3.0, 5.0]` 正好压在阈值上。改回 `<`。

**Bug 6：花式索引的 `+=` 不累加。** `votes` 的形状是 `(n_trees, n)`，`(np.arange(n), votes)` 广播成 `(n_trees, n)` 个 (行, 类) 下标对，其中大量重复。`a[idx] += 1` 等价于 `a[idx] = a[idx] + 1`，先取值、再整体写回，重复的下标只算一次。结果每行记的是"有几种类别得过票"，而不是票数，行和不为 1，按 argmax 选出的类别也不对。改用无缓冲的 `np.add.at`：

```py
np.add.at(counts, (np.arange(len(X)), votes), 1)
```

**和论文对照。** 修好后的代码和 Geurts 等人论文的算法一致：节点样本数小于 $n_{min}$、所有特征为常量、或标签全相同时停止；否则在当前节点非常量的特征里不放回地抽 $K$ 个，每个在节点内的 $[\min, \max]$ 上均匀抽一个切点，取得分最高的切分；分类得分是归一化信息增益 $2 I / (H_s + H_c)$；每棵树用全部训练样本（不做 bootstrap），集成按多数票。唯一的差别是我们用半开区间加"抽到最小值就换成最大值"，避免空子节点（见追问 4）。

修好后的 `trees.py`：

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

**思路。** 先把 $y$ 中心化：$z = y - \bar y$，总平方和 $T = \sum z_i^2 = n\,\mathrm{var}(y)$。左子节点的平方和是 $\sum_{L} z_i^2 - s_l^2 / n_l$，其中 $s_l = \sum_L z_i$；右边同理，且 $s_r = -s_l$（因为 $\sum z_i = 0$）。代回公式，

$$\text{score} = \frac{T - \mathrm{SS}_l - \mathrm{SS}_r}{T} = \frac{s_l^2 / n_l + s_l^2 / n_r}{T}.$$

$s_l$ 对所有候选一次算出：`z @ go_left`，形状 `(k,)`；`n_l = go_left.sum(axis=0)`。空的一边 $s_l = 0$，用 `np.divide(..., where=n > 0)` 让它贡献 0。先中心化，是为了避免"平方的均值减均值的平方"在 $y$ 很大时的抵消误差。复杂度 $O(nk)$，和 `split_scores` 一样。

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

例子：$\bar y = 4$，$T = 9 + 4 + 1 + 36 = 50$。第 1 列 $s_l = -3 - 2 = -5$，得分 $(25/2 + 25/2)/50 = 0.5$；第 2 列 $s_l = -6$，得分 $(36/3 + 36/1)/50 = 0.96$；第 3 列右边为空，$s_l = 0$，得分 0。

**改成回归树还要改什么。** 叶子存目标的均值（或者和与个数），不再存类别计数；停止规则（b）改成"目标值全相同"；单棵树预测返回叶子均值，集成取各棵树预测的平均，而不是投票。论文的默认参数也不同：回归用 $K = d$（全部特征都当候选），$n_{min} = 5$；分类是 $K = \sqrt d$，$n_{min} = 2$。

### 追问

- **修 bug 的顺序有影响吗？** 有，bug 之间会互相掩盖。Bug 2 用全局范围，节点里的常量特征在全局看都不是常量，所以 bug 3（`any` 停止）和 bug 4（`ValueError`）都不会触发。如果先修 bug 6，`test_ensemble` 会直接通过；接着修 bug 2，它又失败了，因为 bug 3、4 露了出来。而 bug 3 让树在出现常量特征时就停，于是也掩盖了 bug 4。所以每修一处都要重跑全部测试，不能只看刚才失败的那个。
- **除了 `np.add.at`，还能怎么数票？** 可以把 (行, 类) 编成一个整数再用 `bincount`：`np.bincount((np.arange(n) * n_classes + votes).ravel(), minlength=n * n_classes).reshape(n, n_classes)`；也可以比较出一个布尔张量再求和：`(votes[:, :, None] == np.arange(n_classes)).sum(axis=0)`，它要 $O(MnC)$ 的内存。三种写法结果相同（验证代码核对过）。
- **集成用多数票，还是平均各棵树的概率？** 论文用多数票，scikit-learn 平均 `predict_proba`（软投票）。在 `min_samples_split = 2` 时，叶子只要还有能切的特征就会继续分到纯，所以叶子几乎都是纯的，两种方式给出相同的预测（验证代码在网格数据上核对过）。叶子不纯（特征全相同、标签不同）或 $n_{min}$ 较大时，两者才会不同，软投票保留了更多信息。
- **为什么切点抽到最小值时要换成最大值？** `rng.uniform(lo, hi)` 的结果在 $[lo, hi)$ 里，但浮点舍入可能正好返回 `lo`，这时 `x < lo` 一个样本也没有，左子节点为空。普通数据上这种概率极小（$2^{-53}$ 量级）；但如果 `lo` 和 `hi` 之间只有很少几个可表示的浮点数，比如 `lo = 1e16, hi = 1e16 + 2`，大约一半的抽样会返回 `lo`（验证代码实测）。换成 `hi` 后，`x < hi` 把除最大值以外的样本都分到左边，两边都不空。
- **复杂度是多少？递归有什么风险？** 一个节点的代价是 $O(|S| \cdot (d + K C))$：求每个特征的最小、最大值 $O(|S| d)$，算 $K$ 个候选的得分 $O(|S| K C)$。同一层的节点加起来最多 $n$ 个样本，所以一棵树是 $O(n \cdot \text{深度} \cdot (d + KC))$。随机切点下，典型数据的深度是 $O(\log n)$（$n$ 个等间距点各自叶子的平均深度正好是 $2(H_n - 1) \approx 2 \ln n$，验证代码用模拟核对过），最坏是 $O(n)$。预测时 `apply` 每轮把所有还在内部节点的样本下移一层，轮数等于最大深度，每轮 $O(n)$。`_grow` 是递归的，深度超过 Python 默认的 1000 层就会出错；要稳妥可以改成显式栈。

<details>
<summary>验证代码（可运行）</summary>

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
