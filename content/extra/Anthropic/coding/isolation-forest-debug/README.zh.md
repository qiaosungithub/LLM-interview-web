# 调试向量化的 Isolation Forest 实现

## 题目

这是 [调试向量化的 Extra-Trees 实现](#/anthropic/coding/extratrees-debug) 的同类练习。证据里没有人说考过 Isolation Forest。我们照那道题的格式和规则另写了一份，给你再练一次"读规格、找 bug"。

下面的 `iforest.py` 用向量化 NumPy 实现了孤立森林（Isolation Forest，Liu、Ting、Zhou，2008），一种无监督的异常检测方法。每棵树在一个不放回抽取的子样本上随机切分，直到点被孤立，或者到达高度上限。一个点在各棵树里的平均路径越短，越容易被孤立，异常分数越高。所有树存在几个形状为 `(n_trees, n_nodes)` 的数组里，打分时所有树、所有点一起往下走。代码大体能用（多数时候能把离群点排在前面），但有几个 bug。`test_iforest.py` 是给你的测试。

规则和 Extra-Trees 那道题一样：

- 只改 `iforest.py`。不要改测试，也不要改 docstring。它们都是对的，docstring 就是规格。
- 保持向量化：不要把 NumPy 的向量运算改写成 Python 嵌套循环。
- 不能用 scikit-learn 或任何现成的实现，不能用 AI。可以查 NumPy / Python 文档。
- 限时 60 分钟。每个测试分值相同。

本地练习：把下面两个代码块分别存成 `iforest.py` 和 `test_iforest.py`，放在同一个目录，运行 `python test_iforest.py`（只需要 NumPy，不需要 pytest）。`python test_iforest.py test_few_points` 只跑一个测试，并打印 traceback。

### Part 1 —— 让测试全部通过

`iforest.py`：

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

`test_iforest.py`：

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

不做任何修改时，`python test_iforest.py` 的输出是：

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

任务：修好 `iforest.py`，让 9 个测试全部通过。修完的代码要符合 docstring 的规格，而不只是让测试变绿。说明你改了哪几处，每处错在哪里。

### Part 2 —— c(n) 从哪里来

设想 $n$ 个等间距的一维点（比如 $0, 1, \dots, n-1$）。不设高度上限，按 docstring 的切分规则一直切，直到每个叶子只剩一个点。记 $D(n)$ 为"随机挑一个点，它所在叶子的深度"的期望，对树的随机性和挑哪个点同时取期望。$D(1) = 0$。

1. 写出 $D(n)$ 的递推式，并实现 `expected_depth(n_max)`：返回长度为 `n_max` 的浮点数组，第 `i` 个元素是 $D(i + 1)$，要求 $O(n_{max})$ 时间。
2. 证明 $D(n) = D(n-1) + 2/n$，从而 $D(n) = 2(H_n - 1)$，其中 $H_n = 1 + 1/2 + \dots + 1/n$ 是精确的调和数。它和 docstring 里的 $c(n)$ 是什么关系？docstring 用 $\ln i + 0.5772156649$ 近似 $H(i)$，$n = 3$ 和 $n = 256$ 时 $c(n)$ 各差多少？为什么要单独规定 $c(2) = 1$？

```py
def expected_depth(n_max):
    """[D(1), D(2), ..., D(n_max)] as a float array; n_max >= 1."""
```

```text
expected_depth(4) = [0, 1, 5/3, 13/6] ≈ [0.0, 1.0, 1.6667, 2.1667]
```

## 参考解答

<details>
<summary>展开参考解答</summary>

### Part 1

**思路。** 先修崩溃：`test_few_points` 的 `IndexError` 指向 `samples_` 里的下标。再拿 docstring 当清单逐条对照：子样本怎么抽、高度上限、叶子规则、`size_` 记什么、`h(x) = e + c(size)` 怎么算、$c(n)$ 的三段定义、分数公式。`test_path_length_manual` 是一棵手算过的小树，最适合用来对照打分这一段：它同时卡住了三个 bug。注意 `test_ranks_outliers_first` 一开始就通过：几个 bug 只改变分数的刻度，不改变排序，只看"能不能找出异常点"是发现不了的。

共有 6 个 bug。下表最后一列是"只剩这一个 bug、其他 5 个都修好"时失败的测试（验证代码里逐个跑过）：

| # | 位置 | 错在哪 | 只剩它时失败的测试 |
|---|---|---|---|
| 1 | `c_factor` | `np.zeros_like(n)` 继承了整数类型，浮点结果被截断 | `test_c_factor`、`test_path_length_manual` |
| 2 | `fit` 抽子样本 | `argsort(..., axis=0)` 排错了轴 | `test_subsamples`、`test_few_points`（IndexError） |
| 3 | `_grow` | 先 `return` 后记 `size_`，叶子大小留成 0 | `test_tree_structure`、`test_score_scale` |
| 4 | `path_length` | 到了叶子的点还在往下走 | `test_path_length_manual`、`test_constant_data` |
| 5 | `score` | 用了 `max_samples` 而不是实际的 `max_samples_` | `test_constant_data`、`test_few_points` |
| 6 | `score` | 运算符优先级：`2.0 ** -mean_path / norm` | `test_path_length_manual`、`test_constant_data`、`test_few_points`、`test_score_scale` |

**Bug 1：`np.zeros_like` 继承了整数类型。** `c_factor` 的输入总是整数：叶子大小 `size_` 是整数数组，`max_samples_` 是 Python 整数。`np.zeros_like(n)` 建出的输出也是整数数组，后面赋进去的浮点数被悄悄截断：$c(3) = 1.207$ 变成 1，$c(256) = 10.245$ 变成 10。docstring 说返回浮点数组。`test_c_factor` 直接查数值；`test_path_length_manual` 里 $c(4) = 1.85$ 被截成 1。改成 `np.zeros(n.shape)`（默认 float64）。

**Bug 2：`argsort` 排错了轴。** 想要的是每棵树（每一行）一个 $0..n-1$ 的随机排列，取前 $\psi$ 个当子样本。`axis=0` 却沿着树的方向排序，每一列是 $0..n_{trees}-1$ 的排列，于是 `samples_` 里装的是树的编号，不是行号。树比样本少时（`test_subsamples`：40 棵树、300 行），只有前 40 行会被用到；树比样本多时（`test_few_points`：200 棵树、8 行），下标越界，抛 `IndexError`。改成 `axis=1`。

**Bug 3：先 return，后记账。** `self.size_[t, node] = len(S)` 被放到了第一个 `return` 后面。因为到达高度上限、或只剩不到 2 个点而成为叶子的节点，大小都没记下来，留着 0。docstring 说 `size_` 是到达该节点的样本数。打分时，这些叶子的 $c(\text{size})$ 变成了 $c(0) = 0$。高度上限处本来还有好几个点的叶子，少加了"没长出来的那部分深度"，于是正常点的路径变短、分数偏高：`test_score_scale` 里内点分数的中位数从约 0.44 升到约 0.58。`test_tree_structure` 则发现了大小为 0 的子节点。把这行赋值移回函数的第一行。

**Bug 4：已经到达叶子的点还在往下走。** `path_length` 用固定的 $l$ 轮把所有点一起往下推。到了叶子的点应该停在原地，即 `np.where(internal, 2 * node + 1 + go_right, node)`；代码却无条件更新 `node`，让它们继续走进不存在的子孙节点。边数没有算错（`edges += internal` 只在内部节点加 1），但最后查到的是不存在节点的 `size_ = 0`，$c(\text{size})$ 丢了。叶子只有 1 个点时 $c(1) = 0$，看不出差别；连续数据上，高度上限以前停下的叶子都只有 1 个点。只有"上限以前就停下、却有好几个点"的叶子才会暴露这个 bug，也就是点完全相同的叶子：`test_constant_data` 里根节点就是叶子，分数应为 0.5，却成了 1；手工树里只有根节点的第 2 棵树也是这样。

**Bug 5：用了请求值，而不是实际值。** 分数应该用实际的子样本大小 $\psi$ = `max_samples_` = `min(max_samples, n)` 来归一化，代码用了 `max_samples`（默认 256）。数据少于 256 行时，$c(256) \approx 10.2$ 远大于 $c(\psi)$，所有分数都被推向 1。`test_few_points` 只有 8 个点，内点的最高分从约 0.5 变成 0.8 以上；`test_constant_data` 的第二段（40 行，默认 `max_samples`）分数不再是 0.5。

**Bug 6：运算符优先级。** 按 Python 的优先级，`2.0 ** -mean_path / norm` 是 `(2.0 ** -mean_path) / norm`，不是 $2^{-E[h]/c(\psi)}$。分数仍然随平均路径单调下降，所以排序不变，`test_ranks_outliers_first` 照样通过；但刻度全错了：平均路径正好等于 $c(\psi)$ 的点本该得 0.5，$\psi = 64$ 时却得到 $2^{-c(64)}/c(64) \approx 0.00075$。加上括号。

**和论文对照。** 修好后的代码和 Liu 等人的 iForest 一致：每棵树用不放回抽取的 $\psi$ 个点，高度上限 $l = \lceil \log_2 \psi \rceil$；在当前节点随机挑一个特征、在它的最小值和最大值之间均匀取切分值，`x[q] < p` 走左边；路径长度是边数加上叶子大小的 $c(\cdot)$ 修正；分数 $s(x) = 2^{-E[h(x)]/c(\psi)}$。$c(2) = 1$、$c(n \le 1) = 0$ 两个特例与 scikit-learn 的实现相同（验证代码逐值比较过 $n = 0..1000$）。论文的默认值是 $\psi = 256$、100 棵树。

修好后的 `iforest.py`：

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

**递推式。** 点是不同的、等间距的，所以根节点的切分值 $p$ 在 $x_1$ 和 $x_n$ 之间均匀分布时（docstring 的规则等于在 $(x_1, x_n]$ 上均匀取值），落进 $n - 1$ 个间隙中的每一个的概率都是 $1/(n-1)$。落在第 $i$ 个间隙时，左边 $i$ 个点、右边 $n - i$ 个点，两边仍然是等间距的点，切分规则也相同，所以子树里的深度分布和"从头开始长一棵 $i$ 个点的树"一样。随机挑的点以 $i/n$ 的概率在左边：

$$D(n) = 1 + \frac{1}{n-1}\sum_{i=1}^{n-1}\left(\frac{i}{n}D(i) + \frac{n-i}{n}D(n-i)\right) = 1 + \frac{2}{n(n-1)}\sum_{k=1}^{n-1} k\,D(k).$$

维护前缀和 $\sum k\,D(k)$，每个 $n$ 只要 $O(1)$：

```python
def expected_depth(n_max):
    depth = np.zeros(n_max)
    weighted = 0.0  # sum of k * D(k) for k < n
    for n in range(2, n_max + 1):
        weighted += (n - 1) * depth[n - 2]
        depth[n - 1] = 1.0 + 2.0 * weighted / (n * (n - 1))
    return depth
```

**闭式解。** 记 $W(m) = \sum_{k=1}^{m} k\,D(k)$。由递推式，$n(n-1)(D(n) - 1) = 2W(n-1)$，$(n-1)(n-2)(D(n-1) - 1) = 2W(n-2)$。两式相减，右边是 $2(n-1)D(n-1)$；两边除以 $n - 1$，整理得 $n\,D(n) = n\,D(n-1) + 2$，即 $D(n) = D(n-1) + 2/n$（$n = 2$ 时直接代入也成立）。从 $D(1) = 0$ 累加：$D(n) = 2(1/2 + \dots + 1/n) = 2(H_n - 1)$。

**和 $c(n)$ 的关系。** 用精确的调和数，$2H_{n-1} - 2(n-1)/n = 2H_{n-1} + 2/n - 2 = 2(H_n - 1)$，正好是 $D(n)$。所以 $c(n)$ 就是"在 $n$ 个点上把树长满，一个点的平均深度"。论文把它说成二叉搜索树里不成功查找的平均路径长度，是同一个量。到达高度上限、还剩 $m$ 个点的叶子，加上 $c(m)$，就是在估计那棵没长出来的子树的平均深度。（对一般的数据，间隙长短不一，各间隙被切中的概率不再相等，$c(n)$ 只是一个近似。）

docstring 用 $\ln i + \gamma$ 近似 $H(i)$。$n = 3$ 时精确值 $5/3 \approx 1.6667$，近似值 1.2074，差 $-0.459$；$n = 256$ 时精确值约 10.2487，近似值 10.2448，差只有 $-0.0039$。$i$ 越小，$H(i) - \ln i$ 离 $\gamma$ 越远（$H(1) - \ln 1 = 1$，而 $\gamma \approx 0.577$）。$n = 2$ 时近似公式给出 $2(\ln 1 + \gamma) - 1 = 2\gamma - 1 \approx 0.154$，而真值是 1，差得太多，所以单独规定 $c(2) = 1$；$n = 1$ 时公式里有 $\ln 0$，没有定义，而一个点的叶子不需要再切，规定 $c(1) = 0$。

### 追问

- **为什么要子采样？$\psi$ 取多大？** 论文说子采样能减轻两个问题：淹没（swamping，正常点离异常点太近而被误判）和掩盖（masking，异常点成团，互相掩护）。$\psi = 256$、100 棵树在论文的实验里基本够用，再加大收益很小。$\psi$ 小还让每棵树都很小，训练代价与 $n$ 无关（抽样本身除外）。
- **为什么到高度上限就停？加 $c(\text{size})$ 有什么用？** 异常点在很浅的地方就被孤立了，深处只关系到正常点，而正常点的精确深度对排序不重要，所以不必把树长满。到上限停下的叶子还有 $m$ 个点，加上 $c(m)$ 估计剩下的深度，让被截断的路径和长满的树可比。少了这一项（bug 3、bug 4），正常点的路径偏短、分数偏高。
- **为什么用 $c(\psi)$ 归一化，而不是 $c(n)$？** 树是在 $\psi$ 个点上长的，一个普通点的期望路径长度大约是 $c(\psi)$，和总样本数 $n$ 无关。用 $c(\psi)$ 归一化后，$E[h] = c(\psi)$ 的点得 0.5，分数越接近 1 越异常，明显低于 0.5 的大致可以看作正常点。用错了数（bug 5），整个刻度就偏了。
- **scikit-learn 的 `score_samples`、`decision_function`、`predict` 和这里的 $s(x)$ 是什么关系？** `score_samples` 返回 $-s(x)$（越小越异常）。`contamination='auto'` 时 `offset_ = -0.5`，`decision_function = score_samples - offset_` $= 0.5 - s(x)$，`predict` 在它小于 0（即 $s(x) > 0.5$）时给 $-1$（异常），否则给 1。指定 `contamination` 为比例 $r$ 时，`offset_` 改为训练集 `score_samples` 的 $100r$ 分位数。（验证代码在 scikit-learn 上核对过这几条。）
- **复杂度和存储？** 每棵树最多 $l$ 层，每层所有节点加起来最多 $\psi$ 个点，求最小、最大值要 $O(\psi d)$，所以训练一棵树最坏 $O(\psi d \log \psi)$（高度上限让它成为最坏情况，而不只是期望）。打分是 $l$ 轮向量化的下移，每轮 $O(n_{trees} \cdot n)$，共 $O(n_{trees}\, n \log \psi)$。堆式布局不存子节点指针，第 $i$ 个节点的孩子就是 $2i+1$、$2i+2$，所以能对所有树、所有点同时下移；代价是不平衡的树会浪费位置。因为 $2^l < 2\psi$，每棵树的数组长度 $2^{l+1} - 1 < 4\psi$，浪费有上界。

<details>
<summary>验证代码（可运行）</summary>

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
