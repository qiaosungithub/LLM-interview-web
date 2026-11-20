"""Judge-side helpers for tests.py: the module docstring restated in plain Python. The forests an answer grows
depend on its random draws, so tests check every node against the rules instead of comparing with a fixed forest."""
import math
from fractions import Fraction

import numpy as np

EULER_GAMMA = 0.5772156649


def c_slow(n):
    """c(n) of the module docstring for one integer n."""
    n = int(n)
    if n > 2:
        return 2.0 * (math.log(n - 1) + EULER_GAMMA) - 2.0 * (n - 1) / n
    return 1.0 if n == 2 else 0.0


def check_forest(forest, X, n_trees, max_samples):
    """None if the fitted forest obeys the module docstring on X, else a Chinese sentence about the first
    violation. Reads samples_, max_samples_, height_limit_, feature_, threshold_, size_ as numpy arrays."""
    n, d = X.shape
    psi = min(max_samples, n)
    limit = math.ceil(math.log2(psi))
    if forest.max_samples_ != psi:
        return f"max_samples_ 应当是 min(max_samples, n) = min({max_samples}, {n}) = {psi}，得到 {forest.max_samples_}"
    if forest.height_limit_ != limit:
        return f"height_limit_ 应当是 ceil(log2({psi})) = {limit}，得到 {forest.height_limit_}"
    samples = np.asarray(forest.samples_)
    if samples.shape != (n_trees, psi):
        return f"samples_ 的形状应当是 (n_trees, max_samples_) = ({n_trees}, {psi})，得到 {samples.shape}"
    if not np.all(np.mod(samples, 1) == 0) or samples.min() < 0 or samples.max() >= n:
        return f"samples_ 里应当是 0 到 {n - 1} 之间的行号，得到的最小值 {samples.min()}、最大值 {samples.max()}"
    samples = samples.astype(int)
    n_nodes = 2 ** (limit + 1) - 1
    arrays = {}
    for name in ("feature_", "threshold_", "size_"):
        a = np.asarray(getattr(forest, name))
        if a.shape != (n_trees, n_nodes):
            return f"{name} 的形状应当是 (n_trees, 2 ** (l + 1) - 1) = ({n_trees}, {n_nodes})，得到 {a.shape}"
        arrays[name] = a
    feature, threshold, size = arrays["feature_"], arrays["threshold_"], arrays["size_"]
    for t in range(n_trees):
        rows = samples[t].tolist()
        if len(set(rows)) != psi:
            dups = sorted({r for r in rows if rows.count(r) > 1})
            return (f"第 {t} 棵树的子样本里有重复的行号（应当不放回地抽 {psi} 个不同的行），重复的有 "
                    f"{dups[:8]}{' 等' if len(dups) > 8 else ''}")
        visited = set()
        stack = [(0, 0, rows)]
        while stack:
            node, depth, rows = stack.pop()
            visited.add(node)
            where = f"第 {t} 棵树的节点 {node}（深度 {depth}）"
            if size[t, node] != len(rows):
                return f"{where}：有 {len(rows)} 个子样本行到达这里，size_[{t}, {node}] 却是 {size[t, node]}"
            sub = X[rows]
            movable = [f for f in range(d) if sub[:, f].min() < sub[:, f].max()]
            q, p = int(feature[t, node]), float(threshold[t, node])
            if depth >= limit or len(rows) <= 1 or not movable:
                rule = (f"(a) 深度 {depth} >= l = {limit}" if depth >= limit else
                        f"(b) 只有 {len(rows)} 行" if len(rows) <= 1 else "(c) 每个特征都是常量")
                if q != -1 or p != 0.0:
                    return (f"{where}满足规则 {rule}，应当是叶子（feature_ = -1，threshold_ = 0.0），"
                            f"得到 feature_ = {q}，threshold_ = {p!r}")
                continue
            if q < 0:
                return (f"{where}有 {len(rows)} 行，特征 {movable} 不是常量，深度没到上限 {limit}，应当切分，"
                        f"却是 feature_ = {q}")
            if q not in movable:
                return f"{where}用特征 {q} 切分，可只能从这个节点的非常量特征 {movable} 里挑"
            lo, hi = float(sub[:, q].min()), float(sub[:, q].max())
            if not lo < p <= hi:
                return (f"{where}用特征 {q} 切分，切分值 {p!r} 不在 (min, max] = ({lo!r}, {hi!r}] 里："
                        f"应当从这个节点子样本的 [min, max) 里取（取到 min 换成 max）")
            stack.append((2 * node + 2, depth + 1, [r for r in rows if not X[r, q] < p]))
            stack.append((2 * node + 1, depth + 1, [r for r in rows if X[r, q] < p]))
        for i in range(n_nodes):
            if i not in visited and (size[t, i] != 0 or feature[t, i] != -1 or threshold[t, i] != 0.0):
                return (f"第 {t} 棵树的节点 {i} 不存在（没有子样本走到这里），应当是 size_ = 0、feature_ = -1、"
                        f"threshold_ = 0.0，得到 {size[t, i]}、{feature[t, i]}、{threshold[t, i]!r}")
    return None


def path_slow(feature, threshold, size, t, x):
    """(edges, leaf) for one point in tree t, walking the heap arrays one node at a time."""
    node, edges = 0, 0
    while feature[t, node] >= 0:
        q, p = int(feature[t, node]), float(threshold[t, node])
        node = 2 * node + 1 if x[q] < p else 2 * node + 2
        edges += 1
    return edges, node


def paths_slow(forest, Q):
    """h(x) for every tree and every row of Q, shape (n_trees, len(Q))."""
    feature, threshold, size = (np.asarray(forest.feature_), np.asarray(forest.threshold_),
                                np.asarray(forest.size_))
    out = np.zeros((feature.shape[0], len(Q)))
    for t in range(feature.shape[0]):
        for j, x in enumerate(Q):
            edges, leaf = path_slow(feature, threshold, size, t, x)
            out[t, j] = edges + c_slow(size[t, leaf])
    return out


def exact_depth(n_max):
    """D(n) = 2 (H_n - 1) for n = 1..n_max, from exact fractions (small n_max only)."""
    out, h = [], Fraction(0)
    for n in range(1, n_max + 1):
        h += Fraction(1, n)
        out.append(float(2 * (h - 1)))
    return out


def literal(a):
    return f"np.array({np.asarray(a).tolist()!r})"
