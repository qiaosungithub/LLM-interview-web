"""Judge-side helpers for tests.py: plain-Python restatements of the docstring rules, used to check the trees an
answer grows (whatever random numbers it draws) and to compute expected scores."""
import math

import numpy as np


# ---------------------------------------------------------------- scores, by brute force

def entropy_slow(counts):
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c > 0) if total else 0.0


def split_score_slow(labels, mask, n_classes):
    """2 I / (H_split + H_class) for one candidate, from the definition."""
    n = len(labels)
    left = [l for l, m in zip(labels, mask) if m]
    right = [l for l, m in zip(labels, mask) if not m]
    count = lambda part: [part.count(c) for c in range(n_classes)]
    h_class = entropy_slow(count(labels))
    h_split = entropy_slow([len(left), len(right)])
    info = h_class - (len(left) * entropy_slow(count(left)) + len(right) * entropy_slow(count(right))) / n
    return 2 * info / (h_split + h_class) if h_split + h_class > 0 else 0.0


def _var(v):
    if not v:
        return 0.0
    m = math.fsum(v) / len(v)
    return math.fsum((t - m) ** 2 for t in v) / len(v)


def variance_score_slow(y, mask):
    """(var(y) - n_l/n var(y_l) - n_r/n var(y_r)) / var(y), two-pass and exactly summed; 0 if all targets equal."""
    y = [float(t) for t in y]
    if all(t == y[0] for t in y):
        return 0.0
    left = [t for t, m in zip(y, mask) if m]
    right = [t for t, m in zip(y, mask) if not m]
    n, total = len(y), _var(y)
    return (total - len(left) / n * _var(left) - len(right) / n * _var(right)) / total


def best_split_score(X, labels, f):
    """The highest score any cut on feature f can reach (every gap between distinct values tried)."""
    values = np.unique(X[:, f])
    n_classes = int(labels.max()) + 1
    best = 0.0
    for cut in values[1:]:
        best = max(best, split_score_slow(labels.tolist(), (X[:, f] < cut).tolist(), n_classes))
    return best


# ---------------------------------------------------------------- the tree specification

def apply_slow(feature, threshold, left, right, x):
    """The leaf a single sample reaches: left if x[f] < threshold, else right."""
    node = 0
    for _ in range(len(feature) + 1):
        f = int(feature[node])
        if f < 0:
            return node
        node = int(left[node]) if x[f] < threshold[node] else int(right[node])
    raise RuntimeError("cycle in the tree")


def check_tree(arrays, X, y, n_min):
    """Route the training samples through the tree in plain Python and check every node against the module
    docstring. `arrays` is (classes, feature, threshold, left, right, value) as numpy arrays of the right
    shapes. Returns None, or a Chinese sentence about the first node that breaks a rule."""
    classes, feature, threshold, left, right, value = arrays
    n_nodes, d = len(feature), X.shape[1]
    want_classes = np.unique(y)
    if not (len(classes) == len(want_classes) and np.array_equal(classes, want_classes)):
        return f"classes_ 应当是排好序的不同标签 {want_classes.tolist()}，得到 {np.asarray(classes).tolist()}"
    codes = np.searchsorted(want_classes, y)
    stack = [(np.arange(len(X)), None, None)]          # (rows, parent, side); popped in pre-order
    next_id = 0
    while stack:
        rows, parent, side = stack.pop()
        node = next_id
        next_id += 1
        if parent is not None:
            pointer = left if side == "left" else right
            name = "left_" if side == "left" else "right_"
            if int(pointer[parent]) != node:
                return (f"节点编号不是先序：节点 {parent} 的{'左' if side == 'left' else '右'}子树按先序应当从编号 {node} "
                        f"开始，可 {name}[{parent}] = {int(pointer[parent])}")
        if node >= n_nodes:
            return f"按先序编号，树至少有 {node + 1} 个节点，但节点数组只有 {n_nodes} 个"
        counts = np.bincount(codes[rows], minlength=len(classes))
        if not np.array_equal(np.asarray(value[node], dtype=float), counts.astype(float)):
            return (f"节点 {node}：到达它的训练样本各类数量是 {counts.tolist()}，value_[{node}] 却是 "
                    f"{np.asarray(value[node]).tolist()}")
        sub = X[rows]
        constant = [f for f in range(d) if sub[:, f].min() == sub[:, f].max()]
        varying = [f for f in range(d) if f not in constant]
        pure = int((counts > 0).sum()) <= 1
        f = int(feature[node])
        if len(rows) < n_min or pure or not varying:
            rule = (f"(a) 样本数 {len(rows)} < min_samples_split={n_min}" if len(rows) < n_min
                    else "(b) 标签全相同" if pure else "(c) 每个特征都是常量")
            if f != -1 or not np.isnan(threshold[node]) or int(left[node]) != -1 or int(right[node]) != -1:
                return (f"节点 {node} 满足规则 {rule}，应当是叶子（feature_ = -1，threshold_ = nan，left_ = right_ = -1），"
                        f"得到 feature_ = {f}，threshold_ = {threshold[node]}，left_ = {int(left[node])}，"
                        f"right_ = {int(right[node])}")
            continue
        if f < 0:
            return (f"节点 {node} 有 {len(rows)} 个样本（min_samples_split={n_min}），标签不全相同，特征 {varying} 不是常量："
                    f"(a)(b)(c) 一条都不满足，应当分裂，却成了叶子")
        if f >= d:
            return f"节点 {node} 的 feature_ = {f}，可数据只有 {d} 个特征"
        if f in constant:
            return f"节点 {node} 用特征 {f} 切分，可这个特征在节点的 {len(rows)} 个样本里是常量（只能从非常量特征里抽）"
        cut = float(threshold[node])
        lo, hi = float(sub[:, f].min()), float(sub[:, f].max())
        if not lo < cut <= hi:
            return (f"节点 {node} 用特征 {f} 切分，切点 {cut!r} 不在 (min, max] = ({lo!r}, {hi!r}] 里：切点要从这个节点"
                    f"样本的 [min, max) 里抽（抽到 min 换成 max），否则有一个子节点没有样本")
        go_left = sub[:, f] < cut
        stack.append((rows[~go_left], node, "right"))
        stack.append((rows[go_left], node, "left"))
    if next_id != n_nodes:
        return f"按规则长出的树有 {next_id} 个节点，节点数组却有 {n_nodes} 个"
    return None


def subtree_depths(feature, left, right):
    depth = np.zeros(len(feature), dtype=int)
    for i in range(len(feature)):          # pre-order: a parent comes before its children
        if feature[i] >= 0:
            depth[left[i]] = depth[right[i]] = depth[i] + 1
    return depth


# ---------------------------------------------------------------- inputs

def literal(a):
    return f"np.array({np.asarray(a).tolist()!r})"


def make_grid_data(n, seed):
    """The statement's test data."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, 6)).astype(float)
    y = 3 + 4 * ((X[:, 0] + X[:, 1] >= 3).astype(int) + (X[:, 2] >= 2).astype(int))
    return X, y
