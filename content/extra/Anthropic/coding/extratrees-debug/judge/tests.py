import numpy as np

from judgelib import Mismatch, case, expect_close, expect_true, rng
import ref


def npgen(salt=""):
    seed = rng(salt).getrandbits(32)
    return seed, np.random.default_rng(seed)


def as_array(x, label, call, dtype=None):
    try:
        return np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)
    except Exception:
        raise Mismatch(f"{label} 不能转成数组：{type(x).__name__}", call, None, x) from None


def arrays_of(tree, call):
    """The tree's node arrays, after checking that they exist and have consistent shapes."""
    names = ("classes_", "feature_", "threshold_", "left_", "right_", "value_")
    for nm in names:
        if not hasattr(tree, nm):
            raise Mismatch(f"fit 之后树没有属性 {nm}（docstring 规定的节点数组）", call, show=False)
    classes = as_array(tree.classes_, "classes_", call)
    feature = as_array(tree.feature_, "feature_", call)
    threshold = as_array(tree.threshold_, "threshold_", call, float)
    left, right = as_array(tree.left_, "left_", call), as_array(tree.right_, "right_", call)
    value = as_array(tree.value_, "value_", call, float)
    n = len(feature)
    for nm, a in (("feature_", feature), ("threshold_", threshold), ("left_", left), ("right_", right)):
        expect_true(a.shape == (n,), f"{nm} 的形状应当是 (n_nodes,) = ({n},)，得到 {a.shape}", call, a)
    expect_true(value.shape == (n, len(classes)),
                f"value_ 的形状应当是 (n_nodes, n_classes) = ({n}, {len(classes)})，得到 {value.shape}", call, value)
    return classes, feature, threshold, left, right, value


def grow(sol, X, y, k, n_min, seed, call):
    tree = sol.ExtraTree(max_features=k, min_samples_split=n_min, seed=seed)
    got = tree.fit(X.copy(), y.copy())
    expect_true(got is tree, "fit 应当返回 self", call, got)
    return tree


def check_grown(sol, X, y, k, n_min, seed, call):
    """Grow a tree and check every node against the docstring; returns the tree and its arrays."""
    tree = grow(sol, X, y, k, n_min, seed, call)
    arrays = arrays_of(tree, call)
    problem = ref.check_tree(arrays, X, y, n_min)
    if problem:
        raise Mismatch(problem, call, show=False)
    return tree, arrays


def data_code(seed, lines):
    return f"rng = np.random.default_rng({seed})\n" + "\n".join(lines)


# ---------------------------------------------------------------- Part 1: the statement's test_trees.py

@case("Part 1", sample=True, title="test_split_scores")
def p1_test_split_scores(sol):
    call = "entropy([[2, 2], [4, 0], [0, 0], [1, 3]])"
    expect_close(sol.entropy([[2, 2], [4, 0], [0, 0], [1, 3]]), [1.0, 0.0, 0.0, 0.8112781244591328],
                 rtol=1e-5, atol=1e-8, call=call)
    onehot = np.eye(2)[[0, 0, 0, 1, 1, 1]]
    go_left = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [1, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 0]], dtype=bool)
    call = ("onehot = np.eye(2)[[0, 0, 0, 1, 1, 1]]\n"
            f"go_left = {ref.literal(go_left)}\nsplit_scores(go_left, onehot)")
    expect_close(sol.split_scores(go_left.copy(), onehot.copy()), [1.0, 0.4787039714, 0.0817041659, 0.0, 0.0],
                 rtol=1e-5, atol=1e-8, call=call)


MANUAL = dict(classes_=np.array(["a", "b", "c"]), feature_=np.array([0, -1, 1, -1, -1]),
              threshold_=np.array([2.0, np.nan, 5.0, np.nan, np.nan]), left_=np.array([1, -1, 3, -1, -1]),
              right_=np.array([2, -1, 4, -1, -1]),
              value_=np.array([[3.0, 3.0, 4.0], [3.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 3.0, 1.0], [0.0, 0.0, 3.0]]))
MANUAL_CODE = ("tree = ExtraTree(max_features=1)\n"
               "tree.classes_ = np.array(['a', 'b', 'c'])\n"
               "tree.feature_ = np.array([0, -1, 1, -1, -1])\n"
               "tree.threshold_ = np.array([2.0, np.nan, 5.0, np.nan, np.nan])\n"
               "tree.left_ = np.array([1, -1, 3, -1, -1])\n"
               "tree.right_ = np.array([2, -1, 4, -1, -1])\n"
               "tree.value_ = np.array([[3.0, 3.0, 4.0], [3.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 3.0, 1.0], "
               "[0.0, 0.0, 3.0]])\n")


def manual_tree(sol, spec):
    tree = sol.ExtraTree(max_features=1)
    for k, v in spec.items():
        setattr(tree, k, v.copy())
    return tree


@case("Part 1", sample=True, title="test_apply_manual_tree")
def p1_test_apply_manual_tree(sol):
    X = np.array([[1.0, 9.0], [2.0, 4.0], [3.0, 5.0], [2.5, 4.9], [-1.0, 0.0], [7.0, 6.0]])
    setup = MANUAL_CODE + f"X = {ref.literal(X)}\n"
    tree = manual_tree(sol, MANUAL)
    call = setup + "tree.apply(X)"
    got = as_array(tree.apply(X.copy()), "apply 的结果", call)
    expect_true(got.tolist() == [1, 3, 4, 3, 1, 4],
                f"apply 应当返回 [1, 3, 4, 3, 1, 4]，得到 {got.tolist()}（x[f] < threshold 才走左边，等于时走右边）",
                call, got)
    call = setup + "tree.predict(X)"
    got = as_array(tree.predict(X.copy()), "predict 的结果", call)
    expect_true(got.tolist() == ["a", "b", "c", "b", "a", "c"],
                f"predict 应当返回 ['a', 'b', 'c', 'b', 'a', 'c']，得到 {got.tolist()}", call, got)
    call = setup + "tree.predict_proba(X)[:2]"
    got = as_array(tree.predict_proba(X.copy()), "predict_proba 的结果", call, float)
    expect_true(got.ndim == 2 and len(got) == 6, f"predict_proba 的形状应当是 (6, 3)，得到 {got.shape}", call, got)
    expect_close(got[:2], [[1.0, 0.0, 0.0], [0.0, 0.75, 0.25]], rtol=1e-5, atol=1e-8, call=call)


NOISY_CODE = "rng = np.random.default_rng(0)\nX = rng.normal(size=(200, 4))\ny = rng.integers(0, 3, size=200)\n"


@case("Part 1", sample=True, title="test_noisy_labels")
def p1_test_noisy_labels(sol):
    gen = np.random.default_rng(0)
    X = gen.normal(size=(200, 4))
    y = gen.integers(0, 3, size=200)
    for n_min in (2, 5, 20):
        call = NOISY_CODE + f"tree = ExtraTree(max_features=2, min_samples_split={n_min}, seed={n_min}).fit(X, y)"
        tree = grow(sol, X, y, 2, n_min, n_min, call)
        _, feature, _, _, _, value = arrays_of(tree, call)
        is_leaf = feature < 0
        size = value.sum(axis=1)
        is_pure = (value > 0).sum(axis=1) == 1
        bad = np.flatnonzero(~is_leaf & (size < n_min))
        expect_true(bad.size == 0, f"内部节点 {int(bad[0]) if bad.size else 0} 只有 "
                                   f"{int(size[bad[0]]) if bad.size else 0} 个样本，少于 min_samples_split={n_min}，不该分裂",
                    call)
        bad = np.flatnonzero(is_leaf & ~is_pure & (size >= n_min))
        expect_true(bad.size == 0, f"叶子 {int(bad[0]) if bad.size else 0} 有 {int(size[bad[0]]) if bad.size else 0} 个样本"
                                   f"（≥ min_samples_split={n_min}）且标签不纯，应当继续分裂", call)
    call = NOISY_CODE + "len(ExtraTree(max_features=2, min_samples_split=201, seed=0).fit(X, y).feature_)"
    n = len(arrays_of(grow(sol, X, y, 2, 201, 0, call), call)[1])
    expect_true(n == 1, f"200 个样本 < min_samples_split=201，根节点应当是叶子，树应当只有 1 个节点，得到 {n} 个", call)
    call = NOISY_CODE + "len(ExtraTree(max_features=2, min_samples_split=200, seed=0).fit(X, y).feature_)"
    n = len(arrays_of(grow(sol, X, y, 2, 200, 0, call), call)[1])
    expect_true(n >= 3, f"200 个样本正好等于 min_samples_split=200，根节点应当分裂（至少 3 个节点），得到 {n} 个", call)
    call = NOISY_CODE + "full = ExtraTree(max_features=2, seed=1).fit(X, y)\nfull.predict(X)"
    full = grow(sol, X, y, 2, 2, 1, call)
    pred = as_array(full.predict(X.copy()), "predict 的结果", call)
    wrong = np.flatnonzero(pred != y) if pred.shape == y.shape else np.arange(1)
    expect_true(pred.shape == y.shape and wrong.size == 0,
                f"完全长开的树应当把训练集全部预测对（样本两两不同），可有 {wrong.size} 个预测错", call, pred)


@case("Part 1", sample=True, title="test_tree_structure")
def p1_test_tree_structure(sol):
    gen = np.random.default_rng(1)
    X = gen.normal(size=(150, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    call = ("rng = np.random.default_rng(1)\nX = rng.normal(size=(150, 3))\ny = (X[:, 0] + X[:, 1] > 0).astype(int)\n"
            "tree = ExtraTree(max_features=3, seed=0).fit(X, y)")
    tree = grow(sol, X, y, 3, 2, 0, call)
    _, feature, threshold, left, right, value = arrays_of(tree, call)
    n_nodes = len(feature)
    is_leaf = feature < 0
    size = value.sum(axis=1)
    expect_true(value[0].tolist() == np.bincount(y).tolist(),
                f"value_[0] 应当是全部样本的类别数 {np.bincount(y).tolist()}，得到 {value[0].tolist()}", call)
    empty = np.flatnonzero(size < 1)
    expect_true(empty.size == 0, f"节点 {int(empty[0]) if empty.size else 0} 一个训练样本都没有（两个子节点都应当至少有一个样本）",
                call)
    expect_true(n_nodes == 2 * is_leaf.sum() - 1,
                f"二叉树的节点数应当是 2 × 叶子数 - 1 = {2 * is_leaf.sum() - 1}，得到 {n_nodes}", call)
    parents = np.flatnonzero(~is_leaf)
    expect_true(np.array_equal(left[parents], parents + 1), "先序编号：内部节点 i 的左子节点应当是 i + 1", call)
    expect_true(np.array_equal(value[parents], value[left[parents]] + value[right[parents]]),
                "内部节点的 value_ 应当等于两个子节点的 value_ 之和", call)
    expect_true(np.all(np.isnan(threshold) == is_leaf), "threshold_ 应当恰好在叶子处是 nan", call)
    leaves = as_array(tree.apply(X.copy()), "apply 的结果", call, int)
    reached = np.bincount(leaves, minlength=n_nodes)
    expect_true(np.array_equal(reached[is_leaf], size[is_leaf]) and np.all(reached[~is_leaf] == 0),
                "用 apply 把训练集再走一遍，每个叶子收到的样本数应当等于它的 value_ 之和", call)


@case("Part 1", sample=True, cpu=10, title="test_duplicate_points")
def p1_test_duplicate_points(sol):
    # cpu=10: the buggy starter recurses until RecursionError (about 5 s at the sandbox's 20000-frame limit); a
    # shorter limit would show a timeout instead of the RecursionError the statement describes.
    X = np.array([[0.0, 1.0]] * 5 + [[1.0, 0.0]] * 3 + [[1.0, 1.0]] * 2)
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 2])
    Q = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    for seed in range(4):
        call = ("X = np.array([[0.0, 1.0]] * 5 + [[1.0, 0.0]] * 3 + [[1.0, 1.0]] * 2)\n"
                "y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 2])\n"
                f"tree = ExtraTree(max_features=1, seed={seed}).fit(X, y)")
        tree = grow(sol, X, y, 1, 2, seed, call)
        feature = arrays_of(tree, call)[1]
        expect_true((feature < 0).sum() == 3,
                    f"三种不同的点应当各成一个叶子（共 3 个叶子），得到 {(feature < 0).sum()} 个", call)
        call2 = call + "\ntree.predict_proba(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))"
        expect_close(as_array(tree.predict_proba(Q.copy()), "predict_proba 的结果", call2, float),
                     [[0.4, 0.6, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], rtol=1e-5, atol=1e-8, call=call2)
        call2 = call + "\ntree.predict(np.array([[1.0, 1.0]]))"
        got = as_array(tree.predict(Q[2:].copy()), "predict 的结果", call2)
        expect_true(got.tolist() == [1], f"叶子里类别 1 和 2 各一个，并列时取较小的类 1，得到 {got.tolist()}", call2, got)


@case("Part 1", sample=True, title="test_small_features")
def p1_test_small_features(sol):
    x = np.linspace(0.0, 1.0, 40)[:, None]
    bands = np.floor(x[:, 0] * 4).astype(int) % 2
    call = ("x = np.linspace(0.0, 1.0, 40)[:, None]\nbands = np.floor(x[:, 0] * 4).astype(int) % 2\n"
            "tree = ExtraTree(max_features=1, seed=0).fit(x, bands)\ntree.predict(x)")
    tree = grow(sol, x, bands, 1, 2, 0, call)
    pred = as_array(tree.predict(x.copy()), "predict 的结果", call)
    expect_true(pred.shape == bands.shape and np.array_equal(pred, bands),
                "40 个点的条带数据：完全长开的树应当把训练集全部预测对", call, pred)
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
    parity = corners.sum(axis=1).astype(int) % 2
    for seed in range(10):
        call = ("corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)\n"
                "parity = corners.sum(axis=1).astype(int) % 2\n"
                f"tree = ExtraTree(max_features=3, seed={seed}).fit(corners, parity)")
        tree = grow(sol, corners, parity, 3, 2, seed, call)
        pred = as_array(tree.predict(corners.copy()), "predict 的结果", call)
        expect_true(pred.shape == parity.shape and np.array_equal(pred, parity),
                    "3 维奇偶数据：树应当把 8 个角全部预测对", call, pred)
        n = len(arrays_of(tree, call)[1])
        expect_true(n == 15, f"3 维奇偶数据要切满 3 层，树应当有 15 个节点，得到 {n} 个", call)


@case("Part 1", sample=True, title="test_ensemble")
def p1_test_ensemble(sol):
    X, y = ref.make_grid_data(250, seed=0)
    X_test, y_test = ref.make_grid_data(400, seed=1)
    setup = ("X, y = make_grid_data(250, seed=0)\nX_test, y_test = make_grid_data(400, seed=1)\n"
             "forest = ExtraTreesClassifier(n_estimators=25, seed=0).fit(X, y)\n")
    call = setup + "# make_grid_data 是 test_trees.py 里的函数"
    forest = sol.ExtraTreesClassifier(n_estimators=25, seed=0).fit(X.copy(), y.copy())
    classes = as_array(forest.classes_, "classes_", call)
    expect_true(classes.tolist() == [3, 7, 11], f"forest.classes_ 应当是 [3, 7, 11]，得到 {classes.tolist()}", call)
    expect_true(len(forest.trees_) == 25, f"应当有 25 棵树，得到 {len(forest.trees_)}", call)
    call = setup + "forest.predict_proba(X_test)"
    proba = as_array(forest.predict_proba(X_test.copy()), "predict_proba 的结果", call, float)
    expect_true(proba.shape == (400, 3), f"predict_proba 的形状应当是 (400, 3)，得到 {proba.shape}", call)
    row = int(np.argmax(np.abs(proba.sum(axis=1) - 1)))
    expect_true(np.allclose(proba.sum(axis=1), 1.0),
                f"每棵树一票，每行之和应当是 1，第 {row} 行是 {proba[row].sum():.6g}", call, proba[row])
    votes = np.stack([as_array(t.predict(X_test.copy()), "predict 的结果", call) for t in forest.trees_])
    expect_true(np.allclose(proba[:, 2], (votes == 11).mean(axis=0)),
                "proba[:, 2] 应当等于投给类别 11 的树所占的比例", call)
    call = setup + "forest.predict(X)"
    pred = as_array(forest.predict(X.copy()), "predict 的结果", call)
    expect_true(pred.shape == y.shape and np.array_equal(pred, y), "集成应当把训练集全部预测对", call, pred)
    call = setup + "np.mean(forest.predict(X_test) == y_test)"
    acc = float(np.mean(as_array(forest.predict(X_test.copy()), "predict 的结果", call) == y_test))
    expect_true(acc > 0.85, f"测试集准确率应当高于 0.85，得到 {acc:.4f}", call, acc)


@case("Part 1", sample=True, title="test_reproducible")
def p1_test_reproducible(sol):
    gen = np.random.default_rng(2)
    X = gen.normal(size=(60, 5))
    y = (X[:, 0] > 0).astype(int)
    call = ("rng = np.random.default_rng(2)\nX = rng.normal(size=(60, 5))\ny = (X[:, 0] > 0).astype(int)\n"
            "a = ExtraTreesClassifier(n_estimators=4, seed=7).fit(X, y)\n"
            "b = ExtraTreesClassifier(n_estimators=4, seed=7).fit(X, y)")
    a = sol.ExtraTreesClassifier(n_estimators=4, seed=7).fit(X.copy(), y.copy())
    b = sol.ExtraTreesClassifier(n_estimators=4, seed=7).fit(X.copy(), y.copy())
    for s, t in zip(a.trees_, b.trees_):
        expect_true(np.array_equal(np.asarray(s.feature_), np.asarray(t.feature_))
                    and np.array_equal(np.asarray(s.threshold_, dtype=float), np.asarray(t.threshold_, dtype=float),
                                       equal_nan=True), "同一个 seed 训练两次，树应当完全相同", call)
    roots = {float(t.threshold_[0]) for t in a.trees_}
    expect_true(len(roots) == 4, f"4 棵树的随机数流应当各不相同，可根节点的切点只有 {len(roots)} 种", call)
    expect_true(a.trees_[0].max_features == 2,
                f"5 个特征时 max_features 默认是 round(sqrt(5)) = 2，得到 {a.trees_[0].max_features}", call)


# ---------------------------------------------------------------- Part 1: each docstring rule on new inputs

@case("Part 1", title="样本数等于 min_samples_split 时要分裂")
def p1_split_at_n_min(sol):
    # Rule (a): a node becomes a leaf only if |S| < min_samples_split. With exactly n_min impure samples whose
    # features all vary, the root must split, whatever the random draws.
    seed, gen = npgen()
    for n_min in (2, 3, 6, 25):
        while True:
            X = np.round(gen.normal(size=(n_min, 3)), 2)
            y = gen.integers(0, 2, size=n_min)
            if len(set(y.tolist())) == 2 and np.all(X.min(axis=0) < X.max(axis=0)):
                break
        call = (f"X = {ref.literal(X)}\ny = {ref.literal(y)}\n"
                f"tree = ExtraTree(max_features=2, min_samples_split={n_min}, seed=0).fit(X, y)")
        check_grown(sol, X, y, 2, n_min, 0, call)
    for n_min in (4, 9):
        X = np.round(gen.normal(size=(150, 3)), 2)
        y = gen.integers(0, 3, size=150)
        call = data_code(seed, ["…（前面的抽样省略）", "X = np.round(rng.normal(size=(150, 3)), 2)",
                                "y = rng.integers(0, 3, size=150)",
                                f"tree = ExtraTree(max_features=2, min_samples_split={n_min}, seed={n_min}).fit(X, y)"])
        check_grown(sol, X, y, 2, n_min, n_min, call)


@case("Part 1", title="切点从当前节点的取值范围里抽")
def p1_cut_in_node_range(sol):
    # Step 2: the cut is drawn from [min_S f, max_S f) of the node's own samples, so both children are non-empty.
    x = np.arange(12, dtype=float)[:, None]
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    for seed in range(8):
        call = (f"x = np.arange(12, dtype=float)[:, None]\ny = {ref.literal(y)}\n"
                f"tree = ExtraTree(max_features=1, seed={seed}).fit(x, y)")
        check_grown(sol, x, y, 1, 2, seed, call)
    seed, gen = npgen()
    X = gen.integers(0, 5, size=(80, 2)).astype(float)
    y = ((X[:, 0] + X[:, 1]) % 3 == 0).astype(int)
    call = data_code(seed, ["X = rng.integers(0, 5, size=(80, 2)).astype(float)",
                            "y = ((X[:, 0] + X[:, 1]) % 3 == 0).astype(int)",
                            "tree = ExtraTree(max_features=2, seed=1).fit(X, y)"])
    check_grown(sol, X, y, 2, 2, 1, call)


@case("Part 1", title="部分特征是常量时照样分裂")
def p1_some_constant_features(sol):
    # Rule (c) makes a leaf only when every feature is constant; a constant column at the root is not enough.
    seed, gen = npgen()
    X = np.column_stack([np.full(60, 1.0), np.round(gen.normal(size=(60, 2)), 2)])
    y = (X[:, 1] > 0).astype(int)
    if len(set(y.tolist())) < 2:
        y[0] = 1 - y[0]
    for k in (1, 2):
        call = data_code(seed, ["X = np.column_stack([np.full(60, 1.0), np.round(rng.normal(size=(60, 2)), 2)])",
                                "y = (X[:, 1] > 0).astype(int)   # 第 0 列是常量",
                                f"tree = ExtraTree(max_features={k}, seed={k}).fit(X, y)"])
        check_grown(sol, X, y, k, 2, k, call)
    X, y = ref.make_grid_data(120, seed=seed % 1000)
    call = (f"X, y = make_grid_data(120, seed={seed % 1000})   # test_trees.py 里的函数\n"
            "tree = ExtraTree(max_features=2, seed=3).fit(X, y)")
    check_grown(sol, X, y, 2, 2, 3, call)


@case("Part 1", title="非常量特征比 max_features 少")
def p1_fewer_features_than_k(sol):
    # Step 1: k = min(max_features, |A|). Here two of four columns are constant, so the root has |A| = 2 < 4.
    seed, gen = npgen()
    X = np.column_stack([np.full(40, 3.0), np.round(gen.normal(size=40), 2), np.full(40, -1.0),
                         gen.integers(0, 3, size=40).astype(float)])
    y = ((X[:, 1] > 0) ^ (X[:, 3] == 1)).astype(int)
    call = data_code(seed, ["X = np.column_stack([np.full(40, 3.0), np.round(rng.normal(size=40), 2), "
                            "np.full(40, -1.0), rng.integers(0, 3, size=40).astype(float)])",
                            "y = ((X[:, 1] > 0) ^ (X[:, 3] == 1)).astype(int)",
                            "tree = ExtraTree(max_features=4, seed=0).fit(X, y)"])
    check_grown(sol, X, y, 4, 2, 0, call)
    corners = np.array([[a, b, c, e] for a in (0, 1) for b in (0, 1) for c in (0, 1) for e in (0, 1)], dtype=float)
    parity = corners.sum(axis=1).astype(int) % 2
    for s in range(3):
        call = ("corners = np.array([[a, b, c, e] for a in (0, 1) for b in (0, 1) for c in (0, 1) for e in (0, 1)], "
                "dtype=float)\nparity = corners.sum(axis=1).astype(int) % 2\n"
                f"tree = ExtraTree(max_features=4, seed={s}).fit(corners, parity)")
        check_grown(sol, corners, parity, 4, 2, s, call)


@case("Part 1", title="随机数据：每个节点都符合 docstring")
def p1_random_trees(sol):
    seed, gen = npgen()
    for trial in range(10):
        n, d = int(gen.integers(1, 120)), int(gen.integers(1, 6))
        if trial % 2:
            X = gen.integers(0, 3, size=(n, d)).astype(float)
            how = f"X = rng.integers(0, 3, size=({n}, {d})).astype(float)"
        else:
            X = np.round(gen.normal(size=(n, d)), 1)
            how = f"X = np.round(rng.normal(size=({n}, {d})), 1)"
        if trial % 3 == 2:
            y = np.array(["cat", "ant", "bee"])[gen.integers(0, 3, size=n)]
            yhow = f"y = np.array(['cat', 'ant', 'bee'])[rng.integers(0, 3, size={n})]"
        else:
            y = gen.integers(0, 3, size=n) * 10 - 5
            yhow = f"y = rng.integers(0, 3, size={n}) * 10 - 5"
        k, n_min = int(gen.integers(1, d + 1)), int(gen.choice([2, 3, 7]))
        call = data_code(seed, [f"…（第 {trial + 1} 组数据，前面的抽样省略）" if trial else "", how, yhow,
                                f"tree = ExtraTree(max_features={k}, min_samples_split={n_min}, seed={trial}).fit(X, y)"])
        check_grown(sol, X, y, k, n_min, trial, call.replace("\n\n", "\n"))
    one = np.array([[0.5, -2.0]])
    call = "tree = ExtraTree(max_features=2, seed=0).fit(np.array([[0.5, -2.0]]), np.array([4]))"
    check_grown(sol, one, np.array([4]), 2, 2, 0, call)


@case("Part 1", title="x 正好等于切点时走右边")
def p1_apply_on_threshold(sol):
    seed, gen = npgen()
    X = np.round(gen.normal(size=(60, 3)), 1)
    y = gen.integers(0, 3, size=60)
    call0 = data_code(seed, ["X = np.round(rng.normal(size=(60, 3)), 1)", "y = rng.integers(0, 3, size=60)",
                             "tree = ExtraTree(max_features=2, seed=5).fit(X, y)"])
    tree, (classes, feature, threshold, left, right, value) = check_grown(sol, X, y, 2, 2, 5, call0)
    inner = np.flatnonzero(feature >= 0)
    Q = np.round(gen.normal(size=(len(inner), 3)), 1)
    for i, node in enumerate(inner):
        Q[i, feature[node]] = threshold[node]          # sits exactly on that node's cut
    Q = np.concatenate([Q, X])
    call = call0 + "\nQ = <每个内部节点一行，切分特征正好等于它的切点；后面接上 X>\ntree.apply(Q)"
    got = as_array(tree.apply(Q.copy()), "apply 的结果", call)
    want = np.array([ref.apply_slow(feature, threshold, left, right, q) for q in Q])
    expect_true(got.shape == want.shape, f"apply 的形状应当是 {want.shape}，得到 {got.shape}", call, got)
    bad = np.flatnonzero(got != want)
    if bad.size:
        i = int(bad[0])
        raise Mismatch(f"第 {i} 行 {Q[i].tolist()} 应当到叶子 {want[i]}，得到 {got[i]}：x[f] < threshold 走左边，"
                       f"x[f] == threshold 走右边", call, want[i], got[i])
    leaf_counts = value[want]
    call = call0 + "\nQ = <同上>\ntree.predict_proba(Q)"
    expect_close(as_array(tree.predict_proba(Q.copy()), "predict_proba 的结果", call, float),
                 leaf_counts / leaf_counts.sum(axis=1, keepdims=True), rtol=1e-12, atol=1e-12, call=call)
    call = call0 + "\nQ = <同上>\ntree.predict(Q)"
    got = as_array(tree.predict(Q.copy()), "predict 的结果", call)
    expect_true(np.array_equal(got, classes[leaf_counts.argmax(axis=1)]), "predict 应当返回叶子里最多的类", call, got)


@case("Part 1", title="单棵树 predict：并列时取最小的类")
def p1_tree_predict_ties(sol):
    spec = dict(classes_=np.array([-1, 4, 9]), feature_=np.array([1, 0, -1, -1, -1]),
                threshold_=np.array([0.5, -2.0, np.nan, np.nan, np.nan]), left_=np.array([1, 2, -1, -1, -1]),
                right_=np.array([4, 3, -1, -1, -1]),
                value_=np.array([[5.0, 5.0, 4.0], [3.0, 3.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0], [2.0, 2.0, 3.0]]))
    X = np.array([[-3.0, 0.0], [-2.0, 0.0], [5.0, 0.4999], [0.0, 0.5], [-9.0, 7.0]])
    call = ("tree = ExtraTree(max_features=1)\n" +
            "".join(f"tree.{k} = {ref.literal(v).replace('nan', 'np.nan')}\n" for k, v in spec.items()) +
            f"X = {ref.literal(X)}\n")
    tree = manual_tree(sol, spec)
    got = as_array(tree.apply(X.copy()), "apply 的结果", call + "tree.apply(X)")
    expect_true(got.tolist() == [2, 3, 3, 4, 4], f"apply 应当返回 [2, 3, 3, 4, 4]，得到 {got.tolist()}",
                call + "tree.apply(X)", got)
    got = as_array(tree.predict(X.copy()), "predict 的结果", call + "tree.predict(X)")
    expect_true(got.tolist() == [-1, -1, -1, 9, 9],
                f"predict 应当返回 [-1, -1, -1, 9, 9]（并列时取最小的类），得到 {got.tolist()}", call + "tree.predict(X)", got)
    expect_close(as_array(tree.predict_proba(X.copy()), "predict_proba 的结果", call, float),
                 [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [2 / 7, 2 / 7, 3 / 7], [2 / 7, 2 / 7, 3 / 7]],
                 rtol=1e-12, atol=1e-12, call=call + "tree.predict_proba(X)")


@case("Part 1", title="选得分最高的候选切分")
def p1_best_candidate(sol):
    # Step 3. With max_features = d every feature is a candidate. Feature 3 equals the label, so any cut on it
    # scores 1; the data are redrawn until no cut on another feature scores 1, so the root must use feature 3.
    seed, gen = npgen()
    while True:
        X = np.column_stack([np.round(gen.normal(size=(40, 3)), 2), np.repeat([0.0, 1.0], 20)])
        y = X[:, 3].astype(int)
        if all(ref.best_split_score(X, y, f) < 1 - 1e-9 for f in range(3)):
            break
    for s in range(5):
        call = (data_code(seed, ["X = <40 行：前 3 列是噪声，第 3 列等于标签>", "y = X[:, 3].astype(int)",
                                 f"tree = ExtraTree(max_features=4, seed={s}).fit(X, y)"]))
        _, (_, feature, _, _, _, _) = check_grown(sol, X, y, 4, 2, s, call)
        expect_true(feature[0] == 3 and len(feature) == 3,
                    f"第 3 列等于标签，切它得分是 1，比别的候选都高：根节点应当用特征 3 切成两个纯叶子，"
                    f"得到根节点特征 {int(feature[0])}、{len(feature)} 个节点", call)


@case("Part 1", title="抽 k 个不同的特征：max_features ≥ |A| 时每个非常量特征都是候选")
def p1_distinct_features(sol):
    # Step 1 draws k = min(max_features, |A|) DISTINCT features, so with max_features >= |A| every non-constant
    # feature is a candidate and the root must take the one column that equals the label (score 1; the data are
    # redrawn until no other column can reach 1). Drawing with replacement (choice(..., replace=True)) misses that
    # column with probability (1 - 1/|A|)^max_features, about 1/3 per tree here, so 30 trees per layout catch it.
    seed, gen = npgen()
    for d, const, k in ((6, (), 6), (6, (0, 4), 6)):
        target = int(gen.choice([c for c in range(d) if c not in const]))
        while True:
            X = np.round(gen.normal(size=(40, d)), 2)
            X[:, list(const)] = 7.0
            X[:, target] = np.repeat([0.0, 1.0], 20)
            y = X[:, target].astype(int)
            if all(ref.best_split_score(X, y, f) < 1 - 1e-9 for f in range(d) if f != target):
                break
        what = "" if not const else f"，第 {list(const)} 列是常量（|A| = {d - len(const)}）"
        for s in range(30):
            call = data_code(seed, [f"X = <40 行 {d} 列：第 {target} 列等于标签{what}>", f"y = X[:, {target}].astype(int)",
                                    f"tree = ExtraTree(max_features={k}, seed={s}).fit(X, y)"])
            _, (_, feature, _, _, _, _) = check_grown(sol, X, y, k, 2, s, call)
            expect_true(feature[0] == target,
                        f"max_features={k} ≥ 非常量特征数 {d - len(const)}，k = min(max_features, |A|) 个不同的特征应当把"
                        f"每个非常量特征都抽到（不放回）；第 {target} 列等于标签，切它得分是 1，根节点应当用它，"
                        f"得到根节点特征 {int(feature[0])}", call)


def forest_votes(forest, Q, call):
    """Votes of each tree (ExtraTree.predict, as the docstring says), counted with plain loops."""
    classes = as_array(forest.classes_, "classes_", call).tolist()
    counts = np.zeros((len(Q), len(classes)))
    for t in forest.trees_:
        for i, label in enumerate(as_array(t.predict(Q.copy()), "predict 的结果", call).tolist()):
            counts[i, classes.index(label)] += 1
    return counts


def check_forest_proba(sol, forest, Q, call, M):
    want = forest_votes(forest, Q, call) / M
    got = as_array(forest.predict_proba(Q.copy()), "predict_proba 的结果", call + "\nforest.predict_proba(Q)", float)
    expect_true(got.shape == want.shape, f"predict_proba 的形状应当是 {want.shape}，得到 {got.shape}",
                call + "\nforest.predict_proba(Q)", got)
    bad = np.flatnonzero(np.abs(got - want).max(axis=1) > 1e-12)
    if bad.size:
        i = int(bad[0])
        raise Mismatch(f"第 {i} 行：{M} 棵树各投一票（投给自己 predict 的类），各类得票比例应当是 "
                       f"{np.round(want[i], 6).tolist()}，得到 {np.round(got[i], 6).tolist()}",
                       call + "\nforest.predict_proba(Q)", want[i], got[i])
    classes = as_array(forest.classes_, "classes_", call)
    pred = as_array(forest.predict(Q.copy()), "predict 的结果", call + "\nforest.predict(Q)")
    expect_true(np.array_equal(pred, classes[want.argmax(axis=1)]),
                "predict 应当返回得票最多的类，并列时取最小的类", call + "\nforest.predict(Q)", pred)


@case("Part 1", title="集成：每棵树一票")
def p1_ensemble_votes(sol):
    seed, gen = npgen()
    X = np.round(gen.normal(size=(80, 3)), 2)
    y = np.array([-2, 5, 9])[gen.integers(0, 3, size=80)]          # labels are not 0..2: votes go by class index
    Q = np.concatenate([X[:20], np.round(gen.normal(size=(60, 3)), 2)])
    for M in (4, 7):
        call = data_code(seed, ["X = np.round(rng.normal(size=(80, 3)), 2)",
                                "y = np.array([-2, 5, 9])[rng.integers(0, 3, size=80)]",
                                "Q = np.concatenate([X[:20], np.round(rng.normal(size=(60, 3)), 2)])",
                                f"forest = ExtraTreesClassifier(n_estimators={M}, seed={M}).fit(X, y)"])
        forest = sol.ExtraTreesClassifier(n_estimators=M, seed=M).fit(X.copy(), y.copy())
        check_forest_proba(sol, forest, Q, call, M)


@case("Part 1", title="叶子不纯时也是一棵树一票")
def p1_ensemble_impure_leaves(sol):
    # With min_samples_split = 40 the leaves hold mixed labels, so averaging the trees' predict_proba (soft voting)
    # gives other numbers than the vote fractions the docstring asks for.
    seed, gen = npgen()
    X = np.round(gen.normal(size=(150, 2)), 2)
    y = (X[:, 0] + gen.normal(scale=1.0, size=150) > 0).astype(int) + (X[:, 1] > 0.5)
    Q = np.round(gen.normal(size=(80, 2)), 2)
    call = data_code(seed, ["X = np.round(rng.normal(size=(150, 2)), 2)",
                            "y = (X[:, 0] + rng.normal(scale=1.0, size=150) > 0).astype(int) + (X[:, 1] > 0.5)",
                            "Q = np.round(rng.normal(size=(80, 2)), 2)",
                            "forest = ExtraTreesClassifier(n_estimators=6, min_samples_split=40, seed=1).fit(X, y)"])
    forest = sol.ExtraTreesClassifier(n_estimators=6, min_samples_split=40, seed=1).fit(X.copy(), y.copy())
    for t in forest.trees_:
        problem = ref.check_tree(arrays_of(t, call), X, y, 40)
        if problem:
            raise Mismatch("forest.trees_ 里的一棵树：" + problem, call, show=False)
    check_forest_proba(sol, forest, Q, call, 6)


@case("Part 1", title="集成：参数传给每棵树")
def p1_ensemble_params(sol):
    seed, gen = npgen()
    for d in (1, 2, 3, 7, 12):
        X = np.round(gen.normal(size=(50, d)), 2)
        y = (X.sum(axis=1) > 0).astype(int)
        call = data_code(seed, ["…（前面的抽样省略）", f"X = np.round(rng.normal(size=(50, {d})), 2)",
                                "y = (X.sum(axis=1) > 0).astype(int)",
                                f"forest = ExtraTreesClassifier(n_estimators=3, min_samples_split=3, seed={d}).fit(X, y)"])
        forest = sol.ExtraTreesClassifier(n_estimators=3, min_samples_split=3, seed=d).fit(X.copy(), y.copy())
        want_k = max(1, round(np.sqrt(d)))
        expect_true(len(forest.trees_) == 3, f"应当有 3 棵树，得到 {len(forest.trees_)}", call)
        for t in forest.trees_:
            expect_true(t.max_features == want_k,
                        f"{d} 个特征时 max_features 默认是 round(sqrt({d})) = {want_k}，得到 {t.max_features}", call)
            problem = ref.check_tree(arrays_of(t, call), X, y, 3)
            if problem:
                raise Mismatch("forest.trees_ 里的一棵树：" + problem, call, show=False)
        roots = {float(t.threshold_[0]) for t in forest.trees_}
        expect_true(len(roots) == 3, "3 棵树应当用各自独立的随机数流，可有两棵树的根切点相同", call)
    call = "forest = ExtraTreesClassifier(n_estimators=2, max_features=3, seed=0).fit(X, y)   # X 有 12 个特征"
    forest = sol.ExtraTreesClassifier(n_estimators=2, max_features=3, seed=0).fit(X.copy(), y.copy())
    expect_true(all(t.max_features == 3 for t in forest.trees_), "给了 max_features=3，每棵树都应当用 3", call)


# ---------------------------------------------------------------- Part 2: variance_scores

SCORE_TOL = dict(rtol=0, atol=1e-9)     # float64 rounding of a correct answer is about 1e-14 on these inputs


def check_scores(sol, go_left, y, call, tol=SCORE_TOL):
    want = [ref.variance_score_slow(y.tolist(), go_left[:, j].tolist()) for j in range(go_left.shape[1])]
    got = sol.variance_scores(go_left.copy(), y.copy())
    expect_close(got, want, call=call, **tol)


@case("Part 2", sample=True, title="题目例子 [0.5, 0.96, 0.0]")
def p2_example(sol):
    go_left = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)
    y = np.array([1.0, 2.0, 3.0, 10.0])
    call = f"variance_scores({ref.literal(go_left)}, np.array([1.0, 2.0, 3.0, 10.0]))"
    expect_close(sol.variance_scores(go_left.copy(), y.copy()), [0.5, 0.96, 0.0], rtol=0, atol=1e-9, call=call)


@case("Part 2", title="随机输入，对比逐个计算")
def p2_random(sol):
    seed, gen = npgen()
    for _ in range(40):
        n, k = int(gen.integers(1, 16)), int(gen.integers(1, 6))
        y = np.round(gen.normal(size=n) * 100 + 1e3, 3)
        go_left = gen.random((n, k)) < 0.5
        check_scores(sol, go_left, y, f"variance_scores({ref.literal(go_left)}, {ref.literal(y)})")


@case("Part 2", title="某一侧没有样本的候选得 0")
def p2_empty_child(sol):
    y = np.array([3.0, -1.0, 4.0, 1.5, -5.0])
    go_left = np.array([[1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]], dtype=bool)
    check_scores(sol, go_left, y, f"variance_scores({ref.literal(go_left)}, {ref.literal(y)})")
    for go in ([[True, False]], [[False]]):
        go = np.array(go)
        call = f"variance_scores({ref.literal(go)}, np.array([5.0]))   # n = 1"
        expect_close(sol.variance_scores(go.copy(), np.array([5.0])), np.zeros(go.shape[1]), rtol=0, atol=1e-12,
                     call=call)


@case("Part 2", title="目标值全相同：全部为 0")
def p2_constant_targets(sol):
    # Values such as 0.1 are not exact in binary: a computed variance can come out as 1e-34 instead of 0, and then
    # dividing by it gives arbitrary numbers. The docstring says every score is 0 when all targets are equal.
    for value, n in ((0.1, 3), (0.7, 6), (-2.5, 4)):
        go_left = np.zeros((n, 3), dtype=bool)
        go_left[0, 0] = go_left[: n // 2, 1] = True
        go_left[:, 2] = True
        call = f"variance_scores({ref.literal(go_left)}, np.full({n}, {value}))"
        expect_close(sol.variance_scores(go_left.copy(), np.full(n, value)), np.zeros(3), rtol=0, atol=1e-12,
                     call=call)


@case("Part 2", title="均值很大、方差很小：避免抵消误差")
def p2_large_offset(sol):
    # y = 1e8 + N(0, 1): "mean of squares minus square of mean" loses about 16 of float64's 16 digits here and is
    # off by 1e-3 or more; centring first keeps the error near 1e-7 (measured below 7e-8 over 200 draws).
    seed, gen = npgen()
    for _ in range(6):
        n, k = int(gen.integers(6, 30)), int(gen.integers(2, 6))
        y = 1e8 + gen.normal(size=n)
        go_left = gen.random((n, k)) < 0.5
        call = (f"variance_scores({ref.literal(go_left)}, {ref.literal(y)})\n"
                "# y 的均值约 1e8，方差约 1：直接用“平方的均值减均值的平方”会严重抵消，先减去均值再算")
        check_scores(sol, go_left, y, call, tol=dict(rtol=0, atol=1e-5))
