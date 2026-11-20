import math

import numpy as np

from judgelib import Mismatch, case, expect_close, expect_true, rng
import ref


def npgen(salt=""):
    seed = rng(salt).getrandbits(32)
    return seed, np.random.default_rng(seed)


def arr(x, label, call, dtype=None):
    try:
        return np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)
    except Exception:
        raise Mismatch(f"{label} 不能转成数组：{type(x).__name__}", call, None, x) from None


def fit(sol, X, call, **kw):
    forest = sol.IsolationForest(**kw)
    got = forest.fit(X.copy())
    expect_true(got is forest, "fit 应当返回 self", call, got)
    return forest


def check_spec(sol, X, call, n_trees, max_samples, seed):
    forest = fit(sol, X, call, n_trees=n_trees, max_samples=max_samples, seed=seed)
    problem = ref.check_forest(forest, X, n_trees, max_samples)
    if problem:
        raise Mismatch(problem, call, show=False)
    return forest


def data_code(seed, lines):
    return f"rng = np.random.default_rng({seed})\n" + "\n".join(lines)


# ---------------------------------------------------------------- Part 1: the statement's test_iforest.py

PAGE_TOL = dict(rtol=1e-5, atol=1e-8)       # np.allclose, as test_iforest.py uses


@case("Part 1", sample=True, title="test_c_factor")
def p1_test_c_factor(sol):
    call = "c_factor(np.array([0, 1, 2, 3, 4, 256]))"
    expect_close(sol.c_factor(np.array([0, 1, 2, 3, 4, 256])),
                 [0.0, 0.0, 1.0, 1.2073923576, 1.8516559071, 10.2447709201], call=call, **PAGE_TOL)
    call = "c_factor(np.array([[2, 5], [9, 1]])).shape"
    shape = arr(sol.c_factor(np.array([[2, 5], [9, 1]])), "c_factor 的结果", call).shape
    expect_true(shape == (2, 2), f"输出形状应当和输入一样是 (2, 2)，得到 {shape}", call)
    call = "float(c_factor(100))"
    expect_close(float(sol.c_factor(100)), 8.3646710301, call=call, **PAGE_TOL)
    call = "np.diff(c_factor(np.arange(2, 300)))"
    c = arr(sol.c_factor(np.arange(2, 300)), "c_factor 的结果", call, float)
    expect_true(c.shape == (298,) and np.all(np.diff(c) > 0), "c(n) 在 n >= 2 上应当严格递增", call, c)


SUB_CODE = ("rng = np.random.default_rng(0)\nX = rng.normal(size=(300, 2))\n"
            "forest = IsolationForest(n_trees=40, max_samples=64, seed=0).fit(X)\n")


@case("Part 1", sample=True, title="test_subsamples")
def p1_test_subsamples(sol):
    X = np.random.default_rng(0).normal(size=(300, 2))
    forest = fit(sol, X, SUB_CODE, n_trees=40, max_samples=64, seed=0)
    call = SUB_CODE + "forest.max_samples_, forest.height_limit_"
    expect_true(forest.max_samples_ == 64 and forest.height_limit_ == 6,
                f"应当是 max_samples_ = 64、height_limit_ = 6，得到 {forest.max_samples_}、{forest.height_limit_}", call)
    call = SUB_CODE + "forest.samples_"
    s = arr(forest.samples_, "samples_", call)
    expect_true(s.shape == (40, 64), f"samples_ 的形状应当是 (40, 64)，得到 {s.shape}", call)
    expect_true(s.min() >= 0 and s.max() < 300, f"行号应当在 0 到 299 之间，得到 {s.min()} 到 {s.max()}", call)
    dup = [t for t, row in enumerate(s) if len(np.unique(row)) != 64]
    expect_true(not dup, f"第 {dup[0] if dup else 0} 棵树的子样本有重复的行号（应当不放回地抽）", call)
    used = len(np.unique(s))
    expect_true(used > 270, f"40 棵树合起来只用到了 {used} 个不同的行（应当超过 270 个：每棵树独立地从全部 300 行里抽）",
                call)
    expect_true(len({tuple(row) for row in s.tolist()}) == 40, "40 棵树的子样本应当各不相同", call)


@case("Part 1", sample=True, title="test_tree_structure")
def p1_test_tree_structure(sol):
    X = np.random.default_rng(1).normal(size=(500, 3))
    call = ("rng = np.random.default_rng(1)\nX = rng.normal(size=(500, 3))\n"
            "forest = IsolationForest(n_trees=20, max_samples=100, seed=1).fit(X)")
    forest = fit(sol, X, call, n_trees=20, max_samples=100, seed=1)
    limit = forest.height_limit_
    expect_true(limit == 7, f"height_limit_ 应当是 ceil(log2(100)) = 7，得到 {limit}", call)
    feature, threshold, size = (arr(forest.feature_, "feature_", call), arr(forest.threshold_, "threshold_", call),
                                arr(forest.size_, "size_", call))
    expect_true(feature.shape == threshold.shape == size.shape == (20, 255),
                f"三个数组的形状都应当是 (20, 255)，得到 {feature.shape}、{threshold.shape}、{size.shape}", call)
    expect_true(np.all(size[:, 0] == 100), f"根节点的 size_ 应当都是 100，得到 {size[:, 0].tolist()}", call)
    internal = feature >= 0
    expect_true(not internal[:, 2 ** limit - 1:].any(), "深度为 l 的节点必须是叶子", call)
    t, i = np.nonzero(internal)
    left, right = size[t, 2 * i + 1], size[t, 2 * i + 2]
    bad = np.flatnonzero((left < 1) | (right < 1))
    expect_true(bad.size == 0, (f"第 {t[bad[0]]} 棵树的内部节点 {i[bad[0]]} 有一个子节点 size_ 是 0（左 {left[bad[0]]}，"
                                f"右 {right[bad[0]]}）：每个到达的子样本行都要记进 size_" if bad.size else ""), call)
    expect_true(np.array_equal(size[t, i], left + right), "内部节点的 size_ 应当等于两个子节点之和", call)
    is_leaf = ~internal & (size > 0)
    expect_true(np.all((size * is_leaf).sum(axis=1) == 100), "每棵树所有叶子的 size_ 加起来应当是 100", call)
    expect_true(np.all(threshold[~internal] == 0.0), "没有切分的节点 threshold_ 应当是 0.0", call)


MANUAL_CODE = ("forest = IsolationForest(n_trees=2, max_samples=4)\n"
               "forest.max_samples_ = 4\nforest.height_limit_ = 2\n"
               "forest.feature_ = np.array([[0, -1, 1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]])\n"
               "forest.threshold_ = np.array([[0.5, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])\n"
               "forest.size_ = np.array([[4, 1, 3, 0, 0, 2, 1], [4, 0, 0, 0, 0, 0, 0]])\n"
               "X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 3.0], [0.5, 2.0]])\n")


def manual_forest(sol):
    forest = sol.IsolationForest(n_trees=2, max_samples=4)
    forest.max_samples_ = 4
    forest.height_limit_ = 2
    forest.feature_ = np.array([[0, -1, 1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]])
    forest.threshold_ = np.array([[0.5, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    forest.size_ = np.array([[4, 1, 3, 0, 0, 2, 1], [4, 0, 0, 0, 0, 0, 0]])
    return forest


@case("Part 1", sample=True, title="test_path_length_manual")
def p1_test_path_length_manual(sol):
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 3.0], [0.5, 2.0]])
    c4 = 1.8516559071
    call = MANUAL_CODE + "forest.path_length(X)"
    expect_close(manual_forest(sol).path_length(X.copy()), [[1.0, 3.0, 2.0, 2.0], [c4, c4, c4, c4]], call=call,
                 **PAGE_TOL)
    call = MANUAL_CODE + "forest.score(X)"
    expect_close(manual_forest(sol).score(X.copy()), [0.5864059, 0.4032969, 0.4863082, 0.4863082], call=call,
                 **PAGE_TOL)


@case("Part 1", sample=True, title="test_constant_data")
def p1_test_constant_data(sol):
    X = np.full((40, 3), 2.5)
    setup = "X = np.full((40, 3), 2.5)\nforest = IsolationForest(n_trees=10, max_samples=16, seed=0).fit(X)\n"
    forest = fit(sol, X, setup, n_trees=10, max_samples=16, seed=0)
    feature = arr(forest.feature_, "feature_", setup)
    expect_true(np.all(feature == -1), "数据全是常量，每棵树的根节点都是叶子，feature_ 应当全是 -1", setup)
    size = arr(forest.size_, "size_", setup)
    expect_true(np.all(size[:, 0] == 16), f"根节点的 size_ 应当是 16，得到 {size[:, 0].tolist()}", setup)
    expect_close(forest.score(X.copy()), np.full(40, 0.5), call=setup + "forest.score(X)", **PAGE_TOL)
    expect_close(forest.score(np.array([[0.0, 9.0, -3.0]])), [0.5],
                 call=setup + "forest.score(np.array([[0.0, 9.0, -3.0]]))", **PAGE_TOL)
    setup = "X = np.full((40, 3), 2.5)\nforest = IsolationForest(n_trees=10, seed=0).fit(X)\n"
    forest = fit(sol, X, setup, n_trees=10, seed=0)
    expect_true(forest.max_samples_ == 40, f"max_samples_ 应当是 min(256, 40) = 40，得到 {forest.max_samples_}", setup)
    expect_close(forest.score(X.copy()), np.full(40, 0.5), call=setup + "forest.score(X)", **PAGE_TOL)


@case("Part 1", sample=True, title="test_few_points")
def p1_test_few_points(sol):
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [50.0]])
    setup = ("X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [50.0]])\n"
             "forest = IsolationForest(n_trees=200, seed=0).fit(X)\n")
    forest = fit(sol, X, setup, n_trees=200, seed=0)
    expect_true(forest.max_samples_ == 8 and forest.height_limit_ == 3,
                f"应当是 max_samples_ = 8、height_limit_ = 3，得到 {forest.max_samples_}、{forest.height_limit_}", setup)
    shape = arr(forest.samples_, "samples_", setup).shape
    expect_true(shape == (200, 8), f"samples_ 的形状应当是 (200, 8)，得到 {shape}", setup)
    call = setup + "forest.score(X)"
    s = arr(forest.score(X.copy()), "score 的结果", call, float)
    expect_true(s.shape == (8,), f"score 的形状应当是 (8,)，得到 {s.shape}", call, s)
    expect_true(np.all((s > 0) & (s <= 1)), "分数应当在 (0, 1] 里", call, s)
    expect_true(s.argmax() == 7, "50.0 是离群点，分数应当最高", call, s)
    expect_true(s[7] > 0.7, f"离群点 50.0 的分数应当大于 0.7，得到 {s[7]:.4f}", call, s)
    expect_true(s[:7].max() < 0.65, f"其余 7 个点的最高分应当小于 0.65，得到 {s[:7].max():.4f}", call, s)


BLOB_CODE = ("def make_blob_with_outliers():\n"
             "    rng = np.random.default_rng(3)\n"
             "    inliers = rng.normal(size=(400, 2))\n"
             "    outliers = rng.uniform(5.0, 8.0, size=(10, 2)) * rng.choice([-1.0, 1.0], size=(10, 2))\n"
             "    return np.concatenate([inliers, outliers])\n"
             "X = make_blob_with_outliers()\n"
             "scores = IsolationForest(n_trees=60, max_samples=64, seed=0).fit(X).score(X)")


def blob():
    gen = np.random.default_rng(3)
    inliers = gen.normal(size=(400, 2))
    outliers = gen.uniform(5.0, 8.0, size=(10, 2)) * gen.choice([-1.0, 1.0], size=(10, 2))
    return np.concatenate([inliers, outliers])


def blob_scores(sol):
    X = blob()
    s = arr(fit(sol, X, BLOB_CODE, n_trees=60, max_samples=64, seed=0).score(X.copy()), "score 的结果", BLOB_CODE,
            float)
    expect_true(s.shape == (410,), f"score 的形状应当是 (410,)，得到 {s.shape}", BLOB_CODE, s)
    return s


@case("Part 1", sample=True, title="test_ranks_outliers_first")
def p1_test_ranks_outliers_first(sol):
    s = blob_scores(sol)
    frac = float(np.mean(s[:400] < s[400:].min()))
    expect_true(frac > 0.97, f"应当有超过 97% 的内点分数低于所有离群点，得到 {frac:.3f}", BLOB_CODE)


@case("Part 1", sample=True, title="test_score_scale")
def p1_test_score_scale(sol):
    s = blob_scores(sol)
    expect_true(np.all((s > 0) & (s <= 1)), "分数应当在 (0, 1] 里", BLOB_CODE, s)
    expect_true(s[400:].min() > 0.6, f"离群点的最低分应当大于 0.6，得到 {s[400:].min():.4f}", BLOB_CODE)
    med = float(np.median(s[:400]))
    expect_true(0.35 < med < 0.53, f"内点分数的中位数应当在 0.35 到 0.53 之间，得到 {med:.4f}", BLOB_CODE)


@case("Part 1", sample=True, title="test_reproducible")
def p1_test_reproducible(sol):
    X = np.random.default_rng(4).normal(size=(120, 4))
    call = ("rng = np.random.default_rng(4)\nX = rng.normal(size=(120, 4))\n"
            "a = IsolationForest(n_trees=8, max_samples=32, seed=5).fit(X)\n"
            "b = IsolationForest(n_trees=8, max_samples=32, seed=5).fit(X)\n"
            "c = IsolationForest(n_trees=8, max_samples=32, seed=6).fit(X)")
    a = fit(sol, X, call, n_trees=8, max_samples=32, seed=5)
    b = fit(sol, X, call, n_trees=8, max_samples=32, seed=5)
    c = fit(sol, X, call, n_trees=8, max_samples=32, seed=6)
    for name in ("samples_", "feature_", "threshold_"):
        expect_true(np.array_equal(arr(getattr(a, name), name, call), arr(getattr(b, name), name, call)),
                    f"同一个 seed 训练两次，{name} 应当完全相同", call)
    expect_true(not np.array_equal(arr(a.threshold_, "threshold_", call), arr(c.threshold_, "threshold_", call)),
                "不同的 seed 应当长出不同的树", call)
    expect_true(np.array_equal(arr(a.score(X.copy()), "score", call), arr(b.score(X.copy()), "score", call)),
                "同一个 seed 的两个森林分数应当完全相同", call)


# ---------------------------------------------------------------- Part 1: each docstring rule on new inputs

C_TOL = dict(rtol=1e-9, atol=1e-9)     # same formula; allows np.euler_gamma (differs from 0.5772156649 by 2e-12)


@case("Part 1", title="c_factor：逐个对照公式")
def p1_c_factor_values(sol):
    n = np.concatenate([np.arange(0, 301), [1000, 65536, 10 ** 6]])
    want = [ref.c_slow(k) for k in n]
    call = "c_factor(np.concatenate([np.arange(0, 301), [1000, 65536, 10 ** 6]]))"
    got = sol.c_factor(n.copy())
    expect_close(got, want, call=call, **C_TOL)
    kind = arr(got, "c_factor 的结果", call).dtype.kind
    expect_true(kind == "f", f"docstring 说返回浮点数组，得到的 dtype 是 {arr(got, '', call).dtype}", call)
    seed, gen = npgen()
    cube = gen.integers(0, 40, size=(2, 3, 4))
    call = f"c_factor({ref.literal(cube)})   # 形状 (2, 3, 4)"
    expect_close(sol.c_factor(cube.copy()), np.vectorize(ref.c_slow)(cube), call=call, **C_TOL)
    for k in (2, 3, 7):
        call = f"c_factor({k})"
        got = arr(sol.c_factor(k), "c_factor 的结果", call, float)
        expect_true(got.shape == (), f"输入是一个整数，输出应当是形状 () 的数，得到形状 {got.shape}", call, got)
        expect_close(got, ref.c_slow(k), call=call, **C_TOL)


@case("Part 1", title="子样本：不放回、各树独立、均匀")
def p1_subsamples(sol):
    X = np.arange(20.0).reshape(10, 2)
    call = "X = np.arange(20.0).reshape(10, 2)\nforest = IsolationForest(n_trees=30, seed=1).fit(X)   # 树比行多"
    forest = check_spec(sol, X, call, 30, 256, 1)
    seed, gen = npgen()
    X = np.round(gen.normal(size=(100, 2)), 3)
    call = data_code(seed, ["X = np.round(rng.normal(size=(100, 2)), 3)",
                            "forest = IsolationForest(n_trees=5, max_samples=32, seed=2).fit(X)   # 树比子样本少"])
    forest = check_spec(sol, X, call, 5, 32, 2)
    sets = {frozenset(row) for row in np.asarray(forest.samples_).tolist()}
    expect_true(len(sets) == 5, "5 棵树应当各自独立地抽子样本，可有几棵树的子样本完全相同", call)
    # Each of 10 rows is in a subsample of 3 with probability 0.3: over 1000 trees its count is 300 +- 14.5 (sd).
    # A band of +-90 (6.2 sd) fails a uniform sampler with probability about 10 * 5e-10.
    X = np.arange(10.0)[:, None]
    call = "X = np.arange(10.0)[:, None]\nforest = IsolationForest(n_trees=1000, max_samples=3, seed=3).fit(X)"
    forest = fit(sol, X, call, n_trees=1000, max_samples=3, seed=3)
    s = arr(forest.samples_, "samples_", call)
    expect_true(s.shape == (1000, 3), f"samples_ 的形状应当是 (1000, 3)，得到 {s.shape}", call)
    expect_true(all(len(set(row)) == 3 for row in s.tolist()), "每棵树的子样本应当是 3 个不同的行号", call)
    counts = np.bincount(s.ravel().astype(int), minlength=10)
    expect_true(len(counts) == 10 and np.all(np.abs(counts - 300) <= 90),
                f"每一行应当以 3/10 的概率被一棵树抽中（1000 棵树里约 300 次），各行被抽中的次数却是 {counts.tolist()}",
                call)


@case("Part 1", title="每棵树都符合 docstring：连续数据")
def p1_trees_continuous(sol):
    for n, d, n_trees, m in ((200, 3, 10, 64), (37, 1, 4, 256), (2, 2, 3, 256), (130, 5, 6, 100)):
        seed, gen = npgen(f"continuous {n}")
        X = np.round(gen.normal(size=(n, d)), 4)
        call = data_code(seed, [f"X = np.round(rng.normal(size=({n}, {d})), 4)",
                                f"forest = IsolationForest(n_trees={n_trees}, max_samples={m}, seed={n}).fit(X)"])
        check_spec(sol, X, call, n_trees, m, n)


@case("Part 1", title="重复点、常量列：叶子大小要记下")
def p1_trees_duplicates(sol):
    seed, gen = npgen("integers")
    X = np.column_stack([gen.integers(0, 3, size=90), np.full(90, 7.0), gen.integers(0, 2, size=90)]).astype(float)
    call = data_code(seed, ["X = np.column_stack([rng.integers(0, 3, size=90), np.full(90, 7.0), "
                            "rng.integers(0, 2, size=90)]).astype(float)",
                            "forest = IsolationForest(n_trees=8, max_samples=32, seed=0).fit(X)"])
    check_spec(sol, X, call, 8, 32, 0)
    X = np.full((12, 2), -1.5)
    call = "X = np.full((12, 2), -1.5)\nforest = IsolationForest(n_trees=3, seed=0).fit(X)"
    check_spec(sol, X, call, 3, 256, 0)
    seed, gen = npgen("rounded")
    X = np.round(gen.normal(size=(300, 2)), 1)
    call = data_code(seed, ["X = np.round(rng.normal(size=(300, 2)), 1)",
                            "forest = IsolationForest(n_trees=5, max_samples=20, seed=1).fit(X)"])
    check_spec(sol, X, call, 5, 20, 1)


def check_paths(sol, forest, Q, setup):
    """Compare path_length(Q) with a node-by-node walk; on a mismatch show the call for the one failing point."""
    want = ref.paths_slow(forest, Q)
    call = setup + f"\nforest.path_length(Q)   # Q 有 {len(Q)} 行：训练行、远处的点、正好落在切分值上的点"
    got = arr(forest.path_length(Q.copy()), "path_length 的结果", call, float)
    expect_true(got.shape == want.shape, f"path_length 的形状应当是 (n_trees, n) = {want.shape}，得到 {got.shape}",
                call, got)
    bad = np.argwhere(~(np.abs(got - want) <= 1e-9))
    if bad.size:
        t, j = (int(v) for v in bad[0])
        edges, leaf = ref.path_slow(np.asarray(forest.feature_), np.asarray(forest.threshold_),
                                    np.asarray(forest.size_), t, Q[j])
        size = int(np.asarray(forest.size_)[t, leaf])
        raise Mismatch(f"第 {t} 棵树：这个点停在叶子 {leaf}（{edges} 条边，叶子大小 {size}），"
                       f"h = {edges} + c({size}) = {want[t, j]:.6g}，得到 {got[t, j]:.6g}",
                       setup + f"\nforest.path_length({ref.literal(Q[j:j + 1])})[{t}, 0]", want[t, j], got[t, j])


def queries(forest, X, gen):
    """Training rows, far-away points, and points sitting exactly on the split value of some internal node."""
    feature, threshold = np.asarray(forest.feature_), np.asarray(forest.threshold_)
    t, i = np.nonzero(feature >= 0)
    pick = gen.permutation(len(t))[:15]
    on_cut = X[gen.integers(0, len(X), size=len(pick))].copy()
    for row, k in enumerate(pick):
        on_cut[row, feature[t[k], i[k]]] = threshold[t[k], i[k]]
    far = gen.normal(size=(5, X.shape[1])) * 10
    return np.concatenate([X[:25], far, on_cut])


@case("Part 1", title="路径长度：逐棵树逐个点对照")
def p1_path_length(sol):
    _, pick = npgen("queries")
    seed, gen = npgen("duplicates")
    X = gen.integers(0, 3, size=(80, 2)).astype(float)
    setup = data_code(seed, ["X = rng.integers(0, 3, size=(80, 2)).astype(float)   # 很多重复点",
                             "forest = IsolationForest(n_trees=6, max_samples=40, seed=0).fit(X)"])
    forest = check_spec(sol, X, setup, 6, 40, 0)
    check_paths(sol, forest, queries(forest, X, pick), setup)
    seed, gen = npgen("continuous")
    X = np.round(gen.normal(size=(150, 3)), 2)
    setup = data_code(seed, ["X = np.round(rng.normal(size=(150, 3)), 2)",
                             "forest = IsolationForest(n_trees=5, max_samples=64, seed=1).fit(X)"])
    forest = check_spec(sol, X, setup, 5, 64, 1)
    check_paths(sol, forest, queries(forest, X, pick), setup)


def check_score(sol, forest, Q, psi, call):
    mean = ref.paths_slow(forest, Q).mean(axis=0)
    want = 2.0 ** (-mean / ref.c_slow(psi))
    expect_close(forest.score(Q.copy()), want, rtol=1e-9, atol=1e-12, call=call)


@case("Part 1", title="分数：用实际的子样本大小 ψ 归一化")
def p1_score(sol):
    # The expected score is 2 ** (-mean / c(psi)) with mean taken from a node-by-node walk of the answer's own
    # trees, so it is the same formula on the same numbers: only float rounding separates them.
    seed, gen = npgen("small")
    X = np.round(gen.normal(size=(50, 2)), 3)
    Q = np.concatenate([X, gen.normal(size=(6, 2)) * 4])
    setup = data_code(seed, ["X = np.round(rng.normal(size=(50, 2)), 3)",
                             "Q = np.concatenate([X, rng.normal(size=(6, 2)) * 4])",
                             "forest = IsolationForest(n_trees=20, seed=4).fit(X)   # 默认 max_samples=256 > 50 行"])
    forest = check_spec(sol, X, setup, 20, 256, 4)
    check_score(sol, forest, Q, 50, setup + "\nforest.score(Q)   # s = 2 ** (-E[h] / c(ψ))，ψ = max_samples_ = 50")
    check_score(sol, forest, Q[-1:], 50, setup + "\nforest.score(Q[-1:])   # 只给一个点打分，ψ 仍是 50")
    seed, gen = npgen("large")
    X = np.round(gen.normal(size=(300, 3)), 3)
    setup = data_code(seed, ["X = np.round(rng.normal(size=(300, 3)), 3)",
                             "forest = IsolationForest(n_trees=15, max_samples=64, seed=5).fit(X)"])
    forest = check_spec(sol, X, setup, 15, 64, 5)
    check_score(sol, forest, X[:40], 64, setup + "\nforest.score(X[:40])   # ψ = 64")


# ---------------------------------------------------------------- Part 2: expected_depth

@case("Part 2", sample=True, title="题目例子 expected_depth(4)")
def p2_example(sol):
    expect_close(sol.expected_depth(4), [0.0, 1.0, 5 / 3, 13 / 6], rtol=1e-9, atol=1e-12, call="expected_depth(4)")


@case("Part 2", title="小 n：对照精确值 2(H_n − 1)")
def p2_small(sol):
    for n_max in (1, 2, 3, 7, 60):
        expect_close(sol.expected_depth(n_max), ref.exact_depth(n_max), rtol=1e-10, atol=1e-12,
                     call=f"expected_depth({n_max})")


@case("Part 2", cpu=2, title="n_max = 200000：要求 O(n_max)")
def p2_large(sol):
    # Measured: the O(n_max) recurrence takes about 0.2 s here; summing k * D(k) afresh for every n (O(n_max^2),
    # even with np.dot) takes about 11 s. Float error of the recurrence or of a cumulative sum is about 1e-14
    # relative, so rtol = 1e-9 leaves a wide margin.
    n_max = 200000
    want = 2.0 * (np.cumsum(1.0 / np.arange(1, n_max + 1)) - 1.0)
    expect_close(sol.expected_depth(n_max), want, rtol=1e-9, atol=1e-12, call=f"expected_depth({n_max})")
