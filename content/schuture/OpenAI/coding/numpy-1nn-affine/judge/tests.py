"""Tests for NumPy 1-NN as an affine layer: predict_1nn (Part 1), affine_layer and predict_1nn_affine (Part 2 (b)).

Part 2 (a) is a proof and is not judged. Labels are checked against ref.nearest_sets: with small integer coordinates
(exact arithmetic) the label of nn(q), the smallest index at the minimum distance, is required; otherwise any index
whose distance is within float64 rounding error of the minimum is accepted, as the statement says. The layer (W, b)
is not unique (any c W_1, c b_1 + t for c > 0 works), so it is checked by what it must do: W has shape (d, n), b has
shape (n,), and the argmax of X_query @ W + b is a nearest training point. probs must be the row-wise softmax of
those logits, within float64 rounding (ref.probs_tolerance).

With integer coordinates the smallest index is required only when the answer's own layer computes X_query @ W + b
without rounding (exact_layer); a layer scaled by, say, 1/3 rounds, and then any exactly tied index is accepted.

Part 1 also forbids Python-level loops. p1_large (n = m = 2000, d = 50) times out a double loop over queries and
training points, and p1_no_loops reads the source of predict_1nn and of the answer's own functions it calls.
"""
import ast
import inspect
import textwrap

import numpy as np

from judgelib import Mismatch, calling, case, expect_close, expect_true, rng
import ref

# The statement's example and its table: nn(q) = 0 (a tie between 0 and 1), 1, 3; the labels are [2, 0, 1].
EX_TRAIN = np.array([[0., 0.], [2., 0.], [0., 2.], [3., 3.]])
EX_Y = np.array([2, 0, 0, 1])
EX_QUERY = np.array([[1., 0.], [2., 1.], [2., 3.]])
EX_NN = [np.array([0]), np.array([1]), np.array([3])]


# ---------------------------------------------------------------- helpers

def copies(X_train, y_train, X_query):
    return X_train.copy(), y_train.copy(), X_query.copy()


def num(v):
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def as_array(got, shape, what, call):
    """got as a float array of the given shape; anything else is a wrong answer."""
    try:
        arr = np.asarray(got, dtype=float)
    except Exception:
        raise Mismatch(f"{what}应是形状为 {shape} 的数值数组，得到的是 {type(got).__name__}", call, None, got) from None
    expect_true(arr.shape == shape, f"{what}的形状应为 {shape}，得到 {arr.shape}", call, got)
    return arr


def pair(got, names, call):
    expect_true(isinstance(got, (tuple, list)) and len(got) == 2, f"应返回二元组 {names}", call, got)
    return got


def explain(j, got_j, X_train, y_train, X_query, sets, exact):
    """Why the label (or index) of query j is wrong, in a sentence."""
    D = ref.row_distances(X_train, X_query[j])
    s = sets[j]
    i = int(s[0])
    if len(s) > 1:
        same = "都是" if np.all(D[s] == D[i]) else "只差在舍入误差以内（约"
        return (f"下标 {s.tolist()} 的平方距离{same} {D[i]:.10g}{'' if same == '都是' else '）'}，"
                f"它们的标签 {sorted({int(v) for v in y_train[s]})} 中任何一个都算对")
    tie = np.flatnonzero(D == D[i])
    if exact and len(tie) > 1:
        return (f"下标 {tie.tolist()[:8]}{'…' if len(tie) > 8 else ''} 的平方距离都是 {D[i]:g}（坐标是小整数，运算精确），"
                f"并列时取最小下标 {i}，它的标签是 {y_train[i]}")
    return f"最近的训练点是下标 {i}（平方距离 {D[i]:.10g}），它的标签是 {y_train[i]}"


def check_labels(labels, X_train, y_train, X_query, sets, exact, call, what):
    m = len(X_query)
    got = as_array(labels, (m,), what, call)
    bad = [j for j in range(m) if not np.any(y_train[sets[j]] == got[j])]
    if not bad:
        return
    j = bad[0]
    D = ref.row_distances(X_train, X_query[j])
    mine = np.flatnonzero(y_train == got[j])
    tail = (f"；标签为 {num(got[j])} 的训练点里最近的是下标 {mine[np.argmin(D[mine])]}，"
            f"平方距离 {D[mine].min():.10g}" if mine.size else f"；y_train 里没有标签 {num(got[j])}")
    more = f"（共 {len(bad)} 个查询点不对）" if len(bad) > 1 else ""
    want = np.array([y_train[s[0]] for s in sets])
    raise Mismatch(f"{what}第 {j} 个元素（查询点 X_query[{j}]）是 {num(got[j])}{more}："
                   f"{explain(j, got[j], X_train, y_train, X_query, sets, exact)}{tail}。期望一栏是一个正确答案。",
                   call, want, labels)


def run_1nn(sol, X_train, y_train, X_query, exact, setup):
    call = calling(f"{setup}\npredict_1nn(X_train, y_train, X_query)")
    sets = ref.nearest_sets(X_train, X_query, exact)
    got = sol.predict_1nn(*copies(X_train, y_train, X_query))
    check_labels(got, X_train, y_train, X_query, sets, exact, call, "返回值")


def get_layer(sol, X_train, call):
    """The answer's (W, b), checked for shape."""
    n, d = X_train.shape
    out = pair(sol.affine_layer(X_train.copy()), "(W, b)", call)
    try:
        W = np.asarray(out[0], dtype=float)
    except Exception:
        raise Mismatch(f"W 应是形状为 ({d}, {n}) 的数值数组，得到的是 {type(out[0]).__name__}", call, None, out) from None
    if W.shape != (d, n):
        hint = "，这是 W_1 的形状，少了转置" if W.shape == (n, d) else ""
        expect_true(False, f"W 的形状应为 (d, n) = ({d}, {n})，得到 {W.shape}{hint}", call, out)
    b = as_array(out[1], (n,), "b ", call)
    return W, b


def exact_layer(X_query, W, b):
    """True when X_query @ W + b involves no rounding, so tied distances give bit-equal logits: integer queries, and
    W, b multiples of 2^-20 small enough that every product and partial sum is a multiple of 2^-20 below 2^52 in
    magnitude. This holds for 2 X^T, -|x|^2 and for any scale such as 1/2 or 3, but not for a scale such as 1/3 or
    0.1, or a shift such as 0.1: then the answer's own layer rounds, and any exactly tied index is accepted."""
    k = 2.0 ** 20
    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(b))):
        return False
    Wk, bk = W * k, b * k
    if not (np.array_equal(X_query, np.round(X_query)) and np.array_equal(Wk, np.round(Wk))
            and np.array_equal(bk, np.round(bk))):
        return False
    return float((np.abs(X_query) @ np.abs(Wk) + np.abs(bk)).max(initial=0.0)) < 2.0 ** 52


def layer_or_none(sol, X_train):
    """The answer's (W, b) as float arrays of the right shapes, or None (used only to pick the tie rule)."""
    n, d = X_train.shape
    try:
        out = sol.affine_layer(X_train.copy())
        W, b = np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)
    except Exception:
        return None
    return (W, b) if W.shape == (d, n) and b.shape == (n,) else None


def run_layer(sol, X_train, y_train, X_query, exact, setup):
    """argmax of X_query @ W + b must be an index a correct answer may return."""
    call = calling(f"{setup}\nW, b = affine_layer(X_train)\nnp.argmax(X_query @ W + b, axis=1)")
    W, b = get_layer(sol, X_train, call)
    exact = exact and exact_layer(X_query, W, b)
    sets = ref.nearest_sets(X_train, X_query, exact)
    nearest = np.argmax(X_query @ W + b, axis=1)
    bad = [j for j in range(len(X_query)) if nearest[j] not in sets[j]]
    if bad:
        j = bad[0]
        D = ref.row_distances(X_train, X_query[j])
        more = f"（共 {len(bad)} 个查询点不对）" if len(bad) > 1 else ""
        n, d = X_train.shape
        hint = ("。你的 W 和 X_train 成正比，是 W_1 的排法；W 应是它的转置，第 i 列对应 x_i（d == n 时少了转置不会报错）"
                if d == n and _looks_untransposed(W, X_train) else "")
        raise Mismatch(f"logits = X_query @ W + b 第 {j} 行的最大值在下标 {nearest[j]}（它的平方距离是 {D[nearest[j]]:.10g}）{more}："
                       f"{explain(j, nearest[j], X_train, y_train, X_query, sets, exact)}{hint}。期望一栏是一组正确的下标。",
                       call, np.array([s[0] for s in sets]), nearest)


def _looks_untransposed(W, X_train):
    """W proportional to X_train itself (the W_1 layout) instead of its transpose; used only to name the mistake."""
    k = np.vdot(X_train, X_train)
    if k == 0:
        return False
    c = np.vdot(W, X_train) / k
    return c > 0 and np.allclose(W, c * X_train) and not np.allclose(W.T, c * X_train)


def probs_hint(probs):
    """A sentence naming the mistake behind a wrong probs, or ''."""
    if not np.all(np.isfinite(probs)):
        return ("probs 里有 nan 或 inf。logits 随 |x|² 增长，np.exp(1000.0) 是 inf，inf / inf 是 nan；"
                "先让每一行减去它的最大值再取 exp，softmax 的值不变。")
    rows, cols = probs.sum(axis=1), probs.sum(axis=0)
    if np.allclose(cols, 1) and not np.allclose(rows, 1):
        return "probs 的每一列之和是 1，每一行却不是：softmax 要沿 axis=1 做，每个查询点（每一行）一个分布。"
    if not np.allclose(rows, 1):
        return "probs 的每一行之和应为 1。"
    return ""


def run_forward(sol, X_train, y_train, X_query, exact, setup):
    """probs = softmax(X_query @ W + b) with the answer's own layer, and acceptable labels."""
    m, n = len(X_query), len(X_train)
    call = calling(f"{setup}\nprobs, labels = predict_1nn_affine(X_train, y_train, X_query)")
    W, b = get_layer(sol, X_train, call)
    exact = exact and exact_layer(X_query, W, b)
    sets = ref.nearest_sets(X_train, X_query, exact)
    want = ref.softmax(X_query @ W + b)
    atol = ref.probs_tolerance(X_query, W, b)
    out = pair(sol.predict_1nn_affine(*copies(X_train, y_train, X_query)), "(probs, labels)", call)
    probs = as_array(out[0], (m, n), "probs ", call)
    try:
        expect_close(probs, want, rtol=0, atol=atol, call=call)
    except Mismatch as e:
        e.msg = "probs 应等于 softmax(X_query @ W + b)，其中 W, b = affine_layer(X_train)：" + (probs_hint(probs) or e.msg)
        e.got = out[0]
        raise
    check_labels(out[1], X_train, y_train, X_query, sets, exact, call, "labels ")


# NumPy functions that call a Python function once per element or per row: loops in disguise.
HIDDEN_LOOPS = {"vectorize": "np.vectorize", "frompyfunc": "np.frompyfunc",
                "apply_along_axis": "np.apply_along_axis（每一行调用一次 Python 函数）"}


def static_loops(sol, fn):
    """(function name, line in answer.py, construct) of the first Python-level loop in fn or in an answer function it
    calls by name, or None. Functions whose source cannot be read are skipped."""
    path = getattr(sol, "__file__", None)
    kinds = {ast.For: "for 循环", ast.AsyncFor: "for 循环", ast.While: "while 循环", ast.ListComp: "列表推导式",
             ast.SetComp: "集合推导式", ast.DictComp: "字典推导式", ast.GeneratorExp: "生成器表达式"}
    seen, todo = set(), [fn]
    while todo:
        f = todo.pop()
        if not inspect.isfunction(f) or f.__code__.co_filename != path or f in seen:
            continue
        seen.add(f)
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        except (OSError, TypeError, SyntaxError):
            continue
        found = []
        for node in ast.walk(tree):
            kind = kinds.get(type(node))
            if isinstance(node, ast.Attribute) and node.attr in HIDDEN_LOOPS:
                kind = HIDDEN_LOOPS[node.attr]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "map" or node.func.id in HIDDEN_LOOPS:
                    kind = HIDDEN_LOOPS.get(node.func.id, "map")
                elif node.func.id in f.__globals__:
                    todo.append(f.__globals__[node.func.id])
            if kind:
                found.append((node.lineno, kind))
        if found:
            line, kind = min(found)
            return f.__name__, f.__code__.co_firstlineno + line - 1, kind
    return None


# ---------------------------------------------------------------- input sets

def fixed_inputs():
    """Hand-made small inputs with integer coordinates: (title, X_train, y_train, X_query)."""
    A = np.array
    return [
        ("n = 1", A([[3., -1.]]), A([4]), A([[0., 0.], [3., -1.], [100., 100.]])),
        ("m = 1、d = 1", A([[-2.], [1.], [4.]]), A([0, 1, 2]), A([[2.]])),
        ("d = 1，正中间并列", A([[-1.], [1.]]), A([1, 0]), A([[0.], [-1.], [1.]])),
        ("d = 1，正中间并列（顺序反过来）", A([[1.], [-1.]]), A([0, 1]), A([[0.], [5.], [-5.]])),
        ("重复的训练点，标签不同", A([[1., 1.], [0., 0.], [1., 1.], [0., 0.]]), A([0, 1, 2, 3]),
         A([[1., 1.], [0., 0.], [1., 0.], [0., 1.], [2., 2.], [-1., -1.]])),
        ("所有训练点相同", A([[2., 2., 2.]] * 5), A([3, 1, 4, 1, 0]), A([[0., 0., 0.], [2., 2., 2.], [5., -1., 2.]])),
        ("负坐标，查询点就是训练点", A([[-3., -3.], [-1., 2.], [4., -5.]]), A([0, 1, 2]),
         A([[-1., 2.], [4., -5.], [-3., -3.]])),
    ]


GRID = [(8, 2, 10, 3, 1), (30, 1, 20, 4, 2), (300, 2, 100, 5, 1), (200, 3, 150, 6, 1), (50, 4, 40, 3, 1),
        (6, 50, 5, 4, 2), (40, 2, 40, 3, 1)]                  # (n, d, m, c, a): integers in [-a, a]
NORMAL = [(1, 1, 5, 2, 1), (1, 3, 7, 3, 1), (5, 1, 9, 3, 1), (40, 50, 30, 5, 1), (200, 3, 100, 4, 1),
          (100, 10, 1, 3, 1), (300, 8, 100, 10, 1), (60, 4, 50, 4, 1000), (60, 4, 50, 4, 0.001)]
SQUARE = [(3, 3, 3, 3, 1), (4, 2, 4, 3, 1), (50, 5, 50, 4, 1), (200, 20, 200, 5, 1), (7, 7, 7, 3, 1)]
DECIMAL = [(50, 1, 50, 3, 10), (100, 2, 100, 4, 10), (200, 4, 100, 5, 10), (150, 3, 100, 2, 20)]


def seeded(configs, salt):
    """(seed, config) pairs with seeds fixed by the judge seed."""
    r = rng(salt)
    return [(r.randrange(10 ** 6), cfg) for cfg in configs]


def random_configs(salt, count, max_n, max_d, max_m, a):
    r = rng(salt)
    out = []
    for _ in range(count):
        n, d, m = r.randint(1, max_n), r.randint(1, max_d), r.randint(1, max_m)
        out.append((r.randrange(10 ** 6), (n, d, m, r.randint(1, 6), a if a else r.randint(1, 2))))
    return out


def with_copies(kind, seed, cfg):
    """Normal inputs where some queries are training points and some training points are duplicated."""
    n, d, m, c, a = cfg
    X_train, y_train, X_query = ref.inputs(kind, seed, n, d, m, c, a)
    h = min(n, m) // 3
    X_query[:h] = X_train[:h]
    X_train[n - h:] = X_train[:h]
    text = ref.recipe(kind, seed, n, d, m, c, a) + f"; X_query[:{h}] = X_train[:{h}]; X_train[{n - h}:] = X_train[:{h}]"
    return X_train, y_train, X_query, text


ROUNDING_TIE = (np.array([[0.0], [0.6]]), np.array([0, 1]), np.array([[0.3]]))     # the page's example


def near_ties(seed, d, m):
    """Queries far apart; next to query j sit training points 2j (label 0) and 2j + 1 (label 1), with point 2j
    farther by a relative 1e-7 in distance. That is far beyond float64 rounding, so 2j + 1 is the only right answer,
    but float32 coordinates cannot tell the two apart."""
    g = np.random.default_rng(seed)
    X_query = 10 * g.standard_normal((m, d))
    V = 0.05 * g.standard_normal((m, d))
    X_train = np.empty((2 * m, d))
    X_train[0::2] = X_query + V * (1 + 5e-8)
    X_train[1::2] = X_query + V
    y_train = np.tile([0, 1], m)
    text = (f"g = np.random.default_rng({seed}); X_query = 10 * g.standard_normal(({m}, {d})); "
            f"V = 0.05 * g.standard_normal(({m}, {d})); X_train = np.empty(({2 * m}, {d})); "
            f"X_train[0::2] = X_query + V * (1 + 5e-8); X_train[1::2] = X_query + V; y_train = np.tile([0, 1], {m})")
    return X_train, y_train, X_query, text


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    call = calling(ref.literal(EX_TRAIN, EX_Y, EX_QUERY) + "\npredict_1nn(X_train, y_train, X_query)  # [2, 0, 1]")
    got = sol.predict_1nn(*copies(EX_TRAIN, EX_Y, EX_QUERY))
    check_labels(got, EX_TRAIN, EX_Y, EX_QUERY, EX_NN, True, call, "返回值")


@case("Part 1", title="小输入：n = 1、重复点、并列")
def p1_small(sol):
    for title, X_train, y_train, X_query in fixed_inputs():
        run_1nn(sol, X_train, y_train, X_query, True, f"# {title}\n" + ref.literal(X_train, y_train, X_query))
    seed = 7
    run_1nn(sol, *ref.inputs("grid", seed, 6, 50, 5, 4, 2), True, ref.recipe("grid", seed, 6, 50, 5, 4, 2))


@case("Part 1", title="整数坐标的并列：取最小下标")
def p1_integer_ties(sol):
    for seed, cfg in seeded(GRID, "grid") + random_configs("grid-random", 6, 200, 4, 60, 0):
        run_1nn(sol, *ref.inputs("grid", seed, *cfg), True, ref.recipe("grid", seed, *cfg))


@case("Part 1", title="随机浮点输入")
def p1_random(sol):
    for seed, cfg in seeded(NORMAL, "normal") + random_configs("normal-random", 6, 200, 50, 60, 1):
        run_1nn(sol, *ref.inputs("normal", seed, *cfg), False, ref.recipe("normal", seed, *cfg))
    for seed, cfg in seeded([(30, 3, 30, 4, 1), (90, 6, 60, 5, 1)], "copies"):
        X_train, y_train, X_query, text = with_copies("normal", seed, cfg)
        run_1nn(sol, X_train, y_train, X_query, False, text)


@case("Part 1", title="m == n")
def p1_square(sol):
    for seed, cfg in seeded(SQUARE, "square"):
        run_1nn(sol, *ref.inputs("normal", seed, *cfg), False, ref.recipe("normal", seed, *cfg))
    seed = rng("square-grid").randrange(10 ** 6)
    run_1nn(sol, *ref.inputs("grid", seed, 6, 2, 6, 3, 1), True, ref.recipe("grid", seed, 6, 2, 6, 3, 1))


@case("Part 1", title="小数坐标：舍入误差内的并列任选")
def p1_decimal(sol):
    run_1nn(sol, *ROUNDING_TIE, False, "# 0.0 和 0.6 到 0.3 的距离精确相等，选下标 0 或 1 都算对\n" + ref.literal(*ROUNDING_TIE))
    for seed, cfg in seeded(DECIMAL, "decimal"):
        run_1nn(sol, *ref.inputs("decimal", seed, *cfg), False, ref.recipe("decimal", seed, *cfg))


@case("Part 1", title="距离只差 1e-7（相对）也要分清")
def p1_near_ties(sol):
    for seed, (d, m) in seeded([(2, 30), (3, 40), (10, 20)], "near"):
        X_train, y_train, X_query, text = near_ties(seed, d, m)
        run_1nn(sol, X_train, y_train, X_query, False, text)


# Broadcasting (m, n, d) differences is loop-free and allowed; with np.linalg.norm it needs about 3 s here, the
# reference 0.25 s, and the double loop over queries and training points about 22 s.
@case("Part 1", cpu=6, title="n = m = 2000、d = 50")
def p1_large(sol):
    seed = rng().randrange(10 ** 6)
    cfg = (2000, 50, 2000, 10, 1)
    run_1nn(sol, *ref.inputs("normal", seed, *cfg), False, ref.recipe("normal", seed, *cfg))


@case("Part 1", title="不用 Python 循环（读源代码）")
def p1_no_loops(sol):
    call = calling("检查 predict_1nn 和它调用的 answer.py 里的函数的源代码")
    found = static_loops(sol, sol.predict_1nn)
    if found:
        name, line, kind = found
        raise Mismatch(f"Part 1 不允许 Python 层面的循环，但 answer.py 第 {line} 行（函数 {name}）用了 {kind}。"
                       f"题目不允许在 Python 层面对训练点、查询点或坐标做迭代（for、while、推导式、map、np.vectorize，"
                       f"以及逐行调用 Python 函数的 np.apply_along_axis），每一步都要是对整个数组的运算。",
                       call, None, None, show=False)


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    call = calling(ref.literal(EX_TRAIN, EX_Y, EX_QUERY) + "\nprobs, labels = predict_1nn_affine(X_train, y_train, X_query)")
    layer = layer_or_none(sol, EX_TRAIN)
    # q_0 ties x_0 and x_1. Only a layer that rounds (a scale such as 1/3) may break the tie either way.
    sets = EX_NN if layer is None or exact_layer(EX_QUERY, *layer) else [np.array([0, 1])] + EX_NN[1:]
    out = pair(sol.predict_1nn_affine(*copies(EX_TRAIN, EX_Y, EX_QUERY)), "(probs, labels)", call)
    probs = as_array(out[0], (3, 4), "probs ", call)
    try:
        expect_close(probs.sum(axis=1), np.ones(3), rtol=0, atol=ref.row_sum_tolerance(4), call=call)
    except Mismatch as e:
        e.msg = "probs 的每一行之和应为 1：" + (probs_hint(probs) or e.msg)
        raise
    check_labels(out[1], EX_TRAIN, EX_Y, EX_QUERY, sets, True, call, "labels ")


@case("Part 2", title="affine_layer：argmax 是最近邻")
def p2_affine_layer(sol):
    run_layer(sol, EX_TRAIN, EX_Y, EX_QUERY, True, ref.literal(EX_TRAIN, EX_Y, EX_QUERY))
    for seed, cfg in seeded([(4, 4, 10, 3, 1), (7, 7, 20, 3, 1), (30, 30, 50, 5, 1)], "layer-square"):
        run_layer(sol, *ref.inputs("normal", seed, *cfg), False, "# d == n\n" + ref.recipe("normal", seed, *cfg))
    for seed, cfg in seeded(NORMAL, "layer-normal"):
        run_layer(sol, *ref.inputs("normal", seed, *cfg), False, ref.recipe("normal", seed, *cfg))
    for seed, cfg in seeded(GRID, "layer-grid"):
        run_layer(sol, *ref.inputs("grid", seed, *cfg), True, ref.recipe("grid", seed, *cfg))


@case("Part 2", title="前向传播：随机浮点输入")
def p2_forward(sol):
    for seed, cfg in (seeded(NORMAL + SQUARE + [(5, 5, 8, 3, 1), (20, 20, 3, 4, 1)], "forward")
                      + random_configs("forward-random", 6, 300, 50, 300, 1)):
        run_forward(sol, *ref.inputs("normal", seed, *cfg), False, ref.recipe("normal", seed, *cfg))
    for seed, cfg in seeded([(30, 3, 30, 4, 1), (90, 6, 60, 5, 1)], "forward-copies"):
        X_train, y_train, X_query, text = with_copies("normal", seed, cfg)
        run_forward(sol, X_train, y_train, X_query, False, text)


@case("Part 2", title="整数坐标的并列：取最小下标")
def p2_integer_ties(sol):
    for title, X_train, y_train, X_query in fixed_inputs():
        run_forward(sol, X_train, y_train, X_query, True, f"# {title}\n" + ref.literal(X_train, y_train, X_query))
    for seed, cfg in seeded(GRID, "forward-grid") + random_configs("forward-grid-random", 6, 300, 4, 100, 0):
        run_forward(sol, *ref.inputs("grid", seed, *cfg), True, ref.recipe("grid", seed, *cfg))


@case("Part 2", title="小数坐标：舍入误差内的并列任选")
def p2_decimal(sol):
    run_forward(sol, *ROUNDING_TIE, False, "# 0.0 和 0.6 到 0.3 的距离精确相等，选下标 0 或 1 都算对\n" + ref.literal(*ROUNDING_TIE))
    for seed, cfg in seeded(DECIMAL, "forward-decimal"):
        run_forward(sol, *ref.inputs("decimal", seed, *cfg), False, ref.recipe("decimal", seed, *cfg))


@case("Part 2", title="坐标大到 logits 上千")
def p2_large_logits(sol):
    for seed, cfg in seeded([(20, 2, 30, 3, 40), (100, 3, 80, 4, 40), (50, 5, 50, 5, 30)], "large-logits"):
        run_forward(sol, *ref.inputs("uniform", seed, *cfg), False, ref.recipe("uniform", seed, *cfg))


@case("Part 2", cpu=4, title="n = m = 2000、d = 50")
def p2_large(sol):
    seed = rng().randrange(10 ** 6)
    cfg = (2000, 50, 2000, 10, 1)
    run_forward(sol, *ref.inputs("normal", seed, *cfg), False, ref.recipe("normal", seed, *cfg))
