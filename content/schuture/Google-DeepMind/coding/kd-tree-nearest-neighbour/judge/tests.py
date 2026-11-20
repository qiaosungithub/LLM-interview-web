import numpy as np

from judgelib import calling, case, expect, expect_raises, expect_true, rng
import ref

P_EX = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)], dtype=np.float64)     # the statement's points
Q_EX = np.array([4.0, 2.0])
LIMIT = 600                     # longest P shown in full in a replay


# ---------------------------------------------------------------- building and checking, with a replay

def build(sol, P, text=None):
    """KDTree(P) on a copy of P; returns (tree, replay line). P is shown in full when short, else as text."""
    full = ref.arr_text(P) if text is None or P.size <= LIMIT // 4 else None      # each entry takes >= 4 chars
    head = f"t = KDTree({full if text is None or (full and len(full) <= LIMIT) else text})"
    calling(head)
    return sol.KDTree(P.copy()), head


def _int(x):
    return int(x) if isinstance(x, np.integer) else x


def _int_list(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, tuple):
        x = list(x)
    return [_int(v) for v in x] if isinstance(x, list) else x


def _d2(P, i, q):
    return float(ref.sq_dists(P[i:i + 1], q)[0])


def check_nearest(sol, t, head, P, q, want=None):
    want = ref.nearest(P, q) if want is None else want
    call = calling(f"{head}\nt.nearest({ref.arr_text(q)})")
    got = _int(t.nearest(q.copy()))
    if got != want and isinstance(got, int) and 0 <= got < len(P):
        dg, dw = _d2(P, got, q), _d2(P, want, q)
        msg = (f"点 {got} 和点 {want} 的平方距离都是 {dw:g}，并列时要返回较小的下标 {want}" if dg == dw else
               f"点 {got} 不是最近的：它的平方距离是 {dg:g}，点 {want} 的是 {dw:g}")
        expect(got, want, call=call, msg=msg)
    expect(got, want, call=call)


def check_knn(sol, t, head, P, q, k, want=None):
    want = ref.knn(P, q, k) if want is None else want
    call = calling(f"{head}\nt.knn({ref.arr_text(q)}, {k})")
    got = _int_list(t.knn(q.copy(), k))
    if isinstance(got, list) and got != want:
        if len(got) != len(want):
            expect(got, want, call=call, msg=f"应返回 k = {k} 个下标，得到 {len(got)} 个")
        if sorted(got, key=str) == sorted(want, key=str):
            expect(got, want, call=call, msg="下标集合对了，但要按 (距离², 下标) 升序排列")
        extra = [i for i in got if i not in want and isinstance(i, int) and 0 <= i < len(P)]
        if extra:
            i = extra[0]
            j = [w for w in want if w not in got][0]
            msg = (f"点 {i} 和点 {j} 的平方距离都是 {_d2(P, j, q):g}，并列时应取下标较小的 {j}"
                   if _d2(P, i, q) == _d2(P, j, q) else
                   f"点 {i} 不在最近的 {k} 个里（平方距离 {_d2(P, i, q):g}），漏掉了点 {j}（平方距离 {_d2(P, j, q):g}）")
            expect(got, want, call=call, msg=msg)
    expect(got, want, call=call)


def check_box(sol, t, head, P, lo, hi, want=None):
    want = ref.count_in_box(P, lo, hi) if want is None else want
    call = calling(f"{head}\nt.count_in_box({ref.arr_text(lo)}, {ref.arr_text(hi)})")
    got = _int(t.count_in_box(lo.copy(), hi.copy()))
    if got != want and isinstance(got, int):
        inside = sum(1 for p in P.tolist() if all(lo[i] < p[i] < hi[i] for i in range(len(lo))))
        if got == inside:
            expect(got, want, call=call, msg=f"矩形是闭的：正好落在边上的点也要算，这里有 {want - got} 个")
        expect(got, want, call=call, msg=f"矩形里有 {want} 个点，得到 {got}")
    expect(got, want, call=call)


def raises(fn, *args, call):
    calling(call)
    expect_raises(ValueError, fn, *args, call=call)


# ---------------------------------------------------------------- big inputs, built once at import time

BIG_N = 20000
BUILD_P, BUILD_TEXT = ref.dyadic_cloud(rng("p1-build"), BIG_N, 2)
PRUNE_P, PRUNE_P_TEXT = ref.dyadic_cloud(rng("p2-points"), BIG_N, 2)
PRUNE_Q, PRUNE_Q_TEXT = ref.dyadic_cloud(rng("p2-queries"), 1000, 2)
PRUNE_WANT = ref.nearest_many(PRUNE_P, PRUNE_Q)


def _larger():
    r = rng("p3-larger")
    P, text = ref.dyadic_cloud(r, 3000, 3)
    knns = []
    for _ in range(40):
        q = ref.dyadic_query(r, 3)
        k = r.choice([1, 2, 5, 20, 100])
        knns.append((q, k, ref.knn(P, q, k)))
    boxes = []
    for _ in range(40):
        lo = ref.dyadic_query(r, 3, bits=6)
        hi = lo + ref.dyadic_query(r, 3, bits=6) / 2
        boxes.append((lo, hi, ref.count_in_box(P, lo, hi)))
    return text, P, knns, boxes


LARGER_TEXT, LARGER_P, LARGER_KNN, LARGER_BOX = _larger()


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子：能建树")
def p1_example(sol):
    t, _ = build(sol, P_EX)
    expect_true(t is not None, "KDTree(P) 没有返回对象")


@case("Part 1", title="各种合法形状")
def p1_shapes(sol):
    for P in (np.array([[3.0, 4.0]]), np.array([[7.0]]), np.array([[1.0], [-2.0], [0.5]]),
              np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
              np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]), np.array([[-5.0, 2.5], [-5.0, -2.5]])):
        build(sol, P)


@case("Part 1", title="形状不是 (n, d) 时抛 ValueError")
def p1_bad_shape(sol):
    for P, text in ((np.zeros((0, 2)), "np.zeros((0, 2))"), (np.zeros((0, 1)), "np.zeros((0, 1))"),
                    (np.array([1.0, 2.0, 3.0]), "np.array([1.0, 2.0, 3.0])"),
                    (np.zeros((3, 2, 2)), "np.zeros((3, 2, 2))"), (np.array(5.0), "np.array(5.0)")):
        raises(sol.KDTree, P, call=f"KDTree({text})")


@case("Part 1", cpu=2, title="2 万个随机点：要求 O(n log² n)")
def p1_big_random(sol):
    build(sol, BUILD_P, text=BUILD_TEXT)


@case("Part 1", cpu=2, title="2 万个点按坐标排好序")
def p1_big_sorted(sol):
    build(sol, np.array([[i, i] for i in range(BIG_N)], dtype=float),
          text=f"np.array([[i, i] for i in range({BIG_N})], dtype=float)")
    build(sol, np.array([[i] for i in range(BIG_N, 0, -1)], dtype=float),
          text=f"np.array([[i] for i in range({BIG_N}, 0, -1)], dtype=float)")


@case("Part 1", cpu=2, title="2 万个点，坐标大量重复")
def p1_big_duplicates(sol):
    build(sol, np.zeros((BIG_N, 2)), text=f"np.zeros(({BIG_N}, 2))")
    build(sol, np.array([[i % 3, i % 2] for i in range(BIG_N)], dtype=float),
          text=f"np.array([[i % 3, i % 2] for i in range({BIG_N})], dtype=float)")


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子 nearest")
def p2_example(sol):
    t, head = build(sol, P_EX)
    check_nearest(sol, t, head, P_EX, Q_EX, 0)


@case("Part 2", title="q 的形状不是 (d,) 时抛 ValueError")
def p2_bad_query(sol):
    t, head = build(sol, P_EX)
    for q in (np.array([1.0, 2.0, 3.0]), np.array([[4.0, 2.0]]), np.array([4.0])):
        raises(t.nearest, q, call=f"{head}\nt.nearest({ref.arr_text(q)})")
    raises(t.nearest, np.array(4.0), call=f"{head}\nt.nearest(np.array(4.0))")


@case("Part 2", title="只有一个点")
def p2_single(sol):
    for P, q in ((np.array([[3.0, 4.0]]), np.array([0.0, 0.0])), (np.array([[7.0]]), np.array([-100.0])),
                 (np.array([[1.0, 2.0, 3.0]]), np.array([1.0, 2.0, 3.0]))):
        t, head = build(sol, P)
        check_nearest(sol, t, head, P, q, 0)


@case("Part 2", title="并列时返回最小下标")
def p2_tie_smallest(sol):
    for P, q, want in ((np.array([[2.0], [-1.0]]), np.array([0.5]), 0),
                       (np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]), np.array([0.0, 2.0]), 0),
                       (np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([0.0, 0.0]), 0),
                       (np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]), np.array([1.0, 1.0]), 0),
                       (np.array([[5.0, 0.0], [3.0, 0.0], [1.0, 0.0]]), np.array([2.0, 0.0]), 1)):
        t, head = build(sol, P)
        check_nearest(sol, t, head, P, q, want)


@case("Part 2", title="等距且下标更小的点在远侧")
def p2_tie_far_side(sol):
    for P, q, want in ((np.array([[0.0], [0.0]]), np.array([3.0]), 0),
                       (np.array([[0.0, 1.0], [3.0, 3.0], [0.0, 1.0]]), np.array([0.5, 1.0]), 0),
                       (np.array([[1.0, 2.0], [1.0, 2.0]]), np.array([3.0, 2.0]), 0)):
        t, head = build(sol, P)
        check_nearest(sol, t, head, P, q, want)


@case("Part 2", title="最近点在分割线的另一侧")
def p2_far_side_closer(sol):
    for P, q, want in ((np.array([[0.125, 0.75], [0.125, 0.625]]), np.array([0.25, 0.9375]), 0),
                       (np.array([[0.0, 0.0], [1.0, 0.0], [1.25, 9.0]]), np.array([0.875, 9.0]), 2),
                       (P_EX, np.array([6.5, 0.0]), 4)):
        t, head = build(sol, P)
        check_nearest(sol, t, head, P, q, want)


@case("Part 2", title="坐标要保持 float64 精度")
def p2_float64_precision(sol):
    # 2**24 + 1 is exact in float64 but rounds to 2**24 in float32, making a false tie that index 0 wins;
    # every squared term stays below 2**53, so |p|^2 - 2 p.q + |q|^2 is exact here too
    P = np.array([[16777216.0, 0.0], [16777217.0, 0.0]])
    t, head = build(sol, P)
    check_nearest(sol, t, head, P, np.array([16777217.0, 0.0]), 1)


@case("Part 2", title="查询已存的点（含重复的点）")
def p2_stored_points(sol):
    r = rng()
    for _ in range(20):
        P, text = ref.grid_cloud(r, r.randint(1, 15), r.randint(1, 3), -1, 1)
        t, head = build(sol, P, text)
        for i in range(len(P)):
            check_nearest(sol, t, head, P, P[i].copy())


@case("Part 2", title="一维")
def p2_one_dim(sol):
    r = rng()
    for _ in range(60):
        P, text = ref.grid_cloud(r, r.randint(1, 20), 1, -5, 5)
        t, head = build(sol, P, text)
        for _ in range(5):
            check_nearest(sol, t, head, P, ref.grid_query(r, 1, -7, 7))


@case("Part 2", title="随机整数点，大量并列")
def p2_random_grid(sol):
    r = rng()
    for _ in range(250):
        d = r.choice([2, 2, 3, 4])
        P, text = ref.grid_cloud(r, r.randint(1, 40), d)
        t, head = build(sol, P, text)
        for _ in range(4):
            check_nearest(sol, t, head, P, ref.grid_query(r, d))


@case("Part 2", title="随机分布在 [0, 1) 的点")
def p2_random_dense(sol):
    r = rng()
    for _ in range(150):
        d = r.choice([2, 3, 5])
        P, text = ref.dyadic_cloud(r, r.randint(1, 60), d)
        t, head = build(sol, P, text)
        for _ in range(4):
            check_nearest(sol, t, head, P, ref.dyadic_query(r, d))


@case("Part 2", title="查询点离所有点都很远")
def p2_far_queries(sol):
    r = rng()
    for _ in range(40):
        d = r.choice([1, 2, 3])
        P, text = ref.grid_cloud(r, r.randint(1, 30), d, -10, 10)
        t, head = build(sol, P, text)
        for _ in range(3):
            q = np.array([float(r.choice([-1, 1]) * r.randint(10 ** 5, 10 ** 6)) for _ in range(d)])
            check_nearest(sol, t, head, P, q)


@case("Part 2", cpu=2, title="2 万个点、1000 次查询：要求剪枝")
def p2_pruning(sol):
    head = f"P = {PRUNE_P_TEXT}\nQ = {PRUNE_Q_TEXT}\nt = KDTree(P)"
    calling(head)
    t = sol.KDTree(PRUNE_P.copy())
    calling(f"{head}\n[t.nearest(q) for q in Q]")
    for j, q in enumerate(PRUNE_Q):
        got = _int(t.nearest(q.copy()))
        if got != PRUNE_WANT[j]:
            check_nearest(sol, t, f"{head}\n# 第 {j} 个查询 Q[{j}]", PRUNE_P, q.copy(), PRUNE_WANT[j])


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子 knn")
def p3_example_knn(sol):
    t, head = build(sol, P_EX)
    check_knn(sol, t, head, P_EX, Q_EX, 3, [0, 1, 5])


@case("Part 3", sample=True, title="题目例子 count_in_box")
def p3_example_box(sol):
    t, head = build(sol, P_EX)
    check_box(sol, t, head, P_EX, np.array([4.0, 1.0]), np.array([8.0, 5.0]), 3)


@case("Part 3", title="k 不在 1..n 时抛 ValueError")
def p3_knn_bad_k(sol):
    t, head = build(sol, P_EX)
    for k in (7, 0, -1, 100):
        raises(t.knn, Q_EX.copy(), k, call=f"{head}\nt.knn(np.array([4.0, 2.0]), {k})")


@case("Part 3", title="k = n 与 k = 1")
def p3_knn_extremes(sol):
    t, head = build(sol, P_EX)
    check_knn(sol, t, head, P_EX, Q_EX, 6, [0, 1, 5, 4, 3, 2])
    check_knn(sol, t, head, P_EX, Q_EX, 1, [0])
    P = np.array([[3.0, 4.0]])
    t, head = build(sol, P)
    check_knn(sol, t, head, P, np.array([0.0, 0.0]), 1, [0])


@case("Part 3", title="第 k 名并列时取小下标")
def p3_knn_ties(sol):
    for P, q, k, want in ((np.array([[1.0], [1.0]]), np.array([0.0]), 1, [0]),
                          (np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), np.array([2.0, 0.0]), 2, [0, 1]),
                          (np.array([[2.0, 2.0], [2.0, 1.0], [2.0, 2.0]]), np.array([2.0, 0.0]), 2, [1, 0]),
                          (np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 1.0], [2.0, -1.0]]), np.array([2.0, 0.0]), 3, [0, 1, 2])):
        t, head = build(sol, P)
        check_knn(sol, t, head, P, q, k, want)


@case("Part 3", title="剪枝要用第 k 近的距离")
def p3_knn_kth_bound(sol):
    for P, q, k, want in ((np.array([[1.0, 2.0], [0.0, 3.0], [0.0, 3.0]]), np.array([3.0, 0.0]), 2, [0, 1]),
                          (np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 0.0], [3.0, 5.0]]), np.array([1.5, 0.0]), 3, [2, 0, 1])):
        t, head = build(sol, P)
        check_knn(sol, t, head, P, q, k, want)


@case("Part 3", title="knn：随机整数点")
def p3_knn_random(sol):
    r = rng()
    for _ in range(200):
        d = r.choice([1, 2, 3])
        P, text = ref.grid_cloud(r, r.randint(1, 30), d)
        t, head = build(sol, P, text)
        for _ in range(4):
            check_knn(sol, t, head, P, ref.grid_query(r, d), r.randint(1, len(P)))


@case("Part 3", title="knn：随机分布在 [0, 1) 的点")
def p3_knn_dense(sol):
    r = rng()
    for _ in range(100):
        d = r.choice([2, 3, 5])
        P, text = ref.dyadic_cloud(r, r.randint(1, 60), d)
        t, head = build(sol, P, text)
        for _ in range(4):
            check_knn(sol, t, head, P, ref.dyadic_query(r, d), r.randint(1, len(P)))


@case("Part 3", title="矩形：边界上的点也算")
def p3_box_edges(sol):
    t, head = build(sol, P_EX)
    for lo, hi, want in (([2.0, 3.0], [2.0, 3.0], 1), ([2.0, 1.0], [9.0, 7.0], 6), ([4.0, 4.0], [5.0, 7.0], 2),
                         ([7.0, 1.0], [8.0, 2.0], 2), ([2.5, 3.0], [8.5, 3.5], 0)):
        check_box(sol, t, head, P_EX, np.array(lo), np.array(hi), want)
    P = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    t, head = build(sol, P)
    check_box(sol, t, head, P, np.array([1.0, 1.0]), np.array([1.0, 1.0]), 2)


@case("Part 3", title="lo > hi 或矩形里没有点时为 0")
def p3_box_empty(sol):
    t, head = build(sol, P_EX)
    for lo, hi in (([5.0, 5.0], [1.0, 1.0]), ([0.0, 5.0], [10.0, 4.0]), ([8.0, 0.0], [7.0, 10.0]),
                   ([100.0, 100.0], [200.0, 200.0]), ([2.1, 3.1], [3.9, 6.9])):
        check_box(sol, t, head, P_EX, np.array(lo), np.array(hi), 0)


@case("Part 3", title="矩形的边可以是无穷")
def p3_box_infinite(sol):
    t, head = build(sol, P_EX)
    for lo, hi, want in (([-np.inf, -np.inf], [np.inf, np.inf], 6), ([5.0, -np.inf], [np.inf, 4.0], 3),
                         ([-np.inf, 7.0], [np.inf, np.inf], 1)):
        check_box(sol, t, head, P_EX, np.array(lo), np.array(hi), want)


@case("Part 3", title="lo、hi 形状不对时抛 ValueError")
def p3_box_bad_shape(sol):
    t, head = build(sol, P_EX)
    for lo, hi in (([0.0, 0.0, 0.0], [9.0, 9.0]), ([0.0, 0.0], [9.0]), ([[0.0, 0.0]], [[9.0, 9.0]])):
        lo, hi = np.array(lo), np.array(hi)
        raises(t.count_in_box, lo, hi, call=f"{head}\nt.count_in_box({ref.arr_text(lo)}, {ref.arr_text(hi)})")


@case("Part 3", title="矩形的边正好在分割值上")
def p3_box_split_ties(sol):
    for P, lo, hi, want in ((np.array([[-2.0], [-2.0]]), [-2.0], [-1.0], 2),
                            (np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 3.0]]), [2.0, 1.0], [2.0, 2.0], 2),
                            (np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [0.0, 0.0]]), [-1.0, 0.0], [1.0, 1.0], 3)):
        t, head = build(sol, P)
        check_box(sol, t, head, P, np.array(lo), np.array(hi), want)


@case("Part 3", title="矩形：随机整数点")
def p3_box_random(sol):
    r = rng()
    for _ in range(200):
        d = r.choice([1, 2, 3])
        P, text = ref.grid_cloud(r, r.randint(1, 30), d)
        t, head = build(sol, P, text)
        for _ in range(4):
            lo = np.array([float(r.randint(-4, 3)) for _ in range(d)])
            hi = lo + np.array([float(r.randint(-1, 4)) for _ in range(d)])
            check_box(sol, t, head, P, lo, hi)


@case("Part 3", title="3000 个点的 knn 和矩形计数")
def p3_larger(sol):
    head = f"P = {LARGER_TEXT}\nt = KDTree(P)"
    calling(head)
    t = sol.KDTree(LARGER_P.copy())
    for q, k, want in LARGER_KNN:
        check_knn(sol, t, head, LARGER_P, q, k, want)
    for lo, hi, want in LARGER_BOX:
        check_box(sol, t, head, LARGER_P, lo, hi, want)
