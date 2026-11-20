from itertools import product

from judgelib import calling, case, expect_true, expect_valid, info, rng, show_call
import ref

FN = {1: "build_basic_schedule", 2: "build_balanced_schedule", 3: "build_doubly_balanced_schedule"}

# Every parameter set with t in -1..5, m in -1..4, h in -1..7, k in 0..6 (k >= 0 is given by the statement).
GRID = list(product(range(-1, 6), range(-1, 5), range(-1, 8), range(0, 7)))


def check(sol, part, t, m, h, k, want=None):
    """Calls the Part's builder and checks it against the statement's contract; returns the schedule (or None)."""
    name = FN[part]
    call = calling(show_call(name, t, m, h, k))
    got = getattr(sol, name)(t, m, h, k)
    what = ref.kind(t, m, h, k)
    if what == "none":
        why = ("t、m、h 中有不大于 0 的，维度不合法：不论 k 是多少都应返回 None" if min(t, m, h) <= 0
               else f"k = {k} > t = {t}，不可行：应返回 None")
        expect_true(got is None, why, call=call, got=got)
        return None
    expect_true(got is not None, f"这组参数可行（k ≤ t），不应返回 None", call=call, got=got)
    s = ref.normalize(got)
    if isinstance(s, str):
        expect_valid(got, lambda _g: s, call=call)
    if what == "empty":
        expect_true(s == [], "k = 0 且维度合法时应返回 []", call=call, got=got)
        return s
    if want is None and h * k <= 60:
        want = ref.one_schedule(t, m, h, k)
    expect_valid(s, lambda g: ref.problem(g, t, m, h, k, task_balance=part >= 2), call=call, want=want)
    return s


def report_human_balance(s, t, m, h, k):
    """Part 3 only reports balance by human; the statement does not require it."""
    bad = ref.human_imbalance(s, m, h)
    if bad is None:
        info(f"(t, m, h, k) = {(t, m, h, k)}：你的调度也按标注员均衡（不计分）")
    else:
        p, u, counts = bad
        info(f"(t, m, h, k) = {(t, m, h, k)}：你的调度不按标注员均衡，长度 {p} 的前缀里标注员 {u} 的模型计数是 "
             f"{counts}（不计分）")


def sweep(sol, part):
    feasible = human_ok = 0
    for t, m, h, k in GRID:
        s = check(sol, part, t, m, h, k)
        if s:
            feasible += 1
            human_ok += ref.human_imbalance(s, m, h) is None
    return feasible, human_ok


def medium(sol, part):
    r = rng()
    for _ in range(25):
        t = r.randint(1, 30)
        k = r.randint(1, t)
        check(sol, part, t, r.randint(1, 9), r.randint(1, 80), k)
    check(sol, part, 40, 6, 150, 20)
    check(sol, part, 7, 50, 120, 7)


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子（任何合法调度都算对）")
def p1_example(sol):
    check(sol, 1, 2, 2, 3, 2, want=[(0, 0, 0), (1, 0, 1), (0, 0, 2), (1, 0, 0), (0, 0, 1), (1, 0, 2)])


@case("Part 1", title="维度不合法：不论 k 都返回 None")
def p1_invalid_dims(sol):
    for t, m, h, k in [(0, 2, 3, 1), (2, 0, 3, 1), (2, 2, 0, 1), (0, 2, 3, 0), (2, -3, 3, 0), (2, 2, 0, 0),
                       (-1, -1, -1, 0), (0, 1, 1, 5), (3, 2, -2, 4)]:
        check(sol, 1, t, m, h, k)


@case("Part 1", title="k = 0：返回空列表")
def p1_k_zero(sol):
    for t, m, h, k in [(1, 1, 1, 0), (3, 2, 5, 0), (10, 4, 1, 0)]:
        check(sol, 1, t, m, h, k)


@case("Part 1", title="k > t：不可行，返回 None")
def p1_k_too_big(sol):
    for t, m, h, k in [(2, 2, 3, 3), (1, 5, 4, 2), (3, 1, 1, 10)]:
        check(sol, 1, t, m, h, k)


@case("Part 1", title="k = t：每个标注员做全部任务")
def p1_k_equals_t(sol):
    for t, m, h, k in [(3, 2, 4, 3), (1, 1, 5, 1), (5, 3, 2, 5), (4, 2, 4, 4)]:
        check(sol, 1, t, m, h, k)


@case("Part 1", title="k < t：长度恰好 h·k，不能多")
def p1_minimal_length(sol):
    for t, m, h, k in [(5, 2, 3, 2), (4, 1, 1, 1), (6, 3, 7, 3), (9, 2, 2, 1)]:
        check(sol, 1, t, m, h, k)


@case("Part 1", title="h 和 t 有公因数时也不能重复")
def p1_common_factor(sol):
    for t, m, h, k in [(4, 2, 2, 3), (6, 2, 4, 4), (4, 3, 6, 4), (6, 1, 3, 5)]:
        check(sol, 1, t, m, h, k)


@case("Part 1", title="穷举小参数（含不合法和不可行）")
def p1_sweep(sol):
    sweep(sol, 1)


@case("Part 1", title="较大的参数")
def p1_medium(sol):
    medium(sol, 1)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子（任何合法调度都算对）")
def p2_example(sol):
    check(sol, 2, 2, 2, 3, 2, want=[(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 0), (0, 0, 1), (1, 0, 2)])


@case("Part 2", title="None 和 [] 的约定与 Part 1 相同")
def p2_contract(sol):
    for t, m, h, k in [(0, 2, 3, 0), (2, 0, 3, 1), (2, 2, -1, 1), (3, 2, 5, 0), (2, 2, 3, 3), (3, 3, 3, 3)]:
        check(sol, 2, t, m, h, k)


@case("Part 2", title="同一任务一轮里出现多次")
def p2_repeat_in_round(sol):
    for t, m, h, k in [(2, 3, 4, 2), (3, 4, 10, 3), (2, 5, 9, 2), (1, 3, 7, 1)]:
        check(sol, 2, t, m, h, k)


@case("Part 2", title="模型比任务出现次数还多")
def p2_many_models(sol):
    for t, m, h, k in [(3, 10, 2, 2), (2, 7, 5, 2), (4, 9, 9, 4)]:
        check(sol, 2, t, m, h, k)


@case("Part 2", title="只有一个模型")
def p2_one_model(sol):
    for t, m, h, k in [(4, 1, 6, 3), (1, 1, 3, 1)]:
        check(sol, 2, t, m, h, k)


@case("Part 2", title="穷举小参数，检查每个前缀")
def p2_sweep(sol):
    sweep(sol, 2)


@case("Part 2", title="较大的参数")
def p2_medium(sol):
    medium(sol, 2)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子（任何合法调度都算对）")
def p3_example(sol):
    s = check(sol, 3, 2, 3, 4, 2,
              want=[(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 0, 3)])
    report_human_balance(s, 2, 3, 4, 2)


@case("Part 3", title="None 和 [] 的约定与 Part 1 相同")
def p3_contract(sol):
    for t, m, h, k in [(0, 2, 3, 0), (2, 0, 3, 1), (2, 2, -1, 1), (3, 2, 5, 0), (2, 2, 3, 3), (3, 3, 3, 3)]:
        check(sol, 3, t, m, h, k)


@case("Part 3", title="(2, 3, 6, 2)：按任务均衡必须保持")
def p3_page_counterexample(sol):
    s = check(sol, 3, 2, 3, 6, 2)
    report_human_balance(s, 2, 3, 6, 2)


@case("Part 3", title="穷举小参数，检查每个前缀")
def p3_sweep(sol):
    feasible, human_ok = sweep(sol, 3)
    info(f"穷举的 {feasible} 组可行参数里，有 {human_ok} 组的调度同时按标注员均衡（不计分）")


@case("Part 3", title="较大的参数")
def p3_medium(sol):
    medium(sol, 3)
