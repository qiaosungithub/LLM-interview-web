import operator

from judgelib import calling, case, expect, rng, show_call
import ref

GRID_A = [[1, 0, 0, 0],         # the example of Part 1
          [0, 0, 0, 0],
          [0, 0, 0, 1]]
GRID_B = [[1, 2, 0],            # the examples of Part 2
          [0, 2, 0],
          [0, 0, 0]]
GRID_B_CUT = [[1, 2, 0],
              [2, 2, 0]]
GRID_D = [[1, 0, 1],            # the example of Part 4 version A
          [0, 0, 0],
          [1, 0, 0]]
GRID_E = [[0, 0, 0, 1],         # the example of Part 5
          [0, 0, 0, 1],
          [0, 0, 1, 0]]


# ---------------------------------------------------------------- calling the answer

def _copy(grid):
    return [list(row) for row in grid]


def _int(x):
    try:
        return operator.index(x)
    except TypeError:
        return x


def _call(fn, grid, src, *args, **kwargs):
    """The call as code that prints its result: the grid in full, or for a big grid the ref.py builder that makes it."""
    if src is None:
        return f"print({show_call(fn, grid, *args, **kwargs)})"
    rest = "".join(f", {a!r}" for a in args) + "".join(f", {k}={v!r}" for k, v in kwargs.items())
    if src.startswith("["):
        return f"print({fn}({src}{rest}))"
    return f"from ref import {src.split('(')[0]}    # 生成网格的函数在 /judge/ref.py\nprint({fn}({src}{rest}))"


def _agree(want, model):
    """A value written in the tests must agree with the model in ref.py."""
    assert want is None or want == model, f"test value {want!r} disagrees with ref {model!r}"
    return model


def check_days(sol, grid, want=None, msg="", src=None, slow=True):
    """Parts 1-2. slow=True also checks the hand value against the whole-grid model."""
    model = _agree(want, ref.spread_days_fast(grid))
    if slow:
        _agree(model, ref.spread_days(grid))
    call = calling(_call("days_until_all_infected", grid, src))
    got = _int(sol.days_until_all_infected(_copy(grid)))
    if not msg and isinstance(got, int) and got != model:
        msg = ("有健康格永远不会被感染，应返回 -1" if model == -1 else
               "所有健康格最终都会被感染，不应返回 -1" if got == -1 else "返回的天数不对")
    expect(got, model, call=call, msg=msg)


def check_end(sol, grid, D, want=None, msg="", src=None, slow=True):
    """Part 3."""
    model = _agree(want, (ref.outbreak if slow else ref.outbreak_fast)(grid, D)[0])
    call = calling(_call("days_until_outbreak_ends", grid, src, D))
    got = _int(sol.days_until_outbreak_ends(_copy(grid), D))
    if not msg and isinstance(got, int) and got != model:
        msg = "疫情结束的那一天不对"
    expect(got, model, call=call, msg=msg)


def _pair(got):
    """A 2-element tuple or list of integers as a tuple; anything else unchanged."""
    if isinstance(got, (tuple, list)) and len(got) == 2:
        return tuple(_int(x) for x in got)
    return got


def check_sim(sol, grid, D, K=1, X=None, want=None, msg="", src=None, slow=True):
    """Part 4: simulate(grid, D[, spread_threshold=K][, death_threshold=X]); the defaults are left out."""
    model = _agree(want, (ref.outbreak if slow else ref.outbreak_fast)(grid, D, K, X))
    kwargs = {}
    if K != 1:
        kwargs["spread_threshold"] = K
    if X is not None:
        kwargs["death_threshold"] = X
    call = calling(_call("simulate", grid, src, D, **kwargs))
    got = _pair(sol.simulate(_copy(grid), D, **kwargs))
    if isinstance(got, tuple) and got != model:
        half = ("死亡格的数量不对" if got[0] == model[0] else
                "疫情结束的那一天不对" if got[1] == model[1] else "结束的那一天和死亡格的数量都不对")
        msg = half + ("：" + msg if msg else "")
    expect(got, model, call=call, msg=msg)


def check_min(sol, grid, D, X, K=1, want=None, msg="", src=None, slow=True):
    """Part 5: min_deaths(grid, D, X[, spread_threshold=K])."""
    best, choice = ref.fewest_deaths(grid, D, X, K, fast=not slow)
    _agree(want, best)
    kwargs = {"spread_threshold": K} if K != 1 else {}
    call = calling(_call("min_deaths", grid, src, D, X, **kwargs))
    got = _int(sol.min_deaths(_copy(grid), D, X, **kwargs))
    if got != best:
        msg = (msg + "；" if msg else "") + f"最好的选择之一是{choice}，共死亡 {best} 个格子"
    expect(got, best, call=call, msg=msg)


def random_small(r, immune):
    rows, cols = r.randint(1, 7), r.randint(1, 7)
    return ref.random_grid(rows, cols, r.randrange(10 ** 6), infected=r.choice([0, 0.05, 0.15, 0.4]),
                           immune=r.choice([0.1, 0.25, 0.4]) if immune else 0)


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    check_days(sol, GRID_A, 2)


@case("Part 1", title="第 0 天没有健康格（包括空网格）")
def p1_no_healthy(sol):
    msg = "第 0 天就没有健康格，答案是 0"
    for grid in ([], [[1]], [[1, 1], [1, 1]]):
        check_days(sol, grid, 0, msg)


@case("Part 1", title="没有已感染格")
def p1_no_infected(sol):
    for grid in ([[0]], [[0, 0, 0], [0, 0, 0]]):
        check_days(sol, grid, -1, "没有已感染格，健康格永远不会被感染")


@case("Part 1", title="一行或一列")
def p1_line(sol):
    check_days(sol, [[0, 0, 1, 0, 0]], 2)
    check_days(sol, [[0, 0, 0, 1]], 3)
    check_days(sol, [[1, 0, 0, 0, 1]], 2)
    check_days(sol, [[1], [0], [0]], 2)


@case("Part 1", title="同步更新：每天只向外扩散一格")
def p1_one_step_per_day(sol):
    msg = "第 t + 1 天才被感染的格子，要到下一步才开始感染邻居"
    check_days(sol, [[1, 0, 0, 0]], 3, msg)
    check_days(sol, [[0, 0, 0, 1]], 3, msg)
    check_days(sol, [[1], [0], [0], [0], [0]], 4, msg)
    check_days(sol, [[0], [0], [0], [1]], 3, msg)


@case("Part 1", title="对角线方向的格子不是邻居")
def p1_no_diagonals(sol):
    msg = "只有上、下、左、右四个格子是邻居"
    check_days(sol, [[1, 0], [0, 0]], 2, msg)
    check_days(sol, [[0, 0, 0], [0, 1, 0], [0, 0, 0]], 2, msg)
    check_days(sol, [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]], 5, msg)


@case("Part 1", title="随机小网格")
def p1_random(sol):
    r = rng()
    for _ in range(300):
        check_days(sol, random_small(r, immune=False))


@case("Part 1", title="大网格")
def p1_big(sol):
    seed = rng().randrange(10 ** 6)
    check_days(sol, ref.random_grid(60, 60, seed, sources=3), src=f"random_grid(60, 60, {seed}, sources=3)",
               slow=False)
    check_days(sol, ref.random_grid(60, 60, seed, infected=0.2), src=f"random_grid(60, 60, {seed}, infected=0.2)",
               slow=False)
    check_days(sol, [[1] + [0] * 499], 499, src="[[1] + [0] * 499]", slow=False)
    check_days(sol, [[0]] * 250 + [[1]] + [[0]] * 249, 250, src="[[0]] * 250 + [[1]] + [[0]] * 249", slow=False)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    check_days(sol, GRID_B, 6)
    check_days(sol, GRID_B_CUT, -1)


@case("Part 2", title="只有免疫格和已感染格")
def p2_no_healthy(sol):
    msg = "第 0 天就没有健康格，答案是 0"
    for grid in ([[2]], [[2, 2]], [[1, 2], [2, 1]]):
        check_days(sol, grid, 0, msg)


@case("Part 2", title="免疫格不传播感染")
def p2_immune_does_not_infect(sol):
    msg = "免疫格不算已感染的邻居"
    check_days(sol, [[2, 0, 0]], -1, msg)
    check_days(sol, [[1, 2, 0]], -1, msg)
    check_days(sol, [[2, 0, 1]], 1, msg)
    check_days(sol, [[0, 2], [2, 1]], -1, msg)


@case("Part 2", title="被隔开的健康格")
def p2_walled_off(sol):
    msg = "有健康格永远不可能被感染"
    check_days(sol, [[1, 0, 0], [0, 2, 2], [0, 2, 0]], -1, msg)
    check_days(sol, [[1, 0, 2, 0, 0]], -1, msg)
    check_days(sol, [[1, 0, 2, 0, 1]], 1)


@case("Part 2", title="绕过免疫格")
def p2_detour(sol):
    check_days(sol, [[1, 2, 0, 0], [0, 2, 2, 0], [0, 0, 0, 0]], 8)
    check_days(sol, ref.snake(9, 7), src="snake(9, 7)")
    check_days(sol, ref.snake(31, 31), 510, src="snake(31, 31)", slow=False)


@case("Part 2", title="随机小网格")
def p2_random(sol):
    r = rng()
    for _ in range(300):
        check_days(sol, random_small(r, immune=True))


@case("Part 2", title="大网格")
def p2_big(sol):
    r = rng()
    for _ in range(3):
        seed, sources = r.randrange(10 ** 6), r.randint(1, 4)
        check_days(sol, ref.random_grid(60, 60, seed, immune=0.3, sources=sources),
                   src=f"random_grid(60, 60, {seed}, immune=0.3, sources={sources})", slow=False)
    seed = r.randrange(10 ** 6)
    check_days(sol, ref.random_grid(60, 60, seed, infected=0.01, immune=0.15),
               src=f"random_grid(60, 60, {seed}, infected=0.01, immune=0.15)", slow=False)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    check_end(sol, [[1, 0, 0]], 2, 4)
    check_end(sol, [[1, 0, 0]], 1, 1)


@case("Part 3", title="第 0 天没有已感染格")
def p3_no_infected(sol):
    msg = "第 0 天就没有已感染格，答案是 0"
    for grid, D in (([], 1), ([[0, 0, 0]], 2), ([[2, 0], [0, 2]], 3), ([[2]], 1)):
        check_end(sol, grid, D, 0, msg)


@case("Part 3", title="D = 1：同一天先康复、后传播")
def p3_d1(sol):
    msg = "D = 1 时一开始的已感染格在第 1 天先康复，没有机会传播"
    for grid in ([[1, 1], [0, 0]], GRID_A, GRID_B, [[0, 1, 0]]):
        check_end(sol, grid, 1, 1, msg)


@case("Part 3", title="被感染满 D 天的那一天康复")
def p3_recover_on_day_d(sol):
    msg = "t − t0 ≥ D 时康复（含等于）"
    check_end(sol, [[1]], 3, 3, msg)
    check_end(sol, [[1, 0]], 3, 4, msg)
    check_end(sol, [[0, 1, 2, 1]], 5, 6, msg)


@case("Part 3", title="同步更新：当天被感染的格子当天不传播")
def p3_one_step_per_day(sol):
    msg = "第 t 天才被感染的格子，要到下一步才开始感染邻居"
    check_end(sol, [[1, 0, 0, 0, 0]], 3, 7, msg)
    check_end(sol, [[1, 0, 0], [0, 0, 0]], 2, 5, msg)
    check_end(sol, [[0, 0, 0, 0], [0, 0, 0, 1]], 4, 8, msg)


@case("Part 3", title="免疫格与留下的健康格")
def p3_immune(sol):
    check_end(sol, GRID_B_CUT, 2, 2, "健康格可以一直留在网格里")
    check_end(sol, [[0, 2, 1, 0]], 2, 3)
    check_end(sol, GRID_B, 3, 9)
    check_end(sol, [[1, 0, 2, 0, 0]], 4, 5)


@case("Part 3", title="随机小网格")
def p3_random(sol):
    r = rng()
    for _ in range(300):
        check_end(sol, random_small(r, immune=r.random() < 0.7), r.randint(1, 5))


@case("Part 3", title="大网格")
def p3_big(sol):
    r = rng()
    check_end(sol, ref.snake(31, 31), 3, 513, src="snake(31, 31)", slow=False)
    seed = r.randrange(10 ** 6)
    check_end(sol, ref.random_grid(60, 60, seed, sources=3), 7, src=f"random_grid(60, 60, {seed}, sources=3)",
              slow=False)
    seed = r.randrange(10 ** 6)
    check_end(sol, ref.random_grid(60, 60, seed, infected=0.05, immune=0.3), 2,
              src=f"random_grid(60, 60, {seed}, infected=0.05, immune=0.3)", slow=False)
    seed = r.randrange(10 ** 6)
    check_end(sol, ref.random_grid(60, 60, seed, infected=0.3), 1, 1,
              src=f"random_grid(60, 60, {seed}, infected=0.3)", slow=False)


# ---------------------------------------------------------------- Part 4
@case("Part 4", sample=True, title="题目例子：版本 A")
def p4_example(sol):
    check_sim(sol, GRID_D, 2, K=2, want=(4, 0))


@case("Part 4", title="传播阈值：恰好 K 个已感染邻居")
def p4_spread_threshold(sol):
    msg = "至少 K 个邻居已感染时就会被感染（含等于）"
    check_sim(sol, [[1, 0, 1]], 2, K=2, want=(3, 0), msg=msg)
    check_sim(sol, [[0, 1, 0], [1, 0, 1], [0, 1, 0]], 2, K=4, want=(3, 0), msg=msg)
    check_sim(sol, [[1, 0, 1], [0, 0, 0], [1, 0, 1]], 3, K=2, msg=msg)
    check_sim(sol, [[1, 1, 1], [0, 0, 0], [1, 1, 1]], 2, K=3, msg=msg)


@case("Part 4", title="阈值按同一天的已感染邻居计数")
def p4_same_day(sol):
    msg = "只有同一天至少 K 个邻居已感染才会被感染，不同天的已感染邻居不累加"
    check_sim(sol, [[1, 0]], 3, K=2, want=(3, 0), msg=msg)
    check_sim(sol, [[1, 0, 0], [0, 0, 1]], 4, K=2, msg=msg)
    check_sim(sol, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1]], 3, K=2, msg=msg)


@case("Part 4", title="阈值大于 4")
def p4_threshold_above_four(sol):
    msg = "一个格子最多只有四个邻居"
    check_sim(sol, [[1, 0], [0, 1]], 3, K=5, want=(3, 0), msg=msg)
    check_sim(sol, [[0, 1, 0], [1, 0, 1], [0, 1, 0]], 2, X=5, want=(3, 0), msg=msg)


@case("Part 4", title="版本 B：死亡")
def p4_deaths(sol):
    check_sim(sol, GRID_D, 2, X=2, want=(4, 4))
    check_sim(sol, GRID_D, 2, K=2, X=1, want=(4, 3), msg="死亡阈值不超过传播阈值时，第 0 天之后被感染的格子全部死亡")
    check_sim(sol, [[1, 0, 0, 0]], 2, X=1, want=(5, 3))
    check_sim(sol, [[0, 1, 0], [1, 0, 1]], 3, X=3)


@case("Part 4", title="死亡只在被感染的那天判定一次")
def p4_counted_once(sol):
    msg = "之后新增的已感染邻居不会让已感染格转为死亡"
    check_sim(sol, [[1, 0, 0, 0, 0, 1]], 5, X=2, want=(7, 0), msg=msg)
    check_sim(sol, [[1, 0, 0], [0, 0, 0], [0, 0, 1]], 4, X=2, msg=msg)
    check_sim(sol, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], 6, X=3, msg=msg)


@case("Part 4", title="死亡计数看康复之后的邻居")
def p4_count_after_recovery(sol):
    # Day 2: the centre gets infected by its left and right neighbours (K = 2) while the cell above it recovers.
    msg = "被感染的那一刻在第 2 阶段：当天第 1 阶段刚康复的邻居已经不算已感染"
    check_sim(sol, [[1, 1, 1], [0, 0, 0], [1, 2, 1]], 2, K=2, X=3, want=(4, 0), msg=msg)


@case("Part 4", title="一开始就已感染的格子不会死亡")
def p4_initial_cells_survive(sol):
    msg = "输入里一开始就已感染的格子不是在模拟中被感染的，期满后变为免疫"
    check_sim(sol, [[1, 1], [1, 1]], 2, X=1, want=(2, 0), msg=msg)
    check_sim(sol, [[1, 1, 0]], 2, X=1, want=(3, 1), msg=msg)
    check_sim(sol, [[1, 1, 1], [1, 1, 1], [0, 0, 0]], 3, X=2, msg=msg)


@case("Part 4", title="版本 C：随机小网格")
def p4_random(sol):
    r = rng()
    for _ in range(300):
        X = r.choice([None, 1, 2, 3, 4, 5])
        check_sim(sol, random_small(r, immune=r.random() < 0.7), r.randint(1, 5), K=r.choice([1, 1, 2, 2, 3, 4, 5]),
                  X=X)


@case("Part 4", title="大网格")
def p4_big(sol):
    r = rng()
    for (infected, immune, D, K, X) in [(0.2, 0.1, 4, 2, 3), (0.4, 0, 3, 3, 4), (0.002, 0.2, 2, 1, 1)]:
        seed = r.randrange(10 ** 6)
        check_sim(sol, ref.random_grid(60, 60, seed, infected=infected, immune=immune), D, K=K, X=X,
                  src=f"random_grid(60, 60, {seed}, infected={infected}, immune={immune})", slow=False)
    check_sim(sol, ref.snake(31, 31), 2, X=1, src="snake(31, 31)", slow=False)


# ---------------------------------------------------------------- Part 5
@case("Part 5", sample=True, title="题目例子")
def p5_example(sol):
    check_min(sol, GRID_E, 2, 1, want=4)


@case("Part 5", title="什么都不烧最好")
def p5_burn_nothing(sol):
    msg = "什么都不烧也是一种选择"
    check_min(sol, [], 2, 1, want=0, msg=msg)
    check_min(sol, [[0, 0], [0, 0]], 2, 1, want=0, msg=msg)
    check_min(sol, [[1, 0], [0, 0]], 1, 1, want=0, msg=msg)
    check_min(sol, [[1, 0, 0]], 2, 2, want=0, msg=msg)
    check_min(sol, GRID_E, 2, 5, want=0, msg=msg)


@case("Part 5", title="烧掉的格子计入死亡数")
def p5_burnt_cells_count(sol):
    msg = "被烧的那条线上的每个格子都计入死亡数"
    check_min(sol, [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], 2, 1, want=5, msg=msg)
    check_min(sol, [[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]], 2, 1, msg=msg)


@case("Part 5", title="烧掉免疫格也计入死亡数")
def p5_burnt_immune_count(sol):
    msg = "被烧的格子无论原来是什么状态都变为死亡，并计入死亡数"
    check_min(sol, [[0, 0, 2, 1], [0, 0, 2, 1], [0, 0, 0, 0]], 2, 1, msg=msg)
    check_min(sol, [[0, 0, 0, 2, 1], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]], 3, 1, msg=msg)


@case("Part 5", title="烧感染最多的那条线不一定最好")
def p5_most_infected_line(sol):
    check_min(sol, [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], 2, 1, want=4)
    check_min(sol, [[1, 0, 0], [0, 0, 0]], 2, 1, want=2)
    check_min(sol, [[0, 1, 0], [0, 0, 1]], 2, 1, want=3)


@case("Part 5", title="烧一条没有已感染格的线当隔离带")
def p5_firebreak(sol):
    msg = "没有已感染格的行或列也可以烧，用来把感染隔开"
    check_min(sol, [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], 3, 2, want=4, msg=msg)
    check_min(sol, [[1, 0, 0], [0, 0, 2], [0, 0, 1]], 2, 1, want=5, msg=msg)


@case("Part 5", title="传播阈值")
def p5_spread_threshold(sol):
    check_min(sol, [[1, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1]], 3, 2, K=2)
    check_min(sol, [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], 2, 2, K=2)


@case("Part 5", title="随机小网格")
def p5_random(sol):
    r = rng()
    for _ in range(150):
        check_min(sol, random_small(r, immune=r.random() < 0.5), r.randint(1, 4), r.randint(1, 5),
                  K=r.choice([1, 1, 2, 3]))


@case("Part 5", title="大一些的网格")
def p5_big(sol):
    r = rng()
    for rows, cols, D, X, K in [(15, 15, 3, 2, 1), (12, 20, 2, 1, 1), (15, 15, 4, 3, 2)]:
        seed = r.randrange(10 ** 6)
        check_min(sol, ref.random_grid(rows, cols, seed, infected=0.03, immune=0.15), D, X, K=K,
                  src=f"random_grid({rows}, {cols}, {seed}, infected=0.03, immune=0.15)", slow=False)
