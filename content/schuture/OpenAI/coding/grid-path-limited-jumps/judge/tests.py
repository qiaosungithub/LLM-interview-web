import operator

from judgelib import calling, case, expect, expect_valid, rng, show_call
import ref

MOD = 10 ** 9 + 7

BOARD_A = [[-3, 1, 4],          # the examples of Parts 1 and 2
           [-1, 4, 4],
           [-3, -2, -3],
           [1, 0, -1],
           [5, 5, 3]]
BOARD_C = [[2, 5],              # the example of Part 3
           [0, -9],
           [3, 1]]
BOARD_D = [[3, 1],              # the example of Part 4
           [-2, 5],
           [1, 2],
           [6, 6],
           [5, 6]]


# ---------------------------------------------------------------- calling the answer

def _copy(board):
    return [list(row) for row in board]


def _call(fn, board, *args, name=None):
    """How the call is shown: the board in full, or described when it is big."""
    if name is None:
        return show_call(fn, board, *args)
    return f"{fn}({name}, {', '.join(repr(a) for a in args)})"


def _big_msg(msg, name):
    """A big board is only described, so say what the expected value was checked against."""
    if msg or name is None:
        return msg
    return "大网格：结果与另一种写法的参考 DP 算出的不同"


def _self_check(board, p, K, X=0, Y=0):
    """Every path enumerated, for hand-written boards: the values written in the tests must agree with it."""
    return ref.brute(board, p, K, X, Y)


def check_max(sol, board, p, K, want, msg="", name=None):
    call = calling(_call("max_score", board, p, K, name=name))
    expect(sol.max_score(_copy(board), p, K), want, call=call, msg=_big_msg(msg, name))


def check_count(sol, board, p, K, want, msg="", name=None):
    call = calling(_call("count_optimal_paths", board, p, K, name=name))
    got = sol.count_optimal_paths(_copy(board), p, K)
    if not msg and isinstance(got, int) and got >= MOD:
        msg = "路径条数要对 10⁹ + 7 取模"
    expect(got, want, call=call, msg=_big_msg(msg, name))


def check_bonus(sol, board, p, K, X, Y, want, msg="", name=None):
    call = calling(_call("max_score_with_bonuses", board, p, K, X, Y, name=name))
    expect(sol.max_score_with_bonuses(_copy(board), p, K, X, Y), want, call=call, msg=_big_msg(msg, name))


def _as_path(got):
    """A list of (row, column) tuples, accepting lists for tuples and numpy integers; None if got is not one."""
    try:
        return [(operator.index(a), operator.index(b)) for a, b in got]
    except Exception:
        return None


def check_path(sol, board, p, K, best, want, msg="", name=None):
    """optimal_path must return a legal path whose base score is best, and the smallest such path (want)."""
    call = calling(_call("optimal_path", board, p, K, name=name))
    got = sol.optimal_path(_copy(board), p, K)
    path = _as_path(got)

    def verdict(_):
        if path is None:
            return "应当返回格子列表，每个格子是 (row, column) 元组，例如 [(0, 0), (1, 1)]"
        bad = ref.path_problem(board, p, K, path)
        if bad:
            return bad
        score = ref.base(board, path)
        if score != best:
            return f"这条路径的基础分数是 {score}，不是最大值 {best}"
        if path != want:
            t = next(t for t, (a, b) in enumerate(zip(path, want)) if a != b)
            return msg or (f"取得最大值 {best} 的路径不止一条，要返回字典序最小的那条：两条路径的第 {t + 1} 个格子"
                           f"不同，{want[t]} < {path[t]}")
        return None

    expect_valid(got, verdict, call=call, want=want)


# ---------------------------------------------------------------- random small boards, drawn once at import time

_r = rng("small")
SMALL = []
for _ in range(300):
    _board, _p, _K = ref.random_small(_r)
    _X = _r.choice([0, 1, 2, 3, 5, 10 ** 9, _r.randint(0, 6)])
    _Y = _r.choice([0, 1, 2, 4, 7, 10 ** 9, _r.randint(0, 6)])
    SMALL.append((_board, _p, _K, _X, _Y, ref.brute(_board, _p, _K, _X, _Y)))

# Big boards. Their expected values are computed inside the cases (a few tenths of a second each), so that
# loading the tests stays fast for 运行样例.
_r = rng("big")
BIG = [[_r.choice([0, 0, 1]) for _ in range(200)] for _ in range(200)]     # many tied optimal paths
BIG_P, BIG_K = _r.randrange(200), 1
BIG_NAME = "<200×200 的网格，值取 0 或 1>"
WIDE = ref.random_board(_r, 60, 60, wide=True)
WIDE_P, WIDE_K = _r.randrange(60), 3
WIDE_NAME = "<60×60 的网格，值在 ±10⁹ 之间>"
TALL = [[_r.choice([-1, 0, 1, 2]) for _ in range(4)] for _ in range(200)]
TALL_P, TALL_K = _r.randrange(4), 200
TALL_NAME = "<200×4 的网格，值取 -1 到 2>"
ZERO = [[0] * 20 for _ in range(200)]
ZERO_NAME = "[[0] * 20 for _ in range(200)]"


def random_check(sol, part):
    """Every small board against the enumeration of all its paths."""
    for board, p, K, X, Y, (best, path, count, bonus) in SMALL:
        msg = "随机小网格：结果与穷举所有路径得到的不同"
        if part == "Part 1":
            check_max(sol, board, p, K, best, msg)
        elif part == "Part 2":
            check_path(sol, board, p, K, best, path)
        elif part == "Part 3":
            check_count(sol, board, p, K, count, msg)
        else:
            check_bonus(sol, board, p, K, X, Y, bonus, msg)


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    check_max(sol, BOARD_A, 0, 1, 6, "题目例子：例如 (0,0) -> (1,1) -> (3,1) -> (4,0)（依次经过 -3, 4, 0, 5，用了一次特殊移动）取得 6")


@case("Part 1", title="N = 1：路径只有起点")
def p1_one_row(sol):
    for board, p, want in [([[-5, 7, 2]], 0, -5), ([[-5, 7, 2]], 1, 7), ([[-1000000000]], 0, -1000000000)]:
        assert want == _self_check(board, p, 1)[0]
        check_max(sol, board, p, 1, want, "N = 1 时路径不走任何一步，分数就是起点的值")


@case("Part 1", title="M = 1：只能直走或跳")
def p1_one_column(sol):
    board = [[1], [-5], [2], [-5], [3]]
    for K, want, msg in [
            (0, -4, "K = 0 时只能一直往下走：1 - 5 + 2 - 5 + 3 = -4"),
            (1, 1, "一次特殊移动跳过一个 -5，例如 (0, 0) -> (1, 0) -> (2, 0) -> (4, 0)，最后一步落在最后一行，共 1 分"),
            (2, 6, "(0, 0) -> (2, 0) -> (4, 0) 用两次特殊移动跳过两个 -5，共 6 分"),
            (5, 6, "K 是上限，用两次特殊移动就够了：(0, 0) -> (2, 0) -> (4, 0)，共 6 分")]:
        assert want == _self_check(board, 0, K)[0]
        check_max(sol, board, 0, K, want, msg)


@case("Part 1", title="第 N−2 行不能发起特殊移动")
def p1_no_jump_off_grid(sol):
    msg = "路径必须走到最后一行；从第 N - 2 行发起的特殊移动会落到网格外，不合法"
    for board, K, want in [([[5], [-100]], 1, -95), ([[1], [2], [-100]], 2, -97),
                           ([[3, 4], [1, 2], [-50, -60]], 3, -45)]:
        assert want == _self_check(board, 0, K)[0]
        check_max(sol, board, 0, K, want, msg)


@case("Part 1", title="特殊移动可以落在最后一行")
def p1_jump_onto_last_row(sol):
    board = [[0, 0], [-9, -9], [5, -9]]
    assert _self_check(board, 0, 1)[0] == 5
    check_max(sol, board, 0, 1, 5, "(0, 0) -> (2, 0) 是合法的特殊移动：它落在最后一行，并没有越界")


@case("Part 1", title="列号不能越界（不会绕回）")
def p1_no_wraparound(sol):
    msg = "普通移动只能走到列号在 [0, M) 内的格子；第 0 列的左边没有格子"
    board = [[0, 0, 0], [-5, -5, 100]]
    assert _self_check(board, 0, 0)[0] == -5
    check_max(sol, board, 0, 0, -5, msg)
    board = [[0, 0, 0], [100, -5, -5]]
    assert _self_check(board, 2, 0)[0] == -5
    check_max(sol, board, 2, 0, -5, "普通移动只能走到列号在 [0, M) 内的格子；最后一列的右边没有格子")


@case("Part 1", title="最多 K 次特殊移动")
def p1_k_limits(sol):
    board = [[0], [-1], [0], [-1], [0], [-1], [0]]     # each special move skips one -1; three skip them all
    for K, want in [(0, -3), (1, -2), (2, -1), (3, 0), (7, 0)]:
        assert want == _self_check(board, 0, K)[0]
        check_max(sol, board, 0, K, want, "K 是特殊移动次数的上限，可以少用：三次就能跳过全部 -1" if K > 3 else
                  f"每次特殊移动可以跳过一个 -1，{K} 次跳过 {K} 个，剩下 {3 - K} 个")


@case("Part 1", title="全是负数：跳过的行越多越好")
def p1_all_negative(sol):
    board = [[-1, -1, -1] for _ in range(6)]
    for K, want, msg in [
            (0, -6, "K = 0 时要经过全部 6 行，共 -6"),
            (1, -5, "一次特殊移动少经过一行，共 -5"),
            (2, -4, "两次特殊移动，例如经过第 0、2、4、5 行，共 -4"),
            (6, -4, "6 行的网格最多用两次特殊移动（第 N - 2 行不能发起特殊移动），最少经过 4 行，共 -4")]:
        assert want == _self_check(board, 1, K)[0]
        check_max(sol, board, 1, K, want, msg)


@case("Part 1", title="随机小网格（对照穷举）")
def p1_random(sol):
    random_check(sol, "Part 1")


@case("Part 1", title="60×60，数值 ±10⁹")
def p1_wide(sol):
    want, _ = ref.forward(WIDE, WIDE_P, WIDE_K)
    check_max(sol, WIDE, WIDE_P, WIDE_K, want, name=WIDE_NAME)


@case("Part 1", title="200 行，K = N")
def p1_tall(sol):
    want, _ = ref.forward(TALL, TALL_P, TALL_K)
    check_max(sol, TALL, TALL_P, TALL_K, want, name=TALL_NAME)


@case("Part 1", cpu=3, title="200×200：要求多项式时间")
def p1_big(sol):
    want, _ = ref.forward(BIG, BIG_P, BIG_K)
    check_max(sol, BIG, BIG_P, BIG_K, want, name=BIG_NAME)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    check_path(sol, BOARD_A, 0, 1, 6, [(0, 0), (1, 1), (3, 1), (4, 0)])


@case("Part 2", title="N = 1：只有起点")
def p2_one_row(sol):
    check_path(sol, [[3, -1]], 1, 0, -1, [(0, 1)])


@case("Part 2", title="并列时普通移动排在特殊移动前")
def p2_step_before_jump(sol):
    assert _self_check(BOARD_C, 0, 1)[:2] == (5, [(0, 0), (1, 0), (2, 0)])
    check_path(sol, BOARD_C, 0, 1, 5, [(0, 0), (1, 0), (2, 0)],
               "两条最优路径的第 2 个格子分别是 (1, 0) 和 (2, 0)：普通移动落在第 i + 1 行，比特殊移动落的第 i + 2 行小")


@case("Part 2", title="特殊移动用得越晚，路径越小")
def p2_jump_late(sol):
    board = [[-1], [-1], [-1], [-1]]
    want = [(0, 0), (1, 0), (3, 0)]
    assert _self_check(board, 0, 1)[:2] == (-3, want)
    check_path(sol, board, 0, 1, -3, want,
               "两条最优路径都用一次特殊移动：[(0, 0), (1, 0), (3, 0)] 和 [(0, 0), (2, 0), (3, 0)]，"
               "第 2 个格子 (1, 0) < (2, 0)，所以先走普通移动的那条更小")


@case("Part 2", title="用掉的特殊移动不能再用")
def p2_jumps_spent(sol):
    board = [[0], [-9], [0], [-5], [0]]
    want = [(0, 0), (2, 0), (3, 0), (4, 0)]
    assert _self_check(board, 0, 1)[:2] == (-5, want)
    check_path(sol, board, 0, 1, -5, want)


@case("Part 2", title="并列时列号小的在前")
def p2_left_first(sol):
    board = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
    want = [(0, 1), (1, 0), (2, 0)]
    assert _self_check(board, 1, 0)[:2] == (1, want)
    check_path(sol, board, 1, 0, 1, want)
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    want = [(0, 2), (1, 1), (2, 0), (3, 0)]
    assert _self_check(board, 2, 1)[:2] == (0, want)
    check_path(sol, board, 2, 1, 0, want)


@case("Part 2", title="随机小网格（对照穷举）")
def p2_random(sol):
    random_check(sol, "Part 2")


@case("Part 2", title="200 行，K = N")
def p2_tall(sol):
    best, _ = ref.forward(TALL, TALL_P, TALL_K)
    check_path(sol, TALL, TALL_P, TALL_K, best, ref.smallest_path(TALL, TALL_P, TALL_K), name=TALL_NAME)


@case("Part 2", cpu=3, title="200×200，大量并列")
def p2_big(sol):
    best, _ = ref.forward(BIG, BIG_P, BIG_K)
    check_path(sol, BIG, BIG_P, BIG_K, best, ref.smallest_path(BIG, BIG_P, BIG_K), name=BIG_NAME)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    check_count(sol, BOARD_C, 0, 1, 2, "题目例子：最大分数 5 恰好由 (0,0) -> (1,0) -> (2,0) 和 (0,0) -> (2,0)（特殊移动）两条路径取得")


@case("Part 3", title="Part 2 例子里的两条最优路径")
def p3_part2_board(sol):
    check_count(sol, BOARD_A, 0, 1, 2, "Part 2 的例子说明这里恰好有两条路径取得最大值 6")


@case("Part 3", title="N = 1：只有一条路径")
def p3_one_row(sol):
    check_count(sol, [[4, -2, 4]], 2, 1, 1, "N = 1 时唯一的路径就是起点本身")


@case("Part 3", title="跳过与走过同一格算两条")
def p3_jump_counts_separately(sol):
    msg = "用特殊移动跳过 (i + 1, j) 和走过它，是两条不同的路径"
    for board, K, want in [([[0], [0], [0]], 1, 2), ([[0], [0], [0]], 0, 1), ([[0], [0], [0], [0]], 1, 3),
                           ([[0], [0], [0], [0]], 2, 3)]:
        assert want == _self_check(board, 0, K)[2]
        check_count(sol, board, 0, K, want, msg if K else "K = 0 时不能用特殊移动")


@case("Part 3", title="出现更优的候选时重新计数")
def p3_better_resets_count(sol):
    board = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    assert _self_check(board, 1, 0)[2] == 3
    check_count(sol, board, 1, 0, 3, "最优路径都经过 (1, 1)，再走到最后一行的三个格子之一；经过 (1, 0)、(1, 2) 的路径分数更低，不能算进来")


@case("Part 3", title="比较真实分数，不能先取模")
def p3_compare_real_scores(sol):
    msg = "10⁹ 和 -7 对 10⁹ + 7 取模后相同，但只有分数为 10⁹ 的那条路径是最优的：比较分数时不能取模"
    for board, p in [([[0, 0], [1000000000, -7]], 0), ([[0, 0], [-7, 1000000000]], 1)]:
        assert _self_check(board, p, 0)[2] == 1
        check_count(sol, board, p, 0, 1, msg)


@case("Part 3", title="路径条数超过 10⁹ + 7")
def p3_modulo(sol):
    board = [[0] * 4 for _ in range(30)]                # every path is optimal
    _, want = ref.forward(board, 1, 3)
    check_count(sol, board, 1, 3, want, name="[[0] * 4 for _ in range(30)]")
    _, want = ref.forward(ZERO, 7, 5)
    check_count(sol, ZERO, 7, 5, want, name=ZERO_NAME)


@case("Part 3", title="随机小网格（对照穷举）")
def p3_random(sol):
    random_check(sol, "Part 3")


@case("Part 3", title="200 行，K = N")
def p3_tall(sol):
    _, want = ref.forward(TALL, TALL_P, TALL_K)
    check_count(sol, TALL, TALL_P, TALL_K, want, name=TALL_NAME)


@case("Part 3", cpu=3, title="200×200，大量并列")
def p3_big(sol):
    _, want = ref.forward(BIG, BIG_P, BIG_K)
    check_count(sol, BIG, BIG_P, BIG_K, want, name=BIG_NAME)


# ---------------------------------------------------------------- Part 4
@case("Part 4", sample=True, title="题目例子")
def p4_example(sol):
    check_bonus(sol, BOARD_D, 0, 1, 3, 6, 29, "题目例子：(0,0) -> (1,1) -> (3,1) -> (4,1) 依次经过 3, 5, 6, 6，基础分数 20，"
                "加 X = 3（最后两个值相等）和 Y = 6（3 < 5 < 6），共 29")


@case("Part 4", title="N = 1：没有奖励")
def p4_one_row(sol):
    check_bonus(sol, [[7, -3]], 1, 0, 10 ** 9, 10 ** 9, -3, "N = 1 时路径只有一个格子，两项奖励都无从谈起")


@case("Part 4", title="X = Y = 0 时就是 Part 1")
def p4_zero_bonus(sol):
    check_bonus(sol, BOARD_A, 0, 1, 0, 0, 6, "X = Y = 0 时没有奖励，结果就是 Part 1 例子的最大分数 6")


@case("Part 4", title="特殊移动的两端也算相邻")
def p4_equal_across_jump(sol):
    board = [[2], [-1], [2]]
    assert _self_check(board, 0, 1, 10, 0)[3] == 14
    check_bonus(sol, board, 0, 1, 10, 0, 14, "(0, 0) -> (2, 0) 的两个值都是 2：特殊移动的两端也是连续访问的，加 X")


@case("Part 4", title="递增三元组可以跨过特殊移动")
def p4_rise_across_jump(sol):
    board = [[1], [-5], [2], [-5], [3]]
    assert _self_check(board, 0, 2, 0, 100)[3] == 106
    check_bonus(sol, board, 0, 2, 0, 100, 106, "两次特殊移动依次访问 1、2、3：严格递增，加 Y")


@case("Part 4", title="起点之前没有格子")
def p4_no_rise_at_start(sol):
    msg = "递增奖励要求 t ≥ 2：起点之前没有格子，第一步不可能结束一个递增三元组"
    for board, want in [([[1], [2]], 3), ([[1], [2], [3]], 106), ([[1], [2], [3], [4]], 210)]:
        assert want == _self_check(board, 0, 0, 0, 100)[3]
        check_bonus(sol, board, 0, 0, 0, 100, want, msg)


@case("Part 4", title="相等不算递增")
def p4_strictly_increasing(sol):
    msg = "递增奖励要求 v[t-2] < v[t-1] < v[t]，相等的值不算递增"
    for board, X, want in [([[1], [1], [1]], 0, 3), ([[1], [1], [1]], 5, 13), ([[1], [2], [2]], 0, 5),
                           ([[1], [2], [2]], 7, 12)]:
        assert want == _self_check(board, 0, 0, X, 100)[3]
        check_bonus(sol, board, 0, 0, X, 100, want, msg)


@case("Part 4", title="奖励可能改变最优路径")
def p4_bonus_changes_path(sol):
    board = [[0, 0], [5, 1], [0, 1]]
    assert _self_check(board, 0, 0, 10, 0)[3] == 12
    check_bonus(sol, board, 0, 0, 10, 0, 12,
                "基础分数最高的路径 0, 5, 1 只有 6 分；0, 1, 1 的基础分数是 2，加上一次 X = 10，共 12 分")


@case("Part 4", title="分数低的走法之后可能拿到奖励")
def p4_lower_arrival_wins(sol):
    board = [[1, 0], [1, 50], [2, -1000], [3, -1000]]
    assert _self_check(board, 0, 0, 0, 100)[3] == 107
    check_bonus(sol, board, 0, 0, 0, 100, 107,
                "经过 1, 1, 2, 3 的路径到 (2, 0) 时只有 4 分，比经过 1, 50, 2 的 53 分低，但它下一步 1 < 2 < 3 能拿到 Y = 100，"
                "共 107 分：同一个格子要按“上一步是否上升”分开记最优值，不能只留分数最高的那个")


@case("Part 4", title="随机小网格（对照穷举）")
def p4_random(sol):
    random_check(sol, "Part 4")


@case("Part 4", title="X = Y = 10⁹，数值 ±10⁹")
def p4_wide(sol):
    want = ref.forward_bonus(WIDE, WIDE_P, WIDE_K, 10 ** 9, 10 ** 9)
    check_bonus(sol, WIDE, WIDE_P, WIDE_K, 10 ** 9, 10 ** 9, want, name=WIDE_NAME)


@case("Part 4", cpu=3, title="200 行，K = N")
def p4_tall(sol):
    want = ref.forward_bonus(TALL, TALL_P, TALL_K, 2, 3)
    check_bonus(sol, TALL, TALL_P, TALL_K, 2, 3, want, name=TALL_NAME)


@case("Part 4", cpu=4, title="200×200")
def p4_big(sol):
    want = ref.forward_bonus(BIG, BIG_P, BIG_K, 1, 2)
    check_bonus(sol, BIG, BIG_P, BIG_K, 1, 2, want, name=BIG_NAME)
