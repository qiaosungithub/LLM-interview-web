from judgelib import case, expect, rng, show_call
import ref

MOD = ref.MOD
EX = [3, 1, 4, 1, 5]


def check_p1(sol, c, G, want=None, call=None):
    want = ref.brute(c, G, G)[0] if want is None else want
    expect(sol.min_cost(list(c), G), want, call=call or show_call("min_cost", c, G))


def check_p2(sol, c, G, H, want=None, call=None):
    want = ref.brute(c, G, H)[0] if want is None else want
    expect(sol.min_cost_limited(list(c), G, H), want, call=call or show_call("min_cost_limited", c, G, H))


def check_p3(sol, c, G, H, want=None, call=None):
    if want is None:
        cost, ways = ref.brute(c, G, H)
        want = (cost, ways % MOD)
    got = sol.count_optimal(list(c), G, H)
    if isinstance(got, list) and len(got) == 2:     # the statement says tuple; a list carries the same answer
        got = tuple(got)
    msg = ""
    if isinstance(got, tuple) and len(got) == 2 and got != want:
        if got[0] != want[0]:
            msg = "最小费用不对"
        elif isinstance(got[1], int) and got[1] >= MOD:
            msg = "最小费用对了，方案数没有对 10⁹ + 7 取模"
        else:
            msg = "最小费用对了，方案数不对"
    expect(got, want, call=call or show_call("count_optimal", c, G, H), msg=msg)


# Big inputs are drawn once at import time, which is not charged to any case.
_big = rng("big")
BIG = [_big.randint(0, 10 ** 9) for _ in range(2000)]
BIG_TIES = [_big.randint(0, 3) for _ in range(2000)]
BIG_DESC = "<2000 个 0..10⁹ 的随机整数>"
TIES_DESC = "<2000 个 0..3 的随机整数>"


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子 G = 2")
def p1_example_g2(sol):
    expect(sol.min_cost(list(EX), 2), 10, call="min_cost([3, 1, 4, 1, 5], 2)")


@case("Part 1", sample=True, title="题目例子 G = 4：间隔正好是 G")
def p1_example_g4(sol):
    expect(sol.min_cost(list(EX), 4), 8, call="min_cost([3, 1, 4, 1, 5], 4)")


@case("Part 1", sample=True, title="题目例子 G = 1：每一帧都得选")
def p1_example_g1(sol):
    expect(sol.min_cost(list(EX), 1), 14, call="min_cost([3, 1, 4, 1, 5], 1)")


@case("Part 1", sample=True, title="题目例子：只有一帧")
def p1_example_one_frame(sol):
    expect(sol.min_cost([7], 3), 7, call="min_cost([7], 3)")


@case("Part 1", sample=True, title="题目例子：没有帧")
def p1_example_empty(sol):
    expect(sol.min_cost([], 1), 0, call="min_cost([], 1)")


@case("Part 1", title="两帧：首尾都必须选")
def p1_two_frames(sol):
    for c, G in [([4, 6], 1), ([4, 6], 2), ([4, 6], 2000), ([0, 0], 1), ([10 ** 9, 0], 5)]:
        check_p1(sol, c, G)


@case("Part 1", title="间隔可以恰好等于 G")
def p1_gap_exactly_g(sol):
    for c, G in [([0, 9, 9, 9, 9, 0], 5), ([0, 9, 9, 9, 9, 0], 4), ([1, 9, 9, 0, 9, 9, 0, 9, 9, 1], 3),
                 ([0, 5, 5, 5, 0, 5, 5, 5, 0], 4)]:
        check_p1(sol, c, G)


@case("Part 1", title="G 不小于 n - 1：只选首尾最便宜")
def p1_g_at_least_n(sol):
    for c, G in [([5] * 6, 2000), ([1, 100, 100, 1], 3), ([1, 100, 100, 1], 7), ([2, 0, 0, 0, 2], 4)]:
        check_p1(sol, c, G)


@case("Part 1", title="不能贪心地跳到够得着的最便宜帧")
def p1_not_greedy(sol):
    for c, G in [([0, 1, 2, 100, 0, 100, 0], 2), ([0, 1, 3, 3, 0], 2), ([0, 1, 4, 7, 0], 2),
                 ([0, 1, 5, 5, 0, 9, 9, 0], 3)]:
        check_p1(sol, c, G)


@case("Part 1", title="费用很大：总和超过 2³¹")
def p1_big_costs(sol):
    check_p1(sol, [10 ** 9] * 9, 1)
    check_p1(sol, [10 ** 9, 10 ** 9 - 1, 10 ** 9, 10 ** 9 - 1, 10 ** 9], 2)
    expect(sol.min_cost([10 ** 9] * 2000, 1), 2 * 10 ** 12, call="min_cost([10 ** 9] * 2000, 1)")


@case("Part 1", title="随机小输入（对照暴力枚举）")
def p1_random_small(sol):
    r = rng()
    for _ in range(250):
        c, G, _H = ref.random_input(r, 0, 12)
        check_p1(sol, c, G)


@case("Part 1", title="随机中等输入（n ≤ 40）")
def p1_random_medium(sol):
    r = rng()
    for _ in range(150):
        c, G, _H = ref.random_input(r, 13, 40)
        check_p1(sol, c, G, want=ref.fast(c, G, G)[0])


@case("Part 1", cpu=2, title="n = 2000：要求 O(nG)")
def p1_large(sol):
    for G in (2000, 999, 37, 1):
        want = ref.fast(BIG, G, G)[0]
        expect(sol.min_cost(list(BIG), G), want, call=f"min_cost({BIG_DESC}, {G})")


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子：两个长间隔不能相邻")
def p2_example_no_two_long(sol):
    expect(sol.min_cost_limited([0, 2, 1, 3, 0], 2, 1), 3, call="min_cost_limited([0, 2, 1, 3, 0], 2, 1)")


@case("Part 2", sample=True, title="题目例子 G = 3, H = 2")
def p2_example_g3_h2(sol):
    expect(sol.min_cost_limited(list(EX), 3, 2), 9, call="min_cost_limited([3, 1, 4, 1, 5], 3, 2)")


@case("Part 2", sample=True, title="题目例子：单独一个长间隔可以")
def p2_example_single_long(sol):
    expect(sol.min_cost_limited(list(EX), 4, 2), 8, call="min_cost_limited([3, 1, 4, 1, 5], 4, 2)")


@case("Part 2", sample=True, title="题目例子 H = G：没有长间隔")
def p2_example_h_equals_g(sol):
    expect(sol.min_cost_limited(list(EX), 2, 2), 10, call="min_cost_limited([3, 1, 4, 1, 5], 2, 2)")


@case("Part 2", title="n = 0、1、2")
def p2_small_n(sol):
    for G, H in [(1, 1), (3, 1), (3, 3)]:
        check_p2(sol, [], G, H)
        check_p2(sol, [9], G, H)
        check_p2(sol, [4, 6], G, H)


@case("Part 2", title="间隔恰好等于 H 不算长间隔")
def p2_gap_equal_h_is_short(sol):
    for c, G, H in [([0, 9, 0, 9, 0], 2, 2), ([0, 9, 9, 0, 9, 9, 0], 3, 3), ([0, 9, 9, 0, 9, 9, 0], 4, 3)]:
        check_p2(sol, c, G, H)


@case("Part 2", title="长间隔可以多次出现，只要不相邻")
def p2_long_gaps_apart(sol):
    for c, G, H in [([0, 5, 5, 0, 0, 5, 5, 0], 3, 2), ([0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0], 3, 1)]:
        check_p2(sol, c, G, H)


@case("Part 2", title="费用更高但最后一步是短间隔的走法不能丢")
def p2_keep_dearer_short_ending(sol):
    for c, G, H in [([0, 1, 0, 3, 1], 2, 1), ([0, 2, 1, 0, 5, 2, 0], 3, 2), ([0, 1, 3, 0, 3, 3, 0], 4, 2)]:
        check_p2(sol, c, G, H)


@case("Part 2", title="H = G：和 Part 1 的答案相同")
def p2_h_equals_g(sol):
    r = rng()
    for _ in range(60):
        c, G, _H = ref.random_input(r, 0, 30)
        check_p2(sol, c, G, G, want=ref.fast(c, G, G)[0])


@case("Part 2", title="随机小输入（对照暴力枚举）")
def p2_random_small(sol):
    r = rng()
    for _ in range(250):
        c, G, H = ref.random_input(r, 0, 12)
        check_p2(sol, c, G, H)


@case("Part 2", title="随机中等输入（n ≤ 40）")
def p2_random_medium(sol):
    r = rng()
    for _ in range(150):
        c, G, H = ref.random_input(r, 13, 40)
        check_p2(sol, c, G, H, want=ref.fast(c, G, H)[0])


@case("Part 2", cpu=3, title="n = 2000：要求 O(nG)")
def p2_large(sol):
    for G, H in [(2000, 1000), (2000, 1), (500, 499), (40, 7)]:
        want = ref.fast(BIG, G, H)[0]
        expect(sol.min_cost_limited(list(BIG), G, H), want, call=f"min_cost_limited({BIG_DESC}, {G}, {H})")


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子：两个最优方案")
def p3_example_two_plans(sol):
    check_p3(sol, list(EX), 3, 2, want=(9, 2), call="count_optimal([3, 1, 4, 1, 5], 3, 2)")


@case("Part 3", sample=True, title="题目例子：最优方案唯一")
def p3_example_unique(sol):
    check_p3(sol, [0, 2, 1, 3, 0], 2, 1, want=(3, 1), call="count_optimal([0, 2, 1, 3, 0], 2, 1)")


@case("Part 3", sample=True, title="题目例子：费用全 0，H = G")
def p3_example_zeros_free(sol):
    check_p3(sol, [0, 0, 0, 0], 3, 3, want=(0, 4), call="count_optimal([0, 0, 0, 0], 3, 3)")


@case("Part 3", sample=True, title="题目例子：费用全 0，H = 1")
def p3_example_zeros_limited(sol):
    check_p3(sol, [0, 0, 0, 0], 2, 1, want=(0, 3), call="count_optimal([0, 0, 0, 0], 2, 1)")


@case("Part 3", sample=True, title="题目例子：没有帧，只有空方案")
def p3_example_empty(sol):
    check_p3(sol, [], 1, 1, want=(0, 1), call="count_optimal([], 1, 1)")


@case("Part 3", title="n = 1、2 或 G = 1：只有一个方案")
def p3_single_plan(sol):
    for c, G, H in [([9], 1, 1), ([9], 5, 2), ([0], 3, 3), ([4, 6], 1, 1), ([0, 0], 4, 2), ([5] * 7, 1, 1)]:
        check_p3(sol, c, G, H)


@case("Part 3", title="费用全为 0：数出全部合法方案")
def p3_all_zero(sol):
    r = rng()
    for _ in range(80):
        n = r.randint(0, 12)
        G = r.randint(1, n + 1)
        check_p3(sol, [0] * n, G, r.randint(1, G))


@case("Part 3", title="费用之和超过 10⁹ + 7：费用不能取模")
def p3_cost_not_reduced(sol):
    for c, G, H in [([10 ** 9] * 5, 2, 2), ([10 ** 9, 10 ** 9, 7, 10 ** 9, 10 ** 9], 2, 2), ([10 ** 9] * 7, 3, 2),
                    ([10 ** 9, 1, 10 ** 9, 10 ** 9, 1, 10 ** 9], 2, 1)]:
        check_p3(sol, c, G, H)


@case("Part 3", title="方案数超过 10⁹ + 7：要取模")
def p3_count_reduced(sol):
    check_p3(sol, [0] * 40, 40, 40, want=(0, pow(2, 38, MOD)), call="count_optimal([0] * 40, 40, 40)")
    check_p3(sol, [0] * 50, 2, 2, want=(0, ref.fibonacci_mod(50)), call="count_optimal([0] * 50, 2, 2)")


@case("Part 3", title="随机小输入（对照暴力枚举）")
def p3_random_small(sol):
    r = rng()
    for _ in range(250):
        c, G, H = ref.random_input(r, 0, 12)
        check_p3(sol, c, G, H)


@case("Part 3", title="随机中等输入，并列很多（n ≤ 40）")
def p3_random_medium(sol):
    r = rng()
    for _ in range(150):
        c, G, H = ref.random_input(r, 13, 40, top=r.choice([0, 1, 2, 5, 10 ** 9]))
        check_p3(sol, c, G, H, want=ref.fast(c, G, H))


@case("Part 3", cpu=5, title="n = 2000、费用全 0：数出 2ⁿ⁻² 等")
def p3_large_zero_cost(sol):
    n = 2000
    for G, H, ways in [(2000, 2000, pow(2, n - 2, MOD)), (2, 2, ref.fibonacci_mod(n)),
                       (2, 1, ref.ones_and_twos_no_adjacent_twos_mod(n - 1))]:
        check_p3(sol, [0] * n, G, H, want=(0, ways), call=f"count_optimal([0] * {n}, {G}, {H})")


@case("Part 3", cpu=5, title="n = 2000：要求 O(nG)")
def p3_large(sol):
    for c, desc, G, H in [(BIG_TIES, TIES_DESC, 2000, 1000), (BIG, BIG_DESC, 2000, 1), (BIG_TIES, TIES_DESC, 60, 20)]:
        check_p3(sol, c, G, H, want=ref.fast(c, G, H), call=f"count_optimal({desc}, {G}, {H})")
