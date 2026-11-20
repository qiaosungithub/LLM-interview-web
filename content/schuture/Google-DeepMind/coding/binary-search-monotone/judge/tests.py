import math
import numbers
from decimal import Decimal

from judgelib import Mismatch, calling, case, expect, expect_true, expect_valid, info, rng, show_call
import ref


# ---------------------------------------------------------------- helpers

def check_min_max(sol, a, want=None, call=None):
    want = (min(a), max(a)) if want is None else want
    call = calling(call or show_call("bitonic_min_max", a))
    got = sol.bitonic_min_max(list(a))
    if isinstance(got, list) and len(got) == 2:          # the statement says tuple; a 2-list carries the same answer
        got = tuple(got)
    msg = ""
    if isinstance(got, tuple) and len(got) == 2 and got != want:
        msg = "最小值不对" if got[1] == want[1] else "最大值不对" if got[0] == want[0] else ""
    expect(got, want, call=call, msg=msg)


class Counted(int):
    """An int that counts every comparison made with it (Part 1: the maximum in O(log n) comparisons)."""
    __slots__ = ()
    count = 0

    def __lt__(self, other):
        Counted.count += 1
        return int.__lt__(self, other)

    def __le__(self, other):
        Counted.count += 1
        return int.__le__(self, other)

    def __gt__(self, other):
        Counted.count += 1
        return int.__gt__(self, other)

    def __ge__(self, other):
        Counted.count += 1
        return int.__ge__(self, other)

    def __eq__(self, other):
        Counted.count += 1
        return int.__eq__(self, other)

    def __ne__(self, other):
        Counted.count += 1
        return int.__ne__(self, other)

    __hash__ = int.__hash__
    __repr__ = int.__repr__


class IntFunction:
    """Part 2's f: the lambda `text`, defined on the positive integers only, counting its calls."""

    def __init__(self, text, call):
        self.text, self.fn, self.call = text, eval(text, {"math": math}), call
        self.calls, self.problem = 0, ""

    def __call__(self, x):
        self.calls += 1
        if isinstance(x, bool) or not isinstance(x, numbers.Integral):
            self.problem = (f"f 的参数应当是正整数（int），却收到了 {x!r}（{type(x).__name__}）："
                            "f 只定义在正整数上，取中点请用 //")
        elif x < 1:
            self.problem = f"f 只定义在正整数上，却被调用为 f({x!r})"
        if self.problem:
            raise Mismatch(self.problem, self.call, show=False)
        return self.fn(x)


def run_p2(sol, fname, text, y, y_text=None):
    call = calling(f"{fname}({text}, {y_text or repr(y)})")
    f = IntFunction(text, call)
    got = getattr(sol, fname)(f, y)
    if f.problem:                                       # the answer caught the Mismatch raised inside f
        raise Mismatch(f.problem, call, show=False)
    return got, f, call


def sample_p2(sol, fname, text, y, want):
    got, _f, call = run_p2(sol, fname, text, y)
    expect(got, want, call=call)


def check_p2(sol, text, y):
    """Both functions of Part 2 on one (f, y), against a linear scan."""
    fn = eval(text, {"math": math})
    want_floor = ref.floor_brute(fn, y)
    want_inv = want_floor if want_floor is not None and fn(want_floor) == y else None
    for fname, want in (("inverse", want_inv), ("inverse_floor", want_floor)):
        got, _f, call = run_p2(sol, fname, text, y)
        expect(got, want, call=call)


def check_plateau(sol, a):
    call = calling(show_call("bitonic_max_with_plateaus", a))
    expect(sol.bitonic_max_with_plateaus(list(a)), max(a), call=call)


def root_call(text, y, lo, hi, eps):
    prefix = "import math; " if text.startswith("math.") else ""
    return f"{prefix}bisect_root({text}, {y!r}, {lo!r}, {hi!r}, {eps!r})"


def check_root(sol, text, y, lo, hi, eps):
    root, slope = ref.exact_root(text, y, lo, hi)
    tol = ref.tolerance(root, y, slope, eps)
    g = eval(text, {"math": math})
    call = calling(root_call(text, y, lo, hi, eps))
    got = sol.bisect_root(g, y, lo, hi, eps)

    def verdict(x):
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            return f"应当返回一个实数（float），得到的是 {type(x).__name__}"
        if not math.isfinite(float(x)):
            return "返回值不是有限的实数"
        err = abs(Decimal(float(x)) - root)
        if err <= Decimal(tol):
            return None
        if eps > 0:
            return f"|x - x*| = {float(err):.3g}，超过了 eps = {eps:g}（真正的根 x* ≈ {float(root)!r}）"
        return (f"eps = 0 时应当逼近到浮点运算能分辨的程度：|x - x*| = {float(err):.3g}，"
                f"而在这里浮点运算能分辨到约 {tol:.2g}（真正的根 x* ≈ {float(root)!r}）")

    expect_valid(got, verdict, call=call, want=float(root))


# Part 1's comparison-count inputs: expressions the user can paste, built once at import time (not charged).
BIG_N = 100_000
BIG_INPUTS = [
    "list(range(100000))",                                  # peak at the right end
    "list(range(100000, 0, -1))",                           # peak at the left end
    "list(range(60000)) + list(range(60000, 20000, -1))",   # peak at index 60000
    "[-x * x for x in range(-30000, 70000)]",               # peak at index 30000
    "list(range(99990)) + list(range(99990, 99980, -1))",   # peak at index 99990
]
BIG_PLAIN = [eval(expr) for expr in BIG_INPUTS]
BIG_WANTS = [(min(a[0], a[-1]), max(a)) for a in BIG_PLAIN]
BIG_LISTS = [[Counted(v) for v in a] for a in BIG_PLAIN]
del BIG_PLAIN


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子 [1, 3, 8, 12, 9, 5, 2]")
def p1_example(sol):
    check_min_max(sol, [1, 3, 8, 12, 9, 5, 2], want=(1, 12), call="bitonic_min_max([1, 3, 8, 12, 9, 5, 2])")


@case("Part 1", title="只有一个元素")
def p1_single(sol):
    for a in ([5], [-7], [0]):
        check_min_max(sol, a)


@case("Part 1", title="两个元素，两种顺序")
def p1_two(sol):
    for a in ([2, 9], [9, 2], [-5, -1], [-1, -5]):
        check_min_max(sol, a)


@case("Part 1", title="峰值在两端：严格递增或递减")
def p1_peak_at_ends(sol):
    for n in range(3, 12):
        check_min_max(sol, list(range(n)))
        check_min_max(sol, list(range(n, 0, -1)))
        check_min_max(sol, [-3 * k for k in range(n)])


@case("Part 1", title="峰值两侧可以有相同的值")
def p1_equal_across_peak(sol):
    for a in ([1, 5, 1], [3, 4, 3], [2, 7, 5, 2], [0, 9, 8, 0], [1, 2, 3, 2, 1], [-4, 6, 5, -4]):
        check_min_max(sol, a)


@case("Part 1", title="负数和很大的数")
def p1_negative_and_big(sol):
    for a in ([-10, -3, -1, -2, -8], [-50, -40, -45], [-(10 ** 18), 10 ** 18, 0], [1, 10 ** 30, 10 ** 30 - 1]):
        check_min_max(sol, a)


@case("Part 1", title="n ≤ 9 的全部严格双调排列")
def p1_exhaustive(sol):
    for n in range(1, 10):
        for a in ref.all_strict_bitonic(n):
            check_min_max(sol, a)


@case("Part 1", title="随机双调数组（峰值位置随机）")
def p1_random(sol):
    r = rng()
    for _ in range(400):
        check_min_max(sol, ref.random_bitonic(r, r.randint(1, 120), 1))


@case("Part 1", title="n = 10⁵：比较次数必须是 O(log n)")
def p1_comparisons(sol):
    limit = 6 * math.ceil(math.log2(BIG_N)) + 10
    for expr, big, want in zip(BIG_INPUTS, BIG_LISTS, BIG_WANTS):
        call = calling(f"bitonic_min_max({expr})")
        Counted.count = 0
        got = sol.bitonic_min_max(list(big))
        used = Counted.count
        if isinstance(got, list) and len(got) == 2:
            got = tuple(got)
        expect(got, want, call=call)
        info(f"{expr}：比较了 {used} 次")
        expect_true(used <= limit,
                    f"对 a 的元素一共比较了 {used} 次。n = {BIG_N} 时，O(log n) 的算法只需要几十次（这里允许 {limit} 次）；"
                    "最大值要用二分查找峰值，最小值只需比较首尾两个元素", call=call)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子 inverse(x * x, 49)")
def p2_example_square(sol):
    sample_p2(sol, "inverse", "lambda x: x * x", 49, 7)


@case("Part 2", sample=True, title="题目例子 inverse(x * x, 50)")
def p2_example_square_missing(sol):
    sample_p2(sol, "inverse", "lambda x: x * x", 50, None)


@case("Part 2", sample=True, title="题目例子 inverse(x + 5, 5)：x = 0 不算")
def p2_example_zero_not_positive(sol):
    sample_p2(sol, "inverse", "lambda x: x + 5", 5, None)


@case("Part 2", sample=True, title="题目例子 inverse(x + 5, 6)")
def p2_example_answer_one(sol):
    sample_p2(sol, "inverse", "lambda x: x + 5", 6, 1)


@case("Part 2", sample=True, title="题目例子 inverse(x - 100, 5)")
def p2_example_answer_above_y(sol):
    sample_p2(sol, "inverse", "lambda x: x - 100", 5, 105)


@case("Part 2", sample=True, title="题目例子 inverse_floor(x * x, 50)")
def p2_example_floor(sol):
    sample_p2(sol, "inverse_floor", "lambda x: x * x", 50, 7)


@case("Part 2", title="f(1) > y：两个函数都返回 None")
def p2_below_range(sol):
    for text, y in [("lambda x: x + 1000", 5), ("lambda x: x * x", 0), ("lambda x: x - 100", -100),
                    ("lambda x: 2 ** x", 1), ("lambda x: 3 * x", -7)]:
        check_p2(sol, text, y)


@case("Part 2", title="答案是 1：f(1) 恰好等于或小于 y")
def p2_answer_one(sol):
    for text, y in [("lambda x: x * x", 1), ("lambda x: x * x", 3), ("lambda x: x - 100", -99),
                    ("lambda x: 10 * x", 15), ("lambda x: 2 ** x", 3)]:
        check_p2(sol, text, y)


@case("Part 2", title="答案恰好是 2 的幂")
def p2_power_of_two(sol):
    for text, y in [("lambda x: x * x", 64), ("lambda x: x * x", 4), ("lambda x: x - 1000", -488),
                    ("lambda x: 3 * x", 3 * 1024), ("lambda x: 3 * x", 3 * 1024 + 2), ("lambda x: x ** 3", 2 ** 18),
                    ("lambda x: x - 5", 3), ("lambda x: 2 * x", 4)]:
        check_p2(sol, text, y)


@case("Part 2", title="y 落在两个函数值之间")
def p2_between_values(sol):
    for text, y in [("lambda x: x * x", 63), ("lambda x: x * x", 65), ("lambda x: 2 ** x", 1000),
                    ("lambda x: 5 * x - 3", 20), ("lambda x: x ** 3 - 50", -20)]:
        check_p2(sol, text, y)


@case("Part 2", title="答案远大于 y：f(x) 可以小于 x")
def p2_answer_beyond_y(sol):
    for text, y in [("lambda x: x - 1000", 0), ("lambda x: x - 1000", 7), ("lambda x: x + x // 2 - 300", 10),
                    ("lambda x: x - 100", -50), ("lambda x: x * x - 10000", -1)]:
        check_p2(sol, text, y)


@case("Part 2", title="随机严格递增函数（对照逐个枚举）")
def p2_random(sol):
    r = rng()
    for _ in range(250):
        text = ref.random_increasing_text(r)
        fn = eval(text, {"math": math})
        # 2 ** x stays at y <= about 10**6: the statement counts calls to f, not their cost, so an answer that
        # also calls f at arguments up to about y (e.g. a search on [1, y - f(1) + 1]) must not hang on 2 ** x.
        x0 = r.randint(1, 20 if "2 ** x" in text else 250)
        y = fn(1) - r.randint(1, 30) if r.random() < 0.1 else fn(x0) + r.choice([0, 0, 0, 1, -1, r.randint(-5, 5)])
        check_p2(sol, text, y)


@case("Part 2", title="答案上千亿：不能逐个试 x")
def p2_huge_answer(sol):
    cases = [("lambda x: x - 10 ** 12", "0", 10 ** 12, 10 ** 12),
             ("lambda x: x", "10 ** 30", 10 ** 30, 10 ** 30),
             ("lambda x: x * x", "10 ** 40", 10 ** 20, 10 ** 20),
             ("lambda x: x * x", "10 ** 40 - 1", 10 ** 20 - 1, None),
             ("lambda x: 3 * x - 10 ** 15", "2", 333333333333334, 333333333333334),
             ("lambda x: 3 * x - 10 ** 15", "3", 333333333333334, None),
             ("lambda x: x ** 10", "10 ** 200", 10 ** 20, 10 ** 20),
             ("lambda x: x ** 10", "10 ** 200 - 1", 10 ** 20 - 1, None),
             ("lambda x: x - 2 ** 300", "5", 2 ** 300 + 5, 2 ** 300 + 5)]
    for text, y_text, want_floor, want_inv in cases:
        for fname, want in (("inverse_floor", want_floor), ("inverse", want_inv)):
            got, f, call = run_p2(sol, fname, text, eval(y_text), y_text)
            expect(got, want, call=call)
            if fname == "inverse_floor":
                info(f"{call}：调用 f {f.calls} 次")


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="(a) 题目例子 [0, 0, 0, 1, 0, 0, 0, 0]")
def p3a_example(sol):
    calling("bitonic_max_with_plateaus([0, 0, 0, 1, 0, 0, 0, 0])")
    expect(sol.bitonic_max_with_plateaus([0, 0, 0, 1, 0, 0, 0, 0]), 1)


@case("Part 3", sample=True, title="(b) 题目例子 x³ = 30，eps = 0.001")
def p3b_example(sol):
    check_root(sol, "lambda x: x ** 3", 30, 0, 4, 0.001)


@case("Part 3", title="(a) 一个元素、全部相等")
def p3a_trivial(sol):
    for a in ([4], [-2], [0, 0], [7, 7, 7, 7, 7], [-1] * 10):
        check_plateau(sol, a)


@case("Part 3", title="(a) 全 0 中只有一个 1，每个位置都试")
def p3a_single_spike(sol):
    for n in range(2, 13):
        for q in range(n):
            a = [0] * n
            a[q] = 1
            check_plateau(sol, a)


@case("Part 3", title="(a) 峰值本身是一段平台")
def p3a_plateau_peak(sol):
    for a in ([1, 3, 3, 3, 2], [5, 5, 4], [0, 1, 1], [2, 2, 2, 9, 9, 1, 1], [-3, -1, -1, -1, -1, -2]):
        check_plateau(sol, a)


@case("Part 3", title="(a) 严格双调也是非严格双调")
def p3a_strict(sol):
    for a in ([1, 3, 8, 12, 9, 5, 2], list(range(10)), list(range(10, 0, -1)), [1, 5, 1]):
        check_plateau(sol, a)


@case("Part 3", title="(a) 取值 {0, 1, 2}、n ≤ 7 的全部情形")
def p3a_exhaustive(sol):
    import itertools
    for n in range(1, 8):
        for a in itertools.product(range(3), repeat=n):
            if ref.is_weak_bitonic(a):
                check_plateau(sol, list(a))


@case("Part 3", title="(a) 随机非严格双调数组")
def p3a_random(sol):
    r = rng()
    for _ in range(300):
        check_plateau(sol, ref.random_bitonic(r, r.randint(1, 80), 0))


@case("Part 3", title="(b) 线性、指数、对数、五次函数")
def p3b_functions(sol):
    for args in [("lambda x: 2 * x + 1", 8, -10, 10, 1e-06), ("lambda x: x ** 3", 500, 0, 10, 1e-09),
                 ("math.exp", 20, 0, 5, 1e-12), ("math.log", 1.5, 1, 100, 1e-08),
                 ("lambda x: x ** 5 + x", -40, -10, 3, 1e-07), ("lambda x: x ** 3", -8, -5, -1, 1e-06)]:
        check_root(sol, *args)


@case("Part 3", title="(b) 根在区间端点")
def p3b_root_at_endpoint(sol):
    for args in [("lambda x: x ** 3", 8, 2, 5, 1e-09), ("lambda x: x ** 3", 125, 2, 5, 1e-09),
                 ("lambda x: 2 * x + 1", -1, -1, 4, 0.01), ("lambda x: 2 * x + 1", 9, -1, 4, 0.01)]:
        check_root(sol, *args)


@case("Part 3", title="(b) eps 不小于区间长度")
def p3b_big_eps(sol):
    for args in [("lambda x: x ** 3", 30, 0, 4, 10), ("lambda x: x ** 3", 30, 0, 4, 4)]:
        check_root(sol, *args)


@case("Part 3", title="(b) g 很平缓：误差按 x 算，不按 g(x)")
def p3b_flat(sol):
    for args in [("lambda x: x / 1e6", 3.3, 0, 1e7, 0.001), ("lambda x: x / 1e6", 0.75, 0, 1e6, 0.5)]:
        check_root(sol, *args)


@case("Part 3", title="(b) g 很陡")
def p3b_steep(sol):
    for args in [("lambda x: 1e6 * x", 3, 0, 1, 1e-09), ("lambda x: 1e6 * x", 1234567, -10, 10, 1e-06)]:
        check_root(sol, *args)


@case("Part 3", title="(b) eps = 0：必须停下，且逼近到浮点精度")
def p3b_eps_zero(sol):
    for args in [("lambda x: x ** 3", 30, 0, 4, 0), ("lambda x: x ** 3", 500, 0, 10, 0), ("math.exp", 20, 0, 5, 0),
                 ("lambda x: 2 * x + 1", 1, -1, 1, 0), ("lambda x: x", 1e-12, 0, 1e6, 0),
                 ("lambda x: x / 1e6", 2.5, 0, 1e7, 0), ("math.log", 1.5, 1, 100, 0),
                 ("lambda x: x ** 3", 125, 2, 5, 0)]:
        check_root(sol, *args)


@case("Part 3", title="(b) 随机函数、区间和 eps")
def p3b_random(sol):
    r = rng()
    for _ in range(60):
        if r.random() < 0.5:
            text = f"lambda x: x ** 3 + {r.randint(0, 20)} * x"
        else:
            text = f"lambda x: {r.randint(2, 9)} * x - {r.randint(1, 50)}"
        g = eval(text)
        lo = r.choice([-1000, -10, -3, 0, 1])
        hi = lo + r.choice([0.5, 2, 7, 100, 1000])
        y = g(lo + (hi - lo) * r.random())
        check_root(sol, text, y, lo, hi, r.choice([0.1, 1e-3, 1e-6, 1e-10, 0.0]))
