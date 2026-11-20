from judgelib import calling, case, expect, expect_raises, expect_true, rng, show_call
import ref

EXAMPLES = [
    ([5, 2, 6, 1], [2, 1, 1, 0]),
    ([3, 3, 1, 3, 1, 2], [3, 3, 0, 2, 0, 0]),
    ([4, 4, 4, 2, 4], [1, 1, 1, 0, 0]),
    ([2, 0, 1], [2, 0, 0]),
    ([-1, -1], [0, 0]),
    ([7], [0]),
    ([], []),
]

# An element type that supports < only, as Part 2 requires. The same source is shown in the call, so the
# user can paste it into 自定义运行.
KEY_SRC = '''class Key:
    def __init__(self, v):
        self.v = v
    def __lt__(self, other):
        return self.v < other.v
    def __le__(self, other):
        raise TypeError("题目要求元素之间只用 < 比较，这里用了 <= 或 >=")
    __ge__ = __le__
    def __eq__(self, other):
        raise TypeError("题目要求元素之间只用 < 比较，这里用了 == 或 !=（元组之间的 < 也会先用 == 比较里面的元素）")
    __hash__ = None
    def __repr__(self):
        return f"Key({self.v!r})"'''
_ns = {}
exec(KEY_SRC, _ns)
Key = _ns["Key"]

# Big inputs are drawn once at import time, which is not charged to any case.
N = 200_000
_big = rng("big")
BIG = [_big.randint(-10 ** 18, 10 ** 18) for _ in range(150_000)] + [_big.randint(-50, 50) for _ in range(50_000)]
_big.shuffle(BIG)
BIG_DESC = "<200000 个随机整数：绝对值可达 10¹⁸，其中 50000 个在 -50..50 之间，有重复>"
STREAM = [_big.randint(0, 999_999) for _ in range(N)]
_big_want = []


def big_want():
    """ref.fast(BIG), computed by the first case that needs it and reused by the other."""
    if not _big_want:
        _big_want.append(ref.fast(BIG))
    return _big_want[0]


def as_list(got):
    """The statement asks for a list; a tuple carries the same answer."""
    return list(got) if isinstance(got, tuple) else got


def check(sol, fn, nums, want=None, call=None):
    """Calls fn on a copy of nums, compares with the definition, and checks that nums was not modified."""
    want = ref.brute(nums) if want is None else want
    call = calling(call or show_call(fn, nums))
    arg = list(nums)
    got = getattr(sol, fn)(arg)
    expect(as_list(got), want, call=call)
    expect_true(arg == nums, "不能修改传入的 nums：调用之后列表的内容变了", call=call)


def check_examples(sol, fn):
    for nums, want in EXAMPLES:
        call = calling(show_call(fn, nums))
        expect(as_list(getattr(sol, fn)(list(nums))), want, call=call)


def check_unchanged(sol, fn):
    for nums in ([5, 2, 6, 1], [3, 3, 1, 3, 1, 2], [9, 8, 7, 6, 5, 4, 3, 2, 1], [1, 2, 3, 4], [-5, 7, 0, -5]):
        check(sol, fn, nums)


def check_ties(sol, fn):
    for nums in ([3, 3, 3], [2, 2, 1, 1], [1, 1, 2, 2], [5, 1, 5, 1, 5], [0] * 12, [4, 4, 4, 2, 4, 2, 4, 2]):
        check(sol, fn, nums)
    r = rng("ties")
    for _ in range(150):
        check(sol, fn, ref.random_nums(r, 2, 30, span=r.choice([0, 1, 2])))


def check_random(sol, fn, count, n_hi):
    r = rng()
    for _ in range(count):
        check(sol, fn, ref.random_nums(r, 0, n_hi))


def check_big(sol, fn, nums, want, desc):
    call = calling(f"{fn}({desc})")
    got = getattr(sol, fn)(list(nums))
    expect(as_list(got), want, call=call)


def replay(head, done, show=12):
    """The calls so far as code to paste into 自定义运行; the last one is the step being checked."""
    lines = []
    for k, (x, want) in enumerate(done):
        shown = "抛出 ValueError" if want is ValueError else f"返回 {want}"
        note = f"# <- 这一步，应当{shown}" if k == len(done) - 1 else f"# -> {shown}"
        lines.append(f"c.add({x})".ljust(14) + note)
    skipped = max(0, len(lines) - show)
    out = [head] + ([f"# …（省略了前面 {skipped} 次 add，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(out + lines[skipped:])


def run_online(sol, lo, hi, ops):
    """Feeds ops to one OnlineSmallerCounter and checks every answer, including the ValueErrors."""
    wants = ref.online_brute(lo, hi, ops)
    head = calling(f"c = OnlineSmallerCounter({lo}, {hi})")
    c = sol.OnlineSmallerCounter(lo, hi)
    done = []
    for x, want in zip(ops, wants):
        done.append((x, want))
        call = calling(replay(head, done))
        if want is ValueError:
            expect_raises(ValueError, c.add, x, call=call, msg=f"{x} 不在 [{lo}, {hi}] 里，应当抛出 ValueError")
        else:
            expect(c.add(x), want, call=call)


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子 [5, 2, 6, 1]")
def p1_example_basic(sol):
    expect(as_list(sol.count_smaller_brute([5, 2, 6, 1])), [2, 1, 1, 0], call="count_smaller_brute([5, 2, 6, 1])")


@case("Part 1", sample=True, title="题目例子：相等的 3 不算")
def p1_example_ties(sol):
    expect(as_list(sol.count_smaller_brute([3, 3, 1, 3, 1, 2])), [3, 3, 0, 2, 0, 0],
           call="count_smaller_brute([3, 3, 1, 3, 1, 2])")


@case("Part 1", sample=True, title="题目例子 [4, 4, 4, 2, 4]")
def p1_example_fours(sol):
    expect(as_list(sol.count_smaller_brute([4, 4, 4, 2, 4])), [1, 1, 1, 0, 0],
           call="count_smaller_brute([4, 4, 4, 2, 4])")


@case("Part 1", sample=True, title="题目例子 [2, 0, 1]")
def p1_example_short(sol):
    expect(as_list(sol.count_smaller_brute([2, 0, 1])), [2, 0, 0], call="count_smaller_brute([2, 0, 1])")


@case("Part 1", sample=True, title="题目例子：两个相等的负数")
def p1_example_negative(sol):
    expect(as_list(sol.count_smaller_brute([-1, -1])), [0, 0], call="count_smaller_brute([-1, -1])")


@case("Part 1", sample=True, title="题目例子：一个元素")
def p1_example_single(sol):
    expect(as_list(sol.count_smaller_brute([7])), [0], call="count_smaller_brute([7])")


@case("Part 1", sample=True, title="题目例子：空列表")
def p1_example_empty(sol):
    expect(as_list(sol.count_smaller_brute([])), [], call="count_smaller_brute([])")


@case("Part 1", title="不能修改传入的 nums")
def p1_unchanged(sol):
    check_unchanged(sol, "count_smaller_brute")


@case("Part 1", title="重复元素：相等的不算")
def p1_ties(sol):
    check_ties(sol, "count_smaller_brute")


@case("Part 1", title="负数和绝对值到 10¹⁸ 的整数")
def p1_big_values(sol):
    check(sol, "count_smaller_brute", [10 ** 18, -10 ** 18, 0, 10 ** 18 - 1, -10 ** 18 + 1, 10 ** 18, -1])
    check(sol, "count_smaller_brute", [-3, -1, -2, -5, -4])


@case("Part 1", title="随机数组（对照定义）")
def p1_random(sol):
    check_random(sol, "count_smaller_brute", 300, 40)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目的 7 个例子")
def p2_examples(sol):
    check_examples(sol, "count_smaller_merge")


@case("Part 2", title="不能修改传入的 nums")
def p2_unchanged(sol):
    check_unchanged(sol, "count_smaller_merge")


@case("Part 2", title="重复元素很多：相等的不算")
def p2_ties(sol):
    check_ties(sol, "count_smaller_merge")


@case("Part 2", title="同一份代码也能用于字符串和浮点数")
def p2_other_types(sol):
    check(sol, "count_smaller_merge", ["b", "a", "c", "a"], want=[2, 0, 1, 0])
    check(sol, "count_smaller_merge", [2.5, -1.0, 2.5, 0.0], want=[2, 0, 1, 0])
    check(sol, "count_smaller_merge", ["pear", "apple", "fig", "apple", "kiwi", "fig"])


@case("Part 2", title="元素只支持 <（题目要求只用 < 比较）")
def p2_less_than_only(sol):
    r = rng()
    inputs = [[3, 3, 1, 3, 1, 2], [5, 2, 6, 1], [1], []] + [ref.random_nums(r, 2, 25, span=3) for _ in range(20)]
    for values in inputs:
        want = ref.brute(values)
        keys = [Key(v) for v in values]
        call = calling(f"{KEY_SRC}\n\ncount_smaller_merge({keys!r})")
        expect(as_list(sol.count_smaller_merge(keys)), want, call=call)


@case("Part 2", title="随机数组（对照定义）")
def p2_random(sol):
    check_random(sol, "count_smaller_merge", 400, 60)


@case("Part 2", cpu=6, title="n = 2×10⁵ 随机：要求 O(n log n)")
def p2_large_random(sol):
    check_big(sol, "count_smaller_merge", BIG, big_want(), BIG_DESC)


@case("Part 2", cpu=6, title="n = 2×10⁵ 升序和降序")
def p2_large_sorted(sol):
    check_big(sol, "count_smaller_merge", list(range(N)), [0] * N, "list(range(200000))")
    check_big(sol, "count_smaller_merge", list(range(N, 0, -1)), list(range(N - 1, -1, -1)),
              "list(range(200000, 0, -1))")


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="(a) 题目的 7 个例子")
def p3_examples(sol):
    check_examples(sol, "count_smaller_fenwick")


@case("Part 3", sample=True, title="(b) 题目例子：在线计数")
def p3_online_example(sol):
    steps = [(3, 0), (-2, 0), (3, 1), (5, 3), (0, 1), (6, ValueError)]
    head = calling("c = OnlineSmallerCounter(-2, 5)")
    c = sol.OnlineSmallerCounter(-2, 5)
    done = []
    for x, want in steps:
        done.append((x, want))
        call = calling(replay(head, done))
        if want is ValueError:
            expect_raises(ValueError, c.add, x, call=call, msg="6 不在 [-2, 5] 里，应当抛出 ValueError")
        else:
            expect(c.add(x), want, call=call)


@case("Part 3", title="(a) 不能修改传入的 nums")
def p3_unchanged(sol):
    check_unchanged(sol, "count_smaller_fenwick")


@case("Part 3", title="(a) 重复元素很多：相等的不算")
def p3_ties(sol):
    check_ties(sol, "count_smaller_fenwick")


@case("Part 3", title="(a) 值到 ±10¹⁸：要先做坐标压缩")
def p3_big_values(sol):
    check(sol, "count_smaller_fenwick", [10 ** 18, -10 ** 18, 0, 10 ** 18 - 1, -10 ** 18 + 1, 10 ** 18, -1])
    check(sol, "count_smaller_fenwick", [-10 ** 18, 10 ** 18])
    r = rng()
    for _ in range(40):
        check(sol, "count_smaller_fenwick", ref.random_nums(r, 1, 30, span=10 ** 18))


@case("Part 3", title="(a) 随机数组（对照定义）")
def p3_random(sol):
    check_random(sol, "count_smaller_fenwick", 400, 60)


@case("Part 3", cpu=6, title="(a) n = 2×10⁵：要求 O(n log n)")
def p3_large(sol):
    check_big(sol, "count_smaller_fenwick", BIG, big_want(), BIG_DESC)


@case("Part 3", title="(b) 值正好等于 lo 或 hi")
def p3_online_bounds(sol):
    run_online(sol, 0, 4, [4, 0, 4, 2, 0, 4, 1, 3])
    run_online(sol, -7, -3, [-3, -7, -5, -3, -7])


@case("Part 3", title="(b) 超出范围：抛 ValueError，不记录")
def p3_online_out_of_range(sol):
    run_online(sol, 10, 20, [15, 21, 16, 9, 15, 100, 20, -10, 10, 11, 21])
    run_online(sol, -5, 5, [6, -6, 0, 6, -6, 1])


@case("Part 3", title="(b) lo = hi，以及很大的 lo")
def p3_online_ranges(sol):
    run_online(sol, 3, 3, [3, 3, 2, 3, 4])
    lo = -10 ** 18
    run_online(sol, lo, lo + 999_999, [lo + 5, lo, lo + 999_999, lo + 5, lo - 1, lo + 6, lo + 1_000_000])
    run_online(sol, 10 ** 18, 10 ** 18 + 10, [10 ** 18 + 3, 10 ** 18, 10 ** 18 + 10, 10 ** 18 + 4])


@case("Part 3", title="(b) 随机序列（对照定义）")
def p3_online_random(sol):
    r = rng()
    for _ in range(200):
        lo = r.choice([0, -3, 7, -10 ** 18])
        hi = lo + r.choice([0, 1, 4, 30, 999_999])
        ops = [r.randint(lo - 2, hi + 2) for _ in range(r.randint(0, 25))]
        run_online(sol, lo, hi, ops)


@case("Part 3", cpu=5, title="(b) 2×10⁵ 次 add，值域 10⁶")
def p3_online_large(sol):
    want = ref.fast(STREAM[::-1])[::-1]         # values that arrived earlier and are strictly smaller
    head = "c = OnlineSmallerCounter(0, 999999)"
    calling(f"{head}\n# 依次 add 200000 个 0..999999 的随机整数")
    c = sol.OnlineSmallerCounter(0, 999_999)
    add = c.add
    for k, x in enumerate(STREAM):
        got = add(x)
        if got != want[k]:
            expect(got, want[k], call=f"{head}\n# 前 {k} 次 add 了 0..999999 的随机整数，之前严格小于 {x} 的有 {want[k]} 个\n"
                                       f"c.add({x})   # <- 第 {k + 1} 次 add")
