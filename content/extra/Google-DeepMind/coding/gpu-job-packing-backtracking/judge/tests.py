"""Tests for the GPU job packing problem: Part 1 pack, Part 2 pack_fast, Part 3 min_peak."""
from judgelib import Mismatch, calling, case, expect_valid, rng
import ref

A = [6, 9, 13, 15, 8, 9, 10, 15, 3, 3, 23, 30]
B = [31, 7, 13, 6, 30, 9, 9, 13, 31, 24, 40, 31]

# n = 14 inputs with their smallest peak, checked offline two ways (the bitmask DP in ref.py and the page's
# branch and bound). The "hard" ones are close in size and cap = peak - 1 is just infeasible.
HARD = [
    ([130, 129, 126, 100, 127, 100, 128, 109, 149, 131, 123, 148, 125, 137], 4, 474),
    ([66, 32, 48, 35, 26, 47, 64, 70, 67, 52, 39, 68, 36, 54], 6, 122),
    ([1049, 1100, 1094, 1065, 1016, 1066, 1099, 1071, 1026, 1054, 1007, 1061, 1046, 1072], 10, 2095),
    ([105, 104, 101, 102, 102, 104, 108, 102, 110, 104, 110, 104, 107, 105], 10, 206),
    ([107, 123, 125, 112, 113, 121, 105, 111, 117, 122, 124, 121, 123, 111], 11, 222),
]
C = ([19, 19, 27, 27, 17, 16, 32, 34, 36, 28, 17, 29], 5, 62)       # greedy 69, lower bound 61, best 62
FIXED = [
    C,
    ([124907, 335004, 526926, 997555, 536686, 678637, 901776, 999242, 107623, 927067, 234131, 938274, 630227,
      651674], 2, 4294941),
    ([85, 76, 86, 85, 81, 87, 78, 65, 50, 89, 55, 57, 68, 56], 3, 340),
    ([29, 1, 32, 21, 14, 26, 17, 23, 23, 25, 33, 5, 22, 6], 4, 70),
    ([371, 368, 337, 336, 358, 318, 382, 390, 391, 373, 339, 303, 390, 347], 5, 1057),
    ([47, 60, 55, 12, 52, 75, 71, 64, 15, 55, 65, 97, 77, 63], 6, 144),
    ([517488, 650138, 371207, 985499, 540366, 691396, 602467, 637866, 643148, 944285, 126608, 703472, 335431,
      903027], 7, 1281014),
    ([1002, 1000, 1008, 1010, 1001, 1009, 1006, 1007, 1002, 1007, 1003, 1001, 1010, 1007], 3, 5018),
]


# ---------------------------------------------------------------- how a call is shown

def _lit(jobs):
    return "[" + ", ".join(map(str, jobs)) + "]"


def _text(fn, jobs, k, cap=None, note="", stats=False, name=None):
    """Code that reproduces one call. Long lists get a name on a line of their own."""
    if name is None and len(jobs) >= 10:
        name = "jobs"
    head = [f"{name} = {_lit(jobs)}"] if name else []
    args = [name or _lit(jobs), str(k)] + ([str(cap)] if cap is not None else [])
    if stats:
        head.append("stats = {}")
        args.append("stats")
    line = f"print({fn}({', '.join(args)}))"
    return "\n".join(head + [line + (f"   # {note}" if note else "")])


def _as_list(x):
    return list(x) if isinstance(x, tuple) else x


def _nodes_problem(stats, fn):
    """None when the function wrote a node count into stats, else what is wrong."""
    if "nodes" not in stats:
        return f'{fn} 返回前应当把节点数写进 stats["nodes"]，但 stats 里没有 "nodes"（stats = {stats!r}）'
    x = stats["nodes"]
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        return f'stats["nodes"] 应当是一个非负整数（递归函数被调用的次数），得到 {x!r}'
    return None


# ---------------------------------------------------------------- the three checks

def _check_pack(sol, jobs, k, cap, want, name=None):
    """Part 1: the answer is exactly the lexicographically smallest valid assignment (or None)."""
    call = calling(_text("pack", jobs, k, cap, f"应当是 {want}", name=name))
    got = _as_list(sol.pack(list(jobs), k, cap))
    if want is None:
        if got is not None:
            raise Mismatch("这个输入没有合法方案，应当返回 None", call, want, got)
        return
    if got is None:
        raise Mismatch(f"存在合法方案（字典序最小的是 {want}），却返回了 None", call, want, got)
    problem = ref.assign_problem(jobs, k, got, cap)
    if problem:
        raise Mismatch(problem, call, want, got)
    if got != want:
        raise Mismatch("返回的方案合法，但不是字典序最小的：先比 assign[0]，相同再比 assign[1]，依此类推", call, want, got)


def _check_fast(sol, jobs, k, cap, ok, name=None, example=None):
    """Part 2: any valid assignment when one exists (ok), None otherwise."""
    call = calling(_text("pack_fast", jobs, k, cap, "应当返回一个合法方案" if ok else "应当是 None", name=name))
    got = sol.pack_fast(list(jobs), k, cap)
    if ok:
        expect_valid(got, lambda g: "存在合法方案，却返回了 None" if g is None else ref.assign_problem(jobs, k, g, cap),
                     call, want=example if example is not None else "任意一个合法方案")
    elif got is not None:
        problem = ref.assign_problem(jobs, k, got, cap)
        raise Mismatch("这个输入没有合法方案，应当返回 None" + (f"；返回的方案不合法：{problem}" if problem else ""),
                       call, None, got)


def _peak_problem(jobs, k, want):
    def check(got):
        if not isinstance(got, (tuple, list)) or len(got) != 2:
            return "应当返回一个二元组 (最小峰值, assign)"
        peak, assign = got
        if isinstance(peak, bool) or peak != want:
            more = "，还有峰值更小的方案" if isinstance(peak, (int, float)) and peak > want else ""
            return f"最小峰值应当是 {want}，得到 {peak!r}{more}"
        problem = ref.assign_problem(jobs, k, assign)
        if problem:
            return problem
        loads = ref.loads_of(jobs, k, assign)
        if max(loads, default=0) != want:
            return f"返回的峰值是 {want}，但 assign 里各块 GPU 的占用是 {loads}，最大的是 {max(loads, default=0)}"
        return None
    return check


def _check_peak(sol, jobs, k, want, name=None):
    """Part 3: the smallest peak, with an assignment whose busiest GPU is exactly that peak."""
    call = calling(_text("min_peak", jobs, k, note=f"峰值应当是 {want}", name=name))
    got = sol.min_peak(list(jobs), k)
    expect_valid(got, _peak_problem(jobs, k, want), call, want=f"({want}, 任意一个峰值为 {want} 的 assign)")


def _check_stats(sol, fn, jobs, k, cap=None):
    """The function writes a node count into stats before it returns, on every path."""
    args = (list(jobs), k) + ((cap,) if cap is not None else ())
    call = calling(_text(fn, jobs, k, cap, stats=True) + '\nprint(stats["nodes"])   # 应当是一个非负整数')
    stats = {}
    got = getattr(sol, fn)(*args, stats)
    problem = _nodes_problem(stats, fn)
    if problem:
        raise Mismatch(problem, call, "一个非负整数", stats.get("nodes", "（没有写）"))
    return got


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子：k = 2, cap = 7")
def p1_example1(sol):
    _check_pack(sol, [3, 5, 2, 4], 2, 7, [0, 1, 1, 0])


@case("Part 1", sample=True, title="题目例子：k = 3, cap = 5")
def p1_example2(sol):
    _check_pack(sol, [3, 5, 2, 4], 3, 5, [0, 1, 0, 2])


@case("Part 1", sample=True, title="题目例子：总量超过 k × cap")
def p1_example3(sol):
    _check_pack(sol, [3, 5, 2, 4], 2, 6, None)


@case("Part 1", sample=True, title="题目例子：单个作业放不下")
def p1_example4(sol):
    _check_pack(sol, [8], 2, 7, None)


@case("Part 1", sample=True, title="题目例子：没有作业")
def p1_example5(sol):
    _check_pack(sol, [], 3, 1, [])


@case("Part 1", title="刚好装满、一块 GPU、k > n")
def p1_edge(sol):
    for jobs, k, cap in (([5], 1, 5), ([5], 1, 4), ([2, 3], 1, 5), ([2, 3], 1, 4), ([1, 1, 1], 5, 1),
                         ([4, 4, 4, 4], 2, 8), ([7, 1, 1], 3, 7), ([], 1, 1), ([10 ** 6, 10 ** 6], 2, 10 ** 6),
                         ([10 ** 6] * 3, 1, 10 ** 7), ([1] * 8, 2, 4), ([1] * 8, 2, 3)):
        _check_pack(sol, jobs, k, cap, ref.brute_first(jobs, k, cap))


@case("Part 1", title="合法但不是字典序最小")
def p1_lex(sol):
    # inputs where trying the emptiest GPU first, or placing large jobs first, finds another valid plan first
    for jobs, k, cap in (([1, 1, 1], 2, 2), ([2, 1, 1], 2, 2), ([1, 2, 1, 2], 2, 3), ([3, 1, 2, 2], 2, 4),
                         ([1, 1, 2, 2], 3, 2), ([2, 3, 1, 4, 2], 3, 5), ([1, 1, 1, 1, 1, 1], 3, 2)):
        _check_pack(sol, jobs, k, cap, ref.brute_first(jobs, k, cap))


@case("Part 1", title="走不通时要撤销再试下一块")
def p1_backtrack(sol):
    # the first GPU that fits is a dead end, so the search has to take the job back off and try another
    for jobs, k, cap in (([1, 2, 3, 2], 2, 4), ([2, 2, 3, 3], 2, 5), ([4, 1, 2, 3, 2], 3, 4), ([5, 1, 1, 4, 3, 2], 2, 8),
                         ([8, 4, 2, 7, 8, 3], 2, 16), ([5, 7, 6, 1, 8, 6, 3], 3, 12)):
        _check_pack(sol, jobs, k, cap, ref.brute_first(jobs, k, cap))


@case("Part 1", title="随机小输入（对照穷举）")
def p1_random(sol):
    r = rng()
    for _ in range(120):
        k = r.randint(1, 4)
        n = r.randint(0, {1: 8, 2: 8, 3: 6, 4: 5}[k])
        jobs = [r.randint(1, r.choice([3, 9, 30, 10 ** 6])) for _ in range(n)]
        lo = max(jobs, default=1)
        cap = r.randint(max(1, lo - 1), max(lo, -(-sum(jobs) // k) + 3))
        _check_pack(sol, jobs, k, cap, ref.brute_first(jobs, k, cap))


@case("Part 1", title='写入 stats["nodes"]')
def p1_stats(sol):
    for jobs, k, cap in (([3, 5, 2, 4], 2, 7), ([3, 5, 2, 4], 2, 6), ([8], 2, 7), ([], 3, 1)):
        got = _as_list(_check_stats(sol, "pack", jobs, k, cap))
        want = ref.brute_first(jobs, k, cap)
        if got != want:
            raise Mismatch("传入 stats 时，返回值也应当是字典序最小的合法方案（没有合法方案时是 None）",
                           _text("pack", jobs, k, cap, stats=True), want, got)


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子 A：有解")
def p2_example_a(sol):
    _check_fast(sol, A, 4, 36, True, name="A", example=[0, 3, 1, 2, 3, 3, 3, 2, 2, 2, 1, 0])


@case("Part 2", sample=True, title="题目例子 B：无解")
def p2_example_b(sol):
    _check_fast(sol, B, 4, 61, False, name="B")


@case("Part 2", title="A、B 上的节点数不超过 1%")
def p2_nodes(sol):
    for jobs, name, cap, ok, part1 in ((A, "A", 36, True, 66005), (B, "B", 61, False, 350105)):
        limit = part1 // 100
        call = calling(_text("pack_fast", jobs, 4, cap, stats=True, name=name) +
                       f'\nprint(stats["nodes"])   # 应当不超过 {part1} 的百分之一，即 {limit}')
        stats = {}
        got = sol.pack_fast(list(jobs), 4, cap, stats)
        problem = (ref.assign_problem(jobs, 4, got, cap) if got is not None else "存在合法方案，却返回了 None") if ok \
            else (None if got is None else "这个输入没有合法方案，应当返回 None")
        if problem:
            raise Mismatch(problem, call, "合法方案" if ok else None, got)
        problem = _nodes_problem(stats, "pack_fast")
        if problem:
            raise Mismatch(problem, call, f"不超过 {limit}", stats.get("nodes", "（没有写）"))
        if stats["nodes"] > limit:
            raise Mismatch(f"{name} 上访问了 {stats['nodes']} 个节点，题目要求不超过 Part 1 写法（{part1} 个）的百分之一，"
                           f"即 {limit} 个", call, f"不超过 {limit}", stats["nodes"])


@case("Part 2", title="Part 1 的例子")
def p2_small(sol):
    for jobs, k, cap in (([3, 5, 2, 4], 2, 7), ([3, 5, 2, 4], 3, 5), ([3, 5, 2, 4], 2, 6), ([8], 2, 7), ([], 3, 1)):
        _check_fast(sol, jobs, k, cap, ref.feasible(jobs, k, cap))


@case("Part 2", title="空输入、刚好装满、k = 1")
def p2_edge(sol):
    # (jobs, k, cap, has a valid plan), checked offline with the DP in ref.py
    for jobs, k, cap, ok in (([], 1, 1, True), ([], 14, 10 ** 7, True), ([5], 1, 5, True), ([5], 3, 4, False),
                             ([2, 3], 1, 5, True), ([2, 3], 1, 4, False), ([1] * 14, 14, 1, True),
                             ([1] * 14, 13, 1, False), ([7, 1, 1, 1], 2, 6, False),
                             ([10 ** 6] * 14, 2, 7 * 10 ** 6, True), ([10 ** 6] * 14, 2, 7 * 10 ** 6 - 1, False),
                             ([10 ** 6] * 14, 1, 10 ** 7, False), ([5] * 14, 4, 19, False), ([5] * 14, 4, 20, True),
                             ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 3, 35, True)):
        _check_fast(sol, jobs, k, cap, ok)


@case("Part 2", title='每条返回路径都写 stats["nodes"]')
def p2_stats(sol):
    # found; searched and failed; total > k * cap; one job > cap; no jobs
    for jobs, k, cap, ok in ((A, 4, 36, True), (B, 4, 61, False), ([3, 5, 2, 4], 2, 6, False), ([8], 2, 7, False),
                             ([], 3, 1, True)):
        got = _check_stats(sol, "pack_fast", jobs, k, cap)
        problem = (ref.assign_problem(jobs, k, got, cap) if got is not None else "存在合法方案，却返回了 None") if ok \
            else (None if got is None else "这个输入没有合法方案，应当返回 None")
        if problem:
            raise Mismatch(problem, _text("pack_fast", jobs, k, cap, stats=True), "合法方案" if ok else None, got)


@case("Part 2", title="随机输入（对照状态压缩 DP）")
def p2_random(sol):
    r = rng()
    for _ in range(30):
        k = r.randint(1, 5)
        n = r.randint(0, 10)
        jobs = [r.randint(1, r.choice([10, 40, 1000, 10 ** 6])) for _ in range(n)]
        best = ref.dp_peak(jobs, k)
        for cap in (best - 1, best, best + r.randint(1, max(1, best // 10))):
            if cap >= 1:
                _check_fast(sol, jobs, k, cap, cap >= best)


@case("Part 2", title="cap 等于最优峰值和少 1")
def p2_fixed(sol):
    for jobs, k, best in FIXED:
        _check_fast(sol, jobs, k, best - 1, False)
        _check_fast(sol, jobs, k, best, True)


@case("Part 2", cpu=2, title="n = 14 大小相近、刚好装不下")
def p2_hard(sol):
    for jobs, k, best in HARD:
        _check_fast(sol, jobs, k, best - 1, False)
        _check_fast(sol, jobs, k, best, True)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子 [3, 3, 2, 2, 2]")
def p3_example1(sol):
    _check_peak(sol, [3, 3, 2, 2, 2], 2, 6)


@case("Part 3", sample=True, title="题目例子 [3, 5, 2, 4]")
def p3_example2(sol):
    _check_peak(sol, [3, 5, 2, 4], 3, 5)


@case("Part 3", sample=True, title="题目例子 A：峰值 36")
def p3_example_a(sol):
    _check_peak(sol, A, 4, 36, name="A")


@case("Part 3", sample=True, title="题目例子 B：峰值 62")
def p3_example_b(sol):
    _check_peak(sol, B, 4, 62, name="B")


@case("Part 3", sample=True, title="题目例子：没有作业")
def p3_example5(sol):
    _check_peak(sol, [], 2, 0)


@case("Part 3", title="空输入、k = 1、k 不小于 n")
def p3_edge(sol):
    for jobs, k, best in (([], 1, 0), ([], 14, 0), ([7], 1, 7), ([7], 14, 7), ([2, 3, 4], 1, 9), ([5, 3], 4, 5),
                          ([7] * 6, 3, 14), ([7] * 7, 3, 21), ([10 ** 6] * 14, 14, 10 ** 6),
                          ([10 ** 6] * 14, 1, 14 * 10 ** 6), ([1, 1, 1, 1, 9], 2, 9)):
        _check_peak(sol, jobs, k, best)


@case("Part 3", title="贪心不是最优")
def p3_greedy(sol):
    # largest-first onto the emptiest GPU (LPT) gives a larger peak on each of these
    for jobs, k, best in (([3, 3, 2, 2, 4], 2, 7), ([6, 7, 4, 9, 4], 2, 15), ([5, 5, 4, 4, 3, 3, 3], 3, 9),
                          ([8, 7, 6, 5, 4], 2, 15), FIXED[2], FIXED[3]):
        _check_peak(sol, jobs, k, best)


@case("Part 3", title="最优峰值大于下界")
def p3_above_lower(sol):
    # the best peak is above max(max job, ceil(total / k)), so the search cannot stop at the bound
    for jobs, k, best in ((C[0], C[1], C[2]), ([4, 5, 5, 4, 6], 2, 13), FIXED[4], FIXED[7], FIXED[1]):
        _check_peak(sol, jobs, k, best)


@case("Part 3", title="随机输入（对照状态压缩 DP）")
def p3_random(sol):
    r = rng()
    for _ in range(40):
        k = r.randint(1, 5)
        n = r.randint(0, 10)
        jobs = [r.randint(1, r.choice([10, 40, 1000, 10 ** 6])) for _ in range(n)]
        _check_peak(sol, jobs, k, ref.dp_peak(jobs, k))


@case("Part 3", title='写入 stats["nodes"]')
def p3_stats(sol):
    # the greedy already reaches the lower bound; the search improves on it; no jobs
    for jobs, k, best in (([5, 3], 4, 5), ([3, 3, 2, 2, 2], 2, 6), ([], 2, 0)):
        got = _check_stats(sol, "min_peak", jobs, k)
        expect_valid(got, _peak_problem(jobs, k, best), _text("min_peak", jobs, k, stats=True), want=best)


@case("Part 3", cpu=2, title="n = 14 大小全相同")
def p3_equal(sol):
    # the greedy plan's peak is above the lower bound, so the search has to prove that no smaller peak exists;
    # without Part 2's equal-load pruning that means trying every way to permute identical GPUs
    for jobs, k, best in (([7] * 14, 6, 21), ([10 ** 6] * 14, 13, 2 * 10 ** 6)):
        _check_peak(sol, jobs, k, best)


@case("Part 3", cpu=2, title="n = 14 难的输入：要用剪枝")
def p3_hard(sol):
    # The page's answer needs well under 0.1 s here. A search with sorting and equal-load pruning but no bound,
    # or with the bound but no equal-load pruning, needs more than 10 s. The other HARD inputs are left out on
    # purpose: a branch and bound that starts without a greedy plan needs 1 to 2.5 s on them, too close to the
    # statement's "about a second" to call either way.
    for jobs, k, best in (HARD[0], HARD[3]):
        _check_peak(sol, jobs, k, best)
