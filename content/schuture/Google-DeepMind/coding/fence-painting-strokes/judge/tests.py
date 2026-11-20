from judgelib import calling, case, expect, expect_true, rng, show_call
import ref

SHOW = 24           # a Fence replay lists at most this many lines


# ---------------------------------------------------------------- Part 1 and Part 3: plain functions

def check_horizontal(sol, h, want=None, call=None):
    want = ref.horizontal_by_ends(h) if want is None else want
    call = calling(call or show_call("min_horizontal_strokes", h))
    expect(sol.min_horizontal_strokes(list(h)), want, call=call)


def check_strokes(sol, h, want, call=None):
    call = calling(call or show_call("min_strokes", h))
    expect(sol.min_strokes(list(h)), want, call=call)


# ---------------------------------------------------------------- Part 2: a Fence driven step by step

def _replay(lines, mark):
    """The steps so far as code to paste into 自定义运行; the last one is the current step."""
    lines = list(lines)
    code, note = lines[-1]
    lines[-1] = (code, f"{note}   {mark}".strip())
    skipped = max(0, len(lines) - SHOW)
    lines = lines[skipped:]
    width = max(len(c) for c, _ in lines) + 3
    head = [f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else []
    return "\n".join(head + [(c.ljust(width) + n).rstrip() for c, n in lines])


def _method(obj, name):
    fn = getattr(obj, name, None)
    expect_true(callable(fn), f"Fence 没有 {name} 方法（方法名要和题目一致）")
    return fn


def run_fence(sol, h, steps, start=None):
    """steps: ("set", i, v) or ("strokes", want), want None meaning recompute it from the current heights.
    Checks every strokes() as soon as it returns."""
    cur = list(h)
    lines = [(start or f"f = Fence({h!r})", "")]
    calling(_replay(lines, "# <- 这一步"))
    f = sol.Fence(list(h))
    for step in steps:
        if step[0] == "set":
            _, i, v = step
            cur[i] = v
            lines.append((f"f.set_height({i}, {v})", f"# h 变为 {cur}" if len(cur) <= 12 else ""))
            fn = _method(f, "set_height")
            calling(_replay(lines, "# <- 这一步"))
            fn(i, v)
        else:
            want = ref.horizontal_by_ends(cur) if step[1] is None else step[1]
            lines.append(("print(f.strokes())", f"# -> {want}"))
            fn = _method(f, "strokes")
            calling(_replay(lines, "# <- 这一步"))
            got = fn()
            if got != want:
                expect(got, want, call=_replay(lines, "# <- 这里不对"),
                       msg=f"strokes() 不对：当前高度下 min_horizontal_strokes 的结果是 {want}")


# ---------------------------------------------------------------- big inputs, built once at import time

def _big_part2():
    r = rng("part2-big")
    n = q = 50000
    h = [r.randint(0, 10 ** 9) for _ in range(n)]
    ops = []
    for _ in range(q):
        i = r.choice([0, n - 1, r.randrange(n)])
        ops.append((i, r.choice([0, r.randint(0, 10 ** 9)])))
    # the expected strokes after each update, by re-evaluating the two terms that read h[i]
    term = lambda hs, k: max(0, hs[k] - (hs[k - 1] if k > 0 else 0)) if 0 <= k < n else 0
    cur, total, wants = list(h), ref.horizontal_by_ends(h), []
    for i, v in ops:
        total -= term(cur, i) + term(cur, i + 1)
        cur[i] = v
        total += term(cur, i) + term(cur, i + 1)
        wants.append(total)
    assert total == ref.horizontal_by_ends(cur)
    return h, ops, wants


BIG_FENCE, BIG_OPS, BIG_WANTS = _big_part2()
LONG_H = [rng("part1-long").randint(0, 10 ** 9) for _ in range(100000)]


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子 [2, 1, 3, 3, 1, 2]")
def p1_example(sol):
    check_horizontal(sol, [2, 1, 3, 3, 1, 2], 5)


@case("Part 1", title="没有木板")
def p1_empty(sol):
    check_horizontal(sol, [], 0)


@case("Part 1", title="全是高度 0 的木板")
def p1_zero_planks(sol):
    for h, want in (([0], 0), ([0, 0, 0], 0), ([0, 0, 4], 4), ([4, 0, 0], 4)):
        check_horizontal(sol, h, want)


@case("Part 1", title="高度 0 的木板把每一行截断")
def p1_zero_splits_rows(sol):
    for h, want in (([1, 0, 1], 2), ([3, 0, 3], 6), ([0, 4, 0, 4, 0], 8), ([2, 2, 0, 2, 2], 4), ([5, 0, 0, 2], 7)):
        check_horizontal(sol, h, want)


@case("Part 1", title="单块木板、单调和尖峰")
def p1_simple_shapes(sol):
    for h, want in (([7], 7), ([3, 1], 3), ([4, 4, 4], 4), ([1, 2, 3], 3), ([3, 2, 1], 3), ([1, 5, 1], 5)):
        check_horizontal(sol, h, want)


@case("Part 1", title="随机栅栏，与逐行数段对照")
def p1_random(sol):
    r = rng()
    for _ in range(300):
        h = ref.random_heights(r, 40, 12)
        check_horizontal(sol, h, ref.horizontal_by_rows(h))


@case("Part 1", title="高度达到 10⁹ 以上")
def p1_huge_heights(sol):
    # 2 ** 60 + 1 and 10 ** 18 + 1 are not floats: an answer computed with / or float() loses the + 1
    for h, want in (([10 ** 9, 1, 10 ** 9], 2 * 10 ** 9 - 1), ([10 ** 18], 10 ** 18),
                    ([0, 10 ** 12, 10 ** 12 - 1, 10 ** 12], 10 ** 12 + 1), ([10 ** 18 + 1], 10 ** 18 + 1),
                    ([2 ** 60 + 1, 1, 2 ** 60 + 1], 2 ** 61 + 1)):
        check_horizontal(sol, h, want)


@case("Part 1", title="10⁵ 块木板")
def p1_long(sol):
    check_horizontal(sol, LONG_H, call="min_horizontal_strokes(<100000 块随机高度的木板，高度不超过 10⁹>)")


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子 Fence([1, 1, 1])")
def p2_example(sol):
    run_fence(sol, [1, 1, 1], [
        ("strokes", 1),
        ("set", 1, 3), ("strokes", 3),
        ("set", 0, 3), ("strokes", 3),
        ("set", 1, 0), ("strokes", 4),
    ])


@case("Part 2", title="没有木板的栅栏")
def p2_empty(sol):
    run_fence(sol, [], [("strokes", 0), ("strokes", 0)])


@case("Part 2", title="改第 0 块木板（左边没有邻居）")
def p2_first_plank(sol):
    run_fence(sol, [1, 2, 5], [
        ("strokes", 5),
        ("set", 0, 4), ("strokes", 7),
        ("set", 0, 6), ("strokes", 9),
        ("set", 0, 0), ("strokes", 5),
    ])


@case("Part 2", title="改最后一块木板")
def p2_last_plank(sol):
    run_fence(sol, [2, 2, 2], [
        ("strokes", 2),
        ("set", 2, 5), ("strokes", 5),
        ("set", 2, 1), ("strokes", 2),
        ("set", 2, 0), ("strokes", 2),
    ])


@case("Part 2", title="只有一块木板")
def p2_single_plank(sol):
    run_fence(sol, [4], [
        ("strokes", 4),
        ("set", 0, 2), ("strokes", 2),
        ("set", 0, 0), ("strokes", 0),
        ("set", 0, 7), ("strokes", 7),
    ])


@case("Part 2", title="降到 0 再升回来")
def p2_zero_and_back(sol):
    run_fence(sol, [3, 3, 3], [
        ("strokes", 3),
        ("set", 1, 0), ("strokes", 6),
        ("set", 1, 3), ("strokes", 3),
        ("set", 1, 1), ("strokes", 5),
        ("set", 0, 0), ("set", 2, 0), ("strokes", 1),
    ])


@case("Part 2", title="高度不变的修改、连续查询")
def p2_same_height(sol):
    run_fence(sol, [2, 5, 1], [
        ("strokes", 5), ("strokes", 5),
        ("set", 1, 5), ("strokes", 5),
        ("set", 0, 2), ("set", 2, 1), ("strokes", 5), ("strokes", 5),
    ])


@case("Part 2", title="随机修改序列，与重新计算对照")
def p2_random(sol):
    r = rng()
    runs = []
    for _ in range(300):
        h = ref.random_heights(r, 8, 6)
        if not h:
            h = [r.randint(0, 6)]
        steps = [("strokes", None)]
        for _ in range(r.randint(1, 10)):
            steps.append(("set", r.randrange(len(h)), r.choice([0, r.randint(0, 6)])))
            if r.random() < 0.8:
                steps.append(("strokes", None))
        steps.append(("strokes", None))
        runs.append((h, steps))
    runs.sort(key=lambda hs: (len(hs[0]), len(hs[1])))      # the first failure shown is a small one
    for h, steps in runs:
        run_fence(sol, h, steps)


@case("Part 2", title="5 万块木板、5 万次修改：要求 O(1)")
def p2_many_updates(sol):
    n = len(BIG_FENCE)
    start = f"f = Fence(<{n} 块随机高度的木板>)"
    calling(start + "   # <- 这一步")
    f = sol.Fence(list(BIG_FENCE))
    set_height, strokes = _method(f, "set_height"), _method(f, "strokes")
    calling(f"{start}\n# 之后 {len(BIG_OPS)} 次 f.set_height(i, v)，每次之后调用 f.strokes()；运行到这里超时或出错")
    for k, ((i, v), want) in enumerate(zip(BIG_OPS, BIG_WANTS)):
        set_height(i, v)
        got = strokes()
        if got != want:
            expect(got, want, call=f"{start}\n# …前面 {k} 次修改都对\nf.set_height({i}, {v})\nprint(f.strokes())   # <- 这里不对",
                   msg=f"第 {k + 1} 次修改之后 strokes() 不对")


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子 [1, 5, 1]")
def p3_example_spike(sol):
    check_strokes(sol, [1, 5, 1], 2)


@case("Part 3", sample=True, title="题目例子 [2, 1, 3, 3, 1, 2]")
def p3_example_rows(sol):
    check_strokes(sol, [2, 1, 3, 3, 1, 2], 5)


@case("Part 3", title="空栅栏和高度为 0 的木板")
def p3_empty_and_zeros(sol):
    for h, want in (([], 0), ([0], 0), ([0, 0, 0], 0), ([0, 1, 0], 1), ([1, 0, 1], 2), ([0, 2, 2, 0, 1], 3)):
        check_strokes(sol, h, want)


@case("Part 3", title="整段都竖着刷最省")
def p3_all_vertical(sol):
    for h, want in (([5], 1), ([5, 5], 2), ([9, 9, 9], 3), ([4, 7], 2), ([3, 1000, 3], 3), ([6, 6, 0, 1], 3)):
        check_strokes(sol, h, want)


@case("Part 3", title="整行横刷最省")
def p3_all_horizontal(sol):
    for h, want in (([1, 1, 1, 1], 1), ([2] * 10, 2), ([3, 3, 3, 3, 3], 3), ([2, 3, 2, 3, 2], 4)):
        check_strokes(sol, h, want)


@case("Part 3", title="子区间从新的基准往上算")
def p3_nested_base(sol):
    for h, want in (([1, 3, 3, 3, 1], 3), ([2, 5, 5, 5, 5, 2], 5), ([1, 4, 4, 4, 4, 4, 1], 4),
                    ([1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1], 5)):
        check_strokes(sol, h, want)


@case("Part 3", title="横笔画和竖笔画混用")
def p3_mixed(sol):
    for h, want in (([1, 5, 1, 5, 1], 3), ([1, 2, 1, 2, 1, 2, 1], 4), ([2, 6, 2, 2, 6, 6, 2], 5),
                    ([1, 9, 2, 9, 2, 9, 1], 5)):
        check_strokes(sol, h, want)


@case("Part 3", title="随机小栅栏，与穷举对照")
def p3_brute(sol):
    r = rng()
    for _ in range(400):
        h = ref.random_heights(r, 10, 5)
        check_strokes(sol, h, ref.brute_min_strokes(h))


@case("Part 3", title="随机栅栏，与二分图匹配对照")
def p3_matching(sol):
    r = rng()
    for _ in range(200):
        h = ref.random_heights(r, 60, 40)
        check_strokes(sol, h, ref.matching_min_strokes(h))


@case("Part 3", title="高度达到 10⁹ 以上")
def p3_huge_heights(sol):
    for h, want in (([10 ** 9, 1, 10 ** 9], 3), ([10 ** 12] * 5, 5), ([1, 10 ** 9, 10 ** 9, 1], 3),
                    ([10 ** 18, 0, 10 ** 18], 2)):
        check_strokes(sol, h, want)


@case("Part 3", title="单调的 1500 块木板（最坏情形）")
def p3_sorted(sol):
    for h, text in ((list(range(1, 1501)), "list(range(1, 1501))"),
                    (list(range(1500, 0, -1)), "list(range(1500, 0, -1))"),
                    (list(range(750, 0, -1)) + list(range(2, 751)), "list(range(750, 0, -1)) + list(range(2, 751))")):
        check_strokes(sol, h, len(h), call=f"min_strokes({text})")


@case("Part 3", title="几千块木板的随机栅栏")
def p3_large(sol):
    r = rng()
    for _ in range(4):
        n = r.randint(2000, 4000)
        h = [r.randint(0, 10 ** 9) if r.random() < 0.9 else r.randint(0, 3) for _ in range(n)]
        check_strokes(sol, h, ref.split_min_strokes(h), call=f"min_strokes(<{n} 块随机高度的木板>)")
