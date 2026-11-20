"""answer.py defines p1 ... p5: one program (text in the checker's syntax) per puzzle. Every program is parsed and
run with the statement's own checker (ref.parse / ref.run); the answer cannot bring its own interpreter."""
import itertools

from judgelib import Mismatch, case, expect, info, rng
import ref
from ref import MAX_LEN, MAX_STEPS, PUZZLES, parse, run     # bound now, before the answer is loaded

MARKS = {"p1": (6, 3), "p2": (8, 5), "p3": (7, 5), "p4": (8, 6), "p5": (11, 7)}   # (及格线, 目标) from the statement


def program(sol, name):
    """The answer's program for one puzzle, parsed by the checker. Fails with a readable message if it is not
    a valid program (this is where the hard limit of 1..12 instructions is enforced, by the checker's parse)."""
    src = getattr(sol, name)
    if not isinstance(src, str):
        raise Mismatch(f"{name} 应当是一段程序文本（str），得到的是 {type(src).__name__}", name, None, src)
    count = sum(1 for line in src.replace(";", "\n").splitlines() if line.split("#")[0].strip())
    if count == 0:
        raise Mismatch(f"{name} 还是空的：还没有写这道题的程序", f"parse({name})", show=False)
    try:
        return parse(src)
    except ValueError as e:
        text = f"parse 报错：ValueError: {e}"
        if str(e).startswith("a program has"):
            why = f"程序有 {count} 条指令，超过上限 {MAX_LEN} 条（硬性限制，超过就无效）"
        elif str(e).startswith("bad instruction"):
            why = "有一条指令写错了：指令名要大写，只有 JMP 带一个参数，前缀只能是 + 或 -"
        elif str(e).startswith("jump target"):
            why = f"JMP 的目标超出范围：要求 0 ≤ k < 程序长度（{count}）"
        else:
            why = "程序无法解析"
        raise Mismatch(f"{why}（{text}）", f"parse({name})", show=False) from None


def report_count(name, prog):
    mark, target = MARKS[name]
    info(f"指令数 {len(prog)}（及格线 {mark}，目标 {target}）")


def check(name, prog, xs, want):
    """One run on the checker's machine; wrong output or no halt within MAX_STEPS is a wrong answer."""
    call = f"run(parse({name}), {xs!r})"
    got = run(prog, xs)
    if got is None:
        raise Mismatch(f"跑满 {MAX_STEPS:,} 步还没有停机（每个测试最多 {MAX_STEPS:,} 步，超过就无效）", call, want, None)
    expect(got, want, call=call)


def check_all(sol, name, inputs):
    prog = program(sol, name)
    spec = PUZZLES[name][0]
    for xs in inputs:
        check(name, prog, list(xs), spec(list(xs)))


def check_random(sol, name, count, max_len):
    r = rng()
    inputs = [ref.random_input(r, name, r.randint(0, max_len)) for _ in range(count)]
    check_all(sol, name, sorted(inputs, key=len))      # shortest first, so a failure shows a short input


def check_worst(sol, name):
    """Part 4 and 5: eight 99s is the input that needs the most steps; report how many."""
    prog = program(sol, name)
    xs = [99] * 8
    steps = ref.count_steps(prog, xs)
    if steps is not None:
        info(f"输入 8 个 99：共 {steps:,} 步（上限 {MAX_STEPS:,}）")
    check(name, prog, xs, PUZZLES[name][0](xs))


def all_short(values, longest):
    return [list(t) for k in range(longest + 1) for t in itertools.product(values, repeat=k)]


SHORT = {"p1": all_short(range(-2, 3), 6), "p2": all_short(range(-2, 3), 6), "p3": all_short(range(0, 4), 7),
         "p4": all_short(range(0, 5), 5), "p5": all_short(range(0, 5), 5)}


# ---------------------------------------------------------------- Part 1 零之后
@case("Part 1", sample=True, title="题目例子 5 0 3 0 -2 7")
def p1_example_1(sol):
    prog = program(sol, "p1")
    report_count("p1", prog)
    check("p1", prog, [5, 0, 3, 0, -2, 7], [3, 0, -2, 7])


@case("Part 1", sample=True, title="题目例子 1 2 3（没有 0）")
def p1_example_2(sol):
    check("p1", program(sol, "p1"), [1, 2, 3], [])


@case("Part 1", sample=True, title="题目例子 0 0")
def p1_example_3(sol):
    check("p1", program(sol, "p1"), [0, 0], [0])


@case("Part 1", sample=True, title="题目例子 -4 0（0 在最后）")
def p1_example_4(sol):
    check("p1", program(sol, "p1"), [-4, 0], [])


@case("Part 1", title="检查器自带的 8 组测试")
def p1_checker_tests(sol):
    check_all(sol, "p1", PUZZLES["p1"][1])


@case("Part 1", title="空输入和只有一个数")
def p1_tiny(sol):
    check_all(sol, "p1", [[], [0], [5], [-99]])


@case("Part 1", title="±99 和 0 混在一起")
def p1_extremes(sol):
    check_all(sol, "p1", [[99, 0, 99, -99, 0, -99], [-99, 99, 0], [0, -99, 99, 0, 0], [1, -1, 0, 1, -1, 99]])


@case("Part 1", title="所有短输入（−2..2，长度 ≤ 6）")
def p1_all_short(sol):
    check_all(sol, "p1", SHORT["p1"])


@case("Part 1", title="随机输入（−99..99，长度 ≤ 24）")
def p1_random(sol):
    check_random(sol, "p1", 800, 24)


# ---------------------------------------------------------------- Part 2 分组求和
@case("Part 2", sample=True, title="题目例子 3 4 0 5 0")
def p2_example_1(sol):
    prog = program(sol, "p2")
    report_count("p2", prog)
    check("p2", prog, [3, 4, 0, 5, 0], [7, 5])


@case("Part 2", sample=True, title="题目例子 0 0 7 0（空组输出 0）")
def p2_example_2(sol):
    check("p2", program(sol, "p2"), [0, 0, 7, 0], [0, 0, 7])


@case("Part 2", sample=True, title="题目例子 2 2 2（没有 0）")
def p2_example_3(sol):
    check("p2", program(sol, "p2"), [2, 2, 2], [])


@case("Part 2", sample=True, title="题目例子 -5 9 0 1")
def p2_example_4(sol):
    check("p2", program(sol, "p2"), [-5, 9, 0, 1], [4])


@case("Part 2", title="检查器自带的 8 组测试")
def p2_checker_tests(sol):
    check_all(sol, "p2", PUZZLES["p2"][1])


@case("Part 2", title="空输入：读数之前不能输出")
def p2_tiny(sol):
    check_all(sol, "p2", [[], [0], [5]])


@case("Part 2", title="组内部分和回到 0（不是分隔符）")
def p2_partial_sum_zero(sol):
    check_all(sol, "p2", [[5, -5, 4, 0], [1, -1, 0], [-3, 3, -3, 3, 0, 2, -2, 7, 0], [7, -7, 0, 0]])


@case("Part 2", title="部分和到达 ±99")
def p2_extremes(sol):
    check_all(sol, "p2", [[99, 0], [-99, 0], [50, 49, 0, -50, -49, 0], [99, -99, 99, 0, -99, 99, -99, 0],
                          [98, 1, -99, -99, 99, 0]])


@case("Part 2", title="所有短输入（−2..2，长度 ≤ 6）")
def p2_all_short(sol):
    check_all(sol, "p2", SHORT["p2"])


@case("Part 2", title="随机输入（部分和都在 −99..99）")
def p2_random(sol):
    check_random(sol, "p2", 800, 24)


# ---------------------------------------------------------------- Part 3 静音开关
@case("Part 3", sample=True, title="题目例子 3 0 5 0 7")
def p3_example_1(sol):
    prog = program(sol, "p3")
    report_count("p3", prog)
    check("p3", prog, [3, 0, 5, 0, 7], [3, 7])


@case("Part 3", sample=True, title="题目例子 0 4 4 0 0 9 0 6")
def p3_example_2(sol):
    check("p3", program(sol, "p3"), [0, 4, 4, 0, 0, 9, 0, 6], [6])


@case("Part 3", sample=True, title="题目例子 0 0 8")
def p3_example_3(sol):
    check("p3", program(sol, "p3"), [0, 0, 8], [8])


@case("Part 3", sample=True, title="题目例子 2 0")
def p3_example_4(sol):
    check("p3", program(sol, "p3"), [2, 0], [2])


@case("Part 3", title="检查器自带的 8 组测试")
def p3_checker_tests(sol):
    check_all(sol, "p3", PUZZLES["p3"][1])


@case("Part 3", title="空输入和只有一个数")
def p3_tiny(sol):
    check_all(sol, "p3", [[], [0], [7], [99]])


@case("Part 3", title="1 和 99（取值范围的两端）")
def p3_extremes(sol):
    check_all(sol, "p3", [[1, 0, 1, 0, 1], [99, 0, 99, 0, 99], [0, 1, 99, 0, 1, 99], [1, 1, 0, 0, 1]])


@case("Part 3", title="所有短输入（0..3，长度 ≤ 7）")
def p3_all_short(sol):
    check_all(sol, "p3", SHORT["p3"])


@case("Part 3", title="随机输入（0..99，长度 ≤ 24）")
def p3_random(sol):
    check_random(sol, "p3", 800, 24)


# ---------------------------------------------------------------- Part 4 倒数
@case("Part 4", sample=True, title="题目例子 3 0 2")
def p4_example_1(sol):
    prog = program(sol, "p4")
    report_count("p4", prog)
    check("p4", prog, [3, 0, 2], [3, 2, 1, 2, 1])


@case("Part 4", sample=True, title="题目例子 1 1")
def p4_example_2(sol):
    check("p4", program(sol, "p4"), [1, 1], [1, 1])


@case("Part 4", title="检查器自带的 8 组测试")
def p4_checker_tests(sol):
    check_all(sol, "p4", PUZZLES["p4"][1])


@case("Part 4", title="x = 99（最大值，注意饱和）")
def p4_max_value(sol):
    check_all(sol, "p4", [[99], [99, 1], [1, 99, 0]])


@case("Part 4", title="8 个 99：每个测试最多 20,000 步")
def p4_eight_99(sol):
    check_worst(sol, "p4")


@case("Part 4", title="全是 0；0 在开头或结尾")
def p4_zeros(sol):
    check_all(sol, "p4", [[], [0] * 8, [0, 3], [3, 0], [0, 0, 1, 0, 0]])


@case("Part 4", title="所有短输入（0..4，至多 5 个数）")
def p4_all_short(sol):
    check_all(sol, "p4", SHORT["p4"])


@case("Part 4", title="随机输入（至多 8 个数，0..99）")
def p4_random(sol):
    check_random(sol, "p4", 300, 8)


# ---------------------------------------------------------------- Part 5 正数
@case("Part 5", sample=True, title="题目例子 3 0 2")
def p5_example_1(sol):
    prog = program(sol, "p5")
    report_count("p5", prog)
    check("p5", prog, [3, 0, 2], [1, 2, 3, 1, 2])


@case("Part 5", sample=True, title="题目例子 1 1")
def p5_example_2(sol):
    check("p5", program(sol, "p5"), [1, 1], [1, 1])


@case("Part 5", title="检查器自带的 8 组测试")
def p5_checker_tests(sol):
    check_all(sol, "p5", PUZZLES["p5"][1])


@case("Part 5", title="x = 99（最大值，注意饱和）")
def p5_max_value(sol):
    check_all(sol, "p5", [[99], [99, 1], [1, 99, 0]])


@case("Part 5", title="8 个 99：每个测试最多 20,000 步")
def p5_eight_99(sol):
    check_worst(sol, "p5")


@case("Part 5", title="全是 0；0 在开头或结尾")
def p5_zeros(sol):
    check_all(sol, "p5", [[], [0] * 8, [0, 3], [3, 0], [0, 0, 1, 0, 0]])


@case("Part 5", title="所有短输入（0..4，至多 5 个数）")
def p5_all_short(sol):
    check_all(sol, "p5", SHORT["p5"])


@case("Part 5", title="随机输入（至多 8 个数，0..99）")
def p5_random(sol):
    check_random(sol, "p5", 300, 8)
