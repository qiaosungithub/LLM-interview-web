"""answer.py defines q1 ... q4: one program (text in the checker's syntax) per puzzle. Every program is parsed and
run with the statement's own checker (ref.parse / ref.run); the answer cannot bring its own interpreter."""
import itertools

from judgelib import Mismatch, case, expect, info, rng
import ref
from ref import CAP, LIMIT, MAX_LEN, PUZZLES, parse, run     # bound now, before the answer is loaded

MARKS = {"q1": (11, 3), "q2": (8, 6), "q3": (8, 6), "q4": (11, 8)}   # (及格线, 目标) from the statement


# ---------------------------------------------------------------- the checker must stay as it is
def _names():
    names = {f"ref.{k}": v for k, v in vars(ref).items() if k != "__warningregistry__"}
    names.update({"parse.__code__": parse.__code__, "run.__code__": run.__code__})
    return names


def _contents():
    return {"ref.NEED": dict(ref.NEED),
            "ref.PUZZLES": {k: (spec, [list(t) for t in tests]) for k, (spec, tests) in ref.PUZZLES.items()}}


ORIGINAL, CONTENTS = _names(), _contents()


def check_untouched():
    """The answer is only the four program texts; validity (queue cap, 1..12 instructions, step limit) is the
    checker's. The answer is loaded into the same process, so it could patch ref at load time (e.g. ref.CAP = 17);
    refuse to score if it did."""
    now = _names()
    changed = [k for k in ORIGINAL.keys() | now.keys() if ORIGINAL.get(k) is not now.get(k)]
    if not changed:
        contents = _contents()
        changed = [k for k, v in CONTENTS.items() if contents[k] != v]
    if changed:
        raise Mismatch(f"答案在运行时改动了检查器：{'、'.join(sorted(changed)[:5])}。每道题只交一段程序文本，"
                       "检查器（题目给的 parse / run / PUZZLES 和各项上限）是固定的，这样跑出来的结果不算数", show=False)


def program(sol, name):
    """The answer's program for one puzzle, parsed by the checker. Fails with a readable message if it is not
    a valid program (this is where the hard limit of 1..12 instructions is enforced, by the checker's parse)."""
    check_untouched()
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
            why = ("有一条指令写错了：指令名要大写，只有 JMP 和 JZ 带一个参数（跳转目标）；"
                   "检查器先按 ; 切开再去掉 # 后面的注释，所以注释里也不能有 ;")
        elif str(e).startswith("jump target"):
            why = f"跳转目标超出范围：要求 0 ≤ k ≤ 程序长度（{count}），k 等于程序长度表示停机"
        else:
            why = "程序无法解析（跳转目标要写成整数）"
        raise Mismatch(f"{why}（{text}）", f"parse({name})", show=False) from None


def report_count(name, prog):
    mark, target = MARKS[name]
    info(f"指令数 {len(prog)}（及格线 {mark}，目标 {target}）")


def check(name, prog, values, want):
    """One run on the checker's machine; a wrong final queue, an overflow or no halt within MAX_STEPS is a
    wrong answer, exactly as the checker decides it."""
    call = f"run(parse({name}), {values!r})"
    got = run(prog, list(values))
    if got is None:
        raise Mismatch(ref.why_invalid(prog, list(values)), call, want, None)
    if got != want and len(got) != len(want):
        raise Mismatch(f"停机时队列里有 {len(got)} 个数，应当恰好 {len(want)} 个（多一个、少一个都不行）", call, want, got)
    expect(got, want, call=call)


def check_all(sol, name, inputs):
    prog = program(sol, name)
    spec = PUZZLES[name][0]
    for values in inputs:
        check(name, prog, list(values), spec(list(values)))


def pairs(values_a, values_b=None):
    return [[a, b] for a in values_a for b in (values_a if values_b is None else values_b)]


EDGE = [[a, b] for a, b in pairs(range(LIMIT + 1)) if {a, b} & {0, LIMIT}]          # a or b is 0 or 99
BIG_SUMS = pairs([49, 50, 51, 60, 70, 98, 99]) + [[1, 99], [99, 1], [2, 98], [98, 2]]   # a + b > 99 for most
EDGE_98 = [[a, b] for a, b in pairs(range(LIMIT)) if {a, b} & {0, LIMIT - 1}]       # Part 4: 0 or 98
NEAR = [[a, a + d] for a in range(LIMIT) for d in (-1, 0, 1) if 0 <= a + d < LIMIT]   # Part 4: |a - b| <= 1
SHORT_LISTS = [list(t) for k in range(1, 7) for t in itertools.product((1, 2, LIMIT), repeat=k)]


# ---------------------------------------------------------------- Part 1 较大值
@case("Part 1", sample=True, title="题目例子 3 8 -> 8")
def q1_example_1(sol):
    prog = program(sol, "q1")
    report_count("q1", prog)
    check("q1", prog, [3, 8], [8])


@case("Part 1", sample=True, title="题目例子 5 5 -> 5")
def q1_example_2(sol):
    check("q1", program(sol, "q1"), [5, 5], [5])


@case("Part 1", sample=True, title="题目例子 99 0 -> 99")
def q1_example_3(sol):
    check("q1", program(sol, "q1"), [99, 0], [99])


@case("Part 1", title="所有 0..9 的数对")
def q1_small(sol):
    check_all(sol, "q1", pairs(range(10)))


@case("Part 1", title="a + b 超过 99（加法饱和）")
def q1_big_sums(sol):
    check_all(sol, "q1", BIG_SUMS)


@case("Part 1", title="有一个数是 0 或 99")
def q1_edges(sol):
    check_all(sol, "q1", EDGE)


@case("Part 1", cpu=10, title="检查器的全部测试（10,000 对）")
def q1_checker_tests(sol):
    check_all(sol, "q1", PUZZLES["q1"][1])


# ---------------------------------------------------------------- Part 2 差的绝对值
@case("Part 2", sample=True, title="题目例子 3 8 -> 5")
def q2_example_1(sol):
    prog = program(sol, "q2")
    report_count("q2", prog)
    check("q2", prog, [3, 8], [5])


@case("Part 2", sample=True, title="题目例子 8 3 -> 5")
def q2_example_2(sol):
    check("q2", program(sol, "q2"), [8, 3], [5])


@case("Part 2", sample=True, title="题目例子 0 99 -> 99")
def q2_example_3(sol):
    check("q2", program(sol, "q2"), [0, 99], [99])


@case("Part 2", title="所有 0..9 的数对")
def q2_small(sol):
    check_all(sol, "q2", pairs(range(10)))


@case("Part 2", title="a + b 超过 99（加法饱和）")
def q2_big_sums(sol):
    check_all(sol, "q2", BIG_SUMS)


@case("Part 2", title="有一个数是 0 或 99")
def q2_edges(sol):
    check_all(sol, "q2", EDGE)


@case("Part 2", cpu=10, title="检查器的全部测试（10,000 对）")
def q2_checker_tests(sol):
    check_all(sol, "q2", PUZZLES["q2"][1])


# ---------------------------------------------------------------- Part 3 数个数
@case("Part 3", sample=True, title="题目例子 4 4 9 -> 3")
def q3_example_1(sol):
    prog = program(sol, "q3")
    report_count("q3", prog)
    check("q3", prog, [4, 4, 9], [3])


@case("Part 3", sample=True, title="题目例子 99 -> 1")
def q3_example_2(sol):
    check("q3", program(sol, "q3"), [99], [1])


@case("Part 3", sample=True, title="题目例子 7 1 2 3 99 5 -> 6")
def q3_example_3(sol):
    check("q3", program(sol, "q3"), [7, 1, 2, 3, 99, 5], [6])


@case("Part 3", title="只有一个数（1..99 每个值）")
def q3_single(sol):
    check_all(sol, "q3", [[x] for x in range(1, LIMIT + 1)])


@case("Part 3", title="15 个数：队列最多只能到 16 个")
def q3_fifteen(sol):
    prog = program(sol, "q3")
    r = rng()
    inputs = [[LIMIT] * 15, [1] * 15, list(range(1, 16)), list(range(99, 84, -1))] + \
        [ref.random_list(r, 15) for _ in range(20)]
    for values in inputs:
        check("q3", prog, values, [15])
    steps, peak = ref.stats(prog, [LIMIT] * 15)
    info(f"输入 15 个 99：队列最长 {peak} 个数（上限 {CAP}），执行了 {steps:,} 条指令")


@case("Part 3", title="长度 1..15，全是 1 或全是 99")
def q3_every_length(sol):
    check_all(sol, "q3", [[x] * k for k in range(1, CAP) for x in (1, LIMIT)])


@case("Part 3", title="1、2、99 组成的所有输入（长度 ≤ 6）")
def q3_all_short(sol):
    check_all(sol, "q3", SHORT_LISTS)


@case("Part 3", title="检查器自带的 316 组测试")
def q3_checker_tests(sol):
    check_all(sol, "q3", PUZZLES["q3"][1])


@case("Part 3", title="随机输入（长度 1..15，每个数 1..99）")
def q3_random(sol):
    r = rng()
    inputs = [ref.random_list(r, r.randint(1, 15)) for _ in range(600)]
    check_all(sol, "q3", sorted(inputs, key=len))      # shortest first, so a failure shows a short input


# ---------------------------------------------------------------- Part 4 相等
@case("Part 4", sample=True, title="题目例子 7 7 -> 1")
def q4_example_1(sol):
    prog = program(sol, "q4")
    report_count("q4", prog)
    check("q4", prog, [7, 7], [1])


@case("Part 4", sample=True, title="题目例子 7 2 -> 0")
def q4_example_2(sol):
    check("q4", program(sol, "q4"), [7, 2], [0])


@case("Part 4", sample=True, title="题目例子 0 98 -> 0")
def q4_example_3(sol):
    check("q4", program(sol, "q4"), [0, 98], [0])


@case("Part 4", title="所有 0..9 的数对")
def q4_small(sol):
    check_all(sol, "q4", pairs(range(10)))


@case("Part 4", title="a = b 和相差 1 的数对")
def q4_near(sol):
    check_all(sol, "q4", NEAR)


@case("Part 4", title="有一个数是 0 或 98")
def q4_edges(sol):
    check_all(sol, "q4", EDGE_98)


@case("Part 4", cpu=10, title="检查器的全部测试（9,801 对，没有 99）")
def q4_checker_tests(sol):
    check_all(sol, "q4", PUZZLES["q4"][1])
