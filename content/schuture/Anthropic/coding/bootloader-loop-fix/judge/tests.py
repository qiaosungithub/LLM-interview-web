from judgelib import calling, case, expect, expect_true, expect_valid, rng
import ref

DEMO = [("next", 0), ("plus", 6), ("next", 5), ("plus", 100), ("jump", -1),
        ("plus", 50), ("jump", -3), ("plus", 7), ("next", 0)]
FIXED_DEMO = DEMO[:2] + [("jump", 5)] + DEMO[3:]

EXAMPLE_TEXT = """\
# module 7 boot sequence
next +0

plus +4      # initial credit
jump +2
plus +999    # skipped trap
plus +3
"""
EXAMPLE_PROGRAM = [("next", 0), ("plus", 4), ("jump", 2), ("plus", 999), ("plus", 3)]


def _norm(x):
    """A list is accepted where the statement says tuple."""
    return tuple(x) if isinstance(x, (list, tuple)) else x


def _exc(sol, name):
    cls = getattr(sol, name, None)
    expect_true(isinstance(cls, type) and issubclass(cls, BaseException),
                f"答案里需要定义题目给出的异常类 {name}")
    return cls


def show_program(program):
    """The program as Python the user can paste; long ones are packed into one string so the call stays short."""
    text = repr(program)
    if len(text) <= 2400:
        return text
    body = ";".join(f"{op} {value:+d}" for op, value in program)
    return f'[(op, int(v)) for op, v in (s.split() for s in "{body}".split(";"))]'


def _op(program, line):
    op, value = program[line]
    return f"{op} {value:+d}"


# ---------------------------------------------------------------- Part 1 helpers

def run_ok(sol, program, want):
    """The program does not leave 0 .. n; want is (terminated, acc, loop_line)."""
    call = calling(f"run_program({show_program(program)})")
    got = _norm(sol.run_program(list(program)))
    msg = ""
    if isinstance(got, tuple) and len(got) == 3 and got != want:
        if got[0] != want[0]:
            msg = "这个程序应当正常结束" if want[0] else f"这个程序应当在第 {want[2]} 行陷入死循环"
        elif got[2] != want[2]:
            msg = "loop_line 应当是即将第二次执行的那一行的下标"
        elif not want[0]:
            msg = "acc 应当是上一条指令执行完之后的值，即重复之前的值"
        else:
            msg = "正常结束时的 acc 不对"
    expect(got, want, call=call, msg=msg)


def run_out(sol, program):
    """Some jump moves pc outside 0 .. n: run_program must raise ProgramError."""
    exc = _exc(sol, "ProgramError")
    call = calling(f"run_program({show_program(program)})")
    kind, _, (line, pc) = ref.simulate(program)
    assert kind == "out"
    where = f"第 {line} 行 {_op(program, line)} 让 pc 变成 {pc}，越出了 0 .. {len(program)}"
    try:
        got = sol.run_program(list(program))
    except exc:
        return
    except NotImplementedError:
        raise
    except Exception as e:
        expect_true(False, f"{where}：应当抛出 ProgramError，实际抛出 {type(e).__name__}: {e}", call=call)
    expect_true(False, f"{where}：应当抛出 ProgramError，但正常返回了", call=call, got=got)


def run_like_ref(sol, program):
    kind, acc, extra = ref.simulate(program)
    if kind == "out":
        run_out(sol, program)
    else:
        run_ok(sol, program, (kind == "end", acc, extra))


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子：第 3 行重复")
def p1_example(sol):
    run_ok(sol, DEMO, (False, 106, 3))


@case("Part 1", title="正常结束：pc 恰好等于 n")
def p1_terminates(sol):
    run_ok(sol, FIXED_DEMO, (True, 13, None))
    run_ok(sol, [("jump", 2), ("plus", 999), ("next", 0)], (True, 0, None))
    run_ok(sol, [("plus", 5), ("jump", 2), ("plus", 1)], (True, 5, None))
    run_ok(sol, [("plus", -4), ("plus", 1)], (True, -3, None))
    run_ok(sol, [("jump", 1)], (True, 0, None))


@case("Part 1", title="越界要抛 ProgramError")
def p1_out_of_range(sol):
    for program in [
        [("plus", 1), ("jump", 5)],                              # three lines past the end
        [("plus", 1), ("jump", 2)],                              # pc = n + 1
        [("plus", 3), ("jump", 2), ("plus", 4), ("jump", 4), ("jump", -2)],
        [("jump", -1)],                                          # negative pc
        [("jump", -1), ("plus", 5)],                             # program[-1] would be the last line
        [("plus", 2), ("next", 0), ("jump", -3), ("plus", 9)],   # wrapping around would even "terminate"
        [("plus", 1), ("jump", -7)],
    ]:
        run_out(sol, program)


@case("Part 1", title="死循环时 acc 是重复之前的值")
def p1_loop_acc(sol):
    run_ok(sol, [("plus", 1), ("plus", 2), ("jump", -2)], (False, 3, 0))
    run_ok(sol, [("plus", 5), ("plus", -2), ("jump", -1)], (False, 3, 1))
    run_ok(sol, [("plus", 10), ("jump", 2), ("plus", 100), ("plus", 1), ("jump", -3)], (False, 11, 1))


@case("Part 1", title="jump +0 立刻重复自己")
def p1_jump_zero(sol):
    run_ok(sol, [("jump", 0)], (False, 0, 0))
    run_ok(sol, [("plus", 5), ("jump", 0)], (False, 5, 1))


@case("Part 1", title="next 的数值不起作用")
def p1_next_value_ignored(sol):
    run_ok(sol, [("next", 5), ("plus", 1)], (True, 1, None))
    run_ok(sol, [("next", -1), ("next", 100), ("plus", 2)], (True, 2, None))
    run_ok(sol, [("next", 0), ("jump", -1)], (False, 0, 0))


@case("Part 1", title="空程序")
def p1_empty(sol):
    run_ok(sol, [], (True, 0, None))


@case("Part 1", title="顺序执行回到执行过的行")
def p1_loop_via_fallthrough(sol):
    run_ok(sol, [("jump", 2), ("plus", 3), ("jump", -1)], (False, 3, 2))
    run_ok(sol, [("plus", 1), ("jump", 2), ("plus", 4), ("next", 3), ("jump", -2), ("jump", -4)], (False, 5, 3))


@case("Part 1", title="随机程序：三种结局都有")
def p1_random(sol):
    r = rng()
    for t in range(300):
        run_like_ref(sol, ref.random_program(r, r.randint(0, 12) if t < 250 else r.randint(13, 80)))


# ---------------------------------------------------------------- Part 2 helpers

def _fix_hint(program, got, want):
    if got == want or not (isinstance(got, tuple) and len(got) == 2):
        return ""
    line = got[0]
    if line == want[0]:
        return "行号对了，但 acc 不对：要的是改回这一行之后整个程序正常结束时的累加器值"
    if not isinstance(line, int) or not 0 <= line < len(program):
        return f"{line!r} 不是这个程序里的行号"
    op, value = program[line]
    if op == "plus":
        return f"第 {line} 行是 plus，plus 行不会被改错"
    trial = list(program)
    trial[line] = (ref.FLIP[op], value)
    kind, _, extra = ref.simulate(trial)
    changed = f"把第 {line} 行改成 {ref.FLIP[op]} {value:+d} 之后，"
    if kind == "loop":
        return changed + f"程序仍会在第 {extra} 行陷入死循环"
    if kind == "out":
        return changed + f"第 {extra[0]} 行会让 pc 越界（变成 {extra[1]}），这不算正常结束"
    return ""


def fix_ok(sol, program, want, shown=None):
    call = calling(f"find_fix({shown or show_program(program)})")
    got = _norm(sol.find_fix(list(program)))
    expect(got, want, call=call, msg=_fix_hint(program, got, want))


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子：第 2 行本该是 jump")
def p2_example(sol):
    fix_ok(sol, DEMO, (2, 13))


@case("Part 2", title="改错的是 jump（本该是 next）")
def p2_jump_should_be_next(sol):
    fix_ok(sol, [("plus", 3), ("jump", 0), ("plus", 4)], (1, 7))
    fix_ok(sol, [("next", 0), ("plus", 2), ("jump", -1), ("plus", 5)], (2, 7))
    fix_ok(sol, [("plus", 1), ("jump", 3), ("plus", 4), ("jump", 4), ("jump", -3), ("plus", 4), ("jump", 2)], (1, 5))


@case("Part 2", title="改错的是 next（本该是 jump）")
def p2_next_should_be_jump(sol):
    fix_ok(sol, [("plus", 1), ("next", 4), ("plus", 10), ("jump", -2), ("jump", -4), ("plus", 20)], (1, 21))
    fix_ok(sol, [("jump", 4), ("jump", 0), ("plus", 5), ("jump", 4), ("next", -2), ("jump", -5), ("jump", 0)], (4, 5))


@case("Part 2", title="有的翻转会越界：那不是修复")
def p2_out_of_range_candidate(sol):
    fix_ok(sol, [("next", 9), ("plus", 1), ("jump", -1)], (2, 1))
    fix_ok(sol, [("next", -21), ("next", 49), ("jump", -1)], (2, 0))
    fix_ok(sol, [("plus", 2), ("next", -5), ("plus", 3), ("jump", -2)], (3, 5))
    fix_ok(sol, [("plus", -22), ("next", -13), ("jump", 0)], (2, -22))


@case("Part 2", title="死循环由顺序执行引起")
def p2_loop_by_fallthrough(sol):
    fix_ok(sol, [("jump", 2), ("plus", 4), ("next", 3), ("jump", -2), ("jump", -4)], (2, 0))
    fix_ok(sol, [("plus", 1), ("jump", 2), ("plus", 4), ("next", 3), ("jump", -2), ("jump", -4)], (3, 1))


@case("Part 2", title="整个程序是一个大循环")
def p2_long_cycle(sol):
    fix_ok(sol, [("plus", 1), ("jump", 1)] * 100 + [("jump", -200)], (200, 100),
           shown='[("plus", 1), ("jump", 1)] * 100 + [("jump", -200)]')
    fix_ok(sol, [("jump", 1)] * 299 + [("jump", -299)], (299, 0), shown='[("jump", 1)] * 299 + [("jump", -299)]')


@case("Part 2", title="随机程序：对照逐行翻转")
def p2_random(sol):
    r = rng()
    for t in range(240):
        instance = ref.single_fix_instance(r, r.randint(4, 40), want_op=("next", "jump", None)[t % 3])
        if instance:
            fix_ok(sol, *instance)


@case("Part 2", title="随机程序：100 到 200 行")
def p2_random_large(sol):
    r = rng()
    for _ in range(12):
        instance = ref.single_fix_instance(r, r.randint(100, 200))
        if instance:
            fix_ok(sol, *instance)


# ---------------------------------------------------------------- Part 3 helpers

def _line(text, number):
    return text.split("\n")[number - 1].rstrip("\r")


def parse_ok(sol, text, want):
    exc = _exc(sol, "ParseError")
    call = calling(f"parse_program({text!r})")
    try:
        got = sol.parse_program(text)
    except exc as e:
        number = getattr(e, "line_number", None)
        shown = f"第 {number} 行 {_line(text, number)!r}" if isinstance(number, int) and \
            1 <= number <= len(text.split("\n")) else f"line_number={number!r}"
        expect_true(False, f"这段文本没有格式错误，不应抛出 ParseError（报告的是{shown}）", call=call)
    if isinstance(got, (list, tuple)):
        got = [_norm(x) for x in got]
    expect(got, want, call=call)


def parse_bad(sol, text, numbers):
    """numbers: the 1-based line numbers of the malformed lines; reporting any one of them is right."""
    exc = _exc(sol, "ParseError")
    call = calling(f"parse_program({text!r})")
    bad = "、".join(f"第 {k} 行 {_line(text, k)!r}" for k in numbers)
    try:
        got = sol.parse_program(text)
    except exc as e:
        error = e
    except NotImplementedError:
        raise
    except Exception as e:
        expect_true(False, f"{bad} 格式错误：应当抛出 ParseError，实际抛出 {type(e).__name__}: {e}", call=call)
    else:
        expect_true(False, f"{bad} 格式错误：应当抛出 ParseError，但正常返回了", call=call, got=got)
    expect_true(hasattr(error, "line_number"), "ParseError 需要带上 line_number 属性（1 起始的行号）", call=call)
    expect_true(isinstance(error, ValueError), "ParseError 要继承 ValueError：题目给出的定义是 class ParseError(ValueError)，"
                f"实际的基类是 {', '.join(b.__name__ for b in type(error).__bases__)}", call=call)
    got = error.line_number
    if len(numbers) == 1:
        expect(got, numbers[0], call=call, msg=f"格式错误的是{bad}（行号 1 起始，空行和注释行都算在内）")
    else:
        expect_valid(got, lambda k: None if k in numbers else f"格式错误的是{bad}，报告其中任意一行都算对",
                     call=call, want=" 或 ".join(map(str, numbers)))


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子：注释和空行")
def p3_example(sol):
    parse_ok(sol, EXAMPLE_TEXT, EXAMPLE_PROGRAM)
    run_ok(sol, EXAMPLE_PROGRAM, (True, 7, None))


@case("Part 3", sample=True, title="题目里的四种格式错误")
def p3_malformed_examples(sol):
    for bad in ["plus 4", "boot +1", "jump +1 +2", "jump"]:
        parse_bad(sol, bad, [1])
        parse_bad(sol, f"next +0\n{bad}\nplus +1\n", [2])


@case("Part 3", title="行号把空行和注释行都算上")
def p3_line_numbers(sol):
    parse_bad(sol, "# header\n\nplus +1\n   \n# note\nplus 2\n", [6])
    parse_bad(sol, "\n\n\njump", [4])
    parse_bad(sol, "next +1\nnext +2\n\n# comment\njump +1 +2\nplus +1\n", [5])
    parse_bad(sol, "".join(f"# line {k}\n" for k in range(1, 41)) + "plus +1\n\nplus three\n", [43])


@case("Part 3", title="各种位置的注释")
def p3_comments(sol):
    parse_ok(sol, "plus +3#no space before the comment\n", [("plus", 3)])
    parse_ok(sol, "jump -2 # one # two\n", [("jump", -2)])
    parse_ok(sol, "plus +1 # jump +7\nplus -1\t# a tab before the comment\n", [("plus", 1), ("plus", -1)])
    parse_ok(sol, "   # indented comment\n\t# after a tab\nnext +1 #\n#\n##\n#plus +5\n", [("next", 1)])
    parse_ok(sol, "   \n\t\n \t \n", [])


@case("Part 3", title="带符号整数：+0、-0 合法")
def p3_signed_int(sol):
    parse_ok(sol, "plus +0\nplus -0\njump -0\nnext +0\n", [("plus", 0), ("plus", 0), ("jump", 0), ("next", 0)])
    parse_ok(sol, "plus -7\nplus +12345678901234567890\n", [("plus", -7), ("plus", 12345678901234567890)])
    for value in ["3", "3.0", "++3", "+-3", "-+3", "--1", "+3.0", "-2x", "+", "-", "+1e3", "3+"]:
        parse_bad(sol, f"next +0\nplus {value}\n", [2])


@case("Part 3", title="操作符必须原样匹配")
def p3_ops_exact(sol):
    for line in ["PLUS +1", "Jump +1", "nextt +1", "plu +1", "add +1", "plus+1", "+1 plus", "jump, +1"]:
        parse_bad(sol, f"plus +1\n{line}\n", [2])


@case("Part 3", title="制表符和多个空格")
def p3_whitespace(sol):
    parse_ok(sol, "plus\t+4\n  jump   -2  \n\tnext\t\t+0\t\n plus \t +1 \n",
             [("plus", 4), ("jump", -2), ("next", 0), ("plus", 1)])
    parse_bad(sol, "plus +1\nplus + 1\n", [2])


@case("Part 3", title="Windows 换行 \\r\\n")
def p3_crlf(sol):
    parse_ok(sol, "# boot\r\nnext +0\r\n\r\nplus +4  # credit\r\njump +2\r\n", [("next", 0), ("plus", 4), ("jump", 2)])
    parse_bad(sol, "next +0\r\n\r\nboot +1\r\n", [3])


@case("Part 3", title="没有指令的文本")
def p3_empty(sol):
    for text in ["", "\n", "# only comments\n\n   \n# another\n", "# no newline at the end"]:
        parse_ok(sol, text, [])


@case("Part 3", title="多行格式错误：报哪行都算对")
def p3_two_bad_lines(sol):
    parse_bad(sol, "plus +1\nplus 2\n# fine\nboot +3\n", [2, 4])


@case("Part 3", title="随机文本，含一行格式错误")
def p3_random(sol):
    r = rng()
    for t in range(160):
        text = ref.random_text(r, r.randint(0, 25), bad=t % 2 == 1)
        want = ref.parse(text)
        if isinstance(want, tuple):
            parse_bad(sol, text, [want[1]])
        else:
            parse_ok(sol, text, want)
