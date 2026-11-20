"""Part 2 (StreamDecoder) and Part 3 (StopStreamDecoder). Part 1 is a written analysis and is not judged.

The statement's given StreamDecoder defines every return value (ref.GivenStreamDecoder), so small inputs are compared
with it call by call. Long replies, built at import time, check that the work per token does not grow with the reply.
"""
from judgelib import calling, case, expect, expect_raises, expect_true, rng
import ref

EX = [b"h", b"i", b" ", b"\xe4\xbd", b"\xa0", b"\xe5\xa5\xbd", b"\xf0\x9f", b"\x98\x80", b"\xff", b""]
EX_IDS = [0, 1, 3, 4, 6, 7, 5, 8, 2, 9, 3]
MISSING = object()


# ---------------------------------------------------------------- replays and drivers

class Replay:
    """The calls made so far, as code the user can paste into 自定义运行 after their answer."""

    def __init__(self, pieces, ctor):
        self.head = [f"tok = ByteTokenizer({pieces!r})", f"d = {ctor}"]
        self.done = []          # ("push", id) or ("finish",): calls that change the decoder

    def code(self, current, note="这一步"):
        lines, run = list(self.head), []
        for op in self.done + [None]:
            if op is not None and op[0] == "push":
                run.append(op[1])
                continue
            if len(run) == 1:
                lines.append(f"d.push({run[0]})")
            elif run:
                lines.append(f"for i in {run}: d.push(i)")
            run = []
            if op is not None:
                lines.append("d.finish()")
        lines.append(f"print(repr({current}))   # <- {note}")
        return "\n".join(lines)


def stopped_flag(d):
    return getattr(d, "stopped", MISSING)


def expect_stopped(d, want, call):
    flag = stopped_flag(d)
    expect_true(flag is not MISSING, "StopStreamDecoder 对象没有 stopped 属性", call=call)
    expect(bool(flag), want, call=call,
           msg="停止串已经出现，stopped 应当是 True" if want else "还没有出现停止串，stopped 应当是 False")


def drive(sol, pieces, ops):
    """Part 2: ops are ("push", id), ("finish",) or ("text",); each result is compared with the given decoder's."""
    given = ref.GivenStreamDecoder(ref.ByteTokenizer(pieces))
    rp = Replay(pieces, "StreamDecoder(tok)")
    calling(rp.code("StreamDecoder(tok)", "构造"))
    d = sol.StreamDecoder(ref.ByteTokenizer(pieces))
    for op in ops:
        if op[0] == "push":
            code = rp.code(f"d.push({op[1]})")
            calling(code)
            if given.done:
                expect_raises(ValueError, d.push, op[1], call=rp.code(f"d.push({op[1]})", "finish() 之后再 push"))
                continue
            want, got = given.push(op[1]), d.push(op[1])
        elif op[0] == "finish":
            code = rp.code("d.finish()")
            calling(code)
            want, got = given.finish(), d.finish()
        else:
            code = rp.code("d.text()")
            calling(code)
            want, got = given.text(), d.text()
        expect(got, want, call=code)
        if op[0] != "text":
            rp.done.append(op)


def full_ops(ids, finish=True, after=True):
    """push every id with text() after each call; then finish, text, a second finish and a push after finish."""
    ops = []
    for i in ids:
        ops += [("push", i), ("text",)]
    if finish:
        ops += [("finish",), ("text",)]
        if after:
            ops += [("finish",), ("text",), ("push", ids[0] if ids else 0)]
    return ops


def drive_stop(sol, pieces, stops, ids, finish=True, after=True):
    """Part 3: after every call, the text returned so far and `stopped` must equal the definition applied to F,
    the text the given decoder has returned so far (ref.stop_ref)."""
    given = ref.GivenStreamDecoder(ref.ByteTokenizer(pieces))
    rp = Replay(pieces, f"StopStreamDecoder(tok, {stops!r})")
    calling(rp.code(f"StopStreamDecoder(tok, {stops!r})", "构造"))
    d = sol.StopStreamDecoder(ref.ByteTokenizer(pieces), list(stops))
    F, prev = "", ""

    def check(op, final):
        nonlocal F, prev
        code = rp.code(f"d.{op}")
        calling(code)
        F += given.push(int(op[5:-1])) if op.startswith("push") else given.finish()
        cur, stopped = ref.stop_ref(F, stops, final)
        got = d.push(int(op[5:-1])) if op.startswith("push") else d.finish()
        expect(got, cur[len(prev):] if cur.startswith(prev) else "", call=code)
        prev = cur
        rp.done.append(("push", int(op[5:-1])) if op.startswith("push") else ("finish",))
        expect_stopped(d, stopped, rp.code("d.stopped"))
        calling(rp.code("d.text()"))
        expect(d.text(), cur, call=rp.code("d.text()"))
        return stopped

    stopped = False
    for i in ids:
        stopped = check(f"push({i})", False)
    if not finish:
        return
    stopped = check("finish()", True)
    if not after:
        return
    if stopped:
        for op in ("push(0)", "finish()"):
            code = rp.code(f"d.{op}", "已经停止：应当返回空串，不报错")
            calling(code)
            got = d.push(0) if op == "push(0)" else d.finish()
            expect(got, "", call=code)
            rp.done.append(("push", 0) if op == "push(0)" else ("finish",))
            expect(d.text(), prev, call=rp.code("d.text()"))
            expect_stopped(d, True, rp.code("d.stopped"))
    else:
        code = rp.code("d.push(0)", "没有停止，finish() 之后再 push：应当抛出 ValueError")
        calling(code)
        expect_raises(ValueError, d.push, 0, call=code)


def check_outputs(got_fn, ids, want, desc, pieces):
    """Long replies: compare every push (then finish) with the precomputed values."""
    for k, i in enumerate(ids):
        got = got_fn(i)
        if got != want[k]:
            before = [pieces[j] for j in ids[max(0, k - 3):k]]
            expect(got, want[k], call=f"{desc}：第 {k + 1} 次 push（这个 token 的字节 {pieces[i]!r:.200}，"
                                      f"前 3 个 token 的字节 {before!r:.400}）")


# Long replies are built once at import time, which is not charged to any case.
_r = rng("long")
TICKET_PIECES, TICKET_IDS = ref.make_reply(_r, 20_000)
TICKET_WANT = ref.fast_outputs(TICKET_PIECES, TICKET_IDS)
LONG_PIECES, LONG_IDS = ref.long_token_reply(_r, 40_000, 100, 300)
LONG_WANT = ref.fast_outputs(LONG_PIECES, LONG_IDS)
LONG_BYTES = sum(len(p) for p in LONG_PIECES)
LONG_STOPS = [" the end", "回答完毕"]          # never complete in the reply, but their starts appear all the time
# Part 3: many short tokens, so rescanning the whole text on every call is far over the limit while a correct answer
# that walks the new text one character at a time (about 2 us per character) still fits comfortably.
STOP_PIECES, STOP_IDS = ref.long_token_reply(_r, 70_000, 3, 12)
STOP_BYTES = sum(len(p) for p in STOP_PIECES)
STOP_WANT = ref.stop_outputs_no_match(ref.fast_outputs(STOP_PIECES, STOP_IDS), LONG_STOPS)
WAIT_PIECES = ["😀".encode() + b"\xff", b"\xff" * 48, b"\xed\xa0\x80" * 16, b"A"]
WAIT_IDS = [0] + [1 + k % 2 for k in range(50_000)] + [3]
WAIT_WANT = ref.fast_outputs(WAIT_PIECES, WAIT_IDS)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子：一条回复的每一步")
def p2_example(sol):
    d = sol.StreamDecoder(ref.ByteTokenizer(EX))
    rp = Replay(EX, "StreamDecoder(tok)")
    for i, want in zip(EX_IDS, ["h", "i", "", "你", "", "😀", "好", "", "\ufffd ", "", ""]):
        code = rp.code(f"d.push({i})")
        calling(code)
        expect(d.push(i), want, call=code)
        rp.done.append(("push", i))
    calling(rp.code("d.finish()"))
    expect(d.finish(), "\ufffd", call=rp.code("d.finish()"))
    rp.done.append(("finish",))
    calling(rp.code("d.text()"))
    expect(d.text(), "hi你😀好\ufffd \ufffd", call=rp.code("d.text()"))


@case("Part 2", sample=True, title="题目例子：半个字符后面直接来 h")
def p2_example_half_then_ascii(sol):
    d = sol.StreamDecoder(ref.ByteTokenizer(EX))
    rp = Replay(EX, "StreamDecoder(tok)")
    for i, want in [(3, ""), (0, "\ufffdh")]:
        code = rp.code(f"d.push({i})")
        calling(code)
        expect(d.push(i), want, call=code)
        rp.done.append(("push", i))


@case("Part 2", title="每一步的 text()；finish 之后的调用")
def p2_text_and_after_finish(sol):
    drive(sol, EX, full_ops(EX_IDS))
    drive(sol, EX, [("push", 3), ("finish",), ("text",), ("finish",), ("push", 4), ("text",)])


@case("Part 2", title="空回复、只有空 token")
def p2_empty(sol):
    drive(sol, EX, [("text",), ("finish",), ("text",), ("finish",)])
    drive(sol, EX, full_ops([9, 9, 9]))


@case("Part 2", title="坏字节后面来空 token：仍要等")
def p2_bad_byte_then_empty(sol):
    drive(sol, EX, full_ops([8, 9, 9, 0]))
    drive(sol, [b"a\xff", b"", b"b", b"\xe4", b"\xbd\xa0"], full_ops([0, 1, 1, 2, 0, 3, 4, 1]))
    drive(sol, [b"\xff", b"", b"\xe4\xbd"], full_ops([0, 1, 2, 1], after=False))


@case("Part 2", title="文本里本来就有 U+FFFD：也要等")
def p2_real_replacement_char(sol):
    fffd = "\ufffd".encode()                            # ef bf bd, a valid character
    drive(sol, [b"x" + fffd, b"y", fffd[:1], fffd[1:], b""], full_ops([0, 4, 1, 2, 3, 4, 1, 0]))


@case("Part 2", title="字符跨 2 到 4 个 token，中间夹空 token")
def p2_byte_by_byte(sol):
    drive(sol, EX, full_ops([9, 0, 9, 9, 3, 9, 4, 9, 6, 9, 7]))
    for text in ["你好", "a😀b", "é世界🚀✨", "😀😀"]:
        data = text.encode()
        pieces = [data[k:k + 1] for k in range(len(data))] + [b""]
        ids = []
        for k in range(len(data)):
            ids += [k, len(data)] if k % 3 == 1 else [k]
        drive(sol, pieces, full_ops(ids))


@case("Part 2", title="非法 UTF-8：代理区、超长编码、截断")
def p2_invalid_utf8(sol):
    pieces = [b"\xe0\x80", b"\xed\xa0\x80", b"\xed\x9f\xbf", b"\xf4\x90\x80\x80", b"\xc0\xaf", b"\xf5", b"\x80",
              b"\xe4\xbd", b"A", b"\xf0\x9f\x98", b"\xc2", b"\xe0", b"\xa0\x80", b""]
    seqs = [[0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8], [7, 7, 8], [9, 8], [9, 6], [10, 6, 8],
            [11, 12, 8], [11, 13, 12], [7, 13, 13, 6], [9, 10, 8], [7], [9], [11], [10, 13], [0], [1]]
    for ids in seqs:
        drive(sol, pieces, full_ops(ids))


@case("Part 2", title="随机字节串（对照原版逐次比较）")
def p2_random_pieces(sol):
    r = rng()
    for _ in range(500):
        pieces = [ref.random_piece(r) for _ in range(r.randint(1, 8))]
        ids = [r.randrange(len(pieces)) for _ in range(r.randint(0, 14))]
        drive(sol, pieces, full_ops(ids, after=r.random() < 0.3))


@case("Part 2", title="真实文本在随机位置切开（对照原版）")
def p2_random_cuts(sol):
    r = rng()
    for _ in range(300):
        text = "".join(r.choice("ab 你好世界éß😀🚀✨\n") for _ in range(r.randint(0, 30)))
        pieces, ids = ref.cut_text(r, text, 20)
        drive(sol, pieces, full_ops(ids, after=False))


@case("Part 2", cpu=2, title="工单复现：2 万 token 的回复")
def p2_ticket(sol):
    desc = "工单那样的 2 万 token 中英文和 emoji 混合回复（题目复现脚本的词表）"
    calling(f"<{desc}>：依次 push 每个 token")
    d = sol.StreamDecoder(ref.ByteTokenizer(TICKET_PIECES))
    check_outputs(d.push, TICKET_IDS, TICKET_WANT, desc, TICKET_PIECES)
    calling(f"<{desc}>：finish()")
    expect(d.finish(), TICKET_WANT[-1], call=f"<{desc}>：最后的 finish()")
    calling(f"<{desc}>：text()")
    expect_true(d.text() == "".join(TICKET_WANT), "全部 push 和 finish 之后，text() 不等于所有返回值拼起来",
                call=f"<{desc}>：text()")


@case("Part 2", cpu=3, title="4 万个长 token：不能累加整段文本")
def p2_long_tokens(sol):
    desc = f"4 万个 token 的回复，每个 token 100 到 300 字节，共 {LONG_BYTES:,} 字节，开头是 emoji"
    calling(f"<{desc}>：依次 push 每个 token")
    d = sol.StreamDecoder(ref.ByteTokenizer(LONG_PIECES))
    check_outputs(d.push, LONG_IDS, LONG_WANT, desc, LONG_PIECES)
    calling(f"<{desc}>：finish()")
    expect(d.finish(), LONG_WANT[-1], call=f"<{desc}>：最后的 finish()")
    calling(f"<{desc}>：text()")
    expect_true(d.text() == "".join(LONG_WANT), "全部 push 和 finish 之后，text() 不等于所有返回值拼起来",
                call=f"<{desc}>：text()")


@case("Part 2", cpu=3, title="5 万个坏字节 token：一直在等也要线性")
def p2_long_wait(sol):
    desc = (f"tok = ByteTokenizer({WAIT_PIECES!r})；push(0)，再交替 push(1)、push(2) 共 5 万次"
            "（全是非法字节，原版一直在等），最后 push(3)")
    calling(f"<{desc}>")
    d = sol.StreamDecoder(ref.ByteTokenizer(WAIT_PIECES))
    check_outputs(d.push, WAIT_IDS, WAIT_WANT, desc, WAIT_PIECES)
    expect(d.finish(), WAIT_WANT[-1], call=f"<{desc}>；finish()")


# ---------------------------------------------------------------- Part 3
def run_stop_example(sol, stops, steps, final_text, stopped_after=None):
    """steps: (call, want) in order; call is "push(i)" or "finish()". Checks each return value, then text()."""
    rp = Replay(EX, f"StopStreamDecoder(tok, {stops!r})")
    calling(rp.code(f"StopStreamDecoder(tok, {stops!r})", "构造"))
    d = sol.StopStreamDecoder(ref.ByteTokenizer(EX), list(stops))
    for k, (call, want) in enumerate(steps):
        code = rp.code(f"d.{call}")
        calling(code)
        got = d.push(int(call[5:-1])) if call.startswith("push") else d.finish()
        expect(got, want, call=code)
        rp.done.append(("push", int(call[5:-1])) if call.startswith("push") else ("finish",))
        if stopped_after is not None:
            expect_stopped(d, k >= stopped_after, rp.code("d.stopped"))
    calling(rp.code("d.text()"))
    expect(d.text(), final_text, call=rp.code("d.text()"))
    return d, rp


@case("Part 3", sample=True, title="题目例子：停在“你好”前面")
def p3_example_stop(sol):
    steps = [("push(0)", "h"), ("push(1)", ""), ("push(3)", ""), ("push(4)", "i"), ("push(5)", ""), ("push(2)", ""),
             ("finish()", "")]
    run_stop_example(sol, ["i h", "你好"], steps, "hi", stopped_after=4)


@case("Part 3", sample=True, title="题目例子：结束最早的停止串优先")
def p3_example_earliest_end(sol):
    steps = [("push(0)", ""), ("push(1)", ""), ("push(2)", ""), ("push(0)", "h"), ("push(1)", "")]
    run_stop_example(sol, ["hi hi", "i h"], steps, "h", stopped_after=3)


@case("Part 3", sample=True, title="题目例子：同时结束取更长的")
def p3_example_tie(sol):
    d, rp = run_stop_example(sol, ["好", "你好"], [("push(3)", ""), ("push(4)", ""), ("push(5)", "")], "")
    expect_stopped(d, True, rp.code("d.stopped"))


@case("Part 3", sample=True, title="题目例子：没有停止，finish 交出留下的")
def p3_example_no_stop(sol):
    run_stop_example(sol, ["i h"], [("push(0)", "h"), ("push(1)", ""), ("finish()", "i")], "hi", stopped_after=99)


@case("Part 3", title="停止串里有空串：抛 ValueError")
def p3_empty_stop_raises(sol):
    for stops in (["a", ""], [""], ["", "", "x"]):
        call = f"StopStreamDecoder(ByteTokenizer([b'a']), {stops!r})"
        calling(call)
        expect_raises(ValueError, sol.StopStreamDecoder, ref.ByteTokenizer([b"a"]), list(stops), call=call)


@case("Part 3", title="没有停止时 finish 之后再 push 抛 ValueError")
def p3_push_after_finish(sol):
    drive_stop(sol, EX, ["xyz"], [0, 1])
    drive_stop(sol, EX, ["i h", "你好"], [0, 1, 3, 4])
    drive_stop(sol, EX, ["hi!"], [])


@case("Part 3", title="停止之后 push / finish 只返回空串")
def p3_after_stop(sol):
    drive_stop(sol, EX, ["hi"], [0, 1, 2, 0, 1, 3, 4])
    drive_stop(sol, EX, ["你"], [0, 3, 4, 5, 8, 2])
    drive_stop(sol, EX, ["h"], [0])


@case("Part 3", title="stops 为空：和 Part 2 完全一样")
def p3_no_stops(sol):
    drive_stop(sol, EX, [], EX_IDS)
    drive_stop(sol, EX, [], [3, 0, 8, 9, 2, 3])
    rp = Replay(EX, "StopStreamDecoder(tok, [])")
    calling(rp.code("StopStreamDecoder(tok, [])", "构造"))
    d = sol.StopStreamDecoder(ref.ByteTokenizer(EX), [])
    for i in [0, 3]:
        calling(rp.code(f"d.push({i})"))
        d.push(i)
        rp.done.append(("push", i))
    calling(rp.code("d.finish()"))
    expect(d.finish(), "\ufffd", call=rp.code("d.finish()"))
    rp.done.append(("finish",))
    code = rp.code("d.finish()", "第二次 finish() 和 Part 2 一样返回空串")
    calling(code)
    expect(d.finish(), "", call=code)


@case("Part 3", title="结束最早的优先，不是开始最早的")
def p3_earliest_end(sol):
    pieces = [b"abcd", b"xabcdx", b"bc", b"a", b"d", b""]
    drive_stop(sol, pieces, ["abcd", "bc"], [0, 1])
    drive_stop(sol, pieces, ["bc", "abcd"], [1, 0])
    drive_stop(sol, pieces, ["abcd", "bc"], [3, 2, 4])
    drive_stop(sol, [b"xx", b"abcdefg", b"y"], ["abcdefg", "cde", "def"], [0, 1, 2])


@case("Part 3", title="同时结束取更长的；一个是另一个的前缀")
def p3_tie_and_prefix(sol):
    pieces = [b"xab", b"x", b"a", b"b", b"c", b""]
    drive_stop(sol, pieces, ["b", "ab"], [0])
    drive_stop(sol, pieces, ["ab", "b"], [1, 2, 3])
    drive_stop(sol, pieces, ["ab", "abc"], [1, 2, 3, 4])
    drive_stop(sol, pieces, ["abc", "ab"], [2, 5, 3, 4])
    drive_stop(sol, [b"aaab", b"a", b"b"], ["aab", "ab"], [0])
    drive_stop(sol, [b"aaab", b"a", b"b"], ["ab", "aab"], [1, 1, 1, 2])


@case("Part 3", title="尽早返回：只留可能是开头的后缀")
def p3_hold_minimal(sol):
    pieces = [b"x", b"a", b"b", b"d", b"c", b"ab", b"aab", b""]
    drive_stop(sol, pieces, ["abc"], [0, 1, 2, 3, 1, 1, 2, 4])
    drive_stop(sol, pieces, ["aab", "abd"], [1, 1, 1, 2, 4])
    drive_stop(sol, pieces, ["xaab", "ac"], [0, 6, 0, 5, 3, 1, 4])
    drive_stop(sol, pieces, ["ba", "aaa"], [1, 1, 7, 2, 0, 1, 4])


@case("Part 3", title="停止串跨很多个 token 和半个字符")
def p3_across_tokens(sol):
    text = "ab你好😀cd你😀"
    data = text.encode()
    pieces = [data[k:k + 1] for k in range(len(data))] + [b""]
    ids = list(range(len(data)))
    for stops in (["你好😀c"], ["好😀", "b你"], ["😀cd你😀x"], ["d你😀"], ["你😀", "😀c"]):
        drive_stop(sol, pieces, stops, ids)


@case("Part 3", title="停止串含 U+FFFD；finish 时才出现")
def p3_replacement_char(sol):
    pieces = [b"x", b"\xe4\xbd", b"\xff", b"y", b""]
    drive_stop(sol, pieces, ["x\ufffd"], [0, 1])               # the U+FFFD only appears when finish() runs
    drive_stop(sol, pieces, ["\ufffdy"], [0, 2, 4, 3, 0])
    drive_stop(sol, pieces, ["\ufffd\ufffd"], [1, 2, 0])
    drive_stop(sol, pieces, ["x\ufffd"], [0, 1, 0])


@case("Part 3", title="随机回复和停止串（对照按定义的暴力）")
def p3_random(sol):
    r = rng()
    for _ in range(500):
        pieces = [ref.random_piece(r) for _ in range(r.randint(1, 8))]
        ids = [r.randrange(len(pieces)) for _ in range(r.randint(0, 16))]
        full = ref.ByteTokenizer(pieces).decode(ids)
        stops = []
        for _ in range(r.randint(0, 3)):
            if full and r.random() < 0.6:
                a = r.randrange(len(full))
                stops.append(full[a:a + r.randint(1, 4)])
            else:
                stops.append("".join(r.choice("a 中\ufffd") for _ in range(r.randint(1, 3))))
        drive_stop(sol, pieces, stops, ids, finish=r.random() < 0.8)


@case("Part 3", title="真实文本随机切开（对照暴力）")
def p3_random_bytes(sol):
    r = rng()
    for _ in range(150):
        text = "".join(r.choice("ab 你好") for _ in range(r.randint(0, 14)))
        pieces, ids = ref.cut_text(r, text, 40)
        stops = []
        for _ in range(r.randint(1, 3)):
            if text and r.random() < 0.7:
                a = r.randrange(len(text))
                stops.append(text[a:a + r.randint(1, 5)])
            else:
                stops.append("".join(r.choice("ab 你好") for _ in range(r.randint(1, 4))))
        drive_stop(sol, pieces, stops, ids)


@case("Part 3", cpu=3, title="7 万个短 token：不能每次扫全文")
def p3_long_tokens(sol):
    desc = (f"StopStreamDecoder(tok, {LONG_STOPS!r})，7 万个 token 的回复，每个 token 3 到 12 字节，"
            f"共 {STOP_BYTES:,} 字节；停止串的开头经常出现，但从不完整出现")
    calling(f"<{desc}>：依次 push 每个 token")
    d = sol.StopStreamDecoder(ref.ByteTokenizer(STOP_PIECES), list(LONG_STOPS))
    check_outputs(d.push, STOP_IDS, STOP_WANT, desc, STOP_PIECES)
    calling(f"<{desc}>：finish()")
    expect(d.finish(), STOP_WANT[-1], call=f"<{desc}>：最后的 finish()")
    expect_stopped(d, False, f"<{desc}>：d.stopped")
