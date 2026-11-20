"""Tests for the streaming tokenizer.

Part 2: every read() list and every pending() value must equal the statement's original class (ref.Given) for any
call sequence, and the total work must be linear (two big cases with CPU limits). Part 3: BoundedStreamTokenizer
used by one writer thread and one reader thread. Part 1 is prose and is not judged.
"""
import sys
import threading
import time

from judgelib import Mismatch, calling, case, expect, expect_raises, expect_true, rng
import ref

EX = ref.EX_VOCAB          # the statement's example vocabulary, L = 5


# ---------------------------------------------------------------- helpers

def head(vocab, unk, cls="StreamTokenizer", extra=""):
    shown = repr(vocab) if len(repr(vocab)) <= 600 else f"<{len(vocab)} 个词条的词表>"
    var = "tok" if cls == "StreamTokenizer" else "b"
    return f"{var} = {cls}({shown}, {unk}{extra})"


def replay(first, done, current=None, keep=12):
    lines = [first]
    if len(done) > keep:
        lines.append(f"# …（省略了中间 {len(done) - keep} 步）")
    lines += done[-keep:]
    if current:
        lines.append(f"{current}   # <- 这一步")
    return "\n".join(lines)


def as_list(got):
    return list(got) if isinstance(got, (list, tuple)) else got


def read_msg(code, got, want):
    if not isinstance(got, list):
        return f"{code} 应当返回一个 id 列表，得到的是 {type(got).__name__}"
    if len(got) < len(want) and got == want[:len(got)]:
        return (f"{code} 少交出了 token：原版此时能确定 {len(want)} 个，你只交出 {len(got)} 个"
                "（只要“已关闭”或“未分词的字符至少 L 个”，当前位置的 token 就已确定）")
    if len(got) > len(want) and got[:len(want)] == want:
        return (f"{code} 多交出了 token：原版此时只能交出 {len(want)} 个"
                "（要么已经取满 max_tokens 个，要么剩下的字符不足 L 个、流也没关闭，后面的文本还可能拼出更长的词条）")
    k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    return f"{code} 返回的第 {k} 个 id（从 0 开始）和原版不同"


def lockstep(sol, vocab, unk, ops):
    """Run ops on the answer and on the statement's original side by side; check every read() and pending()."""
    want_tok = ref.Given(dict(vocab), unk)
    first = head(vocab, unk)
    calling(first)
    tok = sol.StreamTokenizer(dict(vocab), unk)
    done = []
    for op in ops:
        want = ref.apply(want_tok, op)
        code = ref.op_code(op)
        call = replay(first, done, code)
        calling(call)
        got = ref.apply(tok, op)
        if op[0] == "read":
            got = as_list(got)
            expect(got, want, call=call, msg=read_msg(code, got, want) if got != want else "")
        elif op[0] == "pending":
            expect(got, want, call=call,
                   msg=f"{code} 应当是 {want}（已写入、还没变成 token 的字符数），得到 {got!r}" if got != want else "")
        done.append(code + (f"   # -> {got!r}" if op[0] in ("read", "pending") else ""))


def script(sol, vocab, unk, steps):
    """steps: (op, expected) pairs with hard-coded expectations (the statement's examples)."""
    first = head(vocab, unk)
    calling(first)
    tok = sol.StreamTokenizer(dict(vocab), unk)
    done = []
    for op, want in steps:
        code = ref.op_code(op)
        call = replay(first, done, code)
        calling(call)
        got = ref.apply(tok, op)
        if op[0] == "read":
            got = as_list(got)
            expect(got, want, call=call, msg=read_msg(code, got, want) if got != want else "")
        elif op[0] == "pending":
            expect(got, want, call=call, msg=f"{code} 应当是 {want}，得到 {got!r}" if got != want else "")
        done.append(code + (f"   # -> {got!r}" if op[0] in ("read", "pending") else ""))


W, R, P, C = (lambda s: ("write", s)), (lambda n: ("read", n)), ("pending",), ("close",)


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子一")
def p2_example_1(sol):
    script(sol, EX, 0, [
        (W("the"), None), (P, 3), (R(10), []),
        (W("re is"), None), (R(10), [8]),
        (C, None), (R(10), [5, 0, 0]),
        (R(10), []),
    ])


@case("Part 2", sample=True, title="题目例子二：每次只取 1 个")
def p2_example_2(sol):
    script(sol, EX, 0, [
        (W("t"), None), (R(1), []), (W("here i"), None), (R(1), [8]), (W("s"), None), (R(1), []),
        (C, None), (R(1), [5]), (R(1), [0]), (R(1), [0]), (R(1), []),
    ])


@case("Part 2", title="空词表、关闭后写入都抛 ValueError")
def p2_errors(sol):
    expect_raises(ValueError, sol.StreamTokenizer, {}, 0, call="StreamTokenizer({}, 0)",
                  msg="空词表应当抛出 ValueError")
    first = head(EX, 0)
    calling(first)
    tok = sol.StreamTokenizer(dict(EX), 0)
    done = []
    for op in (W("th"), C):
        calling(replay(first, done, ref.op_code(op)))
        ref.apply(tok, op)
        done.append(ref.op_code(op))
    for chunk in ("e", ""):
        call = replay(first, done, f"tok.write({chunk!r})")
        calling(call)
        expect_raises(ValueError, tok.write, chunk, call=call,
                      msg=f"close() 之后 write({chunk!r}) 应当抛出 ValueError（空串也一样）")
        done.append(f"tok.write({chunk!r})   # 抛出 ValueError")
    call = replay(first, done, "tok.read(5)")
    calling(call)
    got = as_list(tok.read(5))
    expect(got, [6], call=call,
           msg="流里只有关闭前写入的 'th'（词条，id 6），被拒绝的 write 不算数，所以 read(5) 应当正好得到 [6]")


@case("Part 2", title="read(0)、空块")
def p2_read_zero_and_empty(sol):
    lockstep(sol, EX, 0, [W(""), P, R(3), R(0), W("there is the"), R(0), P, R(1), P, W(""), R(0), P,
                          C, R(0), P, R(100), R(0), P, R(1)])
    lockstep(sol, {"ab": 7, "b": 9}, 0, [W("ab"), R(0), C, R(0), P, R(1), R(1), P])


@case("Part 2", title="恰好有 L 个字符时 token 才确定")
def p2_final_at_L(sol):
    lockstep(sol, EX, 0, [W("ther"), R(9), P, W("e"), R(9), P, W(" is"), R(9), W("x"), R(1), W("t"), R(9), P])
    r = rng("final")
    for L in (1, 2, 3, 4, 6, 9, 16):
        for _ in range(6):
            vocab = ref.random_vocab(r, "ab", max_keys=6, max_len=L)
            vocab.setdefault("".join(r.choice("ab") for _ in range(L)), 99)
            text = ref.random_text(r, "abc", 2 * L, 3 * L + 3)
            ops = [op for ch in text for op in (W(ch), R(100), P)] + [C, R(100), P]
            lockstep(sol, vocab, 0, ops)


@case("Part 2", title="每一个切点、每一对切点")
def p2_every_cut(sol):
    text = "there is the theatre, there"
    for i in range(len(text) + 1):
        lockstep(sol, EX, 0, [W(text[:i]), R(2), P, W(text[i:]), R(3), P, C, R(100), P])
    text = "the therethe re"
    sizes = (1, 10, 2)
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            k = i + j
            lockstep(sol, EX, 0, [W(text[:i]), R(sizes[k % 3]), P, W(text[i:j]), R(sizes[(k + 1) % 3]), P,
                                  W(text[j:]), R(sizes[(k + 2) % 3]), C, P] + [R(2)] * 9 + [P])


@case("Part 2", title="关闭后剩下的文本比长词条短")
def p2_close_short_tail(sol):
    vocabs = [{"ab": 1, "abc": 2, "abcd": 3, "b": 4}, {"abcd": 1, "c": 2}, {"aaaa": 1, "aa": 2, "a": 3},
              dict(EX)]
    tails = ["", "a", "ab", "abc", "ba", "aaa", "aab", "th", "the", "ther", "t h", "cab"]
    for vocab in vocabs:
        for tail in tails:
            lockstep(sol, vocab, 0, [W(tail), R(5), C, P] + [R(1), P] * (len(tail) + 1))
            lockstep(sol, vocab, 0, [W("abcd" * 3 + tail), R(1), R(100), P, C, R(100), P])
    r = rng("tail")
    for _ in range(300):
        vocab = ref.random_vocab(r, "ab", max_keys=6, max_len=5)
        text = ref.random_text(r, "abc", 0, 7)
        lockstep(sol, vocab, 0, [W(text), R(r.randint(0, 2)), C] + [R(r.randint(1, 2)), P] * (len(text) + 1))


@case("Part 2", title="pending() 要算上还没处理的块")
def p2_pending_queue(sol):
    lockstep(sol, EX, 0, [W("there"), P, W(" is"), P, R(1), P, W(" the"), P, R(1), P, W(" there"), P,
                          R(2), P, W(""), P, C, P, R(1), P, R(100), P])
    r = rng("pending")
    for _ in range(150):
        vocab = ref.random_vocab(r, "abc", max_keys=8, max_len=r.randint(1, 6))
        text = ref.random_text(r, "abcd", 5, 60)
        ops, i = [], 0
        while i < len(text):
            j = min(len(text), i + r.randint(0, 4))
            ops += [W(text[i:j]), P]
            i = j
            if r.random() < 0.3:
                ops += [R(r.randint(1, 3)), P]
        ops += [C, P] + [R(r.randint(1, 3)), P] * (len(text) + 1)
        lockstep(sol, vocab, 0, ops)


@case("Part 2", title="最长键长 L = 16")
def p2_long_keys(sol):
    r = rng("long")
    for _ in range(60):
        vocab = {"a" * k: k for k in range(1, 17)}
        for _ in range(r.randint(0, 8)):
            vocab.setdefault("".join(r.choice("ab") for _ in range(r.randint(2, 16))), 100 + len(vocab))
        if r.random() < 0.5:
            del vocab["a" * r.randint(1, 15)]
        runs = [("a" * r.randint(1, 40)) for _ in range(r.randint(1, 6))]
        text = "".join(run + r.choice(["b", "ab", "c", "bb"]) for run in runs)
        lockstep(sol, vocab, 0, ref.random_ops(r, text, max_piece=9, max_read=3))


@case("Part 2", title="中文、emoji、组合字符")
def p2_unicode(sol):
    vocab = {"你": 1, "你好": 2, "好": 3, "\U0001F642": 4, "\U0001F642\U0001F642": 5, "é": 6, "é": 7,
             "世界": 8, "a": 9, "你好世界": 10}
    lockstep(sol, vocab, 0, [W("你"), R(5), P, W("好世"), R(5), P, W("界\U0001F642"), R(5), P, C, R(5), P])
    r = rng("unicode")
    for _ in range(120):
        text = ref.random_text(r, ["你", "好", "世", "界", "\U0001F642", "é", "e", "́", "a", " "], 0, 30)
        lockstep(sol, vocab, r.choice([0, -1]), ref.random_ops(r, text, max_piece=4, max_read=3))


@case("Part 2", title="随机词表、随机切块和读法")
def p2_random(sol):
    r = rng("random")
    for _ in range(300):
        vocab = ref.random_vocab(r, r.choice(["ab", "abc"]), max_keys=8, max_len=r.randint(1, 5))
        text = ref.random_text(r, "abcd ", 0, 40)
        lockstep(sol, vocab, 0, ref.random_ops(r, text))


# The big document is built at import time, which is not charged to any case.
_pr = rng("perf")
PERF_VOCAB = ref.perf_vocab()
PERF_N = 1_200_000
PERF_TEXT = ref.perf_text(_pr, PERF_VOCAB, PERF_N)
PERF_IDS = ref.tokenize(PERF_TEXT, PERF_VOCAB, 0)
PERF_4K = [PERF_TEXT[i:i + 4096] for i in range(0, PERF_N, 4096)]
PERF_16 = [PERF_TEXT[i:i + 16] for i in range(0, PERF_N, 16)]
PERF_HEAD = (f"tok = StreamTokenizer({PERF_VOCAB!r}, 0)\n"
             f"# 文本：{PERF_N} 个字符，词表里的词（有的两个连在一起）、数字和其他字母，夹着中文词和 emoji，"
             f"共 {len(PERF_IDS)} 个 token")


def check_big(tok, ids):
    expect(ids, PERF_IDS)
    expect(tok.pending(), 0, msg="全部读完之后 pending() 应当是 0")


@case("Part 2", cpu=7, title="120 万字符按工单节奏：要求线性")
def p2_perf_ticket(sol):
    calling(PERF_HEAD + "\n# 像 drive() 一样：每 write 4096 个字符就 read(256)；写完后 close()，再 read(256) 直到返回 []")
    tok = sol.StreamTokenizer(dict(PERF_VOCAB), 0)
    ids = []
    for chunk in PERF_4K:
        tok.write(chunk)
        ids.extend(tok.read(256))
    tok.close()
    while True:
        got = tok.read(256)
        if not got:
            break
        ids.extend(got)
    check_big(tok, ids)


@case("Part 2", cpu=8, title="每次写 16 个字符、取 1 个：仍要线性")
def p2_perf_small_steps(sol):
    calling(PERF_HEAD + "\n# 每 write 16 个字符就 read(1)；写完后 close()，再 read(1) 直到返回 []")
    tok = sol.StreamTokenizer(dict(PERF_VOCAB), 0)
    ids = []
    for chunk in PERF_16:
        tok.write(chunk)
        ids += tok.read(1)
    tok.close()
    while True:
        got = tok.read(1)
        if not got:
            break
        ids += got
    check_big(tok, ids)


# ---------------------------------------------------------------- Part 3

class Pair:
    """Runs a writer and a reader thread; anything the answer raises in them is re-raised in the case."""

    def __init__(self, b, writer, reader):
        self.b, self.errors = b, []
        self.threads = [threading.Thread(target=self._wrap(writer), daemon=True),
                        threading.Thread(target=self._wrap(reader), daemon=True)]

    def _wrap(self, fn):
        def go():
            try:
                fn()
            except BaseException as e:      # noqa: BLE001
                self.errors.append(e)
        return go

    def run(self, call, limit=5.0):
        for t in self.threads:
            t.start()
        deadline = time.monotonic() + limit
        for t in self.threads:
            t.join(max(0.0, deadline - time.monotonic()))
        stuck = [name for t, name in zip(self.threads, ("写线程", "读线程")) if t.is_alive()]
        if stuck:
            try:
                self.b.close()          # wake a writer waiting on a full backlog, so it does not linger
            except BaseException:       # noqa: BLE001
                pass
            raise Mismatch(f"{limit:g} 秒后{'和'.join(stuck)}还没结束：可能死锁，或者在等一个永远不会来的唤醒"
                           "（每次写入、读走、关闭之后都要唤醒等待的一方）", call)
        if self.errors:
            raise self.errors[0]


def background(fn, *args):
    """Start fn(*args) in a daemon thread; returns (thread, box) where box gets 'value' or 'error'."""
    box = {}

    def go():
        try:
            box["value"] = fn(*args)
        except BaseException as e:      # noqa: BLE001
            box["error"] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t, box


def finished(t, box, timeout, reraise=True):
    t.join(timeout)
    if reraise and not t.is_alive() and "error" in box:
        raise box["error"]
    return not t.is_alive()


def split(r, text, lo=0, hi=12):
    pieces, i = [], 0
    while i < len(text):
        j = min(len(text), i + r.randint(lo, hi))
        pieces.append(text[i:j])
        i = j
    return pieces


def threads_run(sol, vocab, text, max_pending, r, writer_pause=0.0, reader_pause=0.0, limit=5.0,
                piece=(0, 12), batch=7, yield_p=0.3):
    """One writer thread writes text in random pieces and closes; one reader thread reads until []. Checks the
    ids, that read gives [] only after close() was called, and the backlog right after each write (only the
    reader can shrink it in between, so the value the writer sees is never above the real one)."""
    pieces = split(r, text, *piece)
    batches = [r.randint(1, batch) for _ in range(len(text) + 2)]
    yields = [r.random() < yield_p for _ in range(len(pieces))]
    want = ref.tokenize(text, vocab, 0)
    first = head(vocab, 0, "BoundedStreamTokenizer", f", {max_pending}")
    desc = (first + f"\n# 写线程：把 {len(text)} 个字符的文本分成 {len(pieces)} 块逐块 write，然后 close()"
            f"\n# 读线程：每次 read({'1' if batch == 1 else f'1..{batch}'}) 直到返回 []")
    if len(repr(text)) <= 600:
        desc += f"\n# 文本：{text!r}"
    calling(desc)
    b = sol.BoundedStreamTokenizer(dict(vocab), 0, max_pending)
    got, over, closing, early = [], [], [], []

    def writer():
        for piece, y in zip(pieces, yields):
            b.write(piece)
            p = b.pending()
            if not over and isinstance(p, int) and p > max_pending - 1 + len(piece):
                over.append((piece, p))
            if writer_pause:
                time.sleep(writer_pause)
            elif y:
                time.sleep(0)
        closing.append(True)            # set before close(), so a correct [] always sees it
        b.close()

    def reader():
        for n in batches:
            out = b.read(n)
            if not out:
                if not closing:
                    early.append(len(got))
                return
            got.extend(out)
            if reader_pause:
                time.sleep(reader_pause)

    try:
        Pair(b, writer, reader).run(desc, limit)
    except Mismatch:
        if not early:
            raise
    if early:
        raise Mismatch(f"读线程拿到 {early[0]} 个 id 之后 read 返回了 []，但写线程还没调用 close()（流没关闭时，"
                       "没有确定的 token 就要等待；只有流已关闭且全部读完才返回 []）", desc)
    if over:
        piece, p = over[0]
        raise Mismatch(f"write({piece!r}) 返回后积压 pending() = {p}，超过了 max_pending - 1 + len(chunk) = "
                       f"{max_pending - 1 + len(piece)}（积压 >= max_pending 时 write 要先等读线程取走一些）", desc)
    expect(got, want, call=desc, msg="" if got == want else "读线程拿到的 id 序列和单线程整段分词的结果不同")


@case("Part 3", sample=True, title="题目例子一（单线程）")
def p3_example(sol):
    first = "b = BoundedStreamTokenizer({'t': 1, 'h': 2, 'e': 3, 'r': 4, ' ': 5, 'th': 6, 'the': 7, 'there': 8}, 0, 8)"
    calling(first)
    b = sol.BoundedStreamTokenizer(dict(EX), 0, 8)
    done = []
    for op, want in [(W("the"), None), (W("re is"), None), (P, 8), (R(10), [8]), (C, None), (R(10), [5, 0, 0]),
                     (R(10), [])]:
        code = ref.op_code(op).replace("tok.", "b.")
        call = replay(first, done, code)
        calling(call)
        got = ref.apply(b, op)
        if op[0] == "read":
            got = as_list(got)
            expect(got, want, call=call, msg=read_msg(code, got, want) if got != want else "")
        elif op[0] == "pending":
            expect(got, want, call=call)
        done.append(code + (f"   # -> {got!r}" if want is not None else ""))


def allowed(sol, vocab, max_pending, call):
    """Construct with max_pending >= L, which the statement allows."""
    calling(call)
    try:
        return sol.BoundedStreamTokenizer(dict(vocab), 0, max_pending)
    except ValueError as e:
        raise Mismatch(f"max_pending = {max_pending} 不小于最长键长 L，是允许的，构造函数不应抛出 ValueError（{e}）",
                       call, show=False) from None


@case("Part 3", title="各种 ValueError")
def p3_errors(sol):
    for mp in (4, 0, -1):
        call = f"BoundedStreamTokenizer(vocab, 0, {mp})   # vocab 是题目例子的词表，L = 5"
        expect_raises(ValueError, sol.BoundedStreamTokenizer, dict(EX), 0, mp, call=call,
                      msg=f"max_pending = {mp} 小于最长键长 L = 5，构造函数应当抛出 ValueError")
    call = "b = BoundedStreamTokenizer(vocab, 0, 5)   # max_pending == L 是允许的"
    calling(call)
    b = allowed(sol, EX, 5, call)
    calling(call + "\nb.write('there is')")
    b.write("there is")
    for n in (0, -1):
        c = call + f"\nb.write('there is')\nb.read({n})"
        calling(c)
        t, box = background(b.read, n)         # in a thread, so an answer that waits here cannot hang the case
        expect_true(finished(t, box, 3.0, reraise=False),
                    f"read({n})：max_tokens 小于 1 时应当立刻抛出 ValueError，但它一直阻塞着", c)
        err = box.get("error")
        if err is not None and not isinstance(err, ValueError):
            raise Mismatch(f"read({n}) 应当抛出 ValueError，实际抛出 {type(err).__name__}: {err}", c, show=False)
        expect_true(err is not None, f"read({n})：max_tokens 小于 1 时应当抛出 ValueError，但它正常返回了", c,
                    got=box.get("value"))
    c = call + "\nb.write('there is')\nb.read(1)"
    calling(c)
    got = as_list(b.read(1))
    expect(got, [8], call=c)
    calling(c + "\nb.close()")
    b.close()
    for chunk in ("x", ""):
        c2 = c + f"\nb.close()\nb.write({chunk!r})"
        expect_raises(ValueError, b.write, chunk, call=c2, msg=f"close() 之后 write({chunk!r}) 应当抛出 ValueError")
    c3 = c + "\nb.close()\nb.read(10)"
    calling(c3)
    expect(as_list(b.read(10)), [5, 0, 0], call=c3)
    calling(c3 + "\nb.read(10)")
    expect(as_list(b.read(10)), [], call=c3 + "\nb.read(10)",
           msg="流已关闭、所有 token 都读完之后，read 应当返回 []")
    allowed(sol, {"abcdefghijklmnop": 1, "a": 2}, 16, "BoundedStreamTokenizer({'abcdefghijklmnop': 1, 'a': 2}, 0, 16)")
    expect_raises(ValueError, sol.BoundedStreamTokenizer, {"abcdefghijklmnop": 1, "a": 2}, 0, 15,
                  call="BoundedStreamTokenizer({'abcdefghijklmnop': 1, 'a': 2}, 0, 15)",
                  msg="max_pending = 15 小于最长键长 L = 16，构造函数应当抛出 ValueError")


@case("Part 3", title="积压满了 write 要等，降下来才写")
def p3_writer_blocks(sol):
    first = "b = BoundedStreamTokenizer({'t': 1, 'h': 2, 'e': 3, 'r': 4, ' ': 5, 'th': 6, 'the': 7, 'there': 8}, 0, 5)"
    log = [first, "b.write('there is the')   # 积压 12 个字符 >= max_pending = 5"]
    calling("\n".join(log))
    b = sol.BoundedStreamTokenizer(dict(EX), 0, 5)
    b.write("there is the")
    log.append("# 另一个线程：b.write(' there')")
    t, box = background(b.write, " there")
    call = "\n".join(log)
    expect_true(not finished(t, box, 0.3), "积压已经 >= max_pending，另一个线程的 write 应当阻塞，但它直接返回了", call)
    for n, want, left in ((1, [8], 7), (2, [5, 0], 5), (1, [0], 4)):
        log.append(f"b.read({n})")
        call = "\n".join(log)
        calling(call)
        got = as_list(b.read(n))
        expect(got, want, call=call)
        log[-1] += f"   # -> {want!r}，积压剩 {left} 个字符"
        call = "\n".join(log)
        if left >= 5:
            expect_true(not finished(t, box, 0.3),
                        f"积压还有 {left} 个字符 >= max_pending = 5，被挡住的 write 应当继续等待，"
                        "但它已经返回了（被唤醒后要重新检查条件）", call)
    expect_true(finished(t, box, 5.0),
                "积压降到 4 < max_pending = 5 之后，被挡住的 write 应当被唤醒并写入，但 5 秒后它还没返回"
                "（读走 token 之后要唤醒写线程）", call)
    log.append("# 另一个线程的 write 返回了")
    log.append("b.pending()")
    call = "\n".join(log)
    calling(call)
    expect(b.pending(), 4 + 6, call=call, msg="write 应当把整块写进去：积压应当是 4 + 6 = 10")
    log += ["b.close()", "b.read(100)"]
    call = "\n".join(log)
    calling(call)
    b.close()
    expect(as_list(b.read(100)), ref.tokenize(" the there", EX, 0), call=call)


@case("Part 3", title="没有确定的 token 时 read 要等")
def p3_reader_waits(sol):
    first = "b = BoundedStreamTokenizer({'t': 1, 'h': 2, 'e': 3, 'r': 4, ' ': 5, 'th': 6, 'the': 7, 'there': 8}, 0, 5)"
    log = [first, "# 另一个线程：b.read(3)"]
    calling("\n".join(log))
    b = sol.BoundedStreamTokenizer(dict(EX), 0, 5)
    t, box = background(b.read, 3)
    expect_true(not finished(t, box, 0.3), "还什么都没写，read 应当等待，但它直接返回了", "\n".join(log),
                got=box.get("value"))
    log.append("b.write('th')")
    calling("\n".join(log))
    b.write("th")
    expect_true(not finished(t, box, 0.3), "只有 2 个字符 < L = 5，后面可能拼成 'the' 或 'there'，"
                "还没有确定的 token，read 应当继续等待，但它返回了", "\n".join(log), got=box.get("value"))
    log.append("b.write('ere i')   # 现在是 'there i'，7 >= L")
    calling("\n".join(log))
    b.write("ere i")
    expect_true(finished(t, box, 5.0), "写入 'ere i' 之后 'there' 已经确定，等待中的 read 应当被唤醒，"
                "但 5 秒后它还没返回（写入之后要唤醒读线程）", "\n".join(log))
    expect(as_list(box.get("value")), [8], call="\n".join(log))
    log += ["# 读线程返回了 [8]", "# 另一个线程：b.read(3)"]
    t, box = background(b.read, 3)
    expect_true(not finished(t, box, 0.3), "剩下 ' i' 只有 2 个字符、流没关闭，read 应当等待，但它返回了",
                "\n".join(log), got=box.get("value"))
    log.append("b.close()")
    calling("\n".join(log))
    b.close()
    expect_true(finished(t, box, 5.0), "close() 之后剩下的 token 都确定了，等待中的 read 应当被唤醒，"
                "但 5 秒后它还没返回（关闭之后也要唤醒读线程）", "\n".join(log))
    expect(as_list(box.get("value")), [5, 0], call="\n".join(log))
    log += ["# 读线程返回了 [5, 0]", "b.read(3)"]
    calling("\n".join(log))
    expect(as_list(b.read(3)), [], call="\n".join(log), msg="流已关闭、所有 token 都读完之后，read 应当返回 []")


@case("Part 3", cpu=3, title="读得慢：积压有上限")
def p3_slow_reader(sol):
    r = rng("slow-reader")
    vocab = ref.perf_vocab()
    for mp in (8, 9, 16, 50):
        threads_run(sol, vocab, ref.perf_text(r, vocab, 1500), mp, r, reader_pause=0.0003)


@case("Part 3", cpu=3, title="写得慢：读线程要等，不能提前结束")
def p3_slow_writer(sol):
    r = rng("slow-writer")
    vocab = ref.perf_vocab()
    for mp in (8, 9, 16, 50):
        threads_run(sol, vocab, ref.perf_text(r, vocab, 600), mp, r, writer_pause=0.0003)


@case("Part 3", cpu=4, title="随机交错 150 次")
def p3_random_threads(sol):
    r = rng("threads")
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)         # switch threads very often, so more interleavings get tried
    try:
        for k in range(150):
            vocab = ref.random_vocab(r, r.choice(["ab", "abc"]), max_keys=8, max_len=r.randint(1, 5))
            L = max(len(key) for key in vocab)
            text = ref.random_text(r, "abcd ", 50, 400)
            threads_run(sol, vocab, text, r.choice([L, L + 1, L + 3, 16, 50]), r)
    finally:
        sys.setswitchinterval(old)


@case("Part 3", cpu=8, title="压力测试：小块写、每次读 1 个，100 轮")
def p3_stress(sol):
    # Many short writes racing with read(1): an answer that touches the shared state outside the lock loses or
    # duplicates text here now and then; each round has a small chance to show it, so there are many rounds.
    r = rng("stress")
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        for k in range(100):
            vocab = ref.random_vocab(r, "abc", max_keys=8, max_len=r.randint(3, 8))
            L = max(len(key) for key in vocab)
            text = ref.random_text(r, "abcd ", 1000, 1000)
            threads_run(sol, vocab, text, r.choice([L + 8, 32, 64]), r, piece=(1, 3), batch=1, yield_p=0.5)
    finally:
        sys.setswitchinterval(old)
