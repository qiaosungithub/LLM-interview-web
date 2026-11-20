import ast
import io
import time
import tokenize

from judgelib import Mismatch, calling, case, expect, rng
import ref

SHOW = 18                        # a random replay lists at most this many calls


class _Raise:
    def __init__(self, exc):
        self.exc = exc

    def __repr__(self):
        return self.exc


IERR, VERR = _Raise("IndexError"), _Raise("ValueError")     # the step must raise this
RAISES = {"IndexError": IERR, "ValueError": VERR}
EXC = {"IndexError": IndexError, "ValueError": ValueError}
SKIP = object()                  # the step's return value is not checked (it must not raise)
MUTATORS = ("insert", "delete", "undo", "redo")


def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    return repr(x)


# ================================================================ Parts 1-3: one TextBuffer

def _code(op):
    name, obj, method, args = op
    if method == "__len__":
        return f"len({obj})"
    call = f"{method}({', '.join(_lit(a) for a in args)})"
    if obj is not None:
        call = f"{obj}.{call}"
    return f"{name} = {call}" if name else call


def _replay(ops, wants, k, mark="这一步", show=SHOW, extra=None):
    """ops[:k + 1] as code to paste into 自定义运行; wants[i] says what step i returned (or that it raised). With
    `extra` (a line such as print(buf.text())), step k is shown like the others and `extra` is the marked line."""
    entries = []
    for i in range(k + 1):
        op, want, code = ops[i], wants[i], _code(ops[i])
        raises = isinstance(want, _Raise)
        if i == k and extra is None:
            note = f"# <- {mark}" + (f"：应当抛出 {want.exc}" if raises else "")
            entries.append([(code if raises or want is SKIP else f"print({code})", note)])
        elif raises:
            entries.append([("try: " + code, ""), (f"except {want.exc}: pass", f"# 抛出 {want.exc}")])
        elif want is SKIP:
            entries.append([(code, "")])
        else:
            entries.append([(f"print({code})", f"# -> {_lit(want)}")])
    head = [e for i, e in enumerate(entries) if ops[i][1] is None]          # constructors are always kept
    body = [e for i, e in enumerate(entries) if ops[i][1] is not None]
    skipped = max(0, len(body) - show)
    rows = [row for e in head for row in e]
    if skipped:
        rows.append((f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）", ""))
    rows += [row for e in body[skipped:] for row in e]
    if extra is not None:
        rows.append((extra, f"# <- {mark}"))
    width = max(len(code) for code, note in rows if note) + 3 if any(note for _, note in rows) else 0
    return "\n".join(code.ljust(width) + note if note else code for code, note in rows)


def _model_step(model, op):
    name, obj, method, args = op
    if obj is None:
        model[name] = ref.TextBuffer(*args)
        return SKIP
    m = model[obj]
    try:
        out = len(m) if method == "__len__" else getattr(m, method)(*args)
    except IndexError:
        return IERR
    except ValueError:
        return VERR
    return SKIP if method == "insert" else out


def _same(got, want):
    if isinstance(got, tuple):
        got = list(got)
    try:
        return bool(got == want)
    except Exception:
        return False


def _outcome(sol, ops, wants=None, track=True, show=SHOW):
    """Runs ops on fresh buffers of the answer, checking each step as soon as it returns: the result (from `wants`,
    or from ref when wants is None) and, with `track`, the buffer's text after every insert/delete/undo/redo.
    Returns None when every step is right, else what went wrong."""
    env, model, seen = {}, {}, []
    for k, op in enumerate(ops):
        name, obj, method, args = op
        want = _model_step(model, op)
        if wants is not None:
            want = wants[k]
        seen.append(want)
        calling(_replay(ops, seen, k, show=show))           # shown if the answer raises or hangs in this call
        if obj is None:
            cls = getattr(sol, method, None)
            if not isinstance(cls, type):
                raise Mismatch(f"答案里没有类 {method}（名字要和题目一致）", show=False)
            env[name] = cls(*args)
            continue
        target = env[obj]
        if method != "__len__" and not callable(getattr(target, method, None)):
            raise Mismatch(f"{type(target).__name__} 没有方法 {method}（名字要和题目一致）",
                           call=_replay(ops, seen, k, show=show), show=False)
        try:
            got = len(target) if method == "__len__" else getattr(target, method)(*args)
        except Exception as e:
            raises = isinstance(want, _Raise)
            if not (raises and isinstance(e, EXC[want.exc])):
                unfinished = isinstance(e, NotImplementedError)            # the starter: report as RE, not WA
                return {"k": k, "kind": "wrong_exc" if raises and not unfinished else "exc", "exc": e, "wants": seen}
        else:
            if isinstance(want, _Raise):
                return {"k": k, "kind": "no_raise", "got": got, "wants": seen}
            if want is not SKIP and not _same(got, want):
                return {"k": k, "kind": "value", "got": got, "want": want, "wants": seen}
        if track and method in MUTATORS:
            calling(_replay(ops, seen, k, show=show, extra=f"print({obj}.text())"))
            text = target.text()
            if text != model[obj].s:
                return {"k": k, "kind": "state", "got": text, "want": model[obj].s, "wants": seen}
    return None


def _report(ops, bad, why, show=SHOW):
    k, kind = bad["k"], bad["kind"]
    want = bad["wants"][k]
    msg = f"{why}。" if why and kind not in ("value", "state") else why
    if kind == "exc":
        calling(_replay(ops, bad["wants"], k, mark="这一步抛出异常", show=show))
        raise bad["exc"]
    if kind == "state":
        call = _replay(ops, bad["wants"], k, show=show, extra=f"print({ops[k][1]}.text())")
        expect(bad["got"], bad["want"], call=call, msg=why or f"{ops[k][2]} 之后缓冲区的内容不对")
    call = _replay(ops, bad["wants"], k, show=show)
    if kind == "wrong_exc":
        e = bad["exc"]
        text = f"{type(e).__name__}: {e}"
        text = text if len(text) <= 300 else text[:300] + " …"
        raise Mismatch(f"{msg}应当抛出 {want.exc}，实际抛出的是 {text}", call=call, show=False)
    if kind == "no_raise":
        raise Mismatch(f"{msg}应当抛出 {want.exc}，却正常返回了", call=call, got=bad["got"])
    expect(bad["got"], bad["want"], call=call, msg=msg)


class Script:
    """A hand-written sequence: the expected result of every step, and why, shown when that step goes wrong."""

    def __init__(self, track):
        self.ops, self.wants, self.whys, self.track = [], [], [], track

    def add(self, op, want, why=""):
        self.ops.append(op)
        self.wants.append(want)
        self.whys.append(why)

    def run(self, sol):
        bad = _outcome(sol, self.ops, self.wants, track=self.track, show=40)   # short enough to show in full
        if bad is not None:
            _report(self.ops, bad, self.whys[bad["k"]], show=40)


def _split(line):
    """(code before '->', expected-value text after it, comment) of one script line."""
    toks = list(tokenize.generate_tokens(io.StringIO(line).readline))
    cut = next((t.start[1] for t in toks if t.type == tokenize.COMMENT), len(line))
    why = line[cut:].lstrip("#").strip()
    arrow = next((t for t in toks if t.type == tokenize.OP and t.string == "->"), None)
    if arrow is None:
        return line[:cut], None, why
    return line[:arrow.start[1]], line[arrow.end[1]:cut].strip(), why


def parse(script, track=True):
    """Lines such as  buf = TextBuffer("ab")  or  buf.insert(1, "x")  or  buf.read(0, 2) -> "ax"   # why  or
    len(buf) -> 3  or  buf.insert(9, "x") -> IndexError   # why."""
    s = Script(track)
    for line in script.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        call, expr, why = _split(line)
        want = SKIP if expr is None else RAISES.get(expr) or ast.literal_eval(expr)
        stmt = ast.parse(call.strip()).body[0]
        name = stmt.targets[0].id if isinstance(stmt, ast.Assign) else None
        node = stmt.value
        if isinstance(node.func, ast.Attribute):
            obj, method, args = node.func.value.id, node.func.attr, node.args
        elif node.func.id == "len":
            obj, method, args = node.args[0].id, "__len__", []
        else:
            obj, method, args = None, node.func.id, node.args
        s.add((name, obj, method, tuple(ast.literal_eval(a) for a in args)), want, why)
    return s


def run(sol, script, track=True):
    parse(script, track).run(sol)


def _shrink(sol, ops, bad, budget=0.5):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 1, False                   # ops[0] is the constructor
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, cand)
            if out is not None and out["kind"] == bad["kind"] and type(out.get("exc")) is type(bad.get("exc")):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def random_check(sol, part, sequences):
    r = rng(f"part{part}")
    for _ in range(sequences):
        ops = ref.random_ops(r, part, r.randint(10, 50))
        bad = _outcome(sol, ops)
        if bad is None:
            continue
        ops, bad = _shrink(sol, ops, bad)
        _report(ops, bad, f"随机操作序列的第 {len(ops) - 1} 步不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子 draft notes")
def p1_example(sol):
    run(sol, """
        buf = TextBuffer("draft notes")
        len(buf) -> 11
        buf.insert(5, " short")
        buf.text() -> "draft short notes"
        buf.read(0, 5) -> "draft"
        buf.read(5, 11) -> " short"
        buf.delete(5, 11) -> " short"
        buf.text() -> "draft notes"
        buf.insert(100, "x") -> IndexError
        buf.delete(3, 1) -> IndexError          # start > end
    """, track=False)


@case("Part 1", title="空缓冲区与默认参数")
def p1_empty(sol):
    run(sol, """
        buf = TextBuffer()
        len(buf) -> 0
        buf.text() -> ""
        buf.read(0, 0) -> ""
        buf.delete(0, 0) -> ""
        buf.insert(0, "")
        buf.text() -> ""
        buf.insert(1, "x") -> IndexError        # 空缓冲区里 pos 只能是 0
        buf.insert(0, "abc")
        buf.text() -> "abc"
        len(buf) -> 3
        buf.delete(0, 3) -> "abc"
        len(buf) -> 0
        buf.text() -> ""
    """)


@case("Part 1", title="两端：在末尾插入、删到末尾")
def p1_ends(sol):
    run(sol, """
        buf = TextBuffer("abc")
        buf.insert(3, "de")                     # pos == len(buf)：接在末尾
        buf.text() -> "abcde"
        buf.insert(0, ">")
        buf.text() -> ">abcde"
        buf.read(6, 6) -> ""
        buf.read(0, 6) -> ">abcde"
        buf.delete(6, 6) -> ""
        buf.delete(5, 6) -> "e"
        buf.delete(0, 1) -> ">"
        buf.text() -> "abcd"
        buf.insert(4, "!")
        buf.text() -> "abcd!"
        len(buf) -> 5
    """)


@case("Part 1", title="越界一律 IndexError，且不改变内容")
def p1_bounds(sol):
    run(sol, """
        buf = TextBuffer("hello")
        buf.insert(-1, "x") -> IndexError       # pos 不能是负数
        buf.insert(6, "x") -> IndexError        # pos 最大是 len(buf) = 5
        buf.delete(-1, 2) -> IndexError         # start 不能是负数
        buf.delete(0, 6) -> IndexError          # end 不能超过 len(buf)
        buf.delete(3, 2) -> IndexError          # start > end
        buf.delete(-3, -1) -> IndexError        # 负数不是“从末尾数”的下标
        buf.read(-2, 5) -> IndexError
        buf.read(4, 9) -> IndexError
        buf.read(2, 1) -> IndexError
        buf.read(-3, -1) -> IndexError
        buf.read(5, 5) -> ""
        buf.text() -> "hello"                   # 抛出 IndexError 的调用不改变缓冲区
        buf.insert(-5, "x") -> IndexError
        buf.text() -> "hello"
        len(buf) -> 5
    """)


@case("Part 1", title="非 ASCII 字符：按字符计数")
def p1_unicode(sol):
    run(sol, """
        buf = TextBuffer("naïve 中文")
        len(buf) -> 8                           # 按字符计数，不是按字节
        buf.read(2, 3) -> "ï"
        buf.insert(6, "😀")
        buf.text() -> "naïve 😀中文"
        len(buf) -> 9
        buf.delete(7, 9) -> "中文"
        buf.insert(0, "é")
        buf.text() -> "énaïve 😀"
        buf.read(7, 8) -> "😀"
    """)


@case("Part 1", title="随机操作序列（对照参考实现）")
def p1_random(sol):
    random_check(sol, 1, 60)


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子 cat / wildcats")
def p2_example(sol):
    run(sol, """
        buf = TextBuffer("cat")
        buf.insert(3, "s")
        buf.text() -> "cats"
        buf.insert(0, "wild")
        buf.text() -> "wildcats"
        buf.undo() -> True
        buf.text() -> "cats"
        buf.undo() -> True
        buf.text() -> "cat"
        buf.redo() -> True
        buf.text() -> "cats"
        buf.insert(4, "!")
        buf.text() -> "cats!"
        buf.redo() -> False                     # 新的编辑丢弃了待重做的 "wild"
        buf.undo() -> True
        buf.text() -> "cats"
        buf.undo() -> True                      # 重做过的 insert(3, "s") 回到了撤销栈上，可以再撤销
        buf.text() -> "cat"
        buf.undo() -> False                     # 没有可撤销的了
        buf.text() -> "cat"
    """, track=False)


@case("Part 2", title="撤销和重做 delete")
def p2_undo_delete(sol):
    run(sol, """
        buf = TextBuffer("hello world")
        buf.delete(5, 11) -> " world"
        buf.undo() -> True                      # 撤销 delete：把删掉的文字放回原处
        buf.text() -> "hello world"
        buf.redo() -> True
        buf.text() -> "hello"
        buf.undo() -> True
        buf.delete(0, 6) -> "hello "
        buf.insert(5, "!")
        buf.text() -> "world!"
        buf.undo() -> True
        buf.text() -> "world"
        buf.undo() -> True
        buf.text() -> "hello world"
        buf.undo() -> False                     # 初始内容不是一次操作
        buf.text() -> "hello world"
    """)


@case("Part 2", title="空操作不被记录")
def p2_noop_not_recorded(sol):
    run(sol, """
        buf = TextBuffer("abc")
        buf.insert(1, "")                       # 什么都没改变，不记录
        buf.delete(2, 2) -> ""                  # 空区间，不记录
        buf.undo() -> False                     # 两个空操作都没有被记录，没有可撤销的
        buf.text() -> "abc"
        buf.insert(3, "d")
        buf.insert(0, "")
        buf.delete(1, 1) -> ""
        buf.undo() -> True                      # 撤销的是 insert(3, "d")
        buf.text() -> "abc"
        buf.undo() -> False                     # 只记录过一次操作，已经撤销了
    """)


@case("Part 2", title="空操作不清空重做栈")
def p2_noop_keeps_redo(sol):
    run(sol, """
        buf = TextBuffer("abc")
        buf.insert(3, "d")
        buf.undo() -> True
        buf.insert(0, "")                       # 空操作不影响任何一个栈
        buf.delete(1, 1) -> ""
        buf.redo() -> True                      # 待重做的 insert(3, "d") 还在
        buf.text() -> "abcd"
        buf.undo() -> True
        buf.text() -> "abc"
    """)


@case("Part 2", title="越界的调用不影响两个栈")
def p2_errors_keep_stacks(sol):
    run(sol, """
        buf = TextBuffer("abc")
        buf.insert(3, "d")
        buf.insert(0, "z")
        buf.undo() -> True
        buf.insert(9, "x") -> IndexError        # 没有改变缓冲区，所以不记录，也不清空重做栈
        buf.delete(2, 9) -> IndexError
        buf.delete(-1, 1) -> IndexError
        buf.redo() -> True                      # 待重做的 insert(0, "z") 还在
        buf.text() -> "zabcd"
        buf.undo() -> True
        buf.undo() -> True
        buf.text() -> "abc"
        buf.undo() -> False                     # insert(3, "d") 和 insert(0, "z") 都撤销了
    """)


@case("Part 2", title="连续撤销、连续重做")
def p2_chain(sol):
    run(sol, """
        buf = TextBuffer()
        buf.insert(0, "a")
        buf.insert(1, "b")
        buf.insert(2, "c")
        buf.delete(0, 1) -> "a"
        buf.text() -> "bc"
        buf.undo() -> True
        buf.undo() -> True
        buf.undo() -> True
        buf.text() -> "a"
        buf.redo() -> True                      # undo 本身不是新的操作，不会清空重做栈
        buf.redo() -> True                      # 第一次 redo 不会清空剩下的重做栈
        buf.text() -> "abc"
        buf.redo() -> True
        buf.text() -> "bc"
        buf.redo() -> False                     # 三次撤销都重做完了
        buf.undo() -> True
        buf.undo() -> True
        buf.undo() -> True
        buf.undo() -> True
        buf.text() -> ""
        buf.undo() -> False                     # 四次操作都撤销了
        buf.redo() -> True
        buf.text() -> "a"
    """)


@case("Part 2", title="重做过的操作还能再撤销")
def p2_redo_then_undo(sol):
    run(sol, """
        buf = TextBuffer("xy")
        buf.insert(1, "123")
        buf.undo() -> True
        buf.redo() -> True
        buf.undo() -> True                      # 重做过的操作回到撤销栈上
        buf.text() -> "xy"
        buf.redo() -> True
        buf.delete(0, 2) -> "x1"
        buf.undo() -> True
        buf.redo() -> True
        buf.undo() -> True
        buf.text() -> "x123y"
        buf.undo() -> True
        buf.text() -> "xy"
        buf.undo() -> False                     # insert(1, "123") 和 delete(0, 2) 都撤销了
    """)


@case("Part 2", title="新的 delete 也清空重做栈")
def p2_new_edit_clears_redo(sol):
    run(sol, """
        buf = TextBuffer("abcdef")
        buf.delete(0, 2) -> "ab"
        buf.undo() -> True
        buf.delete(4, 6) -> "ef"                # 新的编辑清空重做栈，delete 也一样
        buf.redo() -> False
        buf.text() -> "abcd"
        buf.undo() -> True
        buf.text() -> "abcdef"
        buf.undo() -> False                     # delete(0, 2) 早已撤销，delete(4, 6) 也撤销了
    """)


@case("Part 2", title="随机操作序列（对照参考实现）")
def p2_random(sol):
    random_check(sol, 2, 60)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子 the cat sat on the mat")
def p3_example(sol):
    run(sol, """
        buf = TextBuffer("")
        buf.insert(0, "the cat sat on the mat")
        buf.suggest("ca", 5) -> ["cat"]
        buf.suggest("t", 5) -> ["the"]
        buf.suggest("", 3) -> ["the", "cat", "mat"]
        buf.delete(4, 8) -> "cat "
        buf.suggest("ca", 5) -> []
    """, track=False)


@case("Part 3", title="频次相同：大写在小写之前")
def p3_ties(sol):
    run(sol, """
        buf = TextBuffer("apple Zoo apple Zoo bee")
        buf.suggest("", 3) -> ["Zoo", "apple", "bee"]           # 频次相同按字符串顺序，"Zoo" 在 "apple" 之前
        buf.suggest("", 2) -> ["Zoo", "apple"]
        buf.insert(0, "bee ")
        buf.suggest("", 5) -> ["Zoo", "apple", "bee"]           # 三个词都出现两次
        buf.insert(0, "Bee bee ")
        buf.suggest("", 5) -> ["bee", "Zoo", "apple", "Bee"]
        buf.suggest("B", 5) -> ["Bee"]
        other = TextBuffer("b ab a ba aa B A AB Ba")
        other.suggest("", 10) -> ["A", "AB", "B", "Ba", "a", "aa", "ab", "b", "ba"]
    """)


@case("Part 3", title="k = 0、k 大于匹配数、k < 0")
def p3_k(sol):
    run(sol, """
        buf = TextBuffer("one two three two")
        buf.suggest("t", 0) -> []
        buf.suggest("t", 1) -> ["two"]
        buf.suggest("t", 10) -> ["two", "three"]               # 不足 k 个就全部返回
        buf.suggest("", -1) -> ValueError
        buf.suggest("zzz", -1) -> ValueError                   # k < 0 一律抛出 ValueError
        buf.suggest("zzz", 2) -> []
        empty = TextBuffer()
        empty.suggest("", 3) -> []
        empty.suggest("", -2) -> ValueError
    """)


@case("Part 3", title="大小写敏感")
def p3_case(sol):
    run(sol, """
        buf = TextBuffer("The the THE tHe the")
        buf.suggest("t", 5) -> ["the", "tHe"]                  # "The" 不以 "t" 开头
        buf.suggest("T", 5) -> ["THE", "The"]
        buf.suggest("th", 5) -> ["the"]
        buf.suggest("", 9) -> ["the", "THE", "The", "tHe"]
    """)


@case("Part 3", title="只有 A–Z、a–z 组成单词")
def p3_word_chars(sol):
    run(sol, """
        buf = TextBuffer("abc123def don't x_y e-mail 42 A1B2")
        buf.suggest("", 20) -> ["A", "B", "abc", "def", "don", "e", "mail", "t", "x", "y"]    # 数字、_、'、- 都把单词隔开
        buf.suggest("abc", 5) -> ["abc"]                       # 前缀本身就是一个词，也要返回
        buf.suggest("d", 5) -> ["def", "don"]
        buf.suggest("abc1", 5) -> []
        buf.suggest(" ", 5) -> []
    """)


@case("Part 3", title="非 ASCII 字母不算单词的一部分")
def p3_non_ascii(sol):
    run(sol, """
        buf = TextBuffer("café naïve Straße 中文abc ÀB")
        buf.suggest("", 20) -> ["B", "Stra", "abc", "caf", "e", "na", "ve"]    # é、ï、ß、À、汉字都不是 A–Z、a–z
        buf.suggest("caf", 5) -> ["caf"]
        buf.suggest("é", 5) -> []
    """)


@case("Part 3", title="前缀本身就是一个词")
def p3_prefix_is_word(sol):
    run(sol, """
        buf = TextBuffer("cat cats category cat dog")
        buf.suggest("cat", 5) -> ["cat", "category", "cats"]   # 以 "cat" 开头的词包括 "cat" 自己
        buf.suggest("cats", 5) -> ["cats"]
        buf.suggest("dog", 1) -> ["dog"]
        buf.suggest("dogs", 1) -> []
        buf.suggest("c", 2) -> ["cat", "category"]
    """)


@case("Part 3", title="编辑改变单词边界，撤销后词表复原")
def p3_word_boundaries(sol):
    run(sol, """
        buf = TextBuffer("cat dog")
        buf.insert(3, "s")                                     # "cats dog"：cat 变成了 cats
        buf.suggest("c", 5) -> ["cats"]
        buf.delete(4, 5) -> " "                                # "catsdog"：两个词连成一个
        buf.suggest("", 5) -> ["catsdog"]
        buf.insert(2, " ")                                     # "ca tsdog"
        buf.suggest("", 5) -> ["ca", "tsdog"]
        buf.undo() -> True                                     # 撤销之后词表跟着变回去
        buf.suggest("", 5) -> ["catsdog"]
        buf.undo() -> True
        buf.suggest("", 5) -> ["cats", "dog"]
        buf.redo() -> True
        buf.suggest("d", 5) -> []
        buf.insert(0, "dog ")
        buf.suggest("", 5) -> ["catsdog", "dog"]
    """)


@case("Part 3", title="频次随删除变化")
def p3_frequency(sol):
    run(sol, """
        buf = TextBuffer("go go stop go stop")
        buf.suggest("", 5) -> ["go", "stop"]
        buf.delete(0, 6) -> "go go "                           # go 还剩一次，stop 两次
        buf.suggest("", 5) -> ["stop", "go"]
        buf.delete(5, 7) -> "go"
        buf.text() -> "stop  stop"
        buf.suggest("g", 5) -> []                              # go 的最后一次出现被删掉了
        buf.insert(5, "Go")
        buf.suggest("", 5) -> ["stop", "Go"]
    """)


@case("Part 3", title="随机操作序列（对照参考实现）")
def p3_random(sol):
    random_check(sol, 3, 60)


# ================================================================ Part 4: replicas of CRDTDoc

class Replicas:
    """Replicas of the answer's CRDTDoc, driven step by step, with these checks after every step:
    - a local edit lands exactly where it was asked;
    - applying an operation a second time changes nothing;
    - two replicas that have applied the same set of operations (now, or one of them at some earlier moment) show the
      same text;
    - every visible character came from an applied insert, appears once, and disappears only through an applied
      delete; applying an operation does not reorder the other characters;
    - once a replica holds everything its operations depend on (the operations their replica had applied when they
      were made), it shows exactly the characters inserted and not deleted, and an operation applied on top of such a
      state takes effect at once (only its character appears, or only its target disappears).
    An answer may hold back an operation that arrives before what it depends on; only the checks above apply then.
    Characters are distinct, so the test knows which character a local delete removed from the answer's own text."""

    def __init__(self, sol, *names, show=40):
        cls = getattr(sol, "CRDTDoc", None)
        if not isinstance(cls, type):
            raise Mismatch("答案里没有类 CRDTDoc（名字要和题目一致）", show=False)
        self.names, self.show, self.count = names, show, 0
        sites = [n.upper() for n in names]
        self.lines = [f"{', '.join(names)} = {', '.join(f'CRDTDoc({_lit(s)})' for s in sites)}"]
        self.current = calling(self._replay())
        self.docs = {n: cls(s) for n, s in zip(names, sites)}
        self.applied = {n: set() for n in names}        # op names applied on each replica (own edits included)
        self.pending = {n: set() for n in names}        # applied ops whose dependencies are not all applied there
        self.ops = {}                                   # op name -> (operation, char inserted, insert it deletes)
        self.deps = {}                                  # op name -> ops its replica had applied when it was made
        self.char_op = {}                               # char -> name of the insert that created it
        self.history = {}                               # frozenset of op names -> (text, replica that showed it)
        self.texts = {n: "" for n in names}
        self._after()

    # -------------------------------------------------- display
    def _replay(self, extra=None, mark="这一步"):
        body = self.lines[1:] + ([extra] if extra else [])
        skipped = max(0, len(body) - self.show)
        rows = self.lines[:1] + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
        rows += body[skipped:]
        if extra:
            rows[-1] += f"   # <- {mark}"
        return "\n".join(rows)

    def _calling(self, extra):
        self.current = calling(self._replay(extra))

    def _fail(self, msg, extra, want=None, got=None, show=True):
        raise Mismatch(msg, call=self._replay(extra), want=want, got=got, show=show)

    # -------------------------------------------------- bookkeeping
    def _add(self, n, name):
        q = self.applied[n]
        q.add(name)
        self.pending[n] = {o for o in self.pending[n] | {name} if not self.deps[o] <= q}

    def _dead(self, n):
        """Characters whose delete has been applied on replica n."""
        return {self.ops[self.ops[o][2]][1] for o in self.applied[n] if self.ops[o][2] is not None}

    def _new_op(self, n, op, ch, target):
        self.count += 1
        name = f"op{self.count}"
        self.ops[name] = (op, ch, target)
        self.deps[name] = frozenset(self.applied[n])
        self._add(n, name)
        return name

    # -------------------------------------------------- checks after every step
    def _after(self, n=None, check=None):
        before = dict(self.texts)
        for m in self.names:
            self._calling(f"print({m}.text())")
            t = self.docs[m].text()
            if not isinstance(t, str):
                self._fail(f"text() 应当返回 str，却返回了 {type(t).__name__}", f"print({m}.text())", show=False)
            self.texts[m] = t
        if n is not None:
            bad = check(before[n], self.texts[n])
            if bad:
                why, want = bad
                if want is None:
                    self._fail(f"{why}：这一步之前是 {_lit(before[n])}，之后是 {_lit(self.texts[n])}",
                               f"print({n}.text())", show=False)
                self._fail(why, f"print({n}.text())", want=want, got=self.texts[n])
        for m in self.names:
            t, q, line = self.texts[m], self.applied[m], f"print({m}.text())"
            known = {self.ops[o][1] for o in q if self.ops[o][1] is not None}
            if len(set(t)) < len(t):
                c = next(c for i, c in enumerate(t) if c in t[:i])
                self._fail(f"{m} 的文档里 {_lit(c)} 出现了不止一次，但它只被插入过一次：{_lit(t)}", line, show=False)
            for c in set(t) - known:
                self._fail(f"{m} 的文档里出现了 {_lit(c)}，可是 {m} 还没有应用过插入它的操作：{_lit(t)}", line,
                           show=False)
            if n is not None and m != n and t != before[m]:
                self._fail(f"别的副本上的这一步不应改变 {m}：{m} 从 {_lit(before[m])} 变成了 {_lit(t)}", line,
                           show=False)
            if not self.pending[m]:
                dead = self._dead(m)
                want = known - dead
                missing, extra = sorted(want - set(t)), sorted(set(t) - want)
                if missing or extra:
                    parts = ([f"缺少 {_lit(''.join(missing))}"] if missing else []) + \
                            ([f"多出 {_lit(''.join(extra))}"] if extra else [])
                    self._fail(f"{m} 已经应用了它收到的操作所依赖的全部操作，这时应当恰好看到所有插入过、没有被删除的"
                               f"字符，实际 {_lit(t)}：{'，'.join(parts)}", line, show=False)
        for m in self.names:
            key = frozenset(self.applied[m])
            seen = self.history.setdefault(key, (self.texts[m], m))
            if seen[0] != self.texts[m]:
                who = seen[1]
                if who == m:
                    msg = f"{m} 应用过的操作没有变化，text() 却从 {_lit(seen[0])} 变成了 {_lit(self.texts[m])}"
                    self._fail(msg, f"print({m}.text())", show=False)
                if frozenset(self.applied[who]) == key:
                    msg = f"{who} 和 {m} 应用过同一组操作，text() 必须相同"
                    self._fail(f"{msg}：{who} 是 {_lit(self.texts[who])}，{m} 是 {_lit(self.texts[m])}",
                               f"print({who}.text(), {m}.text())", show=False)
                msg = f"{m} 现在应用过的操作，和 {who} 之前某一时刻应用过的是同一组，text() 必须相同"
                self._fail(f"{msg}：那时 {who} 是 {_lit(seen[0])}，现在 {m} 是 {_lit(self.texts[m])}",
                           f"print({m}.text())", show=False)

    # -------------------------------------------------- steps
    def insert(self, n, index, ch):
        assert ch not in self.char_op and 0 <= index <= len(self.texts[n])
        line = f"op{self.count + 1} = {n}.local_insert({index}, {_lit(ch)})"
        self._calling(line)
        op = self.docs[n].local_insert(index, ch)
        self.lines.append(line)
        name = self._new_op(n, op, ch, None)
        self.char_op[ch] = name
        why = f"本地编辑要落在要求的位置：在可见下标 {index} 插入 {_lit(ch)}"
        self._after(n, lambda b, t: None if t == b[:index] + ch + b[index:] else (why, b[:index] + ch + b[index:]))
        return name

    def delete(self, n, index):
        before = self.texts[n]
        assert 0 <= index < len(before)
        line = f"op{self.count + 1} = {n}.local_delete({index})"
        self._calling(line)
        op = self.docs[n].local_delete(index)
        self.lines.append(line)
        name = self._new_op(n, op, None, self.char_op[before[index]])
        why = f"本地编辑要删掉可见下标 {index} 处的 {_lit(before[index])}"
        self._after(n, lambda b, t: None if t == b[:index] + b[index + 1:] else (why, b[:index] + b[index + 1:]))
        return name

    def apply(self, n, name):
        line = f"{n}.apply({name})"
        self._calling(line)
        op, ch, target = self.ops[name]
        q = self.applied[n]
        repeat = name in q
        at_once = not self.pending[n] and self.deps[name] <= q    # everything it depends on is already here
        self.docs[n].apply(op)
        self.lines.append(line)
        self._add(n, name)
        dead = self._dead(n)

        def check(b, t):
            if repeat:
                return None if t == b else (f"apply 是幂等的：{name} 已经在 {n} 上应用过，再应用一次不应有任何变化", b)
            if at_once and ch is not None:
                if len(t) == len(b) + 1 and ch in t and t.replace(ch, "") == b:
                    return None
                return (f"{n} 已经有了这个插入所依赖的全部操作，应用它应当只多出 {_lit(ch)} 这一个字符，其余字符的先后"
                        "不变", None)
            if at_once:
                gone = self.ops[target][1]
                if gone not in b:
                    return None if t == b else (f"{_lit(gone)} 已经被删掉了，这条删除不应再改变文档", b)
                why = f"{n} 已经有了这个删除所依赖的全部操作，应用它应当只去掉 {_lit(gone)} 这一个字符"
                return None if t == b.replace(gone, "") else (why, b.replace(gone, ""))
            lost = [c for c in b if c not in t and c not in dead]
            if lost:
                return f"{_lit(lost[0])} 不见了，可是 {n} 没有应用过删除它的操作", None
            if "".join(c for c in t if c in b) != "".join(c for c in b if c in t):
                return "应用一个操作不应改变其余字符的先后顺序", None
            return None

        self._after(n, check)

    def text(self, n, want, why=""):
        """An explicit check where the statement fixes the whole text."""
        if self.texts[n] != want:
            self._fail(why or "这里的文档由前面的编辑完全确定", f"print({n}.text())", want=want, got=self.texts[n])


def deliver_all(R, ops, to):
    for n in to:
        for name in ops:
            R.apply(n, name)


def type_text(R, n, text, at=None):
    """Local inserts of text's characters, one by one, starting at visible index `at` (default: the end)."""
    start = len(R.texts[n]) if at is None else at
    return [R.insert(n, start + i, ch) for i, ch in enumerate(text)]


def _play(sol, steps, k, holder):
    names = "abcd"[:k]
    R = holder["R"] = Replicas(sol, *names, show=SHOW)
    queues = {(i, j): [] for i in names for j in names if i != j}        # per-channel FIFO, as the statement says
    chars = iter(ref.POOL)
    for st in steps:
        i = names[st[1]]
        n = len(R.texts[i])
        if st[0] == "ins":
            name = R.insert(i, int(st[2] * (n + 1)), next(chars))
        elif st[0] == "del":
            if n == 0:
                continue
            name = R.delete(i, int(st[2] * n))
        elif st[0] == "deliver":
            q = queues[(i, names[st[2]])]
            for name in q[: st[3]]:
                R.apply(names[st[2]], name)
            del q[: st[3]]
            continue
        else:                                                           # repeat an operation already delivered
            done = sorted(R.applied[i], key=lambda m: int(m[2:]))
            if done:
                R.apply(i, done[int(st[2] * len(done))])
            continue
        for j in names:
            if j != i:
                queues[(i, j)].append(name)
    for (i, j), q in queues.items():
        for name in q:
            R.apply(j, name)


def _fails(sol, steps, k):
    holder = {}
    try:
        _play(sol, steps, k, holder)
    except Mismatch as e:
        return e
    except Exception as e:
        e.judge_replay = holder["R"].current if "R" in holder else None
        return e
    return None


def random_crdt(sol, scenarios):
    r = rng("crdt")
    for _ in range(scenarios):
        k = r.choice([2, 3, 3, 4])
        steps = ref.random_scenario(r, k, r.randint(8, 36))
        err = _fails(sol, steps, k)
        if err is None:
            continue
        stop = time.process_time() + 0.5
        size = max(1, len(steps) // 2)
        while time.process_time() < stop:                               # drop steps the failure does not need
            i, progress = 0, False
            while i < len(steps) and time.process_time() < stop:
                cand = steps[:i] + steps[i + size:]
                e = _fails(sol, cand, k)
                if e is not None and type(e) is type(err):
                    steps, err, progress = cand, e, True
                else:
                    i += size
            if size > 1:
                size //= 2
            elif not progress:
                break
        if isinstance(err, Mismatch):
            err.msg = "随机场景（已自动删去与出错无关的步骤）：" + err.msg
            err.args = (err.msg,)
        elif getattr(err, "judge_replay", None):
            calling(err.judge_replay)
        raise err


# ---------------------------------------------------------------- Part 4

@case("Part 4", sample=True, title="题目例子 H / i")
def p4_example(sol):
    R = Replicas(sol, "a", "b")
    op1 = R.insert("a", 0, "H")                 # 副本 A 本地是 "H"
    op2 = R.insert("b", 0, "i")                 # 副本 B 本地是 "i"
    R.apply("b", op1)
    R.apply("a", op2)                           # 两边都应用过 op1、op2，text() 必须相同（每一步之后都检查）


@case("Part 4", title="单个副本的本地编辑")
def p4_local(sol):
    R = Replicas(sol, "a")
    ops = type_text(R, "a", "World")
    R.insert("a", 0, ">")
    R.insert("a", 3, "_")
    d1 = R.delete("a", 0)
    R.delete("a", 5)
    R.text("a", "Wo_rl")
    R.apply("a", ops[0])                        # 自己的操作再 apply 一次：没有变化
    R.apply("a", d1)
    R.insert("a", 5, "!")
    R.insert("a", 0, "<")
    R.text("a", "<Wo_rl!")


@case("Part 4", title="顺序送达：两个副本保持一致")
def p4_in_order(sol):
    R = Replicas(sol, "a", "b")
    ops = type_text(R, "a", "abc")
    deliver_all(R, ops, "b")
    R.text("b", "abc")
    ops = [R.insert("b", 1, "X"), R.insert("b", 2, "Y"), R.delete("b", 4)]
    deliver_all(R, ops, "a")
    R.text("a", "aXYb")


@case("Part 4", title="并发插入同一个位置")
def p4_same_spot(sol):
    R = Replicas(sol, "a", "b")
    left = type_text(R, "a", "abc")             # 两个副本同时从空文档开始打字
    right = type_text(R, "b", "xyz")
    deliver_all(R, right, "a")
    deliver_all(R, left, "b")
    R = Replicas(sol, "a", "b", "c")
    deliver_all(R, type_text(R, "a", "mn"), "bc")
    o1, o2, o3 = R.insert("a", 1, "1"), R.insert("b", 1, "2"), R.insert("c", 1, "3")   # 三个副本同时插在 m、n 之间
    R.apply("a", o3)
    R.apply("b", o1)
    R.apply("c", o2)
    R.apply("a", o2)
    R.apply("b", o3)
    R.apply("c", o1)


@case("Part 4", title="并发的插入和删除")
def p4_insert_delete(sol):
    R = Replicas(sol, "a", "b")
    deliver_all(R, type_text(R, "a", "abc"), "b")
    d = R.delete("a", 1)                        # a 删掉 "b"
    y = R.insert("b", 2, "Y")                   # 同时 b 在自己看到的 "b" 后面插入 "Y"
    R.apply("b", d)
    R.apply("a", y)
    R.text("a", "aYc")
    R = Replicas(sol, "a", "b")
    deliver_all(R, type_text(R, "a", "pqr"), "b")
    d = R.delete("a", 1)                        # a 删掉 "q"
    z = R.insert("b", 0, "Z")                   # 同时 b 在最前面插入 "Z"，"q" 在 b 上的下标变了
    R.apply("b", d)
    R.apply("a", z)
    R.text("b", "Zpr")


@case("Part 4", title="删除或插入比它依赖的字符先到")
def p4_delete_first(sol):
    R = Replicas(sol, "a", "b", "c")
    ins = R.insert("a", 0, "z")
    R.apply("b", ins)
    d = R.delete("b", 0)
    R.apply("c", d)                             # c 先收到删除
    R.apply("c", ins)                           # 再收到插入："z" 不应出现
    R.apply("a", d)
    R = Replicas(sol, "a", "b", "c")
    ops = type_text(R, "a", "uvw")
    deliver_all(R, ops, "b")
    d = R.delete("b", 1)                        # b 删掉 "v"
    R.apply("c", d)                             # b 到 c 的连接比 a 到 c 的快
    R.apply("a", d)                             # a 上是 "uw"
    deliver_all(R, ops, "c")
    R.text("c", "uw")                           # c 和 a 应用过同一组操作
    R = Replicas(sol, "a", "b", "c")
    x = R.insert("a", 0, "x")
    R.apply("b", x)
    y = R.insert("b", 1, "y")                   # b 在 "x" 后面插入 "y"
    R.apply("c", y)                             # c 先收到 "y"，这时还没见过 "x"
    R.apply("c", x)                             # "x" 到了以后，c 和 b 应用过同一组操作
    R.text("c", "xy")
    R.apply("a", y)


@case("Part 4", title="重复送达：apply 是幂等的")
def p4_duplicates(sol):
    R = Replicas(sol, "a", "b")
    o1, o2 = type_text(R, "a", "ab")
    R.apply("a", o1)                            # 自己的操作
    R.apply("b", o1)
    R.apply("b", o1)
    R.apply("b", o2)
    R.apply("b", o1)
    d = R.delete("b", 0)                        # b 删掉 "a"
    R.apply("b", o1)                            # 已删掉的字符的插入再送到一次：不能复活
    R.apply("b", d)
    R.apply("a", d)
    R.apply("a", o1)                            # a 上也一样
    R.apply("a", d)
    R.text("a", "b")
    o3 = R.insert("a", 0, "c")
    R.apply("b", o3)
    R.apply("b", o3)
    R.text("b", "cb")


@case("Part 4", title="同一空隙里并发插入后，再插到两者中间")
def p4_between_twins(sol):
    R = Replicas(sol, "a", "b", "c")
    x, y = R.insert("a", 0, "x"), R.insert("b", 0, "y")    # 两个副本填同一个空隙
    deliver_all(R, [x, y], "abc")
    later = [R.insert(n, 1, ch) for n, ch in zip("abc", "123")]   # 每个副本都插到这两个字符中间
    deliver_all(R, later, "abc")


@case("Part 4", title="连续输入 200 个字符")
def p4_long_typing(sol):
    R = Replicas(sol, "a", "b")
    chars = "".join(ref.POOL[:200])
    ops = type_text(R, "a", chars[:100])                   # 每次接在末尾
    ops += [R.insert("a", 1, ch) for ch in chars[100:]]    # 每次插在第一个字符后面
    R.text("a", chars[0] + chars[100:][::-1] + chars[1:100])
    deliver_all(R, ops, "b")


@case("Part 4", title="随机场景：乱序、延迟、重复送达")
def p4_random(sol):
    random_crdt(sol, 150)
