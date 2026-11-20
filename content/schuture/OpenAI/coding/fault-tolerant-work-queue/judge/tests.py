"""Tests for the fault-tolerant work queue. Every call is checked as soon as it returns. A failure shows the calls so
far as code that can be pasted into 自定义运行: tokens appear as r1.token (the Reservation an earlier reserve returned),
a forged token as object(), time as clock.advance(...) on the statement's ManualClock."""
import ast
import inspect

from judgelib import Mismatch, calling, case, expect, expect_true, rng
import ref
from ref import Tok

SHOW = 16                    # calls shown before the failing one, when there are more than WHOLE
WHOLE = 40                   # a replay up to this long is shown whole
SHRINK_BIG = 3000            # a long sequence that goes wrong within this many steps is shrunk too
BIG = 10 ** 9                # lease_duration / max_attempts in earlier parts' tests: never reached
METHODS = ("submit", "reserve", "complete", "fail")
P3_METHODS = METHODS + ("dead_letters", "requeue_dead")
STATEMENT_ERRORS = ("UnknownTaskError", "InvalidReservationError")


def _lit(x):
    if isinstance(x, str) and x.isprintable() and '"' not in x and "\\" not in x:
        return f'"{x}"'
    return repr(x)


def _arg(a):
    if isinstance(a, Tok):
        return f"{a.name}.token" if a.name else "object()"
    return _lit(a)


def _code(method, args, name, last=False):
    if method == "advance":
        return f"clock.advance({args[0]})"
    text = f"q.{method}({', '.join(_arg(a) for a in args)})"
    if last and method in ("reserve", "dead_letters"):
        return f"print({text})"
    return f"{name} = {text}" if name else text


def _note(method, want, why, now):
    if method == "advance":
        base = f"now = {now}"
    elif want[0] == "err":
        base = f"抛出 {want[1]}"
    elif method == "reserve":
        base = "-> None" if want[1] is None else f"-> {want[1][0]}"
    elif method == "dead_letters":
        base = f"-> {want[1]!r}"
    else:
        base = ""
    text = "；".join(x for x in (base, why) if x)
    return f"# {text}" if text else ""


NAMES = ("clock", "lease_duration", "max_attempts")


def _binds(sig, *args, **kwargs):
    try:
        sig.bind(*args, **kwargs)
        return True
    except TypeError:
        return False


def _form(cls):
    """How the WorkQueue constructor takes the statement's arguments: (3, 2 or 0 of them, i.e. Part 3's, Part 2's or
    Part 1's form; whether to pass them by name). By position, as in the statement, so other parameter names work
    too; by name only when position cannot work (keyword-only parameters) or the parameters have the statement's
    names in another order."""
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return None, False
    positional = [p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    for n in (3, 2, 0):
        by_pos, by_name = _binds(sig, *range(n)), _binds(sig, **dict(zip(NAMES, range(n))))
        reordered = sorted(positional[:n]) == sorted(NAMES[:n]) and positional[:n] != list(NAMES[:n])
        if by_name and (not by_pos or reordered):
            return n, True
        if by_pos:
            return n, False
    return None, False


class _Invalid(Exception):
    """Raised while shrinking: the candidate uses the token of a reservation that no longer happens."""


class Session:
    """One WorkQueue under test. Each call is checked against `want` (from a script) or against ref.QueueModel."""

    def __init__(self, sol, part, lease=None, cap=None, label="", trace=True, fast=False):
        self.part, self.label, self.trace = part, label, trace
        cls = getattr(sol, "WorkQueue", None)
        expect_true(isinstance(cls, type), "answer.py 里没有 WorkQueue 类（类名要和题目一致）")
        self.errors = {"ValueError": ValueError}
        for name in STATEMENT_ERRORS:
            err = getattr(sol, name, None)
            expect_true(isinstance(err, type) and issubclass(err, Exception),
                        f"answer.py 里没有 {name} 异常类（名字要和题目一致）")
            self.errors[name] = err
        self.clock = ref.ManualClock()
        form, by_name = _form(cls)
        if part == 1 and form in (2, 3):
            args, shown = (self.clock, BIG, BIG)[:form], ["clock", "10**9", "10**9"][:form]
            what = "时钟停在 0，租约不会过期" + ("" if form == 2 else "，也到不了重试上限")
            note = f"# 你的 WorkQueue 已是 Part {form} 的形式：{what}"
        elif part == 1:
            args, shown, note = (), [], ""
        elif part == 2 and form != 3:
            args, shown, note = (self.clock, lease), ["clock", str(lease)], f"# lease_duration = {lease}"
        elif part == 2:
            args, shown = (self.clock, lease, BIG), ["clock", str(lease), "10**9"]
            note = f"# lease_duration = {lease}；你的 WorkQueue 已是 Part 3 的形式，max_attempts = 10**9 到不了"
        else:
            args, shown = (self.clock, lease, cap), ["clock", str(lease), str(cap)]
            note = f"# lease_duration = {lease}, max_attempts = {cap}"
        if by_name:
            shown = [f"{k}={v}" for k, v in zip(NAMES, shown)]
        ctor = f"q = WorkQueue({', '.join(shown)})"
        self.head = (["clock = ManualClock()"] if args else []) + [f"{ctor}    {note}" if note else ctor]
        if part >= 2 and form not in (None, 3) and form < len(args):
            expect_true(False, f"WorkQueue 的构造函数还不接受 Part {part} 的参数"
                               f"（{'clock, lease_duration' if part == 2 else 'clock, lease_duration, max_attempts'}）",
                        call="\n".join(self.head))
        if trace:
            calling("\n".join(self.head))
        self.q = cls(**dict(zip(NAMES, args))) if by_name else cls(*args)
        for name in (P3_METHODS if part == 3 else METHODS):
            expect_true(callable(getattr(self.q, name, None)), f"WorkQueue 没有 {name} 方法（方法名要和题目一致）",
                        call="\n".join(self.head))
        self.model = ref.QueueModel(self.clock, lease if part >= 2 else None, cap if part >= 3 else None, fast)
        self.hist = []           # (method, args, name, want, why, clock reading after the call)
        self.tokens = {}         # reservation name -> the answer's token
        self.mtok = {}           # reservation name -> the model's token
        self.payloads = {}       # task_id -> payload of its accepted submit (scripts)

    # ------------------------------------------------------------ display

    def replay(self, mark="# <- 这一步"):
        """The calls so far; the last one is the current call."""
        hist = self.hist if len(self.hist) <= WHOLE else self.hist[-(SHOW + 1):]
        skipped = len(self.hist) - len(hist)
        lines = [_code(m, a, n, last=(k == len(hist) - 1)) for k, (m, a, n, w, y, t) in enumerate(hist)]
        notes = [_note(m, w, y, t) for m, a, n, w, y, t in hist[:-1]] + [mark]
        width = max(len(s) for s in lines) + 3
        head = list(self.head)
        if skipped:
            head.append(f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）")
        return "\n".join(head + [(s.ljust(width) + n).rstrip() for s, n in zip(lines, notes)])

    def wrong(self, msg, why="", got=None, want=None, show=True):
        msg = self.label + msg + (f"。{why}" if why else "")
        call = self.replay() if self.trace else None
        if want is not None:
            expect(got, want, call=call, msg=msg)
        raise Mismatch(msg, call, None, got, show=show and got is not None)

    # ------------------------------------------------------------ one call

    def _model_want(self, method, args):
        if method in ("complete", "fail"):
            tok = args[1]
            if tok.name is not None and tok.name not in self.mtok:
                raise _Invalid
            return getattr(self.model, method)(args[0], self.mtok[tok.name] if tok.name else object())
        return getattr(self.model, method)(*args)

    def step(self, method, args, name=None, want=None, why=""):
        """Makes one call (want None: the model says what is right) and checks it."""
        if method == "advance":
            self.clock.advance(args[0])
            self.hist.append((method, args, name, None, why, self.clock.now()))
            return
        if want is None:
            want = self._model_want(method, args)
        self.hist.append((method, args, name, want, why, None))
        if self.trace:
            calling(self.replay())                    # shown if the answer raises or hangs in this call
        for a in args:
            if isinstance(a, Tok) and a.name is not None and a.name not in self.tokens:
                raise _Invalid
        real = tuple((self.tokens[a.name] if a.name else object()) if isinstance(a, Tok) else a for a in args)
        try:
            got = getattr(self.q, method)(*real)
        except Exception as e:
            if not self._raised(method, e, want, why):
                if self.trace:
                    calling(self.replay(mark="# <- 这一步抛出异常"))
                raise
            return
        self._returned(method, args, name, got, want, why)

    def _raised(self, method, e, want, why):
        """True if raising e here is right; a wrong statement exception fails the case; False re-raises e (any other
        exception is a crash, shown with its traceback)."""
        if want[0] == "err" and isinstance(e, self.errors[want[1]]):
            return True
        mine = [self.errors[n] for n in STATEMENT_ERRORS]
        if method in ("submit", "requeue_dead"):
            mine.append(ValueError)                    # elsewhere a ValueError is just a crash
        if not isinstance(e, tuple(mine)):
            return False
        got = f"{type(e).__name__}: {e}"
        if want[0] == "err":
            self.wrong(f"这一步应抛出 {want[1]}，实际抛出 {got}", why)
        if method == "reserve":
            should = "返回 None" if want[1] is None else f"交出 {want[1][0]}"
            self.wrong(f"reserve 应{should}，却抛出了 {got}", why)
        if method == "dead_letters":
            self.wrong(f"dead_letters() 应返回 {want[1]!r}，却抛出了 {got}", why)
        self.wrong(f"这一步应当成功，却抛出了 {got}", why)

    def _returned(self, method, args, name, got, want, why):
        if want[0] == "err":
            self.wrong(f"这一步应抛出 {want[1]}，但正常返回了", why)
        if method == "submit" and want == ref.OK:
            self.payloads.setdefault(args[0], args[1])
        if method == "reserve":
            self._check_reservation(name, got, want[1], why)
        elif method == "dead_letters":
            if not isinstance(got, (list, tuple)):
                self.wrong("dead_letters() 应返回 task_id 的列表", why, got=got)
            if list(got) != want[1]:
                self.wrong("dead_letters() 应按进入 dead 的先后，返回当前处于 dead 的全部任务", why,
                          got=list(got), want=want[1])

    def _check_reservation(self, name, got, want, why):
        if want is None:
            if got is not None:
                tid = getattr(got, "task_id", got)
                self.wrong(f"此时没有处于 ready 的任务，reserve 应返回 None，实际交出了 {tid!r}", why)
            return
        task_id, payload, mtoken = want
        if got is None:
            self.wrong(f"reserve 应交出 {task_id}，实际返回了 None", why)
        if not all(hasattr(got, f) for f in ("task_id", "token", "payload")):
            self.wrong("reserve 应返回带 task_id、token、payload 字段的 Reservation", why, got=got)
        if got.task_id != task_id:
            self.wrong(f"reserve 应交出 {task_id}（处于 ready 的任务里最早进入 ready 的那个）", why,
                      got=got.task_id, want=task_id)
        try:
            same = got.payload == payload
        except Exception:
            same = False
        if not same:
            self.wrong(f"reserve 交出了 {task_id}，但 payload 不对：应原样交回 submit 时给的值", why,
                      got=got.payload, want=payload)
        if name:
            self.tokens[name], self.mtok[name] = got.token, mtoken


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines like  r3 = reserve("w3") -> "t2"   # why  ,  complete("t2", r2) !! InvalidReservationError  , or
    advance(7). A call with neither '->' nor '!!' must succeed. A bare name argument is a token: r2 means r2.token,
    FORGED a token no reservation issued."""
    steps = []
    for line in script.strip().splitlines():
        code, _, why = line.partition("#")          # the scripts have no '#' inside a string
        code, bang, exc = code.partition("!!")
        call, arrow, expr = code.partition("->")
        node = ast.parse(call.strip()).body[0]
        name = None
        if isinstance(node, ast.Assign):
            name, node = node.targets[0].id, node.value
        else:
            node = node.value
        args = tuple((Tok(None) if a.id == "FORGED" else Tok(a.id)) if isinstance(a, ast.Name)
                     else ast.literal_eval(a) for a in node.args)
        if bang:
            want = ("err", exc.strip())
        elif arrow:
            want = ("val", ast.literal_eval(expr.strip()))
        else:
            want = ref.OK
        steps.append((node.func.id, args, name, want, why.strip()))
    return steps


def run(sol, part, script, lease=None, cap=None):
    s = Session(sol, part, lease, cap)
    for method, args, name, want, why in _parse(script):
        if method == "reserve" and want[0] == "val" and want[1] is not None:
            want = ("val", (want[1], s.payloads.get(want[1]), None))    # the payload its submit gave
        s.step(method, args, name, None if method == "advance" else want, why)
    return s


# ---------------------------------------------------------------- random sequences against ref

def _first_bad(sol, setup, ops):
    """Replays ops quietly; returns (index, kind) of the first wrong call (kind "wa" or the exception type), None if
    every call is right, "skip" if ops is not a valid sequence."""
    s = Session(sol, *setup, trace=False)
    for k, (method, args, name) in enumerate(ops):
        try:
            s.step(method, args, name)
        except _Invalid:
            return "skip"
        except Mismatch:
            return k, "wa"
        except Exception as e:
            return k, type(e)
    return None


def _without(ops, i, size):
    """ops minus ops[i:i+size], and minus every later call that uses the token of a removed reservation."""
    gone = {n for _, _, n in ops[i:i + size] if n}
    return [op for op in ops[:i] + ops[i + size:]
            if not any(isinstance(a, Tok) and a.name in gone for a in op[1])]


def _shrink(sol, setup, ops, bad, budget=1.0):
    """Drops calls the failure does not need, so the replay shown is short."""
    import time
    ops = ops[: bad[0] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = _without(ops, i, size)
            out = _first_bad(sol, setup, cand) if cand else None
            if out not in (None, "skip") and out[1] == bad[1]:
                ops, bad, progress = cand[: out[0] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops


def _drive(sol, setup, ops, label, trace=True):
    s = Session(sol, *setup, label=label, trace=trace)
    for method, args, name in ops:
        s.step(method, args, name)


def random_check(sol, part, sequences, longest):
    r = rng()
    for k in range(sequences):
        lease, cap = r.choice([1, 2, 3, 5]), r.choice([1, 1, 2, 3, 5])
        setup = (part, lease, cap)
        ops = ref.gen_ops(r, r.randint(1, longest), part, lease, cap)
        calling(f"随机生成的第 {k + 1} 组调用（{len(ops)} 步）")
        bad = _first_bad(sol, setup, ops)
        if bad is None:
            continue
        _drive(sol, setup, _shrink(sol, setup, ops, bad), "随机调用序列（已自动删去与出错无关的调用）：")
        _drive(sol, setup, ops, "随机调用序列：")         # only if the shrunk replay somehow passes
        expect_true(False, "随机调用序列出错")


def big_check(sol, part, ops, wants, lease=None, cap=None):
    """Runs a long sequence; wants holds the model's outcomes, recorded while the sequence was generated."""
    calls = sum(1 for op in ops if op[0] != "advance")
    setting = "" if part == 1 else f"，lease_duration = {lease}" + ("" if part == 2 else f", max_attempts = {cap}")
    calling(f"随机生成的 {len(ops)} 步操作（其中 {calls} 次 WorkQueue 调用）{setting}")
    s = Session(sol, part, lease, cap, label=f"{len(ops)} 步随机操作中：", trace=False, fast=True)
    setup = (part, lease, cap)
    for k, ((method, args, name), want) in enumerate(zip(ops, wants)):
        try:
            s.step(method, args, name, want)
        except Exception as e:
            if k < SHRINK_BIG:                    # early enough to shrink into a replay that can be pasted
                bad = (k, "wa" if isinstance(e, Mismatch) else type(e))
                if _first_bad(sol, setup, ops[:k + 1]) == bad:
                    _drive(sol, setup, _shrink(sol, setup, ops, bad),
                           f"{len(ops)} 步随机操作中出错（已自动删去与出错无关的调用）：")
            if isinstance(e, Mismatch):
                e.call = s.replay()
            else:
                calling(s.replay(mark="# <- 这一步抛出异常"))
            raise


# ================================================================ Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    run(sol, 1, """
        submit("t1", "build")
        submit("t2", "test")
        r1 = reserve("w1")            -> "t1"
        r2 = reserve("w2")            -> "t2"
        complete("t1", r1)            # 't1' 变为 completed
        fail("t2", r2)                # 't2' 回到 ready，排到队尾
        r3 = reserve("w3")            -> "t2"    # 又是这个任务，但 token 是新的
        complete("t2", r2)            !! InvalidReservationError    # r2.token 已经失效
        complete("t2", r3)
    """)


@case("Part 1", title="先进先出：失败的任务排到队尾")
def p1_fifo(sol):
    run(sol, 1, """
        submit("a", 1)
        submit("b", 2)
        submit("c", 3)
        ra = reserve("w1")            -> "a"
        rb = reserve("w2")            -> "b"
        fail("a", ra)                 # a 回到 ready，排在 c 后面
        submit("d", 4)
        fail("b", rb)
        r1 = reserve("w1")            -> "c"     # 放回的任务和新提交的任务共用一个先进先出顺序
        r2 = reserve("w1")            -> "a"
        r3 = reserve("w1")            -> "d"
        r4 = reserve("w1")            -> "b"
        reserve("w1")                 -> None
    """)


@case("Part 1", title="没有 ready 的任务时返回 None")
def p1_nothing_ready(sol):
    run(sol, 1, """
        reserve("w1")                 -> None
        submit("a", "x")
        r1 = reserve("w1")            -> "a"
        reserve("w2")                 -> None    # a 正被预约
        complete("a", r1)
        reserve("w2")                 -> None    # a 已完成，不会再回到 ready
    """)


@case("Part 1", title="UnknownTaskError 与重复提交")
def p1_errors(sol):
    run(sol, 1, """
        complete("ghost", FORGED)     !! UnknownTaskError
        fail("ghost", FORGED)         !! UnknownTaskError
        submit("a", "first")
        submit("a", "second")         !! ValueError    # task_id 已经提交过
        r1 = reserve("w1")            -> "a"     # 被拒绝的 submit 没有改动 a：payload 仍是 "first"
        complete("b", r1)             !! UnknownTaskError    # b 从未提交过，不论 token 是什么
        submit("a", "third")          !! ValueError    # a 正被预约，task_id 也不能再用
        reserve("w2")                 -> None    # 被拒绝的 submit 没有放进新任务
        complete("a", r1)
        submit("a", "fourth")         !! ValueError    # a 已完成，task_id 仍然不能再用
        complete("a", r1)             !! InvalidReservationError    # a 提交过：重复 complete 是 InvalidReservationError
        fail("a", r1)                 !! InvalidReservationError
        reserve("w2")                 -> None
    """)


@case("Part 1", title="失效的 token、伪造的 token")
def p1_stale_tokens(sol):
    run(sol, 1, """
        submit("a", "x")
        r1 = reserve("w1")            -> "a"
        complete("a", FORGED)         !! InvalidReservationError    # 不是这次预约签发的 token
        fail("a", r1)
        complete("a", r1)             !! InvalidReservationError    # a 已回到 ready，r1 这次预约已经结束
        fail("a", r1)                 !! InvalidReservationError
        r2 = reserve("w2")            -> "a"
        fail("a", r1)                 !! InvalidReservationError    # a 又被预约了，但不是 r1 这一次
        complete("a", r1)             !! InvalidReservationError
        fail("a", r2)
        r3 = reserve("w3")            -> "a"
        complete("a", r2)             !! InvalidReservationError
        complete("a", r3)
        complete("a", r3)             !! InvalidReservationError    # 已经完成
        fail("a", r3)                 !! InvalidReservationError
    """)


@case("Part 1", title="别的任务的 token 不能用")
def p1_token_other_task(sol):
    run(sol, 1, """
        submit("a", 1)
        submit("b", 2)
        ra = reserve("w1")            -> "a"
        rb = reserve("w2")            -> "b"
        complete("b", ra)             !! InvalidReservationError    # ra.token 只对应“这一次预约 a”，不是 b 的
        fail("a", rb)                 !! InvalidReservationError
        fail("b", rb)
        rb2 = reserve("w2")           -> "b"
        complete("a", rb2)            !! InvalidReservationError
        complete("b", ra)             !! InvalidReservationError
        complete("a", ra)
        complete("b", rb2)
    """)


@case("Part 1", title="payload 原样交回")
def p1_payloads(sol):
    run(sol, 1, """
        submit("a", [1, 2])
        submit("b", {"k": "v"})
        submit("c", None)
        submit("d", [1, 2])
        submit("e", 0)
        r1 = reserve("w")             -> "a"
        r2 = reserve("w")             -> "b"
        r3 = reserve("w")             -> "c"     # payload 是 None，仍然要交出 Reservation
        r4 = reserve("w")             -> "d"
        r5 = reserve("w")             -> "e"
        reserve("w")                  -> None
    """)


@case("Part 1", title="随机操作序列")
def p1_random(sol):
    random_check(sol, 1, 300, 30)


@case("Part 1", cpu=4, title="5 万步随机操作")
def p1_big(sol):
    wants = []
    big_check(sol, 1, ref.gen_ops(rng(), 50_000, 1, fast=True, wants=wants), wants)


# ================================================================ Part 2

@case("Part 2", sample=True, title="题目例子（lease_duration = 5）")
def p2_example(sol):
    run(sol, 2, """
        submit("t1", "x")
        rA = reserve("A")             -> "t1"    # deadline = 0 + 5 = 5，在 t=5 之前（含）都有效
        advance(7)                    # 还没有任何调用：过期要到下一次调用才会被发现
        rB = reserve("B")             -> "t1"    # 先回收 t1（7 > 5）：变回 ready，随即被 B 重新预约
        complete("t1", rA)            !! InvalidReservationError
        complete("t1", rB)
    """, lease=5)


@case("Part 2", title="截止时刻当刻仍有效")
def p2_lease_boundary(sol):
    run(sol, 2, """
        submit("x", 0)
        submit("y", 0)
        rx = reserve("w")             -> "x"
        ry = reserve("w")             -> "y"
        advance(3)
        reserve("w")                  -> None    # now = 3，正是截止时刻：x、y 的预约都还有效，不回收
        complete("x", rx)             # 截止时刻当刻仍然有效
        advance(1)
        complete("y", ry)             !! InvalidReservationError    # now = 4 > 3：这次调用先回收 y
        r3 = reserve("w")             -> "y"
    """, lease=3)


@case("Part 2", title="过期的 token 立即失效")
def p2_expired_token(sol):
    run(sol, 2, """
        submit("a", 1)
        ra = reserve("w1")            -> "a"
        advance(6)
        complete("a", ra)             !! InvalidReservationError    # complete 也先回收：截止时刻 5 < now = 6
        r2 = reserve("w2")            -> "a"
        advance(6)
        fail("a", r2)                 !! InvalidReservationError    # fail 同样先回收
        r3 = reserve("w3")            -> "a"
        fail("a", r3)
        r4 = reserve("w3")            -> "a"
    """, lease=5)


@case("Part 2", title="submit 也先回收")
def p2_submit_reclaims_first(sol):
    run(sol, 2, """
        submit("a", 1)
        ra = reserve("w")             -> "a"
        advance(3)
        submit("b", 2)                # 先回收 a（3 > 2），a 排在 b 前面
        r1 = reserve("w")             -> "a"
        r2 = reserve("w")             -> "b"
    """, lease=2)


@case("Part 2", title="fail 先回收，再放回")
def p2_fail_reclaims_first(sol):
    run(sol, 2, """
        submit("a", 1)
        submit("b", 2)
        ra = reserve("w")             -> "a"     # a 的截止时刻是 5
        advance(3)
        rb = reserve("w")             -> "b"     # b 的截止时刻是 8
        advance(3)
        fail("b", rb)                 # now = 6：先回收 a，再把 b 放回，a 排在前面
        r1 = reserve("w")             -> "a"
        r2 = reserve("w")             -> "b"
    """, lease=5)


@case("Part 2", title="一轮回收：按截止时刻，再按 task_id")
def p2_reclaim_order(sol):
    run(sol, 2, """
        submit("c", 0)
        rc = reserve("w")             -> "c"     # c 的截止时刻是 4
        advance(1)
        submit("b", 0)
        submit("a", 0)
        rb = reserve("w")             -> "b"     # b、a 的截止时刻都是 5
        ra = reserve("w")             -> "a"
        advance(5)
        submit("d", 0)                # now = 6：先回收 c（截止 4），再按 task_id 回收 a、b（截止 5），最后放入 d
        r1 = reserve("w")             -> "c"
        r2 = reserve("w")             -> "a"
        r3 = reserve("w")             -> "b"
        r4 = reserve("w")             -> "d"
    """, lease=4)
    run(sol, 2, """
        submit("p", 0)
        submit("q", 0)
        rp = reserve("w")             -> "p"
        fail("p", rp)                 # p 排到 q 后面
        rq = reserve("w")             -> "q"     # q 的截止时刻是 4
        advance(1)
        rp2 = reserve("w")            -> "p"     # p 的截止时刻是 5
        advance(9)
        submit("z", 0)                # now = 10：按截止时刻先回收 q，再回收 p
        r1 = reserve("w")             -> "q"
        r2 = reserve("w")             -> "p"
        r3 = reserve("w")             -> "z"
    """, lease=4)


@case("Part 2", title="已结束的预约不会再被回收")
def p2_ended_lease(sol):
    run(sol, 2, """
        submit("a", 1)
        r1 = reserve("w")             -> "a"     # 截止时刻 5
        advance(1)
        fail("a", r1)
        r2 = reserve("w")             -> "a"     # 新的一次预约，截止时刻 6
        advance(5)
        complete("a", r2)             # now = 6：r2 仍有效；r1 那次预约早已结束，它的截止时刻 5 与 r2 无关
        submit("b", 2)
        rb = reserve("w")             -> "b"
        complete("b", rb)
        advance(10)
        reserve("w")                  -> None    # a、b 都已完成，它们预约过的截止时刻早已过去，也不会回到 ready
    """, lease=5)


@case("Part 2", title="随机操作序列")
def p2_random(sol):
    random_check(sol, 2, 300, 30)


@case("Part 2", cpu=4, title="5 万步随机操作")
def p2_big(sol):
    wants = []
    big_check(sol, 2, ref.gen_ops(rng(), 50_000, 2, 5, fast=True, wants=wants), wants, lease=5)


@case("Part 2", cpu=5, title="上万个租约同时有效，又不断到期")
def p2_many_leases(sol):
    # Thousands of leases are live and one or two expire before almost every call: finding the expired ones by
    # looking at every reserved task is too slow, even if the look is skipped while the clock has not moved.
    wants = []
    big_check(sol, 2, ref.lease_wave_ops(rng(), 20_000, 20_000, wants), wants, lease=20_000)


# ================================================================ Part 3

@case("Part 3", sample=True, title="题目例子（lease_duration = 3、max_attempts = 2）")
def p3_example(sol):
    run(sol, 3, """
        submit("t1", "x")
        rA = reserve("A")             -> "t1"    # 第 1 次预约
        fail("t1", rA)                # 目前 1 < 2 次：回到 ready
        rB = reserve("B")             -> "t1"    # 第 2 次预约
        fail("t1", rB)                # 目前 2 >= 2 次：进入 dead
        dead_letters()                -> ["t1"]
        reserve("C")                  -> None    # 没有 ready 的任务
        requeue_dead("t1")            # 尝试次数清零
        rC = reserve("C")             -> "t1"    # 又是第 1 次预约
    """, lease=3, cap=2)


@case("Part 3", title="max_attempts 数的是预约次数")
def p3_attempt_cap(sol):
    run(sol, 3, """
        submit("a", 1)
        ra = reserve("w")             -> "a"
        fail("a", ra)                 # max_attempts = 1：第一次失败就进入 dead
        dead_letters()                -> ["a"]
        reserve("w")                  -> None
    """, lease=10, cap=1)
    run(sol, 3, """
        submit("a", 1)
        r1 = reserve("w")             -> "a"
        fail("a", r1)
        r2 = reserve("w")             -> "a"
        fail("a", r2)
        dead_letters()                -> []      # 只预约了 2 次，还没到 max_attempts = 3
        r3 = reserve("w")             -> "a"     # 第 3 次预约
        fail("a", r3)                 # 3 >= 3：进入 dead
        dead_letters()                -> ["a"]
        reserve("w")                  -> None
    """, lease=10, cap=3)


@case("Part 3", title="租约过期也计入尝试次数")
def p3_expiry_counts(sol):
    run(sol, 3, """
        submit("a", 1)
        r1 = reserve("w")             -> "a"     # 第 1 次预约，截止时刻 2
        advance(3)
        r2 = reserve("w")             -> "a"     # 先回收 r1（1 < 2：回到 ready），再预约：第 2 次
        advance(3)
        dead_letters()                -> ["a"]   # dead_letters 也先回收：r2 过期时已预约 2 次，进入 dead
        reserve("w")                  -> None
        complete("a", r2)             !! InvalidReservationError    # dead 不是 reserved
    """, lease=2, cap=2)
    run(sol, 3, """
        submit("a", 1)
        ra = reserve("w")             -> "a"
        advance(3)
        requeue_dead("a")             # requeue_dead 也先回收：a 过期且已预约 1 次，进入 dead，随即被放回 ready
        r2 = reserve("w")             -> "a"
    """, lease=2, cap=1)


@case("Part 3", title="dead_letters 按进入 dead 的先后")
def p3_dead_letters_order(sol):
    run(sol, 3, """
        submit("m", 1)
        submit("k", 2)
        submit("z", 3)
        submit("b", 4)
        rm = reserve("w")             -> "m"
        rk = reserve("w")             -> "k"
        rz = reserve("w")             -> "z"
        rb = reserve("w")             -> "b"     # m、k、z、b 的截止时刻都是 3
        fail("z", rz)
        dead_letters()                -> ["z"]
        fail("m", rm)
        dead_letters()                -> ["z", "m"]    # 按进入 dead 的先后，不按 task_id 排序
        advance(4)
        dead_letters()                -> ["z", "m", "b", "k"]    # 同一轮回收里 b、k 截止时刻相同，按 task_id
        requeue_dead("m")
        dead_letters()                -> ["z", "b", "k"]
        rm2 = reserve("w")            -> "m"
        fail("m", rm2)
        dead_letters()                -> ["z", "b", "k", "m"]    # m 重新进入 dead，排到最后
    """, lease=3, cap=1)


@case("Part 3", title="requeue_dead：排到队尾，次数清零")
def p3_requeue(sol):
    run(sol, 3, """
        submit("a", 1)
        submit("b", 2)
        ra = reserve("w")             -> "a"
        fail("a", ra)                 # a 已预约 1 次
        rb = reserve("w")             -> "b"
        ra2 = reserve("w")            -> "a"
        fail("a", ra2)                # a 已预约 2 次：进入 dead
        fail("b", rb)                 # b 已预约 1 次：回到 ready
        submit("c", 3)
        requeue_dead("a")             # a 排到 ready 队尾，在 b、c 后面
        rb2 = reserve("w")            -> "b"
        rc = reserve("w")             -> "c"
        ra3 = reserve("w")            -> "a"     # 尝试次数已清零：这是第 1 次
        fail("a", ra3)                # 1 < 2：回到 ready，不是 dead
        dead_letters()                -> []      # requeue_dead 已把尝试次数清零，a 只失败了 1 次
        ra4 = reserve("w")            -> "a"     # 第 2 次
        fail("a", ra4)
        dead_letters()                -> ["a"]
        complete("a", ra4)            !! InvalidReservationError
    """, lease=10, cap=2)


@case("Part 3", title="requeue_dead 的两种异常")
def p3_requeue_errors(sol):
    run(sol, 3, """
        requeue_dead("ghost")         !! UnknownTaskError
        submit("a", 1)
        requeue_dead("a")             !! ValueError    # a 处于 ready
        ra = reserve("w")             -> "a"
        requeue_dead("a")             !! ValueError    # a 处于 reserved
        fail("a", ra)                 # max_attempts = 1：进入 dead
        requeue_dead("a")
        requeue_dead("a")             !! ValueError    # 已经放回 ready
        rb = reserve("w")             -> "a"
        complete("a", rb)
        requeue_dead("a")             !! ValueError    # a 处于 completed
        requeue_dead("nope")          !! UnknownTaskError
    """, lease=10, cap=1)


@case("Part 3", title="随机操作序列")
def p3_random(sol):
    random_check(sol, 3, 300, 30)


@case("Part 3", cpu=4, title="5 万步随机操作")
def p3_big(sol):
    wants = []
    big_check(sol, 3, ref.gen_ops(rng(), 50_000, 3, 5, 3, fast=True, wants=wants), wants, lease=5, cap=3)
