import ast
import gc
import random
import re
import time
import types

from judgelib import calling, case, expect, expect_true, rng
import ref

SHOW = 18               # a replay lists at most this many calls
P1, P2, P3 = "RecentEventCounter", "SessionActivityTracker", "OutOfOrderSessionTracker"


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    return repr(x)


def _note(want, why=""):
    """The comment after an earlier call: its expected value, or for a record call the script's remark."""
    return f"# -> {want}" if want is not None else (f"# {why}" if why else "")


def _replay(cls_name, window, done, notes, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step.
    `notes` holds the comment for every earlier call (see _note)."""
    query = ref.QUERY[cls_name]
    lines = [f"t.{m}({', '.join(_lit(a) for a in args)})" for m, args in done]
    if done[-1][0] == query:
        lines[-1] = f"print({lines[-1]})"
    notes = list(notes) + [mark]
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = [f"t = {cls_name}({window})"]
    if skipped:
        head.append(f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）")
    return "\n".join(head + [(s.ljust(width) + n).rstrip() for s, n in zip(lines, notes)])


def _make(sol, cls_name, window):
    cls = getattr(sol, cls_name, None)
    expect_true(isinstance(cls, type), f"answer.py 里没有 {cls_name} 类（类名要和题目一致）")
    calling(f"t = {cls_name}({window})")
    tracker = cls(window)
    for name in ref.MODELS[cls_name].__dict__:
        if not name.startswith("_"):
            expect_true(callable(getattr(tracker, name, None)), f"{cls_name} 没有 {name} 方法（方法名要和题目一致）")
    return tracker


def _same(got, want):
    try:
        return type(got) is not str and bool(got == want)
    except Exception:
        return False


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines like  recent_count("alice", "a1") -> 2   # why  become (method, args, want, why); a line without
    '->' is a record call."""
    steps = []
    for line in script.strip().splitlines():
        code, _, why = line.partition("#")          # the scripts have no '#' inside a string
        call, arrow, expr = code.strip().partition("->")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(ast.literal_eval(a) for a in node.args)
        want = ast.literal_eval(expr.strip()) if arrow else None
        steps.append((node.func.id, args, want, why.strip()))
    return steps


def run(sol, cls_name, window, script):
    """Runs the calls on a fresh tracker, checking each query's answer as soon as it comes back. Returns the
    tracker and the whole sequence as code."""
    t = _make(sol, cls_name, window)
    done, notes = [], []
    for method, args, want, why in _parse(script):
        done.append((method, args))
        calling(_replay(cls_name, window, done, notes))     # shown if the answer raises or hangs in this call
        try:
            got = getattr(t, method)(*args)
        except Exception:
            calling(_replay(cls_name, window, done, notes, mark="# <- 这一步抛出异常"))
            raise
        if want is not None and not _same(got, want):
            expect(got, want, call=_replay(cls_name, window, done, notes), msg=why or "返回值不对")
        notes.append(_note(want, why))
    return t, _replay(cls_name, window, done, notes[:-1], mark=notes[-1])


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, cls_name, window, ops):
    """Replays ops on a fresh answer and a fresh model; None if every answer agrees."""
    t, model, wants = getattr(sol, cls_name)(window), ref.MODELS[cls_name](window), []
    query = ref.QUERY[cls_name]
    for k, (method, args) in enumerate(ops):
        want = getattr(model, method)(*args)
        try:
            got = getattr(t, method)(*args)
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if method == query and not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        wants.append(want)
    return None


def _shrink(sol, cls_name, window, ops, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short. Any subsequence of a valid
    sequence is valid: timestamps stay in order (Parts 1-2) and every `now` stays >= the timestamps before it."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, cls_name, window, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def _report(cls_name, window, ops, bad, what):
    k, notes = bad["k"], [_note(w) for w in bad["wants"]]
    if bad["exc"] is not None:
        calling(_replay(cls_name, window, ops[: k + 1], notes, mark="# <- 这一步抛出异常"))
        raise bad["exc"]
    expect(bad["got"], bad["want"], call=_replay(cls_name, window, ops[: k + 1], notes),
           msg=f"{what}第 {k + 1} 步返回值不对")


def random_check(sol, cls_name, gen, sequences, length):
    r = rng()
    _make(sol, cls_name, 1)
    for _ in range(sequences):
        window = r.choice([1, 2, 3, 5, 8])
        users = ["alice", "bruno", "chen"][: r.randint(1, 3)]
        chats = ["x", "y", "z"][: r.randint(1, 3)]
        ops = gen(r, r.randint(*length), window, users, chats, start=r.randint(0, 3))
        calling(f"随机操作序列（{cls_name}({window})，{len(ops)} 次调用）")
        bad = _outcome(sol, cls_name, window, ops)
        if bad is not None:
            ops, bad = _shrink(sol, cls_name, window, ops, bad)
            _report(cls_name, window, ops, bad, "随机操作序列的（已自动删去与出错无关的调用）")


def big_check(sol, cls_name, gen, window, users, chats, steps):
    """One long sequence over many users and chats; a failure shows its last calls."""
    ops = gen(rng(), steps, window, users, chats, start=1000)
    _make(sol, cls_name, window)
    calling(f"{steps} 次随机调用（{cls_name}({window})，{len(users)} 个用户 × {len(chats)} 个 chat_id）")
    bad = _outcome(sol, cls_name, window, ops)
    if bad is not None:
        _report(cls_name, window, ops, bad, f"{steps} 次随机调用中，")


BIG_USERS = [f"user{i}" for i in range(150)]
BIG_CHATS = [f"chat{i}" for i in range(8)]


# ---------------------------------------------------------------- the memory rule

_OPAQUE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
           types.CodeType, types.FrameType)


def _reachable(root):
    """Every str and dict reachable from the tracker through its attributes and containers (not through its
    class or module). gc.get_referents skips the str keys of a dict, so dict keys are added by hand."""
    seen, stack, strs, dicts = {id(root)}, [root], [], []
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            strs.append(x)
            continue
        if isinstance(x, _OPAQUE):
            continue
        refs = gc.get_referents(x)
        if isinstance(x, dict):
            dicts.append(x)
            refs = list(dict.keys(x)) + refs
        for y in refs:
            if id(y) not in seen:
                seen.add(id(y))
                stack.append(y)
    return strs, dicts


MARK = re.compile(r"m[uc](\d{6})")


def uid(i):
    return f"mu{i:06d}"


def cid(i):
    return f"mc{i:06d}"


def _held(t):
    """The numbers i of the chats (uid(i), cid(i)) whose user_id or chat_id the tracker still refers to."""
    found = set()
    for s in _reachable(t)[0]:
        if "m" in s:
            found.update(int(x) for x in MARK.findall(s))
    return found


def check_memory(t, allowed, code, situation, subset=False):
    """Parts 1-2: the tracker refers to at most len(allowed) chats. Part 3 (subset): only to chats in allowed."""
    held = _held(t)
    extra = sorted(held - allowed)
    if (extra if subset else len(held) > len(allowed)):
        shown = "、".join(f'("{uid(i)}", "{cid(i)}")' for i in extra[:3])
        limit = f"只能记住这 {len(allowed)} 个会话" if allowed else "不能再记住任何会话"
        expect_true(False, f"内存约束：{situation}，tracker {limit}；但从 t 出发还能找到 {len(held)} 个会话的 "
                           f"user_id / chat_id，其中 {len(extra)} 个早已不在窗口内（例如 {shown}）", call=code)


def _check_value(got, want, code, msg):
    if not _same(got, want):
        expect(got, want, call=code, msg=msg)


# ================================================================ Part 1

@case("Part 1", sample=True, title="题目例子（window = 5）")
def p1_example(sol):
    run(sol, P1, 5, """
        record_event("alice", "a1", 1)
        record_event("alice", "a1", 1)
        recent_count("alice", "a1") -> 2     # 允许重复的 timestamp；T = 1，窗口 [-4, 1]
        record_event("alice", "a2", 3)
        recent_count("alice", "a1") -> 2     # T 变成 3；那两条 timestamp=1 的事件仍在 [-2, 3] 内
        recent_count("alice", "a2") -> 1
        record_event("alice", "a1", 9)
        recent_count("alice", "a1") -> 1     # T = 9，窗口 [4, 9]；只剩新事件
        recent_count("alice", "a2") -> 0     # a2 唯一的事件（timestamp=3）已经落在 [4, 9] 之外
    """)


@case("Part 1", title="窗口左端 T - window 是闭的")
def p1_left_edge(sol):
    run(sol, P1, 5, """
        record_event("alice", "a1", 0)
        record_event("alice", "a1", 2)
        record_event("bruno", "b1", 5)
        recent_count("alice", "a1") -> 2     # T = 5，窗口 [0, 5]：timestamp 0 恰好在左端，仍然算
        record_event("bruno", "b1", 6)
        recent_count("alice", "a1") -> 1     # T = 6，窗口 [1, 6]：timestamp 0 出窗，2 还在
        record_event("bruno", "b1", 7)
        recent_count("alice", "a1") -> 1     # T = 7，窗口 [2, 7]：timestamp 2 恰好在左端
        recent_count("bruno", "b1") -> 3
        record_event("bruno", "b1", 8)
        recent_count("alice", "a1") -> 0
    """)


@case("Part 1", title="T 是全体会话的最大时间戳")
def p1_global_clock(sol):
    run(sol, P1, 3, """
        recent_count("alice", "a1") -> 0     # 还没有任何事件
        record_event("alice", "a1", 10)
        record_event("alice", "a1", 11)
        recent_count("alice", "a1") -> 2
        recent_count("alice", "zz") -> 0     # 从没出现过的会话
        record_event("bruno", "b1", 14)
        recent_count("alice", "a1") -> 1     # T 由 bruno 的事件推到 14，窗口 [11, 14]；不是 a1 自己的最大值 11
        record_event("chen", "c1", 20)
        recent_count("alice", "a1") -> 0     # T = 20
        recent_count("bruno", "b1") -> 0
        record_event("alice", "a1", 20)
        recent_count("alice", "a1") -> 1
    """)


@case("Part 1", title="会话由 (user_id, chat_id) 决定")
def p1_pair_key(sol):
    run(sol, P1, 10, """
        record_event("alice", "x", 1)
        record_event("bruno", "x", 2)        # 同一个 chat_id，不同用户：是两个会话
        record_event("bruno", "x", 3)
        recent_count("alice", "x") -> 1      # bruno 的 "x" 是另一个会话
        recent_count("bruno", "x") -> 2
        recent_count("chen", "x") -> 0
        record_event("a:b", "c", 4)          # id 里可以有任何字符：("a:b", "c") 和 ("a", "b:c") 是不同的会话
        record_event("ab", "c", 4)
        recent_count("a", "b:c") -> 0
        recent_count("a", "bc") -> 0
        recent_count("a:b", "c") -> 1
        recent_count("x", "alice") -> 0      # user_id 和 chat_id 不能对调
    """)


@case("Part 1", title="随机操作序列，和逐条统计比较")
def p1_random(sol):
    random_check(sol, P1, ref.p1_ops, 400, (1, 40))


@case("Part 1", title="4 万次随机调用，1200 个会话")
def p1_big(sol):
    big_check(sol, P1, ref.p1_ops, 50, BIG_USERS, BIG_CHATS, 40000)


@case("Part 1", title="2 万个会话同在窗口内：均摊 O(1)")
def p1_many_live_chats(sol):
    window, n = 10 ** 9, 20000
    code = (f"t = {P1}({window})\n"
            f"for i in range({n}):\n"
            f'    t.record_event(f"mu{{i:06d}}", f"mc{{i:06d}}", i)\n'
            f"    if i % 2:\n"
            f'        t.recent_count(f"mu{{i // 2:06d}}", f"mc{{i // 2:06d}}")   # 应当是 1：所有事件都在窗口内')
    t = _make(sol, P1, window)
    calling(code)
    for i in range(n):
        t.record_event(uid(i), cid(i), i)
        if i % 2:
            j = i // 2
            _check_value(t.recent_count(uid(j), cid(j)), 1, code, f"i = {i} 时 recent_count(\"{uid(j)}\", \"{cid(j)}\") 不对")


@case("Part 1", title="内存：只为窗口内的会话保留状态")
def p1_memory_stream(sol):
    window, n = 10, 30000
    code = (f"t = {P1}({window})\n"
            f"for i in range({n}):\n"
            f'    t.record_event(f"mu{{i:06d}}", f"mc{{i:06d}}", i)\n'
            f"    if i % 3 == 0:\n"
            f'        t.recent_count(f"mu{{i // 2:06d}}", f"mc{{i // 2:06d}}")              # 一个更早的会话\n'
            f'        t.recent_count(f"mu{{i + 500000:06d}}", f"mc{{i + 500000:06d}}")      # 从没出现过的会话\n'
            f"# 结束时 T = {n - 1}，窗口 [{n - 1 - window}, {n - 1}]")
    t = _make(sol, P1, window)
    calling(code)
    for i in range(n):
        t.record_event(uid(i), cid(i), i)
        if i % 3 == 0:
            j = i // 2
            _check_value(t.recent_count(uid(j), cid(j)), int(i - j <= window), code,
                         f"i = {i} 时 recent_count(\"{uid(j)}\", \"{cid(j)}\") 不对")
            _check_value(t.recent_count(uid(i + 500000), cid(i + 500000)), 0, code,
                         f"i = {i} 时从没出现过的会话应当返回 0")
    check_memory(t, set(range(n - 1 - window, n)), code, f"结束时只有 {window + 1} 个会话在窗口内有事件")


@case("Part 1", title="内存：一次跳过窗口要清空全部")
def p1_memory_burst(sol):
    window, n = 10, 20000
    code = (f"t = {P1}({window})\n"
            f"for i in range({n}):\n"
            f'    t.record_event(f"mu{{i:06d}}", f"mc{{i:06d}}", 7)\n'
            f't.record_event("mu999999", "mc999999", 18)   # T = 18，窗口 [8, 18]：前面 {n} 个会话全部出窗')
    t = _make(sol, P1, window)
    calling(code)
    for i in range(n):
        t.record_event(uid(i), cid(i), 7)
    _check_value(t.recent_count(uid(0), cid(0)), 1, code, "这时所有事件都在窗口 [-3, 7] 内")
    t.record_event(uid(999999), cid(999999), 18)
    check_memory(t, {999999}, code, "最后一次 record_event 之后只有 1 个会话在窗口内有事件")
    _check_value(t.recent_count(uid(0), cid(0)), 0, code + "\nprint(t.recent_count(\"mu000000\", \"mc000000\"))",
                 "timestamp 7 已不在窗口 [8, 18] 内")


# ================================================================ Part 2

@case("Part 2", sample=True, title="题目例子（window = 6）")
def p2_example(sol):
    run(sol, P2, 6, """
        record("alice", "a1", "ping", 0)
        active_sessions("alice") -> 1
        record("alice", "a2", "ping", 2)
        active_sessions("alice") -> 2
        record("alice", "a1", "close", 3)
        active_sessions("alice") -> 1        # a1 已关闭，a2 仍活跃
        record("alice", "a2", "ping", 5)     # 刷新 a2
        active_sessions("alice") -> 1
        record("bruno", "b1", "ping", 9)
        active_sessions("alice") -> 1        # T 变成 9；a2 的 ping(5) 仍在 [3, 9] 内
        record("bruno", "b2", "ping", 12)
        active_sessions("alice") -> 0        # T 变成 12；a2 的 ping(5) 已经落在 [6, 12] 之外
    """)


@case("Part 2", title="同一时间戳的 ping/close：后记录的为准")
def p2_tie_call_order(sol):
    run(sol, P2, 100, """
        record("alice", "a1", "ping", 5)
        record("alice", "a1", "close", 5)
        active_sessions("alice") -> 0        # 同一 timestamp，close 记录得更晚，以它为准 -> 关闭
        record("bruno", "b1", "close", 5)
        record("bruno", "b1", "ping", 5)
        active_sessions("bruno") -> 1        # 同一 timestamp，ping 记录得更晚，以它为准 -> 活跃
        record("alice", "a1", "ping", 7)
        record("alice", "a1", "close", 7)
        record("alice", "a1", "ping", 7)
        active_sessions("alice") -> 1        # 最后记录的是 ping
        record("bruno", "b1", "close", 7)
        active_sessions("bruno") -> 0        # timestamp 7 晚于 bruno 的 ping(5)
    """)


@case("Part 2", title="反复 ping 只算一个，按最后一次过期")
def p2_refresh(sol):
    run(sol, P2, 4, """
        record("alice", "a1", "ping", 0)
        record("alice", "a1", "ping", 1)
        record("alice", "a1", "ping", 1)
        active_sessions("alice") -> 1        # 同一个会话 ping 几次都只算一个
        record("alice", "a1", "ping", 3)
        record("bruno", "b1", "ping", 6)     # T = 6，窗口 [2, 6]：ping(0)、ping(1) 出窗，ping(3) 还在
        active_sessions("alice") -> 1
        record("bruno", "b1", "ping", 7)     # T = 7，窗口 [3, 7]：ping(3) 恰好在左端
        active_sessions("alice") -> 1
        record("bruno", "b1", "ping", 8)     # T = 8，窗口 [4, 8]
        active_sessions("alice") -> 0
        active_sessions("bruno") -> 1
    """)


@case("Part 2", title="close 的各种情况")
def p2_close_cases(sol):
    run(sol, P2, 5, """
        record("alice", "a1", "close", 0)    # 从没 ping 过的会话被 close：没有影响
        active_sessions("alice") -> 0
        record("alice", "a1", "ping", 1)
        active_sessions("alice") -> 1
        record("alice", "a1", "close", 2)
        record("alice", "a1", "close", 2)    # 重复 close：不会变成负数
        active_sessions("alice") -> 0
        record("alice", "a1", "ping", 3)     # close 之后再 ping：重新活跃
        active_sessions("alice") -> 1
        record("alice", "a2", "ping", 3)
        active_sessions("alice") -> 2
        record("bruno", "b1", "ping", 9)     # T = 9，窗口 [4, 9]：alice 的两个 ping(3) 都出窗
        active_sessions("alice") -> 0
        record("alice", "a1", "close", 10)   # 已经过期的会话再被 close：不会变成负数
        active_sessions("alice") -> 0
        record("alice", "a2", "ping", 10)
        active_sessions("alice") -> 1
        active_sessions("chen") -> 0         # 从没出现过的用户
    """)


@case("Part 2", title="不同用户可以有相同的 chat_id")
def p2_shared_chat_id(sol):
    run(sol, P2, 10, """
        record("alice", "x", "ping", 1)
        record("bruno", "x", "ping", 2)
        record("bruno", "y", "ping", 2)
        active_sessions("alice") -> 1
        active_sessions("bruno") -> 2
        record("bruno", "x", "close", 3)     # 只关闭 bruno 的 x
        active_sessions("alice") -> 1
        active_sessions("bruno") -> 1
        record("alice", "y", "close", 4)     # alice 没有 y：没有影响
        active_sessions("bruno") -> 1
        active_sessions("alice") -> 1
    """)


@case("Part 2", title="随机操作序列，和逐条判定比较")
def p2_random(sol):
    random_check(sol, P2, ref.p2_ops, 400, (1, 40))


@case("Part 2", title="4 万次随机调用，1200 个会话")
def p2_big(sol):
    big_check(sol, P2, ref.p2_ops, 50, BIG_USERS, BIG_CHATS, 40000)


@case("Part 2", title="内存：只为窗口内的会话保留状态")
def p2_memory_stream(sol):
    window, n = 10, 30000
    code = (f"t = {P2}({window})\n"
            f"for i in range({n}):\n"
            f'    t.record(f"mu{{i:06d}}", f"mc{{i:06d}}", "ping", i)\n'
            f"    if i % 4 == 3:\n"
            f'        t.record(f"mu{{i:06d}}", f"mc{{i:06d}}", "close", i)\n'
            f"    if i % 3 == 0:\n"
            f'        t.active_sessions(f"mu{{i // 2:06d}}")              # 一个更早的用户\n'
            f'        t.active_sessions(f"mu{{i + 500000:06d}}")          # 从没出现过的用户\n'
            f"# 结束时 T = {n - 1}，窗口 [{n - 1 - window}, {n - 1}]")
    t = _make(sol, P2, window)
    calling(code)
    for i in range(n):
        t.record(uid(i), cid(i), "ping", i)
        if i % 4 == 3:
            t.record(uid(i), cid(i), "close", i)
        if i % 3 == 0:
            j = i // 2
            _check_value(t.active_sessions(uid(j)), int(j % 4 != 3 and i - j <= window), code,
                         f"i = {i} 时 active_sessions(\"{uid(j)}\") 不对")
            _check_value(t.active_sessions(uid(i + 500000)), 0, code, f"i = {i} 时从没出现过的用户应当返回 0")
    check_memory(t, set(range(n - 1 - window, n)), code, f"结束时只有 {window + 1} 个会话在窗口内有事件")


@case("Part 2", title="内存：活跃数为 0 的用户不留状态")
def p2_memory_users(sol):
    t, code = run(sol, P2, 10, """
        record("u_old", "c1", "ping", 0)
        record("u_closed", "c1", "ping", 1)
        record("u_closed", "c2", "ping", 1)
        active_sessions("u_closed") -> 2
        record("u_closed", "c1", "close", 2)
        record("u_closed", "c2", "close", 3)
        active_sessions("u_closed") -> 0     # 两个会话都被 close 了
        active_sessions("u_ghost") -> 0      # 从没出现过的用户
        record("u_new", "c1", "ping", 11)    # T = 11，窗口 [1, 11]：u_old 唯一的 ping(0) 出窗
        active_sessions("u_old") -> 0
        active_sessions("u_new") -> 1
    """)
    strs, dicts = _reachable(t)
    for d in dicts:
        if dict.__contains__(d, "u_closed") and _empty(dict.__getitem__(d, "u_closed")):
            expect_true(False, "内存约束：u_closed 的 active_sessions 已经是 0，按 user_id 索引的状态不能再为它保留"
                               f"条目；但 tracker 里还有一个以 \"u_closed\" 为键、值为 "
                               f"{dict.__getitem__(d, 'u_closed')!r} 的条目", call=code)
    for user, why in (("u_old", "u_old 唯一的事件已经出窗"), ("u_ghost", "u_ghost 从没记录过事件，只被查询过")):
        hits = [s for s in strs if user in s]
        if hits:
            expect_true(False, f"内存约束：{why}，tracker 里不能再有它的任何状态；但从 t 出发还能找到字符串 "
                               f"{hits[0]!r}", call=code)


def _empty(value):
    """0, None, False or an empty container: an entry that only says 'nothing here'."""
    try:
        return value is None or value is False or (isinstance(value, (int, float)) and value == 0) \
            or (hasattr(value, "__len__") and len(value) == 0)
    except Exception:
        return False


# ================================================================ Part 3

@case("Part 3", sample=True, title="题目例子（window = 4）")
def p3_example(sol):
    run(sol, P3, 4, """
        record("alice", "a1", "ping", 10)
        active_sessions("alice", 10) -> 1
        record("alice", "a1", "close", 8)    # 一条 close，但比那条 ping 更早 -> 没有影响
        active_sessions("alice", 11) -> 1
        record("alice", "a1", "close", 12)   # 一条比 ping 更晚的 close
        active_sessions("alice", 12) -> 0
        record("alice", "a1", "ping", 11)    # 一条迟到的 ping，但仍早于那条 close -> 依然是关闭状态
        active_sessions("alice", 13) -> 0
        record("alice", "a1", "ping", 14)    # 一条比 close 更晚的 ping -> 重新活跃
        active_sessions("alice", 14) -> 1
        record("alice", "a2", "ping", 5)     # C = 14，所以 C - window = 10；5 < 10 -> 已过期，直接丢弃
        active_sessions("alice", 14) -> 1    # 不变：a2 从未真正进入过状态
    """)


@case("Part 3", title="并列归 ping，与到达顺序无关")
def p3_tie_goes_to_ping(sol):
    run(sol, P3, 100, """
        record("alice", "a1", "close", 5)
        record("alice", "a1", "ping", 5)
        active_sessions("alice", 5) -> 1     # last_close = last_ping = 5：并列归 ping
        record("bruno", "b1", "ping", 5)
        record("bruno", "b1", "close", 5)    # 顺序反过来也一样
        active_sessions("bruno", 5) -> 1
        record("bruno", "b1", "close", 6)
        active_sessions("bruno", 6) -> 0
        record("bruno", "b1", "ping", 6)     # 又一次并列
        active_sessions("bruno", 6) -> 1
    """)


@case("Part 3", title="窗口两端：过期与丢弃的边界")
def p3_window_edges(sol):
    run(sol, P3, 4, """
        record("alice", "a1", "ping", 12)
        active_sessions("alice", 16) -> 1    # last_ping = 12 = now - window：仍然活跃
        active_sessions("alice", 17) -> 0
        record("bruno", "b1", "ping", 13)
        active_sessions("bruno", 17) -> 1    # C = 17，C - window = 13：timestamp 13 恰好在边界上，不能丢弃
        record("bruno", "b2", "ping", 12)
        active_sessions("bruno", 17) -> 1    # 12 < 13：这条 ping 丢弃
        record("bruno", "b1", "close", 13)
        active_sessions("bruno", 17) -> 1    # last_close = last_ping = 13：并列归 ping
        record("bruno", "b1", "close", 14)
        active_sessions("bruno", 17) -> 0    # last_close = 14 > last_ping = 13
    """)


@case("Part 3", title="close 也要记住，直到它出窗")
def p3_close_remembered(sol):
    run(sol, P3, 4, """
        record("alice", "a1", "ping", 10)
        record("alice", "a1", "close", 12)
        active_sessions("bruno", 15) -> 0    # 时钟走到 15；a1 的 ping(10) 已出窗，但 close(12) 还在 window 范围内
        record("alice", "a1", "ping", 11)    # 11 >= 15 - 4：要处理
        active_sessions("alice", 15) -> 0    # a1：last_ping = 11 早于 last_close = 12，仍是关闭状态
        record("alice", "a2", "close", 14)   # 只有 close，还没有 ping
        record("alice", "a2", "ping", 13)
        active_sessions("alice", 15) -> 0    # a2：迟到的 ping(13) 早于 close(14)
        record("alice", "a2", "ping", 15)
        active_sessions("alice", 15) -> 1    # a2：ping(15) 晚于 close(14)
        record("alice", "a1", "ping", 16)
        active_sessions("alice", 16) -> 2    # a1：ping(16) 晚于 close(12)，重新活跃
    """)


@case("Part 3", title="同一批事件换顺序到达，答案不变")
def p3_any_order(sol):
    r = rng()
    _make(sol, P3, 1)
    for _ in range(300):
        window = r.choice([1, 2, 4, 8])
        records = [("record", (r.choice(["alice", "bruno"]), r.choice(["x", "y"]), r.choice(["ping", "close"]),
                               r.randint(0, 25))) for _ in range(r.randint(2, 12))]
        now = max(args[3] for _, args in records)
        queries = [("active_sessions", (u, now)) for u in ("alice", "bruno")]
        for _ in range(3):
            ops = records[:]
            r.shuffle(ops)
            ops += queries
            bad = _outcome(sol, P3, window, ops)
            if bad is not None:
                _report(P3, window, ops, bad, "同一批事件以随机顺序到达，")


@case("Part 3", title="随机乱序序列，和规则 2 比较")
def p3_random(sol):
    random_check(sol, P3, ref.p3_ops, 400, (1, 40))


@case("Part 3", title="4 万次随机调用，1200 个会话")
def p3_big(sol):
    big_check(sol, P3, ref.p3_ops, 50, BIG_USERS, BIG_CHATS, 40000)


@case("Part 3", title="内存：乱序记录、没有查询")
def p3_memory_shuffled(sol):
    window, n = 100, 30000
    seed = rng().randrange(1000)
    code = (f"import random\n"
            f"t = {P3}({window})\n"
            f"order = list(range({n}))\n"
            f"random.Random({seed}).shuffle(order)\n"
            f"for ts in order:\n"
            f'    t.record(f"mu{{ts:06d}}", f"mc{{ts:06d}}", "close" if ts % 5 == 0 else "ping", ts)\n'
            f"# 结束时 C = {n - 1}：只有 timestamp >= {n - 1 - window} 的会话还在 C 的 window 范围内")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    t = _make(sol, P3, window)
    calling(code)
    for ts in order:
        t.record(uid(ts), cid(ts), "close" if ts % 5 == 0 else "ping", ts)
    check_memory(t, set(range(n - 1 - window, n)), code, f"结束时 C = {n - 1}，只有 {window + 1} 个会话的时间戳还在 "
                 f"[{n - 1 - window}, {n - 1}] 内", subset=True)
    _check_value(t.active_sessions("nobody", n - 1), 0, code + f'\nprint(t.active_sessions("nobody", {n - 1}))',
                 "从没出现过的用户应当返回 0")


@case("Part 3", title="内存：时钟跳过窗口要清空全部")
def p3_memory_burst(sol):
    window, n = 100, 20000
    code = (f"t = {P3}({window})\n"
            f"for i in range({n}):\n"
            f'    t.record(f"mu{{i:06d}}", f"mc{{i:06d}}", "ping", 7)\n'
            f't.record("probe", "p1", "ping", 7)\n'
            f'print(t.active_sessions("probe", 7))       # 应当是 1\n'
            f'print(t.active_sessions("nobody", 108))    # 应当是 0：now = 108，7 < 108 - {window}')
    t = _make(sol, P3, window)
    calling(code)
    for i in range(n):
        t.record(uid(i), cid(i), "ping", 7)
    t.record("probe", "p1", "ping", 7)
    _check_value(t.active_sessions("probe", 7), 1, code, "这时 probe 的 ping(7) 还在窗口内")
    _check_value(t.active_sessions("nobody", 108), 0, code, "从没出现过的用户应当返回 0")
    check_memory(t, set(), code, "now = 108 之后所有会话的时间戳 7 都已经不在 C 的 window 范围内", subset=True)
