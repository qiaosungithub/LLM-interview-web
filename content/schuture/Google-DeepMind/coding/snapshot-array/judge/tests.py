from collections.abc import Mapping

import judgelib
from judgelib import Mismatch, case, expect, expect_true, rng
import ref
from ref import Raises

IE, VE = Raises(IndexError), Raises(ValueError)
SHOW = 45           # a replay lists at most this many calls

# ---------------------------------------------------------------- driving a sequence of calls

ATTEMPT = ["def attempt(f, *args):      # 出错时返回异常的类型名", "    try:", "        return f(*args)",
           "    except Exception as e:", "        return type(e).__name__"]


def _fmt(want):
    return repr(want) if not isinstance(want, Raises) else str(want)


def replay(length, done, wants):
    """The calls made so far as code to paste into 自定义运行; the last one is the step that failed."""
    skipped = max(0, len(done) - SHOW)
    lines = []
    for k, (op, want) in enumerate(zip(done, wants)):
        if k < skipped:
            continue
        last = k == len(done) - 1
        args = ", ".join(repr(a) for a in op[1:])
        expr = f"attempt(arr.{op[0]}{', ' + args if args else ''})" if isinstance(want, Raises) \
            else f"arr.{op[0]}({args})"
        if last:
            expr = f"print({expr})"
        note = "" if want is None else f"# -> {_fmt(want)}"
        if last:
            note = (note + "   " if note else "# ") + "<- 这一步"
        lines.append((expr, note))
    width = max(len(e) for e, _ in lines) + 3
    head = ATTEMPT if any(isinstance(w, Raises) for w in wants[skipped:]) else []
    head = head + [f"arr = SnapshotArray({length})"]
    if skipped:
        head.append(f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）")
    return "\n".join(head + [(e.ljust(width) + n).rstrip() for e, n in lines])


def norm_diff(got):
    """diff's result with list values read as tuples (the statement's tuple is not the point)."""
    if isinstance(got, Mapping):
        return {k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in got.items()}
    return got


def method(arr, name):
    fn = getattr(arr, name, None)
    expect_true(callable(fn), f"SnapshotArray 没有 {name} 方法（方法名要和题目一致）")
    return fn


def perform(arr, op, want, call):
    """Runs one call and checks its result against `want` (a value, None for set/release, or Raises)."""
    fn = method(arr, op[0])
    try:
        got = fn(*op[1:])
    except Exception as e:
        if isinstance(want, Raises):
            if isinstance(e, want.types):
                return
            raise Mismatch(f"应当抛出 {want}，实际抛出 {type(e).__name__}: {e}", call(), show=False) from None
        judgelib._current["call"] = call()      # shown with the traceback
        raise
    if isinstance(want, Raises):
        raise Mismatch(f"应当抛出 {want}，但正常返回了", call(), f"抛出 {want}", got)
    if op[0] in ("snap", "get", "diff"):
        value = norm_diff(got) if op[0] == "diff" else got
        try:
            same = bool(value == want)
        except Exception:
            same = False
        if not same:                            # the replay is built only when it is needed
            msg = "snap 返回的快照 id 不对：第一次是 0，之后依次加 1" if op[0] == "snap" else ""
            expect(value, want, call(), msg=msg)


def run_steps(sol, length, steps):
    """steps: (op, want) pairs, run in order on a fresh SnapshotArray(length), each result checked at once."""
    arr = sol.SnapshotArray(length)
    done, wants = [], []
    for op, want in steps:
        done.append(op)
        wants.append(want)
        perform(arr, op, want, lambda: replay(length, done, wants))
    return arr


def run_ops(sol, length, ops):
    """Like run_steps, with every expected result taken from the brute force in ref.py."""
    brute = ref.Brute(length)
    return run_steps(sol, length, [(op, brute.step(op)) for op in ops])


def st(i, v, want=None):
    return ("set", i, v), want


def sn(want):
    return ("snap",), want


def gt(i, s, want):
    return ("get", i, s), want


def rl(s, want=None):
    return ("release", s), want


def df(a, b, want):
    return ("diff", a, b), want


def S(i, v):
    return ("set", i, v)


def N():
    return ("snap",)


def G(i, s):
    return ("get", i, s)


def R(s):
    return ("release", s)


def D(a, b):
    return ("diff", a, b)


# The statement's trace, length 4.
TRACE = [st(0, 5), st(0, 6), sn(0), st(1, 9), st(0, 1), sn(1), st(0, 4), st(2, 2), sn(2), sn(3), sn(4), st(0, 7), sn(5),
         gt(0, 0, 6), gt(0, 1, 1), gt(0, 4, 4), gt(0, 5, 7), gt(1, 0, 0), gt(3, 5, 0)]


def random_run(sol, kinds, count, max_ops):
    r = rng()
    for _ in range(count):
        length = r.choice([0, 1, 1, 2, 3, 4, 5, 6])
        run_ops(sol, length, ref.random_ops(r, length, r.randint(5, max_ops), kinds))


def big_call(text):
    judgelib._current["call"] = text        # shown if the answer raises or runs out of time
    return text


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子（length = 4）")
def p1_example(sol):
    run_steps(sol, 4, TRACE)


@case("Part 1", title="snap 之前的写入对 get 不可见")
def p1_pending_invisible(sol):
    run_steps(sol, 2, [sn(0), st(0, 5), gt(0, 0, 0), gt(0, 1, IE), sn(1), gt(0, 1, 5), st(0, 6), st(1, 3),
                       gt(0, 1, 5), gt(1, 1, 0), gt(0, 0, 0), gt(1, 2, IE), sn(2), gt(0, 2, 6), gt(1, 2, 3), gt(0, 1, 5)])


@case("Part 1", title="下标越界（含负数）抛 IndexError")
def p1_bad_index(sol):
    run_steps(sol, 3, [st(-1, 5, IE), st(3, 5, IE), st(10 ** 9, 5, IE), sn(0), gt(-1, 0, IE), gt(3, 0, IE), gt(2, 0, 0),
                       st(2, 4), st(-3, 8, IE), sn(1), gt(-1, 1, IE), gt(2, 1, 4), gt(0, 1, 0), gt(-3, 1, IE)])


@case("Part 1", title="快照 id 没取过（含负数）抛 IndexError")
def p1_bad_snap_id(sol):
    run_steps(sol, 2, [gt(0, 0, IE), sn(0), gt(0, -1, IE), gt(0, 1, IE), gt(1, 5, IE), st(0, 3), gt(0, 1, IE),
                       sn(1), gt(0, 1, 3), gt(0, 2, IE), gt(0, -2, IE), gt(0, 0, 0)])


@case("Part 1", title="同一段里多次写：只有最后一次可见")
def p1_overwrite(sol):
    run_steps(sol, 3, [st(1, 5), st(1, 6), st(1, 0), sn(0), gt(1, 0, 0), st(1, 8), st(2, -4), st(1, 9), sn(1),
                       gt(1, 1, 9), gt(1, 0, 0), gt(2, 1, -4), st(1, 7), st(1, 9), sn(2), gt(1, 2, 9), gt(1, 1, 9),
                       st(2, 1), st(2, -4), sn(3), gt(2, 3, -4), gt(2, 2, -4), gt(2, 0, 0)])


@case("Part 1", title="快照之间没有写入：值保持不变")
def p1_unchanged(sol):
    steps = [st(0, 3), sn(0), sn(1), sn(2), st(0, 4), sn(3), sn(4)]
    steps += [gt(0, s, 3 if s < 3 else 4) for s in range(5)] + [gt(1, s, 0) for s in range(5)]
    run_steps(sol, 2, steps)


@case("Part 1", title="负数、0 和很大的整数")
def p1_values(sol):
    run_steps(sol, 3, [st(0, -7), st(1, 10 ** 18), st(2, 5), sn(0), st(2, 0), st(1, -10 ** 18), sn(1),
                       gt(0, 0, -7), gt(1, 0, 10 ** 18), gt(2, 0, 5), gt(2, 1, 0), gt(1, 1, -10 ** 18)])


@case("Part 1", title="length = 0 和 length = 1")
def p1_tiny(sol):
    run_steps(sol, 0, [sn(0), sn(1), st(0, 1, IE), st(-1, 1, IE), gt(0, 0, IE), gt(0, 1, IE), sn(2)])
    run_steps(sol, 1, [st(0, 2), st(1, 2, IE), sn(0), st(0, 3), gt(0, 0, 2), gt(1, 0, IE), gt(-1, 0, IE), sn(1),
                       gt(0, 1, 3)])


@case("Part 1", title="随机操作序列（对照暴力解）")
def p1_random(sol):
    random_run(sol, ["set", "snap", "get"], 250, 40)


# Big inputs are drawn once at import time, which is not charged to any case.
_big = rng("big")
BIG_LEN = 1_000_000
BIG_WRITES = [(_big.randrange(BIG_LEN), _big.randint(1, 10 ** 9)) for _ in range(1500)]
BIG_READS = [(_big.choice(BIG_WRITES)[0] if _big.random() < 0.7 else _big.randrange(BIG_LEN), _big.randrange(1500))
             for _ in range(300)]


@case("Part 1", cpu=2, title="length = 10⁶、1500 次快照：snap 要 O(1)")
def p1_big_length(sol):
    call = big_call(f"arr = SnapshotArray({BIG_LEN})；1500 轮“随机 set 一个下标再 snap()”，然后 300 次 get")
    arr = sol.SnapshotArray(BIG_LEN)
    history = {}                                # index -> [(snap_id, value)], to compute what get must return
    for s, (i, v) in enumerate(BIG_WRITES):
        arr.set(i, v)
        history.setdefault(i, []).append((s, v))
        expect(arr.snap(), s, call, msg="snap 返回的快照 id 不对")
    for i, s in BIG_READS:
        want = 0
        for sid, v in history.get(i, []):
            if sid <= s:
                want = v
        expect(arr.get(i, s), want, f"{call}\n出错的调用：arr.get({i}, {s})")


MANY = 50_000
_many = rng("many")
MANY_READS = [_many.randrange(MANY // 2, MANY + 1) for _ in range(5000)]


@case("Part 1", cpu=2, title="5×10⁴ 个快照：get 不能逐个往回翻快照")
def p1_many_snaps(sol):
    call = big_call(f"arr = SnapshotArray(10)；set(0, 42)；snap()；之后 {MANY} 轮 set(1, k)、snap()（k = 1, 2, …）；"
                    f"然后 {len(MANY_READS)} 次 get(0, s)，s 在后一半")
    arr = sol.SnapshotArray(10)
    arr.set(0, 42)
    arr.snap()
    for k in range(1, MANY + 1):
        arr.set(1, k)
        arr.snap()
    for s in MANY_READS:
        expect(arr.get(0, s), 42, f"{call}\n出错的调用：arr.get(0, {s})")
    for s in MANY_READS[:50]:
        expect(arr.get(1, s), s, f"{call}\n出错的调用：arr.get(1, {s})")


HEAVY = 60_000
HEAVY_READS = [_many.randrange(HEAVY) for _ in range(8000)]


@case("Part 1", cpu=2, title="一个下标写了 6×10⁴ 次：get 要二分查找")
def p1_many_writes(sol):
    call = big_call(f"arr = SnapshotArray(2)；{HEAVY} 轮 set(0, k)、snap()（k = 0, 1, …）；然后 8000 次随机的 get(0, s)")
    arr = sol.SnapshotArray(2)
    for k in range(HEAVY):
        arr.set(0, k)
        arr.snap()
    for s in HEAVY_READS:
        expect(arr.get(0, s), s, f"{call}\n出错的调用：arr.get(0, {s})")


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子：release(2) 之后")
def p2_example(sol):
    run_steps(sol, 4, TRACE + [rl(2), gt(0, 1, VE), gt(0, 2, VE), gt(0, 4, 4), rl(0)])


@case("Part 2", title="边界之前最后一次写入的值仍然读得到")
def p2_floor_kept(sol):
    run_steps(sol, 3, [st(0, 5), sn(0), st(1, 2), sn(1), sn(2), sn(3), rl(2), gt(0, 3, 5), gt(1, 3, 2), gt(2, 3, 0),
                       st(1, 8), sn(4), rl(3), gt(1, 4, 8), gt(0, 4, 5), gt(1, 3, VE)])
    run_steps(sol, 2, [st(0, 1), sn(0), st(0, 2), sn(1), rl(1), gt(0, 1, VE), sn(2), gt(0, 2, 2), gt(1, 2, 0),
                       st(0, 3), sn(3), gt(0, 3, 3), gt(0, 2, 2)])


@case("Part 2", title="release 更小的 id 不会降低边界")
def p2_boundary_never_lowers(sol):
    run_steps(sol, 2, [st(0, 1), sn(0), st(0, 2), sn(1), sn(2), st(1, 3), sn(3), sn(4), rl(3), rl(1), gt(0, 1, VE),
                       gt(0, 2, VE), gt(1, 3, VE), gt(0, 4, 2), gt(1, 4, 3), rl(3), rl(0), gt(0, 3, VE), gt(1, 4, 3)])


@case("Part 2", title="release 没取过的 id 抛 IndexError")
def p2_release_bad_id(sol):
    run_steps(sol, 2, [rl(0, IE), sn(0), rl(-1, IE), rl(1, IE), st(0, 4), rl(1, IE), sn(1), gt(0, 0, 0), rl(5, IE),
                       gt(0, 1, 4), rl(0), gt(0, 0, VE), rl(-1, IE), rl(0), rl(2, IE), gt(0, 1, 4), gt(1, 0, VE)])


@case("Part 2", title="release 最新的快照之后继续用")
def p2_release_latest(sol):
    run_steps(sol, 3, [st(0, 7), st(2, 1), sn(0), st(2, 2), sn(1), rl(1), gt(0, 1, VE), gt(2, 0, VE), st(1, 5),
                       sn(2), gt(0, 2, 7), gt(1, 2, 5), gt(2, 2, 2), rl(2), sn(3), gt(0, 3, 7), gt(1, 3, 5),
                       gt(2, 3, 2), gt(2, 2, VE), gt(2, 4, IE)])


@case("Part 2", title="length = 0：release 也不能出错")
def p2_length_zero(sol):
    arr = run_steps(sol, 0, [sn(0), sn(1), rl(0), gt(0, 1, IE), rl(1), sn(2)])
    call = "arr = SnapshotArray(0)\narr.snap(); arr.snap(); arr.release(0); arr.release(1); arr.snap()\nprint(arr.stored_entries())"
    expect(method(arr, "stored_entries")(), 0, call, msg="length = 0 时没有任何下标，保存的条目应当是 0")


@case("Part 2", title="release 之后旧条目要被清掉")
def p2_memory_shrinks(sol):
    lines = ["arr = SnapshotArray(3)", "for s in range(200):", "    arr.set(0, s + 1)", "    arr.set(1, -s)",
             "    arr.snap()", "arr.release(195)", "for k in range(12):", "    arr.snap()",
             "    arr.release(196 + k)", "    for i in range(3):", "        arr.get(i, 200 + k)",
             "print(arr.stored_entries())"]
    call = big_call("\n".join(lines))
    arr = sol.SnapshotArray(3)
    for s in range(200):
        arr.set(0, s + 1)
        arr.set(1, -s)
        arr.snap()
    arr.release(195)
    for k in range(12):
        expect(arr.snap(), 200 + k, call, msg="snap 返回的快照 id 不对")
        arr.release(196 + k)
        for i, want in enumerate([200, -199, 0]):
            expect(arr.get(i, 200 + k), want, f"{call}\n出错的调用：arr.get({i}, {200 + k})")
    got = method(arr, "stored_entries")()
    expect_true(isinstance(got, int) and not isinstance(got, bool), "stored_entries() 应当返回 int", call, got)
    expect_true(got <= 20, f"stored_entries() = {got}：快照 0 到 207 都已释放，下标 0 和 1 都只需要保留快照 199 "
                f"时的那一条，下标 2 从没写过；按题意这里最多 2 条（允许到 20）。旧快照的条目一直没有被清掉", call, got)


@case("Part 2", title="随机操作序列（对照暴力解）")
def p2_random(sol):
    random_run(sol, ["set", "snap", "get", "release"], 250, 40)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子 diff(3, 5)、diff(2, 4)")
def p3_example(sol):
    run_steps(sol, 4, TRACE + [df(3, 5, {0: (4, 7)}), df(2, 4, {})])


@case("Part 3", sample=True, title="题目例子：release(2) 之后")
def p3_example_after_release(sol):
    run_steps(sol, 4, TRACE + [df(3, 5, {0: (4, 7)}), df(2, 4, {}), rl(2), df(2, 4, VE), df(3, 5, {0: (4, 7)})])


@case("Part 3", title="a ≥ b 抛 ValueError")
def p3_order(sol):
    run_steps(sol, 2, [st(0, 1), sn(0), st(0, 2), sn(1), df(1, 0, VE), df(1, 1, VE), df(0, 0, VE), df(0, 1, {0: (1, 2)})])


@case("Part 3", title="没取过的 id 抛 IndexError，已释放的抛 ValueError")
def p3_id_errors(sol):
    run_steps(sol, 2, [st(1, 1), sn(0), st(1, 2), sn(1), df(0, 2, IE), df(-1, 1, IE), df(0, 5, IE), st(0, 3),
                       df(1, 2, IE), sn(2), df(0, 2, {0: (0, 3), 1: (1, 2)}), rl(0), df(0, 1, VE), df(0, 2, VE),
                       df(1, 2, {0: (0, 3)}), df(1, 3, IE)])


@case("Part 3", title="改回原值的下标不在结果里")
def p3_write_back(sol):
    run_steps(sol, 3, [st(0, 5), st(1, 5), sn(0), st(0, 9), sn(1), st(0, 5), st(1, 7), st(1, 5), sn(2),
                       df(0, 2, {}), df(0, 1, {0: (5, 9)}), df(1, 2, {0: (9, 5)}), st(2, 3), sn(3), st(2, 0), sn(4),
                       df(2, 4, {}), df(2, 3, {2: (0, 3)}), df(0, 4, {})])


@case("Part 3", title="只比较 a、b 两个快照：之前、之后的写入不算")
def p3_endpoints(sol):
    run_steps(sol, 4, [st(0, 1), sn(0), st(1, 2), sn(1), st(2, 3), sn(2), st(3, 4), sn(3), st(0, 9), st(1, 9),
                       df(1, 3, {2: (0, 3), 3: (0, 4)}), df(0, 1, {1: (0, 2)}), sn(4),
                       df(3, 4, {0: (1, 9), 1: (2, 9)}), df(2, 3, {3: (0, 4)}), df(0, 4, {0: (1, 9), 1: (0, 9),
                                                                                       2: (0, 3), 3: (0, 4)})])


@case("Part 3", title="越界的 set 失败后 diff 不受影响")
def p3_failed_set(sol):
    run_steps(sol, 2, [sn(0), st(2, 5, IE), st(-1, 5, IE), st(0, 1), sn(1), df(0, 1, {0: (0, 1)}), st(7, 1, IE), sn(2),
                       df(1, 2, {}), df(0, 2, {0: (0, 1)})])


@case("Part 3", title="length = 0")
def p3_length_zero(sol):
    run_steps(sol, 0, [sn(0), sn(1), df(0, 1, {}), sn(2), df(0, 2, {}), df(1, 0, VE)])


@case("Part 3", title="随机操作序列（对照暴力解）")
def p3_random(sol):
    random_run(sol, ["set", "snap", "get", "release", "diff"], 250, 40)


DIFF_LEN = 300_000
_dr = rng("diff")
DIFF_WRITES = [[(_dr.randrange(DIFF_LEN), _dr.randint(0, 3)) for _ in range(_dr.randint(0, 4))] for _ in range(200)]
DIFF_PAIRS = [tuple(sorted(_dr.sample(range(200), 2))) for _ in range(40)]


@case("Part 3", cpu=2, title="length = 3×10⁵：diff 不能扫描整个数组")
def p3_big_length(sol):
    call = big_call(f"arr = SnapshotArray({DIFF_LEN})；200 轮“随机 set 0 到 4 个下标（值 0..3）再 snap()”，"
                    f"然后 40 次 diff(a, b)")
    arr = sol.SnapshotArray(DIFF_LEN)
    cur, snaps = {}, []                         # snaps[s]: the values written so far (unwritten indices are 0)
    for s, writes in enumerate(DIFF_WRITES):
        for i, v in writes:
            arr.set(i, v)
            cur[i] = v
        snaps.append(dict(cur))
        expect(arr.snap(), s, call, msg="snap 返回的快照 id 不对")
    for a, b in DIFF_PAIRS:
        sa, sb = snaps[a], snaps[b]
        want = {i: (sa.get(i, 0), sb.get(i, 0)) for i in set(sa) | set(sb) if sa.get(i, 0) != sb.get(i, 0)}
        expect(norm_diff(arr.diff(a, b)), want, f"{call}\n出错的调用：arr.diff({a}, {b})")


# Every index written once before snapshot 0, then a few writes per later snapshot: a diff that looks at every index
# with a history (or every set ever made) costs 10^5 per call, although only the writes between a and b matter.
MW_LEN = 100_000


def _mw_writes(s):                               # the writes made just before snap() returns s (s >= 1)
    return [((s * 7919 + k * 104729) % MW_LEN, (s + k) % 9 + 1) for k in range(s % 4)]


MW_PAIRS = [(1 + j * 37 % 197, 1 + j * 37 % 197 + 1 + j % 3) for j in range(150)]
_hist = {}                                       # index -> [(snap_id, value)], starting with the write before snap 0
for _s in range(1, 201):
    for _i, _v in _mw_writes(_s):
        _h = _hist.setdefault(_i, [(0, _i % 7 + 1)])
        if _h[-1][0] == _s:
            _h[-1] = (_s, _v)
        else:
            _h.append((_s, _v))


def _mw_value(i, s):
    return [v for sid, v in _hist[i] if sid <= s][-1]


MW_WANTS = [{i: (_mw_value(i, a), _mw_value(i, b)) for i in _hist if _mw_value(i, a) != _mw_value(i, b)}
            for a, b in MW_PAIRS]
MW_CODE = [f"arr = SnapshotArray({MW_LEN})", f"for i in range({MW_LEN}):", "    arr.set(i, i % 7 + 1)", "arr.snap()",
           "for s in range(1, 201):", "    for k in range(s % 4):",
           f"        arr.set((s * 7919 + k * 104729) % {MW_LEN}, (s + k) % 9 + 1)", "    arr.snap()"]


@case("Part 3", cpu=2, title="10⁵ 个下标都写过：diff 只看 a 到 b 之间")
def p3_many_written(sol):
    call = big_call("\n".join(MW_CODE) + f"\n# 然后 {len(MW_PAIRS)} 次 diff(a, b)，b - a 不超过 3")
    arr = sol.SnapshotArray(MW_LEN)
    for i in range(MW_LEN):
        arr.set(i, i % 7 + 1)
    expect(arr.snap(), 0, call, msg="snap 返回的快照 id 不对")
    for s in range(1, 201):
        for i, v in _mw_writes(s):
            arr.set(i, v)
        expect(arr.snap(), s, call, msg="snap 返回的快照 id 不对")
    for (a, b), want in zip(MW_PAIRS, MW_WANTS):
        expect(norm_diff(arr.diff(a, b)), want, "\n".join(MW_CODE) + f"\nprint(arr.diff({a}, {b}))")
