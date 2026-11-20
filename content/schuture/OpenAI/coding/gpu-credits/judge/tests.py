import ast
import time

from judgelib import calling, case, expect, rng
import ref

CLASS = {"Part 1": "GPUCreditLedger", "Part 2": "GPUCreditLedgerAnyOrder", "Part 3": "GPUCreditLedgerFast"}
SHOW = 14           # a replay lists at most this many calls

NOT_NONE = "这一时刻 v = 0（没有欠额，有效授予也没有剩余额度），应当返回整数 0，而不是 None"
IS_NONE = "这一时刻还有没偿还的欠额（v < 0），应当返回 None"


# ---------------------------------------------------------------- how a sequence is shown

def _call(op):
    if op[0] == "add":
        return f"ledger.add_credit({op[1]!r}, {op[2]}, {op[3]}, {op[4]})"
    if op[0] == "sub":
        return f"ledger.subtract({op[1]}, {op[2]})"
    return f"ledger.get_balance({op[1]})"


def _line(op):
    return f"print({_call(op)})" if op[0] == "ask" else _call(op)


def _replay(part, done, wants, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step."""
    lines, notes, asked = [], [], iter(wants)
    for op in done[:-1]:
        lines.append(_line(op))
        notes.append(f"# {next(asked)!r}" if op[0] == "ask" else "")
    lines.append(_line(done[-1]))
    notes.append(mark)
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = [f"ledger = {CLASS[part]}()"]
    if skipped:
        head.append(f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）")
    return "\n".join(head + [(s.ljust(width) + n).rstrip() for s, n in zip(lines, notes)])


def _script(part, calls):
    """A whole call sequence as code, results not yet known (shown if the answer hangs on it)."""
    return "\n".join([f"ledger = {CLASS[part]}()"] + [_line(op) for op in calls])


def _hint(got, want):
    if want == 0 and want is not None and got is None:
        return NOT_NONE
    if want is None and got is not None:
        return IS_NONE
    return ""


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


def _do(ledger, op):
    if op[0] == "add":
        return ledger.add_credit(op[1], op[2], op[3], op[4])
    if op[0] == "sub":
        return ledger.subtract(op[1], op[2])
    return ledger.get_balance(op[1])


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines like  get_balance(19) -> 3   # why  and  subtract(3, 16)  become (op, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        code, _, why = line.partition("#")          # no credit id in these scripts contains '#'
        call, arrow, expr = code.strip().partition("->")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(ast.literal_eval(a) for a in node.args)
        name = {"add_credit": "add", "subtract": "sub", "get_balance": "ask"}[node.func.id]
        assert (name == "ask") == bool(arrow), line
        steps.append(((name,) + args, ast.literal_eval(expr.strip()) if arrow else None, why.strip()))
    return steps


def run(sol, part, script):
    """Runs the calls on a fresh ledger of this part's class, checking each get_balance as soon as it returns.
    The expected values are written in the script; they are also checked against ref.balance."""
    ledger = getattr(sol, CLASS[part])()
    done, wants, fed = [], [], []
    for op, want, why in _parse(script):
        done.append(op)
        if op[0] == "ask":
            assert _same(want, ref.balance(fed, op[1])) and (want is None) == (ref.balance(fed, op[1]) is None), op
        calling(_replay(part, done, wants))     # shown if the answer raises or hangs in this call
        try:
            got = _do(ledger, op)
        except Exception:
            calling(_replay(part, done, wants, mark="# <- 这一步抛出异常"))
            raise
        if op[0] != "ask":
            fed.append(op)
            continue
        if not _same(got, want):
            expect(got, want, call=_replay(part, done, wants),
                   msg=why or _hint(got, want) or "这一次 get_balance 的返回值不对（按 timestamp 从小到大重放上面的调用来核对）")
        wants.append(want)


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, part, calls):
    """Replays calls on a fresh answer; None if every get_balance agrees with ref.balance."""
    ledger, fed, wants = getattr(sol, CLASS[part])(), [], []
    for k, op in enumerate(calls):
        want = ref.balance(fed, op[1]) if op[0] == "ask" else None
        try:
            got = _do(ledger, op)
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if op[0] != "ask":
            fed.append(op)
        elif not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        else:
            wants.append(want)
    return None


def _shrink(sol, part, calls, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short. Dropping calls keeps every
    rule the sequence obeyed (timestamp order for Part 1, distinct timestamps, unique credit ids)."""
    calls = calls[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(calls) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(calls) and time.process_time() < stop:
            cand = calls[:i] + calls[i + size:]
            out = _outcome(sol, part, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                calls, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return calls, bad


def random_check(sol, part, make, sequences, n_events):
    r = rng()
    for _ in range(sequences):
        ts_max = r.choice([15, 30, 60])
        calls = make(r, ref.random_events(r, r.randint(*n_events), ts_max), ts_max)
        calling("# 随机调用序列：\n" + _script(part, calls))
        bad = _outcome(sol, part, calls)
        if bad is None:
            continue
        calls, bad = _shrink(sol, part, calls, bad)
        k = bad["k"]
        if bad["exc"] is not None:
            calling(_replay(part, calls[: k + 1], bad["wants"], mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        msg = "随机调用序列中这一次 get_balance 的返回值不对（已自动删去与出错无关的调用）"
        hint = _hint(bad["got"], bad["want"])
        expect(bad["got"], bad["want"], call=_replay(part, calls[: k + 1], bad["wants"]),
               msg=msg + ("：" + hint if hint else ""))


# ---------------------------------------------------------------- big inputs, drawn once at import time

def _big_events(r, n):
    """n calls with distinct timestamps in [0, 10⁹), sorted. Long stretches where grants outweigh spending,
    then shorter ones where spending runs into debt, so the answers mix positive values, 0 and None."""
    ops = []
    for i, ts in enumerate(sorted(r.sample(range(10 ** 9), n))):
        rich = i % (n // 16 or 1) < (n // 20 or 1)
        if r.random() < 0.5:
            amount = r.randint(1, 10 ** 9 if rich else 3 * 10 ** 8)
            if r.random() < 0.02:                   # a few long grants (ts + expiration near 2·10⁹)
                span = r.randint(0, 10 ** 9)
            else:
                span = r.choice([0, r.randint(0, 10 ** 5), r.randint(0, 10 ** 6), r.randint(0, 10 ** 7)])
            ops.append(("add", f"g{i}", amount, ts, span))
        else:
            ops.append(("sub", r.randint(1, 3 * 10 ** 8 if rich else 10 ** 9), ts))
    return ops


def _between(r, events, i):
    """A query time after event i and before event i + 1 (or anywhere after the last one)."""
    lo = ref.when(events[i])
    hi = ref.when(events[i + 1]) - 1 if i + 1 < len(events) else 10 ** 9      # timestamps are at most 10⁹
    return r.randint(lo, hi)


_big = rng("big")
MID_EVENTS = _big_events(_big, 2000)                          # Part 1 and Part 2: long sequences, large amounts
MID_P1 = []
for _i, _op in enumerate(MID_EVENTS):
    MID_P1.append(_op)
    if _big.random() < 0.5:
        MID_P1.append(("ask", _between(_big, MID_EVENTS, _i)))
MID_P1_WANT = ref.sweep(MID_EVENTS, [op[1] for op in MID_P1 if op[0] == "ask"])
MID_P2 = list(MID_EVENTS)
_big.shuffle(MID_P2)
MID_P2_ASK = [_big.randint(0, 10 ** 9) for _ in range(20)]
_sorted = sorted(MID_P2_ASK)
_answer = dict(zip(_sorted, ref.sweep(MID_EVENTS, _sorted)))
MID_P2_WANT = [_answer[t] for t in MID_P2_ASK]

# Part 3 is about sequences dominated by get_balance, and the statement bounds the cost of the queries only, so
# the Part 3 timing cases have U = 5000 and Q close to 2·10⁴: an add_credit/subtract that costs O(U) in C (insort,
# or append and sort, about 0.4 s in all at U = 5000) still passes, while a query that replays the log or sums
# every live grant does not.
BIG_EVENTS = _big_events(_big, 5000)
BIG_SHUFFLED = list(BIG_EVENTS)
_big.shuffle(BIG_SHUFFLED)
BIG_ASK = sorted(_big.randint(0, 10 ** 9) for _ in range(20000))
BIG_WANT = ref.sweep(BIG_EVENTS, BIG_ASK)
BIG_MIXED = []                                                # blocks of events, each followed by queries
_i = 0
while _i < len(BIG_EVENTS):
    _block = BIG_EVENTS[_i:_i + _big.randint(1, 2)]
    _i += len(_block)
    _big.shuffle(_block)
    BIG_MIXED += _block
    for _t in sorted(_between(_big, BIG_EVENTS, _i - 1) for _ in range(_big.randint(2, 8))):
        BIG_MIXED.append(("ask", _t))
        if _big.random() < 0.1:
            BIG_MIXED.append(("ask", _t))                     # the same instant again
BIG_MIXED_WANT = ref.sweep(BIG_EVENTS, [op[1] for op in BIG_MIXED if op[0] == "ask"])
BIG_MIXED_Q = len(BIG_MIXED_WANT)


def _live_events(r, n):
    """n calls with distinct timestamps in [0, 10⁸), sorted: 95 % are grants that stay valid for 9·10⁸ to 10⁹, the
    rest subtracts too small to use many of them up, so thousands of grants are live at once."""
    ops = []
    for i, ts in enumerate(sorted(r.sample(range(10 ** 8), n))):
        if r.random() < 0.95:
            ops.append(("add", f"g{i}", r.randint(1, 10 ** 9), ts, r.randint(9 * 10 ** 8, 10 ** 9)))
        else:
            ops.append(("sub", r.randint(1, 10 ** 8), ts))
    return ops


LIVE_EVENTS = _live_events(_big, 5000)
LIVE_SHUFFLED = list(LIVE_EVENTS)
_big.shuffle(LIVE_SHUFFLED)
LIVE_ASK = sorted(_big.randint(0, 10 ** 9) for _ in range(20000))
LIVE_WANT = ref.sweep(LIVE_EVENTS, LIVE_ASK)


def run_big(sol, part, calls, wants, desc):
    """Feeds a long call sequence; on the first wrong get_balance, says which query it was."""
    head = f"ledger = {CLASS[part]}()\n# {desc}"
    calling(head)
    ledger = getattr(sol, CLASS[part])()
    asked = iter(wants)
    n_events = n_asks = 0
    for op in calls:
        try:
            got = _do(ledger, op)
        except Exception:
            calling(f"{head}\n# 已经做了 {n_events} 次 add_credit/subtract 和 {n_asks} 次查询，下一个调用抛出了异常：\n"
                    f"{_line(op)}")
            raise
        if op[0] != "ask":
            n_events += 1
            continue
        want = next(asked)
        n_asks += 1
        if not _same(got, want):
            hint = _hint(got, want)
            expect(got, want, msg="长序列中这一次 get_balance 的返回值不对" + ("：" + hint if hint else ""),
                   call=f"{head}\n# 已经做了 {n_events} 次 add_credit/subtract 和 {n_asks - 1} 次查询，"
                        f"这是第 {n_asks} 次查询：\n{_line(op)}")


PART1_EXAMPLE = """
    add_credit('r1', 6, 10, 20)
    get_balance(10) -> 6        # r1 的有效区间正好从 10 开始
    add_credit('r2', 5, 14, 6)
    subtract(3, 16)
    add_credit('r3', 4, 18, 8)
    subtract(9, 19)
    get_balance(19) -> 3        # 按过期时间从早到晚扣：r2 扣 2，r3 扣 4，r1 扣 3，只剩 r1 的 3
    get_balance(20) -> 3        # 20 是 r2 区间 [14, 20] 的最后一刻，但 r2 早已扣空
    add_credit('r4', 2, 24, 4)
    get_balance(28) -> 5        # r1（3）和 r4（2）都还有效：区间两端都包含在内
    get_balance(29) -> 3        # r4 的区间 [24, 28] 刚刚结束
    get_balance(31) -> 0        # 没有任何有效授予：答案是整数 0，而不是 None
"""

PART2_EXAMPLE = """
    subtract(5, 42)
    get_balance(8) -> 0             # 还没有 timestamp <= 8 的调用
    add_credit('s1', 7, 35, 25)
    get_balance(35) -> 7            # timestamp 为 42 的 subtract 对这次查询来说还没有发生
    get_balance(42) -> 2            # 按 timestamp 的顺序重放：7 - 5 = 2
    subtract(10, 48)
    get_balance(48) -> None         # 48 时刻只有 2 个有效额度，其余 8 记为欠额：v = -8
    add_credit('s2', 5, 52, 10)
    get_balance(52) -> None         # s2 的 5 全部用来还欠额：v = 0 - 3
    add_credit('s3', 3, 57, 10)
    get_balance(57) -> 0            # 正好还清欠额：返回整数 0，而不是 None
    get_balance(70) -> 0            # s2、s3 过期后，它们还掉的欠额不会重新出现
"""


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    run(sol, "Part 1", PART1_EXAMPLE)


@case("Part 1", title="没有有效授予时返回 0，不是 None")
def p1_zero_not_none(sol):
    run(sol, "Part 1", """
        get_balance(0) -> 0             # 新账本：没有授予，也没有欠额
        add_credit('a', 4, 5, 3)
        get_balance(8) -> 4             # 8 是 a 的区间 [5, 8] 的最后一刻，仍然有效
        get_balance(9) -> 0             # a 的区间 [5, 8] 已经结束
        get_balance(1000000000) -> 0
    """)


@case("Part 1", title="区间两端都有效，expiration 可以是 0")
def p1_inclusive_window(sol):
    run(sol, "Part 1", """
        add_credit('z', 5, 7, 0)
        get_balance(7) -> 5             # expiration = 0：只在 7 这一刻有效
        get_balance(8) -> 0
        add_credit('a', 5, 10, 5)
        subtract(3, 15)                 # 15 是 a 的区间 [10, 15] 的最后一刻，a 仍然有效，可以扣
        get_balance(15) -> 2            # 不能记成欠额
        get_balance(16) -> 0
    """)


@case("Part 1", title="先扣最快过期的，而不是最早发放的")
def p1_soonest_expiry_first(sol):
    run(sol, "Part 1", """
        add_credit('long', 10, 0, 100)
        add_credit('short', 10, 5, 10)
        subtract(10, 6)                 # short 在 15 过期，比 long（100）早：扣 short
        get_balance(6) -> 10
        get_balance(15) -> 10
        get_balance(16) -> 10           # short 过期了，long 的 10 一点没动
        get_balance(101) -> 0
    """)


@case("Part 1", title="一次扣款跨越多笔授予")
def p1_drain_several(sol):
    run(sol, "Part 1", """
        add_credit('c', 4, 0, 30)
        add_credit('a', 2, 1, 10)
        add_credit('b', 3, 2, 18)
        subtract(6, 5)                  # a 扣 2，b 扣 3，c 扣 1
        get_balance(5) -> 3
        get_balance(11) -> 3
        get_balance(20) -> 3
        get_balance(30) -> 3            # 剩下的 3 都在 c 上
        get_balance(31) -> 0
    """)


@case("Part 1", title="两笔授予同时过期")
def p1_same_expiry(sol):
    run(sol, "Part 1", """
        add_credit('a', 3, 0, 10)
        add_credit('b', 4, 2, 8)        # a 和 b 都在 10 过期
        subtract(5, 5)
        get_balance(5) -> 2
        get_balance(10) -> 2            # 10 是两者区间的最后一刻，仍然有效
        get_balance(11) -> 0
    """)


@case("Part 1", title="超支记为欠额，之后的授予先还欠额")
def p1_debt_then_repay(sol):
    run(sol, "Part 1", """
        add_credit('a', 10, 10, 30)
        subtract(100, 20)               # 只有 10 可扣，其余 90 记为欠额
        get_balance(20) -> None
        get_balance(41) -> None         # a 过期了，欠额还在
        get_balance(49) -> None
        add_credit('b', 200, 50, 10)    # 先还 90，剩 110 是 b 的额度
        get_balance(50) -> 110          # 20 时刻 a 的 10 已经扣光，欠额是 90 而不是 100
        get_balance(60) -> 110
        get_balance(61) -> 0            # b 过期了；它还掉的 90 欠额不会重新出现
    """)


@case("Part 1", title="没有任何授予时扣款也记为欠额")
def p1_debt_from_nothing(sol):
    run(sol, "Part 1", """
        get_balance(14) -> 0
        subtract(4, 15)
        get_balance(15) -> None
        get_balance(19) -> None
        add_credit('c', 6, 20, 5)       # 先还 4，剩 2
        get_balance(20) -> 2
        get_balance(25) -> 2
        get_balance(26) -> 0            # c 过期了；它还掉的 4 欠额不会重新出现
    """)


@case("Part 1", title="还掉的欠额在授予过期后不会重新出现")
def p1_repaid_debt_stays_repaid(sol):
    run(sol, "Part 1", """
        subtract(8, 10)
        add_credit('p', 5, 12, 3)       # 5 全部还欠额：8 -> 3
        get_balance(12) -> None
        add_credit('q', 3, 13, 2)       # 正好还清
        get_balance(13) -> 0
        get_balance(15) -> 0
        get_balance(16) -> 0            # p、q 都过期了，欠额不会回来
        add_credit('s', 4, 20, 5)
        subtract(1, 21)
        get_balance(21) -> 3
    """)


@case("Part 1", title="扣款时只扣当时有效的授予")
def p1_skip_expired_on_subtract(sol):
    run(sol, "Part 1", """
        add_credit('old', 10, 0, 5)
        add_credit('new', 3, 8, 100)
        subtract(5, 10)                 # old 在 5 就过期了，不能再扣：new 扣 3，欠 2
        get_balance(10) -> None
        add_credit('more', 4, 11, 100)  # 先还 2，剩 2
        get_balance(11) -> 2
    """)


@case("Part 1", title="数额超过 32 位整数")
def p1_big_amounts(sol):
    run(sol, "Part 1", """
        add_credit('b0', 1000000000, 0, 100)
        add_credit('b1', 1000000000, 1, 100)
        add_credit('b2', 1000000000, 2, 100)
        add_credit('b3', 1000000000, 3, 100)
        add_credit('b4', 1000000000, 4, 100)
        get_balance(4) -> 5000000000
        subtract(1000000000, 5)
        subtract(1000000000, 6)
        subtract(1000000000, 7)
        subtract(1000000000, 8)
        subtract(999999999, 9)
        get_balance(9) -> 1
        subtract(1000000000, 10)
        get_balance(10) -> None
        add_credit('b5', 1000000000, 1000000000, 1000000000)
        get_balance(1000000000) -> 1
    """)


@case("Part 1", title="随机调用序列（对照暴力重放）")
def p1_random(sol):
    random_check(sol, "Part 1", ref.in_order_calls, sequences=300, n_events=(1, 12))


@case("Part 1", title="2000 次调用，大数额")
def p1_long(sol):
    run_big(sol, "Part 1", MID_P1, MID_P1_WANT, "2000 次按时间顺序的 add_credit/subtract，数额最大 10⁹，中间穿插查询")


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    run(sol, "Part 2", PART2_EXAMPLE)


@case("Part 2", title="Part 1 的例子（按时间顺序到达）")
def p2_part1_example(sol):
    run(sol, "Part 2", PART1_EXAMPLE)


@case("Part 2", title="查询只重放 timestamp 不超过它的调用")
def p2_only_past_events(sol):
    run(sol, "Part 2", """
        subtract(4, 50)
        add_credit('a', 10, 0, 100)
        get_balance(49) -> 10           # 50 时刻的扣款还没有发生
        get_balance(50) -> 6
        get_balance(0) -> 10
        get_balance(100) -> 6
        get_balance(101) -> 0
    """)


@case("Part 2", title="晚到的早期授予给更早调用的扣款供额")
def p2_late_grant_funds_subtract(sol):
    run(sol, "Part 2", """
        subtract(10, 30)
        add_credit('late', 4, 40, 10)
        get_balance(45) -> None         # 30 时刻欠 10，late 还掉 4
        get_balance(51) -> None
        add_credit('early', 20, 20, 100)
        get_balance(25) -> 20
        get_balance(30) -> 10           # 按时间重放：30 时刻从 early 扣 10，没有欠额
        get_balance(45) -> 14           # late 的 4 是它自己的额度
        get_balance(51) -> 10           # late 过期了，只剩 early 的 10（30 时刻的扣款由 early 支付，没有欠额）
    """)


@case("Part 2", title="插入更早的事件后，同一时刻的答案要变")
def p2_answer_changes(sol):
    run(sol, "Part 2", """
        add_credit('a', 10, 0, 100)
        get_balance(50) -> 10
        subtract(3, 20)
        get_balance(50) -> 7            # 20 时刻从 a 扣了 3
        add_credit('b', 5, 10, 5)       # [10, 15]：20 时刻的扣款扣不到它
        get_balance(50) -> 7
        get_balance(12) -> 15           # a 的 10 加 b 的 5
        subtract(1, 5)
        get_balance(50) -> 6            # 5 时刻又从 a 扣了 1
        get_balance(12) -> 14
    """)


@case("Part 2", title="随机：任意调用顺序，查询穿插、重复")
def p2_random(sol):
    random_check(sol, "Part 2", ref.any_order_calls, sequences=200, n_events=(1, 12))


@case("Part 2", title="2000 次乱序调用，大数额")
def p2_long(sol):
    run_big(sol, "Part 2", MID_P2 + [("ask", t) for t in MID_P2_ASK], MID_P2_WANT,
            "2000 次乱序的 add_credit/subtract，数额最大 10⁹，之后查询 20 个随机时刻")


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="Part 2 的例子（行为相同）")
def p3_example(sol):
    run(sol, "Part 3", PART2_EXAMPLE)


@case("Part 3", title="Part 1 的例子（按时间顺序到达）")
def p3_part1_example(sol):
    run(sol, "Part 3", PART1_EXAMPLE)


@case("Part 3", title="迟到的事件：之前的结果作废")
def p3_late_event(sol):
    run(sol, "Part 3", """
        add_credit('a', 10, 10, 100)
        add_credit('b', 5, 20, 100)
        get_balance(50) -> 15
        subtract(4, 15)                 # 落在已经查询过的 50 之前
        get_balance(60) -> 11           # 15 时刻从 a 扣了 4：a 6，b 5
        get_balance(50) -> 11           # 同一时刻再问一次，答案已经变了
        add_credit('c', 2, 55, 100)     # 又落在查询过的 60 之前
        get_balance(60) -> 13
        get_balance(50) -> 11           # c 在 55 才发放，50 时刻还没有它
        get_balance(55) -> 13
    """)


@case("Part 3", title="迟到的事件按它自己的时刻生效")
def p3_late_event_own_time(sol):
    run(sol, "Part 3", """
        add_credit('a', 5, 0, 10)       # 在 [0, 10] 内有效
        add_credit('b', 5, 1, 100)      # 在 [1, 101] 内有效
        get_balance(50) -> 5            # a 已经过期，只剩 b
        subtract(4, 5)                  # 迟到的扣款，在 5 时刻生效：那时 a 有效且最快过期，扣 a
        get_balance(50) -> 5            # 扣掉的是 a 的额度，a 本来就会过期；b 的 5 没动
        subtract(9, 40)                 # 又一笔迟到的扣款：40 时刻只有 b 的 5 有效，欠 4
        get_balance(50) -> None
        add_credit('c', 6, 20, 15)      # 迟到的授予，在 [20, 35] 内有效：20 时刻没有欠额可还，40 时刻它已经过期
        get_balance(50) -> None         # 欠额还是 4
        get_balance(30) -> 11           # 往回查：b 5 加 c 6
    """)


@case("Part 3", title="事件正好落在查询过的时刻上")
def p3_event_at_queried_time(sol):
    run(sol, "Part 3", """
        add_credit('a', 10, 10, 100)
        get_balance(30) -> 10
        subtract(4, 30)                 # 时间戳正好等于已经查询过的 30
        get_balance(30) -> 6            # 再问 30：这笔扣款的 timestamp 不超过 30，要算进去
        get_balance(40) -> 6
        add_credit('b', 3, 40, 10)      # 又正好落在查询过的 40 上
        get_balance(40) -> 9
        get_balance(51) -> 6            # b 的区间 [40, 50] 结束了
    """)


@case("Part 3", title="查询时间往回走")
def p3_query_back_in_time(sol):
    run(sol, "Part 3", """
        add_credit('a', 10, 10, 100)
        subtract(4, 30)
        get_balance(40) -> 6
        get_balance(20) -> 10           # 比上一次查询早：30 时刻的扣款还没有发生
        get_balance(5) -> 0
        get_balance(111) -> 0
        get_balance(40) -> 6
        get_balance(110) -> 6
    """)


@case("Part 3", title="随机：任意调用顺序，查询穿插、重复")
def p3_random_any_order(sol):
    random_check(sol, "Part 3", ref.any_order_calls, sequences=200, n_events=(1, 12))


@case("Part 3", title="随机：事件先到，查询非降序")
def p3_random_ahead(sol):
    random_check(sol, "Part 3", ref.ahead_calls, sequences=200, n_events=(1, 16))


@case("Part 3", cpu=2, title="5000 次乱序调用后 2×10⁴ 次非降序查询")
def p3_many_queries(sol):
    run_big(sol, "Part 3", BIG_SHUFFLED + [("ask", t) for t in BIG_ASK], BIG_WANT,
            "先按随机顺序做 5000 次 add_credit/subtract，再按非降序查询 20000 个时刻（不能每次查询都重放）")


@case("Part 3", cpu=2, title="事件与查询交替，事件总在查询之前")
def p3_interleaved(sol):
    run_big(sol, "Part 3", BIG_MIXED, BIG_MIXED_WANT,
            f"5000 次 add_credit/subtract 分成小块到达（块内乱序），每块之后查询几个不早于它的时刻，"
            f"共 {BIG_MIXED_Q} 次查询，查询非降序（不能每来一个事件就从头重放）")


@case("Part 3", cpu=1.5, title="数千笔授予同时有效：查询要 O(1)")
def p3_many_live_grants(sol):
    run_big(sol, "Part 3", LIVE_SHUFFLED + [("ask", t) for t in LIVE_ASK], LIVE_WANT,
            "乱序做 5000 次 add_credit/subtract（大多是长期有效的授予，同时有效的有几千笔），再按非降序查询 20000 个"
            "时刻（每次查询都把有效授予逐笔加一遍就太慢了）")
