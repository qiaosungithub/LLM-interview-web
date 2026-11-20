import threading

from judgelib import case, expect, expect_raises, expect_true, info, rng, show_call
import ref
from ref import HORIZON, SlowInt, SlowStr, Stuck, fast_switching, finishes, hooked, jittering, run_threads, slow_leg

PAUSE_NOTE = ("（测试传入的时间是 int 的子类，比较大小时会让线程停一下。你的代码检查冲突时要拿新区间和已有预订"
              "比较时间，所以线程正好停在检查冲突的那一步。）")
NO_PAUSE = "提示：你的代码检查冲突时没有比较测试传入的时间值，测试插入的暂停点没有触发，这一项只做了普通的检查。"
ROUNDS = 200
JITTER_NOTE = ("测试传入的资源名和时间是 str、int 的子类，你的代码每次用它们查字典、比较大小时，线程都可能随机让出一小会儿，"
               "sys.setswitchinterval(1e-5)")


# ---------------------------------------------------------------- helpers

def trail(log, keep=10):
    """The construction line plus the last `keep` operations, ready to paste into 自定义运行."""
    if len(log) <= keep + 1:
        return "\n".join(log)
    return "\n".join([log[0], f"# …（省略了中间 {len(log) - keep - 1} 步）"] + log[-keep:])


def _is_exc(want):
    if isinstance(want, tuple) and want and all(isinstance(w, type) for w in want):
        return True
    return isinstance(want, type) and issubclass(want, BaseException)


def _copy(args):
    return tuple(list(a) if isinstance(a, list) else a for a in args)


def _exc_name(want):
    return " 或 ".join(w.__name__ for w in want) if isinstance(want, tuple) else want.__name__


STUCK = ("这个调用一直没有返回，在等一把永远不会被释放的锁：前面某个抛异常的调用拿了锁却没有释放（Part 2 第 6 条）。"
         "加锁请用 with，或者把 release 放进 finally。")


def play(sol, resources, steps, svc=None, log=None, guard=False):
    """Run (method, args, want) steps on one object; want is the expected value or an exception class.
    guard=True runs each call in a helper thread and reports a call that never returns (a lock left held)."""
    if svc is None:
        svc = sol.ReservationService(list(resources))
        log = [f"svc = ReservationService({list(resources)!r})"]
    for method, args, want in steps:
        log.append(show_call("svc." + method, *args))
        call = trail(log)
        fn = getattr(svc, method)
        if not guard:
            if _is_exc(want):
                expect_raises(want, fn, *_copy(args), call=call)
            else:
                expect(fn(*_copy(args)), want, call=call)
        else:
            try:
                got = finishes(lambda: fn(*_copy(args)))
            except Stuck:
                expect_true(False, STUCK, call=call)
            except Exception as e:
                if not _is_exc(want):
                    raise
                if not isinstance(e, want):
                    expect_true(False, f"应当抛出 {_exc_name(want)}，实际抛出 {type(e).__name__}: {e}", call=call)
            else:
                if _is_exc(want):
                    expect_true(False, f"应当抛出 {_exc_name(want)}，但正常返回了 {got!r}", call=call)
                expect(got, want, call=call)
        log[-1] += f"  # 抛 {_exc_name(want)}" if _is_exc(want) else f"  # {want!r}"
    return svc, log


def model_steps(r, names, n_ops, multi):
    """Random single-caller operations with the model's answers."""
    model, steps = ref.ScanModel(names), []
    for _ in range(n_ops):
        p = r.random()
        if p < 0.40:
            leg, owner = ref.random_leg(r, names), r.choice("uvw")
            steps.append(("reserve", (*leg, owner), model.reserve(*leg, owner)))
        elif p < 0.55 and multi:
            legs = [ref.random_leg(r, names) for _ in range(r.randint(1, 4))]
            owner = r.choice("xyz")
            steps.append(("reserve_all", (legs, owner), model.reserve_all(legs, owner)))
        elif p < 0.78:
            bid = r.randint(0, model.issued + 2)
            steps.append(("cancel", (bid,), model.cancel(bid)))
        else:
            res, s, e = ref.random_leg(r, names)
            steps.append(("query", (res, s, e), model.query(res, s, e)))
    for res in names:
        steps.append(("query", (res, 0, HORIZON), model.query(res, 0, HORIZON)))
    return steps


def results_line(results):
    return "，".join(f"线程 {i} 得到 {v!r}" for i, v in enumerate(results))


# ---------------------------------------------------------------- the storm (Part 2 and Part 3)

def storm(sol, names, n_threads, n_ops, multi, salt, span=120):
    """Random operations from n_threads threads at once, then checks of every invariant the statement promises.
    Booking X certainly existed from a tick taken after its reserve returned until a tick taken before its
    successful cancel started; two overlapping bookings must never have certainly existed at the same tick."""
    svc = sol.ReservationService(list(names))
    ticker, pool_lock, pool = ref.Ticker(), threading.Lock(), []
    logs = [dict(issued=[], cancelled=[], snaps=[], ids=[]) for _ in range(n_threads)]
    start = threading.Barrier(n_threads)
    p_res, p_all, p_cancel = (0.25, 0.50, 0.82) if multi else (0.45, 0.45, 0.80)

    def worker(k):
        g, log, owner = rng(f"{salt}/{k}"), logs[k], f"t{k}"
        jitter.arm(rng(f"{salt}/jitter/{k}"))
        start.wait(10)
        for _ in range(n_ops):
            p = g.random()
            if p < p_all:
                if p < p_res:
                    legs = [slow_leg(ref.random_leg(g, names, span, 8))]
                    bid = svc.reserve(*legs[0], owner)
                else:
                    legs = [slow_leg(ref.random_leg(g, names, span, 8)) for _ in range(g.randint(2, 3))]
                    bid = svc.reserve_all(list(legs), owner)
                if bid is not None:
                    log["issued"].append((bid, legs, owner, ticker.tick()))
                    log["ids"].append(bid)
                    with pool_lock:
                        pool.append(bid)
            elif p < p_cancel:
                with pool_lock:
                    victim = g.choice(pool) if pool and g.random() < 0.9 else g.choice([0, -1, 10 ** 7])
                before = ticker.tick()
                if svc.cancel(victim):
                    log["cancelled"].append((victim, before))
            else:
                res, s, e = slow_leg(ref.random_leg(g, names, span, 40))
                snap = svc.query(res, s, e)
                log["snaps"].append((res, s, e, snap, list(snap)))

    with jittering(ref.Jitter()) as jitter:
        run_threads([lambda k=k: worker(k) for k in range(n_threads)])

    how = (f"{n_threads} 个线程同时各做 {n_ops} 次随机操作（{'reserve / reserve_all' if multi else 'reserve'} / "
           f"cancel / query），资源 {list(names)}，时间在 0..{span + 8} 之间。{JITTER_NOTE}")
    issued = [x for log in logs for x in log["issued"]]
    cancelled = [x for log in logs for x in log["cancelled"]]

    # 2. ids are exactly 1..K, and one thread's successive ids increase
    ids = sorted(bid for bid, _, _, _ in issued)
    if ids != list(range(1, len(ids) + 1)):
        dup = sorted({b for b in ids if ids.count(b) > 1})[:5]
        miss = sorted(set(range(1, len(ids) + 1)) - set(ids))[:5]
        expect_true(False, f"一共成功了 {len(ids)} 次，发出的编号却不是恰好 1..{len(ids)}"
                           + (f"：编号 {dup} 发了不止一次" if dup else "") + (f"；缺少 {miss}" if miss else "")
                           + "（第 2 条）", call=how)
    for k, log in enumerate(logs):
        seq = log["ids"]
        bad = next((i for i in range(1, len(seq)) if seq[i] <= seq[i - 1]), None)
        expect_true(bad is None, f"线程 {k} 先拿到编号 {seq[bad - 1] if bad else 0}，它下一次成功的预订反而拿到更小的"
                                 f"编号 {seq[bad] if bad else 0}：后开始的调用必须排在先返回的调用后面", call=how)

    # 3. a cancel takes effect exactly once
    ids_set = set(ids)
    cancelled_ids = [bid for bid, _ in cancelled]
    twice = sorted({b for b in cancelled_ids if cancelled_ids.count(b) > 1})
    expect_true(not twice, f"编号 {twice[:5]} 被 cancel 成功（返回 True）了不止一次（第 3 条）", call=how)
    ghost = sorted(set(cancelled_ids) - ids_set)
    expect_true(not ghost, f"cancel({ghost[0] if ghost else 0}) 返回了 True，但这个编号从来没有发出过", call=how)

    # final state = issued minus cancelled
    gone = set(cancelled_ids)
    want = {n: [] for n in names}
    for bid, legs, owner, _ in issued:
        if bid not in gone:
            for res, s, e in legs:
                want[res].append((s, e, bid, owner))
    for n in names:
        got = svc.query(n, 0, HORIZON)
        expect(got, sorted(want[n]), call=f"{how}；全部结束后 svc.query({n!r}, 0, {HORIZON})",
               msg="全部线程结束后，资源上的预订应当恰好是“成功发出的减去成功取消的”")

    # 1. no double booking, also among bookings cancelled again before the end
    end_tick = dict(cancelled)
    for n in names:
        alive = sorted((s, e, bid, owner, born, end_tick.get(bid, float("inf")))
                       for bid, legs, owner, born in issued for res, s, e in legs if res == n)
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                if b[0] >= a[1]:
                    break
                if a[2] != b[2] and a[4] < b[5] and b[4] < a[5]:
                    expect_true(False, f"资源 {n!r} 上的预订 {a[:4]} 和 {b[:4]} 重叠，而且有一段时间两条都有效"
                                       "（两次 reserve 都已返回，两条都还没开始被取消）：同一时段订给了两个人（第 1 条）",
                                call=how)

    # 4. query results are snapshots of real states
    real = {(res, s, e, bid, owner) for bid, legs, owner, _ in issued for res, s, e in legs}
    for log in logs:
        for res, s, e, snap, copy in log["snaps"]:
            call = f"{how}；其中一次 svc.query({res!r}, {s}, {e})"
            expect(snap, copy, call=call, msg="query 返回的列表在返回之后又被别的操作改动了（第 4 条：要返回一份副本）")
            expect_true(copy == sorted(copy, key=lambda t: t[0]), "query 的结果没有按 start 排序", call=call, got=copy)
            for a, b in zip(copy, copy[1:]):
                expect_true(a[1] <= b[0], f"query 返回的 {a} 和 {b} 重叠：这不是任何一个真实存在过的状态（第 4 条）",
                            call=call, got=copy)
            for t in copy:
                expect_true(len(t) == 4 and (res, *t) in real and t[0] < e and s < t[1],
                            f"query 返回了 {t!r}，它不是一条和查询区间重叠的真实预订", call=call, got=copy)
    return len(ids), len(gone)


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    play(sol, ["room-a", "room-b"], [
        ("reserve", ("room-a", 9, 11, "ann"), 1),
        ("reserve", ("room-a", 11, 12, "bob"), 2),
        ("reserve", ("room-a", 10, 13, "cat"), None),
        ("reserve", ("room-b", 10, 13, "cat"), 3),
        ("query", ("room-a", 0, 24), [(9, 11, 1, "ann"), (11, 12, 2, "bob")]),
        ("query", ("room-a", 11, 12), [(11, 12, 2, "bob")]),
        ("query", ("room-a", 12, 24), []),
        ("cancel", (1,), True),
        ("cancel", (1,), False),
        ("cancel", (99,), False),
        ("reserve", ("room-a", 10, 11, "cat"), 4),
        ("query", ("room-a", 0, 24), [(10, 11, 4, "cat"), (11, 12, 2, "bob")]),
        ("reserve", ("room-c", 1, 2, "dan"), KeyError),
        ("reserve", ("room-a", 5, 5, "dan"), ValueError),
    ])


@case("Part 1", title="各种重叠与首尾相接")
def p1_overlap_rules(sol):
    play(sol, ["r"], [
        ("reserve", ("r", 10, 20, "a"), 1),
        ("reserve", ("r", 0, 10, "b"), 2),
        ("reserve", ("r", 20, 30, "c"), 3),
        ("reserve", ("r", 5, 15, "d"), None),
        ("reserve", ("r", 15, 25, "d"), None),
        ("reserve", ("r", 12, 18, "d"), None),
        ("reserve", ("r", 9, 31, "d"), None),
        ("reserve", ("r", 10, 20, "a"), None),
        ("reserve", ("r", 19, 21, "d"), None),
        ("reserve", ("r", 29, 30, "d"), None),
        ("reserve", ("r", 30, 31, "d"), 4),
        ("query", ("r", 0, 100), [(0, 10, 2, "b"), (10, 20, 1, "a"), (20, 30, 3, "c"), (30, 31, 4, "d")]),
        ("cancel", (1,), True),
        ("reserve", ("r", 12, 18, "d"), 5),
        ("reserve", ("r", 11, 19, "e"), None),
        ("reserve", ("r", 10, 12, "e"), 6),
        ("reserve", ("r", 18, 20, "e"), 7),
        ("query", ("r", 9, 21), [(0, 10, 2, "b"), (10, 12, 6, "e"), (12, 18, 5, "d"), (18, 20, 7, "e"),
                                 (20, 30, 3, "c")]),
    ])


@case("Part 1", title="编号：失败不消耗、取消后不复用")
def p1_ids(sol):
    play(sol, ["a", "b"], [
        ("cancel", (1,), False),
        ("cancel", (0,), False),
        ("reserve", ("a", 0, 5, "x"), 1),
        ("reserve", ("a", 3, 8, "x"), None),
        ("reserve", ("a", 4, 5, "y"), None),
        ("cancel", (2,), False),
        ("reserve", ("b", 3, 8, "x"), 2),
        ("cancel", (1,), True),
        ("reserve", ("a", 0, 5, "x"), 3),
        ("cancel", (1,), False),
        ("cancel", (-1,), False),
        ("cancel", (3,), True),
        ("cancel", (2,), True),
        ("cancel", (2,), False),
        ("reserve", ("a", 0, 5, "x"), 4),
        ("query", ("a", 0, 10), [(0, 5, 4, "x")]),
        ("query", ("b", 0, 10), []),
    ])


@case("Part 1", title="非法参数：抛异常且不改变状态")
def p1_errors(sol):
    svc, log = play(sol, ["a", "b"], [
        ("reserve", ("c", 1, 2, "x"), KeyError),
        ("reserve", ("a", 5, 5, "x"), ValueError),
        ("reserve", ("a", 7, 3, "x"), ValueError),
        ("query", ("c", 0, 10), KeyError),
        ("query", ("a", 5, 5), ValueError),
        ("query", ("a", 9, 1), ValueError),
        ("reserve", ("c", 5, 5, "x"), (KeyError, ValueError)),
        ("reserve", ("a", 0, HORIZON, "x"), 1),
        ("query", ("a", 0, 1), [(0, HORIZON, 1, "x")]),
        ("query", ("a", HORIZON - 1, HORIZON), [(0, HORIZON, 1, "x")]),
        ("reserve", ("a", 6, 4, "y"), ValueError),
        ("reserve", ("b", 0, 1, "y"), 2),
        ("query", ("b", 0, HORIZON), [(0, 1, 2, "y")]),
    ], guard=True)
    play(sol, [], [("reserve", ("a", 0, 1, "x"), KeyError)])


@case("Part 1", title="查询：重叠、相接、按 start 排序")
def p1_query(sol):
    play(sol, ["r", "s"], [
        ("reserve", ("r", 5, 8, "a"), 1),
        ("reserve", ("r", 0, 2, "b"), 2),
        ("reserve", ("r", 10, 15, "c"), 3),
        ("reserve", ("s", 0, 100, "d"), 4),
        ("query", ("r", 0, 100), [(0, 2, 2, "b"), (5, 8, 1, "a"), (10, 15, 3, "c")]),
        ("query", ("r", 2, 5), []),
        ("query", ("r", 1, 6), [(0, 2, 2, "b"), (5, 8, 1, "a")]),
        ("query", ("r", 7, 11), [(5, 8, 1, "a"), (10, 15, 3, "c")]),
        ("query", ("r", 6, 7), [(5, 8, 1, "a")]),
        ("query", ("r", 14, 15), [(10, 15, 3, "c")]),
        ("query", ("r", 15, 16), []),
        ("query", ("s", 50, 51), [(0, 100, 4, "d")]),
        ("query", ("s", 100, 200), []),
    ])


@case("Part 1", title="owner 只是标签")
def p1_owner(sol):
    play(sol, ["r"], [
        ("reserve", ("r", 0, 10, "ann"), 1),
        ("reserve", ("r", 5, 6, "ann"), None),
        ("reserve", ("r", 10, 20, ""), 2),
        ("reserve", ("r", 20, 30, "王小明"), 3),
        ("reserve", ("r", 30, 40, "ann"), 4),
        ("query", ("r", 0, 100), [(0, 10, 1, "ann"), (10, 20, 2, ""), (20, 30, 3, "王小明"), (30, 40, 4, "ann")]),
        ("cancel", (3,), True),
        ("reserve", ("r", 20, 30, "bob"), 5),
        ("query", ("r", 15, 25), [(10, 20, 2, ""), (20, 30, 5, "bob")]),
    ])


@case("Part 1", title="随机操作（对照逐条扫描的模型）")
def p1_random(sol):
    r = rng()
    for _ in range(120):
        names = ["r%d" % i for i in range(r.randint(1, 3))]
        play(sol, names, model_steps(r, names, 70, multi=False))


BIG_N = 20_000
BIG_ORDER = list(range(BIG_N))
rng("big").shuffle(BIG_ORDER)
BIG_PROBES = [rng("probe").randrange(BIG_N - 1) for _ in range(BIG_N)]


@case("Part 1", title="2 万条预订：冲突判断 O(log n)")
def p1_many_bookings(sol):
    svc = sol.ReservationService(["big", "other"])
    head = "svc = ReservationService(['big', 'other'])"
    for i, k in enumerate(BIG_ORDER):        # [20k, 20k+10) in random order: every one fits
        got = svc.reserve("big", 20 * k, 20 * k + 10, "u")
        if got != i + 1:
            expect(got, i + 1, call=f"{head}\n# 先按随机顺序订了 {i} 段 [20k, 20k+10)\n"
                                    f"svc.reserve('big', {20 * k}, {20 * k + 10}, 'u')")
    next_id = BIG_N + 1
    for k in BIG_PROBES:
        call = f"{head}\n# 已有 {BIG_N} 段 [20k, 20k+10)（k = 0..{BIG_N - 1}），以及若干填满的空隙\n"
        got = svc.reserve("big", 20 * k + 5, 20 * k + 15, "v")              # hits [20k, 20k+10)
        if got is not None:
            expect(got, None, call=call + f"svc.reserve('big', {20 * k + 5}, {20 * k + 15}, 'v')")
        got = svc.reserve("big", 20 * k + 10, 20 * k + 20, "w")             # the gap after it, touching both sides
        if got is not None:
            if got != next_id:
                expect(got, next_id, call=call + f"svc.reserve('big', {20 * k + 10}, {20 * k + 20}, 'w')")
            next_id += 1
        else:
            gap = svc.query("big", 20 * k + 10, 20 * k + 20)
            if not gap:
                expect(got, next_id, call=call + f"svc.reserve('big', {20 * k + 10}, {20 * k + 20}, 'w')",
                       msg="这段空隙和左右两段只是首尾相接，应当订得上")
    k = BIG_PROBES[0]
    expect(svc.query("big", 20 * k + 9, 20 * k + 21)[0][:3], (20 * k, 20 * k + 10, BIG_ORDER.index(k) + 1),
           call=f"svc.query('big', {20 * k + 9}, {20 * k + 21})[0][:3]")


# ---------------------------------------------------------------- Part 2

def same_slot(sol, n=8, wait=1.0):
    svc = sol.ReservationService(["room"])
    expect(svc.reserve("room", 777, 778, "setup"), 1, call="svc.reserve('room', 777, 778, 'setup')")
    gate = ref.MeetInCheck(n, {777, 778}, wait)
    results = [None] * n

    def worker(i):
        gate.arm()
        results[i] = svc.reserve("room", SlowInt(100), SlowInt(200), f"user{i}")

    with hooked(SlowInt, gate):
        run_threads([lambda i=i: worker(i) for i in range(n)])
    return svc, gate, results


@case("Part 2", sample=True, cpu=1, title="8 个线程同时抢同一时段")
def p2_same_slot(sol):
    svc, gate, results = same_slot(sol)
    call = ("svc = ReservationService(['room'])\nsvc.reserve('room', 777, 778, 'setup')  # 1\n"
            "# 然后 8 个线程同时调用 svc.reserve('room', 100, 200, f'user{i}')")
    if not gate.fired:
        info(NO_PAUSE)
    winners = [(i, v) for i, v in enumerate(results) if v is not None]
    expect_true(len(winners) == 1, f"8 个线程抢同一时段，应当恰好一个拿到编号，其余得到 None；实际 {len(winners)} 个"
                                   f"拿到了编号：{results_line(results)}。" + PAUSE_NOTE +
                                   "（测试等 8 个线程都停在检查这一步才放行。检查和写入在同一把锁里的话，同一时刻只有一个"
                                   "线程能在检查，凑不齐 8 个，1 秒后照常放行。）", call=call)
    i, bid = winners[0]
    expect(bid, 2, call=call, msg="唯一成功的线程应当拿到编号 2")
    expect(svc.query("room", 0, HORIZON), [(100, 200, 2, f"user{i}"), (777, 778, 1, "setup")],
           call=call + f"\nsvc.query('room', 0, {HORIZON})")


@case("Part 2", cpu=1, title="8 个线程同时第一次订同一资源")
def p2_first_use(sol):
    svc = sol.ReservationService(["room", "spare"])
    step, results, name = ref.Lockstep(8, wait=1.0), [None] * 8, SlowStr("room")

    def worker(i):
        step.arm()
        results[i] = svc.reserve(name, 100, 200, f"user{i}")

    with hooked(SlowStr, step):
        run_threads([lambda i=i: worker(i) for i in range(8)])
    call = ("svc = ReservationService(['room', 'spare'])\n"
            "# 8 个线程同时调用 svc.reserve('room', 100, 200, f'user{i}')，这是 room 第一次被用到")
    winners = [i for i, v in enumerate(results) if v is not None]
    expect_true(len(winners) == 1,
                f"8 个线程抢同一时段，应当恰好一个拿到编号；实际 {results_line(results)}。如果锁是第一次用到资源时才创建的，"
                "“没有就建一把”本身就是先检查后行动：几个线程会各建一把锁、各自进入临界区。锁要在 __init__ 里全部建好。"
                "（测试传入的资源名是 str 的子类，每次用它查字典时，线程会等其他线程也走到同一步，所以 8 个线程步调一致。）",
                call=call)
    w = winners[0]
    expect(results[w], 1, call=call, msg="唯一成功的线程应当拿到编号 1")
    expect(svc.query("room", 0, HORIZON), [(100, 200, 1, f"user{w}")], call=call + f"\nsvc.query('room', 0, {HORIZON})")


@case("Part 2", title="重叠区间抢同一时刻，重复多轮")
def p2_overlap_rounds(sol):
    r = rng()
    with jittering(ref.Jitter()) as jitter:
        for rnd in range(ROUNDS):
            svc = sol.ReservationService(["room", "other"])
            legs = [(50 - r.randint(0, 10), 51 + r.randint(0, 10)) for _ in range(6)]
            start, results = threading.Barrier(6), [None] * 6

            def worker(i):
                jitter.arm(rng(f"{rnd}/{i}"))
                start.wait(10)
                results[i] = svc.reserve(*slow_leg(("room", *legs[i])), f"u{i}")

            run_threads([lambda i=i: worker(i) for i in range(6)])
            call = (f"# 第 {rnd + 1} 轮：svc = ReservationService(['room', 'other'])，6 个线程同时调用\n"
                    + "\n".join(f"svc.reserve('room', {s}, {e}, 'u{i}')" for i, (s, e) in enumerate(legs))
                    + f"\n# {JITTER_NOTE}")
            winners = [i for i, v in enumerate(results) if v is not None]
            expect_true(len(winners) == 1 and results[winners[0]] == 1,
                        f"这 6 个区间都包含时刻 50，两两重叠，应当恰好一个拿到编号 1，其余得到 None；实际 "
                        f"{results_line(results)}", call=call)
            w = winners[0]
            expect(svc.query("room", 0, 100), [(*legs[w], 1, f"u{w}")], call=call + "\nsvc.query('room', 0, 100)")


@case("Part 2", title="编号不重复不跳号：16 线程 × 40 次")
def p2_ids(sol):
    names = ["r0", "r1", "r2", "r3"]
    svc = sol.ReservationService(names)
    n, per = 16, 40
    got = [[] for _ in range(n)]
    start = threading.Barrier(n)

    def worker(k):
        jitter.arm(rng(str(k)))
        start.wait(10)
        for j in range(per):
            leg = (names[j % 4], 1000 * k + 10 * j, 1000 * k + 10 * j + 5)
            got[k].append(svc.reserve(*slow_leg(leg), f"t{k}"))

    with jittering(ref.Jitter()) as jitter:
        run_threads([lambda k=k: worker(k) for k in range(n)])
    call = (f"svc = ReservationService({names})；16 个线程同时调用，线程 k 依次订 40 段互不重叠的时段\n"
            "svc.reserve(names[j % 4], 1000*k + 10*j, 1000*k + 10*j + 5, f't{k}')  # j = 0..39\n"
            f"# {JITTER_NOTE}")
    for k in range(n):
        expect_true(None not in got[k], f"线程 {k} 订的时段谁都没占，却得到了 None", call=call, got=got[k])
    ids = sorted(b for g in got for b in g)
    dup = sorted({b for b in ids if ids.count(b) > 1})
    expect_true(not dup, f"编号 {dup[:5]} 发给了不止一次预订（第 2 条）", call=call)
    expect(ids, list(range(1, n * per + 1)), call=call, msg=f"一共成功了 {n * per} 次，编号应当恰好是 1..{n * per}")
    for k in range(n):
        expect_true(got[k] == sorted(got[k]), f"线程 {k} 后一次调用拿到的编号反而更小：一个调用返回之后才开始的调用，"
                                              "必须排在它后面", call=call, got=got[k])
    for res_i, name in enumerate(names):
        want = sorted((1000 * k + 10 * j, 1000 * k + 10 * j + 5, got[k][j], f"t{k}")
                      for k in range(n) for j in range(per) if j % 4 == res_i)
        expect(svc.query(name, 0, HORIZON), want, call=call + f"\nsvc.query({name!r}, 0, {HORIZON})")


@case("Part 2", title="8 个线程同时取消同一编号，重复多轮")
def p2_cancel_race(sol):
    with jittering(ref.Jitter()) as jitter:
        for rnd in range(ROUNDS):
            svc = sol.ReservationService(["a"])
            bid = svc.reserve(*slow_leg(("a", 0, 10)), "u")
            start, answers = threading.Barrier(8), [None] * 8

            def worker(i):
                jitter.arm(rng(f"{rnd}/{i}"))
                start.wait(10)
                answers[i] = svc.cancel(bid)

            run_threads([lambda i=i: worker(i) for i in range(8)])
            call = (f"# 第 {rnd + 1} 轮\nsvc = ReservationService(['a'])\nsvc.reserve('a', 0, 10, 'u')  # 1\n"
                    f"# 8 个线程同时调用 svc.cancel(1)\n# {JITTER_NOTE}")
            expect_true(sum(a is True for a in answers) == 1 and all(a in (True, False) for a in answers),
                        f"应当恰好一个返回 True，其余返回 False；实际 {results_line(answers)}（第 3 条）", call=call)
            expect(svc.query("a", 0, 100), [], call=call + "\nsvc.query('a', 0, 100)")
            expect(svc.reserve("a", 0, 10, "v"), 2, call=call + "\nsvc.reserve('a', 0, 10, 'v')")


@case("Part 2", cpu=1, title="cancel 得到 False 后时段已空出")
def p2_cancel_then_reserve(sol):
    svc = sol.ReservationService(["room"])
    name = SlowStr("room")
    expect(svc.reserve(name, 10, 20, "u"), 1, call="svc.reserve('room', 10, 20, 'u')")
    hold, a_done, out = ref.HoldInside(wait=1.0), threading.Event(), {}

    def a():
        hold.arm()
        try:
            out["a"] = svc.cancel(1)
        finally:
            a_done.set()

    def b():
        out["held"] = hold.wait_entered(a_done)
        out["b_cancel"] = svc.cancel(1)
        out["b_reserve"] = svc.reserve("room", 10, 20, "B")
        hold.release.set()

    with hooked(SlowStr, hold):
        run_threads([a, b])
    call = ("svc = ReservationService(['room'])\nsvc.reserve('room', 10, 20, 'u')  # 1\n"
            "# 线程 A：svc.cancel(1)。线程 B 同时：svc.cancel(1)，返回后接着 svc.reserve('room', 10, 20, 'B')")
    note = ("（测试传入的资源名是 str 的子类，A 在 cancel 里第一次用它查字典或集合时会停住，B 在这段时间里做完它的两次"
            "调用。）") if out["held"] else ""
    if not out["held"]:
        info("提示：你的 cancel 没有用资源名查字典或集合，测试插入的暂停点没有触发，这一项只做了普通的检查。")
    expect_true([out["a"], out["b_cancel"]].count(True) == 1,
                f"两个线程同时取消编号 1，应当恰好一个返回 True；实际 A 得到 {out['a']!r}，B 得到 {out['b_cancel']!r}"
                "（第 3 条）" + note, call=call)
    expect_true(out["b_reserve"] is not None,
                f"B 的 cancel(1) 返回了 {out['b_cancel']!r}，之后才开始的 reserve 却得到 None：时段还没空出来。"
                "B 被告知“已经取消了”（或者自己取消了），时段就必须已经空出来（第 3 条）。常见原因：用 _bookings.pop "
                "决定谁取消成功，而删除区间是之后在另一把锁里做的" + note, call=call)
    expect(out["b_reserve"], 2, call=call)
    expect(svc.query("room", 0, 100), [(10, 20, 2, "B")], call=call + "\nsvc.query('room', 0, 100)")


@case("Part 2", title="cancel 后再订，重复多轮")
def p2_cancel_then_reserve_rounds(sol):
    with jittering(ref.Jitter()) as jitter:
        for rnd in range(ROUNDS):
            svc = sol.ReservationService(["room"])
            bid = svc.reserve(*slow_leg(("room", 10, 20)), "u")
            start, out = threading.Barrier(4), {}

            def canceller(i):
                jitter.arm(rng(f"{rnd}/{i}"))
                start.wait(10)
                out[i] = svc.cancel(bid)

            def booker():
                jitter.arm(rng(f"{rnd}/b"))
                start.wait(10)
                out["b"] = svc.cancel(bid)
                out["r"] = svc.reserve(*slow_leg(("room", 10, 20)), "B")

            run_threads([lambda i=i: canceller(i) for i in range(3)] + [booker])
            call = (f"# 第 {rnd + 1} 轮\nsvc = ReservationService(['room'])\nsvc.reserve('room', 10, 20, 'u')  # 1\n"
                    "# 3 个线程同时 svc.cancel(1)；第 4 个线程同时 svc.cancel(1)，返回后 svc.reserve('room', 10, 20, 'B')"
                    f"\n# {JITTER_NOTE}")
            answers = [out[0], out[1], out[2], out["b"]]
            expect_true(answers.count(True) == 1, f"4 个 cancel(1) 应当恰好一个返回 True；实际 {answers}（第 3 条）",
                        call=call)
            expect_true(out["r"] == 2, f"第 4 个线程的 cancel(1) 返回 {out['b']!r} 之后，reserve 应当拿到编号 2，实际得到 "
                                       f"{out['r']!r}（第 3 条）", call=call)


@case("Part 2", cpu=1, title="不同资源互不等待")
def p2_independent_resources(sol):
    svc = sol.ReservationService(["r0", "r1"])
    expect(svc.reserve("r0", 777, 778, "setup"), 1, call="svc.reserve('r0', 777, 778, 'setup')")
    hold, a_done, out = ref.HoldInside(wait=1.5, triggers={777, 778}), threading.Event(), {}

    def a():
        hold.arm()
        try:
            out["a"] = svc.reserve("r0", SlowInt(100), SlowInt(200), "A")
        finally:
            a_done.set()

    def b():
        out["held"] = hold.wait_entered(a_done)
        out["b"] = svc.reserve("r1", 100, 200, "B")
        hold.release.set()

    with hooked(SlowInt, hold):
        run_threads([a, b])
    call = ("svc = ReservationService(['r0', 'r1'])\nsvc.reserve('r0', 777, 778, 'setup')  # 1\n"
            "# 线程 A：svc.reserve('r0', 100, 200, 'A')，停在检查冲突的那一步\n"
            "# 线程 B 同时：svc.reserve('r1', 100, 200, 'B')")
    if not out["held"]:
        info(NO_PAUSE)
    expect_true(not out["held"] or hold.released_in_time,
                "A 停在 r0 的临界区里（正在检查冲突）时，订 r1 的 B 一直等到 A 做完才返回：订不同资源的调用互相等待了。"
                "第 5 条不允许用一把全局锁把整个 reserve 包起来，要每个资源一把锁。" + PAUSE_NOTE, call=call)
    expect(sorted([out["a"], out["b"]], key=str), [2, 3], call=call, msg="两次预订都应当成功，编号是 2 和 3")


@case("Part 2", title="query 返回快照，之后的操作不影响它")
def p2_query_snapshot(sol):
    svc, log = play(sol, ["r"], [
        ("reserve", ("r", 0, 10, "a"), 1),
        ("reserve", ("r", 20, 30, "b"), 2),
    ])
    log.append("q = svc.query('r', 0, 100)")
    q = svc.query("r", 0, 100)
    want = [(0, 10, 1, "a"), (20, 30, 2, "b")]
    expect(q, want, call=trail(log))
    log[-1] += f"  # {want!r}"
    play(sol, None, [
        ("reserve", ("r", 10, 20, "c"), 3),
        ("cancel", (1,), True),
        ("reserve", ("r", 40, 50, "d"), 4),
        ("query", ("r", 0, 100), [(10, 20, 3, "c"), (20, 30, 2, "b"), (40, 50, 4, "d")]),
    ], svc, log)
    log.append("q")
    expect(q, want, call=trail(log), msg="之前返回的 q 被后来的操作改掉了：query 要返回一份新列表（第 4 条）")


@case("Part 2", title="抛异常后不留下没释放的锁")
def p2_errors_release_locks(sol):
    svc, log = play(sol, ["r", "s"], [
        ("reserve", ("r", 5, 5, "x"), ValueError),
        ("reserve", ("r", 9, 3, "x"), ValueError),
        ("query", ("r", 4, 4), ValueError),
        ("reserve", ("nope", 0, 1, "x"), KeyError),
        ("query", ("nope", 0, 1), KeyError),
        ("query", ("s", 8, 2), ValueError),
    ], guard=True)
    play(sol, None, [
        ("reserve", ("r", 0, 10, "a"), 1),
        ("query", ("r", 0, 100), [(0, 10, 1, "a")]),
        ("query", ("s", 0, 100), []),
        ("reserve", ("s", 0, 10, "b"), 2),
        ("cancel", (1,), True),
        ("reserve", ("r", 0, 10, "c"), 3),
    ], svc, log, guard=True)


@case("Part 2", cpu=4, title="压力测试：随机操作后检查不变量")
def p2_storm(sol):
    for rnd in range(2):
        n_ids, n_gone = storm(sol, ["r0", "r1", "r2", "r3"], 12, 300, multi=False, salt=f"p2/{rnd}")
        expect_true(n_ids > 300 and n_gone > 100, f"压力测试只成功预订了 {n_ids} 次、取消了 {n_gone} 次，少得不正常")


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, cpu=1, title="题目例子")
def p3_example(sol):
    play(sol, ["gpu-0", "gpu-1", "room-a"], [
        ("reserve", ("gpu-1", 5, 8, "ann"), 1),
        ("reserve_all", ([("gpu-0", 0, 10), ("gpu-1", 0, 5)], "bob"), 2),
        ("reserve_all", ([("room-a", 0, 4), ("gpu-1", 7, 9)], "cat"), None),
        ("query", ("room-a", 0, 100), []),
        ("reserve_all", ([("room-a", 0, 4), ("room-a", 3, 6)], "cat"), None),
        ("reserve_all", ([("room-a", 0, 4), ("room-a", 4, 6)], "cat"), 3),
        ("query", ("room-a", 0, 100), [(0, 4, 3, "cat"), (4, 6, 3, "cat")]),
        ("cancel", (2,), True),
        ("query", ("gpu-0", 0, 100), []),
        ("query", ("gpu-1", 0, 100), [(5, 8, 1, "ann")]),
    ])


def opposite_orders(sol, requests):
    """One thread per request, kept in step: each takes its first lock before any asks for its second."""
    names = sorted({res for legs in requests for res, _, _ in legs})
    svc = sol.ReservationService(names)
    step, out = ref.Lockstep(len(requests), wait=1.0), [None] * len(requests)
    slow = [[(SlowStr(res), s, e) for res, s, e in legs] for legs in requests]

    def worker(i):
        step.arm()
        out[i] = svc.reserve_all(slow[i], f"t{i}")

    with hooked(SlowStr, step):
        run_threads([lambda i=i: worker(i) for i in range(len(requests))])
    call = (f"svc = ReservationService({names})\n# {len(requests)} 个线程同时调用：\n"
            + "\n".join(show_call("svc.reserve_all", legs, f"t{i}") for i, legs in enumerate(requests)))
    expect_true(sorted(out, key=str) == list(range(1, len(requests) + 1)),
                f"这些请求的时段互不重叠，应当都成功，编号是 1..{len(requests)}；实际 {results_line(out)}", call=call)
    for name in names:
        want = sorted((s, e, out[i], f"t{i}") for i, legs in enumerate(requests) for res, s, e in legs if res == name)
        expect(svc.query(name, 0, HORIZON), want, call=call + f"\nsvc.query({name!r}, 0, {HORIZON})")


@case("Part 3", sample=True, cpu=1, title="相反顺序的两个请求都要返回")
def p3_opposite_order(sol):
    opposite_orders(sol, [[("a", 0, 1), ("b", 0, 1)], [("b", 5, 6), ("a", 5, 6)]])


@case("Part 3", cpu=1, title="全有或全无：任何一段冲突就一段不订")
def p3_all_or_nothing(sol):
    play(sol, ["a", "b", "c"], [
        ("reserve", ("b", 10, 20, "x"), 1),
        ("reserve_all", ([("a", 0, 30), ("b", 15, 25), ("c", 0, 30)], "y"), None),
        ("query", ("a", 0, 100), []),
        ("query", ("c", 0, 100), []),
        ("reserve_all", ([("b", 0, 12), ("a", 0, 30)], "y"), None),
        ("reserve_all", ([("a", 0, 30), ("c", 0, 30), ("b", 19, 21)], "y"), None),
        ("query", ("a", 0, 100), []),
        ("query", ("c", 0, 100), []),
        ("reserve_all", ([("a", 0, 30), ("b", 20, 25), ("c", 0, 30)], "y"), 2),
        ("query", ("b", 0, 100), [(10, 20, 1, "x"), (20, 25, 2, "y")]),
        ("reserve", ("a", 29, 31, "z"), None),
        ("reserve", ("a", 30, 31, "z"), 3),
        ("cancel", (2,), True),
        ("query", ("a", 0, 100), [(30, 31, 3, "z")]),
        ("query", ("b", 0, 100), [(10, 20, 1, "x")]),
        ("query", ("c", 0, 100), []),
        ("cancel", (2,), False),
    ])


@case("Part 3", cpu=1, title="同一资源出现多次、段之间互相重叠")
def p3_same_resource_twice(sol):
    play(sol, ["room", "gpu"], [
        ("reserve_all", ([("room", 9, 12), ("room", 14, 17)], "a"), 1),
        ("query", ("room", 0, 100), [(9, 12, 1, "a"), (14, 17, 1, "a")]),
        ("reserve_all", ([("room", 12, 13), ("gpu", 0, 5), ("room", 13, 14)], "b"), 2),
        ("reserve_all", ([("gpu", 10, 20), ("room", 20, 24), ("gpu", 15, 25)], "c"), None),
        ("reserve_all", ([("room", 30, 40), ("gpu", 30, 31), ("room", 35, 36)], "c"), None),
        ("reserve_all", ([("room", 50, 60), ("room", 50, 60)], "c"), None),
        ("query", ("gpu", 0, 100), [(0, 5, 2, "b")]),
        ("query", ("room", 18, 100), []),
        ("reserve_all", ([("room", 30, 40), ("room", 20, 30), ("gpu", 30, 31)], "c"), 3),
        ("cancel", (1,), True),
        ("query", ("room", 0, 100), [(12, 13, 2, "b"), (13, 14, 2, "b"), (20, 30, 3, "c"), (30, 40, 3, "c")]),
        ("query", ("gpu", 0, 100), [(0, 5, 2, "b"), (30, 31, 3, "c")]),
    ])


@case("Part 3", cpu=1, title="非法请求：抛异常、一段不订、锁已释放")
def p3_errors(sol):
    svc, log = play(sol, ["a", "b"], [
        ("reserve", ("a", 0, 5, "w"), 1),
        ("reserve_all", ([], "x"), ValueError),
        ("reserve_all", ([("b", 0, 5), ("nope", 0, 1)], "x"), KeyError),
        ("reserve_all", ([("b", 0, 5), ("a", 6, 6)], "x"), ValueError),
        ("reserve_all", ([("b", 0, 5), ("a", 9, 7)], "x"), ValueError),
        ("reserve_all", ([("nope", 0, 1), ("b", 0, 5)], "x"), KeyError),
        ("reserve_all", ([("a", 0, 5), ("nope", 0, 1)], "x"), KeyError),
        ("reserve_all", ([("b", 0, 5), ("nope", 4, 4)], "x"), (KeyError, ValueError)),
    ], guard=True)
    play(sol, None, [
        ("query", ("b", 0, 100), []),
        ("query", ("a", 0, 100), [(0, 5, 1, "w")]),
        ("reserve_all", ([("a", 5, 9), ("b", 0, 5)], "y"), 2),
        ("reserve", ("b", 5, 6, "z"), 3),
        ("cancel", (2,), True),
        ("reserve_all", ([("b", 0, 5), ("a", 5, 9)], "y"), 4),
    ], svc, log, guard=True)


@case("Part 3", cpu=1, title="一个编号覆盖所有段；reserve 等同单段")
def p3_one_id(sol):
    play(sol, ["a", "b", "c"], [
        ("reserve_all", ([("a", 0, 5), ("b", 0, 5), ("c", 0, 5)], "x"), 1),
        ("reserve", ("a", 5, 10, "y"), 2),
        ("reserve_all", ([("b", 5, 10)], "z"), 3),
        ("query", ("a", 0, 100), [(0, 5, 1, "x"), (5, 10, 2, "y")]),
        ("query", ("b", 0, 100), [(0, 5, 1, "x"), (5, 10, 3, "z")]),
        ("query", ("c", 0, 100), [(0, 5, 1, "x")]),
        ("reserve_all", ([("c", 4, 6)], "z"), None),
        ("reserve", ("b", 4, 6, "z"), None),
        ("cancel", (1,), True),
        ("query", ("a", 0, 100), [(5, 10, 2, "y")]),
        ("query", ("b", 0, 100), [(5, 10, 3, "z")]),
        ("query", ("c", 0, 100), []),
        ("cancel", (1,), False),
        ("reserve_all", ([("c", 0, 5), ("a", 0, 5)], "w"), 4),
        ("reserve_all", ([("b", 0, 5)], "v"), 5),
        ("cancel", (3,), True),
        ("query", ("b", 0, 100), [(0, 5, 5, "v")]),
    ])


@case("Part 3", cpu=1, title="16 段的请求")
def p3_sixteen_legs(sol):
    names = ["r0", "r1", "r2", "r3"]
    legs = [(names[i % 4], 10 * i, 10 * i + 5) for i in range(16)]
    blocked = legs[:15] + [("r3", 20, 200)]
    svc, log = play(sol, names, [
        ("reserve", ("r3", 100, 101, "x"), 1),
        ("reserve_all", (blocked, "y"), None),
        ("reserve_all", (legs, "z"), 2),
    ])
    play(sol, None, [("query", (n, 0, HORIZON), sorted(
        [(s, e, 2, "z") for r, s, e in legs if r == n] + ([(100, 101, 1, "x")] if n == "r3" else [])))
        for n in names] + [("cancel", (2,), True)] + [("query", (n, 0, 99), []) for n in names], svc, log)


@case("Part 3", cpu=1, title="随机操作（对照逐条扫描的模型）")
def p3_random(sol):
    r = rng()
    for _ in range(120):
        names = ["r%d" % i for i in range(r.randint(1, 3))]
        play(sol, names, model_steps(r, names, 70, multi=True))


@case("Part 3", cpu=1, title="三个请求首尾相接成一圈")
def p3_three_cycle(sol):
    opposite_orders(sol, [[("a", 0, 1), ("b", 0, 1)], [("b", 5, 6), ("c", 5, 6)], [("c", 9, 10), ("a", 9, 10)]])


@case("Part 3", cpu=1, title="reserve 与 reserve_all 抢同一时段")
def p3_mixed_race(sol):
    names = ["room", "x1", "x3", "x5", "x7"]
    svc = sol.ReservationService(names)
    expect(svc.reserve("room", 777, 778, "setup"), 1, call="svc.reserve('room', 777, 778, 'setup')")
    gate, results = ref.MeetInCheck(8, {777, 778}, 1.0), [None] * 8
    calls = [f"svc.reserve('room', 100, 200, 'u{i}')" if i % 2 == 0 else
             f"svc.reserve_all([('x{i}', 0, 10), ('room', 100, 200)], 'u{i}')" for i in range(8)]

    def worker(i):
        gate.arm()
        if i % 2 == 0:
            results[i] = svc.reserve("room", SlowInt(100), SlowInt(200), f"u{i}")
        else:
            results[i] = svc.reserve_all([(f"x{i}", 0, 10), ("room", SlowInt(100), SlowInt(200))], f"u{i}")

    with hooked(SlowInt, gate):
        run_threads([lambda i=i: worker(i) for i in range(8)])
    call = (f"svc = ReservationService({names})\nsvc.reserve('room', 777, 778, 'setup')  # 1\n# 8 个线程同时调用：\n"
            + "\n".join(calls))
    if not gate.fired:
        info(NO_PAUSE)
    winners = [i for i, v in enumerate(results) if v is not None]
    expect_true(len(winners) == 1, f"8 个请求都要 room 的 [100, 200)，应当恰好一个成功；实际 {results_line(results)}。"
                                   "reserve 和 reserve_all 必须用同一套锁保护同一个资源。" + PAUSE_NOTE, call=call)
    w = winners[0]
    expect(results[w], 2, call=call, msg="唯一成功的请求应当拿到编号 2")
    expect(svc.query("room", 0, HORIZON), [(100, 200, 2, f"u{w}"), (777, 778, 1, "setup")],
           call=call + f"\nsvc.query('room', 0, {HORIZON})")
    for i in range(1, 8, 2):
        expect(svc.query(f"x{i}", 0, HORIZON), [(0, 10, 2, f"u{w}")] if i == w else [],
               call=call + f"\nsvc.query('x{i}', 0, {HORIZON})", msg="失败的 reserve_all 不能留下任何一段")


@case("Part 3", cpu=1, title="资源不相交的请求互不等待")
def p3_independent_requests(sol):
    svc = sol.ReservationService(["r0", "r1", "r2", "r3"])
    expect(svc.reserve("r0", 777, 778, "setup"), 1, call="svc.reserve('r0', 777, 778, 'setup')")
    hold, a_done, out = ref.HoldInside(wait=1.5, triggers={777, 778}), threading.Event(), {}

    def a():
        hold.arm()
        try:
            out["a"] = svc.reserve_all([("r1", 100, 200), ("r0", SlowInt(100), SlowInt(200))], "A")
        finally:
            a_done.set()

    def b():
        out["held"] = hold.wait_entered(a_done)
        out["b"] = svc.reserve_all([("r3", 100, 200), ("r2", 100, 200)], "B")
        hold.release.set()

    with hooked(SlowInt, hold):
        run_threads([a, b])
    call = ("svc = ReservationService(['r0', 'r1', 'r2', 'r3'])\nsvc.reserve('r0', 777, 778, 'setup')  # 1\n"
            "# 线程 A：svc.reserve_all([('r1', 100, 200), ('r0', 100, 200)], 'A')，停在检查冲突的那一步\n"
            "# 线程 B 同时：svc.reserve_all([('r3', 100, 200), ('r2', 100, 200)], 'B')")
    if not out["held"]:
        info(NO_PAUSE)
    expect_true(not out["held"] or hold.released_in_time,
                "A 停在临界区里（拿着 r0、r1 的锁检查冲突）时，只涉及 r2、r3 的 B 一直等到 A 做完才返回：资源没有交集的"
                "两个请求互相等待了（Part 3 里的第 5 条）。" + PAUSE_NOTE, call=call)
    expect(sorted([out["a"], out["b"]], key=str), [2, 3], call=call, msg="两个请求都应当成功，编号是 2 和 3")


@case("Part 3", cpu=1, title="订到一半时别人看不到、不被挡")
def p3_half_booking_hidden(sol):
    svc = sol.ReservationService(["a", "b"])
    expect(svc.reserve("b", 777, 778, "wall"), 1, call="svc.reserve('b', 777, 778, 'wall')")
    hold, x_done, out = ref.HoldInside(wait=1.0, triggers={777, 778}), threading.Event(), {}

    def x():
        hold.arm()
        try:
            out["x"] = svc.reserve_all([("a", 0, 100), ("b", SlowInt(700), SlowInt(800))], "X")
        finally:
            x_done.set()

    def y():
        out["held"] = hold.wait_entered(x_done)
        out["seen"] = svc.query("a", 0, 100)
        out["y"] = svc.reserve("a", 40, 60, "Y")
        hold.release.set()

    with hooked(SlowInt, hold):
        run_threads([x, y])
    call = ("svc = ReservationService(['a', 'b'])\nsvc.reserve('b', 777, 778, 'wall')  # 1\n"
            "# 线程 X：svc.reserve_all([('a', 0, 100), ('b', 700, 800)], 'X')，停在检查 b 冲突的那一步\n"
            "# 线程 Y 同时：svc.query('a', 0, 100)，然后 svc.reserve('a', 40, 60, 'Y')")
    if not out["held"]:
        info(NO_PAUSE)
    note = ("（测试传入的时间是 int 的子类，比较大小时会让线程停一下：X 检查 b 段冲突、拿 700、800 和 b 上已有的 "
            "[777, 778) 比较时停住，这时 a 段已经处理过了。）")
    expect(out["x"], None, call=call, msg="X 的 b 段和 1 号预订冲突，整个请求应当返回 None")
    expect_true(out["seen"] == [], f"X 还没决定成败时，Y 的 query('a', 0, 100) 看到了 {out['seen']!r}：这是“只订了一半”的"
                                   "中间状态，最后 X 失败了，它从来没有真实存在过（全有或全无）。" + note, call=call)
    expect_true(out["y"] == 2, f"Y 的 reserve('a', 40, 60, 'Y') 得到 {out['y']!r}，应当是 2：a 上只有一个最后失败了的请求，"
                               "Y 被它“订了一半”的状态挡住了（全有或全无）。" + note, call=call)


@case("Part 3", cpu=3, title="失败的请求不能被别人看到或挡路")
def p3_failed_request_invisible(sol):
    svc = sol.ReservationService(["a", "b"])
    expect(svc.reserve("b", 0, 100, "wall"), 1, call="svc.reserve('b', 0, 100, 'wall')")
    start, bad = threading.Barrier(7), []
    rounds = 300

    def doomed(k):
        jitter.arm(rng(f"x{k}"))
        start.wait(10)
        for _ in range(rounds):
            got = svc.reserve_all([slow_leg(("a", 0, 100)), slow_leg(("b", 0, 100))], "X")
            if got is not None:
                bad.append(f"reserve_all([('a', 0, 100), ('b', 0, 100)], 'X') 返回了 {got!r}，b 上明明一直有 1 号预订")
                return

    def booker():
        jitter.arm(rng("y"))
        start.wait(10)
        for _ in range(rounds):
            bid = svc.reserve(*slow_leg(("a", 40, 60)), "Y")
            if bid is None:
                bad.append("reserve('a', 40, 60, 'Y') 得到 None：a 上除了它自己没有别人能订成功，它被一个注定失败的 "
                           "reserve_all 的“半个预订”挡住了")
                return
            if svc.cancel(bid) is not True:
                bad.append(f"cancel({bid}) 没有返回 True")
                return

    def reader(k):
        jitter.arm(rng(f"q{k}"))
        start.wait(10)
        for _ in range(rounds):
            for item in svc.query(*slow_leg(("a", 0, 100))):
                if item[3] != "Y":
                    bad.append(f"query('a', 0, 100) 看到了 {item!r}：那是一个最后失败了的 reserve_all 订了一半的状态")
                    return

    with jittering(ref.Jitter()) as jitter:
        run_threads([lambda k=k: doomed(k) for k in range(4)] + [booker] + [lambda k=k: reader(k) for k in range(2)])
    call = ("svc = ReservationService(['a', 'b'])\nsvc.reserve('b', 0, 100, 'wall')  # 1\n"
            f"# 同时：4 个线程各 {rounds} 次 svc.reserve_all([('a', 0, 100), ('b', 0, 100)], 'X')（b 冲突，必然失败）；\n"
            f"# 1 个线程 {rounds} 次 bid = svc.reserve('a', 40, 60, 'Y'); svc.cancel(bid)；"
            f"2 个线程各 {rounds} 次 svc.query('a', 0, 100)\n# {JITTER_NOTE}")
    expect_true(not bad, (bad[0] if bad else "") + "（全有或全无：别的线程不能看到只订了一半的中间状态，也不能被它挡住）",
                call=call)


@case("Part 3", cpu=3, title="多段预订同时取消，重复多轮")
def p3_cancel_race(sol):
    legs = [("a", 0, 10), ("b", 0, 10), ("c", 0, 10)]
    with jittering(ref.Jitter()) as jitter:
        for rnd in range(ROUNDS):
            svc = sol.ReservationService(["a", "b", "c"])
            bid = svc.reserve_all([slow_leg(leg) for leg in legs], "u")
            start, out = threading.Barrier(6), {}

            def canceller(i):
                jitter.arm(rng(f"{rnd}/{i}"))
                start.wait(10)
                out[i] = svc.cancel(bid)

            def booker():
                jitter.arm(rng(f"{rnd}/b"))
                start.wait(10)
                out["b"] = svc.cancel(bid)
                out["r"] = svc.reserve_all([slow_leg(leg) for leg in [("c", 0, 10), ("a", 0, 10), ("b", 0, 10)]], "B")

            run_threads([lambda i=i: canceller(i) for i in range(5)] + [booker])
            call = (f"# 第 {rnd + 1} 轮\nsvc = ReservationService(['a', 'b', 'c'])\n"
                    f"svc.reserve_all({legs}, 'u')  # 1\n# 5 个线程同时 svc.cancel(1)；第 6 个线程同时 svc.cancel(1)，"
                    f"返回后 svc.reserve_all([('c', 0, 10), ('a', 0, 10), ('b', 0, 10)], 'B')\n# {JITTER_NOTE}")
            answers = [out[i] for i in range(5)] + [out["b"]]
            expect_true(answers.count(True) == 1, f"6 个 cancel(1) 应当恰好一个返回 True；实际 {answers}", call=call)
            expect_true(out["r"] == 2, f"第 6 个线程的 cancel(1) 返回 {out['b']!r} 之后，reserve_all 应当拿到编号 2，"
                                       f"实际得到 {out['r']!r}", call=call)


@case("Part 3", cpu=3, title="随机顺序的多资源请求，重复多轮")
def p3_lock_order_rounds(sol):
    r = rng()
    names = ["a", "b", "c", "d"]
    with jittering(ref.Jitter()) as jitter:
        for rnd in range(ROUNDS):
            svc = sol.ReservationService(names)
            reqs = [[(res, 10 * i, 10 * i + 5) for res in r.sample(names, r.randint(2, 4))] for i in range(6)]
            start, out = threading.Barrier(6), [None] * 6

            def worker(i):
                jitter.arm(rng(f"{rnd}/{i}"))
                start.wait(10)
                out[i] = svc.reserve_all([slow_leg(leg) for leg in reqs[i]], f"t{i}")

            run_threads([lambda i=i: worker(i) for i in range(6)])
            call = (f"# 第 {rnd + 1} 轮\nsvc = ReservationService({names})\n# 6 个线程同时调用：\n"
                    + "\n".join(show_call("svc.reserve_all", q, f"t{i}") for i, q in enumerate(reqs))
                    + f"\n# {JITTER_NOTE}")
            expect_true(sorted(out, key=str) == list(range(1, 7)),
                        f"6 个请求的时段互不重叠，应当都成功，编号是 1..6；实际 {results_line(out)}", call=call)


@case("Part 3", cpu=4, title="压力测试：含 reserve_all 的随机操作")
def p3_storm(sol):
    for rnd in range(2):
        n_ids, n_gone = storm(sol, ["r0", "r1", "r2", "r3"], 12, 300, multi=True, salt=f"p3/{rnd}")
        expect_true(n_ids > 300 and n_gone > 100, f"压力测试只成功预订了 {n_ids} 次、取消了 {n_gone} 次，少得不正常")
