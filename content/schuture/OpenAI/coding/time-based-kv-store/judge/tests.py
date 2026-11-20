import threading
import time

from judgelib import Mismatch, calling, case, expect, expect_true, expect_valid, info, rng
import ref
from ref import ManualClock, SlowFloat, SlowStr, hooked, run_threads

OMIT = object()          # the timestamp argument is left out
SHOW = 30                # a replay lists at most this many lines after the setup
OLD, NEW = "Part 1/2", "Part 3"
RULE = {OLD: "Part 1/2 的规则（早的时间戳插到前面，同一时间戳以后一次调用为准）",
        NEW: "Part 3 的规则（每个键的时间戳严格递增，不增大时记为 last + 1）"}
BOTH_OK = ("提示：这一项按 Part 3 的规则检查了你的 TimeMap。Part 1、Part 2 的测试接受按 Part 1/2 或按 Part 3 的规则写的 "
           "TimeMap，只要同一项测试里前后一致。")
ZERO_HINT = "注意：时间戳 0 也是显式传入的值，不能当成没传（timestamp or clock.now() 就会把它当成没传）"
WAIT = 0.6               # seconds a forced pause waits for the other threads before giving up


# ---------------------------------------------------------------- how values are shown

def lit(x):
    """repr(x), with every str in double quotes whenever that needs no extra escaping."""
    if isinstance(x, str):
        r = repr(x)
        return '"' + r[1:-1] + '"' if r[0] == "'" and '"' not in x else r
    if type(x) is list:
        return "[" + ", ".join(map(lit, x)) + "]"
    if type(x) is tuple:
        return "(" + ", ".join(map(lit, x)) + ("," if len(x) == 1 else "") + ")"
    if type(x) is dict:
        return "{" + ", ".join(f"{lit(k)}: {lit(v)}" for k, v in x.items()) + "}"
    return repr(x)


class OneOf:
    """Shown as the expected value when several results are right."""

    def __init__(self, options):
        self.options = options

    def __repr__(self):
        return " 或 ".join(lit(o) for o in self.options)


def _same(got, want):
    if want is None:
        return got is None
    try:
        return type(got) is type(want) and bool(got == want)
    except Exception:
        return False


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _ts_text(ts, kw):
    if ts is OMIT:
        return ""
    return f", timestamp={lit(ts)}" if kw else f", {lit(ts)}"


# ---------------------------------------------------------------- one store

class Session:
    """Drives one store and checks every result against the statement, keeping the replay shown on a failure.
    In Parts 1-2 a result may follow either the Part 1/2 rules or the Part 3 rules (a finished answer.py has only
    Part 3's TimeMap), as long as one store keeps to one of them: each model is dropped at its first disagreement."""

    def __init__(self, sol, part, clock=None, cls="TimeMap", why=""):
        klass = getattr(sol, cls, None)
        expect_true(isinstance(klass, type), f"answer.py 里没有 {cls} 类（类名要和题目一致）")
        self.part, self.clock, self.why = part, clock, why
        self.head, self.lines = [], []
        if clock is not None:
            self.head = ["from ref import ManualClock   # 题目给的 ManualClock（评测用的就是它）",
                         f"clock = ManualClock({lit(clock.now())})"]
            ctor = f"tm = {cls}(clock=clock)"
        else:
            ctor = f"tm = {cls}()"
        self.models = {n: ref.Model(clock, nudge=n == NEW) for n in ((OLD, NEW) if part <= 2 else (NEW,))}
        self.first = OLD if part <= 2 else NEW          # the model the statement's examples follow
        # the same models, fed as if an explicit timestamp 0 had been left out: only to recognise that mistake
        self.zmodels = {n: ref.Model(clock, nudge=n == NEW) for n in self.models}
        self.dropped, self.drop_info = [], {}           # model name -> (get line, its result, the answer's result)
        self.tm = self._call(ctor, (lambda: klass(clock=clock)) if clock is not None else klass)
        self.head.append(ctor)

    # ------------------------------------------------ replay

    def replay(self, last=None, mark="# <- 这一步"):
        lines = self.lines + ([(last, mark)] if last is not None else [])
        keep = lines[-SHOW:]
        head = list(self.head)
        if len(lines) > len(keep):
            head.append(f"# …（省略了中间 {len(lines) - len(keep)} 行，所以不能直接粘贴重现）")
        width = max([len(c) for c, n in keep if n and len(c) <= 56] or [0])
        return "\n".join(head + [(c.ljust(width) + "  " + n) if n else c for c, n in keep])

    def _call(self, line, fn, *args, **kw):
        calling(self.replay(line))
        try:
            return fn(*args, **kw)
        except Exception:
            calling(self.replay(line, "# <- 这一步抛出异常"))
            raise

    def _msg(self, *parts):
        return "；".join(p for p in (self.why,) + parts if p)

    # ------------------------------------------------ operations

    def advance(self, dt):
        self.clock.advance(dt)
        self.lines.append((f"clock.advance({lit(dt)})", f"# clock.now() == {lit(self.clock.now())}"))

    def set(self, key, value, ts=OMIT, kw=False, want=OMIT, show=True):
        line = f"tm.set({lit(key)}, {lit(value)}{_ts_text(ts, kw)})"
        if ts is OMIT:
            got = self._call(line, self.tm.set, key, value)
        elif kw:
            got = self._call(line, self.tm.set, key, value, timestamp=ts)
        else:
            got = self._call(line, self.tm.set, key, value, ts)
        at = None if ts is OMIT else ts
        req = self.clock.now() if at is None else at
        zrec = {n: m.set(key, value, self._zero_missing(at)) for n, m in self.zmodels.items()}[NEW]
        before = [t for t, _ in self.models[NEW].entries.get(key, [])] if NEW in self.models else []
        recs = {n: m.set(key, value, at) for n, m in self.models.items()}
        note = ""
        if self.part >= 3:
            rec = recs[NEW] if want is OMIT else want
            if not (_is_num(got) and got == rec):
                where = "clock.now() 的读数 " if at is None else ""
                if not before:
                    why = f"这是这个键的第一次 set，应当原样记录请求的时间戳（{where}{lit(req)}）"
                elif req > max(before):
                    why = f"请求的时间戳 {where}{lit(req)} 大于这个键上一次记录的 {lit(max(before))}，应当原样记录"
                else:
                    why = (f"请求的时间戳 {where}{lit(req)} 不大于这个键上一次记录的 {lit(max(before))}，应当记为 "
                           f"last + 1 = {lit(rec)}")
                if got is None:
                    why = "从 Part 3 起 set 要返回实际记录的时间戳；" + why
                zero = _is_num(got) and got == zrec != rec
                raise Mismatch(self._msg(why, ZERO_HINT if zero else ""), self.replay(f"print({line})"), rec, got)
            note = f"# -> {lit(rec)}"
        if show:
            self.lines.append((line, note))
        return got

    def bulk(self, code, items):
        """Many set(key, value, ts) calls, shown as one line of code that makes the same calls."""
        calling(self.replay(code, "# <- 这一段"))
        for key, value, ts in items:
            try:
                got = self.tm.set(key, value, ts)
            except Exception:
                calling(self.replay(code, f"# <- 这一段里的 tm.set({lit(key)}, {lit(value)}, {lit(ts)}) 抛出异常"))
                raise
            recs = {n: m.set(key, value, ts) for n, m in self.models.items()}
            for m in self.zmodels.values():
                m.set(key, value, self._zero_missing(ts))
            if self.part >= 3 and not (_is_num(got) and got == recs[NEW]):
                raise Mismatch(self._msg(f"这一段里的 tm.set({lit(key)}, {lit(value)}, {lit(ts)}) 应当返回 "
                                         f"{lit(recs[NEW])}"), self.replay(code, "# <- 这一段"), recs[NEW], got)
        self.lines.append((code, ""))

    def get(self, key, ts=OMIT, kw=False, want=OMIT):
        line = f"tm.get({lit(key)}{_ts_text(ts, kw)})"
        if ts is OMIT:
            got = self._call(line, self.tm.get, key)
        elif kw:
            got = self._call(line, self.tm.get, key, timestamp=ts)
        else:
            got = self._call(line, self.tm.get, key, ts)
        at = None if ts is OMIT else ts
        wants = {n: m.get(key, at) for n, m in self.models.items()}
        if want is not OMIT and self.first in wants:
            wants[self.first] = want
        ok = [n for n, w in wants.items() if _same(got, w)]
        if not ok:
            zat = self._zero_missing(at)
            # only the rule sets still in play, so a dropped one cannot trigger the hint by chance
            zero = any(_same(got, self.zmodels[n].get(key, zat)) for n in wants)
            raise Mismatch(self._get_msg(wants, got, zero, key, at), self.replay(f"print({line})"),
                           self._shown(wants), got)
        note = f"# {lit(wants[ok[0]])}"
        for n in list(self.models):
            if n not in ok:
                del self.models[n]
                self.dropped.append(n)
                self.drop_info[n] = (line, wants[n], wants[ok[0]])
                note += f"（按 {n} 的规则应为 {lit(wants[n])}）"
        self.lines.append((line, note))
        return got

    def _zero_missing(self, at):
        if not (_is_num(at) and at == 0):
            return at
        return self.clock.now() if self.clock is not None else time.time()

    def _shown(self, wants):
        vals = list(wants.values())
        if len(vals) == 2 and not _same(vals[0], vals[1]):
            return OneOf(vals)
        return vals[0]

    def _get_msg(self, wants, got, zero, key, at):
        parts = []
        if len(wants) == 2 and not _same(wants[OLD], wants[NEW]):
            parts.append(f"按 {RULE[OLD]}，应为 {lit(wants[OLD])}；如果你的 TimeMap 已按 {RULE[NEW]} 实现，应为 "
                         f"{lit(wants[NEW])}；结果两者都不是")
        elif self.part <= 2 and self.dropped:
            (n,) = wants
            d = self.dropped[0]
            line, w, g = self.drop_info[d]
            parts.append(f"前面的 {line} 得到 {lit(g)}（按 {d} 的规则应为 {lit(w)}），只符合 {RULE[n]}，"
                         f"所以这一步也按 {n} 的规则检查")
        want = next(iter(wants.values()))
        if want is None and isinstance(got, str) and got == "":
            parts.append("没有值时应当返回 None，不是空字符串")
        elif want is None and len(set(map(repr, wants.values()))) == 1:
            parts.append("这个键在查询的时间戳及以前没有记录（或从未 set 过），应当返回 None")
        if any(self._missed_exact(self.models[n], key, at, got) for n in wants):
            parts.append("记录的时间戳恰好等于查询的时间戳时也算：要找时间戳 ≤ 查询时间戳的最大那条")
        if zero:
            parts.append(ZERO_HINT)
        return self._msg(*parts)

    def _missed_exact(self, model, key, at, got):
        """Did the answer return the value just before an entry recorded exactly at the queried timestamp?"""
        q = model._at(at)
        entries = model.entries.get(key, [])
        if not any(t == q for t, _ in entries):
            return False
        best = None
        for t, v in entries:
            if t < q and (best is None or t >= best[0]):
                best = (t, v)
        return _same(got, None if best is None else best[1])

    def done(self):
        if self.part <= 2 and OLD in self.dropped:
            info(BOTH_OK)


# ---------------------------------------------------------------- random sequences

def play(t, op):
    if op[0] == "set":
        t.set(op[1], op[2], op[3], op[4])
    elif op[0] == "get":
        t.get(op[1], op[2], op[3])
    else:
        t.advance(op[1])


def _attempt(sol, part, cls, start, ops):
    """(failure kind or None, session)."""
    t = None
    try:
        t = Session(sol, part, None if start is None else ManualClock(start), cls)
        for op in ops:
            play(t, op)
    except Mismatch:
        return ("WA",), t
    except Exception as e:      # noqa: BLE001 - the answer's own error
        return ("RE", type(e).__name__), t
    return None, t


def _shrink(sol, part, cls, start, ops, bad, budget=0.6):
    """Drops operations that the failure does not need, so that the replay is short."""
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            if _attempt(sol, part, cls, start, cand)[0] == bad:
                ops, progress = cand, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops


def random_check(sol, part, rounds, gen, cls="TimeMap"):
    Session(sol, part, None, cls)          # the class exists
    r = rng()
    new_rules = False
    for _ in range(rounds):
        start, ops = gen(r)
        bad, t = _attempt(sol, part, cls, start, ops)
        if bad is None:
            new_rules = new_rules or OLD in t.dropped
            continue
        small = _shrink(sol, part, cls, start, ops, bad)
        for cand, why in ((small, "随机操作序列出错（已自动删去与出错无关的操作）"), (ops, "随机操作序列出错")):
            t = Session(sol, part, None if start is None else ManualClock(start), cls, why=why)
            for op in cand:
                play(t, op)
        raise Mismatch("随机操作序列出错，但同一序列重新运行时没有再出错（答案的行为不确定？）", None, None, None, show=False)
    if part <= 2 and new_rules:
        info(BOTH_OK)


KEYS = ["probe-1", "probe-2", "", "温度"]


def _value(r, n):
    """A value unique to the n-th set call, of a random type."""
    return [f"v{n}", n, n + 0.5, [n], {"n": n}, (n, "x")][r.randrange(6)]


def gen1(r):
    """Part 1: explicit timestamps from a small pool, so that late and equal timestamps are common."""
    pool = [-3, 0, 1, 2, 2.5, 4, 7, 10, 10.5]
    ops, n = [], 0
    for _ in range(r.randint(10, 50)):
        key = r.choice(KEYS)
        if r.random() < 0.5:
            n += 1
            ops.append(("set", key, _value(r, n), r.choice(pool), False))
        else:
            ops.append(("get", key, r.choice(pool + [-10, 3.25, 100]), False))
    return None, ops


def gen_clock(r):
    """Parts 2-3: a ManualClock that moves both ways; timestamps omitted, None, or explicit (0 included)."""
    start = r.choice([0.0, 10.0, 1000.0])
    ops, n = [], 0
    for _ in range(r.randint(10, 50)):
        x = r.random()
        key = r.choice(KEYS[:3])
        ts = r.choice([OMIT, OMIT, OMIT, None, 0, start, start + 1, start + 2.5, start - 3])
        kw = ts is not OMIT and r.random() < 0.3
        if x < 0.2:
            ops.append(("adv", r.choice([-5.0, -1.0, 0.5, 1.0, 2.0, 7.0])))
        elif x < 0.6:
            n += 1
            ops.append(("set", key, _value(r, n), ts, kw))
        else:
            ops.append(("get", key, ts, kw))
    return start, ops


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    t = Session(sol, 1)
    t.set("probe-1", 68.0, 100.0)
    t.set("probe-1", 71.5, 140.0)
    t.get("probe-1", 130.0, want=68.0)
    t.get("probe-1", 90.0, want=None)
    t.get("probe-2", 140.0, want=None)
    t.set("probe-1", 65.0, 120.0)
    t.get("probe-1", 130.0, want=65.0)
    t.set("probe-1", 70.0, 140.0)
    t.get("probe-1", 140.0, want=70.0)
    t.done()


@case("Part 1", title="恰好等于记录的时间戳也算")
def p1_exact(sol):
    t = Session(sol, 1)
    for ts, v in ((10, "a"), (20, "b"), (30, "c")):
        t.set("k", v, ts)
    for q in (10, 20, 30, 9.5, 10.5, 29.75, 30.0, 10 ** 9):
        t.get("k", q)
    t.done()


@case("Part 1", title="没有值时返回 None")
def p1_none(sol):
    t = Session(sol, 1)
    t.get("probe-1", 100)
    t.get("", 0)
    t.set("probe-1", "x", 100)
    for key, q in (("probe-1", 99.5), ("probe-1", -100), ("Probe-1", 100), ("probe-1 ", 100), ("", 100),
                   ("probe-1", 100)):
        t.get(key, q)
    t.done()


@case("Part 1", title="晚到的早时间戳插到正确位置")
def p1_out_of_order(sol):
    t = Session(sol, 1)
    for ts in (50, 10, 40, 20, 30, 5, 45, 0):
        t.set("k", f"v{ts}", ts)
        t.get("k", ts)
        t.get("k", ts + 2.5)
    for q in range(-5, 65, 5):
        t.get("k", q)
    t.done()


@case("Part 1", title="同一时间戳：后一次调用为准")
def p1_same_timestamp(sol):
    t = Session(sol, 1)
    t.set("k", "z", 5)
    t.set("k", "a", 5)
    t.get("k", 5)
    t.get("k", 6)
    t.set("k", {"x": 1}, 5)
    t.get("k", 5)
    t.set("k", 9.5, 7)
    t.set("k", 1.5, 7)
    t.get("k", 7)
    t.get("k", 5.5)
    t.set("k", "early", 3)
    t.get("k", 4)
    t.get("k", 5)
    t.set("k", [2], 5)
    t.get("k", 5)
    t.get("k", 7)
    t.done()


@case("Part 1", title='值可以是任意对象，包括 0 和 ""')
def p1_values(sol):
    t = Session(sol, 1)
    values = [0, "", False, [], {}, (), 0.0, "0", [1, [2]], {"a": [1]}]
    for i, v in enumerate(values):
        t.set("k", v, i + 1)
    for i in range(len(values)):
        t.get("k", i + 1)
        t.get("k", i + 1.5)
    t.done()


@case("Part 1", title="多个键互不影响")
def p1_keys(sol):
    t = Session(sol, 1)
    keys = ["a", "b", "", "温度/北京", "a "]
    for ts in (1, 2, 3):
        for i, k in enumerate(keys):
            if (ts + i) % 3:
                t.set(k, f"{k}@{ts}", ts)
    for k in keys + ["A", "c"]:
        t.get(k, 1)
        t.get(k, 2.5)
        t.get(k, 3)
    t.done()


@case("Part 1", title="负数、0、int 与 float")
def p1_numbers(sol):
    t = Session(sol, 1)
    for ts, v in ((-100, "neg"), (-0.5, "neg-half"), (0, "zero"), (2.75, "frac"), (3, "int"), (1e6, "big")):
        t.set("k", v, ts)
    for q in (-1000, -100, -0.75, -0.5, -0.0, 0.1, 2.75, 2.9, 3, 3.0, 999999, 1e6, 1e12):
        t.get("k", q)
    t.done()


@case("Part 1", title="一个键上 600 个值，乱序写入")
def p1_long(sol):
    t = Session(sol, 1)
    items = [("k", f"v{i * 7919 % 600}", i * 7919 % 600) for i in range(600)]
    t.bulk('for i in range(600): tm.set("k", f"v{i * 7919 % 600}", i * 7919 % 600)', items)
    r = rng()
    for q in [-1, 0, 0.5, 299, 299.5, 599, 600, 10 ** 6] + [r.uniform(-2, 602) for _ in range(40)]:
        t.get("k", q)
    t.done()


@case("Part 1", cpu=2, title="随机操作，对照逐条扫描的模型")
def p1_random(sol):
    random_check(sol, 1, 200, gen1)


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    t = Session(sol, 2, ManualClock(start=1000.0))
    t.set("probe-1", 68.0)
    t.advance(40.0)
    t.set("probe-1", 71.5)
    t.get("probe-1", want=71.5)
    t.get("probe-1", 1000.0, want=68.0)
    t.done()


@case("Part 2", title="省略 timestamp 与传 None 一样")
def p2_omitted_or_none(sol):
    t = Session(sol, 2, ManualClock(50.0))
    t.set("k", "a")
    t.advance(10.0)
    t.set("k", "b", None)
    t.advance(10.0)
    t.set("k", "c", None, kw=True)
    t.get("k", 55.0)
    t.get("k", 65.0)
    t.get("k")
    t.get("k", None)
    t.get("k", None, kw=True)
    t.advance(-15.0)
    t.get("k")
    t.get("k", None, kw=True)
    t.done()


@case("Part 2", title="显式时间戳优先于时钟")
def p2_explicit(sol):
    t = Session(sol, 2, ManualClock(1000.0))
    t.set("k", "old", 10.0)
    t.set("k", "new")
    t.get("k", 10.0)
    t.get("k", 999.5)
    t.get("k")
    t.set("f", "future", 2000.0)
    t.get("f")
    t.get("f", 2000.0)
    t.advance(1000.0)
    t.get("f")
    t.get("k")
    t.done()


@case("Part 2", title="时间戳 0 不是“没传”")
def p2_zero(sol):
    t = Session(sol, 2, ManualClock(1000.0))
    t.set("z", "zero", 0)
    t.get("z", 500.0)
    t.get("z", 0)
    t.get("z", -1)
    t.set("f", "five", 5)
    t.get("f", 0)
    t.get("f", 0.0, kw=True)
    t.set("g", "g0", 0.0, kw=True)
    t.get("g", 0.0)
    t.get("g")
    t.done()


@case("Part 2", title="每次调用都重新读时钟")
def p2_each_call(sol):
    t = Session(sol, 2, ManualClock(0.0))
    t.set("k", "a")
    t.advance(5.0)
    t.set("k", "b")
    t.advance(-2.0)
    t.get("k")
    t.advance(10.0)
    t.get("k")
    t.set("j", "x")
    t.get("j", 12.5)
    t.get("j")
    t.done()


@case("Part 2", title="时钟往回走")
def p2_backward(sol):
    t = Session(sol, 2, ManualClock(100.0))
    t.set("k", "a")
    t.advance(-30.0)
    t.set("k", "b")
    t.get("k")
    t.get("k", 100.0)
    t.get("k", 101.0)
    t.advance(30.0)
    t.set("k", "c")
    t.get("k")
    t.get("k", 85.0)
    t.done()


@case("Part 2", title="不传 clock 时用 time.time()")
def p2_real_clock(sol):
    cls = getattr(sol, "TimeMap", None)
    expect_true(isinstance(cls, type), "answer.py 里没有 TimeMap 类（类名要和题目一致）")
    time.sleep(0.05)
    lines = ["import time", "tm = TimeMap()"]
    calling("\n".join(lines + ["# <- 这一步"]))
    tm = cls()
    t0 = time.time()
    lines += ["t0 = time.time()", 'tm.set("probe-1", "now")']
    calling("\n".join(lines) + "  # <- 这一步")
    tm.set("probe-1", "now")
    t1 = time.time()
    lines.append("t1 = time.time()")
    msg = (f"没有传 clock 时，省略 timestamp 的调用应当用 time.time() 的读数（现在大约是 {t0:.0f}，即 1970 年以来的秒数）"
           "代替它")

    def check(line, fn, want):
        calling("\n".join(lines) + f"\n{line}  # <- 这一步")
        got = fn()
        expect(got, want, call="\n".join(lines) + f"\n{line}", msg=msg)
        lines.append(f"{line}  # {lit(want)}")

    check('tm.get("probe-1", t0 - 60)', lambda: tm.get("probe-1", t0 - 60), None)
    check('tm.get("probe-1", t1 + 60)', lambda: tm.get("probe-1", t1 + 60), "now")
    check('tm.get("probe-1")', lambda: tm.get("probe-1"), "now")
    calling("\n".join(lines) + '\ntm.set("probe-1", "later", t1 + 3600)  # <- 这一步')
    tm.set("probe-1", "later", t1 + 3600)
    lines.append('tm.set("probe-1", "later", t1 + 3600)')
    check('tm.get("probe-1")', lambda: tm.get("probe-1"), "now")
    check('tm.get("probe-1", t1 + 3600)', lambda: tm.get("probe-1", t1 + 3600), "later")


@case("Part 2", cpu=2, title="随机操作与时钟，对照模型")
def p2_random(sol):
    random_check(sol, 2, 200, gen_clock)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    t = Session(sol, 3, ManualClock(start=100.0))
    t.set("probe-1", 68.0, want=100.0)
    t.set("probe-1", 70.0, 100.0, want=101.0)
    t.get("probe-1", 100.0, want=68.0)
    t.get("probe-1", 101.0, want=70.0)
    t.advance(-50.0)
    t.set("probe-1", 65.0, want=102.0)
    t.get("probe-1", 102.0, want=65.0)


@case("Part 3", title="时间戳增大时原样记录")
def p3_increasing(sol):
    t = Session(sol, 3, ManualClock(10.0))
    t.set("k", "a")
    t.advance(5.0)
    t.set("k", "b")
    t.set("k", "c", 100)
    t.set("k", "d", 100.5)
    t.advance(1000.0)
    t.set("k", "e")
    for q in (12.0, 15.0, 100.25, 100.5):
        t.get("k", q)
    t.get("k")
    t.set("n", "first", -50)
    t.get("n", -50)
    t.get("n", -50.5)


@case("Part 3", title="显式时间戳不增大也要调整")
def p3_not_larger(sol):
    t = Session(sol, 3)
    for v, ts in (("a", 50), ("b", 50), ("c", 20), ("d", 51.5), ("e", 53.5), ("f", 53.5)):
        t.set("k", v, ts)
    for q in (20, 49.5, 50, 51, 51.75, 52, 53, 53.5, 54, 54.5, 1000):
        t.get("k", q)


@case("Part 3", title="只在不大于 last 时才用 last + 1")
def p3_fraction(sol):
    t = Session(sol, 3, ManualClock(100.0))
    t.set("k", "a")
    t.set("k", "b", 100.5)
    t.set("k", "c", 100.75)
    t.set("k", "d", 100.75)
    t.set("k", "e", 101)
    t.advance(2.875)
    t.set("k", "f")
    for q in (100.25, 100.5, 100.75, 101.5, 101.75, 102.75, 102.8125, 102.875):
        t.get("k", q)


@case("Part 3", title="时钟回拨：连续调整，追上后恢复")
def p3_backward(sol):
    t = Session(sol, 3, ManualClock(100.0))
    t.set("k", "a")
    t.advance(-50.0)
    t.set("k", "b")
    t.set("k", "c")
    t.set("k", "d", 40)
    t.advance(10.0)
    t.set("k", "e")
    t.advance(100.0)
    t.set("k", "f")
    t.get("k")
    for q in (100.5, 103.5, 104, 159):
        t.get("k", q)
    t.advance(-100.0)
    t.get("k")


@case("Part 3", title="每个键各自调整")
def p3_per_key(sol):
    t = Session(sol, 3, ManualClock(100.0))
    for v in ("b1", "b2", "b3", "b4"):
        t.set("busy", v)
    t.set("calm", "c1")
    t.set("calm", "c2", 101)
    t.set("busy", "b5", 101)
    t.set("new", "n1", 50)
    t.get("calm")
    t.get("calm", 101)
    t.get("busy", 101)
    t.get("busy", 104)
    t.get("new")


@case("Part 3", title="不传 clock 时返回 time.time() 读数")
def p3_real_clock(sol):
    cls = getattr(sol, "TimeMap", None)
    expect_true(isinstance(cls, type), "answer.py 里没有 TimeMap 类（类名要和题目一致）")
    lines = ["import time", "tm = TimeMap()", "t0 = time.time()", 'r1 = tm.set("probe-1", "a")']
    calling("\n".join(lines) + "  # <- 这一步")
    tm = cls()
    t0 = time.time()
    r1 = tm.set("probe-1", "a")
    t1 = time.time()
    lines.append("t1 = time.time()")
    call = "\n".join(lines + ["r1"])
    expect_valid(r1, lambda x: None if _is_num(x) and t0 <= x <= t1 else
                 f"这是这个键的第一次 set，应当返回 time.time() 的读数（在 t0 = {t0:.6f} 和 t1 = {t1:.6f} 之间）",
                 call=call, want=f"t0 ≤ r1 ≤ t1，大约 {t0:.6f}")
    lines.append(f"r1  # 约 {t0:.6f}")
    calling("\n".join(lines) + '\nr2 = tm.set("probe-1", "b", r1)  # <- 这一步')
    r2 = tm.set("probe-1", "b", r1)
    expect(r2, r1 + 1, call="\n".join(lines) + '\nr2 = tm.set("probe-1", "b", r1)\nr2',
           msg="显式时间戳 r1 不大于这个键上一次记录的 r1，应当记为 r1 + 1")
    lines += ['r2 = tm.set("probe-1", "b", r1)  # r1 + 1', 'r3 = tm.set("probe-1", "c")']
    calling("\n".join(lines) + "  # <- 这一步")
    r3 = tm.set("probe-1", "c")
    t3 = time.time()
    expect_valid(r3, lambda x: None if _is_num(x) and (x == r2 + 1 or r2 < x <= t3) else
                 "这时 time.time() 的读数还不到 r2 = r1 + 1（除非已经过了一秒），应当记为 r2 + 1",
                 call="\n".join(lines + ["r3"]), want=r2 + 1)
    lines.append("r3")
    for name, q, want in (("r1", r1, "a"), ("r2", r2, "b"), ("r3", r3, "c")):
        expect(tm.get("probe-1", q), want, call="\n".join(lines) + f'\ntm.get("probe-1", {name})')


@case("Part 3", cpu=2, title="随机操作与时钟，对照模型")
def p3_random(sol):
    random_check(sol, 3, 200, gen_clock)


# ---------------------------------------------------------------- Part 4

PAUSE_NOTE = ("（测试传入的键是 str 的子类、时间戳是 float 的子类：线程每次用键查字典、比较时间戳时，都会等其他线程也走到"
              "下一步，所以几个线程步调一致。读 last(key)、比较、写入都在同一把锁里的话，同时只有一个线程在里面，凑不齐，"
              f"{WAIT:g} 秒后照常放行。）")
NO_PAUSE = "提示：测试插入的暂停点（用键查字典、比较时间戳）一次也没有触发，这一项只做了普通的检查。"
JITTER_NOTE = "测试在比较时间戳、读时钟的地方随机让出解释器，以尝试更多的交错"


def ts_class(sol):
    cls = getattr(sol, "ThreadSafeTimeMap", None)
    expect_true(isinstance(cls, type), "answer.py 里没有 ThreadSafeTimeMap 类（类名要和题目一致）")
    return cls


def _nums(rets, call, what):
    bad = {k: v for k, v in rets.items() if not _is_num(v)}
    expect_true(not bad, f"set 应当返回实际记录的时间戳；{what}" + "，".join(f"{k} 得到 {v!r}" for k, v in bad.items()),
                call=call)


def lockstep_sets(cls, clock_start, key, setup, requests, names):
    """Threads names[i] call set(key, f"{name}", requests[i]) at the same time, held in step at every pause point.
    setup: (value, timestamp) pairs set by the main thread first. Returns (store, {name: returned timestamp}, gate)."""
    tm = cls(clock=ManualClock(SlowFloat(clock_start)))
    for v, ts in setup:
        tm.set(key, v, ts)
    step, rets = ref.Lockstep(len(names), WAIT), {}

    def worker(name, req):
        step.arm()
        rets[name] = tm.set(SlowStr(key), name) if req is None else tm.set(SlowStr(key), name, SlowFloat(req))

    with hooked(step):
        run_threads([lambda n=n, q=q: worker(n, q) for n, q in zip(names, requests)])
    return tm, rets, step


def check_lockstep(tm, rets, step, key, want_recs, call, before=None):
    if not step.points:
        info(NO_PAUSE)
    _nums(rets, call, "实际 ")
    got = sorted(rets.values())
    expect_true(got == want_recs,
                f"{len(rets)} 个线程同时 set 同一个键，记录下的时间戳应当是 {', '.join(map(lit, want_recs))}，"
                f"每个各一次（第 3 条）；实际各线程的 set 返回 " + "，".join(f"{n}: {lit(v)}" for n, v in rets.items())
                + "。" + PAUSE_NOTE, call=call)
    for name, rec in rets.items():
        expect(tm.get(key, rec), name, call=call + f"\ntm.get({lit(key)}, {lit(rec)})",
               msg=f"线程 {name} 的 set 返回 {lit(rec)}，这个时间戳上应当是它的值")
    expect(tm.get(key, want_recs[0] - 0.5), before, call=call + f"\ntm.get({lit(key)}, {lit(want_recs[0] - 0.5)})")


@case("Part 4", sample=True, cpu=1, title="题目例子：两个线程同时 set")
def p4_example(sol):
    cls = ts_class(sol)
    call = ("import threading\nfrom ref import ManualClock\ntm = ThreadSafeTimeMap(clock=ManualClock(1040.0))\n"
            'a = threading.Thread(target=tm.set, args=("probe-1", 71.5))   # 线程 A\n'
            'b = threading.Thread(target=tm.set, args=("probe-1", 69.0))   # 线程 B，和 A 同时\n'
            "a.start(); b.start(); a.join(); b.join()")
    calling(call)
    tm = cls(clock=ManualClock(SlowFloat(1040.0)))
    step, rets, key = ref.Lockstep(2, WAIT), {}, SlowStr("probe-1")

    def worker(name, value):
        step.arm()
        rets[name] = tm.set(key, value)

    with hooked(step):
        run_threads([lambda: worker("A", 71.5), lambda: worker("B", 69.0)])
    if not step.points:
        info(NO_PAUSE)
    _nums(rets, call, "实际 ")
    expect_true(sorted(rets.values()) == [1040.0, 1041.0],
                f"两个线程同时 set 同一个键，记录下的时间戳应当是 1040.0 和 1041.0，不能两次都是 1040.0（第 3 条）；实际线程 A "
                f"的 set 返回 {lit(rets['A'])}，线程 B 的返回 {lit(rets['B'])}。" + PAUSE_NOTE, call=call)
    second = "A" if rets["A"] == 1041.0 else "B"
    value = {"A": 71.5, "B": 69.0}
    expect(tm.get("probe-1", 1041.0), value[second], call=call + '\ntm.get("probe-1", 1041.0)',
           msg=f"线程 {second} 的 set 返回 1041.0，即它后生效，1041.0 上应当是它的值")
    expect(tm.get("probe-1", 1040.0), value["B" if second == "A" else "A"], call=call + '\ntm.get("probe-1", 1040.0)')


@case("Part 4", title="单线程时与 Part 3 相同")
def p4_sequential(sol):
    t = Session(sol, 4, ManualClock(start=100.0), cls="ThreadSafeTimeMap")
    t.set("probe-1", 68.0, want=100.0)
    t.set("probe-1", 70.0, 100.0, want=101.0)
    t.get("probe-1", 100.0, want=68.0)
    t.get("probe-1", 101.0, want=70.0)
    t.advance(-50.0)
    t.set("probe-1", 65.0, want=102.0)
    t.get("probe-1", 102.0, want=65.0)
    t.set("other", "x", 0)
    t.get("other", 0)
    random_check(sol, 4, 60, gen_clock, cls="ThreadSafeTimeMap")


@case("Part 4", cpu=1, title="多个线程同时第一次写同一个键")
def p4_first_touch(sol):
    cls = ts_class(sol)
    for n in (8, 3):
        names = [f"t{i}" for i in range(n)]
        call = (f"tm = ThreadSafeTimeMap(clock=ManualClock(1040.0))\n# {n} 个线程同时调用 tm.set(\"sensor\", name)，"
                f"name 是线程名 t0..t{n - 1}；这是 \"sensor\" 第一次被用到")
        calling(call)
        tm, rets, step = lockstep_sets(cls, 1040.0, "sensor", [], [None] * n, names)
        check_lockstep(tm, rets, step, "sensor", [1040.0 + i for i in range(n)], call)


@case("Part 4", cpu=1, title="多个线程同时写已有的键")
def p4_existing_key(sol):
    cls = ts_class(sol)
    for n, req in ((8, 1000.0), (5, None)):
        names = [f"t{i}" for i in range(n)]
        arg = "" if req is None else f", {lit(req)}"
        call = (f"tm = ThreadSafeTimeMap(clock=ManualClock(1040.0))\ntm.set(\"sensor\", \"setup\", 1040.0)\n"
                f"# {n} 个线程同时调用 tm.set(\"sensor\", name{arg})，name 是线程名 t0..t{n - 1}")
        calling(call)
        tm, rets, step = lockstep_sets(cls, 1040.0, "sensor", [("setup", 1040.0)], [req] * n, names)
        check_lockstep(tm, rets, step, "sensor", [1041.0 + i for i in range(n)], call, before="setup")


@case("Part 4", cpu=2, title="set 做到一半时调用 get")
def p4_get_during_set(sol):
    cls = ts_class(sol)
    held_any = False
    for k in range(1, 9):
        tm = cls(clock=ManualClock(SlowFloat(5000.0)))
        key = SlowStr("probe-1")
        tm.set(key, "old-1", 100.0)
        tm.set(key, "old-2", 200.0)
        hold, done, out = ref.HoldAt(k, 0.25), threading.Event(), {}

        def setter():
            hold.arm()
            try:
                out["ret"] = tm.set(key, "new")
            finally:
                done.set()

        def getter():
            out["held"] = hold.wait_entered(done)
            try:
                out["late"] = tm.get("probe-1", 1e9)
                out["at"] = tm.get("probe-1", 5000.0)
                out["early"] = tm.get("probe-1", 150.0)
            finally:
                hold.release.set()

        call = ("tm = ThreadSafeTimeMap(clock=ManualClock(5000.0))\ntm.set(\"probe-1\", \"old-1\", 100.0)\n"
                "tm.set(\"probe-1\", \"old-2\", 200.0)\n# 线程 A：tm.set(\"probe-1\", \"new\")，"
                f"在它第 {k} 次用键查字典或比较时间戳时停住\n# 线程 B 趁这时调用 tm.get(\"probe-1\", 1e9)、"
                "tm.get(\"probe-1\", 5000.0)、tm.get(\"probe-1\", 150.0)")
        calling(call)
        with hooked(hold):
            run_threads([setter, getter])
        held_any = held_any or out["held"]
        note = "（A 正停在 set 中间。）" if out["held"] else ""
        for name, q in (("late", "1e9"), ("at", "5000.0")):
            expect_true(_same(out[name], "old-2") or _same(out[name], "new"),
                        f"B 的 get(\"probe-1\", {q}) 和 A 的 set 同时进行，只能得到 A 之前的 \"old-2\" 或 A 写入的 \"new\"，"
                        f"实际得到 {out[name]!r}（第 1、2 条）" + note, call=call)
        expect(out["early"], "old-1", call=call, msg="B 的 get(\"probe-1\", 150.0) 应当得到 \"old-1\"" + note)
        _nums({"A": out["ret"]}, call, "")
        expect(out["ret"], 5000.0, call=call + "\n# A 的 set 返回值")
        expect(tm.get("probe-1", 5000.0), "new", call=call + "\n# 两个线程都结束后\ntm.get(\"probe-1\", 5000.0)")
    if not held_any:
        info(NO_PAUSE)


@case("Part 4", cpu=1, title="set 做到一半时另一个线程 set")
def p4_set_during_set(sol):
    cls = ts_class(sol)
    held_any = False
    for k in range(1, 9):
        tm = cls(clock=ManualClock(SlowFloat(5000.0)))
        key = SlowStr("probe-1")
        tm.set(key, "old-1", 100.0)
        tm.set(key, "old-2", 200.0)
        hold, done, out = ref.HoldAt(k, 0.25), threading.Event(), {}

        def setter_a():
            hold.arm()
            try:
                out["A"] = tm.set(key, "A")
            finally:
                done.set()

        def setter_b():
            out["held"] = hold.wait_entered(done)
            try:
                out["B"] = tm.set("probe-1", "B")
            finally:
                hold.release.set()

        call = ("tm = ThreadSafeTimeMap(clock=ManualClock(5000.0))\ntm.set(\"probe-1\", \"old-1\", 100.0)\n"
                "tm.set(\"probe-1\", \"old-2\", 200.0)\n# 线程 A：tm.set(\"probe-1\", \"A\")，"
                f"在它第 {k} 次用键查字典或比较时间戳时停住\n# 线程 B 趁这时调用 tm.set(\"probe-1\", \"B\")")
        calling(call)
        with hooked(hold):
            run_threads([setter_a, setter_b])
        held_any = held_any or out["held"]
        note = "（B 调用时 A 正停在 set 中间。）" if out["held"] else ""
        rets = {"A": out["A"], "B": out["B"]}
        _nums(rets, call, "")
        expect_true(sorted(rets.values()) == [5000.0, 5001.0],
                    "两次 set 都不传时间戳，clock.now() 都是 5000.0：先生效的记为 5000.0，后生效的记为 last + 1 = 5001.0，"
                    f"两次的返回值应当分别是这两个数；实际 A 返回 {lit(rets['A'])}，B 返回 {lit(rets['B'])}（第 1、3 条）。"
                    "如果 set 在放开锁之后才去读要返回的时间戳，就可能读到另一个线程刚写入的那个" + note, call=call)
        for name in ("A", "B"):
            expect(tm.get("probe-1", rets[name]), name, call=call + f"\n# 两个线程都结束后\ntm.get(\"probe-1\", {lit(rets[name])})",
                   msg=f"线程 {name} 的 set 返回 {lit(rets[name])}，这个时间戳上应当是它的值" + note)
    if not held_any:
        info(NO_PAUSE)


@case("Part 4", cpu=1, title="clock.now() 在生效那一刻读")
def p4_clock_in_lock(sol):
    cls = ts_class(sol)
    held_any = False
    for rnd, setup in enumerate(([("setup", 10.0)], [], [("s1", 1.0), ("s2", 2.0)])):
        clock = ref.HoldingClock(0.3)
        tm = cls(clock=clock)
        for v, ts in setup:
            tm.set("probe-1", v, ts)
        done, out = threading.Event(), {}

        def a():
            clock.arm()
            try:
                out["A"] = tm.set("probe-1", "A")
            finally:
                done.set()

        def b():
            out["held"] = clock.wait_entered(done)
            try:
                out["B"] = tm.set("probe-1", "B")
            finally:
                clock.release.set()

        call = ("# clock.now() 依次返回 1000.0、2000.0、3000.0……\ntm = ThreadSafeTimeMap(clock=clock)\n"
                + "".join(f"tm.set(\"probe-1\", {lit(v)}, {lit(ts)})\n" for v, ts in setup)
                + "# 线程 A：tm.set(\"probe-1\", \"A\")，它调用的 clock.now() 拿到读数后先停住，等线程 B 做完（最多 0.3 秒）"
                  "才返回\n# 线程 B：A 停住后调用 tm.set(\"probe-1\", \"B\")")
        calling(call)
        run_threads([a, b])
        held_any = held_any or out["held"]
        rets = {"A": out["A"], "B": out["B"]}
        _nums(rets, call, "")
        expect_true(all(v % 1000 == 0 for v in rets.values()) and rets["A"] != rets["B"],
                    f"时钟只往前走，每次 set 记录的都应当是它生效那一刻 clock.now() 的读数（1000 的倍数），不需要 last + 1；"
                    f"实际 A 返回 {lit(rets['A'])}，B 返回 {lit(rets['B'])}。A 在锁外先读了时钟，B 在这期间读到更大的读数"
                    "并先写入，A 拿着旧读数再写入时就只能被调整成 last + 1，这不是“在同一时刻读时钟并生效”（第 1 条）。"
                    "clock.now() 要在加锁之后读", call=call)
    if not held_any:
        info("提示：你的 set 没有调用 clock.now()，测试插入的暂停点没有触发，这一项只做了普通的检查。")


def check_store(tm, sets, gets, call, explicit):
    """After a stress round: every invariant the statement gives, and the final contents through get."""
    bad = [s for s in sets if not _is_num(s[4])]
    expect_true(not bad, f"set 应当返回实际记录的时间戳，实际有 {len(bad)} 次没有返回数，例如 {bad[0][4]!r}" if bad else "",
                call=call)
    rec_of = {s[5]: s[4] for s in sets}
    by_key = {}
    for s in sets:
        by_key.setdefault(s[2], []).append(s)
    for key, calls in by_key.items():
        calls.sort(key=lambda s: s[4])
        recs = [s[4] for s in calls]
        dup = sorted({a for a, b in zip(recs, recs[1:]) if a == b})
        expect_true(not dup, f"键 {lit(key)} 上记录了重复的时间戳，例如 {lit(dup[0])}：同一个键的时间戳必须严格递增（第 3 条）"
                    if dup else "", call=call)
        if explicit:
            prev = None
            for s in calls:
                want = s[3] if prev is None or s[3] > prev else prev + 1
                expect_true(s[4] == want,
                            f"键 {lit(key)} 上按时间戳排好序的 set 依次生效：前一次记录了 {lit(prev)}，这次请求 {lit(s[3])}，"
                            f"应当记为 {lit(want)}，实际返回 {lit(s[4])}（第 1、3 条）", call=call)
                prev = s[4]
        else:
            nudged = [r for r in recs if r % 1000]
            expect_true(not nudged,
                        f"时钟只往前走，set 记录的应当都是 clock.now() 的读数（1000 的倍数），实际键 {lit(key)} 上有 "
                        f"{lit(nudged[0])}：某次 set 在生效之前就读了时钟，拿着旧读数被调整成了 last + 1（第 1 条）"
                        if nudged else "", call=call)
        for tid in {s[0] for s in calls}:
            mine = sorted((s for s in calls if s[0] == tid), key=lambda s: s[1])
            for x, y in zip(mine, mine[1:]):
                expect_true(x[4] < y[4], f"线程 {tid} 先后对键 {lit(key)} 调用 set，后一次返回的时间戳 {lit(y[4])} 却不大于前一次的 "
                            f"{lit(x[4])}：同一线程里后调用的 set 一定后生效", call=call)
    for tid, j, key, q, got in gets:
        if got is None:
            continue
        ok = got in rec_of and got[2] == key and (q is None or rec_of[got] <= q)
        expect_true(ok, f"线程 {tid} 的 get({lit(key)}{'' if q is None else ', ' + lit(q)}) 得到 {got!r}，"
                    "它不是在这个键上、在不晚于查询时间的时间戳上 set 进去的值（第 2 条）", call=call)
    for key, calls in by_key.items():
        prev = None
        for s in calls:
            expect(tm.get(key, s[4]), s[5], call=call + f"\n# 所有线程结束后\ntm.get({lit(key)}, {lit(s[4])})",
                   msg=f"set({lit(key)}, {s[5]!r}) 返回了 {lit(s[4])}，这个时间戳上应当是它的值")
            expect(tm.get(key, s[4] - 0.5), prev, call=call + f"\n# 所有线程结束后\ntm.get({lit(key)}, {lit(s[4] - 0.5)})")
            prev = s[5]


def stress(sol, salt, n_threads, n_ops, explicit):
    cls = ts_class(sol)
    tm = cls(clock=ManualClock(0.0) if explicit else ref.TickingClock())
    keys = ["probe-0", "probe-1", "probe-2", "probe-3"]
    sets, gets = [], []
    start, jitter = threading.Barrier(n_threads), ref.Jitter()

    def worker(tid):
        r = rng(f"{salt}/{tid}")
        jitter.arm(rng(f"{salt}/jitter/{tid}"))
        my_sets, my_gets = [], []
        start.wait(10)
        for j in range(n_ops):
            key = r.choice(keys)
            if r.random() < 0.6:
                value = (tid, j, key)
                if explicit:
                    req = r.choice([1, j, 3 * j + r.randint(-4, 4), 6 * j])
                    my_sets.append((tid, j, key, req, tm.set(key, value, SlowFloat(req)), value))
                else:
                    my_sets.append((tid, j, key, None, tm.set(key, value), value))
            else:
                q = SlowFloat(r.randint(0, 8 * n_ops)) if explicit else None
                my_gets.append((tid, j, key, q, tm.get(key) if q is None else tm.get(key, q)))
        sets.extend(my_sets)
        gets.extend(my_gets)

    with hooked(jitter), ref.fast_switching():
        run_threads([lambda t=t: worker(t) for t in range(n_threads)])
    return tm, sets, gets


@case("Part 4", cpu=2, title="16 线程 × 50 次同一时间戳")
def p4_same_timestamp(sol):
    cls = ts_class(sol)
    n, per = 16, 50
    tm = cls(clock=ManualClock(0.0))
    got = [[] for _ in range(n)]
    start, jitter = threading.Barrier(n), ref.Jitter()

    def worker(k):
        jitter.arm(rng(f"same/{k}"))
        start.wait(10)
        for j in range(per):
            got[k].append(tm.set("probe-1", (k, j), SlowFloat(5.0)))

    call = (f"tm = ThreadSafeTimeMap(clock=ManualClock(0.0))\n# {n} 个线程同时调用，线程 k 依次调用 {per} 次\n"
            f"tm.set(\"probe-1\", (k, j), 5.0)  # j = 0..{per - 1}\n# {JITTER_NOTE}")
    calling(call)
    with hooked(jitter), ref.fast_switching():
        run_threads([lambda k=k: worker(k) for k in range(n)])
    flat = [x for g in got for x in g]
    bad = [x for x in flat if not _is_num(x)]
    expect_true(not bad, f"set 应当返回实际记录的时间戳，实际得到 {bad[0]!r}" if bad else "", call=call)
    want = [5.0 + i for i in range(n * per)]
    recs = sorted(flat)
    dup = sorted({a for a, b in zip(recs, recs[1:]) if a == b})
    expect_true(not dup, f"同一个键上记录了重复的时间戳，例如 {lit(dup[0])}（第 3 条）" if dup else "", call=call)
    expect_true(recs == want, f"一共 {n * per} 次 set，请求的都是 5.0：第一次记为 5.0，之后每次都不大于 last，记为 last + 1，"
                f"所以返回值应当恰好是 5.0、6.0……{lit(want[-1])}；实际最小 {lit(recs[0])}、最大 {lit(recs[-1])}", call=call)
    for k in range(n):
        expect_true(got[k] == sorted(got[k]), f"线程 {k} 后调用的 set 返回的时间戳反而更小", call=call, got=got[k])
    owner = {rec: (k, j) for k in range(n) for j, rec in enumerate(got[k])}
    for rec in want[::7] + [want[-1]]:
        expect(tm.get("probe-1", rec), owner[rec], call=call + f"\n# 所有线程结束后\ntm.get(\"probe-1\", {lit(rec)})")
    expect(tm.get("probe-1", 4.5), None, call=call + "\ntm.get(\"probe-1\", 4.5)")


@case("Part 4", cpu=3, title="压力测试：时间戳来自时钟")
def p4_stress_clock(sol):
    for rnd in range(4):
        tm, sets, gets = stress(sol, f"clock/{rnd}", 12, 120, explicit=False)
        call = ("# clock.now() 每次返回比上一次大的读数：1000.0、2000.0……\ntm = ThreadSafeTimeMap(clock=clock)\n"
                f"# 12 个线程同时各调用 120 次 tm.set(key, (线程号, j, key)) 或 tm.get(key)，key 在 4 个里随机选；"
                f"第 {rnd + 1} 轮\n# {JITTER_NOTE}")
        check_store(tm, sets, gets, call, explicit=False)


@case("Part 4", cpu=3, title="压力测试：显式时间戳")
def p4_stress_explicit(sol):
    for rnd in range(4):
        tm, sets, gets = stress(sol, f"explicit/{rnd}", 12, 120, explicit=True)
        call = ("tm = ThreadSafeTimeMap(clock=ManualClock(0.0))\n"
                f"# 12 个线程同时各调用 120 次 tm.set(key, (线程号, j, key), 随机的整数时间戳) 或 tm.get(key, t)，"
                f"key 在 4 个里随机选；第 {rnd + 1} 轮\n# {JITTER_NOTE}")
        check_store(tm, sets, gets, call, explicit=True)
