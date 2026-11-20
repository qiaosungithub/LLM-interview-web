import threading
import time

from judgelib import Mismatch, calling, case, expect, expect_true, info, rng
import ref
from ref import CYCLE, LONG, NF, lit, normalize

SHOW = 16           # a Level 4 replay lists at most this many steps
ENTER = 2.0         # seconds for the first caller to reach the paused step
ARRIVE = 0.2        # seconds the other callers get to arrive while the first one is paused
WAIT = 4.0          # seconds for every call to return once nothing is paused any more
FREE = 1.5          # seconds a call for another name may take while one name is paused

TOUCH_NOTE = ("（评测传入的每个 CNAMERecord.target 都是 str 的子类，能看到你的代码有没有读取它，以此判断一次调用"
              "有没有沿着链去解析。）")
NO_TOUCH = ("提示：你的 resolve 没有读取评测传入的 CNAME target（大概在构造 Resolver 时就预先处理好了），评测看不出"
            "哪次调用命中了缓存，这一项只检查了返回值。")
NO_GATE = ("提示：你的 resolve 没有读取评测传入的 CNAME target（大概在构造 Resolver 时就预先处理好了），评测没法让"
           "解析停在半路，也看不出哪些调用真的做了解析，这一项只检查了返回值。")
MEMO = ("提示：你的 resolve 只在第一次走到一条 CNAME 记录时读取它的 target，之后用的是自己存下的结果，评测看不出哪些"
        "调用重新解析了，“应当重新解析”的检查跳过了，只检查了命中缓存和返回值。")
PAUSED = "评测让它停在读 CNAME target 的那一步（解析进行到一半）"
ERRORS = ("DNSResolutionError", NF, CYCLE, LONG)
MISSING = object()


class Shown:
    """A value shown as the given text in the result."""

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


# ---------------------------------------------------------------- one call and its check

def _classes(sol):
    return tuple(c for c in (getattr(sol, n, None) for n in ERRORS) if isinstance(c, type))


def _run(sol, fn):
    """("ok", value) or ("err", exc) for the answer's own DNS errors; any other exception propagates."""
    try:
        return ("ok", fn())
    except Exception as e:
        if isinstance(e, _classes(sol)):
            return ("err", e)
        raise


def _as_list(x):
    return list(x) if isinstance(x, tuple) else x


def _want_text(want):
    return lit(want.addresses) if want.ok else f"抛出 {want.error}({lit(want.name)})"


def _got_text(got):
    kind, value = got
    if kind == "ok":
        return value
    name = getattr(value, "name", MISSING)
    shown = "没有 name 属性" if name is MISSING else lit(name) if isinstance(name, str) else repr(name)
    return Shown(f"抛出 {type(value).__name__}({shown})")


def _check(sol, got, want, why=""):
    """got from _run against the model's Outcome."""
    tail = f"（{why}）" if why else ""
    kind, value = got
    if want.ok:
        if kind == "err":
            raise Mismatch(f"应当返回地址列表，实际抛出了 {type(value).__name__}{tail}", None, Shown(_want_text(want)),
                           _got_text(got))
        value = _as_list(value)
        expect(value, want.addresses, msg=f"返回的地址列表不对{tail}" if isinstance(value, list) else "")
        return
    cls = getattr(sol, want.error, None)
    expect_true(isinstance(cls, type), f"没有找到 {want.error} 类（类名要和题目一致）")
    if kind == "ok":
        raise Mismatch(f"应当抛出 {want.error}，但正常返回了{tail}", None, Shown(_want_text(want)), value)
    if not isinstance(value, cls):
        raise Mismatch(f"应当抛出 {want.error}，实际抛出 {type(value).__name__}{tail}", None, Shown(_want_text(want)),
                       _got_text(got))
    if getattr(value, "name", MISSING) != want.name:
        raise Mismatch(f"异常类型对了，但它的 name 应当是 {lit(want.name)}{tail}", None, Shown(_want_text(want)),
                       _got_text(got))


def _outcome(want, hit=False):
    """A hard-coded expectation: a list of addresses, or (error class name, name)."""
    if isinstance(want, list):
        return ref.Outcome(addresses=want, hit=hit)
    return ref.Outcome(error=want[0], name=want[1])


def _resolver(sol, zone, fallback, max_len, touch=None):
    args = [ref.build(sol, zone, touch)] + ([] if fallback is None else [ref.build(sol, fallback, touch)])
    return sol.Resolver(*args, **({} if max_len is None else {"max_chain_length": max_len}))


def _head(zone, fallback, max_len):
    return ref.zone_code(zone, "zone") + ([] if fallback is None else ref.zone_code(fallback, "fallback")) + \
        [ref.resolver_code(fallback, max_len)]


def _rereads(sol):
    """True when the answer reads a CNAME target again every time it walks the chain (a resolve after the cached
    answer expired reads it again). An answer that normalises the targets once, at construction or the first
    time it meets each one, reads nothing on a later walk, so for it a miss that reads no target is no error."""
    touch = ref.Touch()
    try:
        r = _resolver(sol, [("p.t.", ("CNAME", "q.t.", 1)), ("q.t.", ("A", ["192.0.2.1"], 1))], None, None, touch)
        r.resolve("p.t.")
        r.advance_clock(10)
        before = touch.count
        r.resolve("p.t.")
        return touch.count > before
    except Exception:       # noqa: BLE001 - the tests themselves report what is wrong
        return False


class Obs:
    """rereads: the answer reads the targets again on every walk (see _rereads), so a miss that reads none is an
    error. seen: it was seen reading a target inside resolve. blind: a miss that walks a CNAME read nothing and
    rereads is False, so that check was skipped."""

    def __init__(self, seen=False, rereads=False):
        self.seen, self.blind, self.rereads = seen, False, rereads

    def report(self, text=NO_TOUCH):
        if self.blind:
            info(MEMO if self.seen else text)


class Scene:
    """A Resolver built from zones given as data and checked call by call against the model. With touch=True
    (Level 4) every call is also checked to be a cache hit or a miss, and the replay lists the calls so far;
    otherwise the replay shows only the zones and the call being checked."""

    def __init__(self, sol, zone, fallback=None, max_len=None, touch=False, obs=None):
        self.sol = sol
        self.model = ref.Model(zone, fallback, 8 if max_len is None else max_len)
        self.head = _head(zone, fallback, max_len)
        self.steps = []
        self.touch = ref.Touch() if touch else None
        self.obs = obs if obs is not None else Obs(rereads=touch and _rereads(sol))
        calling("\n".join(self.head))
        self.r = _resolver(sol, zone, fallback, max_len, self.touch)

    def _render(self, current, mark="# <- 这一步"):
        if self.touch is None:
            return "\n".join(self.head + [current])
        lines = self.steps + [(current, mark)]
        skipped = max(0, len(lines) - SHOW)
        lines = lines[skipped:]
        width = max(len(c) for c, _ in lines) + 3
        top = self.head + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
        return "\n".join(top + [(c.ljust(width) + n) if n else c for c, n in lines])

    def _reason(self, key):
        m = self.model
        entry = m.cache.get(key)
        if entry is not None:
            if m.now < entry[1]:
                return f"{lit(key)} 在 clock={entry[2]} 存入缓存，过期时间是 {entry[1]}，现在 clock={m.now}，还没到"
            return f"{lit(key)} 的缓存过期时间是 {entry[1]}，现在 clock={m.now}，已经过期"
        if key in m.failed:
            return f"上一次解析 {lit(key)} 失败了，失败的结果不缓存"
        return f"{lit(key)} 之前没有作为 resolve 的参数查询过（顶多在别的名字的链上走到过），没有它的缓存"

    def resolve(self, query, why="", want=None, hit=False):
        """r.resolve(query), expecting the model's outcome, or `want` (and `hit`) for a statement example."""
        key = normalize(query)
        reason = self._reason(key)
        out = self.model.resolve(query)
        if want is not None:
            cnames = out.cnames
            out = _outcome(want, hit)
            out.cnames = cnames
        code = f"r.resolve({lit(query)})"
        calling(self._render(code))
        before = self.touch.count if self.touch is not None else 0
        got = _run(self.sol, lambda: self.r.resolve(query))
        _check(self.sol, got, out, why)
        if self.touch is not None:
            touched = self.touch.count - before
            if out.hit:
                expect_true(touched == 0, f"这次应当命中缓存：{reason}。命中时不应再读任何记录，但这次调用读取了 CNAME "
                                          f"记录，又沿着链解析了一遍。" + TOUCH_NOTE)
            elif out.cnames:
                if touched:
                    self.obs.seen = True
                elif self.obs.rereads:
                    expect_true(False, f"这次应当重新解析：{reason}。但这次调用没有读取任何 CNAME 记录，像是用了缓存。"
                                + TOUCH_NOTE)
                else:
                    self.obs.blind = True
        note = "# -> " + _want_text(out)
        if self.touch is not None:
            note += "，命中缓存" if out.hit else "，重新解析"
        self.steps.append((code, note))

    def advance(self, seconds):
        code = f"r.advance_clock({seconds})"
        calling(self._render(code))
        self.r.advance_clock(seconds)
        self.model.advance_clock(seconds)
        self.steps.append((code, f"# clock = {self.model.now}"))


# ---------------------------------------------------------------- random zones (Levels 1 to 3)

def _fails(sol, zone, fallback, max_len, query):
    """The kind of failure of one fresh resolve call against the model, or None when it agrees."""
    try:
        r = _resolver(sol, zone, fallback, max_len)
        _check(sol, _run(sol, lambda: r.resolve(query)),
               ref.Model(zone, fallback, 8 if max_len is None else max_len).lookup(query))
    except Mismatch:
        return Mismatch
    except Exception as e:
        return type(e)
    return None


def _shrink(sol, zone, fallback, max_len, query, kind, budget=0.6):
    """Drops records (and simplifies spellings) the failure does not need, so the replay is short."""
    stop = time.process_time() + budget

    def still(z, f, q):
        return time.process_time() < stop and _fails(sol, z, f, max_len, q) is kind

    if still(zone, fallback, normalize(query)):
        query = normalize(query)
    progress = True
    while progress and time.process_time() < stop:
        progress = False
        for side in (0, 1):
            i = 0
            while True:
                spec = zone if side == 0 else fallback
                if spec is None or i >= len(spec):
                    break
                trial = spec[:i] + spec[i + 1:]
                z, f = (trial, fallback) if side == 0 else (zone, trial)
                if still(z, f, query):
                    zone, fallback, progress = z, f, True
                else:
                    i += 1
        for side in (0, 1):
            spec = zone if side == 0 else fallback
            for i in range(len(spec or [])):
                spec = zone if side == 0 else fallback
                key, (kind_, value, ttl) = spec[i]
                plain = (normalize(key), (kind_, normalize(value) if kind_ == "CNAME" else value, ttl))
                if plain == spec[i] or any(k == plain[0] for k, _ in spec):
                    continue
                trial = spec[:i] + [plain] + spec[i + 1:]
                z, f = (trial, fallback) if side == 0 else (zone, trial)
                if still(z, f, query):
                    zone, fallback = z, f
    return zone, fallback, query


def random_zones(sol, level, rounds, queries):
    r = rng()
    for _ in range(rounds):
        addrs = ref.Addresses()
        names = r.sample(ref.NAMES, r.randint(3, 8))
        if level == 1 and r.random() < 0.15:
            names.append(".")
        zone = ref.random_zone(r, names, addrs, a_only=level == 1)
        fallback, max_len = None, None
        if level >= 2:
            max_len = r.choice([None, None, 0, 1, 2, 3, 4])
        if level >= 3 and r.random() < 0.8:
            pool = r.sample(names, r.randint(1, len(names))) + r.sample(ref.NAMES, 2)
            fallback = ref.random_zone(r, list(dict.fromkeys(pool)), addrs, p_missing=0.3) if r.random() < 0.9 else []
        for _ in range(queries):
            query = ref.spell(r, r.choice(names + ref.GHOSTS[:1]))
            kind = _fails(sol, zone, fallback, max_len, query)
            if kind is None:
                continue
            zone, fallback, query = _shrink(sol, zone, fallback, max_len, query, kind)
            Scene(sol, zone, fallback, max_len).resolve(query)
            raise Mismatch("随机区域里这次调用和预期不一致，但用新建的 Resolver 重新运行时又对了：结果可能和别的 "
                           "Resolver 有关（比如把缓存放在了类或模块级别）。", "\n".join(
                               _head(zone, fallback, max_len) + [f"r.resolve({lit(query)})"]))


# ---------------------------------------------------------------- random operations (Level 4)

def _play(sol, zone, fallback, max_len, ops, obs):
    """Runs the ops on a fresh Scene; returns (index of the failing op, exception) or None."""
    s = Scene(sol, zone, fallback, max_len, touch=True, obs=obs)
    for k, (method, arg) in enumerate(ops):
        try:
            s.resolve(arg) if method == "resolve" else s.advance(arg)
        except Exception as e:
            return k, e
    return None


def _shrink_ops(sol, zone, fallback, max_len, ops, seen, rereads, kind, budget=0.8):
    """Drops operations and records the failure does not need, so the replay is short."""
    stop = time.process_time() + budget

    def fails(z, f, o):
        if time.process_time() >= stop:
            return None
        out = _play(sol, z, f, max_len, o, Obs(seen, rereads))
        return out if out is not None and type(out[1]) is kind else None

    progress = True
    while progress and time.process_time() < stop:
        progress = False
        i = 0
        while i < len(ops) - 1:
            out = fails(zone, fallback, ops[:i] + ops[i + 1:])
            if out is not None:
                ops, progress = (ops[:i] + ops[i + 1:])[: out[0] + 1], True
            else:
                i += 1
        for side in (0, 1):
            i = 0
            while True:
                spec = zone if side == 0 else fallback
                if spec is None or i >= len(spec):
                    break
                trial = spec[:i] + spec[i + 1:]
                z, f = (trial, fallback) if side == 0 else (zone, trial)
                out = fails(z, f, ops)
                if out is not None:
                    zone, fallback, ops, progress = z, f, ops[: out[0] + 1], True
                else:
                    i += 1
    return zone, fallback, ops


# ---------------------------------------------------------------- threads (Level 5)

class Crowd:
    """resolve calls on one Resolver, one daemon thread each; results[i] is ("ok", value), ("err", the answer's
    DNS error) or ("raise", any other exception). With a touch, call i tags its touches with i, and called[i] is
    the time just before it called resolve."""

    def __init__(self, sol, r, touch=None):
        self.sol, self.r, self.touch = sol, r, touch
        self.results, self.threads, self.called = [], [], []

    def spawn(self, query):
        slot = len(self.results)
        self.results.append(None)
        self.called.append(None)
        classes = _classes(self.sol)

        def run():
            if self.touch is not None:
                self.touch.local.call = slot
            self.called[slot] = time.monotonic()
            try:
                self.results[slot] = ("ok", self.r.resolve(query))
            except Exception as e:      # noqa: BLE001 - checked by the main thread
                self.results[slot] = ("err", e) if isinstance(e, classes) else ("raise", e)
        self.threads.append(ref.start(run))

    def join(self, wait=WAIT):
        return ref.join_all(self.threads, wait)

    def walkers(self, before):
        """The calls made before time `before` that read a CNAME target (so walked a chain)."""
        return sorted(slot for slot in self.touch.calls if self.called[slot] < before)

    def walks(self, before, gate):
        """How many walks of the paused name there were: the calls above, plus the threads that are not callers
        (a thread pool doing the walk for them) and stopped at the gate while it was closed."""
        mine = {t.ident for t in self.threads}
        return len(self.walkers(before)) + len(gate.threads - mine)

    def reraise(self):
        for res in self.results:
            if res is not None and res[0] == "raise":
                raise res[1]


def _call_in_thread(sol, fn, wait):
    """fn() in a helper thread: ("ok", value) / ("err", DNS error), None when it has not returned after `wait`
    seconds; any other exception is raised here."""
    box = {}
    classes = _classes(sol)

    def run():
        try:
            box["res"] = ("ok", fn())
        except Exception as e:          # noqa: BLE001
            box["res"] = ("err", e) if isinstance(e, classes) else ("raise", e)
    t = ref.start(run)
    t.join(wait)
    if t.is_alive():
        return None
    if box["res"][0] == "raise":
        raise box["res"][1]
    return box["res"]


def _wait_entered(gate, leader):
    """'entered' once the first caller is paused, 'done' if it returned without reading the gated target,
    'stuck' if neither happened in time."""
    stop = time.monotonic() + ENTER
    while time.monotonic() < stop:
        if gate.entered.wait(0.01):
            return "entered"
        if not leader.is_alive():
            return "entered" if gate.entered.is_set() else "done"
    return "stuck"


STUCK = "有 {k} 个调用一直没有返回（等了 {s:g} 秒）：可能死锁了，或者等待同一个名字的调用没有被唤醒。"


def _same_name(sol, zone, fallback, max_len, query, spellings, want, lines):
    """The first call for query is paused halfway; the calls for `spellings` arrive meanwhile. Returns
    (whether the pause happened, the touch counter, the resolver)."""
    touch = ref.Touch()
    r = _resolver(sol, zone, fallback, max_len, touch)
    gate = ref.Gate(query)
    touch.hook = gate
    crowd = Crowd(sol, r, touch)
    calling("\n".join(_head(zone, fallback, max_len) + lines))
    try:
        crowd.spawn(query)
        state = _wait_entered(gate, crowd.threads[0])
        if state == "entered":
            for s in spellings:
                crowd.spawn(s)
            time.sleep(ARRIVE)
    finally:
        released = time.monotonic()
        gate.release.set()
    if state == "done":
        for s in spellings:
            crowd.spawn(s)
    finished = crowd.join()
    touch.hook = None
    expect_true(finished, STUCK.format(k=sum(t.is_alive() for t in crowd.threads), s=WAIT))
    crowd.reraise()
    for res in crowd.results:
        _check(sol, res, want)
    if state == "entered":
        n = crowd.walks(released, gate)
        expect_true(n == 1, f"{n} 个调用各自解析了 {lit(normalize(query))}：第一个调用还在解析时，后到的同名调用应当"
                            f"等它的结果（成功或失败都一样），不能自己再解析一遍。" + TOUCH_NOTE)
    lists = [v for k, v in crowd.results if k == "ok" and isinstance(v, list)]
    expect_true(len({id(v) for v in lists}) == len(lists),
                "有几个调用拿到的是同一个 list 对象：每次 resolve 返回的列表都应当是这个调用自己的副本。")
    return state == "entered", touch, r, crowd.results


# ================================================================ Level 1

EX1 = [("api.acme.com.", ("A", ["203.0.113.10"], 60)),
       ("db.acme.com.", ("A", ["203.0.113.20", "203.0.113.21"], 60))]


@case("Level 1", sample=True, title="题目例子：直接查 A 记录")
def l1_example(sol):
    s = Scene(sol, EX1)
    s.resolve("API.acme.com", want=["203.0.113.10"])
    s.resolve("db.acme.com.", want=["203.0.113.20", "203.0.113.21"])
    s.resolve("nope.acme.com.", want=(NF, "nope.acme.com."))


@case("Level 1", title="名称归一化：大小写和结尾的点")
def l1_normalize(sol):
    zone = [("Example.COM", ("A", ["192.0.2.1"], 5)),
            ("shop.example.com..", ("A", ["192.0.2.2", "192.0.2.3"], 5)),
            ("例子.测试", ("A", ["192.0.2.4"], 5)),
            ("MiXeD-1.Case2.org.", ("A", ["192.0.2.5"], 5))]
    s = Scene(sol, zone)
    for q in ["example.com", "EXAMPLE.com..", "example.com.", "Example.Com...."]:
        s.resolve(q, "区域的键和查询的名字都要归一化")
    for q in ["SHOP.Example.com", "shop.example.com.", "例子.测试.", "例子.测试...", "mixed-1.case2.ORG"]:
        s.resolve(q, "区域的键和查询的名字都要归一化")
    s.resolve("example..com", "只把结尾的点合并成一个，中间的点不动")
    s.resolve(".example.com", "开头的点不会去掉")
    s.resolve("examplecom.", "不同的名字")


@case("Level 1", title="空串和全是点都归一化成 \".\"")
def l1_root(sol):
    s = Scene(sol, [("...", ("A", ["192.0.2.9"], 5)), ("a.", ("A", ["192.0.2.10"], 5))])
    for q in ["", ".", "..", "....."]:
        s.resolve(q, "结尾的零个或多个点换成正好一个，空串和全是点的名字都变成 \".\"")
    s.resolve("A", "\"A\" 归一化成 \"a.\"")
    s = Scene(sol, [("b.", ("A", ["192.0.2.11"], 5))])
    s.resolve("", "\"\" 归一化成 \".\"")


@case("Level 1", title="两个键归一化后相同：后出现的生效")
def l1_duplicate_keys(sol):
    s = Scene(sol, [("Dup.acme.com.", ("A", ["203.0.113.200"], 10)),
                    ("dup.acme.com", ("A", ["203.0.113.201"], 10))])
    s.resolve("dup.acme.com.", "遍历 dict 时后出现的键生效")
    s = Scene(sol, [("dup.acme.com", ("A", ["203.0.113.201"], 10)),
                    ("other.acme.com.", ("A", ["203.0.113.202"], 10)),
                    ("DUP.ACME.COM...", ("A", ["203.0.113.200"], 10))])
    s.resolve("Dup.Acme.Com", "遍历 dict 时后出现的键生效")
    s.resolve("other.acme.com", "")
    s = Scene(sol, [("x.t", ("A", ["192.0.2.1"], 1)), ("X.T..", ("A", ["192.0.2.2"], 1)),
                    ("x.T.", ("A", ["192.0.2.3"], 1))])
    s.resolve("x.t", "三个键归一化后都是 \"x.t.\"，最后一个生效")


@case("Level 1", title="NameNotFoundError 的类型和 name")
def l1_errors(sol):
    for n in ("DNSResolutionError", NF):
        expect_true(isinstance(getattr(sol, n, None), type), f"没有找到 {n} 类（类名要和题目一致）")
    expect_true(issubclass(sol.DNSResolutionError, Exception), "DNSResolutionError 应当是 Exception 的子类")
    expect_true(issubclass(sol.NameNotFoundError, sol.DNSResolutionError),
                "NameNotFoundError 应当是 DNSResolutionError 的子类")
    s = Scene(sol, EX1)
    s.resolve("NoPe.Acme.COM..", "name 是归一化后的名字")
    s.resolve("api.acme.co", "name 是归一化后的名字")
    s.resolve("", "\"\" 归一化成 \".\"，区域里没有")
    s = Scene(sol, [])
    s.resolve("Empty.Zone", "区域是空的")


@case("Level 1", cpu=2, title="随机区域与名字")
def l1_random(sol):
    random_zones(sol, 1, rounds=150, queries=8)


# ================================================================ Level 2

EX2 = EX1 + [("www.acme.com.", ("CNAME", "edge.acme.com.", 60)),
             ("edge.acme.com.", ("CNAME", "api.acme.com.", 30))]
CYCLE_ZONE = [("a.acme.com.", ("CNAME", "b.acme.com.", 60)), ("b.acme.com.", ("CNAME", "a.acme.com.", 60))]
LONG_ZONE = [("start.acme.com.", ("CNAME", "mid1.acme.com.", 60)),
             ("mid1.acme.com.", ("CNAME", "mid2.acme.com.", 60)),
             ("mid2.acme.com.", ("CNAME", "mid3.acme.com.", 60)),
             ("mid3.acme.com.", ("A", ["203.0.113.99"], 60))]


def chain(n, tail=("A", ["203.0.113.7"], 9), prefix="h"):
    """h0 -> h1 -> ... -> hn, where hn holds `tail`."""
    return [(f"{prefix}{i}.t.", ("CNAME", f"{prefix}{i + 1}.t.", 9)) for i in range(n)] + [(f"{prefix}{n}.t.", tail)]


RING = [(f"r{i}.t.", ("CNAME", f"r{(i + 1) % 4}.t.", 9)) for i in range(4)]


@case("Level 2", sample=True, title="题目例子：链、环、链长上限")
def l2_example(sol):
    Scene(sol, EX2).resolve("www.acme.com.", want=["203.0.113.10"])
    Scene(sol, CYCLE_ZONE).resolve("a.acme.com.", want=(CYCLE, "a.acme.com."))
    Scene(sol, LONG_ZONE, max_len=2).resolve("start.acme.com.", want=(LONG, "start.acme.com."))


@case("Level 2", title="链长正好等于上限可以，多一步不行")
def l2_length_limit(sol):
    Scene(sol, LONG_ZONE, max_len=3).resolve("start.acme.com.", "链长 3 等于 max_chain_length=3，允许")
    Scene(sol, chain(8)).resolve("h0.t.", "链长 8 等于默认的 max_chain_length=8，允许")
    Scene(sol, chain(9)).resolve("h0.t.", "链长 9 超过默认的 max_chain_length=8")
    Scene(sol, chain(9)).resolve("h1.t.", "从 h1 出发链长是 8，允许")
    Scene(sol, chain(0), max_len=0).resolve("h0.t.", "本身就是 ARecord，链长 0")
    Scene(sol, chain(1), max_len=0).resolve("h0.t.", "max_chain_length=0 时一步 CNAME 也不能走")
    Scene(sol, chain(1), max_len=1).resolve("h0.t.", "链长 1 等于上限，允许")
    Scene(sol, chain(5), max_len=4).resolve("h0.t.", "链长 5 超过上限 4")
    Scene(sol, chain(5), max_len=4).resolve("h1.t.", "从 h1 出发链长是 4，允许")


@case("Level 2", title="环的检查排在链长检查前面")
def l2_cycle_first(sol):
    loop = [("loop.t.", ("CNAME", "loop.t.", 9))]
    Scene(sol, loop, max_len=0).resolve("loop.t.", "这一步的目标已经走过，先报环，哪怕这一步也会超过上限 0")
    Scene(sol, RING, max_len=3).resolve("r0.t.", "第 4 步回到 r0：先查环，所以是 ResolutionCycleError")
    Scene(sol, RING, max_len=2).resolve("r0.t.", "第 3 步会让链长变成 3，超过上限 2，这时还没走到成环的那一步")
    Scene(sol, RING).resolve("r2.t.", "从 r2 出发，第 4 步回到 r2")
    Scene(sol, RING, max_len=4).resolve("r1.t.", "")


@case("Level 2", title="异常带的 name：环、缺失、超长")
def l2_error_names(sol):
    for n in (CYCLE, LONG):
        expect_true(isinstance(getattr(sol, n, None), type), f"没有找到 {n} 类（类名要和题目一致）")
        expect_true(issubclass(getattr(sol, n), sol.DNSResolutionError), f"{n} 应当是 DNSResolutionError 的子类")
    abcb = [("a.t.", ("CNAME", "b.t.", 9)), ("b.t.", ("CNAME", "c.t.", 9)), ("c.t.", ("CNAME", "b.t.", 9))]
    Scene(sol, abcb).resolve("A.t", "name 是重复出现的那个名字 b.t.，不是查询的名字")
    Scene(sol, abcb).resolve("c.t.", "从 c 出发：c -> b -> c，重复的是 c.t.")
    gone = [("x.t.", ("CNAME", "y.t.", 9)), ("y.t.", ("CNAME", "gone.t.", 9))]
    Scene(sol, gone).resolve("x.t.", "name 是链上缺失的那个名字，不是查询的名字")
    Scene(sol, LONG_ZONE, max_len=1).resolve("START.acme.com", "name 是最初的查询名（归一化后），不是超限那一步的名字")
    Scene(sol, chain(6), max_len=3).resolve("H2.T", "name 是最初的查询名（归一化后）")


@case("Level 2", title="超限时还没去查目标：不管目标在不在")
def l2_limit_before_lookup(sol):
    dangling = [("h0.t.", ("CNAME", "h1.t.", 9)), ("h1.t.", ("CNAME", "gone.t.", 9))]
    Scene(sol, dangling, max_len=1).resolve("h0.t.", "第 2 步会超过上限 1，这时还没去查 gone.t.")
    Scene(sol, dangling, max_len=2).resolve("h0.t.", "上限 2 时第 2 步可以走，然后才发现 gone.t. 不存在")
    into_ring = [("h.t.", ("CNAME", "r0.t.", 9)), ("r0.t.", ("CNAME", "r1.t.", 9)), ("r1.t.", ("CNAME", "r0.t.", 9))]
    Scene(sol, into_ring, max_len=1).resolve("h.t.", "第 2 步会超过上限 1，后面才会成的环还没走到")
    Scene(sol, into_ring, max_len=2).resolve("h.t.", "第 3 步回到 r0：先查环")


@case("Level 2", title="CNAME 的 target 也要归一化")
def l2_targets_normalized(sol):
    zone = [("WWW.Acme.com", ("CNAME", "EDGE.acme.COM..", 60)), ("edge.acme.com.", ("CNAME", "Api.Acme.Com", 30)),
            ("api.acme.com", ("A", ["203.0.113.10"], 60))]
    Scene(sol, zone, max_len=2).resolve("www.acme.com", "每一步走到的 target 都先归一化再查")
    loop = [("a.t.", ("CNAME", "B.T", 9)), ("b.t", ("CNAME", "A.t...", 9))]
    Scene(sol, loop).resolve("a.t", "\"A.t...\" 归一化后就是查询的名字 a.t.")
    Scene(sol, [("a.t.", ("CNAME", "MISSING.t", 9))]).resolve("a.t.", "缺失的名字报归一化后的形式")
    Scene(sol, [("self.t", ("CNAME", "SELF.T..", 9))]).resolve("Self.t", "")


@case("Level 2", cpu=2, title="随机区域：链、环、缺失、上限")
def l2_random(sol):
    random_zones(sol, 2, rounds=200, queries=8)


# ================================================================ Level 3

PRIMARY = [("svc.acme.com.", ("CNAME", "internal.acme.com.", 60))]
FALLBACK = [("svc.acme.com.", ("A", ["198.51.100.5"], 30))]
LOOPING = [("loop.acme.com.", ("CNAME", "loop.acme.com.", 60))]
HAS_ANSWER = [("loop.acme.com.", ("A", ["198.51.100.9"], 60))]


@case("Level 3", sample=True, title="题目例子：只有查无此名才回退")
def l3_example(sol):
    Scene(sol, PRIMARY, FALLBACK).resolve("svc.acme.com.", want=["198.51.100.5"])
    Scene(sol, LOOPING, HAS_ANSWER).resolve("loop.acme.com.", want=(CYCLE, "loop.acme.com."))
    Scene(sol, [], []).resolve("nowhere.acme.com.", want=(NF, "nowhere.acme.com."))


@case("Level 3", title="备用区域也失败：报它自己缺的名字")
def l3_fallback_error(sol):
    x_y = [("x.acme.com.", ("CNAME", "y.acme.com.", 10))]
    Scene(sol, x_y, [("x.acme.com.", ("CNAME", "z.acme.com.", 10))]).resolve(
        "x.acme.com.", "第一次尝试的错误被丢弃，报备用区域的链里缺的 z")
    Scene(sol, x_y, []).resolve("x.acme.com.", "备用区域是空的：它那次尝试缺的是查询的名字本身")
    Scene(sol, [("other.t.", ("A", ["192.0.2.1"], 5))], [("q.t.", ("CNAME", "p.t.", 5))]).resolve(
        "Q.T", "主区域没有 q.t.，备用区域里 q.t. 指向的 p.t. 不存在")
    Scene(sol, x_y).resolve("x.acme.com.", "没有给 fallback_zone，直接报主区域的错误")


@case("Level 3", title="备用区域从头重新解析，不接着走")
def l3_fresh_chain(sol):
    primary = [("a.t.", ("CNAME", "b.t.", 9))]
    fallback = [("b.t.", ("A", ["192.0.2.1"], 9)), ("a.t.", ("CNAME", "c.t.", 9)), ("c.t.", ("A", ["192.0.2.2"], 9))]
    Scene(sol, primary, fallback).resolve("a.t.", "在备用区域里从查询的名字 a.t. 重新解析，不是从缺失的 b.t. 接着查")
    Scene(sol, primary, [("b.t.", ("A", ["192.0.2.1"], 9))]).resolve(
        "a.t.", "备用区域里没有 a.t.：重新解析时第一步就缺失")
    mixed = [("a.t.", ("CNAME", "m.t.", 9)), ("m.t.", ("A", ["192.0.2.3"], 9))]
    Scene(sol, [("a.t.", ("CNAME", "gone.t.", 9))], [("a.t.", ("CNAME", "m.t.", 9))]).resolve(
        "a.t.", "备用区域那次解析只用备用区域，m.t. 在备用区域里不存在")
    Scene(sol, mixed, [("a.t.", ("A", ["192.0.2.4"], 9))]).resolve("a.t.", "主区域能解析就不用备用区域")


@case("Level 3", title="环或超长不回退")
def l3_no_fallback_on_bad_chain(sol):
    dangling = [("h0.t.", ("CNAME", "h1.t.", 9)), ("h1.t.", ("CNAME", "gone.t.", 9))]
    rescue = [("h0.t.", ("A", ["203.0.113.8"], 9))]
    Scene(sol, dangling, rescue, max_len=1).resolve("h0.t.", "第一次尝试是 ChainTooLongError，不回退")
    Scene(sol, dangling, rescue, max_len=2).resolve("h0.t.", "上限 2 时第一次尝试是 NameNotFoundError，回退")
    Scene(sol, RING, [("r0.t.", ("A", ["203.0.113.9"], 9))]).resolve("r0.t.", "环不回退")
    Scene(sol, chain(3), [("h0.t.", ("A", ["203.0.113.9"], 9))], max_len=2).resolve("h0.t.", "超长不回退")


@case("Level 3", title="备用区域用同一个链长上限")
def l3_fallback_limit(sol):
    Scene(sol, [], chain(3), max_len=2).resolve("h0.t.", "备用区域那次解析同样受 max_chain_length=2 限制")
    Scene(sol, [], chain(3), max_len=3).resolve("h0.t.", "")
    Scene(sol, [("h0.t.", ("CNAME", "nowhere.t.", 9))], chain(9)).resolve("h0.t.", "默认上限 8 同样适用于备用区域")
    Scene(sol, [], chain(8)).resolve("h0.t.", "")
    Scene(sol, [], RING, max_len=3).resolve("r1.t.", "备用区域里的环")


@case("Level 3", title="备用区域的键和 target 也归一化")
def l3_fallback_normalized(sol):
    fallback = [("SVC.Acme.com", ("CNAME", "Real.ACME.com...", 9)), ("real.acme.com", ("A", ["198.51.100.7"], 9)),
                ("svc.acme.com.", ("CNAME", "REAL.acme.com", 9)), ("Real.Acme.Com.", ("A", ["198.51.100.8"], 9))]
    Scene(sol, PRIMARY, fallback).resolve("Svc.Acme.Com..", "备用区域里同名的键也是后出现的生效")
    Scene(sol, [("q.t", ("A", ["192.0.2.1"], 9))], fallback).resolve("Real.acme.COM", "")


@case("Level 3", cpu=2, title="随机主区域与备用区域")
def l3_random(sol):
    random_zones(sol, 3, rounds=200, queries=8)


# ================================================================ Level 4

@case("Level 4", sample=True, title="题目例子：29 命中，30 过期")
def l4_example(sol):
    s = Scene(sol, EX2, touch=True)
    s.resolve("www.acme.com.", want=["203.0.113.10"], hit=False)
    s.advance(29)
    s.resolve("www.acme.com.", want=["203.0.113.10"], hit=True)
    s.advance(1)
    s.resolve("www.acme.com.", want=["203.0.113.10"], hit=False)
    s.obs.report()


@case("Level 4", title="过期时间取链上 ttl 的最小值")
def l4_min_ttl(sol):
    zone = [("t1.acme.com.", ("CNAME", "t2.acme.com.", 50)), ("t2.acme.com.", ("CNAME", "t3.acme.com.", 7)),
            ("t3.acme.com.", ("A", ["203.0.113.30"], 20)), ("one.acme.com.", ("CNAME", "t3.acme.com.", 1))]
    s = Scene(sol, zone, touch=True)
    s.resolve("t1.acme.com.")
    s.advance(6)
    s.resolve("t1.acme.com.", "min(50, 7, 20) = 7，clock=6 还没过期")
    s.advance(1)
    s.resolve("t1.acme.com.", "clock=7 等于过期时间，已经过期")
    s.advance(6)
    s.resolve("T1.acme.com", "上次在 clock=7 存入，过期时间 14")
    s.advance(1)
    s.resolve("t1.acme.com.", "")
    s.resolve("one.acme.com.")
    s.resolve("one.acme.com.", "ttl=1：同一时刻再查还没过期")
    s.advance(0)
    s.resolve("one.acme.com.", "advance_clock(0) 不改变时钟")
    s.advance(1)
    s.resolve("one.acme.com.", "")
    s.obs.report()


@case("Level 4", title="只算成功那次尝试读到的 ttl")
def l4_successful_attempt_ttl(sol):
    primary = [("svc.t.", ("CNAME", "internal.t.", 2))]
    fallback = [("svc.t.", ("CNAME", "real.t.", 40)), ("real.t.", ("A", ["198.51.100.5"], 50))]
    s = Scene(sol, primary, fallback, touch=True)
    s.resolve("svc.t.")
    s.advance(2)
    s.resolve("svc.t.", "过期时间是 0 + min(40, 50) = 40，主区域那次失败的尝试读到的 ttl=2 不算")
    s.advance(37)
    s.resolve("svc.t.", "")
    s.advance(1)
    s.resolve("svc.t.", "")
    s.obs.report()


@case("Level 4", title="失败的解析不缓存")
def l4_failures_not_cached(sol):
    zone = [("gone.t.", ("CNAME", "missing.t.", 30)), ("a.t.", ("CNAME", "b.t.", 30)), ("b.t.", ("CNAME", "a.t.", 30)),
            ("ok.t.", ("CNAME", "l2.t.", 30))] + chain(3, prefix="l")
    s = Scene(sol, zone, max_len=2, touch=True)
    for q in ["gone.t.", "gone.t.", "GONE.t", "a.t.", "a.t.", "l0.t.", "l0.t."]:
        s.resolve(q, "失败的解析不缓存，每次都要重新解析")
    s.advance(1)
    s.resolve("gone.t.", "")
    s.resolve("ok.t.")
    s.resolve("ok.t.", "成功的结果照常缓存")
    s.obs.report()


@case("Level 4", title="只缓存查询的名字，不缓存链上的名字")
def l4_only_query_cached(sol):
    s = Scene(sol, EX2, touch=True)
    s.resolve("www.acme.com.")
    s.resolve("edge.acme.com.", "edge 只是 www 的链上一环，之前没有被查询过")
    s.resolve("edge.acme.com.", "")
    s.resolve("www.acme.com.", "")
    s.advance(30)
    s.resolve("edge.acme.com.", "")
    s = Scene(sol, PRIMARY, [("svc.acme.com.", ("CNAME", "real.t.", 9)), ("real.t.", ("CNAME", "a.t.", 9)),
                             ("a.t.", ("A", ["198.51.100.6"], 9))], touch=True)
    s.resolve("svc.acme.com.")
    s.resolve("real.t.", "real.t. 只出现在 svc 的链上")
    s.resolve("svc.acme.com.", "")
    s.obs.report()


@case("Level 4", title="缓存按归一化后的名字存")
def l4_cache_key(sol):
    s = Scene(sol, EX2, touch=True)
    s.resolve("WWW.acme.com")
    s.resolve("www.acme.com...", "和上一次是同一个名字（归一化后都是 www.acme.com.）")
    s.resolve("www.ACME.com.", "")
    s.advance(29)
    s.resolve("Www.Acme.Com", "")
    s.obs.report()


@case("Level 4", title="过期后重新解析并覆盖旧的缓存")
def l4_refresh(sol):
    s = Scene(sol, EX2, touch=True)
    s.resolve("www.acme.com.")
    s.advance(30)
    s.resolve("www.acme.com.", "")
    s.advance(1)
    s.resolve("www.acme.com.", "过期后重新解析的结果要覆盖旧的缓存：新的过期时间是 30 + 30 = 60")
    s.advance(28)
    s.resolve("www.acme.com.", "")
    s.advance(1)
    s.resolve("www.acme.com.", "")
    s.advance(100)
    s.resolve("www.acme.com.", "")
    s.advance(29)
    s.resolve("www.acme.com.", "")
    s.obs.report()


@case("Level 4", cpu=2, title="随机的 resolve 与 advance_clock")
def l4_random(sol):
    r = rng()
    obs = Obs(rereads=_rereads(sol))
    for _ in range(120):
        addrs = ref.Addresses()
        names = r.sample(ref.NAMES, r.randint(3, 6))
        zone = ref.random_zone(r, names, addrs, p_missing=0.1, ttl=(1, 12))
        fallback = ref.random_zone(r, names, addrs, p_missing=0.4, ttl=(1, 12)) if r.random() < 0.5 else None
        max_len = r.choice([None, None, 2, 3])
        ops = ref.random_ops(r, zone, fallback, max_len, names, r.randint(12, 30))
        bad = _play(sol, zone, fallback, max_len, ops, obs)
        if bad is None:
            continue
        k, exc = bad
        seen = obs.seen
        zone, fallback, ops = _shrink_ops(sol, zone, fallback, max_len, ops[: k + 1], seen, obs.rereads, type(exc))
        out = _play(sol, zone, fallback, max_len, ops, Obs(seen, obs.rereads))
        raise out[1] if out is not None else exc
    obs.report()


# ================================================================ Level 5

EX5 = EX2 + [("shop.acme.com.", ("CNAME", "edge.acme.com.", 60))]


@case("Level 5", sample=True, cpu=3, title="题目例子：5 个线程查同一个名字")
def l5_example(sol):
    touch = ref.Touch()
    r = _resolver(sol, EX5, None, None, touch)
    gate = ref.Gate("www.acme.com.")
    touch.hook = gate
    crowd = Crowd(sol, r, touch)
    lines = ref.zone_code(EX5, "zone") + ["r = Resolver(zone)", f"# 线程 t1：r.resolve(\"www.acme.com.\")，{PAUSED}",
                                         "# 线程 t2～t5 同时：r.resolve(\"www.acme.com.\")",
                                         "# 线程 t6 同时：r.resolve(\"shop.acme.com.\")"]
    calling("\n".join(lines))
    shop = None
    try:
        crowd.spawn("www.acme.com.")
        state = _wait_entered(gate, crowd.threads[0])
        for _ in range(4):
            crowd.spawn("www.acme.com.")
        if state == "entered":
            time.sleep(ARRIVE)
            shop = _call_in_thread(sol, lambda: r.resolve("shop.acme.com."), FREE)
            expect_true(shop is not None, f"t1 还在解析 www.acme.com. 时，t6 的 resolve(\"shop.acme.com.\") 等了 {FREE:g} "
                                          "秒还没返回：t6 不应当加入 t1～t5 的等待，不同的名字要能同时解析。")
    finally:
        released = time.monotonic()
        gate.release.set()
    if state != "entered":
        shop = _call_in_thread(sol, lambda: r.resolve("shop.acme.com."), WAIT)
        info(NO_GATE)
    finished = crowd.join()
    expect_true(finished and shop is not None, STUCK.format(k=sum(t.is_alive() for t in crowd.threads), s=WAIT))
    crowd.reraise()
    want = _outcome(["203.0.113.10"])
    for res in crowd.results:
        _check(sol, res, want)
    _check(sol, shop, want)
    if state == "entered":
        n = crowd.walks(released, gate)
        expect_true(n == 1, f"www.acme.com. 的链被 {n} 个线程各自走了一遍：t1 还在解析时，t2～t5 应当等它的结果。"
                            + TOUCH_NOTE)


@case("Level 5", cpu=4, title="同名调用合并成一次解析（多轮）")
def l5_coalesce(sol):
    r = rng()
    blind = False
    scenarios = [
        (EX2, None, None, "www.acme.com."),
        (EX2, None, None, "edge.acme.com."),
        (PRIMARY, [("svc.acme.com.", ("CNAME", "real.t.", 9)), ("real.t.", ("A", ["198.51.100.6", "198.51.100.7"], 9))],
         None, "svc.acme.com."),
        (chain(4), None, 4, "h0.t."),
        ([("Mixed.Case.T", ("CNAME", "TARGET.t..", 5)), ("target.t", ("A", ["192.0.2.50"], 5))], None, 1, "mixed.case.t."),
        (chain(2, prefix="例"), [], None, "例0.t."),
    ]
    for zone, fallback, max_len, query in scenarios:
        n = r.randint(24, 32)
        spellings = [ref.spell(r, query) for _ in range(n)]
        want = ref.Model(zone, fallback, 8 if max_len is None else max_len).lookup(query)
        lines = [f"# 线程 1：r.resolve({lit(query)})，{PAUSED}",
                 f"# 同时另外 {n} 个线程也查这个名字，写法各不相同，如 r.resolve({lit(spellings[0])})"]
        entered, _, res, results = _same_name(sol, zone, fallback, max_len, query, spellings, want, lines)
        blind |= not entered
        calling("\n".join(_head(zone, fallback, max_len) + lines +
                          ["# 都返回之后，往每个返回的列表里 append 一个元素，再调用：", f"r.resolve({lit(query)})"]))
        for _, value in results:
            if isinstance(value, list):
                value.append("192.0.2.99")
        _check(sol, _run(sol, lambda: res.resolve(query)), want, "改动之前返回的列表不应当影响缓存")
    if blind:
        info(NO_GATE)


@case("Level 5", cpu=4, title="同名调用一起失败，失败不缓存")
def l5_failure_shared(sol):
    r = rng()
    blind = False
    rereads = _rereads(sol)
    scenarios = [
        ([("svc.t.", ("CNAME", "gone.t.", 9))], None, None, "svc.t."),
        ([("svc.t.", ("CNAME", "gone.t.", 9))], [("svc.t.", ("CNAME", "lost.t.", 9))], None, "svc.t."),
        (RING, None, None, "r0.t."),
        (chain(4), None, 2, "h0.t."),
    ]
    for zone, fallback, max_len, query in scenarios:
        n = r.randint(16, 24)
        spellings = [ref.spell(r, query) for _ in range(n)]
        want = ref.Model(zone, fallback, 8 if max_len is None else max_len).lookup(query)
        lines = [f"# 线程 1：r.resolve({lit(query)})，{PAUSED}",
                 f"# 同时另外 {n} 个线程也查这个名字，写法各不相同，如 r.resolve({lit(spellings[0])})"]
        entered, touch, res, _ = _same_name(sol, zone, fallback, max_len, query, spellings, want, lines)
        blind |= not entered
        calling("\n".join(_head(zone, fallback, max_len) + lines + ["# 都结束之后，再调用一次：",
                                                                    f"r.resolve({lit(query)})"]))
        before = touch.count
        again = _call_in_thread(sol, lambda: res.resolve(query), WAIT)
        expect_true(again is not None, f"上一轮全部结束后，再调用 resolve({lit(query)}) 等了 {WAIT:g} 秒还没返回。")
        _check(sol, again, want, "失败的结果不缓存，再查一次要重新解析")
        if entered and rereads:
            expect_true(touch.count > before, "上一轮的解析失败了，失败的结果不缓存：再调用一次应当重新解析，但这次调用"
                                              "没有读取任何 CNAME 记录，像是直接拿了上一轮的异常。" + TOUCH_NOTE)
    if blind:
        info(NO_GATE)
    elif not rereads:
        info(MEMO)


@case("Level 5", cpu=3, title="别的名字不用等正在进行的解析")
def l5_other_names(sol):
    zone = EX5 + [("cdn.acme.com.", ("CNAME", "api.acme.com.", 60)), ("gone.acme.com.", ("CNAME", "lost.acme.com.", 60))]
    touch = ref.Touch()
    r = _resolver(sol, zone, None, None, touch)
    head = ref.zone_code(zone, "zone") + ["r = Resolver(zone)"]
    calling("\n".join(head + ["r.resolve(\"cdn.acme.com.\")"]))
    _check(sol, _run(sol, lambda: r.resolve("cdn.acme.com.")), _outcome(["203.0.113.10"]))
    gate = ref.Gate("www.acme.com.")
    touch.hook = gate
    crowd = Crowd(sol, r)
    others = [("shop.acme.com.", ["203.0.113.10"], "它的链也经过 edge.acme.com."),
              ("cdn.acme.com.", ["203.0.113.10"], "它已经在缓存里"),
              ("gone.acme.com.", (NF, "lost.acme.com."), "它会失败"),
              ("db.acme.com.", ["203.0.113.20", "203.0.113.21"], "")]
    try:
        crowd.spawn("www.acme.com.")
        state = _wait_entered(gate, crowd.threads[0])
        if state == "entered":
            for q, want, why in others:
                calling("\n".join(head + ["r.resolve(\"cdn.acme.com.\")", f"# 线程 A：r.resolve(\"www.acme.com.\")，{PAUSED}",
                                          f"# 线程 B 同时：r.resolve({lit(q)})"]))
                got = _call_in_thread(sol, lambda: r.resolve(q), FREE)
                expect_true(got is not None, f"www.acme.com. 正在解析（还没结束）时，resolve({lit(q)}) 等了 {FREE:g} 秒还"
                                             f"没返回{'（' + why + '）' if why else ''}：一个调用不应当为别的名字的解析"
                                             "等待，不能用一把锁把整个解析过程包起来。")
                _check(sol, got, _outcome(want))
    finally:
        gate.release.set()
    finished = crowd.join()
    touch.hook = None
    expect_true(finished, STUCK.format(k=1, s=WAIT))
    crowd.reraise()
    _check(sol, crowd.results[0], _outcome(["203.0.113.10"]))
    if state != "entered":
        info(NO_GATE)


@case("Level 5", cpu=3, title="过期时间按存入缓存那一刻的时钟算")
def l5_store_time(sol):
    touch = ref.Touch()
    r = _resolver(sol, EX2, None, None, touch)
    gate = ref.Gate("www.acme.com.")
    touch.hook = gate
    crowd = Crowd(sol, r)
    lines = ref.zone_code(EX2, "zone") + ["r = Resolver(zone)", f"# 线程 A：r.resolve(\"www.acme.com.\")，{PAUSED}",
                                         "# 线程 B 同时：r.advance_clock(5)，返回之后线程 A 才继续，把结果存入缓存"]
    calling("\n".join(lines))
    moved = None
    try:
        crowd.spawn("www.acme.com.")
        state = _wait_entered(gate, crowd.threads[0])
        if state == "entered":
            moved = _call_in_thread(sol, lambda: r.advance_clock(5), 1.0)
    finally:
        gate.release.set()
    finished = crowd.join()
    touch.hook = None
    expect_true(finished, STUCK.format(k=1, s=WAIT))
    crowd.reraise()
    _check(sol, crowd.results[0], _outcome(["203.0.113.10"]))
    if state != "entered":
        info(NO_GATE)
        return
    if moved is None:
        info("提示：别的线程在解析时，你的 advance_clock 一直等到解析结束才返回，这一项看不出过期时间按哪个时刻算，"
             "只检查了返回值。")
        return
    rereads = _rereads(sol)
    steps = lines + ["# 线程 A 返回之后（clock = 5，缓存的过期时间应当是 5 + 30 = 35）："]
    for adv, hit in ((29, True), (1, False)):
        r.advance_clock(adv)
        steps.append(f"r.advance_clock({adv})")
        steps.append("r.resolve(\"www.acme.com.\")")
        calling("\n".join(steps))
        before = touch.count
        _check(sol, _run(sol, lambda: r.resolve("www.acme.com.")), _outcome(["203.0.113.10"]))
        touched = touch.count - before
        if hit:
            expect_true(touched == 0, "这次应当命中缓存：结果是在 clock=5 时存入的，过期时间 5 + 30 = 35，现在 clock=34。"
                                      "过期时间要按存入那一刻的时钟算，不是开始解析时的时钟。" + TOUCH_NOTE)
        elif rereads:
            expect_true(touched > 0, "这次应当重新解析：过期时间是 35，现在 clock=35，已经过期。" + TOUCH_NOTE)
        else:
            info(MEMO)


@case("Level 5", title="返回的列表是调用方自己的副本")
def l5_copies(sol):
    r = _resolver(sol, EX2, None, None)
    steps = ref.zone_code(EX2, "zone") + ["r = Resolver(zone)"]
    want = ["203.0.113.10"]
    seen = []
    for k in range(3):
        steps.append(f"x{k} = r.resolve(\"www.acme.com.\")")
        calling("\n".join(steps))
        got = _run(sol, lambda: r.resolve("www.acme.com."))
        _check(sol, got, _outcome(want), "前面返回的列表被调用方改过，不应当影响缓存" if k else "")
        x = got[1]
        expect_true(all(x is not old for old in seen), "两次 resolve 返回了同一个 list 对象：每次返回的列表都应当是这个调用"
                                                       "自己的副本。")
        seen.append(x)
        if not isinstance(x, list):
            continue
        x.append("192.0.2.99")
        x[0] = "192.0.2.98"
        steps.append(f"x{k}.append(\"192.0.2.99\"); x{k}[0] = \"192.0.2.98\"")


@case("Level 5", cpu=5, title="多线程随机压力测试")
def l5_stress(sol):
    r = rng()
    blind = True
    with ref.fast_switching():
        for rnd in range(3):
            addrs = ref.Addresses()
            names = r.sample(ref.NAMES, 6)
            zone = ref.random_zone(r, names, addrs, p_missing=0.1, ttl=(1, 6))
            fallback = ref.random_zone(r, names, addrs, p_missing=0.4, ttl=(1, 6)) if rnd else None
            model = ref.Model(zone, fallback, 4)
            pool = names + ref.GHOSTS[:1]
            want = {normalize(q): model.lookup(q) for q in pool}
            for phase, (n_threads, n_ops) in enumerate(((40, 24), (40, 12))):
                touch = ref.Touch()
                touch.hook = ref.Jitter(rng(f"jitter {rnd} {phase}"))
                res = _resolver(sol, zone, fallback, 4, touch)
                bad, raised = [], []
                gens = [rng(f"thread {rnd} {phase} {k}") for k in range(n_threads)]

                def worker(k, wr=None):
                    wr = gens[k]
                    for i in range(n_ops):
                        if phase == 0 and wr.random() < 0.12:
                            res.advance_clock(wr.randint(0, 3))
                            continue
                        q = ref.spell(wr, wr.choice(pool))
                        touch.local.call = (normalize(q), k, i)
                        try:
                            got = _run(sol, lambda: res.resolve(q))
                        except Exception as e:      # noqa: BLE001
                            raised.append(e)
                            return
                        finally:
                            touch.local.call = None
                        try:
                            _check(sol, got, want[normalize(q)])
                        except Mismatch as m:
                            bad.append((q, got, m))
                            return

                calling("\n".join(_head(zone, fallback, 4) + [
                    f"# {n_threads} 个线程同时在这个 r 上各调用 {n_ops} 次 r.resolve(...)（名字随机，写法各不相同）"
                    + ("，偶尔调用 r.advance_clock(0～3)" if phase == 0 else "，时钟不动")]))
                threads = [ref.start(lambda k=k: worker(k)) for k in range(n_threads)]
                finished = ref.join_all(threads, 8.0)
                expect_true(finished, STUCK.format(k=sum(t.is_alive() for t in threads), s=8))
                if raised:
                    raise raised[0]
                if bad:
                    q, got, m = bad[0]
                    raise Mismatch(f"多个线程同时调用时，r.resolve({lit(q)}) 的结果不对：{m.msg}", None,
                                   Shown(_want_text(want[normalize(q)])), _got_text(got))
                blind &= touch.count == 0
                if phase == 1:
                    for key, out in want.items():
                        walks = sum(1 for tag in touch.calls if tag[0] == key)
                        expect_true(not out.ok or walks <= 1,
                                    f"时钟一直没动，{lit(key)} 却被解析了 {walks} 次：第一次解析成功后应当命中缓存，"
                                    "解析还没结束时后到的同名调用应当等它的结果。" + TOUCH_NOTE)
    if blind:
        info(NO_GATE)
