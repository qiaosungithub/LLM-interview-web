import time

from judgelib import Mismatch, calling, case, expect, expect_close, rng, show_call
import ref

RTOL = 1e-9          # every factor is a product of floats, so it is compared with a tolerance, never with ==

FACTS = [            # the Part 1 example
    ("km", "m", 1000.0),
    ("m", "cm", 100.0),
    ("inch", "cm", 2.54),
    ("foot", "inch", 12.0),
    ("mile", "foot", 5280.0),
    ("byte", "bit", 8.0),
]
FACTS2 = [("a", "b", 2.0), ("c", "d", 5.0), ("b", "c", 3.0), ("a", "d", 100.0)]   # the Part 2 example
MILE_CM = 5280.0 * 12.0 * 2.54
CLOSE = 1 + 4e-10    # inside rel_tol = 1e-9 by a wide margin
FAR = 1 + 2e-8       # outside it by a wide margin


# ---------------------------------------------------------------- comparing one result
def _same(got, want):
    """want is None, a bool (add_fact) or a factor; a factor matches within RTOL, relative to the true value."""
    try:
        if want is None:
            return got is None
        if isinstance(want, bool):
            return got is not None and bool(got == want)
        return (got is not None and not isinstance(got, (bool, str))
                and abs(float(got) - want) <= RTOL * abs(want))
    except Exception:
        return False


def _fail(got, want, call, msg=""):
    """Raises the Mismatch for a result that _same rejected."""
    if want is None and got is not None:
        msg = msg or "应返回 None（有单位未出现过，或两个单位不连通）"
    elif want is not None and not isinstance(want, bool) and got is None:
        msg = msg or "两个单位是连通的，应返回换算结果，却得到 None"
    elif want is True:
        msg = msg or "这条事实与已知的事实不矛盾（或者带来了新的连接），应接受并返回 True"
    elif want is False:
        msg = msg or "这条事实与已知的系数相差超过 rel_tol = 1e-9，应拒绝并返回 False"
    if want is None or isinstance(want, bool):
        expect(got, want, call, msg)
    else:
        expect_close(got, want, rtol=RTOL, atol=0.0, call=call, msg=msg)
    raise Mismatch(msg or "返回值不对", call, want, got)


# ---------------------------------------------------------------- Part 1 and Part 2
def check_convert(sol, facts, value, src, dst, want, text=None):
    call = calling(text or show_call("convert", facts, value, src, dst))
    got = sol.convert(list(facts), value, src, dst)
    if not _same(got, want):
        _fail(got, want, call)


def check_first(sol, facts, want, text=None, **kw):
    call = calling(text or show_call("first_contradiction", facts, **kw))
    got = sol.first_contradiction(list(facts), **kw)
    if want == -1:
        msg = "没有任何一条事实与之前的事实矛盾，应返回 -1"
    else:
        msg = f"第一条与之前的事实矛盾的是下标 {want} 的事实"
    expect(got, want, call, msg)


# ---------------------------------------------------------------- Part 3: how a sequence is shown
SHOW = 16            # a replay lists at most this many calls


def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    return repr(x)


def _note(want):
    return f"# -> 约 {want:.10g}" if isinstance(want, float) else f"# -> {want!r}"


def _replay(done, wants, mark="# <- 这一步"):
    """The calls made so far as code to paste into 自定义运行; the last one is the current step."""
    lines = [f"c.{m}({', '.join(_lit(a) for a in args)})" for m, args in done]
    lines[-1] = f"print({lines[-1]})"
    notes = [_note(w) for w in wants] + [mark]
    skipped = max(0, len(lines) - SHOW)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = ["c = Converter()"] + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(head + [s.ljust(width) + n for s, n in zip(lines, notes)])


def _call(c, method, args):
    return getattr(c, method)(*args)


def run_ops(sol, steps):
    """Hand-written (method, args, want) steps on a fresh Converter, each result checked as soon as it comes back."""
    c = sol.Converter()
    done, wants = [], []
    for method, args, want in steps:
        done.append((method, args))
        calling(_replay(done, wants))
        got = _call(c, method, args)
        if not _same(got, want):
            _fail(got, want, _replay(done, wants))
        wants.append(want)


# ---------------------------------------------------------------- Part 3: random sequences against ref.Model
def _outcome(sol, ops):
    """Replays ops on a fresh answer and a fresh ref.Model; None if every result agrees."""
    c, model, wants = sol.Converter(), ref.Model(), []
    for k, (method, args) in enumerate(ops):
        want = _call(model, method, args)
        try:
            got = _call(c, method, args)
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        wants.append(want)
    return None


def _shrink(sol, ops, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def random_check(sol, sequences, n_units, length):
    r = rng()
    for _ in range(sequences):
        units = [f"u{i}" for i in range(r.randint(*n_units))]
        ops = ref.random_converter_ops(r, units, ref.sizes(r, units), r.randint(*length))
        bad = _outcome(sol, ops)
        if bad is None:
            continue
        ops, bad = _shrink(sol, ops, bad)
        k = bad["k"]
        if bad["exc"] is not None:
            calling(_replay(ops[: k + 1], bad["wants"], mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        _fail(bad["got"], bad["want"], _replay(ops[: k + 1], bad["wants"]),
              f"随机调用序列的第 {k + 1} 步结果不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- Part 3: long sequences
def run_big(sol, steps, text):
    """A long (method, args, want) sequence, shown as the code that builds it; a failure names the step."""
    c = sol.Converter()
    calling(text)
    for k, (method, args, want) in enumerate(steps):
        got = _call(c, method, args)
        if not _same(got, want):
            _fail(got, want, f"{text}\n# 出错的是第 {k + 1} 次调用：c.{method}({', '.join(_lit(a) for a in args)})")


def h(i):            # the hidden size of unit number i in the long sequences
    return 1 + (i * 37 % 101) / 100


# Two chains of 5 * 10^4 facts. Each fact brings one new unit; on the v chain it is the first unit of the fact, on
# the w chain the second, so a union-find that always hangs one fixed side's root under the other's builds one
# chain as a single path (whichever side it picks).
N_CHAIN = 50_000
CHAIN_QUERIES = [("v0", f"v{N_CHAIN}"), (f"v{N_CHAIN}", "v0"), ("w0", f"w{N_CHAIN}"), (f"w{N_CHAIN}", "w0"),
                 ("v31416", "v2718"), ("w2718", "w31416"), ("v0", "w0")]


def _hidden(unit):
    return h(int(unit[1:]))


CHAIN_STEPS = []
for _i in range(N_CHAIN):
    CHAIN_STEPS.append(("add_fact", (f"v{_i + 1}", f"v{_i}", h(_i + 1) / h(_i)), True))
    CHAIN_STEPS.append(("add_fact", (f"w{_i}", f"w{_i + 1}", h(_i) / h(_i + 1)), True))
CHAIN_STEPS += [("query", (a, b), _hidden(a) / _hidden(b) if a[0] == b[0] else None) for a, b in CHAIN_QUERIES]
CHAIN_TEXT = "\n".join([
    "c = Converter()",
    "h = lambda i: 1 + (i * 37 % 101) / 100",
    f"for i in range({N_CHAIN}):",
    '    c.add_fact(f"v{i + 1}", f"v{i}", h(i + 1) / h(i))   # 每次都应返回 True',
    '    c.add_fact(f"w{i}", f"w{i + 1}", h(i) / h(i + 1))   # 每次都应返回 True',
] + [f'print(c.query("{a}", "{b}"))   # 应约为 h({a[1:]}) / h({b[1:]})' if a[0] == b[0]
     else f'print(c.query("{a}", "{b}"))   # 应为 None' for a, b in CHAIN_QUERIES])

# A random tree of 3 * 10^4 facts, each followed by a query between two units of the tree.
N_MANY = 30_000
MANY_STEPS = []
for _i in range(1, N_MANY + 1):
    _p = _i * 7919 % 100003 % _i
    _a, _b = _i * 7907 % 100019 % (_i + 1), _i * 7883 % 100043 % (_i + 1)
    MANY_STEPS.append(("add_fact", (f"u{_i}", f"u{_p}", h(_i) / h(_p)), True))
    MANY_STEPS.append(("query", (f"u{_a}", f"u{_b}"), h(_a) / h(_b)))
MANY_TEXT = "\n".join([
    "c = Converter()",
    "h = lambda i: 1 + (i * 37 % 101) / 100",
    f"for i in range(1, {N_MANY + 1}):",
    "    p = i * 7919 % 100003 % i",
    "    a, b = i * 7907 % 100019 % (i + 1), i * 7883 % 100043 % (i + 1)",
    '    c.add_fact(f"u{i}", f"u{p}", h(i) / h(p))   # 每次都应返回 True',
    '    c.query(f"u{a}", f"u{b}")                   # 应约为 h(a) / h(b)',
])


# ---------------------------------------------------------------- medium inputs for Parts 1 and 2
_med = rng("medium")
MED_UNITS = [f"c{i}" for i in range(2000)]
MED_SIZE = ref.sizes(_med, MED_UNITS, 0.5, 2.0)
MED_CHAIN = [(MED_UNITS[i], MED_UNITS[i + 1], MED_SIZE[MED_UNITS[i]] / MED_SIZE[MED_UNITS[i + 1]])
             for i in range(len(MED_UNITS) - 1)]
_med.shuffle(MED_CHAIN)


def medium_contradiction(r):
    """About 300 consistent facts over 120 units in 1 to 3 groups, and (usually) one contradicting fact placed
    where its two units are already connected; returns (facts, expected index)."""
    units = [f"w{i}" for i in range(120)]
    size = ref.sizes(r, units)
    facts = ref.consistent_facts(r, size, ref.split_units(r, units, r.randint(1, 3)), 180)
    if r.random() < 0.2:
        return facts, -1
    i = r.randrange(len(facts) // 2, len(facts))
    dsu = ref.Dsu()
    for a, b, _ in facts[:i]:
        dsu.union(a, b)
    a, b = r.choice([(x, y) for x in units for y in units if x != y and dsu.connected(x, y)])
    return facts[:i] + [(a, b, size[a] / size[b] * ref.far_factor(r))] + facts[i:], i


MED_FIRST = [medium_contradiction(_med) for _ in range(6)]


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    check_convert(sol, FACTS, 2.0, "mile", "cm", 321868.8)
    check_convert(sol, FACTS, 5.0, "km", "km", 5.0)
    check_convert(sol, FACTS, 1.0, "m", "parsec", None)
    check_convert(sol, FACTS, 1.0, "km", "byte", None)


@case("Part 1", title="反方向换算：乘 1 / r")
def p1_backwards(sol):
    check_convert(sol, FACTS, 321868.8, "cm", "mile", 2.0)
    check_convert(sol, FACTS, 1.0, "cm", "km", 1e-5)
    check_convert(sol, FACTS, 16.0, "bit", "byte", 2.0)
    check_convert(sol, FACTS, 1.0, "inch", "foot", 1 / 12)
    check_convert(sol, FACTS, 1.0, "m", "inch", 100 / 2.54)
    check_convert(sol, FACTS, 3.0, "km", "mile", 3.0 * 1e5 / MILE_CM)


@case("Part 1", title="未知单位、不连通：返回 None")
def p1_unknown(sol):
    check_convert(sol, FACTS, 1.0, "parsec", "m", None)
    check_convert(sol, FACTS, 1.0, "gram", "gram", None)
    check_convert(sol, [], 1.0, "x", "x", None)
    check_convert(sol, FACTS, 1.0, "bit", "km", None)
    check_convert(sol, FACTS, 1.0, "KM", "m", None)
    check_convert(sol, FACTS, 3.0, "bit", "bit", 3.0)
    check_convert(sol, FACTS, 3.0, "cm", "cm", 3.0)


@case("Part 1", title="value 为 0 或负数")
def p1_value_sign(sol):
    check_convert(sol, FACTS, 0.0, "mile", "cm", 0.0)
    check_convert(sol, FACTS, -2.0, "mile", "cm", -321868.8)
    check_convert(sol, FACTS, -0.5, "byte", "bit", -4.0)


@case("Part 1", title="重复、反向重复、成环的事实")
def p1_repeated_facts(sol):
    facts = [("km", "m", 1000.0), ("m", "km", 0.001), ("km", "m", 1000.0), ("km", "cm", 1e5), ("m", "cm", 100.0)]
    check_convert(sol, facts, 1.0, "km", "cm", 1e5)
    check_convert(sol, facts, 1.0, "cm", "km", 1e-5)
    check_convert(sol, facts, 7.0, "m", "km", 0.007)


@case("Part 1", title="各种单位名")
def p1_unit_names(sol):
    facts = [("公里", "米", 1000.0), ("米", "a b", 4.0), ("a", "ab", 2.0), ("ab", "abc", 3.0), ("1", "2", 0.5)]
    check_convert(sol, facts, 1.0, "公里", "a b", 4000.0)
    check_convert(sol, facts, 1.0, "abc", "a", 1 / 6)
    check_convert(sol, facts, 1.0, "2", "1", 2.0)
    check_convert(sol, facts, 1.0, "a", "米", None)
    check_convert(sol, facts, 1.0, "a", "b", None)


@case("Part 1", title="随机事实（对照隐藏的真实大小）")
def p1_random(sol):
    r = rng()
    for _ in range(150):
        units = [f"u{i}" for i in range(r.randint(2, 10))]
        size = ref.sizes(r, units)
        facts = ref.consistent_facts(r, size, ref.split_units(r, units, r.randint(1, 3)), r.randint(0, 4))
        dsu = ref.Dsu()
        for a, b, _ in facts:
            dsu.union(a, b)
        for _ in range(6):
            src, dst = r.choice(units + ["ghost"]), r.choice(units + ["ghost"])
            value = r.choice([1.0, r.uniform(-5.0, 5.0)])
            want = value * size[src] / size[dst] if dsu.connected(src, dst) else None
            check_convert(sol, facts, value, src, dst, want)


@case("Part 1", title="2000 个单位连成一条链")
def p1_long_chain(sol):
    for a, b in ((0, 1999), (1999, 0), (500, 1500), (1234, 1233)):
        src, dst = MED_UNITS[a], MED_UNITS[b]
        check_convert(sol, MED_CHAIN, 1.0, src, dst, MED_SIZE[src] / MED_SIZE[dst],
                      text=f'convert(<2000 个单位连成一条链的 1999 条事实，顺序打乱>, 1.0, "{src}", "{dst}")')


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    check_first(sol, FACTS2, 3)


@case("Part 2", title="没有矛盾：返回 -1")
def p2_none(sol):
    check_first(sol, FACTS2[:3], -1)
    check_first(sol, [], -1)
    check_first(sol, [("a", "b", 2.0)], -1)
    check_first(sol, [("a", "b", 2.0), ("b", "c", 3.0), ("c", "a", 1 / 6)], -1)
    check_first(sol, [("a", "b", 2.0), ("a", "b", 2.0), ("b", "a", 0.5)], -1)
    check_first(sol, FACTS + [("km", "inch", 1e5 / 2.54), ("mile", "m", MILE_CM / 100)], -1)


@case("Part 2", title="连接两个分量的事实不会矛盾")
def p2_joining(sol):
    check_first(sol, [("a", "b", 2.0), ("c", "d", 5.0), ("b", "c", 123.0), ("x", "y", 1e9)], -1)
    check_first(sol, [("a", "b", 2.0), ("c", "d", 5.0), ("b", "c", 123.0), ("a", "d", 2.0 * 123.0 * 5.0)], -1)
    check_first(sol, [("a", "b", 2.0), ("c", "d", 5.0), ("b", "c", 123.0), ("a", "d", 30.0)], 3)


@case("Part 2", title="返回第一条矛盾")
def p2_first_one(sol):
    check_first(sol, [("a", "b", 2.0), ("b", "a", 2.0), ("a", "b", 3.0)], 1)
    check_first(sol, [("a", "b", 2.0), ("c", "d", 3.0), ("c", "d", 3.0), ("b", "a", 0.25), ("a", "b", 4.0)], 3)
    check_first(sol, [("a", "b", 2.0), ("b", "c", 2.0), ("c", "d", 2.0), ("d", "e", 2.0), ("e", "a", 1 / 16),
                      ("a", "e", 15.0), ("a", "c", 5.0)], 5)


@case("Part 2", title="rel_tol 的边界附近")
def p2_tolerance(sol):
    check_first(sol, [("p", "q", 2.0), ("p", "q", 2.0 * CLOSE)], -1)
    check_first(sol, [("p", "q", 2.0), ("p", "q", 2.0 * FAR)], 1)
    check_first(sol, [("p", "q", 2.0), ("q", "p", 0.5 * CLOSE)], -1)
    check_first(sol, [("p", "q", 2.0), ("q", "p", 0.5 / FAR)], 1)


@case("Part 2", title="很大和很小的系数：相对误差")
def p2_relative_scale(sol):
    check_first(sol, [("a", "b", 1e-12), ("b", "c", 1e-12), ("a", "c", 2e-24)], 2)
    check_first(sol, [("a", "b", 1e-6), ("a", "b", 3e-6)], 1)
    check_first(sol, [("ly", "km", 9.4607e12), ("km", "mm", 1e6), ("ly", "mm", 9.4607e18)], -1)
    check_first(sol, [("x", "y", 3e15), ("y", "z", 7e14), ("x", "z", 2.1e30 * CLOSE)], -1)


@case("Part 2", title="传入的 rel_tol")
def p2_custom_rel_tol(sol):
    facts = [("a", "b", 2.0), ("b", "c", 3.0), ("a", "c", 5.0)]
    check_first(sol, facts, -1, rel_tol=0.25)
    check_first(sol, facts, 2, rel_tol=0.1)
    # |6 - 5| / max(6, 5) ≈ 0.167 <= 0.18, so not a contradiction (|6 - 5| / 5 = 0.2 would wrongly say it is)
    check_first(sol, facts, -1, rel_tol=0.18)
    check_first(sol, FACTS2, -1, rel_tol=0.75)
    check_first(sol, FACTS2, 3, rel_tol=0.6)
    check_first(sol, [("p", "q", 2.0), ("p", "q", 2.0 * CLOSE)], 1, rel_tol=1e-12)


@case("Part 2", title="随机事实（对照逐条暴力检查）")
def p2_random(sol):
    r = rng()
    for _ in range(150):
        units = [f"u{i}" for i in range(r.randint(2, 8))]
        size = ref.sizes(r, units)
        facts = ref.consistent_facts(r, size, ref.split_units(r, units, r.randint(1, 3)), r.randint(0, 4))
        for _ in range(r.randint(0, 2)):
            a, b = r.sample(units, 2)
            facts.insert(r.randint(0, len(facts)), (a, b, size[a] / size[b] * ref.far_factor(r)))
        check_first(sol, facts, ref.first_contradiction(facts))


@case("Part 2", title="约 300 条事实")
def p2_medium(sol):
    for facts, want in MED_FIRST:
        check_first(sol, facts, want, text=f"first_contradiction(<{len(facts)} 条事实>)")


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    run_ops(sol, [
        ("add_fact", ("km", "m", 1000.0), True),
        ("add_fact", ("m", "cm", 100.0), True),
        ("query", ("km", "cm"), 100000.0),
        ("add_fact", ("inch", "cm", 2.54), True),
        ("query", ("km", "inch"), 1e5 / 2.54),
        ("add_fact", ("km", "inch", 39370.0), False),
        ("query", ("km", "inch"), 1e5 / 2.54),
        ("add_fact", ("mile", "foot", 5280.0), True),
        ("query", ("mile", "cm"), None),
        ("add_fact", ("foot", "inch", 12.0), True),
        ("query", ("mile", "cm"), MILE_CM),
    ])


@case("Part 3", title="空的 Converter；同一个单位")
def p3_fresh(sol):
    run_ops(sol, [
        ("query", ("km", "m"), None),
        ("query", ("x", "x"), None),
        ("add_fact", ("x", "y", 2.0), True),
        ("query", ("x", "x"), 1.0),
        ("query", ("y", "x"), 0.5),
        ("query", ("x", "z"), None),
        ("query", ("z", "z"), None),
    ])


@case("Part 3", title="重复已知的事实：接受")
def p3_restate(sol):
    run_ops(sol, [
        ("add_fact", ("a", "b", 2.0), True),
        ("add_fact", ("a", "b", 2.0), True),
        ("add_fact", ("b", "a", 0.5), True),
        ("add_fact", ("b", "c", 3.0), True),
        ("add_fact", ("a", "c", 6.0), True),
        ("add_fact", ("c", "a", 1 / 6), True),
        ("add_fact", ("a", "c", 5.0), False),
        ("add_fact", ("c", "a", 0.2), False),
        ("query", ("c", "a"), 1 / 6),
    ])


@case("Part 3", title="被拒绝的事实不改变任何东西")
def p3_rejected(sol):
    run_ops(sol, [
        ("add_fact", ("a", "b", 2.0), True),
        ("add_fact", ("b", "c", 3.0), True),
        ("add_fact", ("a", "c", 7.0), False),
        ("query", ("a", "c"), 6.0),
        ("query", ("c", "a"), 1 / 6),
        ("add_fact", ("c", "d", 5.0), True),
        ("query", ("a", "d"), 30.0),
        ("add_fact", ("d", "a", 1 / 30), True),
        ("add_fact", ("a", "d", 31.0), False),
        ("query", ("d", "b"), 1 / 15),
    ])


@case("Part 3", title="合并大小不同的分量，两个方向")
def p3_merge_orders(sol):
    run_ops(sol, [
        ("add_fact", ("a1", "a2", 2.0), True),
        ("add_fact", ("a2", "a3", 3.0), True),
        ("add_fact", ("a3", "a4", 5.0), True),
        ("add_fact", ("b1", "a3", 7.0), True),       # a new unit first, the big group second
        ("query", ("a1", "b1"), 6 / 7),
        ("add_fact", ("a2", "c1", 11.0), True),      # the big group first
        ("query", ("c1", "b1"), 3 / 77),
        ("add_fact", ("d1", "d2", 0.5), True),
        ("add_fact", ("d2", "a4", 4.0), True),       # a group of two first
        ("query", ("d1", "a1"), 2 / 30),
        ("add_fact", ("e1", "e2", 10.0), True),
        ("add_fact", ("a4", "e2", 0.1), True),       # a group of two second
        ("query", ("e1", "a1"), 100 / 30),
        ("query", ("c1", "e1"), 15 / 11 * 0.01),
        ("query", ("d1", "e1"), 0.02),
    ])


@case("Part 3", title="rel_tol = 1e-9 的边界附近")
def p3_tolerance(sol):
    run_ops(sol, [
        ("add_fact", ("p", "q", 2.0), True),
        ("add_fact", ("p", "q", 2.0 * CLOSE), True),
        ("add_fact", ("p", "q", 2.0 * FAR), False),
        ("add_fact", ("q", "p", 0.5 * CLOSE), True),
        ("add_fact", ("q", "p", 0.5 / FAR), False),
        ("query", ("q", "p"), 0.5),
    ])


@case("Part 3", title="很大和很小的系数：相对误差")
def p3_relative_scale(sol):
    run_ops(sol, [
        ("add_fact", ("a", "b", 1e-12), True),
        ("add_fact", ("b", "c", 1e-12), True),
        ("add_fact", ("a", "c", 2e-24), False),      # implied 1e-24: off by a factor of 2
        ("query", ("a", "c"), 1e-24),
        ("add_fact", ("x", "y", 3e15), True),
        ("add_fact", ("y", "z", 7e14), True),
        ("add_fact", ("x", "z", 2.1e30 * CLOSE), True),   # relative error 4e-10, inside rel_tol
        ("add_fact", ("z", "x", 1 / 2.1e30), True),
        ("query", ("z", "x"), 1 / 2.1e30),
    ])


@case("Part 3", title="随机交替调用（对照暴力）")
def p3_random(sol):
    random_check(sol, sequences=80, n_units=(2, 9), length=(10, 50))


@case("Part 3", cpu=3, title="10^5 次 add_fact 连成长链")
def p3_long_chain(sol):
    run_big(sol, CHAIN_STEPS, CHAIN_TEXT)


@case("Part 3", cpu=2, title="3×10^4 次 add_fact 与 query 交替")
def p3_many_ops(sol):
    run_big(sol, MANY_STEPS, MANY_TEXT)
