import ast
import textwrap
import time

from judgelib import Mismatch, calling, case, expect, expect_raises, expect_true, rng, short
import ref

SHOW = 16           # a replay lists at most this many steps


# ---------------------------------------------------------------- how steps are shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    if isinstance(x, list):
        return "[" + ", ".join(_lit(v) for v in x) + "]"
    if isinstance(x, tuple):
        return "(" + ", ".join(_lit(v) for v in x) + ("," if len(x) == 1 else "") + ")"
    return repr(x)


def _note(x, limit=150):
    text = _lit(x)
    return text if len(text) <= limit else text[:limit] + " …"


def _lines(head, done, current, show=SHOW):
    """head (always shown), then the last `show` lines of done, then the current step."""
    skipped = max(0, len(done) - show)
    gap = [f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else []
    return "\n".join(list(head) + gap + done[skipped:] + [current])


def _same(got, want):
    if want is None:
        return got is None
    try:
        return bool(got == want)
    except Exception:
        return False


def _what(code):
    """What a checked expression is, for the message when the script gives none."""
    c = code.replace(" ", "")
    if c.startswith("len("):
        return "len 不对"
    if ".get(" in c:
        return "get 返回的值不对"
    if ".delete(" in c:
        return "delete 的返回值不对：应当返回这个键原来是否在表里"
    if c.endswith("inm"):
        return "in 的结果不对"
    if c.startswith("next("):
        return "next(it) 取到的元素不对"
    if "items()" in c or c.startswith("list(") or c.startswith("[kforkin"):
        return "遍历结果不对"
    return "返回值不对"


def _fail(got, want, call, msg, code):
    raise Mismatch(msg or _what(code), call, "None" if want is None else want, got)


# ---------------------------------------------------------------- statement-style scripts

def run_script(sol, script, prelude=""):
    """Runs a script written like the statement's examples, one line at a time, on the answer's HashMap. A line
    `expr -> want` checks the value of expr (`-> raise RuntimeError` checks that it raises); any other line is run as
    it is. The replay shown on a failure is the prelude plus the lines so far, ready to paste into 自定义运行."""
    ns = {"HashMap": sol.HashMap}
    head = textwrap.dedent(prelude).strip().splitlines() if prelude.strip() else []
    if head:
        exec(compile(textwrap.dedent(prelude), "<test>", "exec"), ns)
    done = []
    for raw in textwrap.dedent(script).strip().splitlines():
        if not raw.strip():
            continue
        code, arrow, rest = raw.partition(" -> ")
        code = code.rstrip()
        if not arrow:
            calling(_lines(head, done, f"{code}   # <- 这一步"))
            exec(compile(code, "<test>", "exec"), ns)
            done.append(code)
            continue
        want_src, _, why = rest.partition("#")
        want_src, why = want_src.strip(), why.strip()
        if want_src.startswith("raise "):
            exc = {"RuntimeError": RuntimeError}[want_src[len("raise "):].strip()]
            call = calling(_lines(head, done, f"{code}   # <- 这一步，应当抛出 {exc.__name__}"))
            expect_raises(exc, eval, compile(code, "<test>", "eval"), ns, call=call, msg=why)
            done.append(f"{code}   # -> 抛出 {exc.__name__}")
            continue
        want = ast.literal_eval(want_src)
        call = calling(_lines(head, done, f"print({code})   # <- 这一步，应当是 {want_src}"))
        got = eval(compile(code, "<test>", "eval"), ns)
        if not _same(got, want):
            _fail(got, want, call, why, code)
        done.append(f"{code}   # -> {want_src}")


# ---------------------------------------------------------------- operation sequences checked against a model
#
# An operation is one of
#   ("put", k, v)   m.put(k, v)          ("del", k)   m.delete(k)
#   ("get", k)      m.get(k)             ("len",)     len(m)
#   ("getd", k, d)  m.get(k, d)          ("list",)    list(m)
#   ("in", k)       k in m               ("items",)   [tuple(p) for p in m.items()]

def _op_code(op):
    kind = op[0]
    if kind == "put":
        return f"m.put({_lit(op[1])}, {_lit(op[2])})"
    if kind == "get":
        return f"m.get({_lit(op[1])})"
    if kind == "getd":
        return f"m.get({_lit(op[1])}, {_lit(op[2])})"
    if kind == "in":
        return f"{_lit(op[1])} in m"
    if kind == "del":
        return f"m.delete({_lit(op[1])})"
    if kind == "len":
        return "len(m)"
    if kind == "list":
        return "list(m)"
    return "[tuple(p) for p in m.items()]"


def _fresh(key):
    """A new object equal to key for every call, so that the map never holds the test's own list."""
    return list(key) if isinstance(key, list) else key


def _call_op(m, op):
    kind = op[0]
    if kind == "put":
        return m.put(_fresh(op[1]), op[2])
    if kind == "get":
        return m.get(_fresh(op[1]))
    if kind == "getd":
        return m.get(_fresh(op[1]), op[2])
    if kind == "in":
        return _fresh(op[1]) in m
    if kind == "del":
        return m.delete(_fresh(op[1]))
    if kind == "len":
        return len(m)
    if kind == "list":
        return list(m)
    return [tuple(p) for p in m.items()]


class Expect:
    """Expected results: a ref.Model for the values, and with exact=True a ref.Layout for the iteration order (only
    valid while no key has been deleted). mkey turns a returned key into something hashable for the order-free
    comparison of list(m) and items()."""

    def __init__(self, canon=None, exact=False, hash_fn=hash, eq_fn=None, mkey=None):
        self.model = ref.Model(canon)
        self.layout = ref.Layout(hash_fn, eq_fn) if exact else None
        self.start = "hash(key)" if hash_fn is hash else "hash_fn(key)"
        self.mkey = mkey or (lambda k: k)

    def apply(self, op):
        want = ref.apply_model(self.model, op)
        if self.layout is not None:
            if op[0] == "put":
                self.layout.put(op[1], op[2])
            elif op[0] == "list":
                want = self.layout.order()
            elif op[0] == "items":
                want = self.layout.items()
        return want

    def agrees(self, op, got, want):
        kind = op[0]
        if kind == "put":
            return True
        if kind in ("list", "items") and self.layout is None:
            return self._same_multiset(got, want, kind == "items")
        return _same(got, want)

    def _same_multiset(self, got, want, pairs):
        key = (lambda p: (self.mkey(p[0]), p[1])) if pairs else self.mkey
        try:
            count = {}
            for x in got:
                c = key(x)
                count[c] = count.get(c, 0) + 1
            for x in want:
                c = key(x)
                count[c] = count.get(c, 0) - 1
            return all(v == 0 for v in count.values())
        except Exception:
            return False

    def message(self, op):
        kind = op[0]
        if kind in ("list", "items"):
            if self.layout is not None:
                return (f"没有删除过键时，遍历顺序应当是槽的顺序（按题目的布局规则：起始槽 {self.start} & (capacity - 1)、"
                        "线性探测、插入第几个新键之前扩容）")
            return "遍历结果应当恰好包含表里的每个键各一次（有删除之后顺序不限）"
        return {"get": "get 返回的值不对", "getd": "get 返回的值不对", "in": "in 的结果不对",
                "del": "delete 的返回值不对：应当返回这个键原来是否在表里", "len": "len 不对"}[kind]


def _make(sol, head):
    ns = {"HashMap": sol.HashMap}
    exec(compile("\n".join(head), "<test>", "exec"), ns)
    return ns["m"]


def _replay(head, ops, wants, show=SHOW):
    done = [_op_code(op) + ("" if op[0] == "put" else f"   # -> {_note(w)}") for op, w in zip(ops, wants)]
    last = _op_code(ops[-1])
    if ops[-1][0] != "put":
        last = f"print({last})"
    return _lines(head, done, f"{last}   # <- 这一步", show)


def _outcome(sol, head, ops, expect_factory):
    """Replays ops on a fresh map; None if every result agrees with the expectation."""
    m, exp, wants = _make(sol, head), expect_factory(), []
    for k, op in enumerate(ops):
        want = exp.apply(op)
        try:
            got = _call_op(m, op)
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        except BaseException:
            calling(_replay(head, ops[: k + 1], wants))      # out of time inside this call
            raise
        if not exp.agrees(op, got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants, "msg": exp.message(op)}
        wants.append(want)
    return None


def _shrink(sol, head, ops, bad, expect_factory, budget=0.6):
    """Drops steps that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, head, cand, expect_factory) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def check_ops(sol, head, ops, expect_factory, what=""):
    """Runs ops and fails with a (shortened) replay at the first result that differs from the expectation."""
    bad = _outcome(sol, head, ops, expect_factory)
    if bad is None:
        return
    calling(_replay(head, ops[: bad["k"] + 1], bad["wants"]))   # shown if shrinking runs into a hang
    ops, bad = _shrink(sol, head, ops, bad, expect_factory)
    k = bad["k"]
    call = _replay(head, ops[: k + 1], bad["wants"], show=60)       # shortened: show all of it if possible
    if bad["exc"] is not None:
        calling(call)
        raise bad["exc"]
    note = f"（{what}，已自动删去与出错无关的步骤）" if what else ""
    want = bad["want"]
    raise Mismatch(f"第 {k + 1} 步：{bad['msg']}{note}", call, "None" if want is None else want, bad["got"])


def random_check(sol, head, sequences, make_ops, expect_factory):
    r = rng()
    for _ in range(sequences):
        check_ops(sol, head, make_ops(r), expect_factory, what="随机操作序列")


NEW = ["m = HashMap()"]

# Big inputs are drawn once at import time, which is not charged to any case.
N_BIG = 200_000
_big = rng("big")
_drawn = set()
while len(_drawn) < N_BIG + 100_000:
    _drawn.add(_big.getrandbits(62))
_drawn = list(_drawn)
_big.shuffle(_drawn)
BIG_KEYS, ABSENT_KEYS = _drawn[:N_BIG], _drawn[N_BIG:]
del _drawn


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    run_script(sol, """
        m = HashMap()
        m.put("a", 1); m.put("b", 2); m.put("a", 3)
        len(m) -> 2
        m.get("a") -> 3
        m.get("zzz") -> None
        m.get("zzz", 0) -> 0
        "b" in m -> True
        m.put(1, "int"); m.put(1.0, "float")     # 1 == 1.0 且 hash(1) == hash(1.0)：是同一个键
        len(m) -> 3
        m.get(1) -> "float"
        m.get(True) -> "float"   # True == 1，同理
    """)


@case("Part 1", title="None、0、空串也是普通的键和值")
def p1_none_and_falsy(sol):
    run_script(sol, """
        m = HashMap()
        m.put(None, "none")
        None in m -> True   # None 是一个普通的键
        m.get(None) -> "none"
        m.put(0, "zero"); m.put("", "empty"); m.put((), "tuple")
        len(m) -> 4   # 4 个不同的键（hash("") 和 hash(0) 都是 0，但 "" != 0）
        m.get(0) -> "zero"
        m.get("") -> "empty"
        m.get(()) -> "tuple"
        m.put("v", None)
        m.get("v", "缺省") -> None   # 存的值就是 None，这时不能返回 default
        "v" in m -> True
        m.get("w", "缺省") -> "缺省"
        m.put(None, None)
        m.get(None, "缺省") -> None
        len(m) -> 5
        m.put(False, "false")    # False == 0：覆盖
        len(m) -> 5
        m.get(0) -> "false"
    """)


@case("Part 1", title="a is b 也算同一个键（NaN）")
def p1_identity(sol):
    run_script(sol, """
        nan = float("nan")
        m = HashMap()
        m.put(nan, 1)
        m.get(nan) -> 1   # 同一个对象：a is b，虽然 nan == nan 不成立
        m.put(nan, 2)
        len(m) -> 1   # 同一个对象：a is b，应当覆盖
        m.get(nan) -> 2
        float("nan") in m -> False   # 另一个 nan 对象：不是同一个对象，也不 ==
        m.put(0, "a"); m.put(0.0, "b"); m.put(-0.0, "c")    # 0 == 0.0 == -0.0，哈希也相同
        len(m) -> 2
        m.get(0) -> "c"
    """)


K_CLASS = '''
class K:
    """Every K has hash 42; two K are equal when their names are."""
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return 42
    def __eq__(self, other):
        return isinstance(other, K) and self.name == other.name
    def __repr__(self):
        return f"K({self.name!r})"
'''


@case("Part 1", title="哈希相同但不相等的键要分开存")
def p1_same_hash(sol):
    run_script(sol, """
        m = HashMap()
        m.put(-1, "a"); m.put(-2, "b")      # hash(-1) == hash(-2) == -2，但 -1 != -2
        len(m) -> 2   # 哈希相同只说明可能是同一个键，还要用 == 判断
        m.get(-1) -> "a"
        m.get(-2) -> "b"
        m = HashMap()
        for i in range(40): m.put(K(i), i)
        len(m) -> 40
        [m.get(K(i)) for i in range(40)] == list(range(40)) -> True
        K(40) in m -> False
        m.put(K(7), "seven")
        len(m) -> 40
        m.get(K(7)) -> "seven"
    """, prelude=K_CLASS)


@case("Part 1", title="冲突连成长段，并绕过表尾")
def p1_collisions(sol):
    keys = [7, 15, 23, 31, 39] + [i << 40 for i in range(1, 300)] + [(i << 40) + 7 for i in range(100)]
    ops = []
    for i, k in enumerate(keys):
        ops += [("put", k, i), ("get", k)]
    ops += [("get", k) for k in keys] + [("len",)]
    ops += [("getd", k, "缺省") for k in [47, 6, 0, 1 << 41 | 1] + [(i << 40) + 3 for i in range(50)]]
    ops += [("in", k) for k in [55, (300 << 40), 8]]
    check_ops(sol, NEW, ops, Expect)


@case("Part 1", title="扩容那一次插入的新键也能找到")
def p1_growth(sol):
    r = rng()
    for trial in range(4):
        keys = r.sample(range(10 ** 6), 700)
        layout, inserted, ops = ref.Layout(), [], []
        for i, k in enumerate(keys):
            before = layout.capacity
            layout.put(k, i)
            inserted.append(k)
            ops += [("put", k, i), ("get", k), ("len",)]
            if layout.capacity != before:                  # this put doubled the table: every key must still be there
                ops += [("get", x) for x in inserted]
        check_ops(sol, NEW, ops, Expect)


@case("Part 1", title="随机操作与 dict 对比（没有删除）")
def p1_random(sol):
    def make(r):
        return ref.random_ops(r, ref.pool_default(r, r.choice([3, 10, 40, 150])), r.randint(1, 120), deletes=False)
    random_check(sol, NEW, 300, make, Expect)


@case("Part 1", cpu=8, title="2×10⁵ 个键：要求均摊期望 O(1)")
def p1_large(sol):
    desc = "m = HashMap()\nfor i, k in enumerate(KEYS): m.put(k, i)    # KEYS：2×10⁵ 个不同的随机整数，0 ≤ k < 2⁶²"
    calling(desc)
    m = sol.HashMap()
    put = m.put
    for i, k in enumerate(BIG_KEYS):
        put(k, i)
    call = calling(desc + "\nlen(m)")
    expect(len(m), N_BIG, call=call)
    call = calling(desc + "\n[m.get(k) for k in KEYS]")
    get = m.get
    got = [get(k) for k in BIG_KEYS]
    if got != list(range(N_BIG)):
        i = next(i for i, v in enumerate(got) if not _same(v, i))
        raise Mismatch(f"m.get(KEYS[{i}]) 不对（KEYS[{i}] = {BIG_KEYS[i]}）", call, i, got[i])
    call = calling(desc + "\n[m.get(k, -1) for k in ABSENT]    # ABSENT：10⁵ 个不在表里的随机整数")
    got = [get(k, -1) for k in ABSENT_KEYS]
    if got.count(-1) != len(got):
        i = next(i for i, v in enumerate(got) if not _same(v, -1))
        raise Mismatch(f"m.get(ABSENT[{i}], -1) 不对（ABSENT[{i}] = {ABSENT_KEYS[i]} 不在表里）", call, -1, got[i])


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    run_script(sol, """
        m = HashMap()
        for k in (0, 8, 16): m.put(k, str(k))    # 小整数的 hash 就是它自己，这三个键的起始槽都是 0
        m.delete(8) -> True
        m.get(16) -> "16"
        m.delete(8) -> False
        len(m) -> 2
        m.put(8, "again")
        m.get(8) -> "again"
        len(m) -> 3
    """)


@case("Part 2", title="删掉段首或段中的键，后面的键还在")
def p2_delete_in_run(sol):
    run_script(sol, """
        m = HashMap()
        for k in (7, 15, 23, 31): m.put(k, k)    # 起始槽都是 7：依次落在槽 7、0、1、2
        m.delete(7) -> True
        m.get(15) -> 15
        m.get(23) -> 23
        m.get(31) -> 31
        m.delete(23) -> True
        m.get(31) -> 31
        23 in m -> False
        m.delete(23) -> False
        len(m) -> 2
        m.put(1, "one")    # 起始槽 1，那里原来是 23
        m.get(1) -> "one"
        m.delete(15) -> True
        m.get(31) -> 31
        m.get(1) -> "one"
        m.delete(1.0) -> True   # 1.0 == 1：是同一个键
        1 in m -> False
        len(m) -> 1
        m.put(7, "back"); m.put(23, "back")
        m.get(31) -> 31
        m.get(7) -> "back"
        m.get(23) -> "back"
        len(m) -> 3
    """)


@case("Part 2", title="删除后再存同一个键：不能存两份")
def p2_reput(sol):
    run_script(sol, """
        m = HashMap()
        for k in (0, 8, 16): m.put(k, str(k))    # 依次落在槽 0、1、2
        m.delete(0) -> True
        m.put(16, "x")     # 16 还在表里（槽 2）：只能覆盖它的值
        len(m) -> 2   # 16 被存了两份
        m.get(16) -> "x"
        m.delete(16) -> True
        m.get(16) -> None   # 删掉 16 之后它不能还在表里
        16 in m -> False
        len(m) -> 1
        m.get(8) -> "8"
    """)


@case("Part 2", title="反复插入新键、删除旧键不会死循环")
def p2_no_hang(sol):
    code = ("m = HashMap()\nfor k in range(3): m.put(k, k)\nfor k in range(3, 3000):\n"
            "    m.put(k, k)\n    m.delete(k - 3)\n    m.get(-k)\n    -k in m")
    calling(code)
    m = sol.HashMap()
    k, what = None, "m.put(k, k)"
    try:
        for k in range(3):
            m.put(k, k)
        for k in range(3, 3000):
            what = "m.put(k, k)"
            m.put(k, k)
            what = "m.delete(k - 3)"
            got = m.delete(k - 3)
            if not _same(got, True):
                raise Mismatch(f"k = {k} 这一轮：m.delete(k - 3) 应当返回 True", f"{code}\n# 出错的是 k = {k} 这一轮的 {what}",
                               True, got)
            what = "m.get(-k)"
            got = m.get(-k)
            if got is not None:
                raise Mismatch(f"k = {k} 这一轮：-k 不在表里，m.get(-k) 应当返回 None",
                               f"{code}\n# 出错的是 k = {k} 这一轮的 {what}", "None", got)
            what = "-k in m"
            got = -k in m
            if got is not False:
                raise Mismatch(f"k = {k} 这一轮：-k 不在表里", f"{code}\n# 出错的是 k = {k} 这一轮的 {what}", False, got)
    except Mismatch:
        raise
    except BaseException:
        calling(f"{code}\n# 出错或卡住的是 k = {k} 这一轮的 {what}")
        raise
    call = calling(code + "\nlen(m), m.get(2997), m.get(2998), m.get(2999)")
    expect((len(m), m.get(2997), m.get(2998), m.get(2999)), (3, 2997, 2998, 2999), call=call)


@case("Part 2", title="探测段跨过表尾时删除")
def p2_wrap(sol):
    run_script(sol, """
        m = HashMap()
        for k in (6, 7, 15): m.put(k, k)    # 落在槽 6、7；15 的起始槽是 7，绕到槽 0
        m.delete(6) -> True
        m.get(15) -> 15
        m.get(7) -> 7
        m = HashMap()
        for k in (6, 7, 8, 16): m.put(k, k)    # 槽 6、7、0、1；8 和 16 的起始槽都是 0
        m.delete(6) -> True
        m.get(8) -> 8
        m.get(16) -> 16
        m.delete(7) -> True
        m.get(8) -> 8
        m.get(16) -> 16
        m = HashMap()
        for k in (7, 15, 0, 23): m.put(k, k)    # 槽 7、0；0 的起始槽 0 被占，落到槽 1；23 落到槽 2
        m.delete(15) -> True
        m.get(0) -> 0
        m.get(23) -> 23
        len(m) -> 3
    """)
    # random operations on a few keys whose start slots are 5, 6, 7, 0 and 1 of 8, never more than five at a time
    keys = [5, 13, 21, 6, 14, 22, 7, 15, 23, 31, 0, 8, 16, 1, 9]

    def make(r):
        return ref.random_ops(r, keys, r.randint(1, 80), max_live=5)
    random_check(sol, NEW, 400, make, Expect)


@case("Part 2", title="随机增删查与 dict 对比")
def p2_random(sol):
    def make(r):
        return ref.random_ops(r, ref.pool_default(r, r.choice([3, 10, 40, 150])), r.randint(1, 250))
    random_check(sol, NEW, 300, make, Expect)


@case("Part 2", title="全部删掉，再放回去")
def p2_delete_all(sol):
    keys = list(range(0, 2100, 7))
    ops = [("put", k, k) for k in keys] + [("del", k) for k in keys] + [("len",)]
    ops += [("getd", k, "缺省") for k in range(0, 2100, 5)] + [("in", k) for k in range(1, 400, 3)]
    ops += [("put", k, -k) for k in range(5000, 5300)] + [("put", "x", 1), ("put", 7, "seven")]
    ops += [("get", k) for k in range(5000, 5300)] + [("get", "x"), ("get", 7), ("get", 14), ("len",)]
    check_ops(sol, NEW, ops, Expect)


@case("Part 2", title="反复增删之后表不会越来越大")
def p2_capacity(sol):
    steps = 20_000
    for window in (1, 6, 40):
        peak = window + 1                                  # keys present at once, at most
        code = (f"m = HashMap()\nfor k in range({steps}):\n    m.put(k, None)\n    if k >= {window}:\n"
                f"        m.delete(k - {window})    # 同时存在的键最多 {peak} 个")
        calling(code)
        m = sol.HashMap()
        for k in range(steps):
            m.put(k, None)
            if k >= window:
                got = m.delete(k - window)
                if not _same(got, True):
                    raise Mismatch(f"k = {k} 时 m.delete(k - {window}) 应当返回 True", code, True, got)
        call = calling(code + f"\nlen(m), [k in m for k in range({steps - window - 1}, {steps})]")
        expect((len(m), [k in m for k in range(steps - window - 1, steps)]), (window, [False] + [True] * window), call=call)
        slots = max(8, 4 * peak)
        allowed = 4096 + 400 * slots
        used = ref.footprint(m)
        expect_true(used <= allowed,
                    f"同时存在的键从未超过 {peak} 个，按 Part 2 的要求 2，容量不能超过 max(8, 4×{peak}) = {slots} 个槽。"
                    f"但反复增删 {steps} 次之后，这个 HashMap 对象引用的数据约有 {used} 字节，超过了宽松估计的上限 "
                    f"{allowed} 字节（每个槽算 400 字节，另加 4096）：表在越变越大", call=code)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    run_script(sol, """
        m = HashMap()
        for k in (9, 1, 2): m.put(k, None)
        list(m) -> [9, 1, 2]   # 9 的起始槽是 9 & 7 = 1；1 的起始槽也是 1，被占，落到槽 2；2 落到槽 3
        for k in (3, 4): m.put(k, None)
        list(m) -> [9, 1, 2, 3, 4]
        m.put(5, None)                           # 第 6 个键：先扩到 16，再插入
        list(m) -> [1, 2, 3, 4, 5, 9]
    """)


@case("Part 3", title="keys()、items() 和 for 循环顺序一致")
def p3_views(sol):
    run_script(sol, """
        list(HashMap()) -> []
        list(HashMap().keys()) -> []
        list(HashMap().items()) -> []
        m = HashMap()
        for k in (9, 1, 2, 17): m.put(k, k * 10)    # 槽 1、2、3；17 的起始槽是 1，落到槽 4
        list(m.keys()) -> [9, 1, 2, 17]
        [tuple(p) for p in m.items()] -> [(9, 90), (1, 10), (2, 20), (17, 170)]
        [k for k in m] -> [9, 1, 2, 17]
    """)


@case("Part 3", title="负数键：起始槽是 hash & (capacity-1)")
def p3_negative(sol):
    run_script(sol, """
        m = HashMap()
        for k in (-1, -3, -8, -9, 5): m.put(k, None)    # hash(-1) 是 -2：-2 & 7 = 6，-3 & 7 = 5，-8 & 7 = 0，-9 & 7 = 7
        list(m) -> [-8, 5, -3, -1, -9]   # 5 的起始槽 5 被占，经过 6、7、0，落到槽 1
    """)


@case("Part 3", title="覆盖已有键不会扩容")
def p3_overwrite_no_grow(sol):
    run_script(sol, """
        m = HashMap()
        for k in (9, 1, 2, 3, 4): m.put(k, None)
        m.put(9, "x")       # 已有的键：只覆盖值，表里还是 5 个键
        m.put(1.0, "y")     # 1.0 == 1
        m.put(True, "z")    # True == 1
        list(m) -> [9, 1, 2, 3, 4]   # 只有插入新键之前才检查是否扩容
        m.get(1) -> "z"
        m.put(5, None)      # 第 6 个不同的键：这时才扩到 16
        list(m) -> [1, 2, 3, 4, 5, 9]
    """)


@case("Part 3", title="没有删除时按槽的顺序（随机）")
def p3_order_random(sol):
    def make(r):
        pool = ref.pool_ordered(r, r.choice([10, 40, 200, 1000]))
        return ref.random_ops(r, pool, r.randint(1, 1200), deletes=False, lists=True) + [("list",), ("items",)]
    random_check(sol, NEW, 60, make, lambda: Expect(exact=True))


CHANGES = [("m.put(100, 0)", "加入了一个新键"), ("m.delete(2)", "删除了一个键"),
           ("m.delete(2); m.put(100, 0)", "删除一个键又加入一个键（键数没变）")]


@case("Part 3", title="遍历中增删键：下一次 next 抛 RuntimeError")
def p3_modify(sol):
    for change, what in CHANGES:
        for view in ("iter(m)", "iter(m.keys())", "iter(m.items())"):
            for consumed in (1, 3, 5):
                run_script(sol, f"""
                    m = HashMap()
                    for k in range(5): m.put(k, k)
                    it = {view}
                    {"; ".join(["next(it)"] * consumed)}
                    {change}
                    next(it) -> raise RuntimeError   # 遍历中途{what}，之后第一次 next() 应当抛 RuntimeError
                """)


@case("Part 3", title="取完最后一个槽的键后再修改也要抛")
def p3_last_slot(sol):
    for change in ("m.put(4, 4)", "m.delete(3)"):
        run_script(sol, f"""
            m = HashMap()
            for k in (3, 7): m.put(k, k)    # 槽 3 和槽 7：7 在最后一个槽
            it = iter(m)
            next(it) -> 3
            next(it) -> 7
            {change}
            next(it) -> raise RuntimeError   # 遍历中途有键被加入或删除：即使已经取到了最后一个键，之后第一次 next() 也要抛 RuntimeError
        """)


@case("Part 3", title="遍历时只覆盖值不算修改")
def p3_overwrite_iter(sol):
    run_script(sol, """
        m = HashMap()
        for k in range(5): m.put(k, k)
        for k, v in m.items(): m.put(k, v * 10)
        [tuple(p) for p in m.items()] -> [(0, 0), (1, 10), (2, 20), (3, 30), (4, 40)]
        for k in m: m.put(k, -k)
        m.get(4) -> -4
        for k in m.keys(): m.put(float(k), "f")    # 1.0 == 1：仍是覆盖
        m.get(3) -> "f"
        len(m) -> 5
    """)


@case("Part 3", title="有删除之后：每个键恰好出现一次")
def p3_after_deletes(sol):
    def make(r):
        pool = ref.pool_default(r, r.choice([3, 10, 40, 150]))
        return ref.random_ops(r, pool, r.randint(1, 250), lists=True) + [("list",), ("items",)]
    random_check(sol, NEW, 250, make, Expect)


# ---------------------------------------------------------------- Part 4

CI = ["m = HashMap(hash_fn=lambda s: hash(s.lower()), eq_fn=lambda a, b: a.lower() == b.lower())"]


@case("Part 4", sample=True, title="题目例子：不区分大小写")
def p4_example_case(sol):
    run_script(sol, """
        m = HashMap(hash_fn=lambda s: hash(s.lower()), eq_fn=lambda a, b: a.lower() == b.lower())
        m.put("Key", 1); m.put("KEY", 2)
        len(m) -> 1
        m.get("key") -> 2
        list(m) -> ["Key"]   # 覆盖只换值，保留最先存入的那个键对象
    """)


@case("Part 4", sample=True, title="题目例子：list 作键")
def p4_example_list(sol):
    run_script(sol, """
        m = HashMap(hash_fn=lambda xs: hash(tuple(xs)))
        m.put([1, 2], "a")
        m.get([1, 2]) -> "a"
        [2, 1] in m -> False
    """)


COUNTING = '''
calls = []
def counting_hash(key):
    calls.append(key)
    return hash(key)
m = HashMap(hash_fn=counting_hash)
'''


@case("Part 4", title="每次操作恰好调用一次 hash_fn，扩容时不调用")
def p4_hash_calls(sol):
    ns = {"HashMap": sol.HashMap}
    exec(compile(COUNTING, "<test>", "exec"), ns)
    m, calls = ns["m"], ns["calls"]
    head = COUNTING.strip().splitlines()
    done = []

    def step(code, fn, key, total):
        """total: every call to hash_fn counts (no deletions yet, so a rebuild can only be growth); otherwise only
        the calls on the key passed in count, since the statement does not say what a clean-up after deletions may do."""
        before = len(calls)
        shown = "len(calls) - n" if total else f"sum(c == {key} for c in calls[n:])"
        call = calling(_lines(head, done, f"n = len(calls); {code}; print({shown})   # <- 这一步，应当是 1", show=40))
        fn()
        used = len(calls) - before if total else sum(1 for c in calls[before:] if c == key)
        if used != 1 and total:
            raise Mismatch(f"{code} 调用了 hash_fn {used} 次：put、get、delete、in 每次都应当对传入的键恰好调用一次 "
                           "hash_fn，扩容时一次都不调用", call, 1, used)
        if used != 1:
            raise Mismatch(f"{code} 对传入的键 {key} 调用了 hash_fn {used} 次，应当恰好一次", call, 1, used)

    for k in range(100):                                   # the table grows at the 6th, 11th, 22nd, 43rd, 86th key
        step(f"m.put({k}, {k})", lambda: m.put(k, k), k, True)
        done[:] = [f"for k in range({k + 1}): m.put(k, k)"]
    for code, fn, key, total in [
            ('m.put(7, "x")', lambda: m.put(7, "x"), 7, True), ("m.get(7)", lambda: m.get(7), 7, True),
            ("m.get(-1)", lambda: m.get(-1), -1, True), ("m.get(-1, 0)", lambda: m.get(-1, 0), -1, True),
            ("7 in m", lambda: 7 in m, 7, True), ("-1 in m", lambda: -1 in m, -1, True),
            ("m.delete(7)", lambda: m.delete(7), 7, False), ("m.delete(7)", lambda: m.delete(7), 7, False),
            ('m.put(7, "back")', lambda: m.put(7, "back"), 7, False)]:
        step(code, fn, key, total)
        done.append(code)
    base = list(done)
    for k in range(100, 1100):                             # churn: deleted slots get reused or cleaned up by rebuilds
        step(f"m.put({k}, {k})", lambda: m.put(k, k), k, False)
        done[:] = base + ([f"for k in range(100, {k}): m.put(k, k); m.delete(k - 100)"] if k > 100 else [])
        done.append(f"m.put({k}, {k})")
        step(f"m.delete({k - 100})", lambda: m.delete(k - 100), k - 100, False)
        done[:] = base + [f"for k in range(100, {k + 1}): m.put(k, k); m.delete(k - 100)"]


@case("Part 4", title="只给 eq_fn：1、1.0、True 可以是不同的键")
def p4_eq_only(sol):
    run_script(sol, """
        m = HashMap(eq_fn=lambda a, b: type(a) is type(b) and a == b)
        m.put(1, "int"); m.put(1.0, "float"); m.put(True, "bool")    # 哈希都是 1，但按这个 eq_fn 互不相等
        len(m) -> 3
        m.get(1) -> "int"
        m.get(1.0) -> "float"
        m.get(True) -> "bool"
        m.delete(1.0) -> True
        m.get(1) -> "int"
        1.0 in m -> False
        len(m) -> 2
    """)


@case("Part 4", title="只给 hash_fn：所有键哈希相同")
def p4_hash_only(sol):
    run_script(sol, """
        m = HashMap(hash_fn=lambda k: 7)    # 所有键的哈希都相同，相等仍用 == 判断
        for k in range(20): m.put(k, k)
        len(m) -> 20
        [m.get(k) for k in range(20)] == list(range(20)) -> True
        m.get(20) -> None
        m.put(3.0, "three")    # 3.0 == 3
        m.get(3) -> "three"
        len(m) -> 20
    """)
    head = ["m = HashMap(hash_fn=lambda k: 7)"]

    def make(r):
        return ref.random_ops(r, ref.pool_default(r, r.choice([5, 20, 60])), r.randint(1, 150), lists=True)
    random_check(sol, head, 120, make, Expect)


@case("Part 4", title="list 作键，hash_fn 只有 8 种取值（随机）")
def p4_list_random(sol):
    head = ["m = HashMap(hash_fn=lambda v: hash(tuple(v)) & 7)"]

    def make(r):
        pool = [[r.randint(0, 3) for _ in range(r.randint(0, 3))] for _ in range(r.choice([5, 15, 40]))]
        return ref.random_ops(r, pool, r.randint(1, 200), lists=True, canon=tuple)
    random_check(sol, head, 200, make, lambda: Expect(canon=tuple, mkey=tuple))


@case("Part 4", title="不区分大小写的字符串键（随机）")
def p4_case_random(sol):
    words = ["key", "ab", "x", "", "hello", "q1"]

    def make(r):
        pool = ["".join(c.upper() if r.random() < 0.5 else c for c in r.choice(words)) for _ in range(r.choice([6, 20]))]
        return ref.random_ops(r, pool, r.randint(1, 120), lists=True, canon=str.lower)
    random_check(sol, CI, 200, make, lambda: Expect(canon=str.lower))


@case("Part 4", title="覆盖只换值，保留最先存入的键对象")
def p4_keep_first(sol):
    run_script(sol, """
        m = HashMap()
        m.put(1, "a"); m.put(1.0, "b"); m.put(True, "c")
        [tuple(p) for p in m.items()] -> [(1, "c")]
        type(list(m)[0]).__name__ -> "int"   # 覆盖只换值：键仍是最先存入的 1，不是 1.0 或 True
        m = HashMap(hash_fn=lambda s: hash(s.lower()), eq_fn=lambda a, b: a.lower() == b.lower())
        m.put("Key", 1); m.put("KEY", 2); m.put("key", 3)
        list(m) -> ["Key"]   # 覆盖只换值，保留最先存入的那个键对象
        m.get("kEY") -> 3
        m.delete("KEY") -> True
        m.put("kEy", 4)    # 删掉之后再放进去：这次存的是新的键对象
        list(m) -> ["kEy"]
    """)


MIX64 = '''
def mix64(key):
    x = hash(key) & (1 << 64) - 1
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & (1 << 64) - 1
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & (1 << 64) - 1
    return x ^ (x >> 31)
'''
_mix_ns = {}
exec(MIX64, _mix_ns)


@case("Part 4", title="hash_fn 决定起始槽（没有删除）")
def p4_custom_order(sol):
    tables = [(MIX64.strip().splitlines() + ["m = HashMap(hash_fn=mix64)"], _mix_ns["mix64"]),
              (["m = HashMap(hash_fn=lambda k: -7 * hash(k) - 1)"], lambda k: -7 * hash(k) - 1)]
    for head, fn in tables:
        def make(r):
            pool = ref.pool_ordered(r, r.choice([10, 50, 300])) + [i << 40 for i in range(r.randint(0, 200))]
            return ref.random_ops(r, pool, r.randint(1, 600), deletes=False, lists=True) + [("list",), ("items",)]
        random_check(sol, head, 25, make, lambda: Expect(exact=True, hash_fn=fn))
