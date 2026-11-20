import ast
import math
import textwrap
import time

from judgelib import Mismatch, calling, case, expect_true, rng, short
import ref

SHOW = 16           # a replay lists at most this many steps
NAMES = ("MinHeap", "IndexedPQ", "dijkstra_indexed", "dijkstra_lazy")


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


def _norm(got, want):
    """got in the shape of want: a 2-element list is accepted where the statement says (key, priority), and a tuple
    where it says a list of distances."""
    if isinstance(want, (list, tuple)) and isinstance(got, (list, tuple)) and len(got) == len(want):
        return type(want)(_norm(g, w) for g, w in zip(got, want))
    return got


def _same(got, want):
    if want is None:
        return got is None
    try:
        return bool(_norm(got, want) == want)
    except Exception:
        return False


def _dist_diff(got, want):
    """For a wrong list of distances: the first node that differs."""
    if not isinstance(got, (list, tuple)):
        return f"应当返回一个列表，得到的是 {type(got).__name__}"
    if len(got) != len(want):
        return f"返回的列表长度应当是节点数 {len(want)}，得到 {len(got)}"
    for v, (a, b) in enumerate(zip(got, want)):
        if not _same(a, b):
            return f"节点 {v} 的最短距离应当是 {b}，得到 {short(a, 100)}"
    return "最短距离不对"


def _ns(sol):
    ns = {"math": math}
    for name in NAMES:
        if hasattr(sol, name):
            ns[name] = getattr(sol, name)
    return ns


def _define(src):
    ns = {}
    exec(compile(src, "<test>", "exec"), ns)
    return ns


# Classes the tests use, defined from the same source that a replay shows.

ITEM_SRC = '''
class Item:
    """Only < is defined: <= and >= between two Item raise TypeError."""
    def __init__(self, v):
        self.v = v
    def __lt__(self, other):
        return self.v < other.v
    def __le__(self, other):
        raise TypeError("题目要求元素（或优先级）之间只用 < 比较，这里用了 <= 或 >=")
    __ge__ = __le__
    def __repr__(self):
        return f"Item({self.v})"
'''.strip()

C_SRC = '''
class C:
    """Counts every < between two C."""
    count = 0
    def __init__(self, v):
        self.v = v
    def __lt__(self, other):
        C.count += 1
        return self.v < other.v
    def __le__(self, other):
        raise TypeError("题目要求元素（或优先级）之间只用 < 比较，这里用了 <= 或 >=")
    __ge__ = __le__
    def __repr__(self):
        return f"C({self.v})"
'''.strip()

PK_SRC = '''
class P:
    """A priority that counts every < between two P."""
    count = 0
    def __init__(self, v):
        self.v = v
    def __lt__(self, other):
        P.count += 1
        return self.v < other.v
    def __le__(self, other):
        raise TypeError("题目要求元素（或优先级）之间只用 < 比较，这里用了 <= 或 >=")
    __ge__ = __le__
    def __repr__(self):
        return f"P({self.v})"

class K:
    """A key that counts every call of __hash__ and __eq__; it has no <."""
    calls = 0
    def __init__(self, i):
        self.i = i
    def __hash__(self):
        K.calls += 1
        return self.i
    def __eq__(self, other):
        K.calls += 1
        return isinstance(other, K) and self.i == other.i
    def __repr__(self):
        return f"K({self.i})"
'''.strip()

EDGES_SRC = '''
class Edges(list):
    """A list of edges that counts how many edges are read from it."""
    reads = 0
    def __iter__(self):
        Edges.reads += len(self)
        return super().__iter__()
    def __getitem__(self, i):
        Edges.reads += 1
        return super().__getitem__(i)
'''.strip()

C = _define(C_SRC)["C"]
_pk = _define(PK_SRC)
P, K = _pk["P"], _pk["K"]
Edges = _define(EDGES_SRC)["Edges"]


# ---------------------------------------------------------------- statement-style scripts

EXC = {"IndexError": IndexError, "ValueError": ValueError, "KeyError": KeyError}


def _what(code):
    """What a checked expression is, for the message when the script gives none."""
    c = code.replace(" ", "")
    if c.startswith("dijkstra"):
        return None                                   # _dist_diff says which node
    if c.startswith("len("):
        return "len 不对"
    if " for " in code:
        return "依次 pop 的结果不对：每次 pop 都应当返回当时最小的元素（Part 2 是优先级最小的 (key, priority)）"
    if c.startswith("h."):
        return "应当返回堆里最小的元素"
    if ".peek()" in c or ".pop()" in c:
        return "应当返回优先级最小的 (key, priority)"
    if ".priority(" in c:
        return "priority 应当返回这个键当前的优先级"
    if ".remove(" in c:
        return "remove 应当返回这个键被删之前的优先级"
    if c.endswith("inq"):
        return "in 的结果不对"
    return "返回值不对"


def run_script(sol, script, prelude=""):
    """Runs a script written like the statement's examples, one line at a time. A line `expr -> want` checks the
    value of expr (`-> raise KeyError` checks that it raises); any other line is run as it is. The replay shown on a
    failure is the prelude plus the lines so far, ready to paste into 自定义运行."""
    ns = _ns(sol)
    head = prelude.strip().splitlines() if prelude.strip() else []
    if head:
        exec(compile(prelude.strip(), "<test>", "exec"), ns)
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
            exc = EXC[want_src[len("raise "):].strip()]
            call = calling(_lines(head, done, f"{code}   # <- 这一步，应当抛出 {exc.__name__}"))
            try:
                value = eval(compile(code, "<test>", "eval"), ns)
            except exc:
                done.append(f"{code}   # -> 抛出 {exc.__name__}")
                continue
            except Exception as e:
                raise Mismatch(why or f"应当抛出 {exc.__name__}，实际抛出 {type(e).__name__}: {e}", call,
                               f"抛出 {exc.__name__}", f"抛出 {type(e).__name__}") from None
            raise Mismatch(why or f"应当抛出 {exc.__name__}，但正常返回了", call, f"抛出 {exc.__name__}", value)
        want = eval(want_src, {"__builtins__": {}, "inf": math.inf})
        call = calling(_lines(head, done, f"print({code})   # <- 这一步，应当是 {want_src}"))
        got = eval(compile(code, "<test>", "eval"), ns)
        if not _same(got, want):
            msg = why or _what(code) or _dist_diff(got, want)
            raise Mismatch(msg, call, want, got)
        done.append(f"{code}   # -> {want_src}")


# ---------------------------------------------------------------- operation sequences checked against a model

class Raise:
    """The expected outcome of a step is this exception."""

    def __init__(self, exc):
        self.exc = exc


SKIP = object()     # the step returns nothing to check


class HeapSpec:
    """Part 1 sequences on h = MinHeap(init): ("push", v), ("pop",), ("peek",), ("len",). With item=True every
    element is Item(v), which has only <."""
    var = "h"

    def __init__(self, init, item=False):
        self.init, self.item = list(init), item
        if item:
            self.head = ITEM_SRC.splitlines() + ["h = MinHeap([" + ", ".join(f"Item({v})" for v in init) + "])"]
        else:
            self.head = [f"h = MinHeap({_lit(self.init)})"]

    def call(self, ns, h, op):
        kind = op[0]
        if kind == "push":
            return h.push(ns["Item"](op[1]) if self.item else op[1])
        if kind == "pop":
            return h.pop()
        if kind == "peek":
            return h.peek()
        return len(h)

    def code(self, op):
        if op[0] == "push":
            return f"h.push(Item({op[1]!r}))" if self.item else f"h.push({op[1]!r})"
        return {"pop": "h.pop()", "peek": "h.peek()", "len": "len(h)"}[op[0]]

    def returns(self, op):
        return op[0] != "push"

    def show(self, ns, got):
        return _note(got)

    def model(self, ns):
        return HeapModel(self.init, ns["Item"] if self.item else None)


class HeapModel:
    def __init__(self, init, item_cls):
        self.vals, self.item_cls = list(init), item_cls

    def want(self, op):
        if op[0] == "push":
            return SKIP
        if op[0] == "len":
            return len(self.vals)
        return min(self.vals) if self.vals else Raise(IndexError)

    def check(self, op, got, want):
        if op[0] == "len":
            return None if _same(got, want) else "len 不对"
        if self.item_cls is not None:
            if not isinstance(got, self.item_cls):
                return f"{op[0]} 应当返回堆里的一个 Item 对象"
            got = got.v
        if _same(got, want):
            return None
        return f"{op[0]} 应当返回堆里最小的元素"

    def show_want(self, op, want):
        return self.item_cls(want) if self.item_cls is not None and op[0] != "len" else want

    def apply(self, op, got):
        if op[0] == "push":
            self.vals.append(op[1])
        elif op[0] == "pop":
            self.vals.remove(min(self.vals))


class PQSpec:
    """Part 2 sequences on q = IndexedPQ(): ("push", k, p), ("pop",), ("peek",), ("update", k, p), ("remove", k),
    ("prio", k), ("in", k), ("len",). With keys_var=True a key k is an index into the list `keys` of the prelude."""
    var = "q"

    def __init__(self, prelude=(), keys_var=False):
        self.head = list(prelude) + ["q = IndexedPQ()"]
        self.keys_var = keys_var

    def key(self, ns, t):
        return ns["keys"][t] if self.keys_var else t

    def kcode(self, t):
        return f"keys[{t}]" if self.keys_var else _lit(t)

    def call(self, ns, q, op):
        kind = op[0]
        if kind == "push":
            return q.push(self.key(ns, op[1]), op[2])
        if kind == "pop":
            return q.pop()
        if kind == "peek":
            return q.peek()
        if kind == "update":
            return q.update(self.key(ns, op[1]), op[2])
        if kind == "remove":
            return q.remove(self.key(ns, op[1]))
        if kind == "prio":
            return q.priority(self.key(ns, op[1]))
        if kind == "in":
            return self.key(ns, op[1]) in q
        return len(q)

    def code(self, op):
        kind = op[0]
        if kind in ("push", "update"):
            return f"q.{kind}({self.kcode(op[1])}, {op[2]!r})"
        if kind == "remove":
            return f"q.remove({self.kcode(op[1])})"
        if kind == "prio":
            return f"q.priority({self.kcode(op[1])})"
        if kind == "in":
            return f"{self.kcode(op[1])} in q"
        return {"pop": "q.pop()", "peek": "q.peek()", "len": "len(q)"}[kind]

    def returns(self, op):
        return op[0] not in ("push", "update")

    def show(self, ns, got):
        if self.keys_var and isinstance(got, (tuple, list)) and len(got) == 2:
            for i, k in enumerate(ns["keys"]):
                if k is got[0]:
                    return f"(keys[{i}], {got[1]!r})"
        return _note(got)

    def model(self, ns):
        return PQModel(self, ns)


class PQModel:
    def __init__(self, spec, ns):
        self.spec, self.ns, self.d = spec, ns, {}

    def _k(self, t):
        return self.spec.key(self.ns, t)

    def want(self, op):
        kind = op[0]
        if kind == "push":
            return Raise(ValueError) if self._k(op[1]) in self.d else SKIP
        if kind in ("pop", "peek"):
            if not self.d:
                return Raise(IndexError)
            low = min(self.d.values())
            return next((k, p) for k, p in self.d.items() if p == low)      # one right answer
        if kind == "in":
            return self._k(op[1]) in self.d
        if kind == "len":
            return len(self.d)
        k = self._k(op[1])
        if k not in self.d:
            return Raise(KeyError)
        return SKIP if kind == "update" else self.d[k]

    def check(self, op, got, want):
        kind = op[0]
        if kind in ("pop", "peek"):
            if not (isinstance(got, (tuple, list)) and len(got) == 2):
                return f"{kind} 应当返回二元组 (key, priority)"
            k, p = got
            try:
                present = k in self.d
            except TypeError:
                present = False
            if not present:
                return f"{kind} 返回的键不在队列里"
            if not _same(p, self.d[k]):
                return f"{kind} 返回的优先级不是这个键当前的优先级 {self.d[k]!r}"
            if not _same(p, want[1]):
                return f"{kind} 应当返回优先级最小的元素：队列里最小的优先级是 {want[1]!r}（并列时返回哪一个都可以）"
            return None
        if _same(got, want):
            return None
        return {"in": "in 的结果不对", "len": "len 不对", "remove": "remove 应当返回这个键被删之前的优先级",
                "prio": "priority 应当返回这个键当前的优先级"}[kind]

    def show_want(self, op, want):
        if op[0] in ("pop", "peek") and self.spec.keys_var:
            return self.spec.show(self.ns, want)
        return want

    def apply(self, op, got):
        kind = op[0]
        if kind in ("push", "update"):
            self.d[self._k(op[1])] = op[2]
        elif kind == "pop":
            del self.d[got[0]]
        elif kind == "remove":
            del self.d[self._k(op[1])]


def _make(spec, sol):
    ns = _ns(sol)
    exec(compile("\n".join(spec.head), "<test>", "exec"), ns)
    return ns


def _replay(spec, ops, shown, want=None, show=SHOW):
    done = [spec.code(op) + ("" if s is None else f"   # -> {s}") for op, s in zip(ops, shown)]
    op = ops[-1]
    if isinstance(want, Raise):
        last = f"{spec.code(op)}   # <- 这一步，应当抛出 {want.exc.__name__}"
    else:
        last = (f"print({spec.code(op)})" if spec.returns(op) else spec.code(op)) + "   # <- 这一步"
    return _lines(spec.head, done, last, show)


def _outcome(spec, sol, ops):
    """Replays ops on a fresh object; None if every result agrees with the model."""
    ns = _make(spec, sol)
    obj, model, shown = ns[spec.var], spec.model(ns), []
    for k, op in enumerate(ops):
        want = model.want(op)
        bad = {"k": k, "exc": None, "shown": shown, "raw": want}
        try:
            got = spec.call(ns, obj, op)
        except Exception as e:
            if isinstance(want, Raise) and isinstance(e, want.exc):
                shown.append(f"抛出 {want.exc.__name__}")
                continue
            if isinstance(want, Raise):
                return dict(bad, got=f"抛出 {type(e).__name__}: {e}", want=f"抛出 {want.exc.__name__}",
                            msg=f"应当抛出 {want.exc.__name__}，实际抛出 {type(e).__name__}")
            return dict(bad, exc=e)
        except BaseException:
            calling(_replay(spec, ops[: k + 1], shown, want))     # out of time inside this call
            raise
        if isinstance(want, Raise):
            return dict(bad, got=got, want=f"抛出 {want.exc.__name__}", msg=f"应当抛出 {want.exc.__name__}，但正常返回了")
        if want is not SKIP:
            msg = model.check(op, got, want)
            if msg:
                return dict(bad, got=got, want=model.show_want(op, want), msg=msg)
        model.apply(op, got)
        shown.append(None if want is SKIP else spec.show(ns, got))
    return None


def _shrink(spec, sol, ops, bad, budget=0.6):
    """Drops steps that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(spec, sol, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def check_ops(spec, sol, ops, what=""):
    """Runs ops and fails with a (shortened) replay at the first result that differs from the model."""
    calling("\n".join(spec.head) + "   # <- 这一步")
    bad = _outcome(spec, sol, ops)
    if bad is None:
        return
    calling(_replay(spec, ops[: bad["k"] + 1], bad["shown"], bad["raw"]))   # shown if shrinking runs into a hang
    ops, bad = _shrink(spec, sol, ops, bad)
    k = bad["k"]
    call = _replay(spec, ops[: k + 1], bad["shown"], bad["raw"], show=60)
    if bad["exc"] is not None:
        calling(call)
        raise bad["exc"]
    note = f"（{what}，已自动删去与出错无关的步骤）" if what else ""
    raise Mismatch(f"第 {k + 1} 步：{bad['msg']}{note}", call, bad["want"], bad["got"])


# ---------------------------------------------------------------- random sequences

def heap_ops(r, size_max=12, steps=40, low=-5, high=15):
    init = [r.randint(low, high) for _ in range(r.randint(0, size_max))]
    ops, size = [], len(init)
    for _ in range(r.randint(5, steps)):
        x = r.random()
        if x < 0.45 or (size == 0 and x < 0.8):
            ops.append(("push", r.randint(low, high)))
            size += 1
        elif x < 0.85:
            ops.append(("pop",))
            size = max(0, size - 1)
        elif x < 0.95:
            ops.append(("peek",))
        else:
            ops.append(("len",))
    return init, ops


def pq_ops(r):
    keys = ref.key_names(r.randint(3, 8))
    ops = []
    for _ in range(r.randint(10, 50)):
        x = r.random()
        k, p = r.choice(keys), r.randint(0, 9)
        if x < 0.3:
            ops.append(("push", k, p))
        elif x < 0.45:
            ops.append(("pop",))
        elif x < 0.5:
            ops.append(("peek",))
        elif x < 0.7:
            ops.append(("update", k, p))
        elif x < 0.8:
            ops.append(("remove", k))
        elif x < 0.88:
            ops.append(("prio", k))
        elif x < 0.94:
            ops.append(("in", k))
        else:
            ops.append(("len",))
    return ops


def drain_ops(r, changes):
    """Push n keys, then `changes` steps chosen from ("remove",) or ("update",), then pop until empty (and once
    more, which must raise IndexError)."""
    n = r.randint(4, 15)
    keys = ref.key_names(n)
    prio = {k: r.randint(0, 30) for k in keys}
    ops = [("push", k, prio[k]) for k in keys]
    live = list(keys)
    for _ in range(r.randint(1, 3)):
        kind = r.choice(changes)
        k = r.choice(live)
        if kind == "remove":
            live.remove(k)
            ops.append(("remove", k))
        else:
            old = prio[k]
            prio[k] = r.choice([r.randint(old, 40), r.randint(-10, old)])     # up or down
            ops.append(("update", k, prio[k]))
    ops += [("pop",)] * (len(live) + 1)
    return ops


# ---------------------------------------------------------------- big inputs, drawn once at import time

N_HEAP = 20_000
HEAP_INIT, HEAP_PUSHES, HEAP_POPS = ref.heap_workload(rng("heap"), N_HEAP)
N_PQ = 20_000
PQ_BIG = ref.pq_workload(rng("pq"), N_PQ)
V_BIG, E_BIG = 20_000, 100_000
_g = rng("graph")
ADJ_BIG = ref.random_graph(_g, V_BIG, E_BIG, 0, 1000)
SRC_BIG = _g.randrange(V_BIG)
DIST_BIG = ref.dijkstra_heapq(ADJ_BIG, SRC_BIG)
del _g


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    run_script(sol, """
        h = MinHeap([5, 3, 8, 1])
        h.peek() -> 1
        len(h) -> 4
        h.push(0); h.push(4)
        [h.pop() for _ in range(6)] -> [0, 1, 3, 4, 5, 8]
        h.pop() -> raise IndexError
    """)


@case("Part 1", title="空堆、一个元素、弹空后再用")
def p1_edge(sol):
    run_script(sol, """
        h = MinHeap()
        len(h) -> 0
        h.peek() -> raise IndexError
        h.pop() -> raise IndexError
        h.push(7)
        h.peek() -> 7
        len(h) -> 1
        h.pop() -> 7
        len(h) -> 0
        h.pop() -> raise IndexError
        h.peek() -> raise IndexError
        h.push(-3); h.push(-5)
        h.pop() -> -5
        h.pop() -> -3
        h = MinHeap([7])
        h.peek() -> 7
        h.pop() -> 7
        len(h) -> 0
        h = MinHeap((5, 3, 8, 1))   # items 的默认值是 ()，所以元组也要能用
        [h.pop() for _ in range(4)] -> [1, 3, 5, 8]
    """)


@case("Part 1", title="重复元素、元组、字符串")
def p1_duplicates(sol):
    run_script(sol, """
        h = MinHeap([2, 2, 1, 1, 3, 3, 1])
        [h.pop() for _ in range(7)] -> [1, 1, 1, 2, 2, 3, 3]
        h = MinHeap([4] * 9)
        h.push(4); h.push(2); h.push(4)
        h.peek() -> 2
        [h.pop() for _ in range(12)] -> [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        h = MinHeap([(2, 1), (1, 5), (1, 3), (0, 9)])
        h.pop() -> (0, 9)
        h.pop() -> (1, 3)
        h.push((1, 4))
        [h.pop() for _ in range(3)] -> [(1, 4), (1, 5), (2, 1)]
        h = MinHeap(["pear", "apple", "fig"])
        h.pop() -> "apple"
    """)


@case("Part 1", title="随机操作，和排序的结果对照")
def p1_random(sol):
    r = rng()
    for _ in range(300):
        init, ops = heap_ops(r)
        check_ops(HeapSpec(init), sol, ops, what="随机操作序列")


@case("Part 1", title="元素只支持 <")
def p1_only_lt(sol):
    r = rng()
    for _ in range(120):
        init, ops = heap_ops(r, low=0, high=6)
        check_ops(HeapSpec(init, item=True), sol, ops, what="随机操作序列，元素是只定义了 < 的 Item")


@case("Part 1", title="建堆的比较次数：要求 O(n)")
def p1_build_count(sol):
    n = 2 ** 15
    shapes = [
        (f"range({n}, 0, -1)", "降序", list(range(n, 0, -1))),
        (f"range({n})", "升序", list(range(n))),
        (f"(v * 7919 % {n} for v in range({n}))", "一个打乱的排列", [v * 7919 % n for v in range(n)]),
        (f"(v % 100 for v in range({n}))", "大量重复", [v % 100 for v in range(n)]),
        (f"[0] * {n}", "全部相等", [0] * n),
        (f"range({n - 1}, 0, -1)", "降序", list(range(n - 1, 0, -1))),
    ]
    for gen, what, values in shapes:
        items = [C(v) for v in values]
        limit = 5 * len(items)
        call = calling("\n".join(C_SRC.splitlines() + [
            f"h = MinHeap([C(v) for v in {gen}])   # {what}，{len(items)} 个元素",
            f"print(C.count)   # <- 应当不超过 5n = {limit}"]))
        C.count = 0
        h = sol.MinHeap(items)
        used = C.count
        if used > limit:
            raise Mismatch(f"用 {len(items)} 个元素建堆做了 {used} 次比较，约 {used / len(items):.1f}n。要求建堆是 O(n)："
                           "应当从最后一个有孩子的节点倒着逐个下沉（约 2n 次比较），而不是逐个 push（降序输入时约 n log₂ n 次）",
                           call, f"不超过 {limit}", used)
        call = calling(call.rsplit("\n", 1)[0] + "\nprint(len(h), h.peek())   # <- 这一步")
        got = (len(h), h.peek())
        expect_true(got[0] == len(values) and isinstance(got[1], C) and got[1].v == min(values),
                    f"建堆之后 len(h) 应当是 {len(values)}，h.peek() 应当是最小的 C({min(values)})", call, got)


@case("Part 1", title="2×10⁴ 个元素：要求 O(log n)")
def p1_large(sol):
    call = calling(
        f"h = MinHeap(INIT)          # INIT：{N_HEAP} 个随机整数\n"
        f"for x in PUSHES:           # PUSHES：{N_HEAP} 个随机整数\n"
        "    h.push(x)\n"
        "    print(h.pop())\n"
        "while len(h):\n"
        "    print(h.pop())")
    h = sol.MinHeap(list(HEAP_INIT))
    got = []
    for x in HEAP_PUSHES:
        h.push(x)
        got.append(h.pop())
    while len(h):
        got.append(h.pop())
    if got != HEAP_POPS:
        k = next((i for i, (a, b) in enumerate(zip(got, HEAP_POPS)) if a != b), min(len(got), len(HEAP_POPS)))
        if k >= len(got) or k >= len(HEAP_POPS):
            raise Mismatch(f"一共应当弹出 {len(HEAP_POPS)} 个元素，实际弹出 {len(got)} 个", call, len(HEAP_POPS), len(got))
        raise Mismatch(f"第 {k + 1} 次 pop 应当返回当时堆里最小的元素", call, HEAP_POPS[k], got[k])


def _log2(n):
    return max(1, n).bit_length() - 1


@case("Part 1", title="push、pop、peek 的比较次数")
def p1_op_counts(sol):
    n0, m = 4096, 2048
    build = C_SRC.splitlines() + [f"h = MinHeap([C(v) for v in range({n0})])"]
    calling("\n".join(build) + "   # <- 这一步")
    h = sol.MinHeap([C(v) for v in range(n0)])
    for k in range(1, m + 1):
        item = C(-k)
        call = calling("\n".join(build + [
            f"for k in range(1, {m + 1}):",
            "    C.count = 0",
            "    h.push(C(-k))    # 每次都是新的最小值，一路浮到堆顶",
            "    print(C.count)   # 应当不超过 2⌊log₂ len(h)⌋ + 2",
            f"# 出错的是 k = {k} 这一轮"]))
        C.count = 0
        h.push(item)
        used, limit = C.count, 2 * _log2(n0 + k) + 2
        if used > limit:
            raise Mismatch(f"一次 push 做了 {used} 次比较；push 应当是 O(log n)：上浮每层只比较一次，"
                           f"len(h) = {n0 + k} 时上限按 2⌊log₂ n⌋ + 2 = {limit} 算", call, f"不超过 {limit}", used)
    pushes = "\n".join(build + [f"for k in range(1, {m + 1}): h.push(C(-k))"])
    calling(pushes + "\nprint(h.peek())   # <- 这一步")
    C.count = 0
    top = h.peek()
    used = C.count
    if used > 2:
        raise Mismatch(f"peek 做了 {used} 次比较；peek 应当是 O(1)", pushes + "\nC.count = 0\nh.peek()\nprint(C.count)",
                       "不超过 2", used)
    expect_true(isinstance(top, C) and top.v == -m, f"h.peek() 应当是 C({-m})",
                pushes + "\nprint(h.peek())   # <- 这一步", top)
    want = list(range(-m, 0)) + list(range(n0))
    for j, v in enumerate(want):
        size = len(want) - j
        loop = [
            "j = 0",
            "while len(h):",
            "    n = len(h)",
            "    C.count = 0",
            "    x = h.pop()",
            "    print(x, C.count)   # 比较次数应当不超过 4⌊log₂ n⌋ + 4",
            "    j += 1",
            f"# 出错的是 j = {j} 这一次 pop（len(h) = {size}）"]
        calling(pushes + "\n" + "\n".join(loop))
        C.count = 0
        got = h.pop()
        used, limit = C.count, 4 * _log2(size) + 4
        if used > limit:
            raise Mismatch(f"一次 pop 做了 {used} 次比较；pop 应当是 O(log n)：下沉每层比较两次，"
                           f"len(h) = {size} 时上限按 4⌊log₂ n⌋ + 4 = {limit} 算", pushes + "\n" + "\n".join(loop),
                           f"不超过 {limit}", used)
        if not (isinstance(got, C) and got.v == v):
            raise Mismatch("pop 应当返回堆里最小的元素", pushes + "\n" + "\n".join(loop), C(v), got)


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子 1：remove")
def p2_example1(sol):
    run_script(sol, """
        q = IndexedPQ()
        for key, p in [("a", 1), ("b", 12), ("c", 4), ("d", 18), ("e", 14), ("f", 13), ("g", 7)]: q.push(key, p)
        q.remove("d") -> 18
        [q.pop() for _ in range(6)] -> [("a", 1), ("c", 4), ("g", 7), ("b", 12), ("f", 13), ("e", 14)]
    """)


@case("Part 2", sample=True, title="题目例子 2：update 和出错")
def p2_example2(sol):
    run_script(sol, """
        q = IndexedPQ()
        q.push("x", 5); q.push("y", 3); q.push("z", 9)
        q.update("z", 1)                         # 调小
        q.update("y", 7)                         # 调大
        q.peek() -> ("z", 1)
        q.priority("y") -> 7
        "x" in q -> True
        "w" in q -> False
        q.push("x", 0) -> raise ValueError
        q.update("w", 1) -> raise KeyError
        q.remove("w") -> raise KeyError
        [q.pop() for _ in range(3)] -> [("z", 1), ("x", 5), ("y", 7)]
        q.pop() -> raise IndexError
    """)


@case("Part 2", title="出错的调用不改变队列")
def p2_errors(sol):
    run_script(sol, """
        q = IndexedPQ()
        q.pop() -> raise IndexError
        q.peek() -> raise IndexError
        q.priority("a") -> raise KeyError
        q.update("a", 1) -> raise KeyError
        q.remove("a") -> raise KeyError
        "a" in q -> False
        len(q) -> 0
        q.push("a", 5); q.push("b", 2)
        q.push("a", 1) -> raise ValueError
        q.priority("a") -> 5      # 失败的 push 不改变队列
        len(q) -> 2
        q.peek() -> ("b", 2)
        q.remove("b") -> 2
        "b" in q -> False
        q.remove("b") -> raise KeyError
        q.priority("b") -> raise KeyError
        q.push("b", 9)            # 删掉的键可以重新加入
        q.priority("b") -> 9
        q.pop() -> ("a", 5)
        q.update("a", 0) -> raise KeyError     # 已经弹出
        len(q) -> 1
        q.pop() -> ("b", 9)
        q.pop() -> raise IndexError
        q.push("a", 3)
        q.update("a", 3)          # 改成同样的优先级也可以
        q.peek() -> ("a", 3)
        len(q) -> 1
    """)


@case("Part 2", title="pop 之后按键操作")
def p2_after_pop(sol):
    run_script(sol, """
        q = IndexedPQ()
        q.push("a", 1); q.push("b", 5); q.push("c", 3)
        q.pop() -> ("a", 1)
        q.priority("c") -> 3      # pop 把末尾的 c 换到了堆顶，位置表要跟着改
        q.priority("b") -> 5
        q.update("c", 10)
        q.priority("c") -> 10
        q.peek() -> ("b", 5)
        q.remove("b") -> 5
        q.push("d", 4); q.push("e", 2)
        q.pop() -> ("e", 2)
        q.update("d", 20)
        q.remove("c") -> 10
        q.priority("d") -> 20
        q.pop() -> ("d", 20)
        len(q) -> 0
    """)


KEYS_PRELUDE = ['keys = [object(), "s", 3, (1, 2), None, 2.5, frozenset()]   # 大多两两之间不能比较大小']


@case("Part 2", title="优先级并列时不比较键")
def p2_keys_not_compared(sol):
    r = rng()
    spec = PQSpec(KEYS_PRELUDE, keys_var=True)
    for _ in range(60):
        ops = [("push", t, r.randint(0, 2)) for t in r.sample(range(7), r.randint(2, 7))]
        for _ in range(r.randint(3, 20)):
            x, t, p = r.random(), r.randrange(7), r.randint(0, 2)
            ops.append(("update", t, p) if x < 0.35 else ("remove", t) if x < 0.5 else
                       ("push", t, p) if x < 0.7 else ("pop",))
        ops += [("pop",)] * 8
        check_ops(spec, sol, ops, what="随机操作序列")


@case("Part 2", title="remove 之后两个方向都要调整")
def p2_remove_random(sol):
    r = rng()
    for _ in range(300):
        check_ops(PQSpec(), sol, drain_ops(r, ["remove"]), what="随机：push、remove，然后 pop 到空")


@case("Part 2", title="update 调大调小都对")
def p2_update_random(sol):
    r = rng()
    for _ in range(300):
        check_ops(PQSpec(), sol, drain_ops(r, ["update"]), what="随机：push、update，然后 pop 到空")


@case("Part 2", title="随机操作，和字典模型对照")
def p2_random(sol):
    r = rng()
    for _ in range(300):
        check_ops(PQSpec(), sol, pq_ops(r), what="随机操作序列")


@case("Part 2", cpu=4, title="2×10⁴ 个键的混合操作：要求 O(log n)")
def p2_large(sol):
    call = calling(
        f"q = IndexedPQ()\n"
        f"# 先 push {N_PQ} 个键（0..{N_PQ - 1}），然后 {N_PQ} 轮：pop、push 一个新键、update 一个随机的键，\n"
        "# 每 4 轮再 remove 一个随机的键；最后 pop 到空。优先级是互不相同的随机整数")
    q = sol.IndexedPQ()
    for j, (op, want) in enumerate(PQ_BIG):
        kind = op[0]
        if kind == "push":
            q.push(op[1], op[2])
        elif kind == "update":
            q.update(op[1], op[2])
        elif kind == "pop":
            got = q.pop()
            if not _same(got, want):
                raise Mismatch(f"第 {j + 1} 个操作 q.pop() 应当返回优先级最小的 (key, priority)", call, want, got)
        else:
            got = q.remove(op[1])
            if not _same(got, want):
                raise Mismatch(f"第 {j + 1} 个操作 q.remove({op[1]}) 应当返回它被删之前的优先级", call, want, got)
    expect_true(len(q) == 0, "全部弹出之后 len(q) 应当是 0", call, len(q))


def _counts_check(what, n, used, calls, cmp_limit, call_limit, code):
    if used > cmp_limit:
        raise Mismatch(f"{what} 做了 {used} 次优先级比较（n = {n}），上限按 {cmp_limit} 算：要求 O(log n) 次比较",
                       code, f"不超过 {cmp_limit}", used)
    if calls > call_limit:
        raise Mismatch(f"{what} 调用了 {calls} 次键的 __hash__ / __eq__（n = {n}），上限按 {call_limit} 算：位置表的 dict "
                       "操作应当是 O(log n) 次，查找键的位置不能扫描整个堆", code, f"不超过 {call_limit}", calls)


@case("Part 2", title="比较次数和 dict 操作次数")
def p2_counts(sol):
    n = 2048
    head = PK_SRC.splitlines() + ["q = IndexedPQ()", f"keys = [K(i) for i in range({n})]"]
    phases = [f"for i in range({n}): q.push(keys[i], P({n} - i))"]

    def code(loop, body, where):
        return "\n".join(head + phases + [loop, "    P.count = K.calls = 0", f"    {body}",
                                          "    print(P.count, K.calls)", f"# 出错的是{where}"])

    q = sol.IndexedPQ()
    keys = [K(i) for i in range(n)]
    model = {}
    for i in range(n):
        p = P(n - i)
        call = calling("\n".join(head + [f"for i in range({n}):", "    P.count = K.calls = 0",
                                         f"    q.push(keys[i], P({n} - i))   # 每次都是新的最小值，一路浮到堆顶",
                                         "    print(P.count, K.calls)", f"# 出错的是 i = {i} 这一轮"]))
        P.count = K.calls = 0
        q.push(keys[i], p)
        model[i] = n - i
        L = _log2(i + 1)
        _counts_check("q.push", i + 1, P.count, K.calls, 2 * L + 2, 12 * L + 20, call)
    L = _log2(n)
    for start, expr, note, new in [(0, "P(10**6 + i)", "调大：沉到底", lambda i: 10**6 + i),
                                   (1, "P(-i)", "调小：浮到顶", lambda i: -i)]:
        loop = f"for i in range({start}, {n}, 3):"
        for i in range(start, n, 3):
            p = P(new(i))
            call = calling(code(loop, f"q.update(keys[i], {expr})   # {note}", f" i = {i} 这一轮"))
            P.count = K.calls = 0
            q.update(keys[i], p)
            model[i] = new(i)
            _counts_check("q.update", n, P.count, K.calls, 4 * L + 4, 16 * L + 32, call)
        phases.append(f"{loop} q.update(keys[i], {expr})")
    loop = f"for i in range(2, {n}, 6):"
    for i in range(2, n, 6):
        size = len(model)
        call = calling(code(loop, "q.remove(keys[i])", f" i = {i} 这一轮"))
        P.count = K.calls = 0
        got = q.remove(keys[i])
        used, calls = P.count, K.calls
        L = _log2(size)
        _counts_check("q.remove", size, used, calls, 4 * L + 4, 16 * L + 32, call)
        expect_true(isinstance(got, P) and got.v == model[i], f"q.remove(keys[{i}]) 应当返回 P({model[i]})", call, got)
        del model[i]
    phases.append(f"{loop} q.remove(keys[i])")
    base = "\n".join(head + phases)
    for i in (5, 7, 1000, 2047):
        for text, fn, want in [(f"keys[{i}] in q", lambda: keys[i] in q, i in model),
                               (f"K({i}) in q", lambda: K(i) in q, i in model),
                               (f"q.priority(keys[{i}])", lambda: q.priority(keys[i]), model.get(i)),
                               ("q.peek()", lambda: q.peek(), None),
                               ("len(q)", lambda: len(q), len(model))]:
            if want is None and text.startswith("q.priority"):
                continue                                    # removed key: KeyError is checked elsewhere
            call = calling(base + f"\nP.count = K.calls = 0\nx = {text}\nprint(x, P.count, K.calls)   # <- 这一步")
            P.count = K.calls = 0
            got = fn()
            used, calls = P.count, K.calls
            if used > 2 or calls > 6:
                raise Mismatch(f"{text} 做了 {used} 次优先级比较、调用了 {calls} 次键的 __hash__ / __eq__：peek、priority、"
                               "in、len 应当只做常数次 dict 或列表操作（上限按 2 次比较、6 次调用算）", call,
                               "比较不超过 2 次，调用不超过 6 次", (used, calls))
            if text.startswith("q.priority"):
                expect_true(isinstance(got, P) and got.v == want, f"{text} 应当返回 P({want})", call, got)
            elif text == "q.peek()":
                low = min(model.values())
                ok = isinstance(got, (tuple, list)) and len(got) == 2 and isinstance(got[1], P) and got[1].v == low
                expect_true(ok, f"q.peek() 应当返回优先级最小的 (key, priority)，优先级是 P({low})", call, got)
            else:
                expect_true(_same(got, want), f"{text} 应当是 {want}", call, got)
    order = sorted(model.values())
    loop = "while len(q):"
    for j, v in enumerate(order):
        size = len(order) - j
        call = calling(code(loop, "x = q.pop()", f"第 {j + 1} 次 pop（len(q) = {size}）"))
        P.count = K.calls = 0
        got = q.pop()
        used, calls = P.count, K.calls
        L = _log2(size)
        _counts_check("q.pop", size, used, calls, 4 * L + 4, 16 * L + 32, call)
        ok = isinstance(got, (tuple, list)) and len(got) == 2 and isinstance(got[1], P) and got[1].v == v
        expect_true(ok, f"q.pop() 应当返回优先级最小的 (key, priority)，优先级是 P({v})", call, got)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    run_script(sol, """
        adj = [[(1, 7), (2, 2)], [(3, 1)], [(1, 3), (3, 8)], [], [(0, 1)]]
        dijkstra_indexed(adj, 0) -> [0, 5, 2, 6, inf]
        adj = [[(1, 7), (2, 2)], [(3, 1)], [(1, 3), (3, 8)], [], [(0, 1)]]
        dijkstra_lazy(adj, 0) -> [0, 5, 2, 6, inf]
    """)


@case("Part 3", title="一个节点、到不了、重边、起点不是 0")
def p3_small(sol):
    run_script(sol, """
        dijkstra_indexed([[]], 0) -> [0]
        dijkstra_lazy([[]], 0) -> [0]
        dijkstra_indexed([[(0, 3)], []], 0) -> [0, inf]     # 自环；节点 1 到不了
        dijkstra_lazy([[(0, 3)], []], 0) -> [0, inf]
        dijkstra_indexed([[(1, 5), (1, 2), (1, 9)], []], 0) -> [0, 2]     # 重边取最短的
        dijkstra_lazy([[(1, 5), (1, 2), (1, 9)], []], 0) -> [0, 2]
        dijkstra_indexed([[(1, 5), (1, 2), (1, 9)], []], 1) -> [inf, 0]
        dijkstra_lazy([[(1, 5), (1, 2), (1, 9)], []], 1) -> [inf, 0]
        dijkstra_indexed([[(1, 4)], [(2, 1), (0, 2)], [(0, 1)]], 2) -> [1, 5, 0]   # 起点是 2
        dijkstra_lazy([[(1, 4)], [(2, 1), (0, 2)], [(0, 1)]], 2) -> [1, 5, 0]
        dijkstra_indexed([[(1, 1), (2, 5)], [(2, 1)], [(3, 1)], [], [(3, 0)]], 1) -> [inf, 0, 1, 2, inf]
        dijkstra_lazy([[(1, 1), (2, 5)], [(2, 1)], [(3, 1)], [], [(3, 0)]], 1) -> [inf, 0, 1, 2, inf]
    """)


@case("Part 3", title="边权为 0 的环")
def p3_zero_weight(sol):
    run_script(sol, """
        dijkstra_lazy([[(0, 0)]], 0) -> [0]     # 权为 0 的自环
        dijkstra_indexed([[(0, 0)]], 0) -> [0]
        dijkstra_lazy([[(1, 0)], [(0, 0)]], 0) -> [0, 0]
        dijkstra_indexed([[(1, 0)], [(0, 0)]], 0) -> [0, 0]
        dijkstra_lazy([[(0, 0), (1, 0)], [(1, 0), (2, 0)], [(0, 0), (1, 0)]], 2) -> [0, 0, 0]
        dijkstra_indexed([[(0, 0), (1, 0)], [(1, 0), (2, 0)], [(0, 0), (1, 0)]], 2) -> [0, 0, 0]
        dijkstra_lazy([[(1, 2)], [(2, 0)], [(1, 0), (3, 1)], []], 0) -> [0, 2, 2, 3]
        dijkstra_indexed([[(1, 2)], [(2, 0)], [(1, 0), (3, 1)], []], 0) -> [0, 2, 2, 3]
    """)


EXAMPLE_ADJ = [[(1, 7), (2, 2)], [(3, 1)], [(1, 3), (3, 8)], [], [(0, 1)]]


class _Count:
    """Counts calls of cls.<name> while it is installed; uninstall() puts the original back."""

    def __init__(self, cls, name):
        self.cls, self.name, self.n = cls, name, 0
        self.had = name in cls.__dict__
        self.orig = cls.__dict__[name] if self.had else None
        orig = getattr(cls, name)

        def wrapper(obj, *args, **kwargs):
            self.n += 1
            return orig(obj, *args, **kwargs)
        setattr(cls, name, wrapper)

    def uninstall(self):
        if self.had:
            setattr(self.cls, self.name, self.orig)
        else:
            delattr(self.cls, self.name)


@case("Part 3", title="两种写法各用各的队列")
def p3_uses_queues(sol):
    want = [0, 5, 2, 6, math.inf]
    for name in ("dijkstra_indexed", "dijkstra_lazy"):
        call = calling(f"adj = {_lit(EXAMPLE_ADJ)}\nprint({name}(adj, 0))")
        counts = []
        try:
            for cls, method in [(sol.MinHeap, "__init__"), (sol.IndexedPQ, "__init__"), (sol.IndexedPQ, "update")]:
                counts.append(_Count(cls, method))
            got = getattr(sol, name)(ref.copy_graph(EXAMPLE_ADJ), 0)
        finally:
            for c in reversed(counts):
                c.uninstall()
        if not _same(got, want):
            raise Mismatch(_dist_diff(got, want), call, want, got)
        heaps, pqs, updates = (c.n for c in counts)
        if name == "dijkstra_indexed":
            expect_true(pqs > 0, "dijkstra_indexed 应当用 Part 2 的 IndexedPQ，但这次调用没有创建 IndexedPQ", call)
            expect_true(updates > 0, "dijkstra_indexed 应当用 update() 做 decrease-key：这个例子里节点 1 先得到距离 7，"
                                     "还在队列里时又改成 5，这时应当调用 update()，但一次也没有调用", call)
        else:
            expect_true(heaps > 0, "dijkstra_lazy 应当用 Part 1 的 MinHeap，但这次调用没有创建 MinHeap", call)
            expect_true(pqs == 0, "dijkstra_lazy 只用 Part 1 的 MinHeap，不应当创建 IndexedPQ", call)


def _dij_outcome(fn, adj, src):
    """None if fn gets the right distances on adj, else what went wrong."""
    want = ref.bellman_ford(adj, src)
    try:
        got = fn(ref.copy_graph(adj), src)
    except Exception as e:
        return {"exc": e, "want": want}
    if _same(got, want):
        return None
    return {"exc": None, "got": got, "want": want}


def check_dijkstra(sol, name, adj, src, what):
    fn = getattr(sol, name)
    calling(show_dij(name, adj, src))
    bad = _dij_outcome(fn, adj, src)
    if bad is None:
        return
    stop = time.process_time() + 0.6
    changed = True
    while changed and time.process_time() < stop:          # drop edges the failure does not need
        changed = False
        for u in range(len(adj)):
            j = 0
            while j < len(adj[u]) and time.process_time() < stop:
                cand = ref.copy_graph(adj)
                del cand[u][j]
                out = _dij_outcome(fn, cand, src)
                if out is not None and type(out["exc"]) is type(bad["exc"]):
                    adj, bad, changed = cand, out, True
                else:
                    j += 1
    call = calling(show_dij(name, adj, src))
    if bad["exc"] is not None:
        raise bad["exc"]
    raise Mismatch(f"{_dist_diff(bad['got'], bad['want'])}（{what}，已自动删去与出错无关的边）", call, bad["want"], bad["got"])


def show_dij(name, adj, src):
    return f"print({name}({_lit(adj)}, {src}))"


@case("Part 3", title="随机小图，和 Bellman–Ford 对照")
def p3_random(sol):
    r = rng()
    for _ in range(250):
        n = r.randint(1, 10)
        adj = ref.random_graph(r, n, r.randint(0, 3 * n), 0, 9)
        src = r.randrange(n)
        for name in ("dijkstra_indexed", "dijkstra_lazy"):
            check_dijkstra(sol, name, adj, src, "随机图，含自环、重边和权为 0 的边")


@case("Part 3", title="每条边只扫描一次")
def p3_work(sol):
    V = 80
    E = V * (V - 1) // 2
    limit = 2 * E + V
    want = list(range(V))
    for name in ("dijkstra_indexed", "dijkstra_lazy"):
        call = calling("\n".join(EDGES_SRC.splitlines() + [
            f"adj = [Edges((v, (v - u) ** 2) for v in range(u + 1, {V})) for u in range({V})]   # E = {E} 条边",
            f"print({name}(adj, 0))   # 应当是 list(range({V}))：一步一步走最短",
            f"print(Edges.reads)       # 应当不超过 2E + V = {limit}"]))
        adj = [Edges((v, (v - u) ** 2) for v in range(u + 1, V)) for u in range(V)]
        Edges.reads = 0
        got = getattr(sol, name)(adj, 0)
        reads = Edges.reads
        if not _same(got, want):
            raise Mismatch(_dist_diff(got, want), call, want, got)
        if reads > limit:
            why = ("弹出过期的记录（距离比 dist[u] 大）时要直接跳过，不要再扫描一遍 u 的出边" if name == "dijkstra_lazy"
                   else "每个节点只应当被弹出、处理一次")
            raise Mismatch(f"{name} 一共读了 {reads} 次边，而图只有 E = {E} 条边：每个节点的出边应当只在它的距离确定时扫描一遍。"
                           f"{why}（上限按 2E + V = {limit} 算）", call, f"不超过 {limit}", reads)


@case("Part 3", cpu=3, title="2×10⁴ 个节点、10⁵ 条边")
def p3_large(sol):
    for name in ("dijkstra_indexed", "dijkstra_lazy"):
        call = calling(f"print({name}(ADJ, {SRC_BIG}))   # ADJ：{V_BIG} 个节点、{E_BIG} 条随机边，边权 0..1000")
        got = getattr(sol, name)(ref.copy_graph(ADJ_BIG), SRC_BIG)
        if not _same(got, DIST_BIG):
            raise Mismatch(_dist_diff(got, DIST_BIG), call, None, None, show=False)
