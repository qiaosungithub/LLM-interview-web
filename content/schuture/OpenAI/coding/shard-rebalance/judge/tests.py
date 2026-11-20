import collections
import json
import os
import subprocess
import sys
import tempfile

from judgelib import Mismatch, calling, case, expect, expect_raises, rng, short, show_call
import ref

EXAMPLE = [("north", 5, 40), ("south", 5, 42), ("east", 5, 44), ("west", 5, 120), ("inland", 6, 42),
           ("coast", 130, 150)]
EXAMPLE_OUT = [("north", 5, 40), ("south", 5, 42), ("east", 41, 44), ("west", 43, 129), ("coast", 130, 150)]


# ---------------------------------------------------------------- Part 1 helpers

def make_shards(sol, specs):
    """Fresh input objects: the answer's own Shard(id, start, end) when it has one that works, else the tests' own."""
    cls = getattr(sol, "Shard", None)
    out = []
    for sid, start, end in specs:
        obj = None
        if isinstance(cls, type):
            try:
                obj = cls(sid, start, end)
                if (obj.id, obj.start, obj.end) != (sid, start, end):
                    obj = None
            except Exception:
                obj = None
        out.append(obj if obj is not None else ref.Shard(sid, start, end))
    return out


def as_tuples(got):
    try:
        items = list(got)
    except TypeError:
        raise Mismatch(f"rebalance 应当返回分片列表，得到的是 {type(got).__name__}", got=got) from None
    out = []
    for k, x in enumerate(items):
        try:
            out.append((x.id, x.start, x.end))
        except AttributeError:
            raise Mismatch(f"返回的第 {k} 个元素（下标从 0 开始）没有 id / start / end 属性：应当返回 Shard 对象",
                           got=got) from None
    return out


def diff_message(want, got):
    counts = collections.Counter(t[0] for t in got)
    dup = [i for i, c in counts.items() if c > 1]
    if dup:
        return f"同一个分片在结果里出现了不止一次：{dup[0]!r}"
    want_ids = [t[0] for t in want]
    extra = [i for i in counts if i not in want_ids]
    missing = [i for i in want_ids if i not in counts]
    if extra or missing:
        parts = []
        if extra:
            parts.append("不该保留却出现在结果里：" + "、".join(map(repr, extra)))
        if missing:
            parts.append("应当保留却不在结果里：" + "、".join(map(repr, missing)))
        return "保留的分片不对。" + "；".join(parts)
    got_by_id = {t[0]: t for t in got}
    for sid, s, e in want:
        _, gs, ge = got_by_id[sid]
        if (gs, ge) != (s, e):
            return f"分片 {sid!r} 的区间不对：期望 [{s}, {e}]，得到 [{gs}, {ge}]"
    return "分片和区间都对，但顺序不对：要按最终的 (start, end, id) 排序返回"


def check(sol, limit, specs, want=None, call=None):
    if want is None:
        lo = min((s[1] for s in specs), default=0)
        hi = max((s[2] for s in specs), default=0)
        want = ref.brute(limit, specs) if hi - lo <= 3000 else ref.sweep(limit, specs)
    calling(call or show_call("rebalance", limit, [ref.Shard(*s) for s in specs]))
    got = as_tuples(sol.rebalance(limit, make_shards(sol, specs)))
    if got != want:
        expect([ref.Shard(*t) for t in got], [ref.Shard(*t) for t in want], msg=diff_message(want, got))


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子 limit = 2")
def p1_example(sol):
    check(sol, 2, EXAMPLE, want=EXAMPLE_OUT)


@case("Part 1", title="输入顺序打乱：先按 (start, end, id) 排序")
def p1_shuffled(sol):
    check(sol, 2, EXAMPLE[::-1], want=EXAMPLE_OUT)
    r = rng()
    for _ in range(6):
        specs = list(EXAMPLE)
        r.shuffle(specs)
        check(sol, 2, specs, want=EXAMPLE_OUT)


@case("Part 1", title="两个分片之间的空洞")
def p1_holes(sol):
    check(sol, 2, [("a", 0, 3), ("b", 10, 12)], want=[("a", 0, 9), ("b", 10, 12)])
    check(sol, 1, [("a", 0, 1), ("b", 5, 6), ("c", 9, 9), ("d", 20, 25)],
          want=[("a", 0, 4), ("b", 5, 8), ("c", 9, 19), ("d", 20, 25)])
    check(sol, 1, [("a", 0, 5), ("b", 2, 8), ("c", 12, 15)])         # the hole follows a shifted shard
    check(sol, 2, [("a", 0, 3), ("b", 0, 3), ("c", 1, 3), ("d", 7, 9)])  # the hole follows a dropped shard


@case("Part 1", title="一个区间被另一个完全包含")
def p1_contained(sol):
    check(sol, 1, [("big", 0, 100), ("small", 10, 20)], want=[("big", 0, 100)])
    check(sol, 2, [("big", 0, 100), ("small", 10, 20)], want=[("big", 0, 100), ("small", 10, 20)])
    check(sol, 1, [("a", 0, 5), ("b", 0, 10)], want=[("a", 0, 5), ("b", 6, 10)])
    check(sol, 2, [("a", 0, 10), ("b", 2, 5), ("c", 20, 25)], want=[("a", 0, 19), ("b", 2, 5), ("c", 20, 25)])
    check(sol, 1, [("a", 0, 10), ("b", 2, 5), ("c", 20, 25)], want=[("a", 0, 19), ("c", 20, 25)])


@case("Part 1", title="几个区间完全相同：按 id 决定保留谁")
def p1_identical(sol):
    same = [("d", 0, 5), ("c", 0, 5), ("b", 0, 5), ("a", 0, 5)]
    check(sol, 1, same, want=[("a", 0, 5)])
    check(sol, 2, same, want=[("a", 0, 5), ("b", 0, 5)])
    check(sol, 3, same, want=[("a", 0, 5), ("b", 0, 5), ("c", 0, 5)])
    check(sol, 5, same, want=[("a", 0, 5), ("b", 0, 5), ("c", 0, 5), ("d", 0, 5)])
    check(sol, 2, [("z", 3, 9), ("y", 3, 9), ("x", 3, 9), ("w", 1, 4), ("v", 12, 14)])


@case("Part 1", title="limit = 1：保留的分片互不重叠")
def p1_limit_one(sol):
    check(sol, 1, [("a", 0, 10), ("b", 2, 15), ("c", 4, 6), ("d", 12, 20), ("e", 30, 31)])
    check(sol, 1, [("a", 0, 0), ("b", 0, 0), ("c", 1, 1), ("d", 0, 2)])
    r = rng()
    for _ in range(120):
        check(sol, 1, ref.random_specs(r, r.randint(1, 7), 0, r.choice([8, 20])))


@case("Part 1", title="丢弃一个分片后，后面的分片照常平移")
def p1_after_drop(sol):
    check(sol, 2, [("a", 0, 10), ("b", 0, 10), ("c", 1, 5), ("d", 2, 8)], want=[("a", 0, 10), ("b", 0, 10)])
    check(sol, 2, [("a", 0, 10), ("b", 0, 20), ("c", 1, 5), ("d", 3, 30)],
          want=[("a", 0, 10), ("b", 0, 20), ("d", 11, 30)])
    check(sol, 3, [("a", 0, 9), ("b", 0, 9), ("c", 0, 9), ("d", 2, 4), ("e", 3, 6), ("f", 4, 12)])


@case("Part 1", title="补洞时并列：先比起点，再比 id")
def p1_hole_tie(sol):
    # y (processed first) and x both end up as [3, 6]: the same step-4 start, so the hole goes to x, the smaller id
    check(sol, 2, [("m", 0, 2), ("n", 0, 2), ("y", 1, 6), ("x", 3, 6), ("p", 10, 12)],
          want=[("m", 0, 2), ("n", 0, 2), ("y", 3, 6), ("x", 3, 9), ("p", 10, 12)])
    check(sol, 3, [("a", 0, 5), ("b", 3, 5), ("c", 10, 12)], want=[("a", 0, 9), ("b", 3, 5), ("c", 10, 12)])
    check(sol, 2, [("q", 0, 4), ("p", 0, 4), ("r", 8, 9)], want=[("q", 0, 4), ("p", 0, 7), ("r", 8, 9)])


@case("Part 1", title="空列表")
def p1_empty(sol):
    check(sol, 1, [], want=[])
    check(sol, 3, [], want=[])


@case("Part 1", title="只有一个分片")
def p1_single(sol):
    check(sol, 1, [("only", 7, 7)], want=[("only", 7, 7)])
    check(sol, 2, [("only", -3, 12)], want=[("only", -3, 12)])


@case("Part 1", title="负数键")
def p1_negative(sol):
    specs = [("a", -20, -5), ("b", -25, -10), ("c", -8, 3), ("d", -30, -28), ("e", -27, -26)]
    for limit in (1, 2, 3):
        check(sol, limit, specs)


@case("Part 1", title="limit 不小于分片数：只补洞")
def p1_big_limit(sol):
    specs = [("a", 0, 4), ("b", 2, 9), ("c", 2, 3), ("d", 15, 18), ("e", 16, 30)]
    check(sol, 5, specs, want=[("a", 0, 4), ("c", 2, 3), ("b", 2, 14), ("d", 15, 18), ("e", 16, 30)])
    check(sol, 9, specs)


@case("Part 1", title="随机小输入（对照逐个键模拟规则）")
def p1_random_small(sol):
    r = rng()
    for _ in range(400):
        span = r.choice([6, 12, 40])
        offset = r.choice([0, -15, 100])
        check(sol, r.randint(1, 4), ref.random_specs(r, r.randint(1, 8), offset, offset + span))


@case("Part 1", cpu=2, title="键达到 ±10⁹：不能逐个键扫描")
def p1_huge_keys(sol):
    big = 10 ** 9
    check(sol, 1, [("all", -big, big)], want=[("all", -big, big)])
    check(sol, 1, [("lo", -big, -big + 5), ("hi", big - 5, big)], want=[("lo", -big, big - 6), ("hi", big - 5, big)])
    check(sol, 2, [("a", 0, big), ("b", 0, big), ("c", 5, big)], want=[("a", 0, big), ("b", 0, big)])
    check(sol, 1, [("a", -big, big), ("b", 0, big - 1), ("c", big, big)], want=[("a", -big, big)])


@case("Part 1", title="随机大坐标（±10⁹）")
def p1_random_big(sol):
    r = rng()
    for _ in range(60):
        n, limit = r.randint(1, 80), r.randint(1, 5)
        specs = ref.random_specs(r, n, -10 ** 9, 10 ** 9, max_len=r.choice([10 ** 6, 10 ** 8, 10 ** 9]))
        call = show_call("rebalance", limit, [ref.Shard(*s) for s in specs])
        if len(call) > 2500:
            call = f"rebalance({limit}, <{n} 个随机分片，键在 ±10⁹ 之内>)"
        check(sol, limit, specs, want=ref.sweep(limit, specs), call=call)


# ---------------------------------------------------------------- Part 2 helpers

EXAMPLE_KEYS = range(30_000)
LOW, HIGH = 0.4, 2.0            # "roughly 1/N": every shard owns between 0.4 and 2 times its fair share
KEYS = list(range(1500)) + [-1, -2, -999, -123_456_789, -10 ** 9, 10 ** 9, 10 ** 9 - 1, 987_654_321]


def replay(lines, keep=14):
    """The calls as code; a long replay keeps its first line and its last keep - 1 lines."""
    if len(lines) > keep:
        lines = [lines[0], f"# …（省略了中间 {len(lines) - keep + 1} 步，所以不能直接粘贴重现）"] + lines[-(keep - 1):]
    return "\n".join(lines)


class Script:
    """One router plus the calls made on it so far, kept as code that can be pasted into 自定义运行."""

    def __init__(self, sol, name="r"):
        self.name, self.lines, self.ids = name, [f"{name} = ShardRouter()"], set()
        calling(self.text())
        self.router = sol.ShardRouter()

    def text(self, extra=(), keep=14):
        return replay(self.lines + list(extra), keep)

    def expect_error(self, exc, method, arg, why):
        """The call must raise exc; afterwards the replay keeps it inside try/except so it can still be pasted."""
        line = f"{self.name}.{method}({arg!r})"
        call = calling(self.text([f"{line}   # {why}，应当抛出 {exc.__name__}"]))
        expect_raises(exc, getattr(self.router, method), arg, call=call)
        self.lines.append(f"try:\n    {line}   # {why}\nexcept {exc.__name__}:\n    pass")

    def add(self, x):
        self.lines.append(f"{self.name}.add_shard({x!r})")
        calling(self.text())
        self.router.add_shard(x)
        self.ids.add(x)

    def remove(self, x):
        self.lines.append(f"{self.name}.remove_shard({x!r})")
        calling(self.text())
        self.router.remove_shard(x)
        self.ids.discard(x)

    def owners(self, keys, desc):
        """locate(k) for every k in keys; each must be a current shard id."""
        calling(self.text([f"for k in {desc}: {self.name}.locate(k)"]))
        out = [self.router.locate(k) for k in keys]
        for k, owner in zip(keys, out):
            if not (isinstance(owner, str) and owner in self.ids):
                raise Mismatch(f"locate({k}) 返回了 {short(owner, 80)}，它不是当前的分片 id（当前：{sorted(self.ids)}）",
                               call=self.text([f"print({self.name}.locate({k}))"]), got=owner)
        return out


def check_shares(script, owners, keys, what):
    n = len(script.ids)
    count = collections.Counter(owners)
    for sid in sorted(script.ids):
        share = count[sid] / len(keys)
        if not LOW / n <= share <= HIGH / n:
            raise Mismatch(f"{n} 个分片时，{sid!r} 拥有 {share:.1%} 的键，公平份额是 {1 / n:.1%}（1/{n}）；"
                           f"「大致均匀」这里按每个分片在公平份额的 {LOW:g} 到 {HIGH:g} 倍之间判断",
                           call=script.text([f"# 统计 {what} 中每个分片拥有的键数"]), show=False)


def check_moves(script, keys, before, after, op, x):
    """After add_shard(x) a moved key must now be x's; after remove_shard(x) it must have been x's."""
    for k, b, a in zip(keys, before, after):
        if b == a or (op == "add" and a == x) or (op == "remove" and b == x):
            continue
        lines = script.lines[:-1] + [f"print({script.name}.locate({k}))   # {b!r}",
                                     script.lines[-1] + "   # <- 这一步",
                                     f"print({script.name}.locate({k}))   # {a!r}"]
        rule = (f"add_shard({x!r}) 之后，换了主人的键现在都应当归 {x!r}" if op == "add"
                else f"remove_shard({x!r}) 之后，换了主人的键原来都应当归 {x!r}")
        raise Mismatch(f"键 {k} 从 {b!r} 换到了 {a!r}；题目要求 {rule}（只迁移必须迁移的键）",
                       call=replay(lines, 16), show=False)


def same_routing(sol, first, second, keys, desc):
    """Two routers holding the same set of shard ids must route every key alike."""
    a, b = first.owners(keys, desc), second.owners(keys, desc)
    for k, x, y in zip(keys, a, b):
        if x != y:
            raise Mismatch(f"两个路由器的分片集合相同（{sorted(first.ids)}），locate({k}) 却不同："
                           f"{first.name} 得到 {x!r}，{second.name} 得到 {y!r}；题目要求结果只取决于键和当前的分片集合，"
                           f"与加入的先后次序无关",
                           call=first.text([f"print({first.name}.locate({k}))   # {x!r}"], keep=20) + "\n" +
                           second.text([f"print({second.name}.locate({k}))   # {y!r}"], keep=20), show=False)


CHILD = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("answer", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ids, keys = json.load(open(sys.argv[2]))
r = mod.ShardRouter()
for s in ids:
    r.add_shard(s)
json.dump([r.locate(k) for k in keys], open(sys.argv[3], "w"))
"""


def in_other_process(sol, ids, keys, hash_seed, call):
    """Builds the same router in a fresh Python process with another PYTHONHASHSEED and returns its locate results."""
    folder = tempfile.mkdtemp()
    src, dst = os.path.join(folder, "in.json"), os.path.join(folder, "out.json")
    with open(src, "w") as f:
        json.dump([ids, keys], f)
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    path = getattr(sol, "__file__", None) or "/work/answer.py"
    proc = subprocess.run([sys.executable, "-s", "-c", CHILD, path, src, dst], env=env, cwd=folder,
                          capture_output=True, text=True, timeout=20)
    if proc.returncode != 0 or not os.path.exists(dst):
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise Mismatch(f"在另一个 Python 进程里做同样的调用时出错：\n{tail}", call=call, show=False)
    with open(dst) as f:
        return json.load(f)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, cpu=4, title="题目例子：amber、cobalt、jade，再加 slate、删 cobalt")
def p2_example(sol):
    keys, desc = EXAMPLE_KEYS, "range(30_000)"
    s = Script(sol)
    for x in ("amber", "cobalt", "jade"):
        s.add(x)
    first = s.owners(keys, desc)
    check_shares(s, first, keys, desc)
    s.add("slate")
    second = s.owners(keys, desc)
    check_moves(s, keys, first, second, "add", "slate")
    moved = sum(a != b for a, b in zip(first, second))
    if not LOW / 4 <= moved / len(keys) <= HIGH / 4:
        raise Mismatch(f"add_shard('slate') 之后 {moved / len(keys):.1%} 的键换了主人，题目说大约四分之一"
                       f"（这里要求在 {LOW / 4:.0%} 到 {HIGH / 4:.0%} 之间）", call=s.text(), show=False)
    s.remove("cobalt")
    third = s.owners(keys, desc)
    check_moves(s, keys, second, third, "remove", "cobalt")
    moved = sum(a != b for a, b in zip(second, third))
    if not LOW / 4 <= moved / len(keys) <= HIGH / 4:
        raise Mismatch(f"remove_shard('cobalt') 之后 {moved / len(keys):.1%} 的键换了主人，题目说同样大约四分之一"
                       f"（这里要求在 {LOW / 4:.0%} 到 {HIGH / 4:.0%} 之间）", call=s.text(), show=False)


@case("Part 2", title="ValueError 和 LookupError")
def p2_errors(sol):
    s = Script(sol)
    s.expect_error(LookupError, "locate", 5, "还没有分片")
    s.add("a")
    s.expect_error(ValueError, "add_shard", "a", "已经有 'a'")
    s.expect_error(ValueError, "remove_shard", "b", "没有 'b'")
    s.add("b")
    fresh = Script(sol, "fresh")
    fresh.add("a")
    fresh.add("b")
    same_routing(sol, fresh, s, KEYS[:300], "range(300)")      # the failed calls left the set {a, b} unchanged
    s.remove("a")
    s.expect_error(ValueError, "remove_shard", "a", "'a' 已经删掉了")
    s.remove("b")
    s.expect_error(LookupError, "locate", 5, "分片已经删光")
    s.add("c")
    s.owners(KEYS[:300], "range(300)")


@case("Part 2", title="只有一个分片：每个键都归它")
def p2_single(sol):
    s = Script(sol)
    s.add("solo")
    s.owners(KEYS, "<0..1499 以及几个 ±10⁹ 以内的键>")


@case("Part 2", title="负数键和 ±10⁹ 的键")
def p2_key_range(sol):
    s = Script(sol)
    for x in ("amber", "cobalt", "jade"):
        s.add(x)
    keys = [-1, -2, -17, -999, -123_456_789, -10 ** 9, 10 ** 9, 10 ** 9 - 1, 987_654_321, 0]
    before = s.owners(keys, repr(keys))
    s.add("slate")
    check_moves(s, keys, before, s.owners(keys, repr(keys)), "add", "slate")


@case("Part 2", cpu=4, title="结果与分片加入的先后次序无关")
def p2_order_independent(sol):
    ids = ["amber", "cobalt", "jade", "slate", "ivory", "onyx", "pearl"]
    desc = "<0..1499 以及几个 ±10⁹ 以内的键>"
    base = Script(sol, "r1")
    for x in ids:
        base.add(x)
    other = Script(sol, "r2")
    for x in reversed(ids):
        other.add(x)
    same_routing(sol, base, other, KEYS, desc)
    r = rng()
    shuffled = Script(sol, "r3")
    for x in r.sample(ids, len(ids)):
        shuffled.add(x)
    same_routing(sol, base, shuffled, KEYS, desc)
    history = Script(sol, "r4")
    for x in ["tmp-1", "jade", "onyx", "tmp-2", "amber"]:
        history.add(x)
    history.remove("tmp-1")
    history.remove("jade")
    for x in ["pearl", "cobalt", "slate", "jade", "ivory"]:
        history.add(x)
    history.remove("tmp-2")
    same_routing(sol, base, history, KEYS, desc)


@case("Part 2", cpu=6, title="增删分片只迁移必须迁移的键")
def p2_minimal_moves(sol):
    r = rng()
    pool = ["amber", "cobalt", "jade", "slate", "ivory", "onyx", "pearl", "ruby", "teal", "umber"]
    desc = "<0..1499 以及几个 ±10⁹ 以内的键>"
    for _ in range(6):
        s = Script(sol)
        owners = None
        for _ in range(12):
            present = sorted(s.ids)
            if present and (len(present) >= 7 or r.random() < 0.4):
                x = r.choice(present)
                s.remove(x)
                op = "remove"
            else:
                x = r.choice([p for p in pool if p not in s.ids])
                s.add(x)
                op = "add"
            if not s.ids:
                s.expect_error(LookupError, "locate", 0, "分片已经删光")
                owners = None
                continue
            now = s.owners(KEYS, desc)
            if owners is not None:
                check_moves(s, KEYS, owners, now, op, x)
            owners = now


@case("Part 2", cpu=4, title="20 个分片：每个大约拥有 1/20 的键")
def p2_even_twenty(sol):
    s = Script(sol)
    for i in range(20):
        s.add(f"shard-{i}")
    owners = s.owners(EXAMPLE_KEYS, "range(30_000)")
    check_shares(s, owners, EXAMPLE_KEYS, "range(30_000)")


@case("Part 2", cpu=4, title="先加 10 个再删 3 个：剩下的仍大致均匀")
def p2_even_after_removals(sol):
    s = Script(sol)
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    for x in names:
        s.add(x)
    for x in ("charlie", "golf", "alpha"):
        s.remove(x)
    owners = s.owners(EXAMPLE_KEYS, "range(30_000)")
    check_shares(s, owners, EXAMPLE_KEYS, "range(30_000)")


@case("Part 2", cpu=4, title="换一个进程（重启之后）结果不变")
def p2_other_process(sol):
    ids = ["amber", "cobalt", "jade", "slate", "ivory"]
    keys = list(range(1000)) + [-7, -10 ** 9, 10 ** 9]
    s = Script(sol)
    for x in ids:
        s.add(x)
    here = s.owners(keys, "range(1000)")
    own = int(os.environ.get("PYTHONHASHSEED", "0") or 0)
    for other in (own + 1001, own + 2002):
        there = in_other_process(sol, ids, keys, other, s.text())
        for k, a, b in zip(keys, here, there):
            if a != b:
                raise Mismatch(f"同样的调用放到另一个 Python 进程里（PYTHONHASHSEED={other}）运行，locate({k}) 得到 {b!r}，"
                               f"这个进程得到 {a!r}；题目要求结果与进程无关（重启之后同一个键仍然落到同一个分片）。"
                               f"注意 Python 内置的 hash() 对 str 的结果每个进程都不同",
                               call=s.text([f"print(r.locate({k}))   # 这个进程 {a!r}，另一个进程 {b!r}"]), show=False)
