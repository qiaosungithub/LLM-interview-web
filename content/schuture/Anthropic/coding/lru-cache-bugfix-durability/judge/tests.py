import json
import os
import threading
import time

from judgelib import Mismatch, calling, case, expect, expect_raises, expect_true, info, rng, short, show_call
import ref
from ref import (LOG, Caller, Dirs, Fs, Func, HeldFunc, Model, canon, fast_switching, other_value, sizes,
                 snap_key, snapshot, stats, value_of, wait_parked, with_file)

SHOW = 16           # a replay lists at most this many lines
F_NOTE = "f = lambda *args, **kwargs: repr((args, sorted(kwargs.items())))"
FILL = 900000       # probe calls r(900000), r(900001), ... are never part of a test's own calls


# ---------------------------------------------------------------- how a sequence is shown

class Story:
    """Lines of code with a comment each, shown as the replay of what the test did."""

    def __init__(self, entries=()):
        self.entries = [[code, note] for code, note in entries]

    def add(self, code, note=""):
        self.entries.append([code, note])
        return self

    def note_last(self, note):
        self.entries[-1][1] = note

    def copy(self):
        return Story(self.entries)

    def text(self, mark="<- 这一步"):
        ents = [list(e) for e in self.entries]
        if mark and ents:
            ents[-1][1] = f"{ents[-1][1]}  {mark}" if ents[-1][1] else mark
        skipped = max(0, len(ents) - SHOW)
        if skipped:
            ents = ents[:1] + [[f"# …（省略了中间 {skipped} 行）", ""]] + ents[1 + skipped:]
        width = max([len(c) for c, n in ents if n] or [0]) + 3
        return "\n".join((c.ljust(width) + "# " + n) if n else c for c, n in ents)


def call_of(name, args, kwargs):
    return show_call(name, *args, **kwargs)


def same_value(got, want):
    """A value as it may come back from disk: a tuple may have become a list; everything else is exact (a list
    stays a list)."""
    if isinstance(want, (list, tuple)):
        kinds = (list, tuple) if isinstance(want, tuple) else list
        return isinstance(got, kinds) and len(got) == len(want) and all(map(same_value, got, want))
    if isinstance(want, dict):
        return isinstance(got, dict) and set(got) == set(want) and all(same_value(got[k], want[k]) for k in want)
    return type(got) is type(want) and got == want


def checked_call(inst, fn, model, st, name, args, kwargs, value=None):
    """One call of a cache, checked against the model: hit or miss, the value, and the hits/misses counters."""
    hit, want = model.call(args, kwargs, value or fn.value)
    line = call_of(name, args, kwargs)
    st.add(line)
    calling(st.text())
    n = len(fn.calls)
    got = inst(*args, **kwargs)
    called = len(fn.calls) - n
    if hit and called:
        raise Mismatch(f"{line} 应当命中：同一次调用已经在缓存里（还没被淘汰），却又调用了 func", st.text(),
                       "命中", f"未命中（func 被调用了 {called} 次）")
    if not hit and called == 0:
        raise Mismatch(f"{line} 应当未命中：缓存里没有这次调用，却没有调用 func，返回了 {short(got, 200)}——它被错当成了"
                       "之前的另一次调用（键碰撞），或者那一项本该已被淘汰", st.text(), "未命中", "命中")
    if not hit and called != 1:
        raise Mismatch(f"{line} 未命中时应当只调用 func 一次，实际调用了 {called} 次", st.text())
    expect(got, want, call=st.text(), msg=f"{line} 的返回值不对")
    st.note_last("命中" if hit else "未命中")
    got_counts = (getattr(inst, "hits", None), getattr(inst, "misses", None))
    expect(got_counts, (model.hits, model.misses), call=st.text(),
           msg=f"(hits, misses) 计数不对：应为 ({model.hits}, {model.misses})")
    return got


# ---------------------------------------------------------------- Part 1

def memo_run(sol, capacity, steps):
    fn = Func()
    model = Model(capacity)
    st = Story([(f"cache = Memoize(f, capacity={capacity})", F_NOTE)])
    calling(st.text())
    cache = sol.Memoize(fn, capacity)
    for args, kwargs in steps:
        checked_call(cache, fn, model, st, "cache", args, dict(kwargs))
    return cache, fn, model, st


@case("Part 1", sample=True, title="题目示例")
def part1_statement_example(sol):
    "The statement's example, then two calls that read the final state [f((1,2)), f(1,3)]."
    fn = Func(lambda a, k: (a, k))
    model = Model(2)
    st = Story([("cache = Memoize(lambda *args, **kwargs: (args, kwargs), capacity=2)", "")])
    calling(st.text())
    cache = sol.Memoize(fn, 2)
    steps = [((1,), {"b": 2, "c": 3}), ((1,), {"c": 3, "b": 2}), (((1, 2),), {}), ((1, 2), {}),
             (((1, 2),), {}), ((1, 3), {})]
    for args, kwargs in steps:
        checked_call(cache, fn, model, st, "cache", args, kwargs)
    st.add("# 题目说此时缓存是 [f((1,2)), f(1,3)]，下面两次调用都应命中")
    checked_call(cache, fn, model, st, "cache", ((1, 2),), {})
    checked_call(cache, fn, model, st, "cache", (1, 3), {})


@case("Part 1", title="关键字参数换顺序仍是同一次调用")
def part1_kwargs_order(sol):
    "Keyword arguments in any order hit the same entry, including values that cannot be compared with each other."
    r = rng("order")
    kw = {"a": 1, "b": "x", "c": None, "d": (1, 2), "e": 2.5}
    names = list(kw)
    steps = []
    for _ in range(12):
        r.shuffle(names)
        steps.append(((7,), {k: kw[k] for k in names}))
    steps += [((), {"y": 1, "x": 2}), ((), {"x": 2, "y": 1}), ((1,), {"q": 0, "p": "s"}), ((1,), {"p": "s", "q": 0})]
    memo_run(sol, 4, steps)


@case("Part 1", title="位置参数和关键字参数切分不同的调用互不碰撞")
def part1_split_collisions(sol):
    "Calls whose flattened arguments look alike but are split differently are different calls."
    pairs = [
        (((1, 2),), {}), ((1, 2), {}),          # f((1, 2)) vs f(1, 2)
        ((), {}), (((),), {}),                  # f() vs f(())
        (("b", 2), {}), ((), {"b": 2}),         # f("b", 2) vs f(b=2)
        ((("b", 2),), {}),                      # f(("b", 2))
        ((1, ("b", 2)), {}), ((1,), {"b": 2}),  # f(1, ("b", 2)) vs f(1, b=2)
        ((1,), {}), (((1,),), {}),              # f(1) vs f((1,))
        ((("c", 3), ("b", 2)), {}), ((), {"c": 3, "b": 2}), ((("b", 2), ("c", 3)), {}),
        ((1, ("b", 2), ("c", 3)), {}), ((1,), {"b": 2, "c": 3}),
    ]
    steps = pairs + list(reversed(pairs))       # the second round: every one is still cached (all hits)
    memo_run(sol, len(pairs), steps)


@case("Part 1", title="命中也更新最近使用顺序")
def part1_hit_refreshes(sol):
    "A hit makes its entry the most recently used one, so it is evicted last."
    memo_run(sol, 2, [((1,), {}), ((2,), {}), ((1,), {}), ((3,), {}), ((1,), {}), ((2,), {})])
    memo_run(sol, 3, [((1,), {}), ((2,), {}), ((3,), {}), ((1,), {}), ((2,), {}), ((4,), {}),
                      ((1,), {}), ((2,), {}), ((3,), {})])
    memo_run(sol, 1, [((1,), {}), ((1,), {}), ((2,), {}), ((1,), {})])


@case("Part 1", title="不可哈希的参数抛 TypeError")
def part1_unhashable(sol):
    "An unhashable argument fails the way it fails as a dict key: TypeError."
    for args, kwargs in [(([1, 2],), {}), ((1, {"a": 1}), {}), ((), {"x": [1]}), ((1,), {"b": 2, "c": {3}})]:
        cache = sol.Memoize(Func(), 4)
        line = call_of("cache", args, kwargs)
        expect_raises(TypeError, cache, *args, call=f"cache = Memoize(f, capacity=4)\n{line}", **kwargs)


SHAPES = [
    ((1,), {}), ((1, 2), {}), (((1, 2),), {}), ((1,), {"b": 2}), ((), {"b": 2}), (("b", 2), {}),
    ((("b", 2),), {}), ((), {}), (((),), {}), ((1,), {"b": 2, "c": 3}), ((1,), {"c": 3, "b": 2}), ((2,), {}),
    ((1, ("b", 2)), {}), ((), {"a": 1, "b": 2}), ((), {"b": 2, "a": 1}), ((None,), {}), (("x",), {"k": (1, 2)}),
    ((1.5, "s"), {}), (("s",), {"z": 1.5}),
]


@case("Part 1", title="随机调用序列（对照 LRU 模型）")
def part1_random(sol):
    "Random sequences over call shapes that differ only in how arguments are split, checked call by call."
    r = rng("seq")
    for capacity in (1, 2, 3, 4, 3, 2):
        steps, recent = [], []
        for _ in range(40):
            shape = r.choice(recent) if recent and r.random() < 0.45 else r.choice(SHAPES)
            recent = (recent + [shape])[-4:]
            steps.append(shape)
        memo_run(sol, capacity, steps)


# ---------------------------------------------------------------- Part 2: stores, crashes and recovery

class Disk:
    """One case's temporary directories and fsync records."""

    def __init__(self, sol):
        self.sol = sol
        self.dirs = Dirs()
        self.fs = Fs(sol, self.dirs)

    def __enter__(self):
        self.fs.__enter__()
        return self

    def __exit__(self, *exc):
        self.fs.__exit__(*exc)
        self.dirs.close()

    def recover(self, cls_name, capacity, snap, st, g):
        """A new instance on a copy of snap (in a fresh directory); st ends with its constructor line."""
        d = self.dirs.restore(snap)
        calling(st.text(mark="<- 恢复时出错"))
        r = getattr(self.sol, cls_name)(g, capacity, os.path.join(d, LOG))
        return r, d


def state_text(model, name="f"):
    return "[" + ", ".join(call_of(name, a, k) for a, k, _ in model.entries()) + "]"


def probe(r, g, st, args, kwargs, hit, want=None, why=""):
    """One call on a recovered instance r (whose func is g): it must hit with value `want`, or miss."""
    line = call_of("r", args, kwargs)
    st.add(line, f"期望命中，返回 {short(want, 80)}" if hit else "期望未命中")
    calling(st.text())
    n = len(g.calls)
    got = r(*args, **kwargs)
    called = len(g.calls) - n
    if hit and called:
        raise Mismatch(why or f"{line} 应当命中：崩溃前它在缓存里并且已经持久化，恢复后却不在缓存里（func 被调用了）",
                       st.text(), "命中", "未命中")
    if not hit and not called:
        raise Mismatch(why or f"{line} 应当未命中，却命中了", st.text(), "未命中", f"命中，返回 {short(got, 200)}")
    if hit and not same_value(got, want):
        raise Mismatch(f"{line} 命中了，但返回值和崩溃前缓存的值不一样（恢复时值必须来自磁盘；只有其中的 tuple 可以变成 list，"
                       "其他部分都要原样恢复，list 也必须还是 list）", st.text(), want, got)
    st.note_last("命中" if hit else "未命中")


def check_store(disk, cls_name, capacity, snap, model, universe, story, g_value=other_value):
    """Recover from snap (each probe on its own copy) and check that the cache is exactly `model`: every entry with
    its value (a hit), the order least to most recently used (fill up, add one call, then every entry must have been
    evicted in that order), and no other entry (each other call made so far must miss)."""
    exp = model.entries()
    keys = {canon(a, k) for a, k, _ in exp}
    state = state_text(model)
    ctor = f'r = {cls_name}(g, capacity={capacity}, path="cache.log")'
    ctor_note = f"g 和 f 不同；恢复后的缓存应为（LRU -> MRU）{state}"

    # entries and values; hits/misses start at 0
    st = story.copy().add(ctor, ctor_note)
    g = Func(g_value)
    r, d = disk.recover(cls_name, capacity, snap, st, g)
    expect_true((getattr(r, "hits", None), getattr(r, "misses", None)) == (0, 0),
                "新实例的 hits 和 misses 应当都从 0 开始（它们不持久化）", st.text(),
                (getattr(r, "hits", None), getattr(r, "misses", None)))
    for a, k, v in exp:
        probe(r, g, st, a, k, True, v)
    del r
    disk.dirs.drop(d)

    # order: after filling up and adding one new call, the entries must go in exactly the order of `exp`
    if exp:
        st = story.copy().add(ctor, ctor_note)
        g = Func(g_value)
        r, d = disk.recover(cls_name, capacity, snap, st, g)
        st.add(f"# 先补 {capacity - len(exp) + 1} 个新调用把缓存挤满并淘汰一项，再按 LRU 顺序逐个调用原有条目：每次都应当"
               "未命中，因为每次被淘汰的都正好是下一个")
        for j in range(capacity - len(exp) + 1):
            probe(r, g, st, (FILL + j,), {}, False, why=f"r({FILL + j}) 是一个从未调用过的新调用，却命中了")
        for a, k, v in exp:
            line = call_of("r", a, k)
            probe(r, g, st, a, k, False, why=f"{line} 应当已被淘汰：恢复后的最近使用顺序应为 {state}，补满容量再加入一个"
                  f"新调用之后，它就是被淘汰的那一项；它却命中了，说明恢复出来的最近使用顺序不对")
        del r
        disk.dirs.drop(d)

    # nothing else: every other call made so far must miss
    for a, k in universe:
        if canon(a, k) in keys:
            continue
        st = story.copy().add(ctor, ctor_note)
        g = Func(g_value)
        r, d = disk.recover(cls_name, capacity, snap, st, g)
        line = call_of("r", a, k)
        probe(r, g, st, a, k, False, why=f"{line} 应当未命中：崩溃前一刻它不在缓存里（已被淘汰，或者那次调用没有完成），"
              f"恢复出的缓存却有它")
        del r
        disk.dirs.drop(d)


CRASH = "# —— 在这里模拟崩溃：只保留此刻磁盘上的文件 ——"


class Trial:
    """One store under test: the instance w, its directory, the model of what it must hold, and the story so far."""

    def __init__(self, disk, cls_name, capacity, func=None, note=F_NOTE, head=None):
        self.disk, self.cls_name, self.capacity = disk, cls_name, capacity
        self.d = disk.dirs.new()
        self.f = func or Func()
        self.model = Model(capacity)
        self.universe = []
        self.story = Story([(head or f'w = {cls_name}(f, capacity={capacity}, path="cache.log")', note)])
        calling(self.story.text())
        self.w = getattr(disk.sol, cls_name)(self.f, capacity, os.path.join(self.d, LOG))

    def call(self, args, kwargs=None, disk_checks=True):
        kwargs = dict(kwargs or {})
        if all(canon(a, k) != canon(args, kwargs) for a, k in self.universe):
            self.universe.append((args, kwargs))
        before = snapshot(self.d) if disk_checks else None
        st_before = stats(self.d)
        got = checked_call(self.w, self.f, self.model, self.story, "w", args, kwargs)
        if disk_checks:
            self._check_disk(before, st_before, args, kwargs, got)
        return got

    def compact(self):
        self.story.add("w.compact()")
        calling(self.story.text())
        self.w.compact()

    def _check_disk(self, before, st_before, args, kwargs, value):
        line = call_of("w", args, kwargs)
        after = snapshot(self.d)
        for rel, old in before["files"].items():
            new = after["files"].get(rel)
            if new is None:
                raise Mismatch(f"{line} 删除了文件 {rel}：在 compact() 之外，存储只能在日志末尾追加", self.story.text())
            if not new.startswith(old):
                raise Mismatch(f"{line} 改写了文件 {rel} 里已有的内容，而不是只在末尾追加：在 compact() 之外，日志只能追加、"
                               "不能就地重写（也不能每次调用都重写整个缓存）", self.story.text())
        grown = sum(len(data) - len(before["files"].get(rel, b"")) for rel, data in after["files"].items())
        limit = 256 + 4 * len(json.dumps([args, kwargs, value]))
        if grown > limit:
            raise Mismatch(f"{line} 让存储增长了 {grown} 字节，超过了按这次调用本身的大小给出的上限 {limit} 字节："
                           "记录一次调用的空间只能和这一次调用成正比，不能和 capacity 或调用历史有关", self.story.text())
        st_after = stats(self.d)
        changed = [rel for rel, s in st_after.items() if st_before.get(rel) != s]
        bad = self.disk.fs.unsynced(self.d, changed)
        if bad:
            rel = bad[0]
            dev, ino, size = st_after[rel]
            synced = self.disk.fs.synced.get((dev, ino))
            raise Mismatch(f"{line} 返回时，它写进 {rel} 的内容还没有 fsync（文件现在 {size} 字节，"
                           + (f"最后一次 fsync 时是 {synced} 字节" if synced is not None else "从没有 fsync 过")
                           + "）：题目要求 __call__ 返回之前，这次调用的记录已经 flush 并 fsync", self.story.text())

    def crash_story(self, note=CRASH):
        return self.story.copy().add(note)

    def check(self, snap=None, note=CRASH, model=None, g_value=other_value):
        check_store(self.disk, self.cls_name, self.capacity, snap or snapshot(self.d), model or self.model,
                    self.universe, self.crash_story(note), g_value)


def keys_of(*ns):
    return [((n,), {}) for n in ns]


@case("Part 2", sample=True, title="题目示例：w(5) 的记录写到一半时崩溃")
def part2_statement_example(sol):
    "w(3), w(4), w(3), then a crash while w(5)'s record is written: every cut inside it recovers [4 -> 16, 3 -> 9]."
    square = lambda a, k: a[0] * a[0]          # noqa: E731
    with Disk(sol) as disk:
        t = Trial(disk, "DurableMemoize", 2, func=Func(square), note="",
                  head='w = DurableMemoize(lambda n: n * n, capacity=2, path="cache.log")')
        for n in (3, 4, 3):
            t.call((n,), disk_checks=False)
        before, model3, story3 = snapshot(t.d), t.model.copy(), t.story.copy()
        t.call((5,), disk_checks=False)
        after = snapshot(t.d)
        names = sorted(set(before["files"]) | set(after["files"]))
        changed = [rel for rel in names if before["files"].get(rel) != after["files"].get(rel)]
        expect_true(changed, "w(5) 返回时磁盘上的文件没有任何变化：它的记录没有写到磁盘上", t.story.text())
        rel = changed[0]
        old, new = before["files"].get(rel, b""), after["files"].get(rel)
        if len(changed) > 1 or new is None or not new.startswith(old):
            info("w(5) 不是只在一个文件末尾追加，无法模拟写到一半的记录；只检查了 w(5) 返回之后崩溃的恢复结果")
            check_store(disk, "DurableMemoize", 2, after, t.model, keys_of(3, 4, 5), t.crash_story(), square)
            return
        lo, hi = len(old), len(new)
        for cut in range(lo + 1, hi):
            snap = with_file(before, rel, new[:cut])
            st = story3.copy().add(f"# —— 进程在写 w(5) 的记录时被杀死：这条记录只写了前 {cut - lo} 个字节（共 {hi - lo} 字节）——")
            check_store(disk, "DurableMemoize", 2, snap, model3, keys_of(3, 4, 5), st, square)
        info(f"w(5) 的记录共 {hi - lo} 字节，在其中每一个位置截断都检查了恢复结果")


@case("Part 2", title="命中也要持久化（最近使用顺序）")
def part2_hits_durable(sol):
    "A hit changes the order, and that change must survive a crash too."
    with Disk(sol) as disk:
        t = Trial(disk, "DurableMemoize", 2)
        for n in (1, 2, 1):
            t.call((n,))
        t.check()
        t.call((3,))
        t.check()
        t = Trial(disk, "DurableMemoize", 3)
        for n in (1, 2, 3, 1, 2):
            t.call((n,))
            t.check()


@case("Part 2", title="返回之前已经 fsync")
def part2_fsync_before_return(sol):
    "After every call returns, all it wrote has been fsynced (os.fsync / os.fdatasync, or a file opened O_SYNC)."
    with Disk(sol) as disk:
        t = Trial(disk, "DurableMemoize", 2)
        for n in (1, 2, 1, 3, 3, 2, 1):
            t.call((n,))
        info(f"测试记录到 {disk.fs.syncs} 次 fsync")


def random_ops(r, n_ops, n_keys, compacts):
    """A random list of ("call", args, kwargs) / ("compact",) operations over a few call shapes."""
    shapes = [((i,), {}) for i in range(n_keys)] + [((), {"n": 1}), ((1,), {"tag": "x"})]
    ops, recent = [], []
    for _ in range(n_ops):
        if compacts and r.random() < 0.15:
            ops.append(("compact",))
            continue
        shape = r.choice(recent) if recent and r.random() < 0.4 else r.choice(shapes)
        recent = (recent + [shape])[-3:]
        ops.append(("call",) + shape)
    return ops


def play_ops(t, ops, check_each=True):
    for op in ops:
        if op[0] == "compact":
            t.compact()
        else:
            t.call(op[1], op[2])
        if check_each:
            t.check()


@case("Part 2", title="任意两次调用之间崩溃（含 compact）")
def part2_crash_between_calls(sol):
    "Random calls and compacts; after every one, a crash must recover exactly the cache at that moment."
    r = rng("between")
    with Disk(sol) as disk:
        for capacity in (1, 2, 3):
            t = Trial(disk, "DurableMemoize", capacity)
            play_ops(t, random_ops(r, 24, 5, compacts=True))


def torn_checks(disk, cls_name, r, n_calls=10, per_record=6, follow=2):
    """Calls on a fresh store; then a crash in the middle of each call's record (the record cut short anywhere
    inside), alone or followed by the records of later calls. Either way the cut call and all after it are lost.
    A few of the recovered stores take more calls and crash again: those calls must be there."""
    capacity = 2
    t = Trial(disk, cls_name, capacity)
    calls = [((n,), {}) for n in [r.randrange(4) for _ in range(n_calls)]]
    snaps, zs, models, stories = [snapshot(t.d)], [sizes(t.d)], [t.model.copy()], [t.story.copy()]
    for args, kwargs in calls:
        t.call(args, kwargs)
        snaps.append(snapshot(t.d))
        zs.append(sizes(t.d))
        models.append(t.model.copy())
        stories.append(t.story.copy())
    final = snaps[-1]
    tested = 0
    for i, (args, kwargs) in enumerate(calls):
        grew = [rel for rel in zs[i + 1] if zs[i + 1][rel] != zs[i].get(rel, 0)]
        if len(grew) != 1:
            continue
        rel = grew[0]
        lo, hi = zs[i].get(rel, 0), zs[i + 1][rel]
        if hi - lo < 2 or not snaps[i + 1]["files"][rel].startswith(snaps[i]["files"].get(rel, b"")):
            continue
        inside = list(range(lo + 1, hi))
        cuts = sorted(set([lo + 1, hi - 1] + r.sample(inside, min(per_record, len(inside)))))
        line = call_of("w", args, kwargs)
        for n_cut, cut in enumerate(cuts):
            data = snaps[i + 1]["files"][rel][:cut]
            note = f"# —— 进程在写 {line}（第 {i + 1} 次调用）的记录时被杀死：只写了前 {cut - lo} 个字节（共 {hi - lo} 字节）——"
            snap = with_file(snaps[i], rel, data)
            check_store(disk, cls_name, capacity, snap, models[i], t.universe, stories[i].copy().add(note))
            tested += 1
            later = final["files"].get(rel, b"")
            if i + 1 < len(calls) and later.startswith(snaps[i + 1]["files"][rel]):
                note2 = (f"# —— 日志在 {line}（第 {i + 1} 次调用）的记录中间断开（只剩前 {cut - lo} 个字节），后面却接着之后"
                         f"{len(calls) - i - 1} 次调用的完整记录：断开的记录和它之后的所有内容都要丢弃 ——")
                check_store(disk, cls_name, capacity, with_file(snaps[i], rel, data + later[hi:]), models[i],
                            t.universe, stories[i].copy().add(note2))
            if n_cut < follow:
                # the recovered store keeps working: its new calls are appended after the cut and must survive
                st = stories[i].copy().add(note)
                st.add(f'w2 = {cls_name}(f, capacity={capacity}, path="cache.log")', "恢复，然后继续调用")
                f2 = Func()
                w2, d2 = disk.recover(cls_name, capacity, snap, st, f2)
                model = models[i].copy()
                for a2, k2 in (((7,), {}), ((model.entries()[0][0] if model.entries() else (8,)), {})):
                    checked_call(w2, f2, model, st, "w2", a2, k2)
                del w2
                universe = t.universe + [((7,), {}), ((8,), {})]
                check_store(disk, cls_name, capacity, snapshot(d2), model, universe, st.copy().add(CRASH))
                disk.dirs.drop(d2)
    return tested


@case("Part 2", title="记录写到一半时崩溃：截断处及之后全部丢弃")
def part2_torn_records(sol):
    "A record cut short anywhere is dropped with everything after it; the recovered store keeps appending."
    with Disk(sol) as disk:
        n = torn_checks(disk, "DurableMemoize", rng("torn"))
        info(f"检查了 {n} 个截断位置")


@case("Part 2", title="只追加日志：每次调用的写入量和 capacity 无关")
def part2_append_only(sol):
    "With capacity 40 and a full cache, each call only appends, and only about the size of that one call."
    r = rng("append")
    with Disk(sol) as disk:
        t = Trial(disk, "DurableMemoize", 40)
        for n in range(40):
            t.call((n,))
        for _ in range(40):
            t.call((r.randrange(60),))
        t.check()


def store_size(d):
    return sum(sizes(d).values())


@case("Part 2", title="compact 之后存储大小只取决于当前条目")
def part2_compact_size(sol):
    "A long history and a short one that end in the same cache take the same space after compact()."
    r = rng("size")
    with Disk(sol) as disk:
        final = [(100 + i,) for i in range(5)]
        short_t = Trial(disk, "DurableMemoize", 5)
        for a in final:
            short_t.call(a, disk_checks=False)
        short_t.compact()
        small = store_size(short_t.d)
        long_t = Trial(disk, "DurableMemoize", 5)
        for _ in range(300):
            long_t.call((r.randrange(50),), disk_checks=False)
        for a in final:
            long_t.call(a, disk_checks=False)
        before = store_size(long_t.d)
        long_t.compact()
        big = store_size(long_t.d)
        info(f"compact 后：5 次调用的存储 {small} 字节，305 次调用的存储 {big} 字节（compact 前 {before} 字节）")
        expect_true(big <= 1.2 * small + 64,
                    f"两个实例最后缓存的条目完全相同（5 个），compact() 之后一个占 {small} 字节、一个占 {big} 字节："
                    "compact() 之后存储的数据量只能取决于当前缓存的条目，不能取决于之前发生过多少次调用",
                    long_t.story.text(mark="<- compact 之后"))
        long_t.check()


def compact_with_crashes(t):
    """t.compact(), with a crash simulated at every file-system step inside it (and right after it)."""
    seen, points = set(), []

    def hook(label):
        s = snapshot(t.d)
        if snap_key(s) not in seen:
            seen.add(snap_key(s))
            points.append((label, s))

    with t.disk.fs.steps(hook):
        t.compact()
    hook("compact() 返回之后")
    for label, s in points:
        t.check(s, note=f"# —— 进程在 compact() 内部（{label}）被杀死：只保留此刻磁盘上的文件 ——")
        # the recovered store keeps working after a crash inside compact()
        st = t.crash_story(f"# —— 进程在 compact() 内部（{label}）被杀死 ——")
        st.add(f'w2 = {t.cls_name}(f, capacity={t.capacity}, path="cache.log")', "恢复，然后继续调用")
        f2 = Func()
        w2, d2 = t.disk.recover(t.cls_name, t.capacity, s, st, f2)
        model = t.model.copy()
        for a in ((FILL - 1,), model.entries()[0][0] if model.entries() else (FILL - 2,)):
            checked_call(w2, f2, model, st, "w2", a, {})
        st.add("w2.compact()")
        calling(st.text())
        w2.compact()
        checked_call(w2, f2, model, st, "w2", (FILL - 3,), {})
        del w2
        universe = t.universe + [((FILL - 1,), {}), ((FILL - 2,), {}), ((FILL - 3,), {})]
        check_store(t.disk, t.cls_name, t.capacity, snapshot(d2), model, universe, st.copy().add(CRASH))
        t.disk.dirs.drop(d2)
    return len(points)


@case("Part 2", title="compact 执行到任何一步时崩溃")
def part2_compact_crash(sol):
    "A crash at any file-system step inside compact() recovers the state from just before it, never a mix."
    with Disk(sol) as disk:
        t = Trial(disk, "DurableMemoize", 2)
        t.call((100,))
        t.call((7,))
        n = compact_with_crashes(t)
        for a in (1, 7, 2):
            t.call((a,))
        n += compact_with_crashes(t)        # [7, 2]: a second store of entries must not be replayed onto the old log
        t.call((3,))
        t.check()
        r = rng("compact")
        t = Trial(disk, "DurableMemoize", 3)
        for _ in range(3):
            play_ops(t, [op for op in random_ops(r, 8, 6, compacts=False)], check_each=False)
            n += compact_with_crashes(t)
        info(f"检查了 compact() 内部 {n} 个不同的崩溃时刻")


@case("Part 2", title="compact 之后的调用照样持久")
def part2_after_compact(sol):
    "Calls after compact() (hits and misses) are durable, also after a second compact()."
    with Disk(sol) as disk:
        t = Trial(disk, "DurableMemoize", 3)
        play_ops(t, [("call", (1,), {}), ("call", (2,), {}), ("compact",), ("call", (3,), {}), ("call", (1,), {}),
                     ("call", (4,), {}), ("compact",), ("compact",), ("call", (2,), {}), ("call", (3,), {}),
                     ("call", (5,), {})])


@case("Part 2", title="各种 JSON 参数与返回值的往返")
def part2_round_trip(sol):
    "Arguments (nested tuples, unicode, None, bools, floats, keywords) round-trip exactly; tuples in values may come back as lists."
    calls = [
        ((), {}), (((),), {}), ((1, (2, (3, "x"))), {}), (("中文 ✓", "a\"b\\c\nd"), {}), ((None, True, False), {}),
        ((-5, 2.5, 10 ** 30), {}), ((), {"key": "k", "value": (1, 2)}), (("x",), {"op": "put", "n": None}),
        ((0.1,), {"z": 1, "a": (("p", 1),)}), (("",), {}),
    ]

    def value(a, k):
        return {"args": list(a), "n": len(a), "kw": sorted(k), "t": (1, [2, (3,)]), "s": "é"}
    with Disk(sol) as disk:
        t = Trial(disk, "DurableMemoize", len(calls), func=Func(value),
                  note="f 返回一个由参数算出的 dict，里面有 tuple")
        for a, k in calls:
            t.call(a, k)
        t.call((), {"op": "put", "n": None})          # not the same call as w("x", op="put", n=None)
        t.call(("x",), {"n": None, "op": "put"})      # the same call with keywords in another order: a hit
        t.check(g_value=lambda a, k: ["g", list(a)])


@case("Part 2", title="path=None：不持久化，行为和 Memoize 一样")
def part2_no_path(sol):
    "Without a path nothing is written anywhere; the cache behaves like Memoize and compact() does nothing."
    with Disk(sol) as disk:
        d = disk.dirs.new()
        old = os.getcwd()
        os.chdir(d)
        try:
            for ctor, inst in (("DurableMemoize(f, capacity=2)", lambda f: sol.DurableMemoize(f, 2)),
                               ("DurableMemoize(f, capacity=2, path=None)", lambda f: sol.DurableMemoize(f, 2, None))):
                fn, model = Func(), Model(2)
                st = Story([(f"cache = {ctor}", F_NOTE)])
                calling(st.text())
                cache = inst(fn)
                for n in (1, 2, 1, 3, 2):
                    checked_call(cache, fn, model, st, "cache", (n,), {})
                st.add("cache.compact()")
                calling(st.text())
                cache.compact()
                for n in (1, 3, 2):
                    checked_call(cache, fn, model, st, "cache", (n,), {})
                left = os.listdir(d)
                expect_true(not left, f"path=None 时不应当写任何文件，当前目录里却出现了 {left}", st.text())
        finally:
            os.chdir(old)


# ---------------------------------------------------------------- Part 3

def finished(callers, wait):
    """Whether every thread returns within `wait` seconds in all."""
    end = time.monotonic() + wait
    for c in callers:
        c.join(max(0.0, end - time.monotonic()))
    return all(c.done.is_set() for c in callers)


def fail_from(c, what):
    """A helper thread's exception, reported as a wrong answer (with the replay last passed to calling())."""
    if c.error is not None:
        raise Mismatch(f"{what} 抛出了 {type(c.error).__name__}: {c.error}")


@case("Part 3", sample=True, cpu=3, title="题目示例：同键只算一次，异键不等待")
def part3_statement_example(sol):
    "A computes user 7 (held inside fetch_profile), B waits for it, C computes user 9 meanwhile."
    with Disk(sol) as disk:
        d = disk.dirs.new()
        fetch = HeldFunc(held={7})
        st = Story([('sf = ThreadSafeMemoize(fetch_profile, capacity=100, path="cache.log")',
                     "测试让 fetch_profile(user_id=7) 一直停住，直到最后才放行")])
        calling(st.text())
        sf = sol.ThreadSafeMemoize(fetch, 100, os.path.join(d, LOG))
        try:
            st.add("线程 A：sf(user_id=7)")
            calling(st.text())
            a = Caller(sf, kwargs={"user_id": 7}).go()
            if not fetch.entered[7].wait(2):
                fail_from(a, "sf(user_id=7)")
                raise Mismatch("线程 A 调用 sf(user_id=7) 后 2 秒内都没有调用 fetch_profile", st.text())
            st.add("线程 B：sf(user_id=7)", "A 的 fetch_profile 还没返回")
            b = Caller(sf, kwargs={"user_id": 7}).go()
            wait_parked(b, sol)
            st.add("线程 C：sf(user_id=9)", "应当立刻完成，不必等 A 或 B")
            calling(st.text())
            c = Caller(sf, kwargs={"user_id": 9}).go()
            if not c.done.wait(2):
                raise Mismatch("线程 C 调用一个不同的键 sf(user_id=9)，2 秒内都没有返回：它被正在计算 user_id=7 的线程挡住了"
                               "（func 运行时、或者等待别人的结果时，不能占着无关键需要的锁）", st.text())
            fail_from(c, "线程 C 的 sf(user_id=9)")
            expect(c.result, fetch.value(9), call=st.text(), msg="sf(user_id=9) 的返回值不对")
        finally:
            fetch.release()
        st.add("# 放行 fetch_profile(user_id=7)")
        calling(st.text())
        if not finished([a, b], 3):
            raise Mismatch("放行 fetch_profile 之后，线程 A 或 B 3 秒内都没有返回（死锁？）", st.text())
        fail_from(a, "线程 A 的 sf(user_id=7)")
        fail_from(b, "线程 B 的 sf(user_id=7)")
        expect(a.result, fetch.value(7), call=st.text(), msg="线程 A 的返回值不对")
        expect(b.result, fetch.value(7), call=st.text(), msg="线程 B 的返回值不对")
        expect(fetch.calls[7], 1, call=st.text(),
               msg="fetch_profile(user_id=7) 应当只运行一次：B 到达时 A 正在计算同一个键，B 应当等 A 的结果")
        expect(fetch.calls[9], 1, call=st.text(), msg="fetch_profile(user_id=9) 应当运行一次")


@case("Part 3", cpu=3, title="8 个线程等同一个键：func 只运行一次")
def part3_many_waiters(sol):
    "Eight threads arrive while one key is being computed; func runs once and all get its value."
    with Disk(sol) as disk:
        d = disk.dirs.new()
        fetch = HeldFunc(held={5})
        sf = sol.ThreadSafeMemoize(fetch, 4, os.path.join(d, LOG))
        st = Story([('sf = ThreadSafeMemoize(fetch, capacity=4, path="cache.log")', "fetch(5) 被测试停住"),
                    ("线程 0：sf(5)", ""), ("线程 1..8：sf(5)", "都在 fetch(5) 返回之前到达")])
        calling(st.text())
        try:
            a = Caller(sf, (5,)).go()
            if not fetch.entered[5].wait(2):
                fail_from(a, "sf(5)")
                raise Mismatch("sf(5) 2 秒内都没有调用 func", st.text())
            waiters = [Caller(sf, (5,)).go() for _ in range(8)]
            wait_parked(waiters, sol)
        finally:
            fetch.release()
        st.add("# 放行 fetch(5)")
        calling(st.text())
        if not finished([a] + waiters, 3):
            raise Mismatch("放行 fetch(5) 之后，有线程 3 秒内都没有返回（死锁？）", st.text())
        for i, c in enumerate([a] + waiters):
            fail_from(c, f"线程 {i} 的 sf(5)")
            expect(c.result, fetch.value(5), call=st.text(), msg=f"线程 {i} 的返回值不对")
        expect(fetch.calls[5], 1, call=st.text(), msg="fetch(5) 应当只运行一次：其余线程到达时它正在计算，应当等它的结果")


@case("Part 3", cpu=3, title="有线程在等结果时，无关的键照常命中和计算")
def part3_unrelated_keys(sol):
    "While key 7 is computed and another thread waits for it, a hit on key 5 and a miss on key 9 both finish."
    with Disk(sol) as disk:
        d = disk.dirs.new()
        fetch = HeldFunc(held={7})
        sf = sol.ThreadSafeMemoize(fetch, 4, os.path.join(d, LOG))
        st = Story([('sf = ThreadSafeMemoize(fetch, capacity=4, path="cache.log")', "fetch(7) 被测试停住"),
                    ("sf(5)", "先缓存 5")])
        calling(st.text())
        expect(sf(5), fetch.value(5), call=st.text())
        try:
            st.add("线程 A：sf(7)").add("线程 B：sf(7)", "等 A 的结果")
            a = Caller(sf, (7,)).go()
            if not fetch.entered[7].wait(2):
                fail_from(a, "sf(7)")
                raise Mismatch("sf(7) 2 秒内都没有调用 func", st.text())
            b = Caller(sf, (7,)).go()
            wait_parked(b, sol)
            for key, what in ((5, "命中"), (9, "未命中")):
                st.add(f"线程 C：sf({key})", f"{what}，应当立刻完成")
                calling(st.text())
                c = Caller(sf, (key,)).go()
                if not c.done.wait(2):
                    raise Mismatch(f"sf({key})（{what}，和 7 无关）2 秒内都没有返回：它被计算或等待键 7 的线程挡住了", st.text())
                fail_from(c, f"sf({key})")
                expect(c.result, fetch.value(key), call=st.text(), msg=f"sf({key}) 的返回值不对")
        finally:
            fetch.release()
        if not finished([a, b], 3):
            raise Mismatch("放行 fetch(7) 之后，线程 A 或 B 3 秒内都没有返回（死锁？）", st.text())
        fail_from(a, "线程 A 的 sf(7)")
        fail_from(b, "线程 B 的 sf(7)")
        expect(fetch.calls[5], 1, call=st.text(), msg="fetch(5) 应当只运行一次（之后都是命中）")


@case("Part 3", cpu=3, title="func 抛异常：传给所有等待者，不缓存、不持久化")
def part3_exception(sol):
    "The leader's exception reaches every thread waiting for that call and no one else; nothing is cached or stored."
    with Disk(sol) as disk:
        d = disk.dirs.new()
        fetch = HeldFunc(held={7}, fail_first={7})
        sf = sol.ThreadSafeMemoize(fetch, 4, os.path.join(d, LOG))
        st = Story([('sf = ThreadSafeMemoize(fetch, capacity=4, path="cache.log")',
                     "fetch(7) 第一次调用会抛 RuntimeError('boom-7')，测试先让它停住")])
        try:
            st.add("线程 A：sf(7)").add("线程 B, C：sf(7)", "在 A 的 fetch(7) 返回之前到达").add("线程 D：sf(9)", "无关的键")
            calling(st.text())
            a = Caller(sf, (7,)).go()
            if not fetch.entered[7].wait(2):
                fail_from(a, "sf(7)")
                raise Mismatch("sf(7) 2 秒内都没有调用 func", st.text())
            waiters = [Caller(sf, (7,)).go() for _ in range(2)]
            wait_parked(waiters, sol)
            other = Caller(sf, (9,)).go()
            if not other.done.wait(2):
                raise Mismatch("sf(9)（和 7 无关）2 秒内都没有返回", st.text())
        finally:
            fetch.release()
        st.add("# 放行：fetch(7) 抛出 RuntimeError('boom-7')")
        calling(st.text())
        if not finished([a] + waiters, 3):
            raise Mismatch("fetch(7) 抛出异常之后，有调用 sf(7) 的线程 3 秒内都没有返回（等待者没有被唤醒？）", st.text())
        fail_from(other, "无关键的 sf(9)")
        expect(other.result, fetch.value(9), call=st.text(), msg="sf(9) 的返回值不对")
        expect_true(isinstance(a.error, RuntimeError) and str(a.error) == "boom-7",
                    f"线程 A（领头者）应当抛出 fetch 的异常 RuntimeError('boom-7')，实际"
                    + (f"抛出了 {type(a.error).__name__}: {a.error}" if a.error else f"返回了 {short(a.result, 100)}"), st.text())
        leaders = {ident for key, ident in fetch.callers if key == 7}
        for i, c in enumerate(waiters):
            if c.ident in leaders:
                # it came after A's call had failed and computed the key itself: then it must have the value
                fail_from(c, f"线程 {'BC'[i]} 的 sf(7)（A 失败之后自己重新计算）")
                expect(c.result, fetch.value(7), call=st.text(), msg=f"线程 {'BC'[i]} 的返回值不对")
                continue
            got = f"抛出了 {type(c.error).__name__}: {c.error}" if c.error else f"返回了 {short(c.result, 100)}"
            expect_true(isinstance(c.error, RuntimeError) and str(c.error) == "boom-7",
                        f"线程 {'BC'[i]} 在等 A 那一次 fetch(7) 的结果，那次调用抛出了 RuntimeError('boom-7')，"
                        f"它也应当抛出这个异常；实际{got}", st.text())
        n = fetch.calls[7]
        snap = snapshot(d)
        st.add("sf(7)", "之后的新调用：上次失败没有缓存任何东西，应当重新调用 fetch 并返回它的值")
        calling(st.text())
        expect(sf(7), fetch.value(7), call=st.text(), msg="sf(7) 的返回值不对（失败的那次调用不能被缓存）")
        expect(fetch.calls[7], n + 1, call=st.text(), msg="失败的调用不能被缓存：这次 sf(7) 应当重新调用 fetch(7)")
        st.entries.pop()
        st.add("# —— 在 sf(7) 失败之后、下一次 sf(7) 之前模拟崩溃 ——")
        model = Model(4)
        model.call((9,), {}, lambda a, k: fetch.value(a[0]))
        check_store(disk, "ThreadSafeMemoize", 4, snap, model, [((7,), {}), ((9,), {})], st,
                    lambda a, k: ["g", list(a)])


@case("Part 3", cpu=3, title="等待者在记录持久化之后才返回")
def part3_waiter_after_durable(sol):
    "A thread that waited for another's computation returns only once that entry is durable: at that moment the disk holds its record, fsynced."
    with Disk(sol) as disk:
        d = disk.dirs.new()
        fetch = HeldFunc(held={7})
        sf = sol.ThreadSafeMemoize(fetch, 4, os.path.join(d, LOG))
        st = Story([('sf = ThreadSafeMemoize(fetch, capacity=4, path="cache.log")', "fetch(7) 被测试停住"),
                    ("线程 A：sf(7)", ""), ("线程 B：sf(7)", "等 A 的结果"),
                    ("# 放行 fetch(7)；放行后的第一次 fsync 被测试停住一会儿", "")])
        calling(st.text())
        paused, resume = threading.Event(), threading.Event()
        state = {"armed": False}

        def at_fsync(fd):
            if state["armed"]:
                state["armed"] = False
                paused.set()
                resume.wait(3)

        def at_return(_result):
            # runs in B's thread the moment its sf(7) returns: the files as they are then, and which hold unsynced bytes.
            # B may wait for A's fsync or fsync the log itself; either way the entry must be on disk and fsynced by now.
            snap = snapshot(d)
            return snap, disk.fs.unsynced(d, sorted(snap["files"]))
        try:
            a = Caller(sf, (7,)).go()
            if not fetch.entered[7].wait(2):
                fail_from(a, "sf(7)")
                raise Mismatch("sf(7) 2 秒内都没有调用 func", st.text())
            b = Caller(sf, (7,), after=at_return).go()
            wait_parked(b, sol)
            state["armed"] = True
            disk.fs.fsync_hook = at_fsync
            fetch.release()
            if paused.wait(2):
                wait_parked(b, sol)         # a waiter that does not wait for durability returns now, mid-fsync
            else:
                info("A 放行后 2 秒内没有调用 fsync")
        finally:
            fetch.release()
            resume.set()
            disk.fs.fsync_hook = None
        if not finished([a, b], 3):
            raise Mismatch("线程 A 或 B 3 秒内都没有返回（死锁？）", st.text())
        fail_from(a, "线程 A 的 sf(7)")
        fail_from(b, "线程 B 的 sf(7)")
        expect(a.result, fetch.value(7), call=st.text())
        expect(b.result, fetch.value(7), call=st.text())
        snap, bad = b.extra
        lost = "等待者要在结果缓存好、记录持久化（写进日志并 fsync，等领头者的 fsync 或自己 fsync 都可以）之后才能返回，否则 B 返回后崩溃，这次调用就丢了"
        if bad:
            raise Mismatch(f"线程 B 的 sf(7) 返回时，{bad[0]} 里还有没有 fsync 过的内容：{lost}", st.text())
        st.add("# —— 线程 B 的 sf(7) 刚返回时模拟崩溃：只保留那一刻磁盘上的文件 ——")
        st.add('r = ThreadSafeMemoize(g, capacity=4, path="cache.log")', "g 和 fetch 不同")
        g = Func(lambda a_, k_: ["g", list(a_)])
        r, _d2 = disk.recover("ThreadSafeMemoize", 4, snap, st, g)
        probe(r, g, st, (7,), {}, True, fetch.value(7),
              why=f"线程 B 的 sf(7) 返回那一刻的磁盘上没有 A 算出的这一项（恢复后 r(7) 未命中）：{lost}")


def twin_probe(live, live_count, rec, rec_count, keys, st):
    """Calls live(k) and rec(k) for each k in turn: both must hit or both miss, and a hit in rec on a value it
    recovered from the disk must equal the live one."""
    computed = set()        # keys whose value in rec came from its own func, not from the disk
    for k in keys:
        n_live, n_rec = live_count(k), rec_count()
        got_live, got_rec = live(k), rec(k)
        hit_live, hit_rec = live_count(k) == n_live, rec_count() == n_rec
        st.add(f"sf({k}); r({k})", f"sf {'命中' if hit_live else '未命中'}，r {'命中' if hit_rec else '未命中'}")
        if hit_live != hit_rec:
            raise Mismatch("所有调用都已返回之后，从磁盘恢复出的缓存和内存里的缓存不一致（同样的调用一个命中、一个未命中）："
                           "日志里记录的顺序必须和缓存实际更新的顺序一致", st.text(mark=""))
        if not hit_rec:
            computed.add(k)
        elif k not in computed and not same_value(got_rec, got_live):
            raise Mismatch(f"恢复后 r({k}) 命中了，值却和内存里的不同", st.text(mark=""), got_live, got_rec)


class PausingDict(dict):
    """A func value. json.dumps (and pickle) read a dict subclass through .items(), so the first time a thread
    other than the test's main thread serialises it, that thread pauses until the test resumes it (at most 3 s)."""

    def __init__(self, data, gate):
        super().__init__(data)
        self.gate = gate

    def items(self):
        gate = self.gate
        if gate is not None and threading.current_thread() is not threading.main_thread() and not gate["paused"].is_set():
            gate["paused"].set()
            gate["resume"].wait(3)
        return super().items()

    def __reduce__(self):           # pickle: a plain dict (the gate is not part of the value), read through items()
        return dict, (dict(self.items()),)


class PausingTuple(tuple):
    """An argument. json.dumps (and pickle) read a tuple subclass by iterating it, but hashing and comparing it do
    not; so once armed, the first time a thread other than the main thread serialises a key holding it, that thread
    pauses until the test resumes it (at most 3 s). Equal to, and hashed like, the plain tuple."""

    def __new__(cls, items, gate):
        t = super().__new__(cls, items)
        t.gate = gate
        return t

    def __iter__(self):
        gate = self.gate
        if gate["armed"] and threading.current_thread() is not threading.main_thread() and not gate["paused"].is_set():
            gate["paused"].set()
            gate["resume"].wait(3)
        return super().__iter__()

    def __reduce__(self):           # pickle: a plain tuple (the gate is not part of the key), read by iterating
        return tuple, (tuple(self),)


@case("Part 3", cpu=3, title="日志顺序必须和缓存更新顺序一致")
def part3_log_order(sol):
    "A thread is paused while it serialises its new record (a miss's or a hit's); another thread's call meanwhile must end up in the log in the same order as in the cache."
    for other, what in ((2, "命中 2"), (4, "未命中 4")):
        with Disk(sol) as disk:
            d = disk.dirs.new()
            gate = {"paused": threading.Event(), "resume": threading.Event()}
            func = HeldFunc(value=lambda k: PausingDict({"key": k, "v": f"value-{k}"}, gate if k == 3 else None))
            st = Story([('sf = ThreadSafeMemoize(f, capacity=2, path="cache.log")', ""), ("sf(1); sf(2)", "缓存 [1, 2]")])
            calling(st.text())
            sf = sol.ThreadSafeMemoize(func, 2, os.path.join(d, LOG))
            sf(1)
            sf(2)
            st.add("线程 X：sf(3)", "未命中；测试让 X 在把 f(3) 的返回值序列化（写日志记录）时停住")
            st.add(f"线程 Y：sf({other})", f"{what}，在 X 停住期间到达")
            calling(st.text())
            try:
                x = Caller(sf, (3,)).go()
                paused = gate["paused"].wait(2)
                y = Caller(sf, (other,)).go()
                if paused:
                    wait_parked(y, sol)
                else:
                    info("sf(3) 没有在线程里序列化返回值（没有调用它的 items()），这一轮没有构造出交错")
            finally:
                gate["resume"].set()
            if not finished([x, y], 3):
                raise Mismatch("线程 X 或 Y 3 秒内都没有返回（死锁？）", st.text())
            fail_from(x, "线程 X 的 sf(3)")
            fail_from(y, f"线程 Y 的 sf({other})")
            expect(x.result, func.value(3), call=st.text(), msg="sf(3) 的返回值不对")
            expect(y.result, func.value(other), call=st.text(), msg=f"sf({other}) 的返回值不对")
            snap = snapshot(d)
            g = Func(lambda a, k: ["g", list(a)])
            st.add('# 两个线程都返回后模拟崩溃并恢复：r = ThreadSafeMemoize(g, capacity=2, path="cache.log")；'
                   "然后对 sf 和 r 按同样的顺序调用，命中与否必须一致")
            rec, d2 = disk.recover("ThreadSafeMemoize", 2, snap, st, g)
            twin_probe(sf, lambda k: func.calls[k], rec, lambda: len(g.calls), [FILL, 3, other, 1], st)
            del rec

    # a hit's record too: X hits (1, 2), the least recently used entry, and pauses while serialising that key
    with Disk(sol) as disk:
        d = disk.dirs.new()
        gate = {"armed": False, "paused": threading.Event(), "resume": threading.Event()}
        func = HeldFunc(value=lambda k: f"value-{k}")
        key = PausingTuple((1, 2), gate)
        st = Story([('sf = ThreadSafeMemoize(f, capacity=2, path="cache.log")', ""),
                    ("sf((1, 2)); sf(2)", "缓存 [(1, 2), 2]，(1, 2) 是最久未用的")])
        calling(st.text())
        sf = sol.ThreadSafeMemoize(func, 2, os.path.join(d, LOG))
        sf(key)
        sf(2)
        st.add("线程 X：sf((1, 2))", "命中；测试让 X 在把键 (1, 2) 序列化（写日志记录）时停住")
        st.add("线程 Y：sf(4)", "未命中，在 X 停住期间到达")
        calling(st.text())
        gate["armed"] = True
        try:
            x = Caller(sf, (key,)).go()
            paused = gate["paused"].wait(2)
            y = Caller(sf, (4,)).go()
            if paused:
                wait_parked(y, sol)
            else:
                info("命中的 sf((1, 2)) 没有在线程里序列化这个键（没有遍历它），这一轮没有构造出交错")
        finally:
            gate["resume"].set()
        if not finished([x, y], 3):
            raise Mismatch("线程 X 或 Y 3 秒内都没有返回（死锁？）", st.text())
        fail_from(x, "线程 X 的 sf((1, 2))")
        fail_from(y, "线程 Y 的 sf(4)")
        expect(x.result, func.value((1, 2)), call=st.text(), msg="sf((1, 2)) 的返回值不对")
        expect(y.result, func.value(4), call=st.text(), msg="sf(4) 的返回值不对")
        snap = snapshot(d)
        g = Func(lambda a, k: ["g", list(a)])
        st.add('# 两个线程都返回后模拟崩溃并恢复：r = ThreadSafeMemoize(g, capacity=2, path="cache.log")；'
               "然后对 sf 和 r 按同样的顺序调用，命中与否必须一致")
        rec, d2 = disk.recover("ThreadSafeMemoize", 2, snap, st, g)
        twin_probe(sf, lambda k: func.calls[k], rec, lambda: len(g.calls), [(1, 2), 4, 2, FILL], st)
        del rec


@case("Part 3", cpu=4, title="compact 执行中途有其他线程调用")
def part3_compact_concurrent(sol):
    "compact() is paused at each of its file-system steps while another thread makes a hit or a miss."
    n_points = 0
    with Disk(sol) as disk:
        for op in ("miss", "hit"):
            for step in range(1, 13):
                f = Func()
                t = Trial(disk, "ThreadSafeMemoize", 3, func=f, head='sf = ThreadSafeMemoize(f, capacity=3, path="cache.log")')
                for n in (1, 2, 3):
                    t.call((n,), disk_checks=False)
                key = 4 if op == "miss" else 1
                reached, go_on, labels = threading.Event(), threading.Event(), []

                def hook(label, step=step, labels=labels, reached=reached, go_on=go_on):
                    labels.append(label)
                    if len(labels) == step:
                        reached.set()
                        go_on.wait(3)

                def run_compact(t=t, hook=hook):
                    with disk.fs.steps(hook):
                        t.w.compact()
                comp = Caller(run_compact).go()
                done = threading.Event()
                while not reached.wait(0.01):
                    if comp.done.is_set():
                        done.set()
                        break
                if done.is_set():
                    fail_from(comp, "sf.compact()")
                    break                     # compact() has fewer steps than this
                n_points += 1
                st = t.story.copy().add(f"线程 T：sf.compact()", f"测试让它停在第 {step} 个文件操作（{labels[-1]}）")
                st.add(f"线程 D：sf({key})", "compact 暂停期间调用（" + ("未命中" if op == "miss" else "命中") + "）")
                calling(st.text())
                c = Caller(t.w, (key,)).go()
                wait_parked(c, sol, longest=1.0)
                go_on.set()
                if not finished([comp, c], 3):
                    raise Mismatch("compact() 和另一个线程的调用 3 秒内没有都结束（死锁？）", st.text())
                fail_from(comp, f"compact()（暂停期间另一个线程调用了 sf({key})）")
                fail_from(c, f"sf({key})（在 compact() 执行到一半时调用）")
                hit, want = t.model.call((key,), {}, f.value)
                expect(c.result, want, call=st.text(), msg=f"sf({key}) 的返回值不对")
                if all(canon(a, k) != canon((key,), {}) for a, k in t.universe):
                    t.universe.append(((key,), {}))
                t.story = st
                t.check(note="# —— 两个线程都结束后模拟崩溃 ——")
                disk.dirs.drop(t.d)
        info(f"在 compact() 的 {n_points} 个暂停点各插入了一次并发调用")


@case("Part 3", title="单线程下的持久化约定不变")
def part3_durable_sequential(sol):
    "ThreadSafeMemoize keeps Part 2's contract: crashes between calls, inside compact() and inside a record."
    r = rng("seq3")
    with Disk(sol) as disk:
        t = Trial(disk, "ThreadSafeMemoize", 2)
        play_ops(t, random_ops(r, 16, 4, compacts=True))
        compact_with_crashes(t)
        torn_checks(disk, "ThreadSafeMemoize", r, n_calls=6, per_record=3, follow=1)


@case("Part 3", cpu=4, title="多线程混合压力：结果正确、同键不并发、恢复后与内存一致")
def part3_storm(sol):
    "8 threads × 60 calls over 10 keys with capacity 4, plus compacts; then the recovered cache must match the live one."
    r = rng("storm")
    with Disk(sol) as disk, fast_switching():
        d = disk.dirs.new()
        jit = rng("jitter")
        func = HeldFunc(jitter=lambda: jit.random() < 0.3 and threading.Event().wait(0.0005))
        cap, n_keys = 4, 10
        sf = sol.ThreadSafeMemoize(func, cap, os.path.join(d, LOG))
        plans = [[r.randrange(n_keys) for _ in range(60)] for _ in range(8)]
        results = [[] for _ in plans]

        def worker(i):
            for k in plans[i]:
                results[i].append((k, sf(k)))

        def compactor():
            for _ in range(6):
                threading.Event().wait(0.003)
                sf.compact()
        threads = [Caller(worker, (i,)) for i in range(len(plans))] + [Caller(compactor)]
        st = Story([('sf = ThreadSafeMemoize(f, capacity=4, path="cache.log")', ""),
                    ("# 8 个线程各调用 60 次 sf(k)（k 在 0..9 中随机），另一个线程穿插调用 6 次 sf.compact()", "")])
        calling(st.text())
        for c in threads:
            c.start()
        if not finished(threads, 8):
            raise Mismatch("并发调用 8 秒内没有全部结束（死锁？）", st.text())
        for i, c in enumerate(threads):
            fail_from(c, "并发调用 sf(k) 或 sf.compact()" if i < len(plans) else "sf.compact()")
        for i, res in enumerate(results):
            for k, v in res:
                expect(v, func.value(k), call=st.text(), msg=f"线程 {i} 的 sf({k}) 返回值不对")
        both = [k for k in range(n_keys) if func.max_active[k] > 1]
        expect_true(not both, f"func({both[0] if both else ''}) 曾经同时在两个线程里运行：同一个键的 func 绝不能同时运行超过一次",
                    st.text())
        # the live cache and a recovery from the disk must now behave the same
        func.jitter = None
        snap = snapshot(d)
        g = Func(lambda a, k: ["g", list(a)])
        st.add("# 全部结束后模拟崩溃并恢复：r = ThreadSafeMemoize(g, capacity=4, path=\"cache.log\")；"
               "然后对 sf 和 r 按同样的顺序调用，命中与否必须一致")
        rec, d2 = disk.recover("ThreadSafeMemoize", cap, snap, st, g)
        order = list(range(n_keys)) * 2
        r.shuffle(order)
        twin_probe(sf, lambda k: func.calls[k], rec, lambda: len(g.calls), order, st)
        del rec
        disk.dirs.drop(d2)
