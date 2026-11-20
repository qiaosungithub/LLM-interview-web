import tracemalloc
from collections import Counter

from judgelib import Mismatch, calling, case, expect, expect_true, rng, short
import ref
from ref import Sample as S


class Ev(tuple):
    """An event as a plain tuple that prints the way the statement writes it."""
    __slots__ = ()

    def __repr__(self):
        return ref.show_event(self)


def E(kind, ts, name, depth):
    return Ev((kind, ts, name, depth))


def _copy(samples):
    return [S(s.ts, list(s.stack)) for s in samples]


def _call_text(text, described):
    return text if len(text) <= 2800 else described


# ---------------------------------------------------------------- comparing events

def _as_events(got, what):
    """The answer's events as Ev tuples. Iterating the result runs the answer's own code, so only the iter() call
    is guarded: an error raised while the answer produces its events stays the answer's error."""
    try:
        it = iter(got)
    except TypeError:
        raise Mismatch(f"{what}，实际返回了 {type(got).__name__}", got=got) from None
    out = []
    for e in it:
        try:
            t = tuple(e)
        except TypeError:
            t = ()
        if len(t) != 4:
            raise Mismatch(f"每个事件应是 Event(kind, ts, name, depth)，实际得到 {short(e)}")
        out.append(Ev(t))
    return out


def _diff(got, want):
    """Where got first goes wrong, in words."""
    n = min(len(got), len(want))
    i = next((j for j in range(n) if got[j] != want[j]), n)
    try:
        cg, cw = Counter(got), Counter(want)
        missing = [e for e in want if cw[e] > cg[e]]
        extra = [e for e in got if cg[e] > cw[e]]
    except TypeError:                                    # an unhashable field: fall back to the position
        missing = extra = None
    if missing == [] and extra == []:
        return (f"事件都对，但顺序不对：第 {i + 1} 个应是 {want[i]!r}，实际是 {got[i]!r}。同一时刻要先发 end"
                f"（由内向外），再发 start（由外向内）")
    if not got:
        head = f"没有产出任何事件，第 1 个应是 {want[0]!r}"
    elif i == len(got):
        head = f"前 {len(got)} 个事件都对，但后面还应有 {want[i]!r}"
    elif i == len(want):
        head = f"前 {len(want)} 个事件都对，但多出了 {got[i]!r}"
    else:
        head = f"第 {i + 1} 个事件应是 {want[i]!r}，实际是 {got[i]!r}"
    notes = []
    if missing:
        notes.append(f"缺少 {missing[0]!r}" + (f" 等 {len(missing)} 个事件" if len(missing) > 1 else ""))
    if extra:
        notes.append(f"多出 {extra[0]!r}" + (f" 等 {len(extra)} 个事件" if len(extra) > 1 else ""))
    return head + ("（" + "；".join(notes) + "）" if notes else "")


def check_events(got, want, call):
    want = [Ev(e) for e in want]
    if got != want:
        raise Mismatch(_diff(got, want), call, want, got)


# ---------------------------------------------------------------- Part 1

def p1(sol, samples, end_time, want=None):
    want = ref.events(samples, end_time) if want is None else want
    text = f"convert_samples_to_events({ref.show_samples(samples)}, end_time={end_time!r})"
    call = calling(_call_text(text, f"convert_samples_to_events(<{len(samples)} 个样本>, end_time={end_time!r})"))
    got = _as_events(sol.convert_samples_to_events(_copy(samples), end_time), "应返回事件列表")
    check_events(got, want, call)


EXAMPLE1 = [
    S(0.0, ["main"]),
    S(1.0, ["main", "solve"]),
    S(2.0, ["main", "solve", "solve"]),
    S(2.5, ["main", "solve", "solve"]),
    S(4.0, ["main", "solve", "solve", "solve"]),
    S(6.0, ["main", "solve", "helper"]),
]
EXAMPLE1_EVENTS = [
    E("start", 0.0, "main", 0), E("start", 1.0, "solve", 1), E("start", 2.0, "solve", 2),
    E("start", 4.0, "solve", 3), E("end", 6.0, "solve", 3), E("end", 6.0, "solve", 2),
    E("start", 6.0, "helper", 2), E("end", 9.0, "helper", 2), E("end", 9.0, "solve", 1), E("end", 9.0, "main", 0),
]


@case("Part 1", sample=True, title="题目例子：递归的 solve")
def p1_example(sol):
    p1(sol, EXAMPLE1, 9.0, EXAMPLE1_EVENTS)


@case("Part 1", title="没有样本：返回 []")
def p1_empty(sol):
    p1(sol, [], 4.0, [])


@case("Part 1", title="只有一个样本；end_time 可以等于最后一个样本的时间戳")
def p1_single_sample(sol):
    p1(sol, [S(5.0, ["main", "run", "step"])], 5.0,
       [E("start", 5.0, "main", 0), E("start", 5.0, "run", 1), E("start", 5.0, "step", 2),
        E("end", 5.0, "step", 2), E("end", 5.0, "run", 1), E("end", 5.0, "main", 0)])
    p1(sol, [S(1.0, ["main"]), S(2.0, ["main", "io"])], 2.0,
       [E("start", 1.0, "main", 0), E("start", 2.0, "io", 1), E("end", 2.0, "io", 1), E("end", 2.0, "main", 0)])


@case("Part 1", title="最后仍在栈上的帧在 end_time 结束，而不是在最后一个样本")
def p1_end_time(sol):
    p1(sol, [S(1.0, ["a"]), S(2.0, ["a", "b"]), S(3.0, ["a", "b", "c"])], 10.0,
       [E("start", 1.0, "a", 0), E("start", 2.0, "b", 1), E("start", 3.0, "c", 2),
        E("end", 10.0, "c", 2), E("end", 10.0, "b", 1), E("end", 10.0, "a", 0)])


@case("Part 1", title="递归：同一个名字同时出现在多个深度")
def p1_recursion(sol):
    p1(sol, [S(1, ["f"]), S(2, ["f", "f"]), S(3, ["f", "f", "f"]), S(4, ["f", "f"]), S(5, ["f", "f", "f"]),
             S(6, ["f"]), S(7, ["f", "g", "f"])], 8)


@case("Part 1", title="公共前缀断开后，更深处同名的帧也是新帧")
def p1_chain_break(sol):
    p1(sol, [S(1, ["req", "validate"]), S(2, ["worker", "validate", "req"])], 3,
       [E("start", 1, "req", 0), E("start", 1, "validate", 1),
        E("end", 2, "validate", 1), E("end", 2, "req", 0),
        E("start", 2, "worker", 0), E("start", 2, "validate", 1), E("start", 2, "req", 2),
        E("end", 3, "req", 2), E("end", 3, "validate", 1), E("end", 3, "worker", 0)])
    p1(sol, [S(1, ["a", "b", "c"]), S(2, ["a", "x", "c"]), S(3, ["y", "x", "c"]), S(4, ["y", "x"])], 5)


@case("Part 1", title="相同的栈不产生事件；回到之前的栈也要重新开始")
def p1_repeats(sol):
    p1(sol, [S(t, ["m", "w"]) for t in (1, 2, 3, 4)] + [S(5, ["m"]), S(6, ["m", "w"]), S(7, ["m", "w"])], 7.5)


@case("Part 1", title="空调用栈的样本")
def p1_empty_stack(sol):
    p1(sol, [S(0.5, []), S(1.0, ["a"]), S(2.0, []), S(3.0, ["a", "b"]), S(4.0, [])], 4.0,
       [E("start", 1.0, "a", 0), E("end", 2.0, "a", 0), E("start", 3.0, "a", 0), E("start", 3.0, "b", 1),
        E("end", 4.0, "b", 1), E("end", 4.0, "a", 0)])


@case("Part 1", title="随机调用栈，和逐个样本比较的写法对照")
def p1_random(sol):
    r = rng()
    for _ in range(300):
        names = r.choice([["f", "g"], ["f", "g", "h"], ["main", "read", "parse", "emit", "log"]])
        samples = ref.walk_samples(r, r.randint(0, 14), names, noise=r.choice([0, 0.2]), hold=r.choice([1, 3]))
        end_time = (samples[-1].ts if samples else 0.0) + r.choice([0, 0.5, 3])
        p1(sol, samples, end_time)


# ---------------------------------------------------------------- Part 2

def p2(sol, samples, k, want=None):
    want = ref.debounced(samples, k) if want is None else want
    text = f"list(stream_debounced_events({ref.show_samples(samples)}, k={k}))"
    call = calling(_call_text(text, f"list(stream_debounced_events(<{len(samples)} 个样本>, k={k}))"))
    got = _as_events(sol.stream_debounced_events(_copy(samples), k), "应返回一个迭代器（用 yield 逐个产出事件）")
    check_events(got, want, call)


class Source:
    """Samples that can be read only once, like a live feed; counts how many have been handed out."""

    def __init__(self, samples):
        self.samples, self.pulled, self.iterations = samples, 0, 0

    def __iter__(self):
        self.iterations += 1
        return self._feed() if self.iterations == 1 else iter(())

    def _feed(self):
        for s in self.samples:
            self.pulled += 1
            yield S(s.ts, list(s.stack))


SVC = [S(1, ["svc"]), S(2, ["svc"]), S(3, ["svc", "handle"]), S(4, ["svc"]), S(5, ["svc", "flush"]),
       S(6, ["svc", "flush"]), S(7, ["svc"]), S(8, ["svc"]), S(9, ["svc"])]


@case("Part 2", sample=True, title="题目例子：k = 1，前缀延续")
def p2_example_extend(sol):
    p2(sol, [S(1, ["req", "validate"]), S(2, ["req", "validate", "persist"])], 1,
       [E("start", 1, "req", 0), E("start", 1, "validate", 1), E("start", 2, "persist", 2)])


@case("Part 2", sample=True, title="题目例子：k = 1，前缀断开")
def p2_example_break(sol):
    p2(sol, [S(1, ["req", "validate"]), S(2, ["worker", "validate", "req"])], 1,
       [E("start", 1, "req", 0), E("start", 1, "validate", 1),
        E("end", 2, "validate", 1), E("end", 2, "req", 0),
        E("start", 2, "worker", 0), E("start", 2, "validate", 1), E("start", 2, "req", 2)])


@case("Part 2", sample=True, title="题目例子：k = 2")
def p2_example_k2(sol):
    p2(sol, SVC, 2, [E("start", 2, "svc", 0), E("start", 6, "flush", 1), E("end", 8, "flush", 1)])


@case("Part 2", title="已确认的帧离开后，位置马上被另一帧占上，它的 end 照样要发")
def p2_reoccupied(sol):
    p2(sol, [S(1, ["a"]), S(2, ["a"]), S(3, ["a", "f"]), S(4, ["a", "f"]), S(5, ["a", "g"]), S(6, ["a"])], 2,
       [E("start", 2, "a", 0), E("start", 4, "f", 1), E("end", 6, "f", 1)])
    p2(sol, [S(t, ["a"]) for t in (1, 2, 3)] + [S(t, ["a", "f"]) for t in (4, 5, 6)]
       + [S(t, ["a", "g"]) for t in (7, 8, 9, 10)], 3,
       [E("start", 3, "a", 0), E("start", 6, "f", 1), E("end", 9, "f", 1), E("start", 9, "g", 1)])


@case("Part 2", title="同名的帧离开后很快回来：那是一个新帧，旧帧照样结束")
def p2_same_name_returns(sol):
    p2(sol, [S(t, ["a"]) for t in (1, 2, 3)] + [S(t, ["a", "f"]) for t in (4, 5, 6)] + [S(7, ["a"])]
       + [S(t, ["a", "f"]) for t in (8, 9, 10)], 3,
       [E("start", 3, "a", 0), E("start", 6, "f", 1), E("end", 9, "f", 1), E("start", 10, "f", 1)])
    p2(sol, [S(1, ["a"]), S(2, ["a"]), S(3, ["a", "f"]), S(4, ["a", "f"]), S(5, ["a"]), S(6, ["a", "f"]),
             S(7, ["a", "f"])], 2,
       [E("start", 2, "a", 0), E("start", 4, "f", 1), E("end", 6, "f", 1), E("start", 7, "f", 1)])


@case("Part 2", title="k = 2 时前缀断开：更深处同名的帧也要结束并重新开始")
def p2_chain_break(sol):
    p2(sol, [S(1, ["a", "b"]), S(2, ["a", "b"]), S(3, ["c", "b"]), S(4, ["c", "b"]), S(5, ["c", "b"])], 2,
       [E("start", 2, "a", 0), E("start", 2, "b", 1), E("end", 4, "b", 1), E("end", 4, "a", 0),
        E("start", 4, "c", 0), E("start", 4, "b", 1)])


@case("Part 2", title="没有连续 k 个样本的帧不报告；空输入")
def p2_never_confirmed(sol):
    p2(sol, [S(1, ["m"]), S(2, ["m", "x"]), S(3, ["m", "x"]), S(4, ["m"]), S(5, ["m", "y"]), S(6, ["m"]),
             S(7, ["m"])], 3, [E("start", 3, "m", 0)])
    p2(sol, [S(1, ["m"]), S(2, ["m"]), S(3, ["m"]), S(4, ["m"])], 5, [])
    p2(sol, [], 2, [])


@case("Part 2", title="输入结束时还没数满 k 个缺席样本：只有 start，没有 end")
def p2_start_only(sol):
    p2(sol, [S(1, ["m"]), S(2, ["m", "w"]), S(3, ["m", "w"]), S(4, ["m", "w"]), S(5, ["m"]), S(6, ["m"])], 3,
       [E("start", 3, "m", 0), E("start", 4, "w", 1)])
    p2(sol, [S(1, ["m", "w"]), S(2, ["m", "w"]), S(3, [])], 2, [E("start", 2, "m", 0), E("start", 2, "w", 1)])


@case("Part 2", title="k = 1：和 Part 1 一样，只是没有 end_time 处的关闭")
def p2_k1(sol):
    r = rng()
    for _ in range(150):
        samples = ref.walk_samples(r, r.randint(1, 14), r.choice([["f", "g"], ["f", "g", "h"]]), noise=0.2)
        p2(sol, samples, 1)


@case("Part 2", title="随机样本（带噪声），k 取 1 到 7")
def p2_random(sol):
    r = rng()
    for _ in range(300):
        names = r.choice([["f", "g"], ["f", "g", "h"], ["main", "read", "parse", "emit"]])
        samples = ref.walk_samples(r, r.randint(0, 24), names, noise=r.choice([0.1, 0.25]), hold=r.choice([2, 4, 6]))
        p2(sol, samples, r.choice([1, 2, 2, 3, 3, 4, 7]))


@case("Part 2", title="samples 是只能读一遍的迭代器")
def p2_one_pass(sol):
    r = rng()
    for _ in range(40):
        samples = ref.walk_samples(r, r.randint(1, 20), ["f", "g", "h"], noise=0.2, hold=4)
        k = r.choice([1, 2, 3])
        src = Source(samples)
        call = calling(_call_text(f"list(stream_debounced_events(iter({ref.show_samples(samples)}), k={k}))",
                                  f"list(stream_debounced_events(iter(<{len(samples)} 个样本>), k={k}))"))
        got = _as_events(sol.stream_debounced_events(src, k), "应返回一个迭代器（用 yield 逐个产出事件）")
        expect_true(src.iterations <= 1, f"samples 被遍历了 {src.iterations} 次：它可能是只能读一遍的迭代器"
                    f"（例如生成器），第二遍什么也读不到", call=call)
        check_events(got, ref.debounced(samples, k), call)


@case("Part 2", title="事件一确定就 yield，不等读完全部样本")
def p2_lazy(sol):
    r = rng()
    samples = ref.walk_samples(r, 60, ["f", "g", "h"], noise=0.15, hold=4)
    samples = SVC[:6] + [S(s.ts + 6, s.stack) for s in samples]
    k = 2
    plan = ref.event_index(samples, k)
    call = calling(f"it = stream_debounced_events(iter({ref.show_samples(samples)}), k={k})\nnext(it)")
    src = Source(samples)
    res = sol.stream_debounced_events(src, k)
    try:
        it = iter(res)
    except TypeError:
        raise Mismatch(f"应返回一个迭代器（用 yield 逐个产出事件），实际返回了 {type(res).__name__}") from None
    got = []
    for number, (index, want) in enumerate(plan, start=1):
        e = next(it, None)
        if e is None:
            break
        e = _as_events([e], "")[0]
        if e != Ev(want):
            got.append(e)
            break
        expect_true(src.pulled <= index + 2,
                    f"第 {number} 个事件 {e!r} 读到第 {index + 1} 个样本时就已经确定，但拿到它时已经读取了 "
                    f"{src.pulled} 个样本（共 {len(samples)} 个）：事件应在确定后立刻 yield，而不是先读完输入",
                    call=call)
        got.append(e)
    got += _as_events(it, "")
    check_events(got, [e for _, e in plan], call)


MEMORY_NAMES = ["main", "read", "parse", "emit", "flush"]
MEMORY_TEMPLATES = [s.stack for s in ref.walk_samples(rng("memory"), 240, MEMORY_NAMES, noise=0.15, hold=4)]


def _memory_feed(n):
    for i in range(n):
        yield S(float(i), list(MEMORY_TEMPLATES[i % len(MEMORY_TEMPLATES)]))


MEMORY_COUNTS = {n: len(ref.debounced(list(_memory_feed(n)), 3)) for n in (6000, 18000)}


def _peak(sol, n, k):
    call = calling(f"stream_debounced_events(<{n} 个样本的生成器>, k={k})")
    tracemalloc.start()
    try:
        count = sum(1 for _ in sol.stream_debounced_events(_memory_feed(n), k))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    expect(count, MEMORY_COUNTS[n], call=call, msg=f"{n} 个样本的长输入，事件个数不对")
    return peak


@case("Part 2", cpu=4, title="长输入：保留的状态不随样本数增长")
def p2_memory(sol):
    small, large = _peak(sol, 6000, 3), _peak(sol, 18000, 3)
    expect_true(large - small < 256 * 1024,
                f"读 6000 个样本时额外内存峰值约 {small / 1024:.0f} KB，读 18000 个样本时约 {large / 1024:.0f} KB："
                f"保留的状态随样本数增长了。题目要求状态不随已处理的样本数增长，只能保留和最大栈深度同阶的信息",
                call="stream_debounced_events(<6000 / 18000 个样本的生成器>, k=3)")


# ---------------------------------------------------------------- Part 3

TRACES = [
    ("T1", ["boot", "ingest", "parse", "oom"]),
    ("T2", ["worker", "parse", "oom"]),
    ("T3", ["boot", "cache", "oom"]),
    ("T4", ["boot", "ingest", "oom"]),
]


def p3(sol, traces, queries):
    built = f"index = TraceIndex({traces!r})"
    if len(built) > 2400:
        built = f"index = TraceIndex(<{len(traces)} 条追踪>)"
    calling(built)
    index = sol.TraceIndex([(tid, list(frames)) for tid, frames in traces])
    for query in queries:
        want = ref.match(traces, query)
        call = calling(f"{built}\nindex.match_suffix({query!r})")
        got = index.match_suffix(list(query))
        expect(got, want, call=call, msg="" if got == want else _p3_hint(traces, query, got, want))


def _p3_hint(traces, query, got, want):
    m = len(query)
    if want == "UNKNOWN":
        if got == []:
            return '没有任何追踪匹配时应返回字符串 "UNKNOWN"，而不是空列表'
        return f'没有任何追踪的最后 {m} 帧和查询逐帧相同，应返回 "UNKNOWN"' if isinstance(got, list) else ""
    if got == "UNKNOWN":
        return f"{want[0]!r} 的最后 {m} 帧和查询逐帧相同，应返回匹配的 trace_id 列表"
    if not isinstance(got, list):
        return ""
    frames = {tid: fs for tid, fs in traces}
    extra = [t for t in got if isinstance(t, str) and t in frames and t not in want]
    missing = [t for t in want if t not in got]
    if extra:
        fs = frames[extra[0]]
        if len(fs) < m:
            return f"多返回了 {extra[0]!r}：它只有 {len(fs)} 帧 {fs!r}，比查询短，不算匹配"
        return f"多返回了 {extra[0]!r}：它的最后 {m} 帧是 {fs[len(fs) - m:]!r}，和查询不同"
    if missing:
        return f"漏掉了 {missing[0]!r}：它的帧是 {frames[missing[0]]!r}，最后 {m} 帧和查询逐帧相同"
    if len(got) == len(want):                            # the same ids in another order
        return "trace_id 应按字典序升序排列"
    return ""


@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    p3(sol, TRACES, [["parse", "oom"], ["oom"], ["ingest", "oom"], ["boot", "ingest", "parse", "oom"],
                     ["cache", "parse", "oom"]])


@case("Part 3", title='没有匹配时返回 "UNKNOWN"：开头几帧、未知的帧、比所有追踪都长的查询')
def p3_unknown(sol):
    p3(sol, TRACES, [["boot"], ["boot", "ingest"], ["panic"], ["parse"], ["x", "boot", "ingest", "parse", "oom"]])
    p3(sol, [], [["oom"]])


@case("Part 3", title="trace_id 按字典序排列，与加入的顺序无关")
def p3_sorted_ids(sol):
    p3(sol, [("T9", ["main", "oom"]), ("T10", ["svc", "oom"]), ("T2", ["oom"]), ("T100", ["main", "oom"]),
             ("T1", ["x", "main", "oom"])],
       [["oom"], ["main", "oom"], ["x", "main", "oom"]])


@case("Part 3", title="按整帧比较：名字的一部分相同不算")
def p3_whole_frames(sol):
    p3(sol, [("A", ["main", "preparse", "oom"]), ("B", ["main", "parse", "oom"]), ("C", ["io", "boom"]),
             ("D", ["main", "parse", "oom2"])],
       [["parse", "oom"], ["oom"], ["main", "parse", "oom"], ["boom"]])


@case("Part 3", title="比查询短的追踪不算匹配")
def p3_longer_query(sol):
    p3(sol, [("A", ["parse", "oom"]), ("B", ["main", "parse", "oom"]), ("C", ["oom"])],
       [["main", "parse", "oom"], ["x", "main", "parse", "oom"], ["parse", "oom"], ["oom"]])


@case("Part 3", title="重复的帧列表、递归的帧名")
def p3_duplicates(sol):
    p3(sol, [("r2", ["a", "b", "a", "b"]), ("r1", ["a", "b", "a", "b"]), ("s", ["b", "a", "b"]), ("t", ["a", "b"])],
       [["b"], ["a", "b"], ["b", "a", "b"], ["a", "b", "a", "b"], ["b", "a", "b", "a", "b"], ["a"]])


@case("Part 3", title='帧名可以是任意字符串，包括 "$" 这类常被拿来当内部标记的名字')
def p3_any_frame_name(sol):
    p3(sol, [("T1", ["main", "$", "oom"]), ("T2", ["$", "oom"]), ("T3", ["main", "$"]), ("T4", ["$", "$"])],
       [["$", "oom"], ["oom"], ["main", "$", "oom"], ["$"], ["main", "$"], ["$", "$"], ["x", "$", "$"]])


@case("Part 3", title="随机追踪和查询，和直接切片比较的写法对照")
def p3_random(sol):
    r = rng()
    for _ in range(200):
        names = r.choice([["a", "b"], ["a", "b", "c"], ["main", "read", "parse", "oom", "retry"]])
        traces = ref.random_traces(r, r.randint(0, 10), names)
        p3(sol, traces, [ref.random_query(r, traces, names) for _ in range(8)])
