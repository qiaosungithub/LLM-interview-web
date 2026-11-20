from collections import Counter

from judgelib import calling, case, expect, expect_raises, expect_valid, rng, show_call
import ref

SHOW = 8        # a replay spells out at most this many recent calls; earlier adds are folded into one loop


def _s(x):
    """A literal as the statement writes it: strings in double quotes."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    return repr(x)


def _pair(p):
    return f"({_s(p[0])}, {p[1]!r})"


def _norm_pair(got):
    """The statement says tuple; a 2-element list carries the same answer."""
    if isinstance(got, (tuple, list)) and len(got) == 2:
        return tuple(got)
    return got


# ---------------------------------------------------------------- Part 1 helpers

EVENTS = [("login", 1), ("click", 2), ("login", 4), ("click", 7), ("purchase", 9)]


def _norm_p1(got):
    if isinstance(got, (tuple, list)) and len(got) == 2 and isinstance(got[1], (list, tuple)):
        return got[0], list(got[1])
    return got


def check_p1(sol, events, want=None, call=None):
    want = ref.most_frequent(events) if want is None else want
    call = call or show_call("most_frequent", events)
    got = _norm_p1(sol.most_frequent(list(events)))
    msg = ""
    if isinstance(got, tuple) and len(got) == 2 and got != want:
        counts = Counter(m for m, _ in events)
        if got[0] == want[0]:
            msg = "消息对了，时间戳不对：要按在 events 中出现的顺序列出它每一次出现的时间戳（重复的也要保留）"
        elif counts.get(got[0], 0) == counts[want[0]]:
            msg = f"{_s(got[0])} 和 {_s(want[0])} 出现次数并列最高，要返回在 events 里第一次出现得最早的那一个"
        else:
            msg = f"{_s(got[0])} 出现了 {counts.get(got[0], 0)} 次，不是最多的（最多是 {counts[want[0]]} 次）"
    expect(got, want, call=call, msg=msg)


# ---------------------------------------------------------------- Part 2 helpers

def _window_replay(n, events, wants, i):
    """Code replaying adds 0..i on a fresh WindowTopMessage(n), to paste into 自定义运行; add i is marked."""
    start = max(0, i + 1 - SHOW)
    lines = [f"w = WindowTopMessage({n})"]
    if start:
        pairs = ", ".join(f"({_s(m)}, {t})" for m, t in events[:start])
        lines += [f"for m, t in [{pairs}]:   # 前 {start} 次 add", "    w.add(m, t)"]
    for j in range(start, i):
        m, t = events[j]
        lines.append(f"w.add({_s(m)}, {t})".ljust(28) + f"# -> {_pair(wants[j])}")
    m, t = events[i]
    lines.append(f"print(w.add({_s(m)}, {t}))".ljust(28) + "# <- 这一步")
    return "\n".join(lines)


def _window_msg(got, want, held):
    """Says what is wrong with one add's return value; held = the events held after that add."""
    if not (isinstance(got, tuple) and len(got) == 2):
        return ""
    counts = Counter(m for m, _ in held)
    if got[1] != want[1]:
        return f"最高计数应为 {want[1]}（当前持有的 {len(held)} 个事件里），得到 {got[1]!r}"
    if counts.get(got[0], 0) != want[1]:
        return f"计数 {want[1]} 对了，但 {_s(got[0])} 当前的计数是 {counts.get(got[0], 0)}，不是最高计数"
    return (f"{_s(got[0])} 和 {_s(want[0])} 的计数并列最高，要返回最近一次计数变化编号最大的那条"
            "（被淘汰时的减一也算一次变化，编号排在同一次到达的加一之前）")


def run_window(sol, n, events, wants=None):
    """Feeds the events to a fresh WindowTopMessage(n) and checks every return value (default: the brute force)."""
    wants = ref.window_brute(n, events) if wants is None else wants
    calling(f"WindowTopMessage({n})")
    w = sol.WindowTopMessage(n)
    for i, (m, t) in enumerate(events):
        call = calling(_window_replay(n, events, wants, i))
        got = _norm_pair(w.add(m, t))
        if got != wants[i]:
            expect(got, wants[i], call=call, msg=_window_msg(got, wants[i], events[max(0, i + 1 - n):i + 1]))


# ---------------------------------------------------------------- Part 3 helpers

def _ss_replay(k, adds, last):
    """Code replaying `adds` on a fresh SpaceSaving(k), then `last`, the call being made now (marked)."""
    start = max(0, len(adds) - SHOW)
    lines = [f"ss = SpaceSaving({k})"]
    if start:
        lines += [f"for m in [{', '.join(_s(m) for m in adds[:start])}]:   # 前 {start} 次 add", "    ss.add(m)"]
    lines += [f"ss.add({_s(m)})" for m in adds[start:]]
    lines.append(last.ljust(24) + "# <- 这一步")
    return "\n".join(lines)


def _norm_top(got):
    """The statement says a list of tuples; any sequence of 2-element sequences carries the same answer."""
    if isinstance(got, (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in got):
        return [tuple(p) for p in got]
    return got


def _ss_problem(got, want, counts, n_seen, k, everything):
    """None when top(...) returned `want`; otherwise what is wrong, the statement's guarantees checked first.
    everything: the call asked for more than k messages, so it must return every monitored message."""
    if not (isinstance(got, list) and all(isinstance(p, tuple) for p in got)):
        return "top 应当返回由 (message, count) 二元组组成的列表"
    names = [x for x, _ in got]
    if len(set(names)) != len(names):
        return "top 返回的消息有重复"
    if len(got) > k:
        return f"受监控的消息有 {len(got)} 条，超过了 k = {k}"
    for x, c in got:
        f = counts.get(x, 0)
        if not isinstance(c, int):
            return f"计数应当是 int，得到 {type(c).__name__}"
        if c < f:
            return f"违反“高估有界”：{_s(x)} 真实出现了 {f} 次，汇报的计数 {c} 比它还小"
        if c * k > f * k + n_seen:
            return f"违反“高估有界”：{_s(x)} 真实出现 {f} 次，汇报 {c}，超过了 f + N/k = {f} + {n_seen}/{k}"
    if everything:
        missing = [x for x, f in counts.items() if f * k > n_seen and x not in names]
        if missing:
            x = missing[0]
            return f"违反“高频项覆盖”：{_s(x)} 出现了 {counts[x]} 次，超过 N/k = {n_seen}/{k}，却不在受监控的消息里"
    for (x1, c1), (x2, c2) in zip(got, got[1:]):
        if c1 < c2 or (c1 == c2 and x1 > x2):
            return "top 的顺序不对：按汇报的计数降序，计数相同的按消息字符串升序"
    if got != want:
        if dict(got) == dict(want):
            return "top 的顺序不对：按汇报的计数降序，计数相同的按消息字符串升序"
        if not everything:
            return f"top 应返回汇报计数最高的 {len(want)} 条受监控的消息"
        return ("两条保证都满足，但受监控的消息或计数和题目的规则不一致：表满时淘汰计数最低的那条"
                "（并列时淘汰在这个值上停留最久的），新消息继承被淘汰的计数，再加上这一次出现")
    return None


def check_top(sol_ss, sim, counts, n_seen, k, m, call):
    want = sim.top(m)
    got = _norm_top(sol_ss.top(m))
    expect_valid(got, lambda g: _ss_problem(g, want, counts, n_seen, k, m > k), call=call, want=want)


def run_ss(sol, k, stream, every=1, tops=None):
    """Adds the stream to a fresh SpaceSaving(k). After every `every`-th add and after the last, checks top(m) for
    each m in `tops` (default: k + 1, i.e. every monitored message) against a literal simulation."""
    tops = tops or [k + 1]
    calling(f"SpaceSaving({k})")
    ss, sim, counts = sol.SpaceSaving(k), ref.SpaceSaving(k), Counter()
    for i, msg in enumerate(stream):
        calling(_ss_replay(k, stream[:i], f"ss.add({_s(msg)})"))
        ss.add(msg)
        sim.add(msg)
        counts[msg] += 1
        if (i + 1) % every == 0 or i + 1 == len(stream):
            for m in tops:
                call = calling(_ss_replay(k, stream[:i + 1], f"print(ss.top({m}))"))
                check_top(ss, sim, counts, i + 1, k, m, call)


# ---------------------------------------------------------------- big inputs, built at import time (not charged)

_big = rng("big")
BIG_N = 10_000
BIG_EVENTS = ref.random_events(_big, 30_000, [f"m{i}" for i in range(10_000)], max_gap=1)

ADV_H = 300_000                                       # see p2_adversarial_bucket
ADV_A = [f"a{i}" for i in range(ADV_H + 1)]
ADV_C = [f"c{j}" for j in range(ADV_H)]


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子：login 和 click 并列")
def p1_example(sol):
    check_p1(sol, EVENTS, want=("login", [1, 4]), call=show_call("most_frequent", EVENTS))


@case("Part 1", title="空列表抛出 ValueError")
def p1_empty(sol):
    expect_raises(ValueError, sol.most_frequent, [], call="most_frequent([])")


@case("Part 1", title="只有一个事件")
def p1_single(sol):
    check_p1(sol, [("ping", 42)])
    check_p1(sol, [("ping", -7)])


@case("Part 1", title="消息各不相同：取第一个")
def p1_all_distinct(sol):
    check_p1(sol, [("c", 1), ("b", 2), ("a", 3)])
    check_p1(sol, [("x", 0), ("y", 0), ("z", 5), ("w", 9)])


@case("Part 1", title="并列：比首次出现，不是谁先达到")
def p1_tie_first_occurrence(sol):
    check_p1(sol, [("b", 1), ("a", 2), ("a", 3), ("b", 4)])
    check_p1(sol, [("x", 1), ("y", 1), ("y", 2), ("y", 3), ("x", 4), ("z", 5), ("x", 6)])
    check_p1(sol, [("p", 1), ("q", 2), ("r", 3), ("r", 4), ("q", 5), ("p", 6)])


@case("Part 1", title="并列：不按消息字符串比较")
def p1_tie_not_by_name(sol):
    check_p1(sol, [("apple", 1), ("zebra", 2), ("zebra", 3), ("apple", 4)])
    check_p1(sol, [("zebra", 1), ("apple", 2), ("apple", 3), ("zebra", 4)])


@case("Part 1", title="重复的时间戳和负数时间戳")
def p1_repeated_timestamps(sol):
    check_p1(sol, [("a", -3), ("b", -3), ("a", -3), ("a", 0)])
    check_p1(sol, [("x", 5), ("x", 5), ("y", 5), ("x", 5), ("y", 6)])


@case("Part 1", title="空串和中文消息")
def p1_unusual_messages(sol):
    check_p1(sol, [("", 1), ("登录", 2), ("", 3), ("登录", 3), ("", 8)])
    check_p1(sol, [("登录", 1), ("点击", 2), ("点击", 2), ("登录", 4)])


@case("Part 1", title="随机测试（对照暴力解）")
def p1_random(sol):
    r = rng()
    for _ in range(300):
        alphabet = [f"m{i}" for i in range(r.randint(1, 6))]
        check_p1(sol, ref.random_events(r, r.randint(1, 30), alphabet))


@case("Part 1", title="5000 个事件")
def p1_long(sol):
    r = rng()
    events = ref.random_events(r, 5000, [f"m{i}" for i in range(50)])
    check_p1(sol, events, call="most_frequent(<5000 个事件，50 种消息>)")


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子 n = 3（逐步）")
def p2_example(sol):
    events = [("login", 1), ("click", 2), ("login", 3), ("purchase", 4), ("purchase", 6)]
    wants = [("login", 1), ("click", 1), ("login", 2), ("purchase", 1), ("purchase", 2)]
    run_window(sol, 3, events, wants)


@case("Part 2", title="n = 1：只持有最新的事件")
def p2_n_one(sol):
    run_window(sol, 1, [("a", 1), ("a", 2), ("b", 2), ("a", 3), ("c", 9)])


@case("Part 2", title="同一条消息反复到达")
def p2_same_message(sol):
    run_window(sol, 3, [("a", t) for t in range(7)])


@case("Part 2", title="窗口还没满（n 比事件数大）")
def p2_never_full(sol):
    run_window(sol, 100, [("b", 1), ("a", 1), ("a", 2), ("b", 3), ("c", 3), ("c", 4), ("b", 5), ("a", 6)])


@case("Part 2", title="并列：淘汰造成的减一也算变化")
def p2_tie_eviction_is_a_change(sol):
    # a's eviction at the 6th add is its latest change; b's latest change (its arrival) is older
    run_window(sol, 5, [("a", 1), ("a", 2), ("b", 3), ("a", 4), ("b", 5), ("c", 6)])
    run_window(sol, 4, [("x", 1), ("y", 2), ("x", 3), ("y", 4), ("z", 5), ("w", 6)])


@case("Part 2", title="最高计数因淘汰而下降")
def p2_max_falls(sol):
    run_window(sol, 3, [("a", 1), ("a", 2), ("a", 3), ("b", 4), ("c", 5), ("d", 6), ("e", 7)])
    run_window(sol, 4, [("a", 1), ("a", 1), ("b", 2), ("b", 2), ("c", 3), ("d", 3), ("e", 4), ("f", 4)])


@case("Part 2", title="消息被淘汰光以后再次到达")
def p2_message_returns(sol):
    run_window(sol, 2, [("a", 1), ("b", 2), ("c", 3), ("a", 4), ("a", 5), ("b", 6), ("a", 7)])


@case("Part 2", title="两个对象互不影响")
def p2_two_objects(sol):
    calling("w1 = WindowTopMessage(2); w2 = WindowTopMessage(3)")
    w1, w2 = sol.WindowTopMessage(2), sol.WindowTopMessage(3)
    steps = [(1, "a", 1, ("a", 1)), (2, "b", 1, ("b", 1)), (1, "a", 2, ("a", 2)), (2, "b", 2, ("b", 2)),
             (1, "c", 3, ("c", 1)), (2, "a", 3, ("b", 2)), (2, "a", 4, ("a", 2)), (1, "c", 4, ("c", 2))]
    lines = ["w1 = WindowTopMessage(2)", "w2 = WindowTopMessage(3)"]
    for which, m, t, want in steps:
        lines.append(f"w{which}.add({_s(m)}, {t})")
        call = calling("\n".join(lines[:-1] + [f"print({lines[-1]})".ljust(24) + "# <- 这一步"]))
        got = _norm_pair((w1 if which == 1 else w2).add(m, t))
        expect(got, want, call=call)
        lines[-1] = lines[-1].ljust(24) + f"# -> {_pair(want)}"


@case("Part 2", title="随机测试（对照逐次重算）")
def p2_random(sol):
    r = rng()
    for _ in range(250):
        alphabet = [f"m{i}" for i in range(r.randint(1, 5))]
        n = r.choice([1, 2, 3, 4, 5, 8, 13])
        run_window(sol, n, ref.random_events(r, r.randint(1, 40), alphabet))


@case("Part 2", title="n = 10⁴、3·10⁴ 次 add：不能每次重扫")
def p2_big_window(sol):
    wants = ref.window_fast(BIG_N, BIG_EVENTS)
    desc = f"w = WindowTopMessage({BIG_N})，依次 add <30000 个随机事件，消息从 10000 种里选>"
    calling(desc)
    w = sol.WindowTopMessage(BIG_N)
    for i, (m, t) in enumerate(BIG_EVENTS):
        got = _norm_pair(w.add(m, t))
        if got != wants[i]:
            held = BIG_EVENTS[max(0, i + 1 - BIG_N):i + 1]
            expect(got, wants[i], call=f"{desc} 的第 {i + 1} 次：w.add({_s(m)}, {t})",
                   msg=_window_msg(got, wants[i], held))


@case("Part 2", cpu=12, title="最高桶被从末尾连续删：均摊 O(1)")
def p2_adversarial_bucket(sol):
    # The window holds a_h..a_1, b, b, a_1..a_h (n = 2h + 2): b and every a_i at count 2, a_h changed last.
    # h fresh messages then evict a_h, a_(h-1), ... one at a time, so the count-2 group keeps losing the message
    # that changed last. An answer that keeps each count's group in a plain dict re-skips every deleted slot at
    # that end on each read (the page's audit note); this case catches that.
    h, A, C = ADV_H, ADV_A, ADV_C
    desc = (f"h = {h}; w = WindowTopMessage(2 * h + 2)\n"
            "for m in [f\"a{i}\" for i in range(h, 0, -1)] + [\"b\", \"b\"] + [f\"a{i}\" for i in range(1, h + 1)]:\n"
            "    w.add(m, 0)\n"
            "for j in range(h):\n"
            "    w.add(f\"c{j}\", 0)")
    calling(desc)
    w = sol.WindowTopMessage(2 * h + 2)

    def fill_failed(got, m, c, k):
        expect(_norm_pair(got), (m, c), call=f"{desc}\n# 第一个循环的第 {k} 次 add：w.add({_s(m)}, 0)")

    for i in range(h, 0, -1):                       # a_h .. a_1: each new, so it wins the tie at 1
        w_got = w.add(A[i], 0)
        if w_got != (A[i], 1) and _norm_pair(w_got) != (A[i], 1):
            fill_failed(w_got, A[i], 1, h - i + 1)
    for k, c in ((h + 1, 1), (h + 2, 2)):
        w_got = w.add("b", 0)
        if _norm_pair(w_got) != ("b", c):
            fill_failed(w_got, "b", c, k)
    for i in range(1, h + 1):                       # a_1 .. a_h reach 2 and win the tie with b
        w_got = w.add(A[i], 0)
        if w_got != (A[i], 2) and _norm_pair(w_got) != (A[i], 2):
            fill_failed(w_got, A[i], 2, h + 2 + i)
    for j in range(h):
        want = (A[h - j - 1], 2) if j < h - 1 else ("b", 2)
        got = w.add(C[j], 0)
        if got != want and _norm_pair(got) != want:
            expect(_norm_pair(got), want, call=f"{desc}\n# 第二个循环 j = {j}：w.add(\"c{j}\", 0)",
                   msg=f"淘汰了 a{h - j} 之后，计数 2 的消息里最近变化的是 {_s(want[0])}")


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子 k = 3（逐步看 top）")
def p3_example(sol):
    stream = ["a", "b", "c", "d", "b", "a", "e"]
    states = [[("a", 1)], [("a", 1), ("b", 1)], [("a", 1), ("b", 1), ("c", 1)],
              [("d", 2), ("b", 1), ("c", 1)], [("b", 2), ("d", 2), ("c", 1)],
              [("a", 2), ("b", 2), ("d", 2)], [("e", 3), ("a", 2), ("b", 2)]]
    calling("SpaceSaving(3)")
    ss = sol.SpaceSaving(3)
    for i, (msg, want) in enumerate(zip(stream, states)):
        calling(_ss_replay(3, stream[:i], f"ss.add({_s(msg)})"))
        ss.add(msg)
        call = calling(_ss_replay(3, stream[:i + 1], "print(ss.top(3))"))
        expect(_norm_top(ss.top(3)), want, call=call)


@case("Part 3", title="还没有 add 时 top 返回空列表")
def p3_top_empty(sol):
    calling("ss = SpaceSaving(2)\nprint(ss.top(5))   # <- 这一步")
    expect(_norm_top(sol.SpaceSaving(2).top(5)), [])


@case("Part 3", title="top(m)：只取前 m 条，并列按消息")
def p3_top_m(sol):
    # 4 distinct messages, k = 5: nothing is ever evicted
    run_ss(sol, 5, ["b", "a", "d", "c", "a", "b"], every=6, tops=[1, 2, 3, 4, 10])
    run_ss(sol, 3, ["z", "y", "x", "z", "y", "x"], every=1, tops=[1, 2, 3])


@case("Part 3", title="k 不小于消息种数：计数是精确的")
def p3_exact_when_roomy(sol):
    r = rng()
    stream = [r.choice([f"m{i}" for i in range(12)]) for _ in range(500)]
    calling("SpaceSaving(12)")
    ss, sim, counts = sol.SpaceSaving(12), ref.SpaceSaving(12), Counter()
    for msg in stream:
        ss.add(msg)
        sim.add(msg)
        counts[msg] += 1
    call = calling("ss = SpaceSaving(12)\nfor m in <500 条消息，12 种>:\n    ss.add(m)\nprint(ss.top(20))")
    check_top(ss, sim, counts, len(stream), 12, 20, call)


@case("Part 3", title="最低计数唯一的那条被加一之后")
def p3_min_moves_up(sol):
    run_ss(sol, 2, ["a", "b", "a", "b", "c", "d"])
    run_ss(sol, 3, ["a", "b", "c", "a", "b", "c", "a", "d", "e"])


@case("Part 3", title="表没满时新计数器 1 是最低")
def p3_fresh_counter_is_lowest(sol):
    # a reaches 2 before b arrives; b's fresh counter at 1 is now the lowest, so c must evict b, not a
    run_ss(sol, 2, ["a", "a", "b", "c"])
    run_ss(sol, 3, ["a", "a", "b", "b", "c", "d", "e"])


@case("Part 3", title="新消息继承计数，再加上这一次")
def p3_seed_plus_one(sol):
    run_ss(sol, 1, ["a", "b", "a", "a", "c", "b"])
    run_ss(sol, 2, ["a", "b", "c", "a", "c", "d", "a"])


@case("Part 3", title="最低并列：淘汰停留最久的")
def p3_evict_longest_held(sol):
    run_ss(sol, 2, ["a", "b", "b", "a", "c", "d"])
    run_ss(sol, 3, ["a", "b", "c", "c", "b", "a", "d", "e", "f", "g", "h"])


@case("Part 3", title="两个对象互不影响")
def p3_two_objects(sol):
    calling("s1 = SpaceSaving(1); s2 = SpaceSaving(2)")
    s1, s2 = sol.SpaceSaving(1), sol.SpaceSaving(2)
    for msg in ["a", "b", "a"]:
        s1.add(msg)
    for msg in ["x", "y"]:
        s2.add(msg)
    call = calling('s1 = SpaceSaving(1)\ns2 = SpaceSaving(2)\nfor m in ["a", "b", "a"]:\n    s1.add(m)\n'
                   'for m in ["x", "y"]:\n    s2.add(m)\nprint(s1.top(3), s2.top(3))   # <- 这一步')
    expect((_norm_top(s1.top(3)), _norm_top(s2.top(3))), ([("a", 3)], [("x", 1), ("y", 1)]), call=call)


@case("Part 3", title="随机测试：每次 add 后检查")
def p3_random(sol):
    r = rng()
    for _ in range(200):
        k = r.randint(1, 5)
        alphabet = [chr(ord("a") + i) for i in range(r.randint(2, 8))]
        run_ss(sol, k, [r.choice(alphabet) for _ in range(r.randint(1, 40))])


@case("Part 3", title="长流（偏斜分布）：两条保证")
def p3_zipf(sol):
    r = rng()
    for k, length, kinds in [(10, 20_000, 300), (30, 20_000, 300), (100, 5_000, 2_000)]:
        stream = ref.zipf_stream(r, length, kinds)
        desc = f"ss = SpaceSaving({k})，依次 add <{length} 条偏斜分布的消息，{kinds} 种>"
        calling(desc)
        ss, sim, counts = sol.SpaceSaving(k), ref.SpaceSaving(k), Counter()
        for i, msg in enumerate(stream):
            ss.add(msg)
            sim.add(msg)
            counts[msg] += 1
            if (i + 1) % 1000 == 0 or i + 1 == length:
                check_top(ss, sim, counts, i + 1, k, k + 1, call=calling(f"{desc} 的前 {i + 1} 条后：ss.top({k + 1})"))
