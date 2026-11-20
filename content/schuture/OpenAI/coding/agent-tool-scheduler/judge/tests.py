import numbers

from judgelib import calling, case, expect, expect_raises, expect_true, info, rng, show_call
import ref
from ref import ToolCall as TC

CYCLE = "CycleError"            # stands for "must raise CycleError" wherever an expected value goes


def fresh(x):
    """A copy of a call list or an agents list, so an answer that mutates its input cannot touch ours."""
    if isinstance(x, list):
        return [list(v) if isinstance(v, list) else v for v in x]
    return x


def want_of(fn, *args):
    try:
        return fn(*args)
    except ref.Cycle:
        return CYCLE


def cycle_error(sol):
    exc = getattr(sol, "CycleError", None)
    expect_true(isinstance(exc, type) and issubclass(exc, Exception),
                "题面要求定义 CycleError(Exception)，答案里没有找到这个异常类")
    return exc


def normalize(got):
    """(starts, makespan) as a list of int triples and an int, or None when the shape is not that."""
    def is_int(x):
        return isinstance(x, numbers.Integral) and not isinstance(x, bool)
    try:
        if isinstance(got, (str, bytes, dict)) or len(got) != 2:
            return None
        starts, span = got
        out = []
        for item in starts:
            if isinstance(item, (str, bytes, dict)) or len(item) != 3 or not all(is_int(x) for x in item):
                return None
            out.append(tuple(int(x) for x in item))
        return (out, int(span)) if is_int(span) else None
    except TypeError:
        return None


def check(got, want, call):
    norm = normalize(got)
    expect_true(norm is not None, "返回值应当是 (starts, makespan)：starts 是 (start_time, agent_id, tool_id) "
                "整数三元组的列表，makespan 是整数", call, got)
    starts, span = norm
    want_starts, want_span = want
    if starts != want_starts:
        k = next((i for i, (a, b) in enumerate(zip(starts, want_starts)) if a != b), min(len(starts), len(want_starts)))
        if k < min(len(starts), len(want_starts)):
            msg = f"starts 第 {k} 条（从 0 数）不对：期望 {want_starts[k]}，得到 {starts[k]}"
        elif len(starts) < len(want_starts):
            msg = f"starts 少了：期望 {len(want_starts)} 条，只有 {len(starts)} 条，缺的第一条是 {want_starts[k]}"
        else:
            msg = f"starts 多了：期望 {len(want_starts)} 条，得到 {len(starts)} 条，多出的第一条是 {starts[k]}"
        if span != want_span:
            msg += f"；makespan 也不对（期望 {want_span}，得到 {span}）"
        expect(norm, want, call=call, msg=msg)
    expect(span, want_span, call=call, msg=f"starts 全对，makespan 不对：期望 {want_span}，得到 {span}")


def run(sol, fn, args, want, desc=None):
    """Call sol.<fn>(*copies of args) and compare with `want`, a (starts, makespan) pair or CYCLE."""
    call = calling(desc or show_call(fn, *args))
    if want == CYCLE:
        expect_raises(cycle_error(sol), getattr(sol, fn), *map(fresh, args), call=call)
    else:
        check(getattr(sol, fn)(*map(fresh, args)), want, call)


def p1(sol, calls, cap, want=None):
    run(sol, "schedule_agent", (calls, cap), want_of(ref.brute, [calls], cap) if want is None else want)


def p2(sol, agents, caps, want=None):
    run(sol, "schedule_agents_independently", (agents, caps),
        want_of(ref.independent, agents, caps) if want is None else want)


def p3(sol, agents, cap, want=None):
    run(sol, "schedule_agents_global", (agents, cap), want_of(ref.brute, agents, cap) if want is None else want)


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    p1(sol, [TC(0, 2), TC(1, 1), TC(2, 1, deps=(0,))], 1, ([(0, 0, 0), (2, 0, 1), (3, 0, 2)], 4))


@case("Part 1", title="没有调用：返回 ([], 0)")
def p1_empty(sol):
    p1(sol, [], 1, ([], 0))
    p1(sol, [], 5, ([], 0))


@case("Part 1", title="只有一个调用")
def p1_single(sol):
    p1(sol, [TC(0, 7)], 1, ([(0, 0, 0)], 7))
    p1(sol, [TC(0, 1)], 3, ([(0, 0, 0)], 1))


@case("Part 1", title="完成时刻当场空出槽位、解锁后继")
def p1_free_at_finish(sol):
    # one slot: each call starts the moment the previous one finishes
    p1(sol, [TC(0, 3), TC(1, 2), TC(2, 4)], 1, ([(0, 0, 0), (3, 0, 1), (5, 0, 2)], 9))
    # call 0 finishes at 3: its slot and its dependent call 1 are both usable at 3, not 4
    p1(sol, [TC(0, 3), TC(1, 5, deps=(0,)), TC(2, 10)], 2, ([(0, 0, 0), (0, 0, 2), (3, 0, 1)], 10))
    p1(sol, [TC(0, 4), TC(1, 1, deps=(0,)), TC(2, 1, deps=(1,))], 5, ([(0, 0, 0), (4, 0, 1), (5, 0, 2)], 6))


@case("Part 1", title="按 tool_id 而不是列表位置打破并列")
def p1_tie_by_tool_id(sol):
    calls = [TC(3, 1), TC(2, 1), TC(1, 1), TC(0, 1)]
    p1(sol, calls, 1, ([(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3)], 4))
    p1(sol, calls, 2, ([(0, 0, 0), (0, 0, 1), (1, 0, 2), (1, 0, 3)], 2))
    # calls 1 and 2 both wait for 0; the one with the smaller tool_id gets the only slot
    p1(sol, [TC(2, 5, deps=(0,)), TC(0, 1), TC(1, 2, deps=(0,))], 1, ([(0, 0, 0), (1, 0, 1), (3, 0, 2)], 8))


@case("Part 1", title="同时完成的调用要先全部释放")
def p1_release_all_first(sol):
    # at t=2 calls 0 and 1 both finish; 1 unlocks 2 and 3, which beat call 4 (waiting since t=0) to the two slots
    calls = [TC(0, 2), TC(1, 2), TC(2, 1, deps=(1,)), TC(3, 1, deps=(1,)), TC(4, 1)]
    p1(sol, calls, 2, ([(0, 0, 0), (0, 0, 1), (2, 0, 2), (2, 0, 3), (3, 0, 4)], 4))
    p1(sol, [calls[k] for k in (4, 2, 0, 3, 1)], 2, ([(0, 0, 0), (0, 0, 1), (2, 0, 2), (2, 0, 3), (3, 0, 4)], 4))


@case("Part 1", title="makespan 是最晚的完成时刻")
def p1_makespan(sol):
    # the last call to start (2, at t=1) finishes at 2, long before call 0
    p1(sol, [TC(0, 10), TC(1, 1), TC(2, 1)], 2, ([(0, 0, 0), (0, 0, 1), (1, 0, 2)], 10))
    p1(sol, [TC(0, 1), TC(1, 9), TC(2, 1, deps=(0,)), TC(3, 2, deps=(2,))], 3)


@case("Part 1", title="上限超过调用数")
def p1_big_capacity(sol):
    p1(sol, [TC(0, 3), TC(1, 1), TC(2, 2, deps=(1,)), TC(3, 1, deps=(0, 2))], 10,
       ([(0, 0, 0), (0, 0, 1), (1, 0, 2), (3, 0, 3)], 4))
    r = rng()
    for _ in range(20):
        calls = ref.random_agent(r, r.randint(1, 12), 6, 3)
        p1(sol, calls, len(calls) + r.randint(0, 5))


@case("Part 1", title="依赖多、菱形依赖：没有环就不能报环")
def p1_dag(sol):
    p1(sol, [TC(3, 1, deps=(2, 1)), TC(0, 1), TC(1, 2, deps=(0,)), TC(2, 3, deps=(0,))], 1,
       ([(0, 0, 0), (1, 0, 1), (3, 0, 2), (6, 0, 3)], 7))
    p1(sol, [TC(3, 1, deps=(2, 1)), TC(0, 1), TC(1, 2, deps=(0,)), TC(2, 3, deps=(0,))], 2,
       ([(0, 0, 0), (1, 0, 1), (1, 0, 2), (4, 0, 3)], 5))
    # call 5 waits for everything else
    p1(sol, [TC(5, 1, deps=(4, 0, 3, 1, 2)), TC(4, 2), TC(3, 2, deps=(4,)), TC(2, 1), TC(1, 3, deps=(2,)), TC(0, 1)], 2)


@case("Part 1", title="有环：抛出 CycleError")
def p1_cycle(sol):
    p1(sol, [TC(0, 1, deps=(1,)), TC(1, 1, deps=(0,))], 1, CYCLE)
    p1(sol, [TC(0, 2, deps=(2,)), TC(1, 1, deps=(0,)), TC(2, 3, deps=(1,))], 3, CYCLE)
    # call 0 runs and finishes; 1 and 2 wait on each other and 3 waits on them
    p1(sol, [TC(3, 1, deps=(1,)), TC(0, 5), TC(1, 1, deps=(2,)), TC(2, 1, deps=(1, 0))], 2, CYCLE)
    # a cycle reached only after a long chain of good work
    p1(sol, [TC(0, 4), TC(1, 4, deps=(0,)), TC(2, 4, deps=(1,)), TC(3, 1, deps=(2, 4)), TC(4, 1, deps=(3,)),
             TC(5, 1)], 1, CYCLE)


@case("Part 1", title="时长到 10⁹：不能逐个时间单位推进")
def p1_long_durations(sol):
    p1(sol, [TC(0, 10 ** 9), TC(1, 1), TC(2, 999_999_999, deps=(1,)), TC(3, 10 ** 9, deps=(0, 2))], 2,
       ([(0, 0, 0), (0, 0, 1), (1, 0, 2), (10 ** 9, 0, 3)], 2 * 10 ** 9))
    r = rng()
    for _ in range(10):
        calls = ref.random_agent(r, r.randint(1, 10), 10 ** 9, 3)
        p1(sol, calls, r.randint(1, 4))


@case("Part 1", title="随机小数据，对拍")
def p1_random(sol):
    r = rng()
    for k in range(400):
        n = r.randint(1, 4 + k // 25)
        calls = ref.random_agent(r, n, r.choice((1, 2, 3, 5, 10, 10 ** 9)), r.randint(1, 3),
                                 1 if r.random() < 0.15 else 0)
        p1(sol, calls, r.randint(1, n + 1))


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    agents = [[TC(0, 2), TC(1, 1, deps=(0,))], [TC(0, 1), TC(1, 1)]]
    p2(sol, agents, [1, 2], ([(0, 0, 0), (0, 1, 0), (0, 1, 1), (2, 0, 1)], 3))


@case("Part 2", title="有的 agent 没有调用")
def p2_empty_agents(sol):
    p2(sol, [[]], [1], ([], 0))
    p2(sol, [[], [TC(0, 3)], []], [1, 1, 2], ([(0, 1, 0)], 3))


@case("Part 2", title="各 agent 的上限互不影响")
def p2_own_capacity(sol):
    agents = [[TC(0, 2), TC(1, 2), TC(2, 2)], [TC(0, 1), TC(1, 1), TC(2, 1)]]
    p2(sol, agents, [1, 3], ([(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 2), (2, 0, 1), (4, 0, 2)], 6))
    p2(sol, agents, [3, 1], ([(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (1, 1, 1), (2, 1, 2)], 3))


@case("Part 2", title="相同 tool_id 在不同 agent 里是不同调用")
def p2_own_ids(sol):
    agents = [[TC(0, 5), TC(1, 1, deps=(0,))], [TC(1, 1, deps=(0,)), TC(0, 1)]]
    p2(sol, agents, [2, 2], ([(0, 0, 0), (0, 1, 0), (1, 1, 1), (5, 0, 1)], 6))


@case("Part 2", title="多条时间线按开始时刻交织")
def p2_interleave(sol):
    agents = [[TC(0, 2), TC(1, 2, deps=(0,)), TC(2, 2, deps=(1,))], [TC(0, 1), TC(1, 2, deps=(0,)), TC(2, 1, deps=(1,))]]
    p2(sol, agents, [1, 1], ([(0, 0, 0), (0, 1, 0), (1, 1, 1), (2, 0, 1), (3, 1, 2), (4, 0, 2)], 6))
    r = rng()
    for _ in range(10):
        agents = [ref.random_agent(r, r.randint(1, 6), 4, 2) for _ in range(r.randint(2, 4))]
        p2(sol, agents, [r.randint(1, 3) for _ in agents])


@case("Part 2", title="一个 agent 有环也要抛 CycleError")
def p2_cycle(sol):
    p2(sol, [[TC(0, 1, deps=(1,)), TC(1, 1, deps=(0,))], [TC(0, 1)]], [1, 1], CYCLE)
    p2(sol, [[TC(0, 3)], [TC(0, 1), TC(1, 1, deps=(2,)), TC(2, 1, deps=(1,))]], [1, 2], CYCLE)
    p2(sol, [[TC(0, 1)], [TC(1, 2, deps=(0,)), TC(0, 2, deps=(2,)), TC(2, 2, deps=(1,))], [TC(0, 1), TC(1, 1)]],
       [1, 1, 2], CYCLE)


@case("Part 2", title="时长到 10⁹")
def p2_long_durations(sol):
    agents = [[TC(0, 10 ** 9), TC(1, 10 ** 9, deps=(0,))], [TC(0, 1), TC(1, 999_999_999)]]
    p2(sol, agents, [1, 1], ([(0, 0, 0), (0, 1, 0), (1, 1, 1), (10 ** 9, 0, 1)], 2 * 10 ** 9))
    r = rng()
    for _ in range(10):
        agents = [ref.random_agent(r, r.randint(0, 8), 10 ** 9, 3) for _ in range(r.randint(1, 4))]
        p2(sol, agents, [r.randint(1, 3) for _ in agents])


@case("Part 2", title="随机小数据，对拍")
def p2_random(sol):
    r = rng()
    for k in range(250):
        agents = ref.random_agents(r, r.randint(1, 4), 2 + k // 25, r.choice((1, 2, 3, 5, 10 ** 9)), 0.1)
        p2(sol, agents, [r.randint(1, len(calls) + 1) for calls in agents])


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    agents = [[TC(0, 2), TC(1, 3, deps=(0,)), TC(2, 1, deps=(0,))],
              [TC(0, 4), TC(1, 2, deps=(0,))],
              [TC(0, 2)]]
    p3(sol, agents, 2, ([(0, 0, 0), (0, 1, 0), (2, 0, 1), (4, 0, 2), (5, 1, 1), (5, 2, 0)], 7))


@case("Part 3", title="有的 agent 没有调用")
def p3_empty_agents(sol):
    p3(sol, [[]], 1, ([], 0))
    p3(sol, [[], []], 3, ([], 0))
    p3(sol, [[], [TC(0, 2)]], 1, ([(0, 1, 0)], 2))


@case("Part 3", title="agent 之间抢同一批槽位")
def p3_contention(sol):
    # agent 0's second call beats agent 1's first: (0, 1) < (1, 0)
    p3(sol, [[TC(1, 3), TC(0, 3)], [TC(0, 1)]], 1, ([(0, 0, 0), (3, 0, 1), (6, 1, 0)], 7))
    p3(sol, [[TC(0, 3), TC(1, 3)], [TC(0, 1)], [TC(0, 2)]], 2, ([(0, 0, 0), (0, 0, 1), (3, 1, 0), (3, 2, 0)], 5))
    p3(sol, [[TC(0, 1), TC(1, 1, deps=(0,))], [TC(0, 5)], [TC(0, 1)]], 2,
       ([(0, 0, 0), (0, 1, 0), (1, 0, 1), (2, 2, 0)], 5))


@case("Part 3", title="相同 tool_id 在不同 agent 里是不同调用")
def p3_own_ids(sol):
    p3(sol, [[TC(0, 5), TC(1, 1, deps=(0,))], [TC(1, 1, deps=(0,)), TC(0, 1)]], 4,
       ([(0, 0, 0), (0, 1, 0), (1, 1, 1), (5, 0, 1)], 6))
    p3(sol, [[TC(1, 2, deps=(0,)), TC(0, 1)], [TC(0, 1), TC(1, 1, deps=(0,))]], 1)


@case("Part 3", title="不同 agent 同时完成：先全部释放")
def p3_release_all_first(sol):
    # t=2: (0,0) and (1,0) finish together; (1,0) unlocks (1,1) and (1,2), which beat (2,0) to both slots
    agents = [[TC(0, 2)], [TC(0, 2), TC(1, 1, deps=(0,)), TC(2, 1, deps=(0,))], [TC(0, 1)]]
    p3(sol, agents, 2, ([(0, 0, 0), (0, 1, 0), (2, 1, 1), (2, 1, 2), (3, 2, 0)], 4))


@case("Part 3", title="有环：抛出 CycleError")
def p3_cycle(sol):
    p3(sol, [[TC(0, 1, deps=(1,)), TC(1, 1, deps=(0,))], [TC(0, 1)]], 2, CYCLE)
    p3(sol, [[TC(0, 5)], [TC(0, 1), TC(2, 1, deps=(1,)), TC(1, 1, deps=(2,))], [TC(0, 3), TC(1, 3, deps=(0,))]], 1,
       CYCLE)


@case("Part 3", title="随机小数据，对拍")
def p3_random(sol):
    r = rng()
    for k in range(300):
        agents = ref.random_agents(r, r.randint(1, 5), 1 + k // 30, r.choice((1, 2, 3, 5, 10 ** 9)), 0.1)
        total = sum(map(len, agents))
        p3(sol, agents, r.randint(1, total + 2))


@case("Part 3", cpu=4, title="2.5 万个 agent 抢 1000 个槽位")
def p3_big_contention(sol):
    agents = ref.big_many_agents(rng("big"), 25_000)
    run(sol, "schedule_agents_global", (agents, 1000), ref.fast(agents, 1000),
        desc="schedule_agents_global(<25000 个 agent，每个 2 个调用，时长到 10⁹>, 1000)")


@case("Part 3", cpu=5, title="5 万个调用、12 万条依赖边")
def p3_big_dense(sol):
    agents = ref.big_dense(rng("big"), 500)
    run(sol, "schedule_agents_global", (agents, 50_000), ref.fast(agents, 50_000),
        desc="schedule_agents_global(<500 个 agent，每个 100 个调用、2~3 个依赖，时长到 10⁹>, 50000)")


@case("Part 3", cpu=3, title="一条 5 万层的依赖链")
def p3_long_chain(sol):
    r = rng("chain")
    n = 50_000
    calls = [TC(t, r.randint(1, 10 ** 9), (t - 1,) if t else ()) for t in range(n)]
    starts, t = [], 0
    for c in calls:                 # each call starts the moment the one before it finishes
        starts.append((t, 0, c.tool_id))
        t += c.duration
    r.shuffle(calls)
    try:
        run(sol, "schedule_agents_global", ([calls], 3), (starts, t),
            desc="schedule_agents_global([<一个 agent：50000 个调用，调用 t 依赖调用 t-1>], 3)")
    except RecursionError:
        info("提示：这条依赖链有 50000 层（题目允许 N 到 2×10⁵），沿依赖递归的 DFS 会超过递归深度上限；"
             "改用显式栈，或者按剩余依赖数（入度）推进")
        raise
