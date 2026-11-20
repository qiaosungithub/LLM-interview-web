"""Tests for the fixed RateLimiter.

Only the statement's interface is used: RateLimiter(minute_rule, hour_rule, day_rule, clock=..., checkpoint=...),
should_allow_request(user_id) and rate_limited(limiter). The clock is the tests' own (ref.Clock), moved by hand.
Every single-threaded call is compared with ref.Model, the statement's definition read directly (keep every admitted
time, count those with now - t <= period).

In the thread tests every call of a round happens at the same instant, so the statement's rule ("the same as running
them one after another") fixes how many are admitted per user: exactly as many as there is room for. checkpoint is a
ref.Crowd: where the limiter calls it between deciding and recording, callers that decided before anybody recorded
are held there together, so a limiter without a lock over both admits too many on every run. In race_paced every
thread also sleeps before each line of the answer's code (ref.run_threads(pace=...)), which pulls apart a check and an
action that are not under one lock even where checkpoint is not between them.
"""
from collections import Counter
import threading

from judgelib import Mismatch, calling, case, expect, info, rng

import ref
from ref import Session, rules_text

BIG = 10_000
DAY = 86_400
DEADLINE = 5.0     # seconds a thread round may take; below the case limit (cpu * 3 + 2)
PACE = 0.003       # race_paced: seconds each thread sleeps before every line of the answer's code


def room():
    n = threading.active_count()
    if n > 120:
        raise Mismatch(f"前面失败的测试留下了 {n} 个没有返回的线程，线程数接近上限，这个测试没有运行", show=False)


# ---------------------------------------------------------------- single thread

@case("Part 1", sample=True, title="题目里的测试：三次请求都在上限以内")
def smoke(sol):
    calling("limiter = RateLimiter(minute_rule=(5, 60), hour_rule=(20, 3600), day_rule=(50, 86400))\n"
            "@rate_limited(limiter)\ndef post_message(user_id, text):\n"
            "    return {'status': 200, 'body': {'echo': text}}\n"
            "[post_message('alice', 'hi')['status'] for _ in range(3)]")
    limiter = sol.RateLimiter(minute_rule=(5, 60), hour_rule=(20, 3600), day_rule=(50, 86400))

    @sol.rate_limited(limiter)
    def post_message(user_id, text):
        return {"status": 200, "body": {"echo": text}}

    expect([post_message("alice", "hi")["status"] for _ in range(3)], [200, 200, 200])


class StubLimiter:
    """Answers should_allow_request from a fixed list and records the user ids it was asked about."""

    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    def should_allow_request(self, user_id):
        self.asked.append(user_id)
        return self.answers[len(self.asked) - 1]


@case("Part 1", title="装饰器：放行时返回视图的结果，拒绝时返回 429、不调用视图")
def decorator(sol):
    lines = ["limiter = StubLimiter([True, True, False, True])   # 测试用的替身：按顺序回答 should_allow_request",
             "@rate_limited(limiter)",
             "def view(user_id, text, loud=False):",
             "    return {'status': 200, 'body': {'echo': text.upper() if loud else text}}"]
    calling("\n".join(lines))
    limiter = StubLimiter([True, True, False, True])
    seen = []

    def view(user_id, text, loud=False):
        seen.append((user_id, text, loud))
        return {"status": 200, "body": {"echo": text.upper() if loud else text}}

    wrapped = sol.rate_limited(limiter)(view)
    steps = [("view('ana', 'hi')", ("ana", "hi"), {}, True),
             ("view('ana', 'yo', loud=True)", ("ana", "yo"), {"loud": True}, True),
             ("view('bruno', 'again')", ("bruno", "again"), {}, False),
             ("view('ana', 'back')", ("ana", "back"), {}, True)]
    for text, args, kwargs, allowed in steps:
        call = calling("\n".join(lines + [f"{text}   # ← 这一步"]))
        lines.append(text)
        asked, before = len(limiter.asked), len(seen)
        got = wrapped(*args, **kwargs)
        if limiter.asked[asked:] != [args[0]]:
            raise Mismatch(f"每次请求应当调用一次 limiter.should_allow_request({args[0]!r})", call, [args[0]],
                           limiter.asked[asked:])
        if not allowed:
            if len(seen) != before:
                raise Mismatch("should_allow_request 返回了 False，装饰器应当直接返回 429，不再调用视图，但视图被调用了",
                               call, show=False)
            if not isinstance(got, dict) or got.get("status") != 429:
                raise Mismatch("should_allow_request 返回了 False，装饰器应当返回 429 响应", call,
                               {"status": 429, "body": {"error": "rate limit exceeded"}}, got)
            continue
        want_call = (*args, kwargs.get("loud", False))
        if seen[before:] != [want_call]:
            raise Mismatch(f"should_allow_request 返回了 True，视图应当以同样的参数被调用一次", call, [want_call],
                           seen[before:])
        expect(got, {"status": 200, "body": {"echo": args[1].upper() if kwargs else args[1]}}, call,
               "放行的请求应当原样返回视图的结果")


@case("Part 1", title="窗口滚过之后：过期的请求不再计入，新的请求照常计入")
def window_rolls(sol):
    s = Session(sol, [(3, 60), (BIG, 60), (BIG, 60)])
    s.requests("ana", 3)
    s.advance(61)
    hint = "（一条放行的请求在 now - t <= 60 时计入，过了 60 秒就不再计入；窗口滚过之后放行的请求照样要计入。）"
    s.requests("ana", 10, hint)
    s.advance(30)
    s.requests("ana", 2, hint)
    s.advance(31)
    s.requests("ana", 5, hint)
    s.advance(61)
    for _ in range(4):
        s.requests("ana", 2, hint)
        s.advance(20)
        s.requests("ana", 3, hint)
        s.advance(41)


@case("Part 1", title="边界：now - t 恰好等于 period 时仍然计入")
def boundary(sol):
    s = Session(sol, [(2, 10), (3, 100), (4, 1000)])
    hint = "（题目说只要 now - t <= period 就一直计入，所以恰好过了 period 秒的请求还算在里面，再多一点才不算。）"
    s.requests("ana", 3, hint)
    s.advance(10)
    s.requests("ana", 1, hint)
    s.advance(0.5)
    s.requests("ana", 2, hint)
    s.advance(89.5)                 # +100: the first two are exactly 100 s old
    s.requests("ana", 2, hint)
    s.advance(0.25)
    s.requests("ana", 2, hint)
    s.advance(899.75)               # +1000
    s.requests("ana", 2, hint)
    s.advance(0.5)
    s.requests("ana", 3, hint)
    z = Session(sol, [(1, 0), (BIG, 60), (BIG, 60)])      # period 0: only requests at this very instant count
    z.requests("bruno", 2, hint)
    z.advance(0.5)
    z.requests("bruno", 2, hint)


@case("Part 1", title="题目默认的三档（每分钟 5 次、每小时 20 次、每天 50 次），跨过一分钟、一小时和一天")
def default_tiers(sol):
    s = Session(sol, [(5, 60), (20, 3600), (50, DAY)])
    for step in [0, 61, 61, 61, 61, 61, 600, 3000, 61, 61, 3600, 61, 7200, 61, 61, 3600, 61, 61, 3600, 3600,
                 3600, 61, 61, 3600, 20000, 61, 61, 3600, 30000, 61, 61, 61, 3600, 61, 61]:
        if step:
            s.advance(step)
        s.requests("ana", 7)


@case("Part 1", title="每天 5 次的规则：每隔一整天来一次，8 次都该放行")
def one_a_day(sol):
    s = Session(sol, [(BIG, 60), (BIG, 3600), (5, DAY)])
    hint = "（每日规则数的是最近 86400 秒里被放行的请求，不是这个用户有史以来的总数。）"
    for _ in range(8):
        s.request("bruno", hint)
        s.advance(DAY + 1)
    for _ in range(3):
        s.requests("bruno", 2, hint)
        s.advance(DAY // 2)


@case("Part 1", title="每日规则也是滑动窗口：看最近 period 秒，不按自然日、也不从第一次请求起整段清零")
def sliding_day(sol):
    s = Session(sol, [(BIG, 60), (BIG, 3600), (3, DAY)])
    hint = "（每日规则和另外两条一样：一条请求从被放行起计入整整 86400 秒，然后才不再计入。）"
    s.request("ana", hint)              # +0
    s.advance(6000)
    s.requests("bruno", 4, hint)        # +6000
    s.advance(500)                      # +6500: past midnight UTC
    s.requests("bruno", 1, hint)
    s.requests("ana", 3, hint)
    s.advance(DAY + 1 - 6500)           # +86401: ana's first request is gone, the two from +6500 are not
    s.requests("ana", 2, hint)
    s.requests("bruno", 1, hint)
    s.advance(6000)                     # +92401: bruno's requests from +6000 are gone
    s.requests("bruno", 4, hint)
    s.requests("ana", 3, hint)


@case("Part 1", title="两条规则都要满足：每分钟 4 次加每小时 2 次")
def tighter_rule(sol):
    s = Session(sol, [(4, 60), (2, 3600), (BIG, DAY)])
    hint = "（一次被放行的请求在每条规则里只计一次。）"
    s.requests("chidi", 4, hint)
    s.advance(3601)
    s.requests("chidi", 4, hint)
    s.advance(1800)
    s.requests("chidi", 2, hint)
    s.advance(1801)
    s.requests("chidi", 3, hint)


@case("Part 1", title="被拒绝的请求不计入任何规则")
def rejected_leave_nothing(sol):
    hint = "（被拒绝的请求一条规则也不记，哪怕它满足了其中几条规则。）"
    s = Session(sol, [(3, 100), (1, 10), (BIG, DAY)])
    s.request("dora", hint)
    for _ in range(4):
        s.advance(2)
        s.request("dora", hint)
    s.advance(3)                        # +11: the admitted request is out of the 10 s rule, nothing else counts
    s.requests("dora", 2, hint)
    s.advance(11)
    s.requests("dora", 2, hint)
    d = Session(sol, [(2, 100), (2, 100), (1, 10)])
    d.request("emil", hint)
    for step in (1, 4, 3):
        d.advance(step)
        d.request("emil", hint)
    d.advance(3)                        # +11
    d.requests("emil", 2, hint)
    d.advance(11)
    d.requests("emil", 2, hint)


@case("Part 1", title="不同用户各算各的")
def users_apart(sol):
    s = Session(sol, [(2, 60), (3, 3600), (4, DAY)])
    hint = "（每个用户只数自己被放行的请求。）"
    users = ["ana", "Ana", "bruno", "", 7]
    for step in [0, 30, 31, 61, 3600, 61, 61, DAY, 61]:
        if step:
            s.advance(step)
        for k, u in enumerate(users):
            s.requests(u, 1 + (k + step) % 3, hint)


@case("Part 1", title="随机规则和时间步：每一次调用都和题目的定义比对")
def random_scripts(sol):
    r = rng()
    cover = Counter()
    periods = [0, 1, 2, 3, 5, 10, 30, 60]
    steps = [0, 0, 0, 0.5, 1, 1, 2, 3, 5, 10, 30, 60]
    for _ in range(160):
        rules = [(r.randint(1, 4), r.choice(periods)) for _ in range(3)]
        s = Session(sol, rules)
        users = ["ana", "bruno", "chidi"][:r.randint(1, 3)]
        for _ in range(40):
            step = r.choice(steps)
            if step:
                s.advance(step)
            u = r.choice(users)
            now = s.clock.t
            times = s.model.admitted.get(u, [])
            if any(now - t == p for t in times for _, p in rules):
                cover["boundary"] += 1
            cover["admitted" if s.request(u) else "rejected"] += 1
    info(f"160 段随机脚本，共 {cover['admitted'] + cover['rejected']} 次调用：放行 {cover['admitted']} 次，"
         f"拒绝 {cover['rejected']} 次，其中 {cover['boundary']} 次有已放行的请求恰好在某条规则的边界上（now - t == period）")


# ---------------------------------------------------------------- threads

def race(sol, rules, users, prefill=(), title="", pace=0.0, hint=""):
    """prefill: (advance, user, k) steps run one call at a time; then one call per entry of `users`, each on its
    own thread, all at the same instant, with checkpoint = ref.Crowd. pace, hint: see race_paced."""
    room()
    switch = ref.Switch()
    s = Session(sol, rules, checkpoint=switch)
    s.lines[1] = s.lines[1][:-1] + ", checkpoint=gate)"
    for step, user, k in prefill:
        if step:
            s.advance(step)
        s.requests(user, k)
    now = s.clock.t
    per_user = Counter(users)
    want = {u: min(n, s.model.room(u, now)) for u, n in per_user.items()}
    need = sum(want.values()) + 1
    gate = ref.Crowd(need, len(users))
    who = "、".join(f"{u!r} {n} 个" for u, n in per_user.items())
    text = calling("\n".join(s.lines) + f"\n# 同一时刻起 {len(users)} 个线程，每个线程调用一次 limiter.should_allow_request("
                   f"user_id)：{who}。\n# gate 让做完判断、还没记录的调用在里面等，直到 {need} 个调用同时在里面，"
                   "或者确认同一时刻只能进来一个"
                   + (f"\n# 每个线程在 should_allow_request 里每执行一行代码前都先停 {pace * 1000:g} 毫秒" if pace else ""))
    switch.target = gate
    try:
        results = ref.run_threads(s.limiter, users, gate, deadline=DEADLINE, pace=pace)
    except ref.Stuck as e:
        raise Mismatch(f"{e.args[0]} 个线程 {DEADLINE:g} 秒后仍没有从 should_allow_request 返回：可能死锁了", text,
                       show=False) from None
    finally:
        switch.target = None
    got = Counter(u for u, ok in zip(users, results) if ok)
    for u, n in per_user.items():
        if got[u] != want[u]:
            msg = (f"{n} 个线程同一时刻为 {u!r} 调用 should_allow_request（规则 {rules_text(rules)}），按题目的定义此刻"
                   f"还能放行 {want[u]} 次，实际放行了 {got[u]} 次。题目要求同一个用户的调用落在不同线程上时，结果和"
                   f"它们一前一后跑出来的一样：同一时刻的这 {n} 次请求一前一后跑，正好放行 {want[u]} 次。")
            if gate.reason == "crowd" and got[u] > want[u]:
                msg += (f"测试把 checkpoint 换成了一道闸：调用在判断之后、记录之前停在这里，曾经有 {need} 个调用同时停在"
                        f"里面，也就是说这 {need} 次判断都是在其中任何一次记录之前做出的。")
            if got[u] > want[u]:
                msg += hint
            raise Mismatch(msg, text, want[u], got[u])
    return gate


@case("Part 1", title="20 个线程同时请求，上限是每分钟 5 次")
def race_minute(sol):
    race(sol, [(5, 60), (BIG, 3600), (BIG, DAY)], ["dora"] * 20)


@case("Part 1", title="并发请求，起作用的是小时规则（之前已放行 2 次，还剩 2 次）")
def race_hour(sol):
    race(sol, [(BIG, 60), (4, 3600), (BIG, DAY)], ["emil"] * 16, prefill=[(0, "emil", 2), (1800, "emil", 0)])


@case("Part 1", title="4 个用户各 10 个线程同时请求")
def race_users(sol):
    users = ["ana", "bruno", "chidi", "dora"]
    race(sol, [(5, 60), (7, 3600), (BIG, DAY)], [u for _ in range(10) for u in users],
         prefill=[(0, u, 3) for u in users] + [(61, "ana", 0)])


@case("Part 1", title="放慢每一行：新用户的第一批并发请求")
def race_paced(sol):
    ms = f"{PACE * 1000:g}"
    hint = (f"这个测试让每个线程在 should_allow_request 里每执行一行代码前都先停 {ms} 毫秒（线程本来就可能在任意一行被"
            "系统暂停，这只是其中一种合法的调度），于是每两行之间都会有别的线程插进来。查看共享状态和根据它采取行动"
            "必须在同一次持锁里完成：比如这个用户第一次出现时，用「if user_id not in locks: locks[user_id] = Lock()」"
            "现建一把锁，几个线程会各自建一把、互相拦不住；又比如在一个 with 块里判断、在另一个 with 块里记录，两块"
            "之间别的线程就能拿同样的旧数据做判断。")
    race(sol, [(2, 60), (BIG, 3600), (BIG, DAY)], ["fern"] * 8, pace=PACE, hint=hint)
    race(sol, [(BIG, 60), (2, 3600), (BIG, DAY)], ["gus", "hana"] * 4, pace=PACE, hint=hint)


@case("Part 1", title="多用户、多轮、调用中途随机停顿的真实并发")
def thread_rounds(sol):
    r = rng()
    users = ["ana", "bruno", "chidi", "dora"]
    rules = [(3, 60), (5, 3600), (6, DAY)]
    for rep in range(3):
        room()
        clock = ref.Clock()
        model = ref.Model(rules)
        text = calling(f"clock = FakeClock()\nlimiter = RateLimiter((3, 60), (5, 3600), (6, 86400), clock=clock, "
                       f"checkpoint=jitter)\n# 每一轮：4 个用户各 8 个线程同一时刻调用 should_allow_request；jitter 随机"
                       f"让出线程或停顿不到 0.2 毫秒（第 {rep + 1} 组）")
        limiter = sol.RateLimiter(*rules, clock=clock, checkpoint=ref.Jitter(r))
        with ref.fast_switching():
            for step in [0, 61, 61, 3601, 30, 61, DAY, 3601]:
                clock.advance(step)
                now = clock.t
                want = {u: model.room(u, now) for u in users}
                calls = [u for _ in range(8) for u in users]
                r.shuffle(calls)
                try:
                    results = ref.run_threads(limiter, calls, deadline=DEADLINE)
                except ref.Stuck as e:
                    raise Mismatch(f"{e.args[0]} 个线程 {DEADLINE:g} 秒后仍没有从 should_allow_request 返回：可能死锁了", text,
                                   show=False) from None
                got = Counter(u for u, ok in zip(calls, results) if ok)
                for u in users:
                    if got[u] != min(8, want[u]):
                        raise Mismatch(f"这一轮（t = {ref.rel(now)}）8 个线程同一时刻为 {u!r} 调用 should_allow_request，"
                                       f"按题目的定义应当正好放行 {min(8, want[u])} 次，实际放行了 {got[u]} 次。"
                                       + model.explain(u, now), text, min(8, want[u]), got[u])
                    model.record(u, now, got[u])
