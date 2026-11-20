"""Helpers for tests.py.

- Clock: the tests' own clock (the statement's `clock` parameter): a number the test moves by hand.
- Model: the statement's definition read directly. Every admitted request of a user is kept with its time; a
  request satisfies a rule when fewer than max_requests of them have now - t <= period; it is admitted when it
  satisfies all three rules, and only then recorded. No eviction, no per-rule storage.
- Session: one limiter of the answer driven call by call next to the Model. Every call is compared with the Model;
  the replay of the calls so far is kept as code for the result page.
- Crowd: a checkpoint for the thread tests. Callers wait inside it until `need` of them are there at once, which
  proves `need` decisions were made before any of them recorded (a limiter that does not hold a lock around the
  decision and the record), or until every thread of the round has started its call and nobody new has shown up
  for a short while (a limiter that holds a lock around checkpoint: only one caller can ever be inside). After
  that it stays open. The verdict never depends on it: it only decides which interleaving runs.
- run_threads: one should_allow_request call per thread, all started together, with a deadline. With `pace`, each
  thread sleeps that long before every line of the answer's own code (a per-thread sys.settrace; threading and this
  file run at full speed). That is a legal schedule, so a correct limiter only gets slower; but every step of the
  answer becomes a place where the other threads run, so a check and an action on shared state that are not under
  one lock (a per-user lock created on first use with "if user not in locks: locks[user] = Lock()", a decision and
  a record taken in two separate `with lock` blocks) come apart on almost every run.
"""
from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager

from judgelib import Mismatch, calling

T0 = 1_700_000_000.0
RULE_NAMES = ("minute_rule", "hour_rule", "day_rule")
SHOWN_LINES = 14


def num(x) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def rel(t, start=T0) -> str:
    return "+" + num(t - start)


def rules_text(rules) -> str:
    return ", ".join(f"{n}=({m}, {num(p)})" for n, (m, p) in zip(RULE_NAMES, rules))


class Clock:
    """The statement's FakeClock, owned by the tests."""

    def __init__(self, start: float = T0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class Model:
    def __init__(self, rules):
        self.rules = [tuple(r) for r in rules]
        self.longest = max(p for _, p in self.rules)
        self.admitted: dict[str, list[float]] = {}

    def counts(self, user, now):
        times = self.admitted.get(user, [])
        while times and now - times[0] > self.longest:     # only to keep the scan short: the tests' clock never
            times.pop(0)                                   # goes back, so such a request never counts again
        return [sum(1 for t in times if now - t <= p) for _, p in self.rules]

    def decide(self, user, now) -> bool:
        return all(c < m for c, (m, _) in zip(self.counts(user, now), self.rules))

    def room(self, user, now) -> int:
        """How many requests at this instant are admitted, one after another."""
        return max(0, min(m - c for c, (m, _) in zip(self.counts(user, now), self.rules)))

    def record(self, user, now, k=1):
        self.admitted.setdefault(user, []).extend([now] * k)

    def explain(self, user, now, start=T0) -> str:
        parts = [f"{n}=({m}, {num(p)})：{c} 条{'，已满' if c >= m else ''}"
                 for n, (m, p), c in zip(RULE_NAMES, self.rules, self.counts(user, now))]
        times = self.admitted.get(user, [])
        shown = "、".join(rel(t, start) for t in times[-12:])
        if len(times) > 12:
            shown = "…、" + shown
        return (f"按题目的定义，此刻（t = {rel(now, start)}）{user!r} 已经放行的请求里，仍然计入各条规则"
                f"（now - t <= period）的有：" + "；".join(parts) + f"。{user!r} 最近 {num(self.longest)} 秒内被放行"
                f"的请求时刻：{shown or '没有'}（时刻都是相对时钟起点的秒数）。")


class Session:
    """The answer's RateLimiter next to the Model. request() compares every call; the replay is code."""

    def __init__(self, sol, rules, start=T0, checkpoint=None):
        self.rules = [tuple(r) for r in rules]
        self.start = start
        self.clock = Clock(start)
        self.model = Model(self.rules)
        self.calls = 0
        self.lines = ["clock = FakeClock()" if start == T0 else f"clock = FakeClock({num(start)})",
                      f"limiter = RateLimiter({', '.join(repr(r) for r in self.rules)}, clock=clock)"]
        calling("\n".join(self.lines))
        kwargs = {"clock": self.clock}
        if checkpoint is not None:
            kwargs["checkpoint"] = checkpoint
        self.limiter = sol.RateLimiter(*self.rules, **kwargs)

    def replay(self, line) -> str:
        head, body = self.lines[:2], self.lines[2:]
        cut = ""
        if len(body) > SHOWN_LINES:
            cut = f"# …（前面省略了 {len(body) - SHOWN_LINES} 行）"
            body = body[-SHOWN_LINES:]
        return "\n".join(head + ([cut] if cut else []) + body + [f"{line}   # ← 这一步"])

    def advance(self, seconds):
        self.clock.advance(seconds)
        self.lines.append(f"clock.advance({num(seconds)})")

    def request(self, user, hint="") -> bool:
        now = self.clock.t
        want = self.model.decide(user, now)
        line = f"limiter.should_allow_request({user!r})"
        text = calling(self.replay(line))
        self.calls += 1
        got = self.limiter.should_allow_request(user)
        if bool(got) != want:
            verdict = "应当被放行，实际被拒绝了" if want else "应当被拒绝，实际被放行了"
            raise Mismatch(f"这一步（第 {self.calls} 次调用）{verdict}。" + self.model.explain(user, now, self.start)
                           + hint, text, want, got)
        if want:
            self.model.record(user, now)
        self.lines.append(f"{line}   # {'放行' if want else '拒绝'}")
        return want

    def requests(self, user, k, hint=""):
        return [self.request(user, hint) for _ in range(k)]


# ---------------------------------------------------------------- threads

class Switch:
    """A checkpoint the test can swap while the limiter exists."""

    def __init__(self):
        self.target = None

    def __call__(self):
        if self.target is not None:
            self.target()


class Crowd:
    def __init__(self, need, total, quiet=0.1, limit=3.0):
        self.cond = threading.Condition()
        self.need, self.total, self.quiet, self.limit = need, total, quiet, limit
        self.entered = 0
        self.inside = 0
        self.most = 0
        self.last = time.monotonic()
        self.reason = None          # "crowd": `need` callers were inside at once; "alone": never more than one

    def enter(self):
        """Called by a test thread right before it calls should_allow_request."""
        with self.cond:
            self.entered += 1
            self.last = time.monotonic()
            self.cond.notify_all()

    def __call__(self):
        with self.cond:
            if self.reason is not None:
                return
            self.inside += 1
            self.most = max(self.most, self.inside)
            self.last = time.monotonic()
            self.cond.notify_all()
            end = self.last + self.limit
            while self.reason is None:
                now = time.monotonic()
                if self.inside >= self.need:
                    self.reason = "crowd"
                elif self.entered >= self.total and now - self.last >= self.quiet:
                    self.reason = "alone"
                elif now >= end:
                    self.reason = "late"
                else:
                    wake = min(end, self.last + self.quiet) if self.entered >= self.total else end
                    self.cond.wait(max(1e-3, wake - now))
                    continue
                self.cond.notify_all()
            self.inside -= 1


class Jitter:
    """A checkpoint that sometimes yields or naps (at most 0.2 ms); the draws are fixed in advance."""

    def __init__(self, r, n=4096):
        self.naps = [None if x < 0.4 else 0.0 if x < 0.7 else 2e-4 * r.random() for x in (r.random() for _ in range(n))]
        self.next = itertools.count()

    def __call__(self):
        nap = self.naps[next(self.next) % len(self.naps)]
        if nap is not None:
            time.sleep(nap)


class Stuck(Exception):
    pass


def _pacer(home, pause):
    """A sys.settrace function: sleep `pause` before each line of the frames whose globals are `home`."""
    def local(frame, event, arg):
        if event == "line":
            time.sleep(pause)
        return local

    def call(frame, event, arg):
        return local if frame.f_globals is home else None
    return call


def run_threads(limiter, users, gate=None, deadline=8.0, pace=0.0):
    """limiter.should_allow_request(user) once per entry of `users`, each on its own thread, released together.
    Returns the results in the order of `users`. An exception raised by the answer is raised again here; threads
    still inside after `deadline` seconds raise Stuck(number of them). pace: see the module docstring."""
    n = len(users)
    results, errors = [None] * n, []
    go = threading.Event()
    home = getattr(getattr(type(limiter), "should_allow_request", None), "__globals__", None)

    def work(i, user):
        go.wait()
        if gate is not None:
            gate.enter()
        if pace and home is not None:
            sys.settrace(_pacer(home, pace))
        try:
            results[i] = ("ok", limiter.should_allow_request(user))
        except BaseException as e:      # noqa: BLE001 - raised again in the test's thread
            errors.append(e)
        finally:
            sys.settrace(None)

    threads = [threading.Thread(target=work, args=(i, u), daemon=True) for i, u in enumerate(users)]
    for t in threads:
        t.start()
    go.set()
    end = time.monotonic() + deadline
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    if errors:
        raise errors[0]
    stuck = sum(t.is_alive() for t in threads)
    if stuck:
        raise Stuck(stuck)
    return [r[1] for r in results]


@contextmanager
def fast_switching(interval=1e-5):
    """Ask the interpreter to switch threads far more often than every 5 ms: many more interleavings."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        yield
    finally:
        sys.setswitchinterval(old)
