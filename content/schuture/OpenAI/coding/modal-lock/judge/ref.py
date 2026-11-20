"""Helpers for tests.py.

- Rig: like the statement's Rig. Every name is a thread that calls lock.acquire(mode) when it arrives and
  lock.release(mode) when the test says so. A thread notes its grant in a book; a grant made while another mode
  is in the book is a clash, recorded on the spot. An exception raised inside acquire/release is kept and raised
  again in the test's thread, so the user sees their own traceback.
- FairModel / ModalModel: the statement's rules, with their own bookkeeping, never calling the answer.
  FairModel says exactly who holds the lock after every event. ModalModel does too, except when the lock goes
  idle with callers of several modes waiting: then any one mode may win, and all of its waiting callers must
  get the lock together.
- Script: drives a Rig step by step and checks the answer against a model after every step. It waits until the
  expected state is reached; a deadline only guards against a lock that never gets there. A state that can
  never turn into the expected one (someone holds the lock who should not, two modes at once, an exception)
  fails at once.
- GatedCondition: threading.Condition is replaced by this subclass for the whole run. tests.py (and so this file)
  is loaded before the answer, so it covers every Condition the answer makes: in __init__ or later in acquire(),
  bound at import time, or inside threading.Event / Semaphore / queue.Queue. For a thread the test marks with a
  gate, once the test arms the gate, every return from wait() first lets the Condition's lock go and waits for
  the gate to open, as if the thread had been woken but not scheduled yet. That is a legal delay for any
  Condition, so a correct lock is not affected. It forces the case where a woken member of the head batch has
  not taken the lock back yet. The gate is armed only after the thread has parked, so a Condition-based mutex
  the thread contends for on its way in is never held back. An answer that blocks without any Condition (raw
  Lock hand-off) is just not delayed; the test still holds it to the same expectations. GatedCondition also
  notes, for every Rig thread, the waiter entry its last wait() put in a Condition's queue and how many times it
  called wait(): when a ModalLock goes idle, the callers of the losing modes must each be back in a wait() (or
  never notified) before the next step, so a woken caller that goes on to take the lock is seen doing so instead
  of racing the script's next release.
- Slowed threads: a Rig thread started with slow=t runs acquire() under a trace function that sleeps t before
  every line of the answer's own code. Pausing a thread anywhere is a legal schedule, so a correct lock only gets
  slower. A lock that decides in one critical section and acts in another (sees the lock free, lets its mutex go,
  then claims the mode), or that decides to wait and only then starts waiting on something it did not hold while
  deciding (a release in between wakes nobody), is caught in the gap, whatever kind of lock it uses.
- stress: real, uncontrolled threads with short seeded sleeps; checks that two modes never hold together and
  that every thread finishes.
"""
from __future__ import annotations

import sys
import threading
import time

from judgelib import Mismatch, calling

WAIT = 3.0          # how long a step may take before the lock counts as stuck (a correct lock needs microseconds)
GATE_LIMIT = 6.0    # a gated thread goes on by itself after this long, if the test never opened its gate
SHOWN_STEPS = 12    # replay lines shown in a failure
SLOW_LINE = 0.02    # a slowed Rig thread sleeps this long before each line of the answer it runs in acquire()


def fmt(names) -> str:
    return "{" + ", ".join(sorted(names)) + "}"


def fresh(s: str) -> str:
    """The same text as a new string object (not the interned literal)."""
    return "".join(list(s))


# ---------------------------------------------------------------- gated Condition

_tls = threading.local()
_BaseCondition = threading.Condition


class GatedCondition(_BaseCondition):
    def __init__(self, lock=None):
        super().__init__(lock)
        release_save = self._release_save       # the lock's own (RLock) or Condition's

        def noting():
            # wait() calls this right after putting this thread's waiter at the end of the queue
            w = getattr(_tls, "waiter", None)
            if w is not None and not getattr(_tls, "regating", False) and self._waiters:
                w.parked_on = (self, self._waiters[-1])
                w.waits += 1
            return release_save()
        self._release_save = noting

    def wait(self, timeout=None):
        woke = super().wait(timeout)
        gate = getattr(_tls, "gate", None)
        if gate is not None and gate.armed and not gate.is_set() and not getattr(_tls, "regating", False):
            _tls.regating = True        # the gate's own wait is neither noted nor held back
            try:
                saved = self._release_save()
                try:
                    gate.wait(GATE_LIMIT)
                finally:
                    self._acquire_restore(saved)
            finally:
                _tls.regating = False
        return woke


# for the whole run, before the answer is loaded (see the module docstring)
threading.Condition = GatedCondition


class Gate:
    """Holds a Rig thread's returns from wait() back, from arm() until set()."""

    def __init__(self):
        self.armed = False
        self._open = threading.Event()

    def arm(self):
        self.armed = True

    def set(self):
        self._open.set()

    def is_set(self) -> bool:
        return self._open.is_set()

    def wait(self, timeout=None) -> bool:
        return self._open.wait(timeout)


def _pacer(home, pause):
    """A sys.settrace function: sleep `pause` before each line of the frames whose globals are `home` (the
    answer's module). Other code (threading, this file) runs at full speed."""
    def local(frame, event, arg):
        if event == "line":
            time.sleep(pause)
        return local

    def call(frame, event, arg):
        return local if frame.f_globals is home else None
    return call


# ---------------------------------------------------------------- waiting

def wait_until(pred, bad=None, timeout=WAIT) -> bool:
    """True once pred() holds; False if it still does not after `timeout`. bad() is called on every poll and
    raises when the state can no longer turn out right."""
    end = time.monotonic() + timeout
    delay = 0.0
    while True:
        if bad is not None:
            bad()
        if pred():
            if bad is not None:
                bad()
            return True
        if time.monotonic() > end:
            return False
        time.sleep(delay)
        delay = min(0.002, delay * 2 + 0.00005)


# ---------------------------------------------------------------- the rig

class Waiter:
    def __init__(self, name, mode, gate, slow=0.0):
        self.name, self.mode, self.gate, self.slow = name, mode, gate, slow
        self.acquired = threading.Event()
        self.go = threading.Event()
        self.released = threading.Event()
        self.error = None       # (where, exception) raised inside the answer
        self.waits = 0          # wait() calls on any Condition (see GatedCondition)
        self.parked_on = None   # (condition, waiter entry) of the last of them

    def back_in_wait(self, waits_before) -> bool:
        """Since the snapshot this thread has called wait() again, or it is still queued, never notified. A thread
        that never waited on a Condition (raw Lock hand-off) cannot be told apart, and counts as back."""
        if self.waits > waits_before or self.parked_on is None:
            return True
        cond, entry = self.parked_on
        try:
            return entry in cond._waiters
        except RuntimeError:        # the queue changed while we looked: look again on the next poll
            return False


class Rig:
    def __init__(self, lock):
        self.lock = lock
        self.w: dict[str, Waiter] = {}
        self.book = threading.Lock()
        self.held: dict[str, str] = {}      # name -> mode, from grant until just before release()
        self.clash = []                     # (name, mode, {other name: other mode}) at grant time

    def start(self, name, mode, gate=None, slow=0.0) -> Waiter:
        w = Waiter(name, mode, gate, slow)
        self.w[name] = w
        threading.Thread(target=self._run, args=(w,), daemon=True).start()
        return w

    def _run(self, w):
        _tls.gate = w.gate
        _tls.waiter = w
        home = getattr(getattr(type(self.lock), "acquire", None), "__globals__", None)
        if w.slow and home is not None:
            sys.settrace(_pacer(home, w.slow))
        try:
            self.lock.acquire(w.mode)
        except BaseException as e:      # noqa: BLE001 - raised again in the test's thread
            w.error = ("acquire", e)
            return
        finally:
            sys.settrace(None)
        with self.book:
            other = {n: m for n, m in self.held.items() if m != w.mode}
            if other:
                self.clash.append((w.name, w.mode, other))
            self.held[w.name] = w.mode
        w.acquired.set()
        w.go.wait()
        with self.book:
            self.held.pop(w.name, None)
        try:
            self.lock.release(w.mode)
        except BaseException as e:      # noqa: BLE001
            w.error = ("release", e)
            return
        w.released.set()

    def holders(self) -> frozenset:
        return frozenset(n for n, w in self.w.items() if w.acquired.is_set() and not w.released.is_set())

    def blocked(self) -> frozenset:
        return frozenset(n for n, w in self.w.items() if not w.acquired.is_set() and w.error is None)

    def parked(self, w) -> bool:
        # count the other calls still inside acquire() BEFORE reading waiting_count(): that number can only
        # shrink meanwhile, so a waiting_count() above it can only come from w itself (as in the statement)
        others = sum(1 for v in self.w.values() if v is not w and not v.acquired.is_set() and v.error is None)
        return self.lock.waiting_count() > others

    def raise_errors(self):
        for w in self.w.values():
            if w.error is not None:
                raise w.error[1]

    def finish(self):
        """Let every thread go on (after a failure): open the gates, tell every holder to release."""
        for w in self.w.values():
            if w.gate is not None:
                w.gate.set()
            w.go.set()


# ---------------------------------------------------------------- models of the statement's rules

class FairModel:
    def __init__(self):
        self.queue = []         # [mode, set of names that joined and have not released], oldest first

    def arrive(self, name, mode) -> bool:
        if self.queue and self.queue[-1][0] == mode:
            self.queue[-1][1].add(name)
        else:
            self.queue.append([mode, {name}])
        return name in self.holders()

    def release(self, name):
        for _, members in self.queue:
            members.discard(name)
        while self.queue and not self.queue[0][1]:
            self.queue.pop(0)

    def holders(self) -> frozenset:
        return frozenset(self.queue[0][1]) if self.queue else frozenset()

    def blocked(self) -> frozenset:
        return frozenset(n for _, members in self.queue[1:] for n in members)

    def describe(self) -> str:
        if not self.queue:
            return "按题目的规则，此刻批次队列为空、锁空闲"
        parts = [f"{fmt(m)}:{mode!r}" for mode, m in self.queue]
        return "按题目的规则，此刻的批次队列是 " + " → ".join(parts) + "（第一个批次正在持有锁）"

    def why_blocked(self, mode) -> str:
        head_mode = self.queue[0][0]
        if head_mode == mode:
            return (f"锁虽然正以同一模式 {mode!r} 被持有，但已经有别的模式的请求排在后面，"
                    f"新的请求只能排到它们之后")
        return f"锁正以 {head_mode!r} 被 {fmt(self.queue[0][1])} 持有"

    def why_granted(self, name) -> str:
        if len(self.queue[0][1]) == 1:
            return "锁空闲"
        return f"队列末尾的批次正是正在持有锁的 {self.queue[0][0]!r} 批次，它应当直接加入"


class ModalModel:
    def __init__(self):
        self.holding: dict[str, str] = {}
        self.waiting: dict[str, str] = {}

    def mode(self):
        return next(iter(self.holding.values())) if self.holding else None

    def arrive(self, name, mode) -> bool:
        if not self.holding or self.mode() == mode:
            self.holding[name] = mode
            return True
        self.waiting[name] = mode
        return False

    def release(self, name):
        del self.holding[name]

    def holders(self) -> frozenset:
        return frozenset(self.holding)

    def blocked(self) -> frozenset:
        return frozenset(self.waiting)

    def groups(self) -> dict:
        out = {}
        for n, m in self.waiting.items():
            out.setdefault(m, set()).add(n)
        return out

    def grant(self, names):
        for n in names:
            self.holding[n] = self.waiting.pop(n)

    def describe(self) -> str:
        if not self.holding:
            head = "此刻锁空闲"
        else:
            head = f"此刻锁以 {self.mode()!r} 被 {fmt(self.holding)} 持有"
        if self.waiting:
            head += "，在等待的有 " + "、".join(f"{n}:{m!r}" for n, m in self.waiting.items())
        return head

    def why_blocked(self, mode) -> str:
        return f"锁正以 {self.mode()!r} 被 {fmt(self.holding)} 持有"

    def why_granted(self, name) -> str:
        if len(self.holding) == 1:
            return "锁空闲"
        return f"锁正以同一模式 {self.mode()!r} 被持有，它应当直接加入"


# ---------------------------------------------------------------- the script driver

class Script:
    """Arrive/release steps on a fresh lock, each checked against the model. `fresh=True` hands the answer new
    string objects for the modes (equal text, different identity)."""

    def __init__(self, sol, cls_name, fresh_modes=False, header=()):
        self.cls_name = cls_name
        self.fair = cls_name == "FairModalLock"
        self.fresh = fresh_modes
        self.head = list(header)
        if fresh_modes:
            self.head.append("m = lambda s: ''.join(list(s))   # 内容相同、但不是同一个对象的字符串")
        self.head += [f"lock = {cls_name}()", "rig = Rig(lock)   # 题目里的 Rig"]
        self.lines = []
        calling("\n".join(self.head))
        self.lock = getattr(sol, cls_name)()
        self.rig = Rig(self.lock)
        self.model = FairModel() if self.fair else ModalModel()
        self.count = 0

    # -- replay
    def show_mode(self, mode):
        return f"m({mode!r})" if self.fresh else repr(mode)

    def _step(self, line):
        self.lines.append(line)
        shown = self.lines[-SHOWN_STEPS:]
        skipped = len(self.lines) - len(shown)
        text = list(self.head)
        if skipped:
            text.append(f"# ……前面还有 {skipped} 步")
        text += shown[:-1] + [shown[-1] + "   # ← 这一步"]
        return calling("\n".join(text))

    def fail(self, msg):
        self.rig.finish()
        raise Mismatch(msg, show=False)

    # -- checks shared by every wait
    def _bad(self, allowed=None, extra_msg=""):
        def check():
            try:
                self.rig.raise_errors()
            except BaseException:
                self.rig.finish()
                raise
            if self.rig.clash:
                name, mode, other = self.rig.clash[0]
                others = "、".join(f"{n}:{m!r}" for n, m in other.items())
                self.fail(f"{name} 以模式 {mode!r} 获得了锁，而此时 {others} 还持有着它：任意时刻所有持有者"
                          f"用的必须是同一个模式。{self.model.describe()}。{extra_msg}")
            if allowed is not None:
                extra = self.rig.holders() - allowed
                if extra:
                    self.fail(f"{fmt(extra)} 不应当持有锁：此刻应当持有锁的是 {fmt(allowed)}。"
                              f"{self.model.describe()}。{extra_msg}")
        return check

    def _check_waiting(self):
        want = len(self.model.blocked())
        got = []

        def ok():
            got[:] = [self.lock.waiting_count()]
            return got[0] == want
        if not wait_until(ok, self._bad(self.model.holders())):
            self.fail(f"waiting_count() 返回 {got[0]!r}，但此刻阻塞在 acquire() 里的调用有 {want} 个"
                      f"（{fmt(self.model.blocked())}）。{self.model.describe()}。")

    def settle(self, hint=""):
        """Wait until the holders are exactly the model's (FairModalLock, or ModalLock while it is held)."""
        want = self.model.holders()
        if not wait_until(lambda: self.rig.holders() == want, self._bad(want, hint)):
            missing = want - self.rig.holders()
            self.fail(f"{fmt(missing)} 应当已经获得锁，但 {WAIT:g} 秒后仍停在 acquire 里。"
                      f"{self.model.describe()}。{hint}")
        self._check_waiting()

    def _settle_idle(self, released, waits_before=None, hint=""):
        """ModalLock just went idle: one mode among the waiting callers wins and all its callers get in; the
        callers of the other modes go back to waiting (waits_before: Waiter.waits of everyone, before the release)."""
        groups = self.model.groups()
        if not groups:
            self.settle(hint)
            return
        waiting = frozenset(self.model.waiting)
        modes_of = self.rig.w
        waits_before = waits_before or {}

        def done():
            h = self.rig.holders()
            if not h:
                return False
            modes = {modes_of[n].mode for n in h}
            if len(modes) != 1 or h != groups.get(next(iter(modes))):
                return False
            return all(modes_of[n].back_in_wait(waits_before.get(n, 0)) for n in waiting - h)
        if not wait_until(done, self._bad(waiting, hint)):
            h = self.rig.holders()
            if not h:
                self.fail(f"{released} 释放后锁已经空闲，在等待的 {fmt(waiting)} 中应当有一组同模式的调用获得锁，"
                          f"但 {WAIT:g} 秒后谁也没有获得。{hint}")
            mode = modes_of[next(iter(h))].mode
            missing = groups[mode] - h
            if not missing:
                loose = [n for n in waiting - h if not modes_of[n].back_in_wait(waits_before.get(n, 0))]
                self.fail(f"{fmt(h)} 以 {mode!r} 获得了锁；在等其他模式的 {fmt(loose)} 被唤醒后，{WAIT:g} 秒内既没有"
                          f"获得锁，也没有回到 wait() 里继续等待。{hint}")
            self.fail(f"{fmt(h)} 以 {mode!r} 获得了锁，但同样在等 {mode!r} 的 {fmt(missing)} {WAIT:g} 秒后仍停在 "
                      f"acquire 里：锁的模式已经是 {mode!r}，它们不应当继续阻塞。{hint}")
        h = self.rig.holders()
        self.model.grant(h)
        self._check_waiting()

    # -- steps
    def arrive(self, name, mode, gate=None, note="", slow=0.0):
        self._step(f"rig.arrive({name!r}, {self.show_mode(mode)})" + (f"   # {note}" if note else ""))
        w = self.rig.start(name, fresh(mode) if self.fresh else mode, gate, slow)
        granted = self.model.arrive(name, mode)
        if granted:
            why = self.model.why_granted(name)
            if not wait_until(w.acquired.is_set, self._bad(self.model.holders())):
                self.fail(f"{name} 请求 {mode!r} 时{why}，acquire 应当立即返回，但 {WAIT:g} 秒后它仍停在 "
                          f"acquire 里。{self.model.describe()}。")
        else:
            why = self.model.why_blocked(mode)

            def bad():
                self._bad()()
                if w.acquired.is_set():
                    self.fail(f"{name} 请求 {mode!r} 时{why}，acquire 应当阻塞，但它立即返回了。"
                              f"{self.model.describe()}。")
            if not wait_until(lambda: self.rig.parked(w), bad):
                self.fail(f"{name} 请求 {mode!r} 后，{WAIT:g} 秒内 acquire 既没有返回，waiting_count() 也没有把它"
                          f"计为阻塞中的调用（waiting_count() 返回 {self.lock.waiting_count()!r}）。")
        self.settle()
        self.count += 1

    def release(self, name, settle=True, hint=""):
        w = self.rig.w[name]
        self._step(f"rig.release({name!r})")
        waits_before = {n: v.waits for n, v in self.rig.w.items()}
        w.go.set()
        if not wait_until(w.released.is_set, self._bad()):
            self.fail(f"{name} 调用的 release({w.mode!r}) {WAIT:g} 秒后还没有返回。")
        self.model.release(name)
        self.count += 1
        if not settle:
            return
        if not self.fair and not self.model.holding:
            self._settle_idle(name, waits_before, hint)
        else:
            self.settle(hint)

    def end(self):
        """Every thread has released (the script released them all): nothing went wrong on the way out."""
        everyone = list(self.rig.w.values())
        if not wait_until(lambda: all(w.released.is_set() for w in everyone), self._bad()):
            stuck = [w.name for w in everyone if not w.released.is_set()]
            self.fail(f"脚本最后所有人都应当已经释放，但 {fmt(stuck)} {WAIT:g} 秒后还没有走完 acquire/release。")

    def holders(self):
        return self.rig.holders()

    def bad(self, allowed=None, extra_msg=""):
        return self._bad(allowed, extra_msg)

    def call(self, text, fn, wait=WAIT):
        """fn() in a helper thread, with a deadline; for the direct release() calls of the error tests."""
        self._step(text)
        box, done = {}, threading.Event()

        def run():
            try:
                box["value"] = fn()
            except BaseException as e:      # noqa: BLE001
                box["error"] = e
            finally:
                done.set()
        threading.Thread(target=run, daemon=True).start()
        if not done.wait(wait):
            self.fail(f"{text} {wait:g} 秒后还没有返回。")
        if "error" in box:
            raise box["error"]
        return box.get("value")


# ---------------------------------------------------------------- random scripted runs

def random_trial(script: Script, r, modes, n_names=16, n_events=40, cover=None):
    """The statement's randomized check: arrivals and releases chosen at random, each checked by the Script."""
    names = [f"t{i}" for i in range(n_names)]
    r.shuffle(names)
    nxt = 0

    def step():
        nonlocal nxt
        holding = script.holders()
        if nxt < len(names) and (not holding or r.random() < 0.55):
            name, mode = names[nxt], r.choice(modes)
            nxt += 1
            before = script.model.holders()
            same = bool(before) and script.rig.w[min(before)].mode == mode
            script.arrive(name, mode)
            if cover is not None and same:
                cover["join" if name in script.model.holders() else "queued"] += 1
            return True
        if holding:
            before = script.model.holders()
            script.release(r.choice(sorted(holding)))
            if cover is not None and len(script.model.holders() - before) >= 2:
                cover["together"] += 1
            return True
        return False

    for _ in range(n_events):
        step()
    while step():
        pass


# ---------------------------------------------------------------- real concurrency

def stress(lock, plans, deadline):
    """plans[i]: the list of (mode, sleep) rounds of thread i. Returns (clashes, errors, unfinished)."""
    held, book = {}, threading.Lock()
    clashes, errors = [], []
    done = [False] * len(plans)

    def worker(i):
        try:
            for mode, nap in plans[i]:
                lock.acquire(mode)
                with book:
                    if any(m != mode for m in held.values()) and len(clashes) < 5:
                        clashes.append(dict(held, **{f"线程{i}": mode}))
                    held[f"线程{i}"] = mode
                time.sleep(nap)
                with book:
                    del held[f"线程{i}"]
                lock.release(mode)
            done[i] = True
        except BaseException as e:      # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(plans))]
    for t in threads:
        t.start()
    end = time.monotonic() + deadline
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    return clashes, errors, [i for i, d in enumerate(done) if not d]
