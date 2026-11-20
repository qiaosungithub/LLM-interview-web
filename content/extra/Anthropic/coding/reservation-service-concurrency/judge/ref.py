"""Helpers for tests.py.

- ScanModel: an obviously right single-caller model that scans plain lists.
- run_threads / fast_switching / finishes: running the answer from many threads without ever hanging the judge
  (a deadlock is left to the case's wall-clock limit, which reports TLE).
- SlowInt / SlowStr and the pause controllers: values that behave exactly like the int times and str resource
  names of the statement, except that comparing (SlowInt) or hashing (SlowStr) them can stop the calling thread
  for a moment. The answer compares the new interval's times with the stored bookings while it checks for a
  conflict, so a SlowInt stops a thread right inside that check, i.e. inside its critical section. That turns a
  race that would happen once in a million runs into one that happens every run. With correct locking the
  pauses only cost time (each one gives up after a timeout); they never change what a correct answer returns.
- Jitter / jittering / slow_leg: for the repeated stress tests, the same values pause now and then for a random
  moment (seeded per thread), so many different interleavings get tried instead of the one the scheduler likes.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager

HORIZON = 10 ** 9


# ---------------------------------------------------------------- the model

class ScanModel:
    """One caller at a time; scans lists. Valid inputs only (errors are tested separately)."""

    def __init__(self, resources):
        self.resources = list(resources)
        self.active = {}                       # id -> (owner, legs)
        self.issued = 0

    def reserve_all(self, requests, owner):
        legs = [tuple(leg) for leg in requests]
        taken = [leg for _, booked in self.active.values() for leg in booked]
        for i, (r, s, e) in enumerate(legs):
            if any(r == r2 and s < e2 and s2 < e for r2, s2, e2 in taken + legs[:i]):
                return None
        self.issued += 1
        self.active[self.issued] = (owner, legs)
        return self.issued

    def reserve(self, resource, start, end, owner):
        return self.reserve_all([(resource, start, end)], owner)

    def cancel(self, booking_id):
        return self.active.pop(booking_id, None) is not None

    def query(self, resource, start, end):
        return sorted((s, e, bid, owner) for bid, (owner, legs) in self.active.items()
                      for r, s, e in legs if r == resource and s < end and start < e)


def random_leg(r, names, span=30, longest=6):
    start = r.randrange(span)
    return r.choice(names), start, start + r.randint(1, longest)


# ---------------------------------------------------------------- threads

def run_threads(targets):
    """Start one daemon thread per callable and wait for all of them, with no timeout of our own: a deadlock is
    caught by the case's wall-clock limit (TLE). The first exception raised in any thread is raised again here,
    so an error inside the answer is reported with the answer's own traceback."""
    errors = []

    def wrap(fn):
        def run():
            try:
                fn()
            except BaseException as e:      # noqa: BLE001 - handed to the main thread below
                errors.append(e)
        return run

    threads = [threading.Thread(target=wrap(t), daemon=True) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


@contextmanager
def fast_switching(interval=1e-5):
    """Ask the interpreter to switch threads every `interval` seconds (default 5 ms): many more interleavings."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


class Stuck(Exception):
    pass


def finishes(fn, wait=2.0):
    """fn() run in a helper thread; raises Stuck if it has not returned after `wait` seconds (it waits for a lock
    nobody will release). Single-caller code needs microseconds, so the wait is only a safety margin."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:          # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(wait)
    if t.is_alive():
        raise Stuck()
    if "error" in box:
        raise box["error"]
    return box["value"]


class Ticker:
    """A logical clock shared by the test's threads."""

    def __init__(self):
        self._lock, self._now = threading.Lock(), 0

    def tick(self):
        with self._lock:
            self._now += 1
            return self._now


# ---------------------------------------------------------------- values that pause

class SlowInt(int):
    """An int whose comparisons first call SlowInt.hook(self, other) when a test has set one."""
    hook = None

    def __eq__(self, other):
        _cmp(self, other)
        return int.__eq__(self, other)

    def __ne__(self, other):
        _cmp(self, other)
        return int.__ne__(self, other)

    def __lt__(self, other):
        _cmp(self, other)
        return int.__lt__(self, other)

    def __le__(self, other):
        _cmp(self, other)
        return int.__le__(self, other)

    def __gt__(self, other):
        _cmp(self, other)
        return int.__gt__(self, other)

    def __ge__(self, other):
        _cmp(self, other)
        return int.__ge__(self, other)

    __hash__ = int.__hash__


def _cmp(me, other):
    hook = SlowInt.hook
    if hook is not None:
        hook(me, other)


class SlowStr(str):
    """A str whose hash() (every dict or set lookup with it) first calls SlowStr.hook(self) when one is set."""
    hook = None

    def __hash__(self):
        hook = SlowStr.hook
        if hook is not None:
            hook(self)
        return str.__hash__(self)


@contextmanager
def hooked(cls, hook):
    cls.hook = hook
    try:
        yield hook
    finally:
        cls.hook = None


class _Armed:
    """Only threads that called arm() ever pause (never the main thread, never the answer's own helpers)."""

    def __init__(self):
        self._idents, self._fired, self._lock = set(), set(), threading.Lock()
        self.fired = 0

    def arm(self):
        with self._lock:
            self._idents.add(threading.get_ident())

    def _first_time_here(self):
        me = threading.get_ident()
        with self._lock:
            if me not in self._idents or me in self._fired:
                return False
            self._fired.add(me)
            self.fired += 1
            return True


class MeetInCheck(_Armed):
    """SlowInt hook. Each armed thread, the first time it compares a SlowInt with one of the stored times
    `triggers` (it is checking for a conflict), waits until n threads are waiting together, at most `wait`
    seconds. Without a lock all n get there at once; with one, the gate cannot fill and the first thread
    gives up after `wait` seconds (the gate then stays open for everybody)."""

    def __init__(self, n, triggers, wait):
        super().__init__()
        self.gate, self.triggers, self.wait = threading.Barrier(n), frozenset(triggers), wait

    def __call__(self, me, other):
        if type(other) is not int or other not in self.triggers or not self._first_time_here():
            return
        try:
            self.gate.wait(self.wait)
        except threading.BrokenBarrierError:
            pass

    @property
    def filled(self):
        return not self.gate.broken


class HoldInside(_Armed):
    """SlowInt or SlowStr hook. The first armed thread to get here (a SlowInt compared with one of `triggers`, or
    any hash of a SlowStr) sets `entered` and waits until the test sets `release`, at most `wait` seconds.
    `released_in_time` says whether the release came before the timeout."""

    def __init__(self, wait, triggers=None):
        super().__init__()
        self.triggers = frozenset(triggers) if triggers is not None else None
        self.wait = wait
        self.entered, self.release = threading.Event(), threading.Event()
        self.released_in_time = None

    def __call__(self, me, other=None):
        if self.triggers is not None and (type(other) is not int or other not in self.triggers):
            return
        if self.entered.is_set() or not self._first_time_here():
            return
        self.entered.set()
        self.released_in_time = self.release.wait(self.wait)

    def wait_entered(self, done: threading.Event) -> bool:
        """Wait until the held thread is inside, or until it finished (`done` set) without ever getting here."""
        while not self.entered.wait(0.005):
            if done.is_set():
                return self.entered.is_set()
        return True


class Jitter(_Armed):
    """SlowInt and SlowStr hook for the stress tests: in an armed thread, now and then give up the interpreter for
    a moment (time.sleep(0), or a random sleep of at most `longest` seconds), so that another thread runs right in
    the middle of whatever the caller is doing, typically between its check and its write."""

    def __init__(self, p=0.15, longest=2e-4):
        super().__init__()
        self.p, self.longest, self._local = p, longest, threading.local()

    def arm(self, r):
        """r: this thread's random.Random (from judgelib.rng)."""
        super().arm()
        self._local.r = r

    def __call__(self, me, other=None):
        r = getattr(self._local, "r", None)
        if r is None:
            return
        x = r.random()
        if x < self.p:
            time.sleep(0.0 if x < self.p / 2 else self.longest * x / self.p)


@contextmanager
def jittering(jitter):
    with hooked(SlowInt, jitter), hooked(SlowStr, jitter), fast_switching():
        yield jitter


def slow_leg(leg):
    res, s, e = leg
    return SlowStr(res), SlowInt(s), SlowInt(e)


class Lockstep(_Armed):
    """SlowStr hook. Every hash of a SlowStr in an armed thread waits until all n armed threads have reached
    their next hash, so threads running the same code stay in step: each takes its first lock before any asks
    for its second. Gives up for good after `wait` seconds (a correct answer makes the others wait for a lock)."""

    def __init__(self, n, wait):
        super().__init__()
        self.gate, self.wait = threading.Barrier(n), wait

    def __call__(self, me):
        if self.gate.broken or threading.get_ident() not in self._idents:
            return
        try:
            self.gate.wait(self.wait)
        except threading.BrokenBarrierError:
            pass

    @property
    def held_in_step(self):
        return not self.gate.broken
