"""Helpers for tests.py.

- ManualClock: the statement's manual clock, verbatim.
- Model: an obviously right model of one store, built without bisect (the page's NaiveTimeMap): every set call is
  kept, in call order, in one list per key. nudge=False follows Part 1/2 (a late, earlier timestamp lands in
  place; the later of two calls with the same timestamp wins), nudge=True follows Part 3 (last(key) + 1).
- Part 4: run_threads, fast_switching, and objects that behave exactly like the statement's str keys and float
  timestamps, except that hashing a SlowStr or comparing a SlowFloat can stop the calling thread for a moment.
  A store hashes the key on every dict lookup and compares the new timestamp with last(key) between reading and
  writing it, so these pauses land inside its read-modify-write sequences. The controllers (Lockstep, HoldAt,
  Jitter, HoldingClock) decide who pauses and for how long; only threads that called arm() ever pause. With correct
  locking a pause only costs time (each one gives up after a timeout); it never changes what a correct store returns.
- A thread pauses at most once per bytecode instruction of the store (the first hash or comparison it triggers
  there). A single C-level call such as defaultdict's "missing key: build the value, then store it" hashes the key
  twice; pausing only at the first hash keeps such a call as uninterruptible as it is with a plain str key.
"""
from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager


class ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        self._now += dt   # dt may be negative: this simulates the clock moving backward


class Model:
    """One store, one caller at a time. Every set call is kept in call order; get scans them."""

    def __init__(self, clock=None, nudge=False):
        self.entries = {}           # key -> [(timestamp, value), ...] in call order
        self.clock, self.nudge = clock, nudge

    def _at(self, timestamp):
        return self.clock.now() if timestamp is None else timestamp

    def set(self, key, value, timestamp=None):
        timestamp = self._at(timestamp)
        earlier = [t for t, _ in self.entries.get(key, [])]
        if self.nudge and earlier and timestamp <= max(earlier):
            timestamp = max(earlier) + 1
        self.entries.setdefault(key, []).append((timestamp, value))
        return timestamp

    def get(self, key, timestamp=None):
        timestamp = self._at(timestamp)
        best = None
        for t, v in self.entries.get(key, []):
            if t <= timestamp and (best is None or t >= best[0]):   # >=: the later of two equal timestamps wins
                best = (t, v)
        return None if best is None else best[1]


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


# ---------------------------------------------------------------- pause points

_hook = [None]
_where = threading.local()


def _point():
    """Called by SlowStr.__hash__, SlowFloat's comparisons and the clocks below; frame 2 is the code that hashed,
    compared or read the clock (C functions have no frame, so it is the store's own code)."""
    hook = _hook[0]
    if hook is None:
        return
    f = sys._getframe(2)
    here = (id(f), f.f_lasti)
    if getattr(_where, "here", None) == here:       # a second callback inside the same instruction
        return
    _where.here = here
    hook()


@contextmanager
def hooked(hook):
    _hook[0] = hook
    try:
        yield hook
    finally:
        _hook[0] = None


class SlowStr(str):
    """A str whose hash() (every dict or set lookup with it) is a pause point."""

    def __hash__(self):
        _point()
        return str.__hash__(self)


class SlowFloat(float):
    """A float whose comparisons are pause points."""

    def __eq__(self, other):
        _point()
        return float.__eq__(self, other)

    def __ne__(self, other):
        _point()
        return float.__ne__(self, other)

    def __lt__(self, other):
        _point()
        return float.__lt__(self, other)

    def __le__(self, other):
        _point()
        return float.__le__(self, other)

    def __gt__(self, other):
        _point()
        return float.__gt__(self, other)

    def __ge__(self, other):
        _point()
        return float.__ge__(self, other)

    __hash__ = float.__hash__


class _Armed:
    """Only threads that called arm() ever pause (never the main thread)."""

    def __init__(self):
        self._idents, self._lock = set(), threading.Lock()

    def arm(self):
        _where.here = None
        with self._lock:
            self._idents.add(threading.get_ident())

    def armed(self):
        return threading.get_ident() in self._idents


class Lockstep(_Armed):
    """Every pause point of an armed thread waits until all n armed threads have reached their next one, so
    threads running the same code stay in step: they all read last(key) before any of them writes. With a lock
    around the sequence the others are waiting for that lock, the gate cannot fill, and after `wait` seconds it
    gives up for good."""

    def __init__(self, n, wait):
        super().__init__()
        self.gate, self.wait = threading.Barrier(n), wait
        self.points = 0

    def __call__(self):
        if self.gate.broken or not self.armed():
            return
        self.points += 1
        try:
            self.gate.wait(self.wait)
        except threading.BrokenBarrierError:
            pass

    @property
    def in_step(self):
        return not self.gate.broken


class HoldAt(_Armed):
    """The k-th pause point of the armed thread sets `entered` and waits until the test sets `release`, at most
    `wait` seconds."""

    def __init__(self, k, wait):
        super().__init__()
        self.k, self.wait, self.count = k, wait, 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.released_in_time = None

    def __call__(self):
        if not self.armed():
            return
        self.count += 1
        if self.count == self.k:
            self.entered.set()
            self.released_in_time = self.release.wait(self.wait)

    def wait_entered(self, done):
        """Wait until the held thread is inside, or until it finished (`done` set) without getting there."""
        while not self.entered.wait(0.002):
            if done.is_set():
                return self.entered.is_set()
        return True


class Jitter(_Armed):
    """For the stress tests: at a pause point of an armed thread, now and then give up the interpreter for a moment
    (time.sleep(0), or a random sleep of at most `longest` seconds)."""

    def __init__(self, p=0.15, longest=2e-4):
        super().__init__()
        self.p, self.longest, self._local = p, longest, threading.local()

    def arm(self, r):
        """r: this thread's random.Random (from judgelib.rng)."""
        super().arm()
        self._local.r = r

    def __call__(self):
        r = getattr(self._local, "r", None)
        if r is None:
            return
        x = r.random()
        if x < self.p:
            time.sleep(0.0 if x < self.p / 2 else self.longest * x / self.p)


class TickingClock:
    """now() is a pause point, then returns 1000, 2000, 3000, ...: every reading is larger than the one before, so
    a store that reads it at the instant each set takes effect never needs last(key) + 1."""

    def __init__(self):
        self._ticks = itertools.count(1)        # next() on a counter is atomic, so no two readings are equal

    def now(self) -> float:
        _point()
        return SlowFloat(1000.0 * next(self._ticks))


class HoldingClock(_Armed):
    """Readings 1000, 2000, 3000, ...; the armed thread's first reading is taken, then the thread waits (at most
    `wait` seconds) until the test sets `release` before now() returns it."""

    def __init__(self, wait):
        super().__init__()
        self._ticks = itertools.count(1)
        self.wait = wait
        self.entered, self.release = threading.Event(), threading.Event()
        self.released_in_time = None

    def now(self) -> float:
        reading = 1000.0 * next(self._ticks)
        if self.armed() and not self.entered.is_set():
            self.entered.set()
            self.released_in_time = self.release.wait(self.wait)
        return reading

    def wait_entered(self, done):
        while not self.entered.wait(0.002):
            if done.is_set():
                return self.entered.is_set()
        return True
