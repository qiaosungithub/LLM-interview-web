"""Helpers for tests.py.

- Model: an obviously right LRU cache over the statement's notion of "the same call" (positional values by
  position, keyword values by name). Func: the function being cached; it records every call, so a hit (no call)
  and a miss (one call) can be told apart without looking inside the answer.
- Dirs: one fresh temporary directory per store, removed at the end of the case. snapshot()/Dirs.restore(): a
  process crash is simulated by copying the files exactly as the operating system holds them at that moment
  (what an answer has not flushed out of its own buffers is lost, as in a real crash) into a fresh directory and
  recovering from the copy, so one run can be "crashed" at many points.
- Fs: while active, os.fsync / os.fdatasync / os.sync record what they made durable (and skip the real flush to
  disk, which matters for a power cut, not for the process crashes simulated here), and every file-system step
  (open that creates or truncates, rename, replace, remove, truncate, fsync) can call a hook: a crash test takes a
  snapshot there, a concurrency test pauses there.
- Threads: Caller runs one call in a daemon thread; wait_parked tells "this thread is blocked" from its CPU clock;
  HeldFunc is a func whose calls can be held until the test releases them. Every wait has a timeout.
"""
from __future__ import annotations

import builtins
import fcntl
import io
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager

LOG = "cache.log"           # the store's base path inside each case's directory, as in the statement's example

_open = builtins.open       # the real functions, for the test's own file handling
_ORIG = {name: getattr(os, name) for name in
         ("fsync", "fdatasync", "sync", "replace", "rename", "remove", "unlink", "truncate", "ftruncate", "open")}


# ---------------------------------------------------------------- calls and the model

def value_of(args, kwargs):
    """What the test's func returns: the call itself as a string (JSON-serialisable; equal calls, equal values)."""
    return repr((tuple(args), sorted(kwargs.items())))


def other_value(args, kwargs):
    """A second function, for recovered caches: a hit must return the stored value, never a recomputed one."""
    return "g" + value_of(args, kwargs)


def canon(args, kwargs):
    return tuple(args), tuple(sorted(kwargs.items()))


def show(name, args, kwargs):
    parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    return f"{name}({', '.join(parts)})"


def jsonish(v):
    """v as it looks after a JSON round trip: tuples (at any depth) become lists."""
    if isinstance(v, (list, tuple)):
        return [jsonish(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonish(x) for k, x in v.items()}
    return v


class Model:
    def __init__(self, capacity):
        self.capacity = capacity
        self.od = OrderedDict()             # canon -> (args, kwargs, value), LRU first
        self.hits = self.misses = 0

    def copy(self):
        m = Model(self.capacity)
        m.od = OrderedDict(self.od)
        return m

    def call(self, args, kwargs, value=value_of):
        """Returns (hit, value)."""
        k = canon(args, kwargs)
        if k in self.od:
            self.od.move_to_end(k)
            self.hits += 1
            return True, self.od[k][2]
        self.misses += 1
        v = value(args, kwargs)
        self.od[k] = (tuple(args), dict(kwargs), v)
        if len(self.od) > self.capacity:
            self.od.popitem(last=False)
        return False, v

    def entries(self):
        """[(args, kwargs, value)], least recently used first."""
        return list(self.od.values())


class Func:
    """The function being cached: records every call and returns value(args, kwargs)."""

    def __init__(self, value=value_of):
        self.value = value
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls.append((args, kwargs))
        return self.value(args, kwargs)


# ---------------------------------------------------------------- directories and crash snapshots

def snapshot(d):
    """Every regular file under d (relative path -> bytes) and every directory, as the OS holds them now."""
    files, dirs = {}, []
    for base, subdirs, names in os.walk(d):
        for s in subdirs:
            dirs.append(os.path.relpath(os.path.join(base, s), d))
        for n in names:
            p = os.path.join(base, n)
            if os.path.isfile(p) and not os.path.islink(p):
                try:
                    with _open(p, "rb") as f:
                        files[os.path.relpath(p, d)] = f.read()
                except OSError:
                    pass
    return {"files": files, "dirs": sorted(dirs)}


def snap_key(snap):
    return tuple(sorted(snap["files"].items())), tuple(snap["dirs"])


def sizes(d):
    return {rel: st[2] for rel, st in stats(d).items()}


def stats(d):
    """relative path -> (device, inode, size) of every regular file under d."""
    out = {}
    for base, _subdirs, names in os.walk(d):
        for n in names:
            p = os.path.join(base, n)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if os.path.isfile(p) and not os.path.islink(p):
                out[os.path.relpath(p, d)] = (st.st_dev, st.st_ino, st.st_size)
    return out


def with_file(snap, rel, data):
    """A copy of snap in which file rel holds data."""
    files = dict(snap["files"])
    files[rel] = data
    return {"files": files, "dirs": list(snap["dirs"])}


class Dirs:
    """Fresh temporary directories for one case; all removed when the case ends."""

    def __init__(self):
        self.made = []

    def new(self):
        d = tempfile.mkdtemp(prefix="lru_")
        self.made.append(d)
        return d

    def restore(self, snap):
        d = self.new()
        for rel in snap["dirs"]:
            os.makedirs(os.path.join(d, rel), exist_ok=True)
        for rel, data in snap["files"].items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with _open(p, "wb") as f:
                f.write(data)
        return d

    def drop(self, d):
        shutil.rmtree(d, ignore_errors=True)
        if d in self.made:
            self.made.remove(d)

    def close(self):
        for d in self.made:
            shutil.rmtree(d, ignore_errors=True)
        self.made = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- fsync records and file-system steps

def _sync_flagged():
    """(dev, ino) of every file this process has open with O_SYNC / O_DSYNC (each write durable on its own)."""
    out = set()
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return out
    for name in names:
        try:
            fd = int(name)
            if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_DSYNC:
                st = os.fstat(fd)
                out.add((st.st_dev, st.st_ino))
        except (OSError, ValueError):
            pass
    return out


class Fs:
    """See the module docstring. Use as `with Fs(sol, dirs) as fs:`; fs.steps(hook) arms the step hook for the
    current thread (or for every thread with any_thread=True)."""

    def __init__(self, sol, dirs):
        self.sol, self.dirs = sol, dirs
        self.lock = threading.Lock()
        self.synced = {}            # (dev, ino) -> file size when last fsynced
        self.syncs = 0
        self.hook, self.hook_thread = None, None
        self.fsync_hook = None      # called as fsync_hook(fd) before an fsync is recorded, in any thread
        self._local = threading.local()
        self._saved = []

    # ---- recording
    def _record(self, fd):
        st = os.fstat(fd)
        with self.lock:
            self.synced[(st.st_dev, st.st_ino)] = st.st_size
            self.syncs += 1

    def fsync(self, fd):
        if hasattr(fd, "fileno"):
            fd = fd.fileno()
        hook = self.fsync_hook
        if hook is not None and not self._busy():
            self._enter()
            try:
                hook(fd)
            finally:
                self._leave()
        self._step("fsync")
        self._record(fd)

    def sync(self):
        for d in list(self.dirs.made):
            for base, _s, names in os.walk(d):
                for n in names:
                    p = os.path.join(base, n)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    with self.lock:
                        self.synced[(st.st_dev, st.st_ino)] = st.st_size
        with self.lock:
            self.syncs += 1

    def unsynced(self, d, rels):
        """Which of the files `rels` under d hold bytes that no fsync has covered yet."""
        flagged = None
        bad = []
        for rel in rels:
            p = os.path.join(d, rel)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size == 0 or self.synced.get((st.st_dev, st.st_ino)) == st.st_size:
                continue
            if flagged is None:
                flagged = _sync_flagged()
            if (st.st_dev, st.st_ino) not in flagged:
                bad.append(rel)
        return bad

    # ---- steps
    def _busy(self):
        return getattr(self._local, "busy", False)

    def _enter(self):
        self._local.busy = True

    def _leave(self):
        self._local.busy = False

    def _step(self, label):
        hook = self.hook
        if hook is None or self._busy():
            return
        if self.hook_thread is not None and threading.get_ident() != self.hook_thread:
            return
        self._enter()
        try:
            hook(label)
        finally:
            self._leave()

    @contextmanager
    def steps(self, hook):
        """Arms hook(label) for the file-system steps of the calling thread only."""
        self.hook, self.hook_thread = hook, threading.get_ident()
        try:
            yield
        finally:
            self.hook, self.hook_thread = None, None

    # ---- the replacements
    def _wrap_around(self, name):
        real = _ORIG[name]

        def wrapped(*args, **kwargs):
            self._step(f"before {name}")
            out = real(*args, **kwargs)
            self._step(f"after {name}")
            return out
        return wrapped

    def _os_open(self, path, flags, *args, **kwargs):
        fresh = bool(flags & os.O_TRUNC) or (bool(flags & os.O_CREAT) and not os.path.exists(path))
        fd = _ORIG["open"](path, flags, *args, **kwargs)
        if fresh:
            self._step("after os.open")
        return fd

    def _open(self, file, mode="r", *args, **kwargs):
        fresh = isinstance(file, (str, bytes, os.PathLike)) and (
            "w" in mode or "x" in mode or ("a" in mode and not os.path.exists(file)))
        f = _open(file, mode, *args, **kwargs)
        if fresh:
            self._step("after open")
        return f

    def __enter__(self):
        repl = {
            "fsync": self.fsync, "fdatasync": self.fsync, "sync": self.sync,
            "replace": self._wrap_around("replace"), "rename": self._wrap_around("rename"),
            "remove": self._wrap_around("remove"), "unlink": self._wrap_around("unlink"),
            "truncate": self._wrap_around("truncate"), "ftruncate": self._wrap_around("ftruncate"),
            "open": self._os_open,
        }
        for name, fn in repl.items():
            self._saved.append((os, name, getattr(os, name)))
            setattr(os, name, fn)
        for mod in (builtins, io):
            self._saved.append((mod, "open", mod.open))
            mod.open = self._open
        # names the answer imported directly (from os import fsync)
        by_id = {id(_ORIG[n]): repl[n] for n in repl}
        by_id[id(_open)] = self._open
        for name, value in list(vars(self.sol).items()):
            if id(value) in by_id and (value is _open or any(value is _ORIG[n] for n in _ORIG)):
                self._saved.append((self.sol, name, value))
                setattr(self.sol, name, by_id[id(value)])
        return self

    def __exit__(self, *exc):
        for mod, name, value in reversed(self._saved):
            setattr(mod, name, value)
        self._saved = []
        self.hook = self.fsync_hook = None


# ---------------------------------------------------------------- threads

def cpu_of(thread):
    try:
        return time.clock_gettime(time.pthread_getcpuclockid(thread.ident))
    except (OSError, AttributeError, TypeError, OverflowError):
        return None


def in_answer(thread, sol):
    """Whether the thread is currently running (or blocked) inside a function of the answer's module."""
    frame = sys._current_frames().get(thread.ident)
    g = vars(sol)
    while frame is not None:
        if frame.f_globals is g:
            return True
        frame = frame.f_back
    return False


def wait_parked(callers, sol, still=0.1, longest=1.5):
    """Wait until every Caller has returned ("done") or is "parked": inside the answer's code and using no CPU
    for `still` seconds (blocked on a lock, an event or a sleep); give up after `longest` seconds ("busy").
    Returns the states. Only decides when the test moves on; no verdict depends on how long the scheduler took."""
    if isinstance(callers, Caller):
        return wait_parked([callers], sol, still, longest)[0]
    end = time.monotonic() + longest
    now = time.monotonic()
    last = [cpu_of(c) for c in callers]
    since = [now] * len(callers)
    state = [None] * len(callers)
    while True:
        for i, c in enumerate(callers):
            if c.done.is_set():
                state[i] = "done"
                continue
            cpu = cpu_of(c)
            if cpu != last[i] or not in_answer(c, sol):
                last[i], since[i], state[i] = cpu, time.monotonic(), None
            elif time.monotonic() - since[i] >= still:
                state[i] = "parked"
        if all(s is not None for s in state) or time.monotonic() >= end:
            return [s or "busy" for s in state]
        time.sleep(0.005)


class Caller(threading.Thread):
    """One call of the answer in its own daemon thread; `after(result)` runs in the same thread right after."""

    def __init__(self, fn, args=(), kwargs=None, after=None):
        super().__init__(daemon=True)
        self.fn, self.args, self.kwargs, self.after = fn, args, kwargs or {}, after
        self.done = threading.Event()
        self.result = self.error = self.extra = None

    def run(self):
        try:
            self.result = self.fn(*self.args, **self.kwargs)
            if self.after is not None:
                self.extra = self.after(self.result)
        except BaseException as e:      # noqa: BLE001 - handed to the test
            self.error = e
        finally:
            self.done.set()

    def go(self):
        self.start()
        return self


class HeldFunc:
    """func for Part 3. It takes one argument (positional or keyword), the key. A call for a key in `held` blocks
    until release() (at most `wait` seconds); a key in `fail_first` makes its first call raise
    RuntimeError(f"boom-{key}"). Counts calls per key and how many run at once."""

    def __init__(self, held=(), fail_first=(), wait=6.0, value=None, jitter=None):
        self.lock = threading.Lock()
        self.callers = []               # (key, thread ident) per call
        self.calls, self.active, self.max_active = Counter(), Counter(), Counter()
        self.entered = {k: threading.Event() for k in held}
        self.released = threading.Event()
        self.fail_first = set(fail_first)
        self.wait = wait
        self.value = value or (lambda k: {"key": k, "v": f"value-{k}"})
        self.jitter = jitter            # None, or a function called inside func (random short sleeps)

    def __call__(self, *args, **kwargs):
        key = args[0] if args else next(iter(kwargs.values()))
        with self.lock:
            self.callers.append((key, threading.get_ident()))
            self.calls[key] += 1
            self.active[key] += 1
            self.max_active[key] = max(self.max_active[key], self.active[key])
            first = self.calls[key] == 1
        try:
            if key in self.entered:
                self.entered[key].set()
                self.released.wait(self.wait)
            if self.jitter is not None:
                self.jitter()
            if first and key in self.fail_first:
                raise RuntimeError(f"boom-{key}")
            return self.value(key)
        finally:
            with self.lock:
                self.active[key] -= 1

    def release(self):
        self.released.set()


@contextmanager
def fast_switching(interval=1e-5):
    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        yield
    finally:
        sys.setswitchinterval(old)
