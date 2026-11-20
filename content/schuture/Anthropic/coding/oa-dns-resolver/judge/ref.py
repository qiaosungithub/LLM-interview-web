"""Helpers for tests.py.

- A literal model of the statement (Levels 1 to 4), used for every expected value that is not
  hard-coded. Zones are plain data: a list of (key, record) pairs, a record being ("A", addresses, ttl)
  or ("CNAME", target, ttl). The same data is built into the answer's own record classes, or printed as
  code for a replay.
- Instrumented CNAME targets. The tests hand the answer each CNAMERecord.target as a Target, a str
  subclass that reports every use (a method looked up on it, hashing, comparing, slicing, len,
  iteration, +, formatting) to its Touch. A resolver reads a target only while it follows a chain, so a
  resolve call that touched no target did not walk a chain (unless the answer normalised every target
  ahead of time, which the tests detect and then skip these checks). Touch hooks can stop the thread
  that touches a chosen target (Gate) or make it sleep a little (Jitter).
- Random zones and Level 4 operation sequences.
"""
import json
import sys
import threading
import time
from contextlib import contextmanager

NF, CYCLE, LONG = "NameNotFoundError", "ResolutionCycleError", "ChainTooLongError"


def normalize(name):
    """Every ASCII letter to lower case, then the trailing dots (zero or more) replaced by exactly one."""
    name = "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in name)
    while name.endswith("."):
        name = name[:-1]
    return name + "."


# ---------------------------------------------------------------- the model

class Outcome:
    """What one resolve call does: addresses (and the minimum ttl read on the successful attempt), or the
    class name and .name of the error. cnames counts the CNAME records followed over both attempts; hit
    is True when the answer came from the cache."""

    def __init__(self, addresses=None, ttl=None, error=None, name=None, cnames=0, hit=False):
        self.addresses, self.ttl, self.error, self.name = addresses, ttl, error, name
        self.cnames, self.hit = cnames, hit

    @property
    def ok(self):
        return self.error is None


def zone_dict(spec):
    out = {}
    for key, record in spec:            # a later key that normalises to the same name replaces the earlier
        out[normalize(key)] = record
    return out


def walk(zone, name, max_len):
    """Level 2: follow the chain from name inside one zone."""
    origin = normalize(name)
    current, visited, ttls, length = origin, [origin], [], 0
    while True:
        if current not in zone:
            return Outcome(error=NF, name=current, cnames=len(ttls))
        kind, value, ttl = zone[current]
        ttls.append(ttl)
        if kind == "A":
            return Outcome(addresses=list(value), ttl=min(ttls), cnames=len(ttls) - 1)
        target = normalize(value)
        if target in visited:               # first: the step revisits a name
            return Outcome(error=CYCLE, name=target, cnames=len(ttls))
        if length + 1 > max_len:            # then: the step would make the chain too long
            return Outcome(error=LONG, name=origin, cnames=len(ttls))
        length += 1
        visited.append(target)
        current = target


class Model:
    def __init__(self, zone, fallback=None, max_chain_length=8):
        self.zone = zone_dict(zone)
        self.fallback = None if fallback is None else zone_dict(fallback)
        self.max = max_chain_length
        self.now = 0
        self.cache = {}             # normalised query -> (addresses, expiry, clock when stored)
        self.queried, self.failed = set(), set()

    def lookup(self, name):
        """Level 3, without the cache."""
        first = walk(self.zone, name, self.max)
        if first.error != NF or self.fallback is None:
            return first
        second = walk(self.fallback, name, self.max)
        second.cnames += first.cnames
        return second

    def resolve(self, name):
        """Level 4."""
        key = normalize(name)
        self.queried.add(key)
        entry = self.cache.get(key)
        if entry is not None and self.now < entry[1]:
            return Outcome(addresses=list(entry[0]), hit=True)
        out = self.lookup(key)
        if out.ok:
            self.cache[key] = (list(out.addresses), self.now + out.ttl, self.now)
            self.failed.discard(key)
        else:
            self.failed.add(key)
        return out

    def advance_clock(self, seconds):
        self.now += seconds


# ---------------------------------------------------------------- zones as code and as the answer's objects

def lit(x):
    """A Python literal, strings in double quotes as in the statement."""
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(lit(v) for v in x) + "]"
    return repr(x)


def record_code(record):
    kind, value, ttl = record
    return f"{'ARecord' if kind == 'A' else 'CNAMERecord'}({lit(value)}, ttl={ttl})"


def zone_code(spec, var):
    if not spec:
        return [f"{var} = {{}}"]
    return [f"{var} = {{"] + [f"    {lit(k)}: {record_code(rec)}," for k, rec in spec] + ["}"]


def resolver_code(fallback, max_len, var="r"):
    args = ["zone"] + (["fallback"] if fallback is not None else []) + \
        ([f"max_chain_length={max_len}"] if max_len is not None else [])
    return f"{var} = Resolver({', '.join(args)})"


def build(sol, spec, touch=None):
    """The zone as a dict of the answer's own records; CNAME targets are Targets when touch is given."""
    zone = {}
    for key, (kind, value, ttl) in spec:
        if kind == "A":
            zone[key] = sol.ARecord(list(value), ttl=ttl)
        else:
            zone[key] = sol.CNAMERecord(Target(value, touch, key) if touch is not None else value, ttl=ttl)
    return zone


# ---------------------------------------------------------------- instrumented targets

class Touch:
    """Counts the uses of its Targets. local.call, when a thread sets it, tags the touches made during that
    call (calls collects the tags). hook(owner), when set, runs on every touch outside the lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.local = threading.local()
        self.calls = set()
        self.hook = None

    def note(self, owner):
        tag = getattr(self.local, "call", None)
        with self.lock:
            self.count += 1
            if tag is not None:
                self.calls.add(tag)
        hook = self.hook
        if hook is not None:
            hook(owner)


class Target(str):
    """A CNAME target that reports each use to its Touch, naming the normalised zone key it belongs to."""

    def __new__(cls, value, touch, owner):
        self = super().__new__(cls, value)
        self._touch, self._owner = touch, normalize(owner)
        return self

    def __getattribute__(self, attr):
        if attr[:1] != "_":
            _touched(self)
        return str.__getattribute__(self, attr)

    def __repr__(self):
        return str.__repr__(self)

    def __radd__(self, other):
        _touched(self)
        return other + str.__str__(self)


def _touched(target):
    object.__getattribute__(target, "_touch").note(object.__getattribute__(target, "_owner"))


def _counted(name):
    base = getattr(str, name)

    def method(self, *args):
        _touched(self)
        return base(self, *args)
    method.__name__ = name
    return method


for _name in ("__hash__", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__len__", "__getitem__",
              "__iter__", "__contains__", "__add__", "__mul__", "__mod__", "__format__", "__str__"):
    setattr(Target, _name, _counted(_name))


class Gate:
    """Touch hook: a thread that touches a target of the record whose key is `owner` stops there until
    release is set (or `wait` seconds pass). threads holds the idents of the threads that stopped."""

    def __init__(self, owner, wait=6.0):
        self.owner, self.wait = normalize(owner), wait
        self.entered, self.release = threading.Event(), threading.Event()
        self.lock, self.threads = threading.Lock(), set()

    def __call__(self, owner):
        if owner != self.owner or self.release.is_set():
            return
        with self.lock:
            self.threads.add(threading.get_ident())
        self.entered.set()
        self.release.wait(self.wait)


class Jitter:
    """Touch hook: sometimes sleeps up to `most` seconds, so threads interleave in many orders."""

    def __init__(self, r, p=0.3, most=0.002):
        self.r, self.p, self.most = r, p, most

    def __call__(self, owner):
        if self.r.random() < self.p:
            time.sleep(self.r.random() * self.most)


@contextmanager
def fast_switching(interval=1e-5):
    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


def start(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def join_all(threads, wait):
    """True when every thread finished within `wait` seconds in total."""
    deadline = time.monotonic() + wait
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    return not any(t.is_alive() for t in threads)


# ---------------------------------------------------------------- random zones and operations

NAMES = ["www.acme.com.", "edge.acme.com.", "api.acme.com.", "cdn.acme.com.", "db.acme.com.", "mail.acme.com.",
         "shop.acme.com.", "a-1.b2.example.", "例子.测试.", "x.y."]
GHOSTS = ["ghost.acme.com.", "nope.example."]


class Addresses:
    """Hands out addresses never used before in the same run, so a wrong record shows."""

    def __init__(self):
        self.n = 0

    def take(self, r):
        out = []
        for _ in range(r.choice([1, 1, 1, 2, 3])):
            self.n += 1
            out.append(f"10.{self.n // 250}.{self.n % 250}.{r.randint(1, 254)}")
        return out


def spell(r, name):
    """name with random ASCII letters upper-cased and 0 to 3 trailing dots: the same name once normalised."""
    base = name.rstrip(".")
    base = "".join(c.upper() if "a" <= c <= "z" and r.random() < 0.3 else c for c in base)
    return base + r.choice(["", ".", ".", ".", "..", "..."])


def random_zone(r, names, addrs, a_only=False, p_missing=0.15, ttl=(1, 60)):
    """A zone over `names` (in order): a name is absent, an A record, or a CNAME to a later name (a chain),
    to any name (often a cycle) or to a ghost. Keys and targets are spelled at random; sometimes a second
    key spells an existing name again, and whichever comes later wins."""
    spec, used = [], set()

    def record(i):
        if a_only or r.random() < 0.4:
            return ("A", addrs.take(r), r.randint(*ttl))
        p = r.random()
        if p < 0.6 and i + 1 < len(names):
            target = r.choice(names[i + 1:])
        elif p < 0.9:
            target = r.choice(names)
        else:
            target = r.choice(GHOSTS)
        return ("CNAME", spell(r, target), r.randint(*ttl))

    for i, nm in enumerate(names):
        if r.random() < p_missing:
            continue
        key = spell(r, nm)
        used.add(key)
        spec.append((key, record(i)))
    if spec and r.random() < 0.3:
        key = spell(r, r.choice(spec)[0])
        if key not in used:
            i = names.index(normalize(key)) if normalize(key) in names else 0
            spec.insert(r.randint(0, len(spec)), (key, record(i)))
    return spec


def random_ops(r, zone, fallback, max_len, names, n_ops):
    """Level 4: a sequence of ("resolve", query) and ("advance_clock", seconds). The clock moves often to
    the exact expiry of a cached name, one second before or after it."""
    m = Model(zone, fallback, 8 if max_len is None else max_len)
    ops = []
    for _ in range(n_ops):
        if m.cache and r.random() < 0.35:
            left = [exp - m.now for _, exp, _ in m.cache.values() if exp > m.now]
            if left and r.random() < 0.75:
                step = max(0, r.choice(left) + r.choice([0, 0, -1, 1]))
            else:
                step = r.choice([0, 1, 2, 3, 7])
            ops.append(("advance_clock", step))
            m.advance_clock(step)
        else:
            q = spell(r, r.choice(names + GHOSTS[:1]))
            ops.append(("resolve", q))
            m.resolve(q)
    return ops
