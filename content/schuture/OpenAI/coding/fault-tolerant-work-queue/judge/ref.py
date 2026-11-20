"""The statement's clock, a model of the queue read straight from the statement, and generators of call sequences.

A sequence is a list of (method, args, name). `name` is "r7" for a reserve call (the variable its Reservation is
bound to in the replay), else None. A token argument is a Tok: the token of an earlier reservation, or a forged
one. "advance" is clock.advance, not a queue call."""
import heapq
from bisect import bisect
from collections import deque
from itertools import accumulate


class ManualClock:
    """Exactly the statement's clock."""

    def __init__(self, start: int = 0):
        self._now = start

    def now(self) -> int:
        return self._now

    def advance(self, dt: int) -> None:
        if dt < 0:
            raise ValueError("a clock cannot move backward")
        self._now += dt


class Tok:
    """A token argument: the token of reservation `name` ("r3"), or a forged token when name is None."""
    __slots__ = ("name",)

    def __init__(self, name=None):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Tok) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


OK = ("ok", None)


class QueueModel:
    """The statement, literally. lease_duration None: leases never expire (Part 1). max_attempts None: no cap (Parts
    1-2). Outcomes: ("ok", None), ("err", exception name), ("val", value); reserve's value is (task_id, payload,
    token) or None. fast=False finds expired leases by sorting every reserved task (used for short sequences);
    fast=True keeps a heap of leases instead, for the long ones."""

    def __init__(self, clock, lease_duration=None, max_attempts=None, fast=False):
        self.clock, self.lease, self.cap, self.fast = clock, lease_duration, max_attempts, fast
        self.leases = []                # fast: (deadline, task_id, token) of every reservation issued
        self.status, self.payload, self.attempts, self.token, self.deadline = {}, {}, {}, {}, {}
        self.ready = deque()            # ready task_ids, in the order they entered `ready`
        self.reserved = set()
        self.dead = []                  # dead task_ids, in the order they became dead
        self.issued = 0

    def _reclaim(self):
        """Ends every expired reservation (deadline < now), by (deadline, task_id)."""
        if self.lease is None:
            return
        now = self.clock.now()
        if not self.fast:
            for _, task_id in sorted((self.deadline[t], t) for t in self.reserved if self.deadline[t] < now):
                self._leave(task_id)
            return
        while self.leases and self.leases[0][0] < now:
            _, task_id, token = heapq.heappop(self.leases)
            if self.status[task_id] == "reserved" and self.token[task_id] == token:   # else that reservation ended
                self._leave(task_id)

    def _leave(self, task_id):
        """A reserved task stops being reserved without completing (fail or expiry)."""
        self.reserved.discard(task_id)
        self.token[task_id] = self.deadline[task_id] = None
        if self.cap is not None and self.attempts[task_id] >= self.cap:
            self.status[task_id] = "dead"
            self.dead.append(task_id)
        else:
            self.status[task_id] = "ready"
            self.ready.append(task_id)

    def submit(self, task_id, payload):
        self._reclaim()
        if task_id in self.status:
            return ("err", "ValueError")
        self.status[task_id], self.payload[task_id], self.attempts[task_id] = "ready", payload, 0
        self.token[task_id] = self.deadline[task_id] = None
        self.ready.append(task_id)
        return OK

    def reserve(self, worker_id):
        self._reclaim()
        if not self.ready:
            return ("val", None)
        task_id = self.ready.popleft()
        self.issued += 1
        self.status[task_id], self.token[task_id] = "reserved", self.issued
        self.attempts[task_id] += 1
        if self.lease is not None:
            self.deadline[task_id] = self.clock.now() + self.lease
            if self.fast:
                heapq.heappush(self.leases, (self.deadline[task_id], task_id, self.issued))
        self.reserved.add(task_id)
        return ("val", (task_id, self.payload[task_id], self.issued))

    def _check(self, task_id, token):
        self._reclaim()
        if task_id not in self.status:
            return ("err", "UnknownTaskError")
        if self.status[task_id] != "reserved" or self.token[task_id] != token:
            return ("err", "InvalidReservationError")
        return None

    def complete(self, task_id, token):
        bad = self._check(task_id, token)
        if bad:
            return bad
        self.reserved.discard(task_id)
        self.status[task_id], self.token[task_id], self.deadline[task_id] = "completed", None, None
        return OK

    def fail(self, task_id, token):
        bad = self._check(task_id, token)
        if bad:
            return bad
        self._leave(task_id)
        return OK

    def dead_letters(self):
        self._reclaim()
        return ("val", list(self.dead))

    def requeue_dead(self, task_id):
        self._reclaim()
        if task_id not in self.status:
            return ("err", "UnknownTaskError")
        if self.status[task_id] != "dead":
            return ("err", "ValueError")
        self.dead.remove(task_id)
        self.status[task_id], self.attempts[task_id] = "ready", 0
        self.ready.append(task_id)
        return OK


# ---------------------------------------------------------------- random call sequences

def payload_for(r, n):
    """Payloads of several kinds, unhashable ones included: the queue must hand back whatever it was given."""
    return r.choice([n, f"p{n}", [n, "x"], {"n": n}, None, (n, "y")])


def gen_ops(r, steps, part, lease=None, cap=None, mix=None, jumps=None, fast=False, wants=None):
    """A valid sequence for `part`, built while running a model so that most calls make sense: tokens mostly from
    recent reservations, sometimes older (stale) ones, another task's, or forged ones; now and then a duplicate
    submit, an unknown task_id, or requeue_dead on a task that is not dead. Task ids are drawn so that string order
    differs from submission order ("t12" < "t3"). `jumps`: the clock steps to pick from. With fast=True no step
    scans every task, so long sequences stay cheap. A list passed as `wants` receives the model's outcome of each
    call (None for an advance)."""
    clock = ManualClock()
    model = QueueModel(clock, lease if part >= 2 else None, cap if part >= 3 else None, fast)
    mix = mix or {"submit": 22, "reserve": 25, "token": 35, "advance": 10 if part >= 2 else 0,
                  "dead_letters": 4 if part >= 3 else 0, "requeue_dead": 4 if part >= 3 else 0}
    kinds, cum = list(mix), list(accumulate(mix.values()))
    used = set()
    ops, submitted, issued = [], [], []                   # issued: reservation names, in order
    owner, mtok = {}, {}                                  # reservation name -> its task_id / the model's token
    for step in range(steps):
        kind = kinds[bisect(cum, r.random() * cum[-1])] if submitted else "submit"
        if kind == "token" and not issued:
            kind = "reserve"
        if kind == "submit":
            if submitted and r.random() < 0.08:
                tid = r.choice(submitted)
            else:
                n = r.randrange(1, 10 * steps + 10)
                while n in used:
                    n = r.randrange(1, 10 * steps + 10)
                used.add(n)
                tid = f"t{n}"
            op = ("submit", (tid, payload_for(r, step)), None)
        elif kind == "reserve":
            op = ("reserve", (f"w{r.randrange(3)}",), f"r{len(issued) + 1}")
        elif kind == "token":
            x = r.random()
            name = (r.choice(issued[-3:]) if x < 0.6 else r.choice(issued) if x < 0.9 else None)
            y = r.random()
            tid = (owner[name] if name and y < 0.85 else r.choice(submitted) if y < 0.95 else f"ghost{step}")
            op = (r.choice(["complete", "complete", "fail"]), (tid, Tok(name)), None)
        elif kind == "advance":
            op = ("advance", (r.choice(jumps or [0, 1, 1, 2, max(1, lease - 1), lease, lease + 1, 3 * lease]),), None)
        elif kind == "dead_letters":
            op = ("dead_letters", (), None)
        else:
            y = r.random()
            tid = (r.choice(model.dead) if model.dead and y < 0.75 else r.choice(submitted) if y < 0.93
                   else f"ghost{step}")
            op = ("requeue_dead", (tid,), None)
        method, args, name = op
        out = None
        if method == "advance":
            clock.advance(args[0])
        elif method in ("complete", "fail"):
            tok = args[1]
            out = getattr(model, method)(args[0], mtok[tok.name] if tok.name is not None else object())
        else:
            out = getattr(model, method)(*args)
            if method == "submit" and out == OK:
                submitted.append(args[0])
            elif method == "reserve" and out[1] is None:
                op = (method, args, None)
            elif method == "reserve":
                issued.append(name)
                owner[name], mtok[name] = out[1][0], out[1][2]
        ops.append(op)
        if wants is not None:
            wants.append(out)
    return ops


def lease_wave_ops(r, n, lease, wants):
    """Part 2: n tasks are reserved about one clock tick apart, so thousands of leases are live at once; then the
    clock keeps ticking past their deadlines one by one while every reclaimed task is reserved again, so almost every
    call's reclaiming pass ends a lease or two among thousands of live ones. Some reservations are completed or failed
    on the way (a few with tokens that have just expired). Fills `wants` like gen_ops."""
    clock = ManualClock()
    model = QueueModel(clock, lease, None, fast=True)
    ops, mtok, held = [], {}, []                         # held: (reservation name, task_id), in order

    def do(method, args, name=None):
        out = None
        if method == "advance":
            clock.advance(args[0])
        elif method in ("complete", "fail"):
            out = getattr(model, method)(args[0], mtok[args[1].name])
        else:
            out = getattr(model, method)(*args)
            if method == "reserve" and out[1] is None:
                name = None
            elif method == "reserve":
                mtok[name] = out[1][2]
                held.append((name, out[1][0]))
        ops.append((method, args, name))
        wants.append(out)

    for k in r.sample(range(1, 10 * n), n):
        do("submit", (f"t{k}", payload_for(r, k)))
    for step in range(2 * n):
        if r.random() < 0.9:
            do("advance", (1,))
        if held and r.random() < 0.08:
            name, task_id = held[r.randrange(max(0, len(held) - lease * 6 // 5), len(held))]
            do(r.choice(["complete", "fail"]), (task_id, Tok(name)))
        else:
            do("reserve", (f"w{step % 3}",), f"r{len(held) + 1}")
    return ops
