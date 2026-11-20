# expect: WA
# Part 2 runs the reclaiming pass after each call's own work instead of before it, so reserve misses a task whose lease has just expired.
import heapq
from collections import deque
from typing import Any, NamedTuple


class UnknownTaskError(Exception):
    pass


class InvalidReservationError(Exception):
    pass


class Reservation(NamedTuple):
    task_id: str
    token: object
    payload: Any


class ManualClock:
    def __init__(self, start: int = 0):
        self._now = start

    def now(self) -> int:
        return self._now

    def advance(self, dt: int) -> None:
        if dt < 0:
            raise ValueError("a clock cannot move backward")
        self._now += dt


class _Task:
    __slots__ = ("payload", "state", "serial", "tries")

    def __init__(self, payload):
        self.payload, self.state, self.serial, self.tries = payload, "ready", None, 0


class WorkQueue:
    def __init__(self, clock, lease_duration, max_attempts):
        self.clock, self.lease, self.cap = clock, lease_duration, max_attempts
        self.tasks = {}
        self.ready = deque()
        self.dead = {}                  # insertion-ordered set of dead task ids
        self.leases = []                # (deadline, task_id, serial)
        self.serial = 0

    def _sweep(self):
        now = self.clock.now()
        h = self.leases
        while h and h[0][0] < now:
            _, tid, s = heapq.heappop(h)
            t = self.tasks[tid]
            if t.state == "reserved" and t.serial == s:
                self._release(tid, t)

    def _release(self, tid, t):
        t.serial = None
        if t.tries >= self.cap:
            t.state = "dead"
            self.dead[tid] = None
        else:
            t.state = "ready"
            self.ready.append(tid)

    def submit(self, task_id, payload):
        if task_id in self.tasks:
            raise ValueError(f"duplicate task {task_id!r}")
        self.tasks[task_id] = _Task(payload)
        self.ready.append(task_id)
        self._sweep()

    def reserve(self, worker_id):
        if not self.ready:
            self._sweep()
            return None
        tid = self.ready.popleft()
        t = self.tasks[tid]
        self.serial += 1
        t.state, t.serial = "reserved", self.serial
        t.tries += 1
        heapq.heappush(self.leases, (self.clock.now() + self.lease, tid, self.serial))
        self._sweep()
        return Reservation(tid, self.serial, t.payload)

    def _held(self, task_id, token):
        self._sweep()
        t = self.tasks.get(task_id)
        if t is None:
            raise UnknownTaskError(task_id)
        if t.state != "reserved" or token != t.serial or type(token) is not int:
            raise InvalidReservationError(task_id)
        return t

    def complete(self, task_id, token):
        t = self._held(task_id, token)
        t.state, t.serial = "completed", None

    def fail(self, task_id, token):
        t = self._held(task_id, token)
        self._release(task_id, t)

    def dead_letters(self):
        self._sweep()
        return list(self.dead)

    def requeue_dead(self, task_id):
        self._sweep()
        t = self.tasks.get(task_id)
        if t is None:
            raise UnknownTaskError(task_id)
        if t.state != "dead":
            raise ValueError(f"{task_id!r} is not dead")
        del self.dead[task_id]
        t.state, t.tries = "ready", 0
        self.ready.append(task_id)
