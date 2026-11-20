# expect: TLE
# The page's NaiveWorkQueue: correct, but every call scans every task ever submitted.
from typing import Any, NamedTuple


class UnknownTaskError(Exception):
    """task_id was never passed to submit."""


class InvalidReservationError(Exception):
    """token is not the task's current reservation."""


class Reservation(NamedTuple):
    task_id: str
    token: int
    payload: Any


class WorkQueue:
    """The page's NaiveWorkQueue, renamed: a direct reading of the statement, with no heap.
    Every call scans every task to find expired leases."""

    def __init__(self, clock, lease_duration, max_attempts):
        self.clock = clock
        self.lease_duration = lease_duration
        self.max_attempts = max_attempts
        self.status, self.payload, self.attempts = {}, {}, {}
        self.token, self.deadline, self.order = {}, {}, {}
        self._seq = 0
        self._next_token = 0
        self.dead_order = []

    def _mark_ready(self, task_id):
        self.status[task_id] = "ready"
        self.token[task_id] = None
        self.deadline[task_id] = None
        self._seq += 1
        self.order[task_id] = self._seq

    def _reclaim_all(self):
        now = self.clock.now()
        expired = sorted(                    # ascending deadline, ties by ascending task_id
            (self.deadline[t], t) for t in self.status
            if self.status[t] == "reserved" and self.deadline[t] < now
        )
        for _, task_id in expired:
            self._expire_or_kill(task_id)

    def _expire_or_kill(self, task_id):
        if self.attempts[task_id] >= self.max_attempts:
            self.status[task_id] = "dead"
            self.token[task_id] = None
            self.deadline[task_id] = None
            self.dead_order.append(task_id)
        else:
            self._mark_ready(task_id)

    def submit(self, task_id, payload):
        self._reclaim_all()
        if task_id in self.status:
            raise ValueError(f"duplicate task_id {task_id!r}")
        self.payload[task_id] = payload
        self.attempts[task_id] = 0
        self._mark_ready(task_id)

    def reserve(self, worker_id):
        self._reclaim_all()
        ready_ids = [t for t, st in self.status.items() if st == "ready"]
        if not ready_ids:
            return None
        task_id = min(ready_ids, key=lambda t: self.order[t])
        self.attempts[task_id] += 1
        self._next_token += 1
        self.token[task_id] = self._next_token
        self.status[task_id] = "reserved"
        self.deadline[task_id] = self.clock.now() + self.lease_duration
        return Reservation(task_id, self.token[task_id], self.payload[task_id])

    def _check(self, task_id, token):
        self._reclaim_all()
        if task_id not in self.status:
            raise UnknownTaskError(task_id)
        if self.status[task_id] != "reserved" or self.token[task_id] != token:
            raise InvalidReservationError(task_id, token)

    def complete(self, task_id, token):
        self._check(task_id, token)
        self.status[task_id] = "completed"
        self.token[task_id] = None
        self.deadline[task_id] = None

    def fail(self, task_id, token):
        self._check(task_id, token)
        self._expire_or_kill(task_id)

    def dead_letters(self):
        self._reclaim_all()
        return list(self.dead_order)

    def requeue_dead(self, task_id):
        self._reclaim_all()
        if task_id not in self.status:
            raise UnknownTaskError(task_id)
        if self.status[task_id] != "dead":
            raise ValueError(f"{task_id!r} is not in the dead-letter queue")
        self.dead_order.remove(task_id)
        self.attempts[task_id] = 0
        self._mark_ready(task_id)


class ManualClock:
    def __init__(self, start: int = 0):
        self._now = start

    def now(self) -> int:
        return self._now

    def advance(self, dt: int) -> None:
        if dt < 0:
            raise ValueError("a clock cannot move backward")
        self._now += dt
