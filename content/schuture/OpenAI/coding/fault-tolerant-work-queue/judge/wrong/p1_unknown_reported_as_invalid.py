# expect: WA
# Part 1 raises InvalidReservationError for a task_id that was never submitted, instead of UnknownTaskError.
import heapq
from collections import deque
from typing import Any, NamedTuple


class UnknownTaskError(Exception):
    """task_id was never passed to submit."""


class InvalidReservationError(Exception):
    """token is not the task's current reservation."""


class Reservation(NamedTuple):
    task_id: str
    token: int
    payload: Any


class _Task:
    __slots__ = ("payload", "status", "attempts", "token", "deadline")

    def __init__(self, payload):
        self.payload = payload
        self.status = "ready"
        self.attempts = 0
        self.token = None
        self.deadline = None


class WorkQueue:
    def __init__(self):
        self._tasks: dict[str, _Task] = {}
        self._ready = deque()
        self._next_token = 0

    def submit(self, task_id, payload):
        self._reclaim_expired()                     # NOTE: submit reclaims too, so expired tasks queue ahead of this one
        if task_id in self._tasks:
            raise ValueError(f"duplicate task_id {task_id!r}")
        self._tasks[task_id] = _Task(payload)
        self._ready.append(task_id)

    def reserve(self, worker_id):
        self._reclaim_expired()
        if not self._ready:
            return None
        task_id = self._ready.popleft()
        task = self._tasks[task_id]
        task.attempts += 1                          # NOTE: counted here, at reserve time -- not in fail()
        self._next_token += 1
        task.status, task.token = "reserved", self._next_token
        return Reservation(task_id, task.token, task.payload)

    def _lookup_reserved(self, task_id, token):
        self._reclaim_expired()                     # NOTE: before the token check -- this lease may just have expired
        task = self._tasks.get(task_id)
        if task is None or task.status != "reserved" or task.token != token:
            raise InvalidReservationError(task_id, token)
        return task

    def complete(self, task_id, token):
        task = self._lookup_reserved(task_id, token)
        task.status, task.token, task.deadline = "completed", None, None

    def fail(self, task_id, token):
        task = self._lookup_reserved(task_id, token)
        self._release(task_id, task)

    def _release(self, task_id, task):               # NOTE: Part 3 replaces this with a ready-or-dead version
        task.status, task.token, task.deadline = "ready", None, None
        self._ready.append(task_id)

    def _reclaim_expired(self):                       # NOTE: no-op until Part 2 gives it a real body
        pass


class ManualClock:
    def __init__(self, start: int = 0):
        self._now = start

    def now(self) -> int:
        return self._now

    def advance(self, dt: int) -> None:
        if dt < 0:
            raise ValueError("a clock cannot move backward")
        self._now += dt


def __init__(self, clock, lease_duration):
    self._clock = clock
    self._lease_duration = lease_duration
    self._tasks: dict[str, _Task] = {}
    self._ready = deque()
    self._next_token = 0
    self._leases = []          # min-heap of (deadline, task_id, token)


WorkQueue.__init__ = __init__


def reserve(self, worker_id):
    self._reclaim_expired()
    if not self._ready:
        return None
    task_id = self._ready.popleft()
    task = self._tasks[task_id]
    task.attempts += 1
    self._next_token += 1
    task.status, task.token = "reserved", self._next_token
    task.deadline = self._clock.now() + self._lease_duration
    heapq.heappush(self._leases, (task.deadline, task_id, task.token))
    return Reservation(task_id, task.token, task.payload)


WorkQueue.reserve = reserve


def _reclaim_expired(self):
    now = self._clock.now()
    while self._leases and self._leases[0][0] < now:       # NOTE: '<', not '<=' -- a lease is valid through its deadline
        _, task_id, token = heapq.heappop(self._leases)
        task = self._tasks[task_id]
        if task.status == "reserved" and task.token == token:    # NOTE: otherwise this entry is stale -- discard it
            self._release(task_id, task)


WorkQueue._reclaim_expired = _reclaim_expired


def __init__(self, clock, lease_duration, max_attempts):
    self._clock = clock
    self._lease_duration = lease_duration
    self._max_attempts = max_attempts
    self._tasks: dict[str, _Task] = {}
    self._ready = deque()
    self._dead = []            # task_ids, in the order they entered `dead`
    self._next_token = 0
    self._leases = []


WorkQueue.__init__ = __init__


def _release(self, task_id, task):
    task.token, task.deadline = None, None
    if task.attempts >= self._max_attempts:      # NOTE: attempts already counts the reservation that just ended
        task.status = "dead"
        self._dead.append(task_id)
    else:
        task.status = "ready"
        self._ready.append(task_id)


WorkQueue._release = _release


def dead_letters(self):
    self._reclaim_expired()
    return list(self._dead)


WorkQueue.dead_letters = dead_letters


def requeue_dead(self, task_id):
    self._reclaim_expired()
    task = self._tasks.get(task_id)
    if task is None:
        raise UnknownTaskError(task_id)
    if task.status != "dead":
        raise ValueError(f"{task_id!r} is not in the dead-letter queue")
    self._dead.remove(task_id)
    task.status, task.attempts = "ready", 0
    self._ready.append(task_id)


WorkQueue.requeue_dead = requeue_dead
