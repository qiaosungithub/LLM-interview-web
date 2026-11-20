from typing import Any, NamedTuple


class UnknownTaskError(Exception):
    """Raised by complete / fail / requeue_dead for a task_id that was never submitted."""


class InvalidReservationError(Exception):
    """Raised by complete / fail when the task is not reserved, or the token is not the one this reservation got."""


class Reservation(NamedTuple):
    task_id: str
    token: object
    payload: Any


class ManualClock:
    """The statement's clock (Part 2 on). Tests pass in a clock that behaves exactly like this one."""

    def __init__(self, start: int = 0):
        self._now = start

    def now(self) -> int:
        return self._now

    def advance(self, dt: int) -> None:
        if dt < 0:
            raise ValueError("a clock cannot move backward")
        self._now += dt


class WorkQueue:
    def __init__(self) -> None:
        """Part 1: an empty queue; Part 2 adds the clock and lease_duration parameters, Part 3 max_attempts."""

    def submit(self, task_id: str, payload: Any) -> None:
        """Adds a ready task; ValueError if task_id was ever submitted before."""
        raise NotImplementedError

    def reserve(self, worker_id: str) -> "Reservation | None":
        """Hands out the task that entered ready earliest, with a fresh token; None if no task is ready."""
        raise NotImplementedError

    def complete(self, task_id: str, token: object) -> None:
        """Marks the reserved task completed if token is this reservation's token."""
        raise NotImplementedError

    def fail(self, task_id: str, token: object) -> None:
        """Same token check as complete; the task goes back to the tail of ready (Part 3: or becomes dead)."""
        raise NotImplementedError

    def dead_letters(self) -> list[str]:
        """Part 3: task_ids of every dead task, in the order they became dead."""
        raise NotImplementedError

    def requeue_dead(self, task_id: str) -> None:
        """Part 3: puts a dead task at the tail of ready with its attempt count reset to 0."""
        raise NotImplementedError
