# expect: WA
# FairModalLock.acquire joins any batch of the same mode anywhere in the queue instead of only the tail, so a newcomer cuts in front of callers of other modes.
import threading
from collections import deque


class ModeMismatchError(RuntimeError):
    """release(mode) does not match a reservation this call actually holds."""


class ModalLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._mode = None
        self._holders = 0
        self._waiting = 0

    def acquire(self, mode):
        with self._cond:
            self._waiting += 1
            try:
                # NOTE: while, not if -- notify_all() wakes the parked callers of every mode at once;
                # by the time this one gets back in, another may already hold the lock in its mode
                while self._mode is not None and self._mode != mode:
                    self._cond.wait()
            finally:
                self._waiting -= 1
            self._mode = mode
            self._holders += 1

    def release(self, mode):
        with self._cond:
            if self._holders == 0 or self._mode != mode:
                raise ModeMismatchError(mode)
            self._holders -= 1
            if self._holders == 0:
                self._mode = None
                self._cond.notify_all()
                                            # caller, and the others parked for that same mode
                                            # would stay asleep instead of joining it

    def waiting_count(self):
        with self._cond:
            return self._waiting


class _Batch:
    __slots__ = ("mode", "waiting", "active")

    def __init__(self, mode):
        self.mode = mode
        self.waiting = 0
        self.active = 0


class FairModalLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._queue = deque()

    def acquire(self, mode):
        with self._cond:
            batch = next((b for b in self._queue if b.mode == mode), None)
            if batch is None:
                batch = _Batch(mode)
                self._queue.append(batch)
            batch.waiting += 1
            while self._queue[0] is not batch:
                self._cond.wait()
            batch.waiting -= 1
            batch.active += 1

    def release(self, mode):
        with self._cond:
            batch = self._queue[0] if self._queue else None
            if batch is None or batch.mode != mode or batch.active == 0:
                raise ModeMismatchError(mode)
            batch.active -= 1
            if batch.active == 0 and batch.waiting == 0:
                self._queue.popleft()
                self._cond.notify_all()

    def waiting_count(self):
        with self._cond:
            return sum(batch.waiting for batch in self._queue)
