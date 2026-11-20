# expect: WA
# FairModalLock checks whether its batch is at the head under the queue's mutex but waits on the batch's own Condition only after letting the mutex go: a release() in between notifies nobody and the caller sleeps forever.
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

    def waiting_count(self):
        with self._cond:
            return self._waiting


class _Batch:
    def __init__(self, mode):
        self.mode = mode
        self.waiting = 0
        self.active = 0
        self.cond = threading.Condition()


class FairModalLock:
    def __init__(self):
        self._lock = threading.Lock()
        self._queue = deque()

    def acquire(self, mode):
        with self._lock:
            if self._queue and self._queue[-1].mode == mode:
                batch = self._queue[-1]
            else:
                batch = _Batch(mode)
                self._queue.append(batch)
            batch.waiting += 1
            at_head = self._queue[0] is batch
        if not at_head:
            with batch.cond:
                batch.cond.wait()
        with self._lock:
            batch.waiting -= 1
            batch.active += 1

    def release(self, mode):
        with self._lock:
            batch = self._queue[0] if self._queue else None
            if batch is None or batch.mode != mode or batch.active == 0:
                raise ModeMismatchError(mode)
            batch.active -= 1
            nxt = None
            if batch.active == 0 and batch.waiting == 0:
                self._queue.popleft()
                nxt = self._queue[0] if self._queue else None
        if nxt is not None:
            with nxt.cond:
                nxt.cond.notify_all()

    def waiting_count(self):
        with self._lock:
            return sum(batch.waiting for batch in self._queue)
