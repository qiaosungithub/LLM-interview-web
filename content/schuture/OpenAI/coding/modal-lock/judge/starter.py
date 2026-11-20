class ModeMismatchError(Exception):
    """Raised by release(mode) when the lock has no holders at all, or is held in a mode other than `mode`."""


class ModalLock:
    def __init__(self) -> None:
        """Part 1: a lock any number of threads may hold together, as long as they all ask for the same mode."""

    def acquire(self, mode: str) -> None:
        """Block until the lock is idle or held in `mode`; then hold it in `mode`."""
        raise NotImplementedError

    def release(self, mode: str) -> None:
        """Stop holding the lock; the last holder of the current mode makes it idle again. Raises
        ModeMismatchError if the lock has no holders or its current mode is not `mode`."""
        raise NotImplementedError

    def waiting_count(self) -> int:
        """The number of acquire() calls currently blocked, across every mode."""
        raise NotImplementedError


class FairModalLock:
    def __init__(self) -> None:
        """Part 2: a ModalLock that grants modes in arrival order, one FIFO queue of batches."""

    def acquire(self, mode: str) -> None:
        """Join the tail batch if it asks for `mode`, else start a new batch at the tail; return once that batch
        is at the head of the queue."""
        raise NotImplementedError

    def release(self, mode: str) -> None:
        """Stop holding the lock; the head batch leaves the queue once every member that joined it has released.
        Raises ModeMismatchError under the same conditions as ModalLock.release."""
        raise NotImplementedError

    def waiting_count(self) -> int:
        """The number of acquire() calls currently blocked, across every mode."""
        raise NotImplementedError
