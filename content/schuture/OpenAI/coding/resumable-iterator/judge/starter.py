from abc import ABC, abstractmethod


class ResumableIterator(ABC):
    """Abstract base class: declare __iter__, __next__, get_state and set_state as abstract methods."""

    def __iter__(self) -> "ResumableIterator":
        """Returns the iterator itself, like any Python iterator."""
        raise NotImplementedError

    def __next__(self):
        """Returns the next item. Raises StopIteration once none remain."""
        raise NotImplementedError

    def get_state(self) -> tuple:
        """Returns a copyable, JSON-serializable tuple describing the current position."""
        raise NotImplementedError

    def set_state(self, state) -> None:
        """Moves to the position state describes. Raises TypeError or ValueError as the statement specifies."""
        raise NotImplementedError


class ResumableListIterator(ResumableIterator):
    def __init__(self, items: list) -> None:
        """items may be empty."""

    def __iter__(self) -> "ResumableIterator":
        """Returns the iterator itself, like any Python iterator."""
        raise NotImplementedError

    def __next__(self):
        """Returns the next item. Raises StopIteration once none remain."""
        raise NotImplementedError

    def get_state(self) -> tuple:
        """Returns the index tuple of the next item, or (len(items),) once exhausted."""
        raise NotImplementedError

    def set_state(self, state) -> None:
        """Moves to the position state describes. Raises TypeError or ValueError as the statement specifies."""
        raise NotImplementedError


class Resumable2DIterator(ResumableIterator):
    def __init__(self, items: list) -> None:
        """items is a list of lists; a row may be empty."""

    def __iter__(self) -> "ResumableIterator":
        """Returns the iterator itself, like any Python iterator."""
        raise NotImplementedError

    def __next__(self):
        """Returns the next item. Raises StopIteration once none remain."""
        raise NotImplementedError

    def get_state(self) -> tuple:
        """Returns the index tuple of the next item, or (len(items), 0) once exhausted."""
        raise NotImplementedError

    def set_state(self, state) -> None:
        """Moves to the position state describes. Raises TypeError or ValueError as the statement specifies."""
        raise NotImplementedError


class ResumableNestedIterator(ResumableIterator):
    def __init__(self, items: list, depth: int) -> None:
        """items is nested depth levels deep, depth >= 1; a list at any level may be empty."""

    def __iter__(self) -> "ResumableIterator":
        """Returns the iterator itself, like any Python iterator."""
        raise NotImplementedError

    def __next__(self):
        """Returns the next item. Raises StopIteration once none remain."""
        raise NotImplementedError

    def get_state(self) -> tuple:
        """Returns the index tuple of the next item, or (len(items), 0, ..., 0) once exhausted."""
        raise NotImplementedError

    def set_state(self, state) -> None:
        """Moves to the position state describes. Raises TypeError or ValueError as the statement specifies."""
        raise NotImplementedError
