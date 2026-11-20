class TextBuffer:
    def __init__(self, text: str = "") -> None:
        """Starts with these initial contents."""
        raise NotImplementedError

    def __len__(self) -> int:
        """The current length, in characters."""
        raise NotImplementedError

    def text(self) -> str:
        """The entire contents, equivalent to read(0, len(self))."""
        raise NotImplementedError

    def insert(self, pos: int, text: str) -> None:
        """Inserts text at offset pos (0 <= pos <= len(self)); raises IndexError otherwise."""
        raise NotImplementedError

    def delete(self, start: int, end: int) -> str:
        """Removes [start, end) and returns the removed text; raises IndexError unless 0 <= start <= end <= len."""
        raise NotImplementedError

    def read(self, start: int, end: int) -> str:
        """Returns [start, end) without modifying the buffer. Same bounds as delete."""
        raise NotImplementedError

    # Part 2
    def undo(self) -> bool:
        """Reverses the most recent recorded insert/delete. Returns whether there was one to reverse."""
        raise NotImplementedError

    def redo(self) -> bool:
        """Re-applies the most recently undone operation. Returns whether there was one to redo."""
        raise NotImplementedError

    # Part 3
    def suggest(self, prefix: str, k: int) -> list[str]:
        """Up to k current words starting with prefix, by frequency, ties by string order."""
        raise NotImplementedError


# Part 4
class CRDTDoc:
    def __init__(self, site_id: str) -> None:
        """A single replica, starting from an empty document."""
        raise NotImplementedError

    def text(self) -> str:
        """This replica's current visible document."""
        raise NotImplementedError

    def local_insert(self, index: int, ch: str) -> object:
        """Inserts one character at visible index as a local edit; returns an operation to broadcast."""
        raise NotImplementedError

    def local_delete(self, index: int) -> object:
        """Deletes the visible character at index as a local edit; returns an operation to broadcast."""
        raise NotImplementedError

    def apply(self, op: object) -> None:
        """Applies an operation from any replica (including this one); idempotent."""
        raise NotImplementedError
