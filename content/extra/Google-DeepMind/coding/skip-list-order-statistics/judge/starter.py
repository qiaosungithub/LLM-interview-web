class SkipList:
    def __init__(self, seed: int | None = None, p: float = 0.5) -> None:
        """Creates an empty sorted multiset; seed initializes the structure's own random.Random(seed), p is the
        probability of growing a node's height by one."""

    def add(self, key: int) -> None:
        """Part 1: inserts key; duplicates are kept."""
        raise NotImplementedError

    def __contains__(self, key: int) -> bool:
        """Part 1: `key in sl`."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Part 1: the number of stored keys, duplicates counted."""
        raise NotImplementedError

    def __iter__(self):
        """Part 1: the keys in ascending order, duplicates included."""
        raise NotImplementedError

    def remove(self, key: int) -> bool:
        """Part 2: removes ONE occurrence of key and returns True, or returns False if key is not stored."""
        raise NotImplementedError

    def keys_between(self, lo: int, hi: int) -> list[int]:
        """Part 2: every stored key with lo <= key <= hi, ascending, duplicates included."""
        raise NotImplementedError


class IndexedSkipList(SkipList):
    def kth(self, i: int) -> int:
        """Part 3: the i-th smallest key, counting from 0; IndexError if i is not in [0, len)."""
        raise NotImplementedError

    def rank(self, key: int) -> int:
        """Part 3: how many stored keys are strictly smaller than key."""
        raise NotImplementedError
