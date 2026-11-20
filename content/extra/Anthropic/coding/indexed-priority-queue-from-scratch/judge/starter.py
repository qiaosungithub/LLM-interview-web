import math


class MinHeap:
    def __init__(self, items=()) -> None:
        """Builds a heap holding all of items, in O(n)."""
        raise NotImplementedError

    def push(self, item) -> None:
        """Adds item, in O(log n)."""
        raise NotImplementedError

    def pop(self):
        """Removes and returns a smallest item. Raises IndexError if the heap is empty."""
        raise NotImplementedError

    def peek(self):
        """Returns a smallest item without removing it. Raises IndexError if the heap is empty."""
        raise NotImplementedError

    def __len__(self) -> int:
        """The number of items in the heap."""
        raise NotImplementedError


class IndexedPQ:
    def __init__(self) -> None:
        """An empty queue."""
        raise NotImplementedError

    def push(self, key, priority) -> None:
        """Adds key. Raises ValueError if key is already in the queue."""
        raise NotImplementedError

    def pop(self):
        """Removes and returns (key, priority) with the smallest priority. IndexError if empty."""
        raise NotImplementedError

    def peek(self):
        """(key, priority) with the smallest priority, without removing it. IndexError if empty."""
        raise NotImplementedError

    def update(self, key, priority) -> None:
        """Changes the priority of key; it may go up or down. KeyError if key is not in the queue."""
        raise NotImplementedError

    def remove(self, key):
        """Removes key and returns its priority. KeyError if key is not in the queue."""
        raise NotImplementedError

    def priority(self, key):
        """The current priority of key. KeyError if key is not in the queue."""
        raise NotImplementedError

    def __contains__(self, key) -> bool:
        """Whether key is in the queue."""
        raise NotImplementedError

    def __len__(self) -> int:
        """The number of keys in the queue."""
        raise NotImplementedError


def dijkstra_indexed(adj: list[list[tuple[int, int]]], source: int) -> list:
    """Shortest distances from source. Uses IndexedPQ, with update() as decrease-key,
    so the queue never holds more than one entry per node."""
    raise NotImplementedError


def dijkstra_lazy(adj: list[list[tuple[int, int]]], source: int) -> list:
    """Same result with only the MinHeap of Part 1: push a new entry whenever a distance
    improves, and skip entries that are out of date when they are popped."""
    raise NotImplementedError
