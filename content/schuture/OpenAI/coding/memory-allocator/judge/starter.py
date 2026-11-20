class MemoryAllocator:
    def __init__(self, capacity: int) -> None:
        """Part 1: the whole space [0, capacity) starts free; raises ValueError if capacity <= 0."""

    def allocate(self, size: int) -> int:
        """First fit: takes size bytes from the low end of the lowest-addressed free block that is large enough."""
        raise NotImplementedError

    def free(self, address: int, size: int) -> None:
        """Releases the allocated block (address, size) and merges it with touching free blocks."""
        raise NotImplementedError


class MemoryAllocatorLog:
    def __init__(self, capacity: int) -> None:
        """Part 2: same behaviour as MemoryAllocator."""

    def allocate(self, size: int) -> int:
        """Same as MemoryAllocator.allocate, in O(log n) for n free blocks."""
        raise NotImplementedError

    def free(self, address: int, size: int) -> None:
        """Same as MemoryAllocator.free, in O(log n) for n free blocks."""
        raise NotImplementedError
