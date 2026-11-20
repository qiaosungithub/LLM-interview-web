class IPV4Iterator:
    def __init__(self, ip_or_cidr: str, reverse: bool = False, step: int = 1) -> None:
        """ip_or_cidr is "a.b.c.d" or "a.b.c.d/prefix". Raises ValueError if it is invalid or step <= 0."""
        raise NotImplementedError

    def __iter__(self) -> "IPV4Iterator":
        """Returns the iterator, so that the object works in a for loop."""
        raise NotImplementedError

    def __next__(self) -> str:
        """Returns the next address in the direction reverse selects. Raises StopIteration afterwards."""
        raise NotImplementedError

    def next_batch(self, size: int) -> list[str]:
        """Returns up to size addresses from the current position. Raises ValueError if size < 0."""
        raise NotImplementedError
