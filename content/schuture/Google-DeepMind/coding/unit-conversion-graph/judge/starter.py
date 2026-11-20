def convert(facts: list[tuple[str, str, float]], value: float, src: str, dst: str) -> float | None:
    """Return value expressed in dst, or None if src or dst is unknown or the two are not connected."""
    raise NotImplementedError


def first_contradiction(facts: list[tuple[str, str, float]], rel_tol: float = 1e-9) -> int:
    """Return the index of the first fact that contradicts the facts before it, or -1 if none does."""
    raise NotImplementedError


class Converter:
    def __init__(self) -> None:
        """Start with no facts."""
        raise NotImplementedError

    def add_fact(self, a: str, b: str, r: float) -> bool:
        """Record 1 a = r b; return False (and change nothing) if it contradicts what is already known."""
        raise NotImplementedError

    def query(self, src: str, dst: str) -> float | None:
        """Return the factor such that 1 src = factor dst, or None if unknown or not connected."""
        raise NotImplementedError
