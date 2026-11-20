import random
from collections.abc import Iterable, Iterator, Sequence


def iter_ints(items: Iterable[object] | None) -> Iterator[int]:
    """Lazily yield the int elements of items, skipping None; raise TypeError naming the index of a bad element."""
    raise NotImplementedError


def weighted_sampler(values: Sequence[int], weights: Sequence[float], rng: random.Random) -> Iterator[int]:
    """Validate now, then return an infinite iterator drawing values[i] with probability weights[i] / sum(weights)."""
    raise NotImplementedError


class AliasSampler:
    def __init__(self, values: Sequence[int], weights: Sequence[float], rng: random.Random) -> None:
        """Validate and build the alias tables in O(n)."""
        raise NotImplementedError

    def sample(self) -> int:
        """Draw one value in O(1)."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[int]:
        """Yield sample() forever."""
        raise NotImplementedError
