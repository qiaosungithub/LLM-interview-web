# expect: WA
# AliasSampler.sample keeps column i when random() <= prob[i], so random() == 0.0 returns a zero-weight value.
import math
import random
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from itertools import accumulate


def iter_ints(items: Iterable[object] | None) -> Iterator[int]:
    if items is None:
        return
    for index, x in enumerate(items):
        if x is None:
            continue
        if not isinstance(x, int) or isinstance(x, bool):
            raise TypeError(f"element at index {index} is not an int: {x!r}")
        yield x


def weighted_sampler(values: Sequence[int], weights: Sequence[float], rng: random.Random) -> Iterator[int]:
    if len(values) != len(weights) or len(values) == 0:
        raise ValueError("values and weights must have equal, non-zero length")
    for w in weights:
        if not (math.isfinite(w) and w >= 0):
            raise ValueError(f"weight {w!r} is not finite and non-negative")
    cumulative = list(accumulate(weights))
    total = cumulative[-1]
    if not 0 < total < math.inf:
        raise ValueError("weights must have a finite, positive sum")

    def _draws() -> Iterator[int]:
        while True:
            u = rng.random() * total
            yield values[bisect_right(cumulative, u)]

    return _draws()


class AliasSampler:
    def __init__(self, values: Sequence[int], weights: Sequence[float], rng: random.Random) -> None:
        n = len(values)
        if n != len(weights) or n == 0:
            raise ValueError("values and weights must have equal, non-zero length")
        for w in weights:
            if not (math.isfinite(w) and w >= 0):
                raise ValueError(f"weight {w!r} is not finite and non-negative")
        total = sum(weights)
        if not 0 < total < math.inf:
            raise ValueError("weights must have a finite, positive sum")

        self._values = list(values)
        self._rng = rng
        self._prob = [0.0] * n
        self._alias = [0] * n

        scaled = [w / total * n for w in weights]
        light = [i for i, s in enumerate(scaled) if s < 1.0]
        heavy = [i for i, s in enumerate(scaled) if s >= 1.0]

        while light and heavy:
            s = light.pop()
            l = heavy.pop()
            self._prob[s] = scaled[s]
            self._alias[s] = l
            scaled[l] -= 1.0 - scaled[s]
            if scaled[l] < 1.0:
                light.append(l)
            else:
                heavy.append(l)

        for i in light:
            self._prob[i] = 1.0
        for i in heavy:
            self._prob[i] = 1.0

    def sample(self) -> int:
        i = self._rng.randrange(len(self._values))
        if self._rng.random() <= self._prob[i]:
            return self._values[i]
        return self._values[self._alias[i]]

    def __iter__(self) -> Iterator[int]:
        while True:
            yield self.sample()
