# expect: WA
# weighted_sampler subtracts weights from u = random() * total until it goes negative; rounding can leave u >= 0 after the last weight, and it returns None.
import math
import random
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence


def iter_ints(items: Iterable[object] | None) -> Iterator[int]:
    if items is None:
        return
    it = iter(items)
    i = -1
    while True:
        try:
            x = next(it)
        except StopIteration:
            return
        i += 1
        if x is None:
            continue
        if type(x) is bool or not isinstance(x, int):
            raise TypeError("bad element at position %d" % i)
        yield x


def _check(values, weights):
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("bad lengths")
    total = 0.0
    for w in weights:
        if math.isnan(w) or math.isinf(w) or w < 0:
            raise ValueError("bad weight")
        total += w
    if math.isinf(total) or total <= 0:
        raise ValueError("bad total")
    return total


def weighted_sampler(values: Sequence[int], weights: Sequence[float], rng: random.Random) -> Iterator[int]:
    _check(values, weights)
    vals = list(values)
    ws = list(weights)
    cum = []
    run = 0.0
    for w in weights:
        run += w
        cum.append(run)
    total = cum[-1]
    n = len(vals)

    def pick(u):
        for j in range(n):
            u -= ws[j]
            if u < 0:
                return vals[j]

    def gen():
        while True:
            yield pick(rng.random() * total)
    return gen()


class AliasSampler:
    """Walker alias, one uniform per draw: u * n gives column and coin."""

    def __init__(self, values, weights, rng):
        total = _check(values, weights)
        n = len(values)
        self.vals = list(values)
        self.rng = rng
        self.n = n
        p = [w / total * n for w in weights]
        self.prob = [1.0] * n
        self.alias = list(range(n))
        small = [i for i in range(n) if p[i] < 1.0]
        large = [i for i in range(n) if p[i] >= 1.0]
        while small and large:
            s = small.pop()
            g = large[-1]
            self.prob[s] = p[s]
            self.alias[s] = g
            p[g] = (p[g] + p[s]) - 1.0
            if p[g] < 1.0:
                large.pop()
                small.append(g)
        # leftovers keep prob 1, alias self

    def sample(self):
        u = self.rng.random() * self.n
        i = int(u)
        if i >= self.n:
            i = self.n - 1
        if u - i < self.prob[i]:
            return self.vals[i]
        return self.vals[self.alias[i]]

    def __iter__(self):
        while True:
            yield self.sample()
