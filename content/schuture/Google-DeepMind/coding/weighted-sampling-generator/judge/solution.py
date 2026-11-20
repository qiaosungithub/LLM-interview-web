"""The page's reference answer (Parts 1-3, with its unittest suites), joined into one file."""
from collections.abc import Iterable, Iterator


def iter_ints(items: Iterable[object] | None) -> Iterator[int]:
    """Yields the int elements of items, in order, skipping None. Raises TypeError, naming the
    0-based index, on the first element that is neither None nor an int (bool excluded)."""
    if items is None:
        return
    # NOTE: iterate items directly and yield as each value is ready -- calling list(items) first
    # would force even an infinite items to run forever before this function could produce anything.
    for index, x in enumerate(items):
        if x is None:           # NOTE: must be `is None`, not `if not x` -- the latter would also
            continue            #       skip the valid value 0, and would skip False instead of rejecting it
        # NOTE: isinstance(x, bool) must be checked and rejected separately -- bool is a subclass of
        # int in Python, so isinstance(x, int) alone would silently accept True and False as 1 and 0.
        if not isinstance(x, int) or isinstance(x, bool):
            # NOTE: raised right here, the moment this element is reached -- validating every element
            # of items before yielding any of them would raise before the valid values at earlier
            # positions are yielded, and would never finish on an input that runs forever.
            raise TypeError(f"element at index {index} is not an int: {x!r}")
        yield x


import unittest


class _CountingIterator:
    """Wraps a fixed, finite sequence and records how many of its elements __next__ has actually
    produced -- this is how the laziness test below observes pulls without needing an infinite
    iterator, so an eager bug shows up as a clean assertion failure rather than a hang."""

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)
        self._position = 0
        self.pulled = 0

    def __iter__(self) -> "_CountingIterator":
        return self

    def __next__(self) -> int:
        if self._position >= len(self._values):
            raise StopIteration
        value = self._values[self._position]
        self._position += 1
        self.pulled += 1
        return value


class IterIntsTests(unittest.TestCase):
    FUNC = staticmethod(iter_ints)

    def test_empty_and_none_input(self) -> None:
        self.assertEqual(list(self.FUNC([])), [])
        self.assertEqual(list(self.FUNC(None)), [])

    def test_none_at_start_middle_end_and_all_none(self) -> None:
        self.assertEqual(list(self.FUNC([None, 1, 2])), [1, 2])
        self.assertEqual(list(self.FUNC([1, None, 2])), [1, 2])
        self.assertEqual(list(self.FUNC([1, 2, None])), [1, 2])
        self.assertEqual(list(self.FUNC([None, None, None])), [])

    def test_zero_and_negative_are_ordinary_values(self) -> None:
        self.assertEqual(list(self.FUNC([0, -5, 3, 0])), [0, -5, 3, 0])

    def test_lazy_one_pull_per_next_call(self) -> None:
        source = _CountingIterator([10, 20, 30])
        gen = self.FUNC(source)
        self.assertEqual(source.pulled, 0)              # nothing pulled before the first next()
        self.assertEqual(next(gen), 10)
        self.assertEqual(source.pulled, 1)
        self.assertEqual(next(gen), 20)
        self.assertEqual(source.pulled, 2)

    def test_invalid_element_after_valid_ones(self) -> None:
        gen = self.FUNC([1, 2, True])
        self.assertEqual(next(gen), 1)
        self.assertEqual(next(gen), 2)
        with self.assertRaises(TypeError) as ctx:
            next(gen)
        self.assertIn("index 2", str(ctx.exception))

    def test_bool_float_str_rejected(self) -> None:
        for bad in (True, False, 3.0, "3"):
            with self.assertRaises(TypeError):
                list(self.FUNC([bad]))

    def test_input_list_is_unchanged(self) -> None:
        items = [3, None, 0, -2]
        before = list(items)
        list(self.FUNC(items))
        self.assertEqual(items, before)

    def test_two_calls_are_independent(self) -> None:
        items = [1, 2, 3]
        first, second = self.FUNC(items), self.FUNC(items)
        self.assertEqual(next(first), 1)
        self.assertEqual(next(second), 1)               # unaffected by first's progress
        self.assertEqual(next(first), 2)
        self.assertEqual(list(second), [2, 3])


import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from itertools import accumulate

import scipy.stats


def weighted_sampler(values: Sequence[int], weights: Sequence[float], rng: random.Random) -> Iterator[int]:
    """Infinite generator: each next() returns values[i] with probability weights[i] / sum(weights),
    independently of every earlier draw, using only rng for randomness."""
    # NOTE: weighted_sampler is an ordinary function, not itself a generator -- see above. Validating
    # here and returning a nested generator raises eagerly while the draws it returns stay lazy.
    if len(values) != len(weights) or len(values) == 0:
        raise ValueError("values and weights must have equal, non-zero length")
    for w in weights:
        if not (math.isfinite(w) and w >= 0):
            raise ValueError(f"weight {w!r} is not finite and non-negative")
    cumulative = list(accumulate(weights))
    total = cumulative[-1]
    # NOTE: finite weights can still add up to inf (1e308 + 1e308); rng.random() * inf would then lie
    # past every prefix sum, so the total must be checked for finiteness as well as positivity.
    if not 0 < total < math.inf:
        raise ValueError("weights must have a finite, positive sum")

    def _draws() -> Iterator[int]:
        while True:
            # NOTE: scale by `total`, i.e. by cumulative[-1] -- not by a separately recomputed
            # sum(weights). Floating-point addition is not perfectly associative, so the two sums can
            # differ in their last bit; scaling by a freshly recomputed sum could then push u a hair
            # past cumulative[-1], and bisect_right would return an index one past the end of values.
            u = rng.random() * total
            yield values[bisect_right(cumulative, u)]

    return _draws()


class _StubRandom:
    """A random.Random look-alike whose random() returns a fixed, pre-chosen sequence of floats, so a
    test can pin down exactly which u each draw uses -- rather than which value happens to come out."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._position = 0

    def random(self) -> float:
        value = self._values[self._position]
        self._position += 1
        return value


def _chi_square_p_value(draws: list[int], values: Sequence[int], weights: Sequence[float]) -> float:
    """Independent statistical check: compares observed counts of each value in draws against the
    counts expected under weights / sum(weights), via scipy's chi-square goodness-of-fit test. Assumes
    every weight is strictly positive, so no expected count is ever zero."""
    total = math.fsum(weights)
    counts = Counter(draws)
    observed = [counts[v] for v in values]
    expected = [w / total * len(draws) for w in weights]
    return scipy.stats.chisquare(observed, expected).pvalue


class WeightedSamplerTests(unittest.TestCase):
    def test_validation_is_eager_not_deferred_to_first_next(self) -> None:
        with self.assertRaises(ValueError):
            weighted_sampler([1, 2], [1.0], random.Random(0))           # length mismatch
        with self.assertRaises(ValueError):
            weighted_sampler([], [], random.Random(0))                  # zero length
        with self.assertRaises(ValueError):
            weighted_sampler([1], [-1.0], random.Random(0))             # negative weight
        with self.assertRaises(ValueError):
            weighted_sampler([1], [math.inf], random.Random(0))         # non-finite weight
        with self.assertRaises(ValueError):
            weighted_sampler([1, 2], [0.0, 0.0], random.Random(0))      # sums to zero
        with self.assertRaises(ValueError):
            weighted_sampler([1, 2], [1e308, 1e308], random.Random(0))  # sum overflows to inf

    def test_stub_rng_maps_u_to_value_at_the_boundaries(self) -> None:
        values, weights = [1, 2, 3], [0.1, 0.1, 0.8]     # weights sum to exactly 1.0 here
        boundary_us = [0.0, 0.05, 0.1, 0.15, 0.2, 0.93]
        stub = _StubRandom(boundary_us)
        gen = weighted_sampler(values, weights, stub)
        self.assertEqual([next(gen) for _ in boundary_us], [1, 1, 2, 2, 3, 3])

    def test_only_listed_values_and_zero_weight_never_appears(self) -> None:
        values, weights = [10, 20, 30, 40], [0.0, 2.0, 0.0, 1.0]
        gen = weighted_sampler(values, weights, random.Random(1))
        draws = [next(gen) for _ in range(2_000)]
        self.assertTrue(set(draws) <= {10, 20, 30, 40})
        self.assertNotIn(10, draws)
        self.assertNotIn(30, draws)
        self.assertIn(20, draws)             # sanity: not everything got filtered out by mistake
        self.assertIn(40, draws)

    def test_chi_square_goodness_of_fit(self) -> None:
        values, weights = [1, 2, 3], [0.1, 0.1, 0.8]
        gen = weighted_sampler(values, weights, random.Random(12345))
        draws = [next(gen) for _ in range(150_000)]
        p_value = _chi_square_p_value(draws, values, weights)
        # NOTE: 1e-6 is a low bar on purpose -- a correctly distributed sampler clears it on virtually
        # any fixed seed, while a sampler biased by even a few percentage points fails it overwhelmingly
        # (see the deliberately biased samplers in the checks below).
        self.assertGreater(p_value, 1e-6)


class AliasSampler:
    """Vose's alias method: O(n) preprocessing, then O(1) time and at most two random numbers per
    draw, regardless of n. Same validation and semantics -- including which values ever appear -- as
    weighted_sampler."""

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

        scaled = [w / total * n for w in weights]   # NOTE: n * w first could overflow to inf
        light = [i for i, s in enumerate(scaled) if s < 1.0]
        heavy = [i for i, s in enumerate(scaled) if s >= 1.0]

        while light and heavy:
            s = light.pop()
            l = heavy.pop()
            self._prob[s] = scaled[s]
            self._alias[s] = l
            scaled[l] -= 1.0 - scaled[s]      # NOTE: l donates exactly what s was short of 1.0
            if scaled[l] < 1.0:
                light.append(l)
            else:
                heavy.append(l)

        # NOTE: in exact arithmetic the loop ends with light empty and every column left in heavy at
        # scaled weight exactly 1 (see the invariant in the text below); rounding can also strand a
        # column in light just below 1. Every leftover column's true scaled weight is 1, so it keeps
        # its whole share: prob = 1, and alias is never consulted for it.
        for i in light:
            self._prob[i] = 1.0
        for i in heavy:
            self._prob[i] = 1.0

    def sample(self) -> int:
        i = self._rng.randrange(len(self._values))
        if self._rng.random() < self._prob[i]:
            return self._values[i]
        return self._values[self._alias[i]]

    def __iter__(self) -> Iterator[int]:
        while True:
            yield self.sample()
