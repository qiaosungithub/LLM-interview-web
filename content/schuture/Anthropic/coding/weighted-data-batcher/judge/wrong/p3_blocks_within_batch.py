# expect: WA
# makes the right choices for the batch but emits each dataset's share as one alphabetical block
class DataBatcher:
    def __init__(self, registry: "DataRegistry", weights: dict[str, int], batch_size: int) -> None:
        if not weights or any(w <= 0 for w in weights.values()):
            raise ValueError("weights must be a non-empty mapping of positive integers")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._registry = registry
        self._names = sorted(weights)                      # NOTE: fixed alphabetical order, used everywhere
        self._weights = dict(weights)
        self._total_weight = sum(weights.values())
        self._batch_size = batch_size
        self._offsets = {name: 0 for name in self._names}  # examples drawn so far, per dataset
        self._iters: dict = {}                              # live iterators, opened lazily, never replayed

    def _iterator_for(self, name: str):
        it = self._iters.get(name)
        if it is None:                                       # NOTE: opened once per dataset, at its offset
            it = self._registry.get_iterator(name, self._offsets[name])
            self._iters[name] = it
        return it

    def _take(self, name: str, count: int) -> list:
        it = self._iterator_for(name)
        items = [next(it) for _ in range(count)]
        self._offsets[name] += count
        return items

    def next_batch(self) -> list:
        batch = []
        for name in self._names:
            quota = self._batch_size * self._weights[name] // self._total_weight
            batch.extend(self._take(name, quota))
        return batch


class DataBatcher(DataBatcher):
    def state_dict(self) -> dict:
        return {"offsets": dict(self._offsets)}             # NOTE: a copy -- callers may hold on to this

    def load_state_dict(self, state: dict) -> None:
        self._offsets = dict(state["offsets"])
        self._iters = {}          # NOTE: dropped, not replayed -- reopened lazily at the restored offsets


import random
from bisect import bisect_right


class DataBatcher(DataBatcher):
    def __init__(self, registry: "DataRegistry", weights: dict[str, int], batch_size: int,
                 allocation: str = "deterministic", seed: int = 0) -> None:
        if allocation not in ("deterministic", "stochastic"):
            raise ValueError(f"unknown allocation {allocation!r}")
        super().__init__(registry, weights, batch_size)
        self._allocation = allocation
        self._prefix: list[int] = []          # cumulative weight, same alphabetical order as self._names
        running = 0
        for name in self._names:
            running += self._weights[name]
            self._prefix.append(running)
        self._rng = random.Random(seed)

    def next_batch(self) -> list:
        if self._batch_size % self._total_weight == 0:
            return super().next_batch()                        # NOTE: Part 1's exact rule, unchanged
        pick = self._pick_deterministic if self._allocation == "deterministic" else self._pick_stochastic
        chosen = {name: 0 for name in self._names}
        for _ in range(self._batch_size):
            name = pick()
            chosen[name] += 1
            self._offsets[name] += 1                       # count the choice now, take the examples below
        batch = []
        for name in self._names:                           # each dataset's share as one block
            self._offsets[name] -= chosen[name]
            batch.extend(self._take(name, chosen[name]))
        return batch

    def _pick_deterministic(self) -> str:
        best = self._names[0]
        for name in self._names[1:]:
            # count[name] / weights[name] < count[best] / weights[best], compared by cross-multiplying
            lhs = self._offsets[name] * self._weights[best]
            rhs = self._offsets[best] * self._weights[name]
            if lhs < rhs or (lhs == rhs and name < best):
                best = name
        return best

    def _pick_stochastic(self) -> str:
        draw = self._rng.randrange(self._total_weight)   # NOTE: an int in [0, W) -- no float edge case
        return self._names[bisect_right(self._prefix, draw)]

    def state_dict(self) -> dict:
        state = super().state_dict()
        if self._allocation == "stochastic":
            version, ints, gauss = self._rng.getstate()
            state["rng_state"] = [version, list(ints), gauss]    # NOTE: tuple -> list, so json.dumps works
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if self._allocation == "stochastic" and "rng_state" in state:
            version, ints, gauss = state["rng_state"]
            self._rng.setstate((version, tuple(ints), gauss))
