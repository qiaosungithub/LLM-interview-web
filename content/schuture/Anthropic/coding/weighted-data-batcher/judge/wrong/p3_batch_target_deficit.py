# expect: WA
# deterministic rule picks the largest deficit against the end-of-batch target, which front-loads heavy datasets inside a batch (error grows with batch_size).
import bisect
import random
from itertools import accumulate


class DataBatcher:
    def __init__(self, registry, weights, batch_size, allocation="deterministic", seed=0):
        self.registry = registry
        self.names = sorted(weights)
        self.w = [weights[n] for n in self.names]
        self.W = sum(self.w)
        self.bs = batch_size
        self.allocation = allocation
        self.prefix = list(accumulate(self.w))
        self.rand = random.Random(seed)
        self.taken = [0] * len(self.names)   # examples taken so far from each dataset (never reduced mod L)

    def _pull(self, k, count):
        it = self.registry.get_iterator(self.names[k], self.taken[k])
        self.taken[k] += count
        return [next(it) for _ in range(count)]

    def _choose(self):
        if self.allocation == "stochastic":
            return bisect.bisect_right(self.prefix, self.rand.randrange(self.W))
        n1 = self._batch_end
        # largest deficit (n+1) w_k / W - count_k, scaled by W; ties go to the alphabetically first name
        return max(range(len(self.w)), key=lambda k: (n1 * self.w[k] - self.W * self.taken[k], -k))

    def next_batch(self):
        if self.bs % self.W == 0:
            out = []
            for k, wk in enumerate(self.w):
                out += self._pull(k, self.bs * wk // self.W)
            return out
        self._batch_end = sum(self.taken) + self.bs
        picks = [self._choose_and_count() for _ in range(self.bs)]
        # fetch the examples, preserving the order of the choices
        start = {}
        for k in picks:
            start.setdefault(k, None)
        need = {k: picks.count(k) for k in start}
        base = {k: self.taken[k] - need[k] for k in need}
        fetched = {k: self._fetch(k, base[k], need[k]) for k in need}
        pos = dict.fromkeys(need, 0)
        out = []
        for k in picks:
            out.append(fetched[k][pos[k]])
            pos[k] += 1
        return out

    def _choose_and_count(self):
        k = self._choose()
        self.taken[k] += 1
        return k

    def _fetch(self, k, offset, count):
        it = self.registry.get_iterator(self.names[k], offset)
        return [next(it) for _ in range(count)]

    def state_dict(self):
        state = {"offsets": dict(zip(self.names, self.taken))}
        if self.allocation == "stochastic":
            version, internal, gauss = self.rand.getstate()
            state["rng"] = {"version": version, "internal": list(internal), "gauss": gauss}
        return state

    def load_state_dict(self, state):
        offsets = state["offsets"]
        self.taken = [int(offsets[n]) for n in self.names]
        if "rng" in state:
            rng = state["rng"]
            self.rand.setstate((rng["version"], tuple(rng["internal"]), rng["gauss"]))
