"""A brute force written from the statement alone (a full copy of the array at every snap), used to compute the
expected outcome of every operation in the random tests, plus the random operation generator."""


class Raises:
    """Expected outcome of an operation that must raise one of `types` (several when more than one rule applies
    and the statement does not say which error wins)."""

    def __init__(self, *types):
        self.types = tuple(types)

    def __repr__(self):
        return " 或 ".join(t.__name__ for t in self.types)


class Brute:
    def __init__(self, length):
        self.length = length
        self.cur = [0] * length
        self.snaps = []
        self.released = -1

    def _snap_errors(self, s):
        if not (0 <= s < len(self.snaps)):
            return {IndexError}
        if s <= self.released:
            return {ValueError}
        return set()

    def step(self, op):
        """Applies op and returns its expected result: a value, None, or Raises(...)."""
        kind = op[0]
        if kind == "set":
            _, i, v = op
            if not (0 <= i < self.length):
                return Raises(IndexError)
            self.cur[i] = v
            return None
        if kind == "snap":
            self.snaps.append(list(self.cur))
            return len(self.snaps) - 1
        if kind == "get":
            _, i, s = op
            errs = set() if 0 <= i < self.length else {IndexError}
            errs |= self._snap_errors(s)
            if errs:
                return Raises(*sorted(errs, key=lambda t: t.__name__))
            return self.snaps[s][i]
        if kind == "release":
            _, s = op
            if not (0 <= s < len(self.snaps)):
                return Raises(IndexError)
            self.released = max(self.released, s)
            return None
        if kind == "diff":
            _, a, b = op
            errs = self._snap_errors(a) | self._snap_errors(b)
            if not a < b:
                errs.add(ValueError)
            if errs:
                return Raises(*sorted(errs, key=lambda t: t.__name__))
            sa, sb = self.snaps[a], self.snaps[b]
            return {i: (sa[i], sb[i]) for i in range(self.length) if sa[i] != sb[i]}
        raise AssertionError(kind)


def random_ops(r, length, n_ops, kinds):
    """A random sequence of operations drawn from `kinds`, with a share of invalid indices and snapshot ids."""
    weights = {"set": 40, "snap": 20, "get": 25, "release": 6, "diff": 12}
    pool = [k for k in kinds for _ in range(weights[k])]
    ops, taken = [], 0

    def some_id():
        x = r.random()
        if x < 0.08:
            return r.choice([-1, taken, taken + 3])        # never taken: negative, the pending id, far ahead
        return r.randrange(taken) if taken else 0

    for _ in range(n_ops):
        kind = r.choice(pool)
        if kind == "set":
            i = r.randrange(length) if length and r.random() < 0.93 else r.choice([-1, length])
            ops.append(("set", i, r.choice([0, 0, r.randint(-9, 9), r.randint(-10 ** 12, 10 ** 12)])))
        elif kind == "snap":
            ops.append(("snap",))
            taken += 1
        elif kind == "get":
            i = r.randrange(length) if length and r.random() < 0.93 else r.choice([-1, length])
            ops.append(("get", i, some_id()))
        elif kind == "release":
            x = r.random()
            if taken and x < 0.4:
                ops.append(("release", r.randrange(taken)))
            elif taken and x < 0.85:
                ops.append(("release", r.randrange(max(0, taken - 4), taken)))
            else:
                ops.append(("release", some_id()))
        else:
            a, b = some_id(), some_id()
            if r.random() < 0.8 and a > b:
                a, b = b, a
            ops.append(("diff", a, b))
    return ops
