# expect: WA
# Wrong for Part 4: after growing, put() calls hash_fn on the new key a second time to find its slot in the
# bigger table, so the put that triggers growth calls hash_fn twice.
class _Iter:
    def __init__(self, m, mode):
        self.m, self.mode = m, mode
        self.slots = m._slots
        self.mods = m._mods
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.m._mods != self.mods:
            raise RuntimeError("map changed during iteration")
        slots = self.slots
        while self.i < len(slots):
            e = slots[self.i]
            self.i += 1
            if e is not None:
                return e[1] if self.mode == 0 else (e[1], e[2])
        raise StopIteration


class HashMap:
    def __init__(self, hash_fn=None, eq_fn=None):
        self._h = hash_fn if hash_fn is not None else hash
        self._eq = eq_fn
        self._slots = [None] * 8   # each: [hash, key, value] or None
        self._n = 0
        self._mods = 0

    def _same(self, a, b):
        if a is b:
            return True
        if self._eq is None:
            return a == b
        return self._eq(a, b)

    def _probe(self, key, h):
        slots = self._slots
        mask = len(slots) - 1
        i = h & mask
        while True:
            e = slots[i]
            if e is None:
                return i, None
            if e[0] == h and self._same(e[1], key):
                return i, e
            i = (i + 1) & mask

    def _grow(self):
        old = self._slots
        new = [None] * (2 * len(old))
        mask = len(new) - 1
        for e in old:
            if e is not None:
                j = e[0] & mask
                while new[j] is not None:
                    j = (j + 1) & mask
                new[j] = e
        self._slots = new

    def put(self, key, value):
        h = self._h(key)
        i, e = self._probe(key, h)
        if e is not None:
            e[2] = value
            return
        if 3 * (self._n + 1) > 2 * len(self._slots):
            self._grow()
            h = self._h(key)
            slots = self._slots
            mask = len(slots) - 1
            i = h & mask
            while slots[i] is not None:
                i = (i + 1) & mask
        self._slots[i] = [h, key, value]
        self._n += 1
        self._mods += 1

    def get(self, key, default=None):
        i, e = self._probe(key, self._h(key))
        return default if e is None else e[2]

    def __contains__(self, key):
        return self._probe(key, self._h(key))[1] is not None

    def __len__(self):
        return self._n

    def delete(self, key):
        i, e = self._probe(key, self._h(key))
        if e is None:
            return False
        slots = self._slots
        mask = len(slots) - 1
        hole = i
        j = i
        while True:
            j = (j + 1) & mask
            f = slots[j]
            if f is None:
                break
            home = f[0] & mask
            # f may move into hole iff home is not in the cyclic interval (hole, j]
            if hole <= j:
                inside = hole < home <= j
            else:
                inside = home > hole or home <= j
            if not inside:
                slots[hole] = f
                hole = j
        slots[hole] = None
        self._n -= 1
        self._mods += 1
        return True

    def items(self):
        return _Iter(self, 1)

    def keys(self):
        return _Iter(self, 0)

    def __iter__(self):
        return _Iter(self, 0)
