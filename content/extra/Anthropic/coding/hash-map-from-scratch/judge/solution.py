"""The page's reference answer, joined: LinearProbingMap (Part 1) -> TombstoneMap (Part 2) -> IterableMap
(Part 3) -> HashMap (Part 4), plus the Part 2 alternative ShiftDeleteMap and Part 4's mix64."""


_EMPTY = object()   # a slot that has never held a key: every probe sequence stops here


class LinearProbingMap:
    """Part 1: open addressing with linear probing. Grows by doubling; no deletion yet."""

    def __init__(self) -> None:
        self._allocate(8)

    def _allocate(self, capacity: int) -> None:
        self._capacity = capacity                 # always a power of two
        self._hashes = [0] * capacity             # hash of the key in each slot (unused if no key)
        self._keys = [_EMPTY] * capacity
        self._values = [None] * capacity
        self._size = 0                            # number of keys stored

    def _hash(self, key) -> int:
        return hash(key)

    def _same(self, a, b) -> bool:
        return a is b or a == b

    def _find(self, key, h: int):
        """(slot, True) if key is stored in slot; otherwise (slot, False), where slot is the
        empty slot that ended the probe sequence -- which is where key would be inserted."""
        mask = self._capacity - 1
        i = h & mask                              # NOTE: for a negative h, & still lands in [0, mask]
        while True:
            k = self._keys[i]
            if k is _EMPTY:
                return i, False
            if self._hashes[i] == h and self._same(k, key):   # compare the stored hash first
                return i, True
            i = (i + 1) & mask

    def _entries(self):
        """(hash, key, value) of every stored key, in slot order."""
        for i in range(self._capacity):
            if self._keys[i] is not _EMPTY:
                yield self._hashes[i], self._keys[i], self._values[i]

    def _free_slot(self, h: int) -> int:
        """The first empty slot on the probe sequence of hash h, for a key known to be absent."""
        mask = self._capacity - 1
        i = h & mask
        while self._keys[i] is not _EMPTY:
            i = (i + 1) & mask
        return i

    def _rebuild(self, capacity: int) -> None:
        entries = list(self._entries())
        self._allocate(capacity)
        for h, key, value in entries:
            i = self._free_slot(h)                # the keys are distinct: no comparison needed
            self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size = len(entries)

    def put(self, key, value) -> None:
        h = self._hash(key)
        i, found = self._find(key, h)
        if found:
            self._values[i] = value               # overwrite: the stored key object is kept
            return
        if 3 * (self._size + 1) > 2 * self._capacity:
            self._rebuild(2 * self._capacity)
            i = self._free_slot(h)                # NOTE: the slot found before the rebuild is stale
        self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size += 1

    def get(self, key, default=None):
        i, found = self._find(key, self._hash(key))
        return self._values[i] if found else default

    def __contains__(self, key) -> bool:
        return self._find(key, self._hash(key))[1]

    def __len__(self) -> int:
        return self._size


_TOMB = object()    # a slot whose key was deleted: probe sequences walk past it


class TombstoneMap(LinearProbingMap):
    """Part 2: deletion with tombstones."""

    def _allocate(self, capacity: int) -> None:
        super()._allocate(capacity)
        self._tombs = 0                           # slots holding _TOMB

    def _find(self, key, h: int):
        """(slot, True) if key is stored in slot; otherwise (slot, False), where slot is where key
        should be inserted: the first tombstone passed, or else the empty slot that ended the search."""
        mask = self._capacity - 1
        i = h & mask
        first_tomb = -1
        while True:
            k = self._keys[i]
            if k is _EMPTY:
                return (i if first_tomb < 0 else first_tomb), False
            if k is _TOMB:
                if first_tomb < 0:
                    first_tomb = i                # NOTE: keep walking -- the key may still be further on
            elif self._hashes[i] == h and self._same(k, key):
                return i, True
            i = (i + 1) & mask

    def _entries(self):
        for i in range(self._capacity):
            k = self._keys[i]
            if k is not _EMPTY and k is not _TOMB:
                yield self._hashes[i], k, self._values[i]

    def put(self, key, value) -> None:
        h = self._hash(key)
        i, found = self._find(key, h)
        if found:
            self._values[i] = value
            return
        if self._keys[i] is _TOMB:
            self._tombs -= 1                      # reusing a tombstone uses up no empty slot
        elif 3 * (self._size + self._tombs + 1) > 2 * self._capacity:   # NOTE: tombstones count
            capacity = 8
            while capacity < 2 * (self._size + 1):                      # NOTE: sized by live keys only
                capacity *= 2
            self._rebuild(capacity)               # drops every tombstone
            i = self._free_slot(h)
        self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size += 1

    def delete(self, key) -> bool:
        i, found = self._find(key, self._hash(key))
        if not found:
            return False
        self._keys[i] = _TOMB                     # NOTE: not _EMPTY -- other keys may sit beyond this slot
        self._values[i] = None                    # drop the reference to the value
        self._size -= 1
        self._tombs += 1
        return True


class ShiftDeleteMap(LinearProbingMap):
    """Part 2, the alternative: backward-shift deletion. No tombstones, so put() is unchanged."""

    def delete(self, key) -> bool:
        i, found = self._find(key, self._hash(key))
        if not found:
            return False
        mask = self._capacity - 1
        j = i                                     # i is the hole; j scans the rest of the run
        while True:
            j = (j + 1) & mask
            if self._keys[j] is _EMPTY:
                break
            home = self._hashes[j] & mask
            # distances measured backwards from j, modulo capacity: home is outside (i, j]
            # exactly when home is at least as far back as the hole is
            if ((j - home) & mask) >= ((j - i) & mask):
                self._hashes[i], self._keys[i], self._values[i] = \
                    self._hashes[j], self._keys[j], self._values[j]
                i = j                             # the hole moves to j
        self._keys[i] = _EMPTY
        self._values[i] = None
        self._size -= 1
        return True


class IterableMap(TombstoneMap):
    """Part 3: iteration in slot order; adding or removing a key during iteration is detected."""

    def __init__(self) -> None:
        super().__init__()
        self._version = 0                         # bumped whenever a key is added or removed

    def put(self, key, value) -> None:
        before = self._size
        super().put(key, value)
        if self._size != before:
            self._version += 1

    def delete(self, key) -> bool:
        removed = super().delete(key)
        if removed:
            self._version += 1
        return removed

    def items(self):
        return self._walk(self._version, self._keys, self._values)

    def _walk(self, version: int, keys: list, values: list):
        if self._version != version:
            raise RuntimeError("HashMap changed size during iteration")
        for i in range(len(keys)):
            k = keys[i]
            if k is not _EMPTY and k is not _TOMB:
                yield k, values[i]
                if self._version != version:      # checked when the caller asks for the next item
                    raise RuntimeError("HashMap changed size during iteration")

    def keys(self):
        return (k for k, _ in self.items())

    def __iter__(self):
        return self.keys()


class HashMap(IterableMap):
    """Part 4: caller-supplied hash and equality."""

    def __init__(self, hash_fn=None, eq_fn=None) -> None:
        self._hash_fn = hash if hash_fn is None else hash_fn
        self._eq_fn = eq_fn
        super().__init__()

    def _hash(self, key) -> int:
        return self._hash_fn(key)

    def _same(self, a, b) -> bool:
        if a is b:
            return True
        return a == b if self._eq_fn is None else self._eq_fn(a, b)


_MASK64 = (1 << 64) - 1


def mix64(key) -> int:
    """hash(key) pushed through the splitmix64 finaliser, so that high input bits reach the low bits."""
    x = hash(key) & _MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)
