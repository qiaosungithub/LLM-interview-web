# expect: TLE
# Keeps the entries in a plain list and scans it on every operation: correct results, but O(n) per operation.


class HashMap:
    def __init__(self, hash_fn=None, eq_fn=None) -> None:
        self._hash = hash if hash_fn is None else hash_fn
        self._eq = eq_fn
        self._entries = []                        # [hash, key, value] in insertion order
        self._version = 0

    def _same(self, a, b) -> bool:
        return a is b or (a == b if self._eq is None else self._eq(a, b))

    def _index(self, key):
        h = self._hash(key)
        for i, (hh, k, _) in enumerate(self._entries):
            if hh == h and self._same(k, key):
                return h, i
        return h, -1

    def put(self, key, value) -> None:
        h, i = self._index(key)
        if i >= 0:
            self._entries[i][2] = value
        else:
            self._entries.append([h, key, value])
            self._version += 1

    def get(self, key, default=None):
        i = self._index(key)[1]
        return self._entries[i][2] if i >= 0 else default

    def __contains__(self, key) -> bool:
        return self._index(key)[1] >= 0

    def __len__(self) -> int:
        return len(self._entries)

    def delete(self, key) -> bool:
        i = self._index(key)[1]
        if i < 0:
            return False
        self._entries.pop(i)
        self._version += 1
        return True

    def items(self):
        version = self._version
        for _, k, v in list(self._entries):
            if self._version != version:
                raise RuntimeError("HashMap changed size during iteration")
            yield k, v
        if self._version != version:
            raise RuntimeError("HashMap changed size during iteration")

    def keys(self):
        return (k for k, _ in self.items())

    def __iter__(self):
        return self.keys()
