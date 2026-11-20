"""A literal model of the statement, used for the expected values of the random tests, and the random
operation-sequence generator.

State is one flat dict (key, field) -> (value, expiry), where expiry is None for a permanent field or the
first timestamp at which the field is gone; every query re-checks expiry, nothing is cleaned up."""


class Database:
    def __init__(self):
        self.fields = {}        # (key, field) -> (value, expiry or None)
        self.backups = {}       # backup timestamp -> {(key, field): (value, remaining or None)}
        self.clock = 0
        self.loose = False      # since the last _at call, some backup/restore moved time past the clock

    # ---- calls outside the statement's guarantees
    def ambiguous(self, method, args):
        """True for a call the statement does not cover, which a shortened random sequence can reach:
        a restore with no backup at or before timestamp_to_restore, or a plain call after a backup or restore
        that moved time forward with no _at call in between (the statement only says that the _at twins set
        the clock, so the time such a plain call happens at is open)."""
        if method == "restore":
            return not any(b <= args[1] for b in self.backups)
        return method in PLAIN and self.loose

    # ---- time
    def _at(self, ts):
        assert ts >= self.clock, (ts, self.clock)
        self.clock = ts
        self.loose = False

    def _plain(self, twin, *args):
        """A plain call happens at the clock's current value; the clock then advances by 1."""
        ts = self.clock
        out = twin(ts, *args)
        self.clock = ts + 1
        return out

    def _live(self, key, field, ts):
        entry = self.fields.get((key, field))
        if entry is None or (entry[1] is not None and ts >= entry[1]):
            return None
        return entry[0]

    def _scan(self, key, ts, prefix):
        names = sorted(f for (k, f) in self.fields
                       if k == key and f.startswith(prefix) and self._live(k, f, ts) is not None)
        return ", ".join(f"{f}({self._live(key, f, ts)})" for f in names)

    # ---- Level 3: the _at twins
    def set_at(self, ts, key, field, value):
        self._at(ts)
        self.fields[(key, field)] = (value, None)
        return ""

    def set_at_with_ttl(self, ts, key, field, value, ttl):
        self._at(ts)
        self.fields[(key, field)] = (value, ts + ttl)
        return ""

    def get_at(self, ts, key, field):
        self._at(ts)
        value = self._live(key, field, ts)
        return "" if value is None else value

    def delete_at(self, ts, key, field):
        self._at(ts)
        if self._live(key, field, ts) is None:
            return "false"
        del self.fields[(key, field)]
        return "true"

    def scan_at(self, ts, key):
        self._at(ts)
        return self._scan(key, ts, "")

    def scan_by_prefix_at(self, ts, key, prefix):
        self._at(ts)
        return self._scan(key, ts, prefix)

    # ---- Levels 1 and 2: the same calls at the clock, which then advances by 1
    def set(self, key, field, value):
        return self._plain(self.set_at, key, field, value)

    def get(self, key, field):
        return self._plain(self.get_at, key, field)

    def delete(self, key, field):
        return self._plain(self.delete_at, key, field)

    def scan(self, key):
        return self._plain(self.scan_at, key)

    def scan_by_prefix(self, key, prefix):
        return self._plain(self.scan_by_prefix_at, key, prefix)

    # ---- Level 4
    def _moved(self, ts):
        # Sticky until the next _at call: a second backup/restore at the same timestamp does not settle the clock
        # for an answer whose backup/restore leave the clock alone.
        loose = self.loose or ts > self.clock
        self._at(ts)
        self.loose = loose

    def backup(self, ts):
        self._moved(ts)
        snap = {}
        for (key, field), (value, expiry) in self.fields.items():
            if self._live(key, field, ts) is not None:
                snap[(key, field)] = (value, None if expiry is None else expiry - ts)
        self.backups[ts] = snap
        return ""

    def restore(self, ts, ts_to_restore):
        self._moved(ts)
        best = max(b for b in self.backups if b <= ts_to_restore)
        self.fields = {pair: (value, None if left is None else ts + left)
                       for pair, (value, left) in self.backups[best].items()}
        return ""


PLAIN = ("set", "get", "delete", "scan", "scan_by_prefix")

KEYS = (["k1", "k2", "k3"], [6, 4, 1])
FIELDS = {
    1: ["a", "b", "c"],
    2: ["a", "ab", "abc", "b", "B", "a b", "a&b", "10", "9"],
    3: ["a", "ab", "b", "B", "a b"],
}
FIELDS[4] = FIELDS[3]
PREFIXES = {
    1: [""],
    2: ["", "", "a", "a", "ab", "b", "B", "1", "a ", "z"],
    3: ["", "", "a", "a", "ab", "b", "B", "z"],
}
PREFIXES[4] = PREFIXES[3]
VALUES = ["1", "2", "x", "12.50", "v v"]

KINDS = {
    1: {"set": 5, "get": 4, "delete": 3},
    2: {"set": 5, "get": 2, "delete": 3, "scan": 3, "scan_by_prefix": 3},
    3: {"set": 1, "get": 1, "delete": 1, "scan": 1, "scan_by_prefix": 1,
        "set_at": 4, "set_at_with_ttl": 6, "get_at": 4, "delete_at": 3, "scan_at": 3, "scan_by_prefix_at": 3},
}
KINDS[4] = dict(KINDS[3], backup=3, restore=3)


def random_ops(r, level, n_ops):
    """A sequence of (method, args, kwargs) calls for one InMemoryDatabase, drawn while a model follows along:
    explicit timestamps never go below the clock and often repeat it, TTLs are short enough to expire within
    the sequence, a restore always has a backup to use, and no plain call follows a backup or restore that
    moved time forward until an _at call has set the clock again."""
    m = Database()
    kinds, weights = zip(*KINDS[level].items())
    ops = []

    def ts():
        return m.clock + r.choice([0, 0, 0, 1, 1, 2, 3])

    while len(ops) < n_ops:
        kind = r.choices(kinds, weights)[0]
        if (kind in PLAIN and m.loose) or (kind == "restore" and not m.backups):
            continue
        key = r.choices(*KEYS)[0]
        field = r.choice(FIELDS[level])
        value = r.choice(VALUES)
        prefix = r.choice(PREFIXES[level])
        if kind == "set":
            args = (key, field, value)
        elif kind in ("get", "delete"):
            args = (key, field)
        elif kind == "scan":
            args = (key,)
        elif kind == "scan_by_prefix":
            args = (key, prefix)
        elif kind == "set_at":
            args = (ts(), key, field, value)
        elif kind == "set_at_with_ttl":
            args = (ts(), key, field, value, r.randint(1, 5))
        elif kind in ("get_at", "delete_at"):
            args = (ts(), key, field)
        elif kind == "scan_at":
            args = (ts(), key)
        elif kind == "scan_by_prefix_at":
            args = (ts(), key, prefix)
        elif kind == "backup":
            args = (ts(),)
        else:
            t = ts()
            times = sorted(m.backups)
            target = r.choice(times + [r.randint(times[0], t)])
            args = (t, target)
        assert not m.ambiguous(kind, args)
        getattr(m, kind)(*args)
        ops.append((kind, args, {}))
    return ops
