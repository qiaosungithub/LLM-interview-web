"""A literal model of the statement, used for the expected values of the random tests, and the random
call-sequence generator. The whole database is one flat dict (key, field) -> (value, end), where end is None for a
field without a TTL. After every timestamped call the model keeps a full copy of that dict; get_when reads the last
copy made at or before at_timestamp. Nothing is indexed and every read re-scans what it needs."""


class InMemoryDatabase:
    def __init__(self):
        self.rows = {}          # (key, field) -> (value, end); end is None when the field never expires
        self.backups = []       # (timestamp, {(key, field): (value, time left or None)}), oldest first
        self.copies = []        # (timestamp, copy of self.rows) after every timestamped call, oldest first

    @staticmethod
    def _alive(row, t):
        """Whether the row exists and has not expired at t; t None means "do not look at expiry"."""
        return row is not None and (t is None or row[1] is None or t < row[1])

    def _value(self, key, field, t):
        row = self.rows.get((key, field))
        return row[0] if self._alive(row, t) else ""

    def _scan(self, key, prefix, t):
        found = sorted((f, row[0]) for (k, f), row in self.rows.items()
                       if k == key and f[:len(prefix)] == prefix and self._alive(row, t))
        return ", ".join(f"{f}({v})" for f, v in found)

    def _conditional(self, key, field, expected, t, new_row):
        """new_row None deletes the field; otherwise it becomes the field's row."""
        if self._value(key, field, t) == "" or self._value(key, field, t) != expected:
            return "false"
        if new_row is None:
            del self.rows[(key, field)]
        else:
            self.rows[(key, field)] = new_row
        return "true"

    def _done(self, t, out):
        self.copies.append((t, dict(self.rows)))
        return out

    # Level 1-2, no clock: expiry is never looked at
    def set(self, key, field, value):
        self.rows[(key, field)] = (value, None)
        return ""

    def get(self, key, field):
        return self._value(key, field, None)

    def delete(self, key, field):
        if (key, field) not in self.rows:
            return "false"
        del self.rows[(key, field)]
        return "true"

    def compare_and_set(self, key, field, expected_value, new_value):
        return self._conditional(key, field, expected_value, None, (new_value, None))

    def compare_and_delete(self, key, field, expected_value):
        return self._conditional(key, field, expected_value, None, None)

    def scan(self, key):
        return self._scan(key, "", None)

    def scan_by_prefix(self, key, prefix):
        return self._scan(key, prefix, None)

    # Level 3
    def set_at(self, key, field, value, timestamp):
        self.rows[(key, field)] = (value, None)
        return self._done(timestamp, "")

    def set_at_with_ttl(self, key, field, value, timestamp, ttl):
        self.rows[(key, field)] = (value, timestamp + ttl)
        return self._done(timestamp, "")

    def get_at(self, key, field, timestamp):
        return self._done(timestamp, self._value(key, field, timestamp))

    def delete_at(self, key, field, timestamp):
        if self._value(key, field, timestamp) == "":
            return self._done(timestamp, "false")
        del self.rows[(key, field)]
        return self._done(timestamp, "true")

    def scan_at(self, key, timestamp):
        return self._done(timestamp, self._scan(key, "", timestamp))

    def scan_by_prefix_at(self, key, prefix, timestamp):
        return self._done(timestamp, self._scan(key, prefix, timestamp))

    def compare_and_set_at(self, key, field, expected_value, new_value, timestamp):
        return self._done(timestamp, self._conditional(key, field, expected_value, timestamp, (new_value, None)))

    def compare_and_set_at_with_ttl(self, key, field, expected_value, new_value, timestamp, ttl):
        return self._done(timestamp, self._conditional(key, field, expected_value, timestamp,
                                                       (new_value, timestamp + ttl)))

    def compare_and_delete_at(self, key, field, expected_value, timestamp):
        return self._done(timestamp, self._conditional(key, field, expected_value, timestamp, None))

    # Level 4
    def backup(self, timestamp):
        saved = {kf: (row[0], None if row[1] is None else row[1] - timestamp)
                 for kf, row in self.rows.items() if self._alive(row, timestamp)}
        self.backups.append((timestamp, saved))
        return self._done(timestamp, str(len({key for key, _ in saved})))

    def restore(self, timestamp, timestamp_to_restore):
        saved = [s for when, s in self.backups if when <= timestamp_to_restore][-1]
        self.rows = {kf: (value, None if left is None else timestamp + left) for kf, (value, left) in saved.items()}
        return self._done(timestamp, "")

    def get_when(self, timestamp, key, field, at_timestamp):
        then = {}
        for when, rows in self.copies:
            if when <= at_timestamp:
                then = rows
        row = then.get((key, field))
        return self._done(timestamp, row[0] if self._alive(row, at_timestamp) else "")


# ---------------------------------------------------------------- random call sequences

KEYS = ["k", "kk", "m"]
FIELDS = {1: ["a", "b", "c"], 2: ["a", "ab", "abc", "b", "ba", "B", "A", "a1"]}
VALUES = ["x", "y", "z", "X"]
PREFIXES = ["a", "ab", "abc", "b", "ba", "B", "A", "c"]

KINDS = {   # method -> weight
    1: {"set": 4, "get": 3, "delete": 2, "compare_and_set": 3, "compare_and_delete": 2},
    2: {"set": 4, "get": 2, "delete": 2, "compare_and_set": 2, "compare_and_delete": 2, "scan": 2,
        "scan_by_prefix": 3},
    3: {"set_at": 3, "set_at_with_ttl": 5, "get_at": 3, "delete_at": 2, "scan_at": 2, "scan_by_prefix_at": 2,
        "compare_and_set_at": 2, "compare_and_set_at_with_ttl": 2, "compare_and_delete_at": 2,
        "get": 1, "scan": 1, "scan_by_prefix": 1},
    4: {"set_at": 3, "set_at_with_ttl": 5, "get_at": 2, "delete_at": 2, "scan_at": 2, "scan_by_prefix_at": 1,
        "compare_and_set_at": 1, "compare_and_set_at_with_ttl": 1, "compare_and_delete_at": 1,
        "get": 1, "scan": 1, "backup": 3, "restore": 3, "get_when": 6},
}
# the Level 3 parameters the statement says may be passed by keyword
KEYWORD_OK = ("prefix", "timestamp", "ttl")
PARAMS = {
    "set_at": ("key", "field", "value", "timestamp"),
    "set_at_with_ttl": ("key", "field", "value", "timestamp", "ttl"),
    "get_at": ("key", "field", "timestamp"),
    "delete_at": ("key", "field", "timestamp"),
    "scan_at": ("key", "timestamp"),
    "scan_by_prefix_at": ("key", "prefix", "timestamp"),
    "compare_and_set_at": ("key", "field", "expected_value", "new_value", "timestamp"),
    "compare_and_set_at_with_ttl": ("key", "field", "expected_value", "new_value", "timestamp", "ttl"),
    "compare_and_delete_at": ("key", "field", "expected_value", "timestamp"),
}


def _keywords(r, method, args):
    """Splits args into (positional, keyword) the way a test may: some trailing prefix/timestamp/ttl by name."""
    names = PARAMS.get(method)
    if names is None or r.random() < 0.6:
        return tuple(args), {}
    cut = len(names)
    while cut > 0 and names[cut - 1] in KEYWORD_OK:
        cut -= 1
    cut = r.randint(cut, len(names) - 1)
    kw = list(zip(names[cut:], args[cut:]))
    if r.random() < 0.3:
        r.shuffle(kw)
    return tuple(args[:cut]), dict(kw)


def random_ops(r, level, n_ops):
    """A sequence of (method, args, kwargs) calls for one database, drawn while a model follows along so that the
    draws hit the interesting cases often: calls just before, exactly at and just after a field expires (usually on
    that field), conditional calls with the stored value of an expired field, plain reads of expired fields,
    restores to exactly a backup's time and to times between backups, and get_when on the edges of writes,
    deletes, expiries and restores."""
    m = InMemoryDatabase()
    fields = FIELDS[min(level, 2)]
    kinds, weights = zip(*KINDS[level].items())
    ops, now, marks = [], 0, []

    def draw_key_field():
        if m.rows and r.random() < 0.5:
            return r.choice(sorted(m.rows))
        return r.choice(KEYS), r.choice(fields)

    for _ in range(n_ops):
        kind = r.choices(kinds, weights)[0]
        key, field = draw_key_field()
        if level >= 3:
            ends = [(kf, row[1] + d) for kf, row in sorted(m.rows.items()) if row[1] is not None
                    for d in (-1, 0, 0, 1) if row[1] + d > now]
            if ends and r.random() < 0.35:
                kf, now = r.choice(ends)
                if r.random() < 0.7:
                    key, field = kf
            else:
                now += r.choice([1, 1, 2, 3])
            marks.append(now)
            if kind == "restore" and not m.backups:
                kind = "backup"
        value = r.choice(VALUES)
        stored = m.rows.get((key, field))
        expected = stored[0] if stored is not None and r.random() < 0.6 else r.choice(VALUES)
        prefix = r.choice(PREFIXES)
        ttl = r.choice([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        args = {
            "set": (key, field, value), "get": (key, field), "delete": (key, field),
            "compare_and_set": (key, field, expected, value), "compare_and_delete": (key, field, expected),
            "scan": (key,), "scan_by_prefix": (key, prefix),
            "set_at": (key, field, value, now), "set_at_with_ttl": (key, field, value, now, ttl),
            "get_at": (key, field, now), "delete_at": (key, field, now), "scan_at": (key, now),
            "scan_by_prefix_at": (key, prefix, now),
            "compare_and_set_at": (key, field, expected, value, now),
            "compare_and_set_at_with_ttl": (key, field, expected, value, now, ttl),
            "compare_and_delete_at": (key, field, expected, now),
            "backup": (now,),
        }.get(kind)
        if kind == "restore":
            args = (now, min(now - 1, r.choice(m.backups)[0] + r.choice([0, 0, 1, 2])))
        elif kind == "get_when":
            if r.random() < 0.4:
                seen = sorted({kf for _, rows in m.copies for kf in rows})
                if seen:
                    key, field = r.choice(seen)
            near = [t + d for t in marks for d in (-1, 0, 1) if 1 <= t + d <= now]
            args = (now, key, field, r.choice(near) if near and r.random() < 0.9 else now)
        args, kwargs = _keywords(r, kind, args)
        ops.append((kind, args, kwargs))
        getattr(m, kind)(*args, **kwargs)
        if kind in ("set_at_with_ttl", "compare_and_set_at_with_ttl", "restore"):
            marks.extend(row[1] for row in m.rows.values() if row[1] is not None)
    return ops
