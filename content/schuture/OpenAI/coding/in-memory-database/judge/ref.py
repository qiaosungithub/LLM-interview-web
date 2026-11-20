"""A literal model of the statement, and random call sequences for the tests.

A call is (method, args, kwargs), e.g. ("select", ("t", ["a"]), {"where": [("a", "<", 3)]}). Model.call returns
what the statement says the call returns, or a Raises when it must raise. Indexes (Part 4) never change a result,
so the model ignores them apart from checking create_index's names."""
import functools

OPS = ["=", "!=", "<", "<=", ">", ">="]


class Raises:
    """The expected outcome of a call that must raise `exc`."""

    def __init__(self, exc):
        self.exc = exc

    def __repr__(self):
        return self.exc.__name__


KEY, VALUE = Raises(KeyError), Raises(ValueError)


def compare(a, b):
    """-1, 0 or 1 for two values that are not None: same types compare as Python does, every int < every str."""
    if isinstance(a, str) != isinstance(b, str):
        return 1 if isinstance(a, str) else -1
    return (a > b) - (a < b)


def holds(actual, op, value):
    """Whether the condition (column, op, value) holds for a row whose value in that column is `actual`."""
    if actual is None or value is None:
        return False                    # None fails every operator, "!=" included
    c = compare(actual, value)
    return {"=": c == 0, "!=": c != 0, "<": c < 0, "<=": c <= 0, ">": c > 0, ">=": c >= 0}[op]


def order(a, b):
    """compare() with None below every int and every str (ORDER BY)."""
    if a is None or b is None:
        return (a is not None) - (b is not None)
    return compare(a, b)


class Model:
    def __init__(self):
        self.tables = {}                # name -> (columns, rows in insertion order)

    def call(self, method, args, kwargs):
        return getattr(self, method)(*args, **kwargs)

    def create_table(self, table, columns):
        if table in self.tables:
            return KEY
        self.tables[table] = (list(columns), [])
        return None

    def insert(self, table, row):
        if table not in self.tables:
            return KEY
        columns, rows = self.tables[table]
        if any(k not in columns for k in row):
            return KEY                  # an unknown key is reported before a missing column
        if any(c not in row for c in columns):
            return VALUE
        rows.append(dict(row))
        return None

    def select(self, table, columns=None, where=None, order_by=None):
        if table not in self.tables:
            return KEY
        schema, rows = self.tables[table]
        out = list(schema) if columns is None else list(columns)
        where, order_by = list(where or []), list(order_by or [])
        names = out + [c for c, _, _ in where] + [c for c, _ in order_by]
        if any(c not in schema for c in names):
            return KEY
        rows = [r for r in rows if all(holds(r[c], op, v) for c, op, v in where)]

        def cmp(r1, r2):
            for c, ascending in order_by:
                d = order(r1[c], r2[c])
                if d:
                    return d if ascending else -d
            return 0                    # a full tie: sorted() is stable, so insertion order stays
        rows = sorted(rows, key=functools.cmp_to_key(cmp))
        return [{c: r[c] for c in out} for r in rows]

    def create_index(self, table, column, kind):
        if table not in self.tables or column not in self.tables[table][0]:
            return KEY
        return None


# ---------------------------------------------------------------- random sequences

INTS = [-3, -2, -1, 0, 1, 2, 3]
STRS = ["", "a", "b", "ab", "B", "0", "10"]
BIG = [10 ** 12, -10 ** 12]


def _value(r, kind):
    if r.random() < 0.2:
        return None
    if kind == "mixed":
        kind = r.choice(["int", "str"])
    if kind == "int":
        return r.choice(BIG) if r.random() < 0.03 else r.choice(INTS)
    return r.choice(STRS)


def random_ops(r, part, n, tables=2):
    """n random calls allowed in `part`, starting with create_table for one or two tables."""
    schema, kinds, seen = {}, {}, {}
    ops = []
    for name in ["t", "u"][: r.randint(1, tables)]:
        cols = r.sample(["a", "b", "c", "d"], r.randint(2, 4))
        schema[name] = cols
        kinds[name] = {c: r.choice(["int", "str", "mixed"]) for c in cols}
        seen[name] = {c: [] for c in cols}
        ops.append(("create_table", (name, list(cols)), {}))

    def table():
        return r.choice(list(schema))

    def row(t):
        out = {c: _value(r, kinds[t][c]) for c in schema[t]}
        keys = list(out)
        r.shuffle(keys)                 # the keys of a row may come in any order
        return {k: out[k] for k in keys}

    def value(t, c):
        if seen[t][c] and r.random() < 0.5:
            return r.choice(seen[t][c])
        return _value(r, "mixed")

    weights = {"insert": 6, "bad_insert": 1, "select": 5, "bad_select": 1, "dup_table": 0.3,
               "index": 1.5 if part >= 4 else 0}
    names = list(weights)
    while len(ops) < n:
        kind = r.choices(names, [weights[x] for x in names])[0]
        t = table()
        if kind == "insert":
            rw = row(t)
            for c, v in rw.items():
                seen[t][c].append(v)
            ops.append(("insert", (t, rw), {}))
        elif kind == "bad_insert":
            rw = row(t)
            how = r.choice(["missing", "extra", "both", "table"])
            if how in ("missing", "both"):
                del rw[r.choice(list(rw))]
            if how in ("extra", "both"):
                rw["e"] = _value(r, "mixed")
            ops.append(("insert", ("nope" if how == "table" else t, rw), {}))
        elif kind in ("select", "bad_select"):
            cols = schema[t]
            columns = None if r.random() < 0.3 else r.sample(cols, r.randint(1, len(cols)))
            kwargs = {}
            if part >= 2 and r.random() < 0.8:
                where = []
                for _ in range(r.randint(0, 3)):
                    c = r.choice(cols)
                    where.append((c, r.choice(OPS), value(t, c)))
                kwargs["where"] = where
            if part >= 3 and r.random() < 0.8:
                k = r.randint(0, 3)
                picked = r.sample(cols, min(k, len(cols))) if r.random() < 0.9 else r.choices(cols, k=k)
                kwargs["order_by"] = [(c, r.random() < 0.5) for c in picked]
            args = (t,) if columns is None else (t, columns)
            if kind == "bad_select":
                spots = ["table", "columns"] + (["where"] if part >= 2 else []) + (["order_by"] if part >= 3 else [])
                spot = r.choice(spots)
                if spot == "table":
                    args = ("nope",) + args[1:]
                elif spot == "columns":
                    args = (t, (columns or [])[:1] + ["e"])
                elif spot == "where":
                    kwargs["where"] = list(kwargs.get("where") or []) + [("e", r.choice(OPS), 1)]
                else:
                    kwargs["order_by"] = list(kwargs.get("order_by") or []) + [("e", True)]
            ops.append(("select", args, kwargs))
        elif kind == "dup_table":
            ops.append(("create_table", (t, list(schema[t])), {}))
        else:
            bad = r.random() < 0.1
            ops.append(("create_index", ("nope" if bad and r.random() < 0.5 else t,
                                         "e" if bad else r.choice(schema[t]), r.choice(["hash", "sorted"])), {}))
    return ops
