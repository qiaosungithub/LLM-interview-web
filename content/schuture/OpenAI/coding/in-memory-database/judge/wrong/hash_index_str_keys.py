# expect: WA
# The hash index is keyed by str(value) and the condition it served is not checked again, so ("v", "=", 5) also returns rows holding "5".

from bisect import bisect_left, bisect_right, insort


def _rank(v):
    # int < str; None is handled by callers
    return (1, v) if isinstance(v, str) else (0, v)


def _sort_key(v):
    return (0,) if v is None else (1,) + _rank(v)


class _Table:
    __slots__ = ("cols", "pos", "rows", "hash", "sorted")

    def __init__(self, cols):
        self.cols = list(cols)
        self.pos = {c: i for i, c in enumerate(self.cols)}
        self.rows = []
        self.hash = {}      # col -> {value: [row ids]}
        self.sorted = {}    # col -> list of (rank key, row id)

    def add_to_indexes(self, rid, tup):
        for c, idx in self.hash.items():
            v = tup[self.pos[c]]
            if v is not None:
                idx.setdefault(str(v), []).append(rid)
        for c, idx in self.sorted.items():
            v = tup[self.pos[c]]
            if v is not None:
                insort(idx, (_rank(v), rid))


def _test(actual, op, value):
    if actual is None or value is None:
        return False
    a, b = _rank(actual), _rank(value)
    if op == "=":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    raise ValueError(op)


class Database:
    def __init__(self):
        self._t = {}

    def _table(self, name):
        if name not in self._t:
            raise KeyError(name)
        return self._t[name]

    def create_table(self, table, columns):
        if table in self._t:
            raise KeyError(table)
        self._t[table] = _Table(columns)

    def insert(self, table, row):
        t = self._table(table)
        for k in row:
            if k not in t.pos:
                raise KeyError(k)
        for c in t.cols:
            if c not in row:
                raise ValueError(c)
        tup = tuple(row[c] for c in t.cols)
        rid = len(t.rows)
        t.rows.append(tup)
        t.add_to_indexes(rid, tup)

    def create_index(self, table, column, kind):
        t = self._table(table)
        if column not in t.pos:
            raise KeyError(column)
        p = t.pos[column]
        if kind == "hash":
            idx = {}
            for rid, tup in enumerate(t.rows):
                if tup[p] is not None:
                    idx.setdefault(str(tup[p]), []).append(rid)
            t.hash[column] = idx
        elif kind == "sorted":
            t.sorted[column] = sorted((_rank(tup[p]), rid) for rid, tup in enumerate(t.rows) if tup[p] is not None)
        else:
            raise ValueError(kind)

    def _candidates(self, t, where):
        best = None
        self._used = None
        for c, op, v in where:
            if v is None:
                return []
            ids = None
            if op == "=" and c in t.hash:
                ids = t.hash[c].get(str(v), [])
            elif op in ("<", "<=", ">", ">=") and c in t.sorted:
                idx = t.sorted[c]
                k = _rank(v)
                if op == "<":
                    ids = [rid for _, rid in idx[:bisect_left(idx, (k,))]]
                elif op == "<=":
                    ids = [rid for _, rid in idx[:bisect_right(idx, (k, float("inf")))]]
                elif op == ">":
                    ids = [rid for _, rid in idx[bisect_right(idx, (k, float("inf"))):]]
                else:
                    ids = [rid for _, rid in idx[bisect_left(idx, (k,)):]]
            if ids is not None and (best is None or len(ids) < len(best)):
                best = ids
                self._used = (c, op, v)
        if best is None:
            return range(len(t.rows))
        return sorted(best)

    def select(self, table, columns=None, where=None, order_by=None):
        t = self._table(table)
        out = t.cols if columns is None else list(columns)
        where = list(where or ())
        order_by = list(order_by or ())
        for c in list(out) + [w[0] for w in where] + [o[0] for o in order_by]:
            if c not in t.pos:
                raise KeyError(c)
        cand = self._candidates(t, where)
        rest = list(where)
        if self._used is not None and self._used[1] == "=":
            rest.remove(self._used)         # the hash bucket already holds exactly the matching rows
        conds = [(t.pos[c], op, v) for c, op, v in rest]
        picked = [t.rows[i] for i in cand
                  if all(_test(t.rows[i][p], op, v) for p, op, v in conds)]
        for c, asc in reversed(order_by):
            p = t.pos[c]
            picked.sort(key=lambda tup: _sort_key(tup[p]), reverse=not asc)
        ps = [(c, t.pos[c]) for c in out]
        return [{c: tup[p] for c, p in ps} for tup in picked]
