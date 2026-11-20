# expect: TLE
# Ignores indexes; filters with one list comprehension per condition, specialised per operator, over the whole table.
_RANK = {int: 0, str: 1}


def _k(v):
    return (_RANK[type(v)], v)


class Database:
    def __init__(self):
        self.tables = {}

    def _get(self, t):
        if t not in self.tables:
            raise KeyError(t)
        return self.tables[t]

    def create_table(self, table, columns):
        if table in self.tables:
            raise KeyError(table)
        self.tables[table] = (list(columns), [])

    def insert(self, table, row):
        cols, rows = self._get(table)
        for k in row:
            if k not in cols:
                raise KeyError(k)
        for c in cols:
            if c not in row:
                raise ValueError(c)
        rows.append(dict(row))

    def create_index(self, table, column, kind):
        cols, _ = self._get(table)
        if column not in cols:
            raise KeyError(column)

    def select(self, table, columns=None, where=None, order_by=None):
        cols, rows = self._get(table)
        out = list(cols) if columns is None else list(columns)
        where, order_by = where or [], order_by or []
        for c in out + [w[0] for w in where] + [o[0] for o in order_by]:
            if c not in cols:
                raise KeyError(c)
        for c, op, v in where:
            if v is None:
                rows = []
            elif op == "=":
                rows = [r for r in rows if r[c] == v]
            elif op == "!=":
                rows = [r for r in rows if r[c] is not None and r[c] != v]
            else:
                kv = _k(v)
                if op == "<":
                    rows = [r for r in rows if r[c] is not None and _k(r[c]) < kv]
                elif op == "<=":
                    rows = [r for r in rows if r[c] is not None and _k(r[c]) <= kv]
                elif op == ">":
                    rows = [r for r in rows if r[c] is not None and _k(r[c]) > kv]
                else:
                    rows = [r for r in rows if r[c] is not None and _k(r[c]) >= kv]
        rows = list(rows)
        for c, asc in reversed(order_by):
            rows.sort(key=lambda r: (0,) if r[c] is None else (1,) + _k(r[c]), reverse=not asc)
        return [{c: r[c] for c in out} for r in rows]
