# Part 1

from typing import Any


class Database:
    def __init__(self):
        self._schemas: dict[str, list[str]] = {}
        self._rows: dict[str, list[dict[str, Any]]] = {}

    def create_table(self, table: str, columns: list[str]) -> None:
        if table in self._schemas:
            raise KeyError(f"table already exists: {table!r}")
        self._schemas[table] = list(columns)
        self._rows[table] = []

    def _schema(self, table: str) -> list[str]:
        if table not in self._schemas:
            raise KeyError(f"no such table: {table!r}")
        return self._schemas[table]

    def insert(self, table: str, row: dict[str, Any]) -> None:
        schema = self._schema(table)
        extra = set(row) - set(schema)
        if extra:
            raise KeyError(f"no such column(s) in {table!r}: {sorted(extra)}")
        missing = set(schema) - set(row)
        if missing:
            raise ValueError(f"row is missing column(s) for {table!r}: {sorted(missing)}")
        self._rows[table].append(dict(row))     # NOTE: a copy, so mutating the caller's dict is safe

    def select(self, table: str, columns: list[str] | None = None) -> list[dict[str, Any]]:
        schema = self._schema(table)
        columns = list(columns) if columns is not None else list(schema)
        _check_names(schema, table, columns)
        return [{col: row[col] for col in columns} for row in self._rows[table]]


def _check_names(schema, table, names):
    for col in names:
        if col not in schema:
            raise KeyError(f"no such column in {table!r}: {col!r}")


# Part 2-3

import operator

_OPERATORS = {
    "=": operator.eq, "!=": operator.ne,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}


def _sort_key(value):
    return (0,) if value is None else (1, isinstance(value, str), value)   # NOTE: never compares int with str


def _matches(actual, op, target):
    if actual is None or target is None:       # NOTE: None fails every operator, "!=" included
        return False
    return _OPERATORS[op](_sort_key(actual), _sort_key(target))


def _filtered_rows(self, table, where):     # NOTE: overridden by Part 4
    return [row for row in self._rows[table]
            if all(_matches(row[col], op, value) for col, op, value in where)]


def select(self, table, columns=None, where=None, order_by=None):
    schema = self._schema(table)
    columns = list(columns) if columns is not None else list(schema)
    where, order_by = list(where or []), list(order_by or [])
    _check_names(schema, table, columns + [c for c, _, _ in where] + [c for c, _ in order_by])
    for _, op, _ in where:                      # NOTE: validated up front, even for an empty table
        if op not in _OPERATORS:
            raise ValueError(f"unknown operator: {op!r}")

    rows = self._filtered_rows(table, where)
    for col, ascending in reversed(order_by):    # NOTE: least-significant column first
        rows = sorted(rows, key=lambda row: _sort_key(row[col]), reverse=not ascending)  # NOTE: str has no -x
    return [{col: row[col] for col in columns} for row in rows]   # NOTE: project after filter/sort


Database._filtered_rows = _filtered_rows
Database.select = select


# Part 4

import bisect

_insert_without_indexes = Database.insert   # Part 1's insert, unchanged


def _add_to_index(index, kind, value, pos):
    if kind == "hash":
        index.setdefault(value, []).append(pos)
    elif value is not None:                 # NOTE: sorted index never holds None
        bisect.insort(index, (_sort_key(value), pos))


def insert(self, table, row):
    _insert_without_indexes(self, table, row)
    pos = len(self._rows[table]) - 1
    for (col, kind), index in getattr(self, "_indexes", {}).get(table, {}).items():
        _add_to_index(index, kind, self._rows[table][pos][col], pos)


def create_index(self, table, column, kind):
    schema = self._schema(table)
    _check_names(schema, table, [column])
    if kind not in ("hash", "sorted"):
        raise ValueError(f"unknown index kind: {kind!r}")
    if not hasattr(self, "_indexes"):        # NOTE: lazy init -- predates __init__
        self._indexes = {}
    index = {} if kind == "hash" else []
    for pos, row in enumerate(self._rows[table]):
        _add_to_index(index, kind, row[column], pos)
    self._indexes.setdefault(table, {})[(column, kind)] = index


def _filtered_rows(self, table, where):
    if any(value is None for _, _, value in where):
        return []                                # NOTE: that condition is false on every row
    indexes = getattr(self, "_indexes", {}).get(table, {})
    candidates, remaining = self._rows[table], where
    for i, (col, op, value) in enumerate(where):
        if op == "=" and (col, "hash") in indexes:
            positions = indexes[(col, "hash")].get(value, [])
        elif op in ("<", "<=", ">", ">=") and (col, "sorted") in indexes:
            entries = indexes[(col, "sorted")]
            search = bisect.bisect_right if op in ("<=", ">") else bisect.bisect_left
            cut = search(entries, _sort_key(value), key=lambda e: e[0])
            lo, hi = (0, cut) if op in ("<", "<=") else (cut, len(entries))
            positions = sorted(p for _, p in entries[lo:hi])   # NOTE: restores row-insertion order
        else:
            continue
        candidates = [self._rows[table][p] for p in positions]
        remaining = where[:i] + where[i + 1:]
        break
    return [row for row in candidates
            if all(_matches(row[col], op, value) for col, op, value in remaining)]


Database.insert = insert
Database.create_index = create_index
Database._filtered_rows = _filtered_rows
