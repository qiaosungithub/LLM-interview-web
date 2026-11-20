# expect: WA
# Wrong: Part 2 never drops a cell's old dependency edges when it is overwritten, so a later formula that reads it back is wrongly rejected as a cycle.
import re
from collections import deque


class FormulaError(ValueError):
    """Bad cell name, or bad formula syntax."""


class CycleError(ValueError):
    """A formula depends on itself, directly or indirectly."""


CELL_RE = re.compile(r'[A-Z]+[0-9]+')


def _check_name(name):
    if not CELL_RE.fullmatch(name):  # NOTE: not match() with '$', which also accepts 'A1\n'
        raise FormulaError(f"not a valid cell name: {name!r}")


_TERM = r'(?:[A-Z]+[0-9]+|[0-9]{1,18})'
_FORMULA_RE = re.compile(rf' *{_TERM}(?: *[+-] *{_TERM})* *')  # spaces only around terms
_SIGNED_TERM_RE = re.compile(rf'[+-]?{_TERM}')


def parse_formula(formula):
    """Returns the formula's (sign, kind, value) terms, kind in {'NUM', 'CELL'}."""
    if not isinstance(formula, str) or not formula.startswith('='):
        raise FormulaError(f"formula must start with '=': {formula!r}")
    body = formula[1:]
    if not _FORMULA_RE.fullmatch(body):  # NOTE: a bare term comes first, so no unary minus ("=-A1")
        raise FormulaError(f"not a valid formula: {formula!r}")
    terms = []
    for tok in _SIGNED_TERM_RE.findall(body.replace(' ', '')):
        sign, tok = (-1, tok[1:]) if tok[0] == '-' else (1, tok.lstrip('+'))
        terms.append((sign, 'CELL', tok) if CELL_RE.fullmatch(tok) else (sign, 'NUM', int(tok)))
    return terms


def _referenced_cells(value):
    """Validates value; returns the cells its formula references (empty for a literal)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return set()
    if isinstance(value, str) and value.startswith('='):
        return {item for _, kind, item in parse_formula(value) if kind == 'CELL'}
    raise FormulaError(f"neither an int nor a formula: {value!r}")


def _eval_terms(value, lookup):
    """lookup(cell) returns the value of a cell the formula reads."""
    if isinstance(value, int):
        return value
    total = 0
    for sign, kind, item in parse_formula(value):
        total += sign * (item if kind == 'NUM' else lookup(item))
    return total


class LazySpreadsheet:
    def __init__(self):
        self.contents = {}

    def set(self, name, value):
        _check_name(name)
        _referenced_cells(value)  # validate now; get() parses the formula again
        self.contents[name] = value  # NOTE: no cycle check here; get() finds cycles

    def get(self, name):
        _check_name(name)
        return self._eval(name, set(), {})

    def _eval(self, name, visiting, memo):
        if name in memo:
            return memo[name]
        if name in visiting:
            raise CycleError(f"circular reference through {name!r}")
        value = self.contents.get(name)
        if value is None:
            result = 0
        else:
            visiting.add(name)  # NOTE: mark before recursing, so a path back to `name` is caught
            result = _eval_terms(value, lambda c: self._eval(c, visiting, memo))
            visiting.discard(name)  # NOTE: visiting is the current path; finished cells answer from memo
        memo[name] = result
        return result


class Spreadsheet:
    def __init__(self):
        self.contents = {}
        self.cache = {}
        self.dependencies = {}   # cell -> set of cells it reads
        self.dependents = {}     # cell -> set of cells that read it

    def set(self, name, value):
        _check_name(name)
        new_refs = _referenced_cells(value)  # raises FormulaError; touches nothing yet
        affected = self._affected(name)      # everyone that (transitively) reads `name`, incl. itself
        if new_refs & affected:
            raise CycleError(f"{name} = {value!r} would close a cycle")
        # NOTE: nothing above was mutated, so a rejected set needs no rollback.

        self.dependencies[name] = new_refs
        for r in new_refs:
            self.dependents.setdefault(r, set()).add(name)
        self.contents[name] = value
        for cell in self._topological_order(affected):
            self._recompute_one(cell)

    def get(self, name):
        _check_name(name)
        return self.cache.get(name, 0)

    def _affected(self, name):
        """name plus everything reachable from it by following `dependents` (who reads whom)."""
        affected, stack = {name}, [name]
        while stack:
            cell = stack.pop()
            for dep in self.dependents.get(cell, ()):
                if dep not in affected:
                    affected.add(dep)
                    stack.append(dep)
        return affected

    def _topological_order(self, affected):
        indegree = {c: sum(d in affected for d in self.dependencies.get(c, ())) for c in affected}
        queue = deque(c for c in affected if not indegree[c])
        order = []
        while queue:
            c = queue.popleft()
            order.append(c)
            for dep in self.dependents.get(c, ()):
                if dep in affected:
                    indegree[dep] -= 1
                    if not indegree[dep]:
                        queue.append(dep)
        return order

    def _recompute_one(self, cell):
        value = self.contents.get(cell)
        self.cache[cell] = 0 if value is None else _eval_terms(
            value, lambda c: self.cache.get(c, 0))
