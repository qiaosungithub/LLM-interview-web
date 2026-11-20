# expect: WA
# Wrong for Part 2: finds the cells a value reads with a loose regex and checks for a cycle before checking the
# syntax, so an invalid formula that reads its own cell ('=A1 * 2' for A1) raises CycleError instead of FormulaError.
import re

UPPER = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = "0123456789"


class FormulaError(Exception):
    pass


class CycleError(Exception):
    pass


def check_name(name):
    if not isinstance(name, str):
        raise FormulaError("name must be a string")
    head = name.rstrip(DIGITS)
    if not head or head == name or any(ch not in UPPER for ch in head):
        raise FormulaError("bad cell name %r" % (name,))


_TOKEN = re.compile(r"([A-Z]+[0-9]+)|([0-9]+)|([+-])|( +)|(.)", re.S)


def compile_value(value):
    """-> list of (sign, int | cell name)."""
    if type(value) is int:
        return [(1, value)]
    if not isinstance(value, str) or value[:1] != "=":
        raise FormulaError("not an int or a formula: %r" % (value,))
    out, sign, want_term = [], 1, True
    for m in _TOKEN.finditer(value, 1):
        cell, num, op, space, other = m.groups()
        if space:
            continue
        if other is not None:
            raise FormulaError("unexpected %r in %r" % (other, value))
        if want_term:
            if cell:
                out.append((sign, cell))
            elif num and len(num) <= 18:
                out.append((sign, int(num)))
            else:
                raise FormulaError("expected a term in %r" % (value,))
            want_term = False
        else:
            if not op:
                raise FormulaError("expected + or - in %r" % (value,))
            sign = 1 if op == "+" else -1
            want_term = True
    if want_term:
        raise FormulaError("formula ends without a term: %r" % (value,))
    return out


def refs_of(terms):
    return {t for _, t in terms if isinstance(t, str)}


class LazySpreadsheet:
    def __init__(self):
        self._terms = {}

    def set(self, name, value):
        check_name(name)
        self._terms[name] = compile_value(value)

    def get(self, name):
        check_name(name)
        done = {}
        GRAY = object()
        state = {}
        stack = [(name, False)]
        while stack:
            cell, expanded = stack.pop()
            if expanded:
                done[cell] = sum(s * (t if isinstance(t, int) else done[t]) for s, t in self._terms.get(cell, ()))
                state[cell] = done
                continue
            st = state.get(cell)
            if st is done:
                continue
            if st is GRAY:
                raise CycleError("cycle through %s" % cell)
            state[cell] = GRAY
            stack.append((cell, True))
            for _, t in self._terms.get(cell, ()):
                if isinstance(t, str):
                    s2 = state.get(t)
                    if s2 is GRAY:
                        raise CycleError("cycle through %s" % t)
                    if s2 is not done:
                        stack.append((t, False))
        return done[name]


class Spreadsheet:
    def __init__(self):
        self._terms = {}
        self._value = {}
        self._readers = {}

    def _reaches(self, starts, target):
        """True if some cell in starts reads target, directly or not (following formulas)."""
        seen, todo = set(), list(starts)
        while todo:
            c = todo.pop()
            if c == target:
                return True
            if c in seen:
                continue
            seen.add(c)
            todo.extend(refs_of(self._terms.get(c, ())))
        return False

    def set(self, name, value):
        check_name(name)
        loose = set(re.findall(r"[A-Z]+[0-9]+", value)) if isinstance(value, str) else set()
        if self._reaches(loose, name):
            raise CycleError("%s = %r closes a cycle" % (name, value))
        terms = compile_value(value)
        new_refs = refs_of(terms)
        if self._reaches(new_refs, name):
            raise CycleError("%s = %r closes a cycle" % (name, value))
        for r in refs_of(self._terms.get(name, ())):
            self._readers[r].discard(name)
        for r in new_refs:
            self._readers.setdefault(r, set()).add(name)
        self._terms[name] = terms
        # reverse post-order of a DFS over readers = a topological order of the affected cells
        order, seen = [], set()
        stack = [(name, iter(self._readers.get(name, ())))]
        seen.add(name)
        while stack:
            cell, it = stack[-1]
            nxt = next((c for c in it if c not in seen), None)
            if nxt is None:
                stack.pop()
                order.append(cell)
            else:
                seen.add(nxt)
                stack.append((nxt, iter(self._readers.get(nxt, ()))))
        val = self._value
        for cell in reversed(order):
            val[cell] = sum(s * (t if isinstance(t, int) else val.get(t, 0)) for s, t in self._terms[cell])

    def get(self, name):
        check_name(name)
        return self._value.get(name, 0)
