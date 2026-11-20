"""A model of the statement that shares no code with the page: a hand-written scanner for names and formulas, and
values found by repeatedly evaluating every cell whose references all have a value or were never set. A cell that
needs a cycle never gets a value, which is how both spreadsheets below find cycles.

An operation is a tuple (name, obj, method, args): it calls obj.method(*args), or method(*args) when obj is None
(a constructor), and stores the result under name when name is not None.
"""


class FormulaError(Exception):
    pass


class CycleError(Exception):
    pass


def is_name(s):
    """One or more of A-Z, then one or more of 0-9, and nothing else."""
    if not isinstance(s, str):
        return False
    i = 0
    while i < len(s) and "A" <= s[i] <= "Z":
        i += 1
    j = i
    while j < len(s) and "0" <= s[j] <= "9":
        j += 1
    return 0 < i < j == len(s)


def terms(value):
    """The (sign, term) pairs of a cell's content; a term is an int or a cell name. Raises FormulaError for a value
    that is neither an int nor a valid formula."""
    if isinstance(value, int) and not isinstance(value, bool):
        return [(1, value)]
    if not isinstance(value, str) or not value.startswith("="):
        raise FormulaError(value)
    s, i, out, sign, want_term = value, 1, [], 1, True
    while True:
        while i < len(s) and s[i] == " ":
            i += 1
        if want_term:
            j = i
            while j < len(s) and "A" <= s[j] <= "Z":
                j += 1
            k = j
            while k < len(s) and "0" <= s[k] <= "9":
                k += 1
            if k == j or (j == i and k - j > 18):     # no digits, or a literal longer than 18 digits
                raise FormulaError(value)
            out.append((sign, s[i:k] if j > i else int(s[i:k])))
            i, want_term = k, False
        else:
            if i == len(s):
                return out
            if s[i] not in "+-":
                raise FormulaError(value)
            sign = 1 if s[i] == "+" else -1
            i, want_term = i + 1, True


def values(raw):
    """cell -> value for every cell of raw that needs no cycle; the other cells of raw are missing."""
    parsed = {c: terms(v) for c, v in raw.items()}
    vals, progress = {}, True
    while progress:
        progress = False
        for c, ts in parsed.items():
            if c not in vals and all(isinstance(t, int) or t in vals or t not in raw for _, t in ts):
                vals[c] = sum(sg * (t if isinstance(t, int) else vals.get(t, 0)) for sg, t in ts)
                progress = True
    return vals


def _check(name, value=0):
    if not is_name(name):
        raise FormulaError(name)
    terms(value)


class LazySpreadsheet:
    def __init__(self):
        self.raw = {}

    def set(self, name, value):
        _check(name, value)
        self.raw[name] = value

    def get(self, name):
        _check(name)
        vals = values(self.raw)
        if name in self.raw and name not in vals:
            raise CycleError(name)
        return vals.get(name, 0)


class Spreadsheet:
    def __init__(self):
        self.raw = {}

    def set(self, name, value):
        _check(name, value)
        trial = {**self.raw, name: value}
        if len(values(trial)) < len(trial):       # self.raw has no cycle, so any cycle goes through the new value
            raise CycleError(name)
        self.raw = trial

    def get(self, name):
        _check(name)
        return values(self.raw).get(name, 0)


CLASSES = {"LazySpreadsheet": LazySpreadsheet, "Spreadsheet": Spreadsheet}


def layers(x0, y0, n):
    """Values of X0..X{n-1}, Y0..Y{n-1} where Xi = X{i-1} + Y{i-1} and Yi = X{i-1} - Y{i-1}."""
    xs, ys = [x0], [y0]
    for _ in range(1, n):
        xs.append(xs[-1] + ys[-1])
        ys.append(xs[-2] - ys[-1])
    return xs, ys


# ---------------------------------------------------------------- random sessions

BAD_VALUES = ["=", "= ", "={a}*2", "=({a} + 1)", "={a} ++ {b}", "=-{a}", "=+{a}", "={a} {b}", "={a} 3", "=1 2",
              "={a} +", "={a}{b}", "=4{a}", "{a}", " ={a}", "={a}\t+ 1", "=" + "9" * 19, "=1_000", "=2.5", 3.5, None]
BAD_NAMES = ["a1", "A", "1A", "A1B", "A1\n", " A1", "", "A 1"]


def _spaces(r):
    return r.choice(["", " ", " ", "  "])


def random_formula(r, cells, first=None):
    n = r.choice([1, 1, 2, 2, 3])
    pick = lambda: r.choice(cells) if r.random() < 0.75 else str(r.choice([0, 1, 2, 7, 30, 999]))
    parts = [first or pick()] + [pick() for _ in range(n - 1)]
    r.shuffle(parts)
    text = "=" + _spaces(r) + parts[0]
    for p in parts[1:]:
        text += _spaces(r) + r.choice("+-") + _spaces(r) + p
    return text + r.choice(["", "", " "])


def random_ops(r, cls, length, cells, spare="Z9"):
    """A random session on one sheet: literals, formulas (often closing a cycle), invalid values and names, and gets
    of set cells, of the never-set cell `spare`, and of invalid names."""
    obj = "lz" if cls == "LazySpreadsheet" else "sh"
    ops = [(obj, None, cls, ())]
    held = {}                                   # roughly what each cell holds, to aim formulas at cycles
    for _ in range(length):
        x = r.random()
        if x < 0.4:
            y = r.random()
            name = r.choice(cells) if y < 0.9 else (spare if y < 0.96 else r.choice(BAD_NAMES))
            ops.append((None, obj, "get", (name,)))
            continue
        name = r.choice(cells)
        readers = [c for c, v in held.items() if isinstance(v, str) and name in v.replace("+", " ").replace(
            "-", " ").replace("=", " ").split()]
        y = r.random()
        if y < 0.25 and readers:
            value = random_formula(r, cells, first=r.choice(readers))
        elif y < 0.45:
            value = r.choice([r.randint(-20, 20), r.randint(-20, 20), 10 ** 20, -(10 ** 19)])
        elif y < 0.9:
            value = random_formula(r, cells + [spare])
        elif y < 0.97:
            value = r.choice(BAD_VALUES)
            if isinstance(value, str):
                value = value.format(a=r.choice(cells), b=r.choice(cells))
        else:
            name, value = r.choice(BAD_NAMES), r.randint(0, 9)
        ops.append((None, obj, "set", (name, value)))
        if is_name(name):
            try:
                terms(value)
                held[name] = value
            except FormulaError:
                pass
    return ops
