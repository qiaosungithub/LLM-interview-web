"""Types in the statement's notation ("[T1,[int,bool]]"), parsed to plain Python values: an atomic type is its name
(a str), a tuple type is a list of types. A model of get_return_type that shares no code with the page, and random
calls for the tests."""

PRIMITIVES = ("int", "float", "str", "bool", "char")
ERRORS = ("ArityError", "TypeMismatchError", "GenericConflictError")


def parse(s):
    """'[T1,[int,bool]]' -> ['T1', ['int', 'bool']]."""
    s = s.replace(" ", "")
    t, i = _parse(s, 0)
    if i != len(s):
        raise ValueError(s)
    return t


def _parse(s, i):
    if s[i] == "[":
        items, i = [], i + 1
        if s[i] == "]":
            return items, i + 1
        while True:
            t, i = _parse(s, i)
            items.append(t)
            if s[i] == "]":
                return items, i + 1
            if s[i] != ",":
                raise ValueError(s)
            i += 1
    j = i
    while j < len(s) and s[j] not in ",[]":
        j += 1
    if j == i:
        raise ValueError(s)
    return s[i:j], j


def signature(s):
    """'(T1,bool) -> [T1,T1]' -> (['T1', 'bool'], ['T1', 'T1'])."""
    params, ret = s.split("->")
    params = params.strip()
    assert params[0] == "(" and params[-1] == ")"
    return parse("[" + params[1:-1] + "]"), parse(ret)


def show(t):
    return t if isinstance(t, str) else "[" + ",".join(show(x) for x in t) + "]"


def show_sig(params, ret):
    return "(" + ",".join(show(p) for p in params) + ") -> " + show(ret)


def code(t):
    """Python source that builds t with the answer's Node."""
    if isinstance(t, str):
        return f'Node("{t}")'
    return "Node([" + ", ".join(code(x) for x in t) + "])"


def build(Node, t):
    if isinstance(t, str):
        return Node(t)
    return Node([build(Node, x) for x in t])


def is_generic(t):
    return isinstance(t, str) and t not in PRIMITIVES


def nest(t, depth):
    for _ in range(depth):
        t = [t]
    return t


def substitute(t, bindings):
    if is_generic(t):
        return bindings[t]
    if isinstance(t, str):
        return t
    return [substitute(x, bindings) for x in t]


def analyze(params, ret, args):
    """("ok", result type) for a call without errors. Otherwise ("error", kinds): every error class that some
    correct answer could raise for this call, whatever order it checks things in (so a test only uses calls where
    kinds has one element)."""
    kinds, seen = set(), {}

    def walk(d, a):
        if is_generic(d):
            seen.setdefault(d, []).append(a)
        elif isinstance(d, str):
            if d != a:
                kinds.add("TypeMismatchError")
        elif isinstance(a, str):
            kinds.add("TypeMismatchError")
        else:
            if len(a) != len(d):
                kinds.add("TypeMismatchError")
            for x, y in zip(d, a):                  # an answer might look inside before comparing lengths
                walk(x, y)

    if len(args) != len(params):
        kinds.add("ArityError")
    for d, a in zip(params, args):
        walk(d, a)
    for name, found in seen.items():
        if any(show(x) != show(found[0]) for x in found):
            kinds.add("GenericConflictError")
    if kinds:
        return "error", kinds
    return "ok", substitute(ret, {name: found[0] for name, found in seen.items()})


# ---------------------------------------------------------------- random calls

GENERICS = ["T1", "T2", "T3", "T1", "T2", "K", "value", "Int", "T"]


def random_type(r, atoms, depth):
    if depth == 0 or r.random() < 0.45:
        return r.choice(atoms)
    return [random_type(r, atoms, depth - 1) for _ in range(r.choice([0, 1, 1, 2, 2, 3]))]


def generics_in(t):
    if is_generic(t):
        return [t]
    return [] if isinstance(t, str) else [g for x in t for g in generics_in(x)]


def _paths(d, keep, path=()):
    """Paths (index tuples) of the nodes of d for which keep(node) holds."""
    out = [path] if keep(d) else []
    if isinstance(d, list):
        for i, x in enumerate(d):
            out += _paths(x, keep, path + (i,))
    return out


def _at(t, path):
    for i in path:
        t = t[i]
    return t


def _replace(t, path, new):
    if not path:
        return new
    return [_replace(x, path[1:], new) if i == path[0] else x for i, x in enumerate(t)]


def _different(r, old, depth=2):
    while True:
        new = random_type(r, list(PRIMITIVES), depth)
        if show(new) != show(old):
            return new


def random_call(r):
    """(params, ret, args, mode) with mode one of ok / ArityError / TypeMismatchError / GenericConflictError; the
    call may still come out differently, so the caller asks analyze()."""
    names = r.sample(GENERICS, r.randint(1, 3))
    atoms = list(PRIMITIVES) + names * 2
    params = [random_type(r, atoms, 3) for _ in range(r.randint(0, 4))]
    used = sorted({g for p in params for g in generics_in(p)})
    ret = random_type(r, list(PRIMITIVES) + used * 3, 3)
    bindings = {g: random_type(r, list(PRIMITIVES), 2) for g in used}
    args = [substitute(p, bindings) for p in params]
    mode = r.choices(["ok", "ArityError", "TypeMismatchError", "GenericConflictError"], [45, 12, 23, 20])[0]
    if mode == "ArityError":
        if args and r.random() < 0.5:
            args = args[:-1]
        else:
            args = args + [random_type(r, list(PRIMITIVES), 2)]
    elif mode == "TypeMismatchError":
        spots = [(i, p) for i, d in enumerate(params) for p in _paths(d, lambda n: not is_generic(n))]
        if spots:
            i, path = r.choice(spots)
            old = _at(args[i], path)
            if isinstance(old, str) or r.random() < 0.3:
                new = _different(r, old, 1)
            elif old and r.random() < 0.5:
                new = old[:-1]
            else:
                new = old + [random_type(r, list(PRIMITIVES), 1)]
            args[i] = _replace(args[i], path, new)
    elif mode == "GenericConflictError":
        spots = [(i, p) for i, d in enumerate(params) for p in _paths(d, is_generic)]
        count = {}
        for i, p in spots:
            count[_at(params[i], p)] = count.get(_at(params[i], p), 0) + 1
        spots = [(i, p) for i, p in spots if count[_at(params[i], p)] > 1]
        if spots:
            i, path = r.choice(spots)
            args[i] = _replace(args[i], path, _different(r, _at(args[i], path)))
        else:                                       # no generic occurs twice: add a parameter that repeats one
            g = r.choice(used or names)
            if g not in used:
                bindings[g] = random_type(r, list(PRIMITIVES), 2)
                params, args = params + [g], args + [bindings[g]]
            params, args = params + [g], args + [_different(r, bindings[g])]
    return params, ret, args, mode
