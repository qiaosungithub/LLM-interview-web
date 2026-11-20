"""Helpers for tests.py: an iterator that lists every index tuple up front (the statement's definitions spelled out
directly), random nested data, and random states that are mostly near misses."""


class Naive:
    """All index tuples in walk order; the position is an index into that list."""

    def __init__(self, items, d):
        self.items, self.d, self.pos = items, d, 0
        self.paths = []

        def walk(x, prefix):
            if len(prefix) == d:
                self.paths.append(prefix)
                return
            for i in range(len(x)):
                walk(x[i], prefix + (i,))

        walk(items, ())
        self.end = (len(items),) + (0,) * (d - 1)
        self.where = {p: k for k, p in enumerate(self.paths)}
        self.where[self.end] = len(self.paths)

    def state(self):
        return self.paths[self.pos] if self.pos < len(self.paths) else self.end

    def at(self, path):
        x = self.items
        for i in path:
            x = x[i]
        return x

    def next(self):
        """(True, item) and advance, or (False, None) once exhausted."""
        if self.pos == len(self.paths):
            return False, None
        self.pos += 1
        return True, self.at(self.paths[self.pos - 1])

    def verdict(self, s):
        """None if set_state(s) is accepted, else the name of the exception it must raise."""
        if not isinstance(s, (tuple, list)) or len(s) != self.d or any(type(v) is not int for v in s):
            return "TypeError"
        return None if tuple(s) in self.where else "ValueError"

    def seek(self, s):
        self.pos = self.where[tuple(s)]

    def remaining(self):
        return [self.at(p) for p in self.paths[self.pos:]]


def gen(r, d, max_len=3, p_empty=0.35, list_leaves=False):
    """Random data nested d levels deep; a list at any level may be empty. With list_leaves, some items are lists."""
    if d == 1:
        return [leaf(r, list_leaves) for _ in range(r.randint(0, max_len))]
    return [[] if r.random() < p_empty else gen(r, d - 1, max_len, p_empty, list_leaves)
            for _ in range(r.randint(0, max_len))]


def leaf(r, list_leaves=False):
    if list_leaves and r.random() < 0.3:
        return [r.randint(0, 9) for _ in range(r.randint(0, 2))]
    return r.randint(0, 99)


def relabel(r, x, d):
    """Same shape, new items."""
    if d == 1:
        return [r.randint(100, 199) for _ in x]
    return [relabel(r, sub, d - 1) for sub in x]


def random_state(r, model):
    """Mostly near misses: a real position nudged by one, small ints, the wrong length, a bool or float."""
    d = model.d
    roll = r.random()
    if roll < 0.2:
        s = list(r.choice(model.paths + [model.end]))
    elif roll < 0.5 and model.paths:
        s = list(r.choice(model.paths + [model.end]))
        s[r.randrange(d)] += r.choice([-1, 1])
    elif roll < 0.75:
        s = [r.randint(-1, 3) for _ in range(d)]
    elif roll < 0.85:
        s = [r.randint(0, 3) for _ in range(d + r.choice([-1, 1]))]
    elif roll < 0.97:
        s = list(r.choice(model.paths + [model.end]))
        s[r.randrange(d)] = r.choice([True, False, 1.0, 0.0, "0", None])
    else:
        return r.choice([None, 0, "0", {0: 0}])
    return tuple(s) if r.random() < 0.6 else s
