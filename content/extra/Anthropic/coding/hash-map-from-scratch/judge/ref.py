"""Helpers for tests.py: a slot-for-slot simulator of the statement's table layout (no deletions), a model of what
a correct map returns (a dict, optionally keyed by a canonical form), and random operation generators."""
import gc
import sys
import types


class Layout:
    """The statement's rules without deletion: start slot hash & (capacity - 1), linear probing, capacity 8 to begin
    with, and before a NEW key goes in, double when 3 * (keys + 1) > 2 * capacity, re-inserting the keys in old slot
    order. order() is what iterating a correct map gives."""

    def __init__(self, hash_fn=hash, eq_fn=None):
        self.hash_fn, self.eq_fn = hash_fn, eq_fn
        self.capacity, self.size = 8, 0
        self.slots = [None] * 8                       # (hash, key, value) or None

    def _same(self, a, b):
        return a is b or (a == b if self.eq_fn is None else self.eq_fn(a, b))

    def _place(self, entry):
        mask = self.capacity - 1
        i = entry[0] & mask
        while self.slots[i] is not None:
            i = (i + 1) & mask
        self.slots[i] = entry

    def put(self, key, value):
        h = self.hash_fn(key)
        mask = self.capacity - 1
        i = h & mask
        while self.slots[i] is not None:
            hh, k, _ = self.slots[i]
            if hh == h and self._same(k, key):
                self.slots[i] = (hh, k, value)        # overwrite: the stored key object stays
                return
            i = (i + 1) & mask
        if 3 * (self.size + 1) > 2 * self.capacity:
            old = [s for s in self.slots if s is not None]
            self.capacity *= 2
            self.slots = [None] * self.capacity
            for s in old:
                self._place(s)
        self._place((h, key, value))
        self.size += 1

    def order(self):
        return [s[1] for s in self.slots if s is not None]

    def items(self):
        return [(s[1], s[2]) for s in self.slots if s is not None]


class Model:
    """What a correct map returns. canon=None uses a dict directly (same `a is b or a == b` rule, and an overwrite keeps
    the first key object); otherwise keys are matched by canon(key), e.g. tuple(key) or key.lower()."""

    def __init__(self, canon=None):
        self.canon = canon
        self.d = {}

    def _c(self, key):
        return key if self.canon is None else self.canon(key)

    def put(self, key, value):
        c = self._c(key)
        if self.canon is None:
            self.d[c] = value
        else:
            self.d[c] = (self.d[c][0] if c in self.d else key, value)

    def get(self, key, default=None):
        c = self._c(key)
        if c not in self.d:
            return default
        return self.d[c] if self.canon is None else self.d[c][1]

    def contains(self, key):
        return self._c(key) in self.d

    def delete(self, key):
        return self.d.pop(self._c(key), _MISSING) is not _MISSING

    def __len__(self):
        return len(self.d)

    def keys(self):
        return list(self.d) if self.canon is None else [k for k, _ in self.d.values()]

    def items(self):
        return list(self.d.items()) if self.canon is None else list(self.d.values())


_MISSING = object()


def apply_model(model, op):
    """The expected result of one operation (see tests.OPS for the forms)."""
    kind = op[0]
    if kind == "put":
        model.put(op[1], op[2])
        return None
    if kind == "get":
        return model.get(op[1])
    if kind == "getd":
        return model.get(op[1], op[2])
    if kind == "in":
        return model.contains(op[1])
    if kind == "del":
        return model.delete(op[1])
    if kind == "len":
        return len(model)
    if kind == "list":
        return model.keys()
    if kind == "items":
        return model.items()
    raise ValueError(kind)


# ---------------------------------------------------------------- random keys and operation sequences

def pool_default(r, size):
    """Hashable keys with plenty of collisions and keys that are equal across types (1, 1.0, True)."""
    makers = (
        lambda: r.randint(-50, 50),
        lambda: r.randint(0, 20) * 64,                # many keys sharing a start slot
        lambda: r.randint(0, 7) * 8 + 7,              # start slot 7 of 8: runs wrap around to slot 0
        lambda: str(r.randint(0, 30)),
        lambda: (r.randint(0, 3), "t"),
        lambda: float(r.randint(-5, 5)),              # equal to some of the ints
        lambda: r.choice([None, True, False, ""]),
        lambda: r.randint(-2 ** 70, 2 ** 70),
    )
    return [r.choice(makers)() for _ in range(size)]


def pool_ordered(r, size):
    """Keys whose hash is the same in every process (ints, floats, tuples of ints, bools), so that an iteration order
    shown to the user can be reproduced: no str, no None."""
    makers = (
        lambda: r.randint(-60, 60),
        lambda: r.randint(0, 40) * 64,
        lambda: r.randint(0, 15) * 16 + 15,
        lambda: r.randint(-2 ** 66, 2 ** 66),
        lambda: r.randint(0, 9) << 40,
        lambda: float(r.randint(-5, 5)) + r.choice([0, 0.5]),
        lambda: (r.randint(0, 5), r.randint(-3, 3)),
        lambda: r.choice([True, False]),
    )
    return [r.choice(makers)() for _ in range(size)]


def random_ops(r, pool, steps, deletes=True, lists=False, max_live=None, value=None, canon=None):
    """A random operation sequence over keys from pool. max_live caps how many distinct keys are present at once
    (a put of a new key is turned into a delete when the cap is reached); canon is as for Model. The results are
    checked by the caller's model."""
    ops, live = [], Model(canon)
    value = value or (lambda: r.randint(0, 999))
    for _ in range(steps):
        key, x = r.choice(pool), r.random()
        if x < 0.42:
            if max_live is not None and not live.contains(key) and len(live) >= max_live:
                victim = r.choice(live.keys())
                ops.append(("del", victim))
                live.delete(victim)
            else:
                v = value()
                ops.append(("put", key, v))
                live.put(key, v)
        elif x < 0.70 and deletes:
            ops.append(("del", key))
            live.delete(key)
        elif x < 0.80:
            ops.append(("get", key))
        elif x < 0.86:
            ops.append(("getd", key, "缺省"))
        elif x < 0.94:
            ops.append(("in", key))
        elif x < 0.97 or not lists:
            ops.append(("len",))
        else:
            ops.append((r.choice(["list", "items"]),))
    return ops


# ---------------------------------------------------------------- memory held by a map

_SKIP = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.CodeType,
         types.FrameType, types.GeneratorType)


def footprint(root):
    """Bytes of every object reachable from root (lists, arrays, instances, the keys and values stored), not counting
    classes, modules or functions."""
    seen, stack, total = set(), [root], 0
    while stack:
        o = stack.pop()
        if id(o) in seen or isinstance(o, _SKIP):
            continue
        seen.add(id(o))
        total += sys.getsizeof(o)
        stack.extend(gc.get_referents(o))
    return total
