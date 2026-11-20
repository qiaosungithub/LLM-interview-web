"""The multiset modelled by a sorted Python list, and random operation sequences for the tests.

An operation is a tuple: ("add", k), ("in", k), ("len",), ("list",), ("remove", k), ("between", lo, hi),
("kth", i), ("rank", k). Model.apply returns what the statement says the operation returns, or RAISES for a kth
index outside [0, len)."""
import bisect

RAISES = "IndexError"           # the expected outcome of kth(i) for i outside [0, len)
LO, HI = -10 ** 9, 10 ** 9      # the key range of the statement

PART_OPS = {
    1: ["add", "in", "len", "list"],
    2: ["add", "in", "len", "list", "remove", "between"],
    3: ["add", "in", "len", "list", "remove", "between", "kth", "rank"],
}
WEIGHTS = {"add": 8, "in": 3, "len": 1, "list": 1, "remove": 5, "between": 3, "kth": 4, "rank": 4}


class Model:
    def __init__(self):
        self.keys = []          # every stored key, ascending, duplicates included

    def apply(self, op):
        name, args = op[0], op[1:]
        keys = self.keys
        if name == "add":
            bisect.insort(keys, args[0])
            return None
        if name == "in":
            return args[0] in keys
        if name == "len":
            return len(keys)
        if name == "list":
            return list(keys)
        if name == "remove":
            if args[0] in keys:
                keys.remove(args[0])
                return True
            return False
        if name == "between":
            lo, hi = args
            return [k for k in keys if lo <= k <= hi]
        if name == "kth":
            i = args[0]
            return keys[i] if 0 <= i < len(keys) else RAISES
        if name == "rank":
            return sum(1 for k in keys if k < args[0])
        raise ValueError(name)


def random_ops(r, part, n, spread):
    """n random operations allowed in `part`. Keys come from [-spread, spread] (clipped to the statement's range);
    half of the removes and some queries pick a key that is stored, so they hit."""
    names = PART_OPS[part]
    weights = [WEIGHTS[x] for x in names]
    lo, hi = max(LO, -spread), min(HI, spread)
    stored, ops = [], []

    def key():
        if stored and r.random() < 0.4:
            return r.choice(stored)
        return r.randint(lo, hi)

    for _ in range(n):
        name = r.choices(names, weights)[0]
        if name == "add":
            k = r.randint(lo, hi) if not stored or r.random() < 0.7 else r.choice(stored)
            stored.append(k)
            ops.append(("add", k))
        elif name in ("in", "rank"):
            ops.append((name, key()))
        elif name in ("len", "list"):
            ops.append((name,))
        elif name == "remove":
            k = key()
            if k in stored:
                stored.remove(k)
            ops.append(("remove", k))
        elif name == "between":
            a = key()
            b = a + r.randint(-1, 4) * max(1, spread // 4) if r.random() < 0.8 else key()
            ops.append(("between", a, max(LO, min(HI, b))))
        elif name == "kth":
            ops.append(("kth", r.randint(-2, len(stored) + 1)))
    return ops
