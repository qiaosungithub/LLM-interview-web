"""A model of Parts 1-3 that shares no code with the page (string slicing, whole-text snapshots for undo and redo,
a hand-written word scanner), and random generators for every part.

An operation is a tuple (name, obj, method, args): it calls obj.method(*args), or method(*args) when obj is None
(the constructor), and stores the result under name when name is not None. "__len__" stands for len(obj).
"""


def words(text):
    """The maximal runs of A-Z / a-z in text, in order."""
    out, cur = [], ""
    for ch in text + " ":
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            cur += ch
        elif cur:
            out.append(cur)
            cur = ""
    return out


class TextBuffer:
    def __init__(self, text=""):
        self.s, self.undos, self.redos = text, [], []

    def __len__(self):
        return len(self.s)

    def text(self):
        return self.s

    def _span(self, start, end):
        if not 0 <= start <= end <= len(self.s):
            raise IndexError((start, end))

    def _record(self):
        self.undos.append(self.s)
        self.redos.clear()

    def insert(self, pos, text):
        if not 0 <= pos <= len(self.s):
            raise IndexError(pos)
        if text:
            self._record()
        self.s = self.s[:pos] + text + self.s[pos:]

    def delete(self, start, end):
        self._span(start, end)
        removed = self.s[start:end]
        if removed:
            self._record()
        self.s = self.s[:start] + self.s[end:]
        return removed

    def read(self, start, end):
        self._span(start, end)
        return self.s[start:end]

    def undo(self):
        if not self.undos:
            return False
        self.redos.append(self.s)
        self.s = self.undos.pop()
        return True

    def redo(self):
        if not self.redos:
            return False
        self.undos.append(self.s)
        self.s = self.redos.pop()
        return True

    def suggest(self, prefix, k):
        if k < 0:
            raise ValueError(k)
        counts = {}
        for w in words(self.s):
            counts[w] = counts.get(w, 0) + 1
        found = sorted((w for w in counts if w.startswith(prefix)), key=lambda w: (-counts[w], w))
        return found[:k]


# ---------------------------------------------------------------- random sessions for Parts 1-3

PIECES = ["a", "b", "ab", "ba", "A", "B", "Ab", "aB", " ", " ", "  ", "a b", "b a", "1", "_", "-", "é", "中", "😀",
          "x", "\n"]
PREFIXES = ["", "", "a", "b", "A", "B", "ab", "ba", "Ab", "aa", "a ", "é", "x"]


def random_ops(r, part, length):
    """A session on one buffer: inserts (some empty, some out of range), deletes (some empty, some out of range),
    reads, len/text, and from Part 2 on undo/redo, from Part 3 on suggest (some with k = 0 or k < 0)."""
    ops = [("buf", None, "TextBuffer", (r.choice(["", "", "ab", "a b", "Ab ba"]),))]
    n = len(ops[0][3][0])
    weights = {"insert": 34, "delete": 20, "read": 8, "text": 3, "__len__": 3}
    if part >= 2:
        weights.update(undo=13, redo=9)
    if part >= 3:
        weights.update(suggest=14)
    names, w = list(weights), list(weights.values())
    model = TextBuffer(ops[0][3][0])
    for _ in range(length):
        method = r.choices(names, w)[0]
        n = len(model.s)
        bad = r.random() < 0.06
        if method == "insert":
            pos = r.choice([-1, n + 1, n + 3]) if bad else r.randint(0, n)
            text = "" if r.random() < 0.1 or n > 30 else "".join(r.choice(PIECES) for _ in range(r.randint(1, 3)))
            args = (pos, text)
        elif method in ("delete", "read"):
            if bad:
                start, end = r.choice([(-1, max(0, n - 1)), (0, n + 1), (min(n, 2), min(n, 2) - 1)])
            else:
                start = r.randint(0, n)
                end = start if r.random() < 0.1 else r.randint(start, min(n, start + 6))
            args = (start, end)
        elif method == "suggest":
            args = (r.choice(PREFIXES), r.choice([0, 1, 2, 3, 3, 5, 10, -1] if r.random() < 0.15 else [1, 2, 3, 5]))
        else:
            args = ()
        op = (None, "buf", method, args)
        ops.append(op)
        try:
            getattr(model, method)(*args)
        except (IndexError, ValueError):
            pass
    return ops


# ---------------------------------------------------------------- random scenarios for Part 4

# Distinct characters, so the test can tell which character a local delete removed from the answer's own text.
POOL = [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + \
       [chr(c) for c in range(ord("0"), ord("9") + 1)] + [chr(c) for c in range(0x3b1, 0x3ca)] + \
       [chr(c) for c in range(0x4e00, 0x4e00 + 400)]


def random_scenario(r, replicas, length):
    """Abstract steps on `replicas` replicas: local inserts and deletes (index given as a fraction of the replica's
    current length), deliveries of the next few operations on one channel, and repeat deliveries."""
    steps = []
    for _ in range(length):
        x = r.random()
        i = r.randrange(replicas)
        if x < 0.38:
            steps.append(("ins", i, r.random()))
        elif x < 0.55:
            steps.append(("del", i, r.random()))
        elif x < 0.9:
            j = r.choice([k for k in range(replicas) if k != i])
            steps.append(("deliver", i, j, r.choice([1, 1, 2, 3, 5])))
        else:
            steps.append(("dup", i, r.random()))
    return steps
