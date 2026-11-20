# expect: WA
# Wrong for Part 3: sorts (frequency, word) pairs with reverse=True, so words tied on frequency come out in reverse string order.
def _letter(c):
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


class TextBuffer:
    def __init__(self, text=""):
        self._s = str(text)
        self._undo, self._redo = [], []

    def __len__(self):
        return len(self._s)

    def text(self):
        return self._s

    def _raw_insert(self, pos, text):
        self._s = self._s[:pos] + text + self._s[pos:]

    def _raw_delete(self, start, end):
        gone = self._s[start:end]
        self._s = self._s[:start] + self._s[end:]
        return gone

    def insert(self, pos, text):
        if pos < 0 or pos > len(self._s):
            raise IndexError(pos)
        if text == "":
            return
        self._raw_insert(pos, text)
        self._undo.append((0, pos, text))
        self._redo = []

    def delete(self, start, end):
        if not (0 <= start <= end <= len(self._s)):
            raise IndexError((start, end))
        if start == end:
            return ""
        gone = self._raw_delete(start, end)
        self._undo.append((1, start, gone))
        self._redo = []
        return gone

    def read(self, start, end):
        if not (0 <= start <= end <= len(self._s)):
            raise IndexError((start, end))
        return self._s[start:end]

    def _do(self, rec, forward):
        kind, pos, text = rec
        if (kind == 0) == forward:
            self._raw_insert(pos, text)
        else:
            self._raw_delete(pos, pos + len(text))

    def undo(self):
        if not self._undo:
            return False
        rec = self._undo.pop()
        self._do(rec, False)
        self._redo.append(rec)
        return True

    def redo(self):
        if not self._redo:
            return False
        rec = self._redo.pop()
        self._do(rec, True)
        self._undo.append(rec)
        return True

    def suggest(self, prefix, k):
        if k < 0:
            raise ValueError("negative k")
        freq, i, s = {}, 0, self._s
        while i < len(s):
            if _letter(s[i]):
                j = i
                while j < len(s) and _letter(s[j]):
                    j += 1
                w = s[i:j]
                freq[w] = freq.get(w, 0) + 1
                i = j
            else:
                i += 1
        hits = [(c, w) for w, c in freq.items() if w[:len(prefix)] == prefix]
        hits.sort(reverse=True)
        return [w for _, w in hits[:k]]


class CRDTDoc:
    HEAD = (0, "")

    def __init__(self, site_id):
        self.site = site_id
        self.clock = 0
        self.seq = [self.HEAD]          # ids in document order, tombstones included
        self.char = {}                  # id -> character
        self.dead = set()               # ids deleted (possibly before their insert arrived)
        self.waiting = {}               # missing origin id -> [insert ops]

    def _visible(self):
        return [i for i in self.seq[1:] if i not in self.dead]

    def text(self):
        return "".join(self.char[i] for i in self._visible())

    def _integrate(self, op):
        _, nid, origin, ch = op
        stack = [op]
        while stack:
            _, nid, origin, ch = stack.pop()
            if nid in self.char:
                continue
            if origin != self.HEAD and origin not in self.char:
                self.waiting.setdefault(origin, []).append((_, nid, origin, ch))
                continue
            k = self.seq.index(origin) + 1
            while k < len(self.seq) and self.seq[k] > nid:
                k += 1
            self.seq.insert(k, nid)
            self.char[nid] = ch
            self.clock = max(self.clock, nid[0])
            stack.extend(self.waiting.pop(nid, []))

    def local_insert(self, index, ch):
        if len(ch) != 1:
            raise ValueError(ch)
        vis = self._visible()
        if not 0 <= index <= len(vis):
            raise IndexError(index)
        self.clock += 1
        op = ("i", (self.clock, self.site), vis[index - 1] if index else self.HEAD, ch)
        self._integrate(op)
        return op

    def local_delete(self, index):
        vis = self._visible()
        if not 0 <= index < len(vis):
            raise IndexError(index)
        op = ("d", vis[index])
        self.apply(op)
        return op

    def apply(self, op):
        if op[0] == "i":
            self._integrate(op)
        else:
            self.dead.add(op[1])
            self.clock = max(self.clock, op[1][0])
