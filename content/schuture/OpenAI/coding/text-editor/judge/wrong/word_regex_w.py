# expect: WA
# Splits words with \w+, which also takes digits, underscores and non-ASCII letters.
import bisect
import random
import re
from collections import Counter
from fractions import Fraction


class TextBuffer:
    def __init__(self, text: str = "") -> None:
        self._chars = list(text)

    def __len__(self) -> int:
        return len(self._chars)

    def text(self) -> str:
        return "".join(self._chars)

    def insert(self, pos: int, text: str) -> None:
        if not 0 <= pos <= len(self):
            raise IndexError(f"insert position {pos} out of range for length {len(self)}")
        self._chars[pos:pos] = text                # NOTE: slice-assignment insert, no separate loop

    def delete(self, start: int, end: int) -> str:
        if not 0 <= start <= end <= len(self):
            raise IndexError(f"delete range [{start}, {end}) out of range for length {len(self)}")
        removed = self._chars[start:end]
        del self._chars[start:end]
        return "".join(removed)

    def read(self, start: int, end: int) -> str:
        if not 0 <= start <= end <= len(self):
            raise IndexError(f"read range [{start}, {end}) out of range for length {len(self)}")
        return "".join(self._chars[start:end])


_insert_without_undo = TextBuffer.insert
_delete_without_undo = TextBuffer.delete


def _stacks(self):
    if not hasattr(self, "_undo_stack"):
        self._undo_stack, self._redo_stack = [], []
    return self._undo_stack, self._redo_stack


def insert(self, pos: int, text: str) -> None:
    _insert_without_undo(self, pos, text)
    undo_stack, redo_stack = _stacks(self)
    if text:
        undo_stack.append(("insert", pos, text))
        redo_stack.clear()


def delete(self, start: int, end: int) -> str:
    removed = _delete_without_undo(self, start, end)
    undo_stack, redo_stack = _stacks(self)
    if removed:
        undo_stack.append(("delete", start, removed))
        redo_stack.clear()
    return removed


def undo(self) -> bool:
    undo_stack, redo_stack = _stacks(self)
    if not undo_stack:
        return False
    kind, pos, payload = undo_stack.pop()
    if kind == "insert":
        _delete_without_undo(self, pos, pos + len(payload))
    else:
        _insert_without_undo(self, pos, payload)
    redo_stack.append((kind, pos, payload))
    return True


def redo(self) -> bool:
    undo_stack, redo_stack = _stacks(self)
    if not redo_stack:
        return False
    kind, pos, payload = redo_stack.pop()
    if kind == "insert":
        _insert_without_undo(self, pos, payload)
    else:
        _delete_without_undo(self, pos, pos + len(payload))
    undo_stack.append((kind, pos, payload))
    return True


TextBuffer.insert = insert
TextBuffer.delete = delete
TextBuffer.undo = undo
TextBuffer.redo = redo


_WORD_RE = re.compile(r"\w+")


class _TrieNode:
    __slots__ = ("children", "count")

    def __init__(self):
        self.children: dict[str, "_TrieNode"] = {}
        self.count = 0                  # NOTE: 0 unless this node ends a word that occurs in the buffer


def _build_trie(words: Counter) -> _TrieNode:
    root = _TrieNode()
    for word, count in words.items():
        node = root
        for ch in word:
            node = node.children.setdefault(ch, _TrieNode())
        node.count = count
    return root


def suggest(self, prefix: str, k: int) -> list[str]:
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    root = _build_trie(Counter(_WORD_RE.findall(self.text())))
    node = root
    for ch in prefix:
        if ch not in node.children:
            return []
        node = node.children[ch]
    matches: list[tuple[str, int]] = []

    def collect(n: _TrieNode, built: str) -> None:
        if n.count:
            matches.append((built, n.count))
        for ch, child in n.children.items():
            collect(child, built + ch)

    collect(node, prefix)
    matches.sort(key=lambda item: (-item[1], item[0]))   # NOTE: frequency first, then string order
    return [word for word, _ in matches[:k]]


TextBuffer.suggest = suggest


_RAND_DENOM = 2 ** 32


class CRDTDoc:
    def __init__(self, site_id: str) -> None:
        self.site_id = site_id
        self._counter = 0
        self._entries = []       # (pos, site, seq, ch) tuples, kept sorted
        self._inserted = set()   # (site, seq) of inserts already applied
        self._deleted = set()    # (site, seq) of characters deleted on any replica

    def _live(self) -> list:
        return [e for e in self._entries if (e[1], e[2]) not in self._deleted]

    def text(self) -> str:
        return "".join(e[3] for e in self._live())

    def local_insert(self, index: int, ch: str):
        if len(ch) != 1:
            raise ValueError(f"local_insert takes exactly one character, got {ch!r}")
        live = self._live()
        if not 0 <= index <= len(live):
            raise IndexError(f"insert index {index} out of range for length {len(live)}")
        left = live[index - 1][0] if index > 0 else Fraction(0)
        right = live[index][0] if index < len(live) else Fraction(1)
        pos = left + (right - left) * Fraction(random.randint(1, _RAND_DENOM - 1), _RAND_DENOM)
        self._counter += 1
        op = ("insert", pos, self.site_id, self._counter, ch)
        self.apply(op)
        return op

    def local_delete(self, index: int):
        live = self._live()
        if not 0 <= index < len(live):
            raise IndexError(f"delete index {index} out of range for length {len(live)}")
        op = ("delete", live[index][1], live[index][2])
        self.apply(op)
        return op

    def apply(self, op) -> None:
        if op[0] == "insert":
            _, pos, site, seq, ch = op
            if (site, seq) in self._inserted:    # NOTE: a replayed insert changes nothing
                return
            self._inserted.add((site, seq))
            bisect.insort(self._entries, (pos, site, seq, ch))
        else:
            _, site, seq = op
            # NOTE: the id may arrive before the insert that creates the character
            self._deleted.add((site, seq))
