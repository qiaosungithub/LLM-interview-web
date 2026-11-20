# expect: WA
# Broadcasts plain (index, char) edits, so replicas that apply concurrent edits in different orders diverge.
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


_WORD_RE = re.compile(r"[A-Za-z]+")


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
        self._chars = []
        self._seen = set()       # (site, seq) of operations already applied

    def text(self) -> str:
        return "".join(self._chars)

    def local_insert(self, index: int, ch: str):
        self._counter += 1
        op = ("insert", self.site_id, self._counter, index, ch)
        self.apply(op)
        return op

    def local_delete(self, index: int):
        self._counter += 1
        op = ("delete", self.site_id, self._counter, index)
        self.apply(op)
        return op

    def apply(self, op) -> None:
        if (op[1], op[2]) in self._seen:
            return
        self._seen.add((op[1], op[2]))
        if op[0] == "insert":
            self._chars.insert(min(op[3], len(self._chars)), op[4])
        elif op[3] < len(self._chars):
            del self._chars[op[3]]
