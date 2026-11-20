def tokenize(text: str, vocab: dict[str, int], unk_id: int) -> list[int]:
    ids = []
    i, n = 0, len(text)
    while i < n:
        for length in range(n - i, 0, -1):        # longest possible down to a single character
            piece = text[i:i + length]
            if piece in vocab:
                ids.append(vocab[piece])
                i += length
                break
        else:
            ids.append(unk_id)                     # NOTE: no key matched, not even text[i] alone
            i += 1
    return ids


def tokenize_merged(text: str, vocab: dict[str, int], unk_id: int) -> list[int]:
    max_len = max((len(token) for token in vocab), default=0)   # L: longest key, 0 if vocab is empty
    ids = []
    i, n = 0, len(text)
    while i < n:
        for length in range(min(max_len, n - i), 0, -1):        # NOTE: bounded by L, not by n - i
            piece = text[i:i + length]
            if piece in vocab:
                ids.append(vocab[piece])
                i += length
                break
        else:
            if not (ids and ids[-1] == unk_id):                 # NOTE: merge into the previous UNK
                ids.append(unk_id)
            i += 1
    return ids


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: dict[str, "_TrieNode"] = {}
        self.token_id: int | None = None    # None until some key of vocab ends exactly here


class TrieTokenizer:
    def __init__(self, vocab: dict[str, int], unk_id: int):
        self._unk_id = unk_id
        self._root = _TrieNode()
        for token, token_id in vocab.items():
            node = self._root
            for ch in token:
                node = node.children.setdefault(ch, _TrieNode())
            node.token_id = token_id

    def tokenize(self, text: str) -> list[int]:
        ids = []
        i, n = 0, len(text)
        while i < n:
            node, j, best = self._root, i, None    # best: (end index, id) of the last terminal node
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.token_id is not None:
                    best = (j, node.token_id)
            if best is None:
                if not (ids and ids[-1] == self._unk_id):
                    ids.append(self._unk_id)
                i += 1
            else:
                end, token_id = best
                ids.append(token_id)
                i = end
        return ids
