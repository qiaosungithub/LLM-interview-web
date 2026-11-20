# expect: WA
# Wrong: Part 2 collapses every run of equal ids, not only runs of unk_id. Passes every sample.
def tokenize(text, vocab, unk_id):
    longest = max((len(k) for k in vocab), default=0)
    ids, i, n = [], 0, len(text)
    while i < n:
        for length in range(min(longest, n - i), 0, -1):
            if text[i:i + length] in vocab:
                ids.append(vocab[text[i:i + length]])
                i += length
                break
        else:
            ids.append(unk_id)
            i += 1
    return ids


def tokenize_merged(text, vocab, unk_id):
    out = []
    for x in tokenize(text, vocab, unk_id):
        if not (out and out[-1] == x):
            out.append(x)
    return out


class _Node:
    def __init__(self):
        self.children, self.token_id = {}, None


class TrieTokenizer:
    def __init__(self, vocab, unk_id):
        self.unk_id, self.root = unk_id, _Node()
        for token, token_id in vocab.items():
            node = self.root
            for ch in token:
                node = node.children.setdefault(ch, _Node())
            node.token_id = token_id

    def tokenize(self, text):
        ids, i, n = [], 0, len(text)
        while i < n:
            node, j, best = self.root, i, None
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.token_id is not None:
                    best = (j, node.token_id)
            if best is None:
                if not (ids and ids[-1] == self.unk_id):
                    ids.append(self.unk_id)
                i += 1
            else:
                ids.append(best[1])
                i = best[0]
        return ids
