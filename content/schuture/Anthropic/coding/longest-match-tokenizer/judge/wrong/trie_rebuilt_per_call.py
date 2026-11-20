# expect: TLE
# Wrong for Part 3: right answers, but rebuilds the trie on every call instead of once in __init__.
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
        if not (x == unk_id and out and out[-1] == unk_id):
            out.append(x)
    return out


class TrieTokenizer:
    def __init__(self, vocab, unk_id):
        self.vocab, self.unk_id = vocab, unk_id

    def tokenize(self, text):
        root = {}
        for token, token_id in self.vocab.items():
            node = root
            for ch in token:
                node = node.setdefault(ch, {})
            node[None] = token_id
        ids, i, n = [], 0, len(text)
        while i < n:
            node, j, best = root, i, None
            while j < n and text[j] in node:
                node = node[text[j]]
                j += 1
                if None in node:
                    best = (j, node[None])
            if best is None:
                if not (ids and ids[-1] == self.unk_id):
                    ids.append(self.unk_id)
                i += 1
            else:
                ids.append(best[1])
                i = best[0]
        return ids
