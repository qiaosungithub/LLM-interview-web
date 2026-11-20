# expect: WA
# Wrong for Part 3: keeps the output list on the object, so a second call returns the first call's ids too.
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
        self.unk_id, self.root, self.ids = unk_id, {}, []
        for token, token_id in vocab.items():
            node = self.root
            for ch in token:
                node = node.setdefault(ch, {})
            node[None] = token_id

    def tokenize(self, text):
        i, n = 0, len(text)
        while i < n:
            node, j, best = self.root, i, None
            while j < n and text[j] in node:
                node = node[text[j]]
                j += 1
                if None in node:
                    best = (j, node[None])
            if best is None:
                if not (self.ids and self.ids[-1] == self.unk_id):
                    self.ids.append(self.unk_id)
                i += 1
            else:
                self.ids.append(best[1])
                i = best[0]
        return self.ids
