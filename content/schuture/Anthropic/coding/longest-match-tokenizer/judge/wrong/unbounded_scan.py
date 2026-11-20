# expect: TLE
# Wrong for Part 2: correct output, but tries every length from n - i down at each position (too slow).
def tokenize(text, vocab, unk_id):
    ids, i, n = [], 0, len(text)
    while i < n:
        for length in range(n - i, 0, -1):
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
        return tokenize_merged(text, self.vocab, self.unk_id)
