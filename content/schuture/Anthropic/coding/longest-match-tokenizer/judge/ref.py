"""Independent restatement of the rules, used to compute expected answers for random tests."""


def tokenize(text, vocab, unk_id):
    ids, i, n = [], 0, len(text)
    longest = max((len(k) for k in vocab), default=0)
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


def merge_unk(ids, unk_id):
    out = []
    for x in ids:
        if not (x == unk_id and out and out[-1] == unk_id):
            out.append(x)
    return out


def random_vocab(rng, n_words, alphabet, max_len):
    words = set()
    while len(words) < n_words:
        words.add("".join(rng.choice(alphabet) for _ in range(rng.randint(1, max_len))))
    longer = sorted(w for w in words if len(w) >= 2)
    if longer:                                  # make sure some key is a prefix of another
        base = rng.choice(longer)
        words.add(base[:rng.randint(1, len(base) - 1)])
    return {w: i for i, w in enumerate(sorted(words))}


def random_text(rng, vocab, alphabet, target_len):
    words, out, total = list(vocab), [], 0
    while total < target_len:
        piece = rng.choice(words) if words and rng.random() < 0.6 else rng.choice(alphabet)
        out.append(piece)
        total += len(piece)
    return "".join(out)
