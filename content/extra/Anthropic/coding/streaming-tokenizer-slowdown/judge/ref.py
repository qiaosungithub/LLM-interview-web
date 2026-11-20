"""Helpers for tests.py: the statement's original class (the behaviour every answer must copy), a whole-text
reference tokenizer, and generators for vocabularies, texts and call sequences."""

EX_VOCAB = {"t": 1, "h": 2, "e": 3, "r": 4, " ": 5, "th": 6, "the": 7, "there": 8}


class Given:
    """The statement's StreamTokenizer, unchanged (slow, but its outputs define what is right)."""

    def __init__(self, vocab, unk_id):
        if not vocab:
            raise ValueError("empty vocabulary")
        self.vocab = vocab
        self.unk_id = unk_id
        self.max_len = max(len(key) for key in vocab)
        self.buf = ""
        self.closed = False

    def write(self, chunk):
        if self.closed:
            raise ValueError("write after close")
        self.buf += chunk

    def close(self):
        self.closed = True

    def pending(self):
        return len(self.buf)

    def _ready(self):
        if not self.buf:
            return False
        return self.closed or len(self.buf) >= self.max_len

    def _longest_match(self):
        best = None
        for key in self.vocab:
            if self.buf.startswith(key):
                if best is None or len(key) > len(best):
                    best = key
        if best is None:
            return self.unk_id, 1
        return self.vocab[best], len(best)

    def read(self, max_tokens):
        out = []
        while len(out) < max_tokens and self._ready():
            token_id, n = self._longest_match()
            out.append(token_id)
            self.buf = self.buf[n:]
        return out


def tokenize(text, vocab, unk_id):
    """Greedy longest match over the whole text at once."""
    lengths = sorted({len(k) for k in vocab}, reverse=True)
    out, i, n = [], 0, len(text)
    get = vocab.get
    while i < n:
        for m in lengths:
            if i + m <= n:
                t = get(text[i:i + m])
                if t is not None:
                    out.append(t)
                    i += m
                    break
        else:
            out.append(unk_id)
            i += 1
    return out


# ---------------------------------------------------------------- big inputs

EMOJI_WORDS = ["\U0001F642", "\U0001F680\U0001F680", "\U0001F600", "你\U0001F642"]
CJK_WORDS = ["你好", "世界", "分词器", "文档"]


PERF_WORDS = ["the", "there", "then", "he", "her", "here", "token", "tokens", "stream"]


def perf_vocab():
    """A small vocabulary (16 keys, L = 6): English words that are prefixes of each other, Chinese words and
    emoji. It is small on purpose: scanning every key for each token is allowed by the statement ("per-token
    work depends only on the vocabulary"), so the big tests must not punish it; only the copying decides."""
    vocab = {" ": 1}
    for w in PERF_WORDS + CJK_WORDS[:3] + ["\U0001F642", "\U0001F680\U0001F680", "你\U0001F642"]:
        vocab[w] = len(vocab) + 1
    return vocab


def perf_text(r, vocab, n_chars):
    """Words of perf_vocab (some glued in pairs), numbers and other letters (unk), Chinese words and emoji
    (some not in the vocabulary), joined by spaces; an emoji roughly every 40 characters keeps the backlog a
    4-byte-per-character string the whole time."""
    parts, size = [], 0
    while size < n_chars:
        x = r.random()
        if x < 0.1:
            w = str(r.randint(0, 999))
        elif x < 0.2:
            w = r.choice(EMOJI_WORDS)
        elif x < 0.25:
            w = r.choice(CJK_WORDS)
        elif x < 0.3:
            w = "".join(r.choice("abcdefgh") for _ in range(r.randint(1, 4)))
        elif x < 0.4:
            w = r.choice(PERF_WORDS) + r.choice(PERF_WORDS)
        else:
            w = r.choice(PERF_WORDS)
        parts.append(w)
        size += len(w) + 1
    return " ".join(parts)[:n_chars]


# ---------------------------------------------------------------- small random inputs

def random_vocab(r, alphabet="ab", max_keys=8, max_len=4):
    """1..max_keys distinct non-empty keys over alphabet, with ids 1, 2, ... (unk_id 0 is never a key's id)."""
    vocab = {}
    for _ in range(r.randint(1, max_keys)):
        key = "".join(r.choice(alphabet) for _ in range(r.randint(1, max_len)))
        vocab.setdefault(key, len(vocab) + 1)
    return vocab


def random_text(r, alphabet, lo, hi):
    return "".join(r.choice(alphabet) for _ in range(r.randint(lo, hi)))


def random_ops(r, text, max_piece=5, max_read=4, close=True):
    """A call sequence that writes text in random pieces (empty ones included), reads and asks pending()
    in between, closes, and then reads until empty."""
    ops, i = [], 0
    while i < len(text) or r.random() < 0.2:
        j = min(len(text), i + r.randint(0, max_piece))
        ops.append(("write", text[i:j]))
        i = j
        while r.random() < 0.6:
            ops.append(("read", r.randint(0, max_read)) if r.random() < 0.7 else ("pending",))
        if i >= len(text) and r.random() < 0.5:
            break
    if close:
        ops.append(("close",))
        ops.append(("pending",))
        while r.random() < 0.3:
            ops.append(("read", r.randint(0, max_read)))
        ops += [("read", r.randint(1, max_read)) for _ in range(len(text) + 1)]
        ops.append(("pending",))
    return ops


def op_code(op):
    """One call as Python code."""
    if op[0] == "write":
        return f"tok.write({op[1]!r})"
    if op[0] == "read":
        return f"tok.read({op[1]})"
    return f"tok.{op[0]}()"


def apply(obj, op):
    if op[0] == "write":
        return obj.write(op[1])
    if op[0] == "read":
        return obj.read(op[1])
    if op[0] == "close":
        return obj.close()
    return obj.pending()
