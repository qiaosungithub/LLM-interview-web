"""Helpers for tests.py: the statement's ByteTokenizer and given StreamDecoder (which define the required return
values), an incremental decoder for inputs too long for the given one, a brute force for stop strings written from
the definition, and input generators."""
import codecs


class ByteTokenizer:
    """Byte-level vocabulary: token id i always stands for the byte string pieces[i] (copied from the statement)."""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def token_bytes(self, token_id):
        return self.pieces[token_id]

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")


class GivenStreamDecoder:
    """The statement's StreamDecoder, unchanged: its return values are the specification."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids = []
        self.emitted = ""
        self.done = False

    def push(self, token_id):
        if self.done:
            raise ValueError("push after finish")
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        delta = text[len(self.emitted):]
        self.emitted += delta
        return delta

    def finish(self):
        if self.done:
            return ""
        self.done = True
        delta = self.tok.decode(self.ids)[len(self.emitted):]
        self.emitted += delta
        return delta

    def text(self):
        return self.emitted


def given_outputs(pieces, ids):
    """Every push return value of the given decoder, then finish()."""
    dec = GivenStreamDecoder(ByteTokenizer(pieces))
    return [dec.push(i) for i in ids] + [dec.finish()]


def fast_outputs(pieces, ids):
    """Same values as given_outputs in linear time, for long replies: decode(all ids) ends in U+FFFD exactly when
    the incremental decoder still buffers bytes or the text decoded since the last return ends in U+FFFD."""
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    out, held = [], []
    for i in ids:
        piece = dec.decode(pieces[i])
        if piece:
            held.append(piece)
        if dec.getstate()[0] or (held and held[-1].endswith("\ufffd")):
            out.append("")
        else:
            out.append("".join(held))
            held.clear()
    out.append("".join(held) + dec.decode(b"", final=True))
    return out


def stop_ref(F, stops, final):
    """(text that must have been returned, stopped) when the final text so far is F, straight from the definition:
    the first stop string is the occurrence ending earliest (the longer one on a tie); otherwise hold back the
    longest suffix of F that is a proper prefix of some stop string, unless the reply is finished."""
    for p in range(1, len(F) + 1):
        starts = [p - len(s) for s in stops if len(s) <= p and F[p - len(s):p] == s]
        if starts:
            return F[:min(starts)], True
    if final:
        return F, False
    hold = 0
    for k in range(1, len(F) + 1):
        if any(len(s) > k and s.startswith(F[len(F) - k:]) for s in stops):
            hold = k
    return F[:len(F) - hold], False


def stop_outputs_no_match(deltas, stops):
    """Expected StopStreamDecoder return values when the given decoder returns `deltas` (pushes, then finish) and
    no stop string ever appears: hold back the longest suffix that is a proper prefix of a stop string."""
    full = "".join(deltas)
    assert all(full.find(s) < 0 for s in stops)
    longest = max((len(s) for s in stops), default=1)
    out, done, F = [], 0, 0
    for k, d in enumerate(deltas):
        F += len(d)
        hold = 0
        if k < len(deltas) - 1:
            for h in range(1, min(longest - 1, F) + 1):
                if any(len(s) > h and s.startswith(full[F - h:F]) for s in stops):
                    hold = h
        out.append(full[done:F - hold])
        done = F - hold
    return out


# ---------------------------------------------------------------- generators

PALETTE = [0x00, 0x41, 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf,
           0xe0, 0xe1, 0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf3, 0xf4, 0xf5, 0xff]
CHARS = "a 中é😀\ufffd"


def random_piece(r):
    """Empty, a slice of real UTF-8 (usually cut inside a character), or 1-4 bytes from every UTF-8 byte class."""
    x = r.random()
    if x < 0.1:
        return b""
    if x < 0.5:
        s = "".join(r.choice(CHARS) for _ in range(r.randint(1, 3))).encode()
        a = r.randrange(len(s))
        return s[a:r.randint(a + 1, len(s))]
    return bytes(r.choice(PALETTE) for _ in range(r.randint(1, 4)))


def cut_text(r, text, max_cuts):
    """pieces (the UTF-8 bytes of text cut at random byte positions, plus one empty piece at the end) and ids that
    spell text with the empty token mixed in."""
    data = text.encode()
    k = r.randint(0, max(0, min(max_cuts, len(data) - 1)))
    cuts = sorted(r.sample(range(1, len(data)), k)) if k else []
    pieces = [data[a:b] for a, b in zip([0] + cuts, cuts + [len(data)])] if data else []
    ids = []
    for i in range(len(pieces)):
        if r.random() < 0.2:
            ids.append(len(pieces))
        ids.append(i)
    return pieces + [b""], ids


# The statement's reproduction script, with the random generator passed in.
WORDS = ["the", "model", "stream", "token", "reply", "text", "byte", "fast", "slow", "user", "and", "is"]
HANZI = "你好世界流式分词解码模型输出文本速度用户问题回答"
EMOJI = "😀🚀✨🎉"


def make_pieces():
    pieces = [bytes([b]) for b in range(256)]
    for w in WORDS:
        pieces += [w.encode(), (" " + w).encode()]
    for ch in HANZI + EMOJI:
        b = ch.encode()
        pieces += [b[:2], b[2:]]
    pieces.append(b"")
    return pieces


def piece_index(pieces):
    index = {}
    for i, p in enumerate(pieces):
        if p:
            index.setdefault(p, i)
    return index


def encode(index, text):
    """Greedy longest match over the UTF-8 bytes (the statement's stand-in for the model's tokenizer)."""
    longest = max(len(p) for p in index)
    data, ids, i = text.encode(), [], 0
    while i < len(data):
        for n in range(min(longest, len(data) - i), 0, -1):
            if data[i:i + n] in index:
                ids.append(index[data[i:i + n]])
                i += n
                break
    return ids


def reply_text(r, n_chars):
    """Mixed English / Chinese / emoji text like the statement's make_reply, about n_chars long."""
    parts, total = [], 0
    while total < n_chars:
        x = r.random()
        if x < 0.5:
            s = " " + r.choice(WORDS)
        elif x < 0.9:
            s = "".join(r.choice(HANZI) for _ in range(r.randint(1, 6))) + r.choice("，。")
        else:
            s = r.choice(EMOJI)
        parts.append(s)
        total += len(s)
    return "".join(parts)


def make_reply(r, n_tokens):
    """(pieces, ids): the statement's tokenizer and a reply n_tokens long."""
    pieces = make_pieces()
    index = piece_index(pieces)
    ids = []
    while len(ids) < n_tokens:
        ids += encode(index, reply_text(r, 1))
    return pieces, ids[:n_tokens]


def long_token_reply(r, n_tokens, min_len, max_len):
    """(pieces, ids): a reply of n_tokens long tokens, each min_len..max_len bytes of a mixed reply's UTF-8 cut at
    random positions (often inside a character); every token has its own piece. Starts with an emoji."""
    base = ("😀" + reply_text(r, 20_000)).encode()
    data = base * (n_tokens * max_len // len(base) + 1)
    pieces, i = [], 0
    for _ in range(n_tokens):
        n = r.randint(min_len, max_len)
        pieces.append(data[i:i + n])
        i += n
    return pieces, list(range(n_tokens))
