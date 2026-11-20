# expect: TLE
# Wrong for Part 3: right answers, but searches the whole text so far on every call (the page's direct method).
class ByteTokenizer:
    """Byte-level vocabulary: token id i always stands for the byte string pieces[i].

    A piece may end in the middle of a multi-byte UTF-8 character, and a few
    special ids (end of turn and the like) have empty pieces.
    """

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def token_bytes(self, token_id):
        return self.pieces[token_id]

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")


def incomplete_tail(data):
    """Index where an unfinished UTF-8 character starts at the end of data, or len(data).

    "Unfinished" means: the bytes from there on are a proper prefix of some valid
    UTF-8 encoding, so the next bytes may still complete it.
    """
    n = len(data)
    for i in range(n - 1, max(n - 4, -1), -1):
        b = data[i]
        if 0x80 <= b <= 0xBF:            # continuation byte: keep looking for the lead byte
            continue
        if 0xC2 <= b <= 0xDF:
            need = 2
        elif 0xE0 <= b <= 0xEF:
            need = 3
        elif 0xF0 <= b <= 0xF4:
            need = 4
        else:                            # ASCII, or a byte that can never start a character
            return n
        if n - i >= need:                # long enough: complete (or invalid), decode it now
            return n
        if n - i >= 2:                   # these lead bytes restrict the second byte
            lo, hi = {0xE0: (0xA0, 0xBF), 0xED: (0x80, 0x9F),
                      0xF0: (0x90, 0xBF), 0xF4: (0x80, 0x8F)}.get(b, (0x80, 0xBF))
            if not lo <= data[i + 1] <= hi:
                return n
        return i
    return n


class StreamDecoder:
    """Same return values as the given version; work per token does not grow with the reply."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.tail = b""      # an unfinished character (at most 3 bytes)
        self.held = []       # decoded but not returned; non-empty only while we are waiting
        self.parts = []      # everything returned so far
        self.done = False

    def push(self, token_id):
        if self.done:
            raise ValueError("push after finish")
        data = self.tail + self.tok.token_bytes(token_id)
        cut = incomplete_tail(data)
        piece = data[:cut].decode("utf-8", errors="replace")
        self.tail = data[cut:]
        if piece:
            self.held.append(piece)
        # The given version waits while decode(all ids) ends in U+FFFD, and that text is
        # text() + "".join(held) + (one U+FFFD if tail is not empty).
        if self.tail or (self.held and self.held[-1].endswith("\ufffd")):
            return ""
        delta = "".join(self.held)
        self.held.clear()
        self.parts.append(delta)
        return delta

    def finish(self):
        if self.done:
            return ""
        self.done = True
        delta = "".join(self.held) + self.tail.decode("utf-8", errors="replace")
        self.held.clear()
        self.tail = b""
        self.parts.append(delta)
        return delta

    def text(self):
        return "".join(self.parts)


class StopStreamDecoder:
    """Searches the whole final text for the stop strings on every call."""

    def __init__(self, tokenizer, stops):
        if any(not s for s in stops):
            raise ValueError("empty stop string")
        self.inner = StreamDecoder(tokenizer)
        self.stops = list(stops)
        self.final = ""          # F: all final text so far
        self.sent = 0            # how much of F has been returned
        self.stopped = False

    def push(self, token_id):
        if self.stopped:
            return ""
        self.final += self.inner.push(token_id)
        return self._emit(final=False)

    def finish(self):
        if self.stopped:
            return ""
        self.final += self.inner.finish()
        return self._emit(final=True)

    def text(self):
        return self.final[:self.sent]

    def _emit(self, final):
        F, best = self.final, None
        for s in self.stops:
            i = F.find(s)
            if i >= 0 and (best is None or (i + len(s), i) < best):
                best = (i + len(s), i)
        if best is not None:
            self.stopped = True
            target = best[1]
        else:
            hold = 0
            if not final:
                longest = max((len(s) for s in self.stops), default=0)
                for k in range(1, min(longest, len(F) + 1)):
                    if any(s.startswith(F[len(F) - k:]) for s in self.stops if len(s) > k):
                        hold = k
            target = len(F) - hold
        out = F[self.sent:target]
        self.sent = target
        return out
