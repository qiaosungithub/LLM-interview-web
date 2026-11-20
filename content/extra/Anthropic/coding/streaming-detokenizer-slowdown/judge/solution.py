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
    """Part 2's StreamDecoder, cut just before the first stop string (which is not returned)."""

    def __init__(self, tokenizer, stops):
        if any(not s for s in stops):
            raise ValueError("empty stop string")
        self.inner = StreamDecoder(tokenizer)
        self.stops = list(stops)
        self.keep = max((len(s) for s in self.stops), default=1) - 1
        self.hold = ""           # final text not returned yet: it may be the start of a stop string
        self.stopped = False
        self.parts = []

    def push(self, token_id):
        if self.stopped:
            return ""
        return self._feed(self.inner.push(token_id), final=False)

    def finish(self):
        if self.stopped:
            return ""
        return self._feed(self.inner.finish(), final=True)

    def text(self):
        return "".join(self.parts)

    def _feed(self, new, final):
        window = self.hold + new
        cut = self._first_stop(window)
        if cut >= 0:
            self.stopped = True
            out, self.hold = window[:cut], ""
        else:
            k = 0 if final else self._open_prefix(window)
            out, self.hold = window[:len(window) - k], window[len(window) - k:]
        self.parts.append(out)
        return out

    def _first_stop(self, window):
        """Start of the stop string that ends first in window (the longer one on a tie), or -1."""
        best = None
        for s in self.stops:
            i = window.find(s)
            if i >= 0 and (best is None or (i + len(s), i) < best):
                best = (i + len(s), i)
        return -1 if best is None else best[1]

    def _open_prefix(self, window):
        """Length of the longest suffix of window that is a proper prefix of some stop string."""
        for k in range(min(self.keep, len(window)), 0, -1):
            end = window[len(window) - k:]
            if any(len(s) > k and s.startswith(end) for s in self.stops):
                return k
        return 0
