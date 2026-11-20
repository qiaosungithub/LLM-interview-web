# expect: TLE
# Wrong: the statement's original, unchanged; every push re-decodes the whole reply, O(n^2) in total.
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


class StreamDecoder:
    """The statement's version, unchanged: decodes every id so far on each push."""

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
