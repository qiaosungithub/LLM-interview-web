# expect: TLE
# Wrong: fixes only the vocabulary scan; `self.buf = self.buf[n:]` still copies the backlog for every token.
import threading


class StreamTokenizer:
    """Greedy longest-match tokenizer for text that arrives in pieces.

    vocab maps token strings (all non-empty) to ids. At each position the
    longest key that matches is taken; if no key matches, one character
    becomes unk_id.

        write(chunk)      add text; a cut may fall anywhere, even inside a token
        close()           no more text will be written
        read(max_tokens)  up to max_tokens ids that can no longer change, in
                          order; fewer (maybe none) if the rest needs more text

    Reading until the stream is closed and empty gives the same ids as
    tokenizing the whole text in one piece, however it was cut into chunks.
    """

    def __init__(self, vocab, unk_id):
        if not vocab:
            raise ValueError("empty vocabulary")
        self.vocab = vocab
        self.unk_id = unk_id
        self.max_len = max(len(key) for key in vocab)
        self.lengths = sorted({len(key) for key in vocab}, reverse=True)
        self.buf = ""            # written but not yet tokenized
        self.closed = False
        self.chars_in = 0        # counters for the metrics endpoint
        self.tokens_out = 0

    def write(self, chunk):
        if self.closed:
            raise ValueError("write after close")
        self.buf += chunk
        self.chars_in += len(chunk)

    def close(self):
        self.closed = True

    def pending(self):
        """Characters written but not yet turned into tokens."""
        return len(self.buf)

    def _ready(self):
        """Is the token at the front of the buffer final?"""
        if not self.buf:
            return False
        # With fewer than max_len characters, a longer key could still be
        # completed by text that has not arrived yet.
        return self.closed or len(self.buf) >= self.max_len

    def _longest_match(self):
        """(token id, length in characters) of the token at the front of the buffer."""
        for n in self.lengths:
            if n <= len(self.buf):
                token_id = self.vocab.get(self.buf[:n])
                if token_id is not None:
                    return token_id, n
        return self.unk_id, 1

    def read(self, max_tokens):
        out = []
        while len(out) < max_tokens and self._ready():
            token_id, n = self._longest_match()
            out.append(token_id)
            self.buf = self.buf[n:]
        self.tokens_out += len(out)
        return out


class BoundedStreamTokenizer:
    """StreamTokenizer for one writer thread and one reader thread, with a bounded backlog."""

    def __init__(self, vocab, unk_id, max_pending):
        self.tok = StreamTokenizer(vocab, unk_id)
        if max_pending < self.tok.max_len:
            raise ValueError("max_pending must be at least the longest vocabulary key")
        self.max_pending = max_pending
        self.peak_pending = 0          # largest backlog seen right after a write
        self.cond = threading.Condition()

    def write(self, chunk):
        with self.cond:
            while self.tok.pending() >= self.max_pending and not self.tok.closed:
                self.cond.wait()
            self.tok.write(chunk)      # raises ValueError after close
            self.peak_pending = max(self.peak_pending, self.tok.pending())
            self.cond.notify_all()

    def close(self):
        with self.cond:
            self.tok.close()
            self.cond.notify_all()

    def pending(self):
        with self.cond:
            return self.tok.pending()

    def read(self, max_tokens):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        with self.cond:
            while True:
                out = self.tok.read(max_tokens)
                if out:
                    self.cond.notify_all()   # the writer may have room again
                    return out
                if self.tok.closed and self.tok.pending() == 0:
                    return []
                self.cond.wait()
