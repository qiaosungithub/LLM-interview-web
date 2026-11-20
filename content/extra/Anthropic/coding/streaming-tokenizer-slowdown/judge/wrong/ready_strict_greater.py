# expect: WA
# Wrong: a token counts as final only with more than L characters left (> instead of >=), so reads return too few.
import threading
from collections import deque


class StreamTokenizer:
    """Same contract as the given version; total work is linear in the text length."""

    def __init__(self, vocab, unk_id):
        if not vocab:
            raise ValueError("empty vocabulary")
        self.vocab = vocab
        self.unk_id = unk_id
        self.max_len = max(len(key) for key in vocab)
        # Distinct key lengths, longest first: at most max_len dict lookups per token.
        self.lengths = sorted({len(key) for key in vocab}, reverse=True)
        self.buf = ""              # working text; the next token starts at self.pos
        self.pos = 0
        self.incoming = deque()    # chunks written but not yet moved into buf
        self.incoming_len = 0
        self.closed = False
        self.chars_in = 0
        self.tokens_out = 0

    def write(self, chunk):
        if self.closed:
            raise ValueError("write after close")
        if chunk:
            self.incoming.append(chunk)
            self.incoming_len += len(chunk)
        self.chars_in += len(chunk)

    def close(self):
        self.closed = True

    def pending(self):
        return len(self.buf) - self.pos + self.incoming_len

    def _ready(self):
        left = len(self.buf) - self.pos
        if left < self.max_len and self.incoming:
            # Only now copy: fewer than max_len old characters plus the new chunks.
            self.buf = self.buf[self.pos:] + "".join(self.incoming)
            self.pos = 0
            self.incoming.clear()
            self.incoming_len = 0
            left = len(self.buf)
        if left == 0:
            return False
        return self.closed or left > self.max_len

    def _longest_match(self):
        buf, pos = self.buf, self.pos
        for n in self.lengths:
            if pos + n <= len(buf):
                token_id = self.vocab.get(buf[pos:pos + n])
                if token_id is not None:
                    return token_id, n
        return self.unk_id, 1

    def read(self, max_tokens):
        out = []
        while len(out) < max_tokens and self._ready():
            token_id, n = self._longest_match()
            out.append(token_id)
            self.pos += n
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
