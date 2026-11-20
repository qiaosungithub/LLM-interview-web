class StreamTokenizer:
    """Greedy longest-match tokenizer for text that arrives in pieces: same outputs as the given version, linear total work."""

    def __init__(self, vocab, unk_id):
        """vocab: dict[str, int] with at least one non-empty key; an empty vocabulary raises ValueError."""
        raise NotImplementedError

    def write(self, chunk):
        """Add text; a cut may fall anywhere, even inside a token. Raises ValueError after close()."""
        raise NotImplementedError

    def close(self):
        """No more text will be written."""
        raise NotImplementedError

    def pending(self):
        """Characters written but not yet turned into tokens."""
        raise NotImplementedError

    def read(self, max_tokens):
        """Up to max_tokens (>= 0) ids that can no longer change, in order; fewer (maybe none) if the rest needs more text."""
        raise NotImplementedError


class BoundedStreamTokenizer:
    """For one writer thread and one reader thread, with a bounded backlog."""

    def __init__(self, vocab, unk_id, max_pending):
        """Raises ValueError when max_pending is less than the longest key length L."""
        raise NotImplementedError

    def write(self, chunk):
        """Blocks while pending() >= max_pending, then appends the whole chunk."""
        raise NotImplementedError

    def close(self):
        """No more text will be written."""
        raise NotImplementedError

    def read(self, max_tokens):
        """max_tokens >= 1. Blocks until at least one token is final; returns [] only once the stream is closed and read out."""
        raise NotImplementedError

    def pending(self):
        """Characters written but not yet turned into tokens."""
        raise NotImplementedError
