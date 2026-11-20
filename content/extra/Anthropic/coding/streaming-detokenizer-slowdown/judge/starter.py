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
    """Turns the token ids of one model reply into text deltas, with the same return values as the given version."""

    def __init__(self, tokenizer):
        """Start decoding one reply."""
        raise NotImplementedError

    def push(self, token_id):
        """Text that became final with this token ("" if none)."""
        raise NotImplementedError

    def finish(self):
        """Whatever is left, once the reply is over."""
        raise NotImplementedError

    def text(self):
        """Everything returned so far."""
        raise NotImplementedError


class StopStreamDecoder:
    """Part 2's decoder, cut just before the first stop string; attribute stopped: bool."""

    def __init__(self, tokenizer, stops):
        """stops: a list of non-empty strings (the list may be empty)."""
        raise NotImplementedError

    def push(self, token_id):
        """Text returned for this token (str)."""
        raise NotImplementedError

    def finish(self):
        """Text returned when the reply is over (str)."""
        raise NotImplementedError

    def text(self):
        """Everything returned so far."""
        raise NotImplementedError
