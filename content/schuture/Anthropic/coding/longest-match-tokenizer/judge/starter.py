def tokenize(text: str, vocab: dict[str, int], unk_id: int) -> list[int]:
    """Part 1: greedy longest match from the left. One id per matched token, unk_id for each
    character that matches no key of vocab."""
    raise NotImplementedError


def tokenize_merged(text: str, vocab: dict[str, int], unk_id: int) -> list[int]:
    """Part 2: same token sequence as tokenize, except that a maximal run of unk_id collapses to
    one unk_id. Tries at most L candidate lengths per position (L = the longest key of vocab)."""
    raise NotImplementedError


class TrieTokenizer:
    def __init__(self, vocab: dict[str, int], unk_id: int) -> None:
        """Part 3: builds a trie from vocab once, for reuse across many calls to tokenize."""

    def tokenize(self, text: str) -> list[int]:
        """Returns exactly what tokenize_merged(text, vocab, unk_id) would, computed by walking the trie."""
        raise NotImplementedError
