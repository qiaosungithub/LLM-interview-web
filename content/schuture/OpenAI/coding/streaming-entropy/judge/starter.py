import numpy as np


def softmax_entropy(logits: np.ndarray) -> float:
    """logits: shape (N,) float array, N >= 1, every entry finite. Returns H(softmax(logits))."""
    raise NotImplementedError


def softmax_entropy_stable(logits: np.ndarray) -> float:
    """Same contract as softmax_entropy, correct and finite for every finite input."""
    raise NotImplementedError


def softmax_entropy_blockwise(logits: np.ndarray, block_size: int = 2) -> float:
    """Same contract as softmax_entropy_stable, using O(1) extra space for a fixed block_size."""
    raise NotImplementedError


def softmax_entropy_online(blocks) -> float:
    """blocks: an iterable of 1-D float arrays consumed once, in order. Returns the entropy of their concatenation."""
    raise NotImplementedError
