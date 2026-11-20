import numpy as np


class KDTree:
    def __init__(self, P: np.ndarray) -> None:
        """Build the median-split k-d tree over the rows of P (shape (n, d), n >= 1), else raise ValueError."""
        raise NotImplementedError

    def nearest(self, q: np.ndarray) -> int:
        """Return the index of the point nearest to q (ties -> smallest index), searching the tree with pruning."""
        raise NotImplementedError

    def knn(self, q: np.ndarray, k: int) -> list[int]:
        """Return the k indices with the smallest (distance², index) pairs, ascending; 1 <= k <= n."""
        raise NotImplementedError

    def count_in_box(self, lo: np.ndarray, hi: np.ndarray) -> int:
        """Return how many points satisfy lo[i] <= p[i] <= hi[i] on every axis i (0 if some lo[i] > hi[i])."""
        raise NotImplementedError
