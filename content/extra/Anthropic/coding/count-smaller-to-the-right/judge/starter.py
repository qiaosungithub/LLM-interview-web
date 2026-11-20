def count_smaller_brute(nums: list[int]) -> list[int]:
    """counts[i] = number of j > i with nums[j] < nums[i]."""
    raise NotImplementedError


def count_smaller_merge(nums: list[int]) -> list[int]:
    """Same result as count_smaller_brute, in O(n log n) time. Compares elements with < only."""
    raise NotImplementedError


def count_smaller_fenwick(nums: list[int]) -> list[int]:
    """Same result as count_smaller_brute, with a Fenwick tree over the compressed values."""
    raise NotImplementedError


class OnlineSmallerCounter:
    def __init__(self, lo: int, hi: int) -> None:
        """All values passed to add() lie in [lo, hi]."""

    def add(self, x: int) -> int:
        """Returns how many of the values added before this call are strictly smaller than x,
        then records x. Raises ValueError (and records nothing) if x is outside [lo, hi].
        O(log(hi - lo + 1)) per call."""
        raise NotImplementedError
