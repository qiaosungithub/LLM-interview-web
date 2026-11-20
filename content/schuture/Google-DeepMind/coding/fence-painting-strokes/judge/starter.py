def min_horizontal_strokes(h: list[int]) -> int:
    """Return the fewest horizontal strokes that paint every cell of the fence."""
    raise NotImplementedError


class Fence:
    def __init__(self, h: list[int]):
        """Start from the plank heights h, exactly like Part 1's input."""
        raise NotImplementedError

    def set_height(self, i: int, new_height: int) -> None:
        """Replace h[i] with new_height in O(1) time."""
        raise NotImplementedError

    def strokes(self) -> int:
        """Return, in O(1) time, Part 1's answer for the current heights."""
        raise NotImplementedError


def min_strokes(h: list[int]) -> int:
    """Return the fewest strokes, horizontal and vertical together, that paint every cell of the fence."""
    raise NotImplementedError
