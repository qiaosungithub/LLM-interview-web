class FormulaError(Exception):
    """Raised for a name that is not a cell name, or a value that is neither an int nor a valid formula."""


class CycleError(Exception):
    """Raised for a circular dependency."""


class LazySpreadsheet:
    def __init__(self):
        """Part 1: an empty sheet that evaluates formulas on demand."""

    def set(self, name: str, value: int | str) -> None:
        """Stores value for name. Raises FormulaError if name or value is not of the shape above."""
        raise NotImplementedError

    def get(self, name: str) -> int:
        """Evaluates name recursively; raises CycleError if that needs a cell on a circular dependency."""
        raise NotImplementedError


class Spreadsheet:
    def __init__(self):
        """Part 2: an empty sheet that recomputes affected cells on every set."""

    def set(self, name: str, value: int | str) -> None:
        """Like LazySpreadsheet.set, but recomputes every affected cell and rejects a cycle with CycleError."""
        raise NotImplementedError

    def get(self, name: str) -> int:
        """O(1): returns the already-computed value of name."""
        raise NotImplementedError
