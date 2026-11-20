class Database:
    def __init__(self) -> None:
        """Creates an empty database with no tables."""

    def create_table(self, table: str, columns: list[str]) -> None:
        """Part 1: creates an empty table with these columns, in this order."""
        raise NotImplementedError

    def insert(self, table: str, row: dict[str, int | str | None]) -> None:
        """Part 1: appends a copy of row to table."""
        raise NotImplementedError

    def select(self, table: str, columns: list[str] | None = None,
               where: list[tuple[str, str, int | str | None]] | None = None,
               order_by: list[tuple[str, bool]] | None = None) -> list[dict[str, int | str | None]]:
        """Parts 1-3: the rows of table that satisfy every where condition, sorted by order_by (insertion order
        otherwise), projected to columns (all of the table's columns, in schema order, if columns is None)."""
        raise NotImplementedError

    def create_index(self, table: str, column: str, kind: str) -> None:
        """Part 4: kind is "hash" or "sorted". Rebuilds the index if one already exists for this (column, kind)."""
        raise NotImplementedError
