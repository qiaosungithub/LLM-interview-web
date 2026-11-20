class InMemoryDatabase:
    def __init__(self) -> None:
        """Creates an empty database."""

    # Level 1
    def set(self, key: str, field: str, value: str) -> str:
        """Sets field of record key to value, creating or overwriting it. Returns ""."""
        raise NotImplementedError

    def get(self, key: str, field: str) -> str:
        """Returns the value of the field, or "" if the record or the field does not exist."""
        raise NotImplementedError

    def delete(self, key: str, field: str) -> str:
        """Removes the field. Returns "true" if it existed, "false" otherwise."""
        raise NotImplementedError

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        """If the field exists and equals expected_value, sets it to new_value and returns "true"; else "false"."""
        raise NotImplementedError

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        """If the field exists and equals expected_value, removes it and returns "true"; else "false"."""
        raise NotImplementedError

    # Level 2
    def scan(self, key: str) -> str:
        """All fields of the record as "f1(v1), f2(v2), ..." sorted by field name; "" if none."""
        raise NotImplementedError

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        """The same as scan, but only the fields whose name starts with prefix."""
        raise NotImplementedError

    # Level 3
    def set_at(self, key: str, field: str, value: str, timestamp: int) -> str:
        """set at timestamp; the field never expires."""
        raise NotImplementedError

    def set_at_with_ttl(self, key: str, field: str, value: str, timestamp: int, ttl: int) -> str:
        """set at timestamp; the field lives during [timestamp, timestamp + ttl)."""
        raise NotImplementedError

    def get_at(self, key: str, field: str, timestamp: int) -> str:
        """get at timestamp; an expired field does not exist."""
        raise NotImplementedError

    def delete_at(self, key: str, field: str, timestamp: int) -> str:
        """delete at timestamp; an expired field does not exist."""
        raise NotImplementedError

    def scan_at(self, key: str, timestamp: int) -> str:
        """scan at timestamp, skipping expired fields."""
        raise NotImplementedError

    def scan_by_prefix_at(self, key: str, prefix: str, timestamp: int) -> str:
        """scan_by_prefix at timestamp, skipping expired fields."""
        raise NotImplementedError

    def compare_and_set_at(self, key: str, field: str, expected_value: str, new_value: str,
                           timestamp: int) -> str:
        """compare_and_set at timestamp; on success it writes exactly like set_at."""
        raise NotImplementedError

    def compare_and_set_at_with_ttl(self, key: str, field: str, expected_value: str, new_value: str,
                                    timestamp: int, ttl: int) -> str:
        """compare_and_set at timestamp; on success it writes exactly like set_at_with_ttl."""
        raise NotImplementedError

    def compare_and_delete_at(self, key: str, field: str, expected_value: str, timestamp: int) -> str:
        """compare_and_delete at timestamp; an expired field does not exist."""
        raise NotImplementedError

    # Level 4
    def backup(self, timestamp: int) -> str:
        """Saves the unexpired fields and their remaining lifetimes; returns the count of non-empty records."""
        raise NotImplementedError

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        """Replaces the database with the latest backup made at or before timestamp_to_restore. Returns ""."""
        raise NotImplementedError

    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> str:
        """Returns what get_at(key, field, at_timestamp) returned at the moment at_timestamp."""
        raise NotImplementedError
