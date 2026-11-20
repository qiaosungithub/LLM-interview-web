class InMemoryDatabase:
    def __init__(self) -> None:
        """Creates an empty database."""

    def set(self, key: str, field: str, value: str) -> str:
        """Level 1: sets field of the record key to value, creating the record if key is new and overwriting any
        value already held there. Returns ""."""
        raise NotImplementedError

    def get(self, key: str, field: str) -> str:
        """Level 1: returns the value held in field of key, or "" if key does not exist or holds no such field."""
        raise NotImplementedError

    def delete(self, key: str, field: str) -> str:
        """Level 1: removes field from key if present; "true" if a field was removed, "false" if key did not
        exist or held no such field."""
        raise NotImplementedError

    def scan(self, key: str) -> str:
        """Level 2: every field of key as "name1(value1), name2(value2), ...", sorted by name (Python's default
        string ordering); "" if key does not exist or holds no fields."""
        raise NotImplementedError

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        """Level 2: same as scan, restricted to the fields of key whose name starts with prefix (prefix=""
        behaves exactly like scan)."""
        raise NotImplementedError

    def set_at(self, timestamp: int, key: str, field: str, value: str) -> str:
        """Level 3: same as set, at timestamp; clears any TTL the field held, so the field becomes permanent."""
        raise NotImplementedError

    def set_at_with_ttl(self, timestamp: int, key: str, field: str, value: str, ttl: int) -> str:
        """Level 3: same as set_at, but field is visible from timestamp up to, but not including,
        timestamp + ttl; a later call replaces the window instead of extending it."""
        raise NotImplementedError

    def get_at(self, timestamp: int, key: str, field: str) -> str:
        """Level 3: same as get, at timestamp; a field whose TTL window has elapsed counts as absent."""
        raise NotImplementedError

    def delete_at(self, timestamp: int, key: str, field: str) -> str:
        """Level 3: same as delete, at timestamp; a field whose TTL window has elapsed counts as already absent."""
        raise NotImplementedError

    def scan_at(self, timestamp: int, key: str) -> str:
        """Level 3: same as scan, at timestamp."""
        raise NotImplementedError

    def scan_by_prefix_at(self, timestamp: int, key: str, prefix: str) -> str:
        """Level 3: same as scan_by_prefix, at timestamp."""
        raise NotImplementedError

    def backup(self, timestamp: int) -> str:
        """Level 4: deep-snapshots every field live at timestamp, a TTL field as its remaining time until expiry;
        a second backup at the same timestamp replaces the first. Returns ""."""
        raise NotImplementedError

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        """Level 4: makes the latest backup at or before timestamp_to_restore the entire live state as of
        timestamp, each remaining TTL counting down again from timestamp; no backup is deleted. Returns ""."""
        raise NotImplementedError
