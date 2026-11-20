COMPRESSED_SUFFIX = ".cmp"


class CloudStorage:
    def __init__(self) -> None:
        """Creates an empty store: no files and no users yet."""

    def add_file(self, name: str, size: int) -> bool:
        """Level 1: creates a file called name with this size; returns True, or False, changing nothing, if a
        file called name already exists."""
        raise NotImplementedError

    def copy_file(self, source: str, destination: str) -> bool:
        """Level 1: creates destination with source's size; returns True, or False, changing nothing, if
        source does not exist or destination already exists (Level 3 adds the owner's capacity check)."""
        raise NotImplementedError

    def get_file_size(self, name: str) -> int | None:
        """Level 1: returns the current size of name, or None if no file called name exists."""
        raise NotImplementedError

    def find_files(self, prefix: str, suffix: str) -> list[str]:
        """Level 2: returns "name(size)" for every file whose name starts with prefix and ends with suffix,
        sorted by size descending, then by name ascending."""
        raise NotImplementedError

    def add_user(self, user_id: str, capacity: int) -> bool:
        """Level 3: registers user_id with this capacity and no files; returns True, or False if user_id is
        already a user."""
        raise NotImplementedError

    def add_file_by(self, user_id: str, name: str, size: int) -> int | None:
        """Level 3: creates a file owned by user_id and returns user_id's remaining capacity, or None,
        changing nothing, if it is not allowed."""
        raise NotImplementedError

    def merge_user(self, target: str, source: str) -> int | None:
        """Level 3: moves source's files and capacity to target, removes source, and returns target's
        remaining capacity, or None, changing nothing, if it is not allowed."""
        raise NotImplementedError

    def compress_file(self, user_id: str, name: str) -> str | None:
        """Level 4: replaces name, owned by user_id, with name + COMPRESSED_SUFFIX of size ceil(size / 2);
        returns the new name, or None, changing nothing, if it is not allowed."""
        raise NotImplementedError

    def decompress_file(self, user_id: str, name: str) -> str | None:
        """Level 4: replaces name, owned by user_id, with name minus its trailing COMPRESSED_SUFFIX at double
        the size; returns the new name, or None, changing nothing, if it is not allowed."""
        raise NotImplementedError

    def backup_user(self, user_id: str) -> bool:
        """Level 4: records user_id's files (names and sizes) and capacity as its one backup; returns True, or
        False if user_id is not a user."""
        raise NotImplementedError

    def restore_user(self, user_id: str) -> int | None:
        """Level 4: resets user_id's files and capacity to its most recent backup and returns the remaining
        capacity, or None, changing nothing, if it is not allowed."""
        raise NotImplementedError
