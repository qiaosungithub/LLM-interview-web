class FileSystem:
    def __init__(self) -> None:
        """Creates a file system holding only the root directory "/"."""

    def mkdir(self, path: str) -> bool:
        """Level 1: creates a directory at path and every missing ancestor (like mkdir -p); True once it
        exists, False, without effect, for an invalid path or if path or an ancestor is a file."""
        raise NotImplementedError

    def write(self, path: str, content: str, owner: str = "admin") -> bool:
        """Level 1: creates or overwrites the file at path with content; True on success, False, without
        effect, if it is not allowed (Level 3 adds owner and its quota rules)."""
        raise NotImplementedError

    def read(self, path: str) -> str | None:
        """Level 1: returns the content of the file at path, or None if there is no file there."""
        raise NotImplementedError

    def ls(self, path: str) -> list[str] | None:
        """Level 1: returns the names of path's immediate children, sorted, or None if path is not an
        existing directory."""
        raise NotImplementedError

    def mv(self, src: str, dst: str) -> bool:
        """Level 2: moves src (and its subtree) to dst; True on success, False, without effect, if it is
        not allowed."""
        raise NotImplementedError

    def cp(self, src: str, dst: str) -> bool:
        """Level 2: like mv, but src stays and the copy at dst is independent of it."""
        raise NotImplementedError

    def rm(self, path: str) -> bool:
        """Level 2: deletes whatever is at path, recursively; True if something was deleted, False,
        without effect, otherwise (the root is never deleted)."""
        raise NotImplementedError

    def size(self, path: str) -> int | None:
        """Level 2: returns a file's size, or the total size of every file under a directory, or None if
        nothing exists at path."""
        raise NotImplementedError

    def add_user(self, user_id: str, capacity: int) -> bool:
        """Level 3: registers user_id with this storage capacity and a used total of 0; True on success,
        False, without effect, if it is not allowed."""
        raise NotImplementedError

    def update_capacity(self, user_id: str, capacity: int) -> int | None:
        """Level 3: sets user_id's capacity, deleting its largest files until it fits; returns the number
        of files deleted, or None, without effect, if it is not allowed."""
        raise NotImplementedError

    def snapshot(self, snapshot_id: str) -> bool:
        """Level 4: captures the entire current state under snapshot_id; True, or False, without effect,
        if snapshot_id was already used."""
        raise NotImplementedError

    def restore(self, snapshot_id: str) -> bool:
        """Level 4: replaces the entire current state with an independent copy of the snapshot; True, or
        False, without effect, if no snapshot was captured under snapshot_id."""
        raise NotImplementedError
