class FileSystem:
    """An in-memory mock file system that stores named byte blobs. Files are independent: writing
    "b" never touches "a". write and delete are atomic: if the process dies, the call has either
    taken full effect or had no effect at all. append is not atomic."""

    def __init__(self, max_file_size: int | None = None):
        self.max_file_size = max_file_size
        self._files: dict[str, bytes] = {}

    def write(self, name: str, data: bytes) -> None:
        """Creates or overwrites the file called name with exactly data. Raises ValueError if
        max_file_size is set and len(data) exceeds it; name is then left exactly as it was before
        this call (unchanged, or still absent)."""
        if self.max_file_size is not None and len(data) > self.max_file_size:
            raise ValueError(f"{name}: {len(data)} bytes exceeds max_file_size={self.max_file_size}")
        self._files[name] = data

    def append(self, name: str, data: bytes) -> None:
        """Adds data to the end of the file called name, creating it if absent; raises ValueError,
        changing nothing, if the file would then exceed max_file_size. NOT atomic: if the process
        dies during this call, the file may be left ending in any prefix of data, and the bytes of
        that prefix may be damaged (zeroed, for instance)."""
        old = self._files.get(name, b"")
        if self.max_file_size is not None and len(old) + len(data) > self.max_file_size:
            raise ValueError(f"{name}: {len(old) + len(data)} bytes exceeds max_file_size={self.max_file_size}")
        self._files[name] = old + data

    def read(self, name: str) -> bytes | None:
        """Returns the current bytes of name, or None if no such file exists."""
        return self._files.get(name)

    def list(self) -> list[str]:
        """Names of every file that currently exists, in arbitrary order."""
        return list(self._files)

    def delete(self, name: str) -> None:
        """Removes name if present; a no-op if it does not exist."""
        self._files.pop(name, None)


class KVStore:
    def __init__(self, fs: FileSystem):
        """Part 1: fs is the FileSystem this store persists to and restores from."""

    def put(self, key: str, value: str) -> None:
        """Sets key -> value in memory, overwriting any previous value for key."""
        raise NotImplementedError

    def get(self, key: str) -> str | None:
        """Returns the current value of key, or None if key has no value."""
        raise NotImplementedError

    def save(self) -> None:
        """Persists the entire in-memory store to fs."""
        raise NotImplementedError

    def load(self) -> None:
        """Replaces the in-memory store with what fs currently holds."""
        raise NotImplementedError


class ChunkedKVStore:
    def __init__(self, fs: FileSystem):
        """Part 2: fs is the FileSystem this store persists to and restores from; fs.max_file_size may be set."""

    def put(self, key: str, value: str) -> None:
        """Same contract as KVStore.put."""
        raise NotImplementedError

    def get(self, key: str) -> str | None:
        """Same contract as KVStore.get."""
        raise NotImplementedError

    def save(self) -> None:
        """Persists the entire in-memory store to fs, never writing more than fs.max_file_size bytes to any single file."""
        raise NotImplementedError

    def load(self) -> None:
        """Replaces the in-memory store with what fs currently holds."""
        raise NotImplementedError


class LogKVStore:
    def __init__(self, fs: FileSystem):
        """Part 3: fs is the FileSystem this store appends to and recovers from."""

    def put(self, key: str, value: str) -> None:
        """Durably records key -> value before returning."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Durably records the removal of key before returning."""
        raise NotImplementedError

    def get(self, key: str) -> str | None:
        """Same contract as KVStore.get."""
        raise NotImplementedError

    def load(self) -> None:
        """Rebuilds the in-memory store from what fs currently holds."""
        raise NotImplementedError

    def compact(self) -> None:
        """Rewrites what is stored so that its size no longer depends on the history of overwritten or deleted keys."""
        raise NotImplementedError
