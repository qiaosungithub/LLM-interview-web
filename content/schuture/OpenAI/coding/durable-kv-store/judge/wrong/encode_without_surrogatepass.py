# expect: RE
# Uses the default strict UTF-8 codec, which raises on a lone surrogate such as '\ud800' although any str is allowed.
# Given by the statement, unchanged.
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


# ---------------------------------------------------------------- Part 1

def _encode_str(s):
    raw = s.encode("utf-8")
    return len(raw).to_bytes(8, "big") + raw  # NOTE: the length counts BYTES; len(s) would count code points


def _read_str(data, pos):
    length = int.from_bytes(data[pos:pos + 8], "big")
    raw = data[pos + 8:pos + 8 + length]
    return raw.decode("utf-8"), pos + 8 + length


def _serialize_store(data):
    return b"".join(_encode_str(k) + _encode_str(v) for k, v in data.items())


def _deserialize_store(blob):
    store, pos = {}, 0
    while pos < len(blob):
        key, pos = _read_str(blob, pos)
        value, pos = _read_str(blob, pos)
        store[key] = value
    return store


class KVStore:
    FILE_NAME = "store"

    def __init__(self, fs):
        self.fs = fs
        self._data = {}

    def put(self, key, value):
        self._data[key] = value

    def get(self, key):
        return self._data.get(key)

    def save(self):
        self.fs.write(self.FILE_NAME, _serialize_store(self._data))  # NOTE: one write replaces the whole file

    def load(self):
        blob = self.fs.read(self.FILE_NAME)
        self._data = _deserialize_store(blob) if blob is not None else {}


# ---------------------------------------------------------------- Part 2

import zlib


class CorruptStoreError(RuntimeError):
    pass


def _encode_manifest(*fields):  # generation, num_chunks, total_length, checksum
    return b"".join(field.to_bytes(8, "big") for field in fields)


def _decode_manifest(data):
    return [int.from_bytes(data[i:i + 8], "big") for i in range(0, 32, 8)]


class ChunkedKVStore(KVStore):
    MANIFEST = "manifest"

    def save(self):
        old = self.fs.read(self.MANIFEST)
        generation = _decode_manifest(old)[0] + 1 if old is not None else 0  # NOTE: never the live generation

        blob = _serialize_store(self._data)
        cap = self.fs.max_file_size or len(blob)  # no cap: a single chunk
        chunks = [blob[i:i + cap] for i in range(0, len(blob), cap)] if blob else []
        names = [f"chunk_{generation}_{i}" for i in range(len(chunks))]
        for name, chunk in zip(names, chunks):  # NOTE: a cut may fall inside a length header or a character;
            self.fs.write(name, chunk)          # load() re-joins the chunks before parsing
        # NOTE: the commit point -- one atomic write switches load() to the new generation
        self.fs.write(self.MANIFEST, _encode_manifest(generation, len(chunks), len(blob), zlib.crc32(blob)))

        keep = set(names)
        for name in self.fs.list():  # NOTE: not only the previous generation -- interrupted saves leave others
            if name.startswith("chunk_") and name not in keep:
                self.fs.delete(name)

    def load(self):
        manifest = self.fs.read(self.MANIFEST)
        if manifest is None:
            self._data = {}
            return
        generation, num_chunks, total_length, checksum = _decode_manifest(manifest)
        blob = b"".join(self.fs.read(f"chunk_{generation}_{i}") or b"" for i in range(num_chunks))
        if len(blob) != total_length or zlib.crc32(blob) != checksum:
            raise CorruptStoreError("chunk contents do not match the manifest")
        self._data = _deserialize_store(blob)


# ---------------------------------------------------------------- Part 3

PUT, DELETE = 1, 0


def _encode_record(op, key, value=""):
    payload = bytes([op]) + _encode_str(key) + (_encode_str(value) if op == PUT else b"")
    header = len(payload).to_bytes(8, "big")
    return header + zlib.crc32(header + payload).to_bytes(4, "big") + payload  # NOTE: the checksum covers the length


def _read_record(data, pos):
    """Returns (op, key, value, end), or None if no complete, checksum-valid record starts at pos."""
    if pos + 12 > len(data):
        return None  # NOTE: not even a full header remains
    length = int.from_bytes(data[pos:pos + 8], "big")
    end = pos + 12 + length
    if end > len(data):
        return None  # NOTE: cut off mid-record, or a damaged (huge) length
    payload = data[pos + 12:end]
    if zlib.crc32(data[pos:pos + 8] + payload) != int.from_bytes(data[pos + 8:pos + 12], "big"):
        return None  # NOTE: enough bytes, but not the ones that were written
    op = payload[0]
    key, kpos = _read_str(payload, 1)
    value = _read_str(payload, kpos)[0] if op == PUT else ""
    return op, key, value, end


class LogKVStore(KVStore):
    LOG = "log"

    def put(self, key, value):
        self.fs.append(self.LOG, _encode_record(PUT, key, value))  # NOTE: log first, memory second
        super().put(key, value)

    def delete(self, key):
        self.fs.append(self.LOG, _encode_record(DELETE, key))
        self._data.pop(key, None)

    def load(self):
        data = self.fs.read(self.LOG) or b""
        store, pos = {}, 0
        while (record := _read_record(data, pos)) is not None:  # NOTE: stop at the first bad record, for good
            op, key, value, pos = record
            if op == PUT:
                store[key] = value
            else:
                store.pop(key, None)
        if pos < len(data):  # NOTE: cut the bad tail off, or later appends land behind it and are never replayed
            self.fs.write(self.LOG, data[:pos])
        self._data = store

    def compact(self):
        self.fs.write(self.LOG, b"".join(_encode_record(PUT, k, v) for k, v in self._data.items()))
