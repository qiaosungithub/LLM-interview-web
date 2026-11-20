# expect: WA
# The lock covers only the bookkeeping and func runs outside it, but there is no in-flight table, so concurrent misses on one key all call func.
import json
import os
import threading
import time
import zlib
from collections import OrderedDict
from typing import Any, Callable


class Memoize:
    """Wraps func so that repeated calls with the same arguments return a cached result instead of
    calling func again. Holds at most capacity entries; a call that would add a new one beyond that
    evicts the least recently used entry first. Both a hit and a miss count as a "use" of the key
    they touch, for the purpose of deciding which entry is least recently used."""

    def __init__(self, func: Callable, capacity: int) -> None:
        self.func = func
        self.capacity = capacity
        self.cache: "OrderedDict[Any, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def generate_key(self, *args: Any, **kwargs: Any) -> Any:
        return (args, tuple(sorted(kwargs.items())))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self.generate_key(*args, **kwargs)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)  # NOTE: a hit is a use too -- without this the cache degrades to FIFO
            return self.cache[key]
        self.misses += 1
        result = self.func(*args, **kwargs)
        self.cache[key] = result
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result


PUT, TOUCH = "put", "touch"


def _decode_key(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode_key(item) for item in value)
    return value


def _encode_record(op: str, key: Any, value: Any = None) -> bytes:
    body = {"op": op, "key": key}
    if op == PUT:
        body["value"] = value
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    header = len(payload).to_bytes(8, "big")
    checksum = zlib.crc32(header + payload).to_bytes(4, "big")  # NOTE: covers the header too -- see above
    return header + checksum + payload


def _read_record(data: bytes, pos: int):
    """Returns (op, key, value, end) for the record starting at pos, or None if no complete,
    checksum-valid record starts there."""
    if pos + 12 > len(data):
        return None                                        # not even a full header and checksum remain
    length = int.from_bytes(data[pos:pos + 8], "big")
    end = pos + 12 + length
    if end > len(data):
        return None                                         # cut off mid-record, or a damaged (huge) length
    payload = data[pos + 12:end]
    if zlib.crc32(data[pos:pos + 8] + payload) != int.from_bytes(data[pos + 8:pos + 12], "big"):
        return None                                         # enough bytes, but not the ones that were written
    try:
        body = json.loads(payload)
        op, key = body["op"], _decode_key(body["key"])
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        return None                                          # checksum-valid but not one of our records
    return op, key, body.get("value"), end


def _fsync_directory(path: str) -> None:
    dir_fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    try:
        os.fsync(dir_fd)   # NOTE: the rename below is durable only once the directory entry is synced too
    finally:
        os.close(dir_fd)


class DurableMemoize(Memoize):
    def __init__(self, func: Callable, capacity: int, path: str | None = None) -> None:
        super().__init__(func, capacity)
        self.path = path
        self._log = None
        if path is not None:
            self._load()
            self._log = open(path, "ab")   # NOTE: kept open and appended to for the life of this instance

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as f:
            data = f.read()
        pos = 0
        while (record := _read_record(data, pos)) is not None:
            op, key, value, pos = record
            if op == PUT:
                self.cache[key] = value
                self.cache.move_to_end(key)               # NOTE: reassigning an existing key does not reorder it
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
            elif key in self.cache:                        # TOUCH; always true for a log this class wrote
                self.cache.move_to_end(key)
        if pos < len(data):                                # NOTE: cut the bad tail, or later appends hide behind it
            with open(self.path, "r+b") as f:
                f.truncate(pos)
                f.flush()
                os.fsync(f.fileno())

    def _append(self, op: str, key: Any, value: Any = None) -> None:
        if self._log is None:
            return
        self._log.write(_encode_record(op, key, value))
        self._log.flush()
        os.fsync(self._log.fileno())          # NOTE: the durability point -- __call__ does not return before this

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self.generate_key(*args, **kwargs)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            self._append(TOUCH, key)
            return self.cache[key]
        self.misses += 1
        result = self.func(*args, **kwargs)
        self._append(PUT, key, result)         # NOTE: durable before the entry is even visible in memory
        self.cache[key] = result
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result

    def compact(self) -> None:
        if self.path is None:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:                # NOTE: a fresh log -- one PUT per cached entry, nothing else
            for key, value in self.cache.items():            # oldest (LRU) first, so replay rebuilds the order
                f.write(_encode_record(PUT, key, value))
            f.flush()
            os.fsync(f.fileno())
        self._log.close()
        os.replace(tmp, self.path)                # NOTE: the only commit point -- atomic on the same filesystem
        _fsync_directory(self.path)
        self._log = open(self.path, "ab")


class _InFlight:
    """Coordinates one key's in-progress computation: a thread that is not the leader waits on
    event, then reads result (or re-raises exc)."""

    __slots__ = ("event", "result", "exc")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.exc: BaseException | None = None


class ThreadSafeMemoize(DurableMemoize):
    def __init__(self, func: Callable, capacity: int, path: str | None = None) -> None:
        super().__init__(func, capacity, path)
        self._lock = threading.Lock()
        self._in_flight: dict[Any, _InFlight] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self.generate_key(*args, **kwargs)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
                self._append(TOUCH, key)
                return self.cache[key]
        result = self.func(*args, **kwargs)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self._append(TOUCH, key)
                return self.cache[key]
            self.misses += 1
            self._append(PUT, key, result)
            self.cache[key] = result
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return result

    def compact(self) -> None:
        with self._lock:                            # NOTE: no call may touch the cache or the log mid-swap
            super().compact()
