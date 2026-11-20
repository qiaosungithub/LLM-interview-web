"""The statement's FileSystem, a way to make the "process" die in the middle of a call, and string pools.

A replay shown to the user starts with `from ref import ...`: this folder is on sys.path in 自定义运行 too, so the
replay runs on exactly the file system the tests used."""


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


# ---------------------------------------------------------------- crashes

class Crash(BaseException):
    """The process died. A BaseException, so that `except Exception` in an answer does not swallow it."""


class CrashFS(FileSystem):
    """A FileSystem that crash() can kill. Until then it behaves exactly like FileSystem, except that list() gives
    the names newest first (the statement leaves the order arbitrary). `calls` records every write, append and
    delete as (method, name, data)."""

    def __init__(self, max_file_size: int | None = None):
        super().__init__(max_file_size)
        self.calls = []
        self._left = None           # mutating calls still allowed before the crash; None: no crash planned
        self._tear = None           # (keep, zero) for an append that dies midway
        self._dead = False
        self.torn = None            # (data, what was left of it) for the append that died midway

    def _mutate(self, method, name, data=b""):
        if self._dead:
            raise Crash
        if self._left == 0:
            self._dead = True
            if method == "append" and self._tear is not None:
                keep, zero, hole = self._tear
                frag = data if keep is None else data[:keep]
                zero = min(zero, len(frag))
                frag = frag[:len(frag) - zero] + bytes(zero)
                if hole is not None:
                    i, j = max(0, hole[0]), min(len(frag), hole[1])
                    if i < j:
                        frag = frag[:i] + bytes(j - i) + frag[j:]
                FileSystem.append(self, name, frag)
                self.torn = (bytes(data), frag)
            raise Crash
        if self._left is not None:
            self._left -= 1
        self.calls.append((method, name, bytes(data)))

    def write(self, name, data):
        self._mutate("write", name, data)
        super().write(name, data)

    def append(self, name, data):
        self._mutate("append", name, data)
        super().append(name, data)

    def delete(self, name):
        self._mutate("delete", name)
        super().delete(name)

    def read(self, name):
        if self._dead:
            raise Crash
        return super().read(name)

    def list(self):
        if self._dead:
            raise Crash
        return super().list()[::-1]


def crash(fs, fn, *args, after, keep=None, zero=0, hole=None):
    """Calls fn(*args), and the process dies just before fn's fs call number `after` (counting write, append and
    delete from 0): that call has no effect, except that an append given `keep`, `zero` or `hole` first leaves part
    of its data at the end of the file: data[:keep] (all of it if keep is None) with the last `zero` of those bytes
    zeroed, and with bytes hole[0] to hole[1] - 1 of them zeroed too. Returns True if fn finished before reaching
    that call, so that nothing died."""
    fs._left, fs._dead, fs.torn = after, False, None
    fs._tear = (keep, zero, hole) if keep is not None or zero or hole else None
    try:
        fn(*args)
    except Crash:
        pass
    finally:
        died = fs._dead
        fs._left, fs._tear, fs._dead = None, None, False
    return not died


def copy_fs(fs, without=None):
    """A plain FileSystem holding the same files as fs, minus the file named `without`."""
    out = FileSystem(fs.max_file_size)
    out._files = {k: v for k, v in fs._files.items() if k != without}
    return out


class FastFS(FileSystem):
    """FileSystem with appends that cost O(len(data)) instead of O(file size), to build big logs quickly. Seen
    through its methods it behaves exactly like FileSystem (it has no size cap)."""

    def __init__(self):
        super().__init__(None)
        self._parts = {}

    def _flush(self, name):
        parts = self._parts.pop(name, None)
        if parts:
            self._files[name] = self._files[name] + b"".join(parts)

    def write(self, name, data):
        self._parts.pop(name, None)
        super().write(name, data)

    def append(self, name, data):
        data = bytes(data)
        self._files.setdefault(name, b"")
        self._parts.setdefault(name, []).append(data)

    def read(self, name):
        self._flush(name)
        return super().read(name)

    def delete(self, name):
        self._parts.pop(name, None)
        super().delete(name)

    def plain(self):
        """The same files on a plain FileSystem."""
        for name in list(self._parts):
            self._flush(name)
        return copy_fs(self)


# ---------------------------------------------------------------- strings

# delimiters, escapes, NUL, a length header's worth of zero bytes, emoji, CJK, U+10FFFF
WEIRD = ["", "a", "a:b", "a,b", "a=b", ":", "\n", "key\nwith\nnewlines", "\x00", "\x00null\x00byte", "\x00" * 12,
         "\\", "\\:", "a\\", "\\n", "\\\\", "\\x00", "\"'", ":,=\n\x00", "\r\n", "\t",
         "emoji \U0001F600 \U0001F4A9", "\U0001F600", "中文 and English", "\U0010FFFF", "é", "\x7f\x80\xff"]


def string(r, short=False):
    """A random str: often one of WEIRD, otherwise random characters from every plane (no lone surrogates)."""
    if r.random() < 0.5:
        return r.choice(WEIRD)
    pools = [("a", "e"), ("0", "9"), (":", "="), ("\x00", "\x1f"), ("à", "ÿ"), ("一", "丐"),
             ("\U0001F600", "\U0001F610"), ("\U0010FFF0", "\U0010FFFF")]
    n = r.randint(0, 6 if short else 20)
    out = []
    for _ in range(n):
        lo, hi = r.choice(pools)
        out.append(chr(r.randint(ord(lo), ord(hi))))
    return "".join(out)
