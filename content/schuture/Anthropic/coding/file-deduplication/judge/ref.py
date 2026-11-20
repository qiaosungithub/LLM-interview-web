"""Helpers for tests.py: temporary trees, a brute-force reference, a result checker and a hook that watches reads.

An answer can rebuild a failing case in a custom run: `from ref import make_tree`, then
`find_duplicate_files(make_tree({...}, links={...}, unreadable=[...]))`."""
import builtins
import io
import mmap
import os
import shutil
import stat
import tempfile
import threading
import time
from collections import Counter, defaultdict

from judgelib import Mismatch, short

_open = io.open
_os_open, _os_read, _os_pread, _os_close = os.open, os.read, os.pread, os.close
_mmap = mmap.mmap


# ---------------------------------------------------------------- trees

class Tree:
    """A tree at <fresh temp dir>/root. Relative paths are relative to root; "../outside/x" lies next to it."""

    def __init__(self):
        self.base = tempfile.mkdtemp()
        self.root = os.path.join(self.base, "root")
        os.mkdir(self.root)
        self.contents = {}          # absolute path -> bytes, for every regular file written
        self.symlinks = set()       # absolute paths of symbolic links
        self.linkdirs = set()       # the symbolic links that point at a directory
        self.unreadable = set()
        self.vanished = set()       # files a test deletes just before their first open
        self.lines = []             # human description, in creation order
        self.spec_files, self.spec_links, self.spec_unreadable, self.spec_dirs = {}, {}, [], []
        self.coded = True           # False once a file is too big to show as a literal

    def abs(self, rel: str) -> str:
        return os.path.normpath(os.path.join(self.root, rel))

    def rel(self, p) -> str:
        """How a path is shown to the user: root/..., or outside/... for files next to root."""
        if isinstance(p, str) and (p == self.base or p.startswith(self.base + os.sep)):
            return os.path.relpath(p, self.base)
        return p

    def dir(self, rel: str) -> None:
        os.makedirs(self.abs(rel), exist_ok=True)
        self.spec_dirs.append(rel)

    def file(self, rel: str, data) -> str:
        raw = data.encode() if isinstance(data, str) else bytes(data)
        p = self.abs(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with _open(p, "wb") as f:
            f.write(raw)
        self.contents[p] = raw
        self.spec_files[rel] = data
        if len(raw) > 40:
            self.coded = False
        self.lines.append((rel, "file", raw))
        return p

    def link(self, rel: str, target: str) -> str:
        """A symbolic link at rel to target (relative to root; it may not exist), stored as an absolute path."""
        p, t = self.abs(rel), self.abs(target)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        os.symlink(t, p)
        self.symlinks.add(p)
        if os.path.isdir(t):
            self.linkdirs.add(p)
        self.spec_links[rel] = target
        self.lines.append((rel, "link", target))
        return p

    def deny(self, rel: str) -> None:
        p = self.abs(rel)
        os.chmod(p, 0)
        try:
            with _open(p, "rb"):
                pass
        except PermissionError:
            pass
        else:       # running with privileges: chmod 000 would not make the file unreadable
            raise RuntimeError("judge sandbox can read a chmod-000 file; unreadable-file tests are invalid here")
        self.unreadable.add(p)
        self.spec_unreadable.append(rel)

    def cleanup(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    # ------------------------------------------------------------ display
    def call(self, fname: str, extra: str = "") -> str:
        """The call as code (with make_tree) when the tree is small, else the call plus a listing."""
        if self.coded:
            bits = [repr(self.spec_files)]
            if self.spec_links:
                bits.append(f"links={self.spec_links!r}")
            if self.spec_unreadable:
                bits.append(f"unreadable={self.spec_unreadable!r}")
            if self.spec_dirs:
                bits.append(f"dirs={self.spec_dirs!r}")
            text = f"{fname}(make_tree({', '.join(bits)}){extra})"
            if len(text) <= 600:
                return text
        labels = {}
        rows = []
        for rel, kind, what in self.lines:
            shown = os.path.normpath(os.path.join("root", rel))
            if kind == "link":
                rows.append(f"  {shown} -> {os.path.normpath(os.path.join('root', what))}（符号链接）")
                continue
            label = labels.setdefault(what, len(labels) + 1)
            tail = "（不可读，权限 000）" if self.abs(rel) in self.unreadable else ""
            rows.append(f"  {shown}：{len(what)} 字节，内容 #{label}{tail}")
        if len(rows) > 50:
            rows = rows[:50] + [f"  …… 共 {len(rows)} 项"]
        return (f"{fname}(root{extra})  # root 是下面这棵临时目录树（内容编号相同 = 字节完全相同）：\n"
                + "\n".join(rows))


def make_tree(files: dict, links: dict | None = None, unreadable=(), dirs=()) -> str:
    """Builds a tree in a fresh temporary directory and returns its root. Keys of files are paths relative to root
    (str values are written as UTF-8); links maps a link path to its target, both relative to root; a path
    starting with "../" lies outside root."""
    t = Tree()
    for rel in dirs:
        t.dir(rel)
    for rel, data in files.items():
        t.file(rel, data)
    for rel, target in (links or {}).items():
        t.link(rel, target)
    for rel in unreadable:
        t.deny(rel)
    return t.root


# ---------------------------------------------------------------- brute force

def _regular_files(d: str):
    for name in os.listdir(d):
        p = os.path.join(d, name)
        st = os.lstat(p)
        if stat.S_ISDIR(st.st_mode):
            yield from _regular_files(p)
        elif stat.S_ISREG(st.st_mode):
            yield p


def brute(root: str, skip=()) -> list:
    """Groups the readable regular files under root (except those in skip) by their whole content, never
    following a symlink."""
    by_content = defaultdict(list)
    for p in _regular_files(root):
        if p in skip:
            continue
        try:
            with _open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        by_content[data].append(p)
    groups = [sorted(g) for g in by_content.values() if len(g) >= 2]
    groups.sort(key=lambda g: g[0])
    return groups


# ---------------------------------------------------------------- checking

def _shown(tree: Tree, got):
    if isinstance(got, list) and all(isinstance(g, list) for g in got):
        return [[tree.rel(p) for p in g] for g in got]
    return got


def _why_extra(tree: Tree, p: str) -> str:
    r = tree.rel(p)
    if p in tree.symlinks:
        return f"{r} 是符号链接；符号链接永远不会被跟随，也不会出现在结果里"
    for d in tree.linkdirs:
        if p.startswith(d + os.sep):
            return f"{r} 只能经由指向目录的符号链接 {tree.rel(d)} 到达；这样的目录不应遍历"
    if p in tree.unreadable:
        return f"{r} 不可读（权限 000），应当跳过，不出现在任何分组里"
    if p in tree.vanished:
        return f"{r} 在被打开之前已经被删除，读不到内容，应当跳过"
    if p in tree.contents and p.startswith(tree.root + os.sep):
        return f"{r} 在树里没有内容完全相同的（可读）文件，是单例，不应出现在结果里"
    if not p.startswith(tree.root + os.sep):
        return (f"{r!r} 不是这棵树下文件的路径；路径应当是 os.walk 给出的样子，"
                f"即在传入的 root_path 之下用 os.path.join 拼接")
    return f"{r} 不应出现在结果里"


def normalize(got):
    """Tuples are accepted where lists are asked for, and os.PathLike paths are read as their str form.
    Returns (normalized, None) or (None, Chinese sentence)."""
    if got is None:
        return None, "返回了 None；没有任何重复文件时应当返回空列表 []"
    if not isinstance(got, (list, tuple)):
        return None, f"返回值应当是 list，得到 {type(got).__name__}"
    out = []
    for g in got:
        if not isinstance(g, (list, tuple)):
            return None, f"每一组应当是一个 list，得到 {type(g).__name__}：{short(g, 200)}"
        group = []
        for p in g:
            if isinstance(p, os.PathLike) and not isinstance(p, str):
                p = os.fspath(p)
            if not isinstance(p, str):
                return None, f"路径应当是 str（os.walk 给出的样子），得到 {type(p).__name__}：{short(p, 200)}"
            group.append(p)
        out.append(group)
    return out, None


def problem(got, want: list, tree: Tree):
    """None when got is exactly want (got already normalized), else a Chinese sentence saying what is wrong first."""
    if got == want:
        return None
    for g in got:
        if len(g) < 2:
            return f"每一组至少要有两个路径，结果里却有一组只有 {len(g)} 个：{[tree.rel(p) for p in g]}"
    seen = Counter(p for g in got for p in g)
    again = sorted(p for p, c in seen.items() if c > 1)
    if again:
        return f"{tree.rel(again[0])} 在结果里出现了不止一次"
    want_of = {p: g for g in want for p in g}
    extra = sorted(set(seen) - set(want_of))
    if extra:
        return _why_extra(tree, extra[0])
    missing = sorted(set(want_of) - set(seen))
    if missing:
        p = missing[0]
        mates = [tree.rel(q) for q in want_of[p] if q != p]
        return f"{tree.rel(p)} 和 {'、'.join(mates)} 的字节内容完全相同，应当出现在同一组里，结果里却没有它"
    want_sets = {frozenset(g) for g in want}
    for g in got:
        if frozenset(g) not in want_sets:
            first = want_of[g[0]]
            other = next((q for q in g if want_of[q] is not first), None)
            if other is not None:
                return (f"{tree.rel(g[0])} 和 {tree.rel(other)} 的字节内容不同，却被分在同一组"
                        f"（它们大小{'相同' if len(tree.contents[g[0]]) == len(tree.contents[other]) else '不同'}）")
            lost = sorted(set(first) - set(g))
            return (f"{tree.rel(g[0])} 和 {tree.rel(lost[0])} 的字节内容完全相同，应当在同一组，"
                    f"结果却把它们分到了不同的组")
    for g in got:
        if g != sorted(g):
            return (f"组内的路径要按字典序升序排列：{[tree.rel(p) for p in g]} 应当是 "
                    f"{[tree.rel(p) for p in sorted(g)]}")
    return "每一组都对，但组的顺序不对：各组要按自己（已排好序的）第一个路径的字典序升序排列"


def check(got, want: list, tree: Tree, call: str) -> None:
    norm, msg = normalize(got)
    if msg:
        raise Mismatch(msg, call=call, want=_shown(tree, want), got=got)
    msg = problem(norm, want, tree)
    if msg:
        outside = [p for g in norm for p in g if not (p == tree.base or p.startswith(tree.base + os.sep))]
        if outside:
            # The usual display shortens the real temp dir to "root", which would make a returned literal
            # "root/a.txt" look identical to the expected path; show the real paths instead.
            msg += f"（这次传入的 root_path 实际是 {tree.root!r}；下面的期望显示的是完整路径，没有缩写）"
            raise Mismatch(msg, call=call, want=want, got=norm)
        raise Mismatch(msg, call=call, want=_shown(tree, want), got=_shown(tree, norm))


# ---------------------------------------------------------------- watching file reads

class _Counted:
    """A file object that counts the bytes read through it."""
    __slots__ = ("_f", "_w", "_k")

    def __init__(self, f, w, k):
        self._f, self._w, self._k = f, w, k

    def _n(self, data):
        self._w._add(self._k, len(data))
        return data

    def read(self, *a):
        return self._n(self._f.read(*a))

    def read1(self, *a):
        return self._n(self._f.read1(*a))

    def readline(self, *a):
        return self._n(self._f.readline(*a))

    def readlines(self, *a):
        lines = self._f.readlines(*a)
        self._w._add(self._k, sum(map(len, lines)))
        return lines

    def readinto(self, b):
        n = self._f.readinto(b)
        self._w._add(self._k, n or 0)
        return n

    def readinto1(self, b):
        n = self._f.readinto1(b)
        self._w._add(self._k, n or 0)
        return n

    def readall(self):
        return self._n(self._f.readall())

    def __iter__(self):
        return self

    def __next__(self):
        return self._n(next(self._f))

    def close(self):
        self._w._forget(self._f)
        return self._f.close()

    def __enter__(self):
        self._f.__enter__()
        return self

    def __exit__(self, *exc):
        self._w._forget(self._f)
        return self._f.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._f, name)


class Watch:
    """While active (a with-block), every open of a file under root through open(), io.open() (so pathlib too) or
    os.open() first calls before(path), and every byte read from it (file methods, os.read, os.pread) is counted
    per path. mmap.mmap() of such a file adds the mapped length to `mapped` (what is then read from the map is not
    seen). Reads that bypass these (names imported before the block, io.FileIO, os.readv) are not seen."""

    def __init__(self, root: str, before=None):
        self.prefix = root.rstrip(os.sep) + os.sep
        self.before = before
        self.bytes = defaultdict(int)
        self.mapped = defaultdict(int)
        self.opens = defaultdict(int)
        self._fds = {}
        self._lock = threading.Lock()

    def _key(self, file):
        if isinstance(file, int):
            return self._fds.get(file)
        try:
            p = os.path.abspath(os.fsdecode(os.fspath(file)))
        except TypeError:
            return None
        return p if p.startswith(self.prefix) else None

    def _add(self, key, n):
        if n:
            with self._lock:
                self.bytes[key] += n

    def _opened(self, key):
        with self._lock:
            self.opens[key] += 1
        if self.before is not None:
            self.before(key)

    def _forget(self, f):
        try:
            self._fds.pop(f.fileno(), None)
        except (OSError, ValueError, AttributeError):
            pass

    def __enter__(self):
        w = self

        def w_open(file, *args, **kwargs):
            key = w._key(file)
            if key is not None:
                w._opened(key)
            f = _open(file, *args, **kwargs)
            if key is None:
                return f
            try:
                w._fds[f.fileno()] = key
            except (OSError, ValueError, AttributeError):
                pass
            return _Counted(f, w, key)

        def w_os_open(path, flags, *args, **kwargs):
            key = w._key(path)
            if key is not None:
                w._opened(key)
            fd = _os_open(path, flags, *args, **kwargs)
            if key is not None:
                w._fds[fd] = key
            return fd

        def w_os_read(fd, n):
            data = _os_read(fd, n)
            key = w._fds.get(fd)
            if key is not None:
                w._add(key, len(data))
            return data

        def w_os_pread(fd, n, offset):
            data = _os_pread(fd, n, offset)
            key = w._fds.get(fd)
            if key is not None:
                w._add(key, len(data))
            return data

        def w_os_close(fd):
            w._fds.pop(fd, None)
            return _os_close(fd)

        class w_mmap(_mmap):
            def __new__(cls, *args, **kwargs):
                m = _mmap.__new__(cls, *args, **kwargs)
                key = w._fds.get(args[0] if args else kwargs.get("fileno"))
                if key is not None:
                    with w._lock:
                        w.mapped[key] += len(m)
                return m
        w_mmap.__name__ = w_mmap.__qualname__ = "mmap"
        w_mmap.__module__ = "mmap"

        self._saved = [(builtins, "open", builtins.open), (io, "open", io.open), (os, "open", os.open),
                       (os, "read", os.read), (os, "pread", os.pread), (os, "close", os.close),
                       (mmap, "mmap", mmap.mmap)]
        builtins.open = io.open = w_open
        os.open, os.read, os.pread, os.close = w_os_open, w_os_read, w_os_pread, w_os_close
        mmap.mmap = w_mmap
        return self

    def __exit__(self, *exc):
        for mod, name, value in self._saved:
            setattr(mod, name, value)
        return False


class PairGate:
    """A before-open hook for Watch. An open of a tree file waits until another open arrives (at most `each` seconds
    per open and `budget` seconds in all; once the budget is spent opens no longer wait). Each such overlap goes into
    `pairs` as two (path, n) entries, n meaning the path's n-th open. hashed_together() asks for an overlap of two
    opens that may each be hashing a whole file: the only open of its file, or a later open of it (the first open of
    a file that is opened again may just read its prefix)."""

    def __init__(self, each: float = 1.0, budget: float = 4.0):
        self.cond = threading.Condition()
        self.waiter = None
        self.count = defaultdict(int)
        self.pairs = []
        self.each, self.budget = each, budget

    def __call__(self, key):
        with self.cond:
            self.count[key] += 1
            me = (key, self.count[key])
            if self.waiter is not None:
                self.pairs.append((self.waiter, me))
                self.waiter = None
                self.cond.notify_all()
                return
            if self.budget <= 0:
                return
            self.waiter = me
            seen = len(self.pairs)
            t0 = time.monotonic()
            self.cond.wait_for(lambda: len(self.pairs) > seen, timeout=min(self.each, self.budget))
            self.budget -= time.monotonic() - t0
            if self.waiter == me:
                self.waiter = None

    @property
    def paired(self) -> bool:
        return bool(self.pairs)

    def hashed_together(self) -> bool:
        def whole(entry):
            key, n = entry
            return n > 1 or self.count[key] == 1
        return any(whole(a) and whole(b) for a, b in self.pairs)
