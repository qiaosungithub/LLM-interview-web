import time

from judgelib import Mismatch, calling, case, expect, expect_true, rng
import ref

NAMES = {1: "KVStore", 2: "ChunkedKVStore", 3: "LogKVStore"}
PROBE = "never-put"     # a key no test ever puts; get must return None for it
SHOW = 18               # a replay lists at most this many lines
SHOW_CHARS = 2400       # ... and at most about this many characters


# ---------------------------------------------------------------- how values are shown

def _plain(s):
    """repr(s), in double quotes whenever that needs no extra escaping."""
    r = repr(s)
    if r[0] == "'" and '"' not in s:      # then s holds no ' either, so nothing inside is escaped for the quotes
        return '"' + r[1:-1] + '"'
    return r


def _compressed(s):
    for n in range(1, 9):
        if len(s) >= 2 * n and len(s) % n == 0 and s == s[:n] * (len(s) // n):
            return f"{_plain(s[:n])} * {len(s) // n}"
    runs, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        runs.append((s[i], j - i))
        i = j
    parts, buf = [], ""
    for c, n in runs:
        if n >= 6:
            if buf:
                parts.append(_plain(buf))
                buf = ""
            parts.append(f"{_plain(c)} * {n}")
        else:
            buf += c * n
    if buf:
        parts.append(_plain(buf))
    return " + ".join(parts)


def lit(s):
    """A str as a short Python expression; repetitive strings become "ab" * 500 or "k" + "x" * 300."""
    if not isinstance(s, str):
        return repr(s)
    plain = _plain(s)
    if len(s) >= 6:
        short = _compressed(s)
        if len(short) < len(plain):
            return short
    return plain


class OneOf:
    """Shown as the expected value when several results are right."""

    def __init__(self, options):
        self.options = options

    def __repr__(self):
        return " 或 ".join(repr(o) for o in self.options)


def _same(got, want):
    try:
        return type(got) is type(want) and bool(got == want)
    except Exception:
        return False


def total_size(fs):
    return sum(len(fs.read(name) or b"") for name in fs.list())


# ---------------------------------------------------------------- one store on one file system

class Session:
    """Drives the answer's store class on one file system, checks every result against the statement, and keeps
    the replay shown when something fails. A new instance stands for a restarted process."""

    def __init__(self, sol, part, cap=None, why=""):
        name = NAMES[part]
        cls = getattr(sol, name, None)
        expect_true(isinstance(cls, type), f"answer.py 里没有 {name} 类（类名要和题目一致）")
        self.cls, self.name, self.part, self.cap, self.why = cls, name, part, cap, why
        self.fs = ref.FileSystem(cap) if part == 1 else ref.CrashFS(cap)
        self.items = []             # the replay: ("put", key code, value code) or ("code", text, note)
        self.uses_crash = self.uses_suppress = False
        self.s = None
        self.mem = {}               # what the live instance holds
        self.disk = {}              # what load() on a new instance must give
        self.cands = None           # Part 2, after an interrupted save: every state load() may give
        self.keys, self._seen = [], set()
        self.saved = False          # the last call was a save() that returned
        self.restarts = 0

    # ------------------------------------------------ replay

    def code(self, text, note=""):
        self.items.append(("code", text, note))

    def replay(self, last, mark="# <- 这一步"):
        lines, items, i = [], self.items, 0
        while i < len(items):
            if items[i][0] == "put":
                j = i
                while j < len(items) and items[j][0] == "put":
                    j += 1
                if j - i >= 3:
                    pairs = ", ".join(f"({k}, {v})" for _, k, v in items[i:j])
                    lines.append((f"for k, v in [{pairs}]: s.put(k, v)", ""))
                else:
                    lines += [(f"s.put({k}, {v})", "") for _, k, v in items[i:j]]
                i = j
            else:
                lines.append(items[i][1:])
                i += 1
        lines.append((last, mark))
        keep, size = [], 0
        for code, note in reversed(lines):
            size += len(code) + len(note) + 4
            if keep and (len(keep) >= SHOW or size > SHOW_CHARS):
                break
            keep.append((code, note))
        keep.reverse()
        args = "" if self.cap is None else f"max_file_size={self.cap}"
        head = ["from contextlib import suppress"] if self.uses_suppress else []
        if self.uses_crash:
            head += ["from ref import CrashFS, crash   # 题目给的 FileSystem，加上模拟崩溃的工具（评测用的就是它们）",
                     f"fs = CrashFS({args})"]
        elif self.part != 1:
            head += ["from ref import CrashFS   # 题目给的 FileSystem，只是 list() 先列出新建的文件（评测用的就是它）",
                     f"fs = CrashFS({args})"]
        else:
            head += ["from ref import FileSystem   # 题目给的 FileSystem（评测用的就是它）", f"fs = FileSystem({args})"]
        if len(lines) > len(keep):
            head.append(f"# …（省略了前面 {len(lines) - len(keep)} 行，所以不能直接粘贴重现）")
        width = max([len(c) for c, n in keep if n and len(c) <= 64] or [0]) + 3
        return "\n".join(head + [(c.ljust(width - 2) + "  " + n) if n else c for c, n in keep])

    def _msg(self, *parts):
        return "；".join(p for p in (self.why,) + parts if p)

    def _call(self, line, fn, *args):
        """One call of the answer; the replay is shown if it raises or runs out of time."""
        calling(self.replay(line))
        try:
            return fn(*args)
        except Exception:
            calling(self.replay(line, "# <- 这一步抛出异常"))
            raise

    def _key(self, k):
        if k not in self._seen:
            self._seen.add(k)
            self.keys.append(k)

    # ------------------------------------------------ operations

    def new(self):
        line = f"s = {self.name}(fs)"
        self.s = self._call(line, self.cls, self.fs)
        self.code(line, "# 新实例，相当于重启后的新进程" if self.restarts else "")
        self.restarts += 1
        self.mem, self.saved = {}, False

    def put(self, k, v):
        line = f"s.put({lit(k)}, {lit(v)})"
        self._call(line, self.s.put, k, v)
        self.items.append(("put", lit(k), lit(v)))
        self._key(k)
        self.mem[k] = v
        if self.part == 3:
            self.disk[k] = v
        self.saved = False

    def puts(self, pairs):
        for k, v in pairs.items():
            self.put(k, v)

    def bulk(self, code, pairs):
        """Many puts, shown as one line of code that makes the same calls."""
        calling(self.replay(code, "# <- 这一段"))
        try:
            for k, v in pairs:
                self.s.put(k, v)
        except Exception:
            calling(self.replay(code, "# <- 这一段抛出异常"))
            raise
        self.code(code)
        for k, v in pairs:
            self._key(k)
            self.mem[k] = v
            if self.part == 3:
                self.disk[k] = v
        self.saved = False

    def delete(self, k):
        line = f"s.delete({lit(k)})"
        self._call(line, self.s.delete, k)
        self.code(line)
        self._key(k)
        self.mem.pop(k, None)
        self.disk.pop(k, None)

    def compact(self):
        self._call("s.compact()", self.s.compact)
        self.code("s.compact()")

    def save(self, may_fail=False):
        """save(); with may_fail (a cap below 64) a ValueError counts as an interrupted save. Returns whether the
        save went through."""
        if not may_fail:
            self._call("s.save()", self.s.save)
            self.code("s.save()")
        else:
            self.uses_suppress = True
            line = "with suppress(ValueError): s.save()"
            calling(self.replay(line))
            try:
                self.s.save()
            except ValueError:
                self.code(line, "# 抛出了 ValueError：算作被打断的保存")
                self._interrupted()
                return False
            except Exception:
                calling(self.replay(line, "# <- 这一步抛出异常"))
                raise
            self.code(line, "# 正常返回了")
        self.disk, self.cands, self.saved = dict(self.mem), None, True
        return True

    def _interrupted(self):
        self.cands = (self.cands or [self.disk]) + [dict(self.mem)]
        self.saved, self.s = False, None

    def load(self, why=""):
        self._call("s.load()", self.s.load)
        self.code("s.load()")
        self.saved = False
        if self.cands:
            self._resolve(why)
        else:
            self.mem = dict(self.disk)
            self.check(why)

    def crash(self, method, *args, after, keep=None, zero=0, hole=None):
        """method(*args) on a file system whose process dies just before the call's fs mutation number `after`
        (see ref.crash). Returns True if the call finished first, so that nothing died."""
        self.uses_crash = True
        opts = (f", after={after}" + ("" if keep is None else f", keep={keep}") + (f", zero={zero}" if zero else "")
                + ("" if hole is None else f", hole=({hole[0]}, {hole[1]})"))
        line = f"crash(fs, s.{method}{''.join(', ' + lit(a) for a in args)}{opts})"
        calling(self.replay(line))
        if method in ("put", "delete"):
            self._key(args[0])          # checked later too: a torn record must not leave its value behind
        before = len(self.fs.calls)
        try:
            finished = ref.crash(self.fs, getattr(self.s, method), *args, after=after, keep=keep, zero=zero,
                                 hole=hole)
        except Exception:
            calling(self.replay(line, "# <- 这一步抛出异常"))
            raise
        if finished:
            made = len(self.fs.calls) - before
            self.code(line, f"# {method}() 一共只写入 / 删除了 {made} 次，没有走到崩溃点")
            if method == "save":
                self.disk, self.cands, self.saved = dict(self.mem), None, True
            elif method == "put":
                self.mem[args[0]] = self.disk[args[0]] = args[1]
            elif method == "delete":
                self.mem.pop(args[0], None)
                self.disk.pop(args[0], None)
            elif method == "load":
                if self.cands:
                    self._resolve("")
                else:
                    self.mem = dict(self.disk)
                    self.check()
            return True
        torn = self.fs.torn
        if torn is None:
            note = f"# 进程死在 {method}() 的第 {after + 1} 次写入 / 删除之前（那一次没有生效）"
        else:
            data, frag = torn
            if keep is None:
                part = f"全部 {len(data)} 个字节"
            elif keep < 0:
                part = f" {len(data)} 个字节里除了最后 {-keep} 个以外的部分"
            else:
                part = f" {len(data)} 个字节里的前 {min(keep, len(data))} 个"
            dmg = f"，其中最后 {min(zero, len(frag))} 个被清零" if zero and frag else ""
            if hole is not None and min(hole[1], len(frag)) > hole[0]:
                i, j = hole[0], min(hole[1], len(frag))
                dmg += f"，其中第 {i + 1} 个字节被清零" if j == i + 1 else f"，其中第 {i + 1} 到第 {j} 个字节被清零"
            note = f"# {method}() 的 fs.append 只写进了{part}{dmg}，进程就死了"
        self.code(line, note)
        if method == "save":
            self._interrupted()
        elif method in ("put", "delete") and torn is not None and torn[0] == torn[1]:
            if method == "put":         # the whole record landed undamaged: it counts
                self.disk[args[0]] = args[1]
            else:
                self.disk.pop(args[0], None)
        self.s, self.saved = None, False
        return False

    # ------------------------------------------------ checks

    def check(self, why=""):
        """get() of every key used so far, and of one never used, against what the store must hold."""
        calling(self.replay("for k in <用过的每个键>: print(s.get(k))"))
        for k in self.keys + [PROBE]:
            line = f"print(s.get({lit(k)}))"
            try:
                got = self.s.get(k)
            except Exception:
                calling(self.replay(line, "# <- 这一步抛出异常"))
                raise
            want = self.mem.get(k)
            if not _same(got, want):
                hint = "这个键应当没有值（get 返回 None）" if want is None else ""
                if want is not None and got is not None and not isinstance(got, str):
                    hint = f"get 应当返回 str，得到 {type(got).__name__}"
                raise Mismatch(self._msg(why, hint), self.replay(line), want, got)

    def _resolve(self, why):
        """After an interrupted save, load() must give one of the candidate states completely."""
        keys = self.keys + [PROBE]
        line = f"print([s.get(k) for k in [{', '.join(lit(k) for k in keys)}]])"
        calling(self.replay(line))
        got = []
        for k in keys:
            try:
                got.append(self.s.get(k))
            except Exception:
                calling(self.replay(f"print(s.get({lit(k)}))", "# <- 这一步抛出异常"))
                raise
        options = []
        for c in self.cands:
            o = [c.get(k) for k in keys]
            if _same(got, o):
                self.disk, self.mem, self.cands = dict(c), dict(c), None
                return
            if o not in options:
                options.append(o)
        raise Mismatch(self._msg(why, "被打断的 save() 之后，load() 必须完整地得到上一次保存的内容，或完整地得到被打断的那次"
                                      "保存的内容，不能是混合（期望值里用“或”分开的几种都算对）"),
                       self.replay(line), OneOf(options), got)

    def leftovers(self, why=""):
        """Right after save() returned: every file on fs must be one that load() needs."""
        if not self.saved or not self.disk:
            return
        names = sorted(self.fs.list())
        calling(self.replay("# 检查：逐个去掉 fs 里的一个文件，看新实例的 load() 还需不需要它"))
        keys = self.keys + [PROBE]
        want = [self.disk.get(k) for k in keys]
        for name in names:
            other = ref.copy_fs(self.fs, without=name)
            try:
                st = self.cls(other)
                st.load()
                got = [st.get(k) for k in keys]
            except Exception:
                continue                # load() failed without it: needed
            if _same(got, want):
                raise Mismatch(self._msg(why, f"save() 返回以后，fs 里不能留下 load() 用不到的文件：去掉文件 {name!r} 以后，"
                                              "新实例 load() 得到的内容完全一样"),
                               self.replay("print(sorted(fs.list()))"), None, names)


# ---------------------------------------------------------------- operation lists (random tests)

def play(t, op):
    kind = op[0]
    if kind == "new":
        t.new()
    elif kind == "put":
        t.put(op[1], op[2])
    elif kind == "delete":
        t.delete(op[1])
    elif kind == "save":
        t.save()
    elif kind == "load":
        t.load()
    elif kind == "compact":
        t.compact()
    elif kind == "check":
        t.check()
    elif kind == "leftovers":
        t.leftovers()
    elif kind == "crash":
        t.crash(op[1], *op[2], after=op[3], keep=op[4], zero=op[5])


def _valid(ops, part):
    """A crashed instance is never used again, and in Part 3 a new instance loads first."""
    if not ops or ops[0] != ("new",):
        return False
    for a, b in zip(ops, ops[1:]):
        if a[0] == "crash" and b != ("new",):
            return False
        if part == 3 and a == ("new",) and b != ("load",):
            return False
    return True


def _attempt(sol, part, cap, ops):
    """None if every result agrees with the statement, else the kind of failure."""
    t = Session(sol, part, cap)
    for op in ops:
        try:
            play(t, op)
        except Mismatch:
            return ("WA",)
        except Exception as e:      # noqa: BLE001 - the answer's own error
            return ("RE", type(e).__name__)
    return None


def _shrink(sol, part, cap, ops, bad, budget=0.6):
    """Drops operations that the failure does not need, so that the replay is short."""
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 1, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            if _valid(cand, part) and _attempt(sol, part, cap, cand) == bad:
                ops, progress = cand, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops


def random_check(sol, part, rounds, gen):
    Session(sol, part)          # the class exists
    r = rng()
    for _ in range(rounds):
        cap, ops = gen(r)
        bad = _attempt(sol, part, cap, ops)
        if bad is None:
            continue
        small = _shrink(sol, part, cap, ops, bad)
        for cand, why in ((small, "随机操作序列出错（已自动删去与出错无关的操作）"), (ops, "随机操作序列出错")):
            t = Session(sol, part, cap, why=why)
            for op in cand:
                play(t, op)
        raise Mismatch("随机操作序列出错，但同一序列重新运行时没有再出错（答案的行为不确定？）", None, None, None, show=False)


def _value(r, cap=None):
    if cap and r.random() < 0.15:
        return r.choice("vz中\U0001F600") * r.randint(cap // 2, cap * 3)
    return ref.string(r)


def gen1(r):
    pool = [ref.string(r, short=True) for _ in range(6)]
    ops = [("new",)]
    for _ in range(r.randint(5, 30)):
        x = r.random()
        if x < 0.55:
            ops.append(("put", r.choice(pool), ref.string(r)))
        elif x < 0.72:
            ops.append(("save",))
        elif x < 0.85:
            ops.append(("load",))
        else:
            ops += [("new",), ("load",)] if r.random() < 0.7 else [("new",)]
    return None, ops + [("new",), ("load",)]


def gen2(r):
    cap = r.choice([64, 64, 65, 80, 100, 128, 256, 1000, None])
    pool = [ref.string(r, short=True) for _ in range(5)] + ["k" * r.randint(60, 150)]
    ops = [("new",)]
    for _ in range(r.randint(5, 25)):
        x = r.random()
        if x < 0.45:
            ops.append(("put", r.choice(pool), _value(r, cap or 64)))
        elif x < 0.6:
            ops.append(("save",))
            if r.random() < 0.5:
                ops.append(("leftovers",))
        elif x < 0.77:
            ops += [("crash", "save", (), r.randint(0, 25), None, 0), ("new",)]
            if r.random() < 0.7:
                ops.append(("load",))
        elif x < 0.88:
            ops.append(("load",))
        else:
            ops += [("new",), ("load",)] if r.random() < 0.6 else [("new",)]
    return cap, ops + [("new",), ("load",)]


def gen3(r):
    pool = [ref.string(r, short=True) for _ in range(6)]
    ops = [("new",), ("load",)]
    for _ in range(r.randint(5, 30)):
        x = r.random()
        if x < 0.4:
            ops.append(("put", r.choice(pool), ref.string(r)))
        elif x < 0.55:
            ops.append(("delete", r.choice(pool)))
        elif x < 0.62:
            ops.append(("compact",))
        elif x < 0.74:
            method = r.choice(["put", "put", "delete"])
            args = (r.choice(pool), ref.string(r)) if method == "put" else (r.choice(pool),)
            keep, zero = r.choice([None, 0, 1, 3, 8, 13, -1, -2, -5]), r.choice([0, 0, 0, 1, 2, 5, 100])
            ops += [("crash", method, args, 0, keep, zero), ("new",), ("load",)]
        elif x < 0.78:
            ops += [("crash", "compact", (), r.randint(0, 2), r.choice([None, None, 3, -1]), r.choice([0, 4])),
                    ("new",), ("load",)]
        elif x < 0.81:
            ops += [("crash", "load", (), 0, None, 0), ("new",), ("load",)]
        elif x < 0.9:
            ops += [("new",), ("load",)]
        else:
            ops.append(("check",))
    return None, ops + [("new",), ("load",)]


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    t = Session(sol, 1)
    t.new()
    t.put("time:now", "12:00")
    t.put("", "empty key")
    t.put("emoji", "\U0001F642\n")
    t.save()
    t.new()
    t.load()


DELIMS = [":", ",", "=", "\n", "a:b", "a,b", "a=b", "k=v,k2=v2", "line1\nline2", "\x00", "a\x00b", "\\", "\\:", "a\\",
          "\\n", "\\\\", "\\x00", ":\\", "\"'", "\r\n", "\t", " ", "  x  "]


@case("Part 1", title="分隔符和反斜杠")
def p1_delimiters(sol):
    t = Session(sol, 1, why="键和值里的分隔符、换行、反斜杠都要原样往返")
    t.new()
    for i, k in enumerate(DELIMS):
        t.put(k, DELIMS[-1 - i])
    t.put("value:with:colons", "x:y\n:z\\")
    t.save()
    t.new()
    t.load()


@case("Part 1", title="emoji、中文、NUL 等字符")
def p1_unicode(sol):
    t = Session(sol, 1, why="一个字符可能占好几个字节；所有字符都要原样往返")
    t.new()
    t.puts({"\U0001F600": "emoji \U0001F600 \U0001F4A9", "中文 and English": "\U0010FFFF", "é": "\x7f\x80\xff",
            "\x00" * 12: "\x00", "\U0001F600" * 3 + "a": "ä" * 7, "a": "\U0001F600" * 10})
    t.save()
    t.new()
    t.load()


@case("Part 1", title="孤立的代理字符 \\ud800")
def p1_lone_surrogate(sol):
    t = Session(sol, 1, why='键和值是任意 str，包括 "\\ud800" 这样单独出现的代理字符（str.encode 默认会拒绝它）')
    t.new()
    t.puts({"\ud800": "a\udfffb", "x": "\udc80" * 3, "\U0001F600": "\ud83d"})
    t.save()
    t.new()
    t.load()


@case("Part 1", title="空字符串；覆盖旧值")
def p1_empty_and_overwrite(sol):
    t = Session(sol, 1)
    t.new()
    for k, v in [("", ""), ("k", ""), ("k", "v1"), ("k", "v2"), ("", "x"), ("", ""), ("z", "")]:
        t.put(k, v)
    t.save()
    t.new()
    t.load(why='值是空字符串时 get 返回 ""，不是 None')


LONG_CODE = 'for i in range(2000): s.put(f"key{i}", "x" * (i % 300))'
LONG_PAIRS = [(f"key{i}", "x" * (i % 300)) for i in range(2000)]


@case("Part 1", title="很长的键和值；很多键")
def p1_long(sol):
    t = Session(sol, 1, why="键和值可以任意长")
    t.new()
    t.put("k" * 70000, "v")
    t.put("big", "\U0001F600" * 50000)
    t.put("", "long" * 5000)
    t.bulk(LONG_CODE, LONG_PAIRS)
    t.save()
    t.new()
    t.load()


@case("Part 1", title="什么都没保存时 load() 得到空存储")
def p1_nothing_saved(sol):
    t = Session(sol, 1)
    t.new()
    t.check()
    t.put("a", "1")
    t.load(why="fs 上什么都没保存过，load() 之后存储应当是空的")
    t.new()
    t.load()


@case("Part 1", title="后一次 save() 完全取代前一次")
def p1_later_save_replaces(sol):
    t = Session(sol, 1, why="后一次 save() 完全取代前一次：load() 只得到后一次保存的内容")
    t.new()
    t.puts({"a": "1", "b": "2", "c": "3"})
    t.save()
    t.new()
    t.put("a", "3")
    t.save()
    t.new()
    t.load()
    t.new()
    t.save()
    t.new()
    t.load()


@case("Part 1", title="load() 丢掉没保存的修改")
def p1_load_replaces_memory(sol):
    t = Session(sol, 1)
    t.new()
    t.put("a", "1")
    t.save()
    t.put("a", "2")
    t.put("b", "3")
    t.load(why="load() 用 fs 上的内容替换内存里的内容，没保存的修改要丢掉")
    t.put("c", "4")
    t.save()
    t.new()
    t.load()


@case("Part 1", cpu=3, title="随机内容，对照字典")
def p1_random(sol):
    random_check(sol, 1, 150, gen1)


# ---------------------------------------------------------------- Part 2
EXAMPLE_CODE = 'for i in range(16): s.put(f"key{i}", f"value {i}:" + "\U0001F642" * (i % 8))'
EXAMPLE_PAIRS = [(f"key{i}", f"value {i}:" + "\U0001F642" * (i % 8)) for i in range(16)]


@case("Part 2", sample=True, title="题目例子：上限 64 字节")
def p2_example(sol):
    t = Session(sol, 2, cap=64)
    t.new()
    t.bulk(EXAMPLE_CODE, EXAMPLE_PAIRS)
    t.save()
    names = sorted(t.fs.list())
    expect_true(len(names) > 1, "max_file_size=64、存储有几百字节时，save() 必须产生不止一个文件",
                call=t.replay("print(fs.list())"), got=names)
    t.new()
    t.load()


@case("Part 2", title="和 KVStore 一样的保证")
def p2_basics(sol):
    t = Session(sol, 2, cap=1000)
    t.new()
    t.check()
    t.put("x", "1")
    t.load(why="fs 上什么都没保存过，load() 之后存储应当是空的")
    for i, k in enumerate(ref.WEIRD):
        t.put(k, ref.WEIRD[-1 - i])
    t.save()
    t.new()
    t.load()
    t.new()
    t.put("only", "this")
    t.save()
    t.new()
    t.load(why="后一次 save() 完全取代前一次")
    t.put("unsaved", "u")
    t.load(why="load() 用 fs 上的内容替换内存里的内容")


@case("Part 2", title="比上限还长的键和值")
def p2_long_items(sol):
    t = Session(sol, 2, cap=64, why="单个键或值比 max_file_size 还长时也要能往返")
    t.new()
    t.puts({"k" * 500: "v", "big": "中" * 1000, "\U0001F600" * 40: "\x00" * 200, "small": "s"})
    t.save()
    t.leftovers()
    t.new()
    t.load()
    for n in range(30, 61):         # file boundaries inside a length header or inside a 4-byte character
        t = Session(sol, 2, cap=64, why="单个键或值比 max_file_size 还长时也要能往返")
        t.new()
        t.put("k" * n, "\U0001F600" * 3)
        t.save()
        t.new()
        t.load()


MANY_CODE = 'for i in range(30): s.put(f"key_{i}_" + "x" * (i * 7 % 200), "y" * (i * 13 % 300))'
MANY_PAIRS = [(f"key_{i}_" + "x" * (i * 7 % 200), "y" * (i * 13 % 300)) for i in range(30)]
FEW_CODE = 'for i in range(5): s.put(f"few{i}", "z" * (40 * i))'
FEW_PAIRS = [(f"few{i}", "z" * (40 * i)) for i in range(5)]


@case("Part 2", title="各种上限：64、65、100、10⁶…")
def p2_many_caps(sol):
    for cap in [64, 65, 77, 100, 128, 1000, 10 ** 6]:
        t = Session(sol, 2, cap=cap, why=f"max_file_size={cap}")
        t.new()
        t.bulk(MANY_CODE, MANY_PAIRS)
        t.save()
        t.leftovers()
        t.new()
        t.load()
        t.new()
        t.bulk(FEW_CODE, FEW_PAIRS)
        t.save()
        t.leftovers()
        t.new()
        t.load()


def _crash_grid(sol, cap, s0, s1, s2, why):
    """save s0; then s1, killed before each of its fs calls in turn; then a clean save of s2."""
    after = 0
    while True:
        t = Session(sol, 2, cap=cap, why=why)
        t.new()
        t.puts(s0)
        t.save()
        t.new()
        t.puts(s1)
        if t.crash("save", after=after):
            t.leftovers()
            t.new()
            t.load()
            return
        t.new()
        t.load()
        t.new()
        t.puts(s2)
        t.save()
        t.leftovers()
        t.new()
        t.load()
        after += 1
        expect_true(after < 1000, "一次 save() 调用了上千次 write / delete", call=t.replay("s.save()"))


S0 = {f"k{i}": "v" * 30 for i in range(5)}
S1 = {"big": "z" * 300, "\U0001F600": "", "k1": "changed"}
S2 = {"tiny": "t"}


@case("Part 2", title="没有上限（max_file_size=None）")
def p2_no_cap(sol):
    t = Session(sol, 2, cap=None, why="fs.max_file_size 可以是 None（没有上限）")
    t.new()
    t.save()
    t.new()
    t.load()
    t.puts({"a": "1", "long": "L" * 5000, "": "\n"})
    t.save()
    t.leftovers()
    t.new()
    t.load()
    _crash_grid(sol, None, S0, S1, S2, "fs.max_file_size 是 None 时 save() 也可能被打断")


@case("Part 2", title="上限小于 64：可以抛 ValueError")
def p2_small_caps(sol):
    for cap in [0, 1, 5, 16, 31, 32, 40, 63]:
        t = Session(sol, 2, cap=cap, why=f"max_file_size={cap} 小于 64：save() 可以抛 ValueError，算作被打断的保存")
        t.new()
        t.put("a", "b")
        if t.save(may_fail=True):
            t.leftovers()
        t.new()
        t.load()
        t.new()
        t.puts({"key": "value" * 10, "a": "c"})
        if t.save(may_fail=True):
            t.leftovers()
        t.new()
        t.load()


@case("Part 2", title="save() 中途崩溃：只能是旧的或新的")
def p2_interrupted_save(sol):
    _crash_grid(sol, 64, S0, S1, S2, "")


@case("Part 2", title="连续两次被打断，再正常保存")
def p2_two_crashes(sol):
    first, done1 = 0, False
    while not done1:
        second, done2 = 0, False
        while not done2:
            t = Session(sol, 2, cap=64)
            t.new()
            t.puts(S0)
            t.save()
            t.new()
            t.puts(S1)
            done1 = t.crash("save", after=first)
            if done1:
                t.leftovers()
            t.new()
            t.load()
            t.puts(S2)                          # the instance that recovered goes on and saves again
            done2 = t.crash("save", after=second)
            if done2:
                t.leftovers()
            t.new()
            t.load()
            t.new()
            t.puts(S0)
            t.save()
            t.leftovers(why="前面被打断的保存留下的文件，下一次成功的 save() 返回前都要清理掉")
            t.new()
            t.load()
            second += 1
            expect_true(second < 1000, "一次 save() 调用了上千次 write / delete", call=t.replay("s.save()"))
        first += 1


@case("Part 2", title="被打断的保存留下的文件要清理")
def p2_leftovers(sol):
    why = "被打断的 save() 留下的文件，下一次成功的 save() 返回前必须清理掉"
    for after in [4, 9, 14, 17, 18, 19, 20, 22]:
        t = Session(sol, 2, cap=64, why=why)
        t.new()
        t.puts({"a": "1"})
        t.save()
        t.new()
        t.put("big", "B" * 1000)
        t.crash("save", after=after)
        t.new()
        t.load()
        t.new()
        t.put("s", "x")
        t.save()
        t.leftovers()
        t.new()
        t.load()


@case("Part 2", cpu=4, title="随机保存与崩溃，对照模型")
def p2_random(sol):
    random_check(sol, 2, 150, gen2)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="put / delete 立即持久")
def p3_basic(sol):
    t = Session(sol, 3, why="put 和 delete 返回时就已经持久化，新实例 load() 之后要能看到")
    t.new()
    t.load()
    t.put("a", "1")
    t.put("b", "2")
    t.put("a", "3")
    t.delete("b")
    t.check()
    t.new()
    t.load()
    t.put("b", "again")
    t.new()
    t.load()


@case("Part 3", title='put(k, "") 不是删除')
def p3_empty_value(sol):
    t = Session(sol, 3, why='put(k, "") 存的是空字符串，不是删除：重启后 get 应返回 ""')
    t.new()
    t.load()
    t.puts({"k": "", "": "", "x": "1"})
    t.delete("x")
    t.put("x", "")
    t.check()
    t.new()
    t.load()
    t.delete("")
    t.new()
    t.load()


@case("Part 3", title="任意字符的键和值")
def p3_weird_strings(sol):
    t = Session(sol, 3)
    t.new()
    t.load()
    for i, k in enumerate(ref.WEIRD):
        t.put(k, ref.WEIRD[-1 - i])
    for k in ref.WEIRD[::3]:
        t.delete(k)
    t.put("long", "L" * 3000)
    t.new()
    t.load()


@case("Part 3", title="每次 put / delete 只追加一次")
def p3_one_append(sol):
    t = Session(sol, 3)
    t.new()
    t.load()
    sizes = {}

    def step(method, *args):
        # deleting a missing key, or putting the value a key already has, changes nothing: writing nothing is fine
        same = args[0] not in t.mem if method == "delete" else _same(t.mem.get(args[0]), args[1])
        before = len(t.fs.calls)
        getattr(t, method)(*args)
        made = t.fs.calls[before:]
        if same and not made:
            return None
        if len(made) != 1 or made[0][0] != "append":
            t.items.pop()       # the replay below marks this call itself
            got = [f"{m}({n!r}, {len(d)} 字节)" if m != "delete" else f"delete({n!r})" for m, n, d in made]
            raise Mismatch(f"{method}() 必须恰好调用一次 fs.append，且不能调用 fs.write / fs.delete（不能改写已经存好的内容）；"
                           f"这次的调用是：{', '.join(got) or '没有调用'}", t.replay(f"s.{method}({', '.join(map(lit, args))})"),
                           None, None, show=False)
        return len(made[0][2])

    step("put", "first", "record")
    for method, args in [("put", ("x", "y")), ("delete", ("x",))]:
        sizes[method] = step(method, *args)
    r = rng("one-append")
    for i in range(60):
        if r.random() < 0.7:
            step("put", f"key{i % 9}", "v" * r.randint(0, 40))
        else:
            step("delete", f"key{r.randint(0, 9)}")
    for method, args in [("put", ("x", "y")), ("delete", ("x",))]:
        size = step(method, *args)
        if size != sizes[method]:
            t.items.pop()
            raise Mismatch(f"同样的 {method}({', '.join(map(lit, args))}) 在日志开头附近追加了 {sizes[method]} 字节，"
                           f"在已有 60 多条记录之后追加了 {size} 字节：每次追加的大小只能取决于这次的键和值",
                           t.replay(f"s.{method}({', '.join(map(lit, args))})"), None, None, show=False)


def _torn(sol, history, op, why):
    """Every way the append of op can be cut short or damaged (a prefix, zeroed or not; or all of it with one byte
    zeroed), each followed by recovery on a new instance and one more put that must survive the next load()."""
    method, args = op[0], op[1:]

    def setup():
        t = Session(sol, 3, why=why)
        t.new()
        t.load()
        for h in history:
            getattr(t, h[0])(*h[1:])
        return t

    t = setup()
    before = len(t.fs.calls)
    getattr(t, method)(*args)
    made = t.fs.calls[before:]
    variants = [(None, 0, None)]
    if len(made) == 1 and made[0][0] == "append":
        data = made[0][2]
        n = len(data)
        variants += ([(k, 0, None) for k in range(n)] + [(k, k, None) for k in range(1, n + 1)]
                     + [(None, z, None) for z in range(1, n)]
                     + [(None, 0, (i, i + 1)) for i in range(n)])     # damage is not always at the end
    seen = set()
    for keep, zero, hole in variants:
        t = setup()
        t.crash(method, *args, after=0, keep=keep, zero=zero, hole=hole)
        frag = t.fs.torn[1] if t.fs.torn else None
        if frag in seen:
            continue
        seen.add(frag)
        t.new()
        t.load()
        t.put("after", "recovery")
        t.new()
        t.load(why="恢复之后的操作，下一次 load() 也要能读到")


TORN_WHY = "追加写到一半时进程死了：load() 不能抛异常，要恢复之前完整写入的所有操作，并且不能从残缺的尾巴里取任何东西"


@case("Part 3", title="最后一次追加写到一半")
def p3_torn_tail(sol):
    history = [("put", "a", "1"), ("put", "", ""), ("put", "nul", "\x00" * 12), ("put", "b", "\U0001F600" * 3),
               ("delete", "a"), ("put", "c", "x:y\n")]
    _torn(sol, history, ("put", "d", "value d"), TORN_WHY)
    _torn(sol, [], ("put", "first", "only"), TORN_WHY)
    _torn(sol, [("put", "z", "1")], ("put", "z", "\x00" * 20), TORN_WHY)


@case("Part 3", title="写坏的是 delete 记录")
def p3_torn_delete(sol):
    _torn(sol, [("put", "a", "1"), ("put", "b", "2")], ("delete", "a"), TORN_WHY)


@case("Part 3", title="多次崩溃与恢复")
def p3_repeated_crashes(sol):
    t = Session(sol, 3, why=TORN_WHY)
    t.new()
    t.load()
    t.puts({"a": "1", "b": "2"})
    t.crash("put", "c", "3", after=0, keep=5)
    t.new()
    t.crash("load", after=0)
    t.new()
    t.load()
    t.put("d", "4")
    t.delete("a")
    t.crash("delete", "b", after=0, keep=-1)
    t.new()
    t.load()
    t.compact()
    t.crash("put", "e", "5", after=0, zero=3)
    t.new()
    t.load()
    t.put("f", "6")
    t.crash("put", "g", "7", after=0, keep=2, zero=2)
    t.new()
    t.load()
    t.put("h", "8")
    t.new()
    t.load()


COMPACT_CODE = 'for i in range(50): s.put("x", str(i))'
COMPACT_PAIRS = [("x", str(i)) for i in range(50)]


@case("Part 3", title="compact() 后内容不变、大小只看当前内容")
def p3_compact(sol):
    a = Session(sol, 3, why="compact() 之后存储的内容不能变")
    a.new()
    a.load()
    a.bulk(COMPACT_CODE, COMPACT_PAIRS)
    a.put("y", "keep")
    a.delete("y")
    a.put("", "")
    a.put("z", "\U0001F600" * 5)
    a.delete("z")
    a.put("w", "w")
    a.compact()
    a.check()
    b = Session(sol, 3)
    b.new()
    b.load()
    b.puts({"w": "w", "x": "49", "": ""})
    b.compact()
    size_a, size_b = total_size(a.fs), total_size(b.fs)
    expect(size_a, size_b, call=a.replay("print(sum(len(fs.read(n)) for n in fs.list()))"),
           msg=f"compact() 之后 fs 上文件的总大小只能取决于当前的键和值：同样只有 x、w 和空字符串键这三个键值对的全新存储，"
               f"compact() 之后是 {size_b} 字节")
    a.put("same", "instance")
    a.new()
    a.load(why="compact() 之后，同一个实例上的 put 也要持久")
    a.put("after", "compact")
    a.delete("x")
    a.new()
    a.load(why="compact() 之后的 put / delete 也要持久")
    a.compact()
    a.new()
    a.load()
    c = Session(sol, 3)
    c.new()
    c.load()
    c.compact()
    c.new()
    c.load()


@case("Part 3", title="compact() 中途崩溃不丢数据")
def p3_interrupted_compact(sol):
    history = [("bulk", COMPACT_CODE, COMPACT_PAIRS), ("put", "y", "keep"), ("delete", "y"), ("put", "", ""),
               ("put", "w", "\U0001F600")]

    def setup():
        t = Session(sol, 3, why="被打断的 compact() 不能丢任何数据")
        t.new()
        t.load()
        for h in history:
            getattr(t, h[0])(*h[1:])
        return t

    t = setup()
    before = len(t.fs.calls)
    t.compact()
    made = t.fs.calls[before:]
    for after, (method, _, data) in enumerate(made):
        variants = [(None, 0)] + ([(len(data) // 2, 0), (None, 3)] if method == "append" and data else [])
        for keep, zero in variants:
            t = setup()
            t.crash("compact", after=after, keep=keep, zero=zero)
            t.new()
            t.load()
            t.put("after", "1")
            t.compact()
            t.new()
            t.load()


@case("Part 3", cpu=4, title="随机操作与崩溃，对照字典")
def p3_random(sol):
    random_check(sol, 3, 150, gen3)


# ---------------------------------------------------------------- Part 3, a big log built at import time
BIG_N = 40_000
BIG_CODE = (f"for i in range({BIG_N}):\n"
            "    if i % 7 == 6:\n"
            "        s.delete(f\"k{i * 3 % 5000}\")\n"
            "    else:\n"
            "        s.put(f\"k{i % 5000}\", f\"{i}:\" + \"v\" * (400 + i % 50))")
BIG_OPS, BIG_STATE = [], {}
for _i in range(BIG_N):
    if _i % 7 == 6:
        BIG_OPS.append((False, f"k{_i * 3 % 5000}", None))
        BIG_STATE.pop(BIG_OPS[-1][1], None)
    else:
        BIG_OPS.append((True, f"k{_i % 5000}", f"{_i}:" + "v" * (400 + _i % 50)))
        BIG_STATE[BIG_OPS[-1][1]] = BIG_OPS[-1][2]
BIG_KEYS = [f"k{i}" for i in range(0, 5000, 3)] + [PROBE]


@case("Part 3", cpu=3, title="4·10⁴ 条记录：load() 要 O(N)")
def p3_big_log(sol):
    cls = getattr(sol, "LogKVStore", None)
    expect_true(isinstance(cls, type), "answer.py 里没有 LogKVStore 类（类名要和题目一致）")
    head = ("from ref import FastFS   # 和题目的 FileSystem 行为相同，只是 append 不复制整个文件，建大日志更快\n"
            "fs = FastFS()\ns = LogKVStore(fs)\ns.load()\n" + BIG_CODE)
    calling(head + "   # <- 这一段")
    fs = ref.FastFS()
    s = cls(fs)
    s.load()
    for is_put, k, v in BIG_OPS:
        if is_put:
            s.put(k, v)
        else:
            s.delete(k)
    head += "\nfs = fs.plain()          # 同样的文件，放到题目的 FileSystem 上\ns = LogKVStore(fs)\ns.load()"
    calling(head + "   # <- 这一步：load() 要在 O(N) 时间里重放 N 条记录")
    s = cls(fs.plain())
    s.load()
    calling(head + "\nfor k in <其中 1668 个键>: print(s.get(k))   # <- 这一段")
    for k in BIG_KEYS:
        got = s.get(k)
        want = BIG_STATE.get(k)
        if not _same(got, want):
            raise Mismatch("重放大日志之后内容不对" + ("；这个键应当没有值（get 返回 None）" if want is None else ""),
                           head + f"\nprint(s.get({lit(k)}))   # <- 这一步", want, got)
