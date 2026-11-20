"""Tests for file-deduplication. Every tree is built in its own tempfile.mkdtemp() and removed right after use."""
import os
import random
import threading
import time
from contextlib import contextmanager

from judgelib import Mismatch, calling, case, info, rng
import ref
from ref import PairGate, Tree, Watch

SLOW, FAST, CONC = "find_duplicate_files", "find_duplicate_files_fast", "find_duplicate_files_concurrent"


# ---------------------------------------------------------------- helpers

@contextmanager
def new_tree():
    t = Tree()
    try:
        yield t
    finally:
        t.cleanup()


def run(sol, fname: str, t: Tree, want=None, kwargs=None, watch=None, note: str = ""):
    """Calls the answer's fname on t.root (inside watch, if given) and checks the result against want, by default
    the brute force computed before the call. Returns the call text."""
    if want is None:
        want = ref.brute(t.root, t.vanished)
    kwargs = kwargs or {}
    extra = "".join(f", {k}={v!r}" for k, v in kwargs.items())
    call = calling(t.call(fname, extra) + (f"\n# {note}" if note else ""))
    fn = getattr(sol, fname)
    if watch is None:
        got = fn(t.root, **kwargs)
    else:
        with watch:
            got = fn(t.root, **kwargs)
    ref.check(got, want, t, call)
    return call


def flip(data: bytes, i: int) -> bytes:
    """data with byte i changed."""
    return data[:i] + bytes([data[i] ^ 0x5A]) + data[i + 1:]


def jitter(r: random.Random, most: float = 0.004):
    lock = threading.Lock()

    def before(_path):
        with lock:
            d = r.uniform(0, most)
        time.sleep(d)
    return before


def vanish_hook(t: Tree, victim: str, then=None):
    """A Watch hook that deletes victim just before its first open (then calls then, if given)."""
    done = []

    def before(path):
        if path == victim and not done:
            done.append(True)
            try:
                os.remove(victim)
            except FileNotFoundError:
                pass
        if then is not None:
            then(path)
    return before


# ---------------------------------------------------------------- scenarios
# Each fills a Tree; one that returns a path asks the test to delete that file just before it is first opened.

def sc_example(t):
    t.file("a.txt", "hello world")
    t.file("notes/b.txt", "hello world")
    t.file("notes/c.bin", bytes(range(1, 12)))          # 11 bytes, same length as "hello world"
    t.file("notes/log/d.txt", "goodbye")
    t.file("notes/log/e.txt", "goodbye")
    t.file("empty1.txt", "")
    t.file("notes/empty2.txt", "")


EXAMPLE_WANT = [["a.txt", "notes/b.txt"], ["empty1.txt", "notes/empty2.txt"], ["notes/log/d.txt", "notes/log/e.txt"]]


def nd_empty_dir(t):
    pass


def nd_lone_empty(t):
    t.file("alone.txt", "")


def nd_unique(t):
    for i in range(5):
        t.file(f"unique{i}.txt", f"content number {i}")


def nd_same_size(t):
    t.file("abc", "abc")
    t.file("x/abd", "abd")
    t.file("x/y/abC", "abC")
    t.file("x/y/z/cba", "cba")
    t.dir("x/empty_dir")


def sc_empty(t):
    t.file("e1", "")
    t.file("dir/e2.txt", "")
    t.file("dir/sub/e3", "")
    t.file("nul1", b"\x00")
    t.file("space", " ")
    t.file("newline", "\n")
    t.file("x/nul2", b"\x00")


def sc_bytes(t):
    t.file("lf_a.txt", b"line1\nline2\n")
    t.file("lf_b.txt", b"line1\nline2\n")
    t.file("crlf_a.txt", b"line1\r\nline2\r\n")
    t.file("crlf_b.txt", b"line1\r\nline2\r\n")
    t.file("cr.txt", b"line1\rline2\r")
    t.file("bom.txt", b"\xef\xbb\xbfhello")
    t.file("plain.txt", b"hello")
    t.file("trailing_newline.txt", b"hello\n")
    t.file("nfc.txt", "café")
    t.file("nfd.txt", "café")
    t.file("latin1.txt", b"caf\xe9")
    t.file("bin/x.dat", b"\xff\xfe\x00\x80bad")
    t.file("bin/y.dat", b"\xff\xfe\x00\x80bad")


def sc_file_links(t):
    t.file("data/x.txt", "same")
    t.file("data/y.txt", "same")
    t.file("solo.txt", "solo")
    t.file("../outside/o.txt", "solo")
    t.link("link_x", "data/x.txt")
    t.link("data/link_out", "../outside/o.txt")
    t.link("chain", "link_x")
    t.link("dangling", "missing.txt")
    t.link("loop", "loop")


def sc_dir_links(t):
    t.file("data/x.txt", "same")
    t.file("data/y.txt", "same")
    t.file("other.txt", "pair")
    t.file("../outside/dir/p.bin", "pair")
    t.file("../outside/dir/q.bin", "pair")
    t.link("alias", "data")
    t.link("sub/ext", "../outside/dir")


def sc_unreadable(t):
    t.file("a.txt", "dup")
    t.file("b.txt", "dup")
    t.file("secret.txt", "dup")
    t.file("locked/s1.bin", "other")
    t.file("other.txt", "other")
    t.file("locked/s2.bin", "twin")
    t.file("locked/s3.bin", "twin")
    t.file("z.txt", "after")
    t.file("zz/z.txt", "after")
    t.file("e1", "")
    t.file("zz/e2", "")
    t.file("locked/e3", "")              # an unreadable empty file is skipped like any other unreadable file
    for rel in ("secret.txt", "locked/s1.bin", "locked/s2.bin", "locked/s3.bin", "locked/e3"):
        t.deny(rel)


def sc_vanish(t):
    t.file("gone/v.txt", "vanishing")
    t.file("keep1.txt", "vanishing")
    t.file("keep2.txt", "vanishing")
    t.file("x1", "pair")
    t.file("x2", "pair")
    return t.abs("gone/v.txt")


def sc_order(t):
    t.file("z1", "zz")                    # listed first by os.walk, sorts last
    t.file("z2", "zz")
    t.file("a/x", "aa")
    t.file("a/y", "aa")
    t.file("m/b1", "11")                  # created in reverse order
    t.file("m/a1", "11")
    t.file("n/c2", "22")
    t.file("n/d2", "22")
    t.file("Zeta", "case")
    t.file("readme", "case")
    t.file("README", "case")
    t.file("f9", "num")
    t.file("f10", "num")
    t.file("é.txt", "uni")
    t.file("é.txt", "uni")
    t.file("文件.txt", "uni")


def sc_deep(t):
    deep = "/".join(f"level{i}" for i in range(40))
    t.file(f"{deep}/bottom.txt", "deep")
    t.file("top.txt", "deep")
    t.file("level0/level1/mid.txt", "mid")
    t.file(".git/objects/ab/cdef", "hidden")
    t.file("src/main.py", "hidden")
    t.file(".hidden_dup", "hidden")
    t.dir("empty/dirs/only")


EDGE_SCENARIOS = [nd_empty_dir, nd_lone_empty, nd_unique, nd_same_size, sc_empty, sc_bytes, sc_file_links,
                  sc_dir_links, sc_unreadable, sc_vanish, sc_order, sc_deep]


def random_tree(r: random.Random, t: Tree, n_files=(2, 22), extras: bool = True) -> None:
    """Small random tree: few distinct contents (so many real duplicates), contents that share a 4096-byte prefix,
    empty files, and with extras also symlinks, a symlinked directory and chmod-000 files."""
    dirs = [""]
    for i in range(r.randint(0, 5)):
        d = os.path.join(r.choice(dirs), r.choice(["d", "D", "dir_", "x"]) + str(i))
        t.dir(d)
        dirs.append(d)
    head = r.randbytes(4096)
    pool = [b"", b"a", b"A", r.randbytes(r.randint(1, 30)), r.randbytes(r.randint(1, 30)), r.randbytes(5000),
            head, head + b"x", head + b"y", head + b"xy", flip(head, 4095), head + r.randbytes(900)]
    alphabet = r.sample(pool, r.randint(2, 6))
    files = []
    for i in range(r.randint(*n_files)):
        rel = os.path.join(r.choice(dirs), f"f{i}" + r.choice(["", ".bin", ".TXT"]))
        t.file(rel, r.choice(alphabet))
        files.append(rel)
    if not extras:
        return
    for i in range(r.randint(0, 3)):
        t.link(os.path.join(r.choice(dirs), f"link{i}"), r.choice(files))
    if r.random() < 0.4:
        t.link(os.path.join(r.choice(dirs), "dangling"), "nowhere")
    if r.random() < 0.3 and len(dirs) > 1:
        t.link("dlink", r.choice(dirs[1:]))         # at the top, so it never points at its own ancestor
    if r.random() < 0.3:
        for rel in r.sample(files, min(len(files), r.randint(1, 2))):
            t.deny(rel)


def edge_runs(sol, fname: str, kwargs_list=({},), delays: bool = False) -> None:
    r = rng("delays")
    for build in EDGE_SCENARIOS:
        for kwargs in kwargs_list:
            with new_tree() as t:
                victim = build(t)
                hook, note = (jitter(r), "每次打开树里的文件之前随机等待 0–4 毫秒") if delays else (None, "")
                if victim:
                    t.vanished.add(victim)
                    hook = vanish_hook(t, victim, hook)
                    note = (f"调用过程中，{t.rel(victim)} 在第一次被打开之前被删除（模拟列出之后、读取之前被删掉的文件）"
                            + (f"；{note}" if note else ""))
                watch = Watch(t.root, hook) if hook else None
                run(sol, fname, t, kwargs=kwargs, watch=watch, note=note)


def example_case(sol, fname: str) -> None:
    with new_tree() as t:
        sc_example(t)
        want = [[os.path.join(t.root, p) for p in g] for g in EXAMPLE_WANT]
        assert ref.brute(t.root) == want            # the statement's answer
        run(sol, fname, t, want=want)


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子")
def p1_statement_example(sol):
    example_case(sol, SLOW)


@case("Part 1", title="没有重复时返回 []")
def p1_no_duplicates(sol):
    for build in (nd_empty_dir, nd_lone_empty, nd_unique, nd_same_size):
        with new_tree() as t:
            build(t)
            run(sol, SLOW, t, want=[])


@case("Part 1", title="空文件彼此相同")
def p1_empty_files(sol):
    with new_tree() as t:
        sc_empty(t)
        run(sol, SLOW, t)
    with new_tree() as t:
        t.file("only/one.txt", "")
        t.file("two.txt", "")
        run(sol, SLOW, t)


@case("Part 1", title="逐字节比较，不按文本")
def p1_bytes_not_text(sol):
    with new_tree() as t:
        sc_bytes(t)
        run(sol, SLOW, t)


@case("Part 1", title="符号链接不跟随、不出现")
def p1_symlinks(sol):
    for build in (sc_file_links, sc_dir_links):
        with new_tree() as t:
            build(t)
            run(sol, SLOW, t)


@case("Part 1", title="不可读或中途被删的文件跳过")
def p1_unreadable(sol):
    with new_tree() as t:
        sc_unreadable(t)
        run(sol, SLOW, t)
    with new_tree() as t:
        victim = sc_vanish(t)
        t.vanished.add(victim)
        run(sol, SLOW, t, watch=Watch(t.root, vanish_hook(t, victim)),
            note=f"调用过程中，{t.rel(victim)} 在第一次被打开之前被删除（模拟列出之后、读取之前被删掉的文件）")


@case("Part 1", title="组内与组间的顺序")
def p1_order(sol):
    with new_tree() as t:
        sc_order(t)
        run(sol, SLOW, t)


@case("Part 1", title="深层嵌套与隐藏文件")
def p1_deep_tree(sol):
    with new_tree() as t:
        sc_deep(t)
        run(sol, SLOW, t)


@case("Part 1", title="随机目录树，对照暴力解")
def p1_random(sol):
    r = rng()
    for _ in range(40):
        with new_tree() as t:
            random_tree(r, t)
            run(sol, SLOW, t)


# ---------------------------------------------------------------- Part 2

def read_report(t: Tree, watch: Watch, paths) -> str:
    rows = sorted(paths, key=lambda p: -watch.bytes.get(p, 0))[:5]
    return "；".join(f"{t.rel(p)} 读了 {watch.bytes.get(p, 0)} 字节（文件 {len(t.contents[p])} 字节）" for p in rows)


def map_report(t: Tree, watch: Watch, paths) -> str:
    rows = sorted(paths, key=lambda p: -watch.mapped.get(p, 0))[:5]
    return "；".join(f"{t.rel(p)} 被 mmap 映射了 {watch.mapped.get(p, 0)} 字节（文件 {len(t.contents[p])} 字节）"
                    for p in rows)


@case("Part 2", sample=True, title="题目例子")
def p2_statement_example(sol):
    example_case(sol, FAST)


@case("Part 2", title="边界情况同 Part 1")
def p2_edges(sol):
    edge_runs(sol, FAST)


@case("Part 2", title="前缀相同、后面不同")
def p2_prefix_collisions(sol):
    r = rng()
    head = r.randbytes(4096)
    tail = r.randbytes(6000)
    with new_tree() as t:
        t.file("s4095/a", head[:4095])
        t.file("s4095/b", head[:4095])
        t.file("s4095/c", flip(head[:4095], 4094))
        t.file("s4096/a", head)
        t.file("s4096/b", flip(head, 4095))
        t.file("s4096/c", head)
        t.file("s4097/a", head + b"1")
        t.file("s4097/b", head + b"2")
        t.file("s4097/c", head + b"1")
        t.file("s8192/a", head + tail[:4096])
        t.file("s8192/b", head + flip(tail[:4096], 2000))
        t.file("s10096/a", head + tail)
        t.file("s10096/b", head + flip(tail, 5999))
        t.file("s10096/c", head + flip(tail, 5999))
        t.file("s10096/d", head + flip(tail, 0))
        run(sol, FAST, t)


@case("Part 2", title="随机目录树，对照暴力解")
def p2_random(sol):
    r = rng()
    for _ in range(30):
        with new_tree() as t:
            random_tree(r, t)
            run(sol, FAST, t)


@case("Part 2", title="大小全不同：一个字节都不读")
def p2_distinct_sizes_no_read(sol):
    r = rng()
    body = r.randbytes(9000)
    sizes = [0, 1, 2, 3, 100, 4095, 4096, 4097, 5000, 5001, 8191, 8192, 9000]
    sizes += r.sample(sorted(set(range(10, 3000)) - set(sizes)), 17)
    r.shuffle(sizes)
    with new_tree() as t:
        for i, n in enumerate(sizes):
            t.file(os.path.join(["", "a", "a/b", "c"][i % 4], f"f{i}.bin"), body[:n])   # all share one prefix
        watch = Watch(t.root)
        call = run(sol, FAST, t, want=[], watch=watch)
        total = sum(watch.bytes.values())
        info(f"读取了 {total} 字节（{len(sizes)} 个文件，大小互不相同）")
        if total:
            raise Mismatch("树里每个文件的大小都互不相同，单凭大小就能断定没有重复，不应读取任何文件内容。"
                           + read_report(t, watch, watch.bytes), call=call, show=False)
        if watch.mapped:
            raise Mismatch("树里每个文件的大小都互不相同，单凭大小就能断定没有重复，不应读取任何文件内容，"
                           "却有文件被 mmap 映射进了内存（映射就是为了读它的内容）。"
                           + map_report(t, watch, watch.mapped), call=call, show=False)


@case("Part 2", title="样本不同的文件不能完整读取")
def p2_prefix_reads(sol):
    r = rng()
    size = 512 * 1024
    with new_tree() as t:
        distinct = [t.file(f"same_size/u{i:02d}.bin", bytes([i]) + r.randbytes(size - 1)) for i in range(12)]
        dup = bytes([200]) + r.randbytes(size - 1)
        t.file("same_size/dup_a.bin", dup)
        t.file("other/dup_b.bin", dup)
        near = bytes([201]) + r.randbytes(64 * 1024 - 1)
        t.file("same_size/near_a.bin", near + r.randbytes(size - len(near)))
        t.file("same_size/near_b.bin", near + r.randbytes(size - len(near)))
        unique = [t.file(f"big/w{j}.bin", r.randbytes(size - 1000 * j)) for j in range(1, 6)]
        watch = Watch(t.root)
        call = run(sol, FAST, t, watch=watch)
        info(f"读取了 {sum(watch.bytes.values())} 字节（树里文件共 {sum(map(len, t.contents.values()))} 字节）")
        for p in unique:
            if watch.bytes.get(p, 0) >= len(t.contents[p]):
                raise Mismatch(f"{t.rel(p)} 的大小在树里独一无二，不可能有重复，却被完整读取了。"
                               + read_report(t, watch, unique), call=call, show=False)
            if watch.mapped.get(p, 0) >= len(t.contents[p]):
                raise Mismatch(f"{t.rel(p)} 的大小在树里独一无二，不可能有重复，却被 mmap 整个映射进了内存（等于完整读取）。"
                               + map_report(t, watch, unique), call=call, show=False)
        for p in distinct:
            if watch.bytes.get(p, 0) >= size:
                raise Mismatch(f"{t.rel(p)} 被完整读取了：它和同样大小（{size} 字节）的其他文件开头第一个字节就不同，"
                               f"读开头 k 个字节的样本（k 是固定的小值，比如 4096）就足以排除它，不应完整读取。"
                               + read_report(t, watch, distinct), call=call, show=False)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子")
def p3_statement_example(sol):
    example_case(sol, CONC)


@case("Part 3", title="边界情况，worker 数各异")
def p3_edges(sol):
    edge_runs(sol, CONC, kwargs_list=({}, {"max_workers": 1}, {"max_workers": 3}), delays=True)


@case("Part 3", title="随机延迟打乱完成顺序")
def p3_random_delays(sol):
    r = rng()
    delay_rng = rng("delays")
    for i in range(24):
        with new_tree() as t:
            random_tree(r, t, n_files=(8, 24))
            mw = [None, 2, 3, 4, 8, 16][i % 6]
            run(sol, CONC, t, kwargs={} if mw is None else {"max_workers": mw},
                watch=Watch(t.root, jitter(delay_rng)), note="每次打开树里的文件之前随机等待 0–4 毫秒")


@case("Part 3", cpu=3, title="确实有多个文件同时在哈希")
def p3_hashes_overlap(sol):
    r = rng()
    a, b = r.randbytes(300_000), r.randbytes(320_000)     # far above any small prefix k: these must be hashed
    with new_tree() as t:
        t.file("p/one.bin", a)
        t.file("q/two.bin", a)
        t.file("r/three.bin", b)
        t.file("s/four.bin", b)
        t.file("t/solo.bin", r.randbytes(310_000))
        for kwargs in ({}, {"max_workers": 4}):
            gate = PairGate(each=0.5, budget=6.0)
            call = run(sol, CONC, t, kwargs=kwargs, watch=Watch(t.root, gate),
                       note="测试让每次打开文件最多等 0.5 秒，看这段时间里有没有另一个文件也被打开")
            if not gate.paired:
                raise Mismatch("没有任何两个文件是同时被打开来哈希的：每次打开文件时测试都等了 0.5 秒，"
                               "期间始终没有别的文件被打开，说明文件仍然是一个接一个地哈希。"
                               "Part 3 要求在工作池里并发地计算文件哈希（读文件也是哈希的一部分）。",
                               call=call, show=False)
            if not gate.hashed_together():
                raise Mismatch("只有读开头样本时有文件同时被打开；之后为了完整哈希再打开文件时，每次都等了 0.5 秒，"
                               "期间始终没有别的文件被打开，说明完整哈希仍然是一个接一个地做的。"
                               "Part 3 要求在工作池里并发地计算文件哈希，只把开头样本放进池里不算。",
                               call=call, show=False)


@case("Part 3", title="use_processes=True 用进程池")
def p3_processes(sol):
    r = rng()
    for mw in (2, 3):
        kwargs = {"max_workers": mw, "use_processes": True}
        with new_tree() as t:
            sc_example(t)
            run(sol, CONC, t, want=[[os.path.join(t.root, p) for p in g] for g in EXAMPLE_WANT], kwargs=kwargs)
        for build in (nd_unique, sc_unreadable, sc_file_links):
            with new_tree() as t:
                build(t)
                run(sol, CONC, t, kwargs=kwargs)
        with new_tree() as t:
            random_tree(r, t, n_files=(10, 20))
            run(sol, CONC, t, kwargs=kwargs)
