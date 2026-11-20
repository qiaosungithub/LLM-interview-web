import ast
import time

from judgelib import Mismatch, calling, case, expect, expect_true, rng
import ref

SHOW = 60           # a replay lists at most this many calls (every sequence these tests make fits)


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str):
        if len(x) >= 8 and len(set(x)) == 1 and x[0].isalnum() and x[0].isascii():
            return f'"{x[0]}" * {len(x)}'
        if '"' not in x and "\\" not in x and x.isprintable():
            return f'"{x}"'
        return repr(x)
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_lit(v) for v in x) + "]"
    return repr(x)


def _call(method, args, kwargs):
    parts = [_lit(a) for a in args] + [f"{k}={_lit(v)}" for k, v in kwargs.items()]
    return f"fs.{method}({', '.join(parts)})"


def _replay(done, wants, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step."""
    lines = [_call(*op) for op in done]
    lines[-1] = f"print({lines[-1]})"
    notes = [f"# -> {_lit(w)}" for w in wants] + [mark]
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = ["fs = FileSystem()"] + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(head + [s.ljust(width) + n for s, n in zip(lines, notes)])


def _method(obj, name):
    fn = getattr(obj, name, None)
    expect_true(callable(fn), f"FileSystem 没有 {name} 方法（方法名要和题目一致）")
    return fn


def _same(got, want):
    """got equals want; True/False only match a real bool, and a tuple is accepted where a list is wanted."""
    try:
        if isinstance(want, list) and isinstance(got, (list, tuple)):
            got = list(got)
        if isinstance(want, bool) or isinstance(got, bool):
            return isinstance(got, bool) and isinstance(want, bool) and got == want
        return bool(got == want)
    except Exception:
        return False


def _differs(got, want, call, msg):
    """Fails the case for a return value that _same rejected."""
    if not msg and type(got) is type(want) and not isinstance(want, (str, list)):
        msg = "返回值不对"
    expect(got, want, call=call, msg=msg)       # raises for every difference that == sees
    raise Mismatch(f"类型不对：期望 {type(want).__name__}，得到 {type(got).__name__}", call, want, got)


# ---------------------------------------------------------------- hand-written sequences

def _value(node):
    """A literal such as "a", ["b"], None or "x" * 6 (the scripts below are the only input)."""
    return eval(compile(ast.Expression(node), "<script>", "eval"), {"__builtins__": {}})


def _parse(script):
    """Lines like  write("/a", "x" * 6, owner="u") -> True   # why  become (method, args, kwargs, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        call, arrow, rest = line.strip().partition("->")
        assert arrow, f"missing '->' in {line!r}"
        expr, _, why = rest.partition("#")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(_value(a) for a in node.args)
        kwargs = {k.arg: _value(k.value) for k in node.keywords}
        steps.append((node.func.id, args, kwargs, _value(ast.parse(expr.strip(), mode="eval").body), why.strip()))
    return steps


def run(sol, *scripts):
    """Runs the calls on a fresh FileSystem, checking each return value as soon as it comes back."""
    fs = sol.FileSystem()
    done, wants = [], []
    for method, args, kwargs, want, why in [s for script in scripts for s in _parse(script)]:
        done.append((method, args, kwargs))
        fn = _method(fs, method)
        calling(_replay(done, wants))       # shown if the answer raises or hangs in this call
        try:
            got = fn(*args, **kwargs)
        except Exception:
            calling(_replay(done, wants, mark="# <- 这一步抛出异常"))
            raise
        if not _same(got, want):
            _differs(got, want, _replay(done, wants), why)
        wants.append(want)


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, ops):
    """Replays ops on a fresh answer and a fresh ref.FileSystem; None if every return value agrees. A
    shortened sequence can reach a move the statement leaves open (see ref.ambiguous); it then counts as
    passing, so the shrinker never shows such a step."""
    fs, model, wants = sol.FileSystem(), ref.FileSystem(), []
    for k, (method, args, kwargs) in enumerate(ops):
        if model.ambiguous(method, args):
            return None
        want = getattr(model, method)(*args, **kwargs)
        fn = _method(fs, method)
        try:
            got = fn(*args, **kwargs)
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        wants.append(want)
    return None


def _shrink(sol, ops, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def random_check(sol, level, sequences, length):
    r = rng()
    for _ in range(sequences):
        ops = ref.random_ops(r, level, r.randint(*length))
        bad = _outcome(sol, ops)
        if bad is None:
            continue
        ops, bad = _shrink(sol, ops, bad)
        k = bad["k"]
        if bad["exc"] is not None:
            calling(_replay(ops[: k + 1], bad["wants"], mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        _differs(bad["got"], bad["want"], _replay(ops[: k + 1], bad["wants"]),
                 f"随机操作序列的第 {k + 1} 步返回值不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- the statement's examples

EXAMPLE_L1 = """
    mkdir("/reports/2025")                    -> True      # 创建 /reports 和 /reports/2025
    mkdir("/reports/2025")                    -> True      # 已经是目录：无操作
    write("/reports/2025/q1.txt", "draft")    -> True
    write("/reports/2025/q1.txt", "final")    -> True      # 覆盖内容
    read("/reports/2025/q1.txt")              -> "final"
    ls("/reports")                            -> ["2025"]
    ls("/reports/2025")                       -> ["q1.txt"]
    read("/reports/2025")                     -> None      # 是目录，不是文件
    write("/reports/2025", "oops")            -> False     # 该处已经有一个目录
    ls("/reports/2025/q1.txt")                -> None      # 是文件，不是目录
    mkdir("/reports/2025/q1.txt")             -> False     # 该处已经有一个文件
"""

EXAMPLE_L2 = """
    mkdir("/reports/2025")                               -> True
    write("/reports/2025/q1.txt", "Q1 results")          -> True
    write("/reports/2025/q2.txt", "Q2 results here")     -> True
    mkdir("/archive")                                    -> True
    cp("/reports/2025", "/archive/2025")                 -> True           # 深拷贝
    write("/archive/2025/q1.txt", "changed")             -> True           # 只有副本发生变化
    read("/reports/2025/q1.txt")                         -> "Q1 results"   # 原件未受影响
    size("/reports/2025/q1.txt")                         -> 10
    size("/reports/2025")                                -> 25             # 10 + 15，递归求和
    mv("/reports/2025/q2.txt", "/archive/2025/q2.txt")   -> True
    ls("/reports/2025")                                  -> ["q1.txt"]
    ls("/archive/2025")                                  -> ["q1.txt", "q2.txt"]
    mv("/reports", "/reports/2025/nested")               -> False          # 目标在源的自身子树内
    rm("/reports/2025/q1.txt")                           -> True
    rm("/reports")                                       -> True           # 递归删除
    ls("/reports")                                       -> None           # 已不存在
"""

EXAMPLE_L3 = """
    add_user("alice", 20)                                -> True
    add_user("alice", 50)                                -> False   # 已存在
    write("/data/a.txt", "hello world", owner="alice")   -> True    # 大小 11，11 <= 20
    write("/data/b.txt", "hi", owner="alice")            -> True    # 大小 2，13 <= 20
    write("/data/c.txt", "0123456789", owner="alice")    -> False   # 大小 10，23 > 20
    write("/data/a.txt", "hey", owner="bob")             -> False   # a.txt 属于 alice，不是 bob
    update_capacity("alice", 5)                          -> 1       # 淘汰更大的 a.txt（11）
    read("/data/a.txt")                                  -> None    # 已被淘汰
    read("/data/b.txt")                                  -> "hi"    # 保留，仍然放得下
"""

EXAMPLE_L4 = """
    mkdir("/work")                                       -> True
    write("/work/draft.txt", "v1")                       -> True
    snapshot("before-edit")                              -> True
    write("/work/draft.txt", "v2")                       -> True
    snapshot("before-edit")                              -> False   # id 已被使用
    read("/work/draft.txt")                              -> "v2"
    restore("before-edit")                               -> True
    read("/work/draft.txt")                              -> "v1"
    restore("nope")                                      -> False   # 不存在这个快照
    add_user("carol", 10)                                -> True
    write("/work/notes.txt", "abcde", owner="carol")     -> True
    snapshot("with-carol")                               -> True
    update_capacity("carol", 2)                          -> 1       # 淘汰 notes.txt
    read("/work/notes.txt")                              -> None
    restore("with-carol")                                -> True
    read("/work/notes.txt")                              -> "abcde" # 已恢复
"""


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, EXAMPLE_L1)


@case("Level 1", title="规范化：连续的 / 与末尾的 /")
def l1_normalize(sol):
    run(sol, """
        mkdir("/a//b/c")              -> True
        ls("/a")                      -> ["b"]        # "//" 和 "/" 一样，不会出现空名字
        ls("/a/b")                    -> ["c"]
        ls("/a//b///")                -> ["c"]        # 连续的 / 合并，末尾的 / 去掉
        mkdir("/x/y/")                -> True
        ls("/x")                      -> ["y"]
        write("/x/y//f.txt/", "hi")   -> True         # 同一个路径的另一种写法
        read("/x/y/f.txt")            -> "hi"
        read("//x//y//f.txt")         -> "hi"
        ls("//")                      -> ["a", "x"]   # "//" 规范化后就是根目录
        mkdir("/a/b/c/")              -> True         # 已经是目录
        ls("/a/b")                    -> ["c"]
        ls("/a/b/c")                  -> []
    """)


_BAD_PATHS = ["", "a", "a/f", " /a", "./a", "/.", "/..", "/a/.", "/a/./b", "/a/../a/f", "/a/b/..", "/a/..//",
              "/./a/f"]


@case("Level 1", title="非法路径一律当作不存在")
def l1_invalid(sol):
    for bad in _BAD_PATHS:
        b = _lit(bad)
        run(sol, f"""
            mkdir("/a/b")            -> True
            write("/a/f", "data")    -> True
            mkdir({b})               -> False       # 非法路径
            write({b}, "x")          -> False       # 非法路径
            read({b})                -> None        # 非法路径（"." 和 ".." 不表示导航）
            ls({b})                  -> None        # 非法路径
            ls("/")                  -> ["a"]       # 失败的调用没有创建任何东西
            ls("/a")                 -> ["b", "f"]
            read("/a/f")             -> "data"      # 也没有改动已有的文件
        """)


@case("Level 1", title="含点的名字是普通名字")
def l1_dots(sol):
    run(sol, """
        mkdir("/.../x")          -> True    # "..." 不是 "." 也不是 ".."，是普通的名字
        write("/.a", "1")        -> True
        write("/a..", "2")       -> True
        write("/..b/c", "3")     -> True
        ls("/")                  -> ["...", "..b", ".a", "a.."]
        read("/..b/c")           -> "3"
        ls("/...")               -> ["x"]
        read("/a..")             -> "2"
    """)


@case("Level 1", title="根目录")
def l1_root(sol):
    run(sol, """
        ls("/")                   -> []       # 新建的文件系统只有根目录
        read("/")                 -> None     # 根目录是目录
        mkdir("/")                -> True     # 根目录永远存在，且是目录
        write("/", "x")           -> False    # 根目录不能是文件
        write("//", "x")          -> False    # 规范化后也是根目录
        write("/top", "t")        -> True
        ls("/")                   -> ["top"]
        mkdir("/top")             -> False    # 已经是文件
        mkdir("/top/sub")         -> False    # 祖先是文件
        write("/top/sub", "x")    -> False    # 祖先是文件
        read("/top/sub")          -> None     # 祖先是文件
        ls("/top")                -> None     # 是文件
        ls("/")                   -> ["top"]
    """)


@case("Level 1", title="失败的调用不留下任何东西")
def l1_failed_leaves_nothing(sol):
    run(sol, """
        mkdir("/n1/n2/..")          -> False        # 非法路径：不能先建出 /n1 和 /n1/n2
        write("/w1/w2/./f", "x")    -> False        # 非法路径
        mkdir("/m1/./m2")           -> False        # 非法路径
        ls("/")                     -> []           # 上面三次失败都没有留下目录
        write("/f", "data")         -> True
        mkdir("/f/g/h")             -> False        # /f 是文件
        write("/f/g/h", "x")        -> False        # /f 是文件
        read("/f")                  -> "data"       # 原文件不受影响
        mkdir("/d/e")               -> True
        write("/d", "x")            -> False        # /d 是目录
        ls("/d")                    -> ["e"]        # 目录不受影响
        write("/d/e", "x")          -> False        # /d/e 是目录
        ls("/")                     -> ["d", "f"]
    """)


@case("Level 1", title="区分大小写、按码点排序")
def l1_names(sol):
    run(sol, """
        mkdir("/docs")                     -> True
        mkdir("/Docs")                     -> True     # 区分大小写：另一个目录
        write("/docs/b.txt", "1")          -> True
        write("/docs/B.txt", "2")          -> True
        write("/docs/a.txt", "3")          -> True
        write("/docs/10", "4")             -> True
        write("/docs/9", "5")              -> True
        mkdir("/docs/_x")                  -> True
        write("/docs/my file.txt", "6")    -> True     # 名字可以含空格
        write("/docs/é", "7")              -> True
        write("/docs/文件", "8")            -> True
        ls("/docs")                        -> ["10", "9", "B.txt", "_x", "a.txt", "b.txt", "my file.txt", "é", "文件"]   # 按码点排序：数字 < 大写 < "_" < 小写 < 非 ASCII；文件和目录一起排
        ls("/")                            -> ["Docs", "docs"]
        read("/docs/B.txt")                -> "2"
        read("/DOCS/b.txt")                -> None
        ls("/Docs")                        -> []
        read("/docs/文件")                  -> "8"
    """)


@case("Level 1", title="空内容与覆盖")
def l1_contents(sol):
    run(sol, """
        write("/e", "")              -> True
        read("/e")                   -> ""       # 空文件的内容是 ""，不是 None
        write("/e", "now")           -> True
        read("/e")                   -> "now"
        write("/e", "")              -> True     # 覆盖成空内容
        read("/e")                   -> ""
        write("/u", "héllo 世界")     -> True
        read("/u")                   -> "héllo 世界"
        read("/missing")             -> None
        read("/e/x")                 -> None
        ls("/")                      -> ["e", "u"]
    """)


@case("Level 1", cpu=2, title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=300, length=(5, 30))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, EXAMPLE_L2)


@case("Level 2", title="移动文件与整个目录")
def l2_mv(sol):
    run(sol, """
        write("/a/x.txt", "xx")                   -> True
        write("/a/sub/y.txt", "yyy")              -> True
        mv("/a/x.txt", "/b/c/x2.txt")             -> True    # 目标缺失的祖先目录会被自动创建
        read("/a/x.txt")                          -> None    # 源不再存在
        read("/b/c/x2.txt")                       -> "xx"
        ls("/b")                                  -> ["c"]
        mv("/a", "/z")                            -> True    # 整个目录连同子树一起移动
        ls("/")                                   -> ["b", "z"]
        read("/z/sub/y.txt")                      -> "yyy"
        mv("/z/sub/y.txt", "/z/sub/old_y.txt")    -> True    # 改名
        ls("/z/sub")                              -> ["old_y.txt"]
        mv("/z/sub", "/b/c/sub")                  -> True
        ls("/z")                                  -> []      # 源的父目录还在，只是空了
        ls("/b/c")                                -> ["sub", "x2.txt"]
        size("/b")                                -> 5       # "xx" + "yyy"
    """)


@case("Level 2", title="目标已存在：覆盖文件、替换目录")
def l2_mv_overwrite(sol):
    run(sol, """
        write("/f1", "one")               -> True
        write("/f2", "second")            -> True
        mv("/f1", "/f2")                  -> True    # 文件覆盖文件
        read("/f2")                       -> "one"
        read("/f1")                       -> None    # 源不再存在
        write("/src/keep.txt", "k")       -> True
        write("/src/both.txt", "new")     -> True
        write("/dst/both.txt", "old")     -> True
        write("/dst/only.txt", "o")       -> True
        mkdir("/dst/deep/er")             -> True
        mv("/src", "/dst")                -> True    # 目录替换目录：/dst 原来的整棵子树被丢弃
        ls("/dst")                        -> ["both.txt", "keep.txt"]   # 不是合并：only.txt 和 deep 都没了
        read("/dst/both.txt")             -> "new"
        read("/dst/only.txt")             -> None
        ls("/src")                        -> None
        size("/")                         -> 7       # "one" + "k" + "new"
    """)


@case("Level 2", title="移动失败的各种情况")
def l2_mv_rejected(sol):
    run(sol, """
        write("/p/q/f.txt", "ffff")       -> True
        write("/p/g.txt", "gg")           -> True
        write("/file", "F")               -> True
        mv("/nope", "/x")                 -> False   # 源不存在
        mv("/p/nope/f.txt", "/x")         -> False   # 源的祖先不存在
        mv("/file/x", "/y")               -> False   # 源的祖先是文件
        mv("/p", "/p")                    -> False   # 同一个路径
        mv("/p", "/p/")                   -> False   # 规范化后是同一个路径
        mv("/p", "/p/q/new")              -> False   # 目标在源的子树内
        mv("/p/q", "/p")                  -> False   # 目标是源的祖先
        mv("/p/q/f.txt", "/")             -> False   # 根目录是所有路径的祖先
        mv("/", "/r")                     -> False   # 根目录在任何路径的祖先链上
        mv("/p/g.txt", "/file/g.txt")     -> False   # 目标的祖先是文件
        mv("/p/g.txt", "/p/q")            -> False   # 文件不能覆盖目录
        mv("/p/q", "/file")               -> False   # 目录不能覆盖文件
        mv("/p/../file", "/z")            -> False   # 非法路径
        mv("/file", "/p/./z")             -> False   # 非法路径
        mv("/nope", "/new/deep/x")        -> False
        ls("/")                           -> ["file", "p"]   # 上面的失败都没有创建任何东西
        ls("/p")                          -> ["g.txt", "q"]
        ls("/p/q")                        -> ["f.txt"]       # 尤其没有 /p/q/new
        read("/file")                     -> "F"
        size("/p")                        -> 6
    """)


@case("Level 2", title="同一祖先链按分段判断，不按字符串前缀")
def l2_component_prefix(sol):
    run(sol, """
        write("/a/f", "1")          -> True
        mv("/a", "/ab")             -> True    # "/ab" 不在 "/a" 的子树里，只是名字以 a 开头
        ls("/")                     -> ["ab"]
        read("/ab/f")               -> "1"
        cp("/ab", "/a")             -> True    # "/a" 也不是 "/ab" 的祖先
        mv("/ab/f", "/ab/f2")       -> True
        cp("/ab/f2", "/ab/f")       -> True    # 兄弟之间：f 不是 f2 的祖先
        ls("/ab")                   -> ["f", "f2"]
        mv("/ab", "/abc/d")         -> True
        ls("/")                     -> ["a", "abc"]
        cp("/abc", "/ab")           -> True
        ls("/ab")                   -> ["d"]
    """)


@case("Level 2", title="复制是深拷贝（多层目录）")
def l2_cp_deep(sol):
    run(sol, """
        write("/src/a/b/deep.txt", "deep")       -> True
        write("/src/top.txt", "top")             -> True
        cp("/src", "/copy")                      -> True
        write("/copy/a/b/deep.txt", "CHANGED")   -> True
        read("/src/a/b/deep.txt")                -> "deep"     # 原件不受影响
        write("/src/a/b/new.txt", "n")           -> True
        ls("/copy/a/b")                          -> ["deep.txt"]   # 原件里新建的文件不会出现在副本里
        mkdir("/copy/a/extra")                   -> True
        ls("/src/a")                             -> ["b"]
        rm("/copy/a/b")                          -> True
        ls("/src/a/b")                           -> ["deep.txt", "new.txt"]
        read("/copy/top.txt")                    -> "top"
        size("/src")                             -> 8          # 4 + 3 + 1
        size("/copy")                            -> 3
    """)


@case("Level 2", title="复制到已存在的目标、复制失败")
def l2_cp_overwrite(sol):
    run(sol, """
        write("/f", "abc")            -> True
        write("/g", "zz")             -> True
        cp("/f", "/g")                -> True    # 文件覆盖文件，源保留
        read("/g")                    -> "abc"
        read("/f")                    -> "abc"
        write("/f", "changed")        -> True
        read("/g")                    -> "abc"   # 副本与源互不影响
        write("/d1/x", "1")           -> True
        write("/d2/y", "22")          -> True
        cp("/d1", "/d2")              -> True    # 目录替换目录
        ls("/d2")                     -> ["x"]
        ls("/d1")                     -> ["x"]
        cp("/d1", "/f")               -> False   # 目录不能覆盖文件
        cp("/f", "/d1")               -> False   # 文件不能覆盖目录
        cp("/d1", "/d1/inner")        -> False   # 目标在源的子树内
        cp("/d1/x", "/d1")            -> False   # 目标是源的祖先
        cp("/d1/x", "/d1/x")          -> False   # 同一个路径
        cp("/missing", "/m2/m3")      -> False   # 源不存在
        cp("/f", "/g/h")              -> False   # 目标的祖先是文件
        ls("/")                       -> ["d1", "d2", "f", "g"]   # 失败的复制没有创建任何东西
        size("/")                     -> 12      # 7 + 3 + 1 + 1
    """)


@case("Level 2", title="删除")
def l2_rm(sol):
    run(sol, """
        write("/a/b/c.txt", "abc")      -> True
        write("/a/d.txt", "d")          -> True
        rm("/")                         -> False   # 根目录永远不能删除
        rm("//")                        -> False   # 规范化后是根目录
        rm("/a/nope")                   -> False   # 不存在
        rm("/a/d.txt/x")                -> False   # 祖先是文件
        rm("/a/./d.txt")                -> False   # 非法路径
        rm("/a/d.txt")                  -> True
        rm("/a/d.txt")                  -> False   # 已经删除
        rm("/a/b/")                     -> True    # 末尾的 / 被规范化掉；递归删除
        ls("/a")                        -> []
        read("/a/b/c.txt")              -> None
        write("/a/b", "now a file")     -> True    # 名字空出来，可以换成另一种类型
        mkdir("/a/d.txt")               -> True
        rm("/a")                        -> True
        ls("/")                         -> []
        size("/")                       -> 0
        rm("/a")                        -> False   # 已经删除
    """)


@case("Level 2", title="大小：字符数，目录递归求和")
def l2_size(sol):
    run(sol, """
        mkdir("/empty")                     -> True
        size("/empty")                      -> 0       # 空目录
        write("/empty/zero", "")            -> True
        size("/empty")                      -> 0       # 只有空文件
        write("/u/e.txt", "héllo")          -> True
        size("/u/e.txt")                    -> 5       # 字符数，不是 UTF-8 字节数
        write("/u/cjk.txt", "日本語")        -> True
        size("/u/cjk.txt")                  -> 3
        write("/u/deep/er/f", "0123456789") -> True
        size("/u/deep")                     -> 10
        size("/u")                          -> 18      # 5 + 3 + 10，递归
        size("/")                           -> 18
        size("//u//")                       -> 18
        size("/nope")                       -> None    # 不存在
        size("/u/e.txt/x")                  -> None    # 祖先是文件
        size("/u/../u")                     -> None    # 非法路径
    """)


@case("Level 2", cpu=2, title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=300, length=(5, 30))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, EXAMPLE_L3)


@case("Level 3", title="注册用户与参数检查")
def l3_users(sol):
    run(sol, """
        add_user("admin", 10)                    -> False   # "admin" 永远不能注册
        add_user("u", -1)                        -> False   # 容量不能为负
        add_user("u", 0)                         -> True    # 容量 0 是允许的
        add_user("u", 5)                         -> False   # 已经注册
        write("/e", "", owner="u")               -> True    # 大小 0 <= 0
        write("/f", "x", owner="u")              -> False   # 1 > 0
        update_capacity("admin", 5)              -> None    # "admin" 不是注册用户
        update_capacity("nobody", 5)             -> None
        update_capacity("u", -1)                 -> None    # 容量不能为负
        write("/f", "x", owner="u")              -> False   # 失败的 update_capacity 没有改容量
        update_capacity("u", 3)                  -> 0       # 已用 0 <= 3
        write("/f", "xyz", owner="u")            -> True
        write("/g", "x", owner="u")              -> False   # 3 + 1 > 3
        write("/h", "x", owner="nobody")         -> False   # owner 必须是 "admin" 或已注册用户
        read("/h")                               -> None
        write("/h", "x" * 100)                   -> True    # 默认 owner 是 "admin"，不受限制
        write("/h2", "y" * 100, owner="admin")   -> True
    """)


@case("Level 3", title="所有者固定：只有原所有者能覆盖")
def l3_owner_rules(sol):
    run(sol, """
        add_user("ann", 100)                       -> True
        add_user("bob", 100)                       -> True
        write("/a.txt", "ann's", owner="ann")      -> True
        write("/a.txt", "bob's", owner="bob")      -> False   # a.txt 属于 ann
        write("/a.txt", "admin")                   -> False   # admin 也不能覆盖 ann 的文件
        read("/a.txt")                             -> "ann's"
        write("/sys.txt", "root")                  -> True    # 属于 admin
        write("/sys.txt", "mine", owner="ann")     -> False   # ann 不能覆盖 admin 的文件
        write("/a.txt", "v2", owner="ann")         -> True    # 原所有者可以覆盖
        read("/a.txt")                             -> "v2"
        ls("/")                                    -> ["a.txt", "sys.txt"]   # read、ls、size 不受所有者影响
        size("/")                                  -> 6
        rm("/a.txt")                               -> True    # 所有权不约束 rm
        write("/a.txt", "now bob", owner="bob")    -> True    # 删除之后是新文件，所有者是 bob
        write("/a.txt", "x", owner="ann")          -> False
    """)


@case("Level 3", title="写入配额：恰好放满、覆盖自己的文件")
def l3_write_quota(sol):
    run(sol, """
        add_user("u", 10)                    -> True
        write("/a", "x" * 6, owner="u")      -> True
        write("/b", "x" * 5, owner="u")      -> False   # 6 + 5 > 10
        write("/b", "x" * 4, owner="u")      -> True    # 恰好 10
        write("/c", "", owner="u")           -> True    # 空文件不占配额
        write("/a", "x" * 2, owner="u")      -> True    # 覆盖自己的文件：4 + 2，缩小永远不会失败
        write("/a", "x" * 6, owner="u")      -> True    # 4 + 6 = 10：原来的大小不重复计算
        write("/a", "x" * 7, owner="u")      -> False   # 4 + 7 > 10
        read("/a")                           -> "x" * 6 # 失败的覆盖不改变内容
        write("/d", "x", owner="u")          -> False   # 仍然是满的
    """)


@case("Level 3", title="删除释放配额")
def l3_rm_frees(sol):
    run(sol, """
        add_user("u", 10)                    -> True
        add_user("v", 10)                    -> True
        write("/d/u1", "x" * 4, owner="u")   -> True
        write("/d/u2", "x" * 6, owner="u")   -> True
        write("/d/v1", "x" * 7, owner="v")   -> True
        write("/d/sys", "x" * 50)            -> True
        write("/u3", "x", owner="u")         -> False   # u 已满
        rm("/d/u2")                          -> True
        write("/u3", "x" * 6, owner="u")     -> True    # 删除释放了 6
        write("/u4", "x", owner="u")         -> False   # 4 + 6 = 10，又满了
        rm("/d")                             -> True    # 递归删除：u 释放 4，v 释放 7
        write("/u4", "x" * 4, owner="u")     -> True    # 6 + 4 = 10
        write("/u5", "x", owner="u")         -> False
        write("/v2", "x" * 10, owner="v")    -> True    # v 的 7 也释放了
    """)


@case("Level 3", title="移动保留所有者，不检查容量")
def l3_mv_owner_quota(sol):
    run(sol, """
        add_user("u", 10)                        -> True
        add_user("v", 10)                        -> True
        write("/u/f", "x" * 10, owner="u")       -> True    # u 已满
        mv("/u/f", "/shared/f")                  -> True    # 移动不改变任何人的总量，满了也能移动
        mv("/u", "/u2")                          -> True
        write("/shared/f", "y", owner="v")       -> False   # 移动后仍属于 u
        write("/shared/f", "y" * 10, owner="u")  -> True    # 覆盖自己的文件
        write("/x", "x", owner="u")              -> False   # u 仍然是满的
        write("/v/g", "z" * 3, owner="v")        -> True
        mv("/v/g", "/shared/f")                  -> False   # 目标文件属于 u，来的文件属于 v
        mv("/shared/f", "/v/g")                  -> False   # 反过来也不行
        mv("/shared", "/v/s")                    -> True    # 整个目录移动，满了也行
        write("/v/s/f", "y", owner="u")          -> True    # 仍属于 u（缩小）
        write("/w1", "x" * 3, owner="v")         -> True
        write("/w2", "x" * 4, owner="v")         -> True    # v 已满：3 + 3 + 4
        mv("/w1", "/w2")                         -> True    # 覆盖自己的文件：释放 w2 的 4
        write("/w3", "x" * 4, owner="v")         -> True    # 3 + 3 + 4 = 10
        write("/w4", "x", owner="v")             -> False
    """)


@case("Level 3", title="复制计入配额，失败就整体不复制")
def l3_cp_quota(sol):
    run(sol, """
        add_user("u", 10)                    -> True
        add_user("v", 20)                    -> True
        write("/d/uf", "x" * 4, owner="u")   -> True
        write("/d/vf", "x" * 5, owner="v")   -> True
        write("/d/sys", "x" * 99)            -> True    # admin 的文件不受限制
        cp("/d", "/c1")                      -> True    # u：4 + 4 <= 10，v：5 + 5 <= 20
        cp("/d", "/c2")                      -> False   # u 需要 12 > 10；v 够用也一起失败
        ls("/")                              -> ["c1", "d"]   # 失败的复制什么都没留下
        write("/v2", "x" * 10, owner="v")    -> True    # v：10 + 10，失败的复制没有计入 v
        write("/v3", "x", owner="v")         -> False
        write("/c1/uf", "x" * 2, owner="u")  -> True    # 副本属于 u：4 + 2
        write("/u2", "x" * 4, owner="u")     -> True    # u：4 + 2 + 4 = 10
        cp("/d/sys", "/sys2")                -> True    # 复制 admin 的文件不检查容量
        cp("/u2", "/new/deep/u2")            -> False   # u 已满
        ls("/new")                           -> None    # 失败的复制不会创建目标的祖先目录
    """)


@case("Level 3", title="复制覆盖文件：所有者与配额")
def l3_cp_overwrite(sol):
    run(sol, """
        add_user("u", 10)                    -> True
        add_user("v", 10)                    -> True
        write("/small", "x" * 4, owner="u")  -> True
        write("/big", "y" * 6, owner="u")    -> True    # u 已满
        cp("/small", "/big")                 -> True    # 和 write 一样按结果算：4 + 4 = 8 <= 10
        read("/big")                         -> "x" * 4
        write("/z", "z" * 2, owner="u")      -> True    # u：10
        write("/zz", "z", owner="u")         -> False
        cp("/z", "/small")                   -> True    # 4 + 2 + 2 = 8
        write("/zz", "z" * 2, owner="u")     -> True
        write("/vf", "v", owner="v")         -> True
        cp("/vf", "/big")                    -> False   # 目标文件属于 u，来的文件属于 v
        cp("/big", "/vf")                    -> False
        cp("/big", "/sys")                   -> False   # u 已满，新文件放不下
        write("/sys", "s")                   -> True    # admin 的文件
        cp("/vf", "/sys")                    -> False   # 目标属于 admin
        read("/sys")                         -> "s"
        cp("/big", "/small")                 -> False   # u：10 - 2 + 4 = 12 > 10
        read("/small")                       -> "zz"    # 失败的复制不改动目标
        write("/w", "z", owner="u")          -> False   # 用量也没变：仍是 10
    """)


@case("Level 3", title="替换目录：旧子树的用量被释放")
def l3_dir_replace_quota(sol):
    run(sol, """
        add_user("u", 10)                    -> True
        add_user("v", 10)                    -> True
        write("/old/a", "x" * 6, owner="u")  -> True
        write("/old/b", "x" * 3, owner="v")  -> True
        write("/new/c", "x" * 4, owner="u")  -> True    # u 已满：6 + 4
        cp("/new", "/old")                   -> True    # 目录替换目录，按结果算：u 释放 a 的 6，10 - 6 + 4 = 8 <= 10
        ls("/old")                           -> ["c"]
        write("/u2", "x" * 2, owner="u")     -> True    # u：8 + 2 = 10
        write("/u3", "x", owner="u")         -> False   # u 又满了
        write("/v2", "x" * 10, owner="v")    -> True    # 被替换掉的 b（3）也还给了 v：0 + 10
        mkdir("/empty")                      -> True
        mv("/empty", "/old")                 -> True    # 移动替换目录：释放 /old/c 的 4
        ls("/old")                           -> []
        write("/u3", "x" * 4, owner="u")     -> True    # u：10 - 4 + 4 = 10
        write("/u4", "x", owner="u")         -> False
        size("/")                            -> 20
    """)


@case("Level 3", title="淘汰顺序：大的先删，并列按完整路径")
def l3_evict_order(sol):
    run(sol, """
        add_user("g", 100)                       -> True
        write("/z/bb.txt", "x" * 10, owner="g")  -> True
        write("/z/aa.txt", "y" * 10, owner="g")  -> True
        write("/z/cc.txt", "z" * 5, owner="g")   -> True
        write("/z/big.txt", "a" * 50)            -> True    # 属于 admin，不会被淘汰
        update_capacity("g", 15)                 -> 1       # 两个 10 一样大：路径 "/z/aa.txt" 较小，先删它
        read("/z/aa.txt")                        -> None    # 已被淘汰
        read("/z/bb.txt")                        -> "x" * 10    # 删掉一个后 15 <= 15，保留
        read("/z/big.txt")                       -> "a" * 50
        add_user("h", 100)                       -> True
        write("/a/b", "xxx", owner="h")          -> True
        write("/a.b", "yyy", owner="h")          -> True
        write("/B", "zzz", owner="h")            -> True
        update_capacity("h", 6)                  -> 1       # 一样大：按完整路径字符串比较，"/B" 最小
        read("/B")                               -> None    # 已被淘汰
        update_capacity("h", 3)                  -> 1       # "/a.b" < "/a/b"（"." 在 "/" 前面）
        read("/a.b")                             -> None    # 被淘汰的应当是 "/a.b"
        read("/a/b")                             -> "xxx"   # 保留
        add_user("k", 100)                       -> True
        write("/z/b", "pppp", owner="k")         -> True
        write("/y/x/c", "qqqq", owner="k")       -> True
        update_capacity("k", 4)                  -> 1       # 比较的是完整路径 "/y/x/c" < "/z/b"，不是文件名
        read("/y/x/c")                           -> None    # 已被淘汰
        read("/z/b")                             -> "pppp"  # 保留
    """)


@case("Level 3", title="只删到不超出为止；目录保留")
def l3_evict_amount(sol):
    run(sol, """
        add_user("e", 30)                        -> True
        write("/d1/big", "x" * 10, owner="e")    -> True
        write("/d2/mid", "x" * 8, owner="e")     -> True
        write("/d3/small", "x" * 3, owner="e")   -> True
        write("/d3/empty", "", owner="e")        -> True
        update_capacity("e", 21)                 -> 0       # 已用 21 <= 21：什么都不删
        update_capacity("e", 12)                 -> 1       # 删掉最大的 big 后 11 <= 12 就停止
        ls("/d1")                                -> []      # 只删除文件，目录保留
        read("/d2/mid")                          -> "x" * 8
        write("/n", "x", owner="e")              -> True    # 11 + 1 <= 12
        write("/n2", "x", owner="e")             -> False
        update_capacity("e", 0)                  -> 3       # mid、small、n；空文件不占配额，保留
        read("/d3/empty")                        -> ""
        ls("/d3")                                -> ["empty"]
        update_capacity("e", 50)                 -> 0
        write("/big2", "x" * 50, owner="e")      -> True
    """)


@case("Level 3", title="被拒绝的移动和复制不改变用量")
def l3_failed_keep_usage(sol):
    run(sol, """
        add_user("f", 15)                          -> True
        write("/p/q/f.txt", "x" * 10, owner="f")   -> True
        write("/p/g.txt", "y" * 5, owner="f")      -> True
        mv("/p/q", "/p")                           -> False   # 目标是源的祖先
        cp("/p/q", "/p")                           -> False
        mv("/p", "/p/q/r")                         -> False   # 目标在源的子树内
        write("/h", "x", owner="f")                -> False   # 用量仍是 15
        size("/p")                                 -> 15
        rm("/p/g.txt")                             -> True
        write("/h", "x" * 5, owner="f")            -> True
        write("/h2", "x", owner="f")               -> False
    """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=400, length=(10, 40))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, EXAMPLE_L4)


@case("Level 4", title="快照 id：用过就不能再用")
def l4_snapshot_ids(sol):
    run(sol, """
        snapshot("empty")            -> True    # 空文件系统也可以拍快照
        write("/f", "1")             -> True
        snapshot("s1")               -> True
        write("/f", "2")             -> True
        snapshot("s2")               -> True
        restore("s1")                -> True
        read("/f")                   -> "1"
        snapshot("s2")               -> False   # s2 以前拍过：id 已被使用
        restore("s2")                -> True    # 恢复 s1 不会让之后拍的 s2 消失
        read("/f")                   -> "2"
        snapshot("s1")               -> False
        restore("s3")                -> False   # 从没拍过
        read("/f")                   -> "2"     # 失败的恢复不改变状态
        restore("empty")             -> True
        ls("/")                      -> []
        restore("s1")                -> True
        read("/f")                   -> "1"
    """)


@case("Level 4", title="恢复出来的状态与快照互相独立")
def l4_restore_independent(sol):
    run(sol, """
        write("/d/f", "one")         -> True
        snapshot("s")                -> True
        write("/d/f", "two")         -> True
        restore("s")                 -> True
        write("/d/f", "three")       -> True    # 修改恢复出来的状态
        write("/d/g", "new")         -> True
        restore("s")                 -> True    # 再恢复一次：结果和第一次一样
        read("/d/f")                 -> "one"   # 恢复之后的修改不能改到快照
        ls("/d")                     -> ["f"]
        rm("/d")                     -> True
        restore("s")                 -> True
        ls("/d")                     -> ["f"]
        mv("/d", "/e")               -> True
        cp("/e", "/d")               -> True
        write("/e/f", "four")        -> True
        restore("s")                 -> True
        ls("/")                      -> ["d"]
        read("/d/f")                 -> "one"
        snapshot("t")                -> True    # 拍的是恢复之后的状态
        write("/d/f", "five")        -> True
        restore("t")                 -> True
        read("/d/f")                 -> "one"
    """)


@case("Level 4", title="快照包含用户、容量和用量")
def l4_restore_users(sol):
    run(sol, """
        add_user("u", 10)                    -> True
        write("/u1", "x" * 6, owner="u")     -> True
        snapshot("s")                        -> True
        add_user("late", 5)                  -> True    # 快照之后才注册
        write("/l", "x", owner="late")       -> True
        update_capacity("u", 20)             -> 0
        write("/u2", "x" * 14, owner="u")    -> True    # u：6 + 14 = 20
        restore("s")                         -> True
        read("/u2")                          -> None
        read("/l")                           -> None
        write("/x", "x", owner="late")       -> False   # 快照里没有 "late" 这个用户
        add_user("late", 7)                  -> True    # 可以重新注册
        write("/u3", "x" * 5, owner="u")     -> False   # u 的容量回到 10，已用 6
        write("/u3", "x" * 4, owner="u")     -> True    # 6 + 4 = 10：用量回到快照时的 6
        restore("s")                         -> True
        write("/u3", "x" * 4, owner="u")     -> True    # 再恢复一次：用量又回到 6
        update_capacity("u", 4)              -> 1       # 删掉 u1（6）
        restore("s")                         -> True
        read("/u1")                          -> "x" * 6
        write("/u4", "x" * 5, owner="u")     -> False   # 容量和用量都回到快照时：10 和 6
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=400, length=(15, 50))
