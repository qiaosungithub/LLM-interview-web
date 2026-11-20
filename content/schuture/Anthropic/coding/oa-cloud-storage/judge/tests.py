import ast
import time

from judgelib import Mismatch, calling, case, expect, expect_true, rng
import ref

SHOW = 60           # a replay lists at most this many calls (every sequence these tests make fits)


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_lit(v) for v in x) + "]"
    return repr(x)


def _replay(done, wants, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step."""
    lines = [f"cs.{m}({', '.join(_lit(a) for a in args)})" for m, args in done]
    lines[-1] = f"print({lines[-1]})"
    notes = [f"# -> {_lit(w)}" for w in wants] + [mark]
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = ["cs = CloudStorage()"] + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(head + [s.ljust(width) + n for s, n in zip(lines, notes)])


def _method(obj, name):
    fn = getattr(obj, name, None)
    expect_true(callable(fn), f"CloudStorage 没有 {name} 方法（方法名要和题目一致）")
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
    if want is None:            # the result hides an expected value of None, so say it in the message
        msg = f"{msg}（期望返回 None）" if msg else "期望返回 None"
    if not msg and type(got) is type(want) and not isinstance(want, (str, list)):
        msg = "返回值不对"
    expect(got, want, call=call, msg=msg)       # raises for every difference that == sees
    raise Mismatch(f"类型不对：期望 {type(want).__name__}，得到 {type(got).__name__}", call, want, got)


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines like  add_file_by("u", "f", 4) -> 6   # why  become (method, args, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        call, arrow, rest = line.strip().partition("->")
        assert arrow, f"missing '->' in {line!r}"
        expr, _, why = rest.partition("#")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(ast.literal_eval(a) for a in node.args)
        steps.append((node.func.id, args, ast.literal_eval(expr.strip()), why.strip()))
    return steps


def run(sol, *scripts):
    """Runs the calls on a fresh CloudStorage, checking each return value as soon as it comes back."""
    cs = sol.CloudStorage()
    done, wants = [], []
    for method, args, want, why in [s for script in scripts for s in _parse(script)]:
        done.append((method, args))
        fn = _method(cs, method)
        calling(_replay(done, wants))       # shown if the answer raises or hangs in this call
        try:
            got = fn(*args)
        except Exception:
            calling(_replay(done, wants, mark="# <- 这一步抛出异常"))
            raise
        if not _same(got, want):
            _differs(got, want, _replay(done, wants), why)
        wants.append(want)


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, ops):
    """Replays ops on a fresh answer and a fresh ref.CloudStorage; None if every return value agrees."""
    cs, model, wants = sol.CloudStorage(), ref.CloudStorage(), []
    for k, (method, args) in enumerate(ops):
        want = getattr(model, method)(*args)
        fn = _method(cs, method)
        try:
            got = fn(*args)
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


# ---------------------------------------------------------------- the statement's examples, one store

EXAMPLE_L1 = """
    add_file("report.txt", 200)                   -> True
    add_file("report.txt", 10)                    -> False   # "report.txt" 已经存在
    copy_file("report.txt", "report_backup.txt")  -> True
    copy_file("missing.txt", "x.txt")             -> False   # source 不存在
    copy_file("report.txt", "report_backup.txt")  -> False   # destination 已经存在
    get_file_size("report_backup.txt")            -> 200
    get_file_size("nope.txt")                     -> None
"""

EXAMPLE_L2 = """
    add_file("report_draft.txt", 200)             -> True
    add_file("report_final.txt", 500)             -> True
    add_file("summary.md", 50)                    -> True
    find_files("report", ".txt")                  -> ["report_final.txt(500)", "report.txt(200)", "report_backup.txt(200)", "report_draft.txt(200)"]   # 500 排最前；并列 200 的按名称升序（ASCII 中 "." 在 "_" 前面）
    find_files("summary", "")                     -> ["summary.md(50)"]
"""

EXAMPLE_L3 = """
    add_user("alice", 500)                                  -> True
    add_user("alice", 100)                                  -> False   # "alice" 已经存在
    add_file_by("alice", "alice_a.txt", 300)                -> 200     # 剩余：500 - 300
    add_file_by("alice", "report.txt", 10)                  -> None    # "report.txt" 已经存在，且没有主人
    add_file_by("carol", "c.txt", 10)                       -> None    # "carol" 不是用户
    add_file_by("alice", "alice_b.txt", 250)                -> None    # 250 超过了 alice 剩余的 200
    get_file_size("alice_a.txt")                            -> 300
    copy_file("alice_a.txt", "alice_a_copy.txt")            -> False   # 还需要 300，但 alice 只剩 200
    copy_file("report_final.txt", "report_final_copy.txt")  -> True    # source 没有主人：不检查容量
    add_file_by("alice", "alice_small.txt", 100)            -> 100
    copy_file("alice_small.txt", "alice_small_copy.txt")    -> True    # 正好需要 100，alice 正好剩 100
    get_file_size("alice_small_copy.txt")                   -> 100
    add_file_by("alice", "alice_c.txt", 1)                  -> None    # alice 现在剩 0 字节
    add_user("bob", 50)                                     -> True
    add_file_by("bob", "bob_notes.txt", 20)                 -> 30
    merge_user("alice", "bob")                              -> 30      # alice：容量 500+50，已用 500+20
    add_file_by("bob", "bob_notes2.txt", 5)                 -> None    # "bob" 已经不是用户了
    merge_user("alice", "bob")                              -> None    # source "bob" 已经不存在
    get_file_size("bob_notes.txt")                          -> 20      # 文件本身还在，现在归 alice 所有
    merge_user("alice", "alice")                            -> None    # target 等于 source
    merge_user("dave", "alice")                             -> None    # "dave" 不是用户
"""

EXAMPLE_L4 = """
    add_user("erin", 100)                      -> True
    add_file_by("erin", "photo.raw", 41)       -> 59
    backup_user("erin")                        -> True
    compress_file("erin", "photo.raw")         -> "photo.raw.cmp"   # 大小：ceil(41 / 2) = 21
    get_file_size("photo.raw")                 -> None    # 被压缩操作释放了
    get_file_size("photo.raw.cmp")             -> 21
    add_file_by("erin", "logo.png", 30)        -> 49      # 剩余：100 - 21 - 30
    restore_user("erin")                       -> 59      # 回到 {photo.raw: 41}，容量 100
    get_file_size("photo.raw")                 -> 41      # 被恢复操作重新创建
    get_file_size("photo.raw.cmp")             -> None    # 不在快照里，所以被删除
    get_file_size("logo.png")                  -> None    # 备份之后才添加，所以被删除
    add_user("frank", 60)                      -> True
    add_file_by("frank", "video.raw", 25)      -> 35
    compress_file("frank", "video.raw")        -> "video.raw.cmp"   # 大小：ceil(25 / 2) = 13
    decompress_file("frank", "video.raw.cmp")  -> "video.raw"       # 大小：13 * 2 = 26
    get_file_size("video.raw")                 -> 26
    add_user("gina", 25)                       -> True
    add_file_by("gina", "clip.raw", 25)        -> 0
    compress_file("gina", "clip.raw")          -> "clip.raw.cmp"    # 大小 13，剩余 12
    decompress_file("gina", "clip.raw.cmp")    -> None    # 还需要 13 字节的容量，但只剩 12
"""


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, EXAMPLE_L1)


@case("Level 1", title="失败的调用不改变任何东西")
def l1_failures_change_nothing(sol):
    run(sol, """
        add_file("a.txt", 5)            -> True
        add_file("a.txt", 9)            -> False   # 已经存在
        get_file_size("a.txt")          -> 5       # 失败的 add_file 不能改掉原来的大小
        add_file("b.txt", 7)            -> True
        copy_file("a.txt", "b.txt")     -> False   # destination 已经存在
        get_file_size("b.txt")          -> 7       # 失败的 copy_file 不能覆盖 destination
        copy_file("zzz", "c.txt")       -> False   # source 不存在
        get_file_size("c.txt")          -> None    # 失败的复制不能创建 destination
        get_file_size("zzz")            -> None
        add_file("c.txt", 1)            -> True
    """)


@case("Level 1", title="复制自己、复制副本")
def l1_copies(sol):
    run(sol, """
        add_file("f", 3)                -> True
        copy_file("f", "f")             -> False   # destination（就是 f 本身）已经存在
        copy_file("f", "g")             -> True
        copy_file("g", "h")             -> True    # 副本也是普通文件，可以再复制
        get_file_size("f")              -> 3       # source 不受影响
        get_file_size("h")              -> 3
        add_file("g", 1)                -> False   # 复制出的 g 占用了这个名称
        copy_file("h", "f")             -> False
        get_file_size("g")              -> 3
    """)


@case("Level 1", title="名称区分大小写、大文件")
def l1_names_and_sizes(sol):
    run(sol, """
        add_file("Notes.txt", 10)              -> True
        add_file("notes.txt", 20)              -> True    # 和 "Notes.txt" 是两个不同的文件
        get_file_size("Notes.txt")             -> 10
        get_file_size("notes.txt")             -> 20
        get_file_size("NOTES.TXT")             -> None
        add_file("a-b_c.9", 1)                 -> True
        copy_file("a-b_c.9", "A-B_C.9")        -> True
        get_file_size("A-B_C.9")               -> 1
        add_file("big", 1000000000000)         -> True
        copy_file("big", "big2")               -> True
        get_file_size("big2")                  -> 1000000000000
    """)


@case("Level 1", cpu=2, title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=300, length=(5, 30))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子（接着第 1 关）")
def l2_example(sol):
    run(sol, EXAMPLE_L1, EXAMPLE_L2)


@case("Level 2", title="并列按名称升序，与添加顺序无关")
def l2_ties_by_name(sol):
    run(sol, """
        add_file("c.log", 10)       -> True
        add_file("a.log", 10)       -> True
        add_file("b.log", 30)       -> True
        add_file("d.log", 10)       -> True
        add_file("e.log", 5)        -> True
        find_files("", ".log")      -> ["b.log(30)", "a.log(10)", "c.log(10)", "d.log(10)", "e.log(5)"]   # 大小相同的按名称升序，不按添加顺序
        copy_file("e.log", "0.log") -> True
        find_files("", ".log")      -> ["b.log(30)", "a.log(10)", "c.log(10)", "d.log(10)", "0.log(5)", "e.log(5)"]   # 副本也会被找到；"0" 排在 "e" 前面
    """)


@case("Level 2", title="大小按数值比较；名称按 ASCII")
def l2_ordering(sol):
    run(sol, """
        add_file("z9", 9)           -> True
        add_file("y100", 100)       -> True
        add_file("x10", 10)         -> True
        find_files("", "")          -> ["y100(100)", "x10(10)", "z9(9)"]     # 按大小的数值降序，不是按字符串
        add_file("a.", 4)           -> True
        add_file("_x", 4)           -> True
        add_file("a", 4)            -> True
        add_file("B", 4)            -> True
        add_file("-y", 4)           -> True
        add_file("9", 4)            -> True
        add_file("a-", 4)           -> True
        find_files("", "")          -> ["y100(100)", "x10(10)", "z9(9)", "-y(4)", "9(4)", "B(4)", "_x(4)", "a(4)", "a-(4)", "a.(4)"]   # 名称按 ASCII 升序："-" < "." < 数字 < 大写 < "_" < 小写
    """)


@case("Level 2", title="前缀和后缀可以重叠")
def l2_overlap(sol):
    run(sol, """
        add_file("ab", 1)           -> True
        add_file("aba", 2)          -> True
        add_file("abab", 3)         -> True
        add_file("a", 4)            -> True
        find_files("ab", "ab")      -> ["abab(3)", "ab(1)"]   # "ab" 同时以 "ab" 开头和结尾；"aba" 不以 "ab" 结尾
        find_files("ab", "ba")      -> ["aba(2)"]
        find_files("a", "a")        -> ["a(4)", "aba(2)"]     # 只有一个字符的 "a" 也同时满足两侧
        find_files("aba", "aba")    -> ["aba(2)"]
        find_files("abab", "b")     -> ["abab(3)"]
    """)


@case("Level 2", title="空前缀、空后缀、没有匹配")
def l2_empty_sides(sol):
    run(sol, """
        find_files("", "")          -> []      # 存储为空
        add_file("x.py", 3)         -> True
        add_file("y.py", 3)         -> True
        add_file("z.txt", 8)        -> True
        find_files("", "")          -> ["z.txt(8)", "x.py(3)", "y.py(3)"]   # 两侧都为空：匹配所有文件
        find_files("w", "")         -> []
        find_files("", ".md")       -> []
        find_files("x.py", "")      -> ["x.py(3)"]     # 前缀等于整个名称
        find_files("", "x.py")      -> ["x.py(3)"]
        find_files("x.py.", "")     -> []              # 前缀比名称还长
    """)


@case("Level 2", title="字面匹配，不是通配符或正则")
def l2_literal(sol):
    run(sol, """
        add_file("a.b", 1)          -> True
        add_file("axb", 2)          -> True
        add_file("a.b.c", 3)        -> True
        find_files("a.", "")        -> ["a.b.c(3)", "a.b(1)"]   # "." 就是字符 "."，"axb" 不匹配
        find_files("", ".b")        -> ["a.b(1)"]
        find_files("", "b")         -> ["axb(2)", "a.b(1)"]
        find_files("A", "")         -> []                       # 区分大小写
    """)


@case("Level 2", cpu=2, title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=300, length=(5, 30))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子（接着前两关）")
def l3_example(sol):
    run(sol, EXAMPLE_L1, EXAMPLE_L2, EXAMPLE_L3)


@case("Level 3", title="重复 add_user；容量恰好用完")
def l3_users_and_capacity(sol):
    run(sol, """
        add_user("u", 10)            -> True
        add_user("u", 100)           -> False   # "u" 已经是用户：容量仍是 10
        add_file_by("u", "f1", 11)   -> None    # 11 > 10
        add_file_by("u", "f1", 4)    -> 6
        add_file_by("u", "f2", 6)    -> 0       # size 等于剩余容量是允许的
        add_file_by("u", "f3", 1)    -> None    # 剩余 0
        get_file_size("f3")          -> None    # 失败的调用不能创建文件
        add_user("v", 1)             -> True
        add_file_by("v", "f1", 1)    -> None    # "f1" 已经存在（属于 u）
        add_file_by("v", "g", 1)     -> 0
    """)


@case("Level 3", title="失败的 add_file_by 不占用容量")
def l3_failed_add_by(sol):
    run(sol, """
        add_user("u", 20)              -> True
        add_file("taken", 3)           -> True
        add_file_by("u", "taken", 5)   -> None   # 名称已存在（无主文件也算）
        add_file_by("x", "new", 5)     -> None   # "x" 不是用户
        get_file_size("new")           -> None
        add_file_by("u", "big", 21)    -> None   # 21 > 20
        get_file_size("big")           -> None
        add_file_by("u", "a", 20)      -> 0      # 上面几次失败都没有占用 u 的容量
        get_file_size("taken")         -> 3      # 无主文件的大小也没变
    """)


@case("Level 3", title="无主文件不计入任何人的容量")
def l3_unowned(sol):
    run(sol, """
        add_user("u", 5)               -> True
        add_file("big", 100)           -> True     # add_file 创建的文件没有主人
        copy_file("big", "big2")       -> True     # source 没有主人：不检查容量
        get_file_size("big2")          -> 100
        add_file_by("u", "mine", 5)    -> 0        # u 的容量完全没被占用
        copy_file("big2", "big3")      -> True     # 无主文件的副本也没有主人
        get_file_size("big3")          -> 100
    """)


@case("Level 3", title="副本归 source 的主人并计入容量")
def l3_copy_owned(sol):
    run(sol, """
        add_user("u", 10)               -> True
        add_file_by("u", "f", 4)        -> 6
        copy_file("f", "g")             -> True    # 需要 4，u 剩 6
        copy_file("g", "h")             -> False   # g 也属于 u：还需要 4，但 u 只剩 2
        get_file_size("h")              -> None    # 失败的复制不创建 destination
        add_file_by("u", "x", 3)        -> None    # 副本 g 计入了 u：10-4-4 = 2
        add_file_by("u", "x", 2)        -> 0
        add_user("v", 100)              -> True
        merge_user("v", "u")            -> 100     # 容量 100+10，u 的三个文件 4+4+2 都转给 v
        copy_file("g", "h")             -> True    # g 现在属于 v
        add_file_by("v", "y", 97)       -> None    # 副本 h 计入了 v：110-10-4 = 96
        add_file_by("v", "y", 96)       -> 0
    """)


@case("Level 3", title="合并后文件归 target，旧 id 可重新注册")
def l3_merge_moves_files(sol):
    run(sol, """
        add_user("a", 30)              -> True
        add_user("b", 20)              -> True
        add_file_by("a", "fa", 10)     -> 20
        add_file_by("b", "fb", 15)     -> 5
        merge_user("a", "b")           -> 25     # 容量 30+20，已用 10+15
        add_file_by("b", "x", 1)       -> None   # "b" 已经不是用户
        add_user("b", 8)               -> True   # "b" 空出来了：全新的账户，没有文件
        add_file_by("b", "fb2", 8)     -> 0      # 新 b 的容量里不计旧 b 的文件
        copy_file("fb", "fb_copy")     -> True   # fb 现在属于 a（剩 25），不属于新的 b
        add_file_by("a", "y", 11)      -> None   # a 剩 25-15 = 10
        add_file_by("a", "y", 10)      -> 0
    """)


@case("Level 3", title="合并失败不改变任何东西")
def l3_merge_rejected(sol):
    run(sol, """
        add_user("a", 10)              -> True
        add_user("b", 10)              -> True
        add_file_by("b", "fb", 4)      -> 6
        merge_user("a", "a")           -> None   # target 等于 source
        merge_user("a", "zz")          -> None   # "zz" 不是用户
        merge_user("zz", "b")          -> None   # target 不是用户：b 不能因此被删除
        add_file_by("b", "g", 7)       -> None   # b 仍然是用户，剩 6
        add_file_by("b", "g", 6)       -> 0
        add_file_by("a", "h", 11)      -> None   # a 的容量也没变
        add_file_by("a", "h", 10)      -> 0
        merge_user("b", "a")           -> 0
    """)


@case("Level 3", title="用户 ID 与文件名互不冲突")
def l3_namespaces(sol):
    run(sol, """
        add_user("a", 5)               -> True
        add_file("a", 3)               -> True    # 同一个字符串可以同时是用户 ID 和文件名
        add_file_by("a", "b", 5)       -> 0
        add_user("b", 7)               -> True    # 文件 "b" 存在不影响用户 "b"
        merge_user("b", "a")           -> 7       # 容量 7+5，已用 5
        get_file_size("a")             -> 3       # 删除用户 "a" 不影响文件 "a"
        add_user("a", 1)               -> True
    """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=400, length=(10, 40))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子（接着前三关）")
def l4_example(sol):
    run(sol, EXAMPLE_L1, EXAMPLE_L2, EXAMPLE_L3, EXAMPLE_L4)


@case("Level 4", title="压缩：向上取整，可以重复压缩")
def l4_compress_sizes(sol):
    run(sol, """
        add_user("u", 100)               -> True
        add_file_by("u", "one", 1)       -> 99
        compress_file("u", "one")        -> "one.cmp"
        get_file_size("one.cmp")         -> 1                # ceil(1 / 2) = 1，不会变成 0
        add_file_by("u", "nine", 9)      -> 90
        compress_file("u", "nine")       -> "nine.cmp"
        get_file_size("nine.cmp")        -> 5                # ceil(9 / 2)
        compress_file("u", "nine.cmp")   -> "nine.cmp.cmp"   # 已经以 .cmp 结尾也可以再压缩
        get_file_size("nine.cmp.cmp")    -> 3
        add_file_by("u", "ten", 10)      -> 86               # 已用 1+3+10
        compress_file("u", "ten")        -> "ten.cmp"
        get_file_size("ten.cmp")         -> 5
        add_file_by("u", "rest", 92)     -> None             # 剩余 100-1-3-5 = 91
        add_file_by("u", "rest", 91)     -> 0
    """)


@case("Level 4", title="压缩释放原名称")
def l4_compress_frees_name(sol):
    run(sol, """
        add_user("u", 50)                -> True
        add_file_by("u", "doc", 20)      -> 30
        compress_file("u", "doc")        -> "doc.cmp"
        get_file_size("doc")             -> None   # 原名称被删除
        add_file("doc", 7)               -> True   # 名称空出来了
        compress_file("u", "doc")        -> None   # 新的 "doc" 没有主人
        add_file_by("u", "doc2", 41)     -> None   # 压缩后 u 只用了 10
        add_file_by("u", "doc2", 40)     -> 0
    """)


@case("Level 4", title="压缩失败时不改变任何东西")
def l4_compress_rejected(sol):
    run(sol, """
        add_user("u", 100)               -> True
        add_user("v", 100)               -> True
        add_file_by("u", "a", 10)        -> 90
        add_file_by("v", "b", 10)        -> 90
        add_file("c", 10)                -> True
        compress_file("zz", "a")         -> None   # "zz" 不是用户
        compress_file("u", "nope")       -> None   # 文件不存在
        compress_file("u", "b")          -> None   # b 属于 v
        compress_file("u", "c")          -> None   # c 没有主人
        add_file("a.cmp", 3)             -> True
        compress_file("u", "a")          -> None   # "a.cmp" 已经存在
        get_file_size("a")               -> 10     # 失败的压缩什么都不改
        get_file_size("a.cmp")           -> 3
        compress_file("v", "b")          -> "b.cmp"
        compress_file("v", "b.cmp")      -> "b.cmp.cmp"
        add_file_by("u", "x", 91)        -> None   # u 的已用量仍是 10
        add_file_by("u", "x", 90)        -> 0
    """)


@case("Level 4", title="解压：精确翻倍，只去掉末尾一个 .cmp")
def l4_decompress(sol):
    run(sol, """
        add_user("u", 100)                   -> True
        add_file_by("u", "k", 7)             -> 93
        compress_file("u", "k")              -> "k.cmp"         # 大小 4
        compress_file("u", "k.cmp")          -> "k.cmp.cmp"     # 大小 2
        decompress_file("u", "k.cmp.cmp")    -> "k.cmp"         # 只去掉末尾的一个 ".cmp"
        get_file_size("k.cmp")               -> 4
        decompress_file("u", "k.cmp")        -> "k"
        get_file_size("k")                   -> 8               # 7 -> 4 -> 2 -> 4 -> 8：翻倍不做取整
        get_file_size("k.cmp")               -> None
        add_file_by("u", "z.cmp", 3)         -> 89              # 直接创建的 .cmp 文件也可以解压
        decompress_file("u", "z.cmp")        -> "z"
        get_file_size("z")                   -> 6
        add_file_by("u", "w", 87)            -> None            # 剩余 100-8-6 = 86
        add_file_by("u", "w", 86)            -> 0
    """)


@case("Level 4", title="解压失败时不改变任何东西")
def l4_decompress_rejected(sol):
    run(sol, """
        add_user("u", 100)                -> True
        add_user("v", 100)                -> True
        add_file_by("u", "p", 10)         -> 90
        decompress_file("u", "p")         -> None   # 不以 ".cmp" 结尾
        decompress_file("u", "q.cmp")     -> None   # 不存在
        compress_file("u", "p")           -> "p.cmp"
        decompress_file("v", "p.cmp")     -> None   # 属于 u，不属于 v
        decompress_file("zz", "p.cmp")    -> None   # "zz" 不是用户
        add_file("p", 1)                  -> True
        decompress_file("u", "p.cmp")     -> None   # "p" 已经存在
        get_file_size("p.cmp")            -> 5      # 失败的解压什么都不改
        get_file_size("p")                -> 1
        add_file("x.cmp", 4)              -> True
        decompress_file("u", "x.cmp")     -> None   # 没有主人的文件
        add_file_by("u", "y", 96)         -> None   # u 只用了 5
        add_file_by("u", "y", 95)         -> 0
    """)


@case("Level 4", title="名称中间的 .cmp 不算后缀")
def l4_decompress_inner_suffix(sol):
    run(sol, """
        add_user("u", 100)                   -> True
        add_file_by("u", "a.cmp.txt", 4)     -> 96
        decompress_file("u", "a.cmp.txt")    -> None             # 只是中间含有 ".cmp"，不以 ".cmp" 结尾
        get_file_size("a.cmp.txt")           -> 4
        get_file_size("a.txt")               -> None
        add_file_by("u", "b.cmpx", 2)        -> 94
        decompress_file("u", "b.cmpx")       -> None             # 不以 ".cmp" 结尾
        add_file_by("u", "ccmp", 2)          -> 92
        decompress_file("u", "ccmp")         -> None             # 以 "cmp" 结尾，但不以 ".cmp" 结尾
        compress_file("u", "a.cmp.txt")      -> "a.cmp.txt.cmp"  # 大小 2
        decompress_file("u", "a.cmp.txt.cmp")  -> "a.cmp.txt"    # 只去掉末尾的 ".cmp"，中间的保留
        get_file_size("a.cmp.txt")           -> 4
        get_file_size("a.txt.cmp")           -> None
        add_file_by("u", "z", 93)            -> None             # 剩余 100-4-2-2 = 92
        add_file_by("u", "z", 92)            -> 0
    """)


@case("Level 4", title="解压的容量检查用当前大小")
def l4_decompress_capacity(sol):
    run(sol, """
        add_user("u", 12)                  -> True
        add_file_by("u", "f", 10)          -> 2
        compress_file("u", "f")            -> "f.cmp"     # 大小 5，剩余 7
        decompress_file("u", "f.cmp")      -> "f"         # 比较的是当前大小 5 <= 7，不是翻倍后的 10
        get_file_size("f")                 -> 10
        add_file_by("u", "g", 3)           -> None        # 剩余 12-10 = 2
        add_user("v", 9)                   -> True
        add_file_by("v", "h", 8)           -> 1
        compress_file("v", "h")            -> "h.cmp"     # 大小 4，剩余 5
        add_file_by("v", "pad", 1)         -> 4           # 剩余 4，正好等于 h.cmp 的大小
        decompress_file("v", "h.cmp")      -> "h"         # 4 <= 4：允许
        add_file_by("v", "x", 1)           -> None        # 剩余 9-8-1 = 0
        add_user("w", 7)                   -> True
        add_file_by("w", "j", 7)           -> 0
        compress_file("w", "j")            -> "j.cmp"     # 大小 4，剩余 3
        decompress_file("w", "j.cmp")      -> None        # 4 > 3
        get_file_size("j.cmp")             -> 4
    """)


@case("Level 4", title="恢复：删掉多的、补回少的")
def l4_restore_basic(sol):
    run(sol, """
        add_user("u", 50)                -> True
        add_file_by("u", "a", 10)        -> 40
        add_file_by("u", "b", 5)         -> 35
        restore_user("u")                -> None   # 从未备份过
        backup_user("u")                 -> True   # 快照：{a: 10, b: 5}，容量 50
        backup_user("zz")                -> False  # 不是用户
        restore_user("zz")               -> None
        add_file_by("u", "c", 20)        -> 15
        compress_file("u", "a")          -> "a.cmp"
        restore_user("u")                -> 35     # 删除 c 和 a.cmp，按备份的大小重建 a
        get_file_size("a")               -> 10
        get_file_size("c")               -> None
        get_file_size("a.cmp")           -> None
        get_file_size("b")               -> 5
        add_file("c", 1)                 -> True   # 被删掉的名称空出来了
        restore_user("u")                -> 35     # 同一份备份可以再恢复一次
        add_file_by("u", "d", 1)         -> 34
        backup_user("u")                 -> True   # 新的备份替换旧的
        add_file_by("u", "e", 4)         -> 30
        restore_user("u")                -> 34     # 回到第二份备份：e 被删除，d 保留
        get_file_size("d")               -> 1
        get_file_size("e")               -> None
    """)


@case("Level 4", title="两次都拥有的名称原样保留")
def l4_restore_keeps_current(sol):
    run(sol, """
        add_user("z", 20)                -> True
        add_file_by("z", "doc", 10)      -> 10
        backup_user("z")                 -> True     # 快照：{doc: 10}，容量 20
        compress_file("z", "doc")        -> "doc.cmp"
        add_file_by("z", "doc", 3)       -> 12       # 一个不相关的新文件用了空出来的名称
        restore_user("z")                -> 17       # doc 在备份时和现在都属于 z：原样保留（大小 3）；doc.cmp 被删除
        get_file_size("doc")             -> 3        # 只比较名称，不比较大小
        get_file_size("doc.cmp")         -> None
        add_file_by("z", "x", 18)        -> None
        add_file_by("z", "x", 17)        -> 0
    """)


@case("Level 4", title="恢复冲突时不改变任何东西")
def l4_restore_collision(sol):
    run(sol, """
        add_user("w", 50)                -> True
        add_user("y", 50)                -> True
        add_file_by("w", "f1", 10)       -> 40
        add_file_by("w", "f2", 6)        -> 34
        backup_user("w")                 -> True    # 快照：{f1: 10, f2: 6}，容量 50
        add_file_by("w", "extra", 2)     -> 32
        compress_file("w", "f1")         -> "f1.cmp"    # 名称 f1 空出来，w 现在有 f1.cmp(5)
        compress_file("w", "f2")         -> "f2.cmp"    # f2.cmp(3)
        add_file_by("y", "f1", 3)        -> 47      # 别的用户占用了 f1
        restore_user("w")                -> None    # 要重建的 f1 会和 y 的文件冲突
        get_file_size("extra")           -> 2       # 失败的恢复不能删掉任何东西
        get_file_size("f1.cmp")          -> 5
        get_file_size("f2")              -> None    # 也不能重建任何东西
        get_file_size("f2.cmp")          -> 3
        add_file_by("w", "big", 41)      -> None    # w 剩 50-2-5-3 = 40
        add_file_by("w", "big", 40)      -> 0
    """)


@case("Level 4", title="无主文件占用名称也算冲突")
def l4_restore_collision_unowned(sol):
    run(sol, """
        add_user("w", 50)                -> True
        add_file_by("w", "f", 10)        -> 40
        add_file_by("w", "g", 4)         -> 36
        backup_user("w")                 -> True    # 快照：{f: 10, g: 4}，容量 50
        compress_file("w", "f")          -> "f.cmp"     # 名称 f 空出来，w 现在有 f.cmp(5)
        add_file("f", 7)                 -> True    # 一个没有主人的文件占用了 f
        restore_user("w")                -> None    # 要重建的 f 会和这个无主文件冲突
        get_file_size("f")               -> 7       # 无主文件不能被覆盖
        get_file_size("f.cmp")           -> 5       # 失败的恢复不能删掉任何东西
        get_file_size("g")               -> 4
        add_file_by("w", "big", 42)      -> None    # w 剩 50-5-4 = 41
        add_file_by("w", "big", 41)      -> 0
    """)


@case("Level 4", title="被合并的账户的备份随之失效")
def l4_merge_and_backup(sol):
    run(sol, """
        add_user("w", 50)                -> True
        add_user("x", 10)                -> True
        add_file_by("x", "old", 4)       -> 6
        backup_user("x")                 -> True    # x 的备份：{old: 4}，容量 10
        backup_user("w")                 -> True    # w 的备份：没有文件，容量 50
        merge_user("w", "x")             -> 56      # old 转给 w；账户 x 被删除
        restore_user("x")                -> None    # "x" 不是用户
        compress_file("w", "old")        -> "old.cmp"   # old 属于 w；名称 old 空出来
        add_user("x", 20)                -> True    # 全新的账户
        restore_user("x")                -> None    # 新账户从没有备份过：旧 x 的备份不属于它
        get_file_size("old")             -> None
        add_file_by("x", "n", 20)        -> 0       # 新的 x：容量 20，没有文件
        restore_user("w")                -> 50      # w 自己的备份仍然有效：old.cmp 不在里面，被删除；容量回到 50
        get_file_size("old.cmp")         -> None
    """)


@case("Level 4", title="恢复只影响这个用户；容量回到备份值")
def l4_restore_scope(sol):
    run(sol, """
        add_user("u", 30)                -> True
        add_user("v", 30)                -> True
        add_user("t", 10)                -> True
        add_file_by("u", "uf", 5)        -> 25
        backup_user("u")                 -> True     # 快照：{uf: 5}，容量 30
        add_file_by("v", "vf", 7)        -> 23
        add_file_by("t", "tf", 3)        -> 7
        add_file("free", 9)              -> True
        merge_user("u", "v")             -> 48       # 容量 60，已用 5+7
        restore_user("u")                -> 25       # vf 现在属于 u 但不在备份里：被删除；容量回到 30
        get_file_size("vf")              -> None
        get_file_size("uf")              -> 5
        get_file_size("free")            -> 9        # 无主文件不受影响
        get_file_size("tf")              -> 3        # 别的用户的文件不受影响
        add_file_by("u", "x", 26)        -> None
        add_file_by("u", "x", 25)        -> 0
    """)


@case("Level 4", title="恢复不检查容量，剩余可为负")
def l4_restore_negative(sol):
    run(sol, """
        add_user("u", 20)                -> True
        add_file_by("u", "a", 2)         -> 18
        add_file_by("u", "b", 18)        -> 0
        backup_user("u")                 -> True    # 快照：{a: 2, b: 18}，容量 20
        compress_file("u", "b")          -> "b.cmp"   # 大小 9
        compress_file("u", "a")          -> "a.cmp"   # 大小 1
        add_file_by("u", "a", 10)        -> 0       # 新的 "a"：20-9-1-10
        restore_user("u")                -> -8      # 删除 a.cmp、b.cmp，重建 b(18)，保留现在的 a(10)；题目没有给恢复设容量条件
        add_file_by("u", "c", 1)         -> None    # 剩余是负数
        compress_file("u", "b")          -> "b.cmp"   # 已用 10+9
        add_file_by("u", "c", 1)         -> 0
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=500, length=(15, 60))
