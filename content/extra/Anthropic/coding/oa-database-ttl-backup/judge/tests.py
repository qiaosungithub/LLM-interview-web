import ast
import time

from judgelib import calling, case, expect, expect_true, rng
import ref

SHOW = 18           # a replay lists at most this many calls


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    return repr(x)


def _code(method, args, kwargs):
    parts = [_lit(a) for a in args] + [f"{k}={_lit(v)}" for k, v in kwargs.items()]
    return f"db.{method}({', '.join(parts)})"


def _replay(done, wants, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step."""
    lines = [_code(*step) for step in done]
    lines[-1] = f"print({lines[-1]})"
    notes = [f"# -> {_lit(w)}" for w in wants] + [mark]
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = ["db = InMemoryDatabase()"]
    if skipped:
        head.append(f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）")
    return "\n".join(head + [s.ljust(width) + n for s, n in zip(lines, notes)])


def _method(db, name):
    fn = getattr(db, name, None)
    expect_true(callable(fn), f"InMemoryDatabase 没有 {name} 方法（方法名要和题目一致）")
    return fn


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


def _message(got, want, why):
    if type(got) is not type(want):
        hint = f"所有方法都返回字符串：期望 {_lit(want)}，得到 {type(got).__name__} 类型的 {got!r}"
        return f"{why}；{hint}" if why else hint
    return why or "返回值不对"


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines like  set_at("u1", "a", "1", timestamp=3) -> ""   # why  become (method, args, kwargs, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        call, arrow, rest = line.strip().partition("->")
        assert arrow, f"missing '->' in {line!r}"
        expr, _, why = rest.partition("#")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(ast.literal_eval(a) for a in node.args)
        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        steps.append((node.func.id, args, kwargs, ast.literal_eval(expr.strip()), why.strip()))
    return steps


def run(sol, script):
    """Runs the calls on a fresh InMemoryDatabase, checking each return value as soon as it comes back."""
    db = sol.InMemoryDatabase()
    done, wants = [], []
    for method, args, kwargs, want, why in _parse(script):
        done.append((method, args, kwargs))
        fn = _method(db, method)
        full = len(done)                    # a hand-written script is short: show all of it, so it can be pasted
        calling(_replay(done, wants, show=full))       # shown if the answer raises or hangs in this call
        try:
            got = fn(*args, **kwargs)
        except Exception:
            calling(_replay(done, wants, mark="# <- 这一步抛出异常", show=full))
            raise
        if not _same(got, want):
            expect(got, want, call=_replay(done, wants, show=full),
                   msg=_message(got, want, why or f"第 {len(done)} 步 {method} 的返回值不对"))
        wants.append(want)


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, ops):
    """Replays ops on a fresh answer and a fresh ref model; None if every return value agrees."""
    db, model, wants = sol.InMemoryDatabase(), ref.InMemoryDatabase(), []
    for k, (method, args, kwargs) in enumerate(ops):
        want = getattr(model, method)(*args, **kwargs)
        fn = _method(db, method)
        try:
            got = fn(*args, **kwargs)
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        wants.append(want)
    return None


def _valid(ops):
    """Whether a shortened sequence still keeps the statement's promise that every restore has a backup."""
    made = []
    for method, args, kwargs in ops:
        if method == "backup":
            made.append(args[0])
        elif method == "restore" and not any(t <= args[1] for t in made):
            return False
    return True


def _shrink(sol, ops, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, cand) if cand and _valid(cand) else None
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
        expect(bad["got"], bad["want"], call=_replay(ops[: k + 1], bad["wants"]),
               msg=_message(bad["got"], bad["want"],
                            f"随机操作序列的第 {k + 1} 步返回值不对（已自动删去与出错无关的调用）"))


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, """
        set("u1", "name", "Ann")                        -> ""
        set("u1", "city", "Rome")                       -> ""
        get("u1", "name")                               -> "Ann"
        get("u1", "age")                                -> ""        # 没有这个字段
        get("u2", "name")                               -> ""        # 没有这条记录
        compare_and_set("u1", "name", "Bob", "Cat")     -> "false"   # 当前值是 "Ann"，不是 "Bob"
        compare_and_set("u1", "name", "Ann", "Cat")     -> "true"
        get("u1", "name")                               -> "Cat"
        compare_and_set("u1", "age", "30", "31")        -> "false"   # 字段不存在；条件写入不会创建字段
        compare_and_delete("u1", "city", "Paris")       -> "false"
        compare_and_delete("u1", "city", "Rome")        -> "true"
        get("u1", "city")                               -> ""
        set("u1", "name", "Dan")                        -> ""        # 覆盖
        delete("u1", "name")                            -> "true"
        delete("u1", "name")                            -> "false"   # 已经删掉了
        delete("u9", "name")                            -> "false"
    """)


@case("Level 1", title="覆盖；不同记录互不影响")
def l1_records(sol):
    run(sol, """
        set("a", "f", "1")                  -> ""
        set("b", "f", "2")                  -> ""        # 同一个字段名，在另一条记录里
        set("a", "g", "3")                  -> ""
        get("a", "f")                       -> "1"
        get("b", "f")                       -> "2"
        get("b", "g")                       -> ""        # g 只在记录 a 里
        set("a", "f", "4")                  -> ""        # 覆盖
        get("a", "f")                       -> "4"
        get("b", "f")                       -> "2"
        delete("b", "f")                    -> "true"
        get("a", "f")                       -> "4"       # 删 b 的 f 不影响 a 的 f
        get("b", "f")                       -> ""
        set("b", "f", "5")                  -> ""        # 删掉之后可以再写
        get("b", "f")                       -> "5"
    """)


@case("Level 1", title="delete：只有字段存在才返回 \"true\"")
def l1_delete(sol):
    run(sol, """
        delete("a", "f")                    -> "false"   # 记录不存在
        set("a", "f", "1")                  -> ""
        delete("a", "g")                    -> "false"   # 记录存在，字段不存在
        get("a", "f")                       -> "1"       # 失败的删除什么也没改
        delete("a", "f")                    -> "true"
        delete("a", "f")                    -> "false"
        get("a", "f")                       -> ""
        delete("a", "F")                    -> "false"   # 字段名区分大小写
        set("a", "x", "1")                  -> ""
        set("a", "y", "2")                  -> ""
        delete("a", "x")                    -> "true"
        get("a", "y")                       -> "2"       # 只删这一个字段，不删整条记录
    """)


@case("Level 1", title="compare_and_set：不匹配、不存在都不写")
def l1_compare_and_set(sol):
    run(sol, """
        compare_and_set("a", "f", "1", "2")     -> "false"   # 记录不存在
        get("a", "f")                           -> ""        # 也没有被创建
        set("a", "g", "1")                      -> ""
        compare_and_set("a", "f", "1", "2")     -> "false"   # 记录存在、字段不存在：不会创建字段
        get("a", "f")                           -> ""
        compare_and_set("a", "g", "x", "2")     -> "false"   # 值不相等
        compare_and_set("a", "g", "1 ", "2")    -> "false"   # 必须完全相等
        get("a", "g")                           -> "1"
        compare_and_set("a", "g", "1", "2")     -> "true"
        compare_and_set("a", "g", "1", "3")     -> "false"   # 现在是 "2"
        compare_and_set("a", "g", "2", "2")     -> "true"    # 新值和旧值相同也算成功
        get("a", "g")                           -> "2"
        set("b", "h", "Ann")                    -> ""
        compare_and_set("b", "h", "ann", "x")   -> "false"   # 区分大小写
        compare_and_set("a", "h", "Ann", "x")   -> "false"   # 字段在另一条记录里
        get("b", "h")                           -> "Ann"
    """)


@case("Level 1", title="compare_and_delete")
def l1_compare_and_delete(sol):
    run(sol, """
        compare_and_delete("a", "f", "1")       -> "false"   # 记录不存在
        set("a", "f", "1")                      -> ""
        set("a", "g", "1")                      -> ""
        compare_and_delete("a", "f", "2")       -> "false"
        get("a", "f")                           -> "1"
        compare_and_delete("a", "h", "1")       -> "false"   # 字段不存在
        compare_and_delete("a", "f", "1")       -> "true"
        get("a", "f")                           -> ""
        get("a", "g")                           -> "1"       # 只删这一个字段
        compare_and_delete("a", "f", "1")       -> "false"   # 已经删掉了
        delete("a", "f")                        -> "false"
        set("a", "f", "3")                      -> ""
        compare_and_delete("a", "f", "3")       -> "true"
    """)


@case("Level 1", title="区分大小写、中文字符串")
def l1_case_and_unicode(sol):
    run(sol, """
        set("u", "name", "x")               -> ""
        set("U", "name", "y")               -> ""        # "u" 和 "U" 是不同的记录
        set("u", "Name", "z")               -> ""        # "name" 和 "Name" 是不同的字段
        get("u", "name")                    -> "x"
        get("U", "name")                    -> "y"
        get("u", "Name")                    -> "z"
        get("U", "Name")                    -> ""
        set("用户1", "名字", "张三")        -> ""
        get("用户1", "名字")                -> "张三"
        compare_and_set("用户1", "名字", "张三", "李四")   -> "true"
        get("用户1", "名字")                -> "李四"
        compare_and_delete("用户1", "名字", "李四")        -> "true"
        get("用户1", "名字")                -> ""
    """)


@case("Level 1", title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=300, length=(5, 30))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, """
        set("u1", "name", "Ann")            -> ""
        set("u1", "nick", "an")             -> ""
        set("u1", "age", "30")              -> ""
        set("u1", "Zip", "00100")           -> ""
        scan("u1")                          -> "Zip(00100), age(30), name(Ann), nick(an)"   # 大写字母排在小写前面
        scan_by_prefix("u1", "n")           -> "name(Ann), nick(an)"
        scan_by_prefix("u1", "na")          -> "name(Ann)"
        scan_by_prefix("u1", "name")        -> "name(Ann)"    # 字段名本身也算以它开头
        scan_by_prefix("u1", "z")           -> ""             # 区分大小写：没有以小写 z 开头的字段
        scan("u2")                          -> ""
        delete("u1", "age")                 -> "true"
        scan("u1")                          -> "Zip(00100), name(Ann), nick(an)"
    """)


@case("Level 2", title="按普通字符串比较排序")
def l2_sort_order(sol):
    run(sol, """
        set("k", "b", "1")                  -> ""
        set("k", "ab", "2")                 -> ""
        set("k", "a", "3")                  -> ""
        set("k", "aB", "4")                 -> ""
        set("k", "A", "5")                  -> ""
        set("k", "1x", "6")                 -> ""
        set("k", "a1", "7")                 -> ""
        set("k", "名", "8")                 -> ""
        scan("k")                           -> "1x(6), A(5), a(3), a1(7), aB(4), ab(2), b(1), 名(8)"   # 数字 < 大写 < 小写；"a" 是 "a1" 的前缀，排在前面
        scan_by_prefix("k", "a")            -> "a(3), a1(7), aB(4), ab(2)"
        scan_by_prefix("k", "A")            -> "A(5)"
        scan_by_prefix("k", "aB")           -> "aB(4)"
    """)


@case("Level 2", title="前缀必须在开头")
def l2_prefix_is_start(sol):
    run(sol, """
        set("k", "name", "1")               -> ""
        set("k", "rename", "2")             -> ""
        set("k", "nickname", "3")           -> ""
        set("k", "n", "4")                  -> ""
        scan_by_prefix("k", "n")            -> "n(4), name(1), nickname(3)"   # "rename" 包含 n，但不是以 n 开头
        scan_by_prefix("k", "name")         -> "name(1)"
        scan_by_prefix("k", "ame")          -> ""             # 出现在中间的不算
        scan_by_prefix("k", "names")        -> ""             # 比字段名还长的前缀
        scan_by_prefix("k", "re")           -> "rename(2)"
        scan_by_prefix("q", "n")            -> ""             # 记录不存在
    """)


@case("Level 2", title="按字段名排序，不是按整段文字")
def l2_sort_by_field_name(sol):
    run(sol, """
        set("p", "first name", "Ann")       -> ""
        set("p", "first", "A")              -> ""
        set("p", "first_name", "ann")       -> ""
        scan("p")                           -> "first(A), first name(Ann), first_name(ann)"   # "first" 是 "first name" 的前缀，排在前面
        scan_by_prefix("p", "first")        -> "first(A), first name(Ann), first_name(ann)"
        scan_by_prefix("p", "first ")       -> "first name(Ann)"
        set("q", "a", "1")                  -> ""
        set("q", "a!", "2")                 -> ""
        scan("q")                           -> "a(1), a!(2)"
    """)


@case("Level 2", title="空记录、单个字段、写入后再扫描")
def l2_scan_updates(sol):
    run(sol, """
        set("k", "a", "1")                  -> ""
        scan("k")                           -> "a(1)"         # 只有一个字段：没有多余的分隔符
        set("k", "b", "2")                  -> ""
        compare_and_set("k", "a", "1", "9") -> "true"
        scan("k")                           -> "a(9), b(2)"
        compare_and_delete("k", "b", "2")   -> "true"
        scan("k")                           -> "a(9)"
        delete("k", "a")                    -> "true"
        scan("k")                           -> ""             # 字段删光了，和不存在的记录一样
        scan_by_prefix("k", "a")            -> ""
        set("j", "a", "1")                  -> ""
        scan("k")                           -> ""             # 别的记录的字段不会出现
        set("k", "c", "3")                  -> ""
        scan("k")                           -> "c(3)"
    """)


@case("Level 2", title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=300, length=(5, 30))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, """
        set_at("u1", "name", "Ann", 1)                                   -> ""
        set_at_with_ttl("u1", "token", "t1", 2, 10)                      -> ""        # 有效时间 [2, 12)
        get_at("u1", "token", 11)                                        -> "t1"
        get_at("u1", "token", 12)                                        -> ""        # 恰好到 12：已经过期
        delete_at("u1", "token", 13)                                     -> "false"   # 过期的字段当作不存在
        set_at_with_ttl("u1", "token", "t2", 14, 5)                      -> ""        # [14, 19)
        set_at_with_ttl("u1", "token", "t3", 16, 10)                     -> ""        # 覆盖并重新计时：[16, 26)
        scan_at("u1", 19)                                                -> "name(Ann), token(t3)"
        compare_and_set_at("u1", "token", "t3", "t4", 20)                -> "true"    # 和 set_at 一样写入：t4 永不过期
        get_at("u1", "token", 30)                                        -> "t4"
        compare_and_set_at_with_ttl("u1", "name", "Ann", "Bob", 31, 4)   -> "true"    # name 现在的有效时间是 [31, 35)
        scan_by_prefix_at("u1", "n", 34)                                 -> "name(Bob)"
        compare_and_delete_at("u1", "name", "Bob", 35)                   -> "false"   # 时刻 35 它已经过期
        scan_at("u1", 36)                                                -> "token(t4)"
        compare_and_set_at("u1", "name", "Bob", "Cat", 37)               -> "false"   # 条件写入不能让过期的字段复活
        get("u1", "name")                                                -> "Bob"     # 不带时间戳的读取不看过期时间
    """)


@case("Level 3", title="有效时间 [t, t+ttl)：到点就过期")
def l3_expiry_boundary(sol):
    run(sol, """
        set_at_with_ttl("k", "f", "v", 5, 1)                    -> ""        # 有效时间 [5, 6)
        get_at("k", "f", 6)                                     -> ""        # ttl = 1：下一刻就过期
        set_at_with_ttl("k", "f", "v", 10, 3)                   -> ""        # [10, 13)
        get_at("k", "f", 12)                                    -> "v"
        get_at("k", "f", 13)                                    -> ""        # 恰好到 13：已经过期
        set_at_with_ttl("k", "g", "w", 14, 2)                   -> ""        # [14, 16)
        scan_at("k", 15)                                        -> "g(w)"
        scan_at("k", 16)                                        -> ""
        set_at_with_ttl("k", "f", "v", 20, 5)                   -> ""        # [20, 25)
        delete_at("k", "f", 25)                                 -> "false"   # 到点就不存在了
        set_at_with_ttl("k", "f", "v", 30, 5)                   -> ""        # [30, 35)
        compare_and_set_at("k", "f", "v", "w", 35)              -> "false"
        set_at_with_ttl("k", "f", "v", 40, 5)                   -> ""        # [40, 45)
        compare_and_delete_at("k", "f", "v", 45)                -> "false"
        set_at_with_ttl("k", "f", "v", 50, 5)                   -> ""        # [50, 55)
        scan_by_prefix_at("k", "f", 55)                         -> ""
        compare_and_set_at_with_ttl("k", "f", "v", "w", 56, 3)  -> "false"
    """)


@case("Level 3", title="覆盖时按这一次写入重新计时")
def l3_overwrite_resets_ttl(sol):
    run(sol, """
        set_at_with_ttl("k", "f", "a", 1, 5)            -> ""        # [1, 6)
        set_at("k", "f", "b", 3)                        -> ""        # 用 set_at 覆盖：从此永不过期
        get_at("k", "f", 6)                             -> "b"
        get_at("k", "f", 1000)                          -> "b"
        set_at_with_ttl("k", "f", "c", 1001, 10)        -> ""        # 覆盖永不过期的字段：[1001, 1011)
        get_at("k", "f", 1010)                          -> "c"
        get_at("k", "f", 1011)                          -> ""
        set_at_with_ttl("k", "g", "a", 1020, 100)       -> ""        # [1020, 1120)
        set_at_with_ttl("k", "g", "b", 1021, 3)         -> ""        # 重新计时，可以变短：[1021, 1024)
        get_at("k", "g", 1023)                          -> "b"
        get_at("k", "g", 1024)                          -> ""
        set_at_with_ttl("k", "h", "a", 1030, 3)         -> ""        # [1030, 1033)
        set_at_with_ttl("k", "h", "b", 1032, 3)         -> ""        # 也可以变长：[1032, 1035)
        get_at("k", "h", 1034)                          -> "b"
        get_at("k", "h", 1035)                          -> ""
        set_at_with_ttl("k", "h", "c", 1040, 2)         -> ""        # 过期之后再写：[1040, 1042)
        get_at("k", "h", 1041)                          -> "c"
    """)


@case("Level 3", title="条件写入成功：和对应的 set 一样")
def l3_conditional_resets_ttl(sol):
    run(sol, """
        set_at_with_ttl("k", "f", "v", 1, 10)                       -> ""        # [1, 11)
        compare_and_set_at("k", "f", "v", "w", 5)                   -> "true"    # 和 set_at 一样：w 永不过期
        get_at("k", "f", 11)                                        -> "w"
        get_at("k", "f", 500)                                       -> "w"
        set_at_with_ttl("k", "g", "v", 501, 10)                     -> ""        # [501, 511)
        compare_and_set_at_with_ttl("k", "g", "v", "w", 505, 3)     -> "true"    # 从 505 重新计时：[505, 508)
        get_at("k", "g", 507)                                       -> "w"
        get_at("k", "g", 508)                                       -> ""
        set_at_with_ttl("k", "h", "v", 510, 3)                      -> ""        # [510, 513)
        compare_and_set_at_with_ttl("k", "h", "v", "w", 512, 20)    -> "true"    # [512, 532)
        get_at("k", "h", 520)                                       -> "w"
        get_at("k", "h", 532)                                       -> ""
        set_at("k", "p", "v", 540)                                  -> ""        # 永不过期
        compare_and_set_at_with_ttl("k", "p", "v", "w", 541, 5)     -> "true"    # 和 set_at_with_ttl 一样：[541, 546)
        get_at("k", "p", 545)                                       -> "w"
        get_at("k", "p", 546)                                       -> ""
        set_at_with_ttl("k", "q", "v", 550, 4)                      -> ""        # [550, 554)
        compare_and_set_at("k", "q", "v", "v", 551)                 -> "true"    # 值没变也算写入：TTL 照样作废
        get_at("k", "q", 600)                                       -> "v"
    """)


@case("Level 3", title="过期的字段对带时间戳的方法不存在")
def l3_expired_is_missing(sol):
    run(sol, """
        set_at_with_ttl("k", "f", "v", 1, 4)                        -> ""        # [1, 5)
        compare_and_set_at("k", "f", "v", "w", 6)                   -> "false"   # 已经过期：值虽然还是 "v"，也当作不存在
        get_at("k", "f", 7)                                         -> ""        # 失败的条件写入没有让它复活
        compare_and_set_at_with_ttl("k", "f", "v", "w", 8, 50)      -> "false"
        get_at("k", "f", 9)                                         -> ""
        compare_and_delete_at("k", "f", "v", 10)                    -> "false"
        delete_at("k", "f", 11)                                     -> "false"
        scan_at("k", 12)                                            -> ""
        set_at("k", "f", "n", 13)                                   -> ""        # 普通写入照常生效
        compare_and_set_at("k", "f", "n", "m", 14)                  -> "true"
        get_at("k", "f", 15)                                        -> "m"
    """)


@case("Level 3", title="扫描跳过过期的字段")
def l3_scan_skips_expired(sol):
    run(sol, """
        set_at("u", "b", "1", 1)                        -> ""
        set_at_with_ttl("u", "a", "2", 2, 5)            -> ""        # [2, 7)
        set_at_with_ttl("u", "ab", "3", 3, 2)           -> ""        # [3, 5)
        set_at_with_ttl("u", "B", "4", 4, 10)           -> ""        # [4, 14)
        scan_at("u", 4)                                 -> "B(4), a(2), ab(3), b(1)"
        scan_at("u", 5)                                 -> "B(4), a(2), b(1)"        # ab 在 5 过期
        scan_by_prefix_at("u", "a", 6)                  -> "a(2)"
        scan_by_prefix_at("u", "a", 7)                  -> ""                        # a 也过期了
        scan_at("u", 8)                                 -> "B(4), b(1)"
        set_at_with_ttl("v", "x", "5", 9, 1)            -> ""        # [9, 10)
        scan_at("v", 10)                                -> ""        # 记录里的字段全部过期，和不存在一样
        scan_by_prefix_at("u", "B", 14)                 -> ""
        scan_at("u", 15)                                -> "b(1)"
    """)


@case("Level 3", title="不带时间戳的读取不看过期时间")
def l3_plain_reads_ignore_expiry(sol):
    run(sol, """
        set_at_with_ttl("u", "name", "Ann", 1, 5)       -> ""        # [1, 6)
        set_at_with_ttl("u", "nick", "an", 2, 3)        -> ""        # [2, 5)
        set_at("u", "age", "30", 3)                     -> ""
        get_at("u", "name", 10)                         -> ""        # 带时间戳的读取：已过期
        get("u", "name")                                -> "Ann"     # 不带时间戳的读取：字段还在数据库里，照常返回
        scan_at("u", 11)                                -> "age(30)"
        scan("u")                                       -> "age(30), name(Ann), nick(an)"
        scan_by_prefix("u", "n")                        -> "name(Ann), nick(an)"
        compare_and_set_at("u", "nick", "an", "x", 12)  -> "false"
        get("u", "nick")                                -> "an"      # 失败的条件写入什么也没改
        set_at("u", "name", "Bob", 13)                  -> ""        # 覆盖之后读到新值
        get("u", "name")                                -> "Bob"
    """)


@case("Level 3", title="删除失败的过期字段仍在数据库里")
def l3_failed_delete_keeps_field(sol):
    run(sol, """
        set_at_with_ttl("u", "a", "1", 1, 2)            -> ""        # [1, 3)
        set_at_with_ttl("u", "b", "2", 2, 2)            -> ""        # [2, 4)
        delete_at("u", "a", 5)                          -> "false"   # 过期的字段当作不存在：删除失败
        get("u", "a")                                   -> "1"       # 没有被成功删除，不带时间戳的读取照常返回
        compare_and_delete_at("u", "b", "2", 6)         -> "false"
        scan("u")                                       -> "a(1), b(2)"
        set_at("u", "c", "3", 7)                        -> ""
        delete_at("u", "c", 8)                          -> "true"    # 成功删除
        get("u", "c")                                   -> ""        # 删掉了，不带时间戳的读取也看不到
        scan("u")                                       -> "a(1), b(2)"
    """)


@case("Level 3", title="用关键字传 timestamp、ttl、prefix")
def l3_keyword_args(sol):
    run(sol, """
        set_at("u", "a", "1", timestamp=1)                                      -> ""
        set_at_with_ttl("u", "b", "2", timestamp=2, ttl=5)                      -> ""        # [2, 7)
        set_at_with_ttl("u", "c", "3", ttl=3, timestamp=3)                      -> ""        # [3, 6)；关键字的顺序随意
        set_at_with_ttl("u", "d", "4", 4, ttl=10)                               -> ""        # [4, 14)
        get_at("u", "a", timestamp=5)                                           -> "1"
        scan_at("u", timestamp=6)                                               -> "a(1), b(2), d(4)"
        scan_by_prefix_at("u", prefix="b", timestamp=7)                         -> ""        # b 在 7 过期
        scan_by_prefix_at("u", "d", timestamp=8)                                -> "d(4)"
        scan_by_prefix_at("u", timestamp=9, prefix="a")                         -> "a(1)"
        delete_at("u", "a", timestamp=10)                                       -> "true"
        compare_and_set_at("u", "d", "4", "5", timestamp=11)                    -> "true"
        compare_and_set_at_with_ttl("u", "d", "5", "6", timestamp=12, ttl=2)    -> "true"    # [12, 14)
        compare_and_set_at_with_ttl("u", "d", "6", "7", 13, ttl=1)              -> "true"    # [13, 14)
        compare_and_delete_at("u", "d", "7", timestamp=14)                      -> "false"   # 14 时已经过期
        get_at("u", "d", timestamp=15)                                          -> ""
    """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=500, length=(10, 50))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, """
        set_at("a", "x", "1", 1)                        -> ""
        set_at_with_ttl("a", "y", "2", 2, 10)           -> ""       # 有效时间 [2, 12)
        set_at_with_ttl("b", "z", "3", 3, 4)            -> ""       # 有效时间 [3, 7)
        backup(5)                                       -> "2"      # a 和 b 都有未过期的字段；此刻 y 还剩 7，z 还剩 2
        delete_at("a", "x", 6)                          -> "true"
        backup(8)                                       -> "1"      # b 唯一的字段在 7 过期了；此刻 y 还剩 4
        set_at("c", "w", "9", 9)                        -> ""
        restore(20, 7)                                  -> ""       # 不超过 7 的最新备份是时刻 5 的那份
        scan_at("a", 21)                                -> "x(1), y(2)"   # y 的新过期时刻是 20 + 7 = 27
        get_at("b", "z", 22)                            -> ""       # z 的新过期时刻是 20 + 2 = 22
        get_at("c", "w", 23)                            -> ""       # c 是备份之后才写的，恢复后不存在
        get_at("a", "y", 26)                            -> "2"
        get_at("a", "y", 27)                            -> ""
        backup(28)                                      -> "1"      # 只剩 a（字段 x）
        get_when(30, "a", "x", 5)                       -> "1"
        get_when(31, "a", "x", 6)                       -> ""       # 时刻 6 的删除已经生效
        get_when(32, "a", "x", 19)                      -> ""
        get_when(33, "a", "x", 20)                      -> "1"      # 时刻 20 的 restore 把它带了回来
        get_when(34, "a", "y", 11)                      -> "2"
        get_when(35, "a", "y", 12)                      -> ""       # 原来的有效时间到 12 为止
        get_when(36, "a", "y", 26)                      -> "2"      # 恢复之后的有效时间是 [20, 27)
        get_when(37, "b", "z", 21)                      -> "3"      # 恢复之后的有效时间是 [20, 22)
        get_when(38, "c", "w", 10)                      -> "9"      # 时刻 10 它还在；restore 不改写过去
        get_when(39, "c", "w", 20)                      -> ""
        get_when(40, "b", "z", 2)                       -> ""       # 那时还没写入
        restore(41, 8)                                  -> ""       # 时刻 8 的备份里只有 a 的 y，当时还剩 4
        scan_at("a", 44)                                -> "y(2)"
        scan_at("a", 45)                                -> ""       # 41 + 4 = 45
    """)


@case("Level 4", title="backup 只数有未过期字段的记录")
def l4_backup_count(sol):
    run(sol, """
        backup(1)                                       -> "0"      # 空数据库
        set_at("a", "p", "1", 2)                        -> ""
        set_at_with_ttl("a", "q", "2", 3, 3)            -> ""       # [3, 6)
        set_at_with_ttl("b", "r", "3", 4, 3)            -> ""       # [4, 7)
        set_at_with_ttl("c", "s", "4", 5, 10)           -> ""       # [5, 15)
        backup(6)                                       -> "3"      # a 的 q 恰好过期，但 p 还在；b、c 都还在
        backup(7)                                       -> "2"      # b 唯一的字段恰好在 7 过期
        delete_at("c", "s", 8)                          -> "true"
        backup(9)                                       -> "1"      # c 的字段删光了
        delete_at("a", "p", 10)                         -> "true"
        backup(11)                                      -> "0"
    """)


@case("Level 4", title="恢复后按剩余寿命重算过期时刻")
def l4_restore_remaining(sol):
    run(sol, """
        set_at_with_ttl("a", "x", "1", 1, 10)           -> ""       # [1, 11)
        set_at("a", "y", "2", 2)                        -> ""       # 永不过期
        backup(4)                                       -> "1"      # 此刻 x 还剩 7
        restore(100, 4)                                 -> ""       # x 的新过期时刻是 100 + 7 = 107
        get_at("a", "x", 106)                           -> "1"
        get_at("a", "x", 107)                           -> ""
        get_at("a", "y", 108)                           -> "2"      # 没有 TTL 的字段恢复后仍然没有
        restore(200, 4)                                 -> ""       # 同一份备份再恢复一次：从 200 起又是还剩 7
        scan_at("a", 206)                               -> "x(1), y(2)"
        scan_at("a", 207)                               -> "y(2)"
        restore(208, 150)                               -> ""       # 不超过 150 的最新备份仍是时刻 4 的那份
        get_at("a", "x", 214)                           -> "1"
        get_at("a", "x", 215)                           -> ""
    """)


@case("Level 4", title="恢复：不超过给定时刻的最新备份")
def l4_restore_picks_backup(sol):
    run(sol, """
        set_at("k", "f", "a", 1)                        -> ""
        backup(2)                                       -> "1"
        set_at("k", "f", "b", 3)                        -> ""
        backup(4)                                       -> "1"
        set_at("k", "f", "c", 5)                        -> ""
        backup(6)                                       -> "1"
        set_at("k", "f", "d", 7)                        -> ""
        restore(8, 4)                                   -> ""       # 恰好有一份时刻 4 的备份：就用它
        get_at("k", "f", 9)                             -> "b"
        restore(10, 5)                                  -> ""       # 不超过 5 的最新备份是时刻 4 的那份
        get_at("k", "f", 11)                            -> "b"
        restore(12, 2)                                  -> ""
        get_at("k", "f", 13)                            -> "a"
        restore(14, 6)                                  -> ""
        get_at("k", "f", 15)                            -> "c"
        restore(16, 15)                                 -> ""       # restore 不产生备份：最新的仍是时刻 6 的那份
        get_at("k", "f", 17)                            -> "c"
    """)


@case("Level 4", title="恢复后恰好是备份里的字段")
def l4_restore_replaces_all(sol):
    run(sol, """
        set_at("a", "x", "1", 1)                        -> ""
        set_at_with_ttl("a", "y", "2", 2, 3)            -> ""       # [2, 5)
        set_at_with_ttl("b", "z", "3", 3, 100)          -> ""       # [3, 103)
        backup(5)                                       -> "2"      # y 恰好过期，不进备份；z 还剩 98
        set_at("a", "x", "9", 6)                        -> ""       # 备份之后改写
        delete_at("b", "z", 7)                          -> "true"   # 备份之后删除
        set_at("c", "w", "4", 8)                        -> ""       # 备份之后新写的记录
        set_at("a", "v", "5", 9)                        -> ""       # 备份之后在已有记录里新加的字段
        restore(10, 5)                                  -> ""
        scan_at("a", 11)                                -> "x(1)"   # 改写和新加的都回到备份时的样子
        get_at("b", "z", 12)                            -> "3"      # 删掉的回来了，过期时刻 10 + 98 = 108
        scan_at("c", 13)                                -> ""       # 备份之后才有的记录消失
        get("a", "y")                                   -> ""       # 不带时间戳的读取：y 不在备份里，已被 restore 换掉
        scan("a")                                       -> "x(1)"
        get("c", "w")                                   -> ""
        get_at("b", "z", 107)                           -> "3"
        get_at("b", "z", 108)                           -> ""
        get("b", "z")                                   -> "3"      # 过期了，但不带时间戳的读取不看过期时间
    """)


@case("Level 4", title="备份不受之后的修改影响")
def l4_backup_is_copy(sol):
    run(sol, """
        set_at("a", "x", "1", 1)                        -> ""
        set_at("a", "y", "2", 2)                        -> ""
        backup(3)                                       -> "1"
        set_at("a", "x", "changed", 4)                  -> ""       # 改写备份里已有记录的字段
        delete_at("a", "y", 5)                          -> "true"
        set_at("a", "z", "new", 6)                      -> ""
        restore(7, 3)                                   -> ""
        scan_at("a", 8)                                 -> "x(1), y(2)"
        set_at("a", "x", "again", 9)                    -> ""       # 恢复之后再修改
        delete_at("a", "y", 10)                         -> "true"
        restore(11, 3)                                  -> ""       # 同一份备份可以恢复多次
        scan_at("a", 12)                                -> "x(1), y(2)"
    """)


@case("Level 4", title="get_when：恰好在那一刻的调用已经生效")
def l4_get_when_exact(sol):
    run(sol, """
        set_at("k", "f", "a", 10)                       -> ""
        set_at("k", "f", "b", 20)                       -> ""
        delete_at("k", "f", 30)                         -> "true"
        set_at_with_ttl("k", "f", "c", 40, 5)           -> ""       # [40, 45)
        get_when(50, "k", "f", 9)                       -> ""       # 还没写入
        get_when(51, "k", "f", 10)                      -> "a"      # 恰好是写入的时刻：已经生效
        get_when(52, "k", "f", 19)                      -> "a"
        get_when(53, "k", "f", 20)                      -> "b"
        get_when(54, "k", "f", 29)                      -> "b"
        get_when(55, "k", "f", 30)                      -> ""       # 恰好是删除的时刻：已经删掉
        get_when(56, "k", "f", 40)                      -> "c"
        get_when(57, "k", "f", 44)                      -> "c"
        get_when(58, "k", "f", 45)                      -> ""       # 按 at_timestamp 判断过期
        get_when(59, "k", "f", 59)                      -> ""       # at_timestamp 可以等于 timestamp
        get_when(60, "k", "g", 10)                      -> ""       # 从没写过的字段
        get_when(61, "q", "f", 10)                      -> ""
    """)


@case("Level 4", title="get_when：按那一刻判断过期")
def l4_get_when_expiry(sol):
    run(sol, """
        set_at_with_ttl("k", "f", "a", 1, 10)           -> ""       # [1, 11)
        set_at_with_ttl("k", "f", "b", 5, 2)            -> ""       # 覆盖：[5, 7)
        set_at("k", "f", "c", 9)                        -> ""       # 覆盖：永不过期
        set_at_with_ttl("k", "g", "x", 10, 3)           -> ""       # [10, 13)
        get_when(20, "k", "f", 4)                       -> "a"
        get_when(21, "k", "f", 6)                       -> "b"
        get_when(22, "k", "f", 7)                       -> ""       # b 在 7 过期；被覆盖的 a 不会因此回来
        get_when(23, "k", "f", 8)                       -> ""
        get_when(24, "k", "f", 9)                       -> "c"
        get_when(25, "k", "f", 24)                      -> "c"
        get_when(26, "k", "g", 12)                      -> "x"
        get_when(27, "k", "g", 13)                      -> ""
        get_when(28, "k", "g", 1)                       -> ""
    """)


@case("Level 4", title="get_when：restore 不改写过去")
def l4_get_when_and_restore(sol):
    run(sol, """
        set_at("a", "x", "1", 1)                        -> ""
        set_at_with_ttl("a", "y", "2", 2, 8)            -> ""       # [2, 10)
        backup(3)                                       -> "1"      # y 还剩 7
        set_at("a", "x", "5", 4)                        -> ""
        delete_at("a", "y", 5)                          -> "true"
        set_at("a", "z", "6", 6)                        -> ""
        restore(20, 3)                                  -> ""       # x 回到 "1"，y 在 [20, 27)，z 消失
        get_when(30, "a", "x", 19)                      -> "5"      # restore 之前：当时真实的值，不是备份里的
        get_when(31, "a", "x", 20)                      -> "1"      # 从 restore 那一刻起是备份里的值
        get_when(32, "a", "y", 9)                       -> ""       # 时刻 5 已经删掉了
        get_when(33, "a", "y", 19)                      -> ""
        get_when(34, "a", "y", 20)                      -> "2"      # restore 把它带了回来
        get_when(35, "a", "y", 26)                      -> "2"
        get_when(36, "a", "y", 27)                      -> ""       # 20 + 7
        get_when(37, "a", "z", 19)                      -> "6"      # 备份之后写的，restore 之前一直在
        get_when(38, "a", "z", 20)                      -> ""
        get_when(39, "a", "x", 3)                       -> "1"
    """)


@case("Level 4", title="get_when：条件写入和条件删除")
def l4_get_when_conditional(sol):
    run(sol, """
        set_at("k", "f", "a", 1)                                -> ""
        compare_and_set_at("k", "f", "zz", "b", 2)              -> "false"
        compare_and_set_at_with_ttl("k", "f", "a", "c", 3, 4)   -> "true"    # [3, 7)
        compare_and_delete_at("k", "f", "c", 5)                 -> "true"
        set_at("k", "f", "d", 8)                                -> ""
        get_when(10, "k", "f", 2)                               -> "a"       # 失败的条件写入不改变历史
        get_when(11, "k", "f", 3)                               -> "c"
        get_when(12, "k", "f", 4)                               -> "c"
        get_when(13, "k", "f", 5)                               -> ""        # 条件删除那一刻起不存在
        get_when(14, "k", "f", 7)                               -> ""
        get_when(15, "k", "f", 8)                               -> "d"
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=500, length=(15, 60))
