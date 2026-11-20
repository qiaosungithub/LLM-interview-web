import ast
import time

from judgelib import Mismatch, calling, case, expect_true, rng
import ref

SHOW = 60           # a replay lists at most this many calls (every sequence these tests make fits)


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str):
        if '"' not in x and "\\" not in x and x.isprintable():
            return f'"{x}"'
        return repr(x)
    return repr(x)


def _call(method, args, kwargs):
    parts = [_lit(a) for a in args] + [f"{k}={_lit(v)}" for k, v in kwargs.items()]
    return f"db.{method}({', '.join(parts)})"


def _replay(done, wants, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step."""
    lines = [_call(*op) for op in done]
    lines[-1] = f"print(repr({lines[-1]}))"
    notes = [f"# -> {_lit(w)}" for w in wants] + [mark]
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = ["db = InMemoryDatabase()"] + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(head + [s.ljust(width) + n for s, n in zip(lines, notes)])


def _method(obj, name):
    fn = getattr(obj, name, None)
    expect_true(callable(fn), f"InMemoryDatabase 没有 {name} 方法（方法名要和题目一致）")
    return fn


def _same(got, want):
    """got is the string want (every method returns a string)."""
    return isinstance(got, str) and got == want


def _differs(got, want, call, msg):
    """Fails the case for a return value that _same rejected."""
    if not isinstance(got, str):
        raise Mismatch(f"类型不对：每个方法都返回字符串，这里得到 {type(got).__name__}", call, want, got)
    raise Mismatch(msg or "返回值不对", call, want, got)


# ---------------------------------------------------------------- hand-written sequences

def _value(node):
    """A literal such as "a" or 5 (the scripts below are the only input)."""
    return eval(compile(ast.Expression(node), "<script>", "eval"), {"__builtins__": {}})


def _parse(script):
    """Lines like  get_at(5, "k", "f") -> "v"   # why  become (method, args, kwargs, want, why)."""
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
    """Runs the calls on a fresh InMemoryDatabase, checking each return value as soon as it comes back."""
    db = sol.InMemoryDatabase()
    done, wants = [], []
    for method, args, kwargs, want, why in [s for script in scripts for s in _parse(script)]:
        done.append((method, args, kwargs))
        fn = _method(db, method)
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
    """Replays ops on a fresh answer and a fresh ref.Database; None if every return value agrees. A shortened
    sequence can reach a call the statement does not cover (see ref.Database.ambiguous); it then counts as
    passing, so the shrinker never shows such a step."""
    db, model, wants = sol.InMemoryDatabase(), ref.Database(), []
    for k, (method, args, kwargs) in enumerate(ops):
        if model.ambiguous(method, args):
            return None
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
    set("order-1", "status", "packed")     -> ""
    set("order-1", "carrier", "fedex")     -> ""
    get("order-1", "status")               -> "packed"
    get("order-1", "weight")               -> ""          # 这个字段从未被设置过
    get("order-9", "status")               -> ""          # 这个 key 从未被创建过
    delete("order-1", "carrier")           -> "true"
    delete("order-1", "carrier")           -> "false"     # 已经被删掉了
    set("order-1", "status", "shipped")    -> ""          # 覆盖写入
    get("order-1", "status")               -> "shipped"
"""

EXAMPLE_L2 = """
    set("order-1", "carrier", "fedex")     -> ""
    set("order-1", "cost", "12.50")        -> ""
    set("order-1", "status", "packed")     -> ""
    scan_by_prefix("order-1", "c")         -> "carrier(fedex), cost(12.50)"
    scan("order-1")                        -> "carrier(fedex), cost(12.50), status(packed)"
    scan_by_prefix("order-9", "c")         -> ""          # 这个 key 从未被创建过
"""

EXAMPLE_L3 = """
    set_at_with_ttl(10, "order-1", "carrier", "fedex", 5)   -> ""    # carrier 在 [10, 15) 期间存活
    set_at_with_ttl(12, "order-1", "carrier", "ups", 6)     -> ""    # 替换：窗口变为 [12, 18)，不是延长到 21
    set_at(13, "order-1", "cost", "12.50")                  -> ""    # 没有 ttl：cost 永久存在
    scan_at(17, "order-1")                  -> "carrier(ups), cost(12.50)"   # 17 < 18：carrier 仍然存活
    scan_at(18, "order-1")                  -> "cost(12.50)"                 # 18 == 18：carrier 刚好过期
    delete_at(18, "order-1", "carrier")     -> "false"                       # 已经过期，没有可删除的内容
    set_at_with_ttl(19, "order-1", "carrier", "usps", 3)    -> ""    # carrier 再次存活，在 [19, 22) 期间
    set_at(20, "order-1", "carrier", "dhl")                 -> ""    # 没有 ttl：清除了 TTL，现在永久存在
    scan_at(22, "order-1")                  -> "carrier(dhl), cost(12.50)"   # TTL 已被清除
"""

EXAMPLE_L4 = """
    set_at_with_ttl(0, "order-1", "carrier", "fedex", 10)   -> ""          # 在 [0, 10) 期间存活
    set_at(0, "order-1", "cost", "12.50")                   -> ""          # 永久存在
    backup(4)                                 -> ""          # 快照：carrier 剩余 6（10 - 4），cost 永久
    set_at(5, "order-1", "cost", "15.00")     -> ""          # 快照不受影响
    delete_at(6, "order-1", "carrier")        -> "true"      # 6 < 10：carrier 仍然存活
    restore(100, 4)                           -> ""          # 在时间戳 100 恢复 ts=4 的那次备份
    get_at(100, "order-1", "cost")            -> "12.50"     # 回到备份时的值
    get_at(100, "order-1", "carrier")         -> "fedex"     # 恢复出来，剩余 6，从 100 重新开始倒计时
    get_at(105, "order-1", "carrier")         -> "fedex"     # 100 + 6 = 106：在 105 时仍然存活
    get_at(106, "order-1", "carrier")         -> ""          # 106 正是新的过期时刻
"""


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, EXAMPLE_L1)


@case("Level 1", title="不同 key 的同名字段互不影响")
def l1_keys_and_fields(sol):
    run(sol, """
        set("k1", "f", "one")      -> ""
        set("k2", "f", "two")      -> ""
        get("k1", "f")             -> "one"
        get("k2", "f")             -> "two"
        delete("k1", "f")          -> "true"
        get("k1", "f")             -> ""          # 删掉的是 k1 的 f
        get("k2", "f")             -> "two"       # k2 的 f 不受影响
        delete("k1", "f")          -> "false"     # 已经删掉了
        delete("k3", "f")          -> "false"     # 这个 key 从未被创建过
        delete("k2", "g")          -> "false"     # key 存在，但没有这个字段
        get("k2", "g")             -> ""
        set("k1", "f", "again")    -> ""          # 删除之后可以重新设置
        get("k1", "f")             -> "again"
        set("k1", "g", "f")        -> ""
        get("k1", "g")             -> "f"
        get("k1", "f")             -> "again"
    """)


@case("Level 1", title="任意非空字符串")
def l1_strings(sol):
    run(sol, """
        set("订单 7", "状态", "已发货")    -> ""
        get("订单 7", "状态")             -> "已发货"
        set("x", "x", "x")               -> ""          # key、字段名和值可以相同
        get("x", "x")                    -> "x"
        set("a b", "c d", "e f")         -> ""          # 可以含空格
        get("a b", "c d")                -> "e f"
        get("a b", "c")                  -> ""
        get("a", "c d")                  -> ""
        set("a b", "c d", "12.50")       -> ""
        get("a b", "c d")                -> "12.50"
        delete("a b", "c d")             -> "true"
        get("x", "x")                    -> "x"
    """)


@case("Level 1", cpu=2, title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=300, length=(5, 30))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, EXAMPLE_L2)


@case("Level 2", title="按字段名排序，不按插入顺序")
def l2_sort_order(sol):
    run(sol, """
        set("k", "status", "packed")   -> ""
        set("k", "cost", "9")          -> ""
        set("k", "Weight", "2kg")      -> ""
        set("k", "carrier", "ups")     -> ""
        scan("k")                      -> "Weight(2kg), carrier(ups), cost(9), status(packed)"   # 大写字母排在小写字母之前
        set("k", "9", "b")             -> ""
        set("k", "10", "a")            -> ""
        scan("k")                      -> "10(a), 9(b), Weight(2kg), carrier(ups), cost(9), status(packed)"   # 字符串比较："10" < "9"
        scan_by_prefix("k", "c")       -> "carrier(ups), cost(9)"
    """)


@case("Level 2", title="排序看字段名，而不是格式化后的串")
def l2_sort_by_name(sol):
    run(sol, """
        set("k", "box lid", "1")       -> ""
        set("k", "box", "2")           -> ""
        set("k", "box&tape", "3")      -> ""
        set("k", "boxes", "4")         -> ""
        scan("k")                      -> "box(2), box lid(1), box&tape(3), boxes(4)"   # "box" 是 "box lid" 的前缀，排在它前面
        scan_by_prefix("k", "box")     -> "box(2), box lid(1), box&tape(3), boxes(4)"
        scan_by_prefix("k", "box ")    -> "box lid(1)"
    """)


@case("Level 2", title="前缀：只看开头，区分大小写")
def l2_prefix(sol):
    run(sol, """
        set("k", "account", "a1")      -> ""
        set("k", "cost", "5")          -> ""
        set("k", "c", "x")             -> ""
        set("k", "Cargo", "big")       -> ""
        set("k", "costs", "6")         -> ""
        scan_by_prefix("k", "c")       -> "c(x), cost(5), costs(6)"     # "account" 中间有 c 不算；"Cargo" 是大写 C
        scan_by_prefix("k", "cost")    -> "cost(5), costs(6)"           # 前缀可以等于完整的字段名
        scan_by_prefix("k", "costs!")  -> ""                            # 前缀比字段名还长
        scan_by_prefix("k", "C")       -> "Cargo(big)"
        scan_by_prefix("k", "")        -> "Cargo(big), account(a1), c(x), cost(5), costs(6)"   # 空前缀和 scan 完全一样
        scan("k")                      -> "Cargo(big), account(a1), c(x), cost(5), costs(6)"
        scan_by_prefix("k", "z")       -> ""                            # 没有匹配的字段
        scan_by_prefix("other", "")    -> ""                            # 这个 key 从未被创建过
    """)


@case("Level 2", title="扫描反映覆盖与删除")
def l2_scan_changes(sol):
    run(sol, """
        set("k", "b", "1")             -> ""
        set("k", "a", "2")             -> ""
        scan("k")                      -> "a(2), b(1)"
        set("k", "b", "3")             -> ""          # 覆盖写入
        scan("k")                      -> "a(2), b(3)"
        delete("k", "a")               -> "true"
        scan("k")                      -> "b(3)"      # 删掉的字段不再出现
        scan_by_prefix("k", "a")       -> ""
        delete("k", "b")               -> "true"
        scan("k")                      -> ""          # 所有字段都删掉了
        scan_by_prefix("k", "")        -> ""
        set("k2", "a", "x")            -> ""
        scan("k")                      -> ""          # 别的 key 的字段不会出现
        scan("k2")                     -> "a(x)"
    """)


@case("Level 2", cpu=2, title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=300, length=(5, 30))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, EXAMPLE_L3)


@case("Level 3", title="TTL 窗口不含右端点")
def l3_ttl_boundary(sol):
    run(sol, """
        set_at_with_ttl(5, "k", "f", "v", 1)    -> ""
        get_at(5, "k", "f")                     -> "v"       # 窗口 [5, 6)：5 时存活
        scan_at(5, "k")                         -> "f(v)"
        get_at(6, "k", "f")                     -> ""        # 6 == 5 + 1：刚好过期
        scan_at(6, "k")                         -> ""
        scan_by_prefix_at(6, "k", "f")          -> ""
        delete_at(6, "k", "f")                  -> "false"   # 已经过期
        set_at_with_ttl(6, "k", "g", "w", 3)    -> ""        # [6, 9)
        delete_at(8, "k", "g")                  -> "true"    # 8 < 9：仍然存活
        get_at(8, "k", "g")                     -> ""        # 已删除
    """)


@case("Level 3", title="新 TTL 替换旧窗口；set_at 清除 TTL")
def l3_ttl_replace(sol):
    run(sol, """
        set_at_with_ttl(10, "k", "f", "a", 10)    -> ""      # [10, 20)
        set_at_with_ttl(12, "k", "f", "b", 2)     -> ""      # 替换为 [12, 14)：窗口也可以变短
        get_at(13, "k", "f")                      -> "b"
        get_at(14, "k", "f")                      -> ""      # 旧窗口到 20，但已被替换
        set_at(15, "k", "f", "c")                 -> ""      # 过期之后重新设置：永久存在
        get_at(1000, "k", "f")                    -> "c"
        set_at_with_ttl(1000, "k", "f", "d", 5)   -> ""      # 永久字段也可以重新带上 TTL：[1000, 1005)
        get_at(1004, "k", "f")                    -> "d"
        get_at(1005, "k", "f")                    -> ""
        set_at_with_ttl(1006, "k", "g", "e", 4)   -> ""      # [1006, 1010)
        set_at(1007, "k", "g", "e")               -> ""      # 值相同，但 TTL 被清除
        get_at(1010, "k", "g")                    -> "e"
        scan_at(5000, "k")                        -> "g(e)"
    """)


@case("Level 3", title="delete_at：过期的字段算作不存在")
def l3_delete_at(sol):
    run(sol, """
        set_at(1, "k", "p", "perm")                -> ""
        set_at_with_ttl(1, "k", "t", "temp", 4)    -> ""         # [1, 5)
        delete_at(2, "k", "nope")                  -> "false"    # 没有这个字段
        delete_at(3, "k", "t")                     -> "true"     # 仍然存活
        delete_at(3, "k", "t")                     -> "false"    # 已经删掉了
        get_at(3, "k", "t")                        -> ""
        set_at_with_ttl(4, "k", "t", "temp2", 2)   -> ""         # [4, 6)
        delete_at(9, "k", "t")                     -> "false"    # 早已过期
        set_at(9, "k", "t", "again")               -> ""
        get_at(9, "k", "t")                        -> "again"
        delete_at(9, "k", "p")                     -> "true"     # 永久字段随时可以删除
        scan_at(9, "k")                            -> "t(again)"
        delete_at(9, "other", "p")                 -> "false"    # 这个 key 从未被创建过
    """)


@case("Level 3", title="扫描只列出当时存活的字段")
def l3_scan_at(sol):
    run(sol, """
        set_at_with_ttl(0, "k", "b1", "x", 3)   -> ""      # [0, 3)
        set_at_with_ttl(0, "k", "a1", "y", 5)   -> ""      # [0, 5)
        set_at(0, "k", "c1", "z")               -> ""
        set_at_with_ttl(1, "k", "B", "w", 1)    -> ""      # [1, 2)
        scan_at(1, "k")                         -> "B(w), a1(y), b1(x), c1(z)"
        scan_by_prefix_at(1, "k", "b")          -> "b1(x)"                 # 区分大小写
        scan_at(2, "k")                         -> "a1(y), b1(x), c1(z)"   # B 在 2 过期
        scan_by_prefix_at(3, "k", "")           -> "a1(y), c1(z)"          # b1 在 3 过期
        scan_by_prefix_at(4, "k", "a")          -> "a1(y)"
        scan_at(5, "k")                         -> "c1(z)"                 # a1 在 5 过期
        scan_by_prefix_at(5, "k", "a")          -> ""
        scan_at(5, "zzz")                       -> ""
    """)


@case("Level 3", title="普通调用同样受 TTL 约束")
def l3_plain_ttl(sol):
    run(sol, """
        set_at_with_ttl(0, "k", "f", "v", 1)     -> ""         # [0, 1)
        set_at_with_ttl(0, "k", "t", "u", 100)   -> ""         # [0, 100)
        set("k", "g", "w")                       -> ""         # 发生在 0，之后时钟为 1
        set("k", "h", "z")                       -> ""         # 发生在 1
        get("k", "f")                            -> ""         # 发生在 2：f 已经过期
        get("k", "t")                            -> "u"        # 发生在 3
        scan("k")                                -> "g(w), h(z), t(u)"
        scan_by_prefix("k", "")                  -> "g(w), h(z), t(u)"
        delete("k", "f")                         -> "false"    # 过期的字段算作不存在
    """)
    run(sol, """
        set_at_with_ttl(0, "k", "f", "v", 3)     -> ""         # [0, 3)
        set("k", "f", "w")                       -> ""         # 发生在 0：和 set_at 一样清除 TTL
        get_at(5, "k", "f")                      -> "w"        # f 现在永久存在
    """)


@case("Level 3", title="普通调用发生在时钟的当前值上")
def l3_plain_clock(sol):
    run(sol, """
        set_at_with_ttl(0, "k", "f", "v", 3)    -> ""          # [0, 3)；时钟设为 0
        get("k", "f")                           -> "v"         # 发生在时钟的当前值 0，之后时钟才变为 1
        get("k", "f")                           -> "v"         # 发生在 1
        get("k", "f")                           -> "v"         # 发生在 2：2 < 3，仍然存活
        get("k", "f")                           -> ""          # 发生在 3：刚好过期
    """)
    run(sol, """
        set("k", "a", "1")                      -> ""          # 发生在 0
        set("k", "b", "2")                      -> ""          # 发生在 1，之后时钟为 2
        set_at_with_ttl(2, "k", "c", "3", 2)    -> ""          # [2, 4)
        scan("k")                               -> "a(1), b(2), c(3)"    # 发生在 2
        scan_by_prefix("k", "c")                -> "c(3)"                # 发生在 3：3 < 4，c 仍然存活
        scan("k")                               -> "a(1), b(2)"          # 发生在 4：c 刚好过期
        get_at(5, "k", "a")                     -> "1"         # 普通 set 写入的字段可以被 get_at 读到
    """)


@case("Level 3", title="带时间戳的调用会把时钟设为 timestamp")
def l3_shared_clock(sol):
    run(sol, """
        set_at_with_ttl(50, "k", "f", "v", 2)   -> ""          # [50, 52)；时钟设为 50
        get("k", "f")                           -> "v"         # 发生在 50
        get("k", "f")                           -> "v"         # 发生在 51
        get("k", "f")                           -> ""          # 发生在 52：已经过期
        delete("k", "f")                        -> "false"     # 发生在 53
        set("k", "f", "w")                      -> ""          # 发生在 54，没有 TTL
        get_at(60, "k", "f")                    -> "w"
        set_at_with_ttl(70, "k", "g", "x", 1)   -> ""          # [70, 71)
        scan("k")                               -> "f(w), g(x)"    # 发生在 70
        scan("k")                               -> "f(w)"          # 发生在 71：g 已经过期
    """)
    for twin in ('get_at(24, "k", "f")            -> "v"',
                 'scan_at(24, "k")                -> "f(v)"',
                 'scan_by_prefix_at(24, "k", "")  -> "f(v)"',
                 'delete_at(24, "k", "nope")      -> "false"'):
        run(sol, f"""
            set_at_with_ttl(10, "k", "f", "v", 15)   -> ""     # [10, 25)
            {twin}     # 只读（或什么也没删）的调用同样把时钟设为 24
            get("k", "f")                            -> "v"    # 发生在 24
            get("k", "f")                            -> ""     # 发生在 25：已经过期
        """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=400, length=(10, 40))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, EXAMPLE_L4)


@case("Level 4", title="恢复后从恢复时刻重新倒计时")
def l4_restore_ttl(sol):
    run(sol, """
        set_at_with_ttl(1, "k", "f", "v", 9)    -> ""        # [1, 10)
        set_at_with_ttl(1, "k", "g", "w", 3)    -> ""        # [1, 4)
        set_at_with_ttl(1, "k", "e", "x", 2)    -> ""        # [1, 3)：备份时已经过期
        backup(3)                               -> ""        # f 剩余 7，g 剩余 1
        get_at(20, "k", "f")                    -> ""        # 实时数据库里 f 在 10 就过期了
        restore(20, 3)                          -> ""        # f 的过期时刻变为 27，g 的变为 21
        get_at(20, "k", "g")                    -> "w"
        get_at(20, "k", "e")                    -> ""        # 不在备份里
        scan_at(20, "k")                        -> "f(v), g(w)"
        get_at(21, "k", "g")                    -> ""
        get_at(26, "k", "f")                    -> "v"
        get_at(27, "k", "f")                    -> ""
    """)


@case("Level 4", title="用不超过目标时间的最近一次备份")
def l4_restore_choice(sol):
    run(sol, """
        set_at(1, "k", "f", "one")      -> ""
        backup(3)                       -> ""
        set_at(5, "k", "f", "two")      -> ""
        backup(7)                       -> ""
        set_at(9, "k", "f", "three")    -> ""
        backup(12)                      -> ""
        set_at(13, "k", "f", "live")    -> ""
        restore(14, 13)                 -> ""
        get_at(14, "k", "f")            -> "three"    # 不超过 13 的最近一次备份是 ts=12
        restore(15, 12)                 -> ""
        get_at(15, "k", "f")            -> "three"    # 正好有 ts=12 的备份，用它
        restore(16, 11)                 -> ""
        get_at(16, "k", "f")            -> "two"      # 不超过 11 的最近一次备份是 ts=7
        restore(17, 7)                  -> ""
        get_at(17, "k", "f")            -> "two"      # 正好有 ts=7 的备份，用它
        restore(18, 6)                  -> ""
        get_at(18, "k", "f")            -> "one"      # 不超过 6 的最近一次备份是 ts=3
        restore(19, 3)                  -> ""
        get_at(19, "k", "f")            -> "one"      # 正好有 ts=3 的备份，用它
    """)


@case("Level 4", title="同一时间戳的第二次备份替换第一次")
def l4_backup_same_ts(sol):
    run(sol, """
        set_at(5, "k", "f", "first")    -> ""
        backup(5)                       -> ""
        set_at(5, "k", "f", "second")   -> ""         # 同一个时间戳上又写了一次
        set_at(5, "k", "g", "extra")    -> ""
        backup(5)                       -> ""         # 替换上一次 ts=5 的备份
        set_at(6, "k", "f", "live")     -> ""
        delete_at(6, "k", "g")          -> "true"
        restore(7, 5)                   -> ""
        get_at(7, "k", "f")             -> "second"   # ts=5 的备份是第二次做的那个
        get_at(7, "k", "g")             -> "extra"
        restore(8, 6)                   -> ""
        scan_at(8, "k")                 -> "f(second), g(extra)"   # 不超过 6 的最近一次备份仍是它
    """)


@case("Level 4", title="恢复后的状态就是备份本身")
def l4_restore_replaces(sol):
    run(sol, """
        set_at(1, "a", "x", "1")                -> ""
        set_at_with_ttl(1, "a", "y", "2", 3)    -> ""         # [1, 4)
        backup(2)                               -> ""         # a：x 永久，y 剩余 2
        set_at(3, "a", "z", "3")                -> ""         # 备份之后新增的字段
        set_at(3, "b", "x", "4")                -> ""         # 备份之后新增的 key
        delete_at(3, "a", "x")                  -> "true"     # 备份之后删掉的字段
        restore(10, 2)                          -> ""
        scan_at(10, "a")                        -> "x(1), y(2)"   # z 消失，x 回来，y 从 10 重新倒计时
        scan_at(10, "b")                        -> ""             # 备份时还没有 b
        get_at(11, "a", "y")                    -> "2"
        get_at(12, "a", "y")                    -> ""
    """)


@case("Level 4", title="恢复不会删除任何备份")
def l4_backups_kept(sol):
    run(sol, """
        set_at(1, "k", "f", "old")      -> ""
        backup(2)                       -> ""
        set_at(3, "k", "f", "new")      -> ""
        backup(4)                       -> ""
        restore(5, 2)                   -> ""         # 回到 ts=2 的备份
        get_at(5, "k", "f")             -> "old"
        restore(6, 4)                   -> ""
        get_at(6, "k", "f")             -> "new"      # 上一次恢复没有删掉 ts=4 的备份
        set_at(7, "k", "f", "newest")   -> ""
        backup(8)                       -> ""         # 恢复之后再做备份
        restore(9, 2)                   -> ""
        get_at(9, "k", "f")             -> "old"
        restore(10, 9)                  -> ""
        get_at(10, "k", "f")            -> "newest"   # 不超过 9 的最近一次备份是 ts=8
    """)


@case("Level 4", title="恢复出来的状态与备份互相独立")
def l4_restore_independent(sol):
    run(sol, """
        set_at_with_ttl(0, "k", "f", "v", 10)   -> ""         # [0, 10)
        set_at(0, "k", "g", "w")                -> ""
        backup(2)                               -> ""         # f 剩余 8
        restore(3, 2)                           -> ""         # f 到 11 过期
        set_at(4, "k", "g", "changed")          -> ""         # 修改恢复出来的状态
        delete_at(4, "k", "f")                  -> "true"
        set_at(4, "k", "h", "added")            -> ""
        restore(5, 2)                           -> ""         # 再恢复一次同一个备份
        scan_at(5, "k")                         -> "f(v), g(w)"   # 备份没有被上面的修改影响
        get_at(12, "k", "f")                    -> "v"        # 5 + 8 = 13 才过期
        get_at(13, "k", "f")                    -> ""
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=400, length=(15, 50))
