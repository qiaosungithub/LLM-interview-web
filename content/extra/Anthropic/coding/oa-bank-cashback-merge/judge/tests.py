import ast
import time

from judgelib import calling, case, expect, expect_true, rng
import ref

DAY = ref.DAY
SHOW = 18           # a replay lists at most this many calls


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    if isinstance(x, list):
        return "[" + ", ".join(_lit(v) for v in x) + "]"
    return repr(x)


def _ts(t):
    """A timestamp as the statement writes it: 5, 5 + DAY, 5 + 2 * DAY."""
    k, rest = divmod(t, DAY)
    if k == 0:
        return str(t)
    day = "DAY" if k == 1 else f"{k} * DAY"
    return f"{rest} + {day}" if rest else day


def _args(method, args):
    out = [_ts(a) if (i == 0 or (method == "get_balance" and i == 2)) and isinstance(a, int) else _lit(a)
           for i, a in enumerate(args)]
    return ", ".join(out)


def _replay(done, wants, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step."""
    lines = [f"bank.{m}({_args(m, args)})" for m, args in done]
    lines[-1] = f"print({lines[-1]})"
    notes = [f"# -> {_lit(w)}" for w in wants] + [mark]
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = (["DAY = 86_400_000"] if any("DAY" in s for s in lines) else []) + ["bank = Bank()"]
    if skipped:
        head.append(f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）")
    return "\n".join(head + [s.ljust(width) + n for s, n in zip(lines, notes)])


def _method(bank, name):
    fn = getattr(bank, name, None)
    expect_true(callable(fn), f"Bank 没有 {name} 方法（方法名要和题目一致）")
    return fn


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines like  deposit(4 + DAY, "alice", 0) -> 550   # why  become (method, args, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        call, arrow, rest = line.strip().partition("->")
        assert arrow, f"missing '->' in {line!r}"
        expr, _, why = rest.partition("#")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(eval(compile(ast.Expression(a), "<script>", "eval"), {"__builtins__": {}, "DAY": DAY})
                     for a in node.args)
        steps.append((node.func.id, args, ast.literal_eval(expr.strip()), why.strip()))
    return steps


def run(sol, script):
    """Runs the calls on a fresh Bank, checking each return value as soon as it comes back."""
    bank = sol.Bank()
    done, wants = [], []
    for method, args, want, why in _parse(script):
        done.append((method, args))
        fn = _method(bank, method)
        full = len(done)                    # a hand-written script is short: show all of it, so it can be pasted
        calling(_replay(done, wants, show=full))       # shown if the answer raises or hangs in this call
        try:
            got = fn(*args)
        except Exception:
            calling(_replay(done, wants, mark="# <- 这一步抛出异常", show=full))
            raise
        if not _same(got, want):
            expect(got, want, call=_replay(done, wants, show=full),
                   msg=why or f"第 {len(done)} 步 {method} 的返回值不对")
        wants.append(want)


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, ops):
    """Replays ops on a fresh answer and a fresh ref.Bank; None if every return value agrees."""
    bank, model, wants = sol.Bank(), ref.Bank(), []
    for k, (method, args) in enumerate(ops):
        want = getattr(model, method)(*args)
        fn = _method(bank, method)
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


def random_check(sol, level, sequences, length, ids):
    r = rng()
    for _ in range(sequences):
        ops = ref.random_ops(r, level, r.randint(*length), ids)
        bad = _outcome(sol, ops)
        if bad is None:
            continue
        ops, bad = _shrink(sol, ops, bad)
        k = bad["k"]
        if bad["exc"] is not None:
            calling(_replay(ops[: k + 1], bad["wants"], mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        expect(bad["got"], bad["want"], call=_replay(ops[: k + 1], bad["wants"]),
               msg=f"随机操作序列的第 {k + 1} 步返回值不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, """
        create_account(1, "alice")             -> True
        create_account(2, "alice")             -> False   # 已经有这个账户
        create_account(3, "bob")               -> True
        deposit(4, "carol", 500)               -> None    # 没有这个账户
        deposit(5, "alice", 1200)              -> 1200
        transfer(6, "alice", "bob", 1201)      -> None    # alice 只有 1200
        transfer(7, "alice", "bob", 1200)      -> 0       # 正好转光是允许的；返回的是转出方的新余额
        transfer(8, "bob", "bob", 10)          -> None    # 不能转给自己
        transfer(9, "bob", "carol", 10)        -> None    # 收款账户不存在
        deposit(10, "bob", 0)                  -> 1200    # 存 0 是合法的，结果就是当前余额
    """)


@case("Level 1", title="重复开户返回 False，余额不变")
def l1_duplicate_create(sol):
    run(sol, """
        create_account(1, "A")    -> True
        deposit(2, "A", 50)       -> 50
        create_account(3, "A")    -> False   # "A" 已经存在
        deposit(4, "A", 0)        -> 50      # 失败的 create_account 不改变任何状态：余额仍是 50
        create_account(5, "B")    -> True
        transfer(6, "A", "B", 50) -> 0
        create_account(7, "A")    -> False   # 余额为 0 的账户仍然存在
        deposit(8, "A", 1)        -> 1
    """)


@case("Level 1", title="转账：两边余额都要变")
def l1_transfer_balances(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        deposit(3, "A", 300)           -> 300
        deposit(4, "B", 20)            -> 20
        transfer(5, "A", "B", 120)     -> 180     # 返回转出方 A 的新余额
        deposit(6, "B", 0)             -> 140     # B 收到 120
        transfer(7, "B", "A", 141)     -> None    # B 只有 140
        transfer(8, "B", "A", 140)     -> 0       # 正好转光
        deposit(9, "A", 0)             -> 320
        transfer(10, "A", "B", 321)    -> None
        deposit(11, "B", 0)            -> 0       # 失败的转账不改变余额
    """)


@case("Level 1", title="转账失败的各种情况，都没有效果")
def l1_transfer_rejects(sol):
    run(sol, """
        create_account(1, "A")         -> True
        deposit(2, "A", 100)           -> 100
        transfer(3, "A", "X", 10)      -> None    # 收款账户不存在
        transfer(4, "X", "A", 10)      -> None    # 转出账户不存在
        transfer(5, "X", "Y", 0)       -> None    # 两个都不存在，金额 0 也一样
        transfer(6, "X", "X", 0)       -> None    # id 相同且不存在
        transfer(7, "A", "A", 0)       -> None    # 不能转给自己，金额 0 也不行
        transfer(8, "A", "A", 50)      -> None
        deposit(9, "A", 0)             -> 100     # 上面都没有改变余额
        deposit(10, "X", 5)            -> None    # 失败的转账也不会顺带开户
    """)


@case("Level 1", title="金额为 0 的存款和转账")
def l1_zero_amounts(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        deposit(3, "A", 0)             -> 0
        transfer(4, "A", "B", 0)       -> 0       # 余额 0，转 0 不超过余额：成功，返回 A 的余额
        deposit(5, "B", 7)             -> 7
        transfer(6, "B", "A", 0)       -> 7
        transfer(7, "A", "B", 1)       -> None    # A 的余额是 0
    """)


@case("Level 1", title="多个账户、id 区分大小写、大金额")
def l1_many_accounts(sol):
    run(sol, """
        create_account(1, "a")                     -> True
        create_account(2, "A")                     -> True    # "a" 和 "A" 是不同的 id
        create_account(3, "账户-1")                -> True
        deposit(4, "a", 5)                         -> 5
        deposit(5, "A", 1000000000000)             -> 1000000000000
        transfer(6, "A", "账户-1", 999999999999)   -> 1
        transfer(7, "a", "A", 5)                   -> 0
        deposit(8, "A", 0)                         -> 6
        deposit(9, "账户-1", 1)                    -> 1000000000000
    """)


@case("Level 1", cpu=2, title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=300, length=(5, 30), ids=["A", "B", "C"])


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, """
        create_account(1, "carol")             -> True
        create_account(2, "bob")               -> True
        create_account(3, "alice")             -> True
        deposit(4, "carol", 2000)              -> 2000
        deposit(5, "bob", 3000)                -> 3000
        top_spenders(6, 3)                     -> ["alice(0)", "bob(0)", "carol(0)"]   # 都是 0：按 id 升序
        transfer(7, "carol", "bob", 500)       -> 1500
        transfer(8, "carol", "alice", 1000)    -> 500
        transfer(9, "alice", "bob", 1000)      -> 0
        top_spenders(10, 2)                    -> ["carol(1500)", "alice(1000)"]
        top_spenders(11, 5)                    -> ["carol(1500)", "alice(1000)", "bob(0)"]   # 一共只有 3 个账户
        top_spenders(12, 0)                    -> []
    """)


@case("Level 2", title="并列按 id 升序（普通字符串比较）")
def l2_ties_by_id(sol):
    run(sol, """
        create_account(1, "b")         -> True
        create_account(2, "ab")        -> True
        create_account(3, "a")         -> True
        create_account(4, "B")         -> True
        top_spenders(5, 4)             -> ["B(0)", "a(0)", "ab(0)", "b(0)"]    # 大写字母排在小写前面；"a" < "ab" < "b"
        deposit(6, "b", 100)           -> 100
        deposit(7, "a", 100)           -> 100
        transfer(8, "b", "B", 30)      -> 70
        transfer(9, "a", "B", 30)      -> 70
        top_spenders(10, 3)            -> ["a(30)", "b(30)", "B(0)"]           # 30 打平："a" 在 "b" 前面
        transfer(11, "b", "ab", 1)     -> 69
        top_spenders(12, 2)            -> ["b(31)", "a(30)"]
    """)


@case("Level 2", title="只算成功转出的金额")
def l2_outgoing_definition(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        deposit(3, "A", 1000)          -> 1000
        transfer(4, "A", "B", 300)     -> 700
        top_spenders(5, 2)             -> ["A(300)", "B(0)"]    # 收到的钱不算支出
        transfer(6, "B", "A", 301)     -> None
        transfer(7, "A", "A", 5)       -> None
        transfer(8, "A", "X", 5)       -> None
        deposit(9, "B", 50)            -> 350
        top_spenders(10, 2)            -> ["A(300)", "B(0)"]    # 失败的转账和存款都不计入
        transfer(11, "B", "A", 0)      -> 350
        top_spenders(12, 2)            -> ["A(300)", "B(0)"]    # 转 0 计入 0
        transfer(13, "B", "A", 350)    -> 0
        top_spenders(14, 1)            -> ["B(350)"]
    """)


@case("Level 2", title="n 为 0、超过账户数、没有账户")
def l2_n_edges(sol):
    run(sol, """
        top_spenders(1, 0)             -> []      # 还没有任何账户
        top_spenders(2, 3)             -> []
        create_account(3, "x")         -> True
        top_spenders(4, 0)             -> []
        top_spenders(5, 1)             -> ["x(0)"]
        top_spenders(6, 100)           -> ["x(0)"]
        create_account(7, "y")         -> True
        deposit(8, "y", 1000000000000) -> 1000000000000
        transfer(9, "y", "x", 1000000000000) -> 0
        top_spenders(10, 2)            -> ["y(1000000000000)", "x(0)"]
    """)


@case("Level 2", cpu=2, title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=300, length=(5, 30), ids=["A", "AB", "B", "a"])


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, """
        create_account(1, "alice")                         -> True
        create_account(2, "bob")                           -> True
        deposit(3, "alice", 1000)                          -> 1000
        pay(4, "alice", 1001)                              -> None          # 余额不够；编号不前进
        pay(5, "alice", 450)                               -> "payment1"    # 余额 550；返现 450*2//100 = 9，在 5+DAY 到账
        deposit(6, "bob", 80)                              -> 80
        pay(7, "bob", 49)                                  -> "payment2"    # 余额 31；返现 49*2//100 = 0
        get_payment_status(8, "alice", "payment1")         -> "IN_PROGRESS"
        get_payment_status(9, "bob", "payment1")           -> None          # payment1 是 alice 的
        get_payment_status(10, "alice", "payment7")        -> None          # 没有这笔付款
        get_payment_status(11, "carol", "payment1")        -> None          # 没有这个账户
        top_spenders(12, 2)                                -> ["alice(450)", "bob(49)"]
        deposit(4 + DAY, "alice", 0)                       -> 550           # 还差 1 毫秒，返现没到
        get_payment_status(5 + DAY, "alice", "payment1")   -> "CASHBACK_RECEIVED"   # 恰好到点：先入账，再回答
        deposit(6 + DAY, "alice", 0)                       -> 559
        get_payment_status(8 + DAY, "bob", "payment2")     -> "CASHBACK_RECEIVED"   # 返现是 0，状态照样变
        top_spenders(9 + DAY, 2)                           -> ["alice(450)", "bob(49)"]   # 返现不改变支出总额
    """)


@case("Level 3", title="恰好到点的返现先于这次调用入账")
def l3_due_exactly(sol):
    run(sol, """
        create_account(1, "A")                 -> True
        create_account(2, "B")                 -> True
        deposit(3, "A", 1000)                  -> 1000
        pay(10, "A", 1000)                     -> "payment1"    # 余额 0；返现 20 在 10+DAY 到账
        deposit(9 + DAY, "A", 0)               -> 0             # 还差 1 毫秒
        deposit(10 + DAY, "A", 0)              -> 20            # 恰好到点：先入账，再存款
    """)
    run(sol, """
        create_account(1, "A")                 -> True
        create_account(2, "B")                 -> True
        deposit(3, "A", 500)                   -> 500
        pay(10, "A", 500)                      -> "payment1"    # 余额 0；返现 10 在 10+DAY 到账
        transfer(10 + DAY, "A", "B", 10)       -> 0             # 恰好到点：返现已经在余额里，可以转走
    """)
    run(sol, """
        create_account(1, "A")                 -> True
        deposit(2, "A", 300)                   -> 300
        pay(10, "A", 300)                      -> "payment1"    # 余额 0；返现 6 在 10+DAY 到账
        pay(10 + DAY, "A", 6)                  -> "payment2"    # 恰好到点：返现 6 已经可以用来付款
        get_payment_status(11 + DAY, "A", "payment1") -> "CASHBACK_RECEIVED"
        get_payment_status(12 + DAY, "A", "payment2") -> "IN_PROGRESS"
    """)


@case("Level 3", title="每个方法都先结算到点的返现")
def l3_every_method_settles(sol):
    run(sol, """
        create_account(1, "A")                 -> True
        create_account(2, "B")                 -> True
        deposit(3, "A", 1000)                  -> 1000
        pay(4, "A", 1000)                      -> "payment1"    # 返现 20 在 4+DAY 到账
        transfer(5 + DAY, "A", "B", 20)        -> 0             # 到点后第一次调用就是转账：返现必须已经入账
        deposit(6 + DAY, "B", 0)               -> 20
        pay(7 + DAY, "B", 20)                  -> "payment2"    # 返现 0 在 7+2*DAY 到账
        get_payment_status(8 + 2 * DAY, "B", "payment2") -> "CASHBACK_RECEIVED"   # 到点后第一次调用就是查询
        get_payment_status(9 + 2 * DAY, "A", "payment1") -> "CASHBACK_RECEIVED"
    """)
    run(sol, """
        create_account(1, "A")                 -> True
        deposit(2, "A", 5000)                  -> 5000
        pay(3, "A", 5000)                      -> "payment1"    # 返现 100 在 3+DAY 到账
        create_account(10 + DAY, "B")          -> True          # 这一次调用之前返现就已入账
        pay(11 + DAY, "A", 100)                -> "payment2"
        top_spenders(12 + DAY, 2)              -> ["A(5100)", "B(0)"]
    """)


@case("Level 3", title="返现 = amount*2//100，向下取整")
def l3_cashback_floor(sol):
    run(sol, """
        create_account(1, "A")         -> True
        deposit(2, "A", 10000)         -> 10000
        pay(3, "A", 49)                -> "payment1"    # 返现 0
        pay(4, "A", 50)                -> "payment2"    # 返现 1
        pay(5, "A", 99)                -> "payment3"    # 返现 1（1.98 向下取整）
        pay(6, "A", 125)               -> "payment4"    # 返现 2（2.5 向下取整）
        pay(7, "A", 175)               -> "payment5"    # 返现 3（3.5 向下取整）
        pay(8, "A", 1234)              -> "payment6"    # 返现 24（24.68 向下取整）
        deposit(9, "A", 0)             -> 8268
        deposit(3 + DAY, "A", 0)       -> 8268          # 返现 0
        deposit(4 + DAY, "A", 0)       -> 8269
        deposit(5 + DAY, "A", 0)       -> 8270
        deposit(6 + DAY, "A", 0)       -> 8272
        deposit(7 + DAY, "A", 0)       -> 8275
        deposit(8 + DAY, "A", 0)       -> 8299
    """)


@case("Level 3", title="付款编号全局共享，失败不占号")
def l3_payment_ids(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        deposit(3, "A", 100)           -> 100
        deposit(4, "B", 100)           -> 100
        pay(5, "X", 1)                 -> None          # 没有这个账户：编号不前进
        pay(6, "B", 101)               -> None          # 余额不够：编号不前进
        pay(7, "B", 10)                -> "payment1"
        pay(8, "A", 10)                -> "payment2"    # 所有账户共用一个计数器
        pay(9, "A", 91)                -> None          # A 只剩 90
        pay(10, "A", 90)               -> "payment3"    # 正好付光
        pay(11, "A", 0)                -> "payment4"    # 余额 0 时付 0 也是一笔成功的付款
        pay(12, "B", 90)               -> "payment5"
        deposit(13, "A", 0)            -> 0
        deposit(14, "B", 0)            -> 0
    """)


@case("Level 3", title="付款计入支出总额，返现不计")
def l3_outgoing(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        deposit(3, "A", 1000)          -> 1000
        deposit(4, "B", 1000)          -> 1000
        transfer(5, "A", "B", 100)     -> 900
        pay(6, "B", 150)               -> "payment1"
        top_spenders(7, 2)             -> ["B(150)", "A(100)"]    # 付款和转账都算支出
        pay(8, "A", 50)                -> "payment2"
        top_spenders(9, 2)             -> ["A(150)", "B(150)"]    # 150 打平，按 id
        pay(10, "A", 851)              -> None
        top_spenders(11, 2)            -> ["A(150)", "B(150)"]    # 失败的付款不算
        deposit(6 + DAY, "B", 0)       -> 953                     # 返现 3 到账
        deposit(8 + DAY, "A", 0)       -> 851                     # 返现 1 到账
        top_spenders(9 + DAY, 2)       -> ["A(150)", "B(150)"]    # 返现不改变支出总额
    """)


@case("Level 3", title="查付款状态返回 None 的情况")
def l3_status_none(sol):
    run(sol, """
        get_payment_status(1, "A", "payment1")     -> None      # 没有账户也没有付款
        create_account(2, "A")                     -> True
        create_account(3, "B")                     -> True
        get_payment_status(4, "A", "payment1")     -> None      # 还没有任何付款
        deposit(5, "A", 100)                       -> 100
        pay(6, "A", 60)                            -> "payment1"
        get_payment_status(7, "B", "payment1")     -> None      # 是 A 的付款，不是 B 的
        get_payment_status(8, "A", "payment2")     -> None
        get_payment_status(9, "A", "payment0")     -> None
        get_payment_status(10, "A", "nope")        -> None
        get_payment_status(11, "X", "payment1")    -> None
        get_payment_status(12, "A", "payment1")    -> "IN_PROGRESS"
        get_payment_status(6 + DAY, "B", "payment1") -> None    # 到账之后也一样
        get_payment_status(7 + DAY, "A", "payment1") -> "CASHBACK_RECEIVED"
    """)


@case("Level 3", title="多笔返现各自到点")
def l3_many_cashbacks(sol):
    run(sol, """
        create_account(1, "A")                     -> True
        create_account(2, "B")                     -> True
        deposit(3, "A", 2000)                      -> 2000
        deposit(4, "B", 2000)                      -> 2000
        pay(10, "A", 1000)                         -> "payment1"    # 返现 20 在 10+DAY
        pay(20, "B", 500)                          -> "payment2"    # 返现 10 在 20+DAY
        pay(30, "A", 250)                          -> "payment3"    # 返现 5 在 30+DAY
        deposit(15 + DAY, "A", 0)                  -> 770           # 只有 payment1 到了
        get_payment_status(16 + DAY, "A", "payment3") -> "IN_PROGRESS"
        deposit(25 + DAY, "B", 0)                  -> 1510
        deposit(29 + DAY, "A", 0)                  -> 770
        deposit(5 * DAY, "A", 0)                   -> 775           # 隔了很久之后，中间到点的返现都已入账
        get_payment_status(5 * DAY + 1, "A", "payment1") -> "CASHBACK_RECEIVED"
        get_payment_status(5 * DAY + 2, "B", "payment2") -> "CASHBACK_RECEIVED"
    """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=500, length=(10, 40), ids=["A", "AB", "B", "C"])


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, """
        create_account(1, "alice")                         -> True
        create_account(2, "bob")                           -> True
        deposit(3, "alice", 1000)                          -> 1000
        deposit(4, "bob", 800)                             -> 800
        pay(5, "bob", 300)                                 -> "payment1"   # bob 余额 500；返现 6，在 5+DAY 到账
        transfer(6, "alice", "bob", 100)                   -> 900          # bob 余额 600
        merge_accounts(7, "alice", "alice")                -> False        # 不能和自己合并
        merge_accounts(8, "alice", "carol")                -> False        # carol 不存在
        merge_accounts(9, "alice", "bob")                  -> True         # alice 余额 900 + 600 = 1500
        top_spenders(10, 5)                                -> ["alice(400)"]   # 100（转账）+ 300（bob 的付款）
        get_payment_status(11, "alice", "payment1")        -> "IN_PROGRESS"    # bob 的付款现在归 alice
        get_payment_status(12, "bob", "payment1")          -> None             # bob 不存在了
        deposit(13, "bob", 50)                             -> None
        get_balance(14, "alice", 8)                        -> 900          # 合并之前：alice 自己当时的余额
        get_balance(15, "alice", 9)                        -> 1500         # 从合并那一刻起
        get_balance(16, "bob", 8)                          -> 600          # 时刻 8，bob 还在
        get_balance(17, "bob", 9)                          -> None         # 时刻 9，bob 已被合并掉
        get_balance(18, "bob", 1)                          -> None         # bob 在时刻 2 才开户
        create_account(20, "bob")                          -> True         # 同一个 id，全新的账户
        get_balance(21, "bob", 20)                         -> 0
        get_balance(22, "bob", 15)                         -> None         # 时刻 15 没有叫 bob 的账户
        get_balance(23, "bob", 5)                          -> 500          # 旧 bob 在时刻 5 的余额
        deposit(6 + DAY, "alice", 0)                       -> 1506         # bob 那笔付款的返现进了 alice
        get_balance(7 + DAY, "alice", 4 + DAY)             -> 1500
        get_balance(8 + DAY, "alice", 5 + DAY)             -> 1506         # 返现记在到账时刻，虽然那一刻没有调用
        get_balance(9 + DAY, "bob", 5 + DAY)               -> 0            # 新 bob 没有拿到返现
        get_payment_status(10 + DAY, "alice", "payment1")  -> "CASHBACK_RECEIVED"
    """)


@case("Level 4", title="合并失败时没有任何效果")
def l4_merge_rejected(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        deposit(3, "A", 10)            -> 10
        deposit(4, "B", 20)            -> 20
        merge_accounts(5, "A", "A")    -> False
        merge_accounts(6, "A", "X")    -> False   # 账户 2 不存在
        merge_accounts(7, "X", "B")    -> False   # 账户 1 不存在：B 不能因此消失
        merge_accounts(8, "X", "Y")    -> False
        merge_accounts(9, "X", "X")    -> False
        deposit(10, "B", 0)            -> 20      # B 仍然存在，余额不变
        deposit(11, "A", 0)            -> 10
        top_spenders(12, 5)            -> ["A(0)", "B(0)"]
        get_balance(13, "B", 12)       -> 20
    """)


@case("Level 4", title="余额、支出总额相加，账户 2 消失")
def l4_merge_combines(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        create_account(3, "C")         -> True
        deposit(4, "A", 500)           -> 500
        deposit(5, "B", 800)           -> 800
        transfer(6, "A", "C", 100)     -> 400
        pay(7, "B", 300)               -> "payment1"
        transfer(8, "B", "C", 50)      -> 450
        merge_accounts(9, "A", "B")    -> True
        top_spenders(10, 5)            -> ["A(450)", "C(0)"]   # 100 + 300 + 50；B 不再出现
        deposit(11, "A", 0)            -> 850                  # 400 + 450
        deposit(12, "B", 5)            -> None                 # B 不存在了
        transfer(13, "B", "C", 0)      -> None
        transfer(14, "C", "B", 0)      -> None
        pay(15, "B", 0)                -> None
        merge_accounts(16, "B", "C")   -> False
        merge_accounts(17, "C", "B")   -> False
        transfer(18, "A", "C", 850)    -> 0
        pay(19, "C", 0)                -> "payment2"           # 失败的 pay 没有推进计数器
    """)


@case("Level 4", title="账户 2 未到账的返现进账户 1")
def l4_cashback_follows(sol):
    run(sol, """
        create_account(1, "A")                 -> True
        create_account(2, "B")                 -> True
        deposit(3, "B", 1000)                  -> 1000
        pay(4, "B", 1000)                      -> "payment1"    # 返现 20 在 4+DAY 到账
        merge_accounts(5, "A", "B")            -> True
        deposit(3 + DAY, "A", 0)               -> 0             # 还没到点
        deposit(4 + DAY, "A", 0)               -> 20            # 原定时刻到账，进的是 A
        get_payment_status(5 + DAY, "A", "payment1") -> "CASHBACK_RECEIVED"
        top_spenders(6 + DAY, 5)               -> ["A(1000)"]
    """)
    run(sol, """
        create_account(1, "A")                 -> True
        create_account(2, "B")                 -> True
        create_account(3, "C")                 -> True
        deposit(4, "C", 1000)                  -> 1000
        pay(5, "C", 250)                       -> "payment1"    # 返现 5 在 5+DAY 到账
        merge_accounts(6, "B", "C")            -> True
        merge_accounts(7, "A", "B")            -> True          # 连续两次合并：返现跟着钱走两次
        top_spenders(8, 9)                     -> ["A(250)"]
        deposit(5 + DAY, "A", 0)               -> 755
        get_payment_status(6 + DAY, "A", "payment1") -> "CASHBACK_RECEIVED"
    """)


@case("Level 4", title="合并前已到点的返现留在账户 2")
def l4_cashback_due_before_merge(sol):
    run(sol, """
        create_account(1, "A")                 -> True
        create_account(2, "B")                 -> True
        deposit(3, "B", 1000)                  -> 1000
        pay(4, "B", 500)                       -> "payment1"    # 返现 10 在 4+DAY 到账；之后很久没有调用
        merge_accounts(10 + DAY, "A", "B")     -> True          # 合并之前，返现已经在 4+DAY 进了 B
        get_balance(11 + DAY, "B", 3 + DAY)    -> 500
        get_balance(12 + DAY, "B", 4 + DAY)    -> 510           # 到账时刻 B 还在，返现记在 B 名下
        get_balance(13 + DAY, "A", 4 + DAY)    -> 0             # A 那时还没有拿到这笔钱
        get_balance(14 + DAY, "A", 10 + DAY)   -> 510
        deposit(15 + DAY, "A", 0)              -> 510
        get_payment_status(16 + DAY, "A", "payment1") -> "CASHBACK_RECEIVED"
    """)


@case("Level 4", title="账户 2 的付款从此属于账户 1")
def l4_payments_follow(sol):
    run(sol, """
        create_account(1, "A")                     -> True
        create_account(2, "B")                     -> True
        deposit(3, "A", 100)                       -> 100
        deposit(4, "B", 100)                       -> 100
        pay(5, "A", 50)                            -> "payment1"
        pay(6, "B", 50)                            -> "payment2"
        merge_accounts(7, "A", "B")                -> True
        get_payment_status(8, "A", "payment2")     -> "IN_PROGRESS"   # B 的付款现在归 A
        get_payment_status(9, "A", "payment1")     -> "IN_PROGRESS"   # A 自己的付款还在
        get_payment_status(10, "B", "payment2")    -> None
        create_account(11, "B")                    -> True
        get_payment_status(12, "B", "payment2")    -> None            # 新的 B 名下没有任何付款
        get_payment_status(6 + DAY, "A", "payment2") -> "CASHBACK_RECEIVED"
        get_payment_status(7 + DAY, "B", "payment2") -> None
    """)


@case("Level 4", title="同一个 id 重新开户是全新的账户")
def l4_reopen_fresh(sol):
    run(sol, """
        create_account(1, "A")                 -> True
        create_account(2, "B")                 -> True
        deposit(3, "B", 400)                   -> 400
        pay(4, "B", 400)                       -> "payment1"    # 返现 8 在 4+DAY 到账
        merge_accounts(5, "A", "B")            -> True
        create_account(6, "B")                 -> True          # 余额 0，支出总额 0，没有付款
        top_spenders(7, 5)                     -> ["A(400)", "B(0)"]
        pay(8, "B", 1)                         -> None
        deposit(4 + DAY, "B", 0)               -> 0             # 旧 B 的返现不进新 B
        deposit(5 + DAY, "A", 0)               -> 8
        merge_accounts(6 + DAY, "B", "A")      -> True          # 反过来把 A 并进新 B
        get_payment_status(7 + DAY, "B", "payment1") -> "CASHBACK_RECEIVED"
        top_spenders(8 + DAY, 5)               -> ["B(400)"]
    """)


@case("Level 4", title="被合并掉的账户仍可查合并前的余额")
def l4_balance_of_merged(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        deposit(3, "B", 70)            -> 70
        merge_accounts(4, "A", "B")    -> True
        get_balance(5, "B", 3)         -> 70      # B 现在不存在，但时刻 3 存在
        get_balance(6, "B", 2)         -> 0       # 开户那一刻
        get_balance(7, "B", 1)         -> None    # 还没开户
        get_balance(8, "B", 4)         -> None    # 被合并的那一刻起就不存在了
        get_balance(9, "B", 8)         -> None
        get_balance(10, "A", 3)        -> 0       # 合并不改写 A 的过去
        get_balance(11, "A", 4)        -> 70
    """)


@case("Level 4", title="存在时间是 [创建, 合并)")
def l4_existence_interval(sol):
    run(sol, """
        create_account(10, "A")        -> True
        create_account(20, "B")        -> True
        deposit(30, "B", 5)            -> 5
        get_balance(31, "A", 9)        -> None    # 早于创建
        get_balance(32, "A", 10)       -> 0       # 创建那一刻就存在
        get_balance(33, "B", 19)       -> None
        get_balance(34, "B", 20)       -> 0
        get_balance(35, "B", 29)       -> 0
        get_balance(36, "B", 30)       -> 5
        merge_accounts(40, "A", "B")   -> True
        get_balance(41, "B", 39)       -> 5       # 合并前 1 毫秒还在
        get_balance(42, "B", 40)       -> None    # 合并那一刻起不在
        get_balance(43, "A", 39)       -> 0
        get_balance(44, "A", 40)       -> 5
        get_balance(45, "A", 45)       -> 5       # time_at 可以等于 timestamp
    """)


@case("Level 4", title="同一 id 前后两个账户，中间空档为 None")
def l4_reopen_history(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        deposit(3, "B", 30)            -> 30
        merge_accounts(4, "A", "B")    -> True
        create_account(8, "B")         -> True
        deposit(9, "B", 7)             -> 7
        merge_accounts(10, "A", "B")   -> True
        create_account(12, "B")        -> True
        get_balance(13, "B", 3)        -> 30      # 第一个 B
        get_balance(14, "B", 4)        -> None    # 空档
        get_balance(15, "B", 7)        -> None
        get_balance(16, "B", 8)        -> 0       # 第二个 B
        get_balance(17, "B", 9)        -> 7
        get_balance(18, "B", 10)       -> None
        get_balance(19, "B", 11)       -> None
        get_balance(20, "B", 12)       -> 0       # 第三个 B
        get_balance(21, "A", 10)       -> 37
        get_balance(22, "A", 9)        -> 30
    """)


@case("Level 4", title="返现记在到账时刻，哪怕那一刻没有调用")
def l4_cashback_at_due(sol):
    run(sol, """
        create_account(1, "A")                 -> True
        deposit(2, "A", 1000)                  -> 1000
        pay(3, "A", 300)                       -> "payment1"    # 返现 6 在 3+DAY 到账
        get_balance(4, "A", 3)                 -> 700
        get_balance(5 + DAY, "A", 2 + DAY)     -> 700
        get_balance(6 + DAY, "A", 3 + DAY)     -> 706           # 到账时刻那一刻没有任何调用
        get_balance(7 + DAY, "A", 4 + DAY)     -> 706
    """)
    run(sol, """
        create_account(1, "A")                 -> True
        deposit(2, "A", 1000)                  -> 1000
        pay(3, "A", 500)                       -> "payment1"    # 返现 10 在 3+DAY 到账
        deposit(3 + 3 * DAY, "A", 1)           -> 511           # 两天后才有下一次调用
        get_balance(4 + 3 * DAY, "A", 2 + DAY) -> 500
        get_balance(5 + 3 * DAY, "A", 3 + DAY) -> 510
        get_balance(6 + 3 * DAY, "A", 2 + 3 * DAY) -> 510
        get_balance(7 + 3 * DAY, "A", 3 + 3 * DAY) -> 511
    """)


@case("Level 4", title="到账时刻恰好有调用：两者都算进去")
def l4_same_moment(sol):
    run(sol, """
        create_account(1, "A")                 -> True
        deposit(2, "A", 1000)                  -> 1000
        pay(3, "A", 1000)                      -> "payment1"    # 返现 20 在 3+DAY 到账
        deposit(3 + DAY, "A", 5)               -> 25            # 先入账返现，再存款
        get_balance(4 + DAY, "A", 2 + DAY)     -> 0
        get_balance(5 + DAY, "A", 3 + DAY)     -> 25            # 那一刻所有事情做完之后
    """)
    run(sol, """
        create_account(1, "A")                 -> True
        create_account(2, "B")                 -> True
        deposit(3, "B", 1000)                  -> 1000
        pay(4, "B", 500)                       -> "payment1"    # 返现 10 在 4+DAY 到账
        merge_accounts(4 + DAY, "A", "B")      -> True          # 合并恰好在到账时刻：返现先进 B，再一起并入 A
        get_balance(5 + DAY, "A", 4 + DAY)     -> 510
        get_balance(6 + DAY, "B", 3 + DAY)     -> 500
        get_balance(7 + DAY, "B", 4 + DAY)     -> None
        get_payment_status(8 + DAY, "A", "payment1") -> "CASHBACK_RECEIVED"
    """)


@case("Level 4", title="连续合并后的历史余额")
def l4_chained_history(sol):
    run(sol, """
        create_account(1, "A")         -> True
        create_account(2, "B")         -> True
        create_account(3, "C")         -> True
        deposit(4, "C", 1000)          -> 1000
        pay(5, "C", 250)               -> "payment1"    # 返现 5 在 5+DAY 到账
        merge_accounts(6, "B", "C")    -> True
        merge_accounts(7, "A", "B")    -> True
        get_balance(8, "C", 5)         -> 750
        get_balance(9, "B", 6)         -> 750
        get_balance(10, "B", 5)        -> 0
        get_balance(11, "A", 6)        -> 0
        get_balance(12, "A", 7)        -> 750
        get_balance(13, "C", 6)        -> None
        get_balance(5 + DAY, "A", 5 + DAY) -> 755
        get_balance(6 + DAY, "B", 5 + DAY) -> None
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=500, length=(15, 60), ids=["A", "AB", "B", "C"])
