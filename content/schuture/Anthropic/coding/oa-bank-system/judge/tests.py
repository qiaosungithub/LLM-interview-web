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
    if isinstance(x, list):
        return "[" + ", ".join(_lit(v) for v in x) + "]"
    return repr(x)


def _replay(done, wants, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step."""
    lines = [f"bank.{m}({', '.join(_lit(a) for a in args)})" for m, args in done]
    lines[-1] = f"print({lines[-1]})"
    notes = [f"# -> {_lit(w)}" for w in wants] + [mark]
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = ["bank = Bank()"] + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
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
    """Lines like  withdraw(3, "A", 40) -> 60   # why  become (method, args, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        call, arrow, rest = line.strip().partition("->")
        assert arrow, f"missing '->' in {line!r}"
        expr, _, why = rest.partition("#")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(ast.literal_eval(a) for a in node.args)
        steps.append((node.func.id, args, ast.literal_eval(expr.strip()), why.strip()))
    return steps


def run(sol, script):
    """Runs the calls on a fresh Bank, checking each return value as soon as it comes back."""
    bank = sol.Bank()
    done, wants = [], []
    for method, args, want, why in _parse(script):
        done.append((method, args))
        fn = _method(bank, method)
        calling(_replay(done, wants))       # shown if the answer raises or hangs in this call
        try:
            got = fn(*args)
        except Exception:
            calling(_replay(done, wants, mark="# <- 这一步抛出异常"))
            raise
        if not _same(got, want):
            expect(got, want, call=_replay(done, wants), msg=why)
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
        create_account(0, "A")    -> True
        create_account(1, "A")    -> False   # "A" 已经是存活账户
        deposit(2, "A", 100)      -> 100
        withdraw(3, "A", 40)      -> 60
        withdraw(4, "A", 1000)    -> None    # 只有 60 可用
        deposit(5, "B", 10)       -> None    # "B" 从未被创建过
        withdraw(6, "B", 1)       -> None
    """)


@case("Level 1", title="重复创建返回 False，余额不变")
def l1_duplicate_create(sol):
    run(sol, """
        create_account(0, "A")    -> True
        deposit(1, "A", 50)       -> 50
        create_account(2, "A")    -> False   # "A" 已经是存活账户
        deposit(3, "A", 1)        -> 51      # 失败的 create_account 不能产生任何效果：余额仍是 50
        withdraw(4, "A", 51)      -> 0
        create_account(5, "A")    -> False   # 余额为 0 的账户仍然是存活账户
    """)


@case("Level 1", title="取出全部余额可以，多 1 不行")
def l1_withdraw_boundary(sol):
    run(sol, """
        create_account(0, "A")    -> True
        deposit(1, "A", 30)       -> 30
        withdraw(2, "A", 31)      -> None    # 31 > 30
        withdraw(2, "A", 100)     -> None
        deposit(3, "A", 5)        -> 35      # 失败的取款不能改变余额
        withdraw(4, "A", 35)      -> 0       # amount 等于余额是允许的
        withdraw(5, "A", 1)       -> None    # 余额为 0
        deposit(6, "A", 7)        -> 7
    """)


@case("Level 1", title="未创建的账户；id 区分大小写")
def l1_unknown_account(sol):
    run(sol, """
        deposit(0, "A", 5)        -> None    # "A" 还没有被创建：返回 None，不产生任何效果
        withdraw(0, "A", 5)       -> None
        create_account(1, "A")    -> True    # 前面的 deposit 不能顺带创建账户
        deposit(2, "a", 5)        -> None    # "a" 和 "A" 是不同的 id
        create_account(3, "a")    -> True
        deposit(4, "A", 5)        -> 5       # 新账户余额从 0 开始
        deposit(5, "a", 7)        -> 7
        withdraw(6, "a", 6)       -> 1
        deposit(7, "A", 1)        -> 6
    """)


@case("Level 1", title="多个账户、重复时间戳、大金额")
def l1_many_accounts(sol):
    run(sol, """
        create_account(5, "alice")             -> True
        create_account(5, "bob")               -> True
        create_account(5, "账户-1")            -> True
        deposit(5, "alice", 100)               -> 100
        deposit(5, "bob", 1)                   -> 1
        withdraw(5, "bob", 2)                  -> None
        withdraw(6, "alice", 99)               -> 1
        deposit(6, "bob", 1000000000000)       -> 1000000000001
        withdraw(7, "bob", 1000000000001)      -> 0
        deposit(8, "账户-1", 3)                -> 3
        deposit(8, "alice", 1)                 -> 2
    """)


@case("Level 1", cpu=2, title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=300, length=(5, 30), ids=["A", "B", "C"])


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, """
        create_account(0, "A")    -> True
        create_account(0, "B")    -> True
        create_account(0, "C")    -> True
        deposit(1, "A", 100)      -> 100
        deposit(2, "B", 100)      -> 100
        deposit(3, "C", 100)      -> 100
        withdraw(4, "A", 30)      -> 70      # A 的转出总额现在是 30
        withdraw(5, "B", 30)      -> 70      # B 的转出总额现在是 30，与 A 打平
        top_spenders(6, 2)        -> ["A(30)", "B(30)"]            # 30 分打平："A" 排在 "B" 前面
        top_spenders(6, 5)        -> ["A(30)", "B(30)", "C(0)"]    # 只有 3 个账户存在
        withdraw(7, "A", 1000)    -> None    # 失败：A 的转出总额保持 30
        top_spenders(8, 1)        -> ["A(30)"]
    """)


@case("Level 2", title="并列按 id 升序，与创建顺序无关")
def l2_ties_by_id(sol):
    run(sol, """
        create_account(0, "carol")   -> True
        create_account(0, "alice")   -> True
        create_account(0, "bob")     -> True
        top_spenders(1, 3)           -> ["alice(0)", "bob(0)", "carol(0)"]    # 都是 0：按 account_id 升序
        deposit(2, "carol", 50)      -> 50
        deposit(2, "bob", 50)        -> 50
        withdraw(3, "carol", 10)     -> 40
        withdraw(3, "bob", 10)       -> 40
        top_spenders(4, 3)           -> ["bob(10)", "carol(10)", "alice(0)"]  # 10 分打平："bob" 在 "carol" 前面
        withdraw(5, "carol", 1)      -> 39
        top_spenders(6, 2)           -> ["carol(11)", "bob(10)"]
    """)


@case("Level 2", title="只累计成功的取款，与余额无关")
def l2_outgoing_definition(sol):
    run(sol, """
        create_account(0, "A")    -> True
        create_account(0, "B")    -> True
        deposit(1, "A", 1000)     -> 1000
        deposit(1, "B", 20)       -> 20
        withdraw(2, "A", 5)       -> 995
        withdraw(2, "B", 4)       -> 16
        withdraw(3, "B", 4)       -> 12
        top_spenders(4, 2)        -> ["B(8)", "A(5)"]    # 转出总额是成功取款之和（4+4），不看余额
        withdraw(5, "A", 996)     -> None
        deposit(6, "A", 500)      -> 1495
        top_spenders(7, 2)        -> ["B(8)", "A(5)"]    # 失败的取款和存款都不计入
    """)


@case("Level 2", title="n 超过账户数、没有账户、格式")
def l2_n_and_format(sol):
    run(sol, """
        top_spenders(0, 1)                     -> []     # 还没有任何账户
        top_spenders(0, 3)                     -> []
        create_account(1, "ab")                -> True
        create_account(1, "b")                 -> True
        create_account(1, "abc")               -> True
        top_spenders(2, 3)                     -> ["ab(0)", "abc(0)", "b(0)"]    # 恰好 n 个
        top_spenders(2, 1)                     -> ["ab(0)"]
        top_spenders(2, 100)                   -> ["ab(0)", "abc(0)", "b(0)"]
        deposit(3, "b", 1000000000000)         -> 1000000000000
        withdraw(4, "b", 1000000000000)        -> 0
        top_spenders(5, 2)                     -> ["b(1000000000000)", "ab(0)"]
    """)


@case("Level 2", cpu=2, title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=300, length=(5, 30), ids=["A", "AB", "B", "C"])


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, """
        create_account(0, "A")                    -> True
        create_account(0, "B")                    -> True
        create_account(0, "C")                    -> True
        deposit(1, "A", 500)                      -> 500
        transfer(10, "A", "B", 200)               -> "transfer1"   # 保留 200；窗口在 10+1000=1010 关闭
        transfer(15, "A", "C", 250)               -> "transfer2"   # 保留 250；窗口在 15+1000=1015 关闭
        withdraw(20, "A", 100)                    -> None          # 可用余额 = 500-200-250 = 50 < 100
        accept_transfer(1010, "B", "transfer1")   -> True          # 最后可接受的时刻：1010 <= 1010
        accept_transfer(1010, "B", "transfer1")   -> False         # 已经完成过了
        accept_transfer(1011, "A", "transfer2")   -> False         # A 不是 transfer2 的接收方（"C" 才是）
        accept_transfer(1016, "C", "transfer2")   -> False         # 1016 > 1015：已过期
        withdraw(1300, "A", 300)                  -> 0             # transfer2 的保留额度已不再计入：可用余额 = 300
    """)


@case("Level 3", title="编号全局递增，失败不占号")
def l3_transfer_ids(sol):
    run(sol, """
        create_account(0, "A")         -> True
        create_account(0, "B")         -> True
        create_account(0, "C")         -> True
        deposit(1, "A", 100)           -> 100
        deposit(1, "C", 100)           -> 100
        transfer(2, "A", "B", 10)      -> "transfer1"
        transfer(3, "A", "A", 10)      -> None    # source_id 等于 target_id
        transfer(3, "X", "B", 10)      -> None    # source_id 不是存活账户
        transfer(3, "A", "X", 10)      -> None    # target_id 不是存活账户
        transfer(3, "B", "A", 1)       -> None    # B 的可用余额是 0
        transfer(4, "C", "A", 30)      -> "transfer2"   # 所有账户共用一个计数器，失败的调用不推进它
        transfer(5, "A", "C", 90)      -> "transfer3"   # A 的可用余额 = 100-10 = 90，恰好够
        transfer(6, "A", "C", 1)       -> None    # A 的可用余额已经是 0
    """)


@case("Level 3", title="保留额度不改变余额本身")
def l3_hold_keeps_balance(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        deposit(1, "A", 100)                   -> 100
        transfer(2, "A", "B", 70)              -> "transfer1"
        deposit(3, "A", 5)                     -> 105    # 待处理的转账不改变余额：100+5
        withdraw(4, "A", 36)                   -> None   # 可用余额 = 105-70 = 35
        withdraw(5, "A", 35)                   -> 70     # 恰好等于可用余额；返回的是余额 105-35
        withdraw(6, "B", 1)                    -> None   # 转账还没完成，B 的余额仍是 0
        deposit(7, "B", 3)                     -> 3
        accept_transfer(8, "B", "transfer1")   -> True
        withdraw(9, "B", 73)                   -> 0      # 完成后 B 的余额 = 3+70
        deposit(10, "A", 1)                    -> 1      # 完成后 A 的余额 = 70-70 = 0
    """)


@case("Level 3", title="t0+1000 仍保留，t0+1001 释放")
def l3_hold_window(sol):
    run(sol, """
        create_account(0, "A")                    -> True
        create_account(0, "B")                    -> True
        deposit(0, "A", 100)                      -> 100
        transfer(100, "A", "B", 60)               -> "transfer1"
        withdraw(1100, "A", 41)                   -> None   # 1100 = 100+1000 仍是待处理：可用余额 = 100-60 = 40
        transfer(1100, "A", "B", 41)              -> None   # transfer 同样只能用可用余额
        withdraw(1101, "A", 41)                   -> 59     # 1101 已过期：保留额度不再计入，不需要任何 accept 调用
        accept_transfer(1101, "B", "transfer1")   -> False  # 已过期
        deposit(1102, "B", 1)                     -> 1      # 过期不是完成：B 没有收到钱
        withdraw(1103, "A", 59)                   -> 0      # A 也没有被扣钱
    """)


@case("Level 3", title="创建当刻到 t0+1000 都能接收")
def l3_accept_window(sol):
    run(sol, """
        create_account(0, "A")                    -> True
        create_account(0, "B")                    -> True
        deposit(0, "A", 10)                       -> 10
        transfer(5, "A", "B", 4)                  -> "transfer1"
        accept_transfer(5, "B", "transfer1")      -> True    # 创建的同一时刻就是待处理状态
        transfer(6, "A", "B", 3)                  -> "transfer2"   # A 余额 10-4 = 6，已完成的 transfer1 不再保留额度
        accept_transfer(1006, "B", "transfer2")   -> True    # 1006 = 6+1000，两端都取得到
        transfer(1006, "A", "B", 2)               -> "transfer3"   # A 余额 10-4-3 = 3，已完成的转账不再保留额度
        accept_transfer(2007, "B", "transfer3")   -> False   # 2007 > 1006+1000
        withdraw(2007, "B", 7)                    -> 0       # B 收到了 4+3
        withdraw(2007, "A", 3)                    -> 0       # A 被扣了 4+3，余额 10-7 = 3
    """)


@case("Level 3", title="只有 target 能接收；未知编号")
def l3_accept_rejects(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        create_account(0, "C")                 -> True
        deposit(0, "A", 50)                    -> 50
        accept_transfer(1, "B", "transfer1")   -> False   # 还没有任何转账
        transfer(2, "A", "B", 20)              -> "transfer1"
        accept_transfer(3, "A", "transfer1")   -> False   # source 不能完成它
        accept_transfer(3, "C", "transfer1")   -> False   # 只有 target_id 可以
        accept_transfer(3, "B", "transfer2")   -> False   # 从未被 transfer 返回过
        accept_transfer(3, "B", "transfer0")   -> False
        accept_transfer(3, "B", "nope")        -> False
        withdraw(4, "B", 1)                    -> None    # 上面失败的调用都没有效果
        accept_transfer(5, "B", "transfer1")   -> True
        accept_transfer(6, "B", "transfer1")   -> False   # 已经完成过了
        withdraw(7, "B", 20)                   -> 0       # 钱只到账一次
    """)


@case("Level 3", title="接收后释放保留额度，只扣一次")
def l3_accept_releases_hold(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        deposit(0, "A", 100)                   -> 100
        transfer(1, "A", "B", 60)              -> "transfer1"
        accept_transfer(2, "B", "transfer1")   -> True
        withdraw(3, "A", 40)                   -> 0      # 余额 100-60 = 40；已完成的转账不再保留额度
        deposit(4, "A", 10)                    -> 10
        transfer(5, "A", "B", 10)              -> "transfer2"
    """)


@case("Level 3", title="多笔保留额度累加、各自过期")
def l3_multiple_holds(sol):
    run(sol, """
        create_account(0, "A")             -> True
        create_account(0, "B")             -> True
        create_account(0, "C")             -> True
        deposit(0, "A", 100)               -> 100
        transfer(0, "A", "B", 30)          -> "transfer1"
        transfer(500, "A", "C", 30)        -> "transfer2"
        transfer(500, "A", "B", 41)        -> None          # 可用余额 = 100-30-30 = 40
        withdraw(1000, "A", 41)            -> None          # 1000 = 0+1000：两笔都还在保留
        withdraw(1001, "A", 70)            -> 30            # transfer1 已过期：可用余额 = 100-30 = 70
        transfer(1500, "A", "B", 1)        -> None          # transfer2 在 1500 仍保留：30-30 = 0
        transfer(1501, "A", "B", 30)       -> "transfer3"   # transfer2 也过期了
    """)


@case("Level 3", title="转账不计入转出总额")
def l3_outgoing_unchanged(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        deposit(1, "A", 100)                   -> 100
        withdraw(2, "A", 10)                   -> 90
        transfer(3, "A", "B", 60)              -> "transfer1"
        top_spenders(4, 2)                     -> ["A(10)", "B(0)"]   # 转出总额只算成功的 withdraw
        accept_transfer(5, "B", "transfer1")   -> True
        top_spenders(6, 2)                     -> ["A(10)", "B(0)"]   # 完成转账也不改变它
        withdraw(7, "B", 15)                   -> 45
        top_spenders(8, 2)                     -> ["B(15)", "A(10)"]
    """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=600, length=(10, 40), ids=["A", "AB", "B", "C"])


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, """
        create_account(0, "A")                  -> True
        create_account(0, "B")                  -> True
        deposit(1, "A", 100)                    -> 100
        deposit(2, "B", 50)                     -> 50
        transfer(3, "B", "A", 20)               -> "transfer1"   # B 保留 20，待处理，接收方是 "A"
        merge_accounts(5, "A", "B")             -> True          # A：余额 100+50=150；transfer1 的 source 变为 "A"
        get_balance(6, "B", 4)                  -> None          # "B" 不是存活账户，任何 time_at 都一样
        get_balance(6, "A", 4)                  -> 100           # 合并前一刻 A 自己的余额
        get_balance(6, "A", 5)                  -> 150           # 合并本身发生的那个时间戳
        accept_transfer(10, "A", "transfer1")   -> True          # source 和 target 都是 "A"：净变化为 0
        get_balance(11, "A", 10)                -> 150
        create_account(12, "B")                 -> True          # "B" 的全新一代
        get_balance(13, "B", 4)                 -> None          # 4 早于这一代自己的创建时间（12）
        get_balance(13, "B", 12)                -> 0
        deposit(14, "B", 9)                     -> 9
        get_balance(15, "B", 14)                -> 9
        merge_accounts(16, "A", "A")            -> False         # a 和 b 是同一个账户
        merge_accounts(17, "A", "nope")         -> False         # "nope" 不是存活账户
    """)


@case("Level 4", title="合并失败时没有任何效果")
def l4_merge_rejected(sol):
    run(sol, """
        create_account(0, "A")         -> True
        create_account(0, "B")         -> True
        deposit(1, "A", 10)            -> 10
        deposit(1, "B", 20)            -> 20
        merge_accounts(2, "A", "A")    -> False
        merge_accounts(2, "A", "X")    -> False   # "X" 不是存活账户
        merge_accounts(2, "X", "B")    -> False   # a 不是存活账户：B 不能因此被注销
        merge_accounts(2, "X", "Y")    -> False
        deposit(3, "B", 1)             -> 21      # B 仍然存活，余额不变
        deposit(3, "A", 1)             -> 11
        get_balance(4, "B", 2)         -> 20
    """)


@case("Level 4", title="被合并的 id 不再存活")
def l4_merged_id_dead(sol):
    run(sol, """
        create_account(0, "A")         -> True
        create_account(0, "B")         -> True
        create_account(0, "C")         -> True
        deposit(1, "B", 40)            -> 40
        deposit(1, "C", 10)            -> 10
        merge_accounts(2, "A", "B")    -> True
        deposit(3, "B", 5)             -> None    # "B" 不再存活
        withdraw(3, "B", 1)            -> None
        transfer(3, "B", "C", 1)       -> None
        transfer(3, "C", "B", 1)       -> None
        merge_accounts(3, "A", "B")    -> False
        merge_accounts(3, "B", "C")    -> False
        get_balance(3, "B", 2)         -> None
        top_spenders(4, 5)             -> ["A(0)", "C(0)"]
        transfer(5, "C", "A", 1)       -> "transfer1"   # 失败的 transfer 没有推进计数器
    """)


@case("Level 4", title="余额和转出总额相加")
def l4_merge_combines(sol):
    run(sol, """
        create_account(0, "A")         -> True
        create_account(0, "B")         -> True
        create_account(0, "C")         -> True
        deposit(1, "A", 50)            -> 50
        deposit(1, "B", 80)            -> 80
        deposit(1, "C", 100)           -> 100
        withdraw(2, "A", 10)           -> 40
        withdraw(2, "B", 30)           -> 50
        withdraw(2, "C", 35)           -> 65
        merge_accounts(4, "A", "B")    -> True
        top_spenders(5, 3)             -> ["A(40)", "C(35)"]   # A 的转出总额 = 10+30，"B" 不再出现
        withdraw(6, "A", 91)           -> None                 # A 的余额 = 40+50 = 90
        withdraw(6, "A", 90)           -> 0
        top_spenders(7, 1)             -> ["A(130)"]
    """)


@case("Level 4", title="b 发出的待处理转账改由 a 承担")
def l4_merge_moves_holds(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        create_account(0, "C")                 -> True
        deposit(1, "B", 100)                   -> 100
        transfer(2, "B", "C", 60)              -> "transfer1"
        merge_accounts(3, "A", "B")            -> True
        withdraw(4, "A", 41)                   -> None   # A 余额 100，但 transfer1 的 60 现在保留在 A 名下
        accept_transfer(5, "C", "transfer1")   -> True   # C 仍然可以完成它；被扣钱的是 A
        get_balance(6, "A", 5)                 -> 40
        withdraw(7, "C", 60)                   -> 0
        transfer(8, "A", "C", 40)              -> "transfer2"
        create_account(9, "D")                 -> True
        merge_accounts(10, "D", "A")           -> True   # transfer2 的 source 变为 "D"
        withdraw(1008, "D", 1)                 -> None   # 窗口仍从 transfer2 的创建时刻 8 算起：1008 仍保留
        withdraw(1009, "D", 40)                -> 0      # 1009 已过期
    """)


@case("Level 4", title="发给 b 的待处理转账改由 a 接收")
def l4_merge_moves_incoming(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        create_account(0, "C")                 -> True
        deposit(1, "C", 100)                   -> 100
        transfer(2, "C", "B", 25)              -> "transfer1"
        merge_accounts(3, "A", "B")            -> True   # transfer1 的 target 变为 "A"
        accept_transfer(4, "B", "transfer1")   -> False  # "B" 不再是它的 target
        create_account(5, "B")                 -> True
        accept_transfer(6, "B", "transfer1")   -> False  # 新一代的 "B" 也不是：target 已经是 "A"
        accept_transfer(7, "A", "transfer1")   -> True
        withdraw(8, "A", 25)                   -> 0
        withdraw(8, "C", 75)                   -> 0
        get_balance(9, "B", 8)                 -> 0
    """)


@case("Level 4", title="合并后 source 与 target 相同")
def l4_self_transfer(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        deposit(1, "A", 70)                    -> 70
        transfer(2, "A", "B", 50)              -> "transfer1"
        merge_accounts(3, "A", "B")            -> True   # transfer1 变成 A -> A，仍是待处理
        withdraw(4, "A", 21)                   -> None   # 保留额度还在：可用余额 = 70-50 = 20
        accept_transfer(5, "A", "transfer1")   -> True   # 借记再贷记同一个账户，只释放保留额度
        get_balance(6, "A", 5)                 -> 70
        withdraw(7, "A", 70)                   -> 0
    """)


@case("Level 4", title="重新创建的 id 是全新的一代")
def l4_new_generation(sol):
    run(sol, """
        create_account(0, "A")         -> True
        create_account(0, "B")         -> True
        deposit(1, "B", 30)            -> 30
        withdraw(2, "B", 10)           -> 20
        merge_accounts(3, "A", "B")    -> True
        create_account(3, "B")         -> True                  # 合并的同一时刻就可以开启新一代
        top_spenders(4, 2)             -> ["A(10)", "B(0)"]     # 新一代的转出总额从 0 开始
        withdraw(5, "B", 1)            -> None                  # 新一代的余额从 0 开始
        get_balance(6, "B", 2)         -> None                  # 早于这一代的创建时间
        get_balance(6, "B", 3)         -> 0
        deposit(7, "B", 5)             -> 5
        get_balance(8, "A", 7)         -> 20                    # 新一代 B 的存款与 A 无关
        merge_accounts(9, "B", "A")    -> True                  # 反过来把 A 并进新一代 B
        get_balance(10, "B", 9)        -> 25
        get_balance(10, "B", 8)        -> 5                     # 合并之前只有这一代自己的 5
        top_spenders(11, 2)            -> ["B(10)"]
    """)


@case("Level 4", title="各类变化记在哪个时刻")
def l4_history_points(sol):
    run(sol, """
        create_account(10, "A")                 -> True
        create_account(10, "B")                 -> True
        deposit(10, "A", 100)                   -> 100
        get_balance(10, "A", 9)                 -> None   # 早于创建时间
        get_balance(11, "A", 10)                -> 100    # 时刻 10 的最后一次变化之后
        transfer(20, "A", "B", 40)              -> "transfer1"
        withdraw(30, "A", 50)                   -> 50
        accept_transfer(40, "B", "transfer1")   -> True
        get_balance(41, "A", 25)                -> 100    # 发起转账不改变余额
        get_balance(41, "A", 30)                -> 50
        get_balance(41, "A", 39)                -> 50     # 转账记在完成它的时刻 40
        get_balance(41, "A", 40)                -> 10
        get_balance(41, "B", 39)                -> 0
        get_balance(41, "B", 40)                -> 40
        get_balance(42, "A", 0)                 -> None   # time_at 可以往回走
        get_balance(42, "A", 1000000)           -> 10     # 晚于所有变化：当前余额
    """)


@case("Level 4", title="同一时刻多次变化取最后一次")
def l4_same_timestamp(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        deposit(5, "A", 10)                    -> 10
        withdraw(5, "A", 3)                    -> 7
        transfer(5, "A", "B", 2)               -> "transfer1"
        accept_transfer(5, "B", "transfer1")   -> True
        deposit(5, "A", 1)                     -> 6
        get_balance(6, "A", 5)                 -> 6    # 时刻 5 的最后一次变化之后
        get_balance(6, "A", 4)                 -> 0
        get_balance(6, "B", 5)                 -> 2
        merge_accounts(7, "A", "B")            -> True
        deposit(7, "A", 4)                     -> 12
        get_balance(8, "A", 7)                 -> 12   # 合并贷记和之后的存款都在时刻 7
        get_balance(8, "A", 6)                 -> 6
    """)


@case("Level 4", title="连续合并：转账跟着走")
def l4_chained_merges(sol):
    run(sol, """
        create_account(0, "A")                 -> True
        create_account(0, "B")                 -> True
        create_account(0, "C")                 -> True
        create_account(0, "D")                 -> True
        deposit(1, "D", 100)                   -> 100
        deposit(1, "B", 10)                    -> 10
        withdraw(2, "B", 4)                    -> 6
        transfer(3, "D", "B", 30)              -> "transfer1"
        merge_accounts(4, "A", "B")            -> True    # transfer1 的 target 变为 "A"
        merge_accounts(5, "C", "A")            -> True    # 再变为 "C"
        accept_transfer(6, "A", "transfer1")   -> False
        accept_transfer(6, "C", "transfer1")   -> True
        get_balance(7, "C", 6)                 -> 36
        top_spenders(8, 4)                     -> ["C(4)", "D(0)"]
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=600, length=(15, 60), ids=["A", "AB", "B", "C"])
