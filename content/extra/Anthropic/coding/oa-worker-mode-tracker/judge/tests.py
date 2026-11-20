import ast
import copy
import itertools
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
        if len(x) > 10 and all(item == x[0] for item in x):
            return f"[{_lit(x[0])}] * {len(x)}"           # a long run of one value, still valid Python
        return "[" + ", ".join(_lit(item) for item in x) + "]"
    return repr(x)


def _code(method, args):
    return f"t.{method}({', '.join(_lit(a) for a in args)})"


def _replay(done, wants, mark="# <- 这一步", show=SHOW):
    """The calls made so far as code to paste into 自定义运行; the last one is the failing step."""
    lines = [_code(*step) for step in done]
    lines[-1] = f"print({lines[-1]})"
    notes = [f"# -> {_lit(w)}" for w in wants] + [mark]
    skipped = max(0, len(lines) - show)
    lines, notes = lines[skipped:], notes[skipped:]
    width = max(len(s) for s in lines) + 3
    head = ["t = ModeTracker()"]
    if skipped:
        head.append(f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）")
    return "\n".join(head + [s.ljust(width) + n for s, n in zip(lines, notes)])


def _method(t, name):
    fn = getattr(t, name, None)
    expect_true(callable(fn), f"ModeTracker 没有 {name} 方法（方法名要和题目一致）")
    return fn


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


def _describe(x):
    return "None" if x is None else f"{type(x).__name__} 类型的 {x!r}"


def _message(got, want, why):
    hint = ""
    if type(got) is not type(want):
        hint = f"期望 {_describe(want)}，得到 {_describe(got)}"
        if (got is None and want in (0, False)) or (want is None and got in (0, False)):
            hint += "（0、False 和 None 不是一回事）"
    return "；".join(part for part in (why, hint) if part)


# ---------------------------------------------------------------- hand-written sequences

def _arg(node):
    """A literal, or a list literal times a number such as [1] * 600."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _arg(node.left) * _arg(node.right)
    return ast.literal_eval(node)


def _parse(script):
    """Lines like  report("w1", [5, 3]) -> True   # why  become (method, args, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        if not line.strip():
            continue
        call, arrow, rest = line.strip().partition("->")
        assert arrow, f"missing '->' in {line!r}"
        expr, _, why = rest.partition("#")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(_arg(a) for a in node.args)
        steps.append((node.func.id, args, ast.literal_eval(expr.strip()), why.strip()))
    return steps


QUERIES = {"get_count", "get_worker_count", "get_mode", "top_values"}


def run(sol, script, setup=""):
    """Runs the calls on a fresh ModeTracker, checking each return value as soon as it comes back. The setup
    (the earlier levels' examples, which the statement continues from) contributes its calls that may change the
    state; they are made and shown in the replay with their expected values but not checked, since each earlier
    example is a case of its own. Its pure queries are left out so that the replay stays short."""
    t = sol.ModeTracker()
    steps = [step + (False,) for step in _parse(setup) if step[0] not in QUERIES] if setup else []
    steps += [step + (True,) for step in _parse(script)]
    done, wants = [], []
    for method, args, want, why, checked in steps:
        done.append((method, args))
        fn = _method(t, method)
        full = len(done)                    # a hand-written script is short: show all of it, so it can be pasted
        calling(_replay(done, wants, show=full))       # shown if the answer raises or hangs in this call
        try:
            got = fn(*copy.deepcopy(args))
        except Exception:
            calling(_replay(done, wants, mark="# <- 这一步抛出异常", show=full))
            raise
        if checked and not _same(got, want):
            why = why or f"第 {len(done)} 步 {_code(method, args)} 的返回值不对"
            expect(got, want, call=_replay(done, wants, show=full), msg=_message(got, want, why))
        wants.append(want)


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, ops):
    """Replays ops on a fresh answer and a fresh ref model; None if every return value agrees."""
    t, model, wants = sol.ModeTracker(), ref.ModeTracker(), []
    for k, (method, args) in enumerate(ops):
        want = getattr(model, method)(*copy.deepcopy(args))
        fn = _method(t, method)
        try:
            got = fn(*copy.deepcopy(args))
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        wants.append(want)
    return None


def _shrink(sol, ops, bad, budget=0.6):
    """Drops calls, then reported values, that the failure does not need, so the replay shown is short."""
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
    # then make the reported lists shorter: drop runs of values, then one of each of two different values
    # (a tie between them survives that)
    for i in range(len(ops)):
        if ops[i][0] != "report":
            continue

        def attempt(shorter):
            nonlocal ops, bad
            cand = ops[:i] + [("report", (ops[i][1][0], shorter))] + ops[i + 1:]
            out = _outcome(sol, cand) if shorter else None
            if out is not None and out["k"] == bad["k"] and type(out["exc"]) is type(bad["exc"]):
                ops, bad = cand, out
                return True
            return False

        size = max(1, len(ops[i][1][1]) // 2)
        while time.process_time() < stop:
            j, progress = 0, False
            while j < len(ops[i][1][1]) and time.process_time() < stop:
                values = ops[i][1][1]
                if attempt(values[:j] + values[j + size:]):
                    progress = True
                else:
                    j += size
            if size > 1:
                size //= 2
            elif not progress:
                break
        progress = True
        while progress and time.process_time() < stop:
            values, progress = ops[i][1][1], False
            for a, b in itertools.combinations(sorted(set(values)), 2):
                shorter = list(values)
                for x in (a, b):                            # the last ones, so the order of first sightings stays
                    del shorter[len(shorter) - 1 - shorter[::-1].index(x)]
                if attempt(shorter):
                    progress = True
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


# ---------------------------------------------------------------- the statement's examples (one chain)

EXAMPLE_1 = """
    add_worker("w1")                -> True
    add_worker("w1")                -> False     # 已经注册过
    add_worker("w2")                -> True
    report("w1", [5, 3, 5])         -> True
    report("w2", [3, 3, 9])         -> True
    report("w9", [1])               -> False     # w9 没有注册，这次上报不记录
    get_count(3)                    -> 3         # w1 报了 1 次，w2 报了 2 次
    get_count(5)                    -> 2
    get_count(1)                    -> 0
    get_worker_count("w1", 5)       -> 2
    get_worker_count("w2", 5)       -> 0         # 注册了，但没报过 5
    get_worker_count("w9", 5)       -> None      # 没有注册
"""

EXAMPLE_2 = """
    get_mode()                      -> 3
    top_values(2)                   -> ["3(3)", "5(2)"]
    top_values(10)                  -> ["3(3)", "5(2)", "9(1)"]
    report("w1", [5])               -> True
    get_mode()                      -> 3         # 3 和 5 都是 3 次，取较小的值
    top_values(1)                   -> ["3(3)"]
"""

EXAMPLE_3 = """
    remove_worker("w2")             -> 3         # 抹掉 w2 的 3 个值：3、3、9
    get_count(3)                    -> 1
    get_count(9)                    -> 0
    get_mode()                      -> 5
    top_values(10)                  -> ["5(3)", "3(1)"]     # 9 的计数回到 0，不再出现
    remove_worker("w2")             -> None      # 已经不在了
    report("w2", [9])               -> False
    add_worker("w2")                -> True      # 重新注册，是一个全新的 worker
    get_worker_count("w2", 3)       -> 0
"""

EXAMPLE_4 = """
    report("w2", [7, 7, 5])         -> True
    add_worker("w3")                -> True
    report("w3", [7])               -> True
    merge_workers("w1", "w2")       -> True
    get_worker_count("w1", 7)       -> 2         # 从 w2 并进来的
    get_worker_count("w1", 5)       -> 4         # 自己的 3 次 + 并进来的 1 次
    get_worker_count("w2", 7)       -> None      # w2 不再是注册的 worker
    get_count(7)                    -> 3         # 全局计数不受合并影响
    get_mode()                      -> 5         # 5 出现 4 次，7 出现 3 次
    merge_workers("w1", "w2")       -> False     # w2 不存在
    merge_workers("w1", "w1")       -> False     # 两个 id 相同
    remove_worker("w1")             -> 7         # 自己的 4 个值 + 并进来的 3 个值
    get_mode()                      -> 7         # 只剩 w3 报的那一个 7
    top_values(5)                   -> ["7(1)"]
"""


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, EXAMPLE_1)


@case("Level 1", title="列表里重复的值各算一次")
def l1_duplicates(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        report("a", [4, 4, 4])          -> True      # 列表里的每个元素各算一次
        report("a", [4])                -> True      # 多次上报会累加
        report("b", [4, 6])             -> True
        get_count(4)                    -> 5
        get_worker_count("a", 4)        -> 4
        get_worker_count("b", 4)        -> 1
        get_worker_count("a", 6)        -> 0         # 只有 b 报过 6
        get_count(6)                    -> 1
        get_count(7)                    -> 0
    """)


@case("Level 1", title="重复注册不清空数据")
def l1_add_twice(sol):
    run(sol, """
        add_worker("a")                 -> True
        report("a", [1, 2, 2])          -> True
        add_worker("a")                 -> False     # 已经注册过；失败的调用什么也不改
        get_worker_count("a", 2)        -> 2
        get_count(2)                    -> 2
        report("a", [2])                -> True
        get_worker_count("a", 2)        -> 3
        get_worker_count("a", 1)        -> 1
    """)


@case("Level 1", title="未注册的 worker：None 和 False")
def l1_unregistered(sol):
    run(sol, """
        get_worker_count("x", 1)        -> None      # 还没有任何 worker
        get_count(1)                    -> 0
        report("x", [1, 1])             -> False
        get_count(1)                    -> 0         # 失败的上报不记录
        add_worker("x")                 -> True
        get_worker_count("x", 1)        -> 0         # 之前那次失败的上报也没有记在 x 名下
        add_worker("X")                 -> True      # 区分大小写：这是另一个 worker
        report("X", [1])                -> True
        get_worker_count("x", 1)        -> 0
        get_worker_count("X", 1)        -> 1
        get_worker_count("y", 1)        -> None
        get_count(1)                    -> 1
    """)


@case("Level 1", title="0、负数和很大的值")
def l1_values(sol):
    run(sol, """
        add_worker("w")                                             -> True
        report("w", [0, -1, -1, 1000000000, -1000000000, 0, 0])     -> True
        get_count(0)                                                -> 3     # 0 也是一个普通的值
        get_count(-1)                                               -> 2
        get_count(1)                                                -> 0
        get_count(1000000000)                                       -> 1
        get_count(-1000000000)                                      -> 1
        get_worker_count("w", 0)                                    -> 3
        get_worker_count("w", -1000000000)                          -> 1
        add_worker("0")                                             -> True  # id 是字符串，可以长得像数字
        get_worker_count("0", 0)                                    -> 0
    """)


@case("Level 1", title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=200, length=(5, 40))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子（接着第 1 关）")
def l2_example(sol):
    run(sol, EXAMPLE_2, setup=EXAMPLE_1)


@case("Level 2", title="并列时按数值取小")
def l2_ties_numeric(sol):
    run(sol, """
        add_worker("a")                         -> True
        report("a", [10, 2, -4, 10, 2, -4])     -> True
        get_mode()                              -> -4        # 三个值都是 2 次：取数值最小的
        top_values(3)                           -> ["-4(2)", "2(2)", "10(2)"]     # 并列时按数值从小到大：-4 < 2 < 10（不是按字符串比较）
        report("a", [10])                       -> True
        get_mode()                              -> 10
        top_values(2)                           -> ["10(3)", "-4(2)"]
        report("a", [2])                        -> True
        get_mode()                              -> 2         # 2 和 10 都是 3 次
        top_values(5)                           -> ["2(3)", "10(3)", "-4(2)"]
    """)


@case("Level 2", title="没有数据；k 的边界")
def l2_top_k(sol):
    run(sol, """
        get_count(42)                           -> 0
        get_mode()                              -> None      # 还没有任何数据；查询计数不会让值出现
        top_values(1)                           -> []
        add_worker("a")                         -> True
        get_worker_count("a", 42)               -> 0
        get_mode()                              -> None      # 有 worker 但没有数据
        top_values(5)                           -> []
        report("b", [1])                        -> False
        get_mode()                              -> None      # 失败的上报不算数据
        report("a", [7, 8, 8, 9, 9, 9])         -> True
        top_values(1)                           -> ["9(3)"]
        top_values(3)                           -> ["9(3)", "8(2)", "7(1)"]       # k 恰好等于值的个数
        top_values(1000)                        -> ["9(3)", "8(2)", "7(1)"]       # 不足 k 个时全部返回
        get_mode()                              -> 9
    """)


@case("Level 2", title="计数是所有 worker 的总和")
def l2_global_sum(sol):
    run(sol, """
        add_worker("a")                         -> True
        add_worker("b")                         -> True
        report("a", [5, 5, 5])                  -> True
        report("b", [6, 6])                     -> True
        report("a", [6])                        -> True
        get_mode()                              -> 5         # 5 和 6 都是 3 次（6 来自两个 worker），取较小的
        report("b", [6])                        -> True
        get_mode()                              -> 6         # 6 现在是 4 次
        top_values(2)                           -> ["6(4)", "5(3)"]
        report("a", [-1000000000])              -> True
        top_values(5)                           -> ["6(4)", "5(3)", "-1000000000(1)"]
    """)


@case("Level 2", title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=300, length=(5, 40))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子（接着第 2 关）")
def l3_example(sol):
    run(sol, EXAMPLE_3, setup=EXAMPLE_1 + EXAMPLE_2)


@case("Level 3", title="移除后众数要重新找")
def l3_mode_after_removal(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        report("a", [1, 1, 1])          -> True
        report("b", [2, 2])             -> True
        get_mode()                      -> 1
        remove_worker("a")              -> 3
        get_mode()                      -> 2         # 原来的众数被整个抹掉了
        top_values(3)                   -> ["2(2)"]
        add_worker("c")                 -> True
        report("c", [3, 3, 2])          -> True      # 2 有 3 次，3 有 2 次
        add_worker("d")                 -> True
        report("d", [3, 3])             -> True      # 3 有 4 次
        get_mode()                      -> 3
        remove_worker("d")              -> 2
        get_mode()                      -> 2         # 3 降回 2 次，还在，但不再是第一
        top_values(2)                   -> ["2(3)", "3(2)"]
    """)


@case("Level 3", title="计数回到 0 的值消失")
def l3_zero_counts_vanish(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        report("a", [4, 4])             -> True
        report("b", [5, 4])             -> True
        remove_worker("b")              -> 2
        get_count(5)                    -> 0
        get_count(4)                    -> 2
        top_values(5)                   -> ["4(2)"]      # 计数回到 0 的值不再出现
        remove_worker("a")              -> 2
        get_mode()                      -> None          # 什么数据都没有了
        top_values(5)                   -> []
    """)


@case("Level 3", title="返回抹掉的值的个数")
def l3_remove_count(sol):
    run(sol, """
        add_worker("a")                 -> True
        report("a", [1, 1, 2])          -> True
        report("a", [1])                -> True
        add_worker("b")                 -> True
        report("b", [1])                -> True
        remove_worker("a")              -> 4         # 按上报的次数算，不是不同值的个数
        get_count(1)                    -> 1
        get_count(2)                    -> 0
        remove_worker("b")              -> 1
    """)


@case("Level 3", title="0 和 None 不是一回事")
def l3_none_vs_zero(sol):
    run(sol, """
        remove_worker("a")              -> None      # 没有注册
        add_worker("a")                 -> True
        remove_worker("a")              -> 0         # 注册了但没报过数据：0，不是 None
        remove_worker("a")              -> None      # 已经移除了
        add_worker("a")                 -> True
        report("a", [3])                -> True
        remove_worker("a")              -> 1
    """)


@case("Level 3", title="移除后再注册是全新的 worker")
def l3_readd_is_fresh(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        report("a", [1, 1])             -> True
        report("b", [1])                -> True
        remove_worker("a")              -> 2
        report("a", [1])                -> False     # 移除之后就不是注册的 worker 了
        get_worker_count("a", 1)        -> None
        add_worker("a")                 -> True      # 重新注册：全新的 worker，没有任何数据
        get_worker_count("a", 1)        -> 0
        get_count(1)                    -> 1
        report("a", [2])                -> True
        remove_worker("a")              -> 1         # 只抹掉重新注册之后报的
        get_count(1)                    -> 1
        get_mode()                      -> 1
    """)


@case("Level 3", title="大量上报之后移除，众数仍要对")
def l3_many_changes_then_removal(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        add_worker("c")                 -> True
        report("b", [1] * 300)          -> True
        report("a", [1] * 600)          -> True
        report("c", [2] * 500)          -> True      # 计数变化了很多次（用懒删除堆时，堆多半已经重建过）
        get_mode()                      -> 1         # 1 有 900 次，2 有 500 次
        remove_worker("b")              -> 300
        get_mode()                      -> 1         # 1 降到 600 次，仍然多于 2 的 500 次
        top_values(3)                   -> ["1(600)", "2(500)"]
        report("c", [2] * 100)          -> True
        get_mode()                      -> 1         # 都是 600 次，取较小的值
        report("c", [2])                -> True
        get_mode()                      -> 2
        remove_worker("c")              -> 601
        get_mode()                      -> 1
        remove_worker("a")              -> 600
        get_mode()                      -> None
    """)


@case("Level 3", title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=400, length=(10, 60))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子（接着第 3 关）")
def l4_example(sol):
    run(sol, EXAMPLE_4, setup=EXAMPLE_1 + EXAMPLE_2 + EXAMPLE_3)


@case("Level 4", title="合并时次数相加，source 注销")
def l4_merge_adds_counts(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        report("a", [1, 1, 2])          -> True
        report("b", [1, 3])             -> True
        merge_workers("a", "b")         -> True
        get_worker_count("a", 1)        -> 3         # 两边的次数相加
        get_worker_count("a", 2)        -> 1
        get_worker_count("a", 3)        -> 1
        get_worker_count("b", 1)        -> None      # b 不再是注册的 worker
        report("b", [1])                -> False
        remove_worker("b")              -> None
        remove_worker("a")              -> 5         # 自己的 3 个值 + 并进来的 2 个值
        get_mode()                      -> None
    """)


@case("Level 4", title="合并不改变全局计数")
def l4_global_unchanged(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        report("a", [5, 5])             -> True
        report("b", [6, 6, 6])          -> True
        merge_workers("a", "b")         -> True
        get_count(6)                    -> 3         # 合并不改变全局计数
        get_count(5)                    -> 2
        get_mode()                      -> 6
        top_values(3)                   -> ["6(3)", "5(2)"]
        remove_worker("a")              -> 5
        get_count(6)                    -> 0
    """)


@case("Level 4", title="合并失败的情况")
def l4_merge_failures(sol):
    run(sol, """
        add_worker("a")                 -> True
        report("a", [1])                -> True
        merge_workers("a", "a")         -> False     # 两个 id 相同
        merge_workers("a", "x")         -> False     # source_id 没有注册
        merge_workers("x", "a")         -> False     # target_id 没有注册
        get_worker_count("a", 1)        -> 1         # 失败的合并什么也不改
        get_worker_count("x", 1)        -> None
        get_count(1)                    -> 1
        remove_worker("a")              -> 1
        get_count(1)                    -> 0
    """)


@case("Level 4", title="移除 target 时并进来的一起抹掉")
def l4_remove_after_merge(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        add_worker("c")                 -> True
        report("a", [1])                -> True
        report("b", [1, 2])             -> True
        report("c", [2])                -> True
        merge_workers("a", "b")         -> True
        remove_worker("a")              -> 3         # 并进来的数据一起抹掉
        get_count(1)                    -> 0
        get_count(2)                    -> 1
        get_mode()                      -> 2
        top_values(3)                   -> ["2(1)"]
        remove_worker("b")              -> None
    """)


@case("Level 4", title="source 的 id 可以重新注册")
def l4_source_readd(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        report("b", [4, 4])             -> True
        merge_workers("a", "b")         -> True
        add_worker("b")                 -> True      # b 的 id 空出来了：全新的 worker
        get_worker_count("b", 4)        -> 0
        report("b", [4])                -> True
        get_worker_count("a", 4)        -> 2
        merge_workers("b", "a")         -> True      # 反方向再合并一次
        get_worker_count("b", 4)        -> 3
        get_worker_count("a", 4)        -> None
        remove_worker("b")              -> 3
        get_count(4)                    -> 0
    """)


@case("Level 4", title="连续合并")
def l4_chain(sol):
    run(sol, """
        add_worker("a")                 -> True
        add_worker("b")                 -> True
        add_worker("c")                 -> True
        report("a", [1])                -> True
        report("b", [2, 2])             -> True
        report("c", [3, 3, 3])          -> True
        merge_workers("b", "c")         -> True
        merge_workers("a", "b")         -> True      # a 拿到 b 的全部数据，包括 b 从 c 并进来的
        get_worker_count("a", 3)        -> 3
        get_worker_count("c", 3)        -> None
        get_mode()                      -> 3
        remove_worker("a")              -> 6
        get_mode()                      -> None
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=500, length=(10, 70))
