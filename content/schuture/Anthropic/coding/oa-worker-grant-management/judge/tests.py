import ast
import time

from judgelib import calling, case, expect, expect_true, rng
import ref

SHOW = 18           # a replay lists at most this many steps
UNCHECKED = {"set_double_pay"}      # returns None; its return value is not checked


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    if isinstance(x, list):
        return "[" + ", ".join(_lit(v) for v in x) + "]"
    return repr(x)


def _code(method, args):
    return f"w.{method}({', '.join(_lit(a) for a in args)})"


def _render(steps, current, mark="# <- 这一步"):
    """steps: (code, note) done so far; current: the call being checked."""
    lines = steps + [(current if current.startswith("w.set_double_pay") else f"print({current})", mark)]
    skipped = max(0, len(lines) - SHOW)
    lines = lines[skipped:]
    width = max(len(c) for c, _ in lines) + 3
    top = ["w = Workforce()"] + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(top + [(c.ljust(width) + n) if n else c for c, n in lines])


def _note(method, want):
    return "" if method in UNCHECKED else "# -> " + _lit(want)


def _method(obj, name):
    fn = getattr(obj, name, None)
    expect_true(callable(fn), f"Workforce 没有 {name} 方法（方法名要和题目一致）")
    return fn


def _norm(x):
    return list(x) if isinstance(x, tuple) else x


def _same(got, want):
    try:
        return want is None and got is None or want is not None and got is not None and bool(got == want)
    except Exception:
        return False


class Session:
    """Runs calls on one Workforce, checking each return value as soon as it comes back."""

    def __init__(self, sol):
        calling("w = Workforce()")
        self.w = sol.Workforce()
        self.steps = []

    def call(self, method, args, want=None, why=""):
        code = _code(method, args)
        fn = _method(self.w, method)
        calling(_render(self.steps, code))      # shown if the answer raises or hangs in this call
        try:
            got = _norm(fn(*args))
        except Exception:
            calling(_render(self.steps, code, "# <- 这一步抛出异常"))
            raise
        if method not in UNCHECKED and not _same(got, want):
            expect(got, want, call=_render(self.steps, code), msg=why or ("" if isinstance(want, list) else "返回值不对"))
        self.steps.append((code, _note(method, want)))


def _parse(script):
    """Lines like  calculate_pay("alice", 0, 100) -> 860   # why  become (method, args, want, why);
    a line without '->' is a call whose return value is not checked."""
    steps = []
    for line in script.strip().splitlines():
        call, arrow, rest = line.strip().partition(" -> ")
        if not arrow:
            call, _, rest = call.partition("  #")
        expr, _, why = rest.partition("  #") if arrow else ("None", "", rest)
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(ast.literal_eval(a) for a in node.args)
        assert arrow or node.func.id in UNCHECKED, line
        steps.append((node.func.id, args, ast.literal_eval(expr.strip()), why.strip()))
    return steps


def run(sol, script):
    s = Session(sol)
    for method, args, want, why in _parse(script):
        s.call(method, args, want, why)
    return s


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, ops):
    """Replays ops on a fresh answer and a fresh ref model; None if every return value agrees."""
    calling("w = Workforce()")
    w, model = sol.Workforce(), ref.Workforce()
    wants = []
    for k, (method, args) in enumerate(ops):
        want = getattr(model, method)(*args)
        fn = _method(w, method)
        try:
            got = _norm(fn(*args))
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if method not in UNCHECKED and not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        wants.append(want)
    return None


def _shrink(sol, ops, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short. Any subsequence still has
    non-decreasing clock_in/clock_out timestamps."""
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
        steps = [(_code(*op), _note(op[0], want)) for op, want in zip(ops[:-1], bad["wants"])]
        current = _code(*ops[-1])
        if bad["exc"] is not None:
            calling(_render(steps, current, "# <- 这一步抛出异常"))
            raise bad["exc"]
        expect(bad["got"], bad["want"], call=_render(steps, current),
               msg=f"随机操作序列的第 {len(ops)} 步返回值不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, """
        add_worker("alice", "Engineer", 15) -> True
        add_worker("alice", "Manager", 30) -> False   # "alice" 已经注册过
        clock_in("alice", 100) -> True
        clock_in("alice", 105) -> False   # 已经有一个打开的时段
        clock_out("alice", 140) -> True
        clock_out("alice", 150) -> False   # 没有打开的时段
        clock_in("bob", 160) -> False   # "bob" 从未注册
    """)


@case("Level 1", title="多个员工；首尾相接和长度为 0 的时段")
def l1_sessions(sol):
    run(sol, """
        add_worker("alice", "Engineer", 15) -> True
        add_worker("bob", "Designer", 20) -> True
        clock_out("alice", 0) -> False   # 还没有打开的时段
        clock_in("alice", 0) -> True
        clock_in("bob", 0) -> True   # 不同员工的时段互不影响
        clock_out("alice", 5) -> True
        clock_in("alice", 5) -> True   # 紧接着上一段再次进入
        clock_out("bob", 5) -> True
        clock_out("bob", 6) -> False
        clock_out("alice", 9) -> True
        clock_in("alice", 9) -> True
        clock_out("alice", 9) -> True   # 同一时间戳进出：长度为 0 的时段
        clock_out("alice", 9) -> False
        clock_out("carol", 10) -> False   # carol 没有注册
        clock_in("carol", 10) -> False
        add_worker("carol", "Engineer", 1) -> True
        clock_in("carol", 10) -> True
        clock_in("bob", 12) -> True
    """)


@case("Level 1", title="重复注册失败，没有任何效果")
def l1_duplicate_add(sol):
    run(sol, """
        add_worker("alice", "Engineer", 15) -> True
        clock_in("alice", 3) -> True
        add_worker("alice", "Engineer", 15) -> False   # 参数完全相同也算重复注册
        add_worker("alice", "Manager", 30) -> False
        clock_in("alice", 4) -> False   # 失败的 add_worker 没有效果：时段仍然打开
        clock_out("alice", 8) -> True
        add_worker("Alice", "Engineer", 15) -> True   # 不同的字符串是不同的 worker_id
        add_worker("王五", "工程师", 12) -> True
        clock_in("王五", 8) -> True
        clock_out("Alice", 8) -> False
        clock_out("王五", 9) -> True
    """)


@case("Level 1", cpu=2, title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=250, length=(5, 30))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, """
        add_worker("alice", "Engineer", 15) -> True
        add_worker("bob", "Engineer", 20) -> True
        add_worker("cy", "Designer", 15) -> True
        clock_in("alice", 0) -> True
        clock_out("alice", 40) -> True
        clock_in("bob", 40) -> True
        clock_out("bob", 65) -> True
        clock_in("cy", 65) -> True
        clock_out("cy", 105) -> True
        get_total_work_time("alice") -> 40
        get_total_work_time("dee") -> None   # 没有注册
        top_k_workers(2) -> ["alice", "cy"]   # 同为 40："alice" 在 "cy" 前面
        top_k_workers(10) -> ["alice", "cy", "bob"]   # 只注册了 3 个员工
        clock_in("bob", 110) -> True
        top_k_workers(1) -> ["alice"]   # bob 打开的时段仍然算 0
    """)


@case("Level 2", title="并列时按 worker_id 升序")
def l2_ties_by_id(sol):
    run(sol, """
        add_worker("cy", "Engineer", 1) -> True
        add_worker("bo", "Engineer", 1) -> True
        add_worker("alice", "Engineer", 1) -> True
        add_worker("al", "Engineer", 1) -> True
        top_k_workers(4) -> ["al", "alice", "bo", "cy"]   # 都是 0：按 worker_id 升序，和注册顺序无关
        clock_in("cy", 0) -> True
        clock_out("cy", 10) -> True
        clock_in("al", 10) -> True
        clock_in("bo", 10) -> True
        clock_out("bo", 15) -> True
        clock_out("al", 20) -> True
        top_k_workers(2) -> ["al", "cy"]   # al 和 cy 都是 10，"al" 在前
        top_k_workers(3) -> ["al", "cy", "bo"]
        top_k_workers(5) -> ["al", "cy", "bo", "alice"]
        clock_in("alice", 20) -> True
        clock_out("alice", 35) -> True
        top_k_workers(4) -> ["alice", "al", "cy", "bo"]
    """)


@case("Level 2", title="只累计已关闭的时段")
def l2_closed_only(sol):
    run(sol, """
        add_worker("alice", "Engineer", 15) -> True
        add_worker("bob", "Engineer", 20) -> True
        get_total_work_time("alice") -> 0   # 已注册但还没有时段
        clock_in("alice", 10) -> True
        get_total_work_time("alice") -> 0   # 打开的时段在关闭前不计入
        clock_out("alice", 10) -> True
        get_total_work_time("alice") -> 0   # 长度为 0 的时段
        clock_in("alice", 10) -> True
        clock_out("alice", 17) -> True
        clock_in("bob", 17) -> True
        clock_in("alice", 20) -> True
        clock_out("alice", 23) -> True
        get_total_work_time("alice") -> 10   # 7 + 3：所有已关闭时段的长度之和
        clock_in("alice", 30) -> True
        get_total_work_time("bob") -> 0
        top_k_workers(2) -> ["alice", "bob"]
        clock_out("bob", 40) -> True
        get_total_work_time("bob") -> 23   # [17, 40)
        top_k_workers(2) -> ["bob", "alice"]
        get_total_work_time("alice") -> 10   # alice 在 30 打开的时段还没关闭
        clock_out("alice", 45) -> True
        get_total_work_time("alice") -> 25
        top_k_workers(1) -> ["alice"]
    """)


@case("Level 2", title="k 大于人数；没有员工")
def l2_top_k_sizes(sol):
    run(sol, """
        top_k_workers(1) -> []   # 还没有注册任何员工
        top_k_workers(3) -> []
        add_worker("zed", "Engineer", 5) -> True
        top_k_workers(1) -> ["zed"]
        top_k_workers(100) -> ["zed"]
        add_worker("zed", "Manager", 9) -> False
        top_k_workers(5) -> ["zed"]   # 重复注册失败，不会出现两次
        get_total_work_time("Zed") -> None
        add_worker("m", "Engineer", 5) -> True
        add_worker("a", "Engineer", 5) -> True
        top_k_workers(5) -> ["a", "m", "zed"]
        clock_in("zed", 1000000000) -> True
        clock_out("zed", 1700000000) -> True
        get_total_work_time("zed") -> 700000000
        top_k_workers(3) -> ["zed", "a", "m"]
        top_k_workers(1) -> ["zed"]
    """)


@case("Level 2", cpu=2, title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=250, length=(5, 35))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, """
        add_worker("alice", "Engineer", 10) -> True
        clock_in("alice", 0) -> True
        clock_out("alice", 50) -> True
        set_promotion("alice", "Senior Engineer", 18) -> True   # 待生效：时段 A 不受影响
        clock_in("alice", 80) -> True   # 晋升在这里生效：费率变为 18
        clock_out("alice", 100) -> True
        calculate_pay("alice", 0, 100) -> 860   # 500 (A) + 360 (B)
        calculate_pay("alice", 10, 130) -> 760   # A 截取为 [10,50)=400；B 完整：360
        calculate_pay("bob", 0, 100) -> None   # 没有注册
        clock_in("alice", 150) -> True
        calculate_pay("alice", 120, 200) -> 900   # 打开的时段 C 视为 [150, 200)：50 * 18
        calculate_pay("alice", 0, 130) -> 860   # C 视为 [150, 130)，为空
    """)


@case("Level 3", title="晋升在下一次成功的 clock_in 生效")
def l3_promotion_timing(sol):
    run(sol, """
        add_worker("a", "Engineer", 10) -> True
        set_promotion("ghost", "Senior", 50) -> False   # 没有注册
        clock_in("a", 0) -> True
        set_promotion("a", "Senior", 50) -> True   # 时段打开时设置：要到下一次 clock_in 才生效
        calculate_pay("a", 0, 10) -> 100   # 打开的时段仍按开始时的费率 10
        clock_in("a", 5) -> False   # 被拒绝的 clock_in 没有任何效果，也不会让晋升生效
        calculate_pay("a", 0, 8) -> 80   # 打开的时段仍按 10
        clock_out("a", 10) -> True
        calculate_pay("a", 0, 10) -> 100   # 这个时段按旧费率 10
        clock_in("a", 10) -> True   # 这一次 clock_in 让晋升生效
        clock_out("a", 20) -> True
        calculate_pay("a", 10, 20) -> 500
        calculate_pay("a", 0, 20) -> 600
        clock_in("a", 20) -> True
        clock_out("a", 22) -> True
        calculate_pay("a", 20, 22) -> 100   # 晋升已经用掉：费率保持 50
    """)


@case("Level 3", title="新的晋升整体替换待生效的晋升")
def l3_promotion_replaced(sol):
    run(sol, """
        add_worker("a", "Engineer", 10) -> True
        add_worker("b", "Engineer", 10) -> True
        set_promotion("a", "Mid", 20) -> True
        set_promotion("a", "Senior", 30) -> True   # 替换掉还没生效的 Mid/20
        clock_in("a", 0) -> True
        clock_out("a", 10) -> True
        calculate_pay("a", 0, 10) -> 300
        set_promotion("a", "Lead", 5) -> True
        set_promotion("a", "Lead", 40) -> True
        add_worker("a", "Engineer", 999) -> False   # 失败的 add_worker 不改变费率和待生效的晋升
        add_worker("b", "Manager", 999) -> False
        clock_in("a", 10) -> True
        clock_out("a", 12) -> True
        calculate_pay("a", 0, 12) -> 380   # 300 + 2 * 40
        clock_in("b", 12) -> True
        clock_out("b", 22) -> True
        calculate_pay("b", 0, 100) -> 100   # b 的费率仍是 10
        set_promotion("a", "Junior", 1) -> True   # 费率也可以降低
        clock_in("a", 22) -> True
        clock_out("a", 25) -> True
        calculate_pay("a", 0, 25) -> 383
    """)


@case("Level 3", title="打开的时段视为持续到 end")
def l3_open_session(sol):
    run(sol, """
        add_worker("a", "Engineer", 10) -> True
        clock_in("a", 50) -> True
        calculate_pay("a", 0, 50) -> 0   # 视为 [50, 50)：为空
        calculate_pay("a", 0, 49) -> 0
        calculate_pay("a", 0, 51) -> 10
        calculate_pay("a", 60, 100) -> 400   # [50, 100) 与 [60, 100) 的重叠：40 * 10
        calculate_pay("a", 1000, 1000) -> 0   # 空窗口
        calculate_pay("a", 0, 1000) -> 9500
        clock_out("a", 70) -> True
        calculate_pay("a", 0, 1000) -> 200   # 关闭之后按真实的 clock_out：[50, 70)
        calculate_pay("a", 60, 100) -> 100
        clock_in("a", 70) -> True
        calculate_pay("a", 65, 80) -> 150   # [65,70) 已关闭 50 + [70,80) 打开 100
        calculate_pay("a", 75, 80) -> 50
    """)


@case("Level 3", title="左闭右开的边界")
def l3_windows(sol):
    run(sol, """
        add_worker("a", "Engineer", 3) -> True
        add_worker("b", "Engineer", 7) -> True
        clock_in("a", 10) -> True
        clock_out("a", 20) -> True
        set_promotion("a", "Senior", 5) -> True
        clock_in("a", 20) -> True
        clock_out("a", 30) -> True
        calculate_pay("a", 10, 30) -> 80   # 10 * 3 + 10 * 5
        calculate_pay("a", 19, 21) -> 8   # 刻度 19 按 3，刻度 20 按 5
        calculate_pay("a", 20, 20) -> 0
        calculate_pay("a", 0, 10) -> 0   # [0, 10) 不含 10
        calculate_pay("a", 30, 40) -> 0   # 时段 [20, 30) 不含 30
        calculate_pay("a", 29, 31) -> 5
        calculate_pay("b", 0, 100) -> 0   # 已注册但没有时段
        calculate_pay("c", 0, 100) -> None
        calculate_pay("c", 5, 5) -> None
        clock_in("b", 30) -> True
        clock_out("b", 30) -> True   # 长度为 0 的时段
        calculate_pay("b", 0, 100) -> 0
        calculate_pay("a", 0, 100) -> 80
    """)


@case("Level 3", title="很大的时间戳和费率")
def l3_large_values(sol):
    # Only the values are large. The statement gives no bounds, so sessions and windows stay a few ticks
    # long: an answer that walks the window tick by tick (as ref.py does) is correct and must pass.
    run(sol, """
        add_worker("a", "Engineer", 1000000000000003) -> True
        clock_in("a", 1000000000000) -> True
        clock_out("a", 1000000000011) -> True
        get_total_work_time("a") -> 11
        calculate_pay("a", 999999999990, 1000000000020) -> 11000000000000033   # 11 * 1000000000000003，超过 2**53，要用整数精确计算
        set_promotion("a", "Senior", 3000000000000001) -> True
        clock_in("a", 5000000000000) -> True
        calculate_pay("a", 4999999999995, 5000000000005) -> 15000000000000005   # 打开的时段视为 [5000000000000, 5000000000005)：5 * 3000000000000001
        calculate_pay("a", 1000000000010, 1000000000012) -> 1000000000000003   # 刻度 1000000000010 按旧费率；1000000000011 不在时段内
        calculate_pay("a", 4999999999999, 5000000000001) -> 3000000000000001   # 刻度 5000000000000 按新费率
        clock_out("a", 5000000000007) -> True
        get_total_work_time("a") -> 18
    """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=300, length=(10, 40))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, """
        add_worker("alice", "Engineer", 10) -> True
        clock_in("alice", 0) -> True
        clock_out("alice", 50) -> True
        set_promotion("alice", "Senior Engineer", 20) -> True
        clock_in("alice", 60) -> True
        clock_out("alice", 120) -> True
        set_double_pay(40, 70)
        set_double_pay(65, 100)   # 和前一段重叠：并集是 [40, 100)，重叠部分 [65, 70) 也只翻倍一次
        calculate_pay("alice", 0, 120) -> 2600   # A：400 + 200 = 600；B：1600 + 400 = 2000
    """)


@case("Level 4", title="重叠、包含、相接的双倍时段")
def l4_overlapping_periods(sol):
    run(sol, """
        add_worker("a", "Engineer", 10) -> True
        add_worker("b", "Engineer", 1) -> True
        clock_in("a", 0) -> True
        clock_out("a", 30) -> True
        set_double_pay(0, 12)
        set_double_pay(8, 20)
        set_double_pay(15, 30)   # 三段的并集正好是 [0, 30)
        calculate_pay("a", 0, 30) -> 600   # 每个刻度只翻倍一次：30 * 10 * 2
        set_double_pay(0, 30)
        set_double_pay(0, 30)   # 重复设置同一段
        calculate_pay("a", 0, 30) -> 600
        clock_in("b", 100) -> True
        clock_out("b", 200) -> True
        set_double_pay(100, 140)
        set_double_pay(110, 120)   # 被前一段完全包含：不会让它缩小
        calculate_pay("b", 100, 200) -> 140   # [100, 140) 2 倍 = 80，[140, 200) = 60
        set_double_pay(150, 150)   # 空区间
        calculate_pay("b", 100, 200) -> 140
        set_double_pay(140, 150)   # 和 [100, 140) 首尾相接
        calculate_pay("b", 100, 200) -> 150
        set_double_pay(160, 170)
        set_double_pay(155, 175)   # 后设置的一段包含前一段
        calculate_pay("b", 100, 200) -> 170
        calculate_pay("b", 145, 165) -> 35   # [145,150) 2 倍 10 + [150,155) 5 + [155,165) 2 倍 20
    """)


@case("Level 4", title="只计算查询窗口内的双倍部分")
def l4_window_clips(sol):
    run(sol, """
        add_worker("a", "Engineer", 10) -> True
        clock_in("a", 0) -> True
        clock_out("a", 100) -> True
        set_double_pay(0, 20)
        set_double_pay(80, 120)
        calculate_pay("a", 30, 70) -> 400   # 窗口里没有双倍时段
        calculate_pay("a", 10, 90) -> 1000   # [10,20) 2 倍 200 + [20,80) 600 + [80,90) 2 倍 200
        calculate_pay("a", 90, 150) -> 200   # 时段在 100 结束：[90, 100) 2 倍
        calculate_pay("a", 0, 0) -> 0
        calculate_pay("a", 19, 21) -> 30   # 刻度 19 是 2 倍，刻度 20 不是
        calculate_pay("a", 79, 81) -> 30   # 刻度 79 不是 2 倍，刻度 80 是
        calculate_pay("a", 0, 100) -> 1400
    """)


@case("Level 4", title="对所有员工、所有时段生效")
def l4_all_sessions(sol):
    run(sol, """
        add_worker("a", "Engineer", 10) -> True
        add_worker("b", "Engineer", 3) -> True
        set_double_pay(0, 10)   # 在任何时段之前设置，同样生效
        calculate_pay("ghost", 0, 10) -> None
        calculate_pay("a", 0, 10) -> 0
        clock_in("b", 5) -> True
        clock_out("b", 15) -> True
        calculate_pay("b", 0, 100) -> 45   # [5,10) 2 倍 30 + [10,15) 15
        set_promotion("a", "Senior", 20) -> True
        clock_in("a", 15) -> True
        set_double_pay(30, 40)   # 时段打开期间设置
        calculate_pay("a", 0, 35) -> 500   # 打开的时段视为 [15, 35)：[15,30) 300 + [30,35) 2 倍 200
        calculate_pay("a", 35, 50) -> 400   # [35,40) 2 倍 200 + [40,50) 200
        calculate_pay("a", 0, 25) -> 200   # 打开的时段视为 [15, 25)，不碰双倍时段
        clock_out("a", 45) -> True
        set_double_pay(12, 16)   # 之后设置的双倍时段同样作用于已关闭的时段
        calculate_pay("b", 0, 100) -> 54   # [5,10) 2 倍 30 + [10,12) 6 + [12,15) 2 倍 18
        calculate_pay("a", 0, 100) -> 820   # [15,16) 2 倍 40 + [16,30) 280 + [30,40) 2 倍 400 + [40,45) 100
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=300, length=(10, 45))
