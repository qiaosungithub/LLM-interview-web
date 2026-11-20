import ast
import time

from judgelib import Mismatch, calling, case, expect_true, rng
import ref

SHOW = 40           # a replay lists at most this many calls
ERRORS = {"ValueError": ValueError, "MemoryError": MemoryError}
CLASS = {1: "MemoryAllocator", 2: "MemoryAllocatorLog"}


# ---------------------------------------------------------------- how a sequence is shown

class Text:
    """Shown as-is (no quotes) in the expected / actual lines, e.g. 抛出 ValueError."""

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


def raised(e):
    return Text(f"抛出 {type(e).__name__}: {e}" if str(e) else f"抛出 {type(e).__name__}")


def replay(cls, capacity, done, last_note=None):
    """The calls so far as code to paste into 自定义运行; `done` holds (method, args, want), the last one failing.
    want is an address, None (a free that must succeed) or the name of the exception it must raise."""
    lines = []
    skipped = max(0, len(done) - SHOW)
    for n, (method, args, want) in enumerate(done[skipped:]):
        call = f"a.{method}({', '.join(map(repr, args))})"
        last = n == len(done) - skipped - 1
        if last:
            note = last_note or (f"应当抛出 {want}" if want in ERRORS else
                                 f"应当返回 {want!r}" if method == "allocate" else "应当正常释放")
            shown = f"print({call})" if method == "allocate" and want not in ERRORS else call
            lines.append(f"{shown}   # <- 这一步，{note}")
        elif want in ERRORS:
            lines.append(f"try: {call}   # -> {want}")
            lines.append(f"except {want}: pass")
        else:
            lines.append(call + (f"   # -> {want!r}" if method == "allocate" else ""))
    head = [f"a = {cls}({capacity})"]
    if skipped:
        head.append(f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）")
    return "\n".join(head + lines)


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines like  allocate(5) -> 0   free(5, 5)   free(3, 5) -> ValueError   # why  become (method, args, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        code, _, why = line.strip().partition("#")
        call, arrow, want = code.partition("->")
        node = ast.parse(call.strip(), mode="eval").body
        args = tuple(ast.literal_eval(a) for a in node.args)
        want = want.strip()
        if not arrow:
            want = None
        elif want not in ERRORS:
            want = ast.literal_eval(want)
        steps.append((node.func.id, args, want, why.strip()))
    return steps


def _method(obj, cls, name):
    fn = getattr(obj, name, None)
    if fn is not None and not callable(fn):
        expect_true(False, f"{cls} 对象的 {name} 是 {type(fn).__name__} 而不是方法："
                           f"可能有同名的属性（例如 self.{name} = ...）遮住了 {name}() 方法")
    expect_true(callable(fn), f"{cls} 没有 {name} 方法（方法名要和题目一致）")
    return fn


def make(sol, part, capacity):
    cls = CLASS[part]
    calling(f"a = {cls}({capacity})")
    expect_true(hasattr(sol, cls), f"没有找到类 {cls}（类名要和题目一致）")
    return getattr(sol, cls)(capacity)


def run(sol, part, capacity, script):
    """Runs the calls on a fresh allocator, checking each outcome as soon as it comes back."""
    cls = CLASS[part]
    obj = make(sol, part, capacity)
    done = []
    for method, args, want, why in _parse(script):
        done.append((method, args, want))
        call = calling(replay(cls, capacity, done))
        fn = _method(obj, cls, method)
        if want in ERRORS:
            try:
                got = fn(*args)
            except ERRORS[want]:
                continue
            except (ValueError, MemoryError) as e:
                raise Mismatch(why or f"应当抛出 {want}，实际抛出 {type(e).__name__}", call,
                               Text(f"抛出 {want}"), raised(e)) from None
            raise Mismatch(why or f"应当抛出 {want}，但正常返回了", call, Text(f"抛出 {want}"),
                           got if method == "allocate" else Text("正常返回（没有抛出异常）"))
        try:
            got = fn(*args)
        except (ValueError, MemoryError) as e:
            raise Mismatch(why or ("这次调用合法，不应抛出异常" if method == "free" else
                                   f"应当返回 {want!r}，不应抛出异常"), call,
                           want if method == "allocate" else Text("正常释放"), raised(e)) from None
        if method == "allocate" and not _same(got, want):
            raise Mismatch(why or "返回的地址不对", call, want, got)


def _same(got, want):
    try:
        return bool(got == want) and not isinstance(got, bool)
    except Exception:
        return False


def constructor_errors(sol, part):
    cls = CLASS[part]
    for capacity in (0, -1, -100):
        calling(f"{cls}({capacity})")
        expect_true(hasattr(sol, cls), f"没有找到类 {cls}（类名要和题目一致）")
        try:
            getattr(sol, cls)(capacity)
        except ValueError:
            continue
        raise Mismatch(f"capacity = {capacity} ≤ 0 时构造函数应当抛出 ValueError", f"{cls}({capacity})",
                       Text("抛出 ValueError"), Text("正常返回了"))
    run(sol, part, 1, """
        allocate(1) -> 0
        allocate(1) -> MemoryError
        free(0, 1)
        allocate(1) -> 0
    """)


# ---------------------------------------------------------------- random sequences against ref.Oracle

def _key(method, result):
    kind, value = result
    return (kind, value if method == "allocate" else None)


def _both_rules_apply(method, args, model, got_kind):
    """allocate(size <= 0) while no byte is free meets both of the statement's rules (ValueError for size <= 0,
    MemoryError when no free block is large enough) and the statement does not rank them, so either is accepted."""
    return method == "allocate" and args[0] <= 0 and got_kind == "MemoryError" and not model.gaps()


def _outcome(sol, part, capacity, ops):
    """Replays ops on a fresh answer and a fresh Oracle; None if every outcome agrees."""
    obj, model, wants = getattr(sol, CLASS[part])(capacity), ref.Oracle(capacity), []
    for k, (method, args) in enumerate(ops):
        want = _key(method, ref.outcome(model, method, args))
        try:
            got = _key(method, ref.outcome(obj, method, args))
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if got[0] != want[0] and _both_rules_apply(method, args, model, got[0]):
            got = want
        if got[0] != want[0] or (method == "allocate" and got[0] == "ok" and not _same(got[1], want[1])):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        wants.append(want)
    return None


def _shrink(sol, part, capacity, ops, bad, budget=0.5):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, part, capacity, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def _as_want(method, key):
    kind, value = key
    return kind if kind != "ok" else (value if method == "allocate" else None)


def _shown(method, key):
    kind, value = key
    if kind != "ok":
        return Text(f"抛出 {kind}")
    return value if method == "allocate" else Text("正常释放")


def random_check(sol, part, sequences, capacities, lengths):
    cls = CLASS[part]
    expect_true(hasattr(sol, cls), f"没有找到类 {cls}（类名要和题目一致）")
    r = rng()
    for _ in range(sequences):
        capacity = r.randint(*capacities)
        ops = ref.random_ops(r, capacity, r.randint(*lengths))
        calling(f"a = {cls}({capacity})  # 然后是一串随机操作")
        bad = _outcome(sol, part, capacity, ops)
        if bad is None:
            continue
        ops, bad = _shrink(sol, part, capacity, ops, bad)
        k = bad["k"]
        done = [(m, a, _as_want(m, w)) for (m, a), w in zip(ops[:k], bad["wants"])]
        if bad["exc"] is not None:
            done.append((ops[k][0], ops[k][1], None))
            calling(replay(cls, capacity, done, last_note="这一步抛出了异常"))
            raise bad["exc"]
        method, args = ops[k]
        done.append((method, args, _as_want(method, bad["want"])))
        raise Mismatch(f"随机操作序列的第 {k + 1} 步结果不对（已自动删去与出错无关的调用）",
                       replay(cls, capacity, done), _shown(method, bad["want"]), _shown(method, bad["got"]))


# ---------------------------------------------------------------- scripts shared by both parts

STATEMENT_25 = """
    allocate(5) -> 0
    allocate(5) -> 5
    allocate(5) -> 10
    allocate(5) -> 15
    allocate(5) -> 20
    free(5, 5)
    free(10, 5)
    free(20, 5)
    free(15, 5)
    free(0, 5)
    allocate(25) -> 0       # 全部释放并合并后，空闲块是 [0, 25)
"""

STATEMENT_30 = """
    allocate(12) -> 0
    allocate(3) -> 12
    allocate(6) -> 15
    allocate(9) -> 21
    free(0, 12)
    free(15, 6)
    allocate(5) -> 0        # 首次适配：取地址最低的 [0, 12)，不是更贴合的 [15, 21)
"""

MERGES = """
    allocate(4) -> 0
    allocate(4) -> 4
    allocate(4) -> 8
    allocate(4) -> 12
    allocate(4) -> 16
    allocate(4) -> 20
    free(4, 4)
    free(8, 4)              # 与左边的 [4, 8) 合并成 [4, 12)
    allocate(8) -> 4        # 合并后的块能整块分配
    free(16, 4)
    free(12, 4)             # 左边 [4, 12) 已分配，只与右边的 [16, 20) 合并成 [12, 20)
    allocate(8) -> 12
    free(4, 8)
    free(12, 8)             # 只与左边的 [4, 12) 合并成 [4, 20)
    allocate(16) -> 4
    free(4, 16)
    free(0, 4)              # 与右边合并：空闲 [0, 20)
    allocate(20) -> 0
    allocate(4) -> MemoryError
    free(0, 20)
    free(20, 4)             # 与左边合并：空闲 [0, 24)
    allocate(24) -> 0
"""

MERGE_BOTH = """
    allocate(3) -> 0
    allocate(3) -> 3
    allocate(3) -> 6
    allocate(3) -> 9
    free(0, 3)
    free(6, 3)
    free(3, 3)              # 左右两侧同时合并：空闲 [0, 9)
    allocate(9) -> 0        # [0, 9) 是一整个空闲块
    free(0, 9)
    free(9, 3)              # 与左边的 [0, 9) 合并：空闲 [0, 12)
    allocate(12) -> 0       # 此时应当只有一个空闲块 [0, 12)
"""

BAD_SIZES = """
    allocate(0) -> ValueError
    allocate(-3) -> ValueError
    allocate(4) -> 0
    free(0, 0) -> ValueError
    free(0, -4) -> ValueError
    allocate(6) -> 4        # 前面抛出异常的调用不能改变状态
    allocate(1) -> MemoryError
"""

MEMORY_ERRORS = """
    allocate(11) -> MemoryError
    allocate(10) -> 0
    allocate(1) -> MemoryError
    free(0, 10)
    allocate(2) -> 0
    allocate(3) -> 2
    allocate(2) -> 5
    allocate(3) -> 7
    free(0, 2)
    free(5, 2)
    allocate(3) -> MemoryError   # 一共空闲 4 个字节，但没有一个空闲块有 3 个字节
    allocate(2) -> 0
    allocate(2) -> 5
"""

EXACT_FIT = """
    allocate(4) -> 0
    allocate(2) -> 4
    allocate(4) -> 6
    free(0, 4)
    allocate(4) -> 0        # 大小正好相等的空闲块也要用
    allocate(1) -> 10
    free(6, 4)
    allocate(4) -> 6
    allocate(1) -> 11
    allocate(1) -> MemoryError
"""

INVALID_FREES = """
    allocate(4) -> 0
    allocate(4) -> 4
    allocate(4) -> 8
    free(3, 4) -> ValueError     # 3 从未被 allocate 返回过
    free(2, 4) -> ValueError     # 已分配块内部的地址
    free(4, 3) -> ValueError     # 大小比记录的小
    free(4, 5) -> ValueError     # 大小比记录的大
    free(12, 4) -> ValueError    # 空闲块的起始地址
    free(-4, 4) -> ValueError
    free(100, 4) -> ValueError
    free(4, 4)                   # 前面失败的 free 不能删掉记录；这次大小正确，应当正常释放
    free(4, 4) -> ValueError     # 同一个地址释放两次
    allocate(4) -> 4
    free(4, 4)
    free(0, 8) -> ValueError     # 不能一次释放两个块
    free(0, 4)
    free(8, 4)
    allocate(16) -> 0
"""

NO_EFFECT = """
    allocate(4) -> 0
    allocate(4) -> 4
    free(0, 5) -> ValueError     # 失败的 free 不能把块释放掉，也不能删掉记录
    allocate(4) -> 8             # [0, 4) 仍然是已分配的
    free(0, 4)                   # 大小对了，就能正常释放
    free(4, 0) -> ValueError
    free(4, 4)
    allocate(13) -> MemoryError  # 失败的 allocate 不能占用空间
    allocate(8) -> 0
    free(8, 4)
    free(0, 8)
    allocate(12) -> 0
"""

REUSE = """
    allocate(5) -> 0
    free(0, 5)
    allocate(3) -> 0             # 同一个地址再次被分配，这次大小是 3
    free(0, 5) -> ValueError     # 记录的大小已经是 3
    free(0, 3)
    allocate(7) -> 0
    allocate(3) -> 7
    free(0, 7)
    free(7, 3)
    allocate(10) -> 0
"""

FIRST_FIT_MANY = """
    allocate(2) -> 0
    allocate(1) -> 2
    allocate(5) -> 3
    allocate(1) -> 8
    allocate(3) -> 9
    allocate(1) -> 12
    allocate(7) -> 13
    allocate(1) -> 20
    allocate(4) -> 21
    allocate(1) -> 25
    free(0, 2)
    free(3, 5)
    free(9, 3)
    free(13, 7)
    free(21, 4)
    allocate(4) -> 3             # 空闲块 [0,2) [3,8) [9,12) [13,20) [21,25) [26,30)：地址最低且够大的是 [3, 8)
    allocate(4) -> 13
    allocate(4) -> 21
    allocate(3) -> 9
    allocate(3) -> 17
    allocate(4) -> 26
    allocate(1) -> 0
    allocate(1) -> 1
    allocate(1) -> 7
    allocate(1) -> MemoryError
"""


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子：capacity = 25 的合并")
def p1_example(sol):
    run(sol, 1, 25, STATEMENT_25)


@case("Part 1", title="capacity ≤ 0：构造时抛出 ValueError")
def p1_constructor(sol):
    constructor_errors(sol, 1)


@case("Part 1", title="size ≤ 0：抛出 ValueError")
def p1_bad_size(sol):
    run(sol, 1, 10, BAD_SIZES)


@case("Part 1", title="没有够大的空闲块：MemoryError")
def p1_memory_error(sol):
    run(sol, 1, 10, MEMORY_ERRORS)


@case("Part 1", title="首次适配：取地址最低的块")
def p1_first_fit(sol):
    run(sol, 1, 30, STATEMENT_30)
    run(sol, 1, 30, FIRST_FIT_MANY)


@case("Part 1", title="大小正好相等的空闲块")
def p1_exact_fit(sol):
    run(sol, 1, 12, EXACT_FIT)


@case("Part 1", title="与左边、右边合并")
def p1_merges(sol):
    run(sol, 1, 24, MERGES)


@case("Part 1", title="左右两侧同时合并")
def p1_merge_both(sol):
    run(sol, 1, 12, MERGE_BOTH)


@case("Part 1", title="非法释放：从未分配、重复、大小不符")
def p1_invalid_frees(sol):
    run(sol, 1, 16, INVALID_FREES)


@case("Part 1", title="抛出异常的调用不改变状态")
def p1_no_effect(sol):
    run(sol, 1, 12, NO_EFFECT)


@case("Part 1", title="释放后同一地址可以换个大小再分配")
def p1_reuse(sol):
    run(sol, 1, 10, REUSE)


@case("Part 1", title="随机短序列（对照参考实现）")
def p1_random(sol):
    random_check(sol, 1, 300, (1, 40), (1, 40))


@case("Part 1", title="随机长序列（对照参考实现）")
def p1_random_long(sol):
    random_check(sol, 1, 12, (50, 1000), (200, 400))


# ---------------------------------------------------------------- Part 2
BIG = 10 ** 9


@case("Part 2", sample=True, title="题目例子：首次适配不是最佳适配")
def p2_example(sol):
    run(sol, 2, 30, STATEMENT_30)


@case("Part 2", title="Part 1 的例子：capacity = 25")
def p2_statement_25(sol):
    run(sol, 2, 25, STATEMENT_25)


@case("Part 2", title="构造和 size 的错误与 Part 1 相同")
def p2_errors(sol):
    constructor_errors(sol, 2)
    run(sol, 2, 10, BAD_SIZES)
    run(sol, 2, 10, MEMORY_ERRORS)


@case("Part 2", title="首次适配与正好装满")
def p2_first_fit(sol):
    run(sol, 2, 30, FIRST_FIT_MANY)
    run(sol, 2, 12, EXACT_FIT)


@case("Part 2", title="各种合并")
def p2_merges(sol):
    run(sol, 2, 24, MERGES)
    run(sol, 2, 12, MERGE_BOTH)


@case("Part 2", title="非法释放与失败调用")
def p2_invalid_frees(sol):
    run(sol, 2, 16, INVALID_FREES)
    run(sol, 2, 12, NO_EFFECT)
    run(sol, 2, 10, REUSE)


@case("Part 2", title="随机短序列（对照参考实现）")
def p2_random(sol):
    random_check(sol, 2, 300, (1, 40), (1, 40))


@case("Part 2", title="随机长序列（对照参考实现）")
def p2_random_long(sol):
    random_check(sol, 2, 12, (50, 1000), (200, 400))


@case("Part 2", title="capacity = 10⁹")
def p2_huge_capacity(sol):
    run(sol, 2, BIG, f"""
        allocate({BIG + 1}) -> MemoryError
        allocate({BIG}) -> 0
        free(0, {BIG})
        allocate(1) -> 0
        allocate({BIG - 2}) -> 1
        allocate(2) -> MemoryError
        allocate(1) -> {BIG - 1}
        free(1, {BIG - 2})
        allocate({BIG - 1}) -> MemoryError
        free(0, 1)
        allocate({BIG - 1}) -> 0
    """)


def _fill_then_free_shuffled(sol, n):
    """n blocks of sizes 1..5 fill the front of the space; freeing all of them in random order must merge back."""
    r = rng()
    sizes = [r.randint(1, 5) for _ in range(n)]
    order = list(range(n))
    r.shuffle(order)
    desc = (f"a = MemoryAllocatorLog({BIG})\n"
            f"依次 allocate {n} 个大小为 1..5 的随机块（每个都应紧接在前一个之后）\n"
            f"再按随机顺序把它们全部 free，最后 a.allocate({BIG}) 应返回 0")
    a = make(sol, 2, BIG)
    calling(desc)
    starts, pos = [], 0
    for k, s in enumerate(sizes):
        got = a.allocate(s)
        if got != pos:
            raise Mismatch(f"第 {k + 1} 次 allocate({s}) 的地址不对：前面的块占满了 [0, {pos})", desc, pos, got)
        starts.append(pos)
        pos += s
    for idx in order:
        a.free(starts[idx], sizes[idx])
    try:
        got = a.allocate(BIG)
    except MemoryError:
        raise Mismatch("全部释放后应当合并回一整块 [0, 10⁹)，allocate(10⁹) 却抛出了 MemoryError", desc,
                       0, "抛出 MemoryError") from None
    if got != 0:
        raise Mismatch("全部释放后 allocate(10⁹) 应当返回 0", desc, 0, got)


@case("Part 2", cpu=4, title="2×10⁴ 个块按随机顺序释放后全部合并")
def p2_merge_at_scale(sol):
    _fill_then_free_shuffled(sol, 20_000)


HOLES, SCANS = 25_000, 75_000


@case("Part 2", cpu=11, title="2.5×10⁴ 个空闲块：要求 O(log n)")
def p2_many_holes(sol):
    """2 * HOLES blocks of size 2, every other one freed, then SCANS requests of size 3 that fit no hole."""
    a = make(sol, 2, BIG)
    desc = (f"a = MemoryAllocatorLog({BIG})\n"
            f"for i in range({2 * HOLES}): a.allocate(2)        # 返回 2*i\n"
            f"for i in range(0, {4 * HOLES}, 4): a.free(i, 2)   # 留下 {HOLES} 个大小为 2 的空闲块\n"
            f"for j in range({SCANS}): a.allocate(3)            # 每个都放不进小空闲块，应返回 {4 * HOLES} + 3*j")
    calling(desc)
    for i in range(2 * HOLES):
        got = a.allocate(2)
        if got != 2 * i:
            raise Mismatch(f"第 {i + 1} 次 allocate(2) 的地址不对", desc, 2 * i, got)
    for i in range(0, 4 * HOLES, 4):
        a.free(i, 2)
    base = 4 * HOLES
    for j in range(SCANS):
        got = a.allocate(3)
        if got != base + 3 * j:
            raise Mismatch(f"第 {j + 1} 次 allocate(3) 的地址不对（小空闲块都只有 2 个字节）", desc, base + 3 * j, got)
    got = a.allocate(2)
    if got != 0:
        raise Mismatch("最后再 allocate(2)，应当用地址最低的空闲块 [0, 2)", desc + "\na.allocate(2)", 0, got)


FITTING, ONES = 15_000, 45_000


@case("Part 2", cpu=6, title="1.5×10⁴ 个都够大的空闲块：要求 O(log n)")
def p2_many_fitting_holes(sol):
    """FITTING free blocks of distinct sizes 2, 3, ... between 1-byte blocks, then ONES requests of size 1: every
    free block is large enough, so first fit must not look at all of them (or at every distinct size)."""
    a = make(sol, 2, BIG)
    desc = (f"a = MemoryAllocatorLog({BIG})\n"
            f"starts = []\n"
            f"for i in range({FITTING}):\n"
            f"    starts.append(a.allocate(i + 2))   # 大小 2, 3, 4, … 的块依次紧挨着放\n"
            f"    a.allocate(1)                      # 1 字节的隔板\n"
            f"for i in range({FITTING}): a.free(starts[i], i + 2)   # 留下 {FITTING} 个大小互不相同的空闲块\n"
            f"for j in range({ONES}): print(a.allocate(1))   # 每个空闲块都放得下，首次适配每次取地址最低的那个")
    calling(desc)
    starts, pos = [], 0
    for i in range(FITTING):
        for size in (i + 2, 1):
            got = a.allocate(size)
            if got != pos:
                raise Mismatch(f"第 {i + 1} 轮的 allocate({size}) 的地址不对：前面的块占满了 [0, {pos})", desc, pos, got)
            if size > 1:
                starts.append(pos)
            pos += size
    for i in range(FITTING):
        a.free(starts[i], i + 2)
    i = off = 0
    for j in range(ONES):
        want = starts[i] + off
        got = a.allocate(1)
        if got != want:
            raise Mismatch(f"第 {j + 1} 次 allocate(1) 的地址不对：地址最低的空闲块从 {want} 开始", desc, want, got)
        off += 1
        if off == i + 2:
            i, off = i + 1, 0
    want = starts[-1]
    got = a.allocate(FITTING + 1)
    if got != want:
        raise Mismatch(f"最后再 allocate({FITTING + 1})，应当用唯一够大的那个小空闲块（大小正好 {FITTING + 1}）",
                       desc + f"\na.allocate({FITTING + 1})", want, got)
