import ast
import bisect
import time

from judgelib import Mismatch, calling, case, expect, expect_true, rng
import ref

SHOW = 18           # a replay lists at most this many lines
SHOW_CHARS = 2400   # ... and at most about this many characters, so the failing line is never cut off
METHOD = {"add": "add", "remove": "remove", "between": "keys_between", "kth": "kth", "rank": "rank",
          "len": "__len__", "list": "__iter__"}
RAISE_MARK = "抛出 IndexError"


# ---------------------------------------------------------------- the class under test

def _cls(sol, part):
    """The class a part's tests drive, and its name as shown in a replay."""
    if part < 3:
        cls = getattr(sol, "SkipList", None)
        expect_true(isinstance(cls, type), "answer.py 里没有 SkipList 类（类名要和题目一致）")
        return cls, "SkipList"
    cls = getattr(sol, "IndexedSkipList", None)
    if isinstance(cls, type):
        return cls, "IndexedSkipList"
    cls = getattr(sol, "SkipList", None)
    if isinstance(cls, type) and callable(getattr(cls, "kth", None)) and callable(getattr(cls, "rank", None)):
        return cls, "SkipList"      # Part 3 allows adding kth and rank to the original class
    expect_true(False, "answer.py 里没有 IndexedSkipList 类（可以写成 SkipList 的子类，也可以直接给 SkipList 加上 kth 和 rank）")


def _need(obj, cname, names):
    for name in names:
        m = METHOD.get(name)
        if m:
            expect_true(callable(getattr(obj, m, None)), f"{cname} 没有 {m} 方法（方法名要和题目一致）")


def _ctor_text(cname, ctor):
    return f"sl = {cname}({', '.join(f'{k}={v!r}' for k, v in ctor.items())})"


# ---------------------------------------------------------------- one operation

def _do(sl, op):
    name, a = op[0], op[1:]
    if name == "add":
        return sl.add(a[0])
    if name == "in":
        return a[0] in sl
    if name == "len":
        return len(sl)
    if name == "list":
        return list(sl)
    if name == "remove":
        return sl.remove(a[0])
    if name == "between":
        got = sl.keys_between(a[0], a[1])
        return list(got) if isinstance(got, tuple) else got
    if name == "kth":
        return sl.kth(a[0])
    if name == "rank":
        return sl.rank(a[0])
    raise ValueError(name)


def _code(op):
    name, a = op[0], op[1:]
    return {
        "add": lambda: f"sl.add({a[0]})",
        "in": lambda: f"print({a[0]} in sl)",
        "len": lambda: "print(len(sl))",
        "list": lambda: "print(list(sl))",
        "remove": lambda: f"print(sl.remove({a[0]}))",
        "between": lambda: f"print(sl.keys_between({a[0]}, {a[1]}))",
        "kth": lambda: f"print(sl.kth({a[0]}))",
        "rank": lambda: f"print(sl.rank({a[0]}))",
    }[name]()


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


def _step(sl, op, want):
    """None if the operation behaves as the statement says; otherwise how it failed. An exception the statement
    does not call for comes back as {"exc": e}."""
    try:
        got = _do(sl, op)
    except Exception as e:      # noqa: BLE001 - reported to the user
        if want is ref.RAISES and isinstance(e, IndexError):
            return None
        if want is ref.RAISES:
            return {"msg": f"下标 {op[1]} 不在 [0, len) 内，应当抛出 IndexError，实际抛出 {type(e).__name__}: {e}",
                    "got": None}
        return {"exc": e}
    if op[0] == "add":
        return None                         # add's return value is not checked
    if want is ref.RAISES:
        why = "负数下标也算越界，不能像 Python 列表那样倒着数" if op[1] < 0 else "下标不在 [0, len) 内"
        return {"msg": f"{why}，应当抛出 IndexError，但返回了 {got!r}", "got": got}
    if not _same(got, want):
        return {"msg": "", "got": got}
    return None


# ---------------------------------------------------------------- how a sequence is shown

def _show(want):
    return "IndexError" if want is ref.RAISES else repr(want)


def _replay(head, done, wants, mark="# <- 这一步"):
    """The operations so far as code to paste into 自定义运行; the last one is the failing step. Three or more adds
    in a row are shown as one `for` line."""
    lines, i, prev = [], 0, done[:-1]
    while i < len(prev):
        if prev[i][0] == "add":
            j = i
            while j < len(prev) and prev[j][0] == "add":
                j += 1
            if j - i >= 3:
                lines.append((f"for k in {[op[1] for op in prev[i:j]]}: sl.add(k)", ""))
            else:
                lines += [(_code(op), "") for op in prev[i:j]]
            i = j
        else:
            lines.append((_code(prev[i]), f"# -> {_show(wants[i])}"))
            i += 1
    lines.append((_code(done[-1]), mark))
    keep, size = [], 0
    for code, note in reversed(lines):
        size += len(code) + len(note) + 4
        if keep and (len(keep) >= SHOW or size > SHOW_CHARS):
            break
        keep.append((code, note))
    keep.reverse()
    skipped = len(lines) - len(keep)
    width = max(len(code) for code, _ in keep) + 3
    out = [head] + ([f"# …（省略了前面 {skipped} 行，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(out + [(code.ljust(width) + note).rstrip() for code, note in keep])


def _fail(call, bad, want, msg):
    if want is ref.RAISES:
        m = Mismatch(bad["msg"] + (f"（{msg}）" if msg else ""), call, None, bad["got"])
        m.want = RAISE_MARK
        raise m
    expect(bad["got"], want, call=call, msg=bad["msg"] or msg)


# ---------------------------------------------------------------- hand-written sequences

def _ops_of(code):
    """`sl.add(3)`, `for k in [..]: sl.add(k)`, `3 in sl`, `len(sl)`, `list(sl)`, `sl.remove(3)`, ... as ops."""
    node = ast.parse(code).body[0]
    if isinstance(node, ast.For):
        return [("add", k) for k in ast.literal_eval(node.iter)]
    e = node.value
    if isinstance(e, ast.Compare):
        return [("in", ast.literal_eval(e.left))]
    if isinstance(e.func, ast.Name):
        return [(e.func.id,)]
    name = {"keys_between": "between"}.get(e.func.attr, e.func.attr)
    return [(name, *[ast.literal_eval(a) for a in e.args])]


def _parse(script):
    """Lines like  sl.kth(5) -> IndexError   # why  become (op, want, why); a line without '->' is not checked."""
    steps = []
    for line in script.strip().splitlines():
        code, arrow, rest = line.strip().partition("->")
        if arrow:
            expr, _, why = rest.partition("#")
            expr = expr.strip()
            want = ref.RAISES if expr == "IndexError" else ast.literal_eval(expr)
        else:
            code, _, why = code.partition("#")
            want = None
        for op in _ops_of(code.strip()):
            steps.append((op, want, why.strip()))
    return steps


def run(sol, part, script, **ctor):
    """Runs the operations on a fresh object, checking each result as soon as it comes back."""
    cls, cname = _cls(sol, part)
    head = _ctor_text(cname, ctor)
    calling(f"{head}   # <- 这一步")
    sl = cls(**ctor)
    done, wants = [], []
    for op, want, why in _parse(script):
        done.append(op)
        _need(sl, cname, [op[0]])
        calling(_replay(head, done, wants))         # shown if the answer raises or hangs in this call
        bad = _step(sl, op, want)
        if bad is not None:
            if "exc" in bad:
                calling(_replay(head, done, wants, mark="# <- 这一步抛出异常"))
                raise bad["exc"]
            _fail(_replay(head, done, wants), bad, want, why)
        wants.append(want)


# ---------------------------------------------------------------- random sequences against ref

def _outcome(cls, ctor, ops):
    """Replays ops on a fresh answer object and a fresh ref.Model; None if every result agrees."""
    sl, model, wants = cls(**ctor), ref.Model(), []
    for k, op in enumerate(ops):
        want = model.apply(op)
        bad = _step(sl, op, want)
        if bad is not None:
            bad.update(k=k, want=want, wants=wants)
            return bad
        wants.append(want)
    return None


def _kind(bad):
    return type(bad.get("exc"))


def _shrink(cls, ctor, ops, bad, budget=0.6):
    """Drops operations that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(cls, ctor, cand) if cand else None
            if out is not None and _kind(out) is _kind(bad):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def random_check(sol, part, sequences, length, spread, ps=(0.5,)):
    cls, cname = _cls(sol, part)
    calling(f"{_ctor_text(cname, {'seed': 0})}   # <- 这一步")
    _need(cls(seed=0), cname, ref.PART_OPS[part])
    r = rng()
    for _ in range(sequences):
        ctor = {"seed": r.randrange(1000)}
        p = r.choice(ps)
        if p != 0.5:
            ctor["p"] = p
        ops = ref.random_ops(r, part, r.randint(*length), spread)
        head = _ctor_text(cname, ctor)
        calling(f"{head}\n# …随机操作序列（{len(ops)} 步）")
        bad = _outcome(cls, ctor, ops)
        if bad is None:
            continue
        ops, bad = _shrink(cls, ctor, ops, bad)
        done = ops[: bad["k"] + 1]
        if "exc" in bad:
            calling(_replay(head, done, bad["wants"], mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        _fail(_replay(head, done, bad["wants"]), bad, bad["want"],
              f"随机操作序列的第 {bad['k'] + 1} 步不对（已自动删去与出错无关的操作）")


# ---------------------------------------------------------------- big inputs, drawn once at import time

N1 = 20_000                                     # Part 1: ascending adds, then membership queries
_r = rng("p1-big")
Q1 = [_r.randint(-N1 // 2, N1 + N1 // 2) for _ in range(N1)]

N2 = 20_000                                     # Part 2: random keys, short ranges, then removes
_r = rng("p2-big")
KEYS2 = [_r.randint(ref.LO, ref.HI) for _ in range(N2)]
_sorted2 = sorted(KEYS2)
RANGES2 = []
for _ in range(N2):
    _lo = _r.randint(ref.LO, ref.HI)
    RANGES2.append((_lo, min(ref.HI, _lo + _r.randint(0, 300_000))))   # about 3 keys each; hi stays a legal key
WANT2 = [_sorted2[bisect.bisect_left(_sorted2, lo):bisect.bisect_right(_sorted2, hi)] for lo, hi in RANGES2]
GONE2 = _r.sample(KEYS2, N2 // 2)

N3 = 20_000                                     # Part 3: ascending adds, then kth and rank
_r = rng("p3-big")
KTH3 = [_r.randrange(N3) for _ in range(N3 // 2)]
RANK3 = [_r.randint(-N3 // 2, N3 + N3 // 2) for _ in range(N3 // 2)]

N4 = 100_000                                    # Part 3: the statement's limit of 2 * 10**5 operations
_r = rng("p3-max")
KEYS4 = [_r.randint(-N4 // 2, N4 // 2) for _ in range(N4 - 2)] + [ref.LO, ref.HI]   # many duplicates
_r.shuffle(KEYS4)
SORTED4 = sorted(KEYS4)
KTH4 = [_r.randrange(N4) for _ in range(N4 // 2)]
RANK4 = [_r.randint(-N4, N4) for _ in range(N4 // 2 - 2)] + [ref.LO, ref.HI]     # queries stay in the key range
WANT_RANK4 = [bisect.bisect_left(SORTED4, k) for k in RANK4]


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    run(sol, 1, """
        for k in [30, 10, 20, 10]: sl.add(k)
        list(sl)        -> [10, 10, 20, 30]
        len(sl)         -> 4
        20 in sl        -> True
        25 in sl        -> False
    """, seed=7)


@case("Part 1", title="空集合，默认参数")
def p1_empty(sol):
    run(sol, 1, """
        len(sl)         -> 0
        list(sl)        -> []
        5 in sl         -> False     # 空集合里什么键都没有
        0 in sl         -> False
        sl.add(0)
        0 in sl         -> True
        len(sl)         -> 1
        list(sl)        -> [0]
        1 in sl         -> False
        -1 in sl        -> False
    """)


@case("Part 1", title="重复的键全部保留")
def p1_duplicates(sol):
    run(sol, 1, """
        for k in [7, 7, 7, 7, 7]: sl.add(k)
        len(sl)         -> 5         # add 几次就存几个
        list(sl)        -> [7, 7, 7, 7, 7]
        7 in sl         -> True
        for k in [3, 7, 9]: sl.add(k)
        list(sl)        -> [3, 7, 7, 7, 7, 7, 7, 9]
        len(sl)         -> 8
        8 in sl         -> False
    """, seed=1)


@case("Part 1", title="比所有键都大或都小的键")
def p1_outside_keys(sol):
    run(sol, 1, """
        for k in [10, 20, 30]: sl.add(k)
        31 in sl              -> False     # 比最大的键还大：查找走到了链表末尾
        1000000000 in sl      -> False
        9 in sl               -> False     # 比最小的键还小
        -1000000000 in sl     -> False
        25 in sl              -> False
        30 in sl              -> True
        10 in sl              -> True
        len(sl)               -> 3
    """, seed=2)


@case("Part 1", title="负数与键范围的两端")
def p1_key_range(sol):
    run(sol, 1, """
        for k in [0, -1000000000, 1000000000, -1, 1, -1000000000]: sl.add(k)
        list(sl)              -> [-1000000000, -1000000000, -1, 0, 1, 1000000000]
        -1000000000 in sl     -> True
        1000000000 in sl      -> True
        999999999 in sl       -> False
        -999999999 in sl      -> False
        -1 in sl              -> True
        len(sl)               -> 6
    """, seed=3)


@case("Part 1", title="降序插入，遍历可以重复")
def p1_descending(sol):
    run(sol, 1, """
        for k in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]: sl.add(k)
        list(sl)        -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        list(sl)        -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]    # 遍历不能改变集合
        sl.add(5)
        list(sl)        -> [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
        len(sl)         -> 11
    """, seed=4)


@case("Part 1", title="p=0.99：高度封顶在 32")
def p1_tall_nodes(sol):
    run(sol, 1, """
        for k in [12, 5, 40, 5, 33, 0, 27, 18, 5, 40, 9, 31, 22, 2, 38, 15, 7, 26, 11, 35]: sl.add(k)
        list(sl)        -> [0, 2, 5, 5, 5, 7, 9, 11, 12, 15, 18, 22, 26, 27, 31, 33, 35, 38, 40, 40]
        len(sl)         -> 20
        5 in sl         -> True
        6 in sl         -> False
        40 in sl        -> True
        41 in sl        -> False
    """, seed=3, p=0.99)


@case("Part 1", cpu=2, title="随机操作序列（对照参考实现）")
def p1_random(sol):
    random_check(sol, 1, sequences=150, length=(5, 40), spread=8)
    random_check(sol, 1, sequences=50, length=(20, 80), spread=10 ** 9)


@case("Part 1", cpu=2, title="升序插入 2·10⁴ 个键：要求期望 O(log n)")
def p1_ascending_big(sol):
    cls, cname = _cls(sol, 1)
    head = f"sl = {cname}(seed=5)\nfor k in range({N1}):    # 升序插入\n    sl.add(k)"
    calling(head + "   # <- 这一段（每次 add 要求期望 O(log n)）")
    sl = cls(seed=5)
    for k in range(N1):
        sl.add(k)
    calling(head + f"\nfor k in <{N1} 个随机整数>:\n    k in sl     # <- 这一段（每次查找要求期望 O(log n)）")
    for k in Q1:
        got = k in sl
        if got != (0 <= k < N1):
            expect(got, 0 <= k < N1, call=f"{head}\nprint({k} in sl)   # <- 这一步")
    expect(len(sl), N1, call=f"{head}\nprint(len(sl))   # <- 这一步")
    expect_true(list(sl) == list(range(N1)), f"list(sl) 应当是 0, 1, …, {N1 - 1}",
                call=f"{head}\nprint(list(sl))   # <- 这一步")


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    run(sol, 2, """
        for k in [30, 10, 20, 10]: sl.add(k)
        sl.remove(10)              -> True
        list(sl)                   -> [10, 20, 30]
        sl.remove(15)              -> False     # 没有 15，集合不变
        list(sl)                   -> [10, 20, 30]
        sl.keys_between(10, 25)    -> [10, 20]
        sl.keys_between(20, 20)    -> [20]
        sl.keys_between(31, 99)    -> []
        sl.keys_between(25, 10)    -> []        # lo > hi
    """, seed=7)


@case("Part 2", title="删除后长度和遍历都要更新")
def p2_remove_updates(sol):
    run(sol, 2, """
        for k in [4, 2, 6]: sl.add(k)
        sl.remove(2)       -> True
        len(sl)            -> 2      # 删掉一个，长度减 1
        list(sl)           -> [4, 6]
        2 in sl            -> False
        sl.remove(6)       -> True
        len(sl)            -> 1
        sl.remove(4)       -> True
        len(sl)            -> 0
        list(sl)           -> []
        sl.remove(4)       -> False
        len(sl)            -> 0
    """, seed=8)


@case("Part 2", title="重复的键每次只删一个")
def p2_remove_one_copy(sol):
    run(sol, 2, """
        for k in [5, 1, 5, 9, 5]: sl.add(k)
        sl.remove(5)       -> True
        list(sl)           -> [1, 5, 5, 9]     # 只删掉一个 5
        len(sl)            -> 4
        5 in sl            -> True
        sl.remove(5)       -> True
        sl.remove(5)       -> True
        5 in sl            -> False
        sl.remove(5)       -> False
        list(sl)           -> [1, 9]
    """, seed=9)


@case("Part 2", title="删除不存在的键，集合不变")
def p2_remove_missing(sol):
    run(sol, 2, """
        sl.remove(3)       -> False     # 空集合
        for k in [10, 20, 30]: sl.add(k)
        sl.remove(5)       -> False     # 比所有键都小
        sl.remove(35)      -> False     # 比所有键都大
        sl.remove(25)      -> False     # 夹在两个键中间
        list(sl)           -> [10, 20, 30]
        len(sl)            -> 3
        sl.remove(30)      -> True      # 最后一个
        sl.remove(10)      -> True      # 第一个
        list(sl)           -> [20]
    """, seed=10)


@case("Part 2", title="删空以后再插入")
def p2_empty_again(sol):
    run(sol, 2, """
        for k in [3, 1, 2]: sl.add(k)
        sl.remove(1)       -> True
        sl.remove(2)       -> True
        sl.remove(3)       -> True
        list(sl)           -> []
        len(sl)            -> 0
        3 in sl            -> False
        for k in [8, 8, 2]: sl.add(k)
        list(sl)           -> [2, 8, 8]
        sl.keys_between(-1000000000, 1000000000)   -> [2, 8, 8]
    """, seed=11)


@case("Part 2", title="keys_between：两端都包含，重复照列")
def p2_keys_between(sol):
    run(sol, 2, """
        sl.keys_between(-1000000000, 1000000000)   -> []     # 空集合
        for k in [7, 3, 9, 1, 7, 5, 3, 7]: sl.add(k)
        sl.keys_between(3, 7)      -> [3, 3, 5, 7, 7, 7]     # 两端都包含
        sl.keys_between(4, 6)      -> [5]                    # 两端都不在集合里
        sl.keys_between(7, 7)      -> [7, 7, 7]
        sl.keys_between(0, 0)      -> []
        sl.keys_between(10, 20)    -> []                     # 比所有键都大
        sl.keys_between(-5, 0)     -> []                     # 比所有键都小
        sl.keys_between(1, 1)      -> [1]
        sl.keys_between(9, 9)      -> [9]
        sl.keys_between(9, 1)      -> []                     # lo > hi
        sl.keys_between(8, 7)      -> []
        sl.keys_between(-1000000000, 1000000000)   -> [1, 3, 3, 5, 7, 7, 7, 9]
    """, seed=12)


@case("Part 2", cpu=2, title="随机操作，大量重复键（对照参考实现）")
def p2_random_duplicates(sol):
    random_check(sol, 2, sequences=250, length=(10, 60), spread=4)


@case("Part 2", cpu=2, title="随机操作，整个键范围（对照参考实现）")
def p2_random_wide(sol):
    random_check(sol, 2, sequences=100, length=(10, 60), spread=30)
    random_check(sol, 2, sequences=50, length=(10, 60), spread=10 ** 9)


@case("Part 2", cpu=2, title="2·10⁴ 次短区间查询：要求期望 O(log n + m)")
def p2_ranges_big(sol):
    cls, cname = _cls(sol, 2)
    head = f"sl = {cname}(seed=6)\nfor k in <{N2} 个随机键>:\n    sl.add(k)"
    calling(head + "   # <- 这一段")
    sl = cls(seed=6)
    for k in KEYS2:
        sl.add(k)
    calling(head + f"\nfor lo, hi in <{N2} 个区间，每个约含 3 个键>:\n"
            "    sl.keys_between(lo, hi)   # <- 这一段（每次要求期望 O(log n + m)）")
    for (lo, hi), want in zip(RANGES2, WANT2):
        got = sl.keys_between(lo, hi)
        if not _same(list(got) if isinstance(got, tuple) else got, want):
            expect(got, want, call=f"{head}\nprint(sl.keys_between({lo}, {hi}))   # <- 这一步")
    calling(head + f"\nfor k in <其中 {len(GONE2)} 个键>:\n    sl.remove(k)   # <- 这一段（每次要求期望 O(log n)）")
    for k in GONE2:
        got = sl.remove(k)
        if got is not True:
            expect(got, True, call=f"{head}\n…\nprint(sl.remove({k}))   # <- 这一步（{k} 在集合里）")
    expect(len(sl), N2 - len(GONE2), call=f"{head}\n…\nprint(len(sl))   # <- 这一步")


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    run(sol, 3, """
        for k in [30, 10, 20, 10, 20, 5]: sl.add(k)
        sl.remove(10)      -> True
        list(sl)           -> [5, 10, 20, 20, 30]
        sl.kth(0)          -> 5
        sl.kth(1)          -> 10
        sl.kth(3)          -> 20
        sl.kth(4)          -> 30
        sl.kth(5)          -> IndexError
        sl.kth(-1)         -> IndexError
        sl.rank(20)        -> 2      # 比 20 小的是 5 和 10
        sl.rank(21)        -> 4
        sl.rank(5)         -> 0
        sl.rank(1)         -> 0
        sl.rank(99)        -> 5
    """, seed=7)


@case("Part 3", title="kth 越界都抛 IndexError")
def p3_kth_bounds(sol):
    run(sol, 3, """
        sl.kth(0)          -> IndexError     # 空集合
        sl.kth(-1)         -> IndexError
        sl.add(4)
        sl.kth(0)          -> 4
        sl.kth(1)          -> IndexError     # i == len
        sl.kth(-1)         -> IndexError
        for k in [9, 1, 4]: sl.add(k)
        sl.kth(3)          -> 9
        sl.kth(4)          -> IndexError
        sl.kth(-4)         -> IndexError
        sl.kth(-5)         -> IndexError
        sl.remove(9)       -> True
        sl.kth(2)          -> 4
        sl.kth(3)          -> IndexError     # 删除以后长度是 3
    """, seed=13)


@case("Part 3", title="rank：键不必在集合里，重复各算一次")
def p3_rank(sol):
    run(sol, 3, """
        sl.rank(0)             -> 0       # 空集合
        for k in [5, 8, 5, -3, 8, 5]: sl.add(k)
        sl.rank(5)             -> 1       # 只有 -3 比 5 小
        sl.rank(6)             -> 4       # 三个 5 各算一次
        sl.rank(8)             -> 4
        sl.rank(9)             -> 6
        sl.rank(-3)            -> 0
        sl.rank(-4)            -> 0
        sl.rank(-1000000000)   -> 0
        sl.rank(1000000000)    -> 6
        sl.remove(5)           -> True
        sl.rank(6)             -> 3
        sl.rank(8)             -> 3
        sl.remove(-3)          -> True
        sl.rank(5)             -> 0
    """, seed=14)


@case("Part 3", title="每个名次都对得上")
def p3_kth_rank_agree(sol):
    run(sol, 3, """
        for k in [50, 20, 80, 20, 60, 10, 90, 40, 70, 30]: sl.add(k)
        sl.remove(60)      -> True
        sl.remove(20)      -> True
        sl.add(55)
        sl.add(15)
        list(sl)           -> [10, 15, 20, 30, 40, 50, 55, 70, 80, 90]
        sl.kth(0)          -> 10
        sl.kth(1)          -> 15
        sl.kth(2)          -> 20
        sl.kth(3)          -> 30
        sl.kth(4)          -> 40
        sl.kth(5)          -> 50
        sl.kth(6)          -> 55
        sl.kth(7)          -> 70
        sl.kth(8)          -> 80
        sl.kth(9)          -> 90
        sl.rank(55)        -> 6
        sl.rank(56)        -> 7
        sl.rank(91)        -> 10
        sl.kth(10)         -> IndexError
    """, seed=15)


@case("Part 3", cpu=2, title="随机操作序列（对照参考实现）")
def p3_random(sol):
    random_check(sol, 3, sequences=200, length=(10, 60), spread=5)
    random_check(sol, 3, sequences=100, length=(30, 120), spread=10 ** 9)


@case("Part 3", cpu=2, title="不同 seed 和 p：结果都一样")
def p3_seed_and_p(sol):
    r = rng()
    cls, cname = _cls(sol, 3)
    ops = ref.random_ops(r, 3, 150, 6)
    for ctor in ({}, {"seed": 0}, {"seed": 1}, {"seed": 123456789}, {"seed": 2, "p": 0.25},
                 {"seed": 3, "p": 0.75}, {"seed": 4, "p": 0.9}, {"seed": 5, "p": 0.1}):
        head = _ctor_text(cname, ctor)
        calling(f"{head}\n# …150 步随机操作")
        bad = _outcome(cls, ctor, ops)
        if bad is None:
            continue
        ops2, bad = _shrink(cls, ctor, ops, bad)
        done = ops2[: bad["k"] + 1]
        if "exc" in bad:
            calling(_replay(head, done, bad["wants"], mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        _fail(_replay(head, done, bad["wants"]), bad, bad["want"],
              f"seed 和 p 只能影响内部结构，不能影响返回值（已自动删去与出错无关的操作）")


@case("Part 3", cpu=2, title="2·10⁴ 个键上的 kth / rank：要求期望 O(log n)")
def p3_order_big(sol):
    cls, cname = _cls(sol, 3)
    head = f"sl = {cname}(seed=6)\nfor k in range({N3}):\n    sl.add(k)"
    calling(head + "   # <- 这一段（add 也要保持期望 O(log n)）")
    sl = cls(seed=6)
    for k in range(N3):
        sl.add(k)
    calling(head + f"\nfor i in <{len(KTH3)} 个随机下标>:\n    sl.kth(i)    # <- 这一段（每次要求期望 O(log n)）")
    for i in KTH3:
        got = sl.kth(i)
        if got != i:
            expect(got, i, call=f"{head}\nprint(sl.kth({i}))   # <- 这一步")
    calling(head + f"\nfor k in <{len(RANK3)} 个随机整数>:\n    sl.rank(k)   # <- 这一段（每次要求期望 O(log n)）")
    for k in RANK3:
        want = min(max(k, 0), N3)
        got = sl.rank(k)
        if got != want:
            expect(got, want, call=f"{head}\nprint(sl.rank({k}))   # <- 这一步")


@case("Part 3", cpu=8, title="2·10⁵ 次操作（题目上限）")
def p3_max_size(sol):
    cls, cname = _cls(sol, 3)
    head = f"sl = {cname}(seed=9)\nfor k in <{N4} 个键，有大量重复，包括 ±10⁹>:\n    sl.add(k)"
    calling(head + "   # <- 这一段")
    sl = cls(seed=9)
    for k in KEYS4:
        sl.add(k)
    calling(head + f"\nfor i in <{len(KTH4)} 个随机下标>:\n    sl.kth(i)    # <- 这一段")
    for i in KTH4:
        got = sl.kth(i)
        if got != SORTED4[i]:
            expect(got, SORTED4[i], call=f"{head}\nprint(sl.kth({i}))   # <- 这一步")
    calling(head + f"\nfor k in <{len(RANK4)} 个随机整数>:\n    sl.rank(k)   # <- 这一段")
    for k, want in zip(RANK4, WANT_RANK4):
        got = sl.rank(k)
        if got != want:
            expect(got, want, call=f"{head}\nprint(sl.rank({k}))   # <- 这一步")
