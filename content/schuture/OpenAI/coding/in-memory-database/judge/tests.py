import ast
import copy
import time

from judgelib import Mismatch, calling, case, expect, expect_true, rng
import ref

SHOW = 18           # a replay lists at most this many lines
SHOW_CHARS = 2400   # ... and at most about this many characters
NOTE_CHARS = 70     # an earlier select's result is written next to it only when it is this short

EMP_COLUMNS = ["id", "name", "dept", "salary"]
EMP_ROWS = [
    {"id": 101, "name": "Ana", "dept": "eng", "salary": 95000},
    {"id": 102, "name": "Bo", "dept": "sales", "salary": 71000},
    {"id": 103, "name": "Cy", "dept": "eng", "salary": 88000},
    {"id": 104, "name": "Dee", "dept": "ops", "salary": 60000},
    {"id": 105, "name": "Eli", "dept": "eng", "salary": 88000},
    {"id": 106, "name": "Fo", "dept": "sales", "salary": 71000},
    {"id": 107, "name": "Gia", "dept": "ops", "salary": None},
    {"id": 108, "name": "Hu", "dept": "ops", "salary": None},
]
EMPLOYEES = [("create_table", ("employees", EMP_COLUMNS), {})] + [("insert", ("employees", r), {}) for r in EMP_ROWS]


# ---------------------------------------------------------------- how a call is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    if isinstance(x, list):
        return "[" + ", ".join(_lit(v) for v in x) + "]"
    if isinstance(x, tuple):
        return "(" + ", ".join(_lit(v) for v in x) + ("," if len(x) == 1 else "") + ")"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(v)}" for k, v in x.items()) + "}"
    return repr(x)


def _code(op):
    method, args, kwargs = op
    parts = [_lit(a) for a in args] + [f"{k}={_lit(v)}" for k, v in kwargs.items()]
    return f"db.{method}({', '.join(parts)})"


def _note(want):
    if isinstance(want, ref.Raises):
        return f"# -> {want}"
    text = _lit(want)
    return f"# -> {text}" if len(text) <= NOTE_CHARS else f"# -> {len(want)} 行（已核对）"


def _replay(done, wants, mark="# <- 这一步"):
    """The calls so far as code to paste into 自定义运行; the last one is the failing step. Three or more inserts
    into one table in a row are shown as one `for` line; an earlier call that raised as it should is wrapped in
    suppress(), so the replay runs on past it."""
    prev, lines, i = done[:-1], [], 0
    while i < len(prev):
        method, args, _ = prev[i]
        if method == "insert" and wants[i] is None:
            j = i
            while j < len(prev) and prev[j][0] == "insert" and wants[j] is None and prev[j][1][0] == args[0]:
                j += 1
            if j - i >= 3:
                rows = ", ".join(_lit(op[1][1]) for op in prev[i:j])
                lines.append((f"for row in [{rows}]: db.insert({_lit(args[0])}, row)", ""))
                i = j
                continue
        if isinstance(wants[i], ref.Raises):
            lines.append((f"with suppress({wants[i]}): {_code(prev[i])}", _note(wants[i])))
        elif method == "select":
            lines.append((f"print({_code(prev[i])})", _note(wants[i])))
        else:
            lines.append((_code(prev[i]), ""))
        i += 1
    last = _code(done[-1])
    lines.append((f"print({last})" if done[-1][0] == "select" else last, mark))
    keep, size = [], 0
    for code, note in reversed(lines):
        size += len(code) + len(note) + 4
        if keep and (len(keep) >= SHOW or size > SHOW_CHARS):
            break
        keep.append((code, note))
    keep.reverse()
    skipped = len(lines) - len(keep)
    head = ["db = Database()"]
    if any(code.startswith("with suppress(") for code, _ in keep):
        head.insert(0, "from contextlib import suppress")
    if skipped:
        head.append(f"# …（省略了前面 {skipped} 行，所以不能直接粘贴重现）")
    width = max([len(code) for code, note in keep if note and len(code) <= 72] or [0]) + 3
    return "\n".join(head + [(code.ljust(width - 2) + "  " + note).rstrip() for code, note in keep])


# ---------------------------------------------------------------- one call

def _db(sol):
    cls = getattr(sol, "Database", None)
    expect_true(isinstance(cls, type), "answer.py 里没有 Database 类（类名要和题目一致）")
    calling("db = Database()   # <- 这一步")
    return cls()


def _need(db, method):
    expect_true(callable(getattr(db, method, None)), f"Database 没有 {method} 方法（方法名要和题目一致）")


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


def _key(row):
    return tuple(sorted(row.items()))


def _hint(got, want):
    """A Chinese hint for a wrong select result; empty when judgelib's own element-by-element hint says enough."""
    if not isinstance(got, list):
        return f"select 应当返回 list，得到 {type(got).__name__}"
    try:
        if len(got) == len(want) and sorted(map(_key, got)) == sorted(map(_key, want)):
            return "返回的行都对，但顺序不对"
    except Exception:
        pass
    return ""


def _step(db, op, want):
    """None if the call behaves as the statement says; otherwise how it failed. An exception the statement does
    not call for comes back as {"exc": e}."""
    method, args, kwargs = op
    try:
        got = getattr(db, method)(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    except Exception as e:      # noqa: BLE001 - reported to the user
        if isinstance(want, ref.Raises):
            if isinstance(e, want.exc):
                return None
            return {"msg": f"应当抛出 {want}，实际抛出 {type(e).__name__}: {e}", "got": None, "show": False}
        return {"exc": e}
    if isinstance(want, ref.Raises):
        return {"msg": f"应当抛出 {want}，但正常返回了", "got": got, "show": True}
    if method != "select":
        return None                                 # only select's return value is specified
    if isinstance(got, tuple):
        got = list(got)
    if not _same(got, want):
        return {"msg": _hint(got, want), "got": got, "show": True}
    return None


def _fail(call, bad, want, why):
    msg = "；".join(x for x in (why, bad["msg"]) if x)
    if isinstance(want, ref.Raises):
        raise Mismatch(msg, call, f"抛出 {want}", bad["got"], show=bad["show"])
    expect(bad["got"], want, call=call, msg=msg)


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines like  select("t", ["a"], where=[("a", "<", 3)]) -> [{"a": 1}]   # why  become (op, want, why).
    `-> KeyError` / `-> ValueError` expect that exception; a line without '->' must simply not raise."""
    steps = []
    for line in script.strip().splitlines():
        code, arrow, rest = line.strip().partition("->")
        if arrow:
            expr, _, why = rest.partition("#")
            expr = expr.strip()
            want = {"KeyError": ref.KEY, "ValueError": ref.VALUE}.get(expr) or ast.literal_eval(expr)
        else:
            code, _, why = code.partition("#")
            want = None
        node = ast.parse(code.strip(), mode="eval").body
        args = tuple(ast.literal_eval(a) for a in node.args)
        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        steps.append(((node.func.id, args, kwargs), want, why.strip()))
    return steps


def run(sol, script, employees=False):
    """Runs the calls on a fresh Database, checking each result as soon as it comes back. employees=True first
    builds the statement's employees table."""
    db = _db(sol)
    steps = [(op, None, "") for op in EMPLOYEES] if employees else []
    done, wants = [], []
    for op, want, why in steps + _parse(script):
        done.append(op)
        _need(db, op[0])
        calling(_replay(done, wants))           # shown if the answer raises or hangs in this call
        bad = _step(db, op, want)
        if bad is not None:
            if "exc" in bad:
                calling(_replay(done, wants, mark="# <- 这一步抛出异常"))
                raise bad["exc"]
            _fail(_replay(done, wants), bad, want, why)
        wants.append(want)


# ---------------------------------------------------------------- random sequences against ref

def _outcome(cls, ops):
    """Replays ops on a fresh answer and a fresh ref.Model; None if every result agrees."""
    db, model, wants = cls(), ref.Model(), []
    for k, op in enumerate(ops):
        want = model.call(op[0], copy.deepcopy(op[1]), copy.deepcopy(op[2]))
        bad = _step(db, op, want)
        if bad is not None:
            bad.update(k=k, want=want, wants=wants)
            return bad
        wants.append(want)
    return None


def _kind(bad):
    return type(bad.get("exc")), isinstance(bad.get("want"), ref.Raises)


def _shrink(cls, ops, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(cls, cand) if cand else None
            if out is not None and _kind(out) == _kind(bad):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def random_check(sol, part, sequences, length):
    db = _db(sol)
    for method in ["create_table", "insert", "select"] + (["create_index"] if part >= 4 else []):
        _need(db, method)
    cls, r = type(db), rng()
    for _ in range(sequences):
        ops = ref.random_ops(r, part, r.randint(*length))
        calling(f"db = Database()\n# …随机调用序列（{len(ops)} 步）")
        bad = _outcome(cls, ops)
        if bad is None:
            continue
        ops, bad = _shrink(cls, ops, bad)
        done = ops[: bad["k"] + 1]
        if "exc" in bad:
            calling(_replay(done, bad["wants"], mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        _fail(_replay(done, bad["wants"]), bad, bad["want"],
              f"随机调用序列的第 {bad['k'] + 1} 步不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- big tables for Part 4, drawn at import time

N = 20_000          # rows; the first half is inserted before create_index, the rest after
# Queries per case. A full scan written as one list comprehension per condition costs about 33 ns a row for "="
# and 170 ns for a range, so these counts put it above 3x the 2 s limit, while a select that uses the index
# costs 5-30 us and building the table about 0.1 s.
Q_HASH = 12_000
Q_RANGE = 2_400
_r = rng("p4-big")
HASH_Q = [_r.randrange(N) for _ in range(Q_HASH)]
RANGE_Q = []
for _ in range(Q_RANGE):
    _op = _r.choice(["<", "<=", ">", ">="])
    _c = {"<": _r.randint(0, 2), "<=": _r.randint(0, 1), ">": _r.randint(997, 999), ">=": _r.randint(998, 999)}[_op]
    RANGE_Q.append((_op, _c))
_HOLDS = {"<": lambda v, c: v < c, "<=": lambda v, c: v <= c, ">": lambda v, c: v > c, ">=": lambda v, c: v >= c}
RANGE_WANT = {q: [{"k": i} for i in range(N) if _HOLDS[q[0]](i % 1000, q[1])] for q in set(RANGE_Q)}
BIG_SETUP = (f'db = Database()\ndb.create_table("t", ["k", "v", "s"])\n'
             f'for i in range({N // 2}):\n    db.insert("t", {{"k": i, "v": i % 1000, "s": "x"}})\n'
             f'db.create_index("t", "k", "hash")\ndb.create_index("t", "v", "sorted")\n'
             f'for i in range({N // 2}, {N}):\n    db.insert("t", {{"k": i, "v": i % 1000, "s": "x"}})')


def _big_table(sol):
    db = _db(sol)
    for method in ("create_table", "insert", "create_index", "select"):
        _need(db, method)
    calling(BIG_SETUP + "   # <- 这一段")
    db.create_table("t", ["k", "v", "s"])
    for i in range(N // 2):
        db.insert("t", {"k": i, "v": i % 1000, "s": "x"})
    db.create_index("t", "k", "hash")
    db.create_index("t", "v", "sorted")
    for i in range(N // 2, N):
        db.insert("t", {"k": i, "v": i % 1000, "s": "x"})
    return db


def _big_select(db, where, want):
    got = db.select("t", ["k"], where=where)
    if isinstance(got, tuple):
        got = list(got)
    if not _same(got, want):
        expect(got, want, call=f'{BIG_SETUP}\nprint(db.select("t", ["k"], where={_lit(where)}))   # <- 这一步',
               msg=_hint(got, want))


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    run(sol, """
        select("employees", ["id", "name"]) -> [{"id": 101, "name": "Ana"}, {"id": 102, "name": "Bo"}, {"id": 103, "name": "Cy"}, {"id": 104, "name": "Dee"}, {"id": 105, "name": "Eli"}, {"id": 106, "name": "Fo"}, {"id": 107, "name": "Gia"}, {"id": 108, "name": "Hu"}]
    """, employees=True)


@case("Part 1", title="不给 columns 返回全部列；投影")
def p1_projection(sol):
    run(sol, """
        create_table("t", ["a", "b", "c"])
        select("t") -> []                                  # 空表
        select("t", ["b"]) -> []
        insert("t", {"c": "z", "a": 1, "b": None})         # 键的顺序可以和表的列不同
        insert("t", {"a": 2, "b": "y", "c": ""})
        select("t") -> [{"a": 1, "b": None, "c": "z"}, {"a": 2, "b": "y", "c": ""}]
        select("t", ["c", "a"]) -> [{"c": "z", "a": 1}, {"c": "", "a": 2}]
        select("t", ["b"]) -> [{"b": None}, {"b": "y"}]
        select("t", None) -> [{"a": 1, "b": None, "c": "z"}, {"a": 2, "b": "y", "c": ""}]
    """)


@case("Part 1", title="多张表互不影响；各种取值")
def p1_tables_and_values(sol):
    run(sol, """
        create_table("t", ["x", "y"])
        create_table("u", ["y", "x"])
        insert("t", {"x": -1000000000000, "y": "数据库"})
        insert("u", {"x": 0, "y": ""})
        insert("t", {"x": None, "y": None})
        insert("t", {"x": 7, "y": "7"})
        select("t") -> [{"x": -1000000000000, "y": "数据库"}, {"x": None, "y": None}, {"x": 7, "y": "7"}]
        select("u") -> [{"y": "", "x": 0}]
        select("u", ["x"]) -> [{"x": 0}]
        select("t", ["y"]) -> [{"y": "数据库"}, {"y": None}, {"y": "7"}]
    """)


@case("Part 1", title="表名或列名不存在：KeyError")
def p1_unknown_names(sol):
    run(sol, """
        create_table("t", ["a", "b"])
        insert("t", {"a": 1, "b": 2})
        select("nope") -> KeyError                     # 表不存在
        select("T") -> KeyError                        # 表名区分大小写
        insert("nope", {"a": 1, "b": 2}) -> KeyError
        select("t", ["a", "z"]) -> KeyError            # 列 z 不存在
        select("t", ["A"]) -> KeyError
        select("t") -> [{"a": 1, "b": 2}]
    """)


@case("Part 1", title="空表上的未知列也抛 KeyError")
def p1_unknown_column_empty(sol):
    run(sol, """
        create_table("e", ["a", "b"])
        select("e", ["z"]) -> KeyError         # 表是空的也要检查列名
        select("e", ["a", "z"]) -> KeyError
        select("e", ["a"]) -> []
        select("nope", ["a"]) -> KeyError
    """)


@case("Part 1", title="create_table 重名抛 KeyError")
def p1_duplicate_table(sol):
    run(sol, """
        create_table("t", ["a"])
        insert("t", {"a": 1})
        create_table("t", ["x", "y"]) -> KeyError      # 表名已经存在
        create_table("t", ["a"]) -> KeyError
        select("t") -> [{"a": 1}]                      # 原来的表不受影响
        insert("t", {"a": 2})
        select("t") -> [{"a": 1}, {"a": 2}]
        create_table("u", ["a"])
        select("u") -> []
    """)


@case("Part 1", title="insert 先查多余的键，再查缺的列")
def p1_insert_errors(sol):
    run(sol, """
        create_table("t", ["a", "b"])
        insert("t", {"a": 1, "b": 2, "z": 3}) -> KeyError      # z 不是表的列
        insert("t", {"a": 1}) -> ValueError                    # 缺了 b
        insert("t", {}) -> ValueError
        insert("t", {"a": 1, "z": 3}) -> KeyError              # 既缺 b 又多 z：先检查多余的键，所以是 KeyError
        insert("t", {"z": 3}) -> KeyError
        select("t") -> []                                      # 失败的 insert 不能留下行
        insert("t", {"b": None, "a": None})
        select("t") -> [{"a": None, "b": None}]
    """)


@case("Part 1", title="insert 存的是一份拷贝")
def p1_insert_copies(sol):
    db = _db(sol)
    for method in ("create_table", "insert", "select"):
        _need(db, method)
    head = ('db = Database()\ndb.create_table("t", ["a", "b"])\nrow = {"a": 1, "b": 2}\ndb.insert("t", row)\n'
            'row["a"] = 99      # 插入以后，调用方改了自己的 dict\nrow["b"] = "x"')
    calling(head + "   # <- 这一段")
    db.create_table("t", ["a", "b"])
    row = {"a": 1, "b": 2}
    db.insert("t", row)
    row["a"] = 99
    row["b"] = "x"
    call = head + '\nprint(db.select("t"))   # <- 这一步'
    calling(call)
    got = db.select("t")
    expect(list(got) if isinstance(got, tuple) else got, [{"a": 1, "b": 2}], call=call,
           msg="insert 要存一份拷贝：插入以后再改调用方的 dict，不能改变表里的行")


@case("Part 1", cpu=2, title="随机调用序列（对照参考实现）")
def p1_random(sol):
    random_check(sol, 1, sequences=200, length=(5, 40))


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    run(sol, """
        select("employees", ["name"], where=[("dept", "=", "eng"), ("salary", ">=", 90000)]) -> [{"name": "Ana"}]
        select("employees", ["name"], where=[("salary", "<", 70000)]) -> [{"name": "Dee"}]   # Gia 和 Hu 的 salary 是 None
    """, employees=True)


@case("Part 2", title="None 不满足任何条件，包括 !=")
def p2_none(sol):
    run(sol, """
        select("employees", ["name"], where=[("salary", "!=", 60000)]) -> [{"name": "Ana"}, {"name": "Bo"}, {"name": "Cy"}, {"name": "Eli"}, {"name": "Fo"}]   # Gia 和 Hu 没有 salary，"!=" 也不成立
        select("employees", ["name"], where=[("salary", "=", None)]) -> []          # value 是 None：条件永远为假
        select("employees", ["name"], where=[("salary", "!=", None)]) -> []
        select("employees", ["name"], where=[("name", "!=", None)]) -> []
        select("employees", ["name"], where=[("salary", ">=", 0)]) -> [{"name": "Ana"}, {"name": "Bo"}, {"name": "Cy"}, {"name": "Dee"}, {"name": "Eli"}, {"name": "Fo"}]
        select("employees", ["name"], where=[("dept", "=", "ops"), ("salary", "!=", 1)]) -> [{"name": "Dee"}]
    """, employees=True)


@case("Part 2", title="int 与 str：永不相等，int 小于 str")
def p2_int_vs_str(sol):
    run(sol, """
        select("employees", ["name"], where=[("salary", "<", "0")]) -> [{"name": "Ana"}, {"name": "Bo"}, {"name": "Cy"}, {"name": "Dee"}, {"name": "Eli"}, {"name": "Fo"}]   # 任何 int 都小于任何 str
        select("employees", ["name"], where=[("salary", ">", "0")]) -> []
        select("employees", ["name"], where=[("id", "=", "101")]) -> []                 # 101 和 "101" 不相等
        select("employees", ["id"], where=[("id", "!=", "101")]) -> [{"id": 101}, {"id": 102}, {"id": 103}, {"id": 104}, {"id": 105}, {"id": 106}, {"id": 107}, {"id": 108}]
        select("employees", ["id"], where=[("name", "<=", 1000000000000)]) -> []        # 任何 str 都大于任何 int
    """, employees=True)


@case("Part 2", title="同一列里混有 int 和 str")
def p2_mixed_column(sol):
    run(sol, """
        create_table("m", ["k", "v"])
        insert("m", {"k": 1, "v": 5})
        insert("m", {"k": 2, "v": "5"})
        insert("m", {"k": 3, "v": "a"})
        insert("m", {"k": 4, "v": None})
        insert("m", {"k": 5, "v": -1})
        insert("m", {"k": 6, "v": ""})
        insert("m", {"k": 7, "v": 10})
        insert("m", {"k": 8, "v": "B"})
        select("m", ["k"], where=[("v", "=", 5)]) -> [{"k": 1}]
        select("m", ["k"], where=[("v", "=", "5")]) -> [{"k": 2}]
        select("m", ["k"], where=[("v", "!=", 5)]) -> [{"k": 2}, {"k": 3}, {"k": 5}, {"k": 6}, {"k": 7}, {"k": 8}]
        select("m", ["k"], where=[("v", "<", "")]) -> [{"k": 1}, {"k": 5}, {"k": 7}]       # 所有 int 都小于 ""
        select("m", ["k"], where=[("v", ">=", "")]) -> [{"k": 2}, {"k": 3}, {"k": 6}, {"k": 8}]
        select("m", ["k"], where=[("v", ">", 5)]) -> [{"k": 2}, {"k": 3}, {"k": 6}, {"k": 7}, {"k": 8}]
        select("m", ["k"], where=[("v", ">", "B")]) -> [{"k": 3}]
    """)


@case("Part 2", title="比较的边界；str 按 Python 规则")
def p2_boundaries(sol):
    run(sol, """
        select("employees", ["name"], where=[("salary", ">", 88000)]) -> [{"name": "Ana"}]
        select("employees", ["name"], where=[("salary", ">=", 88000)]) -> [{"name": "Ana"}, {"name": "Cy"}, {"name": "Eli"}]
        select("employees", ["name"], where=[("salary", "<", 71000)]) -> [{"name": "Dee"}]
        select("employees", ["name"], where=[("salary", "<=", 71000)]) -> [{"name": "Bo"}, {"name": "Dee"}, {"name": "Fo"}]
        create_table("s", ["w"])
        insert("s", {"w": "apple"})
        insert("s", {"w": "Banana"})
        insert("s", {"w": "10"})
        insert("s", {"w": "9"})
        insert("s", {"w": "app"})
        insert("s", {"w": ""})
        select("s", where=[("w", "<", "apple")]) -> [{"w": "Banana"}, {"w": "10"}, {"w": "9"}, {"w": "app"}, {"w": ""}]   # 大写字母排在小写前面
        select("s", where=[("w", "<=", "apple")]) -> [{"w": "apple"}, {"w": "Banana"}, {"w": "10"}, {"w": "9"}, {"w": "app"}, {"w": ""}]
        select("s", where=[("w", ">", "9")]) -> [{"w": "apple"}, {"w": "Banana"}, {"w": "app"}]   # "10" < "9"：逐个字符比较
        select("s", where=[("w", ">=", "9")]) -> [{"w": "apple"}, {"w": "Banana"}, {"w": "9"}, {"w": "app"}]
        select("s", where=[("w", "=", "")]) -> [{"w": ""}]
    """, employees=True)


@case("Part 2", title="多个条件同时成立；where 为空")
def p2_and(sol):
    run(sol, """
        select("employees", ["name"], where=[("salary", ">=", 70000), ("salary", "<", 90000)]) -> [{"name": "Bo"}, {"name": "Cy"}, {"name": "Eli"}, {"name": "Fo"}]
        select("employees", ["name"], where=[("salary", ">", 90000), ("salary", "<", 70000)]) -> []
        select("employees", ["name"], where=[("dept", "=", "sales"), ("salary", "=", 71000), ("id", ">", 102)]) -> [{"name": "Fo"}]
        select("employees", ["name"], where=[]) -> [{"name": "Ana"}, {"name": "Bo"}, {"name": "Cy"}, {"name": "Dee"}, {"name": "Eli"}, {"name": "Fo"}, {"name": "Gia"}, {"name": "Hu"}]
        select("employees", ["name"], where=None) -> [{"name": "Ana"}, {"name": "Bo"}, {"name": "Cy"}, {"name": "Dee"}, {"name": "Eli"}, {"name": "Fo"}, {"name": "Gia"}, {"name": "Hu"}]
        select("employees", where=[("name", "=", "Gia")]) -> [{"id": 107, "name": "Gia", "dept": "ops", "salary": None}]
    """, employees=True)


@case("Part 2", title="where 里的未知列：KeyError")
def p2_unknown_column(sol):
    run(sol, """
        create_table("e", ["a", "b"])
        select("e", where=[("a", "=", 1)]) -> []
        select("e", where=[("z", "=", 1)]) -> KeyError                     # 表是空的也要检查
        select("e", ["a"], where=[("a", ">", 0), ("z", "<", 3)]) -> KeyError
        insert("e", {"a": 1, "b": 2})
        select("e", where=[("z", "!=", 0)]) -> KeyError
        select("e", where=[("a", "=", 2), ("z", "=", 0)]) -> KeyError       # 前一个条件已经排除了所有行
        select("nope", where=[("a", "=", 1)]) -> KeyError
        select("e", where=[("b", "=", 2)]) -> [{"a": 1, "b": 2}]
    """)


@case("Part 2", cpu=2, title="随机调用序列（对照参考实现）")
def p2_random(sol):
    random_check(sol, 2, sequences=250, length=(5, 40))


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子：单列排序")
def p3_example_single(sol):
    run(sol, """
        select("employees", ["name"], order_by=[("salary", True)]) -> [{"name": "Gia"}, {"name": "Hu"}, {"name": "Dee"}, {"name": "Bo"}, {"name": "Fo"}, {"name": "Cy"}, {"name": "Eli"}, {"name": "Ana"}]   # None 排最前
    """, employees=True)


@case("Part 3", sample=True, title="题目例子：部门升序、薪水降序")
def p3_example_two_columns(sol):
    run(sol, """
        select("employees", ["name"], order_by=[("dept", True), ("salary", False)]) -> [{"name": "Ana"}, {"name": "Cy"}, {"name": "Eli"}, {"name": "Dee"}, {"name": "Gia"}, {"name": "Hu"}, {"name": "Bo"}, {"name": "Fo"}]
    """, employees=True)


@case("Part 3", title="降序：None 排最后")
def p3_descending_none(sol):
    run(sol, """
        select("employees", ["name"], order_by=[("salary", False)]) -> [{"name": "Ana"}, {"name": "Cy"}, {"name": "Eli"}, {"name": "Bo"}, {"name": "Fo"}, {"name": "Dee"}, {"name": "Gia"}, {"name": "Hu"}]   # Gia 在 Hu 前面：插入顺序
        select("employees", ["name"], order_by=[("salary", True), ("name", False)]) -> [{"name": "Hu"}, {"name": "Gia"}, {"name": "Dee"}, {"name": "Fo"}, {"name": "Bo"}, {"name": "Eli"}, {"name": "Cy"}, {"name": "Ana"}]
    """, employees=True)


@case("Part 3", title="降序时并列的行保持插入顺序")
def p3_descending_ties(sol):
    run(sol, """
        select("employees", ["name"], order_by=[("dept", False)]) -> [{"name": "Bo"}, {"name": "Fo"}, {"name": "Dee"}, {"name": "Gia"}, {"name": "Hu"}, {"name": "Ana"}, {"name": "Cy"}, {"name": "Eli"}]   # 同一部门内按插入顺序，不是倒过来
        select("employees", ["name"], order_by=[("dept", False), ("name", False)]) -> [{"name": "Fo"}, {"name": "Bo"}, {"name": "Hu"}, {"name": "Gia"}, {"name": "Dee"}, {"name": "Eli"}, {"name": "Cy"}, {"name": "Ana"}]
        select("employees", ["id"], where=[("dept", "!=", "eng")], order_by=[("salary", False)]) -> [{"id": 102}, {"id": 106}, {"id": 104}, {"id": 107}, {"id": 108}]
    """, employees=True)


@case("Part 3", title="同一列里混有 int 和 str")
def p3_mixed_column(sol):
    run(sol, """
        create_table("m", ["k", "v"])
        insert("m", {"k": 1, "v": 5})
        insert("m", {"k": 2, "v": "5"})
        insert("m", {"k": 3, "v": "a"})
        insert("m", {"k": 4, "v": None})
        insert("m", {"k": 5, "v": -1})
        insert("m", {"k": 6, "v": ""})
        insert("m", {"k": 7, "v": 10})
        insert("m", {"k": 8, "v": "B"})
        insert("m", {"k": 9, "v": 5})
        insert("m", {"k": 10, "v": None})
        select("m", ["k"], order_by=[("v", True)]) -> [{"k": 4}, {"k": 10}, {"k": 5}, {"k": 1}, {"k": 9}, {"k": 7}, {"k": 6}, {"k": 2}, {"k": 8}, {"k": 3}]   # None，然后 int，然后 str
        select("m", ["k"], order_by=[("v", False)]) -> [{"k": 3}, {"k": 8}, {"k": 2}, {"k": 6}, {"k": 7}, {"k": 1}, {"k": 9}, {"k": 5}, {"k": 4}, {"k": 10}]
        select("m", ["k"], where=[("v", ">", 0)], order_by=[("v", True)]) -> [{"k": 1}, {"k": 9}, {"k": 7}, {"k": 6}, {"k": 2}, {"k": 8}, {"k": 3}]
    """)


@case("Part 3", title="order_by 为空；排序后再投影")
def p3_empty_and_projection(sol):
    run(sol, """
        select("employees", ["name"], order_by=[]) -> [{"name": "Ana"}, {"name": "Bo"}, {"name": "Cy"}, {"name": "Dee"}, {"name": "Eli"}, {"name": "Fo"}, {"name": "Gia"}, {"name": "Hu"}]
        select("employees", ["name"], order_by=None) -> [{"name": "Ana"}, {"name": "Bo"}, {"name": "Cy"}, {"name": "Dee"}, {"name": "Eli"}, {"name": "Fo"}, {"name": "Gia"}, {"name": "Hu"}]
        select("employees", ["id"], where=[("salary", ">=", 71000)], order_by=[("salary", True), ("id", False)]) -> [{"id": 106}, {"id": 102}, {"id": 105}, {"id": 103}, {"id": 101}]
        select("employees", where=[("dept", "=", "ops")], order_by=[("name", False)]) -> [{"id": 108, "name": "Hu", "dept": "ops", "salary": None}, {"id": 107, "name": "Gia", "dept": "ops", "salary": None}, {"id": 104, "name": "Dee", "dept": "ops", "salary": 60000}]
    """, employees=True)


@case("Part 3", title="order_by 里的未知列：KeyError")
def p3_unknown_column(sol):
    run(sol, """
        create_table("e", ["a", "b"])
        select("e", order_by=[("a", True)]) -> []
        select("e", order_by=[("z", True)]) -> KeyError                # 表是空的也要检查
        select("e", ["a"], order_by=[("a", False), ("z", False)]) -> KeyError
        insert("e", {"a": 1, "b": 2})
        select("e", where=[("a", "=", 5)], order_by=[("z", True)]) -> KeyError
        select("nope", order_by=[("a", True)]) -> KeyError
    """)


@case("Part 3", cpu=2, title="随机调用序列（对照参考实现）")
def p3_random(sol):
    random_check(sol, 3, sequences=250, length=(5, 40))


# ---------------------------------------------------------------- Part 4
@case("Part 4", sample=True, title="题目例子")
def p4_example(sol):
    run(sol, """
        create_index("employees", "dept", "hash")
        create_index("employees", "salary", "sorted")
        select("employees", ["name"], where=[("dept", "=", "eng")]) -> [{"name": "Ana"}, {"name": "Cy"}, {"name": "Eli"}]
    """, employees=True)


@case("Part 4", title="建索引之后插入的行也要查得到")
def p4_insert_after_index(sol):
    run(sol, """
        create_table("t", ["k", "v"])
        create_index("t", "k", "hash")
        create_index("t", "v", "sorted")
        insert("t", {"k": 1, "v": 30})
        insert("t", {"k": 2, "v": 10})
        insert("t", {"k": 3, "v": 20})
        insert("t", {"k": 4, "v": 10})
        select("t", ["k"], where=[("k", "=", 3)]) -> [{"k": 3}]                  # 索引要随 insert 更新
        select("t", ["k"], where=[("v", ">=", 20)]) -> [{"k": 1}, {"k": 3}]
        select("t", ["k"], where=[("v", "<", 20)]) -> [{"k": 2}, {"k": 4}]
        create_index("t", "v", "hash")
        insert("t", {"k": 5, "v": 10})
        select("t", ["k"], where=[("v", "=", 10)]) -> [{"k": 2}, {"k": 4}, {"k": 5}]
        select("t", ["k"], where=[("v", "<=", 10)]) -> [{"k": 2}, {"k": 4}, {"k": 5}]
        select("t", ["k"], where=[("k", "=", 5)]) -> [{"k": 5}]
    """)


@case("Part 4", title="用索引时结果仍按插入顺序")
def p4_index_keeps_order(sol):
    run(sol, """
        create_table("t", ["k", "v"])
        insert("t", {"k": 1, "v": 50})
        insert("t", {"k": 2, "v": 10})
        insert("t", {"k": 3, "v": 40})
        insert("t", {"k": 4, "v": 20})
        insert("t", {"k": 5, "v": 30})
        insert("t", {"k": 6, "v": 10})
        create_index("t", "v", "sorted")
        select("t", ["k"], where=[("v", ">", 15)]) -> [{"k": 1}, {"k": 3}, {"k": 4}, {"k": 5}]      # 和没有索引时一样：插入顺序，不是按 v 排序
        select("t", ["k"], where=[("v", "<=", 30)]) -> [{"k": 2}, {"k": 4}, {"k": 5}, {"k": 6}]
        select("t", ["k"], where=[("v", "<", 45)], order_by=[("v", False)]) -> [{"k": 3}, {"k": 5}, {"k": 4}, {"k": 2}, {"k": 6}]
    """)


@case("Part 4", title="sorted 索引：<= 和 > 的边界")
def p4_range_bounds(sol):
    run(sol, """
        create_table("t", ["k", "v"])
        create_index("t", "v", "sorted")
        insert("t", {"k": 1, "v": 5})
        insert("t", {"k": 2, "v": 3})
        insert("t", {"k": 3, "v": 5})
        insert("t", {"k": 4, "v": 7})
        insert("t", {"k": 5, "v": 5})
        insert("t", {"k": 6, "v": 1})
        select("t", ["k"], where=[("v", "<=", 5)]) -> [{"k": 1}, {"k": 2}, {"k": 3}, {"k": 5}, {"k": 6}]   # 等于 5 的行也算
        select("t", ["k"], where=[("v", "<", 5)]) -> [{"k": 2}, {"k": 6}]
        select("t", ["k"], where=[("v", ">", 5)]) -> [{"k": 4}]                                      # 等于 5 的行不算
        select("t", ["k"], where=[("v", ">=", 5)]) -> [{"k": 1}, {"k": 3}, {"k": 4}, {"k": 5}]
        select("t", ["k"], where=[("v", "<=", 4)]) -> [{"k": 2}, {"k": 6}]
        select("t", ["k"], where=[("v", ">", 7)]) -> []
        select("t", ["k"], where=[("v", "<=", 1)]) -> [{"k": 6}]
        select("t", ["k"], where=[("v", "<", 1)]) -> []
    """)


@case("Part 4", title="sorted 索引的列里混有 int 和 str")
def p4_mixed_sorted_index(sol):
    run(sol, """
        create_index("employees", "salary", "sorted")
        insert("employees", {"id": 109, "name": "Ivy", "dept": "eng", "salary": "n/a"})
        select("employees", ["name"], where=[("salary", ">=", 88000)]) -> [{"name": "Ana"}, {"name": "Cy"}, {"name": "Eli"}, {"name": "Ivy"}]   # "n/a" 大于任何 int
        select("employees", ["name"], where=[("salary", "<", "a")]) -> [{"name": "Ana"}, {"name": "Bo"}, {"name": "Cy"}, {"name": "Dee"}, {"name": "Eli"}, {"name": "Fo"}]
        select("employees", ["name"], where=[("salary", ">", "n")]) -> [{"name": "Ivy"}]
        select("employees", ["name"], where=[("salary", "<", 0)]) -> []
    """, employees=True)


@case("Part 4", title="索引列上的 None")
def p4_none_values(sol):
    run(sol, """
        create_table("t", ["k", "c"])
        insert("t", {"k": 1, "c": None})
        insert("t", {"k": 2, "c": 5})
        create_index("t", "c", "hash")
        create_index("t", "c", "sorted")
        insert("t", {"k": 3, "c": None})
        insert("t", {"k": 4, "c": "x"})
        insert("t", {"k": 5, "c": 5})
        select("t", ["k"], where=[("c", "=", None)]) -> []                     # None 不满足任何条件
        select("t", ["k"], where=[("c", "=", 5)]) -> [{"k": 2}, {"k": 5}]
        select("t", ["k"], where=[("c", "!=", 5)]) -> [{"k": 4}]
        select("t", ["k"], where=[("c", "<", 100)]) -> [{"k": 2}, {"k": 5}]
        select("t", ["k"], where=[("c", ">=", 0)]) -> [{"k": 2}, {"k": 4}, {"k": 5}]
        select("t", ["k"], where=[("c", ">", None)]) -> []
    """)


@case("Part 4", title="hash 索引：5 和 \"5\" 不是同一个值")
def p4_hash_mixed_types(sol):
    run(sol, """
        create_table("m", ["k", "v"])
        insert("m", {"k": 1, "v": 5})
        insert("m", {"k": 2, "v": "5"})
        insert("m", {"k": 3, "v": 0})
        create_index("m", "v", "hash")
        insert("m", {"k": 4, "v": "0"})
        insert("m", {"k": 5, "v": 5})
        insert("m", {"k": 6, "v": "5"})
        select("m", ["k"], where=[("v", "=", 5)]) -> [{"k": 1}, {"k": 5}]          # int 与 str 永远不相等，有索引也一样
        select("m", ["k"], where=[("v", "=", "5")]) -> [{"k": 2}, {"k": 6}]
        select("m", ["k"], where=[("v", "=", 0)]) -> [{"k": 3}]
        select("m", ["k"], where=[("v", "=", "0")]) -> [{"k": 4}]
        select("m", ["k"], where=[("k", ">", 1), ("v", "=", 5)]) -> [{"k": 5}]
    """)


@case("Part 4", title="失败的 insert 不能留在索引里")
def p4_failed_insert_index(sol):
    run(sol, """
        create_table("t", ["k", "v"])
        create_index("t", "k", "hash")
        create_index("t", "k", "sorted")
        create_index("t", "v", "hash")
        insert("t", {"k": 1}) -> ValueError                          # 缺了 v：表和索引都不变
        insert("t", {"k": 1, "v": 2, "z": 0}) -> KeyError
        insert("t", {"k": 1, "v": 2})
        select("t", where=[("k", "=", 1)]) -> [{"k": 1, "v": 2}]       # 失败的 insert 不能在索引里留下记录：这一行只出现一次
        select("t", where=[("k", ">=", 1)]) -> [{"k": 1, "v": 2}]
        insert("t", {"k": 9}) -> ValueError
        insert("t", {"v": 7}) -> ValueError
        insert("t", {"k": 3, "v": 5})
        select("t", ["k"], where=[("k", "=", 9)]) -> []
        select("t", ["k"], where=[("v", "=", 7)]) -> []
        select("t", ["k"], where=[("k", ">", 2)]) -> [{"k": 3}]
        select("t", ["k"], where=[("v", "=", 5)]) -> [{"k": 3}]
    """)


@case("Part 4", title="索引条件和其他条件一起")
def p4_index_with_other_conditions(sol):
    run(sol, """
        create_index("employees", "dept", "hash")
        create_index("employees", "salary", "sorted")
        select("employees", ["name"], where=[("salary", ">", 80000), ("dept", "=", "eng")]) -> [{"name": "Ana"}, {"name": "Cy"}, {"name": "Eli"}]
        select("employees", ["name"], where=[("dept", "=", "eng"), ("salary", "<", 90000)]) -> [{"name": "Cy"}, {"name": "Eli"}]
        select("employees", ["name"], where=[("dept", "=", "ops"), ("salary", "!=", 60000)]) -> []
        select("employees", ["name"], where=[("name", "!=", "Cy"), ("salary", ">", 80000)]) -> [{"name": "Ana"}, {"name": "Eli"}]
        select("employees", ["name"], where=[("salary", ">=", 71000), ("salary", "<=", 88000), ("dept", "=", "sales")]) -> [{"name": "Bo"}, {"name": "Fo"}]
        select("employees", ["name"], where=[("dept", "=", "eng"), ("dept", "=", "ops")]) -> []
    """, employees=True)


@case("Part 4", title="重建索引；索引只属于一张表")
def p4_rebuild_and_tables(sol):
    run(sol, """
        create_table("t", ["k", "v"])
        insert("t", {"k": 1, "v": "a"})
        insert("t", {"k": 2, "v": "b"})
        insert("t", {"k": 3, "v": "a"})
        create_index("t", "v", "hash")
        create_index("t", "v", "hash")                                 # 重建：不能让行出现两次
        select("t", ["k"], where=[("v", "=", "a")]) -> [{"k": 1}, {"k": 3}]
        insert("t", {"k": 4, "v": "a"})
        create_index("t", "v", "hash")
        select("t", ["k"], where=[("v", "=", "a")]) -> [{"k": 1}, {"k": 3}, {"k": 4}]
        create_index("t", "k", "sorted")
        create_index("t", "k", "sorted")
        select("t", ["k"], where=[("k", ">=", 2)]) -> [{"k": 2}, {"k": 3}, {"k": 4}]
        create_table("u", ["k", "v"])
        insert("u", {"k": 9, "v": "a"})
        select("u", ["k"], where=[("v", "=", "a")]) -> [{"k": 9}]       # t 的索引不能用在 u 上
        select("u", ["k"], where=[("k", ">", 0)]) -> [{"k": 9}]
    """)


@case("Part 4", title="create_index 的表名或列名不存在")
def p4_unknown_names(sol):
    run(sol, """
        create_table("t", ["a"])
        create_index("nope", "a", "hash") -> KeyError
        create_index("t", "z", "sorted") -> KeyError
        create_index("t", "A", "hash") -> KeyError
        insert("t", {"a": 1})
        select("t", where=[("a", "=", 1)]) -> [{"a": 1}]
    """)


@case("Part 4", cpu=2, title="随机调用序列（对照参考实现）")
def p4_random(sol):
    random_check(sol, 4, sequences=250, length=(5, 40))


@case("Part 4", cpu=2, title="2·10⁴ 行上的 hash 索引：不能扫描全表")
def p4_hash_big(sol):
    db = _big_table(sol)
    calling(BIG_SETUP + f'\nfor x in <{Q_HASH} 个随机的 k>:\n    db.select("t", ["k"], where=[("k", "=", x)])   # <- 这一段')
    for x in HASH_Q:
        _big_select(db, [("k", "=", x)], [{"k": x}])


@case("Part 4", cpu=2, title="有索引的条件不在第一个")
def p4_hash_not_first_big(sol):
    db = _big_table(sol)
    calling(BIG_SETUP + f'\nfor x in <{Q_HASH} 个随机的 k>:\n'
            '    db.select("t", ["k"], where=[("s", "!=", "zz"), ("k", "=", x)])   # <- 这一段（第二个条件有索引，也不能扫描全表）')
    for x in HASH_Q:
        _big_select(db, [("s", "!=", "zz"), ("k", "=", x)], [{"k": x}])


@case("Part 4", cpu=2, title="2·10⁴ 行上的 sorted 索引：范围查询")
def p4_sorted_big(sol):
    db = _big_table(sol)
    calling(BIG_SETUP + f'\nfor op, c in <{Q_RANGE} 个只匹配几十行的范围条件>:\n'
            '    db.select("t", ["k"], where=[("v", op, c)])   # <- 这一段')
    for op, c in RANGE_Q:
        _big_select(db, [("v", op, c)], RANGE_WANT[(op, c)])
