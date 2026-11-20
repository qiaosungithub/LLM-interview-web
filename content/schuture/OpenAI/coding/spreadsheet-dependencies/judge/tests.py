import ast
import time

from judgelib import Mismatch, calling, case, expect, rng
import ref

SHOW = 18                        # a random replay lists at most this many calls
CELLS = ["A1", "A2", "B1", "B2", "C3", "AB7", "Q10"]
N = 120                          # layers of the diamond chain: 240 cells, a longest path of 120


class _Raise:
    def __init__(self, exc):
        self.exc = exc

    def __repr__(self):
        return self.exc


FERR, CERR = _Raise("FormulaError"), _Raise("CycleError")     # the step must raise this
SKIP = object()                  # the step's return value is not checked (it must not raise)


def _errors(sol):
    """The answer's FormulaError and CycleError classes."""
    out = {}
    for n in ("FormulaError", "CycleError"):
        cls = getattr(sol, n, None)
        if not (isinstance(cls, type) and issubclass(cls, BaseException)):
            raise Mismatch(f"答案里没有定义异常类 {n}（题目要求抛出它，名字要一致）", show=False)
        out[n] = cls
    return out


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    return repr(x)


def _code(op):
    name, obj, method, args = op
    call = f"{method}({', '.join(_lit(a) for a in args)})"
    if obj is not None:
        call = f"{obj}.{call}"
    return f"{name} = {call}" if name else call


def _replay(ops, wants, k, mark="这一步", show=SHOW):
    """ops[:k + 1] as code to paste into 自定义运行; wants[i] says what step i returned (or that it raised)."""
    entries = []
    for i in range(k + 1):
        op, want, code = ops[i], wants[i], _code(ops[i])
        raises = isinstance(want, _Raise)
        if i == k:
            note = f"# <- {mark}" + (f"：应当抛出 {want.exc}" if raises else "")
            entries.append([(code if raises or want is SKIP else f"print({code})", note)])
        elif raises:
            entries.append([("try: " + code, ""), (f"except {want.exc}: pass", f"# 抛出 {want.exc}")])
        elif want is SKIP:
            entries.append([(code, "")])
        else:
            entries.append([(f"print({code})", f"# -> {_lit(want)}")])
    head = [e for i, e in enumerate(entries) if ops[i][1] is None]          # constructors are always kept
    body = [e for i, e in enumerate(entries) if ops[i][1] is not None]
    skipped = max(0, len(body) - show)
    rows = [row for e in head for row in e]
    if skipped:
        rows.append((f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）", ""))
    rows += [row for e in body[skipped:] for row in e]
    width = max(len(code) for code, note in rows if note) + 3 if any(note for _, note in rows) else 0
    return "\n".join(code.ljust(width) + note if note else code for code, note in rows)


# ---------------------------------------------------------------- running a sequence

def _model_step(model, op):
    name, obj, method, args = op
    fn = ref.CLASSES[method] if obj is None else getattr(model[obj], method)
    try:
        out = fn(*args)
    except ref.FormulaError:
        return FERR
    except ref.CycleError:
        return CERR
    if name:
        model[name] = out
        return SKIP
    return SKIP if method == "set" else out


def _outcome(sol, ops, wants=None, show=SHOW):
    """Runs ops on fresh objects of the answer, checking each step as soon as it returns. The expected results come
    from `wants`, or from ref when wants is None. Returns None when every step is right, else what went wrong."""
    errs = _errors(sol)
    own = (errs["FormulaError"], errs["CycleError"])
    env, model, seen = {}, {}, []
    for k, op in enumerate(ops):
        name, obj, method, args = op
        want = _model_step(model, op) if wants is None else wants[k]
        seen.append(want)
        target = sol if obj is None else env[obj]
        fn = getattr(target, method, None)
        if not callable(fn):
            where = "答案" if obj is None else type(target).__name__
            raise Mismatch(f"{where} 没有 {method}（名字要和题目一致）", call=_replay(ops, seen, k, show=show), show=False)
        calling(_replay(ops, seen, k, show=show))           # shown if the answer raises or hangs in this call
        try:
            got = fn(*args)
        except Exception as e:
            raises = isinstance(want, _Raise)
            if raises and isinstance(e, errs[want.exc]):
                continue
            unfinished = isinstance(e, NotImplementedError)            # the starter: report as RE, not WA
            kind = "wrong_exc" if (raises or isinstance(e, own)) and not unfinished else "exc"
            return {"k": k, "kind": kind, "exc": e, "wants": seen}
        if isinstance(want, _Raise):
            return {"k": k, "kind": "no_raise", "got": got, "wants": seen}
        if name:
            env[name] = got
        elif want is not SKIP and not _same(got, want):
            return {"k": k, "kind": "value", "got": got, "want": want, "wants": seen}
    return None


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


def _exc_text(e, limit=300):
    text = f"{type(e).__name__}: {e}"
    return text if len(text) <= limit else text[:limit] + " …"


def _report(ops, bad, why, show=SHOW):
    k, kind = bad["k"], bad["kind"]
    want = bad["wants"][k]
    msg = f"{why}。" if why and kind != "value" else why
    if kind == "exc":
        calling(_replay(ops, bad["wants"], k, mark="这一步抛出异常", show=show))
        raise bad["exc"]
    call = _replay(ops, bad["wants"], k, show=show)
    if kind == "wrong_exc":
        got = _exc_text(bad["exc"])
        if isinstance(want, _Raise):
            raise Mismatch(f"{msg}应当抛出 {want.exc}，实际抛出的是 {got}", call=call, show=False)
        raise Mismatch(f"{msg}这一步不应抛出异常，却抛出了 {got}", call=call, show=False)
    if kind == "no_raise":
        raise Mismatch(f"{msg}应当抛出 {want.exc}，却正常返回了", call=call, got=bad["got"])
    expect(bad["got"], bad["want"], call=call, msg=msg)


class Script:
    """A hand-written sequence: the expected result of every step, and why, shown when that step goes wrong."""

    def __init__(self, *classes):
        self.ops, self.wants, self.whys = [], [], []
        for cls in classes:
            self.add((self.obj_of(cls), None, cls, ()), SKIP)

    @staticmethod
    def obj_of(cls):
        return "lz" if cls == "LazySpreadsheet" else "sh"

    def add(self, op, want, why=""):
        self.ops.append(op)
        self.wants.append(want)
        self.whys.append(why)

    def set(self, obj, cell, value, want=SKIP, why=""):
        self.add((None, obj, "set", (cell, value)), want, why)

    def get(self, obj, cell, want, why=""):
        self.add((None, obj, "get", (cell,)), want, why)

    def run(self, sol):
        bad = _outcome(sol, self.ops, self.wants, show=40)       # hand-written sequences are short enough to show
        if bad is not None:
            why = self.whys[bad["k"]]
            _report(self.ops, bad, why, show=40)


def parse(script):
    """Lines such as  sh.set("A1", "=B1 + 2")  or  sh.get("A1") -> 5   # why  or  ... -> CycleError   # why."""
    s = Script()
    for line in script.strip().splitlines():
        if not line.strip():
            continue
        code, _, why = line.strip().partition("#")         # no string in these scripts holds a '#'
        call, arrow, expr = code.partition("->")
        if arrow:
            expr = expr.strip()
            want = {"FormulaError": FERR, "CycleError": CERR}.get(expr) or ast.literal_eval(expr)
        else:
            want = SKIP
        stmt = ast.parse(call.strip()).body[0]
        name = stmt.targets[0].id if isinstance(stmt, ast.Assign) else None
        node = stmt.value
        obj = node.func.value.id if isinstance(node.func, ast.Attribute) else None
        method = node.func.attr if obj else node.func.id
        s.add((name, obj, method, tuple(ast.literal_eval(a) for a in node.args)), want, why.strip())
    return s


def run(sol, script):
    parse(script).run(sol)


# ---------------------------------------------------------------- random sequences against ref

def _shrink(sol, ops, bad, budget=0.5):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 1, False                   # ops[0] is the constructor
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, cand)
            if out is not None and out["kind"] == bad["kind"] and type(out.get("exc")) is type(bad.get("exc")):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def random_check(sol, cls, sequences):
    r = rng(cls)
    for _ in range(sequences):
        ops = ref.random_ops(r, cls, r.randint(15, 60), r.sample(CELLS, r.randint(3, 6)))
        bad = _outcome(sol, ops)
        if bad is None:
            continue
        ops, bad = _shrink(sol, ops, bad)
        _report(ops, bad, f"随机操作序列的第 {bad['k']} 步不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- the syntax rules, for either class

GOOD_NAMES = ["A1", "AB12", "Z0", "XYZ900", "Q007"]
BAD_NAMES = [("a1", "列字母必须大写"), ("A", "没有行号"), ("1A", "必须先字母后数字"), ("A1B", "数字后面不能再有字母"),
             ("", "空字符串不是格子名"), (" A1", "前面不能有空格"), ("A1 ", "后面不能有空格"), ("A 1", "中间不能有空格"),
             ("A1\n", "末尾不能有换行符"), ("A-1", "中间不能有别的字符"), ("Ａ1", "全角字母不是 A–Z"),
             ("É1", "只能用 A–Z"), ("A١", "只能用数字 0–9")]


def names_script(cls):
    o, s = Script.obj_of(cls), Script(cls)
    for i, cell in enumerate(GOOD_NAMES):
        s.set(o, cell, i + 1, why=f"{_lit(cell)} 是格子名")
    for cell, why in BAD_NAMES:
        s.set(o, cell, 5, FERR, f"{_lit(cell)} 不是格子名：{why}")
        s.get(o, cell, FERR, f"get 也要检查格子名：{_lit(cell)} {why}")
    for i, cell in enumerate(GOOD_NAMES):
        s.get(o, cell, i + 1)
    return s


BAD_VALUES = [("=", "等号后面至少要有一项"), ("= ", "只有空格也不行"), ("=D4*2", "不支持 *"), ("=D4/2", "不支持 /"),
              ("=(D4+2)", "不支持括号"), ("=D4+", "最后一项之后不能有运算符"), ("=D4 + ", "最后一项之后不能有运算符"),
              ("=d4+2", "格子名必须大写"), ("=D4E4", "两项之间必须有 + 或 -"), ("=4D", "4D 既不是格子名也不是整数"),
              ("42", "公式必须以 = 开头（字面量要传 int）"), ("D4", "公式必须以 = 开头"), (" =D4", "= 前面不能有空格"),
              ("==D4", "= 只能出现在开头一次"), ("=D4=E4", "不支持 ="), ("=SUM(D4)", "不支持函数"),
              ("=2.5", "只有整数字面量"), ("=1_000", "字面量只能由数字组成"), ("=0x1F", "字面量只能由数字组成"),
              ("", "空字符串不是公式"), (4.0, "float 不是 int"), (None, "None 既不是 int 也不是公式")]


def bad_values_script(cls):
    o, s = Script.obj_of(cls), Script(cls)
    s.set(o, "D4", 2)
    s.set(o, "E4", 3)
    s.set(o, "D5", 1)
    for value, why in BAD_VALUES:
        s.set(o, "D5", value, FERR, why)
    s.get(o, "D5", 1, "抛出 FormulaError 的 set 不能改变 D5 原来的值")
    s.set(o, "F9", "=D4 * E4", FERR, "不支持 *")
    s.get(o, "F9", 0, "F9 从来没有成功 set 过")
    s.get(o, "D4", 2)
    return s


GOOD_SPACING = [("=  D4 +E4 -  3 ", 2), ("=D4+E4", 5), ("=   7", 7), ("=7   ", 7), ("= D4 - E4 ", -1),
                ("=D4  -  1  +  E4", 4)]
BAD_SPACING = [("=D 4", "项的内部不能有空格"), ("=D4 2", "两项之间必须有 + 或 -"), ("=1 000", "项的内部不能有空格"),
               ("=D4 E4", "两项之间必须有 + 或 -"), ("=D4\t+ 2", "只允许空格，不允许制表符"),
               ("=\tD4", "只允许空格，不允许制表符"), ("=D4\n", "末尾不能有换行符")]


def spacing_script(cls):
    o, s = Script.obj_of(cls), Script(cls)
    s.set(o, "D4", 2)
    s.set(o, "E4", 3)
    for value, want in GOOD_SPACING:
        s.set(o, "D5", value, why="项和运算符前后可以有空格")
        s.get(o, "D5", want)
    for value, why in BAD_SPACING:
        s.set(o, "D5", value, FERR, why)
    s.get(o, "D5", 4)
    return s


BAD_SIGNS = [("=-D4", "第一项之前不能有运算符"), ("=+D4", "第一项之前不能有运算符"),
             ("=-5", "字面量是非负整数，第一项之前不能有 -"), ("=- D4", "第一项之前不能有运算符"),
             ("=D4 - -E4", "运算符后面必须是一项"), ("=D4--E4", "两项之间只能有一个运算符"),
             ("=D4++E4", "两项之间只能有一个运算符"), ("=D4+-1", "两项之间只能有一个运算符")]


def signs_script(cls):
    o, s = Script.obj_of(cls), Script(cls)
    s.set(o, "D4", 2)
    s.set(o, "E4", 3)
    s.set(o, "D5", -3, why="int 字面量可以是负数")
    s.get(o, "D5", -3)
    for value, why in BAD_SIGNS:
        s.set(o, "D5", value, FERR, why)
    s.get(o, "D5", -3)
    for value, want in (("=0 - D4", -2), ("=D4 - E4", -1), ("=0-5", -5)):
        s.set(o, "D5", value)
        s.get(o, "D5", want)
    return s


def literals_script(cls):
    o, s = Script.obj_of(cls), Script(cls)
    s.set(o, "D4", "=" + "9" * 18, why="18 位的字面量是合法的")
    s.get(o, "D4", 10 ** 18 - 1)
    s.set(o, "D4", "=" + "9" * 18 + " + " + "9" * 18)
    s.get(o, "D4", 2 * (10 ** 18 - 1))
    s.set(o, "D4", "=0")
    s.get(o, "D4", 0)
    for value in ("=" + "9" * 19, "=1" + "0" * 18, "=D4 + " + "1" * 19):
        s.set(o, "D5", value, FERR, "字面量最多 18 位")
    s.set(o, "D6", 10 ** 30, why="int 本身没有位数限制")
    s.set(o, "D7", "=D6 + 1")
    s.get(o, "D7", 10 ** 30 + 1)
    s.set(o, "D6", -(10 ** 25))
    s.get(o, "D7", -(10 ** 25) + 1)
    s.get(o, "D5", 0, "D5 从来没有成功 set 过")
    return s


SYNTAX = [names_script, bad_values_script, spacing_script, signs_script, literals_script]


# ---------------------------------------------------------------- the diamond chain

def _chain_text(obj, cls, x0=1, y0=2):
    return (f"{obj} = {cls}()\n{obj}.set(\"X0\", {x0})\n{obj}.set(\"Y0\", {y0})\n"
            f"for i in range(1, {N}):\n"
            f"    {obj}.set(f\"X{{i}}\", f\"=X{{i - 1}} + Y{{i - 1}}\")\n"
            f"    {obj}.set(f\"Y{{i}}\", f\"=X{{i - 1}} - Y{{i - 1}}\")\n")


def _chain(sol, cls, x0=1, y0=2):
    sheet = getattr(sol, cls)()
    sheet.set("X0", x0)
    sheet.set("Y0", y0)
    for i in range(1, N):
        sheet.set(f"X{i}", f"=X{i - 1} + Y{i - 1}")
        sheet.set(f"Y{i}", f"=X{i - 1} - Y{i - 1}")
    return sheet


def _get(sol, sheet, cell, want, call, msg=""):
    errs = _errors(sol)
    calling(call)
    try:
        got = sheet.get(cell)
    except (errs["FormulaError"], errs["CycleError"]) as e:
        raise Mismatch(f"{msg}这一步不应抛出异常，却抛出了 {_exc_text(e)}", call=call, show=False) from None
    expect(got, want, call=call, msg=msg)


# ---------------------------------------------------------------- Part 1
EXAMPLE = """
    {o}.set("P4", 6)
    {o}.set("Q2", 17)
    {o}.set("M3", "=Q2 - P4")
    {o}.set("M8", "=M3 - P4 + 30")
    {o}.get("M8") -> 35
    {o}.set("P4", 14)
    {o}.get("M8") -> 19     # M3 现在是 17 - 14 = 3
    {o}.get("M3") -> 3
"""


@case("Part 1", sample=True, title="题目例子：35，改 P4 后 19")
def p1_example(sol):
    run(sol, "lz = LazySpreadsheet()" + EXAMPLE.format(o="lz"))


@case("Part 1", sample=True, title="题目例子：U1、U2 成环")
def p1_example_cycle(sol):
    run(sol, """
        lz = LazySpreadsheet()
        lz.set("U1", "=U2")
        lz.set("U2", "=U1")             # Part 1 的 set 即使成环也要存下
        lz.get("U1") -> CycleError      # U1 读 U2，U2 读 U1
        lz.get("U2") -> CycleError      # U2 的公式读 U1
    """)


@case("Part 1", title="没设置过的格子读作 0")
def p1_unset(sol):
    run(sol, """
        lz = LazySpreadsheet()
        lz.get("A1") -> 0               # 从没 set 过的格子读作 0
        lz.get("ZZ99") -> 0
        lz.set("B2", "=A1 + 7")
        lz.get("B2") -> 7               # A1 没有 set 过，读作 0
        lz.get("A1") -> 0
        lz.set("C3", "=B2 - Q7 - 10")
        lz.get("C3") -> -3
        lz.set("A1", 5)
        lz.get("C3") -> 2               # 之后 set 了 A1，读它的格子要用新值
        lz.set("Q7", -100)
        lz.get("C3") -> 102
    """)


@case("Part 1", title="格子名：大写字母加数字")
def p1_names(sol):
    names_script("LazySpreadsheet").run(sol)


@case("Part 1", title="非法的值抛 FormulaError")
def p1_bad_values(sol):
    bad_values_script("LazySpreadsheet").run(sol)


@case("Part 1", title="空格只能在项和运算符之间")
def p1_spaces(sol):
    spacing_script("LazySpreadsheet").run(sol)


@case("Part 1", title="没有一元正负号")
def p1_signs(sol):
    signs_script("LazySpreadsheet").run(sol)


@case("Part 1", title="18 位字面量与大整数")
def p1_literals(sol):
    literals_script("LazySpreadsheet").run(sol)


@case("Part 1", title="set 之后 get 要看到新值")
def p1_update(sol):
    run(sol, """
        lz = LazySpreadsheet()
        lz.set("A1", 1)
        lz.set("A2", "=A1 + 1")
        lz.set("A3", "=A2 + 1")
        lz.set("A4", "=A3 + A1")
        lz.get("A4") -> 4
        lz.get("A2") -> 2
        lz.set("A1", 10)
        lz.get("A2") -> 11      # A1 改了，读 A1 的格子都要变
        lz.get("A4") -> 22
        lz.set("A2", 0)
        lz.get("A4") -> 11      # A3 = 0 + 1
        lz.get("A3") -> 1
        lz.set("A2", "=A1 - 4")
        lz.get("A3") -> 7
        lz.get("A4") -> 17
        lz.set("A1", -2)
        lz.get("A4") -> -7
    """)


@case("Part 1", title="菱形和重复引用不是环")
def p1_diamond(sol):
    run(sol, """
        lz = LazySpreadsheet()
        lz.set("E5", 2)
        lz.set("F5", "=E5 + 3")
        lz.set("G5", "=E5 - 1")
        lz.set("H5", "=F5 + G5")
        lz.get("H5") -> 6       # E5 经 F5 和 G5 被读了两次，这不是环
        lz.set("J1", "=E5 + E5 - 1")
        lz.get("J1") -> 3       # 一个公式里两次读同一个格子也不是环
        lz.set("K1", "=H5 + J1 + F5")
        lz.get("K1") -> 14
        lz.set("E5", 10)
        lz.get("K1") -> 54
        lz.get("G5") -> 9
    """)


CYCLE_SETUP = """
    lz = LazySpreadsheet()
    lz.set("A1", 5)
    lz.set("B1", "=A1 + C1")
    lz.set("C1", "=D1 - 2")
    lz.set("D1", "=B1")             # B1 -> C1 -> D1 -> B1 成环；Part 1 的 set 照样存下
    lz.set("E1", "=C1 + 1")
    lz.set("F1", "=A1 + 1")
    lz.set("H2", "=H2")             # 自引用也照样存下
    lz.set("H3", "=7 + H2")
"""


@case("Part 1", title="环上的和读环的格子抛 CycleError")
def p1_cycles(sol):
    run(sol, CYCLE_SETUP + """
        lz.get("B1") -> CycleError
        lz.get("C1") -> CycleError
        lz.get("D1") -> CycleError
        lz.get("E1") -> CycleError      # E1 读 C1，C1 在环上
        lz.get("A1") -> 5               # 环读 A1，但求 A1 不需要环上的格子
        lz.get("F1") -> 6
        lz.get("H2") -> CycleError      # 自引用
        lz.get("H3") -> CycleError      # H3 读 H2
        lz.get("Z1") -> 0
    """)


@case("Part 1", title="打破环之后恢复正常")
def p1_break_cycle(sol):
    run(sol, CYCLE_SETUP + """
        lz.get("B1") -> CycleError
        lz.set("D1", 4)                 # D1 不再读 B1，环断开
        lz.get("C1") -> 2
        lz.get("B1") -> 7
        lz.get("E1") -> 3
        lz.set("H2", 3)
        lz.get("H3") -> 10
        lz.set("D1", "=E1")             # D1 -> E1 -> C1 -> D1 又成环
        lz.get("B1") -> CycleError
        lz.get("D1") -> CycleError
        lz.get("F1") -> 6
        lz.get("H3") -> 10
        lz.set("E1", "=F1 - 1")         # E1 不再读 C1
        lz.get("B1") -> 8
    """)


@case("Part 1", title="200 个格子的长链")
def p1_deep_chain(sol):
    loop = "    lz.set(f\"K{i}\", f\"=K{i - 1} + 1\")\n"
    up = "lz = LazySpreadsheet()\nlz.set(\"K1\", 1)\nfor i in range(2, 201):\n" + loop
    down = "lz = LazySpreadsheet()\nfor i in range(200, 1, -1):\n" + loop + "lz.set(\"K1\", 1)\n"
    for text, rows, first in ((up, range(2, 201), True), (down, range(200, 1, -1), False)):
        calling(text)
        lz = sol.LazySpreadsheet()
        if first:
            lz.set("K1", 1)
        for i in rows:
            lz.set(f"K{i}", f"=K{i - 1} + 1")
        if not first:
            lz.set("K1", 1)
        _get(sol, lz, "K200", 200, text + 'print(lz.get("K200"))', "K200 = K1 + 199：")
        _get(sol, lz, "K100", 100, text + 'print(lz.get("K100"))')
        lz.set("K1", 50)
        _get(sol, lz, "K200", 249, text + 'lz.set("K1", 50)\nprint(lz.get("K200"))')


@case("Part 1", title="120 层菱形链：不能按路径重复求值")
def p1_diamond_chain(sol):
    text = _chain_text("lz", "LazySpreadsheet")
    calling(text)
    lz = _chain(sol, "LazySpreadsheet")
    xs, ys = ref.layers(1, 2, N)
    msg = f"从 X{N - 1} 到 X0 有 2^{N - 1} 条路径，每次 get 都要在线性时间内算完："
    for cell, want in ((f"X{N - 1}", xs[-1]), (f"Y{N - 1}", ys[-1]), ("X60", xs[60])):
        _get(sol, lz, cell, want, text + f"print(lz.get(\"{cell}\"))", msg)
    lz.set("X0", -5)
    xs, ys = ref.layers(-5, 2, N)
    for cell, want in ((f"X{N - 1}", xs[-1]), (f"Y{N - 2}", ys[-2])):
        _get(sol, lz, cell, want, text + f"lz.set(\"X0\", -5)\nprint(lz.get(\"{cell}\"))", msg)


@case("Part 1", title="随机操作序列（对照参考实现）")
def p1_random(sol):
    random_check(sol, "LazySpreadsheet", 60)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子：35，改 P4 后 19")
def p2_example(sol):
    run(sol, "sh = Spreadsheet()" + EXAMPLE.format(o="sh"))


@case("Part 2", sample=True, title="题目例子：拒绝 U2 = U1")
def p2_example_cycle(sol):
    run(sol, "sh = Spreadsheet()" + EXAMPLE.format(o="sh") + """
        sh.set("U1", "=U2")
        sh.set("U2", "=U1") -> CycleError   # U2 读 U1、U1 读 U2：成环，要拒绝
        sh.get("U1") -> 0                   # 被拒绝的调用不改变任何格子：U2 仍然没有设置
        sh.get("U2") -> 0
        sh.get("M8") -> 19
        sh.get("M3") -> 3
    """)


@case("Part 2", title="没设置过的格子读作 0")
def p2_unset(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.get("A1") -> 0               # 从没 set 过的格子读作 0
        sh.get("ZZ99") -> 0
        sh.set("B2", "=A1 + 7")
        sh.get("B2") -> 7               # A1 没有 set 过，读作 0
        sh.get("A1") -> 0
        sh.set("C3", "=B2 - Q7 - 10")
        sh.get("C3") -> -3
        sh.set("A1", 5)
        sh.get("C3") -> 2               # set A1 时要更新所有间接读它的格子
        sh.set("Q7", -100)
        sh.get("C3") -> 102
    """)


@case("Part 2", title="格子名和公式语法同 Part 1")
def p2_syntax(sol):
    for make in SYNTAX:
        make("Spreadsheet").run(sol)


@case("Part 2", title="自引用被拒绝")
def p2_reject_self(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.set("H2", 7)
        sh.set("H3", "=H2 + 1")
        sh.set("H2", "=H2") -> CycleError         # 直接自引用
        sh.set("H2", "=H2 + 1") -> CycleError
        sh.set("H9", "=3 - H9") -> CycleError     # 没设置过的格子也一样
        sh.set("H2", "=H2 * 2") -> FormulaError   # 不是合法公式（不支持 *），所以是 FormulaError 而不是 CycleError
        sh.get("H2") -> 7                         # 被拒绝的 set 不改变 H2 原来的值
        sh.get("H3") -> 8
        sh.get("H9") -> 0
        sh.set("H2", 10)
        sh.get("H3") -> 11
    """)


@case("Part 2", title="被拒绝的 set 什么都不改变")
def p2_reject_keeps_state(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.set("A1", 4)
        sh.set("B1", 3)
        sh.set("C1", "=A1 + B1")
        sh.set("D1", "=C1 - 1")
        sh.set("C1", "=D1 + A1") -> CycleError    # C1 原来是 "=A1 + B1"，被拒绝后要原样保留
        sh.get("C1") -> 7
        sh.get("D1") -> 6
        sh.set("B1", 10)
        sh.get("C1") -> 14                        # C1 仍然读 B1：旧的依赖不能丢
        sh.get("D1") -> 13
        sh.set("A1", "=D1") -> CycleError
        sh.get("A1") -> 4
        sh.set("A1", 0)
        sh.get("D1") -> 9                         # C1 仍然读 A1
        sh.set("D1", 2)                           # D1 不再读 C1
        sh.set("C1", "=D1 + A1")                  # 现在不成环了
        sh.get("C1") -> 2
        sh.set("D1", 5)
        sh.get("C1") -> 5
    """)


@case("Part 2", title="经过多个格子的环被拒绝")
def p2_long_cycles(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.set("A1", 1)
        sh.set("A2", "=A1 + 1")
        sh.set("A3", "=A2 + 1")
        sh.set("A4", "=A3 + A2")
        sh.set("A1", "=A4") -> CycleError         # A1 -> A4 -> A3 -> A2 -> A1
        sh.set("A1", "=5 + A3") -> CycleError     # A1 -> A3 -> A2 -> A1
        sh.set("A2", "=A4 - 1") -> CycleError
        sh.set("B1", "=A4")
        sh.set("A1", "=B1 + 1") -> CycleError     # 经过 B1 回到 A1
        sh.get("A1") -> 1
        sh.get("A4") -> 5
        sh.get("B1") -> 5
        sh.set("A1", "=Z9 + 2")
        sh.get("B1") -> 7
        sh.set("Z9", "=B1") -> CycleError         # Z9 -> B1 -> A4 -> A3 -> A2 -> A1 -> Z9
        sh.get("Z9") -> 0
        sh.set("Z9", 1)
        sh.get("B1") -> 9
    """)


@case("Part 2", title="get 必须是 O(1)")
def p2_get_o1(sol):
    text = _chain_text("sh", "Spreadsheet")
    calling(text)
    sh = _chain(sol, "Spreadsheet")
    xs, ys = ref.layers(1, 2, N)
    wants = {f"X{N - 1}": xs[-1], f"Y{N - 1}": ys[-1], "X60": xs[60], "Y3": ys[3], "Z1": 0}
    cells = list(wants)
    gets = 40000
    call = text + f"for _ in range({gets}):\n    sh.get(\"X{N - 1}\")   # 以及 Y{N - 1}、X60、Y3、Z1"
    calling(call)
    for i in range(gets):
        cell = cells[i % len(cells)]
        got = sh.get(cell)
        if got != wants[cell]:
            expect(got, wants[cell], call=text + f"print(sh.get(\"{cell}\"))")


@case("Part 2", title="随机操作序列（对照参考实现）")
def p2_random(sol):
    random_check(sol, "Spreadsheet", 60)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子：自引用 H2")
def p3_example_self(sol):
    # Part 3 statement: set("H2", "=H2") is rejected by Spreadsheet.set; LazySpreadsheet only finds it in get.
    run(sol, """
        sh = Spreadsheet()
        lz = LazySpreadsheet()
        sh.set("H2", "=H2") -> CycleError         # Spreadsheet.set 拒绝自引用
        sh.get("H2") -> 0                         # 被拒绝的 set 不改变 H2
        lz.set("H2", "=H2")                       # LazySpreadsheet.set 照样存下
        lz.get("H2") -> CycleError
    """)


@case("Part 3", title="覆盖公式会去掉旧的依赖")
def p3_overwrite_drops_edges(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.set("J1", 4)
        sh.set("K1", "=J1 + 6")
        sh.get("K1") -> 10
        sh.set("K1", 70)                # K1 不再读 J1
        sh.set("J1", 500)
        sh.get("K1") -> 70
        sh.set("J1", "=K1 + 1")         # 不成环：K1 已经不读 J1 了
        sh.get("J1") -> 71
        sh.set("L1", "=J1")
        sh.set("L1", "=M1 - 2")         # 公式换成公式：L1 不再读 J1
        sh.set("J1", "=L1")             # 不成环
        sh.get("J1") -> -2
        sh.set("M1", 9)
        sh.get("J1") -> 7
        sh.get("L1") -> 7
        sh.get("K1") -> 70
    """)


@case("Part 3", title="三个格子的环：两个类的不同处理")
def p3_cycles_both(sol):
    run(sol, """
        sh = Spreadsheet()
        lz = LazySpreadsheet()
        sh.set("R1", "=R2 + 1")
        sh.set("R2", "=R3 + 1")
        sh.set("R3", "=R1 + 1") -> CycleError     # R1 -> R2 -> R3 -> R1
        lz.set("R1", "=R2 + 1")
        lz.set("R2", "=R3 + 1")
        lz.set("R3", "=R1 + 1")
        lz.get("R1") -> CycleError
        lz.get("R3") -> CycleError
        sh.get("R1") -> 2
        sh.get("R3") -> 0
        sh.set("R3", 10)
        sh.get("R1") -> 12
    """)


@case("Part 3", title="先引用后设置：间接读它的格子都更新")
def p3_forward_ref(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.set("M1", "=N1 + 5")
        sh.set("M2", "=M1 - 1")
        sh.set("M3", "=M2 + M1 + N1")
        sh.get("N1") -> 0
        sh.get("M2") -> 4
        sh.get("M3") -> 9
        sh.set("N1", 3)                 # 设置 N1 之后，所有间接读它的格子都要更新
        sh.get("M1") -> 8
        sh.get("M2") -> 7
        sh.get("M3") -> 18
        sh.set("N1", "=P1 - 10")
        sh.get("M3") -> -21
        sh.set("P1", 10)
        sh.get("M3") -> 9
        sh.get("N1") -> 0
    """)


@case("Part 3", title="菱形：第四个格子的值")
def p3_diamond(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.set("E5", 2)
        sh.set("F5", "=E5 + 3")
        sh.set("G5", "=E5 - 1")
        sh.set("H5", "=F5 + G5")
        sh.get("H5") -> 6
        sh.set("E5", 40)
        sh.get("H5") -> 82              # F5 = 43，G5 = 39
        sh.get("F5") -> 43
        sh.get("G5") -> 39
    """)


@case("Part 3", title="长短不一的路径：按依赖顺序重算")
def p3_uneven(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.set("A1", 1)
        sh.set("B1", "=A1 + 1")
        sh.set("C1", "=B1 + 1")
        sh.set("D1", "=C1 + A1")        # D1 直接读 A1，也经 B1、C1 间接读 A1
        sh.get("D1") -> 4
        sh.set("A1", 10)
        sh.get("D1") -> 22              # 要用更新后的 C1 = 12，而不是旧的 3
        sh.get("C1") -> 12
        sh.set("W1", "=A1 + C1 + B1")
        sh.set("W2", "=W1 - A1")
        sh.set("A1", 0)
        sh.get("W1") -> 3
        sh.get("W2") -> 3
        sh.get("D1") -> 2
    """)


@case("Part 3", title="120 层菱形链：每个格子只重算一次")
def p3_diamond_chain(sol):
    text = _chain_text("sh", "Spreadsheet")
    calling(text)
    sh = _chain(sol, "Spreadsheet")
    msg = f"从 X0 到 X{N - 1} 有 2^{N - 1} 条路径，一次 set 只能把每个格子重算一次："
    for v in range(20):
        x0 = 3 * v - 7
        call = text + f"sh.set(\"X0\", {x0})"
        calling(call)
        sh.set("X0", x0)
        xs, ys = ref.layers(x0, 2, N)
        _get(sol, sh, f"X{N - 1}", xs[-1], call + f"\nprint(sh.get(\"X{N - 1}\"))", msg)
        _get(sol, sh, f"Y{N - 1}", ys[-1], call + f"\nprint(sh.get(\"Y{N - 1}\"))", msg)


@case("Part 3", title="字面量改成公式：依赖立即生效")
def p3_literal_to_formula(sol):
    run(sol, """
        sh = Spreadsheet()
        sh.set("J1", 4)
        sh.set("K1", 70)
        sh.set("L1", "=K1 + 1")
        sh.get("L1") -> 71
        sh.set("K1", "=J1 + 6")         # 字面量改成公式：K1 和读它的 L1 在这次 set 里就要更新
        sh.get("K1") -> 10
        sh.get("L1") -> 11
        sh.set("J1", 500)
        sh.get("K1") -> 506
        sh.get("L1") -> 507
        sh.set("J1", "=L1") -> CycleError   # K1 现在读 J1，所以 J1 读 L1 会成环
        sh.get("J1") -> 500
    """)
