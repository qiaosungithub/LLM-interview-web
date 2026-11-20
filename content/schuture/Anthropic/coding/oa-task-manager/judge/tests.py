import ast
import time

from judgelib import calling, case, expect, expect_true, rng
import ref

SHOW = 18           # a replay lists at most this many steps
FIELDS = ("task_id", "title", "priority", "created_at")


class Task(tuple):
    """How a returned task is compared and shown: its four fields, printed as in the statement."""

    def __new__(cls, task_id, title, priority, created_at):
        return super().__new__(cls, (task_id, title, priority, created_at))

    def __repr__(self):
        return f"Task({_lit(self[0])}, {_lit(self[1])}, {self[2]!r}, {self[3]!r})"


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    if isinstance(x, list):
        return "[" + ", ".join(_lit(v) for v in x) + "]"
    return repr(x)


def _note(want, width=110):
    text = _lit(want)
    return "# -> " + (text if len(text) <= width else text[:width] + " …")


def _code(method, args, kwargs):
    parts = [_lit(a) for a in args] + [f"{k}={_lit(v)}" for k, v in kwargs.items()]
    return f"tm.{method}({', '.join(parts)})"


VIEWS = {   # how a check reads the return value, shown as code
    None: ("{}", lambda g: g),
    "ids": ("[t.task_id for t in {}]", lambda g: [t[0] if isinstance(t, Task) else t for t in g]),
}


def _render(steps, current, mark="# <- 这一步"):
    """steps: (code, note) done so far; current: the expression being checked."""
    lines = steps + [(f"print({current})", mark)]
    skipped = max(0, len(lines) - SHOW)
    lines = lines[skipped:]
    width = max(len(c) for c, _ in lines) + 3
    top = ["tm = TaskManager()"] + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(top + [(c.ljust(width) + n) if n else c for c, n in lines])


def _method(obj, name):
    fn = getattr(obj, name, None)
    expect_true(callable(fn), f"TaskManager 没有 {name} 方法（方法名要和题目一致）")
    return fn


def _norm(x):
    """Any object with the four Task fields compares as a Task; a list or tuple of them as a list."""
    if x is None or isinstance(x, (str, int)):
        return x
    try:
        if all(hasattr(x, f) for f in FIELDS):
            return Task(*(getattr(x, f) for f in FIELDS))
    except Exception:
        return x
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def _from_ref(x):
    if isinstance(x, tuple):
        return Task(*x)
    if isinstance(x, list):
        return [Task(*t) for t in x]
    return x


def _shape_msg(got, want):
    """A message when a Task was expected but something without Task's fields came back."""
    def bad(g):
        return g is not None and not isinstance(g, Task)
    if isinstance(want, Task) and bad(got):
        return "返回值应当是 Task 对象（带 task_id、title、priority、created_at 四个字段），或 None"
    if isinstance(want, list) and isinstance(got, list) and any(bad(g) for g in got):
        return "列表里的每一项都应当是 Task 对象（带 task_id、title、priority、created_at 四个字段）"
    if isinstance(want, list) and not isinstance(got, list):
        return "返回值应当是 Task 组成的列表"
    return ""


def _same(got, want):
    try:
        return bool(got == want)
    except Exception:
        return False


class Session:
    """Runs calls on one TaskManager, checking each return value as soon as it comes back."""

    def __init__(self, sol):
        calling("tm = TaskManager()")
        self.tm = sol.TaskManager()
        self.steps = []

    def call(self, method, args, kwargs, want, why="", view=None):
        expr = VIEWS[view][0].format(_code(method, args, kwargs))
        fn = _method(self.tm, method)
        calling(_render(self.steps, expr))     # shown if the answer raises or hangs in this call
        try:
            raw = fn(*args, **kwargs)
        except Exception:
            calling(_render(self.steps, expr, "# <- 这一步抛出异常"))
            raise
        got = _norm(raw)
        shape = _shape_msg(got, want if view is None else None)
        if not shape:
            try:
                got = VIEWS[view][1](got)
            except Exception:
                shape = "返回值应当是 Task 组成的列表"
        if shape or not _same(got, want):
            expect(got, want, call=_render(self.steps, expr), msg=shape or why or ("" if isinstance(want, list) else "返回值不对"))
        self.steps.append((expr, _note(want)))
        return raw

    def do(self, code, fn):
        """A statement of the replay that is not a checked call."""
        self.steps.append((code, ""))
        try:
            fn()
        except Exception:
            pass


def _parse(script):
    """Lines like  ids(get_task_list(3, "alice", min_priority=3)) -> ["task-2"]   # why  become
    (method, args, kwargs, view, want, why)."""
    steps = []
    for line in script.strip().splitlines():
        call, arrow, rest = line.strip().partition(" -> ")
        assert arrow, f"missing '->' in {line!r}"
        expr, _, why = rest.partition("  #")
        node = ast.parse(call.strip(), mode="eval").body
        view = None
        if node.func.id in VIEWS and isinstance(node.args[0], ast.Call):
            view, node = node.func.id, node.args[0]
        args = tuple(ast.literal_eval(a) for a in node.args)
        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        want = eval(expr.strip(), {"__builtins__": {}}, {"Task": Task, "None": None, "True": True, "False": False})
        steps.append((node.func.id, args, kwargs, view, want, why.strip()))
    return steps


def run(sol, script):
    s = Session(sol)
    for method, args, kwargs, view, want, why in _parse(script):
        s.call(method, args, kwargs, want, why, view)
    return s


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, ops):
    """Replays ops on a fresh answer and a fresh ref model; None if every return value agrees."""
    calling("tm = TaskManager()")
    tm, model = sol.TaskManager(), ref.TaskManager()
    wants = []
    for k, (method, args, kwargs) in enumerate(ops):
        want = _from_ref(getattr(model, method)(*args, **kwargs))
        fn = _method(tm, method)
        try:
            got = _norm(fn(*args, **kwargs))
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants}
        if not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants}
        wants.append(want)
    return None


def _shrink(sol, ops, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short. Every subsequence still
    keeps the statement's timestamp guarantees."""
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
        steps = [(_code(*op), _note(w)) for op, w in zip(ops[:-1], bad["wants"])]
        current = _code(*ops[-1])
        if bad["exc"] is not None:
            calling(_render(steps, current, "# <- 这一步抛出异常"))
            raise bad["exc"]
        expect(bad["got"], bad["want"], call=_render(steps, current),
               msg=_shape_msg(bad["got"], bad["want"])
               or f"随机操作序列的第 {len(ops)} 步返回值不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, """
        add_task(10, "alice", "write report", 3) -> "task-1"
        add_task(12, "alice", "review PR", 5) -> "task-2"
        add_task(14, "bob", "write report", 1) -> "task-3"   # 标题可以重复；属于另一个用户
        get_task(15, "alice", "task-1") -> Task("task-1", "write report", 3, 10)
        get_task(15, "alice", "task-3") -> None   # task-3 属于 bob，不属于 alice
        update_task(16, "alice", "task-1", "write Q3 report", 4) -> True
        get_task(17, "alice", "task-1") -> Task("task-1", "write Q3 report", 4, 10)   # created_at 不变
        complete_task(18, "alice", "task-1") -> True
        complete_task(19, "alice", "task-1") -> False   # 已经完成了
        get_task(20, "alice", "task-1") -> None   # 完成的任务不再活跃
        update_task(21, "alice", "task-1", "x", 1) -> False   # 不活跃的任务不能修改
    """)


@case("Level 1", title="所有用户共用一个 id 计数器")
def l1_shared_counter(sol):
    run(sol, """
        add_task(0, "bob", "write report", 1) -> "task-1"
        add_task(0, "alice", "write report", 1) -> "task-2"   # 所有用户共用一个计数器
        add_task(1, "alice", "write report", 1) -> "task-3"   # 同一用户的同名任务也是新任务
        complete_task(2, "bob", "task-1") -> True
        add_task(3, "bob", "write report", 1) -> "task-4"   # 完成的任务不会让 id 被重用
        update_task(4, "carol", "task-9", "x", 1) -> False
        add_task(4, "carol", "x", 1) -> "task-5"   # 只有 add_task 会用掉 id
        get_task(5, "alice", "task-3") -> Task("task-3", "write report", 1, 1)
        get_task(5, "bob", "task-4") -> Task("task-4", "write report", 1, 3)
        get_task(5, "alice", "task-2") -> Task("task-2", "write report", 1, 0)
        get_task(5, "bob", "task-1") -> None
    """)


@case("Level 1", title="只有拥有者能读、改、完成")
def l1_owner_checks(sol):
    run(sol, """
        add_task(1, "alice", "write report", 3) -> "task-1"
        add_task(2, "bob", "review PR", 5) -> "task-2"
        update_task(3, "bob", "task-1", "hacked", 9) -> False   # task-1 属于 alice
        get_task(3, "alice", "task-1") -> Task("task-1", "write report", 3, 1)   # 失败的 update 没有任何效果
        complete_task(4, "bob", "task-1") -> False   # task-1 不是 bob 的任务
        get_task(4, "alice", "task-1") -> Task("task-1", "write report", 3, 1)   # 失败的 complete 没有任何效果
        get_task(4, "bob", "task-1") -> None
        complete_task(5, "alice", "task-2") -> False   # task-2 不是 alice 的任务
        get_task(5, "bob", "task-2") -> Task("task-2", "review PR", 5, 2)
        update_task(6, "carol", "task-2", "x", 1) -> False
        complete_task(6, "bob", "task-2") -> True
        complete_task(7, "alice", "task-1") -> True
        get_task(8, "alice", "task-1") -> None
    """)


@case("Level 1", title="不存在的 id；完成后的任务")
def l1_unknown_and_inactive(sol):
    run(sol, """
        get_task(0, "alice", "task-1") -> None   # 还没有任何任务
        update_task(0, "alice", "task-1", "x", 1) -> False
        complete_task(0, "alice", "task-1") -> False
        add_task(1, "alice", "write report", 3) -> "task-1"   # 前面的失败调用没有用掉 id
        get_task(1, "alice", "task-0") -> None
        get_task(1, "alice", "task-2") -> None
        complete_task(2, "alice", "task-1") -> True
        update_task(3, "alice", "task-1", "write report", 3) -> False   # 完成的任务不能修改
        complete_task(3, "alice", "task-1") -> False
        get_task(3, "alice", "task-1") -> None   # 失败的 update 也不会让它重新活跃
        add_task(4, "alice", "write report", 3) -> "task-2"
        get_task(4, "alice", "task-2") -> Task("task-2", "write report", 3, 4)
    """)


@case("Level 1", title="修改只换标题和优先级")
def l1_update_semantics(sol):
    run(sol, """
        add_task(5, "alice", "draft", 2) -> "task-1"
        update_task(5, "alice", "task-1", "final", -3) -> True   # 同一时间戳也可以；优先级可以是负数
        get_task(5, "alice", "task-1") -> Task("task-1", "final", -3, 5)
        update_task(9, "alice", "task-1", "final", -3) -> True   # 内容不变的修改也返回 True
        update_task(9, "alice", "task-1", "", 0) -> True
        get_task(10, "alice", "task-1") -> Task("task-1", "", 0, 5)   # created_at 始终是创建时的 5
        update_task(12, "alice", "task-1", "写周报", 7) -> True
        get_task(12, "alice", "task-1") -> Task("task-1", "写周报", 7, 5)
        complete_task(12, "alice", "task-1") -> True   # 同一时间戳完成
        get_task(12, "alice", "task-1") -> None   # 从完成的这次调用起就不再活跃
    """)


@case("Level 1", cpu=2, title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=250, length=(5, 30))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, """
        add_task(0, "alice", "write report", 3) -> "task-1"
        add_task(1, "alice", "review PR", 5) -> "task-2"
        add_task(2, "alice", "file expenses", 5) -> "task-3"   # 和 task-2 同优先级，创建得更晚
        add_task(3, "alice", "renew badge", 1) -> "task-4"
        get_task_list(3, "alice") -> [Task("task-2", "review PR", 5, 1), Task("task-3", "file expenses", 5, 2), Task("task-1", "write report", 3, 0), Task("task-4", "renew badge", 1, 3)]   # 优先级 5、3、1；同为 5 时先创建的 task-2 在前
        get_task_list(3, "alice", min_priority=3) -> [Task("task-2", "review PR", 5, 1), Task("task-3", "file expenses", 5, 2), Task("task-1", "write report", 3, 0)]   # renew badge（优先级 1）被过滤掉
        get_task_list(3, "carol") -> []   # carol 没有任务
    """)


@case("Level 2", title="优先级降序，并列按创建顺序")
def l2_order(sol):
    run(sol, """
        add_task(0, "alice", "a", 2) -> "task-1"
        add_task(0, "bob", "b", 9) -> "task-2"
        add_task(0, "alice", "c", 4) -> "task-3"
        add_task(0, "alice", "d", 2) -> "task-4"
        add_task(1, "alice", "e", 4) -> "task-5"
        get_task_list(1, "alice") -> [Task("task-3", "c", 4, 0), Task("task-5", "e", 4, 1), Task("task-1", "a", 2, 0), Task("task-4", "d", 2, 0)]   # 只有 alice 的任务；同优先级按创建顺序
        update_task(2, "alice", "task-1", "a2", 4) -> True
        ids(get_task_list(2, "alice")) -> ["task-1", "task-3", "task-5", "task-4"]   # task-1 在时刻 2 改成了优先级 4，按创建顺序排在最前
        update_task(3, "alice", "task-3", "c", 4) -> True
        ids(get_task_list(3, "alice")) -> ["task-1", "task-3", "task-5", "task-4"]   # 修改不改变创建顺序
        update_task(4, "alice", "task-5", "e", 10) -> True
        get_task_list(4, "alice") -> [Task("task-5", "e", 10, 1), Task("task-1", "a2", 4, 0), Task("task-3", "c", 4, 0), Task("task-4", "d", 2, 0)]
        get_task_list(4, "bob") -> [Task("task-2", "b", 9, 0)]
    """)


@case("Level 2", title="并列时 task-10 排在 task-9 之后")
def l2_ties_many(sol):
    s = Session(sol)
    for k in range(1, 13):
        user = "bob" if k % 4 == 0 else "alice"
        s.call("add_task", (k // 3, user, f"t{k}", 2 if k == 6 else 1), {}, f"task-{k}")
    alice = ["task-6", "task-1", "task-2", "task-3", "task-5", "task-7", "task-9", "task-10", "task-11"]
    s.call("get_task_list", (4, "alice"), {}, alice, "同优先级按创建顺序（task_id 的数字后缀），不是按字符串比较 task_id",
           view="ids")
    s.call("update_task", (5, "alice", "task-10", "t10", 2), {}, True)
    s.call("get_task_list", (5, "alice"), {"min_priority": 2}, ["task-6", "task-10"],
           "task-6 比 task-10 先创建", view="ids")
    s.call("get_task_list", (5, "bob"), {}, ["task-4", "task-8", "task-12"], view="ids")


@case("Level 2", title="min_priority 包含等号；0 也是过滤条件")
def l2_min_priority(sol):
    run(sol, """
        add_task(0, "alice", "neg", -2) -> "task-1"
        add_task(0, "alice", "zero", 0) -> "task-2"
        add_task(0, "alice", "pos", 3) -> "task-3"
        add_task(0, "alice", "neg2", -1) -> "task-4"
        ids(get_task_list(0, "alice")) -> ["task-3", "task-2", "task-4", "task-1"]
        ids(get_task_list(0, "alice", min_priority=0)) -> ["task-3", "task-2"]   # min_priority=0 是一个过滤条件，和 None 不同
        ids(get_task_list(0, "alice", min_priority=-1)) -> ["task-3", "task-2", "task-4"]   # >= 包含等号
        ids(get_task_list(0, "alice", min_priority=3)) -> ["task-3"]
        ids(get_task_list(0, "alice", min_priority=4)) -> []
        ids(get_task_list(0, "alice", min_priority=None)) -> ["task-3", "task-2", "task-4", "task-1"]
        update_task(1, "alice", "task-1", "neg", 5) -> True
        ids(get_task_list(1, "alice", min_priority=4)) -> ["task-1"]   # 按当前的优先级过滤
        ids(get_task_list(1, "bob", min_priority=0)) -> []
    """)


@case("Level 2", title="只列出自己的活跃任务")
def l2_list_contents(sol):
    s = run(sol, """
        get_task_list(0, "alice") -> []   # 空实例
        add_task(1, "alice", "write report", 3) -> "task-1"
        add_task(1, "alice", "review PR", 5) -> "task-2"
        add_task(2, "bob", "write report", 3) -> "task-3"
        complete_task(3, "alice", "task-2") -> True
        get_task_list(3, "alice") -> [Task("task-1", "write report", 3, 1)]   # 完成的任务不在列表里
        update_task(4, "alice", "task-1", "write Q3 report", 4) -> True
        get_task_list(4, "alice") -> [Task("task-1", "write Q3 report", 4, 1)]   # 反映当前的标题和优先级，created_at 不变
        get_task_list(5, "carol") -> []
        get_task_list(5, "bob") -> [Task("task-3", "write report", 3, 2)]
        complete_task(6, "alice", "task-1") -> True
        get_task_list(6, "alice") -> []
    """)
    s.do('tm.get_task_list(6, "bob").clear()', lambda: s.tm.get_task_list(6, "bob").clear())
    s.call("get_task_list", (6, "bob"), {}, [Task("task-3", "write report", 3, 2)],
           "修改 get_task_list 返回的列表，不能改变已存储的任务")


@case("Level 2", cpu=2, title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=250, length=(5, 30))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, """
        add_task(0, "alice", "flush cache", 2, ttl=5) -> "task-1"   # 在 at_timestamp < 5 时活跃
        add_task(0, "alice", "renew badge", 1) -> "task-2"   # 没有 ttl：永不过期
        get_task_list(4, "alice") -> [Task("task-1", "flush cache", 2, 0), Task("task-2", "renew badge", 1, 0)]   # 4 < 5：task-1 仍然活跃
        get_task_list(5, "alice") -> [Task("task-2", "renew badge", 1, 0)]   # 5 == 0 + 5：task-1 刚好过期
        get_task(5, "alice", "task-1") -> None
        update_task(5, "alice", "task-1", "x", 9) -> False   # 已过期，和不存在一样
        complete_task(6, "alice", "task-2") -> True
        get_task_list(6, "alice") -> []
    """)


@case("Level 3", title="到 created_at + ttl 那一刻就过期")
def l3_ttl_boundary(sol):
    run(sol, """
        add_task(3, "alice", "one tick", 1, ttl=1) -> "task-1"   # 只在时刻 3 活跃
        get_task(3, "alice", "task-1") -> Task("task-1", "one tick", 1, 3)
        get_task(4, "alice", "task-1") -> None   # 3 + 1 = 4：这一刻已经过期
        add_task(4, "alice", "five", 2, ttl=5) -> "task-2"   # 在 at_timestamp < 9 时活跃
        get_task(8, "alice", "task-2") -> Task("task-2", "five", 2, 4)
        update_task(8, "alice", "task-2", "five!", 3) -> True
        complete_task(9, "alice", "task-2") -> False   # 9 == 4 + 5：已经过期
        get_task(9, "alice", "task-2") -> None
        add_task(9, "bob", "x", 1, ttl=2) -> "task-3"
        complete_task(10, "bob", "task-3") -> True   # 10 < 11：还没过期，可以完成
        get_task(11, "bob", "task-3") -> None
    """)


@case("Level 3", title="列表跳过已过期的任务")
def l3_list_skips_expired(sol):
    run(sol, """
        add_task(0, "alice", "a", 5, ttl=3) -> "task-1"
        add_task(0, "alice", "b", 1) -> "task-2"
        add_task(1, "alice", "c", 5, ttl=10) -> "task-3"
        add_task(1, "bob", "d", 5, ttl=1) -> "task-4"
        ids(get_task_list(2, "alice")) -> ["task-1", "task-3", "task-2"]
        get_task_list(2, "bob") -> []   # task-4 在 1 + 1 = 2 过期
        ids(get_task_list(3, "alice")) -> ["task-3", "task-2"]   # task-1 在 0 + 3 = 3 过期
        ids(get_task_list(3, "alice", min_priority=5)) -> ["task-3"]
        ids(get_task_list(10, "alice")) -> ["task-3", "task-2"]   # 10 < 1 + 10
        ids(get_task_list(11, "alice")) -> ["task-2"]
        ids(get_task_list(11, "alice", min_priority=2)) -> []
    """)


@case("Level 3", title="过期等同于完成；无 ttl 永不过期")
def l3_expired_like_completed(sol):
    run(sol, """
        add_task(0, "alice", "a", 5, ttl=3) -> "task-1"
        add_task(0, "alice", "b", 1) -> "task-2"
        update_task(3, "alice", "task-1", "a", 9) -> False   # 过期和完成一样：不能修改
        complete_task(4, "alice", "task-1") -> False   # 也不能再完成
        get_task(4, "alice", "task-1") -> None
        add_task(4, "alice", "d", 0) -> "task-3"   # id 不会被重用
        get_task(1000000000, "alice", "task-2") -> Task("task-2", "b", 1, 0)   # 没有 ttl 的任务永不过期
        add_task(1000000000, "alice", "e", 1, ttl=None) -> "task-4"   # 显式传 ttl=None 也一样
        add_task(1000000000, "alice", "f", 1, ttl=1000000000) -> "task-5"
        get_task_list(1999999999, "alice") -> [Task("task-2", "b", 1, 0), Task("task-4", "e", 1, 1000000000), Task("task-5", "f", 1, 1000000000), Task("task-3", "d", 0, 4)]
        get_task_list(2000000000, "alice") -> [Task("task-2", "b", 1, 0), Task("task-4", "e", 1, 1000000000), Task("task-3", "d", 0, 4)]
    """)


@case("Level 3", title="修改不延长过期时间")
def l3_update_keeps_deadline(sol):
    run(sol, """
        add_task(0, "alice", "flush cache", 2, ttl=5) -> "task-1"
        update_task(4, "alice", "task-1", "flush cache now", 3) -> True
        get_task(4, "alice", "task-1") -> Task("task-1", "flush cache now", 3, 0)
        get_task(5, "alice", "task-1") -> None   # update 不会延长期限：仍在 0 + 5 = 5 过期
        update_task(5, "alice", "task-1", "again", 3) -> False
        add_task(5, "alice", "b", 1, ttl=3) -> "task-2"
        update_task(7, "alice", "task-2", "b2", 2) -> True
        get_task_list(7, "alice") -> [Task("task-2", "b2", 2, 5)]
        get_task_list(8, "alice") -> []   # 5 + 3 = 8
    """)


@case("Level 3", title="同一时刻的多个调用")
def l3_same_instant(sol):
    run(sol, """
        add_task(0, "alice", "a", 1, ttl=2) -> "task-1"
        add_task(2, "alice", "b", 1) -> "task-2"   # 时刻 2 也是 task-1 过期的时刻
        update_task(2, "alice", "task-1", "a", 1) -> False   # 2 == 0 + 2：task-1 已过期
        ids(get_task_list(2, "alice")) -> ["task-2"]
        add_task(2, "alice", "c", 3, ttl=1) -> "task-3"
        ids(get_task_list(2, "alice")) -> ["task-3", "task-2"]
        update_task(2, "alice", "task-3", "c2", 0) -> True
        get_task_list(2, "alice") -> [Task("task-2", "b", 1, 2), Task("task-3", "c2", 0, 2)]
        get_task(3, "alice", "task-3") -> None
    """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=300, length=(10, 40))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, """
        add_task(0, "alice", "write report", 3) -> "task-1"
        add_task(2, "alice", "review PR", 5, ttl=10) -> "task-2"   # 在 at_timestamp < 12 时活跃
        update_task(4, "alice", "task-1", "write Q3 report", 4) -> True
        complete_task(6, "alice", "task-2") -> True
        get_task_list(20, "alice") -> [Task("task-1", "write Q3 report", 4, 0)]   # “现在”：task-2 已完成，task-1 已改名
        get_task_list(1, "alice") -> [Task("task-1", "write report", 3, 0)]   # task-2 还不存在（创建于 2 > 1），task-1 还没改名（改于 4 > 1）
        get_task_list(3, "alice") -> [Task("task-2", "review PR", 5, 2), Task("task-1", "write report", 3, 0)]   # task-2 已存在且优先级更高；task-1 在 4 才改名
        get_task_list(6, "alice") -> [Task("task-1", "write Q3 report", 4, 0)]   # task-2 在 6 完成，已经生效（包含等号）
    """)


@case("Level 4", title="过去的时刻看不到之后的修改")
def l4_past_versions(sol):
    run(sol, """
        add_task(0, "alice", "v1", 1) -> "task-1"
        add_task(0, "alice", "other", 3) -> "task-2"
        update_task(5, "alice", "task-1", "v2", 5) -> True
        update_task(8, "alice", "task-1", "v3", 2) -> True
        get_task_list(4, "alice") -> [Task("task-2", "other", 3, 0), Task("task-1", "v1", 1, 0)]   # 时刻 4 还是第一版
        get_task_list(7, "alice") -> [Task("task-1", "v2", 5, 0), Task("task-2", "other", 3, 0)]   # 第二版的优先级 5 决定顺序
        get_task_list(9, "alice") -> [Task("task-2", "other", 3, 0), Task("task-1", "v3", 2, 0)]
        ids(get_task_list(7, "alice", min_priority=4)) -> ["task-1"]   # 按那个时刻的优先级过滤
        ids(get_task_list(4, "alice", min_priority=2)) -> ["task-2"]
        ids(get_task_list(9, "alice", min_priority=4)) -> []
    """)


@case("Level 4", title="过去的时刻看不到之后的完成")
def l4_past_completion(sol):
    run(sol, """
        add_task(1, "alice", "a", 2) -> "task-1"
        add_task(2, "alice", "b", 2) -> "task-2"
        complete_task(5, "alice", "task-1") -> True
        add_task(6, "bob", "c", 1) -> "task-3"
        complete_task(7, "bob", "task-3") -> True
        ids(get_task_list(4, "alice")) -> ["task-1", "task-2"]   # 时刻 4 task-1 还没完成
        ids(get_task_list(5, "alice")) -> ["task-2"]   # 完成发生在时刻 5（包含等号）
        get_task_list(6, "bob") -> [Task("task-3", "c", 1, 6)]
        get_task_list(7, "bob") -> []
        ids(get_task_list(1, "alice")) -> ["task-1"]
        get_task_list(0, "alice") -> []   # 早于 alice 的第一个任务
        get_task(8, "alice", "task-1") -> None
        get_task(8, "alice", "task-2") -> Task("task-2", "b", 2, 2)
    """)


@case("Level 4", title="恰好在调用时刻查询；同一时刻按调用顺序")
def l4_exact_instant(sol):
    run(sol, """
        add_task(0, "alice", "a", 1) -> "task-1"
        update_task(3, "alice", "task-1", "a2", 2) -> True
        update_task(3, "alice", "task-1", "a3", 3) -> True   # 同一时刻的两次修改：后一次生效
        add_task(3, "alice", "b", 2) -> "task-2"
        update_task(6, "alice", "task-2", "b2", 9) -> True
        complete_task(6, "alice", "task-2") -> True   # 同一时刻先修改后完成
        add_task(9, "alice", "c", 0) -> "task-3"
        get_task_list(3, "alice") -> [Task("task-1", "a3", 3, 0), Task("task-2", "b", 2, 3)]   # 时间戳 <= 3 的调用都算
        get_task_list(2, "alice") -> [Task("task-1", "a", 1, 0)]
        get_task_list(5, "alice") -> [Task("task-1", "a3", 3, 0), Task("task-2", "b", 2, 3)]
        get_task_list(6, "alice") -> [Task("task-1", "a3", 3, 0)]
        get_task_list(0, "alice") -> [Task("task-1", "a", 1, 0)]   # 创建的那一刻就已存在
        get_task_list(9, "alice") -> [Task("task-1", "a3", 3, 0), Task("task-3", "c", 0, 9)]
    """)


@case("Level 4", title="过去的时刻同样应用过期规则")
def l4_past_expiry(sol):
    run(sol, """
        add_task(0, "alice", "short", 4, ttl=3) -> "task-1"
        add_task(2, "alice", "long", 1, ttl=10) -> "task-2"
        update_task(2, "alice", "task-1", "short!", 5) -> True
        add_task(20, "alice", "late", 2) -> "task-3"
        get_task_list(20, "alice") -> [Task("task-3", "late", 2, 20)]
        get_task_list(2, "alice") -> [Task("task-1", "short!", 5, 0), Task("task-2", "long", 1, 2)]
        ids(get_task_list(3, "alice")) -> ["task-2"]   # 0 + 3 = 3：task-1 在时刻 3 已过期
        ids(get_task_list(11, "alice")) -> ["task-2"]
        ids(get_task_list(12, "alice")) -> []   # 2 + 10 = 12
        ids(get_task_list(1, "alice", min_priority=4)) -> ["task-1"]
        get_task_list(19, "bob") -> []
    """)


@case("Level 4", title="查询过去不影响之后的调用")
def l4_queries_keep_state(sol):
    run(sol, """
        add_task(0, "alice", "a", 1) -> "task-1"
        add_task(4, "alice", "b", 2, ttl=4) -> "task-2"
        get_task_list(1, "alice") -> [Task("task-1", "a", 1, 0)]
        update_task(5, "alice", "task-2", "b2", 0) -> True   # 查询过去之后，修改照常进行
        get_task(5, "alice", "task-2") -> Task("task-2", "b2", 0, 4)
        ids(get_task_list(4, "alice")) -> ["task-2", "task-1"]
        complete_task(6, "alice", "task-1") -> True
        get_task_list(6, "alice") -> [Task("task-2", "b2", 0, 4)]
        add_task(6, "alice", "c", 1) -> "task-3"
        get_task_list(5, "alice") -> [Task("task-1", "a", 1, 0), Task("task-2", "b2", 0, 4)]
        get_task_list(8, "alice") -> [Task("task-3", "c", 1, 6)]   # task-2 在 4 + 4 = 8 过期
        get_task(8, "alice", "task-3") -> Task("task-3", "c", 1, 6)
    """)


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=300, length=(10, 45))
