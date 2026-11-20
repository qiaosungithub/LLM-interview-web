import ast
import copy
import inspect
import time

from judgelib import calling, case, expect, expect_true, rng
import ref

SHOW = 18           # a replay lists at most this many steps
OWNER = ref.OWNER


# ---------------------------------------------------------------- how a sequence is shown

class Var:
    """An argument shown by its variable name in the replay and passed as the object itself (not a copy)."""

    def __init__(self, name, value):
        self.name, self.value = name, value


def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, Var):
        return x.name
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    if isinstance(x, (list, tuple)):
        body = ", ".join(_lit(v) for v in x)
        return f"[{body}]" if isinstance(x, list) else f"({body}{',' if len(x) == 1 else ''})"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(v)}" for k, v in x.items()) + "}"
    return repr(x)


def _note(want, width=110):
    text = _lit(want)
    return "# -> " + (text if len(text) <= width else text[:width] + " …")


def _code(method, args):
    return f"mgr.{method}({', '.join(_lit(a) for a in args)})"


VIEWS = {   # how a check reads the return value, shown as code
    None: ("{}", lambda g: g),
    "ids": ('[r["id"] for r in {}]', lambda g: [x["id"] for x in g]),
    "name": ('{}["name"]', lambda g: g["name"]),
    "len": ("len({})", len),
}


def _render(head, steps, current, mark="# <- 这一步"):
    """head: setup lines; steps: (code, note) done so far; current: the expression being checked."""
    lines = steps + [(f"print({current})", mark)]
    skipped = max(0, len(lines) - SHOW)
    lines = lines[skipped:]
    width = max(len(c) for c, _ in lines) + 3
    top = head + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(top + [(c.ljust(width) + n) if n else c for c, n in lines])


def _method(obj, name):
    fn = getattr(obj, name, None)
    expect_true(callable(fn), f"RecipeManager 没有 {name} 方法（方法名要和题目一致）")
    return fn


def _norm(x):
    """Tuples where the statement says list are accepted: compare them as lists."""
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    return x


def _same(got, want):
    try:
        return bool(_norm(got) == want)
    except Exception:
        return False


def _users_form(mgr):
    """True when add_recipe already takes Level 3's owner_id (a fourth argument)."""
    try:
        inspect.signature(mgr.add_recipe).bind("r", "n", [], "u")
        return True
    except (TypeError, ValueError, AttributeError):
        return False


USERS_NOTE = (f'mgr.add_user("{OWNER}", "owner")   # 你的 add_recipe 已是 Level 3 的形式：所有菜谱都由 "{OWNER}" 拥有，'
              f'update_recipe 换成它的 edit_recipe')


def _start(sol, level):
    """A fresh RecipeManager. For Level 1-2 tests on a class already in Level 3 form, registers OWNER first."""
    calling("mgr = RecipeManager()")
    mgr = sol.RecipeManager()
    users = level >= 3 or _users_form(mgr)
    head = ["mgr = RecipeManager()"]
    if level <= 2 and users:
        head.append(USERS_NOTE)
        calling("\n".join(head))
        _method(mgr, "add_user")(OWNER, "owner")
    return mgr, users, head


class Session:
    """Runs calls on one RecipeManager, checking each return value as soon as it comes back."""

    def __init__(self, sol, level):
        self.mgr, self.users, self.head = _start(sol, level)
        self.steps = []

    def call(self, method, args, want, why="", view=None):
        if self.users:
            method, args = ref.users_form(method, args)
        expr = VIEWS[view][0].format(_code(method, args))
        real = tuple(a.value if isinstance(a, Var) else copy.deepcopy(a) for a in args)
        fn = _method(self.mgr, method)
        calling(_render(self.head, self.steps, expr))     # shown if the answer raises or hangs in this call
        try:
            raw = fn(*real)
        except Exception:
            calling(_render(self.head, self.steps, expr, "# <- 这一步抛出异常"))
            raise
        try:
            got = VIEWS[view][1](raw)
        except Exception:
            expect_true(False, "返回值的格式不对：应当是题目规定的 dict / dict 组成的列表",
                        call=_render(self.head, self.steps, expr), got=raw)
        if not _same(got, want):
            msg = why or ("" if isinstance(want, (list, str)) else "返回值不对")
            expect(_norm(got), want, call=_render(self.head, self.steps, expr), msg=msg)
        self.steps.append((expr, _note(want)))
        return raw

    def do(self, code, fn):
        """A statement of the replay that is not a checked call, e.g. mutating a list."""
        self.steps.append((code, ""))
        try:
            fn()
        except Exception:
            pass        # e.g. a returned tuple cannot be mutated: nothing to corrupt


def _parse(script):
    """Lines like  ids(list_recipes()) -> ["r1"]   # why  become (method, args, view, want, why)."""
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
        steps.append((node.func.id, args, view, ast.literal_eval(expr.strip()), why.strip()))
    return steps


def run(sol, level, script):
    s = Session(sol, level)
    for method, args, view, want, why in _parse(script):
        s.call(method, args, want, why, view)
    return s


# ---------------------------------------------------------------- random sequences against ref

def _outcome(sol, level, ops):
    """Replays ops on a fresh answer and a fresh ref model; None if every return value agrees."""
    mgr, users, head = _start(sol, level)
    model = ref.RecipeManager()
    if level <= 2:
        model.add_user(OWNER, "owner")
    done, wants = [], []
    for k, (method, args) in enumerate(ops):
        want = ref.apply(model, level, method, copy.deepcopy(args))
        if users:
            method, args = ref.users_form(method, args)
        done.append((method, args))
        fn = _method(mgr, method)
        try:
            got = fn(*copy.deepcopy(args))
        except Exception as e:
            return {"k": k, "exc": e, "wants": wants, "done": done, "head": head}
        if not _same(got, want):
            return {"k": k, "exc": None, "got": got, "want": want, "wants": wants, "done": done, "head": head}
        wants.append(want)
    return None


def _shrink(sol, level, ops, bad, budget=0.6):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, level, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return bad


def random_check(sol, level, sequences, length):
    r = rng()
    for _ in range(sequences):
        ops = ref.random_ops(r, level, r.randint(*length))
        bad = _outcome(sol, level, ops)
        if bad is None:
            continue
        bad = _shrink(sol, level, ops, bad)
        done = bad["done"]
        steps = [(_code(m, a), _note(w)) for (m, a), w in zip(done[:-1], bad["wants"])]
        current = _code(*done[-1])
        if bad["exc"] is not None:
            calling(_render(bad["head"], steps, current, "# <- 这一步抛出异常"))
            raise bad["exc"]
        expect(_norm(bad["got"]), bad["want"], call=_render(bad["head"], steps, current),
               msg=f"随机操作序列的第 {len(done)} 步返回值不对（已自动删去与出错无关的调用）")


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, 1, """
        add_recipe("r1", "Miso Soup", ["miso paste", "tofu", "scallion"]) -> True
        add_recipe("r2", "MISO SOUP", ["dashi", "kombu"]) -> False   # "miso soup" 已被 r1 占用
        add_recipe("r1", "Ramen", ["noodles", "broth"]) -> False   # "r1" 已经存在
        get_recipe("r1") -> {"id": "r1", "name": "Miso Soup", "ingredients": ["miso paste", "tofu", "scallion"]}
        get_recipe("r9") -> None
        update_recipe("r1", "miso soup", ["miso paste", "tofu"]) -> True   # 换个大小写用自己的名字不算冲突
        name(get_recipe("r1")) -> "miso soup"   # 原样存储
        add_recipe("r2", "Chicken Ramen", ["noodles", "broth", "chicken"]) -> True
        update_recipe("r2", "Miso Soup", ["dashi"]) -> False   # 与 r1 冲突（大小写不敏感）
        delete_recipe("r2") -> True
        delete_recipe("r2") -> False   # 已经不存在了
        get_recipe("r2") -> None
    """)


@case("Level 1", title="查找键是 casefold()")
def l1_lookup_key(sol):
    run(sol, 1, """
        add_recipe("r1", "Straße", ["water"]) -> True
        add_recipe("r2", "STRASSE", ["salt"]) -> False   # "STRASSE".casefold() == "Straße".casefold() == "strasse"
        add_recipe("r2", "strasse", ["salt"]) -> False   # 同上
        get_recipe("r2") -> None   # 失败的 add_recipe 没有任何效果
        add_recipe("r3", "Congee", ["rice"]) -> True
        add_recipe("r4", "Congee ", ["rice"]) -> True   # 末尾多一个空格：查找键不同，不冲突
        add_recipe("r5", "cONGEE", ["rice"]) -> False   # 与 r3 冲突
        add_recipe("r5", " Congee", ["rice"]) -> True   # 开头多一个空格也是不同的名字
        get_recipe("r4") -> {"id": "r4", "name": "Congee ", "ingredients": ["rice"]}
        update_recipe("r3", "CONGEE ", ["rice"]) -> False   # "congee " 属于 r4
        update_recipe("r1", "STRASSE", ["water"]) -> True   # 只是换了自己名字的写法
        name(get_recipe("r1")) -> "STRASSE"
    """)


@case("Level 1", title="改名后旧名字空出来")
def l1_rename_frees_name(sol):
    run(sol, 1, """
        add_recipe("r1", "Miso Soup", ["miso paste"]) -> True
        update_recipe("r1", "Miso Soup", ["miso paste", "tofu"]) -> True   # 保留自己当前的名字不算冲突
        update_recipe("r1", "MISO SOUP", ["miso paste"]) -> True   # 只改大小写也不算
        name(get_recipe("r1")) -> "MISO SOUP"
        update_recipe("r1", "Ramen", ["noodles"]) -> True
        add_recipe("r2", "miso soup", ["dashi"]) -> True   # r1 已经改名，"miso soup" 空出来了
        add_recipe("r3", "RAMEN", ["egg"]) -> False   # "ramen" 现在属于 r1
        update_recipe("r1", "Miso soup", ["tofu"]) -> False   # "miso soup" 现在属于 r2
        get_recipe("r1") -> {"id": "r1", "name": "Ramen", "ingredients": ["noodles"]}   # 失败的 update 没有任何效果
        update_recipe("r9", "Toast", ["bread"]) -> False   # "r9" 不存在
        add_recipe("r9", "Toast", ["bread"]) -> True   # 失败的 update 也没有占用名字
        update_recipe("r2", "ramen ", ["dashi"]) -> True
        add_recipe("r4", "Miso Soup", ["miso paste"]) -> True
    """)


@case("Level 1", title="删除同时释放 id 和名字")
def l1_delete_frees(sol):
    run(sol, 1, """
        add_recipe("r1", "Toast", ["bread"]) -> True
        add_recipe("r2", "Jam", ["fruit", "sugar"]) -> True
        delete_recipe("r1") -> True
        get_recipe("r1") -> None
        add_recipe("r3", "TOAST", ["bread", "butter"]) -> True   # 名字 "toast" 已被释放
        add_recipe("r1", "Bagel", ["flour"]) -> True   # id "r1" 也被释放：这是一个无关的新菜谱
        get_recipe("r1") -> {"id": "r1", "name": "Bagel", "ingredients": ["flour"]}
        delete_recipe("r4") -> False   # 从未存在过
        add_recipe("r2", "Bagel", ["x"]) -> False   # id 和名字都被占用
        delete_recipe("r1") -> True
        delete_recipe("r1") -> False
        add_recipe("r4", "bagel", ["flour"]) -> True
        get_recipe("r2") -> {"id": "r2", "name": "Jam", "ingredients": ["fruit", "sugar"]}
    """)


@case("Level 1", title="失败的 add 不占用 id 和名字")
def l1_failed_add(sol):
    run(sol, 1, """
        add_recipe("r1", "Congee", ["rice"]) -> True
        add_recipe("r1", "Toast", ["bread"]) -> False   # "r1" 已存在
        add_recipe("r2", "Toast", ["bread"]) -> True   # 上一步失败了，"toast" 没有被占用
        add_recipe("r3", "congee", ["rice"]) -> False
        add_recipe("r3", "Jook", ["rice"]) -> True   # "r3" 也没有被占用
        get_recipe("r1") -> {"id": "r1", "name": "Congee", "ingredients": ["rice"]}
        add_recipe("r1", "CONGEE", ["water"]) -> False   # 同 id 同名也不行：id 已存在
        get_recipe("r1") -> {"id": "r1", "name": "Congee", "ingredients": ["rice"]}
    """)


@case("Level 1", title="原料按原样存储")
def l1_ingredients_as_given(sol):
    run(sol, 1, """
        add_recipe("r1", "Salad", ["tomato", "Egg", "tomato", "egg"]) -> True
        get_recipe("r1") -> {"id": "r1", "name": "Salad", "ingredients": ["tomato", "Egg", "tomato", "egg"]}   # 不排序、不去重
        add_recipe("r2", "番茄炒蛋", ["番茄", "鸡蛋"]) -> True
        update_recipe("r1", "Salad", ["lettuce", "Lettuce"]) -> True
        get_recipe("r1") -> {"id": "r1", "name": "Salad", "ingredients": ["lettuce", "Lettuce"]}   # 原料整体替换
        update_recipe("r1", "Plain", []) -> True
        get_recipe("r1") -> {"id": "r1", "name": "Plain", "ingredients": []}
        get_recipe("r2") -> {"id": "r2", "name": "番茄炒蛋", "ingredients": ["番茄", "鸡蛋"]}
    """)


@case("Level 1", title="传入和返回的列表都要复制")
def l1_copies(sol):
    s = Session(sol, 1)
    ings = ["rice", "water"]
    s.do('ings = ["rice", "water"]', lambda: None)
    s.call("add_recipe", ("r1", "Congee", Var("ings", ings)), True)
    s.do('ings.append("salt")', lambda: ings.append("salt"))
    s.call("get_recipe", ("r1",), {"id": "r1", "name": "Congee", "ingredients": ["rice", "water"]},
           "存下原料之后再修改传入的列表，不能改变已存储的内容")
    s.do('mgr.get_recipe("r1")["ingredients"].append("intruder")',
         lambda: s.mgr.get_recipe("r1")["ingredients"].append("intruder"))
    s.call("get_recipe", ("r1",), {"id": "r1", "name": "Congee", "ingredients": ["rice", "water"]},
           "修改 get_recipe 返回的列表，不能改变已存储的内容")
    new = ["rice"]
    s.do('new = ["rice"]', lambda: None)
    s.call("update_recipe", ("r1", "Jook", Var("new", new)), True)
    s.do('new.append("ginger")', lambda: new.append("ginger"))
    s.call("get_recipe", ("r1",), {"id": "r1", "name": "Jook", "ingredients": ["rice"]},
           "修改之后再改动传入的列表，不能改变已存储的内容")


@case("Level 1", cpu=2, title="随机操作序列（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, sequences=250, length=(5, 30))


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, 2, """
        add_recipe("r1", "Ramen", ["noodles", "broth", "egg"]) -> True
        add_recipe("r2", "Fried Rice", ["rice", "egg", "soy sauce"]) -> True
        add_recipe("r3", "Toast", ["bread"]) -> True
        add_recipe("r4", "Congee", ["rice", "water"]) -> True
        ids(list_recipes()) -> ["r3", "r4", "r1", "r2"]   # 按数量 1、2、3、3 排列；r1 比 r2 先创建
        ids(find_by_ingredient("EGG")) -> ["r1", "r2"]   # 大小写不敏感匹配；排序规则相同
        ids(search_by_name_prefix("CO")) -> ["r4"]   # 大小写不敏感的前缀匹配，命中 "Congee"
        ids(search_by_name_prefix("")) -> ["r3", "r4", "r1", "r2"]   # 空前缀匹配所有菜谱
    """)


@case("Level 2", title="并列按创建顺序，不按 id 或名字")
def l2_ties_by_creation(sol):
    run(sol, 2, """
        add_recipe("r3", "Zucchini Bake", ["zucchini", "cheese"]) -> True
        add_recipe("r1", "Bean Stew", ["beans", "onion"]) -> True
        add_recipe("r2", "Apple Pie", ["apple", "flour"]) -> True
        add_recipe("r0", "Yogurt", ["milk"]) -> True
        ids(list_recipes()) -> ["r0", "r3", "r1", "r2"]   # 2 种原料的三个菜谱按创建顺序 r3、r1、r2
        ids(search_by_name_prefix("")) -> ["r0", "r3", "r1", "r2"]
        add_recipe("r4", "Avocado Toast", ["avocado", "bread", "salt"]) -> True
        add_recipe("a", "Muesli", ["oats", "milk", "nuts"]) -> True
        ids(list_recipes()) -> ["r0", "r3", "r1", "r2", "r4", "a"]
        ids(find_by_ingredient("milk")) -> ["r0", "a"]   # 1 种原料排在 3 种之前
    """)


@case("Level 2", title="修改不改变菜谱的先后位置")
def l2_update_keeps_place(sol):
    run(sol, 2, """
        add_recipe("r1", "A-recipe", ["x", "y"]) -> True
        add_recipe("r2", "B-recipe", ["x"]) -> True
        add_recipe("r3", "C-recipe", ["x", "y"]) -> True
        ids(list_recipes()) -> ["r2", "r1", "r3"]
        update_recipe("r2", "B-recipe", ["x", "y"]) -> True   # 三个都是 2 种原料：仍按创建顺序 r1、r2、r3
        ids(list_recipes()) -> ["r1", "r2", "r3"]
        update_recipe("r1", "Renamed", ["y", "x"]) -> True   # 刚被修改过也不会移到最后
        ids(list_recipes()) -> ["r1", "r2", "r3"]
        update_recipe("r3", "C-recipe", ["z"]) -> True
        ids(list_recipes()) -> ["r3", "r1", "r2"]
        update_recipe("r3", "C-recipe", ["z", "w"]) -> True
        ids(list_recipes()) -> ["r1", "r2", "r3"]
    """)


@case("Level 2", title="删除后重用 id 算新创建")
def l2_reused_id_last(sol):
    run(sol, 2, """
        add_recipe("r1", "One", ["a"]) -> True
        add_recipe("r2", "Two", ["b"]) -> True
        add_recipe("r3", "Three", ["c"]) -> True
        delete_recipe("r1") -> True
        add_recipe("r1", "One", ["a"]) -> True   # 新的一次创建，排到最后
        ids(list_recipes()) -> ["r2", "r3", "r1"]
        add_recipe("r2", "Again", ["d"]) -> False   # 失败的 add 不算创建
        ids(list_recipes()) -> ["r2", "r3", "r1"]
        add_recipe("r4", "two", ["e"]) -> False   # 名字冲突：r4 没有被创建
        add_recipe("r5", "Five", ["f"]) -> True
        add_recipe("r4", "Four", ["g"]) -> True   # r4 第一次真正创建，在 r5 之后
        ids(search_by_name_prefix("")) -> ["r2", "r3", "r1", "r5", "r4"]
    """)


@case("Level 2", title="新建的菜谱排在所有现存菜谱之后")
def l2_created_after_survivors(sol):
    run(sol, 2, """
        add_recipe("r1", "One", ["a", "b"]) -> True
        add_recipe("r2", "Two", ["a", "b"]) -> True
        add_recipe("r3", "Three", ["a", "b"]) -> True
        delete_recipe("r1") -> True
        delete_recipe("r2") -> True
        add_recipe("r4", "Four", ["c", "d"]) -> True   # 比 r3 晚创建；与此刻还剩几个菜谱无关
        ids(list_recipes()) -> ["r3", "r4"]
        add_recipe("r5", "Five", ["e", "f"]) -> True
        ids(search_by_name_prefix("")) -> ["r3", "r4", "r5"]
        ids(find_by_ingredient("A")) -> ["r3"]
    """)


@case("Level 2", title="原料整项匹配，大小写不敏感")
def l2_find_exact(sol):
    run(sol, 2, """
        add_recipe("r1", "Boiled Egg", ["egg"]) -> True
        add_recipe("r2", "Baba Ganoush", ["eggplant"]) -> True
        add_recipe("r3", "Omelette", ["Egg", "egg", "EGG"]) -> True
        add_recipe("r4", "Breakfast", ["scrambled egg"]) -> True
        add_recipe("r5", "Odd", ["egg "]) -> True
        ids(find_by_ingredient("egg")) -> ["r1", "r3"]   # 整项相等才算；同一菜谱只出现一次
        ids(find_by_ingredient("eGG")) -> ["r1", "r3"]
        ids(find_by_ingredient("EGGPLANT")) -> ["r2"]
        ids(find_by_ingredient("plant")) -> []   # 不是子串匹配
        ids(find_by_ingredient("egg ")) -> ["r5"]
        ids(find_by_ingredient("tofu")) -> []
    """)


@case("Level 2", title="名称前缀匹配，大小写不敏感")
def l2_prefix(sol):
    run(sol, 2, """
        add_recipe("r1", "Congee", ["rice", "water"]) -> True
        add_recipe("r2", "Corn Soup", ["corn"]) -> True
        add_recipe("r3", "congee ", ["rice"]) -> True
        add_recipe("r4", "Toast", ["bread"]) -> True
        add_recipe("r5", "粥", ["米", "水"]) -> True
        ids(search_by_name_prefix("co")) -> ["r2", "r3", "r1"]
        ids(search_by_name_prefix("CONGEE")) -> ["r3", "r1"]
        ids(search_by_name_prefix("Congee ")) -> ["r3"]
        ids(search_by_name_prefix("Toasted")) -> []   # 前缀比名字还长
        ids(search_by_name_prefix("oast")) -> []   # 只匹配开头
        ids(search_by_name_prefix("粥")) -> ["r5"]
        ids(search_by_name_prefix("")) -> ["r2", "r3", "r4", "r1", "r5"]
    """)


@case("Level 2", title="空管理器；结果反映当前状态")
def l2_current_state(sol):
    s = run(sol, 2, """
        list_recipes() -> []
        find_by_ingredient("rice") -> []
        search_by_name_prefix("") -> []
        add_recipe("r1", "Congee", ["rice", "water"]) -> True
        add_recipe("r2", "Toast", ["bread"]) -> True
        update_recipe("r1", "Jook", ["rice", "ginger", "water"]) -> True
        list_recipes() -> [{"id": "r2", "name": "Toast", "ingredients": ["bread"]}, {"id": "r1", "name": "Jook", "ingredients": ["rice", "ginger", "water"]}]
        ids(search_by_name_prefix("Con")) -> []   # 旧名字不再匹配
        ids(search_by_name_prefix("jo")) -> ["r1"]
        ids(find_by_ingredient("ginger")) -> ["r1"]
        delete_recipe("r2") -> True
        find_by_ingredient("bread") -> []
        list_recipes() -> [{"id": "r1", "name": "Jook", "ingredients": ["rice", "ginger", "water"]}]
    """)
    s.do('mgr.list_recipes()[0]["ingredients"].append("intruder")',
         lambda: s.mgr.list_recipes()[0]["ingredients"].append("intruder"))
    s.do('mgr.find_by_ingredient("rice")[0]["ingredients"].clear()',
         lambda: s.mgr.find_by_ingredient("rice")[0]["ingredients"].clear())
    s.do('mgr.search_by_name_prefix("j")[0]["ingredients"].pop()',
         lambda: s.mgr.search_by_name_prefix("j")[0]["ingredients"].pop())
    s.call("get_recipe", ("r1",), {"id": "r1", "name": "Jook", "ingredients": ["rice", "ginger", "water"]},
           "修改列表方法返回的原料列表，不能改变已存储的内容")


@case("Level 2", cpu=2, title="随机操作序列（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, sequences=250, length=(5, 30))


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, 3, """
        add_user("u1", "Alice") -> True
        add_user("u2", "Bob") -> True
        add_user("u1", "Alice2") -> False   # "u1" 已被占用
        add_recipe("r1", "Congee", ["rice", "water"], "u1") -> True   # u1 拥有 r1
        add_recipe("r2", "Toast", ["bread"], "nope") -> False   # "nope" 不是用户；r2 未被创建
        edit_recipe("u2", "r1", "Rice Congee", ["rice", "water", "ginger"]) -> False   # u2 既不是拥有者也不是编辑者
        grant_editor("u2", "r1", "u2") -> False   # u2 不是 r1 的拥有者，无权授予
        grant_editor("u1", "r1", "u2") -> True   # 拥有者 u1 把编辑权限授予 u2
        edit_recipe("u2", "r1", "Rice Congee", ["rice", "water", "ginger"]) -> True   # u2 现在是被授权的编辑者
        delete_recipe("u2", "r1") -> False   # 编辑者可以编辑，但不能删除
        delete_recipe("u1", "r1") -> True   # 拥有者可以删除
    """)


@case("Level 3", title="注册用户；拥有者必须已注册")
def l3_users_and_owner(sol):
    run(sol, 3, """
        add_user("u1", "Alice") -> True
        add_user("u1", "Bob") -> False
        add_user("u2", "Alice") -> True   # username 不做任何检查，可以重复
        add_user("u3", "") -> True
        add_recipe("r1", "Congee", ["rice"], "nope") -> False   # owner 未注册
        get_recipe("r1") -> None
        add_recipe("r1", "Congee", ["rice"], "u1") -> True   # 失败的 add 没有占用 id 和名字
        add_recipe("r1", "Toast", ["bread"], "u2") -> False   # 第一关的两个条件仍然有效
        add_recipe("r2", "CONGEE", ["rice"], "u2") -> False
        add_recipe("r2", "Toast", ["bread"], "u3") -> True
        get_recipe("r2") -> {"id": "r2", "name": "Toast", "ingredients": ["bread"]}   # 读取对所有人开放
        ids(list_recipes()) -> ["r1", "r2"]
    """)


@case("Level 3", title="只有拥有者能授予编辑权限")
def l3_grant_rules(sol):
    run(sol, 3, """
        add_user("u1", "Alice") -> True
        add_user("u2", "Bob") -> True
        add_user("u3", "Carol") -> True
        add_recipe("r1", "Congee", ["rice"], "u1") -> True
        add_recipe("r2", "Toast", ["bread"], "u2") -> True
        grant_editor("u2", "r1", "u3") -> False   # u2 不是 r1 的拥有者
        grant_editor("u1", "r1", "u2") -> True
        grant_editor("u2", "r1", "u3") -> False   # 编辑者也不能授权
        edit_recipe("u3", "r1", "Jook", ["rice"]) -> False   # 上面的授权都失败了
        grant_editor("u1", "r1", "u2") -> True   # 重复授予：返回 True，没有别的变化
        grant_editor("u1", "r1", "nope") -> False   # user_id 未注册
        grant_editor("u1", "r9", "u2") -> False   # 没有这个菜谱
        grant_editor("nope", "r1", "u3") -> False
        grant_editor("u1", "r2", "u3") -> False   # u1 拥有的是 r1，不是 r2
        edit_recipe("u3", "r2", "Jook", ["rice"]) -> False
        add_user("nope", "Late") -> True
        edit_recipe("nope", "r1", "Jook", ["rice"]) -> False   # 早先对 "nope" 的授权失败了，没有留下任何效果
    """)


@case("Level 3", title="拥有者和编辑者能编辑")
def l3_edit_rules(sol):
    run(sol, 3, """
        add_user("u1", "Alice") -> True
        add_user("u2", "Bob") -> True
        add_user("u3", "Carol") -> True
        add_recipe("r1", "Congee", ["rice", "water"], "u1") -> True
        add_recipe("r2", "Toast", ["bread"], "u3") -> True
        edit_recipe("u1", "r1", "CONGEE", ["rice"]) -> True   # 拥有者；只改大小写不算冲突
        grant_editor("u1", "r1", "u2") -> True
        edit_recipe("u2", "r1", "Jook", ["rice", "ginger"]) -> True
        edit_recipe("u2", "r1", "toast", ["bread"]) -> False   # 名字被 r2 占用
        edit_recipe("u1", "r1", "TOAST", ["bread"]) -> False   # 拥有者也一样
        get_recipe("r1") -> {"id": "r1", "name": "Jook", "ingredients": ["rice", "ginger"]}
        edit_recipe("u2", "r2", "Bread", ["bread"]) -> False   # u2 只是 r1 的编辑者
        edit_recipe("u3", "r1", "Porridge", ["rice"]) -> False   # u3 拥有的是 r2
        edit_recipe("ghost", "r1", "Porridge", ["rice"]) -> False   # 未注册的用户
        edit_recipe("u1", "r9", "Porridge", ["rice"]) -> False   # 没有这个菜谱
        get_recipe("r1") -> {"id": "r1", "name": "Jook", "ingredients": ["rice", "ginger"]}
        add_recipe("r3", "congee", ["rice"], "u2") -> True   # r1 改名后 "congee" 空了出来
        edit_recipe("u3", "r2", "Toast", ["bread", "butter"]) -> True
        ids(find_by_ingredient("BUTTER")) -> ["r2"]
    """)


@case("Level 3", title="只有拥有者能删除")
def l3_delete_rules(sol):
    run(sol, 3, """
        add_user("u1", "Alice") -> True
        add_user("u2", "Bob") -> True
        add_user("u3", "Carol") -> True
        add_recipe("r1", "Congee", ["rice"], "u1") -> True
        grant_editor("u1", "r1", "u2") -> True
        delete_recipe("u2", "r1") -> False   # 编辑者不能删除
        delete_recipe("u3", "r1") -> False
        delete_recipe("ghost", "r1") -> False
        delete_recipe("u1", "r9") -> False   # 没有这个菜谱
        get_recipe("r1") -> {"id": "r1", "name": "Congee", "ingredients": ["rice"]}
        delete_recipe("u1", "r1") -> True
        delete_recipe("u1", "r1") -> False   # 已经删除了
        add_recipe("r2", "CONGEE", ["rice"], "u3") -> True   # 名字已被释放
    """)


@case("Level 3", title="删除后重建：权限不继承")
def l3_recreated_recipe(sol):
    run(sol, 3, """
        add_user("u1", "Alice") -> True
        add_user("u2", "Bob") -> True
        add_user("u3", "Carol") -> True
        add_recipe("r1", "Old", ["x"], "u1") -> True
        grant_editor("u1", "r1", "u2") -> True
        delete_recipe("u1", "r1") -> True
        add_recipe("r1", "New", ["z"], "u3") -> True   # 无关的新菜谱，拥有者是 u3
        edit_recipe("u2", "r1", "New Two", ["z"]) -> False   # u2 只是旧 r1 的编辑者
        edit_recipe("u1", "r1", "New Two", ["z"]) -> False   # u1 拥有的是旧 r1
        delete_recipe("u1", "r1") -> False
        grant_editor("u1", "r1", "u2") -> False
        grant_editor("u3", "r1", "u2") -> True
        edit_recipe("u2", "r1", "New Two", ["z"]) -> True
        get_recipe("r1") -> {"id": "r1", "name": "New Two", "ingredients": ["z"]}
    """)


@case("Level 3", cpu=2, title="随机操作序列（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, sequences=300, length=(10, 40))


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, 4, """
        add_user("u1", "Alice") -> True
        add_recipe("r1", "Congee", ["rice", "water"], "u1") -> True   # v1 = "Congee"
        edit_recipe("u1", "r1", "Rice Porridge", ["rice", "water", "salt"]) -> True   # v2；"Congee" 空出来了
        edit_recipe("u1", "r1", "Jook", ["rice", "water", "ginger"]) -> True   # v3
        get_recipe_history("r1") -> [{"version": 1, "name": "Congee", "ingredients": ["rice", "water"]}, {"version": 2, "name": "Rice Porridge", "ingredients": ["rice", "water", "salt"]}, {"version": 3, "name": "Jook", "ingredients": ["rice", "water", "ginger"]}]
        get_version("r1", 1) -> {"version": 1, "name": "Congee", "ingredients": ["rice", "water"]}
        get_version("r1", 5) -> None   # 目前只有 3 个版本
        add_recipe("r2", "Congee", ["rice"], "u1") -> True   # "Congee" 又空出来了
        rollback_recipe("u1", "r1", 1) -> False   # v1 的名字现在属于 r2
        grant_editor("u1", "r1", "u2") -> False   # "u2" 还不是注册用户
        add_user("u2", "Bob") -> True
        grant_editor("u1", "r1", "u2") -> True
        rollback_recipe("u2", "r1", 2) -> True   # v4 = v2 的副本 = "Rice Porridge"
        get_recipe("r1") -> {"id": "r1", "name": "Rice Porridge", "ingredients": ["rice", "water", "salt"]}
        len(get_recipe_history("r1")) -> 4
        edit_recipe("u2", "r1", "Jook", ["rice", "water", "ginger"]) -> True   # 回滚并没有撤销 u2 的授权
        delete_recipe("u2", "r1") -> False   # u2 依然只是编辑者
        delete_recipe("u1", "r1") -> True
    """)


@case("Level 4", title="只有成功的编辑追加版本")
def l4_versions(sol):
    run(sol, 4, """
        add_user("u1", "Alice") -> True
        add_user("u2", "Bob") -> True
        add_recipe("r1", "Soup", ["water"], "u1") -> True
        add_recipe("r2", "Toast", ["bread"], "u1") -> True
        get_recipe_history("r1") -> [{"version": 1, "name": "Soup", "ingredients": ["water"]}]   # add_recipe 创建第 1 版
        edit_recipe("u2", "r1", "Broth", ["water"]) -> False   # 未授权
        edit_recipe("u1", "r1", "TOAST", ["bread"]) -> False   # 名字冲突
        get_recipe_history("r1") -> [{"version": 1, "name": "Soup", "ingredients": ["water"]}]   # 失败的编辑不留下版本
        edit_recipe("u1", "r1", "Soup", ["water"]) -> True   # 内容不变的编辑也追加一个版本
        get_version("r1", 2) -> {"version": 2, "name": "Soup", "ingredients": ["water"]}
        edit_recipe("u1", "r1", "SOUP", ["water", "salt"]) -> True
        get_recipe("r1") -> {"id": "r1", "name": "SOUP", "ingredients": ["water", "salt"]}   # get_recipe 返回最新版本，不带版本号
        get_version("r1", 3) -> {"version": 3, "name": "SOUP", "ingredients": ["water", "salt"]}
        get_version("r1", 0) -> None   # 版本号从 1 开始
        get_version("r1", -1) -> None
        get_version("r1", 4) -> None
        get_version("r9", 1) -> None
        get_recipe_history("r9") -> None
        get_recipe_history("r2") -> [{"version": 1, "name": "Toast", "ingredients": ["bread"]}]   # 每个菜谱各自编号
    """)


@case("Level 4", title="回滚追加副本，不截断历史")
def l4_rollback_appends(sol):
    run(sol, 4, """
        add_user("u1", "Alice") -> True
        add_recipe("r1", "A", ["a"], "u1") -> True
        edit_recipe("u1", "r1", "B", ["b", "b"]) -> True
        edit_recipe("u1", "r1", "C", ["c"]) -> True
        rollback_recipe("u1", "r1", 1) -> True   # v4 = v1 的副本
        get_recipe_history("r1") -> [{"version": 1, "name": "A", "ingredients": ["a"]}, {"version": 2, "name": "B", "ingredients": ["b", "b"]}, {"version": 3, "name": "C", "ingredients": ["c"]}, {"version": 4, "name": "A", "ingredients": ["a"]}]
        rollback_recipe("u1", "r1", 4) -> True   # 可以回滚到当前最新版本：追加一份相同的副本
        get_version("r1", 5) -> {"version": 5, "name": "A", "ingredients": ["a"]}
        rollback_recipe("u1", "r1", 3) -> True
        get_recipe("r1") -> {"id": "r1", "name": "C", "ingredients": ["c"]}
        rollback_recipe("u1", "r1", 0) -> False   # 版本号越界
        rollback_recipe("u1", "r1", -1) -> False
        rollback_recipe("u1", "r1", 7) -> False   # 只有 6 个版本
        rollback_recipe("u1", "r9", 1) -> False   # 没有这个菜谱
        len(get_recipe_history("r1")) -> 6   # 失败的回滚不留下版本
        get_version("r1", 2) -> {"version": 2, "name": "B", "ingredients": ["b", "b"]}
        ids(list_recipes()) -> ["r1"]
    """)


@case("Level 4", title="回滚的名字冲突规则")
def l4_rollback_collision(sol):
    run(sol, 4, """
        add_user("u1", "Alice") -> True
        add_recipe("r1", "Jook", ["rice"], "u1") -> True
        edit_recipe("u1", "r1", "JOOK", ["rice", "ginger"]) -> True
        rollback_recipe("u1", "r1", 1) -> True   # 自己早先的名字不会和自己当前的名字冲突
        edit_recipe("u1", "r1", "Congee", ["rice", "water"]) -> True
        add_recipe("r2", "jook", ["rice"], "u1") -> True
        rollback_recipe("u1", "r1", 2) -> False   # "JOOK" 现在属于 r2（大小写不敏感）
        rollback_recipe("u1", "r1", 4) -> True   # "Congee" 是自己当前的名字
        edit_recipe("u1", "r2", "Toast", ["bread"]) -> True   # r2 改名，"jook" 空了出来
        rollback_recipe("u1", "r1", 2) -> True
        get_recipe("r1") -> {"id": "r1", "name": "JOOK", "ingredients": ["rice", "ginger"]}
        add_recipe("r3", "Straße", ["water"], "u1") -> True
        edit_recipe("u1", "r3", "Strasse Soup", ["water"]) -> True
        add_recipe("r4", "STRASSE", ["salt"], "u1") -> True
        rollback_recipe("u1", "r3", 1) -> False   # "Straße" 的查找键 "strasse" 属于 r4
        len(get_recipe_history("r3")) -> 2
    """)


@case("Level 4", title="回滚的授权；权限保持不变")
def l4_rollback_permissions(sol):
    run(sol, 4, """
        add_user("u1", "Alice") -> True
        add_user("u2", "Bob") -> True
        add_user("u3", "Carol") -> True
        add_recipe("r1", "Congee", ["rice"], "u1") -> True   # v1：此时还没有编辑者
        edit_recipe("u1", "r1", "Jook", ["rice"]) -> True
        grant_editor("u1", "r1", "u2") -> True
        edit_recipe("u2", "r1", "Porridge", ["rice", "salt"]) -> True   # v3
        rollback_recipe("u3", "r1", 1) -> False   # u3 不是拥有者也不是编辑者
        rollback_recipe("ghost", "r1", 1) -> False
        rollback_recipe("u2", "r1", 1) -> True   # 编辑者可以回滚
        edit_recipe("u2", "r1", "Rice Soup", ["rice"]) -> True   # 回到 v1 不会撤销 u2 的授权
        rollback_recipe("u1", "r1", 1) -> True
        grant_editor("u2", "r1", "u3") -> False   # u2 仍然只是编辑者
        delete_recipe("u2", "r1") -> False
        grant_editor("u1", "r1", "u3") -> True   # u1 仍然是拥有者
        rollback_recipe("u3", "r1", 3) -> True
        len(get_recipe_history("r1")) -> 7
        delete_recipe("u1", "r1") -> True
    """)


@case("Level 4", title="删除后重建：历史从头开始")
def l4_recreated_history(sol):
    run(sol, 4, """
        add_user("u1", "Alice") -> True
        add_user("u2", "Bob") -> True
        add_recipe("r1", "Old", ["x"], "u1") -> True
        edit_recipe("u1", "r1", "Old Two", ["x", "y"]) -> True
        grant_editor("u1", "r1", "u2") -> True
        delete_recipe("u1", "r1") -> True
        get_recipe_history("r1") -> None
        add_recipe("r1", "New", ["z"], "u2") -> True
        get_recipe_history("r1") -> [{"version": 1, "name": "New", "ingredients": ["z"]}]
        get_version("r1", 2) -> None
        rollback_recipe("u1", "r1", 1) -> False   # u1 拥有的是旧 r1
        rollback_recipe("u2", "r1", 2) -> False   # 新 r1 只有 1 个版本
        rollback_recipe("u2", "r1", 1) -> True
        len(get_recipe_history("r1")) -> 2
    """)


@case("Level 4", title="版本里的原料列表要复制")
def l4_copies(sol):
    s = run(sol, 4, """
        add_user("u1", "Alice") -> True
        add_recipe("r1", "Congee", ["rice", "water"], "u1") -> True
    """)
    ings = ["rice", "water", "salt"]
    s.do('ings = ["rice", "water", "salt"]', lambda: None)
    s.call("edit_recipe", ("u1", "r1", "Porridge", Var("ings", ings)), True)
    s.do('ings.append("intruder")', lambda: ings.append("intruder"))
    s.call("get_version", ("r1", 2), {"version": 2, "name": "Porridge", "ingredients": ["rice", "water", "salt"]},
           "编辑之后再修改传入的列表，不能改变已记录的版本")
    s.do('mgr.get_version("r1", 1)["ingredients"].append("intruder")',
         lambda: s.mgr.get_version("r1", 1)["ingredients"].append("intruder"))
    s.do('mgr.get_recipe_history("r1")[1]["ingredients"].clear()',
         lambda: s.mgr.get_recipe_history("r1")[1]["ingredients"].clear())
    s.do('mgr.get_recipe("r1")["ingredients"].pop()', lambda: s.mgr.get_recipe("r1")["ingredients"].pop())
    s.call("get_recipe_history", ("r1",),
           [{"version": 1, "name": "Congee", "ingredients": ["rice", "water"]},
            {"version": 2, "name": "Porridge", "ingredients": ["rice", "water", "salt"]}],
           "修改返回的原料列表，不能改变已记录的版本")
    s.call("rollback_recipe", ("u1", "r1", 1), True)
    s.do('mgr.get_recipe("r1")["ingredients"].append("intruder")',
         lambda: s.mgr.get_recipe("r1")["ingredients"].append("intruder"))
    s.call("get_version", ("r1", 1), {"version": 1, "name": "Congee", "ingredients": ["rice", "water"]},
           "回滚得到的新版本不能和旧版本共用同一个列表")
    s.call("get_recipe", ("r1",), {"id": "r1", "name": "Congee", "ingredients": ["rice", "water"]})


@case("Level 4", cpu=2, title="随机操作序列（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, sequences=300, length=(10, 45))
