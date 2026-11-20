import copy
import itertools
import json

from judgelib import Mismatch, calling, case, expect, expect_true, rng
import ref

CLASSES = {"list": "ResumableListIterator", "2d": "Resumable2DIterator", "nested": "ResumableNestedIterator"}
MODEL = object()   # "take the expected value from ref.Naive"
STOP = object()    # next() must raise StopIteration
ABSTRACT = ("__iter__", "__next__", "get_state", "set_state")
REJECTED = "被拒绝的状态不改变迭代器的位置"
KEEP = 4           # older get_state checks and rejected set_state calls are left out of the replay


class Text:
    """Shown as-is (no quotes) in the expected / actual lines."""

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


def need(sol, *names):
    for name in names:
        expect_true(isinstance(getattr(sol, name, None), type), f"没有找到类 {name}（名字要和题目一致）")


class Session:
    """Iterators driven one operation at a time and checked against ref.Naive after each. The replay of every
    operation so far goes to calling() before the next one. The replay keeps every operation that moves an
    iterator, and only the last KEEP checks (get_state, rejected set_state): each of those was followed by a check
    that the position had not moved.

    The case's `why` is shown only for set_state, where it describes the very state being tried. A wrong next() or
    get_state() can come from any bug, so those messages say what was expected and why from the data itself."""

    def __init__(self, sol, why=""):
        self.sol, self.why = sol, why
        self.lines, self.its, self.models, self.kinds = [], {}, {}, {}
        self.var = None
        self.rejected = False
        self.checked = set()       # iterators whose get_state() was checked with no move since

    def _tail(self, why):
        return f"。这个测试检查的是：{why}" if why else ""

    def _call(self, current):
        checks = [i for i, (_, moves) in enumerate(self.lines) if not moves][-KEEP:]
        shown = [text for i, (text, moves) in enumerate(self.lines) if moves or i in checks]
        return calling("\n".join(shown + [current]))

    def _done(self, text, var, moves=True):
        self.lines.append((text, moves))
        if moves:
            self.checked.discard(var)

    def new(self, kind, items, depth=None, var="it"):
        name = CLASSES[kind]
        need(self.sol, "ResumableIterator", name)
        args = repr(items) if kind != "nested" else f"{items!r}, {depth}"
        code = f"{var} = {name}({args})"
        self._call(f"{code}   # <- 这一步")
        cls = getattr(self.sol, name)
        data = copy.deepcopy(items)
        self.its[var] = cls(data) if kind != "nested" else cls(data, depth)
        self.models[var] = ref.Naive(copy.deepcopy(items), {"list": 1, "2d": 2}.get(kind, depth))
        self.kinds[var] = kind
        self._done(code, var)
        self.var = var
        return self

    def n(self, want=MODEL, var=None):
        var = var or self.var
        it, model = self.its[var], self.models[var]
        has, item = model.next()
        if want is MODEL:
            want = item if has else STOP
        if want is STOP:
            self._call(f"next({var})   # <- 这一步，应当抛出 StopIteration")
            try:
                got = next(it)
            except StopIteration:
                self._done(f"try: next({var})   # StopIteration\nexcept StopIteration: pass", var)
                return self
            except Exception as e:
                raise Mismatch(f"next({var}) 应当抛出 StopIteration（已经没有剩余元素），实际抛出 "
                               f"{type(e).__name__}: {e}", show=False) from None
            raise Mismatch(f"next({var}) 应当抛出 StopIteration（已经没有剩余元素），但返回了一个元素",
                           want=Text("抛出 StopIteration"), got=got)
        where = where_text(model.paths[model.pos - 1])
        if isinstance(want, list) and self.kinds[var] == "nested":
            where += "（取够 depth 层下标得到的就是元素，即使它本身是列表）"
        self._call(f"print(next({var}))   # <- 这一步，应当得到 {want!r}")
        try:
            got = next(it)
        except StopIteration:
            raise Mismatch(f"next({var}) 应当返回 {where}，但抛出了 StopIteration", show=False) from None
        expect(got, want, msg=f"next({var}) 应当返回 {where}")
        self._done(f"print(next({var}))   # {want!r}", var)
        return self

    def s(self, want=MODEL, var=None, why=None):
        var = var or self.var
        it, model = self.its[var], self.models[var]
        if want is MODEL:
            want = model.state()
        if model.pos == len(model.paths):
            hint = f"（已经没有剩余元素，耗尽状态是 {END_TEXT[min(model.d, 3)]}）"
        else:
            hint = f"（下一次 next 将产出 {where_text(model.paths[model.pos])}）"
        tail = self._tail(why)
        self._call(f"print({var}.get_state())   # <- 这一步，应当得到 {want!r}")
        got = it.get_state()
        if not isinstance(got, tuple):
            raise Mismatch(f"{var}.get_state() 应当返回 tuple（题目规定状态是一个元组），得到的是 "
                           f"{type(got).__name__}{tail}", want=want, got=got)
        expect(got, want, msg=f"{var}.get_state() 应当返回 {want!r}{hint}{tail}")
        bad = [v for v in got if type(v) is not int]
        expect_true(not bad, f"{var}.get_state() 里的下标应当是 int，得到了 {type(bad[0]).__name__ if bad else ''}{tail}",
                    got=got)
        self._done(f"print({var}.get_state())   # {want!r}", var, moves=False)
        self.checked.add(var)
        return self

    def set(self, state, want=MODEL, var=None, why=None, note=""):
        """set_state(state); want is None (accepted), "TypeError" or "ValueError"."""
        var = var or self.var
        it, model = self.its[var], self.models[var]
        if want is MODEL:
            want = model.verdict(state)
        tail = self._tail(self.why if why is None else why)
        code = f"{var}.set_state({state!r})"
        self.rejected = want is not None
        if want is None:
            self._call(f"{code}   # <- 这一步{note}，应当成功")
            try:
                it.set_state(copy.deepcopy(state))
            except (TypeError, ValueError) as e:
                raise Mismatch(f"{code} 不应当抛出异常（{state!r} 对这个迭代器的数据是规范的），实际抛出 "
                               f"{type(e).__name__}: {e}{tail}", show=False) from None
            model.seek(state)
            self._done(code + (f"   # {note.strip('（）')}" if note else ""), var)
            return self
        self._call(f"{code}   # <- 这一步{note}，应当抛出 {want}")
        try:
            it.set_state(copy.deepcopy(state))
        except (TypeError if want == "TypeError" else ValueError):
            self._done(f"try: {code}   # {want}\nexcept {want}: pass", var, moves=False)
            return self
        except Exception as e:
            raise Mismatch(f"{code} 应当抛出 {want}，实际抛出 {type(e).__name__}: {e}{tail}", show=False) from None
        raise Mismatch(f"{code} 应当抛出 {want}，但正常返回了{tail}", show=False)

    def bad(self, state, var=None, why=None):
        """A state that must be rejected, with get_state() checked before and after: the position must not move."""
        var = var or self.var
        if var not in self.checked:
            self.s(var=var)
        self.set(state, var=var, why=why)
        expect_true(self.rejected, f"测试自身出错：{state!r} 应当是被拒绝的状态")
        return self.s(var=var, why=REJECTED)

    def rest(self, var=None):
        """list(it) gives every remaining item and leaves the iterator exhausted."""
        var = var or self.var
        it, model = self.its[var], self.models[var]
        want = model.remaining()
        self._call(f"print(list({var}))   # <- 这一步，应当得到 {want!r}")
        got = list(itertools.islice(it, len(want) + 3))
        expect(got, want, msg=f"list({var}) 应当给出剩下的所有元素，并在最后一个之后停止")
        model.pos = len(model.paths)
        self._done(f"print(list({var}))   # {want!r}", var)
        return self


def where_text(path):
    return "items" + "".join(f"[{i}]" for i in path)


END_TEXT = {1: "(len(items),)", 2: "(len(items), 0)", 3: "(len(items), 0, …, 0)"}


def random_session(sol, r, kind, data, depth=None, ops=10):
    d = {"list": 1, "2d": 2}.get(kind, depth)
    t = Session(sol).new(kind, data, depth)
    model, saved = t.models["it"], []
    for _ in range(ops):
        x = r.random()
        if x < 0.4:
            t.n()
        elif x < 0.55:
            t.s()
            saved.append(model.state())
        elif x < 0.7 and saved:
            state = r.choice(saved)
            if r.random() < 0.5:
                t.set(json.loads(json.dumps(state)), note="（经过 JSON 往返）")
            else:
                t.set(state)
        else:
            state = ref.random_state(r, model)
            if model.verdict(state) is None:
                t.set(state)
            else:
                t.bad(state)
    if r.random() < 0.4:              # hand the state to an iterator over other data
        state = model.state()
        t.s()
        other = ref.relabel(r, data, d) if r.random() < 0.5 else ref.gen(r, d, list_leaves=kind == "nested")
        t.new(kind, other, depth, var="other").set(state)
        if not t.rejected:
            t.rest()


# ---------------------------------------------------------------- Part 1

INIT = "    def __init__(self, n=3):\n        self.n, self.pos = n, 0\n"
METHODS = {
    "__iter__": "    def __iter__(self):\n        return self\n",
    "__next__": ("    def __next__(self):\n        if self.pos == self.n:\n            raise StopIteration\n"
                 "        self.pos += 1\n        return self.n - self.pos + 1\n"),
    "get_state": "    def get_state(self):\n        return (self.pos,)\n",
    "set_state": "    def set_state(self, state):\n        self.pos = state[0]\n",
}


def subclass(name, methods):
    return f"class {name}(ResumableIterator):\n" + "\n".join([INIT] + [METHODS[m] for m in methods])


def build(sol, source, name):
    ns = {"ResumableIterator": sol.ResumableIterator}
    exec(source, ns)
    return ns[name]


@case("Part 1", sample=True, title="实现了四个方法的子类可以使用")
def p1_example(sol):
    need(sol, "ResumableIterator")
    source = subclass("Countdown", ABSTRACT)
    call = calling(source + "\nc = Countdown(3)\nprint(list(c))   # [3, 2, 1]")
    cls = build(sol, source, "Countdown")
    try:
        c = cls(3)
    except TypeError as e:
        raise Mismatch(f"实现了 __iter__、__next__、get_state、set_state 的子类应当可以实例化，实际抛出 TypeError: {e}",
                       call, show=False) from None
    expect(list(c), [3, 2, 1], call=call, msg="子类自己实现的迭代应当照常工作")
    expect_true(isinstance(c, sol.ResumableIterator), "子类的实例应当是 ResumableIterator 的实例", call=call)


@case("Part 1", title="四个方法都是抽象方法")
def p1_abstract(sol):
    need(sol, "ResumableIterator")
    call = calling("ResumableIterator()   # 应当抛出 TypeError")
    try:
        sol.ResumableIterator()
    except TypeError:
        pass
    else:
        raise Mismatch("ResumableIterator() 应当抛出 TypeError：它是抽象基类，不能直接实例化", call, show=False)
    for missing in ABSTRACT:
        rest = [m for m in ABSTRACT if m != missing]
        name = "Missing" + "".join(w.capitalize() for w in missing.strip("_").split("_"))
        source = subclass(name, rest)
        call = calling(source + f"\n{name}()   # 应当抛出 TypeError")
        cls = build(sol, source, name)
        try:
            cls()
        except TypeError:
            continue
        raise Mismatch(f"只实现了 {'、'.join(rest)} 的子类应当不能实例化（抛出 TypeError），因为 {missing} 是抽象方法",
                       call, show=False)


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    t = Session(sol).new("list", [68, 71, 74, 70, 73, 69])
    t.n(68).n(71).n(74).s((3,)).n(70).set((3,), want=None).n(70).n(73).n(69).n(STOP)


@case("Part 2", title="空列表、各种元素、list(it)")
def p2_basics(sol):
    need(sol, "ResumableIterator", "ResumableListIterator")
    expect_true(issubclass(sol.ResumableListIterator, sol.ResumableIterator),
                "ResumableListIterator 应当继承 ResumableIterator（Part 1 要求每个迭代器都继承它）")
    Session(sol, "空列表：一开始就耗尽，状态是 (0,)，StopIteration 可以重复抛出").new("list", []).s().n().n().s()
    Session(sol, "只有一个元素").new("list", ["only"]).s().n().s().n().s()
    t = Session(sol, "元素可以是任何值，包括 None、列表和重复值；list(it) 从当前位置取完剩下的元素")
    t.new("list", [None, [1, 2], "x", 0, 0]).n().n().s().rest().s().n()


@case("Part 2", title="列表形式的状态（JSON 往返）")
def p2_json_list(sol):
    t = Session(sol, "列表和元组一样被接受，因为状态经过 JSON 往返之后会变成列表")
    t.new("list", [10, 20, 30, 40]).n().n().s().n()
    t.set(json.loads(json.dumps((2,))), note="（经过 JSON 往返）").n().set([0]).n().set([4]).n().s()


@case("Part 2", title="TypeError：类型、长度、bool")
def p2_type_errors(sol):
    t = Session(sol, "状态不是元组或列表、长度不是 1、或元素不是 int 时抛出 TypeError").new("list", [5, 6, 7]).n()
    for state in (1, None, "1", (), (1, 0), [1, 1], (1.0,), ("1",), (None,), ([1],)):
        t.bad(state)
    for state in ((True,), (False,), [True]):
        t.bad(state, why="bool 不算 int，要抛出 TypeError")
    for state in ((9, 9), (-1.5,), [7.0]):
        t.bad(state, why="TypeError 先于 ValueError 检查：类型或长度不对时，不论数值是否越界都抛出 TypeError")
    t.n().s()


@case("Part 2", title="ValueError：越界和负数")
def p2_value_errors(sol):
    t = Session(sol, "位置必须满足 0 <= i <= len(items)，否则抛出 ValueError").new("list", [5, 6, 7]).n()
    t.bad((-1,), why="下标不能是负数").bad([-1], why="下标不能是负数").bad((-3,), why="下标不能是负数")
    t.bad((4,)).bad((100,))
    t.set((3,), why="(len(items),) 是耗尽元组，是规范的").n().s()
    Session(sol, "空列表只接受耗尽元组 (0,)").new("list", []).bad((1,)).bad((-1,)).set((0,)).n().s()


@case("Part 2", title="向前、向后跳，耗尽之后也能跳")
def p2_jumps(sol):
    t = Session(sol, "set_state 可以向前或向后跳，在抛出过 StopIteration 之后也可以").new("list", [1, 2, 3, 4, 5])
    t.n().n().set((4,)).n().n().n().set((1,)).n().set((0,)).rest().s().set((3,)).s().set((3,)).n().s()


@case("Part 2", title="把状态交给另一个实例")
def p2_other_instance(sol):
    t = Session(sol, "状态只和接收它的迭代器自己的数据比对：在新数据上规范就接受，并从它指向的元素继续")
    t.new("list", [1, 2, 3, 4], var="a").n().n().s()
    t.new("list", [10, 20, 30, 40], var="b").set((2,)).n().s()
    t.new("list", [7, 8], var="c").set((2,)).n().s()
    t.new("list", [9], var="d").bad((2,))
    t.new("list", [4, 5, 6], var="e").set((2,)).n().n()


@case("Part 2", title="随机操作序列")
def p2_random(sol):
    r = rng("p2")
    for _ in range(300):
        random_session(sol, r, "list", ref.gen(r, 1, max_len=6))


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    t = Session(sol).new("2d", [[7, 2], [], [9], [], [3, 5, 1]])
    t.n(7).n(2).s((2, 0)).n(9).set((2, 0), want=None).n(9).n(3).n(5).n(1).n(STOP).s((5, 0))


@case("Part 3", sample=True, title="题目例子 [[], []]")
def p3_example_empty_rows(sol):
    Session(sol).new("2d", [[], []]).s((2, 0)).n(STOP).s((2, 0))


@case("Part 3", sample=True, title="题目里规范元组的例子")
def p3_example_canonical(sol):
    t = Session(sol).new("2d", [[5, 7], [2]])
    t.n(5).n(7).s((1, 0)).set((0, 2), want="ValueError").s((1, 0))
    Session(sol).new("2d", [[], [4]]).set((0, 0), want="ValueError")


@case("Part 3", title="空行在开头、中间、结尾")
def p3_empty_rows(sol):
    need(sol, "ResumableIterator", "Resumable2DIterator")
    expect_true(issubclass(sol.Resumable2DIterator, sol.ResumableIterator),
                "Resumable2DIterator 应当继承 ResumableIterator（Part 1 要求每个迭代器都继承它）")
    Session(sol, "没有行时耗尽元组是 (0, 0)").new("2d", []).s().n().s()
    Session(sol, "开头的空行在构造时就已经跳过").new("2d", [[], [], [4]]).s().n().s().n()
    Session(sol, "产出一行的最后一个元素之后，位置立刻越过后面的空行").new("2d", [[1], [], []]).n().s().n()
    t = Session(sol, "产出一行的最后一个元素之后，位置立刻越过后面的空行")
    t.new("2d", [[1], [], [], [2], [], [3, 4], []]).n().s().n().s().n().s().n().s().n()
    t = Session(sol, "连续很多个空行")
    t.new("2d", [[]] * 30 + [[1, 2]] + [[]] * 30).s().n().s().n().s().n()


@case("Part 3", title="不规范的元组抛出 ValueError")
def p3_canonical(sol):
    t = Session(sol, "两层下标都必须满足 0 <= i < len，或者正好是耗尽元组 (len(items), 0)").new("2d", [[5, 7], [2]]).n()
    t.bad((0, 2), why="(0, 2) 指向非空行的末尾，不是规范的").bad((1, 1), why="(1, 1) 指向非空行的末尾，不是规范的")
    t.bad((2, 1), why="耗尽元组只有 (len(items), 0)").bad((3, 0))
    for state in ((-1, 0), (0, -1), (1, -1), (-2, -1)):
        t.bad(state, why="下标不能是负数")
    t.set((2, 0)).n().s().set((1, 0)).n().set((0, 0)).n()
    t = Session(sol, "指向空行内部的元组不是规范的").new("2d", [[], [4], []])
    t.bad((0, 0)).bad((2, 0)).set((3, 0)).n().set((1, 0)).n().s()
    Session(sol, "没有行时只接受 (0, 0)").new("2d", []).bad((0, 1)).bad((-1, 0)).set((0, 0)).n()


@case("Part 3", title="TypeError：类型、长度、bool")
def p3_type_errors(sol):
    t = Session(sol, "状态不是元组或列表、长度不是 2、或元素不是 int 时抛出 TypeError").new("2d", [[5, 7], [2]]).n()
    for state in ((0,), (0, 0, 0), [1], (0.0, 1), (0, "1"), ([0], 0), None, "01", 5):
        t.bad(state)
    for state in ((0, True), (True, 0), [False, False]):
        t.bad(state, why="bool 不算 int，要抛出 TypeError")
    for state in ((9, 9, 9), (-1, 2.5)):
        t.bad(state, why="TypeError 先于 ValueError 检查")
    t.set([1, 0], note="（列表）").n().s()


@case("Part 3", title="跳转，把状态交给另一份数据")
def p3_jumps_transfer(sol):
    t = Session(sol, "set_state 可以向前或向后跳，在抛出过 StopIteration 之后也可以")
    t.new("2d", [[1, 2], [], [3]], var="a").n().n().s().n().n().set((0, 1)).n().set((2, 0)).n().set((0, 0)).rest()
    t = Session(sol, "状态可以交给形状相同的数据，在形状不同的数据上规范就接受")
    t.new("2d", [[1, 2], [], [3]], var="a").n().n().s()
    t.new("2d", [[10, 20], [], [30]], var="b").set((2, 0)).n().s()
    t.new("2d", [[1], [2, 3]], var="c").set((2, 0)).n().s()
    t.new("2d", [[1, 2, 3]], var="d").bad((2, 0))
    t.new("2d", [[0], [0], [9, 8]], var="e").set((2, 0)).n().n().n()


@case("Part 3", title="随机操作序列")
def p3_random(sol):
    r = rng("p3")
    for _ in range(300):
        random_session(sol, r, "2d", ref.gen(r, 2, max_len=4))


# ---------------------------------------------------------------- Part 4

DATA3 = [[[5, 3], []], [], [[], [9, 1]], [[6]]]


@case("Part 4", sample=True, title="题目例子 depth=3")
def p4_example(sol):
    t = Session(sol).new("nested", DATA3, 3)
    t.n(5).n(3).s((2, 1, 0)).n(9).set((2, 1, 0), want=None).n(9).n(1).n(6).n(STOP)


@case("Part 4", title="耗尽元组 (len(items), 0, …, 0)")
def p4_exhausted(sol):
    need(sol, "ResumableIterator", "ResumableNestedIterator")
    expect_true(issubclass(sol.ResumableNestedIterator, sol.ResumableIterator),
                "ResumableNestedIterator 应当继承 ResumableIterator（Part 1 要求每个迭代器都继承它）")
    why = "耗尽之后状态是 (len(items), 0, …, 0)，长度等于 depth"
    Session(sol, why).new("nested", DATA3, 3).rest().s().n().s()
    Session(sol, why).new("nested", [], 3).s().n().s()
    Session(sol, why).new("nested", [[], [[]]], 3).s().n().s()
    Session(sol, why).new("nested", [[[[1]]]], 4).s().n().s()
    Session(sol, why).new("nested", [[], [[[[7]]], []]], 5).s().n().s().n().s()


@case("Part 4", title="元素本身是列表")
def p4_list_items(sol):
    why = "取 depth 次下标得到的就是元素，即使它本身是列表（空列表也是元素）"
    Session(sol, why).new("nested", [[], [8], []], 1).n().s().n().n().s().n()
    Session(sol, why).new("nested", [[[1], []], [], [[2, 3]]], 2).n().s().n().s().n().s()
    Session(sol, why).new("nested", [[[[]]], [[[], [4]]]], 3).s().n().s().n().n().s()


@case("Part 4", title="每一层都检查规范性")
def p4_canonical(sol):
    t = Session(sol, "每一层的下标都必须满足 0 <= i < len，或者正好是耗尽元组").new("nested", DATA3, 3).n().n()
    for state in ((0, 1, 0), (1, 0, 0), (2, 0, 0)):
        t.bad(state, why=f"{state} 指向空列表内部，不是规范的")
    for state in ((0, 0, 2), (0, 2, 0), (3, 1, 0)):
        t.bad(state, why=f"{state} 在某一层指向非空列表的末尾，不是规范的")
    for state in ((4, 0, 1), (4, 1, 0), (5, 0, 0)):
        t.bad(state, why="耗尽元组只有 (len(items), 0, …, 0)")
    for state in ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (3, 0, -1), (-4, 0, 0)):
        t.bad(state, why="下标不能是负数")
    t.set((2, 1, 1)).n().set((3, 0, 0)).n().set((4, 0, 0)).n().set((0, 0, 0)).n()
    t.set([2, 1, 0], note="（列表）").n().s()


@case("Part 4", title="TypeError：类型、长度、bool")
def p4_type_errors(sol):
    t = Session(sol, "状态不是元组或列表、长度不是 depth、或元素不是 int 时抛出 TypeError").new("nested", DATA3, 3)
    t.n()
    for state in ((2, 1), (2, 1, 0, 0), (2.0, 1, 0), (2, 1, "0"), ((2,), 1, 0), "210", None):
        t.bad(state)
    for state in ((2, 1, True), (False, 0, 0), [0, 0, False]):
        t.bad(state, why="bool 不算 int，要抛出 TypeError")
    for state in ((9, 9), (9, 9, 9, 9), (-1, 0, 0.5)):
        t.bad(state, why="TypeError 先于 ValueError 检查")
    Session(sol, "depth=1 时状态长度是 1").new("nested", [1, 2], 1).bad((0, 0)).bad(()).set((1,)).n()


@case("Part 4", title="depth=1、2 与 Part 2、3 一致")
def p4_depth_1_2(sol):
    t = Session(sol, "depth=1 时与 Part 2 的迭代器完全相同").new("nested", [68, 71, 74, 70, 73, 69], 1)
    t.n().n().n().s().n().set((3,)).n().n().n().n().n().s()
    t = Session(sol, "depth=2 时与 Part 3 的迭代器完全相同").new("nested", [[7, 2], [], [9], [], [3, 5, 1]], 2)
    t.n().n().s().n().set((2, 0)).n().n().n().n().n().s()
    Session(sol, "depth=2 时与 Part 3 的迭代器完全相同").new("nested", [[], []], 2).s().n()
    t = Session(sol, "depth=2 时与 Part 3 的迭代器完全相同").new("nested", [[5, 7], [2]], 2)
    t.n().n().s().bad((0, 2)).bad((1, 1)).set((2, 0)).n()
    r = rng("p4-12")
    for _ in range(100):
        random_session(sol, r, "nested", ref.gen(r, 1, max_len=5), 1)
        random_session(sol, r, "nested", ref.gen(r, 2, max_len=4), 2)


@case("Part 4", title="跳转，把状态交给另一份数据")
def p4_jumps_transfer(sol):
    t = Session(sol, "set_state 可以向前或向后跳，在抛出过 StopIteration 之后也可以").new("nested", DATA3, 3)
    t.rest().n().set((2, 1, 1)).n().set((0, 0, 1)).n().s().n().set((4, 0, 0)).n().set((0, 0, 0)).rest()
    t = Session(sol, "状态可以交给形状相同的数据，在形状不同的数据上规范就接受").new("nested", DATA3, 3, var="a")
    t.n().n().s()
    t.new("nested", [[[50, 30], []], [], [[], [90, 10]], [[60]]], 3, var="b").set((2, 1, 0)).n().n().s()
    t.new("nested", [[[1]], [[2]], [[3], [4, 5]]], 3, var="c").set((2, 1, 0)).n().n().s()
    t.new("nested", [[[1]], [[2]], [[3]]], 3, var="d").bad((2, 1, 0))
    t.new("nested", [[[1]], [], [[2]]], 3, var="e").set((3, 0, 0)).n().s()


@case("Part 4", title="更深的嵌套、大量空列表")
def p4_deep(sol):
    why = "很多层空列表连在一起时，位置要一直移到下一个元素上"
    deep_empty = [[[[[]]]], [[[], []]], [], [[[[], [], [1]]]], [[[]]]]
    Session(sol, why).new("nested", deep_empty, 5).s().n().s().n().s()
    wide = [[[]] * 20, [], [[], [], [7, 8]], [[]] * 20, [[9]]]
    Session(sol, why).new("nested", wide, 3).s().n().s().n().s().n().s().n().s()
    r = rng("p4-deep")
    for _ in range(60):
        random_session(sol, r, "nested", ref.gen(r, 6, max_len=2, p_empty=0.2), 6, ops=8)


@case("Part 4", title="随机操作序列")
def p4_random(sol):
    r = rng("p4")
    for _ in range(400):
        d = r.randint(1, 5)
        random_session(sol, r, "nested", ref.gen(r, d, max_len=3, list_leaves=r.random() < 0.5), d)
