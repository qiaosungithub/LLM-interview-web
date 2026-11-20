from judgelib import Mismatch, calling, case, expect, rng
import ref


def _node(sol, t):
    return ref.build(sol.Node, t)


def _fn_code(params, ret):
    return f"Function([{', '.join(ref.code(p) for p in params)}], {ref.code(ret)})"


def _nest_code(var, name, depth):
    """Code that builds ref.nest(name, depth) into the variable var."""
    return f'{var} = Node("{name}")\nfor _ in range({depth}):\n    {var} = Node([{var}])'


def _shown(x):
    try:
        return str(x)
    except Exception as e:
        return f"<str() 出错：{type(e).__name__}>"


def _str(x, call):
    """str(x); a __str__ that returns something other than a string is a wrong answer, shown with the call."""
    try:
        return str(x)
    except TypeError as e:
        if "returned non-string" not in str(e):
            raise
        raise Mismatch(f"__str__ 应当返回字符串（{e}）", call, show=False) from None


# ================================================================ Part 1

def check_str(sol, t, code=None):
    code = code or f"print({ref.code(t)})"
    calling(code)
    want = ref.show(t)
    got = _str(_node(sol, t), code)
    expect(got, want, call=code)


def check_fn(sol, params, ret, code=None):
    code = code or f"print({_fn_code(params, ret)})"
    calling(code)
    want = ref.show_sig(params, ret)
    got = _str(sol.Function([_node(sol, p) for p in params], _node(sol, ret)), code)
    expect(got, want, call=code)


def check_eq(sol, a, b, code=None):
    code = code or f"print({ref.code(a)} == {ref.code(b)})"
    calling(code)
    sa, sb = ref.show(a), ref.show(b)
    want = sa == sb
    if want:
        msg = f"两个节点的字符串形式都是 {sa}，应当相等"
    else:
        msg = f"两个节点的字符串形式分别是 {sa} 和 {sb}，不应相等"
    got = _node(sol, a) == _node(sol, b)
    expect(got, want, call=code, msg=msg)


@case("Part 1", sample=True, title="题目示例")
def p1_example(sol):
    check_fn(sol, *ref.signature("([T1,bool],T2,char) -> [T2,[T1,int]]"))


@case("Part 1", title="原子类型")
def p1_atoms(sol):
    for name in ["int", "float", "str", "bool", "char", "T1", "T12", "K", "value"]:
        check_str(sol, name)
    check_fn(sol, *ref.signature("(int) -> bool"))
    check_fn(sol, *ref.signature("(T1,T2) -> T1"))


@case("Part 1", title="元组与嵌套")
def p1_tuples(sol):
    for s in ["[int]", "[T1,bool]", "[[int,[char]],str]", "[int,[float,[bool,[char,[str]]]]]", "[T1,T1,T1,T1]"]:
        check_str(sol, ref.parse(s))
    for s in ["([int,float]) -> [bool]", "(T1,[T2,[T3]],char) -> [[T3],T2,T1]"]:
        check_fn(sol, *ref.signature(s))


@case("Part 1", title="空元组与零参数")
def p1_empty(sol):
    for s in ["[]", "[[]]", "[[],[]]", "[int,[]]", "[[[]],T1]"]:
        check_str(sol, ref.parse(s))
    for s in ["() -> int", "() -> []", "([]) -> []", "([],[[]]) -> [int,[]]"]:
        check_fn(sol, *ref.signature(s))


@case("Part 1", title="结构相等")
def p1_eq(sol):
    pairs = [("int", "int"), ("int", "float"), ("T1", "T1"), ("T1", "T2"), ("[int,bool]", "[int,bool]"),
             ("[int,bool]", "[bool,int]"), ("[int]", "int"), ("[]", "[]"), ("[]", "[[]]"), ("[[]]", "[[]]"),
             ("[int,[T1,char]]", "[int,[T1,char]]"), ("[int,[T1,char]]", "[int,[T1,str]]"),
             ("[int,float]", "[int,float,bool]"), ("[[int],float]", "[[int,float]]")]
    for a, b in pairs:
        check_eq(sol, ref.parse(a), ref.parse(b))


@case("Part 1", title="深层嵌套与长列表")
def p1_big(sol):
    deep = ref.nest("T1", 300)
    check_str(sol, deep, _nest_code("t", "T1", 300) + "\nprint(t)")
    for name, depth in [("T1", 300), ("T2", 300), ("T1", 299)]:
        code = _nest_code("t", "T1", 300) + "\n" + _nest_code("u", name, depth) + "\nprint(t == u)"
        check_eq(sol, deep, ref.nest(name, depth), code)
    wide = [ref.PRIMITIVES[i % 5] for i in range(2000)]
    check_str(sol, wide, f"print(Node([Node({list(ref.PRIMITIVES)}[i % 5]) for i in range(2000)]))")
    params = [f"T{i}" for i in range(1000)]
    check_fn(sol, params, [], 'print(Function([Node(f"T{i}") for i in range(1000)], Node([])))')


@case("Part 1", title="随机类型")
def p1_random(sol):
    r = rng("p1")
    atoms = list(ref.PRIMITIVES) + ["T1", "T2", "T3", "K"]
    for _ in range(150):
        check_str(sol, ref.random_type(r, atoms, 4))
        a = ref.random_type(r, atoms, 3)
        b = a if r.random() < 0.4 else ref.random_type(r, atoms, 3)    # equal pairs are still built separately
        check_eq(sol, a, b)
    for _ in range(50):
        params = [ref.random_type(r, atoms, 3) for _ in range(r.randint(0, 4))]
        check_fn(sol, params, ref.random_type(r, atoms, 3))


# ================================================================ Part 2

ERRORS = ref.ERRORS


def _error_classes(sol):
    """The answer's error classes, by name (those it defines)."""
    out = {}
    for name in ERRORS:
        c = getattr(sol, name, None)
        if isinstance(c, type) and issubclass(c, BaseException):
            out[name] = c
    return out


class Calls:
    """The calls of one test, kept as code the user can paste into 自定义运行."""

    def __init__(self, sol):
        self.sol = sol
        self.lines = []
        self.fns = {}

    def function(self, sig, name="f", code=None):
        """sig: a signature in the statement's notation, or (params, ret) as ref values."""
        params, ret = ref.signature(sig) if isinstance(sig, str) else sig
        self.lines.append(code or f"{name} = {_fn_code(params, ret)}  # {ref.show_sig(params, ret)}")
        calling("\n".join(self.lines))
        f = self.sol.Function([_node(self.sol, p) for p in params], _node(self.sol, ret))
        self.fns[name] = (params, ret, f)

    def call(self, args, name="f", want=None, code=None):
        """get_return_type(args, name) must match ref. args: types in the statement's notation or as ref values.
        want (samples only) is the statement's answer, which ref must agree with."""
        sol = self.sol
        args = [ref.parse(a) if isinstance(a, str) else a for a in args]
        params, ret, f = self.fns[name]
        status, result = ref.analyze(params, ret, args)
        assert status == "ok" or len(result) == 1, (params, ret, args, result)    # only calls with one right answer
        expected = ref.show(result) if status == "ok" else next(iter(result))
        assert want is None or want == expected, (want, expected)
        expr = code or f"get_return_type([{', '.join(ref.code(a) for a in args)}], {name})"
        note = "实参：" + (", ".join(ref.show(a) for a in args) or "无") if code is None else ""
        text = "\n".join(self.lines + [f"print({expr})  # <- 这一步" + (f"，{note}" if note else "")])
        calling(text)
        built = [_node(sol, a) for a in args]
        errors = _error_classes(sol)
        if status == "ok":
            try:
                got = sol.get_return_type(built, f)
            except tuple(errors.values()) as e:
                raise Mismatch(f"这次调用没有错误，不应抛出 {type(e).__name__}: {e}", text, expected,
                               f"抛出 {type(e).__name__}") from None
            if not isinstance(got, sol.Node):
                raise Mismatch("get_return_type 应当返回一个 Node", text, expected, got)
            expect(_shown(got), expected, call=text)
            self.lines.append(f"print({expr})  # -> {expected}")
        else:
            self._raises(expected, built, f, text, errors)
            self.lines += [f"try: {expr}", f"except {expected}: pass  # 抛出 {expected}"]

    def _raises(self, kind, built, f, text, errors):
        cls = errors.get(kind)
        if cls is None:
            raise Mismatch(f"答案里没有定义异常类 {kind}（三类错误各自抛出同名的异常类）", text, show=False)
        try:
            got = self.sol.get_return_type(built, f)
        except cls as e:
            for name, other in errors.items():
                if name != kind and isinstance(e, other) and not issubclass(cls, other):
                    raise Mismatch(f"应当抛出 {kind}，实际抛出的 {type(e).__name__} 同时也是 {name}", text,
                                   show=False) from None
            return
        except Exception as e:
            raise Mismatch(f"应当抛出 {kind}，实际抛出 {type(e).__name__}: {e}", text, show=False) from None
        raise Mismatch(f"应当抛出 {kind}，但正常返回了", text, f"抛出 {kind}", _shown(got))


def run_all(sol, calls):
    """calls: (signature, args) pairs, each on a new function; a third item is the statement's answer. The replay
    keeps the earlier calls, so state that leaks from one call into the next can be reproduced."""
    c = Calls(sol)
    for sig, args, *want in calls:
        c.function(sig)
        c.call(args, want=want[0] if want else None)


@case("Part 2", sample=True, title="题目示例：推出返回类型")
def p2_example(sol):
    run_all(sol, [("(T1,bool,T2) -> [T2,T1]", ["char", "bool", "float"], "[float,char]"),
                  ("(T1,[T1,bool],T2) -> [T2,[T1,T1]]", ["char", "[char,bool]", "int"], "[int,[char,char]]"),
                  ("(T1,bool) -> [T1,T1]", ["[int,float]", "bool"], "[[int,float],[int,float]]")])


@case("Part 2", sample=True, title="题目示例：三类错误")
def p2_example_errors(sol):
    run_all(sol, [("(T1,int,bool) -> T1", ["char", "int"], "ArityError"),
                  ("([T1,int],bool) -> T1", ["[char,int,bool]", "bool"], "TypeMismatchError"),
                  ("(int,T3,T3) -> T3", ["int", "bool", "char"], "GenericConflictError")])


@case("Part 2", title="不是 T 加数字的泛型名")
def p2_generic_names(sol):
    run_all(sol, [("(K,[value,bool]) -> [value,K]", ["int", "[float,bool]"]),
                  ("(Int,int) -> [Int,Int]", ["[char]", "int"]),
                  ("(Tfoo,T) -> [T,Tfoo]", ["str", "char"]),
                  ("(T01,T1) -> [T1,T01]", ["int", "bool"]),
                  ("(x,y,x) -> y", ["int", "float", "int"]),
                  ("(Float,integer) -> [integer,Float]", ["float", "[int]"]),
                  ("(K,K) -> K", ["int", "float"])])


@case("Part 2", title="泛型绑定到元组")
def p2_generic_tuple(sol):
    run_all(sol, [("(T1) -> T1", ["[]"]),
                  ("(T1,T1) -> [T1]", ["[]", "[]"]),
                  ("(T1,T1) -> T1", ["[[int],[]]", "[[int],[]]"]),
                  ("([T1,bool],T2) -> [T2,T1]", ["[[char,[int]],bool]", "[]"]),
                  ("(T1,[T1]) -> [T1,[T1]]", ["[int,str]", "[[int,str]]"]),
                  ("(T1) -> [T1,[T1,T1]]", ["[[]]"])])


@case("Part 2", title="泛型冲突")
def p2_conflict(sol):
    run_all(sol, [("(T1,T1) -> T1", ["[]", "[int]"]),
                  ("(T1,T1) -> T1", ["int", "[int]"]),
                  ("(T1,T1) -> T1", ["[int,float]", "[float,int]"]),
                  ("(T1,T1) -> T1", ["[[int]]", "[[float]]"]),
                  ("([T1,[T2,T1]]) -> T2", ["[int,[bool,float]]"]),
                  ("(T1,[bool,[T1]]) -> T1", ["char", "[bool,[str]]"]),
                  ("(T2,T1,T2) -> [T1,T2]", ["[int]", "bool", "[int]"]),
                  ("(T1,[T1,T1]) -> T1", ["[char,[int]]", "[[char,[int]],[char,[int]]]"])])


@case("Part 2", title="类型不匹配")
def p2_mismatch(sol):
    run_all(sol, [("(int) -> int", ["float"]),
                  ("(char,str) -> int", ["char", "char"]),
                  ("(int) -> int", ["[int]"]),
                  ("([T1,int]) -> T1", ["int"]),
                  ("([T1,int]) -> T1", ["[char]"]),
                  ("([T1]) -> T1", ["[]"]),
                  ("([[T1,bool]]) -> T1", ["[[int]]"]),
                  ("(T1,[bool,char]) -> T1", ["int", "[bool,str]"]),
                  ("(T1,[T1,int]) -> T1", ["float", "[float,[int]]"])])


@case("Part 2", title="空元组")
def p2_empty_tuple(sol):
    run_all(sol, [("([]) -> int", ["[]"]),
                  ("([],T1) -> [T1,[]]", ["[]", "[]"]),
                  ("([[],T1]) -> T1", ["[[],bool]"]),
                  ("([]) -> int", ["[int]"]),
                  ("([]) -> int", ["int"]),
                  ("([],[]) -> int", ["[]", "[char]"]),
                  ("([[],T1]) -> T1", ["[[int],bool]"])])


@case("Part 2", title="零参数函数")
def p2_no_params(sol):
    run_all(sol, [("() -> int", []),
                  ("() -> [[],char]", []),
                  ("() -> int", ["int"]),
                  ("(T1) -> T1", [])])


@case("Part 2", title="参数个数")
def p2_arity(sol):
    run_all(sol, [("(T1,T2) -> T1", ["int"]),
                  ("(T1,T2) -> T1", ["int", "int", "int"]),
                  ("(int) -> int", ["int", "int"]),
                  ("(T1,T1) -> T1", ["int", "int", "float"])])


@case("Part 2", title="同一个函数调用多次")
def p2_reuse(sol):
    c = Calls(sol)
    c.function("(T1,T2) -> [T2,T1]", "f")
    c.call(["int", "bool"], "f")
    c.call(["[char]", "float"], "f")
    c.call(["int"], "f")
    c.call(["str", "str"], "f")
    c.function("(T1,T1) -> T1", "g")
    c.call(["int", "float"], "g")
    c.call(["bool", "bool"], "g")
    c.call(["[int]", "[int]"], "g")
    c.call(["[]", "char"], "f")


@case("Part 2", title="深层嵌套与长参数表")
def p2_big(sol):
    # ([[[...T1...]]]) -> [T1,T1], called with [[[...int...]]]
    c = Calls(sol)
    c.function(([ref.nest("T1", 300)], ["T1", "T1"]), code=_nest_code("d", "T1", 300)
               + '\nf = Function([d], Node([Node("T1"), Node("T1")]))  # ([[[…T1…]]]) -> [T1,T1]，300 层')
    c.lines.append(_nest_code("a", "int", 300))
    c.call([ref.nest("int", 300)], code="get_return_type([a], f)")
    c.lines.append(_nest_code("a", "int", 299))
    c.call([ref.nest("int", 299)], code="get_return_type([a], f)")
    # (T1,[[[...T1...]]]) -> T1: the conflict is 300 levels down
    c = Calls(sol)
    c.function((["T1", ref.nest("T1", 300)], "T1"), code=_nest_code("d", "T1", 300)
               + '\nf = Function([Node("T1"), d], Node("T1"))  # (T1,[[[…T1…]]]) -> T1，300 层')
    c.lines.append(_nest_code("a", "float", 300))
    c.call(["int", ref.nest("float", 300)], code='get_return_type([Node("int"), a], f)')
    # 1000 parameters cycling through five generics and five primitives
    names = ["T1", "int", "T2", "float", "T3", "bool", "T4", "char", "T5", "str"]
    values = {"T1": "int", "T2": ["char"], "T3": [], "T4": ["bool", ["str"]], "T5": [["int"], "float"]}
    params = [names[i % 10] for i in range(1000)]
    args = [values.get(p, p) for p in params]
    ret = ["T5", "T4", "T3", "T2", "T1"]
    c = Calls(sol)
    c.function((params, ret), code=f"names = {names}\n"
               "f = Function([Node(names[i % 10]) for i in range(1000)], "
               "Node([Node(\"T5\"), Node(\"T4\"), Node(\"T3\"), Node(\"T2\"), Node(\"T1\")]))")
    c.lines.append("values = {" + ", ".join(f'"{k}": {ref.code(v)}' for k, v in values.items()) + "}")
    c.call(args, code="get_return_type([values.get(n, Node(n)) for n in [names[i % 10] for i in range(1000)]], f)")
    assert params[996] == "T4"
    args[996] = ["char", "float"]                                          # T4's last slot binds differently
    c.lines.append("args = [values.get(n, Node(n)) for n in [names[i % 10] for i in range(1000)]]")
    c.lines.append('args[996] = Node([Node("char"), Node("float")])')
    c.call(args, code="get_return_type(args, f)")


@case("Part 2", title="随机调用")
def p2_random(sol):
    r = rng("p2")
    done = 0
    while done < 300:
        params, ret, args, _ = ref.random_call(r)
        status, result = ref.analyze(params, ret, args)
        if status == "error" and len(result) > 1:
            continue
        c = Calls(sol)
        c.function((params, ret))
        c.call(args)
        done += 1
