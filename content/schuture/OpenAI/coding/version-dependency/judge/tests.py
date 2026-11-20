from judgelib import calling, case, expect, expect_true, expect_valid, info, rng, show_call
import ref

P1, P2, P3 = "find_earliest_supported_monotone", "find_earliest_supported", "find_earliest_supported_hierarchical"


# ---------------------------------------------------------------- how a call is shown

def support_call(fn, support, counting=False):
    """Code that rebuilds is_supported as a dict lookup and makes the call; the dict's order is the order of the
    versions list the answer received."""
    table = repr(support)
    if len(table) > 1500:
        M, N, P = ref.shape(support)
        return (f"{fn}(<{len(support)} 个版本：{M} 个 major，每个 major 至多 {N} 个 minor，"
                f"每组至多 {P} 个 patch>, is_supported)")
    if counting:
        return (f"support = {table}\nprobe = CountingProbe(lambda v: support[v])\n"
                f"print({fn}(list(support), probe), probe.calls)")
    return f"support = {table}\n{fn}(list(support), lambda v: support[v])"


def run_probe(sol, fn, support, call):
    probe = ref.Probe(support)
    calling(call)
    got = getattr(sol, fn)(list(support), probe)
    unknown = probe.unknown()
    expect_true(not unknown, f"调用 is_supported 时传入了不在 versions 里的字符串 {unknown[0]!r}"
                             f"（要传 versions 里的原字符串）" if unknown else "", call)
    return got, probe


def check_result(got, want, support, call):
    msg = ""
    if got != want:
        if want is None:
            msg = "没有任何版本支持该特性，应当返回 None"
        elif got is None:
            msg = f"有版本支持该特性（最早的是 {want!r}），却返回了 None"
        elif isinstance(got, str) and got in support:
            if not support[got]:
                msg = f"{got!r} 并不支持该特性"
            else:
                msg = f"{got!r} 支持该特性，但数值上更早的 {want!r} 也支持"
        elif isinstance(got, str):
            try:
                if ref.parse(got) == ref.parse(want):
                    msg = f"数值上是同一个版本，但要返回 versions 里的原字符串 {want!r}"
            except Exception:
                pass
    expect(got, want, call=call, msg=msg)


def check_scan(sol, fn, support, call=None):
    """Parts 1 and 2: the earliest supported version, by a plain scan."""
    call = call or support_call(fn, support)
    want = ref.earliest(support)
    got, probe = run_probe(sol, fn, support, call)
    check_result(got, want, support, call)
    return probe


def check_p3(sol, support, call=None):
    call = call or support_call(P3, support, counting=True)
    want = ref.earliest(support)
    budget, (M, N, P) = ref.call_budget(support)
    got, probe = run_probe(sol, P3, support, call)
    check_result(got, want, support, call)
    again = probe.repeated()
    expect_true(not again, f"同一个版本 {again[0]!r} 调用了不止一次 is_supported（题目要求同一个版本字符串"
                           f"不能探测两次）" if again else "", call, got=probe.calls)
    n = len(probe.calls)
    expect_true(n <= budget, f"调用了 {n} 次 is_supported，超过对数级的上限 {budget} 次（M = {M}，N_minor = {N}，"
                             f"P = {P}；上限按 2 × (⌊log₂M⌋ + ⌊log₂N_minor⌋ + ⌊log₂P⌋ + 3) + 2 计）", call,
                got=probe.calls)
    return probe


def check_resolve(sol, reg, root, call=None):
    call = call or ref.registry_code(reg, root)
    one = ref.brute_choice(reg, root)
    calling(call)
    got = sol.resolve(reg, root)
    if one is None:
        expect(got, None, call=call, msg="" if got is None else "不存在合法的选择（约束无法同时满足，或者都会成环），应当返回 None")
        return
    expect_true(got is not None, f"存在合法的选择（例如 {one}），却返回了 None", call)
    expect_valid(got, lambda g: ref.check_order(reg, root, g), call=call, want=one)


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子：按数值排序")
def p1_example(sol):
    support = {"2.0.0": True, "1.10.0": True, "1.9.1": True, "1.9.0": False}
    call = calling(f"support = {support!r}\n{P1}([\"2.0.0\", \"1.10.0\", \"1.9.1\", \"1.9.0\"], lambda v: support[v])")
    got, _ = run_probe(sol, P1, support, call)
    check_result(got, "1.9.1", support, call)


@case("Part 1", sample=True, title="题目例子：parse_version 去掉前导零")
def p1_parse_example(sol):
    call = calling('parse_version("103.003.02")')
    got = sol.parse_version("103.003.02")
    if isinstance(got, list):
        got = tuple(got)                    # the statement says tuple; a list carries the same parts
    expect(got, (103, 3, 2), call=call)


@case("Part 1", title="parse_version：多位数与前导零")
def p1_parse_more(sol):
    for s, want in [("0.0.0", (0, 0, 0)), ("1.10.0", (1, 10, 0)), ("007.0.010", (7, 0, 10)), ("12.345.6789", (12, 345, 6789))]:
        call = calling(show_call("parse_version", s))
        got = sol.parse_version(s)
        expect(tuple(got) if isinstance(got, list) else got, want, call=call)


@case("Part 1", title="没有版本支持：返回 None")
def p1_none_supported(sol):
    for support in [{"1.0.0": False}, {"1.2.0": False, "0.9.9": False, "1.10.0": False},
                    {f"2.{k}.0": False for k in (7, 3, 11, 0, 5, 9, 1)}]:
        check_scan(sol, P1, support)


@case("Part 1", title="空列表：返回 None")
def p1_empty(sol):
    check_scan(sol, P1, {})


@case("Part 1", title="全部支持：返回最小的版本")
def p1_all_supported(sol):
    for support in [{"3.0.0": True}, {"1.10.0": True, "1.9.0": True, "1.2.0": True},
                    {f"0.{k}.1": True for k in (12, 3, 40, 7, 25, 1, 9, 30)}]:
        check_scan(sol, P1, support)


@case("Part 1", title="多位数和前导零按数值比较")
def p1_numeric_order(sol):
    check_scan(sol, P1, {"10.0.0": True, "9.0.0": True, "2.0.0": False, "1.0.0": False})
    check_scan(sol, P1, {"1.02.0": False, "1.10.0": True, "1.9.0": False, "01.009.010": True})
    check_scan(sol, P1, {"0.0.10": True, "0.0.9": True, "0.0.2": False, "0.0.1": False})


@case("Part 1", title="支持段开始于任意位置（逐一检查）")
def p1_every_cut(sol):
    tuples = [(1, k, 0) for k in range(13)]
    for cut in range(len(tuples) + 1):
        r = rng(str(cut))
        pairs = [(ref.fmt(t), i >= cut) for i, t in enumerate(tuples)]
        r.shuffle(pairs)
        check_scan(sol, P1, dict(pairs))


@case("Part 1", title="随机测试（对照逐个扫描）")
def p1_random(sol):
    r = rng()
    for _ in range(300):
        check_scan(sol, P1, ref.monotone_support(r, ref.random_tuples(r, r.randint(1, 30))))


@case("Part 1", title="2000 个版本（只报告调用次数）")
def p1_large(sol):
    r = rng()
    support = ref.monotone_support(r, ref.random_tuples(r, 2000, top=(20, 30, 30)))
    probe = check_scan(sol, P1, support)
    info(f"调用 is_supported {len(probe.calls)} 次；二分查找最多 {len(support).bit_length()} 次。"
         f"Part 1 不评测调用次数（Part 3 才有要求）")


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子：支持出现回退")
def p2_example(sol):
    support = {"3.1.4": False, "3.1.0": False, "3.1.2": False, "3.1.1": True, "3.1.3": True}
    call = calling(f"support = {support!r}\n{P2}([\"3.1.4\", \"3.1.0\", \"3.1.2\", \"3.1.1\", \"3.1.3\"], lambda v: support[v])")
    got, _ = run_probe(sol, P2, support, call)
    check_result(got, "3.1.1", support, call)


@case("Part 2", title="二分会落到较晚的 True 上")
def p2_bisection_trap(sol):
    check_scan(sol, P2, {"1.0.0": False, "1.1.0": True, "1.2.0": False, "1.3.0": False, "1.4.0": False,
                         "1.5.0": True, "1.6.0": True})
    check_scan(sol, P2, {"2.0.7": True, "2.0.1": False, "2.0.3": True, "2.0.0": False, "2.0.2": False,
                         "2.0.5": True, "2.0.6": False, "2.0.4": False})


@case("Part 2", title="只有最早或最晚的版本支持")
def p2_ends(sol):
    vs = ["4.0.0", "0.4.0", "1.0.1", "0.10.2", "0.9.9", "3.3.3"]
    for hit in ("0.4.0", "4.0.0"):
        check_scan(sol, P2, {v: v == hit for v in vs})


@case("Part 2", title="没有版本支持、空列表")
def p2_none(sol):
    check_scan(sol, P2, {})
    check_scan(sol, P2, {"1.0.0": False, "1.10.0": False, "1.2.0": False})


@case("Part 2", title="多位数和前导零按数值比较")
def p2_numeric_order(sol):
    check_scan(sol, P2, {"1.10.0": True, "1.9.0": False, "1.2.0": True, "1.1.0": False})
    check_scan(sol, P2, {"01.010.0": True, "1.9.00": True, "1.09.1": False, "001.2.003": False})


@case("Part 2", title="随机测试（对照逐个扫描）")
def p2_random(sol):
    r = rng()
    for _ in range(300):
        check_scan(sol, P2, ref.any_support(r, ref.random_tuples(r, r.randint(1, 30)), p_true=r.choice([0.1, 0.3, 0.6])))


# ---------------------------------------------------------------- Part 3

EXAMPLE3 = {
    "0.9.0": False,
    "1.0.0": False, "1.1.0": False,
    "1.2.0": False, "1.2.1": True,
    "1.3.0": False, "1.3.1": False, "1.3.2": False, "1.3.3": True,
}


@case("Part 3", sample=True, title="题目例子：九个版本")
def p3_example(sol):
    check_p3(sol, dict(EXAMPLE3))


@case("Part 3", title="展平后直接二分会出错")
def p3_flat_bisection_trap(sol):
    check_p3(sol, {"1.2.0": False, "1.2.1": True, "1.3.0": False, "1.3.1": False, "1.3.2": False, "1.3.3": True})
    check_p3(sol, {"1.3.3": True, "1.2.1": True, "1.3.1": False, "1.2.0": False, "1.3.2": False, "1.3.0": False})


@case("Part 3", title="没有版本支持、空列表、只有一个版本")
def p3_small(sol):
    check_p3(sol, {})
    check_p3(sol, {"2.0.0": False})
    check_p3(sol, {"2.0.0": True})
    check_p3(sol, {k: False for k in EXAMPLE3})


@case("Part 3", title="代表元在下一层不能再探测一次")
def p3_no_repeat(sol):
    # support starts exactly at the last minor group and its last patch, so each level ends on the
    # representative the level above already probed
    check_p3(sol, {"1.0.0": False, "1.1.0": False, "1.1.1": False, "1.2.0": False, "1.2.1": False, "1.2.2": True})
    check_p3(sol, {"0.5.0": False, "2.0.0": False, "2.0.1": False, "2.1.0": False, "2.1.4": True})
    r = rng()
    for _ in range(40):
        tuples = ref.tree_tuples(r, r.randint(1, 4), 4, 4)
        n_major = len({t[0] for t in tuples})
        check_p3(sol, ref.a1_a3_support(r, tuples, cuts=(r.randint(0, n_major - 1), -1, -1)))


@case("Part 3", title="分组要按数值：多位数与前导零")
def p3_numeric_groups(sol):
    # "1.10.x" is the latest minor of major 1 although it sorts first as a string
    check_p3(sol, {"1.9.0": False, "1.9.5": True, "1.10.0": True, "1.10.2": True, "1.2.0": False, "1.2.9": False})
    # leading zeros: "01.03.0" and "1.3.1" are one (major, minor) group, "2.0.0" and "02.00.01" another
    check_p3(sol, {"01.03.0": False, "1.3.1": True, "1.03.2": True, "2.0.0": True, "02.00.01": True,
                   "0.4.4": False, "0.04.10": False})
    # patch 10 is the representative of its group, patch 9 is not
    check_p3(sol, {"3.1.9": False, "3.1.10": True, "3.0.10": False, "3.0.9": False, "4.0.0": True})


@case("Part 3", title="随机小输入：枚举所有 A1-A3 模式")
def p3_exhaustive(sol):
    shapes = [[(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
              [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 1), (2, 0, 0)],
              [(5, 0, 1), (5, 0, 3), (5, 2, 0), (10, 0, 0), (10, 1, 0), (10, 1, 7), (10, 11, 2)]]
    import itertools
    for tuples in shapes:
        for bits in itertools.product([False, True], repeat=len(tuples)):
            value = dict(zip(tuples, bits))
            if _a1_a3(value):
                check_p3(sol, {ref.fmt(t): value[t] for t in tuples})


def _a1_a3(value):
    tree = {}
    for t in sorted(value):
        tree.setdefault(t[0], {}).setdefault(t[1], []).append(t)
    mono = lambda seq: all(not (a and not b) for a, b in zip(seq, seq[1:]))
    for minors in tree.values():
        if not all(mono([value[t] for t in group]) for group in minors.values()):
            return False
        if not mono([value[minors[m][-1]] for m in sorted(minors)]):
            return False
    return mono([value[max(t for g in tree[M].values() for t in g)] for M in sorted(tree)])


@case("Part 3", title="随机测试（对照逐个扫描）")
def p3_random(sol):
    r = rng()
    for _ in range(200):
        tuples = ref.tree_tuples(r, r.randint(1, 5), r.randint(1, 5), r.randint(1, 6))
        check_p3(sol, ref.a1_a3_support(r, tuples))


# Big trees are built once at import time, which is not charged to any case.
_big = rng("big")
BIG_TUPLES = ref.tree_tuples(_big, 40, 40, 30, top=400)
_target = sorted({t[0] for t in BIG_TUPLES})[23]
_minors = sorted({t[1] for t in BIG_TUPLES if t[0] == _target})
TARGET_MINOR = len(_minors) - min(3, len(_minors))          # index of the group where support starts
BIG_TUPLES = [t for t in BIG_TUPLES if not (t[0] == _target and t[1] == _minors[TARGET_MINOR])]
BIG_TUPLES += [(_target, _minors[TARGET_MINOR], p) for p in range(0, 400, 2)]       # one group of 200 patches


@case("Part 3", title="约 2 万个版本：调用次数要是对数级")
def p3_large(sol):
    r = rng()
    for cut_patch in (97, 103, 150):
        support = ref.a1_a3_support(r, BIG_TUPLES, zeros=0.05, cuts=(23, TARGET_MINOR, cut_patch))
        probe = check_p3(sol, support)
        info(f"首个支持的 patch 在第 {cut_patch} 个：调用 is_supported {len(probe.calls)} 次")


# ---------------------------------------------------------------- Part 4

EXAMPLE4 = [
    ("toolkit", "1.0.0", {"netlib": ">=2.0.0", "parser": ">=1.0.0"}),
    ("netlib", "1.0.0", {}),
    ("netlib", "2.0.0", {"coreutil": "==1.0.0"}),
    ("netlib", "2.1.0", {"coreutil": ">=1.2.0"}),
    ("parser", "1.0.0", {"coreutil": "<1.2.0"}),
    ("parser", "1.5.0", {"coreutil": "<1.0.0"}),
    ("coreutil", "1.0.0", {}),
    ("coreutil", "1.2.0", {}),
    ("coreutil", "1.3.0", {}),
]


@case("Part 4", sample=True, title="题目例子 toolkit")
def p4_example(sol):
    reg = ref.build(EXAMPLE4)
    call = calling(ref.registry_code(reg, "toolkit"))
    got = sol.resolve(reg, "toolkit")
    expect_true(got is not None, "存在合法的选择，却返回了 None", call)
    expect_valid(got, lambda g: ref.check_order(reg, "toolkit", g), call=call,
                 want=["coreutil@1.0.0", "netlib@2.0.0", "parser@1.0.0", "toolkit@1.0.0"])


@case("Part 4", title="没有依赖：任选一个版本")
def p4_no_deps(sol):
    check_resolve(sol, ref.build([("solo", "1.0.0", {}), ("solo", "2.0.0", {})]), "solo")
    check_resolve(sol, ref.build([("solo", "0.1.0", {}), ("other", "1.0.0", {"solo": ">=0.1.0"})]), "solo")


@case("Part 4", title="根包不存在：返回 None")
def p4_unknown_root(sol):
    check_resolve(sol, ref.build([("lib", "1.0.0", {})]), "app")
    check_resolve(sol, ref.Registry(), "app")


@case("Part 4", title="依赖了注册表里没有的包")
def p4_unknown_dependency(sol):
    check_resolve(sol, ref.build([("app", "2.0.0", {"ghost": ">=1.0.0"}), ("app", "1.0.0", {"lib": ">=1.0.0"}),
                                  ("lib", "1.0.0", {})]), "app")
    check_resolve(sol, ref.build([("app", "1.0.0", {"lib": ">=1.0.0"}), ("lib", "2.0.0", {"ghost": "==1.0.0"}),
                                  ("lib", "1.0.0", {})]), "app")
    check_resolve(sol, ref.build([("app", "1.0.0", {"ghost": ">=0.0.0"})]), "app")


@case("Part 4", title="约束无解：返回 None")
def p4_unsatisfiable(sol):
    check_resolve(sol, ref.build([("app", "1.0.0", {"lib": ">2.0.0"}), ("lib", "2.0.0", {}), ("lib", "1.0.0", {})]), "app")
    check_resolve(sol, ref.build([("app", "1.0.0", {"a": ">=1.0.0", "b": ">=1.0.0"}),
                                  ("a", "1.0.0", {"c": "<=1.0.0"}), ("b", "1.0.0", {"c": ">1.0.0"}),
                                  ("c", "1.0.0", {}), ("c", "2.0.0", {})]), "app")


@case("Part 4", title="运算符边界：> < >= <= ==")
def p4_operators(sol):
    for op in ("==", ">=", "<=", ">", "<"):
        rows = [("app", "1.0.0", {"lib": op + "1.5.0"})] + [("lib", v, {}) for v in ("1.4.9", "1.5.0", "1.5.1")]
        check_resolve(sol, ref.build(rows), "app")
        rows = [("app", "1.0.0", {"lib": op + "1.5.0"}), ("lib", "1.5.0", {})]
        check_resolve(sol, ref.build(rows), "app")


@case("Part 4", title="约束里的版本也按数值比较")
def p4_numeric_constraints(sol):
    check_resolve(sol, ref.build([("app", "1.0.0", {"lib": ">=1.9.0"}), ("lib", "1.10.0", {})]), "app")
    check_resolve(sol, ref.build([("app", "1.0.0", {"lib": "<1.10.0"}), ("lib", "1.9.0", {}), ("lib", "1.11.0", {})]), "app")
    check_resolve(sol, ref.build([("app", "1.0.0", {"lib": "==1.2.0"}), ("lib", "01.02.00", {})]), "app")
    check_resolve(sol, ref.build([("app", "1.0.0", {"lib": ">=02.0.0", "util": "<=010.0.0"}), ("lib", "10.0.0", {}),
                                  ("lib", "9.0.0", {"util": ">9.0.0"}), ("util", "9.5.0", {}), ("util", "10.0.1", {})]), "app")


@case("Part 4", title="冲突要回溯到更早的选择")
def p4_backtrack(sol):
    # the newest app needs lib 2.x, whose only version clashes with util; app 1.0.0 works
    check_resolve(sol, ref.build([("app", "2.0.0", {"lib": ">=2.0.0", "util": ">=1.0.0"}),
                                  ("app", "1.0.0", {"lib": ">=1.0.0", "util": ">=1.0.0"}),
                                  ("lib", "2.0.0", {"base": "==2.0.0"}), ("lib", "1.0.0", {"base": ">=1.0.0"}),
                                  ("util", "1.0.0", {"base": "<2.0.0"}),
                                  ("base", "1.0.0", {}), ("base", "2.0.0", {})]), "app")
    # the clash shows up three levels below the choice that must change
    check_resolve(sol, ref.build([("app", "1.0.0", {"a": ">=1.0.0", "z": ">=1.0.0"}),
                                  ("a", "2.0.0", {"b": ">=1.0.0"}), ("a", "1.0.0", {}),
                                  ("b", "1.0.0", {"c": ">=1.0.0"}), ("c", "1.0.0", {"d": "==2.0.0"}),
                                  ("z", "1.0.0", {"d": "==1.0.0"}), ("d", "1.0.0", {}), ("d", "2.0.0", {})]), "app")


@case("Part 4", title="成环的选择不合法")
def p4_cycle(sol):
    check_resolve(sol, ref.build([("app", "1.0.0", {"x": ">=1.0.0"}), ("x", "1.0.0", {"y": ">=1.0.0"}),
                                  ("y", "1.0.0", {"x": ">=1.0.0"})]), "app")
    check_resolve(sol, ref.build([("app", "1.0.0", {"app": ">=1.0.0"})]), "app")
    check_resolve(sol, ref.build([("app", "1.0.0", {"x": ">=1.0.0"}), ("x", "1.0.0", {"app": "==1.0.0"})]), "app")


@case("Part 4", title="环可以靠换版本绕开")
def p4_avoid_cycle(sol):
    check_resolve(sol, ref.build([("app", "1.0.0", {"x": ">=1.0.0"}), ("x", "1.0.0", {}), ("x", "2.0.0", {"y": ">=1.0.0"}),
                                  ("y", "1.0.0", {"x": ">=1.0.0"})]), "app")
    # the oldest and the newest x both close a cycle; only the middle one does not
    check_resolve(sol, ref.build([("app", "1.0.0", {"x": ">=1.0.0"}), ("x", "1.0.0", {"y": ">=1.0.0"}),
                                  ("x", "2.0.0", {}), ("x", "3.0.0", {"y": ">=1.0.0"}),
                                  ("y", "1.0.0", {"x": ">=1.0.0"})]), "app")
    # a self-dependency rules out that version only
    check_resolve(sol, ref.build([("app", "2.0.0", {"app": ">=1.0.0"}), ("app", "1.0.0", {})]), "app")


@case("Part 4", title="安装顺序：共享依赖排在最前")
def p4_install_order(sol):
    # c needs b too: a breadth-first order reversed would put c before b
    check_resolve(sol, ref.build([("a", "1.0.0", {"b": ">=1.0.0", "c": ">=1.0.0"}), ("b", "1.0.0", {}),
                                  ("c", "1.0.0", {"b": ">=1.0.0"})]), "a")
    check_resolve(sol, ref.build([("a", "1.0.0", {"b": ">=1.0.0", "e": ">=1.0.0"}), ("b", "1.0.0", {"c": ">=1.0.0"}),
                                  ("c", "1.0.0", {"d": ">=1.0.0"}), ("d", "1.0.0", {}),
                                  ("e", "1.0.0", {"d": ">=1.0.0", "c": ">=1.0.0"})]), "a")


@case("Part 4", title="不需要的包不能列出")
def p4_only_needed(sol):
    check_resolve(sol, ref.build([("app", "1.0.0", {"lib": ">=1.0.0"}), ("lib", "1.0.0", {}),
                                  ("unused", "1.0.0", {}), ("lib2", "1.0.0", {"lib": ">=1.0.0"})]), "app")
    # lib 2.0.0 would pull in extra, but it cannot be used; extra must not appear
    check_resolve(sol, ref.build([("app", "1.0.0", {"lib": ">=1.0.0"}), ("lib", "2.0.0", {"extra": ">=1.0.0", "base": "==9.0.0"}),
                                  ("lib", "1.0.0", {}), ("extra", "1.0.0", {}), ("base", "1.0.0", {})]), "app")


@case("Part 4", title="随机注册表（对照枚举所有组合）")
def p4_random(sol):
    r = rng()
    for _ in range(300):
        names = list("ABCDE")[:r.randint(2, 5)]
        check_resolve(sol, ref.random_registry(r, names), names[0])
