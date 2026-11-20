from judgelib import calling, case, expect, rng, show_call
import ref

EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("E", "F")]       # the statement's graph
NAMES = [chr(ord("A") + i) for i in range(26)] + ["a", "b", "ab", "10", "9", "100", "x1", "x10", "x2", "Bob", "bob"]


# ---------------------------------------------------------------- checks, with equivalent forms accepted

def _as_list(x):
    return list(x) if isinstance(x, tuple) else x


def check_above(sol, arr, threshold, want=None):
    want = ref.above_threshold(arr, threshold) if want is None else want
    call = calling(show_call("above_threshold", arr, threshold))
    got = sol.above_threshold(list(arr), threshold)
    if isinstance(got, (tuple, list)) and len(got) == 2:            # (count, indices) as a tuple or a list
        got = (got[0], _as_list(got[1]))
        if got[0] == want[0] and got[1] != want[1]:
            expect(got, want, call=call, msg="个数对了，但下标列表不对")
        if got[0] != want[0] and got[1] == want[1]:
            expect(got, want, call=call, msg="下标列表对了，但个数不对")
    expect(got, want, call=call)


def check_many(sol, arr, thresholds, want=None, call=None):
    want = ref.count_above_brute(arr, thresholds) if want is None else want
    call = calling(call or show_call("count_above_many", arr, thresholds))
    expect(_as_list(sol.count_above_many(list(arr), list(thresholds))), want, call=call)


def check_graph(sol, edges, want=None):
    want = ref.graph(edges) if want is None else want
    call = calling(show_call("build_graph", edges))
    got = sol.build_graph(list(edges))
    if isinstance(got, dict):
        got = {k: _as_list(v) for k, v in got.items()}
        if set(got) != set(want):
            missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
            expect(got, want, call=call,
                   msg="节点集合不对" + (f"：缺少 {missing}" if missing else "") + (f"，多出 {extra}" if extra else ""))
        for node in sorted(want):
            if got[node] != want[node]:
                expect(got, want, call=call, msg=f"节点 {node!r} 的邻居应为 {want[node]}，得到 {got[node]!r}")
    expect(got, want, call=call)


def check_dfs(sol, edges, start, want=None, call=None):
    want = ref.dfs_order(edges, start) if want is None else want
    call = calling(call or show_call("dfs_order", edges, start))
    got = _as_list(sol.dfs_order(list(edges), start))
    if isinstance(got, list) and got != want and len(got) == len(want) > 30:
        k = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b) if sorted(map(str, got)) == sorted(want) else None
        if k is not None:
            expect(got, want, call=call, msg=f"第 {k} 个（下标从 0 开始）访问的节点不对：应为 {want[k]!r}，得到 {got[k]!r}")
    expect(got, want, call=call)


def check_components(sol, edges, want=None):
    want = ref.components(edges) if want is None else want
    call = calling(show_call("components", edges))
    got = sol.components(list(edges))
    if isinstance(got, (list, tuple)):
        got = [_as_list(c) for c in got]
        if got != want and len(got) == len(want) and all(isinstance(c, list) for c in got):
            if sorted(map(sorted, got)) == sorted(map(sorted, want)):
                if [sorted(c) for c in got] == [sorted(c) for c in want]:
                    msg = "分组和分量顺序都对，但分量内部不是从最小节点出发的 DFS 顺序"
                else:
                    msg = "分组对了，但分量之间要按各自最小的节点排序（或某个分量内部顺序不对）"
                expect(got, want, call=call, msg=msg)
    expect(got, want, call=call)


CYCLE_MSG = {True: "这个图里有环（3 个或更多不同节点首尾相连），应返回 True",
             False: "这个图里没有环（重复的边和自环不算环），应返回 False"}


def check_cycle(sol, edges, want=None, call=None):
    want = ref.has_cycle(edges) if want is None else want
    call = calling(call or show_call("has_cycle", edges))
    got = sol.has_cycle(list(edges))
    expect(got, want, call=call, msg=CYCLE_MSG[want] if got in (True, False) else "")


# ---------------------------------------------------------------- big inputs, built once at import time

# n = q = 10**5, written as expressions so a failing call can be pasted back: every value repeats once or
# twice, and most thresholds equal some element (where bisect_left and bisect_right differ)
BIG_ARR_TEXT = "[(i * 7919) % 60013 - 30000 for i in range(100000)]"
BIG_THRESHOLDS_TEXT = "[(i * 104729) % 60107 - 30050 for i in range(100000)]"
BIG_ARR = [(i * 7919) % 60013 - 30000 for i in range(100000)]
BIG_THRESHOLDS = [(i * 104729) % 60107 - 30050 for i in range(100000)]
BIG_COUNTS = ref.count_above_sweep(BIG_ARR, BIG_THRESHOLDS)
CHAIN = 100000
CHAIN_EDGES = [(str(i), str(i + 1)) for i in range(CHAIN - 1)]
CHAIN_TEXT = f"[(str(i), str(i + 1)) for i in range({CHAIN - 1})]"


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子 above_threshold")
def p1_example(sol):
    check_above(sol, [1, 5, 7, 2, 8, 3], 4, (3, [1, 2, 4]))


@case("Part 1", title="空数组")
def p1_empty(sol):
    check_above(sol, [], 0, (0, []))
    check_above(sol, [], -5, (0, []))


@case("Part 1", title="等于阈值的元素不计入")
def p1_equal_not_counted(sol):
    check_above(sol, [4, 4, 4], 4, (0, []))
    check_above(sol, [3, 4, 5, 4], 4, (1, [2]))
    check_above(sol, [5, 5, 1], 4, (2, [0, 1]))


@case("Part 1", title="负数和重复元素")
def p1_negatives(sol):
    check_above(sol, [-1, -5, 0, -1], -2, (3, [0, 2, 3]))
    check_above(sol, [-1, -5, 0, -1], -1, (1, [2]))
    check_above(sol, [-3, -3], -10, (2, [0, 1]))


@case("Part 1", title="随机数组，与逐个比较对照")
def p1_random(sol):
    r = rng()
    for _ in range(300):
        arr = [r.randint(-10, 10) for _ in range(r.randint(0, 15))]
        t = r.choice(arr) if arr and r.random() < 0.5 else r.randint(-12, 12)
        check_above(sol, arr, t)


@case("Part 1", title="多个阈值：基本情况")
def p1_many_basic(sol):
    check_many(sol, [1, 5, 7, 2, 8, 3], [4, 0, 6, 100, -3], [3, 6, 2, 0, 6])


@case("Part 1", title="多个阈值：阈值等于元素时不计入")
def p1_many_equal(sol):
    check_many(sol, [2, 2, 2, 5, 5, 9], [2, 5, 9, 1, 10], [3, 1, 0, 6, 0])
    check_many(sol, [7], [7, 6], [0, 1])


@case("Part 1", title="多个阈值：空数组或没有阈值")
def p1_many_empty(sol):
    check_many(sol, [], [0, -1], [0, 0])
    check_many(sol, [1, 2, 3], [], [])
    check_many(sol, [], [], [])


@case("Part 1", title="多个阈值：按原顺序返回，可重复")
def p1_many_order(sol):
    check_many(sol, [3, 1, 4, 1, 5, 9, 2, 6], [9, 1, 5, 1, -1, 4], [0, 6, 2, 6, 8, 3])


@case("Part 1", title="多个阈值：随机数组，与逐个数对照")
def p1_many_random(sol):
    r = rng()
    for _ in range(300):
        arr = [r.randint(-8, 8) for _ in range(r.randint(0, 15))]
        thresholds = [r.choice(arr) if arr and r.random() < 0.5 else r.randint(-10, 10) for _ in range(r.randint(0, 8))]
        check_many(sol, arr, thresholds)


@case("Part 1", title="n = q = 10⁵：要求 O((n + q) log n)")
def p1_many_big(sol):
    check_many(sol, BIG_ARR, BIG_THRESHOLDS, BIG_COUNTS, call=f"count_above_many({BIG_ARR_TEXT}, {BIG_THRESHOLDS_TEXT})")


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子 build_graph")
def p2_example_graph(sol):
    check_graph(sol, EDGES, {
        "A": ["B", "C"], "B": ["A", "D", "E"], "C": ["A", "F"],
        "D": ["B"], "E": ["B", "F"], "F": ["C", "E"],
    })


@case("Part 2", sample=True, title="题目例子 dfs_order")
def p2_example_dfs(sol):
    check_dfs(sol, EDGES, "A", ["A", "B", "D", "E", "F", "C"])


@case("Part 2", title="没有边")
def p2_empty(sol):
    check_graph(sol, [], {})
    check_dfs(sol, [], "A", ["A"])


@case("Part 2", title="邻居按递增顺序排好")
def p2_sorted_neighbours(sol):
    check_graph(sol, [("A", "D"), ("A", "C"), ("A", "B")], {"A": ["B", "C", "D"], "B": ["A"], "C": ["A"], "D": ["A"]})
    check_dfs(sol, [("A", "D"), ("D", "E"), ("A", "C"), ("A", "B")], "A", ["A", "B", "C", "D", "E"])


@case("Part 2", title="自环只登记节点，不带来邻居")
def p2_self_loops(sol):
    check_graph(sol, [("G", "G")], {"G": []})
    check_graph(sol, [("A", "A"), ("A", "B"), ("B", "B")], {"A": ["B"], "B": ["A"]})
    check_dfs(sol, [("G", "G")], "G", ["G"])
    check_dfs(sol, [("A", "A"), ("A", "B")], "A", ["A", "B"])


@case("Part 2", title="重复的边（含反向）只算一条")
def p2_duplicates(sol):
    check_graph(sol, [("A", "B"), ("B", "A"), ("A", "B")], {"A": ["B"], "B": ["A"]})
    check_graph(sol, [("C", "A"), ("A", "B"), ("A", "C"), ("B", "A")], {"A": ["B", "C"], "B": ["A"], "C": ["A"]})
    check_dfs(sol, [("B", "A"), ("A", "B"), ("A", "C"), ("C", "A")], "A", ["A", "B", "C"])


@case("Part 2", title="节点名按字符串比较")
def p2_string_order(sol):
    edges = [("x", "9"), ("x", "10"), ("x", "a"), ("x", "B"), ("x", "ab"), ("9", "100")]
    check_graph(sol, edges, {"x": ["10", "9", "B", "a", "ab"], "9": ["100", "x"], "10": ["x"], "a": ["x"],
                             "B": ["x"], "ab": ["x"], "100": ["9"]})
    check_dfs(sol, edges, "x", ["x", "10", "9", "100", "B", "a", "ab"])


@case("Part 2", title="start 不在图中")
def p2_start_missing(sol):
    check_dfs(sol, [("X", "Y")], "Z", ["Z"])
    check_dfs(sol, [("A", "B"), ("B", "C")], "a", ["a"])


@case("Part 2", title="已入栈的节点可能先从别处被访问")
def p2_visit_order(sol):
    check_dfs(sol, [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "D")], "A", ["A", "B", "D", "C", "E"])
    check_dfs(sol, [("A", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("D", "C")], "A", ["A", "B", "D", "C"])


@case("Part 2", title="只返回可达的节点，起点可在中间")
def p2_reachable_only(sol):
    edges = [("A", "B"), ("C", "D"), ("D", "E"), ("E", "C"), ("F", "F")]
    check_dfs(sol, edges, "B", ["B", "A"])
    check_dfs(sol, edges, "E", ["E", "C", "D"])
    check_dfs(sol, edges, "F", ["F"])
    check_dfs(sol, EDGES, "E", ["E", "B", "A", "C", "F", "D"])


@case("Part 2", title="随机图，与逐步走最小邻居的 DFS 对照")
def p2_random(sol):
    r = rng()
    for _ in range(300):
        n = r.randint(1, 12)
        edges = ref.random_edges(r, n, r.randint(0, 16), NAMES) if r.random() < 0.6 else \
            ref.random_forest(r, r.sample(NAMES, n), r.randint(0, 2))
        check_graph(sol, edges)
        nodes = sorted({x for e in edges for x in e}) or ["A"]
        check_dfs(sol, edges, r.choice(nodes))


@case("Part 2", cpu=2, title="10 万个节点的链，从一端出发")
def p2_chain_end(sol):
    check_dfs(sol, CHAIN_EDGES, "0", [str(i) for i in range(CHAIN)], call=f'dfs_order({CHAIN_TEXT}, "0")')


@case("Part 2", cpu=2, title="10 万个节点的链，从中间出发")
def p2_chain_middle(sol):
    mid = CHAIN // 2
    want = [str(i) for i in range(mid, -1, -1)] + [str(i) for i in range(mid + 1, CHAIN)]
    check_dfs(sol, CHAIN_EDGES, str(mid), want, call=f'dfs_order({CHAIN_TEXT}, "{mid}")')


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子 components")
def p3_example_components(sol):
    check_components(sol, [("B", "A"), ("A", "C"), ("E", "D"), ("G", "G")], [["A", "B", "C"], ["D", "E"], ["G"]])


@case("Part 3", sample=True, title="题目例子：有环")
def p3_example_cycle(sol):
    check_cycle(sol, EDGES, True)


@case("Part 3", sample=True, title="题目例子：去掉一条边成了树")
def p3_example_tree(sol):
    check_cycle(sol, EDGES[:-1], False)


@case("Part 3", title="没有边")
def p3_empty(sol):
    check_components(sol, [], [])
    check_cycle(sol, [], False)


@case("Part 3", title="分量内部是 DFS 顺序，不是排序")
def p3_components_dfs_order(sol):
    check_components(sol, [("A", "C"), ("C", "B")], [["A", "C", "B"]])
    check_components(sol, [("A", "D"), ("D", "B"), ("A", "C"), ("X", "Z"), ("Z", "Y")], [["A", "C", "D", "B"], ["X", "Z", "Y"]])


@case("Part 3", title="分量按各自最小的节点排序")
def p3_components_order(sol):
    check_components(sol, [("Z", "A"), ("M", "N"), ("Y", "B")], [["A", "Z"], ["B", "Y"], ["M", "N"]])
    check_components(sol, [("x", "9"), ("10", "a"), ("B", "B")], [["10", "a"], ["9", "x"], ["B"]])


@case("Part 3", title="自环和重复的边")
def p3_components_loops(sol):
    check_components(sol, [("G", "G"), ("G", "G"), ("B", "A"), ("A", "B"), ("C", "C")], [["A", "B"], ["C"], ["G"]])


@case("Part 3", title="随机图的连通分量，与并查集对照")
def p3_components_random(sol):
    r = rng()
    for _ in range(300):
        n = r.randint(1, 14)
        edges = ref.random_edges(r, n, r.randint(0, 12), NAMES) if r.random() < 0.5 else \
            ref.random_forest(r, r.sample(NAMES, n), r.randint(0, 2))
        check_components(sol, edges)


@case("Part 3", title="重复的边和自环都不算环")
def p3_cycle_needs_three(sol):
    check_cycle(sol, [("A", "B"), ("B", "A")], False)
    check_cycle(sol, [("A", "A")], False)
    check_cycle(sol, [("A", "B"), ("A", "B"), ("B", "B"), ("B", "C"), ("C", "B")], False)
    check_cycle(sol, [("A", "B"), ("B", "C"), ("C", "A")], True)


@case("Part 3", title="环在另一个连通分量里")
def p3_cycle_other_component(sol):
    check_cycle(sol, [("A", "B"), ("C", "D"), ("D", "E"), ("E", "C")], True)
    check_cycle(sol, [("A", "A"), ("X", "Y"), ("Y", "Z"), ("Z", "X")], True)
    check_cycle(sol, [("A", "B"), ("C", "D"), ("D", "E")], False)


@case("Part 3", title="随机森林加边，与 E > V − C 对照")
def p3_cycle_random(sol):
    r = rng()
    for _ in range(300):
        n = r.randint(1, 14)
        if r.random() < 0.3:
            edges = ref.random_edges(r, n, r.randint(0, 10), NAMES)
        else:
            edges = ref.random_forest(r, r.sample(NAMES, n), r.choice([0, 0, 1, 2]))
        check_cycle(sol, edges)


LARGER = [                       # (expression, cycle or not), each a few thousand nodes
    ("[(str(i), str(i // 2)) for i in range(1, 3000)]", False),                         # a binary tree
    ("[(str(i), str(i // 2)) for i in range(1, 3000)] + [('2999', '1000')]", True),     # plus one edge
    ("[(str(i), str(i + 1000)) for i in range(2000)]", False),                          # 1000 three-node paths
    ("[(str(i), str(i + 1000)) for i in range(2000)] + [('7', '2007')]", True),
    ("[(str(i), str(i)) for i in range(1500)]", False),                                 # self-loops only
]


@case("Part 3", title="较大的图（几千个节点）")
def p3_larger(sol):
    for text, cyclic in LARGER:
        edges = eval(text)
        want = ref.components(edges)
        calling(f"components({text})")
        got = sol.components(list(edges))
        got = [_as_list(c) for c in got] if isinstance(got, (list, tuple)) else got
        expect(got, want, call=f"components({text})")
        check_cycle(sol, edges, cyclic, call=f"has_cycle({text})")
