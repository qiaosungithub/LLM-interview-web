from functools import reduce

from judgelib import calling, case, expect, rng, show_call
import ref

EDGES1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 5), (2, 3), (3, 4), (3, 5), (5, 6)]   # Part 1 example
OUT1 = ["W", "W", "L", "W", "L", "W", "L"]
EDGES2 = [(1, 0), (2, 3), (3, 2), (4, 2), (4, 0), (5, 2), (5, 3), (6, 1)]           # Part 2 example
EDGES3 = [(1, 0), (2, 0)]                                                          # Part 3 example

# Grundy numbers 0, 1, 2, 3 on nodes 0..3 (node k has an edge to every smaller node), node 4 -> 3 has Grundy 0
# although it is not a sink, and node 5 -> {0, 2} has Grundy 1 (the smallest value missing), not 3.
G_EDGES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 3), (5, 0), (5, 2)]


def run(sol, fn, args):
    """Calls sol.fn on copies of the list arguments; a tuple of outcomes counts as the list it holds."""
    got = getattr(sol, fn)(*[list(a) if isinstance(a, list) else a for a in args])
    return list(got) if isinstance(got, tuple) and fn in ("outcomes_dag", "outcomes") else got


def _msg(fn, want):
    """What the right answer means, for results whose mismatch says nothing by itself (a list gets judgelib's
    message naming the first differing position)."""
    if fn == "first_player_wins":
        return "先手能确保获胜，应返回 True" if want else "先手无法确保获胜，应返回 False"
    if fn == "winning_move":
        return ("节点 v 的结果是 \"L\"，不存在必胜走法，应返回 None" if want is None
                else f"应返回结果为 \"L\" 的后继中编号最小的那个：{want}")
    return ""


def check(sol, fn, want, *args):
    """Compares sol.fn(*args) with want; the call shown can be pasted as is."""
    call = calling(show_call(fn, *args))
    expect(run(sol, fn, args), want, call, _msg(fn, want))


def check_big(sol, fn, want, n, edges, text, *extra, extra_text=None):
    """The same for a big graph, shown as the Python expression `text` that builds the edge list (and
    `extra_text` for the remaining arguments when they are long)."""
    rest = extra_text if extra_text is not None else ", ".join(repr(x) for x in extra)
    call = calling(f"{fn}({n}, {text}, {rest})" if rest else f"{fn}({n}, {text})")
    expect(run(sol, fn, (n, edges) + extra), want, call, _msg(fn, want))


def xor_wins(g, tokens):
    return reduce(lambda a, b: a ^ b, (g[t] for t in tokens), 0) != 0


def relabel(edges, perm):
    return [(perm[u], perm[v]) for u, v in edges]


# Big graphs: a hidden chain of 100000 nodes labelled i * 7919 % n (so node numbers are no topological order),
# with edges i -> i + d. They and their answers are built at import time, which is not charged to any case.
N = 100_000
MULT = 7919
PATH = ref.dag_edges(N, MULT, (1,))
PATH_TEXT = ref.dag_text(N, MULT, (1,))
PATH_OUT = ref.linear_outcomes(N, PATH)
PATH_G = ref.grundy(N, PATH)
DENSE_STEPS = (1, 2, 7)
DENSE = ref.dag_edges(N, MULT, DENSE_STEPS)
DENSE_TEXT = ref.dag_text(N, MULT, DENSE_STEPS)
DENSE_OUT = ref.linear_outcomes(N, DENSE)
WIDE_STEPS = (1, 3, 4, 11)
WIDE = ref.dag_edges(N, MULT, WIDE_STEPS)
WIDE_TEXT = ref.dag_text(N, MULT, WIDE_STEPS)
WIDE_G = ref.grundy(N, WIDE)

# The chain's first node (label 0) and second node (label 7919): their outcomes differ, as the chain alternates.
PATH_W = 0 if PATH_OUT[0] == "W" else MULT
PATH_L = MULT if PATH_W == 0 else 0

_med = rng("medium")
MED_DAGS = []
for _ in range(10):
    _n = _med.randint(150, 300)
    _e = ref.random_dag(_med, _n, _med.choice([0.01, 0.02, 0.05]))
    MED_DAGS.append((_n, _e, ref.fixpoint_outcomes(_n, _e)))
MED_GRAPHS = []
for _ in range(10):
    _n = _med.randint(150, 300)
    _e = ref.random_digraph(_med, _n, _med.choice([0.004, 0.007, 0.012]))
    MED_GRAPHS.append((_n, _e, ref.fixpoint_outcomes(_n, _e)))


def small_dags(r, count):
    for _ in range(count):
        n = r.randint(1, 8)
        yield n, ref.random_dag(r, n, r.choice([0.15, 0.3, 0.5, 0.8]))


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    check(sol, "outcomes_dag", OUT1, 7, EDGES1)
    check(sol, "winning_move", 2, 7, EDGES1, 0)
    check(sol, "winning_move", None, 7, EDGES1, 2)
    check(sol, "winning_move", None, 7, EDGES1, 6)


@case("Part 1", title="一个节点；孤立节点；单条边")
def p1_tiny(sol):
    check(sol, "outcomes_dag", ["L"], 1, [])
    check(sol, "winning_move", None, 1, [], 0)
    check(sol, "outcomes_dag", ["W", "L"], 2, [(0, 1)])
    check(sol, "winning_move", 1, 2, [(0, 1)], 0)
    check(sol, "outcomes_dag", ["L", "L", "W", "L"], 4, [(2, 3)])
    check(sol, "winning_move", 3, 4, [(2, 3)], 2)


@case("Part 1", title="节点编号不是拓扑序")
def p1_numbering_not_topo(sol):
    perm = [6, 5, 4, 3, 2, 1, 0]                   # the example with every node v renamed 6 - v
    check(sol, "outcomes_dag", OUT1[::-1], 7, relabel(EDGES1, perm))
    check(sol, "winning_move", 4, 7, relabel(EDGES1, perm), 6)
    check(sol, "outcomes_dag", ["L", "W"], 2, [(1, 0)])
    check(sol, "outcomes_dag", ["L", "W", "L"], 3, [(2, 1), (1, 0)])
    check(sol, "outcomes_dag", ["W", "L", "W", "L"], 4, [(0, 3), (2, 1), (1, 0)])


@case("Part 1", title="返回编号最小的必胜走法")
def p1_winning_move_smallest(sol):
    # 0 -> 4, 3, 1 (listed in that order): 4 and 3 are sinks ("L"), 1 -> 2 is "W"; the answer is 3
    edges = [(0, 4), (0, 3), (0, 1), (1, 2)]
    check(sol, "outcomes_dag", ["W", "W", "L", "L", "L"], 5, edges)
    check(sol, "winning_move", 3, 5, edges, 0)
    check(sol, "winning_move", 2, 5, edges, 1)
    check(sol, "winning_move", 2, 7, [(0, 5), (0, 2), (0, 6), (0, 1), (1, 3), (5, 4)], 0)


@case("Part 1", title="必败节点返回 None")
def p1_winning_move_none(sol):
    check(sol, "winning_move", None, 3, [(0, 1), (1, 2)], 0)       # 0's only move goes to the "W" node 1
    check(sol, "winning_move", None, 3, [(0, 1), (1, 2)], 2)
    check(sol, "winning_move", None, 5, [(0, 1), (0, 2), (1, 3), (2, 4)], 0)
    check(sol, "winning_move", None, 4, [(2, 3)], 1)


@case("Part 1", title="随机小图（对照定义直接迭代）")
def p1_random(sol):
    for n, edges in small_dags(rng(), 200):
        out = ref.fixpoint_outcomes(n, edges)
        check(sol, "outcomes_dag", out, n, edges)
        for v in range(n):
            check(sol, "winning_move", ref.winning_move(n, edges, v, out), n, edges, v)


@case("Part 1", title="随机中等的图，150 到 300 个节点")
def p1_medium(sol):
    for n, edges, out in MED_DAGS:
        check(sol, "outcomes_dag", out, n, edges)
        for v in range(0, n, 37):
            check(sol, "winning_move", ref.winning_move(n, edges, v, out), n, edges, v)


@case("Part 1", cpu=3, title="10^5 个节点的长链：不能递归")
def p1_long_path(sol):
    check_big(sol, "outcomes_dag", PATH_OUT, N, PATH, PATH_TEXT)


@case("Part 1", cpu=4, title="10^5 个节点、约 3×10^5 条边")
def p1_big_dag(sol):
    check_big(sol, "outcomes_dag", DENSE_OUT, N, DENSE, DENSE_TEXT)


@case("Part 1", cpu=5, title="长链上的 winning_move")
def p1_big_winning_move(sol):
    check_big(sol, "winning_move", (PATH_W + MULT) % N, N, PATH, PATH_TEXT, PATH_W)
    check_big(sol, "winning_move", None, N, PATH, PATH_TEXT, PATH_L)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    check(sol, "outcomes", ["L", "W", "D", "D", "W", "D", "L"], 7, EDGES2)


@case("Part 2", title="只能绕圈：平局")
def p2_pure_cycles(sol):
    check(sol, "outcomes", ["D", "D"], 2, [(0, 1), (1, 0)])
    check(sol, "outcomes", ["D", "D", "D"], 3, [(0, 1), (1, 2), (2, 0)])
    check(sol, "outcomes", ["D", "D", "D"], 3, [(2, 0), (0, 1), (1, 0)])        # 2 can only enter the cycle
    check(sol, "outcomes", ["L", "D", "D"], 3, [(1, 2), (2, 1)])


@case("Part 2", title="有环但能分出胜负")
def p2_cycle_with_exit(sol):
    check(sol, "outcomes", ["L", "W", "L"], 3, [(0, 1), (1, 0), (1, 2)])
    check(sol, "outcomes", ["W", "L", "L"], 3, [(0, 1), (1, 0), (0, 2)])
    check(sol, "outcomes", ["W", "L", "W", "L"], 4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    check(sol, "outcomes", ["W", "L", "W", "L"], 4, [(2, 3), (2, 0), (1, 2), (0, 1)])


@case("Part 2", title="后继有 W 也有 D：平局")
def p2_draw_not_loss(sol):
    # 4 -> 1 ("W") and 4 -> 2 (on the 2 <-> 3 cycle, "D"): moving to 1 loses, so the mover keeps cycling
    check(sol, "outcomes", ["L", "W", "D", "D", "D"], 5, [(1, 0), (2, 3), (3, 2), (4, 1), (4, 2)])
    check(sol, "outcomes", ["L", "W", "D", "D", "W"], 5, [(1, 0), (2, 3), (3, 2), (4, 2), (4, 0)])


@case("Part 2", title="已经确定的结果不会再变")
def p2_label_never_changes(sol):
    # 1 -> 0 and 1 -> 2, both sinks, so 1 is "W" (found twice); 3 -> 1 and 3 -> 4, where 4 <-> 5 draw: 3 is "D"
    edges = [(1, 0), (1, 2), (3, 1), (3, 4), (4, 5), (5, 4)]
    check(sol, "outcomes", ["L", "W", "L", "D", "D", "D"], 6, edges)
    edges = [(1, 0), (1, 2), (1, 3), (4, 1), (4, 5), (5, 6), (6, 5)]
    check(sol, "outcomes", ["L", "W", "L", "L", "D", "D", "D"], 7, edges)


@case("Part 2", title="无环图上与 Part 1 相同")
def p2_dag_same_as_part1(sol):
    check(sol, "outcomes", OUT1, 7, EDGES1)
    for n, edges in small_dags(rng(), 100):
        check(sol, "outcomes", ref.fixpoint_outcomes(n, edges), n, edges)


@case("Part 2", title="随机小图（对照定义直接迭代）")
def p2_random(sol):
    r = rng()
    for _ in range(300):
        n = r.randint(1, 9)
        edges = ref.random_digraph(r, n, r.choice([0.1, 0.2, 0.35, 0.5]))
        check(sol, "outcomes", ref.fixpoint_outcomes(n, edges), n, edges)


@case("Part 2", title="随机中等的图，150 到 300 个节点")
def p2_medium(sol):
    for n, edges, out in MED_GRAPHS:
        check(sol, "outcomes", out, n, edges)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    check(sol, "first_player_wins", True, 3, EDGES3, [1])
    check(sol, "first_player_wins", True, 3, EDGES3, [2])
    check(sol, "first_player_wins", False, 3, EDGES3, [1, 2])


@case("Part 3", title="mex 取最小的缺口")
def p3_mex_gap(sol):
    check(sol, "first_player_wins", False, 6, G_EDGES, [4])          # Grundy 0, though 4 has a move
    check(sol, "first_player_wins", False, 6, G_EDGES, [5, 1])       # mex{0, 2} = 1, and 1 xor 1 = 0
    check(sol, "first_player_wins", False, 3, [(2, 1), (1, 0)], [2])


@case("Part 3", title="Grundy 数大于 1")
def p3_grundy_above_one(sol):
    # on nodes 0..3 of G_EDGES the Grundy number is the node number
    check(sol, "first_player_wins", True, 6, G_EDGES, [1, 2])        # 1 xor 2: two "W" tokens, still a win
    check(sol, "first_player_wins", False, 6, G_EDGES, [1, 2, 3])    # 1 xor 2 xor 3 = 0: three "W" tokens, a loss
    check(sol, "first_player_wins", True, 6, G_EDGES, [2, 3])
    check(sol, "first_player_wins", True, 6, G_EDGES, [0, 3, 1])


@case("Part 3", title="一枚棋子：与 Part 1 一致")
def p3_one_token(sol):
    for v in range(7):
        check(sol, "first_player_wins", OUT1[v] == "W", 7, EDGES1, [v])


@case("Part 3", title="多枚棋子在同一节点")
def p3_shared_node(sol):
    check(sol, "first_player_wins", False, 3, EDGES3, [1, 1])
    check(sol, "first_player_wins", True, 3, EDGES3, [1, 1, 1])
    check(sol, "first_player_wins", False, 3, EDGES3, [0, 0])
    check(sol, "first_player_wins", True, 3, EDGES3, [2, 1, 1])
    check(sol, "first_player_wins", False, 6, G_EDGES, [3, 3, 3, 3])


@case("Part 3", title="随机小图（对照逐步搜索）")
def p3_random(sol):
    r = rng()
    for n, edges in small_dags(r, 150):
        tokens = [r.randrange(n) for _ in range(r.randint(1, 4))]
        check(sol, "first_player_wins", ref.token_game(n, edges, tokens), n, edges, tokens)


@case("Part 3", title="随机中等的图，多枚棋子")
def p3_medium(sol):
    r = rng()
    for n, edges, _ in MED_DAGS:
        g = ref.grundy(n, edges)
        for _ in range(5):
            tokens = [r.randrange(n) for _ in range(r.randint(1, 30))]
            check(sol, "first_player_wins", xor_wins(g, tokens), n, edges, tokens)


@case("Part 3", cpu=5, title="10^5 个节点：长链和较密的图")
def p3_big(sol):
    tokens = [0]
    check_big(sol, "first_player_wins", xor_wins(PATH_G, tokens), N, PATH, PATH_TEXT, tokens)
    tokens = list(range(0, N, 997))
    check_big(sol, "first_player_wins", xor_wins(WIDE_G, tokens), N, WIDE, WIDE_TEXT, tokens,
              extra_text=f"list(range(0, {N}, 997))")
