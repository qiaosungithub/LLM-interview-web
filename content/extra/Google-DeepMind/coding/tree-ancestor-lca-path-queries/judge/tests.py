"""Tests for the rooted tree problem: Part 1 kth_ancestor (and the validation), Part 2 lca / distance, Part 3 path_max."""
from judgelib import calling, case, expect, expect_raises, rng
import ref

EX_PARENT = [3, 0, 0, -1, 3, 4, 4, 6, 1]          # the statement's tree, root 3
EX_WEIGHT = [4, 7, 1, 0, 2, 9, 3, 5, 6]


# ---------------------------------------------------------------- how a call is shown

def _args(args):
    return ", ".join(map(repr, args))


class Session:
    """One tree built by the answer, and the code that rebuilds it. `code` replaces the literal lists for big
    trees; `ctor` is the constructor call as written in that code."""

    def __init__(self, sol, parent, weight=None, code=None, pass_none=False):
        if code is None:
            code = f"parent = {parent!r}" + (f"\nweight = {weight!r}" if weight is not None else "")
        if weight is not None:
            ctor = "RootedTree(parent, weight)"
        else:
            ctor = "RootedTree(parent, None)" if pass_none else "RootedTree(parent)"
        self.setup = f"{code}\nt = {ctor}"
        calling(self.setup)
        if weight is not None:
            self.t = sol.RootedTree(list(parent), list(weight))
        elif pass_none:
            self.t = sol.RootedTree(list(parent), None)
        else:
            self.t = sol.RootedTree(list(parent))

    def line(self, method, args, want):
        return f"{self.setup}\nprint(t.{method}({_args(args)}))   # 应当是 {want!r}"

    def check(self, method, args, want):
        """One query; the call text is only built when it is needed (a failure, an exception, a timeout)."""
        try:
            got = getattr(self.t, method)(*args)
        except BaseException:
            calling(self.line(method, args, want))
            raise
        if got is want or (type(got) is int and got == want):
            return
        expect(got, want, call=self.line(method, args, want), msg=f"t.{method}({_args(args)}) 的返回值不对")

    def check_both(self, method, u, v, want):
        self.check(method, (u, v), want)
        self.check(method, (v, u), want)

    def raises(self, exc, method, args):
        call = calling(f"{self.setup}\nt.{method}({_args(args)})   # 应当抛出 {exc.__name__}")
        expect_raises(exc, getattr(self.t, method), *args, call=call)

    def bulk(self, method, us, vs, wants):
        """Many queries in a row (the timed cases): answers are compared once at the end."""
        f = getattr(self.t, method)
        out = []
        add = out.append
        try:
            for u, v in zip(us, vs):
                add(f(u, v))
        except BaseException:
            i = len(out)
            calling(self.line(method, (us[i], vs[i]), wants[i]))
            raise
        try:
            same = out == wants
        except Exception:
            same = False
        if not same:
            for i, (g, w) in enumerate(zip(out, wants)):
                expect(g, w, call=self.line(method, (us[i], vs[i]), w),
                       msg=f"t.{method}({_args((us[i], vs[i]))}) 的返回值不对")


def ctor_raises(sol, exc, parent, weight=None, code=None, why=""):
    if code is None:
        code = f"parent = {parent!r}" + (f"\nweight = {weight!r}" if weight is not None else "")
    ctor = "RootedTree(parent, weight)" if weight is not None else "RootedTree(parent)"
    call = calling(f"{code}\nt = {ctor}   # 应当抛出 {exc.__name__}" + (f"：{why}" if why else ""))
    args = (list(parent),) if weight is None else (list(parent), list(weight))
    expect_raises(exc, sol.RootedTree, *args, call=call)


def depths(parent):
    return [len(ref.up_chain(parent, v)) - 1 for v in range(len(parent))]


def check_all_kth(s, parent, extra=2):
    """Every v and every k from 0 to depth(v) + extra."""
    for v in range(len(parent)):
        chain = ref.up_chain(parent, v)
        for k in range(len(chain) + extra):
            s.check("kth_ancestor", (v, k), chain[k] if k < len(chain) else -1)


def check_all_pairs(s, parent, weight=None):
    """lca and distance (or path_max when weight is given) for every ordered pair."""
    n = len(parent)
    chains = [ref.up_chain(parent, v) for v in range(n)]
    above = [set(c) for c in chains]
    for u in range(n):
        for v in range(n):
            a = next(x for x in chains[v] if x in above[u])
            if weight is None:
                s.check("lca", (u, v), a)
                s.check("distance", (u, v), chains[u].index(a) + chains[v].index(a))
            else:
                ws = [weight[x] for x in chains[u][:chains[u].index(a)] + chains[v][:chains[v].index(a)]]
                s.check("path_max", (u, v), max(ws) if ws else None)


def reversed_chain(n):
    """Node i hangs under node i + 1: the root is n - 1, node 0 is the deepest, and every parent id is larger."""
    return [i + 1 for i in range(n - 1)] + [-1], f"parent = [i + 1 for i in range({n - 1})] + [-1]"


# ---------------------------------------------------------------- the timed inputs (built at import time)

N = ref.N
POS = ref.POS
CHAIN = ref.chain_parent()
ARMS = ref.arms_parent()
W_POS = [i * 2654435761 % (2 * 10**9 + 1) - 10**9 for i in range(N)]    # weight of the edge above position i
WCHAIN = [0] * N
for _i in range(N):
    WCHAIN[POS[_i]] = W_POS[_i]

_POS_CODE = "N = 200000\npos = [i * 7919 % N for i in range(N)]"
CHAIN_CODE = (f"{_POS_CODE}       # pos[i] 是深度为 i 的节点\nparent = [0] * N\nfor i in range(N):\n"
              "    parent[pos[i]] = pos[i - 1] if i else -1")
ARMS_CODE = (f"{_POS_CODE}       # 根是 pos[0]，奇数 i 是一条臂，偶数 i 是另一条\nparent = [0] * N\n"
             "for i in range(N):\n    parent[pos[i]] = pos[max(0, i - 2)] if i else -1")
WCHAIN_CODE = (f"{CHAIN_CODE}\nweight = [0] * N\nfor i in range(N):\n"
               "    weight[pos[i]] = i * 2654435761 % (2 * 10**9 + 1) - 10**9")

Q1, Q2, Q3 = 100_000, 20_000, 50_000


def _p1_queries():
    r = rng("p1_large")
    vs, ks, wants = [], [], []
    for _ in range(Q1):
        i = r.randrange(N)
        k = r.randrange(i + 1) if r.random() < 0.9 else i + 1 + r.randrange(N)
        vs.append(POS[i])
        ks.append(k)
        wants.append(POS[i - k] if k <= i else -1)
    return vs, ks, wants


def _p2_queries():
    """Mostly one node on each arm (the LCA is the root, far above both), some pairs on the same arm."""
    r = rng("p2_large")
    us, vs, lcas, dists = [], [], [], []
    for _ in range(Q2):
        i = 2 * r.randrange(N // 4, N // 2) - 1                      # odd: arm one, depth (i + 1) // 2
        j = 2 * r.randrange(N // 4, N // 2) - (0 if r.random() < 0.8 else 1)
        if r.random() < 0.5:
            i, j = j, i
        di, dj = (i + 1) // 2, (j + 1) // 2
        if (i - j) % 2:
            a, d = 0, di + dj
        else:
            a, d = min(i, j), abs(di - dj)
        us.append(POS[i])
        vs.append(POS[j])
        lcas.append(POS[a])
        dists.append(d)
    return us, vs, lcas, dists


def _p3_queries():
    """Half long paths, half short ones (up to 300 edges), so the answers are not all near the top."""
    r = rng("p3_large")
    pairs = []
    for q in range(Q3):
        if q % 2:
            i, j = r.randrange(N), r.randrange(N)
        else:
            i = r.randrange(N - 301)
            j = i + 1 + r.randrange(300)
        pairs.append((min(i, j), max(i, j), r.random() < 0.5))
    wants = ref.range_max_offline(W_POS, [(i, j) for i, j, _ in pairs if i != j])
    it = iter(wants)
    us, vs, out = [], [], []
    for i, j, swap in pairs:
        u, v = (POS[j], POS[i]) if swap else (POS[i], POS[j])
        us.append(u)
        vs.append(v)
        out.append(next(it) if i != j else None)
    return us, vs, out


P1_V, P1_K, P1_WANT = _p1_queries()
P2_U, P2_V, P2_LCA, P2_DIST = _p2_queries()
P3_U, P3_V, P3_WANT = _p3_queries()


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子：节点 8 和节点 5")
def p1_example(sol):
    s = Session(sol, EX_PARENT)
    for k, want in enumerate([8, 1, 0, 3, -1]):
        s.check("kth_ancestor", (8, k), want)
    s.check("kth_ancestor", (5, 2), 3)


@case("Part 1", sample=True, title="题目例子：根")
def p1_example_root(sol):
    s = Session(sol, EX_PARENT)
    s.check("kth_ancestor", (3, 0), 3)
    s.check("kth_ancestor", (3, 1), -1)


INVALID = [
    ([], "空列表"),
    ([0], "没有根，节点 0 的父节点是它自己"),
    ([5], "没有根"),
    ([1, 0], "没有根，两个节点互为父节点"),
    ([-1, -1], "两个根"),
    ([-1, 0, -1], "两个根"),
    ([-1, 5], "父节点编号超出范围"),
    ([-1, 2], "父节点编号等于 n"),
    ([-1, -2], "父节点是 -2"),
    ([-1, 0, 2], "节点 2 的父节点是它自己"),
    ([-1, 2, 1], "节点 1、2 绕成一圈"),
    ([-1, 3, 1, 2], "节点 1、3、2 绕成一圈"),
    ([1, 2, 0, -1], "0、1、2 绕成一圈，根 3 没有子节点"),
    ([-1, 0, 3, 4, 2], "2、3、4 绕成一圈"),
    ([-1, 0, 1, 2, 3, 4, 7, 8, 6], "6、7、8 绕成一圈，离根很远"),
]


@case("Part 1", title="不合法的 parent 抛出 ValueError")
def p1_invalid(sol):
    for parent, why in INVALID:
        ctor_raises(sol, ValueError, parent, why=why)
    # a long chain from the root plus a separate cycle of 1000 nodes
    parent = [-1] + list(range(999)) + [1000 + (i + 1) % 1000 for i in range(1000)]
    ctor_raises(sol, ValueError, parent, why="节点 1000 .. 1999 绕成一圈",
                code="parent = [-1] + list(range(999)) + [1000 + (i + 1) % 1000 for i in range(1000)]")
    # a tail hanging into a cycle
    parent = [-1] + list(range(99)) + [101, 102, 103, 101, 100]
    ctor_raises(sol, ValueError, parent, why="104 → 100 → 101 → 102 → 103 → 101 绕成一圈",
                code="parent = [-1] + list(range(99)) + [101, 102, 103, 101, 100]")


@case("Part 1", title="父节点编号比子节点大、单个节点、两个节点")
def p1_valid_shapes(sol):
    for parent in ([-1], [1, -1], [1, 2, 3, -1], [3, 3, 4, 4, -1], [2, 2, -1], [4, 0, 1, 2, -1, 1],
                   [1, 2, 3, 4, 5, 6, 7, 8, -1], [6, 6, 0, 1, 2, 3, -1, 5, 7, 8]):
        check_all_kth(Session(sol, parent), parent, extra=1)        # k = 0 .. depth + 1


@case("Part 1", title="编号越界抛出 IndexError，k < 0 抛出 ValueError")
def p1_errors(sol):
    s = Session(sol, EX_PARENT)
    for v, k in ((-1, 0), (-9, 0), (9, 0), (10, 1), (-1, 2), (10**18, 0)):
        s.raises(IndexError, "kth_ancestor", (v, k))
    for v, k in ((8, -1), (3, -1), (0, -10**18)):
        s.raises(ValueError, "kth_ancestor", (v, k))
    s = Session(sol, [-1])
    s.raises(IndexError, "kth_ancestor", (1, 0))
    s.raises(IndexError, "kth_ancestor", (-1, 0))


BIG_KS = [10**18, 2**60, 2**63 - 1, 2**64 + 1, 10**30]


@case("Part 1", title="k 大于深度（含 10**18）返回 -1")
def p1_k_above_depth(sol):
    chain, code = reversed_chain(37)
    trees = [([-1], None), ([1, -1], None), (EX_PARENT, None), (chain, code),
             (ref.random_tree(rng("t"), 50, "random"), None), (ref.random_tree(rng("u"), 50, "thin"), None)]
    for parent, code in trees:
        s = Session(sol, parent, code=code)
        n = len(parent)
        top = 2 ** (max(1, (n - 1).bit_length()) + 1) + 2
        for v, d in enumerate(depths(parent)):
            for k in list(range(d + 1, top)) + BIG_KS:
                s.check("kth_ancestor", (v, k), -1)


@case("Part 1", title="节点数是 2 的幂或多 1 的链")
def p1_power_two(sol):
    for m in range(1, 12):
        for n in (2**m - 1, 2**m, 2**m + 1):
            if n < 2:
                continue
            parent, code = reversed_chain(n)
            s = Session(sol, parent, code=code)
            ks = {n - 2, n - 1, n}
            for j in range(m + 2):
                ks |= {2**j - 1, 2**j, 2**j + 1}
            for k in sorted(ks):
                s.check("kth_ancestor", (0, k), k if k <= n - 1 else -1)        # node k sits k edges above node 0
            s.check("kth_ancestor", (1, n - 2), n - 1)


@case("Part 1", title="5 × 10⁴ 个节点的链（不能用递归）")
def p1_deep_chain(sol):
    n = 50_000
    parent, code = reversed_chain(n)
    s = Session(sol, parent, code=code)
    for v, k in ((0, 0), (0, 1), (0, 25_000), (0, n - 1), (0, n), (1, n - 2), (1, n - 1), (12_345, 37_654),
                 (12_345, 37_655), (40_000, 2**14), (n - 1, 0), (n - 1, 1), (0, 2**16), (0, 2**15 + 1)):
        s.check("kth_ancestor", (v, k), v + k if v + k <= n - 1 else -1)


@case("Part 1", title="随机树：每个节点的每个 k 和暴力比较")
def p1_random(sol):
    r = rng()
    for shape in ref.SHAPES:
        for n in (1, 2, 3, 5, 8, 16, 17, 33, 60):
            parent = ref.random_tree(r, n, shape)
            check_all_kth(Session(sol, parent), parent)


@case("Part 1", cpu=8, title="2 × 10⁵ 个节点的链，10⁵ 次查询")
def p1_large(sol):
    s = Session(sol, CHAIN, code=CHAIN_CODE)
    s.bulk("kth_ancestor", P1_V, P1_K, P1_WANT)


# ---------------------------------------------------------------- Part 2

def _check_lca_dist(s, u, v, a, d):
    s.check_both("lca", u, v, a)
    s.check_both("distance", u, v, d)


@case("Part 2", sample=True, title="题目例子：两个节点在不同的子树里")
def p2_example(sol):
    s = Session(sol, EX_PARENT)
    for u, v, a, d in ((8, 2, 0, 3), (8, 7, 3, 6), (5, 7, 4, 3)):
        _check_lca_dist(s, u, v, a, d)


@case("Part 2", sample=True, title="题目例子：一个节点是另一个的祖先")
def p2_example_ancestor(sol):
    s = Session(sol, EX_PARENT)
    for u, v, a, d in ((6, 7, 6, 1), (3, 8, 3, 3)):
        _check_lca_dist(s, u, v, a, d)


@case("Part 2", sample=True, title="题目例子：同一个节点")
def p2_example_same(sol):
    s = Session(sol, EX_PARENT)
    _check_lca_dist(s, 7, 7, 7, 0)


@case("Part 2", title="一个节点是另一个的祖先")
def p2_ancestor(sol):
    r = rng()
    chain, code = reversed_chain(70)
    trees = [(EX_PARENT, None), (chain, code)] + \
        [(ref.random_tree(r, n, shape), None) for shape in ("random", "binary", "thin", "caterpillar") for n in (12, 40)]
    for parent, code in trees:
        s = Session(sol, parent, code=code)
        for u in range(len(parent)):
            for d, a in enumerate(ref.up_chain(parent, u)):
                _check_lca_dist(s, u, a, a, d)


@case("Part 2", title="深度相同、LCA 在上面好几层")
def p2_lifting(sol):
    s = Session(sol, EX_PARENT)
    for u, v in ((1, 5), (1, 6), (2, 5), (2, 6), (8, 7)):
        _check_lca_dist(s, u, v, 3, 2 * ref.distance(EX_PARENT, u, 3))
    # two arms of 30 nodes under root 0, labels shuffled: every pair at the same depth meets at the root
    r = rng("arms")
    par = [-1] + [max(0, i - 2) for i in range(1, 61)]
    parent = ref.relabel(par, r)
    s = Session(sol, parent)
    root = parent.index(-1)
    dep = depths(parent)
    for u in range(len(parent)):
        for v in range(len(parent)):
            if u != v and dep[u] == dep[v]:
                s.check("lca", (u, v), root)
                s.check("distance", (u, v), 2 * dep[u])
    for n in (31, 63, 64):
        parent = ref.random_tree(r, n, "binary")
        s = Session(sol, parent)
        dep = depths(parent)
        for u in range(n):
            for v in range(n):
                if dep[u] == dep[v]:
                    s.check("lca", (u, v), ref.lca(parent, u, v))


@case("Part 2", title="编号越界抛出 IndexError")
def p2_errors(sol):
    s = Session(sol, EX_PARENT)
    for method in ("lca", "distance"):
        for u, v in ((-1, 0), (0, -1), (9, 0), (0, 9), (9, 9), (-1, -1), (-9, 3)):
            s.raises(IndexError, method, (u, v))


@case("Part 2", title="随机树：所有点对和暴力比较")
def p2_random(sol):
    r = rng()
    for shape in ref.SHAPES:
        for n in (1, 2, 3, 4, 7, 16, 17, 40):
            parent = ref.random_tree(r, n, shape)
            check_all_pairs(Session(sol, parent), parent)


@case("Part 2", cpu=10, title="2 × 10⁵ 个节点、两条长臂，2 × 10⁴ 对")
def p2_large(sol):
    s = Session(sol, ARMS, code=ARMS_CODE)
    s.bulk("lca", P2_U, P2_V, P2_LCA)
    s.bulk("distance", P2_U, P2_V, P2_DIST)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子：一般的路径")
def p3_example(sol):
    s = Session(sol, EX_PARENT, EX_WEIGHT)
    for u, v, want in ((8, 2, 7), (8, 7, 7), (5, 7, 9), (2, 7, 5), (2, 4, 4)):
        s.check_both("path_max", u, v, want)


@case("Part 3", sample=True, title="题目例子：只有一条边")
def p3_example_one_edge(sol):
    s = Session(sol, EX_PARENT, EX_WEIGHT)
    s.check_both("path_max", 6, 7, 5)


@case("Part 3", sample=True, title="题目例子：u == v 返回 None")
def p3_example_none(sol):
    s = Session(sol, EX_PARENT, EX_WEIGHT)
    s.check("path_max", (7, 7), None)


@case("Part 3", title="根的权重被忽略")
def p3_root_weight(sol):
    for root_w in (10**9, -10**9):
        weight = list(EX_WEIGHT)
        weight[3] = root_w
        s = Session(sol, EX_PARENT, weight)
        for u, v, want in ((8, 7, 7), (2, 4, 4), (1, 5, 9), (0, 4, 4), (3, 2, 4), (3, 3, None)):
            s.check_both("path_max", u, v, want)
    s = Session(sol, [-1], [10**9])
    s.check("path_max", (0, 0), None)
    s = Session(sol, [1, -1], [5, 10**9])
    s.check_both("path_max", 0, 1, 5)


@case("Part 3", title="LCA 上方那条边不在路径上")
def p3_lca_edge(sol):
    weight = list(EX_WEIGHT)
    weight[0] = weight[4] = 100                # the edges 0 - 3 and 4 - 3 sit right above an LCA
    s = Session(sol, EX_PARENT, weight)
    for u, v, want in ((8, 2, 7), (1, 2, 7), (5, 7, 9), (5, 6, 9), (8, 7, 100)):
        s.check_both("path_max", u, v, want)
    r = rng()
    for shape in ("random", "binary", "caterpillar", "thin"):
        parent = ref.random_tree(r, 30, shape)
        dep = depths(parent)
        weight = [1000 - 10 * d + r.randrange(10) for d in dep]    # every edge is heavier than the ones below it
        check_all_pairs(Session(sol, parent, weight), parent, weight)


@case("Part 3", title="权重全是负数、有重复")
def p3_negative(sol):
    weight = [-w - 1 for w in EX_WEIGHT]
    s = Session(sol, EX_PARENT, weight)
    for u, v, want in ((8, 2, -2), (5, 7, -4), (6, 7, -6), (2, 4, -2)):
        s.check_both("path_max", u, v, want)
    r = rng()
    for shape, (lo, hi) in (("random", (-5, -1)), ("thin", (-10**9, -10**9 + 3)), ("binary", (-7, -7)),
                            ("path", (-10**9, -1))):
        parent = ref.random_tree(r, 25, shape)
        weight = [r.randint(lo, hi) for _ in parent]
        check_all_pairs(Session(sol, parent, weight), parent, weight)


@case("Part 3", title="weight 长度不对、没有 weight、编号越界")
def p3_errors(sol):
    ctor_raises(sol, ValueError, EX_PARENT, EX_WEIGHT[:-1], why="weight 少一个")
    ctor_raises(sol, ValueError, EX_PARENT, EX_WEIGHT + [0], why="weight 多一个")
    ctor_raises(sol, ValueError, [-1], [], why="weight 是空列表")
    ctor_raises(sol, ValueError, [1, -1], [5], why="weight 少一个")
    for pass_none in (False, True):
        s = Session(sol, EX_PARENT, pass_none=pass_none)
        for u, v in ((8, 2), (5, 7), (3, 0)):
            s.raises(ValueError, "path_max", (u, v))
    s = Session(sol, EX_PARENT, EX_WEIGHT)
    for u, v in ((-1, 0), (0, -1), (9, 0), (0, 9), (9, 9), (-1, -1)):
        s.raises(IndexError, "path_max", (u, v))


@case("Part 3", title="随机树：所有点对和暴力比较")
def p3_random(sol):
    r = rng()
    for shape in ref.SHAPES:
        for n in (2, 3, 5, 16, 17, 33):
            parent = ref.random_tree(r, n, shape)
            for lo, hi in ((-5, 5), (7, 7), (-10**9, 10**9)):
                weight = [r.randint(lo, hi) for _ in range(n)]
                check_all_pairs(Session(sol, parent, weight), parent, weight)


@case("Part 3", cpu=12, title="2 × 10⁵ 个节点的带权链，5 × 10⁴ 次查询")
def p3_large(sol):
    s = Session(sol, CHAIN, WCHAIN, code=WCHAIN_CODE)
    s.bulk("path_max", P3_U, P3_V, P3_WANT)
