"""Tests for weighted-sampling-generator.

Part 1 is deterministic: the rules fix every value, the moment of the TypeError and how far the input is read,
which a finite iterator that counts its pulls (Counting) makes visible. The TypeError message only has to contain
the index as a number; its wording is free. The unittest suite Part 1 asks for is not judged.

Parts 2 and 3 are random. Every frequency check uses the Chernoff bound in ref.py, so a correct sampler fails any
one check with probability below ref.ALPHA = 1e-10 whatever the seed (about 30 checks per run). The bugs targeted
here move some probability by 0.05 or more and fail by a wide margin. What the statement makes exact is checked
exactly: validation at call time, values outside `values` or with weight 0 never drawn, and (Part 2) the same rng
seed giving the same draws. EdgeRandom is a genuine random.Random whose random() also returns the two extreme
legal values 0.0 and 1 - 2**-53, which a correct sampler must survive without drawing a zero-weight value or
indexing past the end."""
import itertools
import math
import random
import re
from collections import Counter
from collections.abc import Iterator

from judgelib import Mismatch, calling, case, expect, expect_true, rng
import ref

# ---------------------------------------------------------------- Part 1 helpers

COUNTING_SRC = '''class Counting:
    # 一个有限迭代器，记录 iter_ints 已经从它拉取了多少个元素
    def __init__(self, items):
        self.items = list(items)
        self.pulled = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.pulled == len(self.items):
            raise StopIteration
        self.pulled += 1
        return self.items[self.pulled - 1]'''
_ns = {}
exec(COUNTING_SRC, _ns)
Counting = _ns["Counting"]

MYINT_SRC = "class MyInt(int):\n    pass"


class MyInt(int):
    pass


class _ReadTooFar(Exception):
    pass


class _Endless:
    """itertools.count() that stops the test, instead of running forever, once it has been read `cap` times."""

    def __init__(self, cap):
        self.cap, self.pulled = cap, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pulled >= self.cap:
            raise _ReadTooFar
        self.pulled += 1
        return self.pulled - 1


def _err(e):
    return f"{type(e).__name__}: {e}"


def _names_index(exc, index):
    return re.search(rf"(?<!\d){index}(?!\d)", str(exc)) is not None


def _start(sol, arg, call):
    calling(call)
    try:
        gen = sol.iter_ints(arg)
    except TypeError as e:
        raise Mismatch(f"调用 iter_ints(...) 时就抛出了 {_err(e)}：iter_ints 是生成器函数，要等生成器运行到"
                       "出错的元素（某一次 next() 时）才抛出", call) from None
    expect_true(isinstance(gen, Iterator), "iter_ints 应当返回一个生成器（可以对它调用 next()）", call, gen)
    return gen


def _next(gen, call, what):
    try:
        return next(gen)
    except StopIteration:
        raise Mismatch(f"{what}，生成器却已经结束了（StopIteration）", call) from None
    except TypeError as e:
        raise Mismatch(f"{what}，却抛出了 {_err(e)}", call) from None


def drain(sol, arg, call, cap, src=None):
    """Pulls iter_ints(arg) one value at a time: (values, the TypeError or None, src.pulled after each value)."""
    gen = _start(sol, arg, call)
    if src is not None and src.pulled:
        raise Mismatch(f"还没有调用 next()，就已经从输入里拉取了 {src.pulled} 个元素：iter_ints 必须惰性地读取，"
                       "第一次 next() 之前不应读取任何元素", call)
    out, pulls = [], []
    while len(out) <= cap:
        try:
            x = next(gen)
        except StopIteration:
            return out, None, pulls
        except TypeError as e:
            return out, e, pulls
        out.append(x)
        pulls.append(src.pulled if src is not None else None)
    raise Mismatch(f"产出的值比输入的元素还多（已经产出 {len(out)} 个）", call, None, out[:20])


def iter_text(expr, items, counting=False, prefix=""):
    """A script for 自定义运行 that runs iter_ints on `expr` (whose elements are `items`) and prints what came out,
    with the expected output in comments."""
    want, bad, positions = ref.ref_iter(items)
    lines = [prefix] if prefix else []
    if counting:
        lines += [COUNTING_SRC, f"src = Counting({expr})", "g = iter_ints(src)", "print(src.pulled)   # -> 0",
                  "out = []", "try:", "    for x in g:", "        out.append((x, src.pulled))   # (产出的值, 此时已拉取的元素个数)",
                  "except TypeError as e:"]
        if bad is None:
            lines.append('    print("TypeError:", e, "| 已拉取", src.pulled)   # 不应走到这里')
        else:
            lines.append(f'    print("TypeError:", e, "| 已拉取", src.pulled)   # 应当走到这里：消息里有下标 {bad}，已拉取 {bad + 1}')
        lines.append(f"print(out)   # -> {list(zip(want, [p + 1 for p in positions]))!r}")
    elif bad is None:
        lines.append(f"list(iter_ints({expr}))   # -> {want!r}")
    else:
        lines += [f"g = iter_ints({expr})", "out = []", "try:", "    for x in g:", "        out.append(x)",
                  "except TypeError as e:", f'    print("TypeError:", e)   # 应当走到这里，消息里要有下标 {bad}',
                  f"print(out)   # -> {want!r}"]
    return "\n".join(lines)


def check_iter(sol, items, call, counting=False, arg=None):
    """iter_ints over `items` (a list) must yield what the rules give and raise where they say. With counting=True
    the input is a Counting over items and every pull count is checked; `arg` passes another iterable instead."""
    want, bad, positions = ref.ref_iter(items)
    src = Counting(items) if counting else None
    if arg is None:
        arg = src if counting else list(items)
    got, err, pulls = drain(sol, arg, call, len(items), src)
    k = len(want)
    if bad is not None and len(got) > k and got[:k] == want:
        raise Mismatch(f"下标 {bad} 处的 {items[bad]!r} 不是合法元素（只接受 None 和 int 的实例；bool、float、str 等都要"
                       f"拒绝），产出 {want!r} 之后应当在这里抛出 TypeError，却把它当作值产出了", call, want, got)
    if err is not None and len(got) < k and got == want[:len(got)]:
        j = positions[len(got)]
        raise Mismatch(f"产出 {got!r} 之后抛出了 {_err(err)}，但下标 {j} 处的 {items[j]!r} 是合法元素（int 的实例，又不是 "
                       "bool），应当把它产出", call, want, got)
    if got != want:
        raise Mismatch("产出的值不对" + (f"：抛出 TypeError 之前应当先产出 {want!r}" if bad is not None else "")
                       + (f"（你的答案随后抛出了 {_err(err)}）" if err is not None else ""), call, want, got)
    if bad is None and err is not None:
        raise Mismatch(f"输入里没有非法元素（每个元素都是 None 或 int 的实例），不应抛出 TypeError，却在产出 {got!r} "
                       f"之后抛出了 {_err(err)}", call)
    if bad is not None:
        if err is None:
            raise Mismatch(f"下标 {bad} 处的 {items[bad]!r} 不是合法元素，产出 {want!r} 之后应当抛出 TypeError，"
                           "却正常结束了", call)
        expect_true(_names_index(err, bad), f"TypeError 的消息应当指出出错元素在输入中从 0 开始的下标 {bad}，"
                                            f"实际的消息是 {str(err)!r}", call)
    if counting:
        for i, (p, pos) in enumerate(zip(pulls, positions)):
            expect_true(p == pos + 1, f"产出第 {i + 1} 个值 {got[i]!r} 之后，已经从输入里拉取了 {p} 个元素，应当恰好是 "
                                      f"{pos + 1} 个：只能读到产出这个值的元素为止（途中的 None 一并读过），不能往后多读",
                        call)
        if bad is not None:
            expect_true(src.pulled == bad + 1, f"抛出 TypeError 时已经从输入里拉取了 {src.pulled} 个元素，应当恰好是 "
                                               f"{bad + 1} 个：出错元素之后的元素一个都不应被读取", call)


def check_lists(sol, lists, counting=False):
    for items in lists:
        check_iter(sol, items, iter_text(repr(items), items, counting), counting=counting)


# ---------------------------------------------------------------- Parts 2 and 3 helpers

EDGE_SRC = '''class EdgeRandom(random.Random):
    # 一个真正的 random.Random，只是 random() 依次返回 1 - 2**-53、正常的随机数、0.0，循环往复；
    # 0.0 和 1 - 2**-53 都是 random() 合法的返回值。重写 getrandbits 只是为了让 randrange、choice 照常工作
    def __init__(self, seed):
        super().__init__(seed)
        self.calls = 0
    def getrandbits(self, k):
        return super().getrandbits(k)
    def random(self):
        self.calls += 1
        r = super().random()
        return (0.0, 1 - 2 ** -53, r)[self.calls % 3]'''
_ns = {"random": random}
exec(EDGE_SRC, _ns)
EdgeRandom = _ns["EdgeRandom"]


def _num(x):
    if isinstance(x, float) and math.isnan(x):
        return "math.nan"
    if isinstance(x, float) and math.isinf(x):
        return "math.inf" if x > 0 else "-math.inf"
    return repr(x)


def _seq(xs):
    if isinstance(xs, list):
        return "[" + ", ".join(map(_num, xs)) + "]"
    if isinstance(xs, tuple):
        return "(" + ", ".join(map(_num, xs)) + ("," if len(xs) == 1 else "") + ")"
    return repr(xs)


def _imports(weights):
    finite = all(not isinstance(w, float) or math.isfinite(w) for w in weights)
    return "import random" if finite else "import math\nimport random"


def _copy(xs):
    return list(xs) if isinstance(xs, list) else xs


def _make(sol, kind, values, weights, rng_obj):
    if kind == "ws":
        return sol.weighted_sampler(_copy(values), _copy(weights), rng_obj)
    return sol.AliasSampler(_copy(values), _copy(weights), rng_obj)


def draw_text(kind, values, weights, rng_expr, n, note, how="sample", extra=""):
    head = _imports(weights) + ("\nimport itertools" if how == "iter" else "") + "\nfrom collections import Counter"
    if extra:
        head += "\n" + extra
    if kind == "ws":
        body = (f"g = weighted_sampler({_seq(values)}, {_seq(weights)}, {rng_expr})\n"
                f"draws = [next(g) for _ in range({n})]")
    else:
        draw = (f"list(itertools.islice(iter(sampler), {n}))" if how == "iter"
                else f"[sampler.sample() for _ in range({n})]")
        body = f"sampler = AliasSampler({_seq(values)}, {_seq(weights)}, {rng_expr})\ndraws = {draw}"
    return f"{head}\n{body}\nprint(Counter(draws))   # {note}"


def get_draws(sol, kind, values, weights, rng_obj, n, call, how="sample"):
    calling(call)
    s = _make(sol, kind, values, weights, rng_obj)
    if kind == "ws":
        expect_true(isinstance(s, Iterator), "weighted_sampler 应当返回一个生成器（可以对它反复调用 next()）", call, s)
        return [next(s) for _ in range(n)]
    if how == "iter":
        it = iter(s)
        expect_true(isinstance(it, Iterator), "iter(sampler) 应当返回一个生成器（可以对它反复调用 next()）", call, it)
        return list(itertools.islice(it, n))
    return [s.sample() for _ in range(n)]


def check_valid(draws, values, weights, call, hint="", weight_of=None):
    """Every draw is one of `values`, and never one whose weight is 0."""
    if weight_of is None:
        weight_of = dict(zip(values, weights))
    for i, x in enumerate(draws):
        try:
            w = weight_of.get(x)
        except TypeError:
            w = None
        if w is None:
            raise Mismatch(f"第 {i + 1} 次抽样得到 {x!r}，它不在 values 里{hint}", call, None, x)
        if w == 0:
            raise Mismatch(f"第 {i + 1} 次抽样得到 {x!r}，它的权重是 0，永远不应被抽到{hint}", call, None, x)


def _worst(cells, n, t):
    """The cell (label, count, probability) the check rejects most clearly, or None when all are accepted."""
    bad = [(n * ref.kl(c / n, q), label, c, q) for label, c, q in cells if not ref.accepts(c, n, q, t)]
    return max(bad, key=lambda b: b[0])[1:] if bad else None


def check_freq(draws, values, weights, call, pairs=False):
    """Valid draws whose counts agree with weights / sum(weights); with pairs=True, the counts of non-overlapping
    consecutive pairs must also agree with independent draws."""
    check_valid(draws, values, weights, call)
    n = len(draws)
    p = ref.probabilities(values, weights)
    live = [v for v in values if 0 < p[v] < 1]
    if not live:
        return
    counts = Counter(draws)
    t = ref.limit(len(live))
    worst = _worst([(v, counts[v], p[v]) for v in live], n, t)
    if worst:
        v, c, q = worst
        lo, hi = ref.count_range(n, q, t)
        raise Mismatch(f"{n} 次抽样中值 {v!r} 出现了 {c} 次；按权重它的概率是 {q:.4g}，应当出现 {lo}~{hi} 次（期望 "
                       f"{n * q:.1f} 次；正确的实现落在这个范围之外的概率小于 1e-10）",
                       call, {v: round(n * p[v], 1) for v in values}, {v: counts[v] for v in values})
    if pairs and len(live) >= 2:
        m = n // 2
        pc = Counter(zip(draws[0:2 * m:2], draws[1:2 * m:2]))
        cells = [((a, b), pc[(a, b)], p[a] * p[b]) for a in live for b in live]
        t2 = ref.limit(len(cells))
        worst = _worst(cells, m, t2)
        if worst:
            (a, b), c, q = worst
            lo, hi = ref.count_range(m, q, t2)
            raise Mismatch(f"把抽样结果两两分组（第 1、2 次，第 3、4 次……）共 {m} 组，({a!r}, {b!r}) 出现了 {c} 次；"
                           f"如果每次抽样都和之前的抽样相互独立，应当出现 {lo}~{hi} 次（期望 {m * q:.1f} 次）", call)


def freq_cases(sol, kind, scenarios, n, how="sample", pairs=False):
    r = rng()
    for values, weights, note in scenarios:
        seed = r.randrange(10 ** 6)
        call = draw_text(kind, values, weights, f"random.Random({seed})", n, note, how)
        draws = get_draws(sol, kind, values, weights, random.Random(seed), n, call, how)
        check_freq(draws, values, weights, call, pairs)


def random_scenarios(r):
    kinds = ref.KINDS + ["uniform"]
    r.shuffle(kinds)
    out = []
    for kind in kinds:
        k = r.randint(2, 12)
        out.append((r.sample(range(-100, 100), k), ref.random_weights(r, kind, k), "每个值的次数应接近 次数 × 权重 / 权重之和"))
    return out


def check_rejects(sol, kind, values, weights, why):
    name = "weighted_sampler" if kind == "ws" else "AliasSampler"
    when = "调用时" if kind == "ws" else "构造时"
    call = (f"{_imports(weights)}\n{name}({_seq(values)}, {_seq(weights)}, random.Random(0))   "
            f"# {why}：{when}应当立即抛出 ValueError")
    calling(call)
    try:
        out = _make(sol, kind, values, weights, random.Random(0))
    except ValueError:
        return
    except NotImplementedError:
        raise
    except Exception as e:
        raise Mismatch(f"{why}：应当抛出 ValueError，实际抛出了 {_err(e)}", call) from None
    later = ""
    if kind == "ws" and isinstance(out, Iterator):
        try:
            next(out)
        except ValueError:
            later = ("；直到第一次 next() 才抛出 ValueError，但校验必须在调用的那一刻完成（生成器函数的函数体要等到"
                     "第一次 next() 才开始执行）")
        except Exception:
            pass
    raise Mismatch(f"{why}：{when}应当立即抛出 ValueError，却正常返回了{later}", call, None, out)


INVALID = [
    ([1, 2], [1.0], "values 和 weights 长度不等"),
    ([1], [1.0, 2.0], "values 和 weights 长度不等"),
    ([], [], "长度为 0"),
    ([1, 2], [1.0, -1.0], "有负的权重"),
    ([1, 2, 3], [2.0, -0.5, 2.0], "有负的权重（即使总和是正的）"),
    ([1, 2], [1.0, math.inf], "有权重是 inf"),
    ([1, 2], [-math.inf, 1.0], "有权重是 -inf"),
    ([1, 2], [math.nan, 1.0], "有权重是 nan"),
    ([1, 2], [0.0, 0.0], "权重之和为 0"),
    ([1], [0.0], "权重之和为 0"),
]
OVERFLOW = [
    ([1, 2], [1e308, 1e308], "每个权重都有限，但浮点数之和溢出成了 inf"),
    ([1, 2, 3], [9e307, 9e307, 0.0], "每个权重都有限，但浮点数之和溢出成了 inf"),
    ([1, 2, 3, 4], [5e307, 5e307, 5e307, 5e307], "每个权重都有限，但浮点数之和溢出成了 inf"),
]
VALID = [
    ([7], [5.0], "只有一个值"),
    ([1, 2], [0.0, 3.0], "权重可以为 0"),
    ([1, 2, 3], [-0.0, 1.0, 0.0], "-0.0 也满足 >= 0"),
    ([1, 2], [1e308, 1e307], "权重很大，但和仍然有限"),
    ((4, 5, 6), (1.0, 2.0, 3.0), "values 和 weights 是元组"),
    (range(3), [1.0, 1.0, 2.0], "values 是 range"),
]
ZERO = [
    ([10, 20, 30, 40], [0.0, 2.0, 0.0, 1.0], 20000),
    ([1, 2, 3, 4], [0.0, 0.0, 1.0, 0.0], 5000),
    ([7, 8], [3.0, 0.0], 5000),
]
EXAMPLE = ([1, 2, 3], [0.1, 0.1, 0.8], "1 约占 10%，2 约占 10%，3 约占 80%")
UNNORMALISED = [
    ([1, 2], [2.0, 6.0], "1 约占 25%，2 约占 75%"),
    ([5, 6, 7], [3.0, 1.0, 4.0], "5 约占 37.5%，6 约占 12.5%，7 约占 50%"),
]
EXTREME = [
    ([1, 2, 3], [6e307, 6e307, 1.0], "1、2 各约占 50%，3 几乎不出现"),
    ([1, 2, 3, 4], [5e307, 5e307, 5e307, 1e307], "1、2、3 各约占 31.25%，4 约占 6.25%"),
    ([1, 2], [1e-300, 3e-300], "1 约占 25%，2 约占 75%"),
    ([1, 2, 3], [1e-3, 1.0, 1e3], "3 约占 99.9%，2 约占 0.1%，1 约占百万分之一"),
]
EDGE = [
    ([5, 6, 7, 8], [0.0, 1.0, 2.0, 0.0]),
    (list(range(10)), [0.1] * 10),
    ([1, 2, 3], [1.0, 0.0, 1.0]),
    ([4, 5, 6, 7, 8], [0.0, 0.0, 3.0, 0.0, 0.5]),
    ([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.0]),
]
EDGE_HINT = "（这个 rng 的 random() 会返回 0.0 和 1 - 2**-53，这两个都是 random() 合法的返回值）"


def valid_cases(sol, kind, n=300):
    r = rng()
    for values, weights, why in VALID:
        seed = r.randrange(10 ** 6)
        call = draw_text(kind, values, weights, f"random.Random({seed})", n, f"{why}：合法输入，不应抛出异常")
        try:
            draws = get_draws(sol, kind, values, weights, random.Random(seed), n, call)
        except ValueError as e:
            raise Mismatch(f"{why}，这是合法的输入，却抛出了 {_err(e)}", call) from None
        check_valid(draws, values, weights, call)


def zero_cases(sol, kind):
    r = rng()
    for values, weights, n in ZERO:
        seed = r.randrange(10 ** 6)
        live = [v for v, w in zip(values, weights) if w > 0]
        call = draw_text(kind, values, weights, f"random.Random({seed})", n, f"只能出现 {live}")
        check_valid(get_draws(sol, kind, values, weights, random.Random(seed), n, call), values, weights, call)


def edge_cases(sol, kind, n=600):
    r = rng()
    for values, weights in EDGE:
        seed = r.randrange(10 ** 6)
        live = [v for v, w in zip(values, weights) if w > 0]
        call = draw_text(kind, values, weights, f"EdgeRandom({seed})", n, f"只能出现 {live}", extra=EDGE_SRC)
        check_valid(get_draws(sol, kind, values, weights, EdgeRandom(seed), n, call), values, weights, call, EDGE_HINT)


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子：跳过 None")
def p1_example(sol):
    call = "list(iter_ints([3, None, 0, -2]))   # -> [3, 0, -2]"
    calling(call)
    expect(list(sol.iter_ints([3, None, 0, -2])), [3, 0, -2], call, "None 要跳过，0 和 -2 是普通的值")


@case("Part 1", sample=True, title="题目例子：遇到 True 抛 TypeError")
def p1_example_error(sol):
    call = ("g = iter_ints([1, 2, True])\nnext(g)   # -> 1\nnext(g)   # -> 2\n"
            "next(g)   # 抛出 TypeError，消息中提到下标 2")
    g = _start(sol, [1, 2, True], call)
    expect(_next(g, call, "第一次 next(g) 应返回 1"), 1, call, "第一次 next(g) 应返回 1")
    expect(_next(g, call, "第二次 next(g) 应返回 2"), 2, call, "第二次 next(g) 应返回 2")
    try:
        x = next(g)
    except TypeError as e:
        expect_true(_names_index(e, 2), f"TypeError 的消息中应当提到下标 2，实际的消息是 {str(e)!r}", call)
        return
    except StopIteration:
        raise Mismatch("第三次 next(g) 遇到 True（bool 不算 int），应当抛出 TypeError，生成器却结束了", call) from None
    raise Mismatch("第三次 next(g) 遇到 True（bool 不算 int），应当抛出 TypeError，却返回了一个值", call, None, x)


@case("Part 1", title="items 为 None")
def p1_none_input(sol):
    call = "list(iter_ints(None))   # -> []"
    expect(list(_start(sol, None, call)), [], call, "items is None 等同于空输入，不产出任何值")


@case("Part 1", title="空输入")
def p1_empty(sol):
    check_iter(sol, [], iter_text("[]", []))
    check_iter(sol, [], iter_text("()", []), arg=())
    check_iter(sol, [], iter_text("iter([])", []), arg=iter([]))


@case("Part 1", title="None 在开头、中间、末尾和全是 None")
def p1_none_positions(sol):
    check_lists(sol, [[None, 1, 2], [1, None, 2], [1, 2, None], [None, None, None], [None], [None, 5, None, None, 6, None]])


@case("Part 1", title="0、负数和大整数是普通的值")
def p1_zero_negative(sol):
    check_lists(sol, [[0, -5, 3, 0], [0], [0, 0, 0], [-1, None, -10 ** 30, 10 ** 30], [None, 0, None]])


@case("Part 1", title="int 子类的实例也是 int")
def p1_int_subclass(sol):
    items = [MyInt(7), None, 8, MyInt(-1)]
    check_iter(sol, items, iter_text("[MyInt(7), None, 8, MyInt(-1)]", items, prefix=MYINT_SRC)
               + "（MyInt 是 int 的实例，又不是 bool）")


@case("Part 1", title="拒绝 bool，消息里有下标")
def p1_bool_rejected(sol):
    check_lists(sol, [[True], [False], [5, -1, False, 4], [0, 1, True]])


@case("Part 1", title="拒绝 float、str 等其他类型")
def p1_float_str_rejected(sol):
    check_lists(sol, [[3.0], ["3"], [1, 2.5, 3], [0, "x", 7], [b"1"], [[1]], [4, 1 + 0j]])


@case("Part 1", title="下标要把 None 也算上")
def p1_index_counts_none(sol):
    check_lists(sol, [[None, None, 8, None, "x", 9], [None, True], [None, None, None, 2.5], [6, None, 7, None, False]])


@case("Part 1", title="惰性：每次只读到需要的元素")
def p1_lazy_pulls(sol):
    check_lists(sol, [[None, 10, None, None, 20, 30, None], [7, 8, 9], [None, None, 0]], counting=True)


@case("Part 1", title="惰性：出错元素之后不再读取")
def p1_lazy_error(sol):
    check_lists(sol, [[4, None, "x", 5, 6], [None, None, 2.5, 1], [True, 1, 2]], counting=True)


@case("Part 1", title="无穷输入 itertools.count()")
def p1_infinite(sol):
    call = "import itertools\nlist(itertools.islice(iter_ints(itertools.count()), 5))   # -> [0, 1, 2, 3, 4]"
    src = _Endless(1000)
    gen = _start(sol, src, call)
    try:
        got = list(itertools.islice(gen, 5))
    except _ReadTooFar:
        raise Mismatch("从无穷的输入里读了 1000 个元素还没有产出前 5 个值：iter_ints 必须惰性地读取，在 itertools.count() "
                       "这样的无穷输入上也要能用（测试用的是一个读到第 1000 个元素就报错的 itertools.count() 替身）",
                       call) from None
    expect(got, [0, 1, 2, 3, 4], call)
    expect_true(src.pulled == 5, f"产出前 5 个值之后已经读取了 {src.pulled} 个元素，应当恰好是 5 个", call)


@case("Part 1", title="元组、range、dict、生成器作输入")
def p1_other_iterables(sol):
    check_iter(sol, [3, None, 4], iter_text("(3, None, 4)", [3, None, 4]), arg=(3, None, 4))
    check_iter(sol, list(range(-2, 3)), iter_text("range(-2, 3)", list(range(-2, 3))), arg=range(-2, 3))
    check_iter(sol, [1, None, 3], iter_text('{1: "a", None: "b", 3: "c"}', [1, None, 3]),
               arg={1: "a", None: "b", 3: "c"})
    check_iter(sol, [None, 1, None, 2], iter_text("(x for x in [None, 1, None, 2])", [None, 1, None, 2]),
               arg=(x for x in [None, 1, None, 2]))
    check_iter(sol, [1, "a", 2], iter_text('(x for x in [1, "a", 2])', [1, "a", 2]), arg=(x for x in [1, "a", 2]))


@case("Part 1", title="不修改传入的列表")
def p1_input_unchanged(sol):
    for items in ([3, None, 0, -2, None], [1, None, "x", 2]):
        before = list(items)
        call = iter_text("items", before, prefix=f"items = {before!r}") + f"\nprint(items)   # -> {before!r}，应当没有变化"
        drain(sol, items, call, len(before))
        expect(items, before, call, "iter_ints 不应修改传入的 items")


@case("Part 1", title="两个生成器互不影响")
def p1_independent(sol):
    call = ("items = [1, None, 2, 3]\ng1, g2 = iter_ints(items), iter_ints(items)\n"
            "next(g1)   # -> 1\nnext(g2)   # -> 1\nnext(g1)   # -> 2\nlist(g2)   # -> [2, 3]\nlist(g1)   # -> [3]")
    items = [1, None, 2, 3]
    g1, g2 = _start(sol, items, call), _start(sol, items, call)
    expect(_next(g1, call, "第一次 next(g1) 应返回 1"), 1, call, "第一次 next(g1) 应返回 1")
    expect(_next(g2, call, "第一次 next(g2) 应返回 1"), 1, call, "第一次 next(g2) 应返回 1：g2 不受 g1 的进度影响")
    expect(_next(g1, call, "第二次 next(g1) 应返回 2"), 2, call, "第二次 next(g1) 应返回 2")
    expect(list(g2), [2, 3], call, "list(g2) 应为 [2, 3]")
    expect(list(g1), [3], call, "list(g1) 应为 [3]")


@case("Part 1", title="随机输入，对照暴力")
def p1_random(sol):
    r = rng()
    for t in range(150):
        items = ref.random_items(r)
        counting = t % 2 == 1
        check_iter(sol, items, iter_text(repr(items), items, counting), counting=counting)


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子：抽样都取自 values")
def p2_example(sol):
    call = ("import random\ng = weighted_sampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random(0))\n"
            "[next(g) for _ in range(10)]   # -> 十个取自 {1, 2, 3} 的值")
    got = get_draws(sol, "ws", [1, 2, 3], [0.1, 0.1, 0.8], random.Random(0), 10, call)
    check_valid(got, [1, 2, 3], [0.1, 0.1, 0.8], call)


@case("Part 2", sample=True, title="题目例子：长度不等立即报错")
def p2_length_mismatch(sol):
    check_rejects(sol, "ws", [1, 2], [1.0], "长度不等")


@case("Part 2", sample=True, title="题目例子：权重之和不必为 1")
def p2_unnormalised(sol):
    call = ("import random\ng = weighted_sampler([1, 2], [2.0, 6.0], random.Random(0))   # 合法：权重之和不必为 1\n"
            "[next(g) for _ in range(10)]   # -> 十个取自 {1, 2} 的值")
    try:
        got = get_draws(sol, "ws", [1, 2], [2.0, 6.0], random.Random(0), 10, call)
    except ValueError as e:
        raise Mismatch(f"权重之和不必为 1，这是合法的输入，却抛出了 {_err(e)}", call) from None
    check_valid(got, [1, 2], [2.0, 6.0], call)


@case("Part 2", title="非法输入：调用时立即 ValueError")
def p2_invalid_inputs(sol):
    for values, weights, why in INVALID:
        check_rejects(sol, "ws", values, weights, why)


@case("Part 2", title="权重之和溢出成 inf")
def p2_sum_overflow(sol):
    for values, weights, why in OVERFLOW:
        check_rejects(sol, "ws", values, weights, why)


@case("Part 2", title="合法的特殊输入")
def p2_valid_inputs(sol):
    valid_cases(sol, "ws")


@case("Part 2", title="权重为 0 的值永远不出现")
def p2_zero_weight_never(sol):
    zero_cases(sol, "ws")


@case("Part 2", title="频率检验：题目例子，且相互独立")
def p2_freq_example(sol):
    freq_cases(sol, "ws", [EXAMPLE], 100000, pairs=True)


@case("Part 2", title="频率检验：权重之和不为 1")
def p2_freq_unnormalised(sol):
    freq_cases(sol, "ws", UNNORMALISED, 60000)


@case("Part 2", title="频率检验：随机权重")
def p2_freq_random(sol):
    freq_cases(sol, "ws", random_scenarios(rng("weights")), 40000)


@case("Part 2", title="频率检验：极大、极小和悬殊的权重")
def p2_freq_extreme(sol):
    freq_cases(sol, "ws", EXTREME, 60000)


@case("Part 2", title="只用 rng：同一种子得到同一序列")
def p2_same_seed(sol):
    r = rng()
    s1, s2 = r.sample(range(1, 10 ** 6), 2)
    make = "weighted_sampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random({}))"
    call = (f"import random\ng1 = {make.format(s1)}\nrandom.seed(1)   # 改变全局 random 的状态：答案不应该用到它\n"
            f"a = [next(g1) for _ in range(200)]\ng2 = {make.format(s1)}\nrandom.seed(2)\n"
            f"b = [next(g2) for _ in range(200)]\nprint(a == b)   # -> True")
    state = random.getstate()
    try:
        calling(call)
        g1 = sol.weighted_sampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random(s1))
        random.seed(1)
        a = [next(g1) for _ in range(200)]
        g2 = sol.weighted_sampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random(s1))
        random.seed(2)
        b = [next(g2) for _ in range(200)]
    finally:
        random.setstate(state)
    expect(b, a, call, "两次都传入 random.Random(同一个种子)，却得到了不同的抽样序列：每一个随机选择都只能来自传入的 rng，"
                       "不能用全局的 random 模块或其他随机来源")
    call = (f"import random\ng1 = {make.format(s1)}\ng3 = {make.format(s2)}\n"
            "print([next(g1) for _ in range(200)] == [next(g3) for _ in range(200)])   # -> False")
    calling(call)
    g3 = sol.weighted_sampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random(s2))
    c = [next(g3) for _ in range(200)]
    expect_true(c != a, "换了 rng 的种子，前 200 次抽样却完全一样：抽样结果应当来自传入的 rng", call)


@case("Part 2", title="rng 返回 0.0 或最接近 1 的数")
def p2_edge_rng(sol):
    edge_cases(sol, "ws")


# ---------------------------------------------------------------- Part 3

BIG_N, BIG_DRAWS, BIG_GROUPS = 200_000, 20_000, 10
BIG_SRC = f"""r = random.Random({rng("big").randrange(10 ** 6)})
values = list(range(10 ** 6, 10 ** 6 + {BIG_N}))
weights = [0.0 if r.random() < 0.1 else round(r.uniform(0.1, 5.0), 3) for _ in values]   # 约 10% 的权重为 0"""
_ns = {"random": random}
exec(BIG_SRC, _ns)
BIG_VALUES, BIG_WEIGHTS = _ns["values"], _ns["weights"]
BIG_WEIGHT_OF = dict(zip(BIG_VALUES, BIG_WEIGHTS))
_size = BIG_N // BIG_GROUPS
BIG_GROUP_P = [math.fsum(BIG_WEIGHTS[g * _size:(g + 1) * _size]) / math.fsum(BIG_WEIGHTS) for g in range(BIG_GROUPS)]


@case("Part 3", sample=True, title="题目例子：权重全为 0")
def p3_zero_sum(sol):
    check_rejects(sol, "alias", [1, 2], [0.0, 0.0], "权重之和为 0")


@case("Part 3", sample=True, title="题目例子：sample() 与 iter()")
def p3_example(sol):
    call = ("import random\nsampler = AliasSampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random(0))\n"
            "sampler.sample()      # -> 1、2 或 3 中的一个\nnext(iter(sampler))   # -> 同样是 1、2 或 3 中的一个")
    calling(call)
    s = sol.AliasSampler([1, 2, 3], [0.1, 0.1, 0.8], random.Random(0))
    check_valid([s.sample()], [1, 2, 3], [0.1, 0.1, 0.8], call)
    it = iter(s)
    expect_true(isinstance(it, Iterator), "iter(sampler) 应当返回一个生成器（可以对它调用 next()）", call, it)
    check_valid([next(it)], [1, 2, 3], [0.1, 0.1, 0.8], call)


@case("Part 3", title="非法输入：构造时立即 ValueError")
def p3_invalid_inputs(sol):
    for values, weights, why in INVALID + OVERFLOW:
        check_rejects(sol, "alias", values, weights, why)


@case("Part 3", title="合法的特殊输入")
def p3_valid_inputs(sol):
    valid_cases(sol, "alias")


@case("Part 3", title="权重为 0 的值永远不出现")
def p3_zero_weight_never(sol):
    zero_cases(sol, "alias")


@case("Part 3", title="频率检验：题目例子，且相互独立")
def p3_freq_example(sol):
    freq_cases(sol, "alias", [EXAMPLE], 100000, pairs=True)


@case("Part 3", title="频率检验：经由 __iter__")
def p3_freq_iter(sol):
    freq_cases(sol, "alias", [EXAMPLE] + UNNORMALISED[:1], 60000, how="iter")


@case("Part 3", title="频率检验：随机权重")
def p3_freq_random(sol):
    freq_cases(sol, "alias", random_scenarios(rng("weights")), 40000)


@case("Part 3", title="频率检验：极大、极小和悬殊的权重")
def p3_huge_weights(sol):
    freq_cases(sol, "alias", EXTREME, 60000)


@case("Part 3", title="rng 返回 0.0 或最接近 1 的数")
def p3_edge_rng(sol):
    edge_cases(sol, "alias")


@case("Part 3", cpu=2, title="2×10⁵ 个值：每次抽样 O(1)")
def p3_big(sol):
    seed = rng().randrange(10 ** 6)
    call = (f"import random\n{BIG_SRC}\nsampler = AliasSampler(values, weights, random.Random({seed}))\n"
            f"draws = [sampler.sample() for _ in range({BIG_DRAWS})]   # 每次抽样应为 O(1)，与 n 无关")
    calling(call)
    s = sol.AliasSampler(list(BIG_VALUES), list(BIG_WEIGHTS), random.Random(seed))
    draws = [s.sample() for _ in range(BIG_DRAWS)]
    check_valid(draws, BIG_VALUES, BIG_WEIGHTS, call, weight_of=BIG_WEIGHT_OF)
    counts = Counter((x - 10 ** 6) // _size for x in draws)
    t = ref.limit(BIG_GROUPS)
    worst = _worst([(g, counts[g], q) for g, q in enumerate(BIG_GROUP_P)], BIG_DRAWS, t)
    if worst:
        g, c, q = worst
        lo, hi = ref.count_range(BIG_DRAWS, q, t)
        raise Mismatch(f"把 values 按位置分成 {BIG_GROUPS} 段，第 {g + 1} 段（values[{g * _size}:{(g + 1) * _size}]）"
                       f"在 {BIG_DRAWS} 次抽样中出现了 {c} 次，按权重应当出现 {lo}~{hi} 次（期望 {BIG_DRAWS * q:.1f} 次）",
                       call)
