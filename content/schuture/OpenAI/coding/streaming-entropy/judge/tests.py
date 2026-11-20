import math
import tracemalloc

import numpy as np

from judgelib import case, expect_close, expect_true, rng
import ref

# Correct float64 code differs from the correctly rounded entropy by a few ulps times the number of terms; inputs
# here have at most 200 000 entries and H <= log N < 13, so 1e-9 leaves many orders of magnitude of slack.
TOL = dict(rtol=1e-9, atol=1e-9)


def arr_code(x):
    x = np.asarray(x, dtype=np.float64)
    return f"np.array({x.tolist()!r})"


class Logits:
    """offset + scale * normal(size=n) from a seeded generator, optionally sorted, with code that rebuilds it."""

    def __init__(self, seed, n, scale, offset=0.0, order=None):
        x = offset + scale * np.random.default_rng(seed).normal(size=n)
        text = f"x = {offset!r} + {scale!r} * np.random.default_rng({seed}).normal(size={n})"
        if order == "asc":
            x, text = np.sort(x), text + "\nx = np.sort(x)"
        elif order == "desc":
            x, text = np.sort(x)[::-1].copy(), text + "\nx = np.sort(x)[::-1]"
        self.x, self.text = x, text


def random_logits(r, mild=False, n_max=60):
    seed, n = r.randrange(10 ** 6), r.randint(1, n_max)
    if mild:                                   # Part 1 must work here: no exp overflow, no p underflow
        return Logits(seed, n, r.choice([0.01, 0.5, 1.0, 3.0]), float(r.choice([0, 2, -5, 15])))
    scale = r.choice([1e-3, 1.0, 50.0, 500.0, 1e4, 1e8])
    offset = r.choice([0.0, 1e3, -1e3, 1e6, -1e12, 1e300])
    return Logits(seed, n, scale, offset, r.choice([None, None, "asc", "desc"]))


def blocks_of(x, block_size):
    return [x[s:s + block_size] for s in range(0, len(x), block_size)]


def one_shot(chunks):
    """A generator: each chunk can be read once, as in Part 4, and is a fresh copy."""
    return (c.copy() for c in chunks)


def traced(fn):
    tracemalloc.start()
    try:
        got = fn()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return got, peak


# ============================================================================== Part 1

@case("Part 1", sample=True, title="例子：[0.5, 2.5, -1.0] ≈ 0.4761")
def p1_example(sol):
    expect_close(sol.softmax_entropy(np.array([0.5, 2.5, -1.0])), 0.4761, rtol=0, atol=5e-5,
                 call="softmax_entropy(np.array([0.5, 2.5, -1.0]))", msg="应当约为 0.4761（题目保留四位小数）")


@case("Part 1", title="一个元素、全部相等")
def p1_degenerate(sol):
    for x, want, why in (([3.0], 0.0, "只有一个类别时 p = [1]，熵是 0"),
                         ([0.7] * 5, math.log(5), "logits 全相等时 p 是均匀分布，熵是 log 5"),
                         ([-2.0, -2.0], math.log(2), "logits 全相等时 p 是均匀分布，熵是 log 2")):
        x = np.array(x)
        expect_close(sol.softmax_entropy(x.copy()), want, call=f"softmax_entropy({arr_code(x)})", msg=why, **TOL)


@case("Part 1", title="随机的温和输入")
def p1_random(sol):
    r = rng()
    for _ in range(30):
        lg = random_logits(r, mild=True)
        expect_close(sol.softmax_entropy(lg.x.copy()), ref.entropy(lg.x), call=f"{lg.text}\nsoftmax_entropy(x)",
                     msg="熵 H = -Σ p_i log p_i（自然对数，返回非负值）", **TOL)


# ============================================================================== Part 2

@case("Part 2", sample=True, title="例子：[1000, 1002, 998] ≈ 0.4411")
def p2_example(sol):
    expect_close(sol.softmax_entropy_stable(np.array([1000.0, 1002.0, 998.0])), 0.4411, rtol=0, atol=5e-5,
                 call="softmax_entropy_stable(np.array([1000.0, 1002.0, 998.0]))",
                 msg="应当约为 0.4411（题目保留四位小数）")


@case("Part 2", title="温和输入上与第 1 部分一致")
def p2_matches_direct(sol):
    r = rng()
    for _ in range(30):
        lg = random_logits(r, mild=True)
        expect_close(sol.softmax_entropy_stable(lg.x.copy()), ref.entropy(lg.x),
                     call=f"{lg.text}\nsoftmax_entropy_stable(x)", **TOL)


@case("Part 2", title="很大的 logit：exp 不能溢出")
def p2_large(sol):
    for x in ([1e6, 1e6, 1e6], [0.0, 1000.0, -1000.0], [700.0, 710.0, 720.0, 730.0], [1e300], [5e299, 1e300]):
        x = np.array(x)
        expect_close(sol.softmax_entropy_stable(x.copy()), ref.entropy(x),
                     call=f"softmax_entropy_stable({arr_code(x)})",
                     msg="先减去最大的 logit 再取 exp，结果才不会溢出", **TOL)


@case("Part 2", title="p_i 下溢成 0 时不能是 nan")
def p2_underflow(sol):
    for x in ([0.0, -800.0, -900.0], [5.0, -1e5, -2e5, 4.0], [0.0, -745.2, -1e4, -0.5]):
        x = np.array(x)
        expect_close(sol.softmax_entropy_stable(x.copy()), ref.entropy(x),
                     call=f"softmax_entropy_stable({arr_code(x)})",
                     msg="log p_i 要按 (x_i - m) - log Σ e^(x_k - m) 计算，不能对下溢成 0 的 p_i 取 log", **TOL)


@case("Part 2", title="10 万个 logits：log p_i 不能加 ε")
def p2_many(sol):
    # log(p + eps) is off by about eps in every term, N * eps in all: 1e-7 here for eps = 1e-12, far outside TOL.
    seed, n = rng().randrange(10 ** 6), 100_000
    x = np.random.default_rng(seed).normal(size=n)
    expect_close(sol.softmax_entropy_stable(x.copy()), ref.entropy(x),
                 call=f"x = np.random.default_rng({seed}).normal(size={n})\nsoftmax_entropy_stable(x)",
                 msg="log p_i 要按 (x_i - m) - log Σ e^(x_k - m) 计算；写成 log(p_i + ε) 时每一项偏差约 ε，"
                     "N 很大时累积起来超出了舍入误差", **TOL)


@case("Part 2", title="随机输入（各种尺度和偏移）")
def p2_random(sol):
    r = rng()
    for _ in range(40):
        lg = random_logits(r)
        expect_close(sol.softmax_entropy_stable(lg.x.copy()), ref.entropy(lg.x),
                     call=f"{lg.text}\nsoftmax_entropy_stable(x)", **TOL)


# ============================================================================== Part 3

@case("Part 3", sample=True, title="例子：5 个 logits、block_size = 2")
def p3_example(sol):
    expect_close(sol.softmax_entropy_blockwise(np.array([1.5, -2.0, 4.0, -0.5, 3.0]), block_size=2), 0.8168,
                 rtol=0, atol=5e-5, call="softmax_entropy_blockwise(np.array([1.5, -2.0, 4.0, -0.5, 3.0]), block_size=2)",
                 msg="应当约为 0.8168（题目保留四位小数）")


@case("Part 3", title="最后一块可以更短")
def p3_partial_last(sol):
    for x, b in (([0.3, 9.0, -1.0, 2.0, 8.5, 7.0, 10.0], 3), ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 50.0], 7),
                 ([2.0, 1.0, 0.0, 6.0, -3.0], 4)):
        x = np.array(x)
        expect_close(sol.softmax_entropy_blockwise(x.copy(), b), ref.entropy(x),
                     call=f"softmax_entropy_blockwise({arr_code(x)}, {b})",
                     msg=f"block_size 不整除 N 时，最后一块只有 {len(x) % b} 个元素，也要算进去", **TOL)


@case("Part 3", title="不传 block_size（默认 2）")
def p3_default_block(sol):
    lg = Logits(rng().randrange(10 ** 6), 9, 2.0)
    expect_close(sol.softmax_entropy_blockwise(lg.x.copy()), ref.entropy(lg.x),
                 call=f"{lg.text}\nsoftmax_entropy_blockwise(x)", **TOL)


@case("Part 3", title="全是很负的 logits")
def p3_all_negative(sol):
    for x, b in (([-1000.0, -1001.0, -1003.0], 2), ([-5e5, -5e5 - 2.0, -5e5 + 1.0, -5e5 - 0.5], 3)):
        x = np.array(x)
        expect_close(sol.softmax_entropy_blockwise(x.copy(), b), ref.entropy(x),
                     call=f"softmax_entropy_blockwise({arr_code(x)}, {b})",
                     msg="所有 logit 都远小于 0：最大值要从 -inf（或第一块的最大值）开始，不能从 0 开始", **TOL)


@case("Part 3", title="随机输入、各种 block_size")
def p3_random(sol):
    r = rng()
    for _ in range(25):
        lg = random_logits(r)
        n = len(lg.x)
        for b in sorted({1, 2, 3, 7, n, n + 5}):
            expect_close(sol.softmax_entropy_blockwise(lg.x.copy(), b), ref.entropy(lg.x),
                         call=f"{lg.text}\nsoftmax_entropy_blockwise(x, {b})", **TOL)


N_BIG, BLOCK_BIG = 200_000, 1024


def check_blockwise_memory(sol, x, call, tol):
    want = ref.entropy(x.astype(np.float64))
    arg = x.copy()
    got, peak = traced(lambda: sol.softmax_entropy_blockwise(arg, BLOCK_BIG))
    expect_close(got, want, call=call, **tol)
    # A length-N array of this dtype is x.nbytes; the reference peaks at about 35 KB. Any array whose size depends
    # on N (a float64 copy of the input, the vector of probabilities, x - m for all of x) needs at least x.nbytes.
    limit = x.nbytes // 4
    expect_true(peak <= limit,
                f"调用期间分配内存的峰值是 {peak / 1e6:.2f} MB，而一个长度 N = {len(x)} 的 {x.dtype} 数组是 "
                f"{x.nbytes / 1e6:.2f} MB：题目要求不分配任何大小依赖 N 的数组（这个测试允许到 {limit / 1e6:.2f} MB）", call)


@case("Part 3", title="20 万个 float64：不分配长度 N 的数组")
def p3_memory_float64(sol):
    seed = rng().randrange(10 ** 6)
    x = np.random.default_rng(seed).normal(size=N_BIG)
    call = f"x = np.random.default_rng({seed}).normal(size={N_BIG})\nsoftmax_entropy_blockwise(x, {BLOCK_BIG})"
    check_blockwise_memory(sol, x, call, TOL)


@case("Part 3", title="20 万个 float32：也不能整体转成 float64")
def p3_memory_float32(sol):
    seed = rng().randrange(10 ** 6)
    x = np.random.default_rng(seed).normal(size=N_BIG).astype(np.float32)
    call = (f"x = np.random.default_rng({seed}).normal(size={N_BIG}).astype(np.float32)\n"
            f"softmax_entropy_blockwise(x, {BLOCK_BIG})")
    # The statement leaves the accumulation dtype open. All-float32 accumulation over these ~200 blocks was measured
    # at most 2.2e-7 relative (about 4 float32 epsilons) over 100 such inputs; 1e-5 leaves a wide margin.
    check_blockwise_memory(sol, x, call, dict(rtol=1e-5, atol=0))


# ============================================================================== Part 4

@case("Part 4", sample=True, title="例子：三个块依次到达 ≈ 0.8168")
def p4_example(sol):
    chunks = [np.array([1.5, -2.0]), np.array([4.0, -0.5]), np.array([3.0])]
    expect_close(sol.softmax_entropy_online(one_shot(chunks)), 0.8168, rtol=0, atol=5e-5,
                 call="softmax_entropy_online(c for c in [np.array([1.5, -2.0]), np.array([4.0, -0.5]), "
                      "np.array([3.0])])   # 生成器：每个块只能读一次",
                 msg="应当约为 0.8168（题目保留四位小数）")


def check_online(sol, chunks, call, msg=""):
    want = ref.entropy(np.concatenate(chunks))
    expect_close(sol.softmax_entropy_online(one_shot(chunks)), want, call=call, msg=msg, **TOL)


def cut(r, x, lo, hi):
    """Splits x into consecutive chunks of random lengths lo..hi (the last may be shorter); returns the chunks and
    code that rebuilds them from x."""
    seed, n = r.randrange(10 ** 6), len(x)
    k = n // lo + 1
    edges = np.cumsum(np.random.default_rng(seed).integers(lo, hi + 1, size=k))
    text = (f"edges = np.cumsum(np.random.default_rng({seed}).integers({lo}, {hi + 1}, size={k}))\n"
            f"chunks = np.split(x, edges[edges < {n}])")
    return np.split(x, edges[edges < n]), text


def chunks_code(chunks):
    return "chunks = [" + ", ".join(arr_code(c) for c in chunks) + "]"


@case("Part 4", title="只有一个块")
def p4_single_chunk(sol):
    for x in ([2.0], [0.5, 2.5, -1.0], [1000.0, 1002.0, 998.0]):
        x = np.array(x)
        check_online(sol, [x], f"softmax_entropy_online(iter([{arr_code(x)}]))")


@case("Part 4", title="最大值逐块变大或变小")
def p4_monotone_max(sol):
    r = rng()
    for order in ("asc", "desc"):
        for scale in (1.0, 300.0):
            lg = Logits(r.randrange(10 ** 6), 24, scale, 0.0, order)
            chunks = blocks_of(lg.x, 5)
            check_online(sol, chunks, f"{lg.text}\nsoftmax_entropy_online(x[s:s + 5] for s in range(0, len(x), 5))",
                         msg="新块的最大值变化时，已有的 s 和 u 都要按新的最大值重新缩放（u 还要加上 δ·s）")


@case("Part 4", title="全是很负的 logits")
def p4_all_negative(sol):
    chunks = [np.array([-1000.0, -1001.0]), np.array([-999.5]), np.array([-1003.0, -998.0, -1000.0])]
    check_online(sol, chunks, f"{chunks_code(chunks)}\nsoftmax_entropy_online(iter(chunks))",
                 msg="所有 logit 都远小于 0：running max 要从 -inf（或第一块的最大值）开始，不能从 0 开始")


@case("Part 4", title="随机长度的块、各种尺度")
def p4_random(sol):
    r = rng()
    for _ in range(30):
        lg = random_logits(r)
        max_len = r.choice([1, 2, 4, 10, 60])
        chunks, cut_text = cut(r, lg.x, 1, max_len)
        text = chunks_code(chunks) if len(chunks_code(chunks)) <= 600 else f"{lg.text}\n{cut_text}"
        check_online(sol, chunks, f"{text}\nsoftmax_entropy_online(iter(chunks))")


@case("Part 4", title="20 万个 logits 分块到达：不保存旧块")
def p4_memory(sol):
    r = rng()
    seed = r.randrange(10 ** 6)
    x = np.random.default_rng(seed).normal(size=N_BIG)
    chunks, cut_text = cut(r, x, 500, 1500)
    want = ref.entropy(x)
    call = (f"x = np.random.default_rng({seed}).normal(size={N_BIG})\n"
            f"{cut_text}   # 每段 500 到 1500 个（最后一段可能更短）\n"
            "softmax_entropy_online(c.copy() for c in chunks)   # 每个块是新分配的数组")
    gen = one_shot(chunks)
    got, peak = traced(lambda: sol.softmax_entropy_online(gen))
    expect_close(got, want, call=call, **TOL)
    # The reference peaks at about 50 KB (one chunk and its temporaries). Keeping every chunk, or concatenating
    # them, needs at least x.nbytes = 1.6 MB.
    limit = x.nbytes // 4
    expect_true(peak <= limit,
                f"调用期间分配内存的峰值是 {peak / 1e6:.2f} MB，而所有块加起来是 {x.nbytes / 1e6:.2f} MB：每个块只能读一次，"
                f"处理完就不能再保留（这个测试允许到 {limit / 1e6:.2f} MB）", call)
