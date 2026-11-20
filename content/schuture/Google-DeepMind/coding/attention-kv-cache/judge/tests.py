import numbers
import tracemalloc

import numpy as np

from judgelib import Mismatch, case, expect_close, expect_raises, expect_true, rng, show_call
import ref

# The statement prints its worked examples to 4 decimals, so a correct value is within 5e-5 of each.
EXAMPLE_TOL = dict(rtol=0.0, atol=5e-5)
# A correct float64 answer differs from the reference (explicit loops, fsum) only by rounding: about 1e-15 for the
# O(1) values below. Every mistake these tests target moves some output by 1e-3 or more.
TOL = dict(rtol=1e-9, atol=1e-9)
# Part 2: the statement's own bound between decode_step and causal_mha.
DECODE_TOL = dict(rtol=0.0, atol=1e-10)


def seeds(salt, k):
    r = rng(salt)
    return [r.getrandbits(32) for _ in range(k)]


def arr(x, label, call):
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        raise Mismatch(f"{label} 不是数值数组：{type(x).__name__}", call, None, x) from None


def random_problem(seed, T, d):
    gen = np.random.default_rng(seed)
    x = gen.normal(size=(T, d)) * 0.7
    Ws = [gen.normal(size=(d, d)) * 0.6 for _ in range(4)]
    setup = (f"rng = np.random.default_rng({seed})\nx = rng.normal(size=({T}, {d})) * 0.7\n"
             f"Wq, Wk, Wv, Wo = (rng.normal(size=({d}, {d})) * 0.6 for _ in range(4))\n")
    return x, Ws, setup


def copies(Ws):
    return [W.copy() for W in Ws]


def lit(a):
    return f"np.array({np.asarray(a).tolist()!r})"


# ---------------------------------------------------------------- Part 1: causal_mha

EX_X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
EX_ROWS = [[1.0, 0.0], [0.3302, 0.6698], [0.7517, 0.7517]]
EX_SETUP = "x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])\nI2 = np.eye(2)\n"


@case("Part 1", sample=True, title="题目例子：单头、单位矩阵")
def p1_example(sol):
    call = EX_SETUP + "causal_mha(x, I2, I2, I2, I2, n_heads=1)"
    I2 = np.eye(2)
    expect_close(sol.causal_mha(EX_X.copy(), I2.copy(), I2.copy(), I2.copy(), I2.copy(), 1), EX_ROWS, call=call,
                 **EXAMPLE_TOL)


# (T, n_heads, d_head): one position, one feature per head, many heads, a longer sequence.
P1_SHAPES = [(1, 1, 1), (1, 3, 2), (2, 2, 1), (5, 2, 3), (7, 4, 1), (8, 6, 2), (6, 3, 3), (4, 1, 5), (40, 3, 4)]


@case("Part 1", title="随机输入，1 到 6 个头")
def p1_random(sol):
    for seed, (T, h, d_h) in zip(seeds("p1", len(P1_SHAPES)), P1_SHAPES):
        x, Ws, setup = random_problem(seed, T, h * d_h)
        want = ref.causal_mha_slow(x, *Ws, h)
        call = setup + f"causal_mha(x, Wq, Wk, Wv, Wo, n_heads={h})   # d_h = {d_h}"
        expect_close(sol.causal_mha(x.copy(), *copies(Ws), h), want, call=call, **TOL)


@case("Part 1", title="得分上千：要先减去每行最大值")
def p1_large_scores(sol):
    # Integer inputs and identity weights keep every score exact (d_h = 4, so dividing by sqrt(d_h) = 2 is exact);
    # row 0 attends only to itself with a scaled score of 2679, where an unshifted exp overflows.
    gen = np.random.default_rng(seeds("p1 large", 1)[0])
    x = np.vstack([[40, -35, 38, 33, 39, -40, 36, 37], gen.integers(-40, 41, size=(5, 8))]).astype(float)
    I8 = np.eye(8)
    want = ref.causal_mha_slow(x, I8, I8, I8, I8, 2)
    call = f"x = {lit(x)}\nI8 = np.eye(8)\n" \
           f"causal_mha(x, I8, I8, I8, I8, n_heads=2)   # 除以 sqrt(d_h) 之后的得分最高到 {int((x[:, :4] ** 2).sum(1).max() / 2)}"
    expect_close(sol.causal_mha(x.copy(), I8.copy(), I8.copy(), I8.copy(), I8.copy(), 2), want, call=call,
                 rtol=1e-12, atol=1e-9)


@case("Part 1", title="d 不能被 n_heads 整除：ValueError")
def p1_value_error(sol):
    for T, d, h in ((2, 3, 2), (4, 6, 4), (1, 10, 3), (3, 4, 8)):
        I = np.eye(d)
        call = f"causal_mha(np.zeros(({T}, {d})), *[np.eye({d})] * 4, n_heads={h})"
        expect_raises(ValueError, sol.causal_mha, np.zeros((T, d)), I.copy(), I.copy(), I.copy(), I.copy(), h,
                      call=call)


# ---------------------------------------------------------------- Part 2: KVCache, decode_step, kv_cache_bytes

def decode_all(sol, x, Ws, h, want, setup, cache=None):
    """Feed x row by row through decode_step on a fresh cache; check every row and len(cache) after every step."""
    T, d = x.shape
    if cache is None:
        cache = sol.KVCache(n_heads=h, d_head=d // h)
    start = f"cache = KVCache(n_heads={h}, d_head={d // h})\n"
    expect_true(len(cache) == 0, f"新建的 cache 应当是空的，len(cache) 却是 {len(cache)}", setup + start + "len(cache)")
    for t in range(T):
        call = (setup + start + f"outs = [decode_step(x[t], cache, Wq, Wk, Wv, Wo, n_heads={h}) for t in range({t + 1})]\n"
                f"outs[{t}]   # 应当等于 causal_mha(x, Wq, Wk, Wv, Wo, n_heads={h})[{t}]")
        got = sol.decode_step(x[t].copy(), cache, *copies(Ws), h)
        expect_close(got, want[t], call=call, **DECODE_TOL)
        expect_true(len(cache) == t + 1, f"第 {t} 步之后 cache 里应当有 {t + 1} 个位置，len(cache) 却是 {len(cache)}",
                    call.rsplit("\n", 1)[0] + "\nlen(cache)")


@case("Part 2", sample=True, title="题目例子：逐行复现 Part 1")
def p2_example(sol):
    I2 = np.eye(2)
    cache = sol.KVCache(n_heads=1, d_head=2)
    start = EX_SETUP + "cache = KVCache(n_heads=1, d_head=2)\n"
    expect_true(len(cache) == 0, f"新建的 cache 应当是空的，len(cache) 却是 {len(cache)}", start + "len(cache)")
    for t in range(3):
        call = start + "".join(f"decode_step(x[{s}], cache, I2, I2, I2, I2, n_heads=1)\n" for s in range(t + 1))
        got = sol.decode_step(EX_X[t].copy(), cache, I2.copy(), I2.copy(), I2.copy(), I2.copy(), 1)
        expect_close(got, EX_ROWS[t], call=call.rstrip(), **EXAMPLE_TOL)
        expect_true(len(cache) == t + 1, f"第 {t} 步之后 len(cache) 应当是 {t + 1}，得到 {len(cache)}", call + "len(cache)")


@case("Part 2", sample=True, title="题目例子：4 GiB 的 cache")
def p2_bytes_example(sol):
    call = "kv_cache_bytes(32, 8, 128, 32_768, 1, 2)"
    got = sol.kv_cache_bytes(32, 8, 128, 32_768, 1, 2)
    expect_true(isinstance(got, numbers.Number) and not isinstance(got, bool) and got == 2 ** 32,
                f"应当恰好是 2**32 = {2 ** 32}，得到 {got!r}", call, got)


@case("Part 2", title="两个 cache 同时解码，互不影响")
def p2_two_caches(sol):
    h, d_h, T = 2, 2, 6
    sa, sb = seeds("two caches", 2)
    xa, Ws, setup_a = random_problem(sa, T, h * d_h)
    xb = np.random.default_rng(sb).normal(size=(T, h * d_h)) * 0.7
    want_a, want_b = ref.causal_mha_slow(xa, *Ws, h), ref.causal_mha_slow(xb, *Ws, h)
    setup = (setup_a + f"xb = np.random.default_rng({sb}).normal(size=({T}, {h * d_h})) * 0.7\n"
             f"a = KVCache(n_heads={h}, d_head={d_h})\nb = KVCache(n_heads={h}, d_head={d_h})\n")
    a, b = sol.KVCache(n_heads=h, d_head=d_h), sol.KVCache(n_heads=h, d_head=d_h)
    for name, cache in (("a", a), ("b", b)):
        expect_true(len(cache) == 0, f"新建的 cache 应当是空的，len({name}) 却是 {len(cache)}", setup + f"len({name})")
    step = "decode_step({}[{}], {}, Wq, Wk, Wv, Wo, n_heads=" + str(h) + ")"
    for t in range(T):
        history = setup + (f"for s in range({t}):\n    {step.format('x', 's', 'a')}\n    {step.format('xb', 's', 'b')}\n"
                           if t else "")
        for name, cache, x, want in (("a", a, xa, want_a), ("b", b, xb, want_b)):
            xs = "x" if name == "a" else "xb"
            call = history + f"{step.format(xs, t, name)}   # 应当等于 causal_mha({xs}, Wq, Wk, Wv, Wo, n_heads={h})[{t}]"
            expect_close(sol.decode_step(x[t].copy(), cache, *copies(Ws), h), want[t], call=call, **DECODE_TOL)
            expect_true(len(cache) == t + 1, f"len({name}) 应当是 {t + 1}，得到 {len(cache)}：每个 cache 只记自己的位置",
                        call.split("   #")[0] + f"\nlen({name})")
            history += step.format(xs, t, name) + "\n"
    c = sol.KVCache(n_heads=h, d_head=d_h)
    expect_true(len(c) == 0, f"另外新建一个 cache，len 应当是 0，得到 {len(c)}",
                history + f"len(KVCache(n_heads={h}, d_head={d_h}))")


P2_SHAPES = [(1, 1, 1), (6, 2, 3), (12, 4, 2), (30, 3, 4), (9, 1, 5), (20, 6, 1)]


@case("Part 2", title="随机输入：每一步都等于 causal_mha 的那一行")
def p2_decode_random(sol):
    for seed, (T, h, d_h) in zip(seeds("p2", len(P2_SHAPES)), P2_SHAPES):
        x, Ws, setup = random_problem(seed, T, h * d_h)
        decode_all(sol, x, Ws, h, ref.causal_mha_slow(x, *Ws, h), setup)


@case("Part 2", title="append 形状不对：ValueError")
def p2_append_shapes(sol):
    good = (2, 3)
    for k, v in (((2, 3), (2, 4)), ((2, 4), (2, 3)), ((3, 3), (2, 3)), ((2, 3), (1, 3)), ((6,), (2, 3)),
                 ((2, 3), (6,)), ((3, 2), (3, 2)), ((1, 2, 3), (1, 2, 3))):
        cache = sol.KVCache(n_heads=2, d_head=3)
        call = f"cache = KVCache(n_heads=2, d_head=3)\ncache.append(np.zeros({k}), np.zeros({v}))"
        expect_raises(ValueError, cache.append, np.zeros(k), np.zeros(v), call=call)
    cache = sol.KVCache(n_heads=2, d_head=3)
    for n in (1, 2):
        cache.append(np.ones(good), np.ones(good))
        expect_true(len(cache) == n, f"形状正确的 append 做了 {n} 次，len(cache) 应当是 {n}，得到 {len(cache)}",
                    "cache = KVCache(n_heads=2, d_head=3)\n" + "cache.append(np.ones((2, 3)), np.ones((2, 3)))\n" * n
                    + "len(cache)")


@case("Part 2", title="字节数公式：键和值都要算")
def p2_bytes_formula(sol):
    # Distinct primes: dropping, doubling or swapping any factor changes the product.
    for args in ((32, 64, 128, 32_768, 1, 2), (3, 5, 7, 11, 13, 17), (1, 1, 1, 1, 1, 1), (80, 8, 128, 4096, 16, 2),
                 (7, 2, 64, 1, 3, 4)):
        want = args[0] * 2 * args[1] * args[2] * args[3] * args[4] * args[5]
        call = show_call("kv_cache_bytes", *args)
        got = sol.kv_cache_bytes(*args)
        expect_true(isinstance(got, numbers.Number) and not isinstance(got, bool) and got == want,
                    f"应当是 n_layers · 2 · n_kv_heads · d_head · seq_len · batch · bytes_per_elem = {want}，得到 {got!r}",
                    call, got)


# ---------------------------------------------------------------- Part 3: attention_blockwise

def blockwise_call(Q, K, V, block, setup=None):
    if setup is None:
        return f"attention_blockwise({lit(Q)}, {lit(K)}, {lit(V)}, {block})"
    return setup + f"attention_blockwise(Q, K, V, {block})"


@case("Part 3", sample=True, title="题目例子：两块，结果 30.8562")
def p3_example(sol):
    Q, K, V = np.array([[1.0]]), np.array([[0.0], [1.0], [3.0], [2.0]]), np.array([[10.0], [20.0], [30.0], [40.0]])
    expect_close(sol.attention_blockwise(Q.copy(), K.copy(), V.copy(), 2), [[30.8562]],
                 call=blockwise_call(Q, K, V, 2), **EXAMPLE_TOL)


@case("Part 3", title="随机输入，block 从 1 到 T_k + 3")
def p3_every_block(sol):
    shapes = [(1, 1, 1), (3, 7, 2), (2, 12, 1), (4, 10, 3), (1, 11, 4), (5, 5, 2), (2, 9, 3), (3, 1, 2)]
    for seed, (T_q, T_k, d_h) in zip(seeds("p3", len(shapes)), shapes):
        gen = np.random.default_rng(seed)
        Q, K, V = (gen.normal(size=(T, d_h)) * 0.8 for T in (T_q, T_k, T_k))
        setup = (f"rng = np.random.default_rng({seed})\n"
                 f"Q, K, V = (rng.normal(size=(T, {d_h})) * 0.8 for T in ({T_q}, {T_k}, {T_k}))\n")
        want = ref.attention_slow(Q, K, V)
        for block in range(1, T_k + 4):
            expect_close(sol.attention_blockwise(Q.copy(), K.copy(), V.copy(), block), want,
                         call=blockwise_call(Q, K, V, block, setup), rtol=1e-9, atol=1e-10)


@case("Part 3", title="最后一块更短，也要算进去")
def p3_short_last_block(sol):
    # The largest score sits in the last, shorter block, so leaving that block out changes the result a lot.
    for T_k, block in ((7, 3), (10, 4), (5, 2)):
        Q = np.array([[1.0, 0.5]])
        K = np.column_stack([np.arange(T_k, dtype=float) * 0.5, np.zeros(T_k)])
        V = np.column_stack([np.arange(T_k, dtype=float), -np.arange(T_k, dtype=float)])
        want = ref.attention_slow(Q, K, V)
        expect_close(sol.attention_blockwise(Q.copy(), K.copy(), V.copy(), block), want,
                     call=blockwise_call(Q, K, V, block), **TOL)


def offset_scores(M, sign, order, gen):
    """K for d_h = 4 and Q = [[2, 0, 0, 0]]: the scaled score q.k_j / sqrt(4) is exactly sign * (M + offset_j),
    with small integer offsets."""
    offsets = gen.integers(-6, 1, size=9).astype(float)
    offsets[0] = 0.0
    if order == "升序":
        offsets.sort()
    elif order == "降序":
        offsets[::-1].sort()
    K = np.zeros((9, 4))
    K[:, 0] = sign * (M + offsets)
    K[:, 1:] = gen.integers(-3, 4, size=(9, 3))
    return K


def check_offset_case(sol, M, sign, salt):
    gen = np.random.default_rng(seeds(salt, 1)[0])
    Q = np.array([[2.0, 0.0, 0.0, 0.0]])
    V = np.round(gen.normal(size=(9, 4)), 3)
    for order in ("升序", "降序", "乱序"):
        K = offset_scores(M, sign, order, gen)
        want = ref.attention_slow(Q, K, V)
        for block in (1, 2, 4, 9, 12):
            call = blockwise_call(Q, K, V, block) + f"   # 得分 q·k/sqrt(d_h) 在 {sign * M:g} 附近，{order}"
            expect_close(sol.attention_blockwise(Q.copy(), K.copy(), V.copy(), block), want, call=call,
                         rtol=1e-9, atol=1e-9)


@case("Part 3", title="得分达到 1e3、1e5、1e8 也不溢出")
def p3_large_scores(sol):
    # Integer scores are exact in float64, so a max-shifted answer reproduces the reference to rounding.
    for M in (1e3, 1e5, 1e8):
        check_offset_case(sol, M, 1.0, f"large {M}")


@case("Part 3", title="得分全是很大的负数")
def p3_negative_scores(sol):
    # Every scaled score is about -1000: exp of an unshifted (or zero-shifted) score underflows to 0 for all keys.
    check_offset_case(sol, 1e3, -1.0, "negative")


# Scaled scores of the first query (the second sees their negatives). After the 1000 at the start, a later block whose
# own max is more than ~709 lower must still be rescaled by exp(m_old - m_new) <= 1: an answer that takes the block's
# own max as the new reference (instead of max(m, block max)) multiplies by exp(> 709) and overflows to inf / nan.
SPREAD = [1000, 999, -1000, -1200, 998, 0, -800, -900, 997, -3000, 100, 50, -500, -600, -700, -800]


@case("Part 3", title="后面的块比前面低上千：m 只增不减")
def p3_spread_scores(sol):
    gen = np.random.default_rng(seeds("spread", 1)[0])
    Q = np.array([[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]])
    K = np.zeros((len(SPREAD), 4))
    K[:, 0] = SPREAD                     # q·k / sqrt(4) is exactly ±SPREAD[j]
    K[:, 1:] = gen.integers(-3, 4, size=(len(SPREAD), 3))
    V = np.round(gen.normal(size=(len(SPREAD), 4)), 3)
    want = ref.attention_slow(Q, K, V)
    for block in (1, 2, 3, 4, 5, 16, 20):
        call = blockwise_call(Q, K, V, block) + "   # 第一个查询的得分 q·k/sqrt(d_h) = " + str(SPREAD)
        expect_close(sol.attention_blockwise(Q.copy(), K.copy(), V.copy(), block), want, call=call,
                     rtol=1e-9, atol=1e-9)


T_Q, T_K, D_H, BLOCK = 128, 16_384, 2, 128


@case("Part 3", cpu=5, title="不保留完整的 T_q × T_k 得分矩阵")
def p3_memory(sol):
    # tracemalloc sees every NumPy buffer. The full float64 score matrix is 128 x 16384 x 8 B = 16.8 MB; one block
    # of scores is 0.13 MB and a copy of K or V 0.26 MB. The reference peaks at 0.53 MB, a two-pass version
    # (global max first) at 1.25 MB; anything that holds all T_q x T_k scores at once needs at least 16.8 MB.
    # The limit is a quarter of that.
    # cpu=5: tracemalloc slows Python-level loops; a valid per-query, per-block loop takes about 1 s here.
    seed = seeds("memory", 1)[0]
    gen = np.random.default_rng(seed)
    Q, K, V = gen.normal(size=(T_Q, D_H)), gen.normal(size=(T_K, D_H)), gen.normal(size=(T_K, D_H))
    full_mb = T_Q * T_K * 8 / 1e6
    limit_mb = full_mb / 4
    s = Q @ K.T / np.sqrt(D_H)
    w = np.exp(s - s.max(axis=1, keepdims=True))
    want = (w @ V) / w.sum(axis=1, keepdims=True)
    del s, w
    call = (f"rng = np.random.default_rng({seed})\n"
            f"Q, K, V = rng.normal(size=({T_Q}, {D_H})), rng.normal(size=({T_K}, {D_H})), rng.normal(size=({T_K}, {D_H}))\n"
            f"attention_blockwise(Q, K, V, {BLOCK})")
    args = Q.copy(), K.copy(), V.copy()
    tracemalloc.start()
    try:
        got = sol.attention_blockwise(*args, BLOCK)
        peak = tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()
    expect_close(got, want, call=call, rtol=1e-9, atol=1e-9)
    expect_true(peak <= limit_mb,
                f"调用期间分配内存的峰值是 {peak:.1f} MB，而完整的 {T_Q} × {T_K} 得分矩阵是 {full_mb:.1f} MB："
                f"题目要求一次只保留一块 (T_q, block) 的得分（这里约 {T_Q * BLOCK * 8 / 1e6:.2f} MB），"
                f"这个测试允许到 {limit_mb:.1f} MB", call)
