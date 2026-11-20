import contextlib

import numpy as np

from judgelib import Mismatch, case, expect_close, expect_true, rng
import ref

# The worked examples are printed to 4 decimals.
EX_ATOL = 1e-4
# Float64 answers that evaluate the formulas in another order (einsum, @, log-sum-exp, a different max) differ from
# the index loops in ref.py by about 1e-14 on these inputs, outputs being at most ~3 in size. The statement asks for
# atol=1e-8 in Part 3 and its own checks use atol=1e-8 throughout.
RTOL, ATOL = 1e-8, 1e-8
P3_ATOL = 1e-8
GUESS_RTOL, GUESS_ATOL = 1e-6, 1e-8     # an answer "looks like" a known mistake when it matches it this closely


def build(code, s):
    """Run `code` with rng = np.random.default_rng(s): the inputs, and the same code as the start of the call."""
    ns = {"np": np, "rng": np.random.default_rng(s)}
    exec(code, ns)
    return ns, f"rng = np.random.default_rng({s}); {code}; "


def seeds(n):
    r = rng()
    return [r.randrange(2 ** 31) for _ in range(n)]


def as_float(got):
    try:
        return np.asarray(got, dtype=float)
    except Exception:
        return None


def close(a, b, rtol=GUESS_RTOL, atol=GUESS_ATOL):
    return a is not None and a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, rtol=rtol, atol=atol)


@contextlib.contextmanager
def prefixed(text):
    """Put `text` in front of the message of a failing check."""
    try:
        yield
    except Mismatch as e:
        e.msg = text + (e.msg or "")
        e.args = (e.msg,)
        raise


# ---------------------------------------------------------------- Part 1
def check_attention(sol, Q, K, V, mask, call, want=None, rtol=RTOL, atol=ATOL):
    """mask=False: call without the mask argument."""
    want = ref.attention(Q, K, V, None if mask is False else mask) if want is None else np.asarray(want, dtype=float)
    args = (Q.copy(), K.copy(), V.copy()) + (() if mask is False else (None if mask is None else mask.copy(),))
    got = sol.scaled_dot_product_attention(*args)
    a = as_float(got)
    if mask is not None and mask is not False and a is not None and a.shape == want.shape:
        full = ~mask.any(axis=-1)                       # (B, T_q): queries that may attend to no key
        if full.any():
            rows = a[full]
            if not np.isfinite(rows).all():
                raise Mismatch("所有键都被掩码的查询，结果是 nan / inf：题目约定它的权重全是 0、输出零向量，而不是 0/0",
                               call, want=want, got=got)
            if np.abs(rows).max() > atol:
                uniform = np.broadcast_to(V.mean(axis=1)[:, None, :], a.shape)[full]
                if np.allclose(rows, uniform, rtol=GUESS_RTOL, atol=GUESS_ATOL):
                    raise Mismatch("所有键都被掩码的查询，输出像是所有值向量的平均（权重变成了均匀分布）：题目约定它的权重"
                                   "全是 0、输出零向量", call, want=want, got=got)
                raise Mismatch("所有键都被掩码的查询，输出应当是零向量", call, want=want, got=got)
    expect_close(got, want, rtol=rtol, atol=atol, call=call)


@case("Part 1", sample=True, title="题目例子：一行全被掩码")
def p1_example(sol):
    Q = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    K = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    V = np.array([[[10.0], [20.0], [30.0]]])
    mask = np.array([[[False, False, False], [True, False, True]]])
    call = ("Q = np.array([[[1., 0.], [0., 1.]]]); K = np.array([[[1., 0.], [0., 1.], [1., 1.]]]); "
            "V = np.array([[[10.], [20.], [30.]]]); mask = np.array([[[False, False, False], [True, False, True]]]); "
            "scaled_dot_product_attention(Q, K, V, mask)")
    check_attention(sol, Q, K, V, mask, call, want=[[[0.0], [23.3952]]], rtol=0.0, atol=EX_ATOL)


@case("Part 1", title="不带掩码：T_q ≠ T_k、d_v ≠ d")
def p1_no_mask(sol):
    shapes = [(1, 1, 1, 1, 1), (2, 3, 5, 4, 2), (3, 4, 2, 1, 3), (1, 6, 6, 5, 4)]
    for (B, Tq, Tk, d, dv), s in zip(shapes, seeds(len(shapes))):
        ns, pre = build(f"Q = rng.normal(size=({B}, {Tq}, {d})); K = rng.normal(size=({B}, {Tk}, {d})); "
                        f"V = rng.normal(size=({B}, {Tk}, {dv}))", s)
        if Tq == Tk:
            check_attention(sol, ns["Q"], ns["K"], ns["V"], False, pre + "scaled_dot_product_attention(Q, K, V)")
        else:
            check_attention(sol, ns["Q"], ns["K"], ns["V"], None, pre + "scaled_dot_product_attention(Q, K, V, None)")


@case("Part 1", title="随机掩码")
def p1_mask(sol):
    shapes = [(2, 3, 4, 2, 3, 0.5), (3, 5, 5, 4, 1, 0.3), (1, 4, 6, 3, 2, 0.8), (2, 2, 3, 2, 2, 1.0)]
    for (B, Tq, Tk, d, dv, p), s in zip(shapes, seeds(len(shapes))):
        ns, pre = build(f"Q = rng.normal(size=({B}, {Tq}, {d})); K = rng.normal(size=({B}, {Tk}, {d})); "
                        f"V = rng.normal(size=({B}, {Tk}, {dv})); mask = rng.random(({B}, {Tq}, {Tk})) < {p}; "
                        f"mask[:, :, 0] = True", s)
        check_attention(sol, ns["Q"], ns["K"], ns["V"], ns["mask"], pre + "scaled_dot_product_attention(Q, K, V, mask)")


@case("Part 1", title="整行被掩码：输出零向量")
def p1_fully_masked(sol):
    s1, s2 = seeds(2)
    ns, pre = build("Q = rng.normal(size=(2, 3, 2)); K = rng.normal(size=(2, 4, 2)); V = rng.normal(size=(2, 4, 3)); "
                    "mask = np.zeros((2, 3, 4), dtype=bool)", s1)
    check_attention(sol, ns["Q"], ns["K"], ns["V"], ns["mask"], pre + "scaled_dot_product_attention(Q, K, V, mask)")
    ns, pre = build("Q = rng.normal(size=(3, 4, 3)); K = rng.normal(size=(3, 5, 3)); V = rng.normal(size=(3, 5, 2)); "
                    "mask = rng.random((3, 4, 5)) < 0.5; mask[:, 0, :] = False; mask[:, 1, :] = False; "
                    "mask[:, 1, 2] = True; mask[:, 2, :] = True", s2)
    check_attention(sol, ns["Q"], ns["K"], ns["V"], ns["mask"], pre + "scaled_dot_product_attention(Q, K, V, mask)")


@case("Part 1", title="被掩码的键得分很大")
def p1_masked_large(sol):
    # Keys 0 and 1 score about 2500 or more against every query, and are masked for every query. The visible keys
    # score a few units, so softmax over all keys gives them weight exp(-2500) = 0 in float64: zeroing the masked
    # weights after normalizing leaves nothing to renormalize.
    (s,) = seeds(1)
    ns, pre = build("Q = rng.normal(size=(2, 3, 4)); Q[..., 0] = np.abs(Q[..., 0]) + 0.5; "
                    "K = rng.normal(size=(2, 5, 4)); K[:, :2, :] = 0.0; K[:, :2, 0] = 1e4; V = rng.normal(size=(2, 5, 3)); "
                    "mask = rng.random((2, 3, 5)) < 0.6; mask[:, :, :2] = False; mask[:, :, 2] = True", s)
    with prefixed("可见的键得分正常、被掩码的键得分很大时结果不对。被掩码的键要在 softmax 之前排除（得分设为 −∞），"
                  "不能先对所有键做 softmax 再把它们的权重清零。"):
        check_attention(sol, ns["Q"], ns["K"], ns["V"], ns["mask"], pre + "scaled_dot_product_attention(Q, K, V, mask)")


@case("Part 1", title="得分很大：先减去每行最大值")
def p1_large_scores(sol):
    # The scores have a standard deviation of about 1600; np.exp overflows past about 709.
    s1, s2 = seeds(2)
    for code, s, call in [
        ("Q = 40 * rng.normal(size=(2, 4, 4)); K = 40 * rng.normal(size=(2, 5, 4)); V = rng.normal(size=(2, 5, 3))",
         s1, "scaled_dot_product_attention(Q, K, V)"),
        ("Q = 40 * rng.normal(size=(2, 3, 4)); K = 40 * rng.normal(size=(2, 6, 4)); V = rng.normal(size=(2, 6, 2)); "
         "mask = rng.random((2, 3, 6)) < 0.6; mask[:, :, 1] = True", s2, "scaled_dot_product_attention(Q, K, V, mask)"),
    ]:
        ns, pre = build(code, s)
        mask = ns.get("mask", False)
        top = np.abs(np.einsum("bqd,bkd->bqk", ns["Q"], ns["K"])).max() / 2.0
        got = as_float(sol.scaled_dot_product_attention(ns["Q"].copy(), ns["K"].copy(), ns["V"].copy(),
                                                         *(() if mask is False else (mask.copy(),))))
        if got is not None and not np.isfinite(got).all():
            raise Mismatch(f"得分很大（绝对值最大约 {top:.0f}）时结果里出现了 nan / inf：np.exp 的参数超过约 709 就会溢出。"
                           f"在 np.exp 之前先减去每行的最大值", pre + call, show=False)
        check_attention(sol, ns["Q"], ns["K"], ns["V"], mask, pre + call)


# ---------------------------------------------------------------- Part 2
def mha_guesses(X, Ws, H):
    W_q, W_k, W_v, W_o = Ws
    dm = X.shape[2]
    return [
        (lambda: ref.multi_head(X, *Ws, H, strided=True),
         "头 h 应当用 Q、K、V 的第 h·d_head 到 h·d_head + d_head − 1 列（连续的一段），现在像是用了第 h、h + H、h + 2H、… 列"),
        (lambda: ref.multi_head(X, *Ws, H, scale_dim=dm),
         "每个头的得分应当除以 √d_head，现在像是除以了 √d_model"),
        (lambda: ref.multi_head(X, *Ws, H, causal=False),
         "没有加因果掩码：位置 i 只能关注 j ≤ i 的位置"),
        (lambda: ref.multi_head(X, W_q, W_k, W_v, W_o.T, H),
         "输出投影应当是（拼接后的输出）· W_o，现在像是乘了 W_o 的转置"),
        (lambda: ref.multi_head(X, W_q.T, W_k.T, W_v.T, W_o, H),
         "应当是 Q = X·W_q（K、V 同理），现在像是乘了权重矩阵的转置"),
        (lambda: ref.multi_head(X, W_q.T, W_k.T, W_v.T, W_o.T, H),
         "应当是 Q = X·W_q（K、V 和输出投影同理），现在像是乘了权重矩阵的转置"),
    ]


def check_mha(sol, X, Ws, H, call, want=None, rtol=RTOL, atol=ATOL):
    want = ref.multi_head(X, *Ws, H) if want is None else np.asarray(want, dtype=float)
    got = sol.multi_head_self_attention(X.copy(), *(W.copy() for W in Ws), H)
    a = as_float(got)
    if (a is not None and a.shape == want.shape and np.isfinite(a).all()
            and not np.allclose(a, want, rtol=rtol, atol=atol)):
        for guess, msg in mha_guesses(X, Ws, H):
            if close(a, guess()):
                raise Mismatch(msg, call, want=want, got=got)
    expect_close(got, want, rtol=rtol, atol=atol, call=call)
    return a


def random_weights(dm):
    return "; ".join(f"{w} = 0.5 * rng.normal(size=({dm}, {dm}))" for w in ("W_q", "W_k", "W_v", "W_o"))


def mha_case(sol, B, T, dm, H, s):
    ns, pre = build(f"X = 0.5 * rng.normal(size=({B}, {T}, {dm})); {random_weights(dm)}", s)
    Ws = [ns[w] for w in ("W_q", "W_k", "W_v", "W_o")]
    check_mha(sol, ns["X"], Ws, H, pre + f"multi_head_self_attention(X, W_q, W_k, W_v, W_o, {H})")


@case("Part 2", sample=True, title="题目例子：权重都是单位矩阵")
def p2_example(sol):
    X = np.array([[[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]])
    I4 = np.eye(4)
    call = "X = np.array([[[1., 0., 2., 0.], [0., 1., 0., 2.]]]); I4 = np.eye(4); multi_head_self_attention(X, I4, I4, I4, I4, 2)"
    check_mha(sol, X, [I4] * 4, 2, call, want=[[[1.0, 0.0, 2.0, 0.0], [0.3302, 0.6698, 0.1116, 1.8884]]],
              rtol=0.0, atol=EX_ATOL)


@case("Part 2", title="随机权重：多个头")
def p2_random(sol):
    shapes = [(2, 5, 6, 2), (3, 4, 4, 4), (1, 6, 6, 3), (2, 3, 8, 2)]
    for (B, T, dm, H), s in zip(shapes, seeds(len(shapes))):
        mha_case(sol, B, T, dm, H, s)


@case("Part 2", title="H = 1、d_head = 1、T = 1")
def p2_edge(sol):
    shapes = [(2, 4, 5, 1), (2, 4, 4, 4), (3, 1, 6, 3)]
    for (B, T, dm, H), s in zip(shapes, seeds(len(shapes))):
        mha_case(sol, B, T, dm, H, s)


@case("Part 2", title="因果性：改后面的 token 不影响前面")
def p2_causal(sol):
    (s,) = seeds(1)
    ns, pre = build(f"X = rng.normal(size=(2, 5, 6)); {random_weights(6)}; X2 = X.copy(); X2[:, 4, :] += rng.normal(size=6); "
                    "X3 = X.copy(); X3[:, 2, :] += rng.normal(size=6)", s)
    Ws = [ns[w] for w in ("W_q", "W_k", "W_v", "W_o")]
    before = as_float(sol.multi_head_self_attention(ns["X"].copy(), *(W.copy() for W in Ws), 3))
    for name, t in (("X2", 4), ("X3", 2)):
        call = pre + (f"np.abs(multi_head_self_attention({name}, W_q, W_k, W_v, W_o, 3)[:, :{t}] "
                      f"- multi_head_self_attention(X, W_q, W_k, W_v, W_o, 3)[:, :{t}]).max()")
        after = as_float(sol.multi_head_self_attention(ns[name].copy(), *(W.copy() for W in Ws), 3))
        expect_true(before is not None and after is not None and before.shape == after.shape == (2, 5, 6),
                    "返回值的形状应当是 (B, T, d_model) = (2, 5, 6)", call)
        change = float(np.abs(after[:, :t] - before[:, :t]).max())
        expect_true(change <= ATOL, f"只改了第 {t} 个 token，位置 0..{t - 1} 的输出就变了（最大变化 {change:.3g}）："
                                    f"位置 i 只能关注 j ≤ i 的位置", call)


# ---------------------------------------------------------------- Part 3
def diagnose_block(sol, x, gamma, beta, Ws):
    """Which of the two functions disagrees with its formula on this input; '' when neither does."""
    want = ref.layer_norm(x, gamma, beta)
    got = as_float(sol.layer_norm(x.copy(), gamma.copy(), beta.copy()))
    if not close(got, want, rtol=0.0, atol=P3_ATOL):
        if close(got, ref.layer_norm_batch_mean(x, gamma, beta)):
            return ("layer_norm 和公式不一致：μ 应当是每一行自己 d 个特征的平均，现在像是沿批次轴求的平均"
                    "（批次大小为 1 时 x − μ 恒为 0，结果只剩 β）")
        return "layer_norm(x, gamma, beta) 和 LN 的公式不一致"
    return diagnose_attention(sol, want, Ws)


def diagnose_attention(sol, x, Ws):
    want = ref.causal_attention(x, *Ws)
    got = as_float(sol.causal_self_attention(x.copy(), *(W.copy() for W in Ws)))
    if close(got, want, rtol=0.0, atol=P3_ATOL):
        return ""
    if close(got, ref.causal_attention_buggy(x, *Ws, floor=True)):
        return "causal_self_attention 和公式不一致：softmax 看到的得分只剩下整数部分，QKᵀ/√d 被向下取整了"
    if close(got, ref.causal_attention_buggy(x, *Ws, max_axis=-2)):
        return ("causal_self_attention 和公式不一致：exp 之前减去的最大值应当是每个查询（每一行）一个常数，"
                "现在它随键的下标 j 变化，softmax 的结果因此变了")
    return "causal_self_attention 和 Part 2 的公式（H = 1）不一致"


def check_block(sol, x, gamma, beta, Ws, call, want=None, atol=P3_ATOL):
    want = ref.block(x, gamma, beta, *Ws) if want is None else np.asarray(want, dtype=float)
    got = sol.transformer_block_forward(x.copy(), gamma.copy(), beta.copy(), *(W.copy() for W in Ws))
    a = as_float(got)
    hint = ""
    if not close(a, want, rtol=0.0, atol=atol):
        hint = diagnose_block(sol, x, gamma, beta, Ws)
    expect_close(got, want, rtol=0.0, atol=atol, call=call, msg=hint)


def block_inputs(B, T, d, s, extra=""):
    return build(f"x = rng.normal(size=({B}, {T}, {d})); gamma = rng.normal(size={d}); beta = rng.normal(size={d}); "
                 f"W_q = rng.normal(size=({d}, {d})); W_k = rng.normal(size=({d}, {d})); W_v = rng.normal(size=({d}, {d}))"
                 + extra, s)


@case("Part 3", sample=True, title="题目例子")
def p3_example(sol):
    x = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    I2 = np.eye(2)
    call = ("x = np.array([[[1., 0.], [0., 1.]]]); gamma = np.array([1., 1.]); beta = np.array([0., 0.]); I2 = np.eye(2); "
            "transformer_block_forward(x, gamma, beta, I2, I2, I2)")
    check_block(sol, x, np.ones(2), np.zeros(2), [I2] * 3, call, want=[[[2.0, -1.0], [-0.8884, 1.8884]]], atol=EX_ATOL)


@case("Part 3", title="layer_norm：每行用自己的 d 个特征")
def p3_layer_norm(sol):
    shapes = [(3, 5, 4, None), (1, 4, 6, None), (2, 3, 3, None), (2, 1, 5, None), (2, 3, 4, 0.5)]
    for (B, T, d, eps), s in zip(shapes, seeds(len(shapes))):
        ns, pre = build(f"x = rng.normal(size=({B}, {T}, {d})); gamma = rng.normal(size={d}); beta = rng.normal(size={d})", s)
        x, gamma, beta = ns["x"], ns["gamma"], ns["beta"]
        kw = {} if eps is None else {"eps": eps}
        call = pre + "layer_norm(x, gamma, beta" + ("" if eps is None else f", eps={eps}") + ")"
        want = ref.layer_norm(x, gamma, beta, **kw)
        got = sol.layer_norm(x.copy(), gamma.copy(), beta.copy(), **kw)
        a = as_float(got)
        if eps is None and not close(a, want, rtol=0.0, atol=P3_ATOL) and close(a, ref.layer_norm_batch_mean(x, gamma, beta)):
            raise Mismatch("μ 应当是每一行自己 d 个特征的平均，现在像是沿批次轴求的平均（同一位置、不同序列的平均）",
                           call, want=want, got=got)
        expect_close(got, want, rtol=0.0, atol=P3_ATOL, call=call)


@case("Part 3", title="因果性：改后面的 token 不影响前面")
def p3_causal(sol):
    for (B, T, d), s in zip([(3, 5, 4), (2, 6, 3)], seeds(2)):
        ns, pre = block_inputs(B, T, d, s, f"; x2 = x.copy(); x2[:, {T - 1}, :] = rng.normal(size=({B}, {d})); "
                                           f"x3 = x.copy(); x3[:, 2, :] = rng.normal(size=({B}, {d}))")
        Ws = [ns[w] for w in ("W_q", "W_k", "W_v")]

        def run(name):
            return as_float(sol.transformer_block_forward(ns[name].copy(), ns["gamma"].copy(), ns["beta"].copy(),
                                                          *(W.copy() for W in Ws)))

        before = run("x")
        for name, t in (("x2", T - 1), ("x3", 2)):
            call = pre + (f"np.abs(transformer_block_forward({name}, gamma, beta, W_q, W_k, W_v)[:, :{t}] "
                          f"- transformer_block_forward(x, gamma, beta, W_q, W_k, W_v)[:, :{t}]).max()")
            after = run(name)
            expect_true(before is not None and after is not None and before.shape == after.shape == (B, T, d),
                        f"返回值的形状应当是 x 的形状 {(B, T, d)}", call)
            change = float(np.abs(after[:, :t] - before[:, :t]).max())
            expect_true(change <= P3_ATOL, f"只改了 x 的第 {t} 个 token，位置 0..{t - 1} 的输出就变了（最大变化 {change:.3g}）："
                                           f"位置 i 的输出只能依赖第 0..i 个 token", call)


@case("Part 3", title="causal_self_attention 和公式一致")
def p3_attention(sol):
    shapes = [(3, 5, 4), (1, 1, 3), (2, 6, 2), (2, 4, 8)]
    for (B, T, d), s in zip(shapes, seeds(len(shapes))):
        ns, pre = build(f"x = rng.normal(size=({B}, {T}, {d})); W_q = rng.normal(size=({d}, {d})); "
                        f"W_k = rng.normal(size=({d}, {d})); W_v = rng.normal(size=({d}, {d}))", s)
        x, Ws = ns["x"], [ns[w] for w in ("W_q", "W_k", "W_v")]
        call = pre + "causal_self_attention(x, W_q, W_k, W_v)"
        want = ref.causal_attention(x, *Ws)
        got = sol.causal_self_attention(x.copy(), *(W.copy() for W in Ws))
        hint = "" if close(as_float(got), want, rtol=0.0, atol=P3_ATOL) else diagnose_attention(sol, x, Ws)
        expect_close(got, want, rtol=0.0, atol=P3_ATOL, call=call, msg=hint)


@case("Part 3", title="得分很大：每行减自己的最大值")
def p3_attention_large(sol):
    # x is scaled by 30, so the scores reach thousands and the rows' maxima differ by far more than 745: with one
    # maximum for the whole array (or per batch) every exp of some row underflows to 0, and that row becomes 0/0.
    # W_v is scaled down so the outputs stay O(1); correct answers differ from ref.py by about 1e-13 here.
    shapes = [(2, 5, 4), (1, 6, 3), (3, 4, 2)]
    for (B, T, d), s in zip(shapes, seeds(len(shapes))):
        ns, pre = build(f"x = 30 * rng.normal(size=({B}, {T}, {d})); W_q = rng.normal(size=({d}, {d})); "
                        f"W_k = rng.normal(size=({d}, {d})); W_v = rng.normal(size=({d}, {d})) / 30", s)
        x, Ws = ns["x"], [ns[w] for w in ("W_q", "W_k", "W_v")]
        call = pre + "causal_self_attention(x, W_q, W_k, W_v)"
        want = ref.causal_attention(x, *Ws)
        got = sol.causal_self_attention(x.copy(), *(W.copy() for W in Ws))
        a = as_float(got)
        if a is not None and a.shape == want.shape and not np.isfinite(a).all():
            raise Mismatch("得分很大时结果里出现了 nan / inf：exp 之前要减去每一行（每个查询）自己的最大值。整个数组或"
                           "整个批次共用一个最大值时，得分低的那些行 exp 全部下溢成 0，归一化变成 0/0", call, want=want, got=got)
        hint = "" if close(a, want, rtol=0.0, atol=P3_ATOL) else diagnose_attention(sol, x, Ws)
        expect_close(got, want, rtol=0.0, atol=P3_ATOL, call=call, msg=hint)


@case("Part 3", title="整个 block 和下标循环一致")
def p3_block(sol):
    shapes = [(3, 5, 4), (2, 4, 4), (1, 3, 6), (2, 1, 3)]
    for (B, T, d), s in zip(shapes, seeds(len(shapes))):
        ns, pre = block_inputs(B, T, d, s)
        check_block(sol, ns["x"], ns["gamma"], ns["beta"], [ns[w] for w in ("W_q", "W_k", "W_v")],
                    pre + "transformer_block_forward(x, gamma, beta, W_q, W_k, W_v)")
