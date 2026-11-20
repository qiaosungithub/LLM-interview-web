import numpy as np

from judgelib import Mismatch, case, expect_true, info, rng
import ref

NAMES = ref.NAMES

# The statement's worked example (Part 1), B = d = m = C = 2.
EX_X = np.array([[1.0, -1.0], [0.5, 2.0]])
EX_Y = np.array([0, 1])
EX_P = {"W1": np.array([[0.5, -0.5], [-1.0, 1.0]]), "b1": np.array([0.1, -0.2]),
        "W2": np.array([[1.0, -1.0], [-1.0, 2.0]]), "b2": np.array([0.0, 0.5])}
EX_SETUP = ref.setup_code(EX_P, EX_X, EX_Y)

# Tolerances against the reference (an exact log-sum-exp computation). Every input below keeps the label
# probability >= 1e-3 and every pre-activation >= 1e-3 away from the kink (the all-zero case sits on the kink, but
# there W2 = 0 makes dZ1 = 0 either way), so a correct answer differs from the reference only by rounding, measured
# below 1e-3 of these tolerances; the bugs these tests target exceed them by a factor of 1000 or more.
LOSS_TOL = dict(rtol=1e-7, atol=1e-9)
GRAD_TOL = dict(rtol=1e-6, atol=1e-9)
# Central differences with eps = 1e-5 in float64 are accurate to about eps^2/6 * |L'''| + 2.2e-16 * |L| / eps,
# roughly 1e-10 here; gradient bugs move entries by 1e-3 or more.
FD_TOL = dict(rtol=1e-4, atol=1e-6)


def npgen(salt=""):
    return np.random.default_rng(rng(salt).getrandbits(64))


def cp(params):
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in params.items()}


def as_float_array(x, label, call):
    try:
        return np.asarray(x, dtype=np.float64)
    except Exception:
        raise Mismatch(f"{label} 不是数值数组：{type(x).__name__}", call, None, x) from None


def close_or_fail(got, want, label, call, rtol, atol):
    """got must have want's shape, be finite, and match within atol + rtol * |want|; the message names the entry."""
    a = as_float_array(got, label, call)
    want = np.asarray(want, dtype=np.float64)
    shown = float(want) if want.ndim == 0 else want
    label += " " if label[-1].isascii() else ""
    if a.shape != want.shape:
        raise Mismatch(f"{label}的形状应当是 {want.shape}，得到 {a.shape}", call, shown, got)
    if not np.all(np.isfinite(a)):
        idx = tuple(int(i) for i in np.argwhere(~np.isfinite(a))[0])
        at = f"位置 {idx} " if a.ndim else ""
        raise Mismatch(f"{label}里有 nan 或 inf（{at}是 {a[idx] if a.ndim else float(a)}）", call, shown, got)
    err = np.abs(a - want)
    bad = err > atol + rtol * np.abs(want)
    if bad.any():
        idx = np.unravel_index(np.argmax(np.where(bad, err, -1.0)), err.shape) if a.ndim else ()
        idx = tuple(int(i) for i in idx)
        at = f"位置 {idx} " if a.ndim else ""
        raise Mismatch(f"{label}不对：{at}期望 {float(want[idx]):.10g}，得到 {float(a[idx]):.10g}"
                       f"（允许误差 atol={atol:g}, rtol={rtol:g}）", call, shown, got)


def unpack(result, call):
    """forward_backward's return value as (loss: float, grads: dict)."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise Mismatch("forward_backward 应当返回 (L, grads) 两个值", call, "(L, grads)", result)
    loss, grads = result
    if np.ndim(loss) != 0:
        raise Mismatch(f"L 应当是一个标量，得到形状 {np.shape(loss)}", call, None, loss)
    try:
        loss = float(loss)
    except Exception:
        raise Mismatch(f"L 应当是一个数，得到 {type(loss).__name__}", call, None, loss) from None
    if not isinstance(grads, dict):
        raise Mismatch(f"grads 应当是 dict，得到 {type(grads).__name__}", call, None, grads)
    missing = [k for k in NAMES if k not in grads]
    if missing:
        raise Mismatch(f"grads 缺少键 {missing}（应当有 W1、b1、W2、b2）", call, None, sorted(map(str, grads)))
    return loss, grads


def check_fb(sol, params, X, y, label=""):
    """forward_backward on copies of the inputs, against the reference: loss, then every gradient."""
    want_loss, want_grads = ref.forward_backward(cp(params), X.copy(), y.copy())
    call = f"{label}{ref.setup_code(params, X, y)}\nloss, grads = forward_backward(params, X, y)"
    loss, grads = unpack(sol.forward_backward(cp(params), X.copy(), y.copy()), call)
    close_or_fail(loss, want_loss, "L", call, **LOSS_TOL)
    for k in NAMES:
        close_or_fail(grads[k], want_grads[k], f"grads['{k}']", call, **GRAD_TOL)


def check_params_dict(got, shapes, what, call):
    if not isinstance(got, dict):
        raise Mismatch(f"{what} 应当返回参数 dict，得到 {type(got).__name__}", call, None, got)
    for k in NAMES:
        if k not in got:
            raise Mismatch(f"{what} 返回的 dict 缺少键 {k!r}", call, None, sorted(map(str, got)))
        a = as_float_array(got[k], f"{what} 返回的 {k}", call)
        if a.shape != shapes[k]:
            raise Mismatch(f"{what} 返回的 {k} 形状应当是 {shapes[k]}，得到 {a.shape}", call, None, got[k])


def shapes_of(d, m, C):
    return {"W1": (d, m), "b1": (m,), "W2": (m, C), "b2": (C,)}


# ---------------------------------------------------------------- Part 1: forward_backward

@case("Part 1", sample=True, title="题目例子：loss ≈ 0.0354")
def p1_example_loss(sol):
    call = f"{EX_SETUP}\nloss, grads = forward_backward(params, X, y)"
    loss, _ = unpack(sol.forward_backward(cp(EX_P), EX_X.copy(), EX_Y.copy()), call)
    close_or_fail(loss, 0.0354, "L", call, rtol=0, atol=5e-5)


@case("Part 1", sample=True, title="题目例子：梯度对上中心差分")
def p1_example_fd(sol):
    call = (f"{EX_SETUP}\nloss, grads = forward_backward(params, X, y)\n"
            "# 每一项梯度都和你自己的 loss 的中心差分 (L(θ+1e-5) - L(θ-1e-5)) / 2e-5 比较")
    _, grads = unpack(sol.forward_backward(cp(EX_P), EX_X.copy(), EX_Y.copy()), call)
    numeric = ref.numeric_grads(sol.forward_backward, EX_P, EX_X, EX_Y)
    for k in NAMES:
        close_or_fail(grads[k], numeric[k], f"grads['{k}']（和中心差分比）", call, **FD_TOL)


@case("Part 1", title="题目例子：四个梯度的值")
def p1_example_grads(sol):
    check_fb(sol, EX_P, EX_X, EX_Y)


@case("Part 1", title="d、m、C 各不相同：键和形状")
def p1_shapes(sol):
    p, X, y = ref.random_net(npgen(), d=3, m=5, C=4, B=6, margin=1e-3, min_prob=1e-3)
    call = f"{ref.setup_code(p, X, y)}\nloss, grads = forward_backward(params, X, y)"
    _, grads = unpack(sol.forward_backward(cp(p), X.copy(), y.copy()), call)
    for k in NAMES:
        a = as_float_array(grads[k], f"grads['{k}']", call)
        expect_true(a.shape == p[k].shape, f"grads['{k}'] 的形状应当和 params['{k}'] 一样是 {p[k].shape}，得到 {a.shape}",
                    call, grads[k])
    check_fb(sol, p, X, y)


@case("Part 1", title="批大小 B = 1")
def p1_batch_one(sol):
    gen = npgen()
    for d, m, C in [(3, 4, 5), (1, 2, 3)]:
        p, X, y = ref.random_net(gen, d=d, m=m, C=C, B=1, margin=1e-3, min_prob=1e-3)
        check_fb(sol, p, X, y)


@case("Part 1", title="随机网络和批次，对比参考答案")
def p1_random(sol):
    gen = npgen()
    for _ in range(12):
        d, m, C, B = int(gen.integers(1, 7)), int(gen.integers(1, 9)), int(gen.integers(2, 7)), int(gen.integers(1, 21))
        p, X, y = ref.random_net(gen, d=d, m=m, C=C, B=B, margin=1e-3, min_prob=1e-3)
        check_fb(sol, p, X, y)


@case("Part 1", title="有的隐藏单元整批都不激活")
def p1_dead_units(sol):
    gen = npgen()
    done = 0
    while done < 3:
        p, X, y = ref.random_net(gen, d=3, m=6, C=3, B=8, margin=1e-3, min_prob=1e-3)
        dead = gen.choice(6, size=2, replace=False)
        p["b1"][dead] = -20.0                      # Z1 < -10 on every example: ReLU output and gradient are 0
        if ref.label_probs(p, X, y).min() >= 1e-3:
            check_fb(sol, p, X, y)
            done += 1


@case("Part 1", title="全零参数")
def p1_zero_params(sol):
    gen = npgen()
    for d, m, C, B in [(2, 3, 2, 4), (4, 5, 3, 6)]:
        p = {k: np.zeros(s) for k, s in shapes_of(d, m, C).items()}
        X = np.round(gen.normal(size=(B, d)), 2) + 0.0
        y = gen.integers(0, C, size=B)
        check_fb(sol, p, X, y)


@case("Part 1", title="logits 很大（±1000）：数值稳定")
def p1_large_logits(sol):
    gen = npgen()
    for shift in (1000.0, -1000.0):
        p, X, y = ref.random_net(gen, d=3, m=4, C=3, B=5, margin=1e-3, min_prob=1e-3, shift=shift)
        check_fb(sol, p, X, y, label=f"# b2 的每一项都加了 {shift:+g}，softmax 的结果不变\n")


@case("Part 1", title="梯度对上你自己 loss 的中心差分")
def p1_finite_difference(sol):
    gen = npgen()
    for d, m, C, B in [(3, 4, 3, 5), (1, 3, 2, 1), (4, 6, 5, 7)]:
        p, X, y = ref.random_net(gen, d=d, m=m, C=C, B=B, margin=1e-3, min_prob=1e-3)
        call = (f"{ref.setup_code(p, X, y)}\nloss, grads = forward_backward(params, X, y)\n"
                "# 每一项梯度都和你自己的 loss 的中心差分 (L(θ+1e-5) - L(θ-1e-5)) / 2e-5 比较")
        _, grads = unpack(sol.forward_backward(cp(p), X.copy(), y.copy()), call)
        numeric = ref.numeric_grads(sol.forward_backward, p, X, y)
        for k in NAMES:
            close_or_fail(grads[k], numeric[k], f"grads['{k}']（和中心差分比）", call, **FD_TOL)


# ---------------------------------------------------------------- Part 2: grad_check
# grad_check is defined against forward_backward. Apart from the sample, these cases replace sol.forward_backward by a
# version with an exact loss and one analytic gradient entry deliberately wrong, so the expected return value is known
# from the statement's formula: a gradient entry g reported as 3g has rel_err |3g - g| / (|3g| + |g|) = 0.5.

GC_RTOL, GC_ATOL = 1e-4, 1e-6     # finite-difference noise moves rel_err by about 1e-9 here


def run_grad_check(sol, fb, params, X, y, call, eps=None):
    original = sol.forward_backward
    sol.forward_backward = fb
    try:
        args = (cp(params), X.copy(), y.copy())
        got = sol.grad_check(*args) if eps is None else sol.grad_check(*args, eps=eps)
    finally:
        sol.forward_backward = original
    if np.ndim(got) != 0:
        raise Mismatch(f"grad_check 应当返回一个数，得到形状 {np.shape(got)}", call, None, got)
    try:
        return float(got)
    except Exception:
        raise Mismatch(f"grad_check 应当返回一个数，得到 {type(got).__name__}", call, None, got) from None


def check_bugged(sol, params, X, y, name, idx, factor, eps=None, note=""):
    fb = ref.with_bug(name, idx, factor)
    want = ref.grad_check(fb, params, X, y, 1e-5 if eps is None else eps)
    call = (f"{ref.setup_code(params, X, y)}\n{ref.bug_code(name, idx, factor)}\n"
            f"grad_check(params, X, y{'' if eps is None else f', eps={eps!r}'}){note}")
    got = run_grad_check(sol, fb, params, X, y, call, eps)
    close_or_fail(got, want, "grad_check 的返回值", call, rtol=GC_RTOL, atol=GC_ATOL)


def largest_entry(grad):
    return tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(grad)), grad.shape))


@case("Part 2", sample=True, title="题目例子：结果小于 1e-7")
def p2_example(sol):
    call = f"{EX_SETUP}\ngrad_check(params, X, y)"
    got = sol.grad_check(cp(EX_P), EX_X.copy(), EX_Y.copy())
    if np.ndim(got) != 0:
        raise Mismatch(f"grad_check 应当返回一个数，得到形状 {np.shape(got)}", call, None, got)
    got = float(got)
    expect_true(np.isfinite(got) and 0 <= got < 1e-7,
                f"在题目例子上 grad_check 应当小于 1e-7（经验法则里的“通过”），得到 {got:.3g}；"
                "如果 Part 1 的梯度还有错，这里也会超标", call, got)


@case("Part 2", title="分母是 |g_a| + |g_n|")
def p2_denominator(sol):
    _, g = ref.forward_backward(EX_P, EX_X, EX_Y)
    idx = largest_entry(g["W2"])
    check_bugged(sol, EX_P, EX_X, EX_Y, "W2", idx, 1.1,
                 note="\n# 期望 0.1 / 2.1 ≈ 0.047619：分母是 |g_a| + |g_n|，不是 max(|g_a|, |g_n|) 或 |g_n|")


@case("Part 2", title="四个参数数组都要检查")
def p2_every_array(sol):
    p, X, y = ref.random_net(npgen(), d=3, m=4, C=3, B=5, margin=1e-3, min_prob=1e-3, min_grad=1e-3, live=True)
    _, g = ref.forward_backward(p, X, y)
    for name in NAMES:
        check_bugged(sol, p, X, y, name, largest_entry(g[name]), 3.0,
                     note=f"\n# 只有 grads['{name}'] 的一项被乘以 3，期望 |3g - g| / (|3g| + |g|) = 0.5")


@case("Part 2", title="b2 梯度整体乘以 B")
def p2_bias_times_batch(sol):
    p, X, y = ref.random_net(npgen(), d=2, m=3, C=4, B=6, margin=1e-3, min_prob=1e-3, min_grad=1e-3)
    check_bugged(sol, p, X, y, "b2", None, 6.0, note="\n# 期望 (B - 1) / (B + 1) = 5 / 7")


@case("Part 2", title="用传入的 eps 做中心差分")
def p2_eps(sol):
    # With eps = 1e-2 the central difference has a visible O(eps^2) truncation error, so the result (about 1e-5 to
    # 1e-4 here) pins down both the eps that was used and the two-sided formula: a one-sided difference is O(eps),
    # about 100 times larger, and ignoring eps gives about 1e-10. Pre-activations stay 0.2 away from the kink, more
    # than eps times any input, and every nonzero gradient is at least 1e-2, so rounding stays far below GC_RTOL.
    gen = npgen()
    for d, m, C, B in [(2, 3, 2, 4), (3, 2, 3, 3)]:
        p, X, y = ref.random_net(gen, d=d, m=m, C=C, B=B, scale=1.0, margin=0.2, min_prob=1e-2, min_grad=1e-2)
        want = ref.grad_check(ref.forward_backward, p, X, y, 1e-2)
        call = f"{ref.setup_code(p, X, y)}\ngrad_check(params, X, y, eps=0.01)   # 期望 {want:.6g}"
        got = run_grad_check(sol, ref.forward_backward, p, X, y, call, eps=1e-2)
        close_or_fail(got, want, "grad_check 的返回值", call, rtol=GC_RTOL, atol=1e-12)


@case("Part 2", title="随机网络，某一项梯度错成 3 倍")
def p2_random(sol):
    gen = npgen()
    for _ in range(6):
        d, m, C, B = int(gen.integers(1, 5)), int(gen.integers(1, 6)), int(gen.integers(2, 5)), int(gen.integers(1, 7))
        p, X, y = ref.random_net(gen, d=d, m=m, C=C, B=B, margin=1e-3, min_prob=1e-3, min_grad=1e-3)
        _, g = ref.forward_backward(p, X, y)
        choices = [(k, i) for k in NAMES for i in np.ndindex(g[k].shape) if g[k][i] != 0]
        name, idx = choices[int(gen.integers(len(choices)))]
        check_bugged(sol, p, X, y, name, tuple(int(i) for i in idx), 3.0,
                     note="\n# 期望 |3g - g| / (|3g| + |g|) = 0.5")


# ---------------------------------------------------------------- Part 3: data, initialisation, training

# (make_spirals seed, he_init seed, train seed): the page's own seeds first, then independent retries.
PAGE_RUNS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]


def page_call(init, seeds):
    s, i, t = seeds
    return (f"X, y = make_spirals(150, n_turns=1.5, noise=0.08, seed={s})\n"
            f"params = {init if init != 'he' else f'he_init(2, 64, 2, seed={i})'}\n"
            f"trained = train(params, X, y, steps=4000, batch_size=32, lr=0.15, momentum=0.9, seed={t})\n"
            "# 训练准确率：argmax(S) 与 y 在同一份 X、y 上对比")


def spirals(sol, n, turns, noise, seed, call):
    out = sol.make_spirals(n, turns, noise, seed)
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise Mismatch("make_spirals 应当返回 (X, y) 两个值", call, "(X, y)", out)
    X = as_float_array(out[0], "X", call)
    y = np.asarray(out[1])
    expect_true(X.shape == (2 * n, 2), f"X 的形状应当是 {(2 * n, 2)}，得到 {X.shape}", call, out[0])
    expect_true(y.shape == (2 * n,), f"y 的形状应当是 {(2 * n,)}，得到 {y.shape}", call, out[1])
    return X, y


@case("Part 3", sample=True, cpu=12, title="题目例子：He 初始化练到 95%")
def p3_example_he(sol):
    # The page reaches 99% with these seeds, but the result depends on how an answer consumes its random numbers, and
    # over 200 other seed triples about 9% of the reference's runs land below 95% (lowest 0.90). So a run below 95%
    # is retried with independent seeds; a correct answer fails all four with probability about 1e-4.
    accs = []
    for seeds in PAGE_RUNS:
        call = page_call("he", seeds)
        X, y = spirals(sol, 150, 1.5, 0.08, seeds[0], call)
        params = sol.he_init(2, 64, 2, seeds[1])
        check_params_dict(params, shapes_of(2, 64, 2), "he_init", call)
        trained = sol.train(cp(params), X.copy(), y.copy(), steps=4000, batch_size=32, lr=0.15, momentum=0.9,
                            seed=seeds[2])
        check_params_dict(trained, shapes_of(2, 64, 2), "train", call)
        acc = ref.accuracy(cp(trained), X, y.astype(int))
        accs.append(acc)
        info(f"种子 {seeds}：训练准确率 {acc:.4f}")
        if acc >= 0.95:
            return
    raise Mismatch(f"{len(PAGE_RUNS)} 组种子的训练准确率都低于 95%：{', '.join(f'{a:.4f}' for a in accs)}",
                   page_call("he", PAGE_RUNS[0]), ">= 0.95", max(accs))


@case("Part 3", sample=True, cpu=6, title="题目例子：全零初始化停在 50%")
def p3_example_zero(sol):
    call = page_call("zero_init(2, 64, 2)", PAGE_RUNS[0])
    X, y = spirals(sol, 150, 1.5, 0.08, 0, call)
    params = sol.zero_init(2, 64, 2)
    check_params_dict(params, shapes_of(2, 64, 2), "zero_init", call)
    trained = sol.train(cp(params), X.copy(), y.copy(), steps=4000, batch_size=32, lr=0.15, momentum=0.9, seed=2)
    check_params_dict(trained, shapes_of(2, 64, 2), "train", call)
    for k in ("W1", "b1", "W2"):
        a = np.asarray(trained[k], dtype=np.float64)
        expect_true(np.all(a == 0.0), f"训练后 {k} 应当逐位等于 0，最大绝对值是 {np.max(np.abs(a)):.3g}", call, trained[k])
    acc = ref.accuracy(cp(trained), X, y.astype(int))
    expect_true(acc == 0.5, f"训练准确率应当恰好是 0.5，得到 {acc}", call, acc)


@case("Part 3", title="make_spirals：形状和标签")
def p3_spirals_labels(sol):
    gen = npgen()
    for n in (1, 7, 100):
        seed = int(gen.integers(1 << 30))
        call = f"X, y = make_spirals({n}, n_turns=1.5, noise=0.1, seed={seed})"
        _, y = spirals(sol, n, 1.5, 0.1, seed, call)
        expect_true(np.issubdtype(y.dtype, np.integer), f"y 是整数类别标签，dtype 应当是整数，得到 {y.dtype}", call, y)
        want = np.repeat([0, 1], n)
        expect_true(np.array_equal(y, want), f"y 应当是 {n} 个 0 后面跟 {n} 个 1", call, y)


@case("Part 3", title="make_spirals：noise=0 时点都在臂上")
def p3_spirals_arms(sol):
    # Without noise every class-0 point is (r cos t, r sin t) with r = t / T, T = n_turns * 2 * pi, so r = |p| and
    # t = r * T are read back from the point itself; class-1 points are the same after rotating back by pi.
    gen = npgen()
    for turns in (1.5, 2.25, 0.75):
        n, seed = 400, int(gen.integers(1 << 30))
        call = f"X, y = make_spirals({n}, n_turns={turns}, noise=0.0, seed={seed})"
        X, _ = spirals(sol, n, turns, 0.0, seed, call)
        T = turns * 2 * np.pi
        for cls, pts in ((0, X[:n]), (1, -X[n:])):
            r = np.hypot(pts[:, 0], pts[:, 1])
            expect_true(r.max() <= 1 + 1e-9, f"类别 {cls} 有点的半径 r = {r.max():.6g} > 1", call, X)
            on_arm = np.stack([r * np.cos(r * T), r * np.sin(r * T)], axis=1)
            off = np.hypot(*(pts - on_arm).T)
            i = int(np.argmax(off))
            expect_true(off[i] < 1e-9,
                        f"类别 {cls} 的第 {i} 个点 {tuple(round(float(v), 6) for v in X[i + cls * n])} 不在"
                        f"{'臂' if cls == 0 else '转了 π 的臂'}上（偏离 {off[i]:.3g}）", call, X)


@case("Part 3", title="make_spirals：t 正比于 sqrt(u)")
def p3_spirals_density(sol):
    # t = T * sqrt(u) means r = sqrt(u), so r^2 is uniform on [0, 1). Kolmogorov-Smirnov distance of r^2 from
    # uniform, n = 4000: a correct answer exceeds 0.06 with probability about 1e-12; t = T * u gives about 0.25.
    seed = int(npgen().integers(1 << 30))
    n = 4000
    call = f"X, y = make_spirals({n}, n_turns=1.5, noise=0.0, seed={seed})"
    X, _ = spirals(sol, n, 1.5, 0.0, seed, call)
    r2 = np.sort(np.sum(X[:n] ** 2, axis=1))
    k = np.arange(1, n + 1)
    D = float(max(np.max(k / n - r2), np.max(r2 - (k - 1) / n)))
    expect_true(D < 0.06, f"类别 0 的 r² 应当在 [0, 1) 上均匀分布（t = n_turns·2π·sqrt(u)），"
                          f"但 KS 距离是 {D:.3f}（均匀时大约 0.01）", call, D)


@case("Part 3", title="make_spirals：噪声 N(0, noise²)")
def p3_spirals_noise(sol):
    # Independent noise with variance noise^2 per coordinate adds 2 * noise^2 to E|x|^2, whatever the clean points are.
    # Compared with a noise-free call (different seed), per class and with 10000 points each, the difference of the
    # means has a standard error of about 0.005; the tolerance is 0.04.
    gen = npgen()
    s1, s2 = int(gen.integers(1 << 30)), int(gen.integers(1 << 30))
    n, noise = 10000, 0.3
    call = (f"X, y = make_spirals({n}, n_turns=1.5, noise={noise}, seed={s1})\n"
            f"X0, y0 = make_spirals({n}, n_turns=1.5, noise=0.0, seed={s2})\n"
            "# 每个类别比较 |x|² 的平均值")
    X, _ = spirals(sol, n, 1.5, noise, s1, call)
    X0, _ = spirals(sol, n, 1.5, 0.0, s2, call)
    want = 2 * noise ** 2
    for cls in (0, 1):
        part = slice(0, n) if cls == 0 else slice(n, 2 * n)
        got = float(np.mean(np.sum(X[part] ** 2, axis=1)) - np.mean(np.sum(X0[part] ** 2, axis=1)))
        expect_true(abs(got - want) < 0.04,
                    f"类别 {cls}：加噪声后 |x|² 的平均值应当比 noise=0 时大约多 2·noise² = {want:.3f}，"
                    f"实际多了 {got:.3f}", call, got)


@case("Part 3", title="make_spirals / he_init：同一个种子结果相同")
def p3_seeded(sol):
    seed = int(npgen().integers(1 << 30))
    call = f"make_spirals(20, 1.5, 0.1, seed={seed}) 调用两次"
    a, b = sol.make_spirals(20, 1.5, 0.1, seed), sol.make_spirals(20, 1.5, 0.1, seed)
    expect_true(np.array_equal(np.asarray(a[0]), np.asarray(b[0])), "同一个 seed 两次调用 make_spirals，X 不一样", call)
    call = f"he_init(3, 5, 2, seed={seed}) 调用两次"
    a, b = sol.he_init(3, 5, 2, seed), sol.he_init(3, 5, 2, seed)
    check_params_dict(a, shapes_of(3, 5, 2), "he_init", call)
    for k in NAMES:
        expect_true(np.array_equal(np.asarray(a[k]), np.asarray(b[k])), f"同一个 seed 两次调用 he_init，{k} 不一样", call)


@case("Part 3", title="he_init：形状和零偏置")
def p3_he_shapes(sol):
    for d, m, C in [(2, 64, 2), (3, 5, 4), (1, 1, 2)]:
        call = f"he_init({d}, {m}, {C}, seed=0)"
        p = sol.he_init(d, m, C, 0)
        check_params_dict(p, shapes_of(d, m, C), "he_init", call)
        for k in ("b1", "b2"):
            a = np.asarray(p[k], dtype=np.float64)
            expect_true(np.all(a == 0.0), f"{k} 应当全为 0", call, p[k])


@case("Part 3", title="he_init：W ~ N(0, 2 / fan_in)")
def p3_he_stats(sol):
    # d, m, C all differ, so fan_in is distinguishable from fan_out. With 120000 and 60000 entries the sample variance
    # is within 0.6% of the truth (one standard error) and the tolerance is 5%; the sample kurtosis of a normal is
    # 3 +- 0.02 (uniform gives 1.8) and the tolerance is 0.2.
    d, m, C = 400, 300, 200
    seed = int(npgen().integers(1 << 30))
    call = f"params = he_init({d}, {m}, {C}, seed={seed})"
    p = sol.he_init(d, m, C, seed)
    check_params_dict(p, shapes_of(d, m, C), "he_init", call)
    for k, fan_in in (("W1", d), ("W2", m)):
        w = np.asarray(p[k], dtype=np.float64).ravel()
        var = 2.0 / fan_in
        mean, v = float(w.mean()), float(w.var())
        expect_true(abs(v / var - 1) < 0.05,
                    f"{k} 的方差应当是 2 / fan_in = 2 / {fan_in} = {var:.4g}，样本方差是 {v:.4g}", call, v)
        expect_true(abs(mean) < 6 * np.sqrt(var / w.size), f"{k} 的均值应当是 0，样本均值是 {mean:.3g}", call, mean)
        kurt = float(np.mean((w - mean) ** 4) / v ** 2)
        expect_true(2.8 < kurt < 3.2, f"{k} 应当服从正态分布（峰度 3），样本峰度是 {kurt:.3f}", call, kurt)


@case("Part 3", title="zero_init：全是 0.0")
def p3_zero_init(sol):
    for d, m, C in [(2, 64, 2), (3, 5, 4)]:
        call = f"zero_init({d}, {m}, {C})"
        p = sol.zero_init(d, m, C)
        check_params_dict(p, shapes_of(d, m, C), "zero_init", call)
        for k in NAMES:
            a = np.asarray(p[k])
            expect_true(np.issubdtype(a.dtype, np.floating), f"{k} 应当是浮点数 0.0，dtype 得到 {a.dtype}", call, p[k])
            expect_true(np.all(a == 0.0), f"{k} 应当全为 0.0", call, p[k])


@case("Part 3", title="train：动量更新逐步对上")
def p3_train_update(sol):
    # Every row of X is the same example, so every mini-batch has that example's gradient however it is drawn, and
    # the result is fixed by the update rule alone: v <- momentum * v - lr * grad, theta <- theta + v, v0 = 0.
    gen = npgen()
    for steps, bs, lr, mom in [(25, 4, 0.1, 0.9), (10, 1, 0.5, 0.0), (3, 3, 0.2, 0.5)]:
        p, X1, y1 = ref.random_net(gen, d=3, m=5, C=4, B=1, margin=1e-3, min_prob=1e-3, live=True)
        X, y = np.repeat(X1, 5, axis=0), np.repeat(y1, 5)
        want = ref.train_same_example(p, X1[0], int(y1[0]), steps, lr, mom)
        seed = int(gen.integers(1000))
        call = (f"{ref.setup_code(p, X, y)}\n"
                f"trained = train(params, X, y, steps={steps}, batch_size={bs}, lr={lr}, momentum={mom}, seed={seed})\n"
                "# X 的每一行都是同一个样本，所以不论怎么抽，每一批的梯度都一样")
        got = sol.train(cp(p), X.copy(), y.copy(), steps=steps, batch_size=bs, lr=lr, momentum=mom, seed=seed)
        check_params_dict(got, shapes_of(3, 5, 4), "train", call)
        for k in NAMES:
            close_or_fail(got[k], want[k], f"训练后的 {k}", call, rtol=1e-7, atol=1e-10)


@case("Part 3", title="train：有放回地均匀抽小批量")
def p3_train_sampling(sol):
    # Two different examples a, b and batch_size = 3: after one step with momentum 0 and lr 1 the update is
    # -(k/3 * g_a + (3-k)/3 * g_b), where k is how many times a was drawn. k must be an integer in 0..3 (a full-batch
    # step gives k = 1.5; drawing without replacement cannot fill a batch of 3 from 2 examples), and over 40 seeds
    # the average k/3 must be near 1/2 (standard error 0.046, tolerance 0.25).
    gen = npgen()
    while True:
        p, X, y = ref.random_net(gen, d=2, m=3, C=2, B=2, margin=1e-3, min_prob=1e-2, live=True)
        _, ga = ref.forward_backward(p, X[:1], y[:1])
        _, gb = ref.forward_backward(p, X[1:], y[1:])
        flat_a = np.concatenate([ga[k].ravel() for k in NAMES])
        flat_b = np.concatenate([gb[k].ravel() for k in NAMES])
        if np.linalg.norm(flat_a - flat_b) > 0.05:
            break
    theta0 = np.concatenate([p[k].ravel() for k in NAMES])
    diff = flat_a - flat_b
    fractions = []
    for seed in range(40):
        call = (f"{ref.setup_code(p, X, y)}\n"
                f"trained = train(params, X, y, steps=1, batch_size=3, lr=1.0, momentum=0.0, seed={seed})")
        got = sol.train(cp(p), X.copy(), y.copy(), steps=1, batch_size=3, lr=1.0, momentum=0.0, seed=seed)
        check_params_dict(got, shapes_of(2, 3, 2), "train", call)
        step = -(np.concatenate([np.asarray(got[k], dtype=np.float64).ravel() for k in NAMES]) - theta0)
        alpha = float(np.dot(step - flat_b, diff) / np.dot(diff, diff))
        resid = float(np.max(np.abs(step - (alpha * flat_a + (1 - alpha) * flat_b))))
        expect_true(resid < 1e-9 and abs(3 * alpha - round(3 * alpha)) < 1e-6 and -1e-6 < alpha < 1 + 1e-6,
                    "这一步的更新不是“3 个有放回抽到的样本”的平均梯度：" +
                    (f"更新量相当于样本 0 占 {alpha:.4f}、样本 1 占 {1 - alpha:.4f}（应当是 0、1/3、2/3 或 1）"
                     if resid < 1e-9 else "它不是这两个样本梯度的加权平均"), call, got)
        fractions.append(alpha)
    mean = float(np.mean(fractions))
    expect_true(abs(mean - 0.5) < 0.25,
                f"40 个种子里样本 0 平均占批次的 {mean:.3f}，均匀抽样时应当约为 0.5",
                "for seed in range(40): train(params, X, y, steps=1, batch_size=3, lr=1.0, momentum=0.0, seed=seed)",
                fractions)
