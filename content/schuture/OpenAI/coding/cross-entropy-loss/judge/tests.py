import math

import numpy as np

from judgelib import Mismatch, case, expect_close, expect_true, rng
import ref

# Tolerances. The reference computes every position in float64 with an exactly rounded sum (math.fsum); a correct
# float64 answer differs from it only by rounding, a few ulp of the largest logit (|z| <= 1000 here), i.e. below
# 1e-12. The bugs these tests target move the result by 1e-2 or more (or to nan / inf).
LOSS_TOL = dict(rtol=1e-9, atol=1e-9)
GRAD_TOL = dict(rtol=1e-9, atol=1e-12)
# The statement's examples are printed to 4 decimals ("约 0.2152"): exact values 0.215207959 and 0.645207959.
EXAMPLE_TOL = dict(rtol=0, atol=5e-5)
# Central differences with h = 1e-6 in float64: truncation h^2/6 |l'''| ~ 1e-13, rounding ~ 2.2e-16 |l| / h ~ 1e-10.
FD_H = 1e-6
FD_TOL = dict(rtol=1e-5, atol=1e-6)

EX_LOGITS = np.array([[[2.0, 1.0, 0.1], [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]]])
EX_TARGETS = np.array([[0, 2, 7]])
EX_MASK = np.array([[1, 1, 0]])


def npgen(salt=""):
    return np.random.default_rng(rng(salt).getrandbits(64))


def src(x):
    """An array as code the user can paste into 自定义运行, or a description when it is too long."""
    x = np.asarray(x)
    text = f"np.array({x.tolist()!r})"
    return text if len(text) <= 900 else f"<形状 {x.shape} 的数组>"


def scalar(got, call):
    """The returned loss as a Python float; the statement asks for one number (the mean)."""
    if np.ndim(got) != 0:
        raise Mismatch(f"应当返回一个数（所有计入位置的均值），得到的是形状 {np.shape(got)} 的数组", call, None, got)
    try:
        return float(got)
    except Exception:
        raise Mismatch(f"应当返回一个数，得到的是 {type(got).__name__}", call, None, got) from None


def hint(value, alternatives):
    """A sentence naming the mistake when `value` equals what a known wrong formula gives."""
    for text, alt in alternatives:
        if alt is not None and math.isfinite(alt) and math.isclose(value, alt, rel_tol=1e-9, abs_tol=1e-9):
            return text
    return ""


def check_loss(got, want, call, alternatives=(), tol=LOSS_TOL):
    value = scalar(got, call)
    if not math.isfinite(value):
        raise Mismatch(f"结果是 {value}，期望有限值 {want:.10g}" + hint_stability(), call, want, value)
    if not math.isclose(value, want, rel_tol=tol["rtol"], abs_tol=tol["atol"]):
        extra = hint(value, alternatives)
        raise Mismatch(f"损失不对：期望 {want:.10g}，得到 {value:.10g}" + (f"。{extra}" if extra else ""),
                       call, want, value)


def hint_stability():
    return "（softmax 或 log 溢出、下溢了？）"


def as_grad(got, shape, call):
    try:
        a = np.asarray(got, dtype=np.float64)
    except Exception:
        raise Mismatch(f"应当返回形状 {shape} 的数值数组，得到 {type(got).__name__}", call, None, got) from None
    if a.shape != shape:
        raise Mismatch(f"梯度的形状应当和 logits 一样是 {shape}，得到 {a.shape}", call, None, got)
    return a


def call_ce(logits, targets):
    return f"cross_entropy({src(logits)}, {src(targets)})"


def call_masked(logits, targets, mask):
    return f"cross_entropy_masked({src(logits)}, {src(targets)}, {src(mask)})"


def call_smoothed(logits, targets, mask, eps):
    return f"cross_entropy_smoothed({src(logits)}, {src(targets)}, {src(mask)}, {eps!r})"


def call_grad(logits, targets, mask, eps):
    return f"cross_entropy_grad({src(logits)}, {src(targets)}, {src(mask)}, {eps!r})"


def run_ce(sol, logits, targets, alternatives=()):
    want = ref.plain(logits, targets)
    check_loss(sol.cross_entropy(logits.copy(), targets.copy()), want, call_ce(logits, targets), alternatives)


def denominators(logits, targets, mask, eps=0.0):
    total, count = ref.position_losses(logits, targets, mask, eps)
    return [(f"这是计入位置的损失之和除以 B×T = {targets.size}：分母应当是 mask == 1 的位置个数 {count}", total / targets.size),
            ("这是计入位置的损失之和，没有除以位置个数", total)]


def run_masked(sol, logits, targets, mask):
    want = ref.masked(logits, targets, mask)
    alts = denominators(logits, targets, mask) if np.count_nonzero(mask) else ()
    check_loss(sol.cross_entropy_masked(logits.copy(), targets.copy(), mask.copy()), want,
               call_masked(logits, targets, mask), alts)


def smoothing_alternatives(logits, targets, mask, eps):
    V = logits.shape[-1]
    alts = denominators(logits, targets, mask, eps) if np.count_nonzero(mask) else []
    if V > 1 and np.count_nonzero(mask):
        # eps spread over the V - 1 other classes only: p_y = 1 - eps, p_v = eps / (V - 1)
        total, count = ref.position_losses(logits, targets, mask, eps * V / (V - 1))
        alts = list(alts) + [("这是把 ε 只分给其余 V - 1 个类别的结果；题目的约定是 ε/V 分给全部 V 个类别（包括目标类）",
                              total / count)]
    return alts


def run_smoothed(sol, logits, targets, mask, eps):
    want = ref.smoothed(logits, targets, mask, eps)
    check_loss(sol.cross_entropy_smoothed(logits.copy(), targets.copy(), mask.copy(), eps), want,
               call_smoothed(logits, targets, mask, eps), smoothing_alternatives(logits, targets, mask, eps))


def run_grad(sol, logits, targets, mask, eps):
    want = ref.grad(logits, targets, mask, eps)
    call = call_grad(logits, targets, mask, eps)
    got = as_grad(sol.cross_entropy_grad(logits.copy(), targets.copy(), mask.copy(), eps), logits.shape, call)
    keep = np.asarray(mask).astype(bool)
    if not np.isfinite(got).all():
        idx = tuple(int(i) for i in np.argwhere(~np.isfinite(got))[0])
        raise Mismatch(f"梯度里有 {got[idx]}（位置 {idx}）" + hint_stability(), call, want, got)
    if got[~keep].any():
        b, t = (int(i) for i in np.argwhere(~keep & got.any(axis=-1))[0])
        raise Mismatch(f"位置 ({b}, {t}) 被 mask 了，它对损失没有贡献，梯度这一行应当全为 0，得到 {got[b, t].tolist()}",
                       call, want, got)
    count = int(keep.sum())
    if count and not np.allclose(got, want, **GRAD_TOL):
        for text, factor in ((f"梯度整体只有正确值的 {count}/{keep.size}：分母应当是未被 mask 的位置个数 {count}，不是 B×T = {keep.size}",
                              count / keep.size),
                             (f"梯度整体是正确值的 {count} 倍：均值损失的梯度要除以未被 mask 的位置个数 {count}", count)):
            if factor != 1 and np.allclose(got, want * factor, **GRAD_TOL):
                raise Mismatch(text, call, want, got)
    expect_close(got, want, call=call, **GRAD_TOL)


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子：约 0.2152")
def p1_example(sol):
    logits, targets = EX_LOGITS[:, :2].copy(), np.array([[0, 2]])
    check_loss(sol.cross_entropy(logits, targets), 0.2152, call_ce(EX_LOGITS[:, :2], np.array([[0, 2]])),
               tol=EXAMPLE_TOL)


@case("Part 1", title="logits 达到 ±1000：有限且正确")
def p1_large_logits(sol):
    row = [1000.0, 0.0, -1000.0]
    for y, want in ((0, 0.0), (1, 1000.0), (2, 2000.0)):
        logits, targets = np.array([[row]]), np.array([[y]])
        check_loss(sol.cross_entropy(logits.copy(), targets.copy()), want, call_ce(logits, targets))
    logits, targets = np.array([[row, row, row]]), np.array([[0, 1, 2]])
    check_loss(sol.cross_entropy(logits.copy(), targets.copy()), 1000.0, call_ce(logits, targets))
    for value in (1000.0, -1000.0):
        logits, targets = np.full((1, 1, 4), value), np.array([[3]])
        check_loss(sol.cross_entropy(logits.copy(), targets.copy()), math.log(4), call_ce(logits, targets))


@case("Part 1", title="同一批里有的行在 1000 附近，有的在 -1000 附近")
def p1_mixed_scales(sol):
    logits = np.array([[[1000.0, 998.0, 990.0], [-1000.0, -1001.0, -1003.0]],
                       [[0.5, -0.5, 0.0], [-995.0, -1000.0, -999.0]]])
    for targets in (np.array([[1, 0], [2, 2]]), np.array([[0, 2], [1, 1]])):
        run_ce(sol, logits, targets)


@case("Part 1", title="各种形状：B、T、V 不同或相等，V = 1")
def p1_shapes(sol):
    gen = npgen()
    for B, T, V in [(1, 1, 1), (1, 1, 5), (2, 3, 3), (3, 3, 3), (1, 4, 4), (4, 2, 2), (2, 1, 7), (3, 4, 1)]:
        logits = np.round(gen.normal(scale=4.0, size=(B, T, V)), 2) + 0.0
        run_ce(sol, logits, gen.integers(0, V, size=(B, T)))


@case("Part 1", title="随机批次，对比参考实现")
def p1_random(sol):
    gen = npgen()
    for _ in range(40):
        B, T, V = (int(v) for v in gen.integers(1, [5, 6, 9]))
        scale = float(gen.choice([0.5, 4.0, 50.0, 300.0]))
        logits = np.clip(np.round(gen.normal(scale=scale, size=(B, T, V)), 2), -1000.0, 1000.0) + 0.0
        run_ce(sol, logits, gen.integers(0, V, size=(B, T)))


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子：第三个位置不计入")
def p2_example(sol):
    check_loss(sol.cross_entropy_masked(EX_LOGITS.copy(), EX_TARGETS.copy(), EX_MASK.copy()), 0.2152,
               call_masked(EX_LOGITS, EX_TARGETS, EX_MASK), denominators(EX_LOGITS, EX_TARGETS, EX_MASK),
               tol=EXAMPLE_TOL)


@case("Part 2", title="分母是 mask == 1 的个数，不是 B×T")
def p2_denominator(sol):
    logits = np.array([[[2.0, 1.0, 0.1], [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]],
                       [[0.3, -1.2, 2.5], [4.0, 1.0, -2.0], [0.0, 3.0, 0.0]]])
    targets = np.array([[0, 2, 1], [2, 0, 1]])
    for mask in (np.array([[1, 1, 0], [1, 0, 0]]), np.array([[0, 0, 0], [0, 1, 0]]), np.array([[1, 0, 1], [0, 1, 1]])):
        run_masked(sol, logits, targets, mask)


@case("Part 2", title="被 mask 位置的 target 越界：负数、很大")
def p2_masked_junk_targets(sol):
    logits = np.array([[[0.5, 1.5, -1.0, 2.0], [3.0, 0.0, 0.0, 1.0]],
                       [[-2.0, 0.0, 1.0, 0.5], [1.0, 1.0, 1.0, 4.0]]])
    for junk in (-1, -100, -999, 4, 10 ** 9):
        targets = np.array([[1, junk], [junk, 3]])
        run_masked(sol, logits, targets, np.array([[1, 0], [0, 1]]))


@case("Part 2", title="0/1 整数 mask 与布尔 mask 结果相同")
def p2_int_and_bool_mask(sol):
    gen = npgen()
    for B, T, V in [(3, 2, 4), (2, 2, 3), (4, 3, 5)]:
        logits, targets, mask = ref.random_batch(gen, B, T, V)
        run_masked(sol, logits, targets, mask)
        run_masked(sol, logits, targets, mask.astype(bool))


@case("Part 2", title="全部位置都被 mask：返回 0.0")
def p2_all_masked(sol):
    gen = npgen()
    for mask in (np.zeros((2, 3), dtype=np.int64), np.zeros((2, 3), dtype=bool)):
        logits = np.round(gen.normal(scale=4.0, size=(2, 3, 4)), 2) + 0.0
        targets = gen.integers(-9, 9, size=(2, 3))
        call = call_masked(logits, targets, mask)
        value = scalar(sol.cross_entropy_masked(logits.copy(), targets.copy(), mask.copy()), call)
        expect_true(value == 0.0, f"没有任何位置计入时题目要求返回 0.0，得到 {value}", call, value)


@case("Part 2", title="mask 全为 1 时与 Part 1 相同")
def p2_all_kept(sol):
    gen = npgen()
    for B, T, V in [(2, 3, 4), (1, 5, 2)]:
        logits = np.round(gen.normal(scale=4.0, size=(B, T, V)), 2) + 0.0
        run_masked(sol, logits, gen.integers(0, V, size=(B, T)), np.ones((B, T), dtype=np.int64))


@case("Part 2", title="logits 达到 ±1000")
def p2_large_logits(sol):
    logits = np.array([[[1000.0, 0.0, -1000.0], [-1000.0, -999.0, -1000.0], [5.0, 1000.0, 1000.0]]])
    run_masked(sol, logits, np.array([[2, 0, -1]]), np.array([[1, 1, 0]]))
    run_masked(sol, logits, np.array([[1, 50, 2]]), np.array([[True, False, True]]))


@case("Part 2", title="随机批次与随机 mask，对比参考实现")
def p2_random(sol):
    gen = npgen()
    for k in range(40):
        B, T, V = (int(v) for v in gen.integers(1, [5, 6, 9]))
        logits, targets, mask = ref.random_batch(gen, B, T, V, bool_mask=bool(k % 2))
        run_masked(sol, logits, targets, mask)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目例子：ε = 0.2 时约 0.6452")
def p3_example(sol):
    check_loss(sol.cross_entropy_smoothed(EX_LOGITS.copy(), EX_TARGETS.copy(), EX_MASK.copy(), 0.2), 0.6452,
               call_smoothed(EX_LOGITS, EX_TARGETS, EX_MASK, 0.2),
               smoothing_alternatives(EX_LOGITS, EX_TARGETS, EX_MASK, 0.2), tol=EXAMPLE_TOL)


@case("Part 3", title="ε = 0 时与 cross_entropy_masked 一致")
def p3_eps_zero(sol):
    gen = npgen()
    for k in range(6):
        logits, targets, mask = ref.random_batch(gen, 2, 3, 4 + k, bool_mask=bool(k % 2))
        call = call_smoothed(logits, targets, mask, 0.0)
        mine = scalar(sol.cross_entropy_masked(logits.copy(), targets.copy(), mask.copy()),
                      call_masked(logits, targets, mask))
        got = scalar(sol.cross_entropy_smoothed(logits.copy(), targets.copy(), mask.copy(), 0.0), call)
        expect_true(math.isclose(got, mine, rel_tol=1e-12, abs_tol=1e-12),
                    f"label_smoothing = 0.0 时应当与你的 cross_entropy_masked 结果一致：那边是 {mine!r}，这里是 {got!r}",
                    call, got)
        run_smoothed(sol, logits, targets, mask, 0.0)


@case("Part 3", title="ε/V 分给全部 V 个类别，包括目标类")
def p3_convention(sol):
    logits = np.array([[[1.0, 2.0, 0.5, -1.0, 0.0], [0.3, 0.3, 2.2, 1.1, -0.4]]])
    targets = np.array([[1, 3]])
    mask = np.array([[1, 1]])
    for eps in (0.1, 0.5, 0.9):
        run_smoothed(sol, logits, targets, mask, eps)
    two = np.array([[[0.0, 1.0]], [[2.0, -1.0]]])
    run_smoothed(sol, two, np.array([[0], [0]]), np.array([[1], [1]]), 0.3)


@case("Part 3", title="logits 达到 ±1000：有限且正确")
def p3_large_logits(sol):
    logits, mask = np.array([[[1000.0, 0.0, -1000.0]]]), np.array([[1]])
    for y, eps, want in ((0, 0.3, 300.0), (2, 0.0, 2000.0), (1, 0.6, 1000.0)):
        targets = np.array([[y]])
        check_loss(sol.cross_entropy_smoothed(logits.copy(), targets.copy(), mask.copy(), eps), want,
                   call_smoothed(logits, targets, mask, eps))
    mixed = np.array([[[1000.0, 998.0, 990.0], [-1000.0, -1001.0, -1003.0]],
                      [[0.5, -0.5, 0.0], [-995.0, -1000.0, -999.0]]])
    for eps in (0.0, 0.1):
        run_smoothed(sol, mixed, np.array([[1, 0], [-5, 2]]), np.array([[1, 1], [0, 1]]), eps)


@case("Part 3", title="mask：越界 target、全部被 mask")
def p3_masking(sol):
    logits = np.array([[[0.5, 1.5, -1.0, 2.0], [3.0, 0.0, 0.0, 1.0]],
                       [[-2.0, 0.0, 1.0, 0.5], [1.0, 1.0, 1.0, 4.0]]])
    run_smoothed(sol, logits, np.array([[1, -100], [10 ** 9, 3]]), np.array([[True, False], [False, True]]), 0.25)
    for mask in (np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=bool)):
        targets = np.array([[-1, 7], [2, -100]])
        call = call_smoothed(logits, targets, mask, 0.25)
        value = scalar(sol.cross_entropy_smoothed(logits.copy(), targets.copy(), mask.copy(), 0.25), call)
        expect_true(value == 0.0, f"没有任何位置计入时题目要求返回 0.0，得到 {value}", call, value)


@case("Part 3", title="V = 1：损失恒为 0")
def p3_single_class(sol):
    logits = np.array([[[3.5], [-2.0]], [[1000.0], [0.0]]])
    run_smoothed(sol, logits, np.array([[0, 0], [0, 4]]), np.array([[1, 1], [1, 0]]), 0.4)


@case("Part 3", title="随机批次、随机 ε，对比参考实现")
def p3_random(sol):
    gen = npgen()
    for k in range(40):
        B, T, V = (int(v) for v in gen.integers(1, [5, 6, 9]))
        logits, targets, mask = ref.random_batch(gen, B, T, V, bool_mask=bool(k % 2))
        eps = float(gen.choice([0.0, round(float(gen.uniform(0, 0.4)), 3), round(float(gen.uniform(0.4, 0.99)), 3)]))
        run_smoothed(sol, logits, targets, mask, eps)


# ---------------------------------------------------------------- Part 4 (c)

@case("Part 4", sample=True, title="题目要求：梯度对上中心差分")
def p4_finite_difference(sol):
    eps = 0.2
    call = (f"logits = {src(EX_LOGITS)}; targets = {src(EX_TARGETS)}; mask = {src(EX_MASK)}\n"
            f"cross_entropy_grad(logits, targets, mask, {eps})\n"
            f"# 每个元素都和你自己的 cross_entropy_smoothed 的中心差分 (ℓ(z+h) - ℓ(z-h)) / 2h（h = {FD_H:g}）比较")
    got = as_grad(sol.cross_entropy_grad(EX_LOGITS.copy(), EX_TARGETS.copy(), EX_MASK.copy(), eps),
                  EX_LOGITS.shape, call)
    numeric = np.zeros(EX_LOGITS.shape)
    for idx in np.ndindex(EX_LOGITS.shape):
        plus, minus = EX_LOGITS.copy(), EX_LOGITS.copy()
        plus[idx] += FD_H
        minus[idx] -= FD_H
        f_plus = scalar(sol.cross_entropy_smoothed(plus, EX_TARGETS.copy(), EX_MASK.copy(), eps), call)
        f_minus = scalar(sol.cross_entropy_smoothed(minus, EX_TARGETS.copy(), EX_MASK.copy(), eps), call)
        numeric[idx] = (f_plus - f_minus) / (2 * FD_H)
    expect_close(got, numeric, call=call, **FD_TOL)


@case("Part 4", title="被 mask 的行梯度全为 0")
def p4_masked_rows(sol):
    logits = np.array([[[0.5, 1.5, -1.0, 2.0], [3.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
                       [[-2.0, 0.0, 1.0, 0.5], [1.0, 1.0, 1.0, 4.0], [2.0, -3.0, 0.5, 1.0]]])
    targets = np.array([[1, -100, 2], [10 ** 9, 3, 0]])
    run_grad(sol, logits, targets, np.array([[1, 0, 1], [0, 1, 1]]), 0.1)
    run_grad(sol, logits, targets, np.array([[True, False, False], [False, True, False]]), 0.0)


@case("Part 4", title="除以未被 mask 的位置个数")
def p4_denominator(sol):
    logits = np.array([[[2.0, 1.0, 0.1], [0.0, 0.0, 5.0], [1.0, 1.0, 1.0]],
                       [[0.3, -1.2, 2.5], [4.0, 1.0, -2.0], [0.0, 3.0, 0.0]]])
    targets = np.array([[0, 2, 1], [2, 0, 1]])
    for mask, eps in ((np.array([[1, 1, 0], [1, 0, 0]]), 0.0), (np.array([[0, 1, 1], [1, 1, 1]]), 0.3)):
        run_grad(sol, logits, targets, mask, eps)


@case("Part 4", title="全部位置都被 mask：梯度全为 0")
def p4_all_masked(sol):
    logits = np.array([[[0.5, 1.5, -1.0], [3.0, 0.0, 0.0]]])
    for mask in (np.zeros((1, 2), dtype=np.int64), np.zeros((1, 2), dtype=bool)):
        targets = np.array([[-1, 9]])
        call = call_grad(logits, targets, mask, 0.2)
        got = as_grad(sol.cross_entropy_grad(logits.copy(), targets.copy(), mask.copy(), 0.2), logits.shape, call)
        expect_true(np.isfinite(got).all() and not got.any(),
                    "没有任何位置计入时损失恒为 0.0，梯度应当全为 0", call, got)


@case("Part 4", title="ε = 0、没有 mask：softmax − one-hot 再除以 B×T")
def p4_plain(sol):
    gen = npgen()
    for B, T, V in [(2, 3, 4), (1, 1, 2), (3, 3, 3)]:
        logits = np.round(gen.normal(scale=3.0, size=(B, T, V)), 2) + 0.0
        run_grad(sol, logits, gen.integers(0, V, size=(B, T)), np.ones((B, T), dtype=np.int64), 0.0)


@case("Part 4", title="logits 达到 ±1000：梯度有限且正确")
def p4_large_logits(sol):
    logits = np.array([[[1000.0, 0.0, -1000.0], [-1000.0, -1001.0, -1003.0]],
                       [[0.5, -0.5, 0.0], [1000.0, 1000.0, 999.0]]])
    for eps in (0.0, 0.2):
        run_grad(sol, logits, np.array([[1, 0], [2, -7]]), np.array([[1, 1], [1, 0]]), eps)


@case("Part 4", title="随机批次，对比参考实现")
def p4_random(sol):
    gen = npgen()
    for k in range(30):
        B, T, V = (int(v) for v in gen.integers(1, [5, 6, 9]))
        logits, targets, mask = ref.random_batch(gen, B, T, V, bool_mask=bool(k % 2))
        eps = float(gen.choice([0.0, round(float(gen.uniform(0, 0.99)), 3)]))
        run_grad(sol, logits, targets, mask, eps)
