import warnings

import numpy as np

from judgelib import Mismatch, calling, case, rng
import ref

# Every factor of FL and of its closed-form gradient is a product of stably computed positive quantities (no
# cancellation: the bracket gamma * p_t * log p_t - (1 - p_t) adds two non-positive terms), so a float64 answer that
# follows the statement is within about 1e-13 relative of the exact value, even for gradients near 1e-260 at
# |z| = 100. 1e-9 leaves four orders of magnitude; every targeted mistake is off by a factor, or gives 0 or inf.
RTOL = 1e-9


def lit(a):
    return f"np.array({np.asarray(a).tolist()!r})"


def call_text(z_text, y_text, alpha, gamma):
    kw = "".join([f", alpha={alpha!r}" if alpha != 0.25 else "", f", gamma={gamma!r}" if gamma != 2.0 else ""])
    return f"focal_loss_and_grad({z_text}, {y_text}{kw})"


def check(sol, z, y, alpha=0.25, gamma=2.0, setup="", note=""):
    """Call the answer on (z, y) and compare with the exact mean loss and gradient."""
    z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    n = len(z)
    if setup:
        call = setup + call_text("z", "y", alpha, gamma) + note
    else:
        call = call_text(lit(z), lit(y), alpha, gamma) + note
    want_loss, want_grad = ref.focal(z, y, alpha, gamma)
    want = (want_loss, want_grad)
    calling(call)
    kw = {}
    if alpha != 0.25:
        kw["alpha"] = alpha
    if gamma != 2.0:
        kw["gamma"] = gamma
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = sol.focal_loss_and_grad(z.copy(), y.copy(), **kw)
    overflow = [w for w in caught if "overflow" in str(w.message)]
    if overflow:
        raise Mismatch(f"触发了溢出警告：{overflow[0].category.__name__}: {overflow[0].message}。|z| ≤ 100 时不能溢出",
                       call, want, got)
    if not isinstance(got, (tuple, list)) or len(got) != 2:
        raise Mismatch("应当返回 (mean_loss, grad) 两个值", call, want, got)
    try:
        loss = np.asarray(got[0], dtype=float)
        grad = np.asarray(got[1], dtype=float)
    except Exception:
        raise Mismatch("mean_loss 和 grad 都应当是数值", call, want, got) from None
    if loss.shape != ():
        raise Mismatch(f"mean_loss 应当是一个数（整个批量的平均），得到形状 {loss.shape}", call, want, got)
    if grad.shape != (n,):
        raise Mismatch(f"grad 的形状应当是 ({n},)，得到 {grad.shape}", call, want, got)
    loss = float(loss)
    if not np.isfinite(loss):
        raise Mismatch(f"mean_loss = {loss}，不是有限数：log p_t 要用 -softplus(-s z) 算，不能对已经舍入成 0 的 p_t 取 log",
                       call, want, got)
    bad = ~np.isfinite(grad)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise Mismatch(f"grad[{i}] = {grad[i]}（z = {float(z[i])!r}，y = {y[i]:g}），不是有限数", call, want, got)
    if not abs(loss - want_loss) <= RTOL * abs(want_loss):
        hint = ""
        if n > 1 and abs(loss - n * want_loss) <= 1e-6 * n * want_loss:
            hint = "：看起来返回的是总和，题目要的是批量平均"
        elif loss == 0.0:
            hint = "：结果正好是 0。p_t 接近 1 时 log p_t 是一个很小的负数，不能先算 p_t 再取 log（会得到 log 1 = 0）"
        raise Mismatch(f"mean_loss 不对：期望 {want_loss!r}，得到 {loss!r}{hint}", call, want, got)
    off = np.abs(grad - want_grad) > RTOL * np.abs(want_grad)
    if off.any():
        i = int(np.flatnonzero(off)[0])
        hint = ""
        if n > 1 and np.allclose(grad, n * want_grad, rtol=1e-6, atol=0):
            hint = "：整个 grad 正好是期望的 N 倍，要对批量平均求导，每一项要除以 N"
        elif grad[i] == 0.0:
            hint = "：结果正好是 0。1 - p_t 要用 sigma(-s z) 直接算，不能用 1 减去已经舍入成 1.0 的 p_t"
        raise Mismatch(f"grad[{i}] 不对（z = {float(z[i])!r}，y = {y[i]:g}）：期望 {float(want_grad[i])!r}，得到 {float(grad[i])!r}{hint}",
                       call, want, got)


@case("Part 2", sample=True, title="题目例子：一个易样本、一个难样本，都是正类")
def p2_example(sol):
    z = np.array([np.log(9.0), -np.log(9.0)])
    check(sol, z, [1.0, 1.0], setup="z = np.array([np.log(9.0), -np.log(9.0)])\ny = np.array([1.0, 1.0])\n",
          note="   # 题目给出 (0.2333, [-0.0004, -0.1378])")


# (z, y, alpha, gamma): both labels on both sides of 0, so alpha_t, s and p_t are each exercised for y = 0.
SINGLES = [(1.3, 1.0, 0.25, 2.0), (1.3, 0.0, 0.25, 2.0), (-0.8, 0.0, 0.25, 2.0), (-0.8, 1.0, 0.25, 2.0),
           (0.0, 0.0, 0.6, 1.0), (2.5, 0.0, 0.6, 3.0), (-2.5, 0.0, 0.3, 0.5), (4.0, 1.0, 0.9, 2.0),
           (-4.0, 1.0, 0.9, 2.0), (0.4, 0.0, 0.1, 2.0)]


@case("Part 2", title="单个样本：两种标签、logit 正负两侧")
def p2_single(sol):
    for z, y, alpha, gamma in SINGLES:
        check(sol, [z], [y], alpha, gamma)


@case("Part 2", title="批量：损失取平均，梯度是平均损失的梯度")
def p2_batch_mean(sol):
    r = rng("batch")
    for n in (2, 3, 5, 8, 13):
        z = [round(r.uniform(-6, 6), 3) for _ in range(n)]
        y = [float(r.random() < 0.5) for _ in range(n)]
        y[0], y[1] = 0.0, 1.0
        alpha, gamma = (0.25, 2.0) if n < 5 else (round(r.uniform(0.05, 0.95), 2), round(r.uniform(0.5, 4), 2))
        check(sol, z, y, alpha, gamma)


@case("Part 2", title="gamma = 0：正好是按 alpha_t 加权的交叉熵")
def p2_gamma_zero(sol):
    r = rng("gamma0")
    for alpha in (0.25, 0.5, 0.8):
        z = [round(r.uniform(-5, 5), 3) for _ in range(6)]
        check(sol, z, [0.0, 1.0, 0.0, 1.0, 1.0, 0.0], alpha, 0.0, note="   # 等于 mean(-alpha_t * log p_t)")


@case("Part 2", title="gamma 不是整数")
def p2_fractional_gamma(sol):
    r = rng("fractional")
    for gamma in (0.5, 1.5, 4.7):
        z = [round(r.uniform(-8, 8), 3) for _ in range(6)]
        check(sol, z, [1.0, 0.0, 1.0, 0.0, 0.0, 1.0], round(r.uniform(0.1, 0.9), 2), gamma)


@case("Part 2", title="自信但错误：|z| 到 100，p_t 极小")
def p2_confident_wrong(sol):
    # p = sigma(40) rounds to 1.0, so 1 - p (= p_t for y = 0) becomes 0 and log p_t becomes -inf.
    for gamma in (2.0, 0.5, 0.0):
        check(sol, [40.0, -40.0, 100.0, -100.0, 37.5, -60.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 0.25, gamma,
              note="   # 每个样本都站在错误的一侧：p_t 小到 e^-100")


@case("Part 2", title="自信且正确：1 - p_t 极小但不是 0")
def p2_confident_right(sol):
    # Only easy examples: 1 - p_t is e^-37 ... e^-100, and log p_t is about -(1 - p_t). Both must keep their relative
    # precision; computing either one from the other by subtraction gives exactly 0.
    for gamma in (2.0, 1.0, 0.0):
        check(sol, [37.0, 45.0, 100.0, -40.0, -100.0], [1.0, 1.0, 1.0, 0.0, 0.0], 0.25, gamma,
              note="   # 每个样本都站在正确的一侧：1 - p_t 小到 e^-100，结果都很小但不是 0")


@case("Part 2", title="z 从 -100 到 100：结果有限、准确，不溢出")
def p2_grid(sol):
    r = rng("grid")
    for gamma in (0.0, 1.0, 2.0, 5.0):
        seed = r.randrange(10_000)
        alpha = round(r.uniform(0.05, 0.95), 2)
        z = np.linspace(-100, 100, 401)
        y = (np.random.default_rng(seed).random(401) < 0.5).astype(float)
        setup = f"z = np.linspace(-100, 100, 401)\ny = (np.random.default_rng({seed}).random(401) < 0.5).astype(float)\n"
        check(sol, z, y, alpha, gamma, setup=setup)


@case("Part 2", title="随机批量")
def p2_random(sol):
    r = rng("random")
    for _ in range(25):
        n = r.randint(1, 20)
        z = [round(r.gauss(0, 8), 4) for _ in range(n)]
        y = [float(r.random() < 0.5) for _ in range(n)]
        check(sol, z, y, round(r.uniform(0.05, 0.95), 3), round(r.uniform(0, 5), 3))
