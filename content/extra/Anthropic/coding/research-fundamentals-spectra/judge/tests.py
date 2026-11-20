import numpy as np
import torch

from judgelib import Mismatch, case, expect, expect_close, expect_true, rng
import ref

torch.set_num_threads(1)        # the tensors are tiny: a second thread only adds CPU time

F64 = torch.float64

# The statement's reference numbers for make_data() with seed 0, rounded to 4 and 2 decimals.
QUOTED_EIGVALS = [3.8102, 2.0014, 1.0244, 0.4941, 0.2652, 0.1310, 0.0649, 0.0310]
QUOTED_MEAN_SQ = 72.59

# Tolerances. Every input below has C's eigenvalues apart by at least 1e-5 of the largest (ref.well_separated),
# so a correct answer's eigenvalues differ from eigh's by about 1e-14 and its eigenvectors by at most
# 2.2e-16 / 1e-5 = 2e-11; the statement's own check allows 1e-10 for the eigenvalues. The iterations in Parts 2
# and 3 are deterministic, so a correct answer differs from the formulas only by round-off (measured below 1e-14).
EIG_ATOL = 1e-10
VEC_ATOL = 1e-8
STEP_ATOL = 1e-12
ILL_RTOL = 1e-6     # SVD of the centred data: measured relative error <= 4e-9; eigh(C): >= 1e-4 (kappa(C) = 1e14)


def as_tensor(x, label, call):
    """Accept a tensor, a numpy array, or a list of numbers or 0-d / 1-d tensors; return a float64 tensor."""
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().to(F64)
        if isinstance(x, (list, tuple)) and x and all(isinstance(e, torch.Tensor) for e in x):
            return torch.stack([e.detach().to(F64) for e in x])
        return torch.as_tensor(np.asarray(x, dtype=np.float64))
    except Exception:
        raise Mismatch(f"{label} 应当是 tensor，得到的是 {type(x).__name__}", call, got=x) from None


def unpack(got, names, call):
    k = len(names)
    expect_true(isinstance(got, (tuple, list)) and len(got) == k,
                f"应当返回 {k} 个值 ({', '.join(names)})，得到的是 {type(got).__name__}", call, got=got)
    return [as_tensor(v, name, call) for v, name in zip(got, names)]


def expect_shape(t, shape, label, call):
    expect(tuple(t.shape), shape, call=call, msg=f"{label} 的形状应当是 {shape}，得到 {tuple(t.shape)}")


def lam_code(lam):
    return f"torch.tensor({lam.tolist()}, dtype=torch.float64)"


def part1_inputs(r):
    """Data sets from the statement's generator: d = 1 .. 8, n = 40 .. 2000, assorted spectra."""
    out = []
    for d, n in [(1, 40), (2, 40), (3, 100), (5, 100), (8, 500), (8, 2000)]:
        for _ in range(100):
            ratio, scale, seed = r.choice([0.3, 0.5, 0.7]), r.choice([0.5, 1.0, 4.0, 10.0]), r.randrange(10 ** 6)
            lam = torch.tensor([round(scale * ratio ** i, 6) for i in range(d)], dtype=F64)
            X, _ = ref.make_data(lam, n, seed)
            if ref.well_separated(X):
                break
        else:
            raise RuntimeError("no well-separated data set found")
        out.append((X, f"make_data(lam={lam_code(lam)}, n={n}, seed={seed})[0]"))
    return out


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目数据：特征值和 ‖mean‖²")
def p1_example(sol):
    X, _ = ref.make_data()
    call = "X, y = make_data(); mean, eigvals, components = pca_svd(X)"
    mean, eigvals, comps = unpack(sol.pca_svd(X.clone()), ("mean", "eigvals", "components"), call)
    expect_shape(mean, (8,), "mean", call)
    expect_shape(eigvals, (8,), "eigvals", call)
    expect_shape(comps, (8, 8), "components", call)
    expect_close(eigvals, QUOTED_EIGVALS, rtol=0, atol=5e-5, call=call,
                 msg=f"eigvals 和题目给的对照数字 {QUOTED_EIGVALS} 不符（允许 4 位小数的舍入误差 5e-5）")
    mean_sq = float(mean @ mean)
    expect_true(abs(mean_sq - QUOTED_MEAN_SQ) <= 5e-3,
                f"‖mean‖² 应当约为 {QUOTED_MEAN_SQ}，得到 {mean_sq:.6g}", call, got=mean_sq)


@case("Part 1", title="特征值：与 eigh(C) 相差不超过 1e-10")
def p1_eigvals(sol):
    for X, code in part1_inputs(rng()):
        n, d = X.shape
        call = f"pca_svd({code})"
        want, _ = ref.eig_desc(X)
        mean, eigvals, comps = unpack(sol.pca_svd(X.clone()), ("mean", "eigvals", "components"), call)
        expect_shape(mean, (d,), "mean", call)
        expect_close(mean, X.mean(0), rtol=0, atol=1e-12, call=call, msg="mean 应当是 X 每一列的平均值")
        expect_shape(eigvals, (d,), "eigvals", call)
        if d > 1 and torch.allclose(eigvals, want.flip(0), rtol=0, atol=EIG_ATOL):
            raise Mismatch("eigvals 应当从大到小排列", call, want, eigvals)
        if torch.allclose(eigvals, want * n / (n - 1), rtol=0, atol=EIG_ATOL):
            raise Mismatch("eigvals 是 XcᵀXc / (n − 1) 的特征值；题目约定 C = XcᵀXc / n", call, want, eigvals)
        expect_close(eigvals, want, rtol=0, atol=EIG_ATOL, call=call)


@case("Part 1", title="components：对应的单位特征向量")
def p1_vectors(sol):
    for X, code in part1_inputs(rng()):
        d = X.shape[1]
        call = f"pca_svd({code})"
        want_vals, want_rows = ref.eig_desc(X)
        _, _, comps = unpack(sol.pca_svd(X.clone()), ("mean", "eigvals", "components"), call)
        expect_shape(comps, (d, d), "components", call)
        expect_close(comps @ comps.T, torch.eye(d, dtype=F64), rtol=0, atol=1e-10, call=call,
                     msg="components 的行应当两两正交、长度为 1")
        for i in range(d):
            cos = float((comps[i] @ want_rows[i]).abs())
            if cos < 1 - 1e-10:
                if all(float((comps[:, k] @ want_rows[k]).abs()) >= 1 - 1e-10 for k in range(d)):
                    raise Mismatch("特征向量排在 components 的列上；题目要求第 i 行是第 i 个特征向量"
                                   "（torch.linalg.svd 返回的 Vh 的行就是，不要转置成 V）", call, want_rows[i], comps[i])
                raise Mismatch(f"components[{i}] 不是第 {i + 1} 大的特征值 {float(want_vals[i]):.6g} 对应的特征向量"
                               f"（与它夹角的余弦是 {cos:.6g}）", call, want_rows[i], comps[i])


@case("Part 1", title="components 的符号约定")
def p1_signs(sol):
    for X, code in part1_inputs(rng()):
        d = X.shape[1]
        call = f"pca_svd({code})"
        _, want_rows = ref.eig_desc(X)
        _, _, comps = unpack(sol.pca_svd(X.clone()), ("mean", "eigvals", "components"), call)
        expect_shape(comps, (d, d), "components", call)
        for i in range(d):
            j = int(comps[i].abs().argmax())
            if comps[i, j] < 0:
                raise Mismatch(f"components[{i}] 里绝对值最大的是第 {j} 个元素 {float(comps[i, j]):.6g}，"
                               f"题目规定它为正", call, want_rows[i], comps[i])
        expect_close(comps, want_rows, rtol=0, atol=VEC_ATOL, call=call)


@case("Part 1", title="病态数据 κ(C)=1e14：不构造 C")
def p1_ill_conditioned(sol):
    r = rng()
    for seed in [r.randrange(10 ** 6) for _ in range(3)]:
        X, want = ref.ill_conditioned(seed)
        call = f"import ref; X, want = ref.ill_conditioned({seed}); pca_svd(X)   # 中心化后奇异值 1 … 1e-7"
        _, eigvals, _ = unpack(sol.pca_svd(X.clone()), ("mean", "eigvals", "components"), call)
        expect_shape(eigvals, (6,), "eigvals", call)
        rel = ((eigvals - want) / want).abs()
        i = int(rel.argmax())
        msg = f"κ(C) = 1e14 的数据上 eigvals[{i}] 的相对误差是 {float(rel[i]):.3g}（允许 {ILL_RTOL:g}）"
        if rel[0] <= ILL_RTOL:      # only the small eigenvalues are off: the signature of squaring the condition number
            msg += "。题目要求对中心化后的数据做 SVD、不构造 C：构造 C 会把条件数平方，小特征值就不准了"
        expect_true(bool(rel.max() <= ILL_RTOL), msg, call, got=eigvals)


# ---------------------------------------------------------------- Part 2
def pi_outputs(got, d, steps, call):
    V, rho = unpack(got, ("V", "rho"), call)
    expect_shape(V, (steps + 1, d), "V", call)
    expect_shape(rho, (steps + 1,), "rho", call)
    return V, rho


def start_vector(d, seed):
    g = torch.randn(d, generator=torch.Generator().manual_seed(seed), dtype=F64)
    return g / g.norm()


def expect_v0(V, d, seed, call, msg):
    """V[0] must be the statement's start vector; name the shared-storage mistake when every row is the last v."""
    v0 = start_vector(d, seed)
    if not torch.allclose(V[0], v0, rtol=0, atol=1e-14) and V.shape[0] > 1 and bool((V == V[-1]).all()):
        raise Mismatch("V 的每一行都相同，等于最后的 v：存进去的是同一个 tensor，之后原地更新 v 把前面的行也改了。"
                       "保存时要 .clone()，或者每步新建 v（v = w / w.norm()）而不是 v /= ...", call, v0, V[0])
    expect_close(V[0], v0, rtol=0, atol=1e-14, call=call, msg=msg)


def spd_inputs(r):
    """(C, code, d, seed): symmetric positive definite matrices whose ratio lambda_2 / lambda_1 is 0.9, so 15 steps
    are far from convergence and ||C v_t|| still differs from v_t^T C v_t."""
    out = []
    for d in (2, 6, 12):
        mseed, seed = r.randrange(10 ** 6), r.randrange(10 ** 6)
        out.append((ref.spd(mseed, d, 0.9), f"import ref; C = ref.spd({mseed}, {d}, 0.9)", d, seed))
    return out


@case("Part 2", sample=True, title="题目设置：跑 60 步")
def p2_example(sol):
    X, _ = ref.make_data()
    Xc = X - X.mean(0)
    n = X.shape[0]
    call = "X, y = make_data(); Xc = X - X.mean(0); V, rho = power_iteration(lambda v: Xc.T @ (Xc @ v) / 2000, 8, 60)"
    V, rho = pi_outputs(sol.power_iteration(lambda v: Xc.T @ (Xc @ v) / n, 8, 60), 8, 60, call)
    expect_v0(V, 8, 0, call,
              msg="V[0] 应当是 g / ‖g‖，g = torch.randn(8, generator=torch.Generator().manual_seed(0), dtype=torch.float64)")
    expect_close(V.norm(dim=1), torch.ones(61, dtype=F64), rtol=0, atol=1e-12, call=call, msg="V 的每一行应当是单位向量")
    expect_true(abs(float(rho[60]) - QUOTED_EIGVALS[0]) <= 5e-5,
                f"60 步后 rho[60] 应当收敛到 λ̂₁ ≈ {QUOTED_EIGVALS[0]}，得到 {float(rho[60]):.6g}", call, got=rho)


@case("Part 2", title="起点 v_0 = g / ‖g‖")
def p2_start(sol):
    r = rng()
    for d in (1, 3, 8, 20):
        seed = r.randrange(10 ** 6)
        C = ref.spd(seed + 1, d, 0.7)
        call = f"import ref; C = ref.spd({seed + 1}, {d}, 0.7); power_iteration(lambda v: C @ v, {d}, 3, seed={seed})"
        V, _ = pi_outputs(sol.power_iteration(lambda v: C @ v, d, 3, seed=seed), d, 3, call)
        expect_v0(V, d, seed, call,
                  msg=f"V[0] 应当是 g / ‖g‖，g = torch.randn({d}, generator=torch.Generator().manual_seed({seed}), "
                      f"dtype=torch.float64)")


@case("Part 2", title="每一步 v_{t+1} = C v_t / ‖C v_t‖")
def p2_steps(sol):
    for C, code, d, seed in spd_inputs(rng()):
        call = f"{code}; power_iteration(lambda v: C @ v, {d}, 15, seed={seed})"
        V, _ = pi_outputs(sol.power_iteration(lambda v: C @ v, d, 15, seed=seed), d, 15, call)
        expect_v0(V, d, seed, call, msg="V[0] 不对")
        for t in range(15):
            want = C @ V[t]
            want = want / want.norm()
            if not torch.allclose(V[t + 1], want, rtol=0, atol=STEP_ATOL):
                raise Mismatch(f"V[{t + 1}] 应当等于 C v_{t} / ‖C v_{t}‖（v_{t} 取你返回的 V[{t}]）",
                               call, want, V[t + 1])


@case("Part 2", title="rho[t] 是 v_t 的 Rayleigh 商")
def p2_rayleigh(sol):
    for C, code, d, seed in spd_inputs(rng()):
        call = f"{code}; power_iteration(lambda v: C @ v, {d}, 15, seed={seed})"
        V, rho = pi_outputs(sol.power_iteration(lambda v: C @ v, d, 15, seed=seed), d, 15, call)
        want = ((V @ C) * V).sum(dim=1)                 # v_t^T C v_t for the answer's own v_t
        bad = ((rho - want).abs() > STEP_ATOL).nonzero()
        if len(bad):
            t = int(bad[0])
            raise Mismatch(f"rho[{t}] 应当是 v_{t}ᵀ C v_{t} = {float(want[t]):.12g}（v_{t} 取你返回的 V[{t}]），"
                           f"得到 {float(rho[t]):.12g}", call, want, rho)


@case("Part 2", title="steps = 0：只有 v_0")
def p2_zero_steps(sol):
    r = rng()
    d, mseed, seed = 5, r.randrange(10 ** 6), r.randrange(10 ** 6)
    C = ref.spd(mseed, d, 0.5)
    call = f"import ref; C = ref.spd({mseed}, {d}, 0.5); power_iteration(lambda v: C @ v, {d}, 0, seed={seed})"
    V, rho = pi_outputs(sol.power_iteration(lambda v: C @ v, d, 0, seed=seed), d, 0, call)
    v0 = start_vector(d, seed)
    expect_close(V[0], v0, rtol=0, atol=1e-14, call=call, msg="V[0] 不对")
    expect_close(rho, (v0 @ C @ v0).reshape(1), rtol=0, atol=STEP_ATOL, call=call, msg="rho[0] 应当是 v_0ᵀ C v_0")


# ---------------------------------------------------------------- Part 3
def check_iterates(got, Xc, y, lr, steps, call):
    """W must have steps + 1 rows and follow w_{t+1} = w_t - lr * grad L(w_t) from w_0 = 0. Each row may differ from
    the hand-written recurrence by round-off only: 1e-9 relative (measured: below 1e-14)."""
    d = Xc.shape[1]
    W = as_tensor(got, "W", call)
    expect_shape(W, (steps + 1, d), "W", call)
    want = ref.gd_iterates(Xc, y, lr, steps)
    for t in range(steps + 1):
        err = float((W[t] - want[t]).abs().max())
        if not err <= 1e-9 * (1.0 + float(want[t].abs().max())):
            what = "w_0 应当是全 0" if t == 0 else f"W[{t}] 不等于 w_{t - 1} − lr · ∇L(w_{t - 1})（w_{t - 1} 取正确的值）"
            if t == 0 and steps > 0 and bool((W == W[-1]).all()):
                what = ("W 的每一行都相同，等于最后的 w：存进去的是 w 本身（共享内存），之后原地更新 w 把前面的行也改了，"
                        "保存时要 .clone()")
            raise Mismatch(f"{what}；最大误差 {err:.3g}", call, want[t], W[t])


def part3_inputs(r):
    out = []
    for d, n in [(1, 20), (4, 30), (6, 200)]:
        seed = r.randrange(10 ** 6)
        lam = torch.tensor([round(2.0 * 0.6 ** i, 6) for i in range(d)], dtype=F64)
        X, y = ref.make_data(lam, n, seed)
        Xc = X - X.mean(0)
        lam1 = float(torch.linalg.eigvalsh(Xc.T @ Xc / n)[-1])
        code = (f"X, y = make_data(lam={lam_code(lam)}, n={n}, seed={seed}); Xc = X - X.mean(0); "
                f"lam1 = torch.linalg.eigvalsh(Xc.T @ Xc / {n})[-1].item()")
        out.append((Xc, y, lam1, code))
    return out


@case("Part 3", sample=True, title="题目设置：η = 1/λ̂₁，1200 步")
def p3_example(sol):
    X, y = ref.make_data()
    Xc = X - X.mean(0)
    lr = 1.0 / float(torch.linalg.eigvalsh(Xc.T @ Xc / 2000)[-1])
    call = ("X, y = make_data(); Xc = X - X.mean(0); lr = 1.0 / torch.linalg.eigvalsh(Xc.T @ Xc / 2000)[-1].item(); "
            "W = gd_least_squares(Xc, y, lr, 1200)")
    check_iterates(sol.gd_least_squares(Xc.clone(), y.clone(), lr, 1200), Xc, y, lr, 1200, call)


@case("Part 3", title="从 w = 0 开始，返回 steps + 1 行")
def p3_start(sol):
    for Xc, y, lam1, code in part3_inputs(rng()):
        for steps in (0, 1, 3):
            call = f"{code}; gd_least_squares(Xc, y, 1.0 / lam1, {steps})"
            check_iterates(sol.gd_least_squares(Xc.clone(), y.clone(), 1.0 / lam1, steps), Xc, y, 1.0 / lam1,
                           steps, call)


@case("Part 3", title="每一步都是一次梯度下降")
def p3_steps(sol):
    for Xc, y, lam1, code in part3_inputs(rng()):
        for k in (0.3, 1.0, 1.9, 2.1):            # 2.1 / lambda_1 is past the stability threshold: a_1 grows
            call = f"{code}; gd_least_squares(Xc, y, {k} / lam1, 40)"
            check_iterates(sol.gd_least_squares(Xc.clone(), y.clone(), k / lam1, 40), Xc, y, k / lam1, 40, call)
