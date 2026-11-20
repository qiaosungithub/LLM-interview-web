import copy
import math

import numpy as np

from judgelib import Mismatch, calling, case, expect_close, rng
import ref

# Part 3: BIC is one formula; a correct float64 answer is within 1e-13 relative of it. Every mistake the tests
# target moves it by log(n) >= 3 or more.
BIC_TOL = dict(rtol=1e-9, atol=0.0)
# k-means++ frequency tests fail only when a deviation this unlikely (two-sided binomial tail) would be needed.
P_FAIL = 1e-9


def seeds(salt, k):
    r = rng(salt)
    return [r.getrandbits(32) for _ in range(k)]


def lit(a):
    return f"np.array({np.asarray(a).tolist()!r})"


def mixture(seed, n, d, K, sep, spread=1.0):
    """n points around K random centers; the setup text rebuilds exactly this X."""
    gen = np.random.default_rng(seed)
    centers = gen.normal(size=(K, d)) * sep
    X = centers[gen.integers(K, size=n)] + gen.normal(size=(n, d)) * spread
    setup = (f"rng = np.random.default_rng({seed})\ncenters = rng.normal(size=({K}, {d})) * {sep}\n"
             f"X = centers[rng.integers({K}, size={n})] + rng.normal(size=({n}, {d})) * {spread}\n")
    return X, setup


def stable_mixture(salt, n, d, K, sep, reg, tol):
    """The first mixture (seeds drawn from salt) on which EM keeps every covariance well away from collapse."""
    for seed in seeds(salt, 20):
        X, setup = mixture(seed, n, d, K, sep)
        if ref.stable(X, K, reg, tol):
            return X, setup
    raise RuntimeError("no stable mixture")      # never reached: at most about 1 draw in 10 is rejected


def summary(got):
    """What fit_gmm returned, rounded for display."""
    if not isinstance(got, dict):
        return got
    out = {}
    for key, v in got.items():
        try:
            a = np.asarray(v, dtype=float)
            out[key] = a.tolist() if key == "log_likelihoods" else np.round(a, 6).tolist()
        except Exception:
            out[key] = v
    return out


def fit_call(setup, K, n_iter=200, tol=1e-8, reg=1e-6, seed=0):
    args = [str(K)]
    if n_iter != 200:
        args.append(f"n_iter={n_iter}")
    if tol != 1e-8:
        args.append(f"tol={tol!r}")
    if reg != 1e-6:
        args.append(f"reg={reg!r}")
    args.append(f"seed={seed}")
    return setup + f"fit_gmm(X, {', '.join(args)})"


def run_fit(sol, X, K, call, **kw):
    calling(call)
    return sol.fit_gmm(X.copy(), K, **kw)


def check_fit(sol, X, K, setup, n_iter=200, tol=1e-8, reg=1e-6, seed=0, note=""):
    call = fit_call(setup, K, n_iter, tol, reg, seed) + note
    got = run_fit(sol, X, K, call, n_iter=n_iter, tol=tol, reg=reg, seed=seed)
    msg = ref.check_fit(X, K, n_iter, tol, reg, got)
    if msg:
        raise Mismatch(msg, call, None, summary(got))


def check_init(sol, X, K, setup, reg=1e-6, seed=0, note=""):
    call = fit_call(setup, K, 1, 1e-8, reg, seed) + note
    got = run_fit(sol, X, K, call, n_iter=1, reg=reg, seed=seed)
    msg = ref.check_init(X, K, reg, got)
    if msg:
        raise Mismatch(msg, call, None, summary(got))


# ---------------------------------------------------------------- Part 2: fit_gmm

EX_X = np.array([[-2.0], [0.0], [3.0]])
EX_SETUP = "X = np.array([[-2.0], [0.0], [3.0]])\n"


@case("Part 2", sample=True, title="题目例子的数据：n_iter=1 只返回初始参数")
def p2_example_init(sol):
    # Sigma_hat of (-2, 0, 3) is 114/27; the means are two of the three rows, whichever k-means++ draws.
    check_init(sol, EX_X, 2, EX_SETUP, note="   # 权重 (0.5, 0.5)，协方差都是 114/27 + 1e-6，均值是 X 的两行")


@case("Part 2", sample=True, title="题目例子的数据：跑到收敛")
def p2_example_fit(sol):
    check_fit(sol, EX_X, 2, EX_SETUP)


# (n, d, K, reg): reg = 0.25 makes a forgotten + reg * I obvious.
P2_INITS = [(12, 1, 1, 0.25), (15, 1, 3, 1e-6), (20, 2, 2, 0.25), (25, 2, 4, 1e-6), (30, 3, 3, 0.25), (10, 3, 2, 1e-6)]


@case("Part 2", title="初始化：权重 1/K、协方差 Sigma_hat + reg·I、均值是不同的数据点")
def p2_init(sol):
    for seed, (n, d, K, reg) in zip(seeds("init", len(P2_INITS)), P2_INITS):
        X, setup = mixture(seed, n, d, K, 3.0)
        check_init(sol, X, K, setup, reg=reg, seed=seed % 1000)


# (d, K, n, sep, reg, tol)
P2_FITS = [(1, 1, 30, 3.0, 1e-6, 1e-8), (1, 2, 40, 4.0, 1e-6, 1e-8), (1, 3, 50, 5.0, 1e-6, 1e-8),
           (2, 1, 30, 3.0, 1e-6, 1e-8), (2, 2, 45, 4.0, 1e-6, 1e-8), (2, 3, 60, 5.0, 1e-3, 1e-8),
           (3, 2, 50, 4.0, 1e-6, 1e-4), (3, 1, 35, 3.0, 1e-3, 1e-8)]


@case("Part 2", cpu=4, title="随机数据跑到收敛：逐个 E 步对照")
def p2_fits(sol):
    for i, (d, K, n, sep, reg, tol) in enumerate(P2_FITS):
        X, setup = stable_mixture(f"fit {i}", n, d, K, sep, reg, tol)
        check_fit(sol, X, K, setup, tol=tol, reg=reg, seed=seeds(f"fit seed {i}", 1)[0] % 1000)


@case("Part 2", title="没收敛就到了 n_iter：恰好 n_iter 个 E 步，参数对应最后一个 ℓ")
def p2_n_iter(sol):
    # Heavily overlapping clusters: EM is still improving ell by far more than tol after a handful of steps.
    for i, (d, K, n_iter) in enumerate(((2, 2, 2), (1, 2, 3), (2, 3, 6), (1, 3, 4))):
        X, setup = mixture(seeds(f"n_iter {i}", 1)[0], 50, d, K, 0.8)
        check_fit(sol, X, K, setup, n_iter=n_iter, seed=i)


@case("Part 2", title="tol 是 ℓ 的绝对变化量")
def p2_tol_absolute(sol):
    # |ell| is in the hundreds, so a relative test (change / |ell| < tol) stops hundreds of times too early.
    for i, (n, d, tol) in enumerate(((150, 1, 1e-3), (120, 2, 1e-2), (200, 1, 1e-4))):
        X, setup = mixture(seeds(f"tol {i}", 1)[0], n, d, 2, 1.2)
        check_fit(sol, X, 2, setup, tol=tol, seed=i)


@case("Part 2", title="reg 很大：每次 M 步之后都要加 reg·I")
def p2_large_reg(sol):
    # With a reg this large ell can dip at the last step; the check accepts stopping there or going on.
    for i, (n, d, K, reg) in enumerate(((40, 2, 3, 0.3), (30, 1, 2, 1.0), (36, 3, 2, 0.5))):
        X, setup = mixture(seeds(f"reg {i}", 1)[0], n, d, K, 3.0)
        check_fit(sol, X, K, setup, reg=reg, seed=i)


@case("Part 2", title="log 密度低到 -790：E 步要用 log-sum-exp")
def p2_underflow(sol):
    # 1599 points with spread 0.1 and one at 40: Sigma_hat is about 1 along that axis, so the far point sits about
    # sqrt(n) standard deviations out and its log density is about -n/2 = -800, where exp() underflows to 0.
    for d, (seed, row) in zip((1, 2), ((s % 1000, s % 1600) for s in seeds("underflow", 2))):
        gen = np.random.default_rng(seed)
        X = gen.normal(size=(1600, d)) * 0.1
        X[row, 0] = 40.0
        setup = (f"X = np.random.default_rng({seed}).normal(size=(1600, {d})) * 0.1\nX[{row}, 0] = 40.0\n")
        check_fit(sol, X, 1, setup, seed=seed, note="   # 那个离群点的 log N 大约是 -790")


@case("Part 2", title="同一个 seed 结果完全相同")
def p2_same_seed(sol):
    X, setup = mixture(seeds("same seed data", 1)[0], 30, 2, 3, 3.0)
    for seed in (s % 10000 for s in seeds("same seed", 3)):
        call = setup + f"a = fit_gmm(X, 3, n_iter=1, seed={seed})\nfit_gmm(X, 3, n_iter=1, seed={seed + 1})\n" \
                       f"b = fit_gmm(X, 3, n_iter=1, seed={seed})\n# a 和 b 应当完全相同"
        a = run_fit(sol, X, 3, call, n_iter=1, seed=seed)
        run_fit(sol, X, 3, call, n_iter=1, seed=seed + 1)
        b = run_fit(sol, X, 3, call, n_iter=1, seed=seed)
        for got in (a, b):
            msg = ref.check_init(X, 3, 1e-6, got)
            if msg:
                raise Mismatch(msg, call, None, summary(got))
        ma, mb = (np.asarray(m["means"], dtype=float) for m in (a, b))
        if not np.array_equal(ma, mb):
            raise Mismatch("同一个 seed 的两次调用选出了不同的初始均值：所有随机性都要来自 np.random.default_rng(seed)，"
                           "不能用全局的 np.random", call, np.round(ma, 6).tolist(), np.round(mb, 6).tolist())


def kmeanspp_frequencies(sol, X, K, groups, salt, N=240):
    """Draw the initial means for N seeds (n_iter = 1) and compare how often each row, and each group of rows, is
    among them with the exact k-means++ probabilities."""
    n = len(X)
    base = rng(salt).getrandbits(20)
    setup = f"X = {lit(X)}\n"
    call = setup + f"[fit_gmm(X, {K}, n_iter=1, seed=s)['means'] for s in range({base}, {base + N})]"
    draws = []
    for s in range(base, base + N):
        one = setup + f"fit_gmm(X, {K}, n_iter=1, seed={s})"
        got = run_fit(sol, X, K, one, n_iter=1, seed=s)
        model = ref.as_model(got, X, K)
        if isinstance(model, str):
            raise Mismatch(model, one, None, summary(got))
        rows = ref.chosen_rows(X, model[1])
        if rows is None or len(set(rows)) < K:
            raise Mismatch(f"初始均值应当是 X 的 {K} 行不同的数据点", one, None, summary(got))
        draws.append(set(rows))
    sets = ref.kmeanspp_sets(X, K)
    events = [(f"第 {i} 行（x = {X[i, 0]:g}）", {i}) for i in range(n)] + groups
    for label, rows in events:
        p = sum(prob for s, prob in sets.items() if rows & set(s))
        c = sum(1 for dr in draws if rows & dr)
        if ref.binom_two_sided(c, N, p) < P_FAIL:
            raise Mismatch(f"初始均值的分布不像 k-means++：{label}在这 {N} 次里被选中 {c} 次，按 k-means++ 应当约 "
                           f"{N * p:.1f} 次（正确的实现偏离这么多的概率不到 {P_FAIL:g}）。第一个均值从 n 行里均匀地抽，"
                           f"之后每个按“到最近的已选均值的距离平方”成比例地抽", call)


@case("Part 2", title="k-means++：第一个均匀抽，之后按距离平方抽")
def p2_kmeanspp_squared(sol):
    # 19 points in [-1, 1] and one at 12 (row 7): squared distances pick it about 92% of the time, plain distances
    # 51%, uniform draws 10%; a first mean that is always row 0 makes row 0 appear every time instead of 6%.
    X = np.insert(np.linspace(-1, 1, 19), 7, 12.0).reshape(-1, 1)
    kmeanspp_frequencies(sol, X, 2, [], "kpp2")


@case("Part 2", title="k-means++：按到*最近*已选均值的距离抽")
def p2_kmeanspp_nearest(sol):
    # 14 points in [-1, 1], 3 near 5 and 3 near 10. After one seed near 0 and one near 10, the points near 5 are
    # 25 away (squared) from both; weighting by the distance to the last seed only, or by the sum over seeds, makes
    # the 14 points near 0 (100 away from 10) win instead: the group near 5 is covered 36% of the time, not 89%.
    X = np.concatenate([np.linspace(-1, 1, 14), [4.8, 5.0, 5.2, 9.8, 10.0, 10.2]])
    X = X[[14, 11, 6, 16, 5, 0, 12, 2, 4, 1, 17, 15, 3, 18, 7, 13, 9, 10, 8, 19]].reshape(-1, 1)
    groups = [(f"{name}（第 {sorted(rows)} 行）的某一行", rows) for name, lo, hi in
              (("-1 到 1 那组", -1.5, 1.5), ("5 附近那组", 4.0, 6.0), ("10 附近那组", 9.0, 11.0))
              for rows in [set(np.flatnonzero((X[:, 0] > lo) & (X[:, 0] < hi)).tolist())]]
    kmeanspp_frequencies(sol, X, 3, groups, "kpp3")


# ---------------------------------------------------------------- Part 3: bic

def bic_formula(n, d, K, ell):
    return -2 * ell + ((K - 1) + K * d + K * d * (d + 1) // 2) * math.log(n)


def model_text(m):
    return (f"model = {{\"weights\": {lit(m['weights'])}, \"means\": {lit(m['means'])},\n"
            f"         \"covs\": {lit(m['covs'])},\n         \"log_likelihoods\": {m['log_likelihoods']!r}}}\n")


def check_bic(sol, X, model, setup, want, note=""):
    call = setup + model_text(model) + "bic(X, model)" + note
    calling(call)
    got = sol.bic(X.copy(), copy.deepcopy(model))
    expect_close(got, want, call=call, **BIC_TOL)


# Covariance chosen so that ell at these parameters is -140.0 (to 3e-14): the dict is consistent with X, as one
# returned by fit_gmm would be.
EX_S2 = 0.5848274256260623


@case("Part 3", sample=True, title="题目例子：K=2、d=1、n=100、ℓ=-140，BIC ≈ 303.03")
def p3_example(sol):
    X = np.linspace(-1.5, 1.5, 100).reshape(100, 1)
    model = {"weights": np.array([0.5, 0.5]), "means": np.array([[-1.0], [1.0]]),
             "covs": np.array([[[EX_S2]], [[EX_S2]]]), "log_likelihoods": [-140.0]}
    check_bic(sol, X, model, "X = np.linspace(-1.5, 1.5, 100).reshape(100, 1)\n", 280 + 5 * math.log(100),
              "   # p = 5，BIC = 280 + 5 log(100) ≈ 303.03")


def fitted_model(X, K, gen, reg=1e-6):
    w, mu, cov, lls = ref.fit(X, X[ref.kmeanspp(X, K, gen)], 200, 1e-8, reg)
    return {"weights": w, "means": mu, "covs": cov, "log_likelihoods": lls}


@case("Part 3", title="参数个数：d = 1, 2, 3，K = 1 到 4")
def p3_formula(sol):
    for i, (n, d, K) in enumerate(((40, 1, 3), (50, 2, 2), (60, 3, 2), (45, 2, 4), (25, 3, 1))):
        X, setup = mixture(seeds(f"bic {i}", 1)[0], n, d, K, 6.0)
        model = fitted_model(X, K, np.random.default_rng(i))
        p = (K - 1) + K * d + K * d * (d + 1) // 2
        check_bic(sol, X, model, setup, bic_formula(n, d, K, model["log_likelihoods"][-1]),
                  f"   # n = {n}, d = {d}, K = {K}：p = {p}")


@case("Part 3", title="ℓ 取 log_likelihoods 的最后一个，不是最大的")
def p3_last_value(sol):
    # reg = 0.5 is large next to this data's unit variance, so the last EM step lowers ell (by 0.56) and the fit
    # stops there: log_likelihoods[-1] is not the largest entry.
    X = np.random.default_rng(211).normal(size=(24, 2))
    w, mu, cov, lls = ref.fit(X, X[[20, 8, 22]], 200, 1e-8, 0.5)
    model = {"weights": w, "means": mu, "covs": cov, "log_likelihoods": lls}
    check_bic(sol, X, model, "X = np.random.default_rng(211).normal(size=(24, 2))\n",
              bic_formula(24, 2, 3, lls[-1]), "   # ℓ 最后一步下降了：要用 log_likelihoods[-1]")
