"""Judge-side references for tests.py.

fit_gmm's initial means depend on its random draws, which the statement does not pin down (only that they come
from np.random.default_rng(seed) and follow k-means++). So check_fit does not compare with one fixed run. It finds
every legal initialisation (K distinct rows of X as means, weights 1/K, covariances Sigma_hat + reg * I) whose
log-likelihood equals the answer's log_likelihoods[0], replays EM from it with the statement's rules, and accepts
the answer if one replay reproduces every logged value and the returned parameters (up to relabelling the
components)."""
import itertools
import math

import numpy as np

# A correct float64 fit differs from this replay only by rounding. EM amplifies it most when a component collapses
# onto a few points (its covariance shrinks to reg * I). Measured with the page's reference and an independent
# implementation (inverse instead of solve, einsum M-step, another k-means++ draw), 6000 fits of problems like the
# tests' (n 30-60, d <= 3, K <= 3): the worst |delta ell| was 0.015 of ll_tol (typically 1e-6 of it) and the worst
# parameter error 4e-7 of PARAM_TOL; on 3000 harder ones (n down to 8, K up to 4, many collapsed components) the
# worst was 0.41 of ll_tol. Every mistake the tests target moves ell or a parameter by 1e-4 or more.
LL_RTOL = 1e-9
PARAM_TOL = 1e-6


def ll_tol(ell, n):
    return LL_RTOL * (abs(ell) + n)


def log_gauss(X, mean, cov):
    """log N(x_i; mean, cov) for every row, through a Cholesky factor."""
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)
    return -0.5 * (X.shape[1] * math.log(2 * math.pi) + 2 * np.log(np.diag(L)).sum() + (z * z).sum(axis=0))


def e_step(X, weights, means, covs):
    with np.errstate(divide="ignore"):
        a = np.stack([np.log(weights[k]) + log_gauss(X, means[k], covs[k]) for k in range(len(weights))], axis=1)
    m = a.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(a - m).sum(axis=1))
    return np.exp(a - lse[:, None]), float(lse.sum())


def m_step(X, resp, reg):
    n, d = X.shape
    Nk = resp.sum(axis=0)
    means = (resp.T @ X) / Nk[:, None]
    covs = np.stack([(resp[:, k, None] * (X - means[k])).T @ (X - means[k]) / Nk[k] + reg * np.eye(d)
                     for k in range(resp.shape[1])])
    return Nk / len(X), means, covs


def base_cov(X, reg):
    d = X.shape[1]
    diff = X - X.mean(axis=0)
    return diff.T @ diff / len(X) + reg * np.eye(d)


def fit(X, means0, n_iter, tol, reg):
    """The statement's EM from the initial means means0: returns (weights, means, covs, log_likelihoods)."""
    K = len(means0)
    w, mu, cov = np.full(K, 1.0 / K), np.array(means0, dtype=float), np.stack([base_cov(X, reg)] * K)
    lls, prev = [], -math.inf
    for it in range(n_iter):
        resp, ll = e_step(X, w, mu, cov)
        lls.append(ll)
        if ll - prev < tol or it == n_iter - 1:
            break
        prev = ll
        w, mu, cov = m_step(X, resp, reg)
    return w, mu, cov, lls


def kmeanspp(X, K, gen):
    """One k-means++ draw of K row indices (the judge's own fixtures and the stability guard)."""
    rows = [int(gen.integers(len(X)))]
    d2 = ((X - X[rows[0]]) ** 2).sum(axis=1)
    for _ in range(K - 1):
        rows.append(int(gen.choice(len(X), p=d2 / d2.sum())))
        d2 = np.minimum(d2, ((X - X[rows[-1]]) ** 2).sum(axis=1))
    return rows


def stable(X, K, reg, tol, tries=3):
    """False if EM from one of a few k-means++ starts lets a component collapse (its smallest covariance eigenvalue
    falls below 1e-3), where rounding differences between correct implementations grow fastest."""
    gen = np.random.default_rng(12345)
    for _ in range(tries):
        cov = fit(X, X[kmeanspp(X, K, gen)], 200, tol, reg)[2]
        if np.linalg.eigvalsh(cov).min() < 1e-3:
            return False
    return True


def init_candidates(X, K, reg, ll0):
    """Every set of K rows of X with distinct values whose initial log-likelihood is within tolerance of ll0."""
    n, d = X.shape
    _, first = np.unique(X, axis=0, return_index=True)
    rows = np.sort(first)
    L = np.linalg.cholesky(base_cov(X, reg))
    Z = np.linalg.solve(L, X.T).T                                      # whitened rows
    maha = ((Z[:, None, :] - Z[None, rows, :]) ** 2).sum(axis=2)         # (n, len(rows))
    logN = -0.5 * (d * math.log(2 * math.pi) + 2 * np.log(np.diag(L)).sum() + maha)
    combos = np.array(list(itertools.combinations(range(len(rows)), K)))
    lls = np.empty(len(combos))
    step = max(1, 400_000 // (n * K))
    for s in range(0, len(combos), step):
        a = logN[:, combos[s:s + step]] - math.log(K)                    # (n, chunk, K)
        m = a.max(axis=2, keepdims=True)
        lls[s:s + step] = (m[:, :, 0] + np.log(np.exp(a - m).sum(axis=2))).sum(axis=0)
    hits = np.flatnonzero(np.abs(lls - ll0) <= ll_tol(ll0, n))
    hits = hits[np.argsort(np.abs(lls[hits] - ll0))]
    return [rows[combos[h]] for h in hits[:20]]


def best_perm(got_means, want_means):
    K = len(want_means)
    return min(itertools.permutations(range(K)),
               key=lambda p: np.abs(got_means - want_means[list(p)]).max())


def as_model(got, X, K):
    """(weights, means, covs, lls) as float arrays, or a Chinese sentence saying what is malformed."""
    n, d = X.shape
    if not isinstance(got, dict):
        return f"应当返回 dict，得到 {type(got).__name__}"
    for key in ("weights", "means", "covs", "log_likelihoods"):
        if key not in got:
            return f"返回的 dict 缺少键 {key!r}"
    out = []
    for key, shape in (("weights", (K,)), ("means", (K, d)), ("covs", (K, d, d))):
        try:
            a = np.asarray(got[key], dtype=float)
        except Exception:
            return f"{key} 不是数值数组"
        if a.shape != shape:
            return f"{key} 的形状应当是 {shape}，得到 {a.shape}"
        if not np.all(np.isfinite(a)):
            return f"{key} 里有 nan 或 inf：{a.tolist()}"
        out.append(a)
    try:
        lls = [float(v) for v in got["log_likelihoods"]]
    except Exception:
        return "log_likelihoods 应当是数值列表"
    if not lls:
        return "log_likelihoods 是空的：至少要记录初始参数处的那一次 E 步"
    if not all(math.isfinite(v) for v in lls):
        return f"log_likelihoods 里有 nan 或 inf：{lls[:12]}{' …' if len(lls) > 12 else ''}"
    return out + [lls]


def model_ll(X, w, mu, cov):
    """ell at the given parameters, or a Chinese sentence if they are not a valid mixture."""
    if np.any(w < 0) or abs(w.sum() - 1) > 1e-9:
        return f"weights 应当非负且和为 1，得到 {w.tolist()}（和为 {w.sum()!r}）"
    for k in range(len(cov)):
        if np.abs(cov[k] - cov[k].T).max() > 1e-9 * (1 + np.abs(cov[k]).max()):
            return f"covs[{k}] 不对称：{cov[k].tolist()}"
        try:
            np.linalg.cholesky(cov[k])
        except np.linalg.LinAlgError:
            return f"covs[{k}] 不是正定矩阵：{cov[k].tolist()}"
    return e_step(X, w, mu, cov)[1]


def check_ll_consistent(X, model):
    """log_likelihoods[-1] must be ell at the returned parameters."""
    w, mu, cov, lls = model
    ell = model_ll(X, w, mu, cov)
    if isinstance(ell, str):
        return ell
    if abs(ell - lls[-1]) > ll_tol(ell, len(X)):
        return (f"log_likelihoods[-1] = {lls[-1]!r}，但返回的 weights/means/covs 处的 ℓ 是 {ell!r}：最后记录的 ℓ "
                f"应当恰好是返回参数处的 ℓ（停下时不再做这一轮的 M 步）")
    return None


def replay(X, rows, model, n_iter, tol, reg):
    """(matched E-steps, None) if the answer is EM from means X[rows]; else (matched E-steps, Chinese message)."""
    n = len(X)
    w_got, mu_got, cov_got, lls = model
    K = len(rows)
    w, mu, cov = np.full(K, 1.0 / K), X[rows].astype(float), np.stack([base_cov(X, reg)] * K)
    prev = -math.inf
    for j, got_ll in enumerate(lls):
        resp, ll = e_step(X, w, mu, cov)
        if abs(got_ll - ll) > ll_tol(ll, n):
            return j, (f"log_likelihoods[{j}] = {got_ll!r}，按题目的规则从同一个初始化做 EM，第 {j} 个 E 步的 ℓ "
                       f"应当是 {ll!r}" + ("（初始均值是 X 的第 " + ", ".join(map(str, rows)) + " 行）" if j else ""))
        delta = ll - prev
        amb = 2 * ll_tol(ll, n)
        last = j == len(lls) - 1
        if last and not (j == n_iter - 1 or delta <= tol + amb):
            return j, (f"第 {j} 个 E 步的 ℓ 比上一个提升了 {delta!r} ≥ tol = {tol!r}，也还没到 n_iter = {n_iter} 个 E 步，"
                       f"不应在这里停下")
        if not last and not (j < n_iter - 1 and (delta >= tol - amb or delta < -tol)):
            why = (f"已经跑了 n_iter = {n_iter} 个 E 步" if j == n_iter - 1 else
                   f"第 {j} 个 E 步的 ℓ 只比上一个提升了 {delta!r} < tol = {tol!r}")
            return j, f"{why}，应当在这里停下，可 log_likelihoods 还有 {len(lls)} 项"
        if last:
            break
        prev = ll
        w, mu, cov = m_step(X, resp, reg)
    p = list(best_perm(mu_got, mu))
    for name, got_a, want_a in (("weights", w_got, w[p]), ("means", mu_got, mu[p]), ("covs", cov_got, cov[p])):
        err = np.abs(got_a - want_a).max()
        if err > PARAM_TOL * (1 + np.abs(want_a).max()):
            return len(lls), (f"log_likelihoods 都对，但返回的 {name} 和从同一个初始化做 {len(lls)} 个 E 步后的参数不符"
                              f"（分量顺序已对齐）：应当是 {np.round(want_a, 6).tolist()}，得到 {np.round(got_a, 6).tolist()}")
    return len(lls), None


def chosen_rows(X, means):
    """Indices of the rows of X that the given means copy, or None if some mean is not a row of X."""
    out = []
    for m in means:
        hit = np.flatnonzero(np.abs(X - m).max(axis=1) <= 1e-12 * (1 + np.abs(X).max()))
        if not len(hit):
            return None
        out.append(int(hit[0]))
    return out


def check_init(X, K, reg, got):
    """fit_gmm(X, K, n_iter=1, reg=reg) returns the initial parameters: None if they follow the statement, else a
    Chinese sentence."""
    model = as_model(got, X, K)
    if isinstance(model, str):
        return model
    w, mu, cov, lls = model
    n, d = X.shape
    if len(lls) != 1:
        return f"n_iter = 1 时只做一次 E 步（在初始参数处），log_likelihoods 应当只有 1 项，得到 {len(lls)} 项"
    if np.abs(w - 1.0 / K).max() > 1e-12:
        return f"初始权重都应当是 1/K = {1.0 / K!r}，得到 {w.tolist()}"
    want = base_cov(X, reg)
    if np.abs(cov - want).max() > 1e-9 * (1 + np.abs(want).max()):
        diff = X - X.mean(axis=0)
        hint = ""
        if n > 1 and np.allclose(cov, diff.T @ diff / (n - 1) + reg * np.eye(d), rtol=1e-9, atol=1e-12):
            hint = "：看起来除以了 n - 1（np.cov 的默认 ddof=1），题目的 Sigma_hat 除以 n"
        elif np.allclose(cov, diff.T @ diff / n, rtol=1e-9, atol=1e-15):
            hint = "：看起来漏了 + reg·I"
        return (f"每个初始协方差都应当是 Sigma_hat + reg·I = {np.round(want, 9).tolist()}（reg = {reg!r}），"
                f"得到 {np.round(cov, 9).tolist()}{hint}")
    rows = chosen_rows(X, mu)
    if rows is None:
        return f"初始均值应当是 X 的 {K} 行数据点，得到 {np.round(mu, 6).tolist()}，其中有的不是 X 的行"
    if len({tuple(X[r]) for r in rows}) < K:
        return f"初始均值应当是 X 的 {K} 行*不同*的数据点，得到的是第 {rows} 行"
    return check_ll_consistent(X, model)


def check_fit(X, K, n_iter, tol, reg, got):
    """None if got is a correct fit_gmm(X, K, n_iter, tol, reg, seed) for some legal draw of the initial means,
    else a Chinese sentence about what is wrong."""
    model = as_model(got, X, K)
    if isinstance(model, str):
        return model
    lls = model[3]
    if len(lls) > n_iter:
        return f"log_likelihoods 有 {len(lls)} 项，可最多只跑 n_iter = {n_iter} 个 E 步"
    msg = check_ll_consistent(X, model)
    if msg:
        return msg
    cands = init_candidates(X, K, reg, lls[0])
    if not cands:
        return (f"log_likelihoods[0] = {lls[0]!r} 不是任何一种合法初始化处的 ℓ：初始权重都是 1/K，协方差都是 "
                f"Sigma_hat + reg·I（Sigma_hat 除以 n），均值是 X 的 {K} 个不同的行")
    best = (-1, None)
    for rows in cands:
        matched, msg = replay(X, rows, model, n_iter, tol, reg)
        if msg is None:
            return None
        best = max(best, (matched, msg), key=lambda t: t[0])
    return best[1]


def kmeanspp_sets(X, K, weight=None, first=None):
    """Exact distribution of the set of K k-means++ seeds, as {sorted row tuple: probability}: the first row uniform
    over all n rows (or always row `first`), each next one with probability proportional to weight(squared distances
    to every chosen row), by default the squared distance to the nearest chosen row. Enumerates every ordered draw,
    so keep n ** K small."""
    n = len(X)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    weight = weight or (lambda d2s: d2s.min(axis=0))
    sets = {}

    def walk(chosen, prob):
        if len(chosen) == K:
            key = tuple(sorted(chosen))
            sets[key] = sets.get(key, 0.0) + prob
            return
        wts = weight(D2[chosen]).astype(float)
        wts[chosen] = 0.0
        wts /= wts.sum()
        for j in np.flatnonzero(wts):
            walk(chosen + [int(j)], prob * wts[j])

    for f in (range(n) if first is None else [first]):
        walk([f], 1.0 / n if first is None else 1.0)
    return sets


def binom_two_sided(k, N, p):
    """Two-sided tail probability of k successes out of N under Binomial(N, p): twice the smaller tail, capped at 1."""
    def logpmf(j):
        if p <= 0.0 or p >= 1.0:
            return 0.0 if j == round(N * p) else -math.inf
        return (math.lgamma(N + 1) - math.lgamma(j + 1) - math.lgamma(N - j + 1)
                + j * math.log(p) + (N - j) * math.log1p(-p))
    lo = sum(math.exp(logpmf(j)) for j in range(0, k + 1))
    hi = sum(math.exp(logpmf(j)) for j in range(k, N + 1))
    return min(1.0, 2 * min(lo, hi))
