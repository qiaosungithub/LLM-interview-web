# expect: WA
# Divides each weighted scatter matrix by n instead of N_k.
import numpy as np


def make_gmm_clusters(n_per_cluster: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, true_means): X stacks n_per_cluster points drawn from each of 3 two-dimensional
    Gaussians with well-separated means and a shared covariance; true_means holds the 3 generating
    means, shape (3, 2)."""
    rng = np.random.default_rng(seed)
    true_means = np.array([[0.0, 0.0], [8.0, 0.5], [3.5, 7.0]])
    true_cov = np.array([[1.0, 0.3], [0.3, 0.8]])
    X = np.concatenate([rng.multivariate_normal(m, true_cov, size=n_per_cluster) for m in true_means])
    return X, true_means


def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: returns K distinct row indices into X."""
    n = X.shape[0]
    centers = [int(rng.integers(n))]
    dist_sq = np.sum((X - X[centers[0]]) ** 2, axis=1)
    for _ in range(K - 1):
        probs = dist_sq / dist_sq.sum()
        # NOTE: every already-chosen point has distance 0 to itself, hence probability 0 -- never re-picked
        next_idx = int(rng.choice(n, p=probs))
        centers.append(next_idx)
        dist_sq = np.minimum(dist_sq, np.sum((X - X[next_idx]) ** 2, axis=1))
    return np.array(centers)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    # NOTE: a row that is -inf everywhere (every component underflows to 0 density for that point) would
    # otherwise subtract -inf from -inf and give nan; exp(-inf - 0) == 0 leaves the correct value, 0.
    m = np.where(np.isneginf(m), 0.0, m)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    diff = X - mean                                          # (n, d)
    sign, logdet = np.linalg.slogdet(cov)                    # NOTE: log-determinant via slogdet, never det(cov)
    solved = np.linalg.solve(cov, diff.T)                     # NOTE: solve, not an explicit matrix inverse
    mahalanobis = np.sum(diff.T * solved, axis=0)              # (n,): one (x-mu)^T Sigma^-1 (x-mu) per row
    return -0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis)


def _e_step(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, float]:
    n, K = X.shape[0], len(weights)
    log_joint = np.empty((n, K))
    for k in range(K):
        log_joint[:, k] = np.log(weights[k]) + _log_gaussian_pdf(X, means[k], covs[k])
    log_norm = _logsumexp(log_joint, axis=1)                   # (n,): log p(x_i) under the mixture
    resp = np.exp(log_joint - log_norm[:, None])                # (n, K): gamma_ik
    return resp, float(log_norm.sum())


def _m_step(X: np.ndarray, resp: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    K = resp.shape[1]
    Nk = resp.sum(axis=0)                                       # (K,)
    weights = Nk / n
    means = (resp.T @ X) / Nk[:, None]                           # (K, d)
    covs = np.empty((K, d, d))
    for k in range(K):
        diff = X - means[k]
        covs[k] = (resp[:, k, None] * diff).T @ diff / n
        covs[k] += reg * np.eye(d)                                # NOTE: guards a collapsing component's Sigma_k
    return weights, means, covs


def fit_gmm(X: np.ndarray, K: int, n_iter: int = 200, tol: float = 1e-8, reg: float = 1e-6,
            seed: int = 0) -> dict:
    n, d = X.shape
    rng = np.random.default_rng(seed)
    means = X[_kmeans_pp_init(X, K, rng)].copy()                 # (K, d): K distinct data points
    weights = np.full(K, 1.0 / K)
    base_cov = np.cov(X, rowvar=False, ddof=0).reshape(d, d) + reg * np.eye(d)
    covs = np.stack([base_cov.copy() for _ in range(K)])

    log_likelihoods: list = []
    prev_ll = -np.inf
    for iteration in range(n_iter):
        resp, ll = _e_step(X, weights, means, covs)
        log_likelihoods.append(ll)
        # NOTE: stop BEFORE this iteration's M-step, so the returned parameters always match
        # log_likelihoods[-1] exactly -- never one M-step ahead of the last logged value.
        if ll - prev_ll < tol or iteration == n_iter - 1:
            break
        prev_ll = ll
        weights, means, covs = _m_step(X, resp, reg)

    return {"weights": weights, "means": means, "covs": covs, "log_likelihoods": log_likelihoods}


def bic(X: np.ndarray, model: dict) -> float:
    n, d = X.shape
    K = model["weights"].shape[0]
    p = (K - 1) + K * d + K * d * (d + 1) // 2
    ell = model["log_likelihoods"][-1]
    return -2 * ell + p * np.log(n)
