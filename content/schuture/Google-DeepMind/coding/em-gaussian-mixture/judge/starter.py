import numpy as np


def fit_gmm(X: np.ndarray, K: int, n_iter: int = 200, tol: float = 1e-8, reg: float = 1e-6,
            seed: int = 0) -> dict:
    """X: (n, d). Fits a K-component full-covariance Gaussian mixture by EM. Returns a dict with
    weights (K,), means (K, d), covs (K, d, d), and log_likelihoods, a list holding ell(theta) at
    every E-step from the initial parameters through the returned ones (so log_likelihoods[-1] is
    ell at the returned weights/means/covs)."""
    raise NotImplementedError


def bic(X: np.ndarray, model: dict) -> float:
    """model: a dict as returned by fit_gmm, fitted on this same X. Returns the BIC score."""
    raise NotImplementedError
