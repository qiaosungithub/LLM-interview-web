"""Helpers for tests.py: the statement's data generator, reference computations and test inputs."""
import torch

F64 = torch.float64
LAM = 4.0 * 0.5 ** torch.arange(8, dtype=F64)


def make_data(lam=LAM, n=2000, seed=0):
    """The statement's generator, unchanged."""
    g = torch.Generator().manual_seed(seed)
    d = lam.numel()
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=F64))
    Z = torch.randn(n, d, generator=g, dtype=F64)
    X = (Z * lam.sqrt()) @ Q.T + 3.0
    w_true = torch.randn(d, generator=g, dtype=F64)
    y = (X - X.mean(0)) @ w_true + 0.1 * torch.randn(n, generator=g, dtype=F64)
    return X, y


def eig_desc(X):
    """Eigenvalues (largest first) and unit eigenvectors (rows) of C = Xc^T Xc / n, via eigh, signs as in the
    statement: the largest-magnitude entry of each row is positive."""
    Xc = X - X.mean(0)
    evals, evecs = torch.linalg.eigh(Xc.T @ Xc / X.shape[0])
    vals, rows = evals.flip(0), evecs.flip(1).T.clone()
    idx = rows.abs().argmax(dim=1)
    rows *= torch.sign(rows[torch.arange(rows.shape[0]), idx])[:, None]
    return vals, rows


def well_separated(X, rel_gap=1e-3, entry_gap=1e-3):
    """True if C's eigenvalues are pairwise apart by rel_gap (relative to the largest), so each eigenvector is
    determined to about 1e-13, and in every eigenvector the two largest |entries| differ by entry_gap, so the
    sign convention is not decided by round-off."""
    vals, rows = eig_desc(X)
    if vals.numel() > 1 and float((vals[:-1] - vals[1:]).min()) < rel_gap * float(vals[0]):
        return False
    if rows.shape[1] > 1:
        top2 = rows.abs().sort(dim=1, descending=True).values[:, :2]
        if float((top2[:, 0] - top2[:, 1]).min()) < entry_gap:
            return False
    return True


def ill_conditioned(seed):
    """X (200, 6) with mean 3 in every column; the centred data has singular values logspace(0, -7, 6), so
    kappa(C) = 1e14. Returns (X, the eigenvalues of C = s^2 / n, largest first)."""
    g = torch.Generator().manual_seed(seed)
    n, k = 200, 6
    B = torch.cat([torch.ones(n, 1, dtype=F64), torch.randn(n, k, generator=g, dtype=F64)], dim=1)
    U = torch.linalg.qr(B)[0][:, 1:]                 # orthonormal columns, each orthogonal to the ones vector
    V = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=F64))[0]
    s = torch.logspace(0, -7, k, dtype=F64)
    return (U * s) @ V.T + 3.0, s ** 2 / n


def spd(seed, d, ratio):
    """A symmetric positive definite (d, d) matrix with eigenvalues 3 * ratio^i, in a random basis."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=F64))[0]
    lam = 3.0 * ratio ** torch.arange(d, dtype=F64)
    return (Q * lam) @ Q.T


def gd_iterates(Xc, y, lr, steps):
    """w_{t+1} = w_t - lr * Xc^T (Xc w_t - y) / n from w_0 = 0, with the gradient written out by hand."""
    n, d = Xc.shape
    w = torch.zeros(d, dtype=F64)
    W = [w]
    for _ in range(steps):
        w = w - lr * (Xc.T @ (Xc @ w - y)) / n
        W.append(w)
    return torch.stack(W)
