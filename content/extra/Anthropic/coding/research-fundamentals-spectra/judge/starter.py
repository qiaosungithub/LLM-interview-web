import torch

LAM = 4.0 * 0.5 ** torch.arange(8, dtype=torch.float64)   # population eigenvalues: 4, 2, 1, ..., 1/32


def make_data(lam=LAM, n=2000, seed=0):
    """X: (n, d) float64, population covariance Q diag(lam) Q^T, mean 3 in every coordinate.
    y: (n,) float64, a noisy linear function of the centred X (noise std 0.1)."""
    g = torch.Generator().manual_seed(seed)
    d = lam.numel()
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    Z = torch.randn(n, d, generator=g, dtype=torch.float64)
    X = (Z * lam.sqrt()) @ Q.T + 3.0
    w_true = torch.randn(d, generator=g, dtype=torch.float64)
    y = (X - X.mean(0)) @ w_true + 0.1 * torch.randn(n, generator=g, dtype=torch.float64)
    return X, y


X, y = make_data()
n, d = X.shape          # 2000, 8


def pca_svd(X):
    """Part 1: PCA of the rows of X through the SVD of the centred data; return (mean, eigvals, components)."""
    raise NotImplementedError


def power_iteration(matvec, d, steps, seed=0):
    """Part 2: power iteration with matvec(v) = C @ v; return (V, rho) for v_0 .. v_steps."""
    raise NotImplementedError


def gd_least_squares(Xc, y, lr, steps):
    """Part 3: gradient descent on ||Xc @ w - y||^2 / (2 n) from w = 0; return the iterates w_0 .. w_steps."""
    raise NotImplementedError
