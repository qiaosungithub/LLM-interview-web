# expect: WA
# Wrong for Part 1: takes the SVD of X itself, without subtracting the mean (question (c) of Part 1).
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


# ---------------------------------------------------------------- Part 1
def pca_svd(X):
    n = X.shape[0]
    mean = X.mean(dim=0)
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)   # S comes out largest first
    rows = torch.arange(Vh.shape[0])
    signs = torch.sign(Vh[rows, Vh.abs().argmax(dim=1)])         # sign of the largest-magnitude entry
    return mean, S ** 2 / n, Vh * signs[:, None]


# ---------------------------------------------------------------- Part 2
def power_iteration(matvec, d, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g, dtype=torch.float64)
    v = v / v.norm()
    V, rho = [v], []
    for _ in range(steps):
        w = matvec(v)
        rho.append(v @ w)              # v has unit norm, so this is the Rayleigh quotient of v_t
        v = w / w.norm()
        V.append(v)
    rho.append(v @ matvec(v))          # the Rayleigh quotient of the last vector
    return torch.stack(V), torch.stack(rho)


# ---------------------------------------------------------------- Part 3
def gd_least_squares(Xc, y, lr, steps):
    n, d = Xc.shape
    w = torch.zeros(d, dtype=Xc.dtype, requires_grad=True)
    W = [w.detach().clone()]
    for _ in range(steps):
        loss = ((Xc @ w - y) ** 2).sum() / (2 * n)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()                 # backward() accumulates into .grad, so clear it every step
        W.append(w.detach().clone())
    return torch.stack(W)
