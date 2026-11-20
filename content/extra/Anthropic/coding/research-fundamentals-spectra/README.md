# Spectra and convergence rates: PCA by SVD, power iteration, gradient descent

## Problem

This round takes place in a shared notebook and tests "ML research fundamentals": write a few small functions in PyTorch, make plots, and **say what each plot will look like before you draw it**. The interviewer gives you the data-generating function below; use it as is, without changes.

```py
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
```

Conventions shared by all three parts:

- $X_c$ is $X$ with each column's mean subtracted. The covariance matrix is always $C = \frac{1}{n} X_c^\top X_c$ (divide by $n$, not $n-1$).
- The eigenvalues of $C$, from largest to smallest, are $\hat\lambda_1 \ge \dots \ge \hat\lambda_d$, with unit eigenvectors $u_1, \dots, u_d$. `LAM` holds the population eigenvalues $\lambda_i$ used to generate the data; $\hat\lambda_i$ are computed from 2000 samples, and the two are not equal.
- Use float64 throughout, CPU only. Plot with matplotlib.

Each part has three steps: write the function; state the shape of the plot, then draw it; answer a few questions that call for numbers.

### Part 1 — PCA by SVD

```py
def pca_svd(X):
    """PCA of the rows of X through the SVD of the centred data. Do not form C.

    Return (mean, eigvals, components):
      mean        (d,)    column means of X
      eigvals     (d,)    eigenvalues of C, largest first
      components  (d, d)  row i is the unit eigenvector for eigvals[i]; its sign is fixed so that
                          the entry with the largest absolute value is positive
    """
```

1. Implement `pca_svd`. Then check it against `torch.linalg.eigh(C)`: the eigenvalues agree to within $10^{-10}$, and the eigenvectors are the same up to sign.
2. Draw a figure with two panels, both with a log-scale y-axis. Left panel: x-axis is the index $i = 1..8$; plot two lines, the sample eigenvalues $\hat\lambda_i$ and the population eigenvalues $\lambda_i$. Right panel: x-axis is the number of principal components kept, $k = 0..7$; plot the measured relative reconstruction error

   $$e_k = \frac{\lVert X_c - X_c V_k V_k^\top \rVert_F}{\lVert X_c \rVert_F}$$

   together with your prediction computed from the eigenvalues. Here the columns of $V_k$ are the first $k$ principal components, and $V_0$ is empty ($e_0 = 1$).
3. Before plotting, answer:
   - (a) What shape does each panel have? How does the slope of the right panel relate to the left? Roughly what percentage apart are the sample and population eigenvalues?
   - (b) How do you write $e_k$ in terms of the eigenvalues? Why can no other $k$-dimensional subspace do better?
   - (c) If you forget to subtract the mean and take the SVD of $X$ directly, roughly where does the first right singular vector point? Roughly what is $s_1^2/n$?

Reference numbers for the default arguments (`seed=0`), to confirm that `pca_svd` is right:

```text
eigvals  ≈ [3.8102, 2.0014, 1.0244, 0.4941, 0.2652, 0.1310, 0.0649, 0.0310]
‖mean‖²  ≈ 72.59
```

### Part 2 — Power iteration and its convergence rate

```py
def power_iteration(matvec, d, steps, seed=0):
    """Power iteration for the top eigenpair of a symmetric positive semi-definite matrix C that is
    only available as a function matvec(v) = C @ v, for v of shape (d,), float64.

    v_0 = g / ||g||, where g = torch.randn(d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).
    Then repeat `steps` times: v_{t+1} = matvec(v_t) / ||matvec(v_t)||.

    Return (V, rho):
      V    (steps + 1, d)  the unit vectors v_0 .. v_steps
      rho  (steps + 1,)    rho[t] = v_t^T C v_t, the Rayleigh quotient of v_t
    """
```

Let `Xc = X - mean`. Run 60 steps with `matvec = lambda v: Xc.T @ (Xc @ v) / n` (without forming $C$ explicitly). Taking $u_1$ and $\hat\lambda_1$ from Part 1 as ground truth, define two errors:

$$\sin\theta_t = \lVert v_t - (u_1^\top v_t)\,u_1 \rVert, \qquad \delta_t = \hat\lambda_1 - \rho_t .$$

1. Implement `power_iteration`.
2. Draw one plot: x-axis $t = 0..60$, log-scale y-axis, with two curves, $\sin\theta_t$ and $\delta_t$.
3. Before plotting, answer:
   - (a) What is the slope of the descending part of each curve? Write it in terms of the eigenvalues and give the number. What conditions does this result need?
   - (b) How low does each curve go before it stops falling? Which one bottoms out first?
   - (c) Without looking at the measurements, using only $\tan\theta_0$ and the eigenvalues, give an upper bound on the number of steps needed for $\sin\theta_t \le 10^{-8}$, then compare it with the measured number of steps.

### Part 3 — Gradient descent on least squares

```py
def gd_least_squares(Xc, y, lr, steps):
    """Gradient descent on L(w) = ||Xc @ w - y||^2 / (2 n), starting from w = 0, with the gradient
    taken from autograd (loss.backward()).
    Return W of shape (steps + 1, d): the iterates w_0 .. w_steps."""
```

Let $\hat w$ be the least-squares solution (computed with `torch.linalg.lstsq`). Project the error $w_t - \hat w$ onto the eigendirections from Part 1:

$$a_i(t) = u_i^\top (w_t - \hat w), \qquad \ell_i(t) = \tfrac{1}{2}\,\hat\lambda_i\, a_i(t)^2 .$$

1. Implement `gd_least_squares` and run 1200 steps with learning rate $\eta = 1/\hat\lambda_1$.
2. Draw a figure with two panels, both with a log-scale y-axis over $[10^{-20}, 10]$. The left panel's x-axis is $t = 0..60$, the right panel's $t = 0..1200$. Each panel has 9 lines: the 8 curves $\ell_i(t)$, plus the total excess loss $L(w_t) - L(\hat w)$.
3. Before plotting, answer:
   - (a) What shape is each $\ell_i$, and what is its slope? What about the total? Why plot the excess loss rather than $L(w_t)$ itself?
   - (b) How many steps does it take for the slowest component $|a_8(t)|$ to shrink to $1/1000$ of its initial value? Write the formula, give the number, then compare with the measurement.
   - (c) What is the largest usable learning rate? Run 200 steps each with $\eta = 1.9/\hat\lambda_1$ and $\eta = 2.1/\hat\lambda_1$. What happens to $a_1$?

## Reference solution

<details>
<summary>Show the reference solution</summary>

First, paste in the given code unchanged (plus a few imports needed for plotting). All figures are saved to a temporary directory; `plt.show()` is never called.

```python
import math
import os
import tempfile

import matplotlib.pyplot as plt
import torch

OUT_DIR = tempfile.mkdtemp(prefix="spectra_")              # figures are written here
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#73726c"

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
```

### Part 1

**Idea.** Take the SVD of the centred data, $X_c = U S V^\top$; then

$$C = \tfrac{1}{n} X_c^\top X_c = V\,\frac{S^2}{n}\,V^\top .$$

So the eigenvalues of $C$ are $s_i^2/n$ and the eigenvectors are the columns of $V$, i.e. the rows of the `Vh` returned by `torch.linalg.svd`. Three easy mistakes:

- `svd` returns singular values from largest to smallest, while `eigh` returns eigenvalues from smallest to largest; flip one of them before comparing.
- The sign of an eigenvector is arbitrary, and the two algorithms may give different signs. Either compare $|u_i^\top u_i'|$, or fix the sign as the problem does.
- `torch.cov` divides by $n-1$ by default. To match the problem's convention you have to pass `correction=0`.

Why not just call `eigh(C)`: forming $C$ squares the condition number, $\kappa(C) = \kappa(X_c)^2$. The smallest eigenvalue from `eigh` has a relative error of about machine precision times $\kappa(C)$; going through the SVD gives machine precision times $\sqrt{\kappa(C)}$. Here $\kappa(C) \approx 123$, so both routes are accurate to within $10^{-13}$ and you cannot see a difference; on badly conditioned data the difference is large (the verification code has an example with $\kappa(C) = 10^{14}$). Both routes cost $O(nd^2)$.

```python
def pca_svd(X):
    n = X.shape[0]
    mean = X.mean(dim=0)
    _, S, Vh = torch.linalg.svd(X - mean, full_matrices=False)   # S comes out largest first
    rows = torch.arange(Vh.shape[0])
    signs = torch.sign(Vh[rows, Vh.abs().argmax(dim=1)])         # sign of the largest-magnitude entry
    return mean, S ** 2 / n, Vh * signs[:, None]


mean, eigvals, comps = pca_svd(X)
Xc = X - mean
C = Xc.T @ Xc / n


def rank_k_error(Xc, basis, k):
    """Relative Frobenius error after projecting every row of Xc onto the first k rows of `basis`
    (the rows must be orthonormal)."""
    Vk = basis[:k]
    return ((Xc - Xc @ Vk.T @ Vk).norm() / Xc.norm()).item()


err_measured = torch.tensor([rank_k_error(Xc, comps, k) for k in range(d)], dtype=torch.float64)
tail = eigvals.flip(0).cumsum(0).flip(0)               # tail[k] = eigvals[k] + ... + eigvals[d - 1]
err_predicted = (tail / eigvals.sum()).sqrt()          # e_k for k = 0 .. d - 1
```

**(a) The shape of the plots.** Left panel: each population eigenvalue is half the previous one, so on a log y-axis they form a straight line that drops by $\ln 2$ per step. The sample eigenvalues hug this line, a few percent above or below. For Gaussian data with distinct eigenvalues and $n \gg d$, $\hat\lambda_i/\lambda_i - 1$ is approximately $\mathcal N(0,\, 2/n)$, with standard deviation $\sqrt{2/2000} \approx 3.2\%$. With `seed=0` the largest deviation is $6.1\%$.

Right panel: also roughly a straight line, but with only half the slope of the left panel. The reason is in (b): $e_k^2$ is proportional to the sum of the remaining eigenvalues, and the tail sum of a geometric series is about twice its first term, so $e_k \approx 2^{-k/2}$: each additional component kept divides the error by $\sqrt 2$. Because the tail sum is truncated at $d = 8$, the rightmost few points sit slightly below the line: for the population eigenvalues the exact value is $e_k^2 = (2^{-k} - 2^{-8})/(1 - 2^{-8})$.

**(b) The Eckart–Young–Mirsky theorem.** Let $A$ have singular values $s_1 \ge s_2 \ge \dots$, and let $A_k$ be the truncated SVD that keeps only the first $k$ singular values. Then for every matrix $B$ of rank at most $k$,

$$\lVert A - B \rVert_F \ge \lVert A - A_k \rVert_F = \sqrt{\textstyle\sum_{i>k} s_i^2}, \qquad \lVert A - B \rVert_2 \ge \lVert A - A_k \rVert_2 = s_{k+1}.$$

When $s_k > s_{k+1}$ the optimum in the Frobenius norm is unique; when $s_k = s_{k+1} > 0$ it is not. Take $A = X_c$: $X_c V_k V_k^\top$ is exactly $A_k$, and using $s_i^2 = n\hat\lambda_i$ gives

$$e_k = \sqrt{\frac{\sum_{i>k} \hat\lambda_i}{\sum_{i} \hat\lambda_i}} .$$

"No other subspace does better" follows directly from the theorem: projecting every row onto any $k$-dimensional subspace gives a matrix of rank at most $k$. With `seed=0`, $e_k$ is $1,\ 0.716,\ 0.507,\ 0.355,\ 0.251,\ 0.170,\ 0.111,\ 0.063$.

**(c) Forgetting to centre.** The SVD then factors $\frac1n X^\top X = C + \bar x \bar x^\top$. Every coordinate of the mean is close to 3, so $\lVert \bar x \rVert^2 \approx 8 \times 3^2 = 72$ ($72.6$ with `seed=0`), much larger than $\hat\lambda_1 \approx 3.8$. So the first right singular vector is almost exactly the mean direction $\bar x / \lVert \bar x \rVert$ (cosine above $0.999$), and $s_1^2/n \approx 74$. More precisely,

$$\lVert \bar x \rVert^2 \;\le\; s_1^2/n \;\le\; \lVert \bar x \rVert^2 + \hat\lambda_1 .$$

In other words, the "first principal component" explains the mean, not the variance. The other 7 values are not equal to any $\hat\lambda_i$ either; they are only sandwiched between them (the interlacing property of a rank-one update): the $(i+1)$-th uncentred value is at most $\hat\lambda_i$, and the $i$-th uncentred value is at least $\hat\lambda_i$.

```python
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.0, 3.6))
index = torch.arange(1, d + 1)
ax_l.semilogy(index, LAM, "--", color=GRAY, lw=1.5, label="population")
ax_l.semilogy(index, eigvals, "o-", color=BLUE, lw=2, ms=5, label="sample (SVD)")
ax_l.set_xlabel("index i")
ax_l.set_ylabel("eigenvalue of C")
ax_l.legend(frameon=False)
ks = torch.arange(d)
ax_r.semilogy(ks, err_predicted, "--", color=GRAY, lw=1.5, label="predicted from eigenvalues")
ax_r.semilogy(ks, err_measured, "o", color=BLUE, ms=5, label="measured")
ax_r.set_xlabel("components kept, k")
ax_r.set_ylabel("relative reconstruction error")
ax_r.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "part1_spectrum.png"), dpi=120)
plt.close(fig)
```

### Part 2

**Idea.** Expand the starting vector in the eigenvectors, $v_0 = \sum_i c_i u_i$. Each step multiplies by $C$ once and normalizes, so

$$v_t \;\propto\; C^t v_0 = \sum_i c_i\, \hat\lambda_i^{\,t}\, u_i .$$

Every component other than $u_1$ shrinks relative to the $u_1$ component by a factor $\hat\lambda_i/\hat\lambda_1$ per step; the slowest is $i = 2$. This gives the exact expressions:

$$\tan\theta_t = \frac{\sqrt{\sum_{i \ge 2} c_i^2 \hat\lambda_i^{2t}}}{|c_1|\,\hat\lambda_1^{\,t}}, \qquad \delta_t = \hat\lambda_1 - \rho_t = \frac{\sum_{i \ge 2} c_i^2 \hat\lambda_i^{2t}\,(\hat\lambda_1 - \hat\lambda_i)}{\sum_{i} c_i^2 \hat\lambda_i^{2t}} .$$

The implementation needs only one `matvec` per step, and one `matvec` costs $O(nd)$.

```python
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


def angle_errors(V, u):
    """sin and tan of the angle between each row of V and the unit vector u."""
    cos = V @ u
    sin = (V - cos[:, None] * u).norm(dim=1)       # accurate down to 1e-16; sqrt(1 - cos^2) is not
    return sin, sin / cos.abs()


matvec = lambda v: Xc.T @ (Xc @ v) / n             # C @ v without forming C
STEPS = 60
V, rho = power_iteration(matvec, d, STEPS)
u1, lam1 = comps[0], eigvals[0]
sin_t, tan_t = angle_errors(V, u1)
gap_t = lam1 - rho                                  # delta_t
rate = (eigvals[1] / eigvals[0]).item()             # 0.5253 for seed 0
ts = torch.arange(STEPS + 1, dtype=torch.float64)
tan_bound = tan_t[0] * rate ** ts                   # tan(theta_t) <= rate^t * tan(theta_0)
steps_bound = math.ceil(math.log(tan_t[0].item() / 1e-8) / math.log(1.0 / rate))
steps_measured = int((sin_t <= 1e-8).nonzero()[0])
```

**(a) Slopes.** Conditions: $C$ is symmetric positive semi-definite; the largest eigenvalue is strictly larger than the second, $\hat\lambda_1 > \hat\lambda_2$; the starting vector is not orthogonal to $u_1$, $c_1 \ne 0$ (a random start satisfies this with probability 1). Under these conditions, for every $t$,

$$\tan\theta_t \le \left(\frac{\hat\lambda_2}{\hat\lambda_1}\right)^{t} \tan\theta_0, \qquad (\hat\lambda_1 - \hat\lambda_2)\sin^2\theta_t \;\le\; \delta_t \;\le\; (\hat\lambda_1 - \hat\lambda_d)\sin^2\theta_t .$$

If the starting vector also has a component in the eigenspace of $\hat\lambda_2$ (again true with probability 1 for a random start), the ratio between consecutive steps converges to this bound: $\sin\theta_t$ is multiplied by $\hat\lambda_2/\hat\lambda_1$ per step, and $\delta_t$ by $(\hat\lambda_2/\hat\lambda_1)^2$. So on a log y-axis they are two straight lines:

- $\sin\theta_t$ has slope $\ln(\hat\lambda_2/\hat\lambda_1) = \ln 0.5253 = -0.644$ per step, about one order of magnitude every 3.6 steps.
- $\delta_t$ has twice that slope, $-1.288$ per step, about one order of magnitude every 1.8 steps. The eigenvalue error is of the order of the square of the angle error; this is a general property of the Rayleigh quotient for symmetric matrices.

Note that the ratio must be computed from the **sample** eigenvalues. The matrix being iterated is $C$, whose ratio is $0.5253$, not the population value $0.5$. A $5\%$ difference sounds small, but after 50 steps the prediction is off by a factor of $(0.5253/0.5)^{50} \approx 12$, more than an order of magnitude on the plot.

**(b) Where they bottom out.** Both lines fall until they hit round-off. $\sin\theta_t$ stops around $10^{-15}$, and $\delta_t$ at a few times $10^{-15}$ (the last digit of $\hat\lambda_1 \approx 3.8$); $\delta_t$ can also be 0 or slightly negative, so take its absolute value and clip it before plotting on a log scale. $\delta_t$ has twice the slope, so it bottoms out first: around $t = 25$; $\sin\theta_t$ only after $t = 50$.

There is a numerical trap here: if you compute the sine as $\sqrt{1 - \cos^2\theta_t}$, the difference between $\cos\theta_t$ and 1 is rounded away once it drops below $10^{-16}$, so the result is distorted near $10^{-8}$ and then becomes exactly 0, and the curve looks as if it suddenly stops at $10^{-8}$. The definition in the problem (subtract the projection onto $u_1$, then take the norm) does not have this problem.

**(c) An upper bound on the steps.** Since $\sin\theta_t \le \tan\theta_t \le (\hat\lambda_2/\hat\lambda_1)^t \tan\theta_0$, as soon as

$$t \;\ge\; \frac{\ln(\tan\theta_0 / \varepsilon)}{\ln(\hat\lambda_1/\hat\lambda_2)}$$

we have $\sin\theta_t \le \varepsilon$. With `seed=0`, $\tan\theta_0 = 2.35$; taking $\varepsilon = 10^{-8}$ gives a bound of 30 steps, and the measurement reaches it at step 28. The measurement beats the bound because the bound charges all components outside $u_1$ to the slowest one, $u_2$, whereas in fact some of them lie in directions that decay faster.

```python
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.semilogy(ts, sin_t, color=BLUE, lw=2, label="sin(theta_t)")
ax.semilogy(ts, gap_t.abs().clamp(min=1e-16), color=ORANGE, lw=2, label="lambda_1 - rho_t")   # can be 0 or slightly negative at the floor
ax.semilogy(ts, tan_bound, "--", color=GRAY, lw=1.2, label="bound: tan(theta_0) * (lambda_2 / lambda_1)^t")
ax.semilogy(ts, (lam1 - eigvals[-1]) * tan_bound ** 2, ":", color=GRAY, lw=1.2,
            label="bound: (lambda_1 - lambda_d) * (previous line)^2")
ax.set_ylim(1e-17, 1e2)
ax.set_xlabel("iteration t")
ax.set_ylabel("error")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "part2_power_iteration.png"), dpi=120)
plt.close(fig)
```

### Part 3

**Idea.** $\hat w$ satisfies the normal equations $X_c^\top (X_c \hat w - y) = 0$, so the gradient can be written $\nabla L(w) = \frac1n X_c^\top (X_c w - y) = C\,(w - \hat w)$. The Hessian is exactly the $C$ from Part 1. One gradient descent step is

$$w_{t+1} - \hat w = (I - \eta C)(w_t - \hat w),$$

and along the eigendirections of $C$ the components do not interact; each follows its own geometric sequence:

$$a_i(t) = (1 - \eta \hat\lambda_i)^t\, a_i(0), \qquad L(w_t) - L(\hat w) = \tfrac12 (w_t - \hat w)^\top C\,(w_t - \hat w) = \sum_i \ell_i(t).$$

Each step costs $O(nd)$.

```python
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


def loss_fn(W):
    """L(w) for one w of shape (d,), or for every row of a stack of shape (T, d)."""
    return ((W @ Xc.T - y) ** 2).sum(dim=-1) / (2 * n)


w_hat = torch.linalg.lstsq(Xc, y[:, None]).solution[:, 0]
loss_floor = loss_fn(w_hat)                        # about sigma^2 / 2 = 0.005, not 0
lr = 1.0 / eigvals[0].item()
GD_STEPS = 1200
W = gd_least_squares(Xc, y, lr, GD_STEPS)
A = (W - w_hat) @ comps.T                          # A[t, i] = a_i(t)
per_dir = 0.5 * eigvals * A ** 2                   # per_dir[t, i] = l_i(t)
excess = loss_fn(W) - loss_floor

kappa = (eigvals[0] / eigvals[-1]).item()          # condition number of C: 122.75 for seed 0
T_pred = math.ceil(math.log(1000.0) / -math.log(1.0 - 1.0 / kappa))
T_meas = int((A[:, -1].abs() <= 1e-3 * A[0, -1].abs()).nonzero()[0])

# learning rates just below and just above the stability threshold 2 / lambda_1
A_below = (gd_least_squares(Xc, y, 1.9 / eigvals[0].item(), 200) - w_hat) @ comps.T
A_above = (gd_least_squares(Xc, y, 2.1 / eigvals[0].item(), 200) - w_hat) @ comps.T
```

**(a) The shape of the plots.** With $\eta = 1/\hat\lambda_1$ the per-step factor is $1 - \hat\lambda_i/\hat\lambda_1$, so

$$\ell_i(t) = \ell_i(0)\left(1 - \frac{\hat\lambda_i}{\hat\lambda_1}\right)^{2t} .$$

On a log y-axis every $\ell_i$ is a straight line with slope $2\ln(1 - \hat\lambda_i/\hat\lambda_1)$; the smaller the eigenvalue, the flatter the line. The 8 lines fan out like an open fan:

- Direction 1 has factor 0: it goes to zero in one step and drops straight off the plot.
- Direction 2 has slope $2\ln(1 - 0.5253) = -1.49$ per step and falls below $10^{-10}$ within a dozen or so steps.
- Direction 8 has slope $2\ln(1 - 1/122.75) = -0.0164$ per step and falls only 8.5 orders of magnitude in 1200 steps.

The total excess loss is the sum of these 8 lines. On a log plot a sum of straight lines roughly follows the highest of them, so it is a convex curve: steep at first and flatter and flatter later (with `seed=0` the initial error sits mostly in direction 3); it flattens a little each time a faster direction has decayed away, and finally runs along the direction-8 line (all other directions have decayed by then). The left panel shows the fan of the first few directions clearly; the right panel shows the long tail set by directions 7 and 8 at the end.

Why not plot $L(w_t)$ itself: $y$ is noisy, so $L(w_t)$ tends to $L(\hat w) \approx \sigma^2/2 = 0.005$ ($0.0049$ with `seed=0`), not to 0. Plotted directly, the curve flattens out at $0.005$ after a few dozen steps and hides all the later decay.

**(b) Steps and the condition number.** The slowest direction is the one with the smallest eigenvalue, with per-step factor $1 - 1/\hat\kappa$, where $\hat\kappa = \hat\lambda_1/\hat\lambda_8$ is the condition number of $C$. For $|a_8(t)| \le \varepsilon\,|a_8(0)|$ you need

$$t \;\ge\; \frac{\ln(1/\varepsilon)}{-\ln(1 - 1/\hat\kappa)} \;\approx\; \hat\kappa \,\ln(1/\varepsilon).$$

With `seed=0`, $\hat\kappa = 122.75$; taking $\varepsilon = 10^{-3}$ and rounding up gives 845 steps, and the measurement is also exactly step 845. The approximation $\hat\kappa \ln 1000 = 848$ is off by only 3 steps. The takeaway: the number of steps is proportional to the condition number, and each further factor of 10 in accuracy costs another $\hat\kappa \ln 10 \approx 283$ steps.

**(c) The stability threshold.** When $C$ is positive definite (here $\hat\lambda_8 > 0$), gradient descent converges from every starting point if and only if every factor has absolute value below 1, that is,

$$0 < \eta < \frac{2}{\hat\lambda_1} .$$

- $\eta = 1.9/\hat\lambda_1$: $a_1$ is multiplied by $1 - 1.9 = -0.9$ per step; it alternates in sign and shrinks in magnitude, so it converges.
- $\eta = 2.1/\hat\lambda_1$: $a_1$ is multiplied by $-1.1$ per step; it alternates in sign and grows in magnitude. After 200 steps it has grown by $1.1^{200} \approx 1.9 \times 10^{8}$, and the excess loss is more than $10^{15}$ times its initial value.

The other 7 directions still converge at $\eta = 2.1/\hat\lambda_1$, because $\eta \hat\lambda_i \le 2.1 \times 0.5253 < 2$. When the learning rate is only slightly too large, the divergence happens only along the single steepest direction, showing up as an oscillation back and forth along that direction that grows with every step.

```python
RAMP = ["#0d366b", "#104281", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#6da7ec", "#86b6ef"]   # dark = large eigenvalue

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
for ax, t_max in zip(axes, (60, GD_STEPS)):
    for i in range(d):
        ax.semilogy(per_dir[: t_max + 1, i].clamp(min=1e-300), color=RAMP[i], lw=1.5, label=f"direction {i + 1}")
    ax.semilogy(excess[: t_max + 1].clamp(min=1e-300), "--", color="black", lw=1.8, label="total excess loss")
    ax.set_xlim(0, t_max)
    ax.set_ylim(1e-20, 10)
    ax.set_xlabel("step t")
axes[0].set_ylabel("loss above the minimum")
axes[1].legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "part3_gd_directions.png"), dpi=120)
plt.close(fig)
```

### Follow-up questions

**1. For a constant step size, is $1/\hat\lambda_1$ the best choice?** No. The best constant step makes the factors of the largest and smallest eigenvalues equal in absolute value: $\eta = 2/(\hat\lambda_1 + \hat\lambda_d)$, at which every direction's factor is at most $(\hat\kappa - 1)/(\hat\kappa + 1)$ in absolute value. The same factor-of-1000 reduction then takes 424 steps, half of 845. The price is that direction 1 no longer vanishes in one step; it is now as slow as direction 8 and flips sign every step.

**2. Can all directions be made equally fast?** Whiten the inputs with the PCA from Part 1: $Z = X_c V \hat\Lambda^{-1/2}$ (the columns of $V$ are the eigenvectors of the covariance matrix, i.e. the right singular vectors of the SVD). Its covariance is the identity, with condition number 1. Then with $\eta = 1$ a single gradient descent step lands on the least-squares solution. Input standardization and preconditioning all serve this purpose: bringing the condition number down.

**3. How much faster is momentum?** Gradient descent with momentum (heavy ball), with optimal parameters, has a per-step factor of about $(\sqrt{\hat\kappa} - 1)/(\sqrt{\hat\kappa} + 1)$, so the number of steps goes from proportional to $\hat\kappa$ to proportional to $\sqrt{\hat\kappa}$. Here, shrinking the error norm by 1000 takes only 49 steps. Conjugate gradient reaches the exact solution in at most $d = 8$ steps in exact arithmetic.

**4. What if the power iteration's starting vector happens to be orthogonal to $u_1$?** In exact arithmetic it converges to $u_2$. In floating point, round-off introduces a small $u_1$ component, which grows relative to the rest by a factor $\hat\lambda_1/\hat\lambda_2 \approx 1.9$ per step and takes over only after several dozen steps. Until then the Rayleigh quotient sits still at $\hat\lambda_2$. So a stopping rule like "stop when $\rho_t$ barely changes between consecutive steps" can stop at the wrong eigenvalue. The verification code demonstrates this with the starting vector $u_2 + 10^{-10} u_1$.

**5. How do you get the second eigenvector?** Deflate: replace `matvec` with $v \mapsto Cv - \hat\lambda_1 u_1 (u_1^\top v)$, whose largest eigenvalue is $\hat\lambda_2$. Run power iteration again; the convergence rate is set by $\hat\lambda_3/\hat\lambda_2 \approx 0.512$. When you need the top several eigenvectors at once, subspace iteration (one QR per step) is more common, or just use Lanczos.

**6. How do the plots change in float32?** The slopes stay the same, but the floor is much higher: $\sin\theta_t$ stops around $10^{-7}$ instead of $10^{-15}$. That is why experiments like these, which need to see a dozen or more orders of magnitude, all use float64.

<details>
<summary>Verification code (runnable)</summary>

```python
def fit_slope(t, values):
    """Least-squares slope of log(values) against t."""
    t = t.double()
    design = torch.stack([t, torch.ones_like(t)], dim=1)
    return torch.linalg.lstsq(design, values.double().log()[:, None]).solution[0, 0].item()


eye = torch.eye(d, dtype=torch.float64)

# ---------- the data, and the reference numbers quoted in the statement ----------
assert X.shape == (2000, 8) and y.shape == (2000,) and X.dtype == y.dtype == torch.float64
assert torch.equal(LAM, torch.tensor([4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=torch.float64))
quoted = torch.tensor([3.8102, 2.0014, 1.0244, 0.4941, 0.2652, 0.1310, 0.0649, 0.0310], dtype=torch.float64)
assert torch.allclose(eigvals, quoted, rtol=0, atol=6e-5)
assert abs((mean @ mean).item() - 72.59) < 5e-3

# ---------- Part 1: SVD against the eigendecomposition ----------
evals, evecs = torch.linalg.eigh(C)                       # ascending order, arbitrary signs
assert bool((evals[1:] >= evals[:-1]).all())
assert torch.allclose(eigvals, evals.flip(0), rtol=0, atol=1e-10)
assert bool((eigvals[:-1] > eigvals[1:]).all())           # pca_svd returns them largest first
assert torch.allclose(comps @ comps.T, eye, atol=1e-12)
assert torch.allclose(C @ comps.T, comps.T * eigvals, atol=1e-10)        # C u_i = lambda_i u_i
assert bool(((comps @ evecs.flip(1)).diagonal().abs() > 1 - 1e-10).all())   # same vectors up to sign
rows = torch.arange(d)
assert bool((comps[rows, comps.abs().argmax(dim=1)] > 0).all())          # the sign convention
assert torch.equal(mean, X.mean(dim=0))
# the 1/n convention: torch.cov divides by n - 1 unless told otherwise
assert torch.allclose(torch.cov(X.T, correction=0), C, atol=1e-12)
assert torch.allclose(torch.cov(X.T), C * n / (n - 1), atol=1e-12)

# sample eigenvalues scatter around the population ones with relative standard deviation sqrt(2 / n)
rel_dev = ((eigvals - LAM) / LAM).abs().max().item()
assert abs(rel_dev - 0.061) < 5e-4 and rel_dev < 0.15
ratios = torch.stack([pca_svd(make_data(seed=1000 + s)[0])[1] / LAM for s in range(200)])
assert bool(((ratios.std(dim=0) / math.sqrt(2 / n) - 1).abs() < 0.25).all())
assert bool(((ratios.mean(dim=0) - 1).abs() < 0.02).all())
assert abs(math.sqrt(2 / n) - 0.032) < 5e-4

# why SVD rather than eigh(C): an ill-conditioned example with known singular values, kappa(C) = 1e14
g = torch.Generator().manual_seed(7)
U_ill, _ = torch.linalg.qr(torch.randn(200, 6, generator=g, dtype=torch.float64))
V_ill, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
s_ill = torch.logspace(0, -7, 6, dtype=torch.float64)
A_ill = (U_ill * s_ill) @ V_ill.T
err_svd = abs(torch.linalg.svdvals(A_ill)[-1].item() ** 2 / s_ill[-1].item() ** 2 - 1)
err_eigh = abs(torch.linalg.eigvalsh(A_ill.T @ A_ill)[0].item() / s_ill[-1].item() ** 2 - 1)
assert err_svd < 1e-6 and err_eigh > 1e-4 and err_eigh > 100 * err_svd
# on the well-conditioned C of this problem both routes are accurate
assert ((eigvals - evals.flip(0)).abs() / eigvals).max().item() < 1e-13

# Eckart-Young: the measured error equals the formula, in both norms
assert torch.allclose(err_measured, err_predicted, rtol=0, atol=1e-12)
quoted_err = torch.tensor([1, 0.716, 0.507, 0.355, 0.251, 0.170, 0.111, 0.063], dtype=torch.float64)
assert torch.allclose(err_measured, quoted_err, rtol=0, atol=6e-4)
S_c = torch.linalg.svdvals(Xc)
assert torch.allclose(S_c ** 2, n * eigvals, rtol=1e-12)
for k in range(d):
    resid = Xc - Xc @ comps[:k].T @ comps[:k]
    assert abs(resid.norm().item() - S_c[k:].pow(2).sum().sqrt().item()) < 1e-9
    assert abs(torch.linalg.matrix_norm(resid, ord=2).item() - S_c[k].item()) < 1e-9

# ... and no other k-dimensional subspace does better: random ones, the "next" k components, a small rotation
g = torch.Generator().manual_seed(1)
for k in range(1, d):
    best = err_predicted[k].item()
    for _ in range(100):
        Qk, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
        assert rank_k_error(Xc, Qk.T, k) > best
    assert rank_k_error(Xc, comps[1:], k) > best
    rotated = comps.clone()
    rotated[k - 1] = math.cos(0.01) * comps[k - 1] + math.sin(0.01) * comps[k]
    assert rank_k_error(Xc, rotated, k) > best

# the shape of the right panel: about 2^(-k/2), i.e. half the slope of the spectrum
population_err = torch.tensor([math.sqrt((2.0 ** -k - 2.0 ** -8) / (1 - 2.0 ** -8)) for k in range(d)],
                              dtype=torch.float64)
assert bool(((err_predicted / population_err - 1).abs() < 0.05).all())
step_ratio = err_predicted[1:6] / err_predicted[:5]
assert bool(((step_ratio - 2 ** -0.5).abs() < 0.04).all())
assert abs(fit_slope(ks[:6], err_predicted[:6]) / fit_slope(index, eigvals) - 0.5) < 0.03

# forgetting to centre: the top singular direction is the mean, and s_1^2 / n is about ||mean||^2
_, S_raw, Vh_raw = torch.linalg.svd(X, full_matrices=False)
ev_raw = S_raw ** 2 / n
mean_sq = (mean @ mean).item()
assert torch.allclose(X.T @ X / n, C + torch.outer(mean, mean), atol=1e-10)
assert abs((Vh_raw[0] @ mean / mean.norm()).item()) > 0.999
assert mean_sq <= ev_raw[0].item() <= mean_sq + eigvals[0].item()
assert abs(ev_raw[0].item() - 74) < 0.5 and abs(mean_sq - 72.6) < 0.05
assert bool((ev_raw[1:] <= eigvals[:-1] + 1e-10).all()) and bool((eigvals <= ev_raw + 1e-10).all())   # interlacing
assert (ev_raw[1:] - eigvals[1:]).abs().min().item() > 1e-3            # none of them equals a centred eigenvalue

# ---------- Part 2: power iteration ----------
assert V.shape == (STEPS + 1, d) and rho.shape == (STEPS + 1,)
g0 = torch.randn(d, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
assert torch.allclose(V[0], g0 / g0.norm(), rtol=0, atol=1e-15)
assert torch.allclose(V.norm(dim=1), torch.ones(STEPS + 1, dtype=torch.float64), atol=1e-12)
assert torch.allclose(V[1], C @ V[0] / (C @ V[0]).norm(), atol=1e-12)
assert torch.allclose(rho, ((V @ C) * V).sum(dim=1), atol=1e-12)
assert bool((rho[1:] >= rho[:-1] - 1e-12).all())           # for a PSD matrix the Rayleigh quotient never decreases
assert abs(rate - 0.5253) < 5e-5 and abs(math.log(rate) + 0.644) < 5e-4

# the exact formulas for tan(theta_t) and delta_t
c = comps @ V[0]                                           # c_i = u_i^T v_0
weight = c ** 2 * eigvals ** (2 * ts[:, None])             # weight[t, i] = c_i^2 lambda_i^(2t)
tan_exact = (weight[:, 1:].sum(dim=1) / weight[:, 0]).sqrt()
gap_exact = (weight * (lam1 - eigvals)).sum(dim=1) / weight.sum(dim=1)
assert torch.allclose(tan_t, tan_exact, rtol=1e-6, atol=1e-13)
assert torch.allclose(gap_t, gap_exact, rtol=1e-6, atol=1e-13)

# the bounds hold at every step (the absolute slack covers the round-off floor)
assert bool((tan_t <= tan_bound * (1 + 1e-9) + 1e-13).all())
assert bool((gap_t <= (lam1 - eigvals[-1]) * sin_t ** 2 * (1 + 1e-9) + 1e-13).all())
assert bool((gap_t >= (lam1 - eigvals[1]) * sin_t ** 2 * (1 - 1e-9) - 1e-13).all())

# the slopes of the two straight segments
window = (sin_t < 1e-3) & (sin_t > 1e-12)
slope_sin = fit_slope(ts[window], sin_t[window])
assert abs(slope_sin / math.log(rate) - 1) < 0.01
window_gap = (gap_t < 1e-4) & (gap_t > 1e-11)
slope_gap = fit_slope(ts[window_gap], gap_t[window_gap])
assert abs(slope_gap / (2 * math.log(rate)) - 1) < 0.01
assert abs(slope_gap + 1.288) < 0.01
assert abs(math.log(10) / -math.log(rate) - 3.6) < 0.05 and abs(math.log(10) / (-2 * math.log(rate)) - 1.8) < 0.05
# the population ratio 0.5 predicts a visibly different slope
assert abs(slope_sin / math.log(0.5) - 1) > 0.05
assert abs((rate / 0.5) ** 50 - 12) < 0.5

# floors, and which curve gets there first
assert sin_t[55:].max().item() < 1e-14 and gap_t[25:].abs().max().item() < 1e-13
assert sin_t[25].item() > 1e-8 and sin_t[50].item() > 2e-15
naive_sin = (1 - (V @ u1) ** 2).clamp(min=0).sqrt()       # sqrt(1 - cos^2): useless below 1e-8
assert bool(((naive_sin[32:] == 0) | (naive_sin[32:] > 1e-9)).all()) and sin_t[32].item() < 1e-9
assert sin_t[32:45].min().item() > 1e-13                  # ... while the true value is still far from 0

# number of iterations: bound against measurement
assert abs(tan_t[0].item() - 2.35) < 5e-3
assert steps_bound == 30 and steps_measured == 28
assert sin_t[steps_measured].item() <= 1e-8 < sin_t[steps_measured - 1].item()

# ---------- Part 3: gradient descent on least squares ----------
assert W.shape == (GD_STEPS + 1, d) and bool((W[0] == 0).all())
# the gradient that autograd computes is C (w - w_hat), so the Hessian is C
w_probe = torch.randn(d, generator=torch.Generator().manual_seed(2), dtype=torch.float64).requires_grad_(True)
(grad_probe,) = torch.autograd.grad(((Xc @ w_probe - y) ** 2).sum() / (2 * n), w_probe)
assert torch.allclose(grad_probe, C @ (w_probe.detach() - w_hat), atol=1e-10)
assert torch.allclose(Xc.T @ (Xc @ w_hat - y), torch.zeros(d, dtype=torch.float64), atol=1e-8)   # normal equations

# every component follows its own geometric sequence, and the losses add up
steps_axis = torch.arange(GD_STEPS + 1, dtype=torch.float64)
factors = 1 - lr * eigvals
assert torch.allclose(A, A[0] * factors ** steps_axis[:, None], rtol=0, atol=1e-12)
assert torch.allclose(excess, per_dir.sum(dim=1), rtol=0, atol=1e-12)
assert abs(A[1, 0].item()) < 1e-12 and abs(A[0, 0].item()) > 0.1        # direction 1 is gone after one step
assert per_dir[0].argmax().item() == 2 and per_dir[0, 2].item() > 0.7 * excess[0].item()   # direction 3 dominates at t = 0
for i in range(1, d):
    visible = per_dir[:, i] > 1e-16
    assert int(visible.sum()) >= 10
    slope = fit_slope(steps_axis[visible], per_dir[visible, i])
    assert abs(slope / (2 * math.log(factors[i].item())) - 1) < 1e-4
assert abs(2 * math.log(factors[1].item()) + 1.49) < 5e-3 and abs(2 * math.log(factors[-1].item()) + 0.0164) < 5e-5
assert abs(-2 * math.log(factors[-1].item()) * GD_STEPS / math.log(10) - 8.5) < 0.05
assert per_dir[15, 1].item() < 1e-10
# the total ends parallel to the slowest direction
assert abs((excess[-1] / excess[-2]).item() / factors[-1].item() ** 2 - 1) < 1e-4
assert bool((excess[1:] < excess[:-1]).all())

# the loss itself levels off at sigma^2 / 2, not at 0
assert abs(loss_floor.item() - 0.0049) < 5e-5 and abs(loss_floor.item() / 0.005 - 1) < 0.15
assert loss_fn(W)[100].item() < 2 * loss_floor.item()

# steps and the condition number
assert abs(kappa - 122.75) < 5e-3
assert T_pred == T_meas == 845
assert abs(kappa * math.log(1000) - 848) < 0.5 and abs(kappa * math.log(10) - 283) < 0.5

# the stability threshold 2 / lambda_1
assert bool(((A_below[1:6, 0] / A_below[:5, 0] + 0.9).abs() < 1e-9).all())    # factor -0.9: shrinking, alternating
assert bool(((A_above[1:6, 0] / A_above[:5, 0] + 1.1).abs() < 1e-9).all())    # factor -1.1: growing, alternating
excess_below = (0.5 * eigvals * A_below ** 2).sum(dim=1)
excess_above = (0.5 * eigvals * A_above ** 2).sum(dim=1)
assert bool((excess_below[1:] < excess_below[:-1]).all()) and excess_below[-1] < 1e-3 * excess_below[0]
assert abs(A_above[200, 0] / A_above[0, 0]).item() > 1.8e8 and excess_above[-1] > 1e15 * excess_above[0]
assert abs(1.1 ** 200 / 1.9e8 - 1) < 0.02
assert bool((A_above[-1, 1:].abs() < A_above[0, 1:].abs()).all())            # the other 7 directions still shrink
assert bool((2.1 / eigvals[0] * eigvals[1:] < 2).all())

# ---------- follow-up 1: the best constant step ----------
lr_opt = 2.0 / (eigvals[0] + eigvals[-1]).item()
rate_opt = (kappa - 1) / (kappa + 1)
assert abs((1 - lr_opt * eigvals).abs().max().item() - rate_opt) < 1e-12
assert abs(abs(1 - lr_opt * eigvals[0].item()) - rate_opt) < 1e-12 and 1 - lr_opt * eigvals[0].item() < 0
T_opt = math.ceil(math.log(1000.0) / -math.log(rate_opt))
assert T_opt == 424
A_opt = (gd_least_squares(Xc, y, lr_opt, T_opt) - w_hat) @ comps.T
assert A_opt[T_opt, -1].abs() <= 1e-3 * A_opt[0, -1].abs() < A_opt[T_opt - 1, -1].abs()
assert bool((A_opt[T_opt].abs() <= 1e-3 * A_opt[0].abs() + 1e-12).all())

# ---------- follow-up 2: whitening makes the condition number 1 ----------
Z_white = Xc @ comps.T / eigvals.sqrt()
assert torch.allclose(Z_white.T @ Z_white / n, eye, atol=1e-10)
beta_hat = torch.linalg.lstsq(Z_white, y[:, None]).solution[:, 0]
assert torch.allclose(gd_least_squares(Z_white, y, 1.0, 1)[1], beta_hat, atol=1e-10)   # one step is enough
assert torch.allclose(Z_white @ beta_hat, Xc @ w_hat, atol=1e-9)                        # same fitted values

# ---------- follow-up 3: momentum and conjugate gradient ----------
b_vec = Xc.T @ y / n
beta_hb = ((math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)) ** 2
lr_hb = 4.0 / (eigvals[0].sqrt() + eigvals[-1].sqrt()).item() ** 2
w_prev = w_cur = torch.zeros(d, dtype=torch.float64)
hb_steps = None
for t in range(1, 200):
    w_prev, w_cur = w_cur, w_cur - lr_hb * (C @ w_cur - b_vec) + beta_hb * (w_cur - w_prev)
    if (w_cur - w_hat).norm() <= 1e-3 * w_hat.norm():
        hb_steps = t
        break
assert hb_steps == 49
w_cg = torch.zeros(d, dtype=torch.float64)
r = b_vec - C @ w_cg
p = r.clone()
for _ in range(d):
    Cp = C @ p
    alpha = (r @ r) / (p @ Cp)
    w_cg = w_cg + alpha * p
    r_new = r - alpha * Cp
    p = r_new + (r_new @ r_new) / (r @ r) * p
    r = r_new
assert (w_cg - w_hat).norm() < 1e-8 * w_hat.norm()

# ---------- follow-up 4: a start (almost) orthogonal to u_1 stalls at lambda_2 first ----------
v = comps[1] + 1e-10 * comps[0]
v = v / v.norm()
rho_path = []
for _ in range(80):
    w_next = matvec(v)
    rho_path.append((v @ w_next).item())
    v = w_next / w_next.norm()
assert abs(rho_path[10] - eigvals[1].item()) < 1e-9 and abs(rho_path[10] - rho_path[9]) < 1e-12
assert abs(rho_path[-1] - eigvals[0].item()) < 1e-9
assert abs((eigvals[0] / eigvals[1]).item() - 1.9) < 0.05

# ---------- follow-up 5: deflation gives the second eigenpair at rate lambda_3 / lambda_2 ----------
V2, rho2 = power_iteration(lambda v: matvec(v) - lam1 * u1 * (u1 @ v), d, STEPS)
sin2, _ = angle_errors(V2, comps[1])
window2 = (sin2 < 1e-3) & (sin2 > 1e-12)
rate2 = (eigvals[2] / eigvals[1]).item()
assert abs(rate2 - 0.512) < 5e-4
assert abs(fit_slope(ts[window2], sin2[window2]) / math.log(rate2) - 1) < 0.01
assert abs(rho2[-1].item() - eigvals[1].item()) < 1e-10 and sin2[-1].item() < 1e-12

# ---------- follow-up 6: float32 has the same slope but a much higher floor ----------
Xc32, v32 = Xc.float(), V[0].float()
for _ in range(STEPS):
    w32 = Xc32.T @ (Xc32 @ v32) / n
    v32 = w32 / w32.norm()
sin32 = angle_errors(v32.double()[None], u1)[0].item()
assert 1e-8 < sin32 < 1e-5

# ---------- the core facts hold for other seeds as well ----------
for seed in (1, 2, 3):
    Xs, ys = make_data(seed=seed)
    mean_s, ev_s, comps_s = pca_svd(Xs)
    Xcs = Xs - mean_s
    assert torch.allclose(ev_s, torch.linalg.eigvalsh(Xcs.T @ Xcs / n).flip(0), rtol=0, atol=1e-10)
    Vs, _ = power_iteration(lambda v: Xcs.T @ (Xcs @ v) / n, d, STEPS, seed=seed)
    sin_s, tan_s = angle_errors(Vs, comps_s[0])
    rate_s = (ev_s[1] / ev_s[0]).item()
    win = (sin_s < 1e-3) & (sin_s > 1e-12)
    assert abs(fit_slope(ts[win], sin_s[win]) / math.log(rate_s) - 1) < 0.01
    assert bool((tan_s <= tan_s[0] * rate_s ** ts * (1 + 1e-9) + 1e-13).all())
    assert int((sin_s <= 1e-8).nonzero()[0]) <= math.ceil(math.log(tan_s[0].item() / 1e-8) / math.log(1 / rate_s))
    kappa_s = (ev_s[0] / ev_s[-1]).item()
    w_hat_s = torch.linalg.lstsq(Xcs, ys[:, None]).solution[:, 0]
    a8 = ((gd_least_squares(Xcs, ys, 1.0 / ev_s[0].item(), GD_STEPS) - w_hat_s) @ comps_s[-1]).abs()
    assert int((a8 <= 1e-3 * a8[0]).nonzero()[0]) == math.ceil(math.log(1000.0) / -math.log(1 - 1 / kappa_s))

# ---------- the three figures were written ----------
for name in ("part1_spectrum.png", "part2_power_iteration.png", "part3_gd_directions.png"):
    assert os.path.getsize(os.path.join(OUT_DIR, name)) > 10_000

print("all checks passed")
```

</details>

</details>
