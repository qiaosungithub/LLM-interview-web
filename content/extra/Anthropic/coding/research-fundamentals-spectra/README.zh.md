# 谱与收敛速度：PCA、幂迭代和梯度下降

## 题目

这一轮在共享的 notebook 里做，考的是“ML 研究基本功”：用 PyTorch 写几个小函数，画图，并且**在画图之前先说出图会长什么样**。面试官给出下面的数据生成函数，直接用，不要改。

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

三个部分共用的约定：

- $X_c$ 是每一列减去均值之后的 $X$。协方差矩阵一律取 $C = \frac{1}{n} X_c^\top X_c$（除以 $n$，不是 $n-1$）。
- $C$ 的特征值从大到小记作 $\hat\lambda_1 \ge \dots \ge \hat\lambda_d$，对应的单位特征向量记作 $u_1, \dots, u_d$。`LAM` 是生成数据用的总体（population）特征值 $\lambda_i$；$\hat\lambda_i$ 是从 2000 个样本算出来的，两者不相等。
- 全程用 float64，只用 CPU。画图用 matplotlib。

每个部分做三件事：写函数；先说出图的形状，再把图画出来；回答几个要给出数字的问题。

### Part 1 —— 用 SVD 做 PCA

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

1. 实现 `pca_svd`。再用 `torch.linalg.eigh(C)` 核对：特征值相差不超过 $10^{-10}$；特征向量在相差一个正负号的意义下相同。
2. 画一张两格的图，纵轴都取对数。左格：横轴是序号 $i = 1..8$，画样本特征值 $\hat\lambda_i$ 和总体特征值 $\lambda_i$ 两条线。右格：横轴是保留的主成分个数 $k = 0..7$，画相对重构误差

   $$e_k = \frac{\lVert X_c - X_c V_k V_k^\top \rVert_F}{\lVert X_c \rVert_F}$$

   的实测值，和你用特征值算出来的预测值。这里 $V_k$ 的列是前 $k$ 个主成分，$V_0$ 为空（$e_0 = 1$）。
3. 画之前回答：
   - (a) 两格各是什么形状？右格的斜率和左格是什么关系？样本特征值和总体特征值大约差百分之几？
   - (b) $e_k$ 用特征值怎么写？为什么换成任何别的 $k$ 维子空间都不会更好？
   - (c) 如果忘了减均值，直接对 $X$ 做 SVD，第一个右奇异向量大致指向哪里？$s_1^2/n$ 大约是多少？

默认参数（`seed=0`）下的对照数字，用来确认 `pca_svd` 没写错：

```text
eigvals  ≈ [3.8102, 2.0014, 1.0244, 0.4941, 0.2652, 0.1310, 0.0649, 0.0310]
‖mean‖²  ≈ 72.59
```

### Part 2 —— 幂迭代和它的收敛速度

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

记 `Xc = X - mean`。用 `matvec = lambda v: Xc.T @ (Xc @ v) / n`（不显式构造 $C$）跑 60 步。以 Part 1 的 $u_1$ 和 $\hat\lambda_1$ 为标准答案，定义两个误差：

$$\sin\theta_t = \lVert v_t - (u_1^\top v_t)\,u_1 \rVert, \qquad \delta_t = \hat\lambda_1 - \rho_t .$$

1. 实现 `power_iteration`。
2. 画一张图：横轴 $t = 0..60$，纵轴取对数，画 $\sin\theta_t$ 和 $\delta_t$ 两条曲线。
3. 画之前回答：
   - (a) 两条曲线下降段的斜率各是多少？用特征值写出来，并给出数值。这个结论需要哪些条件？
   - (b) 两条曲线降到多低就不再降了？哪一条先到底？
   - (c) 不看实测结果，只用 $\tan\theta_0$ 和特征值，给出“$\sin\theta_t \le 10^{-8}$ 最多需要多少步”的上界，再和实测的步数比较。

### Part 3 —— 最小二乘上的梯度下降

```py
def gd_least_squares(Xc, y, lr, steps):
    """Gradient descent on L(w) = ||Xc @ w - y||^2 / (2 n), starting from w = 0, with the gradient
    taken from autograd (loss.backward()).
    Return W of shape (steps + 1, d): the iterates w_0 .. w_steps."""
```

记 $\hat w$ 为最小二乘解（用 `torch.linalg.lstsq` 求）。把误差 $w_t - \hat w$ 投影到 Part 1 的特征方向上：

$$a_i(t) = u_i^\top (w_t - \hat w), \qquad \ell_i(t) = \tfrac{1}{2}\,\hat\lambda_i\, a_i(t)^2 .$$

1. 实现 `gd_least_squares`，取学习率 $\eta = 1/\hat\lambda_1$ 跑 1200 步。
2. 画一张两格的图，纵轴都取对数，范围 $[10^{-20}, 10]$。左格横轴 $t = 0..60$，右格横轴 $t = 0..1200$。每格画 9 条线：8 条 $\ell_i(t)$，加一条总的超额损失 $L(w_t) - L(\hat w)$。
3. 画之前回答：
   - (a) 每条 $\ell_i$ 是什么形状，斜率是多少？总的那一条呢？为什么画超额损失，而不是 $L(w_t)$ 本身？
   - (b) 最慢的分量 $|a_8(t)|$ 缩小到初值的 $1/1000$ 需要多少步？写出公式，给出数值，再和实测比较。
   - (c) 学习率最大能取多少？分别用 $\eta = 1.9/\hat\lambda_1$ 和 $\eta = 2.1/\hat\lambda_1$ 各跑 200 步，$a_1$ 会怎样？

## 参考解答

<details>
<summary>展开参考解答</summary>

先把题目给的代码原样放进来（外加画图要用的几个 import）。图一律存到临时目录，不调用 `plt.show()`。

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

**思路。** 对中心化后的数据做 SVD：$X_c = U S V^\top$，于是

$$C = \tfrac{1}{n} X_c^\top X_c = V\,\frac{S^2}{n}\,V^\top .$$

所以 $C$ 的特征值就是 $s_i^2/n$，特征向量就是 $V$ 的列，也就是 `torch.linalg.svd` 返回的 `Vh` 的行。有三处容易错：

- `svd` 的奇异值从大到小排，`eigh` 的特征值从小到大排，核对时要把其中一个翻过来。
- 特征向量的正负号是任意的，两个算法给出的符号可以不同。要么比较 $|u_i^\top u_i'|$，要么像题目这样规定符号。
- `torch.cov` 默认除以 $n-1$。要和题目的约定一致，得写 `correction=0`。

为什么不直接 `eigh(C)`：构造 $C$ 会把条件数平方，$\kappa(C) = \kappa(X_c)^2$。`eigh` 算出的最小特征值，相对误差大约是机器精度乘 $\kappa(C)$；走 SVD 则是机器精度乘 $\sqrt{\kappa(C)}$。本题 $\kappa(C) \approx 123$，两种做法都准到 $10^{-13}$ 以内，看不出差别；数据很病态时差别很大（验证代码里有一个 $\kappa(C) = 10^{14}$ 的例子）。两种做法的代价都是 $O(nd^2)$。

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

**(a) 图的形状。** 左格：总体特征值每个是前一个的一半，在对数纵轴上是一条直线，每走一格下降 $\ln 2$。样本特征值的点贴着这条线，上下偏差百分之几。对高斯数据，特征值互不相同、$n \gg d$ 时，$\hat\lambda_i/\lambda_i - 1$ 近似服从 $\mathcal N(0,\, 2/n)$，标准差 $\sqrt{2/2000} \approx 3.2\%$。`seed=0` 时偏差最大的一个是 $6.1\%$。

右格：也近似是直线，但斜率只有左格的一半。原因见 (b)：$e_k^2$ 正比于剩下的特征值之和，等比数列的尾和约等于首项的 2 倍，所以 $e_k \approx 2^{-k/2}$，每多保留一个主成分误差除以 $\sqrt 2$。因为尾和在 $d = 8$ 处截断，最右边几个点比直线略低：对总体特征值，精确值是 $e_k^2 = (2^{-k} - 2^{-8})/(1 - 2^{-8})$。

**(b) Eckart–Young–Mirsky 定理。** 设 $A$ 的奇异值为 $s_1 \ge s_2 \ge \dots$，$A_k$ 是只保留前 $k$ 个奇异值的截断 SVD。那么对任何秩不超过 $k$ 的矩阵 $B$，

$$\lVert A - B \rVert_F \ge \lVert A - A_k \rVert_F = \sqrt{\textstyle\sum_{i>k} s_i^2}, \qquad \lVert A - B \rVert_2 \ge \lVert A - A_k \rVert_2 = s_{k+1}.$$

$s_k > s_{k+1}$ 时，Frobenius 范数下的最优解唯一；$s_k = s_{k+1} > 0$ 时不唯一。取 $A = X_c$：$X_c V_k V_k^\top$ 正是 $A_k$，再用 $s_i^2 = n\hat\lambda_i$，得到

$$e_k = \sqrt{\frac{\sum_{i>k} \hat\lambda_i}{\sum_{i} \hat\lambda_i}} .$$

“别的子空间不会更好”是定理的直接推论：把每一行投影到任何一个 $k$ 维子空间，得到的矩阵秩都不超过 $k$。`seed=0` 时 $e_k$ 依次是 $1,\ 0.716,\ 0.507,\ 0.355,\ 0.251,\ 0.170,\ 0.111,\ 0.063$。

**(c) 忘了减均值。** 这时 SVD 分解的是 $\frac1n X^\top X = C + \bar x \bar x^\top$。均值的每个坐标都接近 3，所以 $\lVert \bar x \rVert^2 \approx 8 \times 3^2 = 72$（`seed=0` 时是 $72.6$），比 $\hat\lambda_1 \approx 3.8$ 大得多。因此第一个右奇异向量几乎就是均值方向 $\bar x / \lVert \bar x \rVert$（夹角余弦大于 $0.999$），而 $s_1^2/n \approx 74$。更准确地说，

$$\lVert \bar x \rVert^2 \;\le\; s_1^2/n \;\le\; \lVert \bar x \rVert^2 + \hat\lambda_1 .$$

也就是说，“第一主成分”解释的是均值，不是方差。其余 7 个值也不等于任何一个 $\hat\lambda_i$，只是被它们两两夹住（秩一修正的交错性质）：不减均值时的第 $i+1$ 个值不超过 $\hat\lambda_i$，第 $i$ 个值不小于 $\hat\lambda_i$。

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

**思路。** 把起点按特征向量展开，$v_0 = \sum_i c_i u_i$。每一步乘一次 $C$ 再归一化，所以

$$v_t \;\propto\; C^t v_0 = \sum_i c_i\, \hat\lambda_i^{\,t}\, u_i .$$

$u_1$ 以外的每个分量，相对 $u_1$ 的分量每步缩小 $\hat\lambda_i/\hat\lambda_1$ 倍，最慢的是 $i = 2$。由此可以写出精确的式子：

$$\tan\theta_t = \frac{\sqrt{\sum_{i \ge 2} c_i^2 \hat\lambda_i^{2t}}}{|c_1|\,\hat\lambda_1^{\,t}}, \qquad \delta_t = \hat\lambda_1 - \rho_t = \frac{\sum_{i \ge 2} c_i^2 \hat\lambda_i^{2t}\,(\hat\lambda_1 - \hat\lambda_i)}{\sum_{i} c_i^2 \hat\lambda_i^{2t}} .$$

实现时每步只需要一次 `matvec`，一次 `matvec` 的代价是 $O(nd)$。

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

**(a) 斜率。** 条件：$C$ 对称半正定；最大特征值严格大于第二个，$\hat\lambda_1 > \hat\lambda_2$；起点和 $u_1$ 不正交，$c_1 \ne 0$（随机起点以概率 1 满足）。在这些条件下，对每个 $t$ 都有

$$\tan\theta_t \le \left(\frac{\hat\lambda_2}{\hat\lambda_1}\right)^{t} \tan\theta_0, \qquad (\hat\lambda_1 - \hat\lambda_2)\sin^2\theta_t \;\le\; \delta_t \;\le\; (\hat\lambda_1 - \hat\lambda_d)\sin^2\theta_t .$$

如果起点在 $\hat\lambda_2$ 的特征空间里也有分量（随机起点同样以概率 1 满足），相邻两步的比值就收敛到这个界：$\sin\theta_t$ 每步乘 $\hat\lambda_2/\hat\lambda_1$，$\delta_t$ 每步乘 $(\hat\lambda_2/\hat\lambda_1)^2$。所以在对数纵轴上是两条直线：

- $\sin\theta_t$ 的斜率是 $\ln(\hat\lambda_2/\hat\lambda_1) = \ln 0.5253 = -0.644$ 每步，大约每 3.6 步下降一个数量级。
- $\delta_t$ 的斜率是它的两倍，$-1.288$ 每步，大约每 1.8 步下降一个数量级。特征值的误差是角度误差的平方量级，这是对称矩阵 Rayleigh 商的一般性质。

注意比值要用**样本**特征值算。迭代的矩阵是 $C$，它的比值是 $0.5253$，不是总体的 $0.5$。两者差 $5\%$，听起来不多，但走 50 步之后预测值会差 $(0.5253/0.5)^{50} \approx 12$ 倍，在图上超过一个数量级。

**(b) 到底的位置。** 两条线都降到舍入误差为止。$\sin\theta_t$ 停在 $10^{-15}$ 附近，$\delta_t$ 停在几个 $10^{-15}$（$\hat\lambda_1 \approx 3.8$ 的最后一位），而且可能是 0 或很小的负数，画对数图之前要取绝对值并截断。$\delta_t$ 的斜率大一倍，所以先到底：大约 $t = 25$；$\sin\theta_t$ 要到 $t = 50$ 以后。

这里有一个数值上的坑：如果用 $\sqrt{1 - \cos^2\theta_t}$ 算正弦，$\cos\theta_t$ 和 1 的差小于 $10^{-16}$ 时就被舍掉了，算出的结果在 $10^{-8}$ 附近失真，之后直接变成 0，曲线看起来像是在 $10^{-8}$ 处突然中断。题目里的定义（减去在 $u_1$ 上的投影再取范数）没有这个问题。

**(c) 步数上界。** 由 $\sin\theta_t \le \tan\theta_t \le (\hat\lambda_2/\hat\lambda_1)^t \tan\theta_0$，只要

$$t \;\ge\; \frac{\ln(\tan\theta_0 / \varepsilon)}{\ln(\hat\lambda_1/\hat\lambda_2)}$$

就有 $\sin\theta_t \le \varepsilon$。`seed=0` 时 $\tan\theta_0 = 2.35$，取 $\varepsilon = 10^{-8}$ 得到上界 30 步；实测第 28 步达到。实测比上界少，是因为上界把 $u_1$ 以外的分量全都算在最慢的 $u_2$ 上，实际上有一部分落在衰减更快的方向里。

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

**思路。** $\hat w$ 满足正规方程 $X_c^\top (X_c \hat w - y) = 0$，所以梯度可以写成 $\nabla L(w) = \frac1n X_c^\top (X_c w - y) = C\,(w - \hat w)$。Hessian 正是 Part 1 的 $C$。梯度下降一步是

$$w_{t+1} - \hat w = (I - \eta C)(w_t - \hat w),$$

在 $C$ 的特征方向上，各个分量互不干扰，各自按等比数列变化：

$$a_i(t) = (1 - \eta \hat\lambda_i)^t\, a_i(0), \qquad L(w_t) - L(\hat w) = \tfrac12 (w_t - \hat w)^\top C\,(w_t - \hat w) = \sum_i \ell_i(t).$$

每步的代价是 $O(nd)$。

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

**(a) 图的形状。** 取 $\eta = 1/\hat\lambda_1$ 时，每步的因子是 $1 - \hat\lambda_i/\hat\lambda_1$，所以

$$\ell_i(t) = \ell_i(0)\left(1 - \frac{\hat\lambda_i}{\hat\lambda_1}\right)^{2t} .$$

在对数纵轴上，每条 $\ell_i$ 都是直线，斜率是 $2\ln(1 - \hat\lambda_i/\hat\lambda_1)$，特征值越小线越平。8 条线像一把打开的扇子：

- 方向 1 的因子是 0，走一步就归零，直接掉出图外。
- 方向 2 的斜率是 $2\ln(1 - 0.5253) = -1.49$ 每步，十几步就降到 $10^{-10}$ 以下。
- 方向 8 的斜率是 $2\ln(1 - 1/122.75) = -0.0164$ 每步，1200 步只下降 8.5 个数量级。

总的超额损失是这 8 条线的和。在对数图上，一组直线的和大致贴着其中最高的那条走，所以它是一条凸的曲线：开头陡、后面越来越平（`seed=0` 时初始误差主要在方向 3 上），每当一个较快的方向衰减完就变平一点，最后贴着方向 8 那条线走（别的方向都已经衰减完了）。左格看得清前几个方向的扇形，右格看得清最后由方向 7、8 决定的长尾。

为什么不画 $L(w_t)$ 本身：$y$ 里有噪声，$L(w_t)$ 的极限是 $L(\hat w) \approx \sigma^2/2 = 0.005$（`seed=0` 时是 $0.0049$），不是 0。直接画的话，曲线几十步之后就在 $0.005$ 处变平，后面的衰减全被盖住。

**(b) 步数和条件数。** 最慢的是特征值最小的方向，每步的因子是 $1 - 1/\hat\kappa$，其中 $\hat\kappa = \hat\lambda_1/\hat\lambda_8$ 是 $C$ 的条件数。要 $|a_8(t)| \le \varepsilon\,|a_8(0)|$，需要

$$t \;\ge\; \frac{\ln(1/\varepsilon)}{-\ln(1 - 1/\hat\kappa)} \;\approx\; \hat\kappa \,\ln(1/\varepsilon).$$

`seed=0` 时 $\hat\kappa = 122.75$，取 $\varepsilon = 10^{-3}$，向上取整得到 845 步，实测也正好是第 845 步。近似式 $\hat\kappa \ln 1000 = 848$ 只差 3 步。结论是：步数和条件数成正比，精度每提高 10 倍要多走 $\hat\kappa \ln 10 \approx 283$ 步。

**(c) 稳定阈值。** $C$ 正定时（这里 $\hat\lambda_8 > 0$），梯度下降从任何起点都收敛，当且仅当所有因子的绝对值小于 1，即

$$0 < \eta < \frac{2}{\hat\lambda_1} .$$

- $\eta = 1.9/\hat\lambda_1$：$a_1$ 每步乘 $1 - 1.9 = -0.9$，正负交替，幅度缩小，收敛。
- $\eta = 2.1/\hat\lambda_1$：$a_1$ 每步乘 $-1.1$，正负交替，幅度放大。200 步之后放大了 $1.1^{200} \approx 1.9 \times 10^{8}$ 倍，超额损失比初值大 $10^{15}$ 倍以上。

其余 7 个方向在 $\eta = 2.1/\hat\lambda_1$ 时仍然收敛，因为 $\eta \hat\lambda_i \le 2.1 \times 0.5253 < 2$。学习率略微过大时，发散只发生在最陡的那一个方向上，表现为沿这个方向来回振荡并且越振越大。

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

### 追问

**1. 固定步长的话，$1/\hat\lambda_1$ 是最好的吗？** 不是。最好的固定步长让最大和最小两个特征值的因子绝对值相等：$\eta = 2/(\hat\lambda_1 + \hat\lambda_d)$，这时所有方向的因子绝对值都不超过 $(\hat\kappa - 1)/(\hat\kappa + 1)$。同样缩小 1000 倍，需要 424 步，是 845 步的一半。代价是方向 1 不再一步归零，而是和方向 8 一样慢，并且每步变号。

**2. 能不能让所有方向一样快？** 用 Part 1 的 PCA 把输入白化（whitening）：$Z = X_c V \hat\Lambda^{-1/2}$（$V$ 的列是协方差矩阵的特征向量，也就是 SVD 的右奇异向量），它的协方差是单位阵，条件数是 1。这时取 $\eta = 1$，梯度下降一步就到最小二乘解。输入标准化、预条件（preconditioning）起的都是这个作用：把条件数降下来。

**3. 加动量能快多少？** 带动量的梯度下降（heavy ball）取最优参数时，每步的因子大约是 $(\sqrt{\hat\kappa} - 1)/(\sqrt{\hat\kappa} + 1)$，步数从正比于 $\hat\kappa$ 变成正比于 $\sqrt{\hat\kappa}$。本题里误差范数缩小 1000 倍只要 49 步。共轭梯度法（conjugate gradient）在精确算术下最多 $d = 8$ 步就得到精确解。

**4. 幂迭代的起点恰好和 $u_1$ 正交会怎样？** 精确算术下它收敛到 $u_2$。浮点运算中舍入误差会带进一点 $u_1$ 的分量，这个分量每步相对放大 $\hat\lambda_1/\hat\lambda_2 \approx 1.9$ 倍，几十步之后才占上风。在这之前 Rayleigh 商停在 $\hat\lambda_2$ 上不动。所以“相邻两步的 $\rho_t$ 几乎不变就停”这个停止条件可能停在错误的特征值上。验证代码里用起点 $u_2 + 10^{-10} u_1$ 演示了这一点。

**5. 怎么求第二个特征向量？** 做收缩（deflation）：把 `matvec` 换成 $v \mapsto Cv - \hat\lambda_1 u_1 (u_1^\top v)$，新矩阵的最大特征值是 $\hat\lambda_2$。再跑幂迭代，收敛速度由 $\hat\lambda_3/\hat\lambda_2 \approx 0.512$ 决定。一次要求前几个特征向量时，更常用的是子空间迭代（每步做一次 QR），或者干脆用 Lanczos。

**6. 换成 float32 图会怎样变？** 斜率不变，但到底的位置高得多：$\sin\theta_t$ 停在 $10^{-7}$ 附近，而不是 $10^{-15}$。所以这类要看十几个数量级的实验都用 float64。

<details>
<summary>验证代码（可运行）</summary>

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
