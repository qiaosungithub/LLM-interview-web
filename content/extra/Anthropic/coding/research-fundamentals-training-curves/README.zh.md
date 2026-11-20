# 训练曲线：最小训练循环、三处 bug 和学习率阈值

## 题目

这一轮在共享的 notebook 里做，考的是“ML 研究基本功”：用 PyTorch 写一个很小的训练循环，看懂它画出来的曲线，并且**在画图之前先说出图会长什么样**。面试官给出数据和模型，直接用，不要改。

```py
import torch
import torch.nn as nn


def make_data(n, seed):
    """x ~ N(0, I_8) in float32; y = relu(x0 - x1) - relu(x2) + 0.5 * x3 + 0.1 * noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 8, generator=g)
    clean = torch.relu(X[:, 0] - X[:, 1]) - torch.relu(X[:, 2]) + 0.5 * X[:, 3]
    y = clean + 0.1 * torch.randn(n, generator=g)
    return X, y


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


X_train, y_train = make_data(1000, seed=0)
X_val, y_val = make_data(1000, seed=1000)
```

损失一律是均方误差（mean squared error），对样本取平均，没有 $\frac12$：$\frac1n\sum_i (f(x_i) - y_i)^2$。只用 CPU。画图用 matplotlib，纵轴都取对数。

### Part 1 —— 最小训练循环和它的曲线

```py
def evaluate(model, X, y):
    """Mean squared error of model on (X, y), computed in eval mode without building a graph.
    Must leave model in the same mode (train / eval) it was in before the call."""


def train(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    """Minibatch SGD (torch.optim.SGD, no momentum, no weight decay) on the mean squared error.

    - call torch.manual_seed(seed) once at the start (it drives the dropout masks);
    - create g = torch.Generator().manual_seed(seed) once; every epoch calls model.train(), draws
      perm = torch.randperm(n, generator=g) and uses the batches perm[0:50], perm[50:100], ...

    Return a dict of lists:
      "train":       length epochs,     the minibatch losses of each epoch, averaged with weight = batch size
      "val":         length epochs + 1, evaluate(model, X_val, y_val) before training and after every epoch
      "train_eval":  length epochs + 1, evaluate(model, X_train, y_train) at the same moments
    """
```

1. 实现 `evaluate` 和 `train`，用 `train(make_model(0), X_train, y_train, X_val, y_val)` 训练一次。
2. 画一张图，纵轴取对数。画两条曲线：`history["val"]` 画在横坐标 $0, 1, \dots, 40$；`history["train"]` 是整个 epoch 的平均，画在横坐标 $0.5, 1.5, \dots, 39.5$。再画三条水平参考线，取总体（population）的值：$\operatorname{Var}(y)$、最好的线性模型（带截距）的均方误差、噪声方差 $0.1^2 = 0.01$。
3. 画之前回答：
   - (a) 三条参考线各是多少？从 `make_data` 的公式推出精确值。
   - (b) 验证曲线是什么形状？大概从哪里开始，第几个 epoch 穿过线性模型那条线，最后停在哪里？
   - (c) 训练曲线和验证曲线谁高？画出来以后，用几行代码量出原因，并给出一个能精确解释两者之差的式子。

### Part 2 —— 同事的训练脚本

同事用同样的数据和模型写了下面这个函数。他用 `train_colleague(make_model(0), X_train, y_train, X_val, y_val)` 跑了一次，说曲线不对，最后还出了 `nan`。

```py
def train_colleague(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mse = lambda pred, target: ((pred - target) ** 2).mean()
    n = X_train.shape[0]
    history = {"train": [], "val": []}
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = mse(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        with torch.no_grad():
            history["val"].append(mse(model(X_val), y_val).item())
    return history
```

这段代码有 3 处错误，叠加在一起。请一处一处地找：每一步先画出当前版本的训练、验证曲线（坐标和 Part 1 一样），说出你从图上看到了什么、猜是哪里错了，再用几行测量代码确认，然后修好它，再看下一张图。

修好全部错误之后，`train_colleague` 返回的 `history["train"]` 应当和 Part 1 `train` 返回的 `history["train"]` 逐项相等，`history["val"]` 应当和 Part 1 的 `history["val"][1:]` 逐项相等（同样的 seed，结果完全一样，不只是接近）。

### Part 3 —— 梯度核对和学习率阈值

换成线性模型。用 Part 1 的训练集，转成 float64：`X = X_train.double()`，`y = y_train.double()`。参数 $\theta = (w_0, \dots, w_7, b)$ 共 9 个数，损失

$$L(\theta) = \frac1n \sum_i (w^\top x_i + b - y_i)^2 .$$

```py
def linear_loss(theta, X, y):                  # given
    return ((X @ theta[:-1] + theta[-1] - y) ** 2).mean()


def linear_grad(theta, X, y):
    """Gradient of linear_loss with respect to theta, derived by hand (no autograd). Shape (9,)."""


def fd_grad(f, theta, eps):
    """Central differences: entry i is (f(theta + eps * e_i) - f(theta - eps * e_i)) / (2 * eps)."""


def gd_linear(X, y, lr, steps):
    """Full-batch gradient descent with torch.optim.SGD on nn.Linear(8, 1) in float64, weight and bias
    initialised to zero, loss = mean squared error of the whole set.
    Return a tensor of shape (steps + 1, 9): theta = (weight, bias) after 0, 1, ..., steps updates."""
```

1. 取 `theta0 = torch.randn(9, generator=torch.Generator().manual_seed(0), dtype=torch.float64)`。比较三种梯度：`linear_grad`、`fd_grad(eps=1e-6)`、autograd，相对误差定义为 $\lVert a - b \rVert / \lVert b \rVert$，$b$ 取 `linear_grad`。再把 `X`、`y`、`theta0` 都换成 float32，分别用 `eps=1e-6` 和 `eps=1e-2` 做有限差分。先预测这几个误差的数量级，再算出来解释。
2. 手推 Hessian $H$，用 `torch.autograd.functional.hessian` 核对。求出最大特征值 $h_1$、条件数 $\kappa(H)$，以及你认为的学习率稳定阈值 $\eta^*$。
3. 记 $\hat\theta$ 为最小二乘解。取 $\eta = c\,\eta^*$，$c \in \{0.9,\ 1.0,\ 1.1\}$，各跑 100 步。画一张图：横轴 $t = 0..100$，纵轴取对数，三条曲线都是 $L(\theta_t) - L(\hat\theta)$（低于 $10^{-15}$ 的部分是舍入误差，不用管）。
4. 画之前回答：
   - (a) 三条曲线各是什么形状？后期每步乘多少？$c = 1.0$ 那条最后停在哪个值（给出公式和数值）？
   - (b) 把阈值写成定理：在什么条件下，梯度下降从任何起点都收敛？
   - (c) 如果损失改成 `reduction="sum"`（对样本求和而不是平均），阈值变成多少？

默认数据下的对照数字：

```text
L(theta_hat) ≈ 0.2893
h_1          ≈ 2.453
```

## 参考解答

<details>
<summary>展开参考解答</summary>

先把题目给的代码原样放进来（外加画图用的 import）。图一律存到临时目录，不调用 `plt.show()`。

```python
import copy
import math
import os
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

OUT_DIR = tempfile.mkdtemp(prefix="training_curves_")       # figures are written here
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#73726c"


def make_data(n, seed):
    """x ~ N(0, I_8) in float32; y = relu(x0 - x1) - relu(x2) + 0.5 * x3 + 0.1 * noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 8, generator=g)
    clean = torch.relu(X[:, 0] - X[:, 1]) - torch.relu(X[:, 2]) + 0.5 * X[:, 3]
    y = clean + 0.1 * torch.randn(n, generator=g)
    return X, y


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))


X_train, y_train = make_data(1000, seed=0)
X_val, y_val = make_data(1000, seed=1000)
```

### Part 1

**思路。** 训练循环本身没有难点，容易错的是几个细节：

- 模型输出的形状是 `(B, 1)`，标签是 `(B,)`，相减之前要 `squeeze(-1)`，否则会广播成 `(B, B)`。
- 每一步先 `zero_grad()`。PyTorch 的 `backward()` 是把梯度**加到** `.grad` 上，不是覆盖。
- `evaluate` 要切到 `eval()` 模式（关掉 dropout），用 `torch.no_grad()`（不建计算图），算完再把模式恢复原样。
- 训练损失按 batch 大小加权平均。这里 1000 正好被 50 整除，加不加权结果一样；最后一个 batch 不满时才有区别。

每个 epoch 的代价是一次前向加一次反向，$O(n \cdot 8 \cdot 64)$。

```python
def evaluate(model, X, y):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = ((model(X).squeeze(-1) - y) ** 2).mean().item()
    model.train(was_training)
    return loss


def train(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = X_train.shape[0]
    history = {"train": [], "val": [evaluate(model, X_val, y_val)], "train_eval": [evaluate(model, X_train, y_train)]}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            opt.zero_grad()
            loss = ((model(xb).squeeze(-1) - yb) ** 2).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        history["val"].append(evaluate(model, X_val, y_val))
        history["train_eval"].append(evaluate(model, X_train, y_train))
    return history


model = make_model(0)
history = train(model, X_train, y_train, X_val, y_val)
```

**(a) 三条参考线。** 记 $u = x_0 - x_1 \sim \mathcal N(0, 2)$。对 $Z \sim \mathcal N(0, s^2)$，有 $\mathbb E[\mathrm{relu}(Z)] = s/\sqrt{2\pi}$，$\mathbb E[\mathrm{relu}(Z)^2] = s^2/2$。$y$ 的四项分别只用到 $(x_0, x_1)$、$x_2$、$x_3$ 和噪声，彼此独立，方差直接相加：

$$\operatorname{Var}(y) = \underbrace{\left(1 - \tfrac{1}{\pi}\right)}_{\mathrm{relu}(u)} + \underbrace{\left(\tfrac12 - \tfrac{1}{2\pi}\right)}_{\mathrm{relu}(x_2)} + \underbrace{\tfrac14}_{0.5 x_3} + \underbrace{0.01}_{\text{noise}} = 1.76 - \frac{3}{2\pi} \approx 1.2825 .$$

最好的线性模型就是把 $y$ 投影到 $\{1, x_0, \dots, x_7\}$ 张成的空间上。各项用到的输入互不相交且独立，所以可以一项一项地投影。$\mathrm{relu}(u)$ 对 $u$ 的回归系数是 $\mathbb E[u\,\mathrm{relu}(u)]/\mathbb E[u^2] = (s^2/2)/s^2 = \frac12$，$\mathrm{relu}(x_2)$ 对 $x_2$ 的系数同样是 $\frac12$。所以最好的线性模型是

$$\hat y = \tfrac12 (x_0 - x_1) - \tfrac12 x_2 + \tfrac12 x_3 + \frac{\sqrt2 - 1}{\sqrt{2\pi}}, \qquad \text{截距} \approx 0.165 .$$

线性部分的方差是 $\frac14 \cdot 2 + \frac14 + \frac14 = 1$，剩下的就是线性模型的均方误差：$0.76 - \frac{3}{2\pi} \approx 0.2825$。噪声下限是 $0.01$，任何模型的期望误差都不会低于它。

**(b) 验证曲线的形状。** 初始化时网络的输出很小，验证误差大约是 $\mathbb E[y^2]$ 再加上随机输出的方差，在 $\operatorname{Var}(y)$ 附近或略高（`seed=0` 时是 $1.39$）。第一个 epoch 就降到 $0.21$，已经低于线性模型的 $0.28$：relu 网络很快学到了线性部分，又学到一部分折线。之后下降越来越慢，最后停在 $0.020$ 左右，大约是噪声下限的 2 倍。验证曲线不会低于 $0.01$。

**(c) 训练曲线为什么一直更高。** 画出来会看到，40 个 epoch 里训练曲线**每一个点**都在验证曲线上面，最后是 $0.053$ 对 $0.020$。这不是欠拟合，也不是数据有问题。原因有两个：

- 训练损失是一个 epoch 里各个 batch 的平均，而模型在这个 epoch 里一直在变好，所以它比 epoch 结束时的损失偏高。这只在前几个 epoch 明显。
- 训练损失是在 `train()` 模式下算的，dropout 是开着的；验证损失在 `eval()` 模式下算。后期的差距几乎全来自这一点。

测量办法：在 `eval()` 模式下算训练集上的损失（就是 `history["train_eval"]`），最后是 $0.017$，和验证集的 $0.020$ 差不多。所以模型在训练集和验证集上的表现几乎一样，差距来自 dropout 本身。

精确的式子：dropout 只在最后一个线性层前面。记隐藏层激活为 $h_j$，最后一层的权重为 $w_j$，掩码 $m_j \sim \mathrm{Bernoulli}(1-p)$ 相互独立，那么

$$f_{\text{drop}}(x) = b + \sum_j w_j h_j \frac{m_j}{1-p}, \qquad \mathbb E\left[\frac{m_j}{1-p}\right] = 1, \quad \operatorname{Var}\left[\frac{m_j}{1-p}\right] = \frac{p}{1-p} .$$

所以 $\mathbb E[f_{\text{drop}}] = f_{\text{eval}}$，再用“均值平方加方差”：

$$\mathbb E_m\left[(f_{\text{drop}}(x) - y)^2\right] = (f_{\text{eval}}(x) - y)^2 + \frac{p}{1-p} \sum_j (w_j h_j)^2 .$$

这是恒等式，对每一个样本都精确成立，不需要任何近似。右边第二项在训练集上的平均，`seed=0` 时是 $0.036$，正好等于训练曲线最后一点和 `train_eval` 之差 $0.053 - 0.017$。

```python
def dropout_penalty(model, X, p=0.2):
    """p / (1 - p) * mean over samples of sum_j (w_j h_j)^2: the extra loss that dropout adds in expectation."""
    with torch.no_grad():
        h = torch.relu(model[0](X))
        return (p / (1 - p) * ((h * model[3].weight) ** 2).sum(dim=1).mean()).item()


VAR_Y = 1.76 - 3 / (2 * math.pi)                 # 1.2825
LINEAR_MSE = 0.76 - 3 / (2 * math.pi)            # 0.2825
NOISE = 0.01
MEAN_Y = (math.sqrt(2) - 1) / math.sqrt(2 * math.pi)

penalty_train = dropout_penalty(model, X_train)
gap_end = history["train"][-1] - history["train_eval"][-1]


def plot_curves(ax, train_curve, val_curve, val_x, title=None, train_eval=None):
    E = len(train_curve)
    ax.semilogy([e + 0.5 for e in range(E)], train_curve, color=BLUE, lw=2, label="train (epoch average, train mode)")
    ax.semilogy(val_x, val_curve, color=ORANGE, lw=2, label="val")
    if train_eval is not None:
        ax.semilogy(range(len(train_eval)), train_eval, ":", color=AQUA, lw=2, label="train set, eval mode")
    for value, name in ((VAR_Y, "Var(y)"), (LINEAR_MSE, "best linear model"), (NOISE, "noise floor")):
        ax.axhline(value, color=GRAY, ls="--", lw=1)
        ax.text(E, value * 1.1, name, ha="right", va="bottom", fontsize=8, color=GRAY)
    ax.set_ylim(5e-3, 10)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    if title:
        ax.set_title(title, fontsize=10)


fig, ax = plt.subplots(figsize=(6.4, 4.0))
plot_curves(ax, history["train"], history["val"], range(len(history["val"])), train_eval=history["train_eval"])
ax.legend(frameon=False, fontsize=8, loc="upper center")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "part1_curves.png"), dpi=120)
plt.close(fig)
```

### Part 2

三处错误按从图上看到的顺序排列。每一处都写出：在哪里，出了什么问题，在什么输入上暴露，怎么修。

**第一张图（原样运行）：损失一直不降，几个 epoch 后越涨越快，最后变成 `nan`。** `seed=0` 时训练损失从没低于 1.3（和 $\operatorname{Var}(y)$ 差不多高），后几个 epoch 涨到 3.2，第 10 个 epoch 出现 `nan`。损失先不降、后爆炸，第一个怀疑对象是梯度本身。

**错误 A：循环里没有 `opt.zero_grad()`。** `loss.backward()` 把这一步的梯度加到 `.grad` 上，于是 `.grad` 是从第一步到现在所有梯度的和，每一步都沿着这个越来越大的和走。测量：走两步之后，`p.grad` 等于两个 batch 各自梯度之和，而不是第二个 batch 的梯度（验证代码里有这个对比）。数学上，它和 `torch.optim.SGD(momentum=1.0)` 完全等价，就是没有阻尼的重球法。修复：在 `backward()` 之前加 `opt.zero_grad()`。

**第二张图（修了 A）：两条曲线都平躺在 $\operatorname{Var}(y)$ 那条参考线上。** 训练损失停在 $1.32$，验证在 $1.26$，40 个 epoch 几乎不动。再看模型在训练集上的输出：标准差只有 $0.03$，而 $y$ 的标准差是 $1.14$，模型几乎输出一个常数。网络明明能学到 $0.02$，却只学到了均值，说明它优化的根本不是我们以为的那个损失。

**错误 B：`mse` 里 `(B, 1)` 减 `(B,)` 广播成了 `(B, B)`。** 测量：打印 `(model(xb) - yb).shape`，得到 `(50, 50)`。这个“损失”是所有预测和所有标签两两之差的平方平均，可以拆成

$$\frac{1}{B^2}\sum_{i,j} (f_i - y_j)^2 = \frac1B \sum_i (f_i - \bar y)^2 + \frac1B \sum_j (y_j - \bar y)^2 .$$

第二项和模型无关，第一项在 $f_i \equiv \bar y$ 时最小。所以模型学会输出常数 $\bar y$，损失停在 $y$ 的方差上，和图上一模一样。验证集上也是同一个函数，所以验证曲线也停在 $\operatorname{Var}(y_{\text{val}})$。修复：`pred.squeeze(-1) - target`。

**第三张图（修了 A 和 B）：训练曲线和 Part 1 差不多，但验证曲线也在 $0.06$ 左右，而且一跳一跳的。** Part 1 里验证曲线最后是 $0.020$；现在它和训练曲线一样高（`seed=0` 时最后一点是 $0.062$）。Part 1 (c) 已经量过，训练曲线比 eval 模式下的损失高出的部分正是 dropout 带来的 $0.036$。验证曲线现在也高出差不多这么多，自然想到验证时 dropout 也开着。

**错误 C：验证时只用了 `torch.no_grad()`，没有切到 `model.eval()`。** `no_grad` 只是不建计算图，不会关掉 dropout。测量：此时 `model.training` 是 `True`；同一个模型在验证集上连算两次，结果不一样；用 `evaluate`（eval 模式）算，得到 $0.023$，和错误的值相差 $0.039$，接近验证集上的 dropout 项 $0.038$。修复：验证前 `model.eval()`，验证后 `model.train()` 切回来。

**为什么修完之后和 Part 1 逐位相同。** 两段代码消耗随机数的顺序完全一样：开头 `torch.manual_seed(seed)`，打乱顺序用独立的生成器 `g`，dropout 掩码只在训练的前向里抽取，eval 模式下不抽。错误 C 还有一个副作用：验证时抽了掩码，打乱了后面训练用的随机数，所以只修 A、B 时，连训练出的模型都和 Part 1 不同。

```python
def train_colleague_fixed(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mse = lambda pred, target: ((pred.squeeze(-1) - target) ** 2).mean()   # B: (n, 1) - (n,) broadcasts to (n, n)
    n = X_train.shape[0]
    history = {"train": [], "val": []}
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()                                               # A: backward() adds to .grad
            loss = mse(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        model.eval()                                                      # C: no_grad() does not turn dropout off
        with torch.no_grad():
            history["val"].append(mse(model(X_val), y_val).item())
        model.train()
    return history


history_fixed = train_colleague_fixed(make_model(0), X_train, y_train, X_val, y_val)
```

下面把同事的原始代码原样放进一个字符串，用字符串替换逐个修错误，画出四个阶段的图。每个替换都断言恰好命中一次。

```python
BUGGY_SRC = '''
def train_colleague(model, X_train, y_train, X_val, y_val, lr=0.1, epochs=40, batch_size=50, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mse = lambda pred, target: ((pred - target) ** 2).mean()
    n = X_train.shape[0]
    history = {"train": [], "val": []}
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = mse(model(X_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        with torch.no_grad():
            history["val"].append(mse(model(X_val), y_val).item())
    return history
'''

FIXES = {
    "A": ("            loss = mse(model(X_train[idx]), y_train[idx])\n",
          "            opt.zero_grad()\n            loss = mse(model(X_train[idx]), y_train[idx])\n"),
    "B": ("((pred - target) ** 2).mean()", "((pred.squeeze(-1) - target) ** 2).mean()"),
    "C": ('        with torch.no_grad():\n            history["val"].append(mse(model(X_val), y_val).item())\n',
          '        model.eval()\n        with torch.no_grad():\n            history["val"].append(mse(model(X_val), y_val).item())\n'
          '        model.train()\n'),
}


def colleague_version(fixes, src=BUGGY_SRC, extra=()):
    """train_colleague with the named fixes applied (and optional extra (old, new) replacements)."""
    for old, new in [FIXES[name] for name in fixes] + list(extra):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    namespace = {"torch": torch}
    exec(src, namespace)
    return namespace["train_colleague"]


stages = {}
for fixes in ((), ("A",), ("A", "B"), ("A", "B", "C")):
    stage_model = make_model(0)
    stages[fixes] = (colleague_version(fixes)(stage_model, X_train, y_train, X_val, y_val), stage_model)

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True, sharey=True)
titles = ["as written", "after fixing A", "after fixing A and B", "after fixing A, B and C"]
for ax, (fixes, (h, _)), title in zip(axes.flat, stages.items(), titles):
    plot_curves(ax, h["train"], h["val"], range(1, len(h["val"]) + 1), title=title)
axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "part2_stages.png"), dpi=120)
plt.close(fig)
```

### Part 3

**梯度。** 记 $\tilde X = [X,\ \mathbf 1]$（加一列 1），残差 $r = \tilde X \theta - y$，则 $L = \frac1n \lVert r \rVert^2$，

$$\nabla L(\theta) = \frac2n \tilde X^\top r, \qquad H = \nabla^2 L = \frac2n \tilde X^\top \tilde X .$$

注意因子 2：损失没有 $\frac12$，所以 Hessian 是 $[x, 1]$ 的二阶矩矩阵的 **2 倍**。

**有限差分的误差。** 中心差分有两种误差：

- 截断误差 $O(\varepsilon^2)$，来自三阶导数。$L$ 是二次函数，三阶导数为零，所以这里截断误差**恰好是 0**，$\varepsilon$ 取大一点也没关系。
- 舍入误差：$f$ 本身有大约 $u\,|f|$ 的误差（$u$ 是单位舍入误差，float64 约 $1.1 \times 10^{-16}$，float32 约 $6 \times 10^{-8}$），除以 $2\varepsilon$ 之后被放大成 $u\,|f|/\varepsilon$。

`theta0` 处 $L \approx 9$，梯度的范数约 6，所以相对误差大约是 $u\,|f| / (\varepsilon \lVert \nabla L \rVert) \approx 1.5\,u/\varepsilon$。float64、$\varepsilon = 10^{-6}$ 时它在 $10^{-10}$ 量级（实测 $3.6 \times 10^{-10}$）。autograd 和手推公式算的是同一个式子，只差舍入（实测 $10^{-15}$ 量级）。float32、$\varepsilon = 10^{-6}$ 时舍入误差被放大到和梯度只差一个数量级，结果不能用（实测 $0.15$）。float32、$\varepsilon = 10^{-2}$ 时降到 $10^{-5}$ 量级（实测 $2.4 \times 10^{-5}$）。所以做梯度核对要用 float64；对一般的非二次函数，$\varepsilon$ 取 $u^{1/3}$ 左右能平衡两种误差。

```python
X = X_train.double()
y = y_train.double()
n = X.shape[0]


def linear_loss(theta, X, y):
    return ((X @ theta[:-1] + theta[-1] - y) ** 2).mean()


def linear_grad(theta, X, y):
    n = X.shape[0]
    r = X @ theta[:-1] + theta[-1] - y
    return torch.cat([2.0 / n * (X.T @ r), (2.0 / n * r.sum()).reshape(1)])


def fd_grad(f, theta, eps):
    g = torch.zeros_like(theta)
    for i in range(theta.numel()):
        e = torch.zeros_like(theta)
        e[i] = eps
        g[i] = (f(theta + e) - f(theta - e)) / (2 * eps)
    return g


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


theta0 = torch.randn(9, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
g_hand = linear_grad(theta0, X, y)
g_fd = fd_grad(lambda t: linear_loss(t, X, y), theta0, 1e-6)
theta_req = theta0.clone().requires_grad_(True)
(g_auto,) = torch.autograd.grad(linear_loss(theta_req, X, y), theta_req)
X32, y32, theta32 = X.float(), y.float(), theta0.float()
g_fd32_small = fd_grad(lambda t: linear_loss(t, X32, y32), theta32, 1e-6)
g_fd32_large = fd_grad(lambda t: linear_loss(t, X32, y32), theta32, 1e-2)
errors = {
    "fd, float64, eps=1e-6": rel_err(g_fd, g_hand),
    "autograd, float64": rel_err(g_auto, g_hand),
    "fd, float32, eps=1e-6": rel_err(g_fd32_small.double(), g_hand),
    "fd, float32, eps=1e-2": rel_err(g_fd32_large.double(), g_hand),
}

X_tilde = torch.cat([X, torch.ones(n, 1, dtype=torch.float64)], dim=1)
H = 2.0 / n * X_tilde.T @ X_tilde
h, U = torch.linalg.eigh(H)
h, U = h.flip(0), U.flip(1)                       # largest first
lr_star = 2.0 / h[0].item()
kappa_H = (h[0] / h[-1]).item()
theta_hat = torch.linalg.lstsq(X_tilde, y[:, None]).solution[:, 0]
loss_hat = linear_loss(theta_hat, X, y).item()
```

**阈值。** 梯度下降一步是 $\theta_{t+1} - \hat\theta = (I - \eta H)(\theta_t - \hat\theta)$。把误差投影到 $H$ 的特征向量上，$a_i(t) = u_i^\top(\theta_t - \hat\theta)$，每个分量各自按等比数列变化：

$$a_i(t) = (1 - \eta h_i)^t\, a_i(0), \qquad L(\theta_t) - L(\hat\theta) = \tfrac12 (\theta_t - \hat\theta)^\top H (\theta_t - \hat\theta) = \sum_i \tfrac12 h_i\, a_i(t)^2 .$$

**(b) 定理。** 设 $H$ 正定（这里 $\tilde X$ 列满秩）。固定步长的梯度下降从任何起点都收敛到 $\hat\theta$，当且仅当所有 $|1 - \eta h_i| < 1$，即

$$0 < \eta < \eta^* = \frac{2}{h_1} = \frac{2}{\lambda_{\max}(H)} .$$

$\eta = \eta^*$ 时，$h_1$ 方向的因子是 $-1$：这个分量每步变号，大小不变，既不收敛也不发散。只要 $h_1$ 严格大于其余特征值，其它分量都会衰减。

`seed=0` 时 $h_1 = 2.453$，$\eta^* = 0.8153$。常见的坑是忘了因子 2，把阈值写成 $2/\lambda_{\max}(\tilde X^\top \tilde X / n)$。这个值是真正阈值的 2 倍，最陡方向的因子变成 $1 - 4 = -3$，几步就发散。

为什么 $H$ 的特征值都在 2 附近（$\kappa(H) = 1.44$）：输入是独立的标准正态，再加一列 1，所以 $\tilde X^\top \tilde X / n \approx I_9$，$H \approx 2I$。有限样本让特征值散开，散开的范围大约是 $2(1 \pm \sqrt{d/n})^2 \approx [1.64,\ 2.40]$（$d = 9$，$n = 1000$，这是 Marchenko–Pastur 分布的边缘）。`seed=0` 时实测是 $[1.70,\ 2.45]$。

**(a) 三条曲线。** 取 $\eta = c\,\eta^*$，最陡方向的因子是 $1 - 2c$。其它方向因 $h_i < h_1$，因子是 $1 - 2c\,h_i/h_1$。

- $c = 0.9$：最陡方向的因子是 $-0.8$。其它方向的因子都在 $-0.8$ 和 $-0.25$ 左右之间（因为 $h_i/h_1 \ge 0.69$），衰减得更快。所以后期每步乘 $0.8^2 = 0.64$，是一条斜率 $\ln 0.64 = -0.45$ 的直线，大约每 5 步下降一个数量级，七八十步之后碰到 $10^{-16}$ 附近的舍入下限。
- $c = 1.0$：最陡方向每步乘 $-1$，它对应的那部分损失 $\frac12 h_1 a_1(0)^2$ 永远不变；其它方向都在衰减。所以曲线先下降，然后变平，停在 $\frac12 h_1 a_1(0)^2$。`seed=0` 时这个值是 $0.151$，起点是 $1.058$。
- $c = 1.1$：最陡方向每步乘 $-1.2$，损失每步乘 $1.44$，后期是一条斜率 $\ln 1.44 = +0.36$ 的直线，100 步后到 $10^{15}$ 量级。$h_i > h_1/1.1$ 的方向也在发散，只是慢一些（`seed=0` 时第二个方向每步乘 $-1.02$）。开头几步曲线先往下走，因为其余方向的误差正在衰减，`seed=0` 时最低点在第 2 步。

学习率只比阈值大 10%，损失就从“收敛很快”变成“指数爆炸”。阈值附近的行为是突变的。

**(c) `reduction="sum"`。** 求和的损失是平均的 $n$ 倍，梯度和 Hessian 也都是 $n$ 倍，所以阈值变成 $\eta^*/n = 8.2 \times 10^{-4}$。把学习率同时除以 $n$，轨迹和平均损失完全一样。反过来，学习率不变而换成求和，最陡方向的因子是 $1 - 0.9 \times 2 \times 1000 = -1799$，第一步就飞出去。

```python
def gd_linear(X, y, lr, steps):
    model = nn.Linear(X.shape[1], 1).to(X.dtype)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    thetas = [torch.cat([model.weight.detach().reshape(-1), model.bias.detach()]).clone()]
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()
        thetas.append(torch.cat([model.weight.detach().reshape(-1), model.bias.detach()]).clone())
    return torch.stack(thetas)


GD_STEPS = 100
runs = {c: gd_linear(X, y, c * lr_star, GD_STEPS) for c in (0.9, 1.0, 1.1)}
excess = {c: torch.stack([linear_loss(t, X, y) for t in TH]) - loss_hat for c, TH in runs.items()}
a0 = U.T @ (torch.zeros(9, dtype=torch.float64) - theta_hat)        # a_i(0): the start is theta = 0
plateau = 0.5 * h[0].item() * a0[0].item() ** 2

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for c, color in ((0.9, BLUE), (1.0, AQUA), (1.1, ORANGE)):
    ax.semilogy(range(GD_STEPS + 1), excess[c].abs().clamp(min=1e-16), color=color, lw=2, label=f"lr = {c} x 2 / h_1")
ax.axhline(plateau, color=GRAY, ls="--", lw=1)
ax.text(GD_STEPS, plateau * 1.5, "h_1 a_1(0)^2 / 2", ha="right", va="bottom", fontsize=8, color=GRAY)
ax.set_xlabel("step t")
ax.set_ylabel("L(theta_t) - L(theta_hat)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "part3_threshold.png"), dpi=120)
plt.close(fig)
```

### 追问

**1. 修错误 C 时，如果只加了 `model.eval()`，忘了切回 `model.train()`，会怎样？** 从第二个 epoch 起 dropout 就一直关着，等于在训练一个没有 dropout 的网络。训练曲线降到 $0.012$，而且跑到了验证曲线下面；验证曲线最后是 $0.017$，反而比 Part 1 的 $0.020$ 更好。这说明这个任务上 $p = 0.2$ 的 dropout 并没有帮助：1000 个样本、平滑的目标、小网络，本来就不怎么过拟合。顺带一提，这样“修”完之后 history 就和 Part 1 对不上了。

**2. 少了 `zero_grad` 为什么会出 `nan`？** 它等价于 `SGD(momentum=1.0)`。在二次函数上用满批量，每个分量满足 $a_{t+1} = (2 - \eta h)\,a_t - a_{t-1}$。当 $0 < \eta h < 4$ 时，解是不衰减的振荡：

$$a_t = a_0 \, \frac{\cos(\omega (t + \frac12))}{\cos(\omega/2)}, \qquad \cos\omega = 1 - \frac{\eta h}{2} .$$

每个分量的振幅是 $|a_0| / \sqrt{1 - \eta h / 4}$，比初值略大，而且永远不衰减。所以损失在 0 和略高于初值之间来回振荡，既不收敛也不发散。实测在满批量下跑 2000 步，损失从没超过初值的 1.1 倍，最后 200 步里也还会回到初值的一半以上。换成小批量后，每一步的梯度和曲率都在变，没有阻尼的振荡不但压不下去，还会被放大。同一个线性模型跑 800 个小批量步（相当于 40 个 epoch），损失涨了 $10^6$ 倍以上。MLP 上是同样的情况，Part 2 的第一张图里第 10 个 epoch 就出现了 `nan`。

**3. 训练循环里把损失改成 `reduction="sum"` 会怎样？** 梯度变成 50 倍，等效学习率从 0.1 变成 5，第一个 epoch 就是 `nan`。学习率除以 50 就和原来完全一样。只是打印出来的训练损失是 50 倍，验证损失（对 1000 个样本求和）是 1000 倍。

**4. 能不能也对 MLP 做有限差分核对？** 能，但要注意三点：用 `eval()` 模式，否则两次前向的 dropout 掩码不同，差分的结果完全没有意义（实测相对误差 $10^5$ 以上）；用 float64；ReLU 有折点，$\varepsilon$ 跨过折点的那个坐标会出错，所以只核对少数几个坐标，或者换一个随机点再试。按这三点做，相对误差在 $10^{-8}$ 以下。

**5. $2/\lambda_{\max}$ 对 MLP 还成立吗？** 不再是严格的定理，因为 Hessian 随参数变化，只能在当前点附近局部地用。可以用 Hessian-向量乘积（double backward）加幂迭代估计 $\lambda_{\max}$，不需要构造 Hessian。训练结束时，全训练集、eval 模式下损失的 $\lambda_{\max}$ 约为 9.8，局部阈值 $2/\lambda_{\max} \approx 0.20$，我们用的学习率 0.1 大约是它的一半。

<details>
<summary>验证代码（可运行）</summary>

```python
def first_nonfinite(values):
    return next((i for i, v in enumerate(values) if not math.isfinite(v)), None)


def all_finite(h):
    return first_nonfinite(h["train"]) is None and first_nonfinite(h["val"]) is None


# ---------- the data ----------
assert X_train.shape == (1000, 8) and y_train.shape == (1000,) and X_train.dtype == torch.float32
assert not torch.equal(X_train, X_val)

# ---------- Part 1 (a): the reference lines, on a large sample ----------
X_big, y_big = make_data(400_000, seed=12345)
X_big, y_big = X_big.double(), y_big.double()
clean_big = torch.relu(X_big[:, 0] - X_big[:, 1]) - torch.relu(X_big[:, 2]) + 0.5 * X_big[:, 3]
assert abs(VAR_Y - 1.2825) < 5e-5 and abs(LINEAR_MSE - 0.2825) < 5e-5 and abs(MEAN_Y - 0.165) < 5e-4
assert abs(y_big.var().item() - VAR_Y) < 0.01
assert abs(y_big.mean().item() - MEAN_Y) < 0.01
assert abs((y_big - clean_big).var().item() - NOISE) < 5e-4
design_big = torch.cat([X_big, torch.ones(len(X_big), 1, dtype=torch.float64)], dim=1)
coef_big = torch.linalg.lstsq(design_big, y_big[:, None]).solution[:, 0]
expected_coef = torch.tensor([0.5, -0.5, -0.5, 0.5, 0, 0, 0, 0, MEAN_Y], dtype=torch.float64)
assert torch.allclose(coef_big, expected_coef, atol=0.01)
assert abs(((design_big @ coef_big - y_big) ** 2).mean().item() - LINEAR_MSE) < 0.005
# relu moments used in the derivation
z = X_big[:, 0] * math.sqrt(2)
assert abs(torch.relu(z).mean().item() - math.sqrt(2) / math.sqrt(2 * math.pi)) < 0.005
assert abs((torch.relu(z) ** 2).mean().item() - 1.0) < 0.01
assert abs((z * torch.relu(z)).mean().item() / (z ** 2).mean().item() - 0.5) < 0.005

# ---------- Part 1 (b): the shape of the validation curve ----------
E = 40
assert len(history["train"]) == E and len(history["val"]) == E + 1 and len(history["train_eval"]) == E + 1
val = history["val"]
linear_val = ((torch.cat([X_val, torch.ones(1000, 1)], 1).double()
               @ torch.linalg.lstsq(torch.cat([X_train, torch.ones(1000, 1)], 1).double(),
                                    y_train.double()[:, None]).solution[:, 0] - y_val.double()) ** 2).mean().item()
assert abs(linear_val - 0.29) < 0.005 and abs(linear_val - LINEAR_MSE) < 0.03
assert val[0] > 1.0 and abs(val[0] - 1.39) < 0.01
assert abs(val[1] - 0.21) < 0.01 and val[1] < linear_val < val[0]       # below the linear model after one epoch
assert abs(val[-1] - 0.020) < 0.001 and min(val) > NOISE                 # about twice the noise floor, never below it
assert val[-1] < 0.1 * val[1]

# ---------- Part 1 (c): train above val, and why ----------
assert all(history["train"][e] > val[e + 1] for e in range(E))           # every single epoch
assert abs(history["train"][-1] - 0.053) < 0.001 and abs(history["train_eval"][-1] - 0.017) < 0.001
assert abs(history["train_eval"][-1] - val[-1]) < 0.005                  # train set in eval mode looks like val
assert abs(penalty_train - 0.036) < 0.001 and abs(gap_end - penalty_train) < 0.002
# the identity E_mask[(f_drop - y)^2] = (f_eval - y)^2 + p/(1-p) sum_j (w_j h_j)^2, by Monte Carlo over masks
model.train()
torch.manual_seed(123)
with torch.no_grad():
    mc_loss = sum(((model(X_train).squeeze(-1) - y_train) ** 2).mean().item() for _ in range(400)) / 400
assert abs(mc_loss - (history["train_eval"][-1] + penalty_train)) < 0.002
with torch.no_grad():                                                    # per-sample mean and variance over masks
    x5 = X_train[:5]
    draws = torch.stack([model(x5).squeeze(-1) for _ in range(4000)]).double()
    model.eval()
    f_eval5 = model(x5).squeeze(-1).double()
    h5 = torch.relu(model[0](x5)).double()
    var_pred = 0.2 / 0.8 * ((h5 * model[3].weight.double()) ** 2).sum(dim=1)
assert bool(((draws.mean(0) - f_eval5).abs() < 5 * (var_pred / 4000).sqrt()).all())
assert bool(((draws.var(0) / var_pred - 1).abs() < 0.12).all())
assert not model.training
# evaluate() restores the mode it found
model.train()
evaluate(model, X_val, y_val)
assert model.training
model.eval()
evaluate(model, X_val, y_val)
assert not model.training
# training is reproducible bit for bit
assert train(make_model(0), X_train, y_train, X_val, y_val) == history

# ---------- Part 2: the three bugs ----------
assert all(BUGGY_SRC.count(old) == 1 for old, _ in FIXES.values())
h_all, _ = stages[()]
h_a, model_a = stages[("A",)]
h_ab, model_ab = stages[("A", "B")]
h_abc, _ = stages[("A", "B", "C")]

# as written: no progress, then nan at epoch 10
assert first_nonfinite(h_all["train"]) == 9 and first_nonfinite(h_all["val"]) == 9
assert min(h_all["train"][:9]) > 1.3 and abs(max(h_all["train"][:9]) - 3.2) < 0.1

# bug A: .grad holds the sum of all gradients so far
probe = make_model(0)
probe_opt = torch.optim.SGD(probe.parameters(), lr=0.1)
xb1, yb1, xb2, yb2 = X_train[:50], y_train[:50], X_train[50:100], y_train[50:100]
probe.eval()                                   # no dropout, so the two gradients are reproducible
((probe(xb1).squeeze(-1) - yb1) ** 2).mean().backward()
g1 = probe[0].weight.grad.clone()
probe_opt.step()
(g2,) = torch.autograd.grad(((probe(xb2).squeeze(-1) - yb2) ** 2).mean(), probe[0].weight)
((probe(xb2).squeeze(-1) - yb2) ** 2).mean().backward()
assert torch.allclose(probe[0].weight.grad, g1 + g2, atol=1e-7) and not torch.allclose(probe[0].weight.grad, g2, atol=1e-4)
# ... which is SGD with momentum 1, exactly
only_a_left = colleague_version(("B", "C"))(make_model(0), X_train, y_train, X_val, y_val)
fixed_src = BUGGY_SRC
for old, new in FIXES.values():
    fixed_src = fixed_src.replace(old, new)
momentum_one = colleague_version((), src=fixed_src, extra=[("lr=lr)", "lr=lr, momentum=1.0)")])(
    make_model(0), X_train, y_train, X_val, y_val)
same = lambda a, b: a == b or (math.isnan(a) and math.isnan(b))
assert all(same(a, b) for a, b in zip(only_a_left["train"] + only_a_left["val"], momentum_one["train"] + momentum_one["val"]))
assert first_nonfinite(only_a_left["train"]) is not None                  # bug A alone still blows up

# after fixing A: flat at Var(y), the model predicts a constant
var_train = y_train.var(unbiased=False).item()
assert all_finite(h_a)
assert abs(h_a["train"][-1] - 1.32) < 0.01 and abs(h_a["val"][-1] - 1.26) < 0.01
assert all(abs(v - var_train) < 0.1 for v in h_a["train"][5:])
model_a.eval()
with torch.no_grad():
    pred_a = model_a(X_train).squeeze(-1)
assert pred_a.std().item() < 0.05 and abs(pred_a.std().item() - 0.03) < 0.005 and abs(y_train.std().item() - 1.14) < 0.005
# bug B: the shape, and the decomposition of the broadcast loss
xb, yb = X_train[:50], y_train[:50]
with torch.no_grad():
    out = make_model(0)(xb)
assert out.shape == (50, 1) and (out - yb).shape == (50, 50)
f64, y64 = out.squeeze(-1).double(), yb.double()
lhs = ((f64[:, None] - y64[None, :]) ** 2).mean()
rhs = ((f64 - y64.mean()) ** 2).mean() + y64.var(unbiased=False)
assert abs((lhs - rhs).item()) < 1e-12
only_b_left = colleague_version(("A", "C"))(make_model(0), X_train, y_train, X_val, y_val)
assert all_finite(only_b_left) and min(only_b_left["train"]) > 1.0 and min(only_b_left["val"]) > 1.0   # bug B alone still fails

# after fixing A and B: val is as high as train, and noisy (bug C)
true_val_ab = evaluate(model_ab, X_val, y_val)
penalty_val_ab = dropout_penalty(model_ab, X_val)
assert all_finite(h_ab)
assert abs(h_ab["val"][-1] - 0.062) < 0.001 and abs(true_val_ab - 0.023) < 0.001
assert abs(h_ab["val"][-1] - true_val_ab - 0.039) < 0.001 and abs(penalty_val_ab - 0.038) < 0.001
assert h_ab["val"][-1] > 2 * true_val_ab and abs(h_ab["val"][-1] - (true_val_ab + penalty_val_ab)) < 0.01
assert model_ab.training                                                  # the colleague's model is still in train mode
with torch.no_grad():
    v1 = ((model_ab(X_val).squeeze(-1) - y_val) ** 2).mean().item()
    v2 = ((model_ab(X_val).squeeze(-1) - y_val) ** 2).mean().item()
assert v1 != v2 and abs(v1 - v2) < 0.02                                   # two evaluations of the same model differ
assert history["train"] != h_ab["train"]                                  # bug C even changes the trained model

# all three fixed: identical to Part 1, bit for bit
assert h_abc["train"] == history["train"] and h_abc["val"] == history["val"][1:]
assert history_fixed == h_abc

# ---------- Part 3: gradients ----------
assert torch.allclose(theta0, torch.randn(9, generator=torch.Generator().manual_seed(0), dtype=torch.float64))
assert abs(linear_loss(theta0, X, y).item() - 8.74) < 0.01 and abs(g_hand.norm().item() - 5.74) < 0.01
u64, u32 = torch.finfo(torch.float64).eps / 2, torch.finfo(torch.float32).eps / 2
assert abs(u64 * 8.74 / (1e-6 * 5.74) / (1.5 * u64 / 1e-6) - 1) < 0.02   # the rough error model in the text
assert 1e-11 < errors["fd, float64, eps=1e-6"] < 1e-9 and abs(errors["fd, float64, eps=1e-6"] / 3.6e-10 - 1) < 0.05
assert errors["autograd, float64"] < 1e-14
# float32 roundoff depends on the summation order of the build, so the quoted values get a wide margin
assert 0.03 < errors["fd, float32, eps=1e-6"] < 0.5 and abs(errors["fd, float32, eps=1e-6"] / 0.15 - 1) < 0.3
assert 1e-6 < errors["fd, float32, eps=1e-2"] < 1e-4 and abs(errors["fd, float32, eps=1e-2"] / 2.4e-5 - 1) < 0.5
for name, u, eps in (("fd, float64, eps=1e-6", u64, 1e-6), ("fd, float32, eps=1e-6", u32, 1e-6), ("fd, float32, eps=1e-2", u32, 1e-2)):
    assert 0.3 < errors[name] / (1.5 * u / eps) < 10                      # within an order of magnitude of the model
# zero truncation error on a quadratic: a huge eps is still exact in float64
assert rel_err(fd_grad(lambda t: linear_loss(t, X, y), theta0, 10.0), g_hand) < 1e-12
# the Hessian
H_auto = torch.autograd.functional.hessian(lambda t: linear_loss(t, X, y), theta0)
assert torch.allclose(H, H_auto, atol=1e-12)
assert torch.allclose(U @ torch.diag(h) @ U.T, H, atol=1e-12)
assert abs(h[0].item() - 2.453) < 5e-4 and abs(lr_star - 0.8153) < 5e-4 and abs(kappa_H - 1.44) < 0.005
assert abs(loss_hat - 0.2893) < 5e-5
assert abs(h[-1].item() - 1.70) < 0.005 and (h[1] / h[0]).item() > 1 / 1.1
mp_low, mp_high = 2 * (1 - math.sqrt(9 / n)) ** 2, 2 * (1 + math.sqrt(9 / n)) ** 2
assert abs(mp_low - 1.64) < 0.005 and abs(mp_high - 2.40) < 0.005
assert mp_low * 0.95 < h[-1].item() and h[0].item() < mp_high * 1.05
assert torch.allclose(theta_hat, expected_coef, atol=0.1)                 # close to the population linear fit
# the factor-2 trap: 2 / lambda_max(X~^T X~ / n) is twice the real threshold
assert abs(2.0 / torch.linalg.eigvalsh(X_tilde.T @ X_tilde / n)[-1].item() - 2 * lr_star) < 1e-12
trap = gd_linear(X, y, 2 * lr_star, 20)
assert linear_loss(trap[-1], X, y).item() > 1e8

# ---------- Part 3: gradient descent around the threshold ----------
steps_axis = torch.arange(GD_STEPS + 1, dtype=torch.float64)[:, None]
for c, TH in runs.items():
    assert TH.shape == (GD_STEPS + 1, 9) and bool((TH[0] == 0).all())
    A = (TH - theta_hat) @ U
    closed = a0 * (1 - c * lr_star * h) ** steps_axis
    assert (A - closed).abs().max().item() < 1e-12 * A.abs().max().item()   # relative to the largest component
    quad = 0.5 * (h * A ** 2).sum(dim=1)
    assert torch.allclose(excess[c], quad, rtol=1e-9, atol=1e-14)
assert abs(excess[1.0][0].item() - 1.058) < 5e-4 and abs(plateau - 0.151) < 5e-4
assert abs(abs(1 - 0.9 * lr_star * h[0].item()) - 0.8) < 1e-12
assert bool(((1 - 0.9 * lr_star * h).abs() <= 0.8 + 1e-12).all()) and (1 - 0.9 * lr_star * h).max().item() < -0.24
quad09 = 0.5 * (h * ((runs[0.9] - theta_hat) @ U) ** 2).sum(dim=1)
assert abs((quad09[40] / quad09[39]).item() - 0.64) < 1e-3                # the late ratio for c = 0.9
assert bool((quad09[1:60] < quad09[:59]).all())
assert excess[0.9][60].item() < 1e-11 and excess[0.9][GD_STEPS].abs().item() < 1e-15
assert abs(math.log(0.64) + 0.45) < 0.01 and abs(math.log(10) / -math.log(0.64) - 5) < 0.2
assert bool((excess[1.0][1:] <= excess[1.0][:-1] + 1e-15).all())          # c = 1.0: non-increasing, flattens at the plateau
assert abs(excess[1.0][-1].item() / plateau - 1) < 1e-6 and excess[1.0][10].item() > 1.01 * plateau
assert abs((excess[1.1][-1] / excess[1.1][-2]).item() - 1.44) < 1e-3     # c = 1.1: x 1.44 per step
assert 1e14 < excess[1.1][-1].item() < 1e16 and abs(math.log(1.44) - 0.36) < 0.01
assert int(excess[1.1].argmin()) == 2 and bool((excess[1.1][3:] > excess[1.1][2:-1]).all())
assert abs((1 - 1.1 * lr_star * h[1]).item() + 1.02) < 0.005
# reduction="sum": the threshold divides by n
def gd_linear_sum(lr, steps):
    m = nn.Linear(8, 1).double()
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    out = [torch.cat([m.weight.detach().reshape(-1), m.bias.detach()]).clone()]
    for _ in range(steps):
        opt.zero_grad()
        ((m(X).squeeze(-1) - y) ** 2).sum().backward()
        opt.step()
        out.append(torch.cat([m.weight.detach().reshape(-1), m.bias.detach()]).clone())
    return torch.stack(out)
assert abs(lr_star / n - 8.2e-4) < 5e-6
assert torch.allclose(gd_linear_sum(0.9 * lr_star / n, GD_STEPS), runs[0.9], atol=1e-12)
blown = gd_linear_sum(0.9 * lr_star, 3)
assert ((blown[2] - theta_hat).norm() / (blown[1] - theta_hat).norm()).item() > 1000
assert abs(1 - 0.9 * lr_star * n * h[0].item() + 1799) < 1e-6

# ---------- follow-up 1: model.eval() without model.train() afterwards ----------
half_fixed = colleague_version(("A", "B"), extra=[FIXES["C"][0:1] + (FIXES["C"][1].replace("        model.train()\n", ""),)])
h_half = half_fixed(make_model(0), X_train, y_train, X_val, y_val)
assert abs(h_half["train"][-1] - 0.012) < 0.001 and abs(h_half["val"][-1] - 0.017) < 0.001
assert h_half["train"][-1] < h_half["val"][-1] and h_half["val"][-1] < history["val"][-1]
assert h_half["train"] != history["train"]

# ---------- follow-up 2: no zero_grad on a quadratic ----------
def linear_without_zero_grad(lr, steps, batch_size=None, seed=0):
    m = nn.Linear(8, 1).double()
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    thetas = [torch.zeros(9, dtype=torch.float64)]
    for _ in range(steps):
        idx = torch.arange(n) if batch_size is None else torch.randperm(n, generator=g)[:batch_size]
        ((m(X[idx]).squeeze(-1) - y[idx]) ** 2).mean().backward()           # no zero_grad
        opt.step()
        thetas.append(torch.cat([m.weight.detach().reshape(-1), m.bias.detach()]).clone())
    return torch.stack(thetas)
lr_nz = 0.1
A_nz = (linear_without_zero_grad(lr_nz, 2000) - theta_hat) @ U
omega = torch.arccos(1 - lr_nz * h / 2)
t_axis = torch.arange(2001, dtype=torch.float64)[:, None]
assert bool((lr_nz * h < 4).all())
assert torch.allclose(A_nz, a0 * torch.cos(omega * (t_axis + 0.5)) / torch.cos(omega / 2), atol=1e-9)
assert bool((A_nz.abs().max(dim=0).values <= a0.abs() / (1 - lr_nz * h / 4).sqrt() + 1e-9).all())   # the amplitude
assert bool((A_nz.abs().max(dim=0).values >= 0.99 * a0.abs() / (1 - lr_nz * h / 4).sqrt()).all())   # ... is reached
loss_nz = 0.5 * (h * A_nz ** 2).sum(dim=1)
assert loss_nz[-200:].max().item() > 0.5 * loss_nz[0].item()               # never decays
assert loss_nz.max().item() < 1.1 * loss_nz[0].item()                     # never diverges either
loss_mb = 0.5 * (h * ((linear_without_zero_grad(lr_nz, 800, batch_size=50) - theta_hat) @ U) ** 2).sum(dim=1)
assert loss_mb[-1].item() > 1e6 * loss_mb[0].item()                         # minibatches: grows by more than 1e6

# ---------- follow-up 3: reduction="sum" in the minibatch loop ----------
sum_src = (fixed_src, [("((pred.squeeze(-1) - target) ** 2).mean()", "((pred.squeeze(-1) - target) ** 2).sum()")])
h_sum = colleague_version((), src=sum_src[0], extra=sum_src[1])(make_model(0), X_train, y_train, X_val, y_val)
assert first_nonfinite(h_sum["train"]) == 0
h_sum_small = colleague_version((), src=sum_src[0], extra=sum_src[1])(make_model(0), X_train, y_train, X_val, y_val, lr=0.1 / 50)
assert all(abs(a / 50 - b) < 1e-5 * b for a, b in zip(h_sum_small["train"], history["train"]))
assert all(abs(a / 1000 - b) < 1e-5 * b for a, b in zip(h_sum_small["val"], history["val"][1:]))

# ---------- follow-up 4: finite differences on the MLP ----------
mlp64 = copy.deepcopy(model).double().eval()
X64_train, y64_train = X_train.double(), y_train.double()
W1 = mlp64[0].weight
mlp_loss = lambda: ((mlp64(X64_train).squeeze(-1) - y64_train) ** 2).mean()
(grad_W1,) = torch.autograd.grad(mlp_loss(), W1)


def fd_entry(i, j, eps=1e-6):
    with torch.no_grad():
        W1[i, j] += eps
        plus = mlp_loss().item()
        W1[i, j] -= 2 * eps
        minus = mlp_loss().item()
        W1[i, j] += eps
    return (plus - minus) / (2 * eps)


coords = [(0, 0), (3, 5), (10, 2), (63, 7)]
assert all(abs(fd_entry(i, j) - grad_W1[i, j].item()) < 1e-8 * max(abs(grad_W1[i, j].item()), 1e-3) for i, j in coords)
mlp64.train()
torch.manual_seed(0)
assert all(abs(fd_entry(i, j) - grad_W1[i, j].item()) > 1e5 * abs(grad_W1[i, j].item()) for i, j in coords[:2])

# ---------- follow-up 5: the local threshold of the MLP, by Hessian-vector products and power iteration ----------
mlp64.eval()
params = list(mlp64.parameters())
grads = torch.autograd.grad(mlp_loss(), params, create_graph=True)


def top_hessian_eigenvalue(iters, seed):
    gen = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=gen, dtype=torch.float64) for p in params]
    norm = math.sqrt(sum((t ** 2).sum().item() for t in v))
    v = [t / norm for t in v]
    lam = None
    for _ in range(iters):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = sum((a * b).sum().item() for a, b in zip(hv, v))
        norm = math.sqrt(sum((t ** 2).sum().item() for t in hv))
        v = [t / norm for t in hv]
    return lam


lam_mlp = top_hessian_eigenvalue(100, seed=0)
assert abs(top_hessian_eigenvalue(150, seed=1) - lam_mlp) < 1e-3 * lam_mlp
assert abs(lam_mlp - 9.8) < 0.05 and abs(2 / lam_mlp - 0.20) < 0.005 and 2 / lam_mlp > 0.1

# ---------- the main facts hold for other seeds ----------
for seed in (1, 2):
    Xs_tr, ys_tr = make_data(1000, seed)
    Xs_va, ys_va = make_data(1000, seed + 1000)
    ms = make_model(seed)
    hs = train(ms, Xs_tr, ys_tr, Xs_va, ys_va, seed=seed)
    assert 0.01 < hs["val"][-1] < 0.04 and all(hs["train"][e] > hs["val"][e + 1] for e in range(E))
    assert abs(hs["train"][-1] - hs["train_eval"][-1] - dropout_penalty(ms, Xs_tr)) < 0.01
    run = lambda fixes: colleague_version(fixes)(make_model(seed), Xs_tr, ys_tr, Xs_va, ys_va, seed=seed)
    assert first_nonfinite(run(())["train"]) is not None                    # as written: nan
    assert first_nonfinite(run(("B", "C"))["train"]) is not None            # only A left: nan
    hb = run(("A", "C"))                                                    # only B left: stuck at Var(y)
    assert all_finite(hb) and abs(hb["train"][-1] - ys_tr.var(unbiased=False).item()) < 0.1
    mc_model = make_model(seed)
    hc = colleague_version(("A", "B"))(mc_model, Xs_tr, ys_tr, Xs_va, ys_va, seed=seed)   # only C left
    assert hc["val"][-1] > 2 * evaluate(mc_model, Xs_va, ys_va)
    full = run(("A", "B", "C"))
    assert full["train"] == hs["train"] and full["val"] == hs["val"][1:]

# ---------- the figures were written ----------
for name in ("part1_curves.png", "part2_stages.png", "part3_threshold.png"):
    assert os.path.getsize(os.path.join(OUT_DIR, name)) > 10_000

print("all checks passed")
```

</details>

</details>
