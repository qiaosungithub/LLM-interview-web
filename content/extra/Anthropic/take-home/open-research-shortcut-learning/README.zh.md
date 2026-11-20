# 72 小时开放研究：小视觉模型什么时候走捷径，能不能从数据事先预测

## 题目

这是一道约 72 小时的开放研究 take-home。题目只有一个问题，而且故意没有说清楚。数据怎么造、模型怎么选、做到哪里为止，都由你来定。

**问题。** 训练数据里常常有一个和标签相关、但不是我们想要的特征，比如背景、水印、拍摄设备。模型可能学这个虚假的捷径（shortcut），而不学本该学的特征。请研究：一个小的视觉模型什么时候会学捷径而不学本该学的特征？能不能从数据的性质事先预测这一点？

**条件。**

- 从拿到题目起，约 72 小时内交卷。只用 CPU，不提供 GPU。
- 图片用代码合成，不下载现成数据集。这样每个变量都在你手里。
- 可以读公开论文。用到别人的结论要注明出处，并且要用你自己的实验检验。
- 不能用 AI 编程助手，除非邀请信另有说明。

**交付物。**

- 一份不超过 4 页的书面报告（write-up），附录不限。
- 能从头跑到尾的代码，能复现报告里的每一个数字。
- 交卷后有一次约 45 分钟的评审通话：你先讲 10 分钟，然后回答问题。

**评审看什么。**

- "学了捷径"有没有一个可以测量的定义。
- "从数据预测"是不是真的只用了数据，并且在训练之前就写下了。
- 有没有对照（control），有没有在多个随机种子上重复；有没有区分"模型用了捷径"和"模型做错了"。
- 结论的强弱和证据是否相称；报告里有没有写清楚"还不知道什么"。
- 时间花在了哪里：砍掉了什么，为什么砍。

### Part 1 —— 设计一个可控的数据集

造一个数据集，里面有一个本该学的核心特征（core feature）和一个捷径特征。至少要能调两样东西：捷径和标签一致的频率，以及核心特征有多难。定义怎样测量"模型依赖捷径的程度"。

### Part 2 —— 训练并测量

在几组设置上训练小模型，测量它们依赖捷径的程度。要有一个捷径毫无用处的对照。

### Part 3 —— 从数据预测

给出一个只用数据就能算的预测量，在训练之前算好。检验它预测得怎么样：哪些方面准，哪些方面不准。

### Part 4 —— 解释差距，写报告

如果预测和实际有差距，给出一个解释并检验它。写出结论、对照、其他可能的解释、局限，以及"再给你一周，你会做什么"。

## 参考解答

<details>
<summary>展开参考解答</summary>

### 先定范围：72 小时怎么用

72 小时里真正能用的大约 20 到 25 小时。题目里的"捷径"可以是背景、纹理、颜色，模型可以是 CNN 或 ViT，还可以研究缓解方法。每一条都能做一个月。先把问题收窄成：

> 一个核心特征、一个捷径、两个旋钮。捷径依赖的程度，能不能用一个**只靠数据分布算出来的量**事先排出顺序、估出大小？

砍掉的东西：

- 不用真实数据集（如 Waterbirds）。真实数据里算不出"理想情况下该依赖多少捷径"，而这正是回答"能不能预测"的关键。
- 只用一种很小的 CNN，不比较架构。
- 不做缓解方法（重新加权、group DRO 等），放进"再给一周"。
- 捷径的强度（标记的亮度）和位置固定，只调两个旋钮。

| 时间 | 做什么 | 产出 |
| --- | --- | --- |
| 第 1 天上午 | 写下定义和预测；写数据生成器；推出贝叶斯最优观察者并用模拟核对 | 一页"定义与预测"笔记；数据代码和预测表 |
| 第 1 天下午到晚上 | 搭小 CNN，把多个模型并成一次训练，调到几分钟内跑完；先跑一个设置 | 能跑的训练代码 |
| 第 2 天 | 扫两个旋钮，每格 3 个模型；冲突测试；和预测比较 | 主结果表 |
| 第 3 天上午 | 解释差距：保存训练过程中的快照，看核心特征学得多快；换种子重跑 | 学习曲线表 |
| 第 3 天下午 | 写报告；从头重跑一遍代码，逐个核对报告里的数字 | 报告 + 代码 |

### Part 1 —— 数据、测量方法和事先的预测

**数据。** 图片是 12×12 的灰度图。

- **核心特征：** 一根长度为 5 的亮条，横着是标签 0，竖着是标签 1。位置在第 2 行以下随机（行号从 0 开始），共 80 种横条、72 种竖条，等概率。
- **捷径：** 左上角 2×2 的标记，亮（+2）或暗（−2）。标记和标签一致的概率是 $\rho$。亮条永远碰不到这个角落，所以两个特征在像素上分得开。
- 每个像素再加标准差为 $\sigma$ 的高斯噪声。

$\rho = 0.5$ 时标记毫无用处，这就是对照。$\sigma$ 越大，横竖越难分。

**怎么测"依赖捷径"。** 主指标是**冲突准确率**：测试图的标记故意指向错误的标签，看模型还能不能答对。1 表示完全不看标记，0 表示完全跟着标记走。另外两个辅助指标：

- 一致准确率：标记指向正确标签时的准确率。
- 核心 AUC：把标记固定（全亮或全暗），只看模型输出能不能把横竖分开。它回答的是"模型输出里还有没有核心特征的信息"。

**只用数据的预测：贝叶斯最优观察者（ideal observer）。** 数据是我们自己生成的，所以后验概率可以精确算出来。核心特征的对数似然比是对所有亮条模板求和：

$$\log\frac{p(x\mid 1)}{p(x\mid 0)} = \log \frac{1}{72}\sum_{t \in \text{竖}} e^{\langle x, t\rangle/\sigma^2} - \log \frac{1}{80}\sum_{t \in \text{横}} e^{\langle x, t\rangle/\sigma^2}$$

（每个模板的 $\|t\|^2$ 都是 5，在比值里抵消了。）再加上标记带来的对数似然比，就是最优分类器。它在冲突图上的准确率只取决于数据分布（$\rho$、$\sigma$、标记亮度），训练之前就能算。我们用 2000 张测试图来估计它。

一个容易忽略的事实：**跟着标记走不一定是错。** 下表里 $\sigma = 1.2$、$\rho = 0.8$ 时，最优分类器自己在冲突图上也只对 0.082。所以"模型用了捷径"本身不是问题，问题是它用得比数据所支持的多还是少。

```python
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)          # one thread: the run is bit-for-bit reproducible
SEED = 0
SIZE, BAR_LEN, TOP, AMP = 12, 5, 2, 2.0     # 12x12 images; bars stay below row TOP; the marker is the top-left TOP x TOP patch


def bar_templates():
    """Every clean bar image: horizontal bars (label 0) first, then vertical bars (label 1)."""
    horiz, vert = [], []
    for r in range(TOP, SIZE):
        for c in range(SIZE - BAR_LEN + 1):
            t = torch.zeros(SIZE, SIZE)
            t[r, c:c + BAR_LEN] = 1.0
            horiz.append(t)
    for r in range(TOP, SIZE - BAR_LEN + 1):
        for c in range(SIZE):
            t = torch.zeros(SIZE, SIZE)
            t[r:r + BAR_LEN, c] = 1.0
            vert.append(t)
    return torch.stack(horiz), torch.stack(vert)


T_H, T_V = bar_templates()
TEMPLATES, N_H, N_V = torch.cat([T_H, T_V]), len(T_H), len(T_V)


def bars(y, sigma, g):
    """Noisy bar images without a marker. y: integer labels of any shape; sigma broadcastable to y.shape + (1, 1)."""
    idx = torch.where(y == 1, N_H + torch.randint(0, N_V, y.shape, generator=g),
                      torch.randint(0, N_H, y.shape, generator=g))
    return TEMPLATES[idx] + sigma * torch.randn(*y.shape, SIZE, SIZE, generator=g)


def add_marker(x, m, amp=AMP):
    """Add +amp (m = 1) or -amp (m = 0) to the top-left patch."""
    x = x.clone()
    x[..., :TOP, :TOP] += amp * (2 * m[..., None, None] - 1)
    return x


def core_llr(x, sigma):
    """Exact log p(x | vertical) - log p(x | horizontal) from the bar pixels. x: [n, SIZE, SIZE]."""
    scores = x.flatten(1) @ TEMPLATES.flatten(1).T / sigma ** 2
    return ((torch.logsumexp(scores[:, N_H:], 1) - math.log(N_V))
            - (torch.logsumexp(scores[:, :N_H], 1) - math.log(N_H)))


def marker_llr(x, sigma, rho, amp=AMP):
    """Exact log-odds for label 1 carried by the marker patch, when the marker agrees with the label w.p. rho."""
    u = amp * x[:, :TOP, :TOP].sum((1, 2)) / sigma ** 2
    lr, l1r = math.log(rho), math.log(1 - rho)
    return torch.logaddexp(lr + u, l1r - u) - torch.logaddexp(l1r + u, lr - u)


def acc(score, y):
    return float(((score > 0).long() == y).float().mean())


SIGMAS, RHOS, N_TEST = [0.5, 0.85, 1.2], [0.5, 0.8, 0.9, 0.97], 2000
g_test = torch.Generator().manual_seed(10_000 + SEED)
y_test = torch.randint(0, 2, (N_TEST,), generator=g_test)
x_test = {s: bars(y_test, s, g_test) for s in SIGMAS}       # the same bars and noise for every model at a given sigma

bayes_core, bayes_conf = {}, {}
for s in SIGMAS:
    cl = core_llr(x_test[s], s)
    bayes_core[s] = acc(cl, y_test)
    x_conf = add_marker(x_test[s], 1 - y_test)                 # conflict images: the marker points to the wrong label
    for r in RHOS:
        bayes_conf[s, r] = acc(cl + marker_llr(x_conf, s, r), y_test)
table_bayes = [f"| {s} | {bayes_core[s]:.3f} | " + " | ".join(f"{bayes_conf[s, r]:.3f}" for r in RHOS[1:]) + " |"
               for s in SIGMAS]
print("\n".join(table_bayes))
```

最优观察者的预测（`SEED = 0`）：

| $\sigma$ | 只看亮条的准确率 | 冲突准确率 $\rho = 0.8$ | $\rho = 0.9$ | $\rho = 0.97$ |
| --- | --- | --- | --- | --- |
| 0.5 | 0.975 | 0.949 | 0.924 | 0.868 |
| 0.85 | 0.783 | 0.413 | 0.243 | 0.101 |
| 1.2 | 0.647 | 0.082 | 0.015 | 0.002 |

训练前写下的预测：冲突准确率随 $\rho$ 和 $\sigma$ 增大而下降；9 个 $\rho > 0.5$ 的格子里，CNN 的排序和上表一致。大小上，CNN 不会比最优观察者好，但差多少不知道。

### Part 2 —— 训练 36 个小 CNN 并测量

**模型。** 3×3 卷积（步长 2，8 个通道）→ ReLU → 3×3 卷积（16 个通道）→ ReLU → 全局最大池化 → 线性输出。3 个 $\sigma$ × 4 个 $\rho$ × 每格 3 个模型，共 36 个。

**一次训练 36 个模型。** 把 36 个模型拼成一个分组卷积（`groups=36`）：每组只看自己的输入通道，损失是 36 个模型损失之和，Adam 又是逐参数更新，所以和分开训练在数学上一样，只是快得多。

**训练。** Adam，学习率固定为 0.002（固定学习率，后面才能比较学习速度），1600 步，每步每个模型 64 张新生成的图。没有固定的训练集，所以不存在过拟合训练集的问题。在第 50、100、200、400、800、1600 步各存一份权重。

```python
class StackedCNN(nn.Module):
    """K independent CNNs: conv3x3 (stride 2) -> ReLU -> conv3x3 -> ReLU -> global max pool -> linear."""

    def __init__(self, K, c1=8, c2=16):
        super().__init__()
        self.K, self.c2 = K, c2
        self.conv1 = nn.Conv2d(K, K * c1, 3, stride=2, padding=1, groups=K)   # groups=K keeps the K models apart
        self.conv2 = nn.Conv2d(K * c1, K * c2, 3, padding=1, groups=K)
        self.head_w = nn.Parameter(torch.randn(K, c2) / math.sqrt(c2))
        self.head_b = nn.Parameter(torch.zeros(K))

    def forward(self, x):                                       # x: [n, K, SIZE, SIZE] -> logits [n, K]
        h = F.relu(self.conv2(F.relu(self.conv1(x))))
        h = h.amax(dim=(2, 3)).view(-1, self.K, self.c2)
        return (h * self.head_w).sum(-1) + self.head_b


def train_stack(rho, sigma, steps, batch, lr, seed, snapshots=()):
    """rho, sigma: [K] tensors, one entry per model. Fresh images every step; keeps a copy of the weights at `snapshots`."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)                                     # weight init
    K = len(rho)
    model = StackedCNN(K)
    opt = torch.optim.Adam(model.parameters(), lr=lr)           # constant lr, so learning speed can be compared
    snaps = {}
    for step in range(1, steps + 1):
        y = torch.randint(0, 2, (batch, K), generator=g)
        m = torch.where(torch.rand(batch, K, generator=g) < rho, y, 1 - y)
        x = add_marker(bars(y, sigma.view(1, K, 1, 1), g), m)
        loss = F.binary_cross_entropy_with_logits(model(x), y.float(), reduction="none").mean(0).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in snapshots:
            snaps[step] = {k: v.clone() for k, v in model.state_dict().items()}
    return model, snaps


N_REP, STEPS, BATCH, LR = 3, 1600, 64, 2e-3
SNAPS = [50, 100, 200, 400, 800, 1600]
conds = [(s, r, rep) for s in SIGMAS for r in RHOS for rep in range(N_REP)]     # 36 models
K = len(conds)
t0 = time.time()
model, snaps = train_stack(torch.tensor([c[1] for c in conds]), torch.tensor([c[0] for c in conds]),
                           STEPS, BATCH, LR, SEED, snapshots=SNAPS)
print(f"trained {K} CNNs in {time.time() - t0:.1f} s")
```

**测试。** 每个 $\sigma$ 一套 2000 张测试图，同一 $\sigma$ 的所有模型用同一套。每张图做三个版本：标记指向错误标签（冲突）、指向正确标签（一致）、没有标记。

```python
t0 = time.time()
x0 = torch.stack([x_test[c[0]] for c in conds], 1)            # [N_TEST, K, 12, 12]
yk = y_test[:, None].expand(N_TEST, K)
logits = {}
with torch.no_grad():
    for st in SNAPS:
        model.load_state_dict(snaps[st])
        logits[st] = {"conflict": model(add_marker(x0, 1 - yk)), "aligned": model(add_marker(x0, yk)), "none": model(x0)}
print(f"evaluated {len(SNAPS)} snapshots in {time.time() - t0:.1f} s")


def ks(s, r):
    return [i for i, c in enumerate(conds) if c[0] == s and c[1] == r]


def auc(pos, neg):
    ranks = torch.cat([pos, neg]).argsort().argsort().float() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def core_auc(st, k):
    """AUC for the true label with the marker held fixed (marker = 1 images, then marker = 0 images)."""
    lc, la = logits[st]["conflict"][:, k], logits[st]["aligned"][:, k]
    return (auc(la[y_test == 1], lc[y_test == 0]) + auc(lc[y_test == 1], la[y_test == 0])) / 2


sign = (2 * y_test - 1).float()
cnn_each, cnn_conf, cnn_align, own, cauc = {}, {}, {}, {}, {}
for s in SIGMAS:
    twins = ks(s, 0.5)                                          # same sigma, no shortcut in training
    for r in RHOS:
        cnn_each[s, r] = [acc(logits[STEPS]["conflict"][:, k], y_test) for k in ks(s, r)]
        cnn_conf[s, r] = float(np.mean(cnn_each[s, r]))
        cnn_align[s, r] = float(np.mean([acc(logits[STEPS]["aligned"][:, k], y_test) for k in ks(s, r)]))
        shift = math.log(r / (1 - r))                           # log-odds the marker carries in training
        own[s, r] = float(np.mean([((logits[STEPS]["none"][:, k] * sign - shift) > 0).float().mean() for k in twins]))
        cauc[s, r] = float(np.mean([core_auc(STEPS, k) for k in ks(s, r)]))
table_main = [f"| {s} | {r} | {bayes_conf[s, r]:.3f} | {', '.join(f'{a:.3f}' for a in cnn_each[s, r])} | "
              f"{cnn_conf[s, r]:.3f} | {own[s, r]:.3f} | {cauc[s, r]:.3f} |" for s in SIGMAS for r in RHOS]
print("\n".join(table_main))
```

结果（`SEED = 0`，训练结束时）。"自感知预测"一列在 Part 3 解释。

| $\sigma$ | $\rho$ | 最优观察者 冲突准确率 | CNN 冲突准确率（3 个模型） | CNN 平均 | 自感知预测 | CNN 核心 AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.5 | 0.975 | 0.959, 0.956, 0.956 | 0.957 | 0.959 | 0.993 |
| 0.5 | 0.8 | 0.949 | 0.887, 0.840, 0.891 | 0.873 | 0.902 | 0.988 |
| 0.5 | 0.9 | 0.924 | 0.792, 0.646, 0.765 | 0.734 | 0.848 | 0.979 |
| 0.5 | 0.97 | 0.868 | 0.211, 0.087, 0.176 | 0.158 | 0.738 | 0.894 |
| 0.85 | 0.5 | 0.783 | 0.722, 0.717, 0.733 | 0.724 | 0.737 | 0.815 |
| 0.85 | 0.8 | 0.413 | 0.156, 0.171, 0.263 | 0.197 | 0.285 | 0.774 |
| 0.85 | 0.9 | 0.243 | 0.004, 0.007, 0.036 | 0.016 | 0.129 | 0.697 |
| 0.85 | 0.97 | 0.101 | 0.001, 0.000, 0.000 | 0.000 | 0.027 | 0.629 |
| 1.2 | 0.5 | 0.647 | 0.589, 0.567, 0.551 | 0.569 | 0.586 | 0.633 |
| 1.2 | 0.8 | 0.082 | 0.002, 0.007, 0.002 | 0.003 | 0.012 | 0.614 |
| 1.2 | 0.9 | 0.015 | 0.001, 0.002, 0.001 | 0.001 | 0.000 | 0.574 |
| 1.2 | 0.97 | 0.002 | 0.001, 0.001, 0.001 | 0.001 | 0.000 | 0.540 |

怎么读：

- **对照成立。** $\rho = 0.5$ 时，冲突准确率和一致准确率相差不到 0.05，模型确实不看标记。它比最优观察者略差（$\sigma = 0.5$ 时 0.957 对 0.975），没有捷径时模型也不是最优的。
- **训练目标达到了。** 每个 $\rho > 0.5$ 的格子，一致准确率都在 0.95 以上。
- **依赖捷径的程度随 $\rho$、$\sigma$ 上升。** 冲突准确率随 $\rho$ 增大而下降，也随 $\sigma$ 增大而下降。$\sigma = 0.5$、$\rho = 0.97$ 时只剩 0.158；$\sigma = 1.2$ 时只要 $\rho \ge 0.8$ 就接近 0。
- **"学了捷径"不等于"没学核心特征"。** $\sigma = 0.5$、$\rho = 0.97$ 时冲突准确率只有 0.158，但固定标记后的核心 AUC 还有 0.894。模型学到了一部分横竖信息，只是在输出里，标记的分量压过了它。核心 AUC 也随 $\rho$ 增大而下降。
- **个别格子很不稳定。** $\sigma = 0.5$、$\rho = 0.97$ 这一格，三个模型从 0.087 到 0.211。只看一个模型的数字，在这里不可靠。

### Part 3 —— 从数据预测：排序大体准，大小不准

**排序。** 在 9 个 $\rho > 0.5$ 的格子上，最优观察者和 CNN 的冲突准确率的斯皮尔曼等级相关（Spearman rank correlation）是 0.88。也就是说，只用数据分布算出来的量，能把"谁更依赖捷径"大体排对，但事先写下的"排序一致"没有完全兑现。0.88 主要来自两个旋钮各自的方向：$\rho$ 或 $\sigma$ 单独变大，两边的冲突准确率都下降。一个旋钮变大、另一个变小时就不一定对：最优观察者认为 $\sigma = 0.5$、$\rho = 0.97$（0.868）比 $\sigma = 0.85$、$\rho = 0.8$（0.413）依赖捷径少得多，CNN 却是 0.158 对 0.197，反了过来；`SEED = 0` 到 4 里有 3 次是反的。

**大小。** 每一个 CNN 的冲突准确率都不比最优观察者高（容差 0.01），在中间的格子里低得多：$\sigma = 0.5$、$\rho = 0.97$ 时，最优观察者是 0.868，CNN 平均只有 0.158。9 个格子的平均绝对误差是 0.179。所以在这个实验里，只用数据的预测相当于一个**上限**（这是观察到的规律，不是定理）：模型依赖捷径的程度，至少有数据所支持的那么多，实际上更多。

**一个考虑模型能力的预测：自感知预测。** 最优观察者假设模型看亮条的能力是完美的。换成模型自己的能力：取同一个 $\sigma$ 下 $\rho = 0.5$ 训练的"孪生"模型，它没见过有用的标记，它在无标记图上的 logit $z(x)$ 可以看作"这个模型能从亮条里读出多少证据"。冲突图上，标记把对数几率往错的方向推了 $\log\frac{\rho}{1-\rho}$。于是预测：

$$\text{答对} \iff y_\pm \cdot z(x) > \log\frac{\rho}{1-\rho},\qquad y_\pm = \pm 1.$$

它每个 $\sigma$ 只要训练一个孪生模型，不需要在每个 $\rho$ 上训练。它和 CNN 的等级相关也是 0.88，排序上没有更好；但平均绝对误差降到 0.107。它在极端处仍然太乐观：$\sigma = 0.5$、$\rho = 0.97$ 时预测 0.738，实际 0.158。

```python
from scipy.stats import spearmanr

sel = [(s, r) for s in SIGMAS for r in RHOS if r != 0.5]
spear = float(spearmanr([bayes_conf[c] for c in sel], [cnn_conf[c] for c in sel])[0])
spear_own = float(spearmanr([own[c] for c in sel], [cnn_conf[c] for c in sel])[0])
err_bayes = float(np.mean([abs(bayes_conf[c] - cnn_conf[c]) for c in sel]))
err_own = float(np.mean([abs(own[c] - cnn_conf[c]) for c in sel]))
print(f"spearman(Bayes, CNN) = {spear:.2f}, spearman(own, CNN) = {spear_own:.2f}, "
      f"mean |Bayes - CNN| = {err_bayes:.3f}, mean |own - CNN| = {err_own:.3f}")


def steps_to(traj, thr):
    """First crossing of thr, interpolated linearly in log(step) between snapshots."""
    for i in range(1, len(SNAPS)):
        if traj[i - 1] < thr <= traj[i]:
            f = (thr - traj[i - 1]) / (traj[i] - traj[i - 1])
            return float(np.exp(np.log(SNAPS[i - 1]) + f * (np.log(SNAPS[i]) - np.log(SNAPS[i - 1]))))
    return float(SNAPS[0]) if traj[0] >= thr else float("nan")


traj = {r: [float(np.mean([core_auc(st, k) for k in ks(0.5, r)])) for st in SNAPS] for r in RHOS}
t75 = {r: steps_to(traj[r], 0.75) for r in RHOS}
table_time = [f"| {r} | " + " | ".join(f"{a:.3f}" for a in traj[r]) + f" | {t75[r]:.0f} | {t75[r] / t75[0.5]:.1f} |"
              for r in RHOS]
print("\n".join(table_time))
```

### Part 4 —— 为什么比预测的更依赖捷径：核心特征学得更慢

**假设。** 标记比亮条容易学：它在固定位置，亮度大。模型先学会标记以后，一致的样本（占 $\rho$）损失已经很小，推动模型学亮条的梯度主要来自少数冲突样本。所以 $\rho$ 越大，亮条学得越慢；训练步数固定时，亮条就学得不够。

**检验。** 用存下来的快照，看 $\sigma = 0.5$ 时核心 AUC 怎样随步数上升，以及升到 0.75 要多少步（在相邻快照之间按对数步数插值）：

| $\rho$ | 50 步 | 100 步 | 200 步 | 400 步 | 800 步 | 1600 步 | 到 0.75 的步数 | 相对 $\rho = 0.5$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.708 | 0.767 | 0.882 | 0.962 | 0.989 | 0.993 | 82 | 1.0 |
| 0.8 | 0.605 | 0.670 | 0.807 | 0.913 | 0.976 | 0.988 | 150 | 1.8 |
| 0.9 | 0.535 | 0.559 | 0.632 | 0.738 | 0.933 | 0.979 | 417 | 5.1 |
| 0.97 | 0.532 | 0.538 | 0.551 | 0.586 | 0.701 | 0.894 | 953 | 11.6 |

$\rho$ 越大，核心特征学得越慢：$\rho = 0.97$ 时要多花 11.6 倍的步数。这个顺序在换种子以后不变（个别种子上 $\rho = 0.8$ 和 0.9 几乎打平）。它和 Part 3 的差距方向一致：自感知预测用的孪生模型在 1600 步时已经把亮条学得差不多了，而 $\rho = 0.97$ 的模型还没有，所以预测太乐观。$\rho = 0.97$ 的核心 AUC 在最后 800 步里从 0.701 升到 0.894，还在涨。再训久一点，差距可能会缩小。这一点我们没有测，不能下结论。

**结果说明了什么。** 在这个合成任务上：

1. 依赖捷径的程度随"捷径有多可靠"（$\rho$）和"核心特征有多难"（$\sigma$）上升。
2. 只用数据算出的最优观察者能把依赖程度大体排对（等级相关 0.88；两个旋钮反向变化时会排错），但在这里只起到上限的作用：训练出的 CNN 更依赖捷径。
3. 差距和学习速度有关：捷径越可靠，核心特征学得越慢。

**结果没有说明什么。**

- 没有说明"更依赖捷径"在训练足够久以后是否还在。我们只看了一个训练长度。
- 没有把"标记容易检测"和"标记可靠"分开。标记的亮度和位置是固定的。
- 没有说明真实图片上的情况。真实数据里算不出最优观察者，Part 3 的上限也就算不出来。
- 只用了一种很小的 CNN。全局最大池化让它天然对位置不敏感，这对亮条有利；换架构可能改变结果。

**其他解释和对应的对照。**

| 另一种解释 | 对照 | 结果 |
| --- | --- | --- |
| 模型根本分不清横竖（容量不够） | $\rho = 0.5$ 的孪生模型 | $\sigma = 0.5$ 时冲突准确率 0.957，核心 AUC 0.993 |
| 冲突图在训练里很少见，所以测试时答错 | 最优观察者面对的是同一批测试图，也知道 $\rho$ | 它已经把"冲突少见"算进去了，CNN 仍然比它低 |
| 模型完全没学核心特征 | 固定标记后的核心 AUC | $\sigma = 0.5$、$\rho = 0.97$ 时还有 0.894：学到了一部分，只是被标记压过 |
| 只是训练不够 | 学习曲线 | 部分是：核心 AUC 在最后还在涨；训久以后差距会不会消失，没有测 |
| 种子运气 | 每格 3 个模型；把 `SEED` 改成 1、2 重跑整页 | 下面验证代码里的所有定性检查在 `SEED = 0`、1、2 都通过 |

**再给一周会做什么。**

- **把"容易"和"可靠"分开。** 扫标记亮度：亮度降低时，最优观察者的预测会变，CNN 的多余依赖是否跟着变？
- **训练长度。** 训到 10 倍步数，或者在相同训练损失处比较，看 CNN 和最优观察者的差距是否收窄。
- **不需要最优观察者的预测量。** 真实数据上可以分别测"只有核心特征时多快学会"和"只有捷径时多快学会"，用二者的学习速度之比预测依赖程度。可以对照 Hermann 和 Lampinen（2020）关于特征难易（可解码性）的工作，以及 Shah 等人（2020）关于简单性偏置（simplicity bias）的工作。
- **缓解方法。** 给冲突样本加权、group DRO，看排序预测是否仍然成立。
- **换架构和真实数据。** 小 ResNet、小 ViT；再到 Waterbirds 这类有背景捷径的数据集，只检验排序是否成立。

### 报告提纲

1. **问题和定义**（半页）：收窄后的问题；冲突准确率、核心 AUC；为什么要和最优观察者比。
2. **数据和预测**（半页）：数据生成；最优观察者的推导和预测表。
3. **主结果**（一页）：主结果表，配一张"冲突准确率对 $\rho$"的图，每个 $\sigma$ 一条线，最优观察者用虚线画在同一张图上。
4. **从数据预测**（半页）：排序准、大小不准；自感知预测。
5. **解释差距**（半页）：学习曲线表。
6. **对照、其他解释和局限**（半页）：上面那张五行的表。
7. **下一步**（几行）。
8. **附录**：超参数、每个种子的结果、复现命令。

讲 10 分钟时，只讲一张图（CNN 对比最优观察者）、一个结论（排序大体可预测、程度被低估）和一个解释（核心特征学得慢）。

### 追问

- **冲突准确率接近 0，就说明模型做错了吗？** 不一定。$\sigma = 1.2$、$\rho = 0.8$ 时，最优分类器在冲突图上也只对 0.082。在训练分布上，跟着标记走是对的。要判断模型是否"过度"依赖捷径，必须和最优观察者比，而不是和 1 比。
- **真实数据上算不出最优观察者，你的方法还有什么用？** 排序的结论可以迁移：用"核心特征和捷径各自有多容易学"来代替最优观察者，比如分别只用一种特征训练小模型或线性探针（linear probe），比较学到某个准确率所需的步数。这个替代量能不能像这里一样排对顺序，需要在合成数据上先验证，因为只有在合成数据上才有标准答案可比。
- **为什么用固定学习率？** 我们要比较不同 $\rho$ 下核心特征学得多快。余弦退火会让后期的步长越来越小，同样的步数在不同阶段代表不同的"学习量"，比较起来不干净。
- **36 个模型拼成一个分组卷积，会不会互相影响？** 不会。`groups=36` 时每组卷积核只连到自己那一组通道；全局池化和线性输出也按模型分开；损失是各模型损失之和，所以每个参数的梯度只来自它所属的模型。Adam 是逐参数的，更新也互不影响。每个模型每步拿到的图片也是各自独立生成的。
- **能不能从学习速度定量预测依赖程度？** 我们试过一个粗略的估计：学亮条的有效信号大约按 $4\rho(1-\rho)$ 缩小，所以变慢的倍数约为 $1/(4\rho(1-\rho))$，$\rho = 0.97$ 时是 8.6 倍。实测在三个 $\rho$ 上都比这个估计更慢，所以我们只报告顺序，不报告定量关系。
- **如果只有 24 小时，你会砍什么？** 保留最优观察者、一个 $\sigma$ 上的完整 $\rho$ 扫描和 $\rho = 0.5$ 对照，这是"能不能预测"这个问题的最低要求。砍掉学习曲线和多种子重跑，在报告里写明差距的原因没有检验、只跑了一个种子。

<details>
<summary>验证代码（可运行）</summary>

```python
# every table in the text is exactly what the code printed (for SEED = 0)
QUOTED_BAYES = [
    "| 0.5 | 0.975 | 0.949 | 0.924 | 0.868 |",
    "| 0.85 | 0.783 | 0.413 | 0.243 | 0.101 |",
    "| 1.2 | 0.647 | 0.082 | 0.015 | 0.002 |",
]
QUOTED_MAIN = [
    "| 0.5 | 0.5 | 0.975 | 0.959, 0.956, 0.956 | 0.957 | 0.959 | 0.993 |",
    "| 0.5 | 0.8 | 0.949 | 0.887, 0.840, 0.891 | 0.873 | 0.902 | 0.988 |",
    "| 0.5 | 0.9 | 0.924 | 0.792, 0.646, 0.765 | 0.734 | 0.848 | 0.979 |",
    "| 0.5 | 0.97 | 0.868 | 0.211, 0.087, 0.176 | 0.158 | 0.738 | 0.894 |",
    "| 0.85 | 0.5 | 0.783 | 0.722, 0.717, 0.733 | 0.724 | 0.737 | 0.815 |",
    "| 0.85 | 0.8 | 0.413 | 0.156, 0.171, 0.263 | 0.197 | 0.285 | 0.774 |",
    "| 0.85 | 0.9 | 0.243 | 0.004, 0.007, 0.036 | 0.016 | 0.129 | 0.697 |",
    "| 0.85 | 0.97 | 0.101 | 0.001, 0.000, 0.000 | 0.000 | 0.027 | 0.629 |",
    "| 1.2 | 0.5 | 0.647 | 0.589, 0.567, 0.551 | 0.569 | 0.586 | 0.633 |",
    "| 1.2 | 0.8 | 0.082 | 0.002, 0.007, 0.002 | 0.003 | 0.012 | 0.614 |",
    "| 1.2 | 0.9 | 0.015 | 0.001, 0.002, 0.001 | 0.001 | 0.000 | 0.574 |",
    "| 1.2 | 0.97 | 0.002 | 0.001, 0.001, 0.001 | 0.001 | 0.000 | 0.540 |",
]
QUOTED_TIME = [
    "| 0.5 | 0.708 | 0.767 | 0.882 | 0.962 | 0.989 | 0.993 | 82 | 1.0 |",
    "| 0.8 | 0.605 | 0.670 | 0.807 | 0.913 | 0.976 | 0.988 | 150 | 1.8 |",
    "| 0.9 | 0.535 | 0.559 | 0.632 | 0.738 | 0.933 | 0.979 | 417 | 5.1 |",
    "| 0.97 | 0.532 | 0.538 | 0.551 | 0.586 | 0.701 | 0.894 | 953 | 11.6 |",
]
if SEED == 0:
    assert table_bayes == QUOTED_BAYES, table_bayes
    assert table_main == QUOTED_MAIN, table_main
    assert table_time == QUOTED_TIME, table_time
    assert (f"{spear:.2f}", f"{spear_own:.2f}", f"{err_bayes:.3f}", f"{err_own:.3f}") == ("0.88", "0.88", "0.179", "0.107")
    assert bayes_conf[0.5, 0.97] > bayes_conf[0.85, 0.8] and cnn_conf[0.5, 0.97] < cnn_conf[0.85, 0.8]   # the cross-knob pair the text calls reversed
assert (N_H, N_V) == (80, 72)                                   # 80 horizontal and 72 vertical bars
assert all((t.sum() == BAR_LEN) and (t[:TOP, :TOP].sum() == 0) for t in TEMPLATES)   # |t|^2 = 5; never under the marker
assert f"{1 / (4 * 0.97 * 0.03):.1f}" == "8.6"

# the qualitative claims below hold for SEED = 0, 1, 2 (each run trains 36 fresh CNNs)
# controls: with rho = 0.5 the marker is ignored; with rho > 0.5 aligned images are easy
for s in SIGMAS:
    assert abs(cnn_conf[s, 0.5] - cnn_align[s, 0.5]) < 0.05
    assert cnn_conf[s, 0.5] < bayes_core[s]
    assert all(cnn_align[s, r] > 0.95 for r in RHOS[1:])
# reliance on the marker grows with rho and with sigma (0.01 slack for cells stuck at the floor)
for s in SIGMAS:
    assert all(cnn_conf[s, b] <= cnn_conf[s, a] + 0.01 for a, b in zip(RHOS, RHOS[1:]))
    assert all(bayes_conf[s, b] < bayes_conf[s, a] for a, b in zip(RHOS, RHOS[1:]))
for r in RHOS:
    assert all(cnn_conf[b, r] <= cnn_conf[a, r] + 0.01 for a, b in zip(SIGMAS, SIGMAS[1:]))
    assert all(bayes_conf[b, r] < bayes_conf[a, r] for a, b in zip(SIGMAS, SIGMAS[1:]))
assert all(cnn_conf[1.2, r] < 0.02 for r in RHOS[1:])
# every single CNN leans on the marker at least as much as the ideal observer, and much more in the middle
assert all(a <= bayes_conf[c] + 0.01 for c in sel for a in cnn_each[c])
assert all(bayes_conf[c] - cnn_conf[c] > 0.1 for c in [(0.5, 0.9), (0.5, 0.97), (0.85, 0.8), (0.85, 0.9)])
# the data-only prediction gets the order roughly right; the model-aware prediction is closer but too optimistic at the extreme
assert spear > 0.85 and abs(spear_own - spear) < 0.05
assert err_own < 0.75 * err_bayes
assert own[0.5, 0.97] - cnn_conf[0.5, 0.97] > 0.3
# the core feature is still in the logit, but weaker when the marker is more predictive
for s in SIGMAS:
    assert all(cauc[s, b] <= cauc[s, a] + 0.01 for a, b in zip(RHOS, RHOS[1:]))
    assert cauc[s, 0.97] < cauc[s, 0.5] - 0.05
assert cauc[0.5, 0.97] > 0.8
# the core feature is learned later when the marker is more predictive, and later than the rough estimate says
assert all(not math.isnan(t75[r]) for r in RHOS)
assert t75[0.5] < t75[0.8] < t75[0.9] < t75[0.97]
assert t75[0.97] / t75[0.5] > 8
assert all(t75[r] / t75[0.5] > 1 / (4 * r * (1 - r)) for r in RHOS[1:])
assert traj[0.97][-1] - traj[0.97][-2] > 0.05                   # still rising at the end of training
print("all checks passed")
```

</details>

</details>
