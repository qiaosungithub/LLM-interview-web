# 72 小时开放研究：小模型什么时候能存下比维度更多的特征

## 题目

这是一道约 72 小时的开放研究 take-home。题目只有一个问题，而且故意没有说清楚。怎么定义、怎么搭实验、做到哪里为止，都由你来定。

**问题。** 一个隐藏层只有 $m$ 维的小模型，有时能表示多于 $m$ 个特征。这种现象叫叠加（superposition）。请研究：模型什么时候会这样做？是什么在控制它？

**条件。**

- 从拿到题目起，约 72 小时内交卷。一台普通笔记本的 CPU 就够用，不提供 GPU。
- 可以读公开论文。用到别人的结论要注明出处，并且要用你自己的实验检验，不能只引用。
- 不能用 AI 编程助手，除非邀请信另有说明。

**交付物。**

- 一份不超过 4 页的书面报告（write-up），附录不限。
- 能从头跑到尾的代码，能复现报告里的每一个数字。
- 交卷后有一次约 45 分钟的评审通话：你先讲 10 分钟，然后回答问题。

**评审看什么。**

- 有没有把问题收窄成一个能回答的问题；"表示了一个特征"这类说法有没有可以测量的定义。
- 跑实验之前有没有写下预测；结果和预测对不上时，你怎么处理。
- 每个结论有没有对照（control），有没有在多个随机种子上重复。
- 结论的强弱和证据是否相称；报告里有没有写清楚"还不知道什么"。
- 时间花在了哪里：砍掉了什么，为什么砍。

### Part 1 —— 搭一个最小的模型

选一个能出现叠加的最小设置。写清楚数据怎么生成、模型长什么样、损失怎么算。给"模型表示了第 $i$ 个特征"一个可以测量的定义，并说明定义里的阈值换一换，结论会不会变。

### Part 2 —— 找出控制变量

至少改变一个数据性质，测量模型表示的特征数怎样跟着变。跑之前先写下预测。要有一个按道理不会出现叠加的对照模型。

### Part 3 —— 解释机制

给出一个能定量预测"存几个特征"的解释，并用实验检验它。说清楚它在什么范围里成立，在哪里不成立。

### Part 4 —— 写报告

写出结论、对照、其他可能的解释、局限，以及"再给你一周，你会做什么"。

## 参考解答

<details>
<summary>展开参考解答</summary>

### 先定范围：72 小时怎么用

72 小时不是 72 个工作小时。扣掉睡觉和吃饭，真正能用的大约 20 到 25 小时。这道题最大的风险不是做不出来，而是题目太大：叠加可以在大语言模型里找，可以用稀疏自编码器（sparse autoencoder）找，也可以研究它对鲁棒性的影响。每一条都能做一个月。所以第一件事是把问题收窄：

> 在一个最小的玩具模型里，**哪个数据性质**决定模型存几个特征？能不能**事先算出**这个数？

收窄以后，砍掉的东西也就清楚了：

- 不碰真实语言模型和稀疏自编码器。问的是"什么在控制叠加"，玩具模型里变量最干净。
- 主实验只扫一个变量：特征的稀疏度（sparsity）。第二个变量（特征重要性衰减得多快）只做探索，结果不干净就放进"局限"。
- 不做花哨的可视化。两张表、一两张图就够。

| 时间 | 做什么 | 产出 |
| --- | --- | --- |
| 第 1 天上午 | 读题，写下问题定义和预测；搭最小模型，在一个稀疏度上确认能训练 | 一页"定义与预测"笔记；能跑的训练代码 |
| 第 1 天下午到晚上 | 扫稀疏度，加线性对照，3 次重复；检查结论是否依赖阈值 | 主结果表 |
| 第 2 天 | 推导成本收益模型，直接构造它的解，和训练结果比；做训练步数的对照；换 5 个种子重跑 | 机制表、对照表 |
| 第 3 天上午 | 探索第二个变量；不干净就停 | 附录或"局限"里的一段 |
| 第 3 天下午 | 写报告；从头重跑一遍代码，逐个核对报告里的数字 | 报告 + 代码 |

### Part 1 —— 最小模型和测量方法

**设置。** 沿用 Elhage 等人 2022 年的 Toy Models of Superposition 里第一组基本实验的设置（他们也用 $n = 20$、$m = 5$），只是重要性衰减从 $0.7^i$ 改成了 $0.8^i$：

- $n = 20$ 个特征。每个特征以概率 $p = 1 - S$ 出现，出现时取值服从 $U[0, 1)$，否则为 0。$S$ 就是稀疏度。特征之间相互独立。
- 模型：$h = Wx \in \mathbb{R}^5$，$\hat{x} = \mathrm{ReLU}(W^\top h + b)$。隐藏层 $m = 5$ 维，唯一的非线性是输出端的 ReLU。
- 损失：$L = \mathbb{E}\sum_i I_i (x_i - \hat{x}_i)^2$，重要性（importance）$I_i = 0.8^{i}$（$i$ 从 0 开始）。重要性递减，于是"最该存哪 5 个"没有歧义。
- 对照：去掉 ReLU 的线性模型 $\hat{x} = W^\top W x + b$。线性模型最多做到主成分分析（PCA）的效果，按道理只会存 5 个特征。

**什么叫"存了一个特征"。** 主定义：$\|W_i\|^2 > 0.5$（$W_i$ 是第 $i$ 列）。备用定义：这个特征的重构 $R^2 > 0.5$。两个定义要互相核对。

**基线（baseline）。** 只存最重要的 5 个、其余 15 个一律输出均值，这是 5 维里不叠加能做到的最好结果，损失为 $\sum_{i \ge 5} I_i \operatorname{Var}(x_i)$。损失除以这个基线，小于 1 说明多存的特征真的有用，而不只是范数变大了。

**实现。** 2 种模型 × 9 个稀疏度 × 3 次重复，共 54 个模型。损失是各模型损失之和，模型之间没有任何耦合，所以可以放进一个张量一次训练完。Adam，学习率 0.01，余弦退火，4000 步，每步 256 个新样本。评估用另外 2 万个样本。只用一个线程，结果逐位可复现。

```python
import time

import numpy as np
import torch

torch.set_num_threads(1)          # one thread: the run is bit-for-bit reproducible
SEED = 0


def sample(p, batch, n, g):
    """x[k, j, i] is 0 with probability 1 - p[k], otherwise uniform on [0, 1). p: [K, 1, 1]."""
    u = torch.rand(p.shape[0], batch, n, generator=g)
    return torch.where(u < p, u / p, torch.zeros(()))      # given u < p, u / p is uniform on [0, 1)


def forward(W, b, x, is_relu):
    """x_hat = ReLU(W^T W x + b), without the ReLU where is_relu is False. W: [K, m, n], b: [K, 1, n], x: [K, batch, n]."""
    out = (x @ W.transpose(1, 2)) @ W + b
    return torch.where(is_relu, torch.relu(out), out)


def train(p, is_relu, importance, n_hid, steps, batch, lr, seed):
    """Train K independent models at once. p: K floats, is_relu: K bools, importance: [n]."""
    g = torch.Generator().manual_seed(seed)
    K, n = len(p), len(importance)
    imp = torch.tensor(importance, dtype=torch.float32)
    p_t = torch.tensor(p, dtype=torch.float32).view(K, 1, 1)
    relu_t = torch.tensor(is_relu).view(K, 1, 1)
    W = (torch.randn(K, n_hid, n, generator=g) / n_hid ** 0.5).requires_grad_()
    b = torch.zeros(K, 1, n, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x = sample(p_t, batch, n, g)
        loss = (imp * (x - forward(W, b, x, relu_t)) ** 2).sum(-1).mean(-1).sum()   # a sum over models: no interaction
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return W.detach(), b.detach()


def per_feature_mse(W, b, x, is_relu):
    """Mean squared error of every feature on the sample x, shape [K, n]."""
    relu_t = torch.tensor(is_relu).view(-1, 1, 1)
    with torch.no_grad():
        return ((x - forward(W, b, x, relu_t)) ** 2).mean(1).numpy()


N_FEAT, N_HID, DECAY = 20, 5, 0.8
SPARSITY = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
N_REP, STEPS, BATCH, LR = 3, 4000, 256, 1e-2
importance = DECAY ** np.arange(N_FEAT)

cells = [(kind, s, rep) for kind in ("relu", "linear") for s in SPARSITY for rep in range(N_REP)]
p_all = [1 - s for _, s, _ in cells]
relu_all = [kind == "relu" for kind, _, _ in cells]
t0 = time.time()
W, b = train(p_all, relu_all, importance, N_HID, STEPS, BATCH, LR, SEED)
print(f"trained {len(cells)} models in {time.time() - t0:.1f} s")

g_eval = torch.Generator().manual_seed(1000 + SEED)
x_eval = sample(torch.tensor(p_all, dtype=torch.float32).view(-1, 1, 1), 20000, N_FEAT, g_eval)   # fresh data per model
mse = per_feature_mse(W, b, x_eval, relu_all)
var = x_eval.var(1).numpy()
loss = (importance * mse).sum(1)
baseline = (importance[N_HID:] * var[:, N_HID:]).sum(1)     # store the top 5, predict the other 15 by their mean
norm2 = (W ** 2).sum(1).numpy()                              # ||W_i||^2 for every model and feature
count_norm = (norm2 > 0.5).sum(1)
count_r2 = (1 - mse / var > 0.5).sum(1)


def idx(kind, s):
    return [i for i, c in enumerate(cells) if c[0] == kind and c[1] == s]


def f2(v):
    return f"{round(float(v), 2) + 0.0:.2f}"                 # + 0.0 turns -0.00 into 0.00


table1 = []
for s in SPARSITY:
    r, l = idx("relu", s), idx("linear", s)
    table1.append(f"| {s:.2f} | {', '.join(str(count_norm[i]) for i in r)} | {f2(np.mean(count_norm[r]))} | "
                  f"{', '.join(str(count_norm[i]) for i in l)} | {f2(np.mean(loss[r] / baseline[r]))} | "
                  f"{f2(np.mean(loss[l] / baseline[l]))} |")
print("\n".join(table1))
```

### Part 2 —— 稀疏度控制叠加

跑之前写下的预测是：特征越稀疏，两个特征同时出现的机会越少，共用方向的代价越小，所以 ReLU 模型会随稀疏度增加而存下更多特征；线性模型不管多稀疏都只存 5 个。

结果（`SEED = 0`，每格 3 个独立训练的模型）：

| 稀疏度 $S$ | ReLU：存下的特征数（3 次） | ReLU 平均 | 线性：特征数 | ReLU 损失 / 基线 | 线性 损失 / 基线 |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 5, 5, 5 | 5.00 | 5, 5, 5 | 1.00 | 1.00 |
| 0.30 | 5, 5, 5 | 5.00 | 5, 5, 5 | 1.00 | 1.00 |
| 0.50 | 7, 8, 7 | 7.33 | 5, 5, 5 | 1.03 | 1.00 |
| 0.60 | 8, 9, 8 | 8.33 | 5, 5, 5 | 0.97 | 1.00 |
| 0.70 | 10, 9, 10 | 9.67 | 5, 5, 5 | 0.82 | 1.00 |
| 0.80 | 10, 10, 10 | 10.00 | 5, 5, 5 | 0.65 | 1.00 |
| 0.90 | 10, 11, 10 | 10.33 | 5, 5, 5 | 0.46 | 1.00 |
| 0.95 | 11, 12, 11 | 11.33 | 5, 5, 5 | 0.37 | 1.00 |
| 0.98 | 15, 14, 14 | 14.33 | 5, 5, 5 | 0.29 | 1.00 |

怎么读：

- **稠密时没有叠加。** $S \le 0.3$ 时，ReLU 模型和线性模型一样，正好存 5 个，损失等于基线。
- **越稀疏，存得越多。** 从 $S = 0.5$ 开始，ReLU 模型存下的特征超过 5 个，到 $S = 0.98$ 时平均 14.33 个。换种子时相邻两档的平均值偶尔会倒过来，但差距不到 1 个。$S \ge 0.8$ 时每个模型都至少存 9 个。线性对照在所有稀疏度下都正好是 5 个，损失和基线相差不到 2%。最重要的 5 个特征，每个 ReLU 模型都存了。
- **多存的特征是有用的。** $S = 0.98$ 时，ReLU 模型的损失只有基线的 0.29。但在 $S = 0.5$ 到 0.6，损失和基线差不多：多存了两三个特征，几乎没换来好处。这一点 Part 3 会再解释。
- **结论不依赖阈值。** 用 $R^2 > 0.5$ 数出来的特征数，和用范数数出来的，在全部 54 个模型里最多差 1 个。每个 ReLU 模型最多只有 2 个特征的 $\|W_i\|^2$ 落在 0.2 到 0.8 之间，其余要么接近 0，要么接近 1。所以阈值取 0.5 还是 0.3、0.7，数出来几乎一样。

### Part 3 —— 一个能算出特征数的成本收益模型

光说"越稀疏越多"还不够。评审会问：能不能事先算出存几个？下面是一个只用纸笔就能推的模型。

**反向配对（antipodal pair）。** 让两个特征 $a$、$c$ 共用一个隐藏方向，符号相反：$W_a = e$，$W_c = -e$，偏置为 0。

- 只有 $a$ 出现时，$\hat{x}_a = \mathrm{ReLU}(x_a) = x_a$，$\hat{x}_c = \mathrm{ReLU}(-x_a) = 0$，两个都对。ReLU 把负的干扰截掉了。
- 两个同时出现（概率 $p^2$）时，$\hat{x}_a = \mathrm{ReLU}(x_a - x_c)$，两个特征的误差都是 $\min(x_a, x_c)^2$。两个独立 $U[0,1)$ 的最小值，平方的期望是 $1/6$。

所以一对的代价是 $p^2 (I_a + I_c)/6$。不存 $c$、只输出它的均值，代价是 $I_c \operatorname{Var}(x_c) = I_c\,(p/3 - p^2/4)$。配对更划算，当且仅当

$$\frac{p^2 (I_a + I_c)}{6} < I_c\left(\frac{p}{3} - \frac{p^2}{4}\right) \iff p < p^* = \frac{4r}{2 + 5r},\qquad r = \frac{I_c}{I_a}.$$

5 维里最多放 5 对，也就是最多 10 个特征。第 $k$ 对应该把排第 $6-k$ 的特征和排第 $5+k$ 的配在一起，$r = 0.8^{2k-1}$。对每个稀疏度取总代价最小的 $k$，预测的特征数就是 $5 + k$。

```python
def var_x(p):
    return p / 3 - p * p / 4                                 # variance of a feature that is U[0, 1) with probability p


def pair_loss(p, k):
    """Expected loss: the top 5 - k features alone, the next 2k in antipodal pairs, the rest predicted by their mean."""
    return p * p / 6 * importance[N_HID - k: N_HID + k].sum() + var_x(p) * importance[N_HID + k:].sum()


def best_k(p):
    return int(np.argmin([pair_loss(p, k) for k in range(N_HID + 1)]))


def pair_solution(p, k):
    """W and b of that solution: the feature ranked 5 - j shares a direction, with opposite sign, with the one ranked 6 + j."""
    Wc, bc = torch.zeros(N_HID, N_FEAT), torch.zeros(N_FEAT)
    Wc[torch.arange(N_HID), torch.arange(N_HID)] = 1.0
    for j in range(k):
        Wc[N_HID - 1 - j, N_HID + j] = -1.0
    bc[N_HID + k:] = p / 2                                   # a dropped feature is predicted by its mean
    return Wc, bc


ratios = [DECAY ** (2 * k - 1) for k in range(1, N_HID + 1)]
p_star = [4 * r / (2 + 5 * r) for r in ratios]
thresholds = [f"| {k} | 第 {N_HID + 1 - k} 和第 {N_HID + k} | {r:.3f} | {ps:.3f} | {1 - ps:.3f} |"
              for k, r, ps in zip(range(1, N_HID + 1), ratios, p_star)]
print("\n".join(thresholds))

# 1) the algebra: the constructed solution, measured on a large sample, has the loss the formula says
g_big = torch.Generator().manual_seed(2000 + SEED)
formula_ratio = {}
for s in SPARSITY:
    p = 1 - s
    Wc, bc = pair_solution(p, best_k(p))
    xb = sample(torch.tensor([[[p]]]), 200_000, N_FEAT, g_big)
    formula_ratio[s] = float((importance * per_feature_mse(Wc[None], bc.view(1, 1, -1), xb, [True])[0]).sum()
                             / pair_loss(p, best_k(p)))

# 2) the trained ReLU models against the constructed solution, on the same evaluation sample
relu_idx = [i for i, c in enumerate(cells) if c[0] == "relu"]
built = [pair_solution(p_all[i], best_k(p_all[i])) for i in relu_idx]
Wc_all = torch.stack([w for w, _ in built])
bc_all = torch.stack([v for _, v in built]).unsqueeze(1)
loss_c = (importance * per_feature_mse(Wc_all, bc_all, x_eval[relu_idx], [True] * len(relu_idx))).sum(1)
ratio = dict(zip(relu_idx, loss[relu_idx] / loss_c))


def n_pairs(i):
    """Pairs of stored features whose directions point almost exactly opposite (cosine < -0.9)."""
    keep = norm2[i] > 0.5
    Wn = W[i][:, keep] / W[i][:, keep].norm(dim=0)
    return int((np.triu((Wn.T @ Wn).numpy(), 1) < -0.9).sum())


pairs = {i: n_pairs(i) for i in relu_idx}
bias_kept = {i: float(b[i, 0][norm2[i] > 0.5].mean()) for i in relu_idx}
table2 = []
for s in SPARSITY:
    r = idx("relu", s)
    table2.append(f"| {s:.2f} | {N_HID + best_k(1 - s)} | {f2(np.mean(count_norm[r]))} | {f2(np.mean([ratio[i] for i in r]))} | "
                  f"{', '.join(str(pairs[i]) for i in r)} | {f2(np.mean([bias_kept[i] for i in r]))} |")
print("\n".join(table2))
```

每一对开始划算的位置：

| 第 $k$ 对 | 配对的特征 | $r = I_c / I_a$ | $p^*$ | 稀疏度阈值 $S^* = 1 - p^*$ |
| --- | --- | --- | --- | --- |
| 1 | 第 5 和第 6 | 0.800 | 0.533 | 0.467 |
| 2 | 第 4 和第 7 | 0.512 | 0.449 | 0.551 |
| 3 | 第 3 和第 8 | 0.328 | 0.360 | 0.640 |
| 4 | 第 2 和第 9 | 0.210 | 0.275 | 0.725 |
| 5 | 第 1 和第 10 | 0.134 | 0.201 | 0.799 |

先检查推导本身：把这个配对解直接构造出来，在 20 万个样本上测它的损失，和公式相差不到 3%。然后拿训练出来的模型和它比：

| 稀疏度 $S$ | 配对模型预测 | 训练所得（平均） | 训练损失 / 构造解损失 | 反向对数（3 次） | 已存特征的平均偏置 |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 5 | 5.00 | 1.00 | 0, 0, 0 | 0.00 |
| 0.30 | 5 | 5.00 | 1.00 | 0, 0, 0 | 0.00 |
| 0.50 | 6 | 7.33 | 1.05 | 2, 3, 2 | 0.06 |
| 0.60 | 7 | 8.33 | 1.07 | 3, 4, 3 | 0.04 |
| 0.70 | 8 | 9.67 | 1.05 | 5, 4, 5 | 0.02 |
| 0.80 | 10 | 10.00 | 1.02 | 5, 5, 5 | 0.01 |
| 0.90 | 10 | 10.33 | 1.00 | 5, 3, 5 | -0.03 |
| 0.95 | 10 | 11.33 | 0.98 | 5, 2, 3 | -0.11 |
| 0.98 | 10 | 14.33 | 0.86 | 0, 1, 1 | -0.20 |

"反向对"指两个已存特征方向的余弦小于 −0.9。表里能读出三段：

1. **两头吻合：$S \le 0.3$ 和 $S = 0.8$、0.9。** 平均特征数和预测相差不到 1 个，损失和构造解相差在 10% 以内。$S = 0.8$ 时每个模型都有至少 4 对反向对，偏置接近 0，正是配对解的样子。注意 $S = 0.8$ 恰好压在第 5 对的阈值 0.799 上，9 个还是 10 个几乎打平。
2. **过渡区不吻合：$S = 0.5$ 到 0.7。** 训练出的模型大致比预测多存 0 到 2 个特征，损失却没有因此更低。这个种子在这三档都比构造解高（$S = 0.6$ 时 1.07）；换种子以后，$S = 0.5$ 和 0.7 有时和构造解持平，但 $S = 0.6$ 在 5 个种子里都更高。也就是说，梯度下降停在了一个存得更多、却不更好的解上。配对模型给出的是"最优解大概长什么样"，训练不一定找得到它。
3. **高稀疏区超出模型：$S \ge 0.95$。** 训练出的模型平均存了 11 个以上，超过了配对模型的上限 10 个。到 $S = 0.98$，每个模型都超过 10 个（平均 14.33）；损失比构造解低（0.86）；反向对反而变少（最多 2 对）；偏置变成负的（-0.20）。这说明这里有比"两两反向"更好的排法：多个特征以非反向的角度共用一个子空间，彼此会漏进一点正的干扰，负偏置把这点干扰压到 ReLU 的零点以下。这和 Elhage 等人描述的五边形之类的几何结构一致，但我们没有直接检查几何形状，只能说"一致"。

### Part 4 —— 怎么读这些结果

**结果说明了什么。** 在这个玩具设置里，稀疏度控制叠加：ReLU 模型存下的特征数随稀疏度总体上升（相邻两档偶尔回落不到 1 个），从 5 个到 14 个左右；线性对照始终是 5 个。在稠密一端（$S \le 0.3$）和 $S = 0.8$ 到 0.9，一个两两反向配对的成本收益模型能预测特征数，误差在 1 个以内。它预测第一个多余特征在 $S > 0.467$ 时出现，和观察到的起点（0.3 到 0.5 之间）一致，但我们的网格太粗，没法更精确地检验。

**结果没有说明什么。**

- 没有说明真实网络里的情况。这里只有输出端一个 ReLU，特征相互独立，取值是均匀分布。
- 没有说明过渡区为什么训练不到最优解。"局部最优"只是最可能的解释，没有被证明。
- 没有说明重要性的作用。我们试过改变重要性衰减的速度：衰减很快或很慢时，配对模型的预测明显偏离训练结果，三天内没查清原因，所以没写进结论。
- 只扫了一个规模（$n = 20$，$m = 5$）。阈值的位置会随 $n$、$m$ 和重要性变化，这里的具体数字不能外推。

**其他解释和对应的对照。**

| 另一种解释 | 对照 | 结果 |
| --- | --- | --- |
| 特征数变多只是因为 ReLU 模型训练得乱，范数乱长 | 同样训练的线性模型；损失除以基线 | 线性模型始终 5 个、损失等于基线；ReLU 多存的特征让损失降到基线的 0.29 |
| 阈值 0.5 选得巧 | $R^2$ 定义；看范数分布 | 两种定义最多差 1 个；每个模型最多 2 个特征的范数落在 0.2 到 0.8 之间 |
| 没训够，换个步数结论就变 | 只训一半步数重来一遍 | 每个模型的特征数最多差 2 个，随稀疏度上升的趋势不变 |
| 只是这个种子运气好 | 每格 3 次重复；把 `SEED` 改成 1 到 4 重跑整页 | 下面验证代码里的所有定性检查在 `SEED = 0` 到 4 都通过 |
| 配对公式推错了 | 直接构造配对解，在 20 万个样本上测损失 | 和公式相差不到 3% |

```python
t0 = time.time()
W_half, _ = train([p_all[i] for i in relu_idx], [True] * len(relu_idx), importance, N_HID, STEPS // 2, BATCH, LR, SEED)
print(f"half-step control: {time.time() - t0:.1f} s")
count_half = dict(zip(relu_idx, ((W_half ** 2).sum(1).numpy() > 0.5).sum(1)))
half_diff = [int(count_half[i] - count_norm[i]) for i in relu_idx]
print("half - full:", half_diff, "max |diff| =", max(abs(d) for d in half_diff))
print("formula check:", {s: round(v, 3) for s, v in formula_ratio.items()})
```

**局限，以及再给一周会做什么。**

- **过渡区的优化问题。** 从构造好的配对解出发继续训练：如果它停在原地，说明它是训练损失的一个极小值，只是随机初始化找不到它。再加多次重启，取最好的一次，看特征数是否回到预测值。
- **高稀疏区的几何。** 算每个特征的"维度占用" $D_i = \|W_i\|^2 / \sum_j (\hat{W}_i \cdot W_j)^2$（$\hat{W}_i$ 是单位化的 $W_i$），看是否出现 2/5 这类分数，直接检查五边形结构。
- **重要性这个变量。** 查清衰减很快或很慢时配对模型为什么失效，可能需要把"三个或更多特征共用方向"加进成本模型。
- **更接近真实网络的设置。** 隐藏层也加 ReLU；特征之间有相关性；更大的 $n$ 和 $m$，看阈值怎么缩放。
- **学习过程。** 记录每个特征在训练中什么时候"进入"表示，看是否按重要性顺序。

### 报告提纲

1. **问题和定义**（半页）：收窄后的问题；"存了一个特征"的两个定义；基线。
2. **设置**（半页）：数据、模型、损失、对照、超参数、种子。
3. **主结果**（一页）：稀疏度和特征数的表，配一张"平均特征数对稀疏度"的图，线性对照画在同一张图上。
4. **机制**（一页）：配对模型的推导、阈值表、和训练结果的对比表。
5. **对照和其他解释**（半页）：上面那张五行的表。
6. **不成立的地方和局限**（半页）：过渡区、高稀疏区、重要性、规模。
7. **下一步**（几行）。
8. **附录**：完整超参数、每个种子的结果、复现命令。

讲 10 分钟时，只讲第 3、4、6 节：一张主图，一个公式，两处不成立。

### 追问

- **为什么要给特征不同的重要性？全都一样会怎样？** 重要性递减让"该存哪 5 个"没有歧义，也让每一对的阈值错开，特征数一个一个往上加，方便和预测对比。如果全都一样，$r = 1$，五对的阈值都是 $p^* = 4/7$，配对模型预测特征数会从 5 个直接跳到 10 个。这个设置我们没有跑，所以只能说是预测。
- **为什么线性模型永远只存 5 个？** 对线性模型，叠加带来的干扰没法去掉：两个特征共用方向，一个出现就会在另一个的输出上留下等量的误差。最优的线性重构是投影到加权方差最大的 5 维子空间上，这里就是最重要的 5 个特征。ReLU 的作用是把负的干扰截成 0，反向配对正是利用了这一点。
- **过渡区里训练解比构造解差，你凭什么说是局部最优，而不是优化器设置的问题？** 现在还不能下结论。半数步数的对照只说明继续用同样的优化器多训，特征数几乎不变。要区分两种可能：从构造解出发训练，看它是否稳定；换学习率、多次重启，看能不能找到更好的解。两个都做完，才能说是哪一种。
- **$S = 0.98$ 时偏置为什么是负的？** 反向配对时，干扰是负的，ReLU 自己就能截掉，不需要偏置。三个以上特征以非反向的角度共用子空间时，一个特征出现会给别的特征带来正的干扰。负偏置让这点干扰落到 0 以下，代价是特征真的出现时，小的取值也会被截掉。特征越稀疏，挤进同一子空间的特征越多；和这个特征自己真正出现的次数相比，干扰出现得越多，这笔交易越划算。
- **这个结论能推广到大模型吗？** 不能直接推广。它说明的是：特征稀疏、又有能截掉干扰的非线性时，叠加是划算的。真实模型里特征有多稀疏、有多少，我们不知道。这正是要用字典学习、稀疏自编码器去估计的东西。玩具模型给出的是一个可以检验的假设，不是结论。
- **如果只有 24 小时，你会砍什么？** 保留稀疏度扫描、线性对照和阈值检查，这是主结论的最低要求。配对模型保留推导和阈值表，但不做构造解的对比。半数步数对照和多种子重跑砍掉，在报告里写明"只跑了一个种子"。

<details>
<summary>验证代码（可运行）</summary>

```python
# every table in the text is exactly what the code printed
QUOTED_TABLE1 = [
    "| 0.00 | 5, 5, 5 | 5.00 | 5, 5, 5 | 1.00 | 1.00 |",
    "| 0.30 | 5, 5, 5 | 5.00 | 5, 5, 5 | 1.00 | 1.00 |",
    "| 0.50 | 7, 8, 7 | 7.33 | 5, 5, 5 | 1.03 | 1.00 |",
    "| 0.60 | 8, 9, 8 | 8.33 | 5, 5, 5 | 0.97 | 1.00 |",
    "| 0.70 | 10, 9, 10 | 9.67 | 5, 5, 5 | 0.82 | 1.00 |",
    "| 0.80 | 10, 10, 10 | 10.00 | 5, 5, 5 | 0.65 | 1.00 |",
    "| 0.90 | 10, 11, 10 | 10.33 | 5, 5, 5 | 0.46 | 1.00 |",
    "| 0.95 | 11, 12, 11 | 11.33 | 5, 5, 5 | 0.37 | 1.00 |",
    "| 0.98 | 15, 14, 14 | 14.33 | 5, 5, 5 | 0.29 | 1.00 |",
]
QUOTED_THRESHOLDS = [
    "| 1 | 第 5 和第 6 | 0.800 | 0.533 | 0.467 |",
    "| 2 | 第 4 和第 7 | 0.512 | 0.449 | 0.551 |",
    "| 3 | 第 3 和第 8 | 0.328 | 0.360 | 0.640 |",
    "| 4 | 第 2 和第 9 | 0.210 | 0.275 | 0.725 |",
    "| 5 | 第 1 和第 10 | 0.134 | 0.201 | 0.799 |",
]
QUOTED_TABLE2 = [
    "| 0.00 | 5 | 5.00 | 1.00 | 0, 0, 0 | 0.00 |",
    "| 0.30 | 5 | 5.00 | 1.00 | 0, 0, 0 | 0.00 |",
    "| 0.50 | 6 | 7.33 | 1.05 | 2, 3, 2 | 0.06 |",
    "| 0.60 | 7 | 8.33 | 1.07 | 3, 4, 3 | 0.04 |",
    "| 0.70 | 8 | 9.67 | 1.05 | 5, 4, 5 | 0.02 |",
    "| 0.80 | 10 | 10.00 | 1.02 | 5, 5, 5 | 0.01 |",
    "| 0.90 | 10 | 10.33 | 1.00 | 5, 3, 5 | -0.03 |",
    "| 0.95 | 10 | 11.33 | 0.98 | 5, 2, 3 | -0.11 |",
    "| 0.98 | 10 | 14.33 | 0.86 | 0, 1, 1 | -0.20 |",
]
if SEED == 0:                                                # the quoted numbers are for SEED = 0
    assert table1 == QUOTED_TABLE1, table1
    assert table2 == QUOTED_TABLE2, table2
assert thresholds == QUOTED_THRESHOLDS                       # pure arithmetic, no seed involved
assert abs(4 / 7 - 4 * 1 / (2 + 5 * 1)) < 1e-12              # the equal-importance follow-up: r = 1 gives p* = 4/7

# the qualitative claims below hold for SEED = 0, 1, 2, 3, 4 (each run trains 54 fresh models)
mean_count = {s: float(np.mean(count_norm[idx("relu", s)])) for s in SPARSITY}
mean_half = {s: float(np.mean([count_half[i] for i in idx("relu", s)])) for s in SPARSITY}
pred = {s: N_HID + best_k(1 - s) for s in SPARSITY}
assert [pred[s] for s in SPARSITY] == [5, 5, 6, 7, 8, 10, 10, 10, 10]

# controls: the linear model never goes beyond 5 and sits on the baseline; the dense ReLU model stores exactly 5
for s in SPARSITY:
    assert all(count_norm[i] == 5 for i in idx("linear", s))
    assert all(abs(loss[i] / baseline[i] - 1) < 0.02 for i in idx("linear", s))
for s in (0.0, 0.3):
    assert all(count_norm[i] == 5 and loss[i] / baseline[i] < 1.05 for i in idx("relu", s))
assert all((norm2[i][:N_HID] > 0.5).all() for i in relu_idx)                  # the top 5 are always stored
# counts grow with sparsity
key = [0.0, 0.5, 0.8, 0.95, 0.98]
assert all(mean_count[a] < mean_count[c] for a, c in zip(key, key[1:]))
assert all(mean_count[c] > mean_count[a] - 1 for a, c in zip(SPARSITY, SPARSITY[1:]))
assert all(count_norm[i] >= 9 for s in SPARSITY if s >= 0.8 for i in idx("relu", s))
assert all(count_norm[i] >= 12 for i in idx("relu", 0.98))                    # beyond the pair model's ceiling of 10
# the threshold does not matter: the R^2 criterion agrees, and few norms sit in the middle
assert all(abs(int(count_r2[i]) - int(count_norm[i])) <= 1 for i in range(len(cells)))
assert all(((norm2[i] > 0.2) & (norm2[i] < 0.8)).sum() <= 2 for i in relu_idx)
# the extra features are useful: the loss falls well below the best 5-feature solution
lb = {s: float(np.mean(loss[idx("relu", s)] / baseline[idx("relu", s)])) for s in SPARSITY}
assert all(lb[a] > lb[c] for a, c in zip(SPARSITY[3:], SPARSITY[4:]))
assert lb[0.98] < 0.35
assert all(abs(lb[s] - 1) < 0.1 for s in (0.5, 0.6))                         # little gain at the start
# the cost-benefit model
assert all(abs(mean_count[s] - pred[s]) <= 1 for s in (0.0, 0.3, 0.8, 0.9))
assert all(-0.5 <= mean_count[s] - pred[s] <= 2.5 for s in (0.5, 0.6, 0.7))
assert mean_count[0.95] - pred[0.95] >= 1 and mean_count[0.98] - pred[0.98] >= 3
assert all(abs(formula_ratio[s] - 1) < 0.03 for s in SPARSITY)
assert all(0.9 <= ratio[i] <= 1.1 for s in (0.0, 0.3, 0.8, 0.9) for i in idx("relu", s))
assert np.mean([ratio[i] for i in idx("relu", 0.6)]) > 1.0
assert all(np.mean([ratio[i] for i in idx("relu", s)]) > 0.97 for s in (0.5, 0.7))   # never clearly better
assert mean_count[0.95] >= 11
assert all(ratio[i] < 0.97 for i in idx("relu", 0.98))
assert all(pairs[i] == 0 for s in (0.0, 0.3) for i in idx("relu", s))
assert all(pairs[i] >= 4 for i in idx("relu", 0.8))
assert all(pairs[i] <= 2 for i in idx("relu", 0.98))
assert all(abs(bias_kept[i]) < 0.1 for s in SPARSITY if s <= 0.9 for i in idx("relu", s))
assert all(bias_kept[i] < -0.12 for i in idx("relu", 0.98))
# half the training steps: the same picture
assert all(abs(d) <= 2 for d in half_diff)
assert all(mean_half[a] < mean_half[c] for a, c in zip(key, key[1:]))
print("all checks passed")
```

</details>

</details>
