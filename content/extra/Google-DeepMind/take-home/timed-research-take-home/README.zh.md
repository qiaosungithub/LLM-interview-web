# 限时 3 小时的研究小课题：上下文加噪能不能稳住自回归滚动预测

## 题目

这是一道限时的 take-home（带回家做的作业）。打开题目后你有连续 3 小时，要交一份一条命令就能跑完的代码和一页报告。
研究问题、数据和基线都已经给定，新代码不到 60 行。考的是：三小时内能不能把一个小实验做干净，把结论说准。

**研究问题。** 下一帧预测模型（next-frame predictor）训练时看到的上下文都是真实帧，这叫 teacher forcing。
用它生成长序列时，要把自己的预测当作下一步的输入，一步步滚下去，这叫自回归滚动（autoregressive rollout）。
滚动时的输入带着模型自己的误差，训练时从没见过，误差就会越滚越大。一个常见的补救办法是：训练时给上下文帧加一点
高斯噪声，目标帧保持干净。请在下面这个小数据上回答：

> 给上下文加噪能不能降低长程滚动误差？从第几步开始有用？代价是什么？噪声加多了会怎样？

**数据和基线（直接给你，不要改）。** 数据是一维“视频”：每帧 16 个像素，画面里有一个高斯形状的亮斑，以恒定速度移动，
碰到边界反弹。基线是一个小 MLP，看最近 3 帧，预测下一帧。

```py
import numpy as np
import torch

torch.set_num_threads(1)      # same numbers on every run, and polite on a shared machine

W = 16                        # pixels per frame
K = 3                         # context length: the model sees the last K frames
LO, HI = 2.0, 13.0            # the blob centre moves between these two walls
BLOB = 1.2                    # blob width in pixels (std of the Gaussian bump)
HORIZON = 30                  # longest rollout we evaluate


def make_sequences(n, length, seed):
    """n clips of a 1-D "video": one Gaussian bump moving at constant speed, bouncing off the walls.

    Returns the frames, float32 (n, length, W) with values in [0, 1], and the true centres, (n, length).
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(LO, HI, n)
    vel = rng.uniform(0.3, 1.2, n) * rng.choice([-1.0, 1.0], n)     # pixels per frame
    grid = np.arange(W)
    frames = np.zeros((n, length, W), dtype=np.float32)
    centres = np.zeros((n, length))
    for t in range(length):
        frames[:, t] = np.exp(-0.5 * ((grid[None, :] - pos[:, None]) / BLOB) ** 2)
        centres[:, t] = pos
        pos = pos + vel
        hit = pos > HI
        pos[hit], vel[hit] = 2 * HI - pos[hit], -vel[hit]
        hit = pos < LO
        pos[hit], vel[hit] = 2 * LO - pos[hit], -vel[hit]
    return frames, centres


class Predictor(torch.nn.Module):
    """Next frame = last frame + MLP(all K context frames)."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(K * W, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, W))

    def forward(self, context):                      # context: (B, K, W)
        return context[:, -1] + self.net(context.flatten(1))


def windows(frames):
    """Every (context, target) pair in the clips: arrays of shape (N, K, W) and (N, W)."""
    n, length, _ = frames.shape
    context = np.stack([frames[:, i:length - K + i] for i in range(K)], axis=2)
    target = frames[:, K:]
    return context.reshape(-1, K, W), target.reshape(-1, W)


def train_baseline(frames, seed, steps=2000, batch=128, lr=1e-3):
    """Teacher forcing: the context always consists of true frames."""
    torch.manual_seed(seed)                              # fixes the initial weights
    model = Predictor()
    context, target = (torch.from_numpy(a) for a in windows(frames))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = torch.Generator().manual_seed(seed)          # fixes the batch order
    for _ in range(steps):
        idx = torch.randint(0, context.shape[0], (batch,), generator=order)
        loss = ((model(context[idx]) - target[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def one_step_mse(model, frames):
    """Mean squared error of one-step predictions from true contexts, over all windows."""
    context, target = (torch.from_numpy(a) for a in windows(frames))
    return float(((model(context) - target) ** 2).mean())


def load_data(seed):
    """The train / test split for one seed. Test clips are K + HORIZON frames long."""
    train_frames, _ = make_sequences(256, 24, 1000 + seed)
    test_frames, test_centres = make_sequences(500, K + HORIZON, 2000 + seed)
    return train_frames, test_frames, test_centres
```

先跑一下，确认环境没问题：

```py
train_frames, test_frames, test_centres = load_data(seed=0)
model = train_baseline(train_frames, seed=0)
print(f"{one_step_mse(model, test_frames):.1e}")                    # 2.5e-05
context, target = windows(test_frames)
print(round(float(((context[:, -1] - target) ** 2).mean()), 3))    # 0.025 -- what "copy the last frame" gets
```

基线的单步误差比“直接复制上一帧”小约一千倍，看起来很好。题目问的是：滚动 30 步之后还好不好。

**计时和规则。**

- 关于这类题，能查到的只有一句话：一位研究科学家候选人说，约十轮面试里有一轮是“带计时器的 take-home，3 小时”。
  平台、题目内容、能不能暂停，都没有记录。下面几条是为练习定的假设，取的是最常见也最严格的做法。
- 打开题目就开始计时，连续 3 小时，不能暂停。到点就交当时已有的东西。
- 交两个文件：`solution.py`（运行 `python solution.py` 能从头跑完，并打印报告里出现的每一个数字）和 `report.md`（不超过一页）。
- 只用 CPU，只用 PyTorch 和 NumPy。全部实验从头跑一遍不超过 10 分钟。
- 不用 AI 工具。DeepMind 的官方面试指南写明：除非另有通知，面试任务中不要使用任何 AI 工具。能不能查文档以招聘方的
  通知为准；练习时建议只查 PyTorch 和 NumPy 的官方文档。
- 你的数字不必和参考解答逐位相同，因为噪声的随机数流不同就会有差别。应当一致的是 Part 1 的数字（那里没有你自己引入的
  随机性；但换了 CPU 或 PyTorch 版本，浮点舍入不同，起始代码打印的单步 MSE 和第 20、30 步的数可能在第二位有效数字上
  略有出入）和 Part 2 主要结论的方向。哪些结论对随机数敏感，本身就是报告该写的内容。

### Part 1 —— 滚动评估和参照线

实现滚动：

```py
def rollout(model, frames, horizon):
    """frames: test clips, shape (n, K + H, W) with H >= horizon. Returns a numpy array (n, horizon, W)."""
```

- 每个片段的前 `K` 帧（真实帧）是初始上下文。
- 每一步：把当前上下文交给 `model`，得到下一帧；然后丢掉上下文里最旧的一帧，把**刚预测出的帧**接在最后。
- 返回值的第 $h$ 个时间位置（$h$ 从 1 数起）是对片段第 $K + h - 1$ 帧（下标从 0 数起）的预测。
- `model` 可以是任何把形状 `(n, K, W)` 的张量映射成 `(n, W)` 张量的可调用对象。不要修改传入的 `frames`。

例子：令 `model = lambda c: 2 * c[:, -1] - c[:, -2]`，也就是用最后两帧做直线外推。记片段的前三帧为 `x0, x1, x2`，那么

```text
h = 1:  2*x2 - x1                          = x2 + 1*(x2 - x1)
h = 2:  2*(2*x2 - x1) - x2                 = x2 + 2*(x2 - x1)
h = 3:  2*(3*x2 - 2*x1) - (2*x2 - x1)      = x2 + 3*(x2 - x1)
```

指标都在测试片段上算，对 $h = 1, \dots, 30$ 分别给出：

- $\text{MSE}@h$：第 $h$ 步的预测帧与真实帧之差的平方，对所有测试片段和 16 个像素取平均。它是第 $h$ 步这一帧的误差，
  不是前 $h$ 步的累计。
- 位置误差：把预测帧里的负值当作 0，按像素下标做加权平均，得到质心（centre of mass）；取它与真实中心之差的绝对值，
  对片段取平均。单位是像素。
- 质量比（mass ratio）：把预测帧里的负值当作 0，求像素和，对片段取平均；再除以真实帧像素和的平均值（对所有片段和全部
  30 步取平均）。等于 1 表示亮斑的总亮度没有变。

两条不需要学习的参照线：

- 复制最后一帧：每一步都输出上下文的最后一帧，即片段的第 $K - 1$ 帧。
- 平均帧：每一步都输出训练集所有帧的逐像素平均。

交付：seed 0 的基线模型在 $h = 1, 5, 10, 20, 30$ 的 MSE 和质量比，两条参照线的 MSE；并说出基线从第几步开始不如“平均帧”。

### Part 2 —— 上下文加噪训练，做一次受控对比

实现加噪训练：

```py
def train(frames, seed, noise_std):
    """Like train_baseline, but every batch of context frames gets fresh N(0, noise_std^2) noise. Targets stay clean."""
```

- 只改一处：每个 batch 的上下文加上独立同分布的 $\mathcal N(0, \sigma^2)$ 噪声，每个 batch 重新采样；目标帧不加噪。
  初始权重、batch 顺序、步数、学习率都与 `train_baseline` 相同。
- `noise_std = 0` 时得到的权重必须与 `train_baseline` **逐位相同**。
- 同一个 seed 下，不同的 $\sigma$ 用同样的初始权重和同样的 batch 顺序。

实验网格：$\sigma \in \{0,\ 0.01,\ 0.03,\ 0.1\}$，seed 取 0 到 4（数据用 `load_data(seed)`），每个模型滚动 30 步。
汇总 5 个 seed 时，要给出中间水平和离散程度。用均值还是中位数由你决定，并说明理由。

如果还有时间，加**一个**你认为最能排除别的解释的对照实验。只加一个。

### Part 3 —— 一页报告

`report.md` 不超过一页，分三段：

1. 做了什么：实验设置，改了哪一个因素，怎么保证对比公平。
2. 结果：一张表；每个结论都带不确定性；说清哪些结论有数据支持，哪些没有。
3. 如果再有三小时，下一步做什么，按优先级排。

**评分看什么。** 对比是不是只改了一个因素；有没有参照线；数字有没有带不确定性；结论有没有超出证据；代码是不是一条命令
就能复现；时间是不是花在了回答问题上。

## 参考解答

<details>
<summary>展开参考解答</summary>

这道题的新代码很少。时间主要花在三件事上：确认评估是对的，把对比做干净，把结论写准。

### 三小时怎么分

| 时间 | 做什么 | 这一段结束时手里有什么 |
| --- | --- | --- |
| 0:00–0:15 | 读题。原样跑一次起始代码，确认单步 MSE 与题面一致。打印两三段数据看一眼 | 环境没问题，知道数据长什么样 |
| 0:15–0:50 | Part 1：`rollout`、三个指标、两条参照线。先用“假模型”检验 `rollout`，再跑真模型 | seed 0 基线的滚动误差 |
| 0:50–1:20 | Part 2：`train`。验证 $\sigma = 0$ 与基线权重逐位相同。先只在 seed 0 上跑完四个 $\sigma$ | 知道现象存在不存在 |
| 1:20–1:50 | 5 个 seed 全部跑完（约 1 分钟），整理成表，决定用中位数还是均值 | 主表 |
| 1:50–2:10 | 一个便宜的对照：滚动时裁剪输出。再只换噪声随机流，把 $\sigma = 0.01$ 重跑两遍（约半分钟） | 对照表，知道哪些结论稳 |
| 2:10–2:45 | 写报告 | `report.md` |
| 2:45–3:00 | 在干净目录里重跑一遍，逐个核对报告里的数字，打包 | 可以提交 |

两条纪律。第一，先在一个 seed 上把整条链路跑通，再加 seed；否则一个评估上的错误会浪费掉所有 seed 的时间。
第二，最后 15 分钟不写新代码，只核对和打包。

### Part 1

思路：滚动本身只有几行，容易错的是上下文怎么滑动。写完先不要跑真模型，用两个答案已知的“假模型”检验：
题面里的直线外推（它用到最后两帧，滑动错了就对不上），以及“复制最后一帧”（应当每一步都输出同一帧）。再确认
滚动的第 1 步等于模型在真实上下文上的单步预测。这三项检查都在本页末尾的验证代码里。

先抄一遍题目给的起始代码，与题面完全相同：

```python
import numpy as np
import torch

torch.set_num_threads(1)      # same numbers on every run, and polite on a shared machine

W = 16                        # pixels per frame
K = 3                         # context length: the model sees the last K frames
LO, HI = 2.0, 13.0            # the blob centre moves between these two walls
BLOB = 1.2                    # blob width in pixels (std of the Gaussian bump)
HORIZON = 30                  # longest rollout we evaluate


def make_sequences(n, length, seed):
    """n clips of a 1-D "video": one Gaussian bump moving at constant speed, bouncing off the walls.

    Returns the frames, float32 (n, length, W) with values in [0, 1], and the true centres, (n, length).
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(LO, HI, n)
    vel = rng.uniform(0.3, 1.2, n) * rng.choice([-1.0, 1.0], n)     # pixels per frame
    grid = np.arange(W)
    frames = np.zeros((n, length, W), dtype=np.float32)
    centres = np.zeros((n, length))
    for t in range(length):
        frames[:, t] = np.exp(-0.5 * ((grid[None, :] - pos[:, None]) / BLOB) ** 2)
        centres[:, t] = pos
        pos = pos + vel
        hit = pos > HI
        pos[hit], vel[hit] = 2 * HI - pos[hit], -vel[hit]
        hit = pos < LO
        pos[hit], vel[hit] = 2 * LO - pos[hit], -vel[hit]
    return frames, centres


class Predictor(torch.nn.Module):
    """Next frame = last frame + MLP(all K context frames)."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(K * W, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, W))

    def forward(self, context):                      # context: (B, K, W)
        return context[:, -1] + self.net(context.flatten(1))


def windows(frames):
    """Every (context, target) pair in the clips: arrays of shape (N, K, W) and (N, W)."""
    n, length, _ = frames.shape
    context = np.stack([frames[:, i:length - K + i] for i in range(K)], axis=2)
    target = frames[:, K:]
    return context.reshape(-1, K, W), target.reshape(-1, W)


def train_baseline(frames, seed, steps=2000, batch=128, lr=1e-3):
    """Teacher forcing: the context always consists of true frames."""
    torch.manual_seed(seed)                              # fixes the initial weights
    model = Predictor()
    context, target = (torch.from_numpy(a) for a in windows(frames))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = torch.Generator().manual_seed(seed)          # fixes the batch order
    for _ in range(steps):
        idx = torch.randint(0, context.shape[0], (batch,), generator=order)
        loss = ((model(context[idx]) - target[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def one_step_mse(model, frames):
    """Mean squared error of one-step predictions from true contexts, over all windows."""
    context, target = (torch.from_numpy(a) for a in windows(frames))
    return float(((model(context) - target) ** 2).mean())


def load_data(seed):
    """The train / test split for one seed. Test clips are K + HORIZON frames long."""
    train_frames, _ = make_sequences(256, 24, 1000 + seed)
    test_frames, test_centres = make_sequences(500, K + HORIZON, 2000 + seed)
    return train_frames, test_frames, test_centres
```

然后是滚动、指标和参照线。`rollout` 多了一个 `clip` 参数，Part 2 的对照实验会用到。

```python
@torch.no_grad()
def rollout(model, frames, horizon, clip=False):
    """Give the model the first K true frames, then feed it its own predictions.

    Returns the predictions for frames K .. K+horizon-1, shape (n, horizon, W).
    `model` is any callable from a (n, K, W) tensor to a (n, W) tensor.
    clip=True clamps every predicted frame to [0, 1] before it is fed back (a control used in Part 2).
    """
    context = torch.from_numpy(frames[:, :K]).clone()
    out = []
    for _ in range(horizon):
        nxt = model(context)
        if clip:
            nxt = nxt.clamp(0.0, 1.0)
        out.append(nxt)
        context = torch.cat([context[:, 1:], nxt[:, None]], dim=1)
    return torch.stack(out, dim=1).numpy()


def centre_of_mass(frames):
    """Position readout: intensity-weighted mean pixel index (negative values count as zero)."""
    f = np.clip(frames, 0.0, None)
    return (f * np.arange(W)).sum(-1) / (f.sum(-1) + 1e-8)


def evaluate(pred, test_frames, test_centres):
    """Per-horizon metrics; each value is an array of length H, index 0 is horizon 1."""
    h = pred.shape[1]
    truth = test_frames[:, K:K + h]
    return {
        "mse": ((pred - truth) ** 2).mean(axis=(0, 2)),
        "pos": np.abs(centre_of_mass(pred) - test_centres[:, K:K + h]).mean(axis=0),
        "mass": np.clip(pred, 0.0, None).sum(-1).mean(axis=0) / truth.sum(-1).mean(),
    }


def reference_levels(train_frames, test_frames, test_centres):
    """Two predictors that learn nothing: repeat the last context frame; output the mean training frame."""
    shape = test_frames[:, K:].shape
    copy_last = np.repeat(test_frames[:, K - 1:K], shape[1], axis=1)
    mean_frame = np.broadcast_to(train_frames.mean(axis=(0, 1)), shape)
    return {"copy": evaluate(copy_last, test_frames, test_centres),
            "mean": evaluate(mean_frame, test_frames, test_centres)}


train0, test0, centres0 = load_data(0)
baseline0 = train_baseline(train0, seed=0)
base0 = evaluate(rollout(baseline0, test0, HORIZON), test0, centres0)
refs0 = reference_levels(train0, test0, centres0)

SHOW = (1, 5, 10, 20, 30)


def part1_table():
    rows = [("基线模型", base0["mse"]), ("复制最后一帧", refs0["copy"]["mse"]), ("平均帧", refs0["mean"]["mse"])]
    lines = ["| " + name + " | " + " | ".join(f"{v[h - 1]:.2g}" for h in SHOW) + " |" for name, v in rows]
    lines.append("| 基线模型的质量比 | " + " | ".join(f"{base0['mass'][h - 1]:.2f}" for h in SHOW) + " |")
    return lines


print("\n".join(part1_table()))
```

seed 0 的结果（各列是 $h$）：

| | 1 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| 基线模型 | 2.4e-05 | 0.00087 | 0.013 | 0.22 | 1.9 |
| 复制最后一帧 | 0.025 | 0.18 | 0.21 | 0.17 | 0.18 |
| 平均帧 | 0.089 | 0.089 | 0.088 | 0.087 | 0.087 |
| 基线模型的质量比 | 1.00 | 1.01 | 1.06 | 1.37 | 2.69 |

怎么读：

- 第 1 步，基线比“复制最后一帧”好约一千倍。到第 10 步仍然远好于两条参照线。
- 第 16 步起，基线的 MSE 超过“平均帧”。换成另外四个 seed，这个位置在第 15 到 19 步之间。
- 第 30 步，基线的 MSE 是 1.9，约为“平均帧”的 21 倍。质量比是 2.69：预测帧的总亮度涨到了真实值的 2.7 倍。
  所以这不只是位置慢慢漂走，而是幅度在发散。只看 MSE 分不清这两种失败，质量比能分清。
- “平均帧”这条参照线很重要。没有它，0.22 这个数看不出好坏；有了它，就知道模型在第 20 步已经不如什么都不学。

### Part 2

思路：对比要干净，靠三点。

1. **配对。** 同一个 seed 下，四个 $\sigma$ 共用训练集、测试集、初始权重和 batch 顺序。噪声用单独的随机数发生器，
   所以加不加噪都不会改变 batch 顺序。这样同一个 seed 里两个模型的差别只来自噪声。`noise_seed` 参数只给最后的
   稳健性检查用。
2. **对照组走同一条代码路径。** $\sigma = 0$ 也调用 `train`，并验证它与 `train_baseline` 的权重逐位相同。否则“基线”
   和“方法”之间可能差着一个没注意到的实现细节。
3. **网格事先定好，全部汇报。** 没有验证集，所以不在测试集上挑“最好的 $\sigma$”再报它的数字。

```python
def train(frames, seed, noise_std, steps=2000, batch=128, lr=1e-3, noise_seed=None):
    """train_baseline plus Gaussian noise on the context frames. The targets stay clean.
    noise_seed is only used by the robustness check below; by default it is 10_000 + seed."""
    torch.manual_seed(seed)
    model = Predictor()
    context, target = (torch.from_numpy(a) for a in windows(frames))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = torch.Generator().manual_seed(seed)
    noise = torch.Generator().manual_seed(10_000 + seed if noise_seed is None else noise_seed)   # its own stream: same batch order for every sigma
    for _ in range(steps):
        idx = torch.randint(0, context.shape[0], (batch,), generator=order)
        x = context[idx]
        if noise_std > 0:
            x = x + noise_std * torch.randn(x.shape, generator=noise)
        loss = ((model(x) - target[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


SIGMAS = (0.0, 0.01, 0.03, 0.1)
SEEDS = (0, 1, 2, 3, 4)

runs, refs, clipped, sigma0_models = {}, {}, {}, {}
for seed in SEEDS:
    train_frames, test_frames, test_centres = load_data(seed)
    refs[seed] = reference_levels(train_frames, test_frames, test_centres)
    for sigma in SIGMAS:
        model = train(train_frames, seed, sigma)
        runs[sigma, seed] = evaluate(rollout(model, test_frames, HORIZON), test_frames, test_centres)
        if sigma == 0.0:                                   # control: same weights, outputs clamped at rollout time
            sigma0_models[seed] = model
            clipped[seed] = evaluate(rollout(model, test_frames, HORIZON, clip=True), test_frames, test_centres)


def at(metric, sigma, h):
    """One number per seed: the metric of the sigma model at horizon h."""
    return np.array([runs[sigma, s][metric][h - 1] for s in SEEDS], dtype=float)


def ref_at(metric, name, h):
    return np.array([refs[s][name][metric][h - 1] for s in SEEDS], dtype=float)


def cell(values, scale=1.0, fmt=".3g"):
    """median (min–max) over seeds"""
    v = np.asarray(values, dtype=float) * scale
    return f"{format(np.median(v), fmt)} ({format(v.min(), fmt)}–{format(v.max(), fmt)})"


def mse_table():
    lines = []
    for sigma in SIGMAS:
        lines.append(f"| {sigma:g} | {cell(at('mse', sigma, 1), 1e4)} | {cell(at('mse', sigma, 10), 1e2)} | "
                     f"{cell(at('mse', sigma, 20))} | {cell(at('mse', sigma, 30))} | {cell(at('mass', sigma, 30), fmt='.2f')} |")
    for label, name in (("复制最后一帧", "copy"), ("平均帧", "mean")):
        lines.append(f"| {label} | {cell(ref_at('mse', name, 1), 1e4)} | {cell(ref_at('mse', name, 10), 1e2)} | "
                     f"{cell(ref_at('mse', name, 20))} | {cell(ref_at('mse', name, 30))} | {cell(ref_at('mass', name, 30), fmt='.2f')} |")
    return lines


def pos_table():
    lines = [f"| {sigma:g} | " + " | ".join(cell(at("pos", sigma, h), fmt=".2f") for h in (10, 20, 30)) + " |" for sigma in SIGMAS]
    lines.append("| 0，滚动时裁剪到 [0, 1] | " + " | ".join(
        cell([clipped[s]["pos"][h - 1] for s in SEEDS], fmt=".2f") for h in (10, 20, 30)) + " |")
    for label, name in (("复制最后一帧", "copy"), ("平均帧", "mean")):
        lines.append(f"| {label} | " + " | ".join(cell(ref_at("pos", name, h), fmt=".2f") for h in (10, 20, 30)) + " |")
    return lines


def clip_table():
    return ["| " + label + " | " + " | ".join(cell(vals(h)) for h in (10, 20, 30)) + " |" for label, vals in (
        ("σ = 0", lambda h: at("mse", 0.0, h)),
        ("σ = 0，滚动时裁剪到 [0, 1]", lambda h: [clipped[s]["mse"][h - 1] for s in SEEDS]),
        ("σ = 0.01，不裁剪", lambda h: at("mse", 0.01, h)))]


print("\n".join(mse_table()))
print("\n".join(pos_table()))
print("\n".join(clip_table()))


# Robustness: is the sigma = 0.01 gain a property of the method or of one lucky noise stream?
# Same data, same initial weights, same batch order; only the noise stream changes (two more streams per seed).
EXTRA_STREAMS = (20_100, 20_200)
gains = {}
for seed in SEEDS:
    train_frames, test_frames, test_centres = load_data(seed)
    base = runs[0.0, seed]["mse"]
    curves = [runs[0.01, seed]["mse"]]
    for stream in EXTRA_STREAMS:
        model = train(train_frames, seed, 0.01, noise_seed=stream + seed)
        curves.append(evaluate(rollout(model, test_frames, HORIZON), test_frames, test_centres)["mse"])
    gains[seed] = [(float(base[9] / m[9]), float(base[19] / m[19])) for m in curves]    # baseline / noisy, h = 10 and 20


def stream_table():
    return [f"| {s} | " + ", ".join(f"{g10:.2f}" for g10, _ in gains[s]) + " | "
            + ", ".join(f"{g20:.2f}" for _, g20 in gains[s]) + " |" for s in SEEDS]


print("\n".join(stream_table()))
```

**用中位数。** 基线的 MSE@30 在 5 个 seed 上从 0.521 到 10.3，相差约 20 倍，是重尾的。均值和标准差会被一个 seed 主导。
所以每格写“中位数（最小–最大）”。

MSE 和质量比，5 个 seed：

| $\sigma$ | MSE@1（×10⁻⁴） | MSE@10（×10⁻²） | MSE@20 | MSE@30 | 质量比@30 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.295 (0.237–0.376) | 1.28 (1.14–1.75) | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) | 2.69 (1.67–3.04) |
| 0.01 | 0.468 (0.389–0.526) | 0.735 (0.607–0.797) | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) | 1.05 (1.02–1.10) |
| 0.03 | 1.17 (1.04–1.31) | 1.13 (0.721–1.58) | 0.0563 (0.0441–0.0617) | 0.0939 (0.0923–0.115) | 1.01 (0.97–1.08) |
| 0.1 | 6.56 (5.87–8.43) | 6.71 (6.12–7.33) | 0.142 (0.135–0.153) | 0.166 (0.162–0.179) | 1.00 (0.99–1.01) |
| 复制最后一帧 | 259 (249–267) | 21.1 (20.3–21.6) | 0.174 (0.171–0.177) | 0.18 (0.18–0.184) | 1.00 (1.00–1.00) |
| 平均帧 | 881 (867–886) | 8.78 (8.65–8.89) | 0.0875 (0.0868–0.0882) | 0.0879 (0.0874–0.0881) | 1.00 (1.00–1.00) |

位置误差（像素），5 个 seed：

| $\sigma$ | 位置误差@10 | 位置误差@20 | 位置误差@30 |
| --- | --- | --- | --- |
| 0 | 0.39 (0.38–0.56) | 2.08 (1.43–2.93) | 3.07 (2.76–3.87) |
| 0.01 | 0.31 (0.28–0.35) | 0.90 (0.83–1.02) | 1.62 (1.39–2.00) |
| 0.03 | 0.40 (0.31–0.50) | 1.04 (0.90–1.12) | 1.62 (1.47–1.82) |
| 0.1 | 1.21 (1.14–1.29) | 2.52 (2.42–2.70) | 3.64 (3.48–3.78) |
| 0，滚动时裁剪到 [0, 1] | 0.39 (0.35–0.50) | 1.72 (1.28–2.31) | 2.97 (2.45–3.42) |
| 复制最后一帧 | 4.63 (4.42–4.70) | 3.75 (3.59–3.88) | 3.97 (3.89–4.04) |
| 平均帧 | 2.75 (2.60–2.90) | 2.70 (2.67–2.80) | 2.73 (2.69–2.80) |

**结果解读。**

1. **长程误差确实降低了，幅度远大于 seed 之间的波动。** 第 20 步，$\sigma = 0.01$ 的 MSE 是 0.043（0.0396–0.0616），
   基线是 0.198（0.125–0.223），两个范围不重叠。按 seed 配对看，基线分别是 $\sigma = 0.01$ 的 3.5、5.2、2.4、4.8、3.6 倍。
   $\sigma = 0.03$ 的结果相近。
2. **第 20 步肯定有用；第 10 步还说不准；第 5 步没有用。** 主表里，第 10 步 $\sigma = 0.01$ 在 5 个 seed 上都更好，
   基线是它的 1.4 到 2.6 倍。但下面只换噪声随机流的重复实验显示，这个结论不稳。第 5 步，两者之比在 0.79 到 1.39
   之间，方向不一致；$\sigma = 0.03$ 在第 5 步 5 个 seed 都比基线差。
3. **加噪消除了发散。** 基线的质量比@30 是 2.69（1.67–3.04），三个加噪模型都在 0.97 到 1.10 之间。
4. **代价是单步误差。** $\sigma = 0.01$ 的单步 MSE 是基线的 1.1 到 1.8 倍，$\sigma = 0.1$ 是 16 到 29 倍。即便如此，
   $\sigma = 0.1$ 的单步误差仍远小于“复制最后一帧”。
5. **噪声太大得不偿失。** $\sigma = 0.1$ 在第 10 步的 MSE 是基线的 3.8 到 5.7 倍；第 20 步是 0.142，比“平均帧”的 0.0875
   还差；第 30 步的位置误差是 3.64 像素，比“平均帧”的 2.73 还大，位置信息已经丢了。一个可能的解释：亮斑每帧只移动
   0.3 到 1.2 像素，相邻帧的差本来就小，噪声大到与它可比时，模型就读不准速度。这个解释我没有单独验证。
6. **第 30 步，按 MSE 谁也没有赢过“平均帧”。** $\sigma = 0.01$ 和 $0.03$ 的 MSE@30 中位数是 0.0976 和 0.0939，
   “平均帧”是 0.0879。但它们的位置误差是 1.62 像素，“平均帧”是 2.73，5 个 seed 都更小。也就是说，模型还大致知道亮斑
   在哪，只是 MSE 对“清晰但偏了一点”的预测，罚得比“模糊的平均”更重。所以长程不能只看 MSE。

**一个便宜的对照：滚动时把输出裁剪到 [0, 1]。** 基线的失败里有发散。那加噪的好处会不会只是“限制了幅度”？像素值本来就
在 [0, 1] 里，所以可以在滚动时把基线的每个预测帧裁剪到这个范围，再喂回去。这不需要重新训练，只多一次滚动。

| | MSE@10 | MSE@20 | MSE@30 |
| --- | --- | --- | --- |
| σ = 0 | 0.0128 (0.0114–0.0175) | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) |
| σ = 0，滚动时裁剪到 [0, 1] | 0.0091 (0.0081–0.011) | 0.101 (0.0787–0.13) | 0.262 (0.157–0.267) |
| σ = 0.01，不裁剪 | 0.00735 (0.00607–0.00797) | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) |

裁剪把基线第 30 步的 MSE 从 1.85 压到 0.262，不再爆炸；但质量比@30 仍是 1.75（1.48–1.91），总亮度还是明显偏高。
第 20 步它仍是 $\sigma = 0.01$ 的 1.9 到 3.0 倍，位置误差@20 是 1.72 像素，而 $\sigma = 0.01$ 是 0.90。所以加噪的作用不只是限制幅度，它还减小了位置漂移。裁剪只是推理时的一行代码，
本身也值得用；它和加噪叠加的效果，这里没有跑。

**只换噪声随机流，结论还在不在？** 主表里每个 seed 同时决定了数据、初始权重、batch 顺序和噪声。为了看噪声本身的
随机性有多大，固定前三样，只换噪声的随机数流，把 $\sigma = 0.01$ 每个 seed 再训两遍（多 10 个模型，约半分钟）。
表中是“基线的 MSE ÷ 加噪模型的 MSE”，大于 1 表示加噪更好；每格第一个数是主表那次。

| seed | 第 10 步，三条噪声流 | 第 20 步，三条噪声流 |
| --- | --- | --- |
| 0 | 2.05, 1.66, 1.48 | 3.49, 4.01, 3.61 |
| 1 | 2.22, 1.59, 1.24 | 5.20, 3.29, 3.10 |
| 2 | 1.44, 0.99, 1.02 | 2.39, 1.31, 1.84 |
| 3 | 2.63, 1.29, 3.55 | 4.83, 2.85, 4.36 |
| 4 | 1.55, 0.92, 1.31 | 3.61, 2.28, 2.69 |

第 20 步，15 次全都更好，比值在 1.31 到 5.20 之间。第 10 步，15 次里有 2 次反而更差（0.99 和 0.92）。所以报告里
只能写“到第 20 步有明确的好处”；“从第 10 步起就有好处”要更多重复才能说。这也说明，只汇报一条噪声流时，
seed 之间的波动会低估真实的不确定性。

**这份数据不支持的结论。**

- 第 10 步就有好处。理由见上一段。
- 精确的倍数。只有 5 个 seed。方向 5 比 0，符号检验（sign test）的双侧 p 值是 $2 \times 0.5^5 = 0.0625$，单靠它到不了 0.05。
  结论的说服力来自效应量（2.4 倍以上）和两组范围不重叠。
- $\sigma = 0.01$ 和 $0.03$ 哪个更好。两者分不出高下。
- “起作用的是噪声本身”。没有跑权重衰减（weight decay）之类的对照，不能排除“任何正则化都有类似效果”。
- 测试集只有 500 个片段带来的抽样误差。这一项没有量化。
- 能不能推广到真实视频。这里是单个物体、确定性运动、MLP。

### Part 3

下面是一份范例，篇幅约一页。实际提交用英文写，结构相同。

**问题。** 下一帧预测器训练时只见过真实上下文，滚动时却要吃自己的输出。训练时给上下文加高斯噪声（目标不加），能否降低
长程滚动误差，代价是什么？

**做法。** 数据和基线用题目给的：一维亮斑视频，MLP，看 3 帧。只改一个因素：上下文噪声的标准差
$\sigma \in \{0, 0.01, 0.03, 0.1\}$。同一个 seed 下，四个 $\sigma$ 共用训练集、测试集、初始权重和 batch 顺序；$\sigma = 0$
与题目基线的权重逐位相同。5 个 seed，每个模型从 3 帧真实上下文滚动 30 步，在 500 个测试片段上评估。网格在看结果之前
定好，四个 $\sigma$ 全部汇报。基线的长程误差是重尾的（MSE@30 在 seed 之间相差约 20 倍），所以写中位数（最小–最大）。

**结果。**

| $\sigma$ | MSE@1（×10⁻⁴） | MSE@20 | MSE@30 | 位置误差@20（像素） |
| --- | --- | --- | --- | --- |
| 0（基线） | 0.295 | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) | 2.08 |
| 0.01 | 0.468 | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) | 0.90 |
| 0.03 | 1.17 | 0.0563 (0.0441–0.0617) | 0.0939 (0.0923–0.115) | 1.04 |
| 0.1 | 6.56 | 0.142 (0.135–0.153) | 0.166 (0.162–0.179) | 2.52 |
| 平均帧 | 881 | 0.0875 | 0.0879 | 2.70 |

- 加噪降低了长程误差。第 20 步，基线的 MSE 是 $\sigma = 0.01$ 的 2.4 到 5.2 倍，5 个 seed 方向一致，两组范围不重叠。
  只换噪声随机流再重复两次，第 20 步 15 次全都更好；第 10 步有 2 次反而更差，第 5 步方向不一致。
- 基线在第 15 到 19 步之后比“永远输出平均帧”还差；到第 30 步，预测帧的总亮度是真实值的 1.7 到 3.0 倍。加噪后总亮度
  保持在 0.97 到 1.10 倍。
- 代价：$\sigma = 0.01$ 的单步误差是基线的 1.1 到 1.8 倍。$\sigma = 0.1$ 太大：单步误差是 16 到 29 倍，第 10 步的误差是
  基线的 3.8 到 5.7 倍。
- 对照：滚动时把基线的输出裁剪到 [0, 1]，MSE 不再爆炸（MSE@30 从 1.85 降到 0.262），但总亮度仍偏高（质量比@30 为 1.75），
  第 20 步的误差仍是 $\sigma = 0.01$ 的 1.9 到 3.0 倍。加噪的作用不只是限制幅度。
- 不该读出的结论：第 30 步，没有哪个设置的 MSE 低于平均帧（0.0976、0.0939 对 0.0879），此时模型只是位置比平均帧准
  （1.62 对 2.73 像素）。$\sigma = 0.01$ 和 $0.03$ 分不出高下。5 个 seed 的符号检验双侧 p 值是 0.0625，结论靠的是效应量，
  不是显著性。没有排除“任何正则化都有类似效果”。

**下一步（按优先级）。**

1. 排除“只是正则化”：在同样的训练步数下加一个权重衰减的对照。
2. seed 加到 20 个，$\sigma$ 在 0.001 到 0.03 之间加密，找出够用的最小噪声。
3. 与直接针对问题的做法比较：训练时展开 2 到 4 步，对滚动误差求损失（multi-step loss）。
4. 换更难的数据：两个亮斑，或带随机性的运动。带随机性时 MSE 本身就不合适，要换指标。

### 追问

**为什么给上下文加噪会有用？** 滚动到第 $h$ 步时，上下文是“真实帧加上累积的误差”，训练集里没有这样的输入。网络在没见过的
输入上怎么外推，训练不提供任何约束；在这里的表现就是亮度被一步步放大。加噪让模型在训练时就见到“真实帧附近的偏离”，
而目标仍是干净的下一帧，于是它学到把小偏离拉回去，而不是放大。不过独立的像素噪声只是滚动误差的粗糙替身：真实的
滚动误差是有结构的，比如整体偏移。这也解释了为什么加噪消除了发散，却没有把长程的位置误差消掉。

**为什么目标帧不加噪？** 给目标加零均值的独立噪声不会改变 MSE 的最优解，最优预测仍是条件均值，只是梯度更吵。起作用的是
输入端的噪声：输入带噪时，最优预测是“给定带噪上下文时，干净下一帧的条件均值”，模型被迫顺带学会去噪。

**长程的 MSE 为什么会骗人？** 当预测的位置与真实位置已经无关时，“清晰但放错位置”的帧，期望 MSE 恰好是“平均帧”的两倍：
设 $a$、$b$ 是两帧独立同分布的图像，均值为 $\mu$，则 $\mathbb E\|a-b\|^2 = 2\,\mathbb E\|a-\mu\|^2$。表里“复制最后一帧”
在第 30 步是“平均帧”的 2.04 到 2.10 倍，就是这个关系。所以 MSE 会奖励模糊。评估视频预测时，要同时看位置、清晰度
这类不奖励模糊的指标。

**没有验证集，怎么选 $\sigma$ 才不算在测试集上调参？** 这页的做法是不选：网格事先定好，四个值全部汇报，结论是“0.01 到 0.03
都有效”，而不是“最优值是多少”。如果必须交出一个值，就从训练片段里留出一部分做验证，用验证集上的滚动误差选，再在测试集上
只报这一个值。

**真实的视频生成模型里，对应的做法是什么？** 同一类思路有几种：训练时给条件帧（或它们的隐变量）加噪，并把噪声强度作为输入
告诉模型，推理时再指定一个小的强度；学习型物理模拟器在训练时给输入状态加随机游走噪声；训练时按一定概率用模型自己的输出
替换真实输入（scheduled sampling）；以及直接展开多步求损失。这些是凭记忆列出的方向，具体细节请在面试前对照原论文核实。

**如果只剩 30 分钟，报告还没写，砍什么？** 砍位置误差表和裁剪对照。保留 5 个 seed、两条参照线、“$\sigma = 0$ 与基线一致”的
验证。没有参照线和不确定性的结果，别人没法用；多一个对照只是加分。

<details>
<summary>验证代码（可运行）</summary>

```python
# ---- starter code: the generator does what its docstring says ---------------------------------
f, c = make_sequences(4, 40, seed=7)
assert f.shape == (4, 40, W) and f.dtype == np.float32 and c.shape == (4, 40)
assert f.min() >= 0.0 and f.max() <= 1.0 and c.min() >= LO and c.max() <= HI
assert np.allclose(f, np.exp(-0.5 * ((np.arange(W) - c[..., None]) / BLOB) ** 2), atol=1e-6)
step = np.abs(np.diff(c, axis=1))                       # distance moved per frame
speed = step.max(axis=1, keepdims=True)
assert (speed >= 0.3).all() and (speed <= 1.2).all()
assert (np.isclose(step, speed).mean(axis=1) >= 0.8).all()      # constant speed except in the bounce frames
assert np.array_equal(make_sequences(4, 40, seed=7)[0], f)      # same seed, same data

ctx, tgt = windows(f)
assert ctx.shape == (4 * 37, K, W) and tgt.shape == (4 * 37, W)
assert np.array_equal(ctx[0], f[0, 0:3]) and np.array_equal(tgt[0], f[0, 3])
assert np.array_equal(ctx[37 + 5], f[1, 5:8]) and np.array_equal(tgt[37 + 5], f[1, 8])

# ---- the two numbers quoted in the statement ---------------------------------------------------
assert train0.shape == (256, 24, W) and test0.shape == (500, K + HORIZON, W)
assert f"{one_step_mse(baseline0, test0):.1e}" == "2.5e-05"
c0, t0 = windows(test0)
assert round(float(((c0[:, -1] - t0) ** 2).mean()), 3) == 0.025

# ---- rollout: the worked example of Part 1, and its basic properties ---------------------------
linear = lambda c: 2 * c[:, -1] - c[:, -2]              # straight-line extrapolation from the last two frames
x1, x2 = test0[:2, 1], test0[:2, 2]
r = rollout(linear, test0[:2], 3)
assert r.shape == (2, 3, W)
for h in (1, 2, 3):
    assert np.allclose(r[:, h - 1], x2 + h * (x2 - x1), atol=1e-5)
assert np.array_equal(rollout(lambda c: c[:, -1], test0, 5), np.repeat(test0[:, K - 1:K], 5, axis=1))
with torch.no_grad():
    direct = baseline0(torch.from_numpy(test0[:, :K])).numpy()
assert np.array_equal(rollout(baseline0, test0, 1)[:, 0], direct)            # horizon 1 is the one-step prediction
before = test0.copy()
full = rollout(baseline0, test0, HORIZON)
assert np.array_equal(test0, before)                                         # the input clips are not modified
assert np.array_equal(full[:, :7], rollout(baseline0, test0, 7))             # a longer rollout extends a shorter one
clamped = rollout(baseline0, test0, HORIZON, clip=True)
assert clamped.min() >= 0.0 and clamped.max() <= 1.0 and full.max() > 1.5


def slow_rollout(model, clip_frames, horizon):
    """One clip at a time, a plain Python list as the context. Shares no code with rollout()."""
    frames_so_far = [clip_frames[i] for i in range(K)]
    preds = []
    for _ in range(horizon):
        with torch.no_grad():
            nxt = model(torch.from_numpy(np.stack(frames_so_far[-K:]))[None])[0].numpy()
        preds.append(nxt)
        frames_so_far.append(nxt)
    return np.stack(preds)


for i in range(5):
    assert np.allclose(slow_rollout(baseline0, test0[i], 8), full[i, :8], atol=1e-4)

# ---- metrics -------------------------------------------------------------------------------------
perfect = evaluate(test0[:, K:], test0, centres0)
assert (perfect["mse"] == 0).all() and perfect["pos"].max() < 0.01 and np.abs(perfect["mass"] - 1).max() < 0.01
assert np.isclose(refs0["copy"]["mse"][0], ((test0[:, K - 1] - test0[:, K]) ** 2).mean())
assert np.isclose(refs0["mean"]["mse"][4], ((train0.mean(axis=(0, 1)) - test0[:, K + 4]) ** 2).mean())

# ---- Part 1 table (seed 0), to the printed precision ---------------------------------------------
# These exact digits hold for this page's PyTorch build and CPU. Another build rounds differently and can
# move the second significant figure (the worst seed's MSE@30 moves the most); the conclusions do not change.
assert part1_table() == [
    "| 基线模型 | 2.4e-05 | 0.00087 | 0.013 | 0.22 | 1.9 |",
    "| 复制最后一帧 | 0.025 | 0.18 | 0.21 | 0.17 | 0.18 |",
    "| 平均帧 | 0.089 | 0.089 | 0.088 | 0.087 | 0.087 |",
    "| 基线模型的质量比 | 1.00 | 1.01 | 1.06 | 1.37 | 2.69 |",
]
ratio_copy = refs0["copy"]["mse"][0] / base0["mse"][0]
assert 900 < ratio_copy < 1200                                    # "about a thousand times better than copying"
assert round(float(base0["mse"][29] / refs0["mean"]["mse"][29])) == 21
crossing = [int(np.argmax(runs[0.0, s]["mse"] > refs[s]["mean"]["mse"])) + 1 for s in SEEDS]
assert crossing == [16, 15, 19, 16, 18]                           # first horizon where the baseline is worse than the mean frame

# ---- Part 2 controls -----------------------------------------------------------------------------
assert all(torch.equal(p, q) for p, q in zip(baseline0.parameters(), sigma0_models[0].parameters()))
assert all(np.array_equal(runs[0.0, 0][k], base0[k]) for k in base0)
other = train(train0, 0, 0.01, steps=50)
again = train(train0, 0, 0.01, steps=50)
assert all(torch.equal(p, q) for p, q in zip(other.parameters(), again.parameters()))          # reproducible
plain = train(train0, 0, 0.0, steps=50)
assert not all(torch.equal(p, q) for p, q in zip(other.parameters(), plain.parameters()))      # the noise is really applied

# ---- Part 2 tables, to the printed precision -------------------------------------------------------
assert mse_table() == [
    "| 0 | 0.295 (0.237–0.376) | 1.28 (1.14–1.75) | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) | 2.69 (1.67–3.04) |",
    "| 0.01 | 0.468 (0.389–0.526) | 0.735 (0.607–0.797) | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) | 1.05 (1.02–1.10) |",
    "| 0.03 | 1.17 (1.04–1.31) | 1.13 (0.721–1.58) | 0.0563 (0.0441–0.0617) | 0.0939 (0.0923–0.115) | 1.01 (0.97–1.08) |",
    "| 0.1 | 6.56 (5.87–8.43) | 6.71 (6.12–7.33) | 0.142 (0.135–0.153) | 0.166 (0.162–0.179) | 1.00 (0.99–1.01) |",
    "| 复制最后一帧 | 259 (249–267) | 21.1 (20.3–21.6) | 0.174 (0.171–0.177) | 0.18 (0.18–0.184) | 1.00 (1.00–1.00) |",
    "| 平均帧 | 881 (867–886) | 8.78 (8.65–8.89) | 0.0875 (0.0868–0.0882) | 0.0879 (0.0874–0.0881) | 1.00 (1.00–1.00) |",
]
assert pos_table() == [
    "| 0 | 0.39 (0.38–0.56) | 2.08 (1.43–2.93) | 3.07 (2.76–3.87) |",
    "| 0.01 | 0.31 (0.28–0.35) | 0.90 (0.83–1.02) | 1.62 (1.39–2.00) |",
    "| 0.03 | 0.40 (0.31–0.50) | 1.04 (0.90–1.12) | 1.62 (1.47–1.82) |",
    "| 0.1 | 1.21 (1.14–1.29) | 2.52 (2.42–2.70) | 3.64 (3.48–3.78) |",
    "| 0，滚动时裁剪到 [0, 1] | 0.39 (0.35–0.50) | 1.72 (1.28–2.31) | 2.97 (2.45–3.42) |",
    "| 复制最后一帧 | 4.63 (4.42–4.70) | 3.75 (3.59–3.88) | 3.97 (3.89–4.04) |",
    "| 平均帧 | 2.75 (2.60–2.90) | 2.70 (2.67–2.80) | 2.73 (2.69–2.80) |",
]
assert clip_table() == [
    "| σ = 0 | 0.0128 (0.0114–0.0175) | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) |",
    "| σ = 0，滚动时裁剪到 [0, 1] | 0.0091 (0.0081–0.011) | 0.101 (0.0787–0.13) | 0.262 (0.157–0.267) |",
    "| σ = 0.01，不裁剪 | 0.00735 (0.00607–0.00797) | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) |",
]

# ---- every claim in the reading, checked seed by seed ----------------------------------------------
ratio = lambda a, b: [float(x / y) for x, y in zip(a, b)]
for s in SEEDS:                                           # one-step error grows with sigma, in every seed
    one = [runs[sigma, s]["mse"][0] for sigma in SIGMAS]
    assert one == sorted(one) and len(set(one)) == len(one)
cost_small = ratio(at("mse", 0.01, 1), at("mse", 0.0, 1))
cost_large = ratio(at("mse", 0.1, 1), at("mse", 0.0, 1))
assert (round(min(cost_small), 1), round(max(cost_small), 1)) == (1.1, 1.8)
assert (round(min(cost_large)), round(max(cost_large))) == (16, 29)
assert at("mse", 0.1, 1).max() * 25 < ref_at("mse", "copy", 1).min()          # still far better than copying

gain5 = ratio(at("mse", 0.0, 5), at("mse", 0.01, 5))
assert [round(g, 2) for g in gain5] == [1.04, 1.39, 0.79, 1.22, 1.06]         # no consistent direction at h = 5
assert min(gain5) < 1 < max(gain5)
assert max(ratio(at("mse", 0.0, 5), at("mse", 0.03, 5))) < 0.9                # sigma = 0.03 is worse at h = 5, 5 of 5
gain10 = ratio(at("mse", 0.0, 10), at("mse", 0.01, 10))
assert (round(min(gain10), 1), round(max(gain10), 1)) == (1.4, 2.6) and min(gain10) > 1.25
gain20 = ratio(at("mse", 0.0, 20), at("mse", 0.01, 20))
assert [round(g, 1) for g in gain20] == [3.5, 5.2, 2.4, 4.8, 3.6] and min(gain20) > 2.0
assert at("mse", 0.01, 20).max() < at("mse", 0.0, 20).min()                   # the two ranges do not overlap
assert at("mse", 0.03, 20).max() < at("mse", 0.0, 20).min()
assert 2 * 0.5 ** len(SEEDS) == 0.0625                                         # two-sided sign test with 5 of 5

loss10 = ratio(at("mse", 0.1, 10), at("mse", 0.0, 10))                         # too much noise
assert (round(min(loss10), 1), round(max(loss10), 1)) == (3.8, 5.7)
assert at("mse", 0.1, 20).min() > 1.5 * ref_at("mse", "mean", 20).max()
assert at("pos", 0.1, 30).min() > ref_at("pos", "mean", 30).max()

assert at("mass", 0.0, 30).min() > 1.5                                         # the baseline blows up in every seed
for sigma in SIGMAS[1:]:
    assert 0.95 < at("mass", sigma, 30).min() and at("mass", sigma, 30).max() < 1.12

mean30 = float(np.median(ref_at("mse", "mean", 30)))                           # at h = 30 nobody beats the mean frame on MSE
for sigma in (0.01, 0.03):
    assert 1.0 < float(np.median(at("mse", sigma, 30))) / mean30 < 1.15
    assert at("pos", sigma, 30).max() < ref_at("pos", "mean", 30).min()        # ... but the position is still better, 5 of 5
assert np.median(at("pos", 0.0, 30)) > np.median(ref_at("pos", "mean", 30))
double = ratio(ref_at("mse", "copy", 30), ref_at("mse", "mean", 30))
assert (round(min(double), 2), round(max(double), 2)) == (2.04, 2.1)
assert abs(np.median(ref_at("pos", "mean", 30)) - (HI - LO) / 4) < 0.1         # uniform position vs the midpoint
assert abs(np.median(ref_at("pos", "copy", 30)) - (HI - LO) / 3) < 0.35        # two independent uniform positions

clip20 = [float(clipped[s]["mse"][19] / runs[0.01, s]["mse"][19]) for s in SEEDS]      # the clipping control
assert (round(min(clip20), 1), round(max(clip20), 1)) == (1.9, 3.0) and min(clip20) > 1.5
assert max(clipped[s]["mse"][29] for s in SEEDS) < 0.3 < at("mse", 0.0, 30).min()
assert all(clipped[s]["pos"][19] > runs[0.01, s]["pos"][19] * 1.4 for s in SEEDS)
clip_mass = [float(clipped[s]["mass"][29]) for s in SEEDS]                     # clipping alone leaves it too bright
assert cell(clip_mass, fmt=".2f") == "1.75 (1.48–1.91)" and min(clip_mass) > 1.3
# ---- the remaining numbers quoted in the reading and in the report --------------------------------
def report_table():
    lines = []
    for label, sigma in (("0（基线）", 0.0), ("0.01", 0.01), ("0.03", 0.03), ("0.1", 0.1)):
        lines.append(f"| {label} | {np.median(at('mse', sigma, 1)) * 1e4:.3g} | {cell(at('mse', sigma, 20))} | "
                     f"{cell(at('mse', sigma, 30))} | {np.median(at('pos', sigma, 20)):.2f} |")
    lines.append(f"| 平均帧 | {np.median(ref_at('mse', 'mean', 1)) * 1e4:.3g} | {np.median(ref_at('mse', 'mean', 20)):.3g} | "
                 f"{np.median(ref_at('mse', 'mean', 30)):.3g} | {np.median(ref_at('pos', 'mean', 20)):.2f} |")
    return lines


assert report_table() == [
    "| 0（基线） | 0.295 | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) | 2.08 |",
    "| 0.01 | 0.468 | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) | 0.90 |",
    "| 0.03 | 1.17 | 0.0563 (0.0441–0.0617) | 0.0939 (0.0923–0.115) | 1.04 |",
    "| 0.1 | 6.56 | 0.142 (0.135–0.153) | 0.166 (0.162–0.179) | 2.52 |",
    "| 平均帧 | 881 | 0.0875 | 0.0879 | 2.70 |",
]
assert round(float(at("mse", 0.0, 30).max() / at("mse", 0.0, 30).min())) == 20      # heavy tail across seeds
assert (min(crossing), max(crossing)) == (15, 19)
assert (round(float(at("mass", 0.0, 30).min()), 1), round(float(at("mass", 0.0, 30).max()), 1)) == (1.7, 3.0)
noisy_mass = np.concatenate([at("mass", sigma, 30) for sigma in SIGMAS[1:]])
assert (round(float(noisy_mass.min()), 2), round(float(noisy_mass.max()), 2)) == (0.97, 1.10)
assert (round(min(gain20), 1), round(max(gain20), 1)) == (2.4, 5.2)
assert (round(min(gain5), 2), round(max(gain5), 2)) == (0.79, 1.39)
all_windows = float(((c0[:, -1] - t0) ** 2).mean()) / one_step_mse(baseline0, test0)
assert 900 < all_windows < 1100                                    # "about a thousand times" in the statement
# ---- robustness to the noise stream (sigma = 0.01): h = 20 holds every time, h = 10 does not ------
assert stream_table() == [
    "| 0 | 2.05, 1.66, 1.48 | 3.49, 4.01, 3.61 |",
    "| 1 | 2.22, 1.59, 1.24 | 5.20, 3.29, 3.10 |",
    "| 2 | 1.44, 0.99, 1.02 | 2.39, 1.31, 1.84 |",
    "| 3 | 2.63, 1.29, 3.55 | 4.83, 2.85, 4.36 |",
    "| 4 | 1.55, 0.92, 1.31 | 3.61, 2.28, 2.69 |",
]
all10 = [g10 for s in SEEDS for g10, _ in gains[s]]
all20 = [g20 for s in SEEDS for _, g20 in gains[s]]
assert len(all10) == 15 and sum(g < 1 for g in all10) == 2 and min(all20) > 1.3
assert (round(min(all20), 2), round(max(all20), 2)) == (1.31, 5.2)
assert np.allclose([gains[s][0] for s in SEEDS], list(zip(gain10, gain20)), rtol=1e-6)     # stream 0 is the main run
print("all checks passed")
```

</details>

</details>
