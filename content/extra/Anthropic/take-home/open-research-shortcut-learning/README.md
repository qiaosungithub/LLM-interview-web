# 72-Hour Open Research Take-Home: When Does a Small Vision Model Learn a Shortcut, and Can the Data Predict It?

## Problem

This is an open research take-home of about 72 hours. The prompt is a single question, and it is deliberately underspecified. How you build the data, which models you pick, and where you stop are all up to you.

**The question.** Training data often contains a feature that correlates with the label but is not the one we care about: a background, a watermark, the camera that took the photo. A model may learn this spurious shortcut instead of the feature it is supposed to learn. Investigate: when does a small vision model learn the shortcut instead of the intended feature? Can you predict this in advance from properties of the data?

**Conditions.**

- Submit within about 72 hours of receiving the prompt. CPU only; no GPU is provided.
- Generate images with code; do not download an existing dataset. That way every variable is under your control.
- You may read published papers. Cite anything you take from someone else, and test it with your own experiments.
- No AI coding assistants unless your invitation says otherwise.

**Deliverables.**

- A write-up of at most 4 pages, with an appendix of any length.
- Code that runs end to end and reproduces every number in the write-up.
- After you submit, a review call of about 45 minutes: you present for 10 minutes, then take questions.

**What reviewers look for.**

- Whether "learned the shortcut" has a measurable definition.
- Whether "predict from the data" really uses only the data, and was written down before training.
- Whether there are controls and repeats over multiple random seeds; whether you distinguish "the model used the shortcut" from "the model got it wrong".
- Whether the strength of the conclusions matches the evidence, and whether the write-up says clearly what is still unknown.
- Where the time went: what you cut, and why.

### Part 1 — Design a controllable dataset

Build a dataset with one core feature the model should learn and one shortcut feature. You need at least two knobs: how often the shortcut agrees with the label, and how hard the core feature is. Define how you will measure "how much the model relies on the shortcut".

### Part 2 — Train and measure

Train small models on several settings and measure how much they rely on the shortcut. Include a control in which the shortcut is useless.

### Part 3 — Predict from the data

Propose a predictor that can be computed from the data alone, and compute it before training. Test how well it predicts: where it is right and where it is wrong.

### Part 4 — Explain the gap and write it up

If prediction and reality disagree, propose an explanation and test it. Write up the conclusions, the controls, alternative explanations, the limitations, and what you would do with one more week.

## Reference solution

<details>
<summary>Show the reference solution</summary>

### Scope first: how to spend 72 hours

Of the 72 hours, roughly 20 to 25 are actually usable. The "shortcut" in the prompt could be a background, a texture or a color; the model could be a CNN or a ViT; you could also study mitigations. Each of these is a month of work. First narrow the question to:

> One core feature, one shortcut, two knobs. Can the degree of shortcut reliance be ranked, and its size estimated, in advance by a quantity **computed only from the data distribution**?

What gets cut:

- No real datasets (such as Waterbirds). On real data you cannot compute "how much shortcut reliance is ideal", and that is exactly what answering "can it be predicted" hinges on.
- A single very small CNN; no architecture comparison.
- No mitigations (reweighting, group DRO, etc.); they go into "one more week".
- The shortcut's strength (the marker's brightness) and position are fixed; only the two knobs vary.

| Time | What | Output |
| --- | --- | --- |
| Day 1 morning | Write down the definitions and predictions; write the data generator; derive the Bayes-optimal observer and check it by simulation | A one-page "definitions and predictions" note; data code and the prediction table |
| Day 1 afternoon to evening | Build the small CNN, fuse many models into one training run, tune until it finishes in a few minutes; run one setting first | Working training code |
| Day 2 | Sweep the two knobs, 3 models per cell; conflict test; compare with the prediction | Main results table |
| Day 3 morning | Explain the gap: save snapshots during training and see how fast the core feature is learned; rerun with other seeds | Learning-curve table |
| Day 3 afternoon | Write the report; rerun the code from scratch and check every number in the report one by one | Report + code |

### Part 1 — Data, the measurement, and the prediction made in advance

**Data.** Images are 12×12 grayscale.

- **Core feature:** a bright bar of length 5, horizontal for label 0 and vertical for label 1. Its position is random below row 2 (rows counted from 0): 80 horizontal and 72 vertical placements, all equally likely.
- **Shortcut:** a 2×2 marker in the top-left corner, bright (+2) or dark (−2). The marker agrees with the label with probability $\rho$. The bar never touches that corner, so the two features are separate in pixel space.
- Every pixel gets Gaussian noise with standard deviation $\sigma$.

At $\rho = 0.5$ the marker is useless; that is the control. The larger $\sigma$, the harder it is to tell horizontal from vertical.

**How to measure "relying on the shortcut".** The main metric is **conflict accuracy**: on test images whose marker deliberately points to the wrong label, can the model still answer correctly? 1 means it ignores the marker completely; 0 means it follows the marker completely. Two auxiliary metrics:

- Aligned accuracy: accuracy when the marker points to the correct label.
- Core AUC: hold the marker fixed (all bright or all dark) and ask only whether the model's output separates horizontal from vertical. It answers "is there still core-feature information in the model's output?"

**A data-only prediction: the Bayes-optimal (ideal) observer.** We generate the data ourselves, so the posterior can be computed exactly. The log-likelihood ratio of the core feature is a sum over all bar templates:

$$\log\frac{p(x\mid 1)}{p(x\mid 0)} = \log \frac{1}{72}\sum_{t \in \text{vertical}} e^{\langle x, t\rangle/\sigma^2} - \log \frac{1}{80}\sum_{t \in \text{horizontal}} e^{\langle x, t\rangle/\sigma^2}$$

(Every template has $\|t\|^2 = 5$, which cancels in the ratio.) Adding the log-likelihood ratio from the marker gives the optimal classifier. Its accuracy on conflict images depends only on the data distribution ($\rho$, $\sigma$, the marker's brightness) and can be computed before training. We estimate it on 2000 test images.

An easily overlooked fact: **following the marker is not necessarily wrong.** In the table below, at $\sigma = 1.2$, $\rho = 0.8$ the optimal classifier itself gets only 0.082 on conflict images. So "the model used the shortcut" is not the problem in itself; the question is whether it uses it more or less than the data supports.

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

The ideal observer's predictions (`SEED = 0`):

| $\sigma$ | Accuracy from the bar alone | Conflict accuracy $\rho = 0.8$ | $\rho = 0.9$ | $\rho = 0.97$ |
| --- | --- | --- | --- | --- |
| 0.5 | 0.975 | 0.949 | 0.924 | 0.868 |
| 0.85 | 0.783 | 0.413 | 0.243 | 0.101 |
| 1.2 | 0.647 | 0.082 | 0.015 | 0.002 |

Predictions written down before training: conflict accuracy falls as $\rho$ and $\sigma$ grow; over the 9 cells with $\rho > 0.5$, the CNNs rank the same way as the table above. On magnitude: the CNN will not beat the ideal observer, but by how much it falls short is unknown.

### Part 2 — Train 36 small CNNs and measure

**Model.** 3×3 convolution (stride 2, 8 channels) → ReLU → 3×3 convolution (16 channels) → ReLU → global max pooling → linear output. 3 values of $\sigma$ × 4 values of $\rho$ × 3 models per cell, 36 in total.

**Training 36 models at once.** The 36 models are packed into one grouped convolution (`groups=36`): each group sees only its own input channels, the loss is the sum of the 36 models' losses, and Adam updates each parameter independently, so this is mathematically identical to training them separately, only much faster.

**Training.** Adam with a constant learning rate of 0.002 (constant, so that learning speed can be compared later), 1600 steps, 64 freshly generated images per model per step. There is no fixed training set, so overfitting a training set is not an issue. A copy of the weights is kept at steps 50, 100, 200, 400, 800 and 1600.

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

**Testing.** One set of 2000 test images per $\sigma$, shared by every model at that $\sigma$. Each image comes in three versions: marker pointing to the wrong label (conflict), marker pointing to the correct label (aligned), and no marker.

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

Results (`SEED = 0`, at the end of training). The "model-aware prediction" column is explained in Part 3.

| $\sigma$ | $\rho$ | Ideal observer conflict accuracy | CNN conflict accuracy (3 models) | CNN mean | Model-aware prediction | CNN core AUC |
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

How to read it:

- **The control holds.** At $\rho = 0.5$, conflict and aligned accuracy differ by less than 0.05: the model really ignores the marker. It is slightly worse than the ideal observer (0.957 vs 0.975 at $\sigma = 0.5$); even without a shortcut the model is not optimal.
- **The training objective was met.** In every cell with $\rho > 0.5$, aligned accuracy is above 0.95.
- **Shortcut reliance grows with $\rho$ and $\sigma$.** Conflict accuracy falls as $\rho$ grows, and also as $\sigma$ grows. At $\sigma = 0.5$, $\rho = 0.97$ only 0.158 is left; at $\sigma = 1.2$ it is close to 0 as soon as $\rho \ge 0.8$.
- **"Learned the shortcut" does not mean "did not learn the core feature".** At $\sigma = 0.5$, $\rho = 0.97$ conflict accuracy is only 0.158, yet the core AUC with the marker held fixed is still 0.894. The model learned some of the horizontal/vertical information; in its output, the marker's contribution simply outweighs it. Core AUC also falls as $\rho$ grows.
- **Some cells are very unstable.** In the $\sigma = 0.5$, $\rho = 0.97$ cell the three models range from 0.087 to 0.211. Looking at a single model's number is not reliable here.

### Part 3 — Predicting from the data: the order is roughly right, the magnitude is not

**Order.** Over the 9 cells with $\rho > 0.5$, the Spearman rank correlation between the ideal observer's and the CNNs' conflict accuracy is 0.88. In other words, a quantity computed only from the data distribution ranks "who relies more on the shortcut" roughly correctly, but the "same order" prediction written down in advance did not fully come true. The 0.88 comes mostly from each knob's own direction: when $\rho$ or $\sigma$ alone increases, conflict accuracy falls on both sides. When one knob goes up and the other goes down, it is not necessarily right: the ideal observer says $\sigma = 0.5$, $\rho = 0.97$ (0.868) relies on the shortcut far less than $\sigma = 0.85$, $\rho = 0.8$ (0.413), but for the CNNs it is 0.158 vs 0.197, the other way round; the pair comes out reversed in 3 of the 5 runs with `SEED = 0` to 4.

**Magnitude.** No CNN has a higher conflict accuracy than the ideal observer (with a tolerance of 0.01), and in the middle cells it is much lower: at $\sigma = 0.5$, $\rho = 0.97$ the ideal observer gets 0.868 and the CNNs average only 0.158. The mean absolute error over the 9 cells is 0.179. So in this experiment the data-only prediction acts as a **bound** (an observed regularity, not a theorem): models rely on the shortcut at least as much as the data supports, and in practice more.

**A prediction that accounts for the model's ability: the model-aware prediction.** The ideal observer assumes a perfect ability to read the bar. Substitute the model's own ability: take the "twin" model trained at $\rho = 0.5$ with the same $\sigma$. It has never seen a useful marker, so its logit $z(x)$ on marker-free images can be read as "how much evidence this model can extract from the bar". On a conflict image the marker pushes the log-odds by $\log\frac{\rho}{1-\rho}$ in the wrong direction. Hence the prediction:

$$\text{correct} \iff y_\pm \cdot z(x) > \log\frac{\rho}{1-\rho},\qquad y_\pm = \pm 1.$$

It needs only one twin model per $\sigma$, not one model trained at every $\rho$. Its rank correlation with the CNNs is also 0.88, so it ranks no better; but the mean absolute error drops to 0.107. It is still too optimistic at the extreme: at $\sigma = 0.5$, $\rho = 0.97$ it predicts 0.738, against an actual 0.158.

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

### Part 4 — Why models rely on the shortcut more than predicted: the core feature is learned more slowly

**Hypothesis.** The marker is easier to learn than the bar: it sits in a fixed position and is bright. Once the model has learned the marker, the loss on aligned examples (a fraction $\rho$) is already small, and the gradient that pushes the model to learn the bar comes mostly from the few conflict examples. So the larger $\rho$, the more slowly the bar is learned; with a fixed number of training steps, the bar ends up under-learned.

**Test.** Using the saved snapshots, look at how the core AUC rises with steps at $\sigma = 0.5$, and how many steps it takes to reach 0.75 (interpolating in log steps between adjacent snapshots):

| $\rho$ | 50 steps | 100 steps | 200 steps | 400 steps | 800 steps | 1600 steps | Steps to 0.75 | Relative to $\rho = 0.5$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.708 | 0.767 | 0.882 | 0.962 | 0.989 | 0.993 | 82 | 1.0 |
| 0.8 | 0.605 | 0.670 | 0.807 | 0.913 | 0.976 | 0.988 | 150 | 1.8 |
| 0.9 | 0.535 | 0.559 | 0.632 | 0.738 | 0.933 | 0.979 | 417 | 5.1 |
| 0.97 | 0.532 | 0.538 | 0.551 | 0.586 | 0.701 | 0.894 | 953 | 11.6 |

The larger $\rho$, the more slowly the core feature is learned: at $\rho = 0.97$ it takes 11.6 times as many steps. The order is the same across seeds (on some seeds $\rho = 0.8$ and 0.9 are nearly tied). It matches the direction of the gap in Part 3: the twin model used by the model-aware prediction has mostly learned the bar by step 1600, while the $\rho = 0.97$ models have not, so the prediction is too optimistic. The core AUC at $\rho = 0.97$ rises from 0.701 to 0.894 over the last 800 steps and is still climbing. With longer training the gap might shrink. We did not measure this, so we cannot draw a conclusion.

**What the results show.** On this synthetic task:

1. Shortcut reliance grows with "how reliable the shortcut is" ($\rho$) and "how hard the core feature is" ($\sigma$).
2. The ideal observer, computed from the data alone, ranks the degree of reliance roughly correctly (rank correlation 0.88; it gets the order wrong when the two knobs move in opposite directions), but here it only works as a bound: the trained CNNs rely on the shortcut more.
3. The gap is tied to learning speed: the more reliable the shortcut, the more slowly the core feature is learned.

**What the results do not show.**

- Whether the "more reliance" persists after training long enough. We looked at only one training length.
- Which matters, "the marker is easy to detect" or "the marker is reliable". The marker's brightness and position were fixed.
- What happens on real images. On real data the ideal observer cannot be computed, and so neither can Part 3's bound.
- Anything beyond one very small CNN. Global max pooling makes it naturally insensitive to position, which favors the bar; a different architecture could change the results.

**Alternative explanations and the matching controls.**

| Alternative explanation | Control | Result |
| --- | --- | --- |
| The model simply cannot tell horizontal from vertical (not enough capacity) | The $\rho = 0.5$ twin models | At $\sigma = 0.5$: conflict accuracy 0.957, core AUC 0.993 |
| Conflict images are rare in training, so the model gets them wrong at test time | The ideal observer faces the same test images and also knows $\rho$ | It already accounts for "conflicts are rare", and the CNNs still fall below it |
| The model did not learn the core feature at all | Core AUC with the marker held fixed | Still 0.894 at $\sigma = 0.5$, $\rho = 0.97$: partly learned, just outweighed by the marker |
| It is just under-training | Learning curves | Partly: the core AUC is still rising at the end; whether the gap disappears with longer training was not measured |
| Seed luck | 3 models per cell; rerun the whole page with `SEED` set to 1 and 2 | Every qualitative check in the verification code below passes at `SEED = 0`, 1 and 2 |

**What I would do with one more week.**

- **Separate "easy" from "reliable".** Sweep the marker's brightness: as it dims, the ideal observer's prediction changes; does the CNNs' excess reliance follow?
- **Training length.** Train for 10 times as many steps, or compare at equal training loss, and see whether the gap between the CNNs and the ideal observer narrows.
- **A predictor that does not need the ideal observer.** On real data one can measure separately "how fast the model learns with only the core feature" and "how fast it learns with only the shortcut", and predict reliance from the ratio of the two learning speeds. Compare with Hermann and Lampinen (2020) on feature difficulty (decodability) and Shah et al. (2020) on simplicity bias.
- **Mitigations.** Upweight conflict examples, group DRO; see whether the ranking prediction still holds.
- **Other architectures and real data.** A small ResNet, a small ViT; then a dataset with a background shortcut such as Waterbirds, testing only whether the ranking holds.

### Report outline

1. **Question and definitions** (half a page): the narrowed question; conflict accuracy, core AUC; why compare against the ideal observer.
2. **Data and prediction** (half a page): data generation; derivation of the ideal observer and the prediction table.
3. **Main results** (one page): the main results table, plus a plot of conflict accuracy against $\rho$, one line per $\sigma$, with the ideal observer as dashed lines on the same plot.
4. **Predicting from the data** (half a page): the order is right, the magnitude is not; the model-aware prediction.
5. **Explaining the gap** (half a page): the learning-curve table.
6. **Controls, alternative explanations and limitations** (half a page): the five-row table above.
7. **Next steps** (a few lines).
8. **Appendix**: hyperparameters, per-seed results, reproduction commands.

In the 10-minute presentation, show only one plot (CNNs versus the ideal observer), one conclusion (the order is roughly predictable, the degree is underestimated) and one explanation (the core feature is learned slowly).

### Follow-up questions

- **If conflict accuracy is close to 0, does that mean the model is wrong?** Not necessarily. At $\sigma = 1.2$, $\rho = 0.8$ the optimal classifier also gets only 0.082 on conflict images. On the training distribution, following the marker is correct. To judge whether a model relies on the shortcut "too much", you have to compare it with the ideal observer, not with 1.
- **The ideal observer cannot be computed on real data. What use is your method then?** The ranking result can transfer: replace the ideal observer with "how easy the core feature and the shortcut each are to learn", for example by training small models or linear probes on one feature at a time and comparing the steps needed to reach some accuracy. Whether this proxy ranks as well as it does here needs to be validated on synthetic data first, because only there is there a ground truth to compare against.
- **Why a constant learning rate?** We want to compare how fast the core feature is learned at different $\rho$. Cosine annealing makes the step size smaller and smaller late in training, so the same number of steps represents a different "amount of learning" at different stages, which muddies the comparison.
- **Do the 36 models packed into one grouped convolution interfere with each other?** No. With `groups=36` each group's kernels connect only to that group's channels; the global pooling and linear output are also separate per model; the loss is the sum of the per-model losses, so each parameter's gradient comes only from the model it belongs to. Adam is per-parameter, so the updates do not interact either. Each model also gets its own independently generated images at every step.
- **Can learning speed predict the degree of reliance quantitatively?** We tried a rough estimate: the effective signal for learning the bar shrinks roughly as $4\rho(1-\rho)$, so the slowdown factor is about $1/(4\rho(1-\rho))$, which is 8.6 at $\rho = 0.97$. The measurement is slower than this estimate at all three values of $\rho$, so we report only the order, not a quantitative relationship.
- **With only 24 hours, what would you cut?** Keep the ideal observer, a full $\rho$ sweep at one $\sigma$, and the $\rho = 0.5$ control: that is the minimum for the question "can it be predicted". Cut the learning curves and the multi-seed reruns, and state in the report that the cause of the gap was not tested and only one seed was run.

<details>
<summary>Verification code (runnable)</summary>

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
