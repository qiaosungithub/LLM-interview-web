# 72-Hour Open Research Take-Home: When Does a Small Model Store More Features Than It Has Dimensions?

## Problem

This is an open-ended research take-home of about 72 hours. There is a single question, and it is deliberately underspecified. How to define it, how to set up the experiments, and where to stop are all up to you.

**The question.** A small model whose hidden layer has only $m$ dimensions can sometimes represent more than $m$ features. This phenomenon is called superposition. Investigate: when does a model do this? What controls it?

**Conditions.**

- Submit within about 72 hours of receiving the prompt. An ordinary laptop CPU is enough; no GPU is provided.
- You may read public papers. Cite anyone else's results you use, and test them with your own experiments rather than just citing them.
- No AI coding assistants, unless the invitation says otherwise.

**Deliverables.**

- A write-up of at most 4 pages, with an unlimited appendix.
- Code that runs end to end and reproduces every number in the write-up.
- After you submit, a review call of about 45 minutes: you present for 10 minutes, then answer questions.

**What the reviewers look for.**

- Whether you narrowed the question down to one that can be answered, and whether claims like "represents a feature" have a measurable definition.
- Whether you wrote down predictions before running experiments, and what you did when the results didn't match them.
- Whether each conclusion has a control, and whether it was repeated across multiple random seeds.
- Whether the strength of each conclusion matches the evidence, and whether the write-up states clearly "what we don't know yet."
- Where the time went: what you cut, and why.

### Part 1 — Build a minimal model

Pick the smallest setup in which superposition can appear. Spell out how the data is generated, what the model looks like, and how the loss is computed. Give a measurable definition of "the model represents feature $i$," and show whether changing the threshold in that definition changes the conclusions.

### Part 2 — Find the controlling variable

Vary at least one property of the data and measure how the number of features the model represents changes with it. Write down your prediction before running. Include a control model that, by design, should not show superposition.

### Part 3 — Explain the mechanism

Give an explanation that quantitatively predicts "how many features get stored," and test it experimentally. Say clearly over what range it holds and where it breaks down.

### Part 4 — Write it up

Write up the conclusions, the controls, alternative explanations, limitations, and "what you would do with another week."

## Reference solution

<details>
<summary>Show the reference solution</summary>

### Scope first: how to spend 72 hours

72 hours is not 72 working hours. After sleep and meals, roughly 20 to 25 hours are actually usable. The biggest risk in this problem is not failing to get results; it is that the question is too big: you could look for superposition in large language models, look for it with sparse autoencoders, or study its effect on robustness. Each of those is a month of work. So the first thing to do is narrow the question:

> In a minimal toy model, **which property of the data** determines how many features the model stores? Can we **calculate that number in advance**?

Once it is narrowed, what gets cut is also clear:

- No real language models and no sparse autoencoders. The question is "what controls superposition," and the variables are cleanest in a toy model.
- The main experiment sweeps only one variable: feature sparsity. A second variable (how fast feature importance decays) is exploratory only; if the results aren't clean, it goes under "limitations."
- No fancy visualizations. Two tables and one or two plots are enough.

| Time | What to do | Output |
| --- | --- | --- |
| Day 1 morning | Read the prompt, write down the problem definition and predictions; build the minimal model and confirm it trains at one sparsity | A one-page "definitions and predictions" note; working training code |
| Day 1 afternoon to evening | Sweep sparsity, add the linear control, 3 repeats; check whether conclusions depend on the threshold | Main results table |
| Day 2 | Derive the cost-benefit model, construct its solution directly, compare with trained results; run the training-steps control; rerun with 5 seeds | Mechanism table, controls table |
| Day 3 morning | Explore the second variable; stop if it isn't clean | A paragraph in the appendix or under "limitations" |
| Day 3 afternoon | Write the report; rerun the code from scratch and check every number in the report one by one | Report + code |

### Part 1 — The minimal model and how to measure it

**Setup.** Follow the setup of the first basic experiment in Elhage et al.'s 2022 Toy Models of Superposition (they also use $n = 20$, $m = 5$), except that the importance decay is changed from $0.7^i$ to $0.8^i$:

- $n = 20$ features. Each feature is present with probability $p = 1 - S$, in which case its value is drawn from $U[0, 1)$; otherwise it is 0. $S$ is the sparsity. Features are independent of each other.
- Model: $h = Wx \in \mathbb{R}^5$, $\hat{x} = \mathrm{ReLU}(W^\top h + b)$. The hidden layer has $m = 5$ dimensions, and the only nonlinearity is the ReLU at the output.
- Loss: $L = \mathbb{E}\sum_i I_i (x_i - \hat{x}_i)^2$, with importance $I_i = 0.8^{i}$ ($i$ starting from 0). Importance decreases, so there is no ambiguity about "which 5 should be stored first."
- Control: the linear model without the ReLU, $\hat{x} = W^\top W x + b$. A linear model can at best do what principal component analysis (PCA) does, and by design should store only 5 features.

**What "storing a feature" means.** Primary definition: $\|W_i\|^2 > 0.5$ ($W_i$ is the $i$-th column). Backup definition: the reconstruction $R^2$ of that feature is $> 0.5$. The two definitions should be checked against each other.

**Baseline.** Store only the 5 most important features and output the mean for the other 15. This is the best you can do in 5 dimensions without superposition, with loss $\sum_{i \ge 5} I_i \operatorname{Var}(x_i)$. Divide the loss by this baseline: a value below 1 means the extra stored features genuinely help, rather than norms just getting bigger.

**Implementation.** 2 model types × 9 sparsities × 3 repeats, 54 models in total. The loss is the sum of the models' losses and there is no coupling of any kind between models, so they can all go into one tensor and be trained in a single run. Adam, learning rate 0.01, cosine annealing, 4,000 steps, 256 fresh samples per step. Evaluation uses a separate 20,000 samples. Single thread only, so results reproduce bit for bit.

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

### Part 2 — Sparsity controls superposition

The prediction written down before running: the sparser the features, the less often two of them appear together, and the lower the cost of sharing a direction, so the ReLU model will store more features as sparsity increases; the linear model will store only 5 no matter how sparse.

Results (`SEED = 0`, 3 independently trained models per cell):

| Sparsity $S$ | ReLU: features stored (3 runs) | ReLU mean | Linear: features | ReLU loss / baseline | Linear loss / baseline |
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

How to read it:

- **No superposition when dense.** At $S \le 0.3$ the ReLU model behaves like the linear model, storing exactly 5, with loss equal to the baseline.
- **The sparser, the more it stores.** From $S = 0.5$ on, the ReLU model stores more than 5 features, reaching 14.33 on average at $S = 0.98$. With other seeds the means of two adjacent levels occasionally invert, but by less than 1. At $S \ge 0.8$ every model stores at least 9. The linear control stores exactly 5 at every sparsity, with loss within 2% of the baseline. Every ReLU model stores the 5 most important features.
- **The extra features are useful.** At $S = 0.98$ the ReLU model's loss is only 0.29 of the baseline. But at $S = 0.5$ to 0.6 the loss is about the same as the baseline: storing two or three extra features bought almost nothing. Part 3 comes back to this.
- **The conclusions don't depend on the threshold.** The feature count from $R^2 > 0.5$ differs from the norm-based count by at most 1 across all 54 models. Each ReLU model has at most 2 features with $\|W_i\|^2$ between 0.2 and 0.8; the rest are either close to 0 or close to 1. So a threshold of 0.5, 0.3, or 0.7 gives nearly the same count.

### Part 3 — A cost-benefit model that predicts the feature count

"Sparser means more" isn't enough on its own. The reviewers will ask: can you calculate in advance how many get stored? Here is a model you can derive with just pen and paper.

**Antipodal pairs.** Let two features $a$ and $c$ share one hidden direction with opposite signs: $W_a = e$, $W_c = -e$, bias 0.

- When only $a$ is present, $\hat{x}_a = \mathrm{ReLU}(x_a) = x_a$ and $\hat{x}_c = \mathrm{ReLU}(-x_a) = 0$; both are correct. The ReLU cuts off the negative interference.
- When both are present (probability $p^2$), $\hat{x}_a = \mathrm{ReLU}(x_a - x_c)$, and both features have error $\min(x_a, x_c)^2$. For the minimum of two independent $U[0,1)$ variables, the expected square is $1/6$.

So the cost of a pair is $p^2 (I_a + I_c)/6$. Not storing $c$ and just outputting its mean costs $I_c \operatorname{Var}(x_c) = I_c\,(p/3 - p^2/4)$. Pairing is better if and only if

$$\frac{p^2 (I_a + I_c)}{6} < I_c\left(\frac{p}{3} - \frac{p^2}{4}\right) \iff p < p^* = \frac{4r}{2 + 5r},\qquad r = \frac{I_c}{I_a}.$$

At most 5 pairs fit in 5 dimensions, so at most 10 features. The $k$-th pair should match the feature ranked $6-k$ with the one ranked $5+k$, giving $r = 0.8^{2k-1}$. For each sparsity, take the $k$ with the lowest total cost; the predicted feature count is $5 + k$.

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
thresholds = [f"| {k} | {N_HID + 1 - k} and {N_HID + k} | {r:.3f} | {ps:.3f} | {1 - ps:.3f} |"
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

Where each pair starts to pay off:

| Pair $k$ | Paired features (by rank) | $r = I_c / I_a$ | $p^*$ | Sparsity threshold $S^* = 1 - p^*$ |
| --- | --- | --- | --- | --- |
| 1 | 5 and 6 | 0.800 | 0.533 | 0.467 |
| 2 | 4 and 7 | 0.512 | 0.449 | 0.551 |
| 3 | 3 and 8 | 0.328 | 0.360 | 0.640 |
| 4 | 2 and 9 | 0.210 | 0.275 | 0.725 |
| 5 | 1 and 10 | 0.134 | 0.201 | 0.799 |

First check the derivation itself: construct this paired solution directly and measure its loss on 200,000 samples; it is within 3% of the formula. Then compare the trained models against it:

| Sparsity $S$ | Pair-model prediction | Trained (mean) | Trained loss / constructed loss | Antipodal pairs (3 runs) | Mean bias of stored features |
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

An "antipodal pair" means two stored features whose directions have cosine below −0.9. The table splits into three regimes:

1. **Agreement at both ends: $S \le 0.3$ and $S = 0.8$, 0.9.** The mean feature count is within 1 of the prediction, and the loss is within 10% of the constructed solution. At $S = 0.8$ every model has at least 4 antipodal pairs and the biases are close to 0, which is exactly what the paired solution looks like. Note that $S = 0.8$ sits right on the fifth pair's threshold of 0.799, so 9 versus 10 is nearly a tie.
2. **Disagreement in the transition: $S = 0.5$ to 0.7.** The trained models store roughly 0 to 2 more features than predicted, yet their loss is no lower for it. For this seed it is higher than the constructed solution at all three levels (1.07 at $S = 0.6$); with other seeds, $S = 0.5$ and 0.7 sometimes match the constructed solution, but $S = 0.6$ is higher in all 5 seeds. In other words, gradient descent stops at a solution that stores more but is not better. The pair model describes "roughly what the optimum looks like"; training doesn't necessarily find it.
3. **Beyond the model at high sparsity: $S \ge 0.95$.** The trained models store more than 11 on average, exceeding the pair model's ceiling of 10. At $S = 0.98$, every model exceeds 10 (mean 14.33); the loss is lower than the constructed solution (0.86); antipodal pairs actually become rarer (at most 2); and the bias turns negative (-0.20). This shows there is a better arrangement than "opposite pairs": several features share a subspace at non-antipodal angles, each leaking a little positive interference into the others, and the negative bias pushes that interference below the ReLU's zero point. This is consistent with the pentagon-like geometric structures Elhage et al. describe, but we did not check the geometry directly, so we can only say "consistent."

### Part 4 — How to read these results

**What the results show.** In this toy setup, sparsity controls superposition: the number of features the ReLU model stores rises overall with sparsity (occasionally dipping by less than 1 between adjacent levels), from 5 to around 14; the linear control is always 5. At the dense end ($S \le 0.3$) and at $S = 0.8$ to 0.9, a cost-benefit model of antipodal pairs predicts the feature count to within 1. It predicts the first extra feature appears at $S > 0.467$, consistent with the observed onset (between 0.3 and 0.5), but our grid is too coarse to test this more precisely.

**What the results don't show.**

- They say nothing about real networks. Here there is only one ReLU at the output, features are independent, and values are uniformly distributed.
- They don't explain why training fails to reach the optimum in the transition region. "Local optimum" is just the most likely explanation; it has not been proven.
- They don't establish the role of importance. We tried changing how fast importance decays: when it decays very fast or very slowly, the pair model's predictions deviate clearly from the trained results. We didn't track down why within three days, so it isn't in the conclusions.
- Only one scale was swept ($n = 20$, $m = 5$). The threshold locations will change with $n$, $m$, and importance, so the specific numbers here can't be extrapolated.

**Alternative explanations and the corresponding controls.**

| Alternative explanation | Control | Result |
| --- | --- | --- |
| The feature count grows only because the ReLU model trains erratically and norms grow at random | The linear model trained the same way; loss divided by the baseline | The linear model is always 5 with loss equal to the baseline; the extra features the ReLU stores bring the loss down to 0.29 of the baseline |
| The 0.5 threshold was conveniently chosen | The $R^2$ definition; look at the distribution of norms | The two definitions differ by at most 1; each model has at most 2 features with norms between 0.2 and 0.8 |
| Undertrained: a different number of steps would change the conclusion | Rerun with half the training steps | Each model's feature count differs by at most 2, and the upward trend with sparsity is unchanged |
| This seed just got lucky | 3 repeats per cell; change `SEED` to 1 through 4 and rerun the whole page | All qualitative checks in the verification code below pass for `SEED = 0` through 4 |
| The pairing formula is derived wrong | Construct the paired solution directly and measure its loss on 200,000 samples | Within 3% of the formula |

```python
t0 = time.time()
W_half, _ = train([p_all[i] for i in relu_idx], [True] * len(relu_idx), importance, N_HID, STEPS // 2, BATCH, LR, SEED)
print(f"half-step control: {time.time() - t0:.1f} s")
count_half = dict(zip(relu_idx, ((W_half ** 2).sum(1).numpy() > 0.5).sum(1)))
half_diff = [int(count_half[i] - count_norm[i]) for i in relu_idx]
print("half - full:", half_diff, "max |diff| =", max(abs(d) for d in half_diff))
print("formula check:", {s: round(v, 3) for s, v in formula_ratio.items()})
```

**Limitations, and what to do with another week.**

- **The optimization problem in the transition region.** Continue training from the constructed paired solution: if it stays put, it is a minimum of the training loss that random initialization just doesn't find. Then add multiple restarts, take the best one, and see whether the feature count returns to the predicted value.
- **The geometry at high sparsity.** Compute each feature's "dimensionality" $D_i = \|W_i\|^2 / \sum_j (\hat{W}_i \cdot W_j)^2$ ($\hat{W}_i$ is $W_i$ normalized to unit length), check whether fractions like 2/5 show up, and examine the pentagon structure directly.
- **Importance as a variable.** Find out why the pair model fails when decay is very fast or very slow; it may be necessary to add "three or more features sharing a direction" to the cost model.
- **Setups closer to real networks.** Add a ReLU to the hidden layer too; correlated features; larger $n$ and $m$, to see how the thresholds scale.
- **The learning dynamics.** Record when each feature "enters" the representation during training, and see whether it happens in order of importance.

### Report outline

1. **Question and definitions** (half a page): the narrowed question; the two definitions of "storing a feature"; the baseline.
2. **Setup** (half a page): data, model, loss, control, hyperparameters, seeds.
3. **Main results** (one page): the sparsity vs feature-count table, with one plot of "mean feature count vs sparsity," the linear control on the same plot.
4. **Mechanism** (one page): the pair-model derivation, the threshold table, and the comparison table against trained results.
5. **Controls and alternative explanations** (half a page): the five-row table above.
6. **Where it breaks down, and limitations** (half a page): the transition region, the high-sparsity region, importance, scale.
7. **Next steps** (a few lines).
8. **Appendix**: full hyperparameters, per-seed results, reproduction commands.

For the 10-minute presentation, cover only sections 3, 4, and 6: one main plot, one formula, two places where it breaks down.

### Follow-up questions

- **Why give features different importances? What happens if they are all equal?** Decreasing importance removes ambiguity about "which 5 to store," and also staggers the thresholds of the pairs, so the feature count climbs one at a time, which makes it easy to compare with the prediction. If they were all equal, $r = 1$, all five pairs would have threshold $p^* = 4/7$, and the pair model predicts the feature count jumps straight from 5 to 10. We did not run this setup, so it is only a prediction.
- **Why does the linear model always store just 5?** For a linear model, interference from superposition can't be removed: if two features share a direction, whenever one is present it leaves an equal error on the other's output. The optimal linear reconstruction projects onto the 5-dimensional subspace with the largest weighted variance, which here is the 5 most important features. The ReLU's role is to clip negative interference to 0, and antipodal pairing exploits exactly that.
- **In the transition region the trained solution is worse than the constructed one. What makes you say it is a local optimum rather than a problem with the optimizer settings?** We can't conclude that yet. The half-steps control only shows that training longer with the same optimizer barely changes the feature count. To tell the two possibilities apart: train starting from the constructed solution and see whether it is stable; change the learning rate and use multiple restarts to see whether a better solution can be found. Only after doing both can we say which it is.
- **Why is the bias negative at $S = 0.98$?** With antipodal pairs, the interference is negative and the ReLU clips it by itself, with no bias needed. When three or more features share a subspace at non-antipodal angles, one feature being present causes positive interference on the others. A negative bias pushes that interference below 0, at the cost of also clipping small values when the feature really is present. The sparser the features, the more of them get packed into the same subspace; and the more often interference occurs relative to how often the feature itself actually appears, the better this trade becomes.
- **Does this conclusion generalize to large models?** Not directly. What it shows is: when features are sparse and there is a nonlinearity that can clip interference, superposition pays off. How sparse and how numerous the features are in real models, we don't know. That is exactly what dictionary learning and sparse autoencoders are for estimating. The toy model gives a testable hypothesis, not a conclusion.
- **If you had only 24 hours, what would you cut?** Keep the sparsity sweep, the linear control, and the threshold check; that is the minimum for the main conclusion. Keep the pair model's derivation and threshold table, but skip the comparison with the constructed solution. Cut the half-steps control and the multi-seed reruns, and state in the report "only one seed was run."

<details>
<summary>Verification code (runnable)</summary>

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
    "| 1 | 5 and 6 | 0.800 | 0.533 | 0.467 |",
    "| 2 | 4 and 7 | 0.512 | 0.449 | 0.551 |",
    "| 3 | 3 and 8 | 0.328 | 0.360 | 0.640 |",
    "| 4 | 2 and 9 | 0.210 | 0.275 | 0.725 |",
    "| 5 | 1 and 10 | 0.134 | 0.201 | 0.799 |",
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
