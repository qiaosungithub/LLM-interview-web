# Training curves: a minimal PyTorch loop, three planted bugs, and the stability threshold

## Problem

This round takes place in a shared notebook and tests "ML research fundamentals": write a very small training loop in PyTorch, understand the curves it produces, and **say what the plot will look like before you draw it**. The interviewer provides the data and the model; use them as they are, without changes.

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

The loss is always the mean squared error, averaged over samples, with no $\frac12$: $\frac1n\sum_i (f(x_i) - y_i)^2$. CPU only. Plot with matplotlib, with a log-scale vertical axis throughout.

### Part 1 — A minimal training loop and its curves

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

1. Implement `evaluate` and `train`, and train once with `train(make_model(0), X_train, y_train, X_val, y_val)`.
2. Draw one plot with a log-scale vertical axis. Plot two curves: `history["val"]` at x-coordinates $0, 1, \dots, 40$; `history["train"]` is an average over a whole epoch, so plot it at $0.5, 1.5, \dots, 39.5$. Add three horizontal reference lines, using population values: $\operatorname{Var}(y)$, the mean squared error of the best linear model (with intercept), and the noise variance $0.1^2 = 0.01$.
3. Before plotting, answer:
   - (a) What is each of the three reference lines? Derive the exact values from the formula in `make_data`.
   - (b) What shape does the validation curve have? Roughly where does it start, at which epoch does it cross the linear-model line, and where does it end up?
   - (c) Which is higher, the training curve or the validation curve? Once plotted, measure the reason with a few lines of code, and give a formula that exactly explains the difference between them.

### Part 2 — A colleague's training script

A colleague wrote the function below with the same data and model. They ran `train_colleague(make_model(0), X_train, y_train, X_val, y_val)` once and say the curves look wrong, and at the end it even produced `nan`.

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

This code has 3 bugs, stacked on top of each other. Find them one at a time: at each step, first plot the training and validation curves of the current version (same axes as Part 1), say what you see in the plot and where you guess the bug is, confirm it with a few lines of measurement code, then fix it and look at the next plot.

After all the bugs are fixed, the `history["train"]` returned by `train_colleague` should equal the `history["train"]` returned by Part 1's `train` element by element, and `history["val"]` should equal Part 1's `history["val"][1:]` element by element (same seed, exactly the same results, not just close).

### Part 3 — Gradient checking and the learning-rate threshold

Switch to a linear model. Use the Part 1 training set, converted to float64: `X = X_train.double()`, `y = y_train.double()`. The parameters $\theta = (w_0, \dots, w_7, b)$ are 9 numbers, and the loss is

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

1. Take `theta0 = torch.randn(9, generator=torch.Generator().manual_seed(0), dtype=torch.float64)`. Compare three gradients: `linear_grad`, `fd_grad(eps=1e-6)`, and autograd, with relative error defined as $\lVert a - b \rVert / \lVert b \rVert$, taking $b$ to be `linear_grad`. Then switch `X`, `y`, and `theta0` all to float32 and do finite differences with `eps=1e-6` and with `eps=1e-2`. First predict the order of magnitude of these errors, then compute them and explain.
2. Derive the Hessian $H$ by hand and check it with `torch.autograd.functional.hessian`. Find the largest eigenvalue $h_1$, the condition number $\kappa(H)$, and what you believe is the learning-rate stability threshold $\eta^*$.
3. Let $\hat\theta$ be the least-squares solution. Take $\eta = c\,\eta^*$ with $c \in \{0.9,\ 1.0,\ 1.1\}$ and run 100 steps of each. Draw one plot: horizontal axis $t = 0..100$, log-scale vertical axis, all three curves showing $L(\theta_t) - L(\hat\theta)$ (anything below $10^{-15}$ is rounding error and can be ignored).
4. Before plotting, answer:
   - (a) What shape does each of the three curves have? What factor is applied per step late on? Where does the $c = 1.0$ curve end up (give the formula and the number)?
   - (b) State the threshold as a theorem: under what condition does gradient descent converge from any starting point?
   - (c) If the loss is changed to `reduction="sum"` (summed over samples instead of averaged), what does the threshold become?

Reference numbers for the default data:

```text
L(theta_hat) ≈ 0.2893
h_1          ≈ 2.453
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

First, paste in the code the problem provides, unchanged (plus the imports needed for plotting). All plots are saved to a temporary directory; `plt.show()` is never called.

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

**Approach.** The training loop itself is not hard; what is easy to get wrong is a handful of details:

- The model's output has shape `(B, 1)` and the labels have shape `(B,)`. You need `squeeze(-1)` before subtracting, otherwise it broadcasts to `(B, B)`.
- Call `zero_grad()` at every step. PyTorch's `backward()` **adds** gradients into `.grad`; it doesn't overwrite them.
- `evaluate` has to switch to `eval()` mode (turning dropout off), use `torch.no_grad()` (no computation graph), and restore the original mode afterwards.
- The training loss is averaged weighted by batch size. Here 1000 is exactly divisible by 50, so weighting makes no difference; it only matters when the last batch is not full.

Each epoch costs one forward and one backward pass, $O(n \cdot 8 \cdot 64)$.

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

**(a) The three reference lines.** Let $u = x_0 - x_1 \sim \mathcal N(0, 2)$. For $Z \sim \mathcal N(0, s^2)$, $\mathbb E[\mathrm{relu}(Z)] = s/\sqrt{2\pi}$ and $\mathbb E[\mathrm{relu}(Z)^2] = s^2/2$. The four terms of $y$ use only $(x_0, x_1)$, $x_2$, $x_3$, and the noise respectively, so they are independent and their variances simply add:

$$\operatorname{Var}(y) = \underbrace{\left(1 - \tfrac{1}{\pi}\right)}_{\mathrm{relu}(u)} + \underbrace{\left(\tfrac12 - \tfrac{1}{2\pi}\right)}_{\mathrm{relu}(x_2)} + \underbrace{\tfrac14}_{0.5 x_3} + \underbrace{0.01}_{\text{noise}} = 1.76 - \frac{3}{2\pi} \approx 1.2825 .$$

The best linear model is the projection of $y$ onto the space spanned by $\{1, x_0, \dots, x_7\}$. The inputs used by the different terms are disjoint and independent, so each term can be projected separately. The regression coefficient of $\mathrm{relu}(u)$ on $u$ is $\mathbb E[u\,\mathrm{relu}(u)]/\mathbb E[u^2] = (s^2/2)/s^2 = \frac12$, and the coefficient of $\mathrm{relu}(x_2)$ on $x_2$ is likewise $\frac12$. So the best linear model is

$$\hat y = \tfrac12 (x_0 - x_1) - \tfrac12 x_2 + \tfrac12 x_3 + \frac{\sqrt2 - 1}{\sqrt{2\pi}}, \qquad \text{intercept} \approx 0.165 .$$

The variance of the linear part is $\frac14 \cdot 2 + \frac14 + \frac14 = 1$, and what remains is the linear model's mean squared error: $0.76 - \frac{3}{2\pi} \approx 0.2825$. The noise floor is $0.01$; no model's expected error can go below it.

**(b) The shape of the validation curve.** At initialization the network's outputs are small, so the validation error is roughly $\mathbb E[y^2]$ plus the variance of the random outputs, near or slightly above $\operatorname{Var}(y)$ ($1.39$ with `seed=0`). After the first epoch it is already down to $0.21$, below the linear model's $0.28$: the relu network quickly learns the linear part, plus some of the kinks. After that it falls more and more slowly and ends up around $0.020$, about twice the noise floor. The validation curve never goes below $0.01$.

**(c) Why the training curve is always higher.** Once plotted, you will see that over the 40 epochs **every single point** of the training curve is above the validation curve, ending at $0.053$ versus $0.020$. This is not underfitting, and nothing is wrong with the data. There are two reasons:

- The training loss is an average over the batches of an epoch, and the model keeps improving during the epoch, so it is higher than the loss at the end of the epoch. This only matters in the first few epochs.
- The training loss is computed in `train()` mode, with dropout on; the validation loss is computed in `eval()` mode. Late in training, almost all of the gap comes from this.

How to measure it: compute the loss on the training set in `eval()` mode (that is `history["train_eval"]`); it ends at $0.017$, about the same as the validation set's $0.020$. So the model performs almost identically on the training and validation sets, and the gap comes from dropout itself.

The exact formula: dropout sits only before the last linear layer. Let the hidden activations be $h_j$, the last layer's weights $w_j$, and the masks $m_j \sim \mathrm{Bernoulli}(1-p)$ independent. Then

$$f_{\text{drop}}(x) = b + \sum_j w_j h_j \frac{m_j}{1-p}, \qquad \mathbb E\left[\frac{m_j}{1-p}\right] = 1, \quad \operatorname{Var}\left[\frac{m_j}{1-p}\right] = \frac{p}{1-p} .$$

So $\mathbb E[f_{\text{drop}}] = f_{\text{eval}}$, and using "squared mean plus variance":

$$\mathbb E_m\left[(f_{\text{drop}}(x) - y)^2\right] = (f_{\text{eval}}(x) - y)^2 + \frac{p}{1-p} \sum_j (w_j h_j)^2 .$$

This is an identity: it holds exactly for every sample, with no approximation. The average of the second term on the right over the training set is $0.036$ with `seed=0`, exactly the difference $0.053 - 0.017$ between the last point of the training curve and `train_eval`.

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

The three bugs are listed in the order they show up in the plots. For each one: where it is, what goes wrong, on what input it shows, and how to fix it.

**First plot (run as is): the loss never goes down, after a few epochs it climbs faster and faster, and finally becomes `nan`.** With `seed=0` the training loss never drops below 1.3 (about as high as $\operatorname{Var}(y)$), climbs to 3.2 in the later epochs, and hits `nan` at epoch 10. A loss that first fails to fall and then blows up makes the gradient itself the first suspect.

**Bug A: no `opt.zero_grad()` in the loop.** `loss.backward()` adds this step's gradient to `.grad`, so `.grad` is the sum of all gradients from the first step until now, and every step moves along this ever-growing sum. Measurement: after two steps, `p.grad` equals the sum of the two batches' gradients, not the second batch's gradient (the verification code includes this comparison). Mathematically it is exactly equivalent to `torch.optim.SGD(momentum=1.0)`, the heavy-ball method with no damping. Fix: add `opt.zero_grad()` before `backward()`.

**Second plot (A fixed): both curves lie flat on the $\operatorname{Var}(y)$ reference line.** The training loss sits at $1.32$ and validation at $1.26$, barely moving over 40 epochs. Now look at the model's outputs on the training set: their standard deviation is only $0.03$, while the standard deviation of $y$ is $1.14$; the model is outputting almost a constant. The network is clearly capable of reaching $0.02$ but has only learned the mean, which means it isn't optimizing the loss we think it is at all.

**Bug B: in `mse`, `(B, 1)` minus `(B,)` broadcasts to `(B, B)`.** Measurement: print `(model(xb) - yb).shape` and you get `(50, 50)`. This "loss" is the mean squared difference between every prediction and every label, which decomposes as

$$\frac{1}{B^2}\sum_{i,j} (f_i - y_j)^2 = \frac1B \sum_i (f_i - \bar y)^2 + \frac1B \sum_j (y_j - \bar y)^2 .$$

The second term doesn't depend on the model, and the first is minimized when $f_i \equiv \bar y$. So the model learns to output the constant $\bar y$ and the loss sits at the variance of $y$, exactly as in the plot. The validation set uses the same function, so the validation curve also sits at $\operatorname{Var}(y_{\text{val}})$. Fix: `pred.squeeze(-1) - target`.

**Third plot (A and B fixed): the training curve looks about like Part 1, but the validation curve is also around $0.06$, and jumpy.** In Part 1 the validation curve ended at $0.020$; now it is as high as the training curve (the last point is $0.062$ with `seed=0`). Part 1 (c) already measured that the part of the training curve above the eval-mode loss is exactly the $0.036$ due to dropout. The validation curve is now higher by about that much too, which naturally suggests dropout is also on during validation.

**Bug C: validation uses only `torch.no_grad()`, without switching to `model.eval()`.** `no_grad` only skips building the computation graph; it doesn't turn dropout off. Measurement: at this point `model.training` is `True`; computing the same model on the validation set twice in a row gives different results; computing with `evaluate` (eval mode) gives $0.023$, which differs from the wrong value by $0.039$, close to the dropout term on the validation set, $0.038$. Fix: `model.eval()` before validation, and `model.train()` to switch back afterwards.

**Why, once fixed, it matches Part 1 bit for bit.** Both pieces of code consume random numbers in exactly the same order: `torch.manual_seed(seed)` at the start, a separate generator `g` for shuffling, and dropout masks drawn only in the training forward pass, never in eval mode. Bug C has a side effect too: drawing masks during validation shifts the random numbers used by subsequent training, so with only A and B fixed, even the trained model differs from Part 1.

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

Below, the colleague's original code is put verbatim into a string, the bugs are fixed one by one with string replacements, and the plots for all four stages are drawn. Each replacement asserts that it matches exactly once.

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

**The gradient.** Let $\tilde X = [X,\ \mathbf 1]$ (append a column of ones) and the residual $r = \tilde X \theta - y$. Then $L = \frac1n \lVert r \rVert^2$,

$$\nabla L(\theta) = \frac2n \tilde X^\top r, \qquad H = \nabla^2 L = \frac2n \tilde X^\top \tilde X .$$

Note the factor of 2: the loss has no $\frac12$, so the Hessian is **2 times** the second-moment matrix of $[x, 1]$.

**The error of finite differences.** Central differences have two kinds of error:

- Truncation error $O(\varepsilon^2)$, from the third derivative. $L$ is quadratic, so its third derivative is zero, and the truncation error here is **exactly 0**; a larger $\varepsilon$ does no harm.
- Rounding error: $f$ itself carries an error of about $u\,|f|$ ($u$ is the unit roundoff, about $1.1 \times 10^{-16}$ for float64 and about $6 \times 10^{-8}$ for float32), which dividing by $2\varepsilon$ amplifies to $u\,|f|/\varepsilon$.

At `theta0`, $L \approx 9$ and the gradient norm is about 6, so the relative error is roughly $u\,|f| / (\varepsilon \lVert \nabla L \rVert) \approx 1.5\,u/\varepsilon$. For float64 with $\varepsilon = 10^{-6}$ that is on the order of $10^{-10}$ (measured: $3.6 \times 10^{-10}$). Autograd and the hand-derived formula compute the same expression and differ only by rounding (measured: on the order of $10^{-15}$). For float32 with $\varepsilon = 10^{-6}$ the rounding error is amplified to within an order of magnitude of the gradient, and the result is unusable (measured: $0.15$). For float32 with $\varepsilon = 10^{-2}$ it drops to the order of $10^{-5}$ (measured: $2.4 \times 10^{-5}$). So gradient checks should be done in float64; for general non-quadratic functions, $\varepsilon$ around $u^{1/3}$ balances the two kinds of error.

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

**The threshold.** One step of gradient descent is $\theta_{t+1} - \hat\theta = (I - \eta H)(\theta_t - \hat\theta)$. Project the error onto the eigenvectors of $H$, $a_i(t) = u_i^\top(\theta_t - \hat\theta)$; each component evolves as its own geometric sequence:

$$a_i(t) = (1 - \eta h_i)^t\, a_i(0), \qquad L(\theta_t) - L(\hat\theta) = \tfrac12 (\theta_t - \hat\theta)^\top H (\theta_t - \hat\theta) = \sum_i \tfrac12 h_i\, a_i(t)^2 .$$

**(b) The theorem.** Assume $H$ is positive definite (here $\tilde X$ has full column rank). Gradient descent with a fixed step size converges to $\hat\theta$ from any starting point if and only if $|1 - \eta h_i| < 1$ for all $i$, i.e.

$$0 < \eta < \eta^* = \frac{2}{h_1} = \frac{2}{\lambda_{\max}(H)} .$$

At $\eta = \eta^*$, the factor in the $h_1$ direction is $-1$: that component flips sign every step while keeping its size, neither converging nor diverging. As long as $h_1$ is strictly larger than the other eigenvalues, all other components decay.

With `seed=0`, $h_1 = 2.453$ and $\eta^* = 0.8153$. A common pitfall is forgetting the factor of 2 and writing the threshold as $2/\lambda_{\max}(\tilde X^\top \tilde X / n)$. That value is twice the true threshold; the factor in the steepest direction becomes $1 - 4 = -3$, and it diverges within a few steps.

Why the eigenvalues of $H$ are all near 2 ($\kappa(H) = 1.44$): the inputs are independent standard normals plus a column of ones, so $\tilde X^\top \tilde X / n \approx I_9$ and $H \approx 2I$. The finite sample spreads the eigenvalues out, over a range of about $2(1 \pm \sqrt{d/n})^2 \approx [1.64,\ 2.40]$ ($d = 9$, $n = 1000$; this is the edge of the Marchenko–Pastur distribution). With `seed=0` the measured range is $[1.70,\ 2.45]$.

**(a) The three curves.** With $\eta = c\,\eta^*$, the factor in the steepest direction is $1 - 2c$. In the other directions, since $h_i < h_1$, the factor is $1 - 2c\,h_i/h_1$.

- $c = 0.9$: the factor in the steepest direction is $-0.8$. The factors in the other directions all lie between $-0.8$ and about $-0.25$ (since $h_i/h_1 \ge 0.69$), so they decay faster. So late on each step multiplies the loss by $0.8^2 = 0.64$: a straight line with slope $\ln 0.64 = -0.45$, dropping about one order of magnitude every 5 steps, and hitting the rounding floor near $10^{-16}$ after seventy or eighty steps.
- $c = 1.0$: the steepest direction is multiplied by $-1$ each step, so its share of the loss, $\frac12 h_1 a_1(0)^2$, never changes; all other directions decay. So the curve falls first, then flattens out at $\frac12 h_1 a_1(0)^2$. With `seed=0` that value is $0.151$, starting from $1.058$.
- $c = 1.1$: the steepest direction is multiplied by $-1.2$ each step, so the loss is multiplied by $1.44$ each step; late on it is a straight line with slope $\ln 1.44 = +0.36$, reaching the order of $10^{15}$ after 100 steps. Directions with $h_i > h_1/1.1$ also diverge, just more slowly (with `seed=0` the second direction is multiplied by $-1.02$ each step). In the first few steps the curve goes down first, because the error in the other directions is decaying; with `seed=0` the minimum is at step 2.

A learning rate only 10% above the threshold turns the loss from "converges fast" to "blows up exponentially." The behavior near the threshold changes abruptly.

**(c) `reduction="sum"`.** The summed loss is $n$ times the mean, and so are the gradient and the Hessian, so the threshold becomes $\eta^*/n = 8.2 \times 10^{-4}$. Divide the learning rate by $n$ at the same time and the trajectory is exactly the same as with the mean loss. Conversely, switching to the sum while keeping the learning rate, the factor in the steepest direction is $1 - 0.9 \times 2 \times 1000 = -1799$, and it flies off on the first step.

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

### Follow-up questions

**1. When fixing bug C, what happens if you add `model.eval()` but forget to switch back with `model.train()`?** From the second epoch on, dropout stays off, which amounts to training a network without dropout. The training curve falls to $0.012$ and ends up below the validation curve; the validation curve ends at $0.017$, actually better than Part 1's $0.020$. This shows that $p = 0.2$ dropout doesn't help on this task: 1000 samples, a smooth target, and a small network don't overfit much to begin with. Incidentally, after a "fix" like this the history no longer matches Part 1.

**2. Why does a missing `zero_grad` produce `nan`?** It is equivalent to `SGD(momentum=1.0)`. On a quadratic with full batches, each component satisfies $a_{t+1} = (2 - \eta h)\,a_t - a_{t-1}$. When $0 < \eta h < 4$, the solution is an undamped oscillation:

$$a_t = a_0 \, \frac{\cos(\omega (t + \frac12))}{\cos(\omega/2)}, \qquad \cos\omega = 1 - \frac{\eta h}{2} .$$

Each component's amplitude is $|a_0| / \sqrt{1 - \eta h / 4}$, slightly larger than its initial value, and it never decays. So the loss oscillates between 0 and slightly above its initial value, neither converging nor diverging. Measured over 2000 full-batch steps, the loss never exceeds 1.1 times its initial value, and even in the last 200 steps it still comes back above half its initial value. With minibatches, the gradient and curvature change at every step, and the undamped oscillation not only fails to die down but gets amplified. The same linear model run for 800 minibatch steps (equivalent to 40 epochs) sees its loss grow by more than a factor of $10^6$. The MLP behaves the same way: in the first plot of Part 2, `nan` appears at epoch 10.

**3. What happens if the training loop's loss is changed to `reduction="sum"`?** The gradient becomes 50 times larger, the effective learning rate goes from 0.1 to 5, and it is `nan` in the first epoch. Divide the learning rate by 50 and it is exactly the same as before. Only the printed training loss is 50 times larger, and the validation loss (summed over 1000 samples) is 1000 times larger.

**4. Can you also do a finite-difference check on the MLP?** Yes, but watch three things: use `eval()` mode, otherwise the two forward passes have different dropout masks and the difference is completely meaningless (measured relative error above $10^5$); use float64; and ReLU has kinks, so any coordinate where $\varepsilon$ crosses a kink will be wrong, so check only a few coordinates, or try again at a different random point. Done this way, the relative error is below $10^{-8}$.

**5. Does $2/\lambda_{\max}$ still hold for the MLP?** It is no longer a strict theorem, because the Hessian changes with the parameters; it can only be used locally around the current point. You can estimate $\lambda_{\max}$ with Hessian-vector products (double backward) plus power iteration, without building the Hessian. At the end of training, $\lambda_{\max}$ of the loss on the full training set in eval mode is about 9.8, giving a local threshold $2/\lambda_{\max} \approx 0.20$; the learning rate we used, 0.1, is about half of that.

<details>
<summary>Verification code (runnable)</summary>

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
