# Timed 3-hour research take-home: does context noise stabilise autoregressive rollouts?

## Problem

This is a timed take-home. Once you open the problem you have 3 consecutive hours to hand in code that runs end to end with a single command, plus a one-page report.
The research question, the data and the baseline are all given, and the new code is under 60 lines. What is being tested: can you run a small experiment cleanly in three hours and state the conclusions precisely?

**The research question.** A next-frame predictor sees only real frames as context during training; this is called teacher forcing.
When you use it to generate a long sequence, you feed its own predictions back in as the next input and roll forward step by step; this is called an autoregressive rollout.
During a rollout the inputs carry the model's own errors, which it never saw in training, so the errors grow as the rollout goes on. A common remedy is to add a little
Gaussian noise to the context frames during training while keeping the target frame clean. Answer the following on the small dataset below:

> Does adding noise to the context reduce long-horizon rollout error? From which step on does it help? What does it cost? What happens with too much noise?

**Data and baseline (given; do not change them).** The data is a one-dimensional “video”: each frame has 16 pixels, and the picture contains one Gaussian-shaped bright blob that moves at a constant speed
and bounces off the edges. The baseline is a small MLP that looks at the last 3 frames and predicts the next one.

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

Run it first to make sure your environment works:

```py
train_frames, test_frames, test_centres = load_data(seed=0)
model = train_baseline(train_frames, seed=0)
print(f"{one_step_mse(model, test_frames):.1e}")                    # 2.5e-05
context, target = windows(test_frames)
print(round(float(((context[:, -1] - target) ** 2).mean()), 3))    # 0.025 -- what "copy the last frame" gets
```

The baseline's one-step error is about a thousand times smaller than “just copy the previous frame”, which looks great. The question is whether it is still good after rolling out 30 steps.

**Timing and rules.**

- All that is known about this kind of problem is one sentence: a research scientist candidate said that one of their roughly ten interview rounds was a “take-home with a timer, 3 hours”.
  The platform, the content of the problem and whether you can pause were not recorded. The points below are assumptions made for practice, taking the most common and strictest setup.
- The clock starts when you open the problem and runs for 3 consecutive hours, with no pausing. When time is up, you hand in whatever you have.
- Hand in two files: `solution.py` (running `python solution.py` runs everything from scratch and prints every number that appears in the report) and `report.md` (at most one page).
- CPU only, PyTorch and NumPy only. Running all experiments from scratch takes no more than 10 minutes.
- No AI tools. DeepMind's official interview guide states that unless told otherwise, you should not use any AI tools during interview tasks. Whether you may look up documentation depends on what the recruiter
  tells you; for practice, we suggest consulting only the official PyTorch and NumPy documentation.
- Your numbers don't need to match the reference solution digit for digit, because a different random stream for the noise will produce differences. What should agree are the Part 1 numbers (there is no randomness
  you introduce yourself there; but on a different CPU or PyTorch version, floating-point rounding differs, so the one-step MSE printed by the starter code and the numbers at steps 20 and 30 may differ slightly
  in the second significant figure) and the direction of Part 2's main conclusions. Which conclusions are sensitive to the random numbers is itself something the report should cover.

### Part 1 — Rollout evaluation and reference levels

Implement the rollout:

```py
def rollout(model, frames, horizon):
    """frames: test clips, shape (n, K + H, W) with H >= horizon. Returns a numpy array (n, horizon, W)."""
```

- The first `K` frames (real frames) of each clip are the initial context.
- At each step: give the current context to `model` to get the next frame; then drop the oldest frame from the context and append the **frame just predicted** at the end.
- Time position $h$ of the return value ($h$ counting from 1) is the prediction for frame $K + h - 1$ of the clip (indices counting from 0).
- `model` can be any callable that maps a tensor of shape `(n, K, W)` to a tensor of shape `(n, W)`. Do not modify the `frames` passed in.

Example: let `model = lambda c: 2 * c[:, -1] - c[:, -2]`, i.e. straight-line extrapolation from the last two frames. Calling the first three frames of a clip `x0, x1, x2`, we get

```text
h = 1:  2*x2 - x1                          = x2 + 1*(x2 - x1)
h = 2:  2*(2*x2 - x1) - x2                 = x2 + 2*(x2 - x1)
h = 3:  2*(3*x2 - 2*x1) - (2*x2 - x1)      = x2 + 3*(x2 - x1)
```

All metrics are computed on the test clips, separately for $h = 1, \dots, 30$:

- $\text{MSE}@h$: the squared difference between the predicted frame and the real frame at step $h$, averaged over all test clips and all 16 pixels. It is the error of the frame at step $h$,
  not the cumulative error over the first $h$ steps.
- Position error: treat negative values in the predicted frame as 0 and take the average pixel index weighted by intensity, which gives the center of mass; take the absolute difference between it and the true center,
  averaged over clips. The unit is pixels.
- Mass ratio: treat negative values in the predicted frame as 0, sum the pixels, and average over clips; then divide by the average pixel sum of the real frames (averaged over all clips and all
  30 steps). A value of 1 means the blob's total brightness has not changed.

Two reference levels that require no learning:

- Copy the last frame: at every step, output the last frame of the context, i.e. frame $K - 1$ of the clip.
- Mean frame: at every step, output the per-pixel mean of all frames in the training set.

Deliverables: the MSE and mass ratio of the seed-0 baseline model at $h = 1, 5, 10, 20, 30$, the MSE of the two reference levels, and the step from which the baseline becomes worse than the “mean frame”.

### Part 2 — Training with context noise, as one controlled comparison

Implement noisy training:

```py
def train(frames, seed, noise_std):
    """Like train_baseline, but every batch of context frames gets fresh N(0, noise_std^2) noise. Targets stay clean."""
```

- Change exactly one thing: add i.i.d. $\mathcal N(0, \sigma^2)$ noise to the context of every batch, resampled for each batch; the target frames get no noise.
  Initial weights, batch order, number of steps and learning rate are all the same as in `train_baseline`.
- With `noise_std = 0`, the resulting weights must be **bit-for-bit identical** to those of `train_baseline`.
- Under the same seed, different values of $\sigma$ use the same initial weights and the same batch order.

Experiment grid: $\sigma \in \{0,\ 0.01,\ 0.03,\ 0.1\}$, seeds 0 to 4 (data from `load_data(seed)`), with each model rolled out 30 steps.
When summarizing over the 5 seeds, report both a central value and the spread. Whether to use the mean or the median is up to you; explain why.

If you have time left, add **one** control experiment that you think best rules out an alternative explanation. Just one.

### Part 3 — One-page report

`report.md` is at most one page, in three sections:

1. What you did: the experimental setup, which single factor you changed, and how you made sure the comparison is fair.
2. Results: one table; every conclusion comes with its uncertainty; be clear about which conclusions the data supports and which it doesn't.
3. If you had three more hours, what you would do next, in priority order.

**What the grading looks at.** Whether the comparison changes only one factor; whether there are reference levels; whether the numbers come with uncertainty; whether the conclusions go beyond the evidence; whether the code is reproducible with a single
command; and whether the time was spent answering the question.

## Reference solution

<details>
<summary>Show the reference solution</summary>

There is very little new code in this problem. The time goes mainly into three things: confirming the evaluation is correct, keeping the comparison clean, and stating the conclusions precisely.

### How to split the three hours

| Time | What to do | What you have at the end of it |
| --- | --- | --- |
| 0:00–0:15 | Read the problem. Run the starter code as is and confirm the one-step MSE matches the statement. Print two or three clips and look at them | A working environment, and a sense of what the data looks like |
| 0:15–0:50 | Part 1: `rollout`, the three metrics, the two reference levels. Test `rollout` with “fake models” first, then run the real model | Rollout error of the seed-0 baseline |
| 0:50–1:20 | Part 2: `train`. Verify that $\sigma = 0$ is bit-for-bit identical to the baseline weights. First run all four values of $\sigma$ on seed 0 only | Whether the effect exists at all |
| 1:20–1:50 | Run all 5 seeds (about 1 minute), put them in a table, decide between median and mean | The main table |
| 1:50–2:10 | One cheap control: clip the outputs during rollout. Then rerun $\sigma = 0.01$ twice, changing only the noise random stream (about half a minute) | The control table, and which conclusions are robust |
| 2:10–2:45 | Write the report | `report.md` |
| 2:45–3:00 | Rerun everything in a clean directory, check each number in the report one by one, package it up | Ready to submit |

Two rules of discipline. First, get the whole pipeline working on one seed before adding seeds; otherwise one evaluation bug wastes the time spent on every seed.
Second, write no new code in the last 15 minutes; only check and package.

### Part 1

Approach: the rollout itself is only a few lines; the error-prone part is how the context slides. Once it's written, don't run the real model yet; test it with two “fake models” whose answers are known:
the straight-line extrapolation from the statement (it uses the last two frames, so if the sliding is wrong the numbers won't match), and “copy the last frame” (which should output the same frame at every step). Then confirm
that step 1 of the rollout equals the model's one-step prediction on the real context. All three checks are in the checks at the end of this page.

First, copy the starter code given in the problem, exactly as in the statement:

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

Then the rollout, the metrics and the reference levels. `rollout` has an extra `clip` parameter, which the control experiment in Part 2 uses.

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
    rows = [("Baseline model", base0["mse"]), ("Copy last frame", refs0["copy"]["mse"]), ("Mean frame", refs0["mean"]["mse"])]
    lines = ["| " + name + " | " + " | ".join(f"{v[h - 1]:.2g}" for h in SHOW) + " |" for name, v in rows]
    lines.append("| Baseline mass ratio | " + " | ".join(f"{base0['mass'][h - 1]:.2f}" for h in SHOW) + " |")
    return lines


print("\n".join(part1_table()))
```

Results for seed 0 (columns are $h$):

| | 1 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| Baseline model | 2.4e-05 | 0.00087 | 0.013 | 0.22 | 1.9 |
| Copy last frame | 0.025 | 0.18 | 0.21 | 0.17 | 0.18 |
| Mean frame | 0.089 | 0.089 | 0.088 | 0.087 | 0.087 |
| Baseline mass ratio | 1.00 | 1.01 | 1.06 | 1.37 | 2.69 |

How to read it:

- At step 1, the baseline is about a thousand times better than “copy last frame”. At step 10 it is still far better than both reference levels.
- From step 16 on, the baseline's MSE exceeds the “mean frame”. On the other four seeds, this crossover falls between steps 15 and 19.
- At step 30, the baseline's MSE is 1.9, about 21 times the “mean frame”. The mass ratio is 2.69: the total brightness of the predicted frame has grown to 2.7 times the true value.
  So this is not just the position slowly drifting away; the amplitude is diverging. MSE alone can't tell these two failures apart; the mass ratio can.
- The “mean frame” reference level matters. Without it, a number like 0.22 tells you nothing about whether it's good or bad; with it, you know that by step 20 the model is already worse than learning nothing at all.

### Part 2

Approach: a clean comparison rests on three things.

1. **Pairing.** Under the same seed, the four values of $\sigma$ share the training set, test set, initial weights and batch order. The noise uses its own random number generator,
   so adding noise or not never changes the batch order. That way, the difference between two models under the same seed comes only from the noise. The `noise_seed` parameter is only for the final
   robustness check.
2. **The control group goes through the same code path.** $\sigma = 0$ also calls `train`, and we verify that its weights are bit-for-bit identical to those of `train_baseline`. Otherwise the “baseline”
   and the “method” might differ by some implementation detail nobody noticed.
3. **The grid is fixed in advance, and everything is reported.** There is no validation set, so we don't pick the “best $\sigma$” on the test set and then report its numbers.

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
    for label, name in (("Copy last frame", "copy"), ("Mean frame", "mean")):
        lines.append(f"| {label} | {cell(ref_at('mse', name, 1), 1e4)} | {cell(ref_at('mse', name, 10), 1e2)} | "
                     f"{cell(ref_at('mse', name, 20))} | {cell(ref_at('mse', name, 30))} | {cell(ref_at('mass', name, 30), fmt='.2f')} |")
    return lines


def pos_table():
    lines = [f"| {sigma:g} | " + " | ".join(cell(at("pos", sigma, h), fmt=".2f") for h in (10, 20, 30)) + " |" for sigma in SIGMAS]
    lines.append("| 0, clipped to [0, 1] during rollout | " + " | ".join(
        cell([clipped[s]["pos"][h - 1] for s in SEEDS], fmt=".2f") for h in (10, 20, 30)) + " |")
    for label, name in (("Copy last frame", "copy"), ("Mean frame", "mean")):
        lines.append(f"| {label} | " + " | ".join(cell(ref_at("pos", name, h), fmt=".2f") for h in (10, 20, 30)) + " |")
    return lines


def clip_table():
    return ["| " + label + " | " + " | ".join(cell(vals(h)) for h in (10, 20, 30)) + " |" for label, vals in (
        ("σ = 0", lambda h: at("mse", 0.0, h)),
        ("σ = 0, clipped to [0, 1] during rollout", lambda h: [clipped[s]["mse"][h - 1] for s in SEEDS]),
        ("σ = 0.01, no clipping", lambda h: at("mse", 0.01, h)))]


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

**Use the median.** The baseline's MSE@30 ranges from 0.521 to 10.3 across the 5 seeds, about a 20-fold spread, which is heavy-tailed. The mean and standard deviation would be dominated by a single seed.
So each cell reads “median (min–max)”.

MSE and mass ratio, 5 seeds:

| $\sigma$ | MSE@1 (×10⁻⁴) | MSE@10 (×10⁻²) | MSE@20 | MSE@30 | Mass ratio@30 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.295 (0.237–0.376) | 1.28 (1.14–1.75) | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) | 2.69 (1.67–3.04) |
| 0.01 | 0.468 (0.389–0.526) | 0.735 (0.607–0.797) | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) | 1.05 (1.02–1.10) |
| 0.03 | 1.17 (1.04–1.31) | 1.13 (0.721–1.58) | 0.0563 (0.0441–0.0617) | 0.0939 (0.0923–0.115) | 1.01 (0.97–1.08) |
| 0.1 | 6.56 (5.87–8.43) | 6.71 (6.12–7.33) | 0.142 (0.135–0.153) | 0.166 (0.162–0.179) | 1.00 (0.99–1.01) |
| Copy last frame | 259 (249–267) | 21.1 (20.3–21.6) | 0.174 (0.171–0.177) | 0.18 (0.18–0.184) | 1.00 (1.00–1.00) |
| Mean frame | 881 (867–886) | 8.78 (8.65–8.89) | 0.0875 (0.0868–0.0882) | 0.0879 (0.0874–0.0881) | 1.00 (1.00–1.00) |

Position error (pixels), 5 seeds:

| $\sigma$ | Position error@10 | Position error@20 | Position error@30 |
| --- | --- | --- | --- |
| 0 | 0.39 (0.38–0.56) | 2.08 (1.43–2.93) | 3.07 (2.76–3.87) |
| 0.01 | 0.31 (0.28–0.35) | 0.90 (0.83–1.02) | 1.62 (1.39–2.00) |
| 0.03 | 0.40 (0.31–0.50) | 1.04 (0.90–1.12) | 1.62 (1.47–1.82) |
| 0.1 | 1.21 (1.14–1.29) | 2.52 (2.42–2.70) | 3.64 (3.48–3.78) |
| 0, clipped to [0, 1] during rollout | 0.39 (0.35–0.50) | 1.72 (1.28–2.31) | 2.97 (2.45–3.42) |
| Copy last frame | 4.63 (4.42–4.70) | 3.75 (3.59–3.88) | 3.97 (3.89–4.04) |
| Mean frame | 2.75 (2.60–2.90) | 2.70 (2.67–2.80) | 2.73 (2.69–2.80) |

**Reading the results.**

1. **Long-horizon error really does go down, by far more than the seed-to-seed variation.** At step 20, the MSE for $\sigma = 0.01$ is 0.043 (0.0396–0.0616)
   and the baseline's is 0.198 (0.125–0.223); the two ranges don't overlap. Paired by seed, the baseline is 3.5, 5.2, 2.4, 4.8 and 3.6 times the $\sigma = 0.01$ value.
   $\sigma = 0.03$ gives similar results.
2. **At step 20 it definitely helps; at step 10 it's not yet clear; at step 5 it doesn't help.** In the main table, at step 10 $\sigma = 0.01$ is better on all 5 seeds,
   with the baseline at 1.4 to 2.6 times its value. But the repeat below, which changes only the noise random stream, shows this conclusion is not robust. At step 5, the ratio between the two ranges from 0.79 to 1.39,
   with no consistent direction; $\sigma = 0.03$ is worse than the baseline at step 5 on all 5 seeds.
3. **Noise eliminates the divergence.** The baseline's mass ratio@30 is 2.69 (1.67–3.04); all three noisy models fall between 0.97 and 1.10.
4. **The cost is one-step error.** The one-step MSE for $\sigma = 0.01$ is 1.1 to 1.8 times the baseline's, and for $\sigma = 0.1$ it is 16 to 29 times. Even so,
   the one-step error for $\sigma = 0.1$ is still far smaller than “copy last frame”.
5. **Too much noise isn't worth it.** At step 10, the MSE for $\sigma = 0.1$ is 3.8 to 5.7 times the baseline's; at step 20 it is 0.142, worse than the “mean frame” at 0.0875;
   at step 30 its position error is 3.64 pixels, larger than the “mean frame” at 2.73, so the position information is lost. One possible explanation: the blob moves only
   0.3 to 1.2 pixels per frame, so the difference between adjacent frames is small to begin with, and once the noise is comparable to it, the model can no longer read the velocity accurately. I did not verify this explanation separately.
6. **At step 30, nothing beats the “mean frame” on MSE.** The median MSE@30 for $\sigma = 0.01$ and $0.03$ is 0.0976 and 0.0939,
   versus 0.0879 for the “mean frame”. But their position error is 1.62 pixels versus 2.73 for the “mean frame”, smaller on all 5 seeds. In other words, the model still roughly knows where the blob
   is; it's just that MSE penalizes a “sharp but slightly off” prediction more heavily than a “blurry average”. So at long horizons you can't look at MSE alone.

**A cheap control: clip the outputs to [0, 1] during rollout.** Part of the baseline's failure is divergence. So could the benefit of noise just be “limiting the amplitude”? Pixel values lie
in [0, 1] anyway, so during rollout we can clip each of the baseline's predicted frames to that range before feeding it back. This needs no retraining, only one more rollout.

| | MSE@10 | MSE@20 | MSE@30 |
| --- | --- | --- | --- |
| σ = 0 | 0.0128 (0.0114–0.0175) | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) |
| σ = 0, clipped to [0, 1] during rollout | 0.0091 (0.0081–0.011) | 0.101 (0.0787–0.13) | 0.262 (0.157–0.267) |
| σ = 0.01, no clipping | 0.00735 (0.00607–0.00797) | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) |

Clipping brings the baseline's MSE at step 30 down from 1.85 to 0.262, so it no longer blows up; but the mass ratio@30 is still 1.75 (1.48–1.91), so the total brightness is still clearly too high.
At step 20 it is still 1.9 to 3.0 times the $\sigma = 0.01$ value, and its position error@20 is 1.72 pixels versus 0.90 for $\sigma = 0.01$. So noise does more than limit the amplitude; it also reduces positional drift. Clipping is just one line of code at inference time
and is worth using in its own right; the effect of combining it with noise was not run here.

**Does the conclusion survive changing only the noise random stream?** In the main table, each seed determines the data, initial weights, batch order and noise all at once. To see how much randomness the noise itself
contributes, fix the first three and change only the noise random stream, training $\sigma = 0.01$ twice more for each seed (10 more models, about half a minute).
The table shows “baseline MSE ÷ noisy-model MSE”; above 1 means noise is better. The first number in each cell is the main-table run.

| seed | Step 10, three noise streams | Step 20, three noise streams |
| --- | --- | --- |
| 0 | 2.05, 1.66, 1.48 | 3.49, 4.01, 3.61 |
| 1 | 2.22, 1.59, 1.24 | 5.20, 3.29, 3.10 |
| 2 | 1.44, 0.99, 1.02 | 2.39, 1.31, 1.84 |
| 3 | 2.63, 1.29, 3.55 | 4.83, 2.85, 4.36 |
| 4 | 1.55, 0.92, 1.31 | 3.61, 2.28, 2.69 |

At step 20, all 15 runs are better, with ratios between 1.31 and 5.20. At step 10, 2 of the 15 are actually worse (0.99 and 0.92). So the report
can only say “there is a clear benefit by step 20”; claiming “it helps from step 10 on” would need more repeats. It also shows that when only one noise stream is reported,
the seed-to-seed variation underestimates the true uncertainty.

**Conclusions this data does not support.**

- That it helps already at step 10. See the previous paragraph for why.
- Precise multiples. There are only 5 seeds. The direction is 5 to 0, and the two-sided p-value of a sign test is $2 \times 0.5^5 = 0.0625$, which on its own doesn't reach 0.05.
  The conclusion's strength comes from the effect size (2.4 times or more) and the non-overlapping ranges of the two groups.
- Whether $\sigma = 0.01$ or $0.03$ is better. The two can't be told apart.
- That “it's the noise itself that does the work”. No control such as weight decay was run, so “any regularization would have a similar effect” can't be ruled out.
- The sampling error from having only 500 clips in the test set. This was not quantified.
- Whether it generalizes to real video. This is a single object, deterministic motion and an MLP.

### Part 3

Below is a sample, about one page long. The actual submission is written in English, with the same structure.

**Question.** A next-frame predictor sees only real context during training but has to consume its own outputs during rollout. Does adding Gaussian noise to the context during training (not to the targets) reduce
long-horizon rollout error, and what does it cost?

**Method.** Data and baseline as given in the problem: 1-D blob videos, an MLP looking at 3 frames. Only one factor is changed: the standard deviation of the context noise,
$\sigma \in \{0, 0.01, 0.03, 0.1\}$. Under the same seed, the four values of $\sigma$ share the training set, test set, initial weights and batch order; $\sigma = 0$
is bit-for-bit identical to the given baseline's weights. 5 seeds; each model is rolled out 30 steps from 3 frames of real context and evaluated on 500 test clips. The grid was fixed before looking at
the results, and all four values of $\sigma$ are reported. The baseline's long-horizon error is heavy-tailed (MSE@30 varies about 20-fold across seeds), so I report the median (min–max).

**Results.**

| $\sigma$ | MSE@1 (×10⁻⁴) | MSE@20 | MSE@30 | Position error@20 (pixels) |
| --- | --- | --- | --- | --- |
| 0 (baseline) | 0.295 | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) | 2.08 |
| 0.01 | 0.468 | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) | 0.90 |
| 0.03 | 1.17 | 0.0563 (0.0441–0.0617) | 0.0939 (0.0923–0.115) | 1.04 |
| 0.1 | 6.56 | 0.142 (0.135–0.153) | 0.166 (0.162–0.179) | 2.52 |
| Mean frame | 881 | 0.0875 | 0.0879 | 2.70 |

- Noise reduces long-horizon error. At step 20, the baseline's MSE is 2.4 to 5.2 times that of $\sigma = 0.01$, in the same direction on all 5 seeds, with non-overlapping ranges.
  Repeating twice more with only the noise random stream changed, all 15 runs are better at step 20; at step 10, 2 are actually worse, and at step 5 the direction is inconsistent.
- After steps 15 to 19, the baseline is worse than “always output the mean frame”; by step 30, the total brightness of the predicted frame is 1.7 to 3.0 times the true value. With noise, total brightness
  stays between 0.97 and 1.10 times.
- Cost: the one-step error for $\sigma = 0.01$ is 1.1 to 1.8 times the baseline's. $\sigma = 0.1$ is too large: its one-step error is 16 to 29 times, and its step-10 error is
  3.8 to 5.7 times the baseline's.
- Control: clipping the baseline's outputs to [0, 1] during rollout stops the MSE from blowing up (MSE@30 drops from 1.85 to 0.262), but the total brightness is still too high (mass ratio@30 of 1.75),
  and the step-20 error is still 1.9 to 3.0 times that of $\sigma = 0.01$. Noise does more than limit the amplitude.
- Conclusions not to draw: at step 30, no setting has a lower MSE than the mean frame (0.0976 and 0.0939 vs 0.0879); at that point the models are only more accurate than the mean frame on position
  (1.62 vs 2.73 pixels). $\sigma = 0.01$ and $0.03$ can't be told apart. The two-sided sign-test p-value over 5 seeds is 0.0625, so the conclusion rests on the effect size,
  not on significance. “Any regularization would have a similar effect” has not been ruled out.

**Next steps (in priority order).**

1. Rule out “it's just regularization”: add a weight-decay control at the same number of training steps.
2. Increase to 20 seeds and refine the $\sigma$ grid between 0.001 and 0.03 to find the smallest noise level that is enough.
3. Compare with a method that targets the problem directly: unroll 2 to 4 steps during training and put the loss on the rollout error (multi-step loss).
4. Move to harder data: two blobs, or stochastic motion. With stochasticity, MSE itself is no longer appropriate, and the metric needs to change.

### Follow-ups

**Why would adding noise to the context help?** At step $h$ of a rollout, the context is “real frames plus accumulated error”, an input that doesn't exist in the training set. How the network extrapolates on inputs
it has never seen is not constrained by training at all; here, that shows up as the brightness being amplified step by step. Adding noise lets the model see “deviations near real frames” during training,
while the target is still the clean next frame, so it learns to pull small deviations back rather than amplify them. However, independent pixel noise is only a crude stand-in for rollout error: real
rollout error is structured, for example an overall shift. This also explains why noise eliminated the divergence but did not eliminate the long-horizon position error.

**Why not add noise to the target frames?** Adding zero-mean independent noise to the target doesn't change the MSE optimum; the optimal prediction is still the conditional mean, just with noisier gradients. What does the work is
noise on the input side: with noisy inputs, the optimal prediction is “the conditional mean of the clean next frame given the noisy context”, which forces the model to learn denoising along the way.

**Why is long-horizon MSE misleading?** Once the predicted position is unrelated to the true position, a “sharp but misplaced” frame has an expected MSE of exactly twice that of the “mean frame”:
if $a$ and $b$ are two i.i.d. images with mean $\mu$, then $\mathbb E\|a-b\|^2 = 2\,\mathbb E\|a-\mu\|^2$. In the table, “copy last frame”
at step 30 is 2.04 to 2.10 times the “mean frame”, which is exactly this relationship. So MSE rewards blurriness. When evaluating video prediction, also look at metrics such as position and sharpness
that don't reward blurriness.

**Without a validation set, how do you choose $\sigma$ without tuning on the test set?** This page's approach is not to choose: the grid is fixed in advance, all four values are reported, and the conclusion is “0.01 to 0.03
all work”, not “the optimal value is X”. If you must hand in a single value, hold out part of the training clips as a validation set, choose by rollout error on the validation set, and then report
only that one value on the test set.

**What is the equivalent in real video generation models?** There are several variants of the same idea: add noise to the conditioning frames (or their latents) during training and give the noise level to the model
as an input, then specify a small level at inference time; learned physics simulators add random-walk noise to the input states during training; replace real inputs with the model's own outputs with some probability
during training (scheduled sampling); and unroll multiple steps directly and compute the loss on them. These directions are listed from memory; check the details against the original papers before the interview.

**If you have only 30 minutes left and haven't written the report, what do you cut?** Cut the position-error table and the clipping control. Keep the 5 seeds, the two reference levels, and the check that “$\sigma = 0$ matches the baseline”.
Results without reference levels and uncertainty are of no use to anyone else; an extra control is only a bonus.

<details>
<summary>Checks (runnable)</summary>

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
    "| Baseline model | 2.4e-05 | 0.00087 | 0.013 | 0.22 | 1.9 |",
    "| Copy last frame | 0.025 | 0.18 | 0.21 | 0.17 | 0.18 |",
    "| Mean frame | 0.089 | 0.089 | 0.088 | 0.087 | 0.087 |",
    "| Baseline mass ratio | 1.00 | 1.01 | 1.06 | 1.37 | 2.69 |",
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
    "| Copy last frame | 259 (249–267) | 21.1 (20.3–21.6) | 0.174 (0.171–0.177) | 0.18 (0.18–0.184) | 1.00 (1.00–1.00) |",
    "| Mean frame | 881 (867–886) | 8.78 (8.65–8.89) | 0.0875 (0.0868–0.0882) | 0.0879 (0.0874–0.0881) | 1.00 (1.00–1.00) |",
]
assert pos_table() == [
    "| 0 | 0.39 (0.38–0.56) | 2.08 (1.43–2.93) | 3.07 (2.76–3.87) |",
    "| 0.01 | 0.31 (0.28–0.35) | 0.90 (0.83–1.02) | 1.62 (1.39–2.00) |",
    "| 0.03 | 0.40 (0.31–0.50) | 1.04 (0.90–1.12) | 1.62 (1.47–1.82) |",
    "| 0.1 | 1.21 (1.14–1.29) | 2.52 (2.42–2.70) | 3.64 (3.48–3.78) |",
    "| 0, clipped to [0, 1] during rollout | 0.39 (0.35–0.50) | 1.72 (1.28–2.31) | 2.97 (2.45–3.42) |",
    "| Copy last frame | 4.63 (4.42–4.70) | 3.75 (3.59–3.88) | 3.97 (3.89–4.04) |",
    "| Mean frame | 2.75 (2.60–2.90) | 2.70 (2.67–2.80) | 2.73 (2.69–2.80) |",
]
assert clip_table() == [
    "| σ = 0 | 0.0128 (0.0114–0.0175) | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) |",
    "| σ = 0, clipped to [0, 1] during rollout | 0.0091 (0.0081–0.011) | 0.101 (0.0787–0.13) | 0.262 (0.157–0.267) |",
    "| σ = 0.01, no clipping | 0.00735 (0.00607–0.00797) | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) |",
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
    for label, sigma in (("0 (baseline)", 0.0), ("0.01", 0.01), ("0.03", 0.03), ("0.1", 0.1)):
        lines.append(f"| {label} | {np.median(at('mse', sigma, 1)) * 1e4:.3g} | {cell(at('mse', sigma, 20))} | "
                     f"{cell(at('mse', sigma, 30))} | {np.median(at('pos', sigma, 20)):.2f} |")
    lines.append(f"| Mean frame | {np.median(ref_at('mse', 'mean', 1)) * 1e4:.3g} | {np.median(ref_at('mse', 'mean', 20)):.3g} | "
                 f"{np.median(ref_at('mse', 'mean', 30)):.3g} | {np.median(ref_at('pos', 'mean', 20)):.2f} |")
    return lines


assert report_table() == [
    "| 0 (baseline) | 0.295 | 0.198 (0.125–0.223) | 1.85 (0.521–10.3) | 2.08 |",
    "| 0.01 | 0.468 | 0.043 (0.0396–0.0616) | 0.0976 (0.0797–0.128) | 0.90 |",
    "| 0.03 | 1.17 | 0.0563 (0.0441–0.0617) | 0.0939 (0.0923–0.115) | 1.04 |",
    "| 0.1 | 6.56 | 0.142 (0.135–0.153) | 0.166 (0.162–0.179) | 2.52 |",
    "| Mean frame | 881 | 0.0875 | 0.0879 | 2.70 |",
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
