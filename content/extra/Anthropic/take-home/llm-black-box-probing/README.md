# Black-Box Probing of an LLM Through Its API: Why Does Long Addition Fail?

## Problem

This is a 5-hour research take-home. You get API access to a language model and nothing else: no weights, no logits, no fine-tuning.

Someone has noticed that **this model gets multi-digit addition wrong more often the longer the numbers are.** Your task is to find out what actually determines the errors, and to give an explanation that quantitatively predicts the probability of getting a given problem right. The scope is the sum of two positive integers, each with at most 20 digits.

**Conditions**

- You can reach the model only through `client.complete(prompt) -> str`. Each call is a brand-new conversation, with no system prompt and no sampling parameters. Asking the same prompt twice may give different replies.
- 3,000 calls in total, exploratory calls included. Past that, the client raises an error.
- 5 hours, counted from when you open the prompt.
- The model here is the object of study, accessed only through the client. Whether you may use an AI assistant to write code is set by your invitation; if it does not say, assume you may not.
- Deliver two things: a notebook that runs end to end, and a 10-minute presentation (followed by Q&A).

**What the grading looks at**

- Whether the conclusions are backed by evidence: are there controls, do the numbers have confidence intervals, were sample sizes thought through in advance?
- Whether factors that change together were separated, rather than drawing a conclusion from one falling curve.
- Whether the hypothesis was tested on **data not used to fit it**, and what was done when the test failed.
- Whether you read the raw replies and the wrong answers themselves, not just the accuracy.
- Whether the conclusions are robust to how the prompt is worded.
- Whether the budget and time were used methodically; whether the notebook reproduces; whether the presentation makes clear what is known and what is not.

### Part 1 — Build the measuring tools and confirm the phenomenon

Write three things:

1. A budgeted client wrapper: it counts calls, raises an error once the budget is exceeded, and saves (tag, prompt, raw reply) for every call.
2. A parser that extracts the answer from a reply. The model's reply format is not fixed. Replies with no parseable number must be counted separately, not silently counted as "wrong" and not silently dropped.
3. A pilot: 50 random addition problems at each of six lengths, 2, 4, 8, 12, 16 and 20 digits (both operands with the same number of digits, no leading zero); report the accuracy and a 95% confidence interval for each length.

```py
class BudgetExceeded(RuntimeError): ...

class BudgetedClient:
    def __init__(self, client, budget: int): ...
    def complete(self, prompt: str, tag: str = "") -> str: ...   # raises BudgetExceeded once the budget is used up

def parse_answer(reply: str) -> int | None: ...                  # None: the reply contains no number
```

### Part 2 — Separate the factors that are mixed together

As the number of digits grows, at least two things change together: the length of the numbers themselves, and the number of carries in column addition. When the two digits in a column, plus the carry from the column below, add up to 10 or more, that column has a carry. For example:

```text
  4821
+ 3579
------
  8400      units 1+9, tens 2+7(+1) and hundreds 8+5(+1) all carry; thousands 4+3(+1)=8 does not: this problem has 3 carries
```

Design a set of experiments that separates the two factors, length and number of carries, and give the size of each effect with an interval. Then propose a quantitative hypothesis: given a problem, what is the probability that the model gets it right? Estimate the hypothesis's parameters, with intervals.

### Part 3 — Test your hypothesis on data you have not seen

Before sending the queries, use the Part 2 hypothesis to write down predictions (values and intervals) for a batch of new problems. The new problems must come from a problem type not used for fitting. Then query and compare. If the prediction misses: read the wrong answers themselves, find what the hypothesis left out, revise it, and test the revised version on a batch of **new** data.

### Part 4 — Robustness and deliverables

1. Repeat the measurement with at least three other prompt wordings on **the same set of problems**, and say which conclusions depend on the wording and which do not.
2. Give the structure of the notebook and an outline of the 10-minute presentation.

### A stand-in for offline practice

The real take-home gives you an API key and a model name. This page cannot call a real API at run time, so it uses a **simulated model** instead: a deterministic Python class with some addition "bugs" planted in it, its randomness fixed by a seed. It is only a stand-in for practice and does not represent the behavior of any real model. Your analysis code depends only on the single interface `complete(prompt) -> str`; to switch to a real API you replace the client and not one other line of code.

For practice, copy the code in the collapsed block below and run it as is, **without reading it first**; reading it amounts to peeking at the answer. It understands any prompt that contains an expression of the form `a + b`.

<details>
<summary>Source of the simulated model (do not expand before solving)</summary>

```python
import hashlib
import re


class SimulatedModel:
    """Offline practice stand-in for "the model behind the API". Deterministic given `seed`."""

    Q_GEN, Q_PROP, SLIP, NO_ANSWER, SHARED, GROUPED = 0.04, 0.25, 0.015, 0.03, 0.7, 0.5

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._seen: dict[str, int] = {}

    def _u(self, *key) -> float:                     # uniform in [0, 1), a pure function of (seed, key)
        digest = hashlib.blake2b(repr((self.seed,) + key).encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") / 2**64

    def complete(self, prompt: str) -> str:
        nth = self._seen.get(prompt, 0)              # asking the same prompt again draws fresh call-level noise
        self._seen[prompt] = nth + 1
        m = re.search(r"(\d[\d,]*)\s*\+\s*(\d[\d,]*)", prompt)
        if m is None:
            return "I can help with addition problems such as 12 + 34."
        a, b = (int(t.replace(",", "")) for t in m.groups())
        scale = self.GROUPED if "," in m.group(0) else 1.0
        xs, ys = str(a)[::-1], str(b)[::-1]
        total = carry = 0
        for k in range(max(len(xs), len(ys))):
            s = int(xs[k:k + 1] or 0) + int(ys[k:k + 1] or 0)
            t = s + carry
            total += (t % 10) * 10**k
            carry = 0
            if t >= 10:                              # this column should carry; sometimes the carry is lost
                q = scale * (self.Q_GEN if s >= 10 else self.Q_PROP)
                shared = self._u("mix", prompt, nth, k) < self.SHARED
                u = self._u("problem", a, b, k) if shared else self._u("call", prompt, nth, k)
                carry = 0 if u < q else 1
        total += carry * 10**(k + 1)
        if self._u("slip", prompt, nth) < self.SLIP:  # rare unrelated slip: one digit of the answer is wrong
            digits = list(str(total))
            pos = int(self._u("slip-pos", prompt, nth) * len(digits))
            digits[pos] = str((int(digits[pos]) + 1 + int(self._u("slip-digit", prompt, nth) * 9)) % 10)
            total = int("".join(digits))
        style = self._u("style", prompt, nth)
        if style < self.NO_ANSWER:
            return "I am not fully certain about this one; please double-check it with a calculator."
        style = (style - self.NO_ANSWER) / (1 - self.NO_ANSWER)
        bare, sentence = (0.75, 0.13) if re.search(r"\b(just|only)\b", prompt, re.I) else (0.20, 0.45)
        if style < bare:
            return str(total)
        if style < bare + sentence:
            return f"The sum is {total:,}."
        return f"{m.group(1)} + {m.group(2)} = {total}"
```

</details>

Usage:

```py
client = BudgetedClient(SimulatedModel(seed=0), 3000)
print(client.complete("What is 4821 + 3579?"))
```

With a real API the client looks roughly like this (official Anthropic Python SDK; this snippet is not run on this page). `MODEL` is the model name given in the prompt.

```py
import anthropic

class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 1024):
        self.client = anthropic.Anthropic()          # reads the API key from the environment
        self.model, self.max_tokens = model, max_tokens

    def complete(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if msg.stop_reason == "max_tokens":          # cut off: a format failure, not a wrong answer
            return ""
        return "".join(block.text for block in msg.content if block.type == "text")

client = BudgetedClient(AnthropicClient(MODEL), 3000)
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

There is no single correct answer to this kind of problem. Graders look at how you get from "a falling curve" to "an explanation that has been tested". Below is one complete route; every number was produced by this page's code running on the stand-in.

### Plan the time and the budget first

Before you start, spend 10 minutes splitting up the time and the calls. The most common failure is to spend the first two hours of budget on one curve drawn in ever finer detail, leaving no calls for testing.

| Time | What | Calls |
| --- | --- | --- |
| 0:00–0:30 | Read the prompt; ask a few questions by hand to see what replies look like; write the client wrapper, parser, logging | A few dozen manual probes at most |
| 0:30–1:00 | Pilot: does the phenomenon exist, what reply formats are there, how many replies give no answer | 300 |
| 1:00–2:15 | Controlled experiment: separate length from carries; fit the first hypothesis | 960 |
| 2:15–3:15 | Write down predictions, run a held-out test; read the wrong answers; revise the hypothesis; run another held-out test | 300 + 300 |
| 3:15–4:00 | Sensitivity to prompt wording; replicate the surprising result | 750 + 390 |
| 4:00–5:00 | Tidy the notebook, write conclusions and limitations, prepare the presentation | 0 |

The reference solution books all 3,000 calls. In the real thing, manual probing and rework also come out of the 3,000, and the final 390 are the slack kept for exactly that.

How to choose sample sizes: the half-width of a 95% interval for a proportion is about $1.96\sqrt{p(1-p)/n}$. At the worst case $p=0.5$, that is about ±0.14 for $n=50$, ±0.11 for $n=80$ and ±0.06 for $n=250$. The pilot only looks at the trend, so 50 problems per length is enough. The controlled experiment has to resolve differences of about 0.1, so 80 problems per cell, then pooled by factor. The prompt comparison uses the same 250 problems, paired.

### Part 1

**Idea.** Get the measuring tools right first. There are three parts: the budget ledger, the parser and the problem generator. If any of them is wrong, none of the later numbers can be trusted.

- Ledger: every call carries a tag, and every raw reply is saved. All later analysis reads only the log, so rerunning the notebook costs no calls.
- Parser: look at what the replies look like before writing rules. One way is to replace every number in a reply with `N` and count how many "shapes" there are. The rule here is "take the last integer in the reply, thousands separators allowed". Take the last one because the model sometimes copies the expression back before giving the result.
- A reply with no number returns `None` and is counted separately.
- Intervals are Wilson score intervals. With small samples or proportions near 0 or 1 they are more reliable than "estimate ± 1.96 standard errors".

```python
from collections import Counter

import numpy as np


class BudgetExceeded(RuntimeError):
    pass


class BudgetedClient:
    """Wraps any object with .complete(prompt) -> str: counts calls, enforces the budget, keeps every raw reply."""

    def __init__(self, client, budget: int):
        self.client, self.budget, self.log = client, budget, []

    def complete(self, prompt: str, tag: str = "") -> str:
        if len(self.log) >= self.budget:
            raise BudgetExceeded(f"budget of {self.budget} calls is used up")
        reply = self.client.complete(prompt)
        self.log.append((tag, prompt, reply))
        return reply

    @property
    def used(self) -> int:
        return len(self.log)

    def spent(self) -> dict[str, int]:
        return dict(Counter(tag for tag, _, _ in self.log))


NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


def parse_answer(reply: str) -> int | None:
    """The last integer in the reply (thousands separators allowed); None when the reply contains no number."""
    found = NUMBER.findall(reply)
    return int(found[-1].replace(",", "")) if found else None


def parse_strict(reply: str) -> int | None:
    """What a first draft usually does: the whole reply must be an integer."""
    try:
        return int(reply)
    except ValueError:
        return None


def shape(reply: str) -> str:
    """Reply with every number replaced by N: a quick way to list all the formats the model uses."""
    return NUMBER.sub("N", reply)


TEMPLATES = {
    "plain": "What is {a} + {b}? Reply with just the number.",
    "terse": "{a} + {b} = ? Give only the final result.",
    "open": "Compute {a} + {b}.",
    "grouped": "What is {a:,} + {b:,}? Reply with just the number.",
}


def ask(client, problems, tag: str, template: str = "plain") -> list[tuple[int, int, str]]:
    """One call per problem; returns rows (a, b, raw reply)."""
    text = TEMPLATES[template]
    return [(a, b, client.complete(text.format(a=a, b=b), tag)) for a, b in problems]


def score(rows, parser=parse_answer):
    """Boolean arrays (answered, correct). An unparsed reply is not answered and not correct."""
    parsed = [parser(reply) for _, _, reply in rows]
    answered = np.array([p is not None for p in parsed])
    correct = np.array([p == a + b for (a, b, _), p in zip(rows, parsed)])
    return answered, correct


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    p, denom = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half
```

The pilot: six lengths, 50 random problems each. Compute each problem's number of carries along the way. `carry_profile` splits carries into two kinds, which only matters in Part 3; here only their sum is used.

```python
def carry_profile(a: int, b: int) -> tuple[int, int]:
    """(generated, propagated) carries of a + b: digit sum >= 10, or digit sum 9 pushed over by a carry-in."""
    gen = prop = carry = 0
    while a or b:
        s = a % 10 + b % 10
        if s >= 10:
            gen, carry = gen + 1, 1
        elif s == 9 and carry:
            prop += 1
        else:
            carry = 0
        a, b = a // 10, b // 10
    return gen, prop


def random_problem(n_digits: int, rng) -> tuple[int, int]:
    """Two uniform n-digit operands, built digit by digit (rng.integers cannot go past 2**63)."""
    pair = []
    for _ in range(2):
        digits = [int(rng.integers(1, 10))] + [int(d) for d in rng.integers(0, 10, n_digits - 1)]
        pair.append(int("".join(map(str, digits))))
    return pair[0], pair[1]


BUDGET = 3000
client = BudgetedClient(SimulatedModel(seed=0), BUDGET)       # with a real API only this line changes
rng = np.random.default_rng(0)

PILOT_DIGITS = (2, 4, 8, 12, 16, 20)
pilot = {n: ask(client, [random_problem(n, rng) for _ in range(50)], "pilot") for n in PILOT_DIGITS}
pilot_table = {}
print("digits  carries  strict parser      tolerant parser")
for n, rows in pilot.items():
    answered, correct = score(rows)
    strict = int(score(rows, parse_strict)[1].sum())
    carries = float(np.mean([sum(carry_profile(a, b)) for a, b, _ in rows]))
    lo, hi = wilson(int(correct.sum()), len(rows))
    pilot_table[n] = (carries, strict, int(correct.sum()), lo, hi, int(answered.sum()))
    print(f"{n:6d}  {carries:7.2f}  {strict:2d}/50              {int(correct.sum()):2d}/50  [{lo:.2f}, {hi:.2f}]")

pilot_rows = [row for rows in pilot.values() for row in rows]
shapes = Counter(shape(reply) for _, _, reply in pilot_rows)
for text, count in shapes.most_common():
    print(f"{count:4d}  {text}")
pilot_digits = np.repeat(PILOT_DIGITS, 50)
pilot_carries = np.array([sum(carry_profile(a, b)) for a, b, _ in pilot_rows])
pilot_corr = float(np.corrcoef(pilot_digits, pilot_carries)[0, 1])
print(f"corr(digits, carries) in the pilot = {pilot_corr:.2f}")
```

```text
digits  carries  strict parser      tolerant parser
     2     1.12  29/50              47/50  [0.84, 0.98]
     4     2.24  31/50              42/50  [0.71, 0.92]
     8     4.30  27/50              41/50  [0.69, 0.90]
    12     6.44  23/50              32/50  [0.50, 0.76]
    16     7.98  20/50              27/50  [0.40, 0.67]
    20     9.52  22/50              28/50  [0.42, 0.69]
 207  N
  49  The sum is N.
  34  N + N = N
  10  I am not fully certain about this one; please double-check it with a calculator.
corr(digits, carries) in the pilot = 0.88
```

Reading this table:

- The phenomenon is real. 47/50 of the 2-digit problems are right but only 28/50 of the 20-digit ones, and the two intervals do not overlap.
- Replies come in four shapes: 207 are just the number, 49 are a full sentence with thousands separators, 34 copy the expression back, and 10 give no answer.
- The middle column is a strict parser that requires the whole reply to be an integer. On the 2-digit problems it accepts only 29/50. The 18 problems it loses are not wrong sums; their format simply does not suit it. With it, accuracy is squashed to 40–60%, and the downward trend is nearly invisible (29 to 22).
- In the pilot the correlation between number of digits and number of carries is 0.88. This curve cannot tell whether length or carries is doing the work.

### Part 2

**Idea.** To separate the two factors you cannot use random problems; you have to build problems yourself, specifying the number of digits and then the number of carries. Make a 3×4 grid: 8, 14 and 20 digits by 0, 2, 4 and 6 carries, 80 problems per cell.

How to build them: first choose which columns carry. In a carrying column the two digits sum to at least 10; in every other column they sum to at most 8. Why not "at most 9"? Because a column summing to 9 gets pushed to 10 by a carry from below, and the number of carries is then no longer under control. This comes at a cost: the grid has no columns summing to 9 at all. Make a note of it.

The effects are estimated as stratified differences: at each number of carries, compare 20 digits with 8, then average over the four strata; that is the length effect. At each number of digits, compare 6 carries with 0, then average over the three strata; that is the carry effect. Intervals use the Agresti–Caffo approach (add one success and one failure to each group before computing), which stays stable when the proportions are close to 1.

```python
def controlled_problem(n_digits: int, n_carries: int, rng) -> tuple[int, int]:
    """n-digit operands with exactly n_carries carries: carry columns sum to >= 10, the others to <= 8."""
    carry_cols = set(rng.choice(n_digits, size=n_carries, replace=False).tolist())
    a = b = 0
    for k in range(n_digits):
        low = 1 if k == n_digits - 1 else 0                   # no leading zero
        while True:
            x, y = int(rng.integers(low, 10)), int(rng.integers(low, 10))
            if x + y != 9 and (x + y >= 10) == (k in carry_cols):
                break
        a, b = a + x * 10**k, b + y * 10**k
    return a, b


def diff_ci(strata, z: float = 1.96) -> tuple[float, float, float]:
    """Difference p2 - p1 averaged over strata [(k1, n1, k2, n2), ...] with an Agresti-Caffo style interval."""
    w = np.array([n1 * n2 / (n1 + n2) for _, n1, _, n2 in strata], dtype=float)
    w /= w.sum()
    est = sum(wi * (k2 / n2 - k1 / n1) for wi, (k1, n1, k2, n2) in zip(w, strata))
    centre = var = 0.0
    for wi, (k1, n1, k2, n2) in zip(w, strata):
        p1, p2 = (k1 + 1) / (n1 + 2), (k2 + 1) / (n2 + 2)     # one success and one failure added to each group
        centre += wi * (p2 - p1)
        var += wi * wi * (p1 * (1 - p1) / (n1 + 2) + p2 * (1 - p2) / (n2 + 2))
    return est, centre - z * np.sqrt(var), centre + z * np.sqrt(var)


GRID_DIGITS, GRID_CARRIES, PER_CELL = (8, 14, 20), (0, 2, 4, 6), 80
grid = {(n, c): ask(client, [controlled_problem(n, c, rng) for _ in range(PER_CELL)], "grid")
        for n in GRID_DIGITS for c in GRID_CARRIES}
cell = {}                                                     # (digits, carries) -> (correct, answered)
print("carries " + "".join(f"  {n:2d} digits" for n in GRID_DIGITS))
for c in GRID_CARRIES:
    line = f"{c:7d} "
    for n in GRID_DIGITS:
        answered, correct = score(grid[n, c])
        cell[n, c] = (int(correct.sum()), int(answered.sum()))
        line += f"   {cell[n, c][0]:2d}/{cell[n, c][1]:2d}  "
    print(line)

length_effect = diff_ci([cell[8, c] + cell[20, c] for c in GRID_CARRIES])     # 20 digits minus 8, carries fixed
carry_effect = diff_ci([cell[n, 0] + cell[n, 6] for n in GRID_DIGITS])        # 6 carries minus 0, digits fixed
print("length effect at fixed carries: %+.3f  [%+.3f, %+.3f]" % length_effect)
print("carry effect at fixed length:   %+.3f  [%+.3f, %+.3f]" % carry_effect)
```

```text
carries    8 digits  14 digits  20 digits
      0    75/78     77/77     75/77  
      2    74/77     70/79     73/77  
      4    65/78     64/75     66/79  
      6    64/78     63/78     53/77  
length effect at fixed carries: -0.033  [-0.082, +0.019]
carry effect at fixed length:   -0.206  [-0.259, -0.143]
```

- With the number of carries fixed, 20 digits differ from 8 by only −0.033, with interval [−0.082, +0.019], which contains 0. This is a "no effect" control.
- "The interval contains 0" does not mean "there is no effect". The accurate statement is: length by itself has an effect of at most about 8 percentage points, and probably less.
- With the number of digits fixed, 6 carries are 0.206 lower than 0 carries, with interval [−0.259, −0.143]. The same estimator does detect an effect for carries, which shows that failing to detect a length effect is not because the method is blunt.

**Quantitative hypothesis H1.** Each carry is lost independently with probability $q$; in addition there is a small error rate $r_0$ unrelated to carries. The probability of getting a problem with $c$ carries right is

$$P(\text{correct}) = (1-r_0)\,(1-q)^{c}.$$

Estimate by maximum likelihood. With $\theta = -\log(1-r) \ge 0$ the log-likelihood is concave in $\theta$, so bounded L-BFGS solves it reliably. The code pools problems with the same carry counts before computing, so it is fast.

Parameter intervals use the profile likelihood: fix one parameter at some value and refit the rest; the values whose log-likelihood is less than 1.92 below the optimum form the 95% interval (1.92 is half of 3.84, the 95% quantile of $\chi^2_1$).

Why not the handier bootstrap? In a draft I checked coverage by simulation with known parameters: over 200 repetitions, bootstrap percentile intervals covered $r_0$ only 88% to 89% of the time, because $r_0$ is pinned down mainly by a handful of errors in the zero-carry cells and sits right against the boundary at 0. Over 600 repetitions, the profile-likelihood intervals covered each parameter between 94.7% and 96% of the time. So parameter intervals use the profile likelihood. The bootstrap is used only to supply parameter draws for the prediction intervals in the next step; over 250 repetitions the prediction intervals covered between 95.6% and 96.4% of the time. These simulations take over ten minutes to run and are not included on this page.

```python
from scipy.optimize import brentq, minimize


def fit_rates(counts, correct, pin=None) -> tuple[np.ndarray, float]:
    """MLE of (r0, r1, ...) in P(correct) = (1 - r0) * prod_j (1 - r_j) ** counts[:, j]; returns (rates, min nll).
    pin=(j, value) holds r_j at value while the other rates are fitted (needed for a profile likelihood)."""
    X = np.column_stack([np.ones(len(correct)), counts]).astype(float)
    rows, inverse = np.unique(X, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    ok = np.bincount(inverse, weights=correct, minlength=len(rows))
    bad = np.bincount(inverse, minlength=len(rows)) - ok

    def nll(theta):                                           # theta_j = -log(1 - r_j) >= 0, so z is linear
        z = np.clip(rows @ theta, 1e-12, 50.0)
        return ok @ z - bad @ np.log(-np.expm1(-z)), rows.T @ (ok - bad / np.expm1(z))

    bounds = [(1e-9, None)] * X.shape[1]
    if pin is not None:
        bounds[pin[0]] = (-np.log1p(-pin[1]),) * 2
    best = minimize(nll, np.full(X.shape[1], 0.05), jac=True, method="L-BFGS-B", bounds=bounds)
    return 1 - np.exp(-best.x), float(best.fun)


def profile_ci(counts, correct, crit: float = 3.841) -> np.ndarray:
    """95% profile-likelihood interval for every rate: all values whose best fit loses < crit / 2 in log-likelihood."""
    best, floor = fit_rates(counts, correct)
    out = []
    for j, r in enumerate(best):
        def excess(v):
            return 2 * (fit_rates(counts, correct, pin=(j, v))[1] - floor) - crit
        r = float(np.clip(r, 1e-8, 0.98))
        lo = 0.0 if excess(1e-9) < 0 else brentq(excess, 1e-9, r, xtol=1e-6)
        out.append((lo, brentq(excess, r, 0.99, xtol=1e-6)))
    return np.array(out)


def bootstrap_rates(counts, correct, rng, reps: int = 500) -> np.ndarray:
    """Refit on problems resampled with replacement; one row of rates per resample."""
    n = len(correct)
    picks = (rng.integers(0, n, n) for _ in range(reps))
    return np.array([fit_rates(counts[i], correct[i])[0] for i in picks])


def predict_count(draws, counts, rng, sims_per_draw: int = 5) -> tuple[float, float, float]:
    """Expected number correct on new problems and a 95% predictive interval (parameter + sampling noise)."""
    X = np.column_stack([np.ones(len(counts)), counts]).astype(float)
    p = np.exp(X @ np.log1p(-draws).T)                        # (problems, draws)
    sims = np.concatenate([(rng.random(p.shape) < p).sum(axis=0) for _ in range(sims_per_draw)])
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return float(p.sum(axis=0).mean()), float(lo), float(hi)


def table(rows):
    """Per-problem arrays for the answered problems: (generated, propagated) counts and correctness."""
    answered, correct = score(rows)
    profile = np.array([carry_profile(a, b) for a, b, _ in rows])
    return profile[answered], correct[answered].astype(float)


grid_rows = [row for rows in grid.values() for row in rows]
grid_profile, grid_correct = table(grid_rows)
grid_total = grid_profile.sum(axis=1, keepdims=True)
h1, _ = fit_rates(grid_total, grid_correct)                   # H1: one rate for every carry
h1_ci = profile_ci(grid_total, grid_correct)
h1_draws = bootstrap_rates(grid_total, grid_correct, rng)     # parameter draws, used for predictions below
print("H1  slip %.3f [%.3f, %.3f]   per-carry loss %.3f [%.3f, %.3f]"
      % (h1[0], h1_ci[0, 0], h1_ci[0, 1], h1[1], h1_ci[1, 0], h1_ci[1, 1]))
```

```text
H1  slip 0.019 [0.007, 0.039]   per-carry loss 0.037 [0.028, 0.045]
```

The estimated probability of losing a carry is 0.037, with interval [0.028, 0.045]. At this point the story looks complete: length does not matter, and each carry has about a 4% chance of being dropped.

### Part 3

**Idea.** A hypothesis that does well on the data it was fitted to is not evidence. Make it predict a batch of problems it has not seen, and write the prediction down before querying.

The held-out data is 300 fully random 16-digit problems. The grid problems were deliberately constructed; random problems are "natural", a different problem type. The prediction interval has to include two kinds of uncertainty at once: the parameters not being estimated exactly, and the sampling noise of the new data itself. For each set of bootstrap parameters, simulate every problem with the probabilities the hypothesis gives, and take the 2.5% and 97.5% quantiles of the simulated results.

```python
natural = ask(client, [random_problem(16, rng) for _ in range(300)], "natural")
natural_profile, natural_correct = table(natural)
natural_pred = predict_count(h1_draws, natural_profile.sum(axis=1, keepdims=True), rng)
natural_seen = int(natural_correct.sum())
print(f"natural 16-digit problems, {len(natural_correct)} answered: H1 predicts {natural_pred[0]:.0f} correct "
      f"[{natural_pred[1]:.0f}, {natural_pred[2]:.0f}], observed {natural_seen}")


def dropped_column(a: int, b: int, answer: int):
    """If the answer is exactly one lost carry away from a + b, return the kind of the column that lost it."""
    gap, k = a + b - answer, 0
    if gap < 10 or gap != 10 ** (len(str(gap)) - 1):
        return None
    k = len(str(gap)) - 2                                     # carry out of column k is worth 10**(k+1)
    s = a // 10**k % 10 + b // 10**k % 10
    return "generated" if s >= 10 else "propagated" if s == 9 else None


errors = Counter()
for a, b, reply in natural:
    answer = parse_answer(reply)
    if answer is not None and answer != a + b:
        errors[dropped_column(a, b, answer) or "other"] += 1
share_prop = natural_profile[:, 1].sum() / natural_profile.sum()
print(dict(errors), f"| propagated carries are {share_prop:.1%} of all carries in these problems")
```

```text
natural 16-digit problems, 294 answered: H1 predicts 216 correct [196, 236], observed 177
{'generated': 59, 'propagated': 31, 'other': 27} | propagated carries are 9.5% of all carries in these problems
```

294 problems got an answer. H1 predicts 216 correct, interval [196, 236]; only 177 were actually right, far below the interval. **H1 is rejected.**

Do not rush to tune parameters now. H1 is fine on the grid and overestimates on the natural problems, which means the natural problems contain something the grid does not. The answer is hidden in the errors; go and read the wrong answers.

Addition has a handy property: if the carry out of column $k$ is lost, the answer is exactly $10^{k+1}$ less than the correct value. So whenever "correct value − model's answer" is a power of 10, you can locate the column that lost the carry directly, and then look at what that column's two digits sum to.

- Of the 117 errors, 90 are exactly "one lost carry".
- 59 of those are lost in a column whose digits sum to at least 10: the column produces the carry itself (a generated carry).
- The other 31 are lost in a column whose digits sum to exactly 9. Such a column does not carry on its own; it is pushed past 10 by the carry from below and merely passes it up (a propagated carry).
- Propagated carries are only 9.5% of all carries in this batch, yet account for 31 of the 90 errors, a third.

This is exactly the grid's blind spot: to control the number of carries, the generator excluded every column summing to 9.

**Revised hypothesis H2.** The two kinds of carry each have their own loss rate. For a problem with $g$ generated carries and $p$ propagated carries, the probability of getting it right is

$$P(\text{correct}) = (1-r_0)\,(1-q_g)^{g}\,(1-q_p)^{p}.$$

Fit it on the grid and natural problems together:

```python
both_profile = np.vstack([grid_profile, natural_profile])
both_correct = np.concatenate([grid_correct, natural_correct])
h2, _ = fit_rates(both_profile, both_correct)                 # H2: generated and propagated carries differ
h2_ci = profile_ci(both_profile, both_correct)
h2_draws = bootstrap_rates(both_profile, both_correct, rng)
for name, est, (lo, hi) in zip(("slip", "generated", "propagated"), h2, h2_ci):
    print(f"H2  {name:10s} {est:.3f}  [{lo:.3f}, {hi:.3f}]")
```

```text
H2  slip       0.020  [0.007, 0.041]
H2  generated  0.036  [0.029, 0.044]
H2  propagated 0.301  [0.212, 0.392]
```

The loss rate for propagated carries is 0.301, interval [0.212, 0.392]; for generated carries it is 0.036, interval [0.029, 0.044]. The two intervals are far apart.

But H2 was proposed after looking at the errors on the natural problems, so they cannot be used to test it. You need a batch of new data, and for the test to have discriminating power you should pick a problem type on which H1 and H2 predict very differently. Here that is a carry chain: 12-digit problems where one column generates a carry, the $L-1$ columns right above it all sum to 9, and no other column carries. Take $L = 3, 5, 7$, 100 problems each. Chains this long are rare in the earlier data.

```python
def chain_problem(n_digits: int, chain: int, rng) -> tuple[int, int]:
    """One run of `chain` carrying columns: the lowest generates the carry, the rest only pass it on."""
    start = int(rng.integers(0, n_digits - chain))
    a = b = 0
    for k in range(n_digits):
        low = 1 if k == n_digits - 1 else 0
        while True:
            x, y = int(rng.integers(low, 10)), int(rng.integers(low, 10))
            want = (x + y >= 10) if k == start else (x + y == 9) if start < k < start + chain else (x + y <= 8)
            if want:
                break
        a, b = a + x * 10**k, b + y * 10**k
    return a, b


CHAINS = (3, 5, 7)
chains = {L: ask(client, [chain_problem(12, L, rng) for _ in range(100)], "chains") for L in CHAINS}
chain_table = {}
print("chain  observed   H2 predicts        H1 predicts")
for L, rows in chains.items():
    profile, correct = table(rows)
    assert (profile == (1, L - 1)).all()
    p2 = predict_count(h2_draws, profile, rng)
    p1 = predict_count(h1_draws, profile.sum(axis=1, keepdims=True), rng)
    chain_table[L] = (int(correct.sum()), len(correct), p2, p1)
    print(f"{L:5d}  {int(correct.sum()):3d}/{len(correct):3d}   {p2[0]:5.1f} [{p2[1]:.0f}, {p2[2]:.0f}]"
          f"     {p1[0]:5.1f} [{p1[1]:.0f}, {p1[2]:.0f}]")
```

```text
chain  observed   H2 predicts        H1 predicts
    3   51/ 99    46.1 [31, 62]      86.9 [80, 93]
    5   29/ 96    22.4 [10, 39]      78.3 [70, 86]
    7   15/ 96    11.4 [3, 24]      72.7 [63, 82]
```

At all three lengths the observed result falls inside H2's prediction interval and far below H1's. At chain length 7, H1 predicts about 73 correct; only 15 actually were.

Be careful how you phrase the conclusion: H2 passed a test it could have failed. That does not mean "H2 is right", let alone that we know how the model computes internally. It is just a description that holds up at the level of behavior.

### Part 4

**Prompt sensitivity.** Take the first 250 natural problems and ask each one again in three other wordings:

- `terse`: different phrasing, still asking for only the result.
- `open`: just "Compute a + b.", without asking for only the number.
- `grouped`: the same phrasing as the original, but with the operands written with thousands separators.

Compare in a paired way: for each problem, subtract its correctness under one wording from its correctness under the other, and compute a t interval over the 250 differences. Pairing cancels out how hard each problem is, so the interval is narrower than for two independent samples. A problem is left out of a comparison if either wording got no answer.

```python
from scipy import stats


def paired_diff(base, other) -> tuple[float, float, float]:
    """Mean of other - base over the same problems, with a t interval on the paired differences."""
    d = np.asarray(other, dtype=float) - np.asarray(base, dtype=float)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


shared_problems = [(a, b) for a, b, _ in natural[:250]]
variants = {"plain": natural[:250]}
for name in ("terse", "open", "grouped"):
    variants[name] = ask(client, shared_problems, "prompts", name)

prompt_table = {}
base_answered, base_correct = score(variants["plain"])
base_strict = score(variants["plain"], parse_strict)[1]
print("template  correct  no answer   tolerant parser: diff vs plain      strict parser: diff vs plain")
print(f"plain     {int(base_correct.sum()):3d}/250  {int((~base_answered).sum()):3d}         (baseline)")
for name in ("terse", "open", "grouped"):
    answered, correct = score(variants[name])
    both = answered & base_answered
    tolerant = paired_diff(base_correct[both], correct[both])
    strict = paired_diff(base_strict, score(variants[name], parse_strict)[1])
    prompt_table[name] = (int(correct.sum()), int((~answered).sum()), tolerant, strict)
    print(f"{name:8s}  {int(correct.sum()):3d}/250  {int((~answered).sum()):3d}         "
          f"{tolerant[0]:+.3f} [{tolerant[1]:+.3f}, {tolerant[2]:+.3f}]       "
          f"{strict[0]:+.3f} [{strict[1]:+.3f}, {strict[2]:+.3f}]")
```

```text
template  correct  no answer   tolerant parser: diff vs plain      strict parser: diff vs plain
plain     146/250    5         (baseline)
terse     158/250    9         +0.072 [+0.017, +0.127]       -0.008 [-0.081, +0.065]
open      145/250    8         +0.004 [-0.059, +0.067]       -0.336 [-0.404, -0.268]
grouped   198/250    9         +0.215 [+0.152, +0.279]       +0.144 [+0.070, +0.218]
```

- `open` is no different from the original wording: +0.004, interval [−0.059, +0.067]. But with the strict parser you would get −0.336. The reason is that when not asked to "reply with just the number", the model more often replies with a full sentence. This is a spurious effect created by the measuring tool and has nothing to do with the model's arithmetic.
- `grouped` is a real effect, and a large one: +0.215, interval [+0.152, +0.279]. So the conclusion needs a qualifier: the probability of losing a carry depends on how the numbers are written.
- `terse` is a surprise: +0.072, interval [+0.017, +0.127], which excludes 0. Yet there is no mechanistic reason for it. With three comparisons, the chance of at least one false positive is $1-0.95^3 \approx 14\%$, so this is not unusual.

A surprising result must neither be cherry-picked into the report nor quietly deleted. The right move is to replicate it on new problems. The remaining 390 calls are exactly enough for 195 new problems in two wordings:

```python
fresh = [random_problem(16, rng) for _ in range((BUDGET - client.used) // 2)]      # replicate the surprise
again = {name: score(ask(client, fresh, "replicate", name)) for name in ("plain", "terse")}
both = again["plain"][0] & again["terse"][0]
replication = paired_diff(again["plain"][1][both], again["terse"][1][both])
print("terse vs plain on %d fresh problems: %+.3f [%+.3f, %+.3f]" % ((int(both.sum()),) + replication))
print("calls used:", client.used, client.spent())
```

```text
terse vs plain on 179 fresh problems: +0.011 [-0.067, +0.089]
calls used: 3000 {'pilot': 300, 'grid': 960, 'natural': 300, 'chains': 300, 'prompts': 750, 'replicate': 390}
```

The replication gives +0.011, interval [−0.067, +0.089]. The report says "`terse` has no stable effect", and both results stay in the notebook.

**Comparing with the ground truth planted in the stand-in.** The real take-home has no such step. In practice you can use it to check your conclusions:

| Quantity | Planted value | Estimate above |
| --- | --- | --- |
| Loss rate of generated carries | 0.04 | 0.036, [0.029, 0.044] |
| Loss rate of propagated carries | 0.25 | 0.301, [0.212, 0.392] |
| Error rate unrelated to carries | 0.015 | 0.020, [0.007, 0.041] |
| Effect of length itself | 0 | −0.033, [−0.082, +0.019] |
| Effect of wording | 0 | `open` +0.004, [−0.059, +0.067]; `terse` replication +0.011, [−0.067, +0.089] |
| Thousands grouping | Both loss rates halved, +0.164 on average on this batch | +0.215, [+0.152, +0.279] |
| Share of replies with no answer | 0.03 | 97/3000 |

**Structure of the notebook.**

```text
1. Question and summary of conclusions: five lines at most, each conclusion with a number and an interval
2. Setup: client, budget ledger, parser; tests for the parser; counts of each reply shape
3. Pilot: the phenomenon; evidence that digits and carries are confounded
4. Controlled experiment: how problems are built, the grid table, intervals for the two effects
5. Hypothesis H1 and its parameters
6. Held-out test one: the prediction written down in advance -> result -> error analysis
7. Hypothesis H2; held-out test two (carry chains)
8. Prompt sensitivity; replication of the surprising result
9. Limitations and next steps
Appendix: budget usage table; log file of all raw replies
```

**The 10-minute presentation.**

```text
0:00-1:00  One-sentence conclusion with one plot: carry-chain length vs accuracy, with the H1 and H2 prediction lines on top
1:00-3:00  How it was measured: parsing rule, replies with no answer, intervals; the strict parser as a counterexample
3:00-5:00  Separating the factors: the grid table; the length interval contains 0, the carry interval does not
5:00-7:30  H1's prediction fails -> read the errors -> H2 -> the new test passes
7:30-9:00  Robustness: wording does not matter, grouping does, one surprise that did not replicate
9:00-10:00 What we don't know: only a behavioral description; only addition, up to 20 digits, English prompts; what to do next
```

**The easiest places to trip up.**

- Drawing a conclusion from one curve of "accuracy falls with the number of digits". Digits and carries change together.
- A parser that accepts only one format, counting format problems as arithmetic errors; or a change of prompt that changes the format and produces a fake prompt effect.
- Numbers without intervals; or describing "the interval contains 0" as "there is no effect".
- Testing the hypothesis only on the data it was fitted to; or looking at the data first and then "predicting".
- A test without discriminating power: the old and new hypotheses predict about the same on the test problems.
- Writing a surprising significant result straight into the conclusions, or quietly deleting it.
- Spending the whole budget in the first half, leaving no calls for the held-out tests and the replication.
- Not reading the raw replies or the wrong answers.
- Treating several answers to the same problem as independent samples.
- A notebook that cannot be rerun from scratch; raw replies not saved.

### Follow-up questions

**If you ask the same problem 5 times, can you treat that as 5 independent samples?** No. Answers to the same problem are correlated. On the stand-in, asking each of 400 problems twice, the correlation between the two answers' correctness is 0.46. With the design effect $1+(m-1)\rho$, for $m=5$ and $\rho=0.46$ that is 2.84, so 5 repeats are worth only 1.8 independent samples. With a fixed budget, asking more distinct problems beats repeating the same one. Repeating is worth it only when you want to estimate the variability within a single problem.

**The API has no temperature or similar parameters and replies are nondeterministic. How do you ensure reproducibility?** What can be reproduced is the analysis, not the model's replies. Save the raw reply of every call, and have the analysis read only the log; fix the seed of the problem generator; when the notebook is rerun, replay the log instead of calling again. State the model name and the dates of the calls in the report. Also, do not cache identical prompts. A cache wipes out the variability on the same problem and makes repeated measurements meaningless.

**How do you count replies with no answer?** Report both conventions. One counts them as wrong, answering "what is the probability the user gets the correct answer". The other computes only over problems that got an answer, answering "does it compute correctly". More important is to check whether "no answer" is related to problem difficulty. Here 97 of the 3,000 calls gave no answer, about 3%; in the grid, the half with fewer carries and the half with more carries had 15 each. So it can be treated as missing at random, and both conventions give the same conclusions. If hard problems got no answer more often, looking only at answered problems would overestimate accuracy.

**What changes with a real model?** In the code only the client line changes, but the analysis cannot be copied over as is. There will be more reply shapes: worked steps, scientific notation, refusals. The parser has to be rewritten against the real raw replies. A real model might make no errors at all at this length, or err in a completely different way; the hypothesis must come from its own errors, so do not go in looking for "propagated carries". Calls have latency and rate limits, so run them concurrently and handle retries. With a thinking model, leave enough `max_tokens`; a truncated reply (`stop_reason` is `max_tokens`) counts as a format failure, not a wrong answer.

**You compared several prompts and several hypotheses. How do you handle multiple comparisons?** Split the comparisons into two kinds. Confirmatory comparisons are fixed before looking at the data, here the length effect, the carry effect and the two held-out predictions; use 95% intervals as usual. Exploratory comparisons come up along the way, such as the three prompt wordings. Every exploratory finding is replicated on new data before it counts as a conclusion. Alternatively, tighten the intervals directly: use a 98.3% interval for each of the three comparisons (Bonferroni correction).

<details>
<summary>Verification code (runnable)</summary>

```python
M = SimulatedModel                                            # the planted constants, read only here

# the budget is enforced and every call is accounted for
assert client.used == BUDGET == 3000
assert client.spent() == {"pilot": 300, "grid": 960, "natural": 300, "chains": 300, "prompts": 750, "replicate": 390}
try:
    client.complete("What is 1 + 1?")
    raise AssertionError("the budget should be used up")
except BudgetExceeded:
    pass

# the example in the statement, and the parser on every reply shape seen in the pilot
assert 4821 + 3579 == 8400 and carry_profile(4821, 3579) == (2, 1)
assert carry_profile(1234, 4321) == (0, 0) and carry_profile(999, 1) == (1, 2)
assert parse_answer("8400") == 8400 and parse_answer("The sum is 8,400.") == 8400
assert parse_answer("4821 + 3579 = 8400") == 8400 and parse_answer("4,821 + 3,579 = 8400") == 8400
assert parse_answer("I am not fully certain about this one; please double-check it with a calculator.") is None
assert parse_strict("8400") == 8400 and parse_strict("The sum is 8,400.") is None
assert [shapes[k] for k in ("N", "The sum is N.", "N + N = N")] == [207, 49, 34] and sum(shapes.values()) == 300
assert sum(n for text, n in shapes.items() if "N" not in text) == 10

# every generator builds what it claims to build
check_rng = np.random.default_rng(1)
for _ in range(300):
    n, c = int(check_rng.integers(6, 21)), int(check_rng.integers(0, 7))
    a, b = controlled_problem(n, c, check_rng)
    assert len(str(a)) == len(str(b)) == n and carry_profile(a, b) == (c, 0)
    a, b = chain_problem(12, 3 + c % 5, check_rng)
    assert len(str(a)) == len(str(b)) == 12 and carry_profile(a, b) == (1, 2 + c % 5)
    a, b = random_problem(n, check_rng)
    assert len(str(a)) == len(str(b)) == n

# dropped_column against a direct simulation of "lose the carry out of column k"
for _ in range(300):
    a, b = random_problem(10, check_rng)
    xs, ys = str(a)[::-1], str(b)[::-1]
    carry_in, kinds_by_column = 0, {}
    for k in range(10):
        s = int(xs[k]) + int(ys[k])
        if s + carry_in >= 10:
            kinds_by_column[k] = "generated" if s >= 10 else "propagated"
        carry_in = int(s + carry_in >= 10)
    for k, kind_k in kinds_by_column.items():
        assert dropped_column(a, b, a + b - 10 ** (k + 1)) == kind_k
    assert dropped_column(a, b, a + b) is None and dropped_column(a, b, a + b - 3) is None

# Part 1: the phenomenon is real, and a strict parser would have hidden it
assert pilot_table[2][2] == 47 and pilot_table[20][2] == 28
assert pilot_table[2][3] > pilot_table[20][4]                 # the two Wilson intervals do not even touch
assert [pilot_table[n][1] for n in PILOT_DIGITS] == [29, 31, 27, 23, 20, 22]      # strict parser: 40% to 62%
assert all(0.4 <= pilot_table[n][1] / 50 <= 0.62 for n in PILOT_DIGITS)
assert f"{pilot_corr:.2f}" == "0.88"

# Part 2: the control shows no length effect, the same estimator does show the carry effect
assert "%+.3f [%+.3f, %+.3f]" % length_effect == "-0.033 [-0.082, +0.019]" and length_effect[1] < 0 < length_effect[2]
assert "%+.3f [%+.3f, %+.3f]" % carry_effect == "-0.206 [-0.259, -0.143]" and carry_effect[2] < 0
assert "%.3f [%.3f, %.3f]" % (h1[1], *h1_ci[1]) == "0.037 [0.028, 0.045]"
assert h1_ci[1, 0] <= M.Q_GEN <= h1_ci[1, 1] and h1_ci[0, 0] <= M.SLIP <= h1_ci[0, 1]

# Part 3: H1 fails on data it has not seen; the errors point at propagated carries; H2 survives a new test
assert (round(natural_pred[0]), natural_pred[1], natural_pred[2], natural_seen, len(natural_correct)) == (216, 196, 236, 177, 294)
assert natural_seen < natural_pred[1]
assert dict(errors) == {"generated": 59, "propagated": 31, "other": 27} and f"{share_prop:.3f}" == "0.095"
assert errors["propagated"] / (errors["propagated"] + errors["generated"]) > 3 * share_prop
for planted, (lo, hi) in zip((M.SLIP, M.Q_GEN, M.Q_PROP), h2_ci):
    assert lo <= planted <= hi
assert ["%.3f [%.3f, %.3f]" % (est, lo, hi) for est, (lo, hi) in zip(h2, h2_ci)] == \
    ["0.020 [0.007, 0.041]", "0.036 [0.029, 0.044]", "0.301 [0.212, 0.392]"]
assert h2_ci[2, 0] > h2_ci[1, 1]                               # the two rates are clearly different
assert [(chain_table[L][0], chain_table[L][1]) for L in CHAINS] == [(51, 99), (29, 96), (15, 96)]
for L in CHAINS:
    seen, n, p2, p1 = chain_table[L]
    assert p2[1] <= seen <= p2[2] and seen < p1[1]             # H2 predicts the new data, H1 does not
    truth_p = (1 - M.SLIP) * (1 - M.Q_GEN) * (1 - M.Q_PROP) ** (L - 1)
    lo, hi = wilson(seen, n)
    assert lo <= truth_p <= hi

# Part 4: wording does not matter, digit grouping does, and the strict parser invents a wording effect
assert "%+.3f [%+.3f, %+.3f]" % prompt_table["terse"][2] == "+0.072 [+0.017, +0.127]"       # the surprise ...
assert "%+.3f [%+.3f, %+.3f]" % replication == "+0.011 [-0.067, +0.089]" and replication[1] < 0 < replication[2]
assert int(both.sum()) == 179
assert prompt_table["open"][2][1] < 0 < prompt_table["open"][2][2]
assert "%+.3f" % prompt_table["open"][3][0] == "-0.336" and prompt_table["open"][3][2] < -0.2
profile = np.array([carry_profile(a, b) for a, b in shared_problems])
answered_both = score(variants["grouped"])[0] & base_answered
p_plain = (1 - M.SLIP) * (1 - M.Q_GEN) ** profile[:, 0] * (1 - M.Q_PROP) ** profile[:, 1]
p_grouped = (1 - M.SLIP) * (1 - M.GROUPED * M.Q_GEN) ** profile[:, 0] * (1 - M.GROUPED * M.Q_PROP) ** profile[:, 1]
planted_gain = float((p_grouped - p_plain)[answered_both].mean())
assert "%+.3f [%+.3f, %+.3f]" % prompt_table["grouped"][2] == "+0.215 [+0.152, +0.279]"
assert 0 < prompt_table["grouped"][2][1] <= planted_gain <= prompt_table["grouped"][2][2] and f"{planted_gain:+.3f}" == "+0.164"
assert f"{1 - 0.95 ** 3:.2f}" == "0.14"                        # chance of at least one false alarm in three comparisons

# replies without an answer: about 3% everywhere, not tied to difficulty
silent = Counter(tag for tag, _, reply in client.log if parse_answer(reply) is None)
assert sum(silent.values()) == 97 and wilson(sum(silent.values()), 3000)[0] <= M.NO_ANSWER <= wilson(sum(silent.values()), 3000)[1]
easy = sum(80 - cell[n, c][1] for n in GRID_DIGITS for c in (0, 2))
hard = sum(80 - cell[n, c][1] for n in GRID_DIGITS for c in (4, 6))
assert (easy, hard) == (15, 15)
silent_gap = diff_ci([(easy, 480, hard, 480)])
assert silent_gap[1] < 0 < silent_gap[2]

# the back-of-envelope half-widths quoted in the planning section
assert [round(1.96 * np.sqrt(0.25 / n), 2) for n in (50, 80, 250)] == [0.14, 0.11, 0.06]

# the interval methods: exact coverage of Wilson at n = 80, and asking the same problem twice is not a new sample
ks = np.arange(81)
bounds = np.array([wilson(int(k), 80) for k in ks])
coverage = [stats.binom.pmf(ks, 80, p)[(bounds[:, 0] <= p) & (p <= bounds[:, 1])].sum() for p in np.linspace(0.05, 0.95, 91)]
assert 0.93 < min(coverage) and 0.945 < np.mean(coverage) < 0.96
twin = SimulatedModel(seed=1)                                  # a separate copy: these calls are not in the budget
first, second = [], []
for _ in range(400):
    a, b = random_problem(16, check_rng)
    first.append(parse_answer(twin.complete(TEMPLATES["plain"].format(a=a, b=b))) == a + b)
    second.append(parse_answer(twin.complete(TEMPLATES["plain"].format(a=a, b=b))) == a + b)
repeat_corr = float(np.corrcoef(first, second)[0, 1])
assert f"{repeat_corr:.2f}" == "0.46" and f"{1 + 4 * 0.46:.2f}" == "2.84" and f"{5 / 2.84:.1f}" == "1.8"
print(f"correlation between two answers to the same problem: {repeat_corr:.2f}")
print("all checks passed")
```

</details>

</details>
