# Evaluating LLMs on a Toy Task, Then Testing LLM-Generated Data

## Problem

This is a 5-hour research take-home. The task itself is tiny, but what you have to answer are the two most common questions in evaluation work: how to score a model reliably, and whether "having an LLM generate more data" actually helps.

**The toy task: bracket balancing.** Strings consist only of the four characters `(`, `)`, `[`, `]`. Reading left to right, every closing bracket must be the same kind as the nearest unmatched opening bracket before it (`()` or `[]`), and when you finish reading there must be no unmatched opening brackets left. A string that meets both conditions is called balanced; otherwise it is unbalanced. The empty string counts as balanced, but there are no empty strings in the data.

```text
([])[]    balanced
([)]      unbalanced    one pair each of ( and [, so the counts are right; but at ), the nearest unmatched opening bracket is [
(()       unbalanced    one ( too many
```

**What you get**

- Two models, `"small"` and `"large"`, accessible only through `client.complete(model, prompt) -> str`. Each call is a brand-new conversation, with no system prompt and no sampling parameters. Asking the same prompt twice may give different replies.
- `load_task()` returns three datasets:
  - `dev`: 12 labeled strings, a list of `(string, label)`;
  - `test`: 300 labeled test strings, of length 5 to 14;
  - `colleague_file`: a 300-line text. A colleague says it is extra data he "generated for this task with an LLM" last week, and said nothing else.
- The labels in `dev` and `test` have been checked by hand and are correct.
- The two prompt-building functions below.

**Constraints**

- 3,000 calls in total, shared by both models; exploratory calls count too. Beyond that, the client raises an error.
- 5 hours, counted from when you open the problem.
- The models here are the object of study and can only be called through the client. Whether you may use an AI assistant while writing code is governed by the invitation email; if it doesn't say, assume you may not.
- Deliver a notebook that runs from top to bottom, plus a 10-minute presentation (followed by Q&A).

**What is graded**

- Whether every number has a confidence interval; whether comparisons are made on the same data; whether there are controls and baselines.
- Whether you read the raw replies and the errors themselves, not just the accuracy.
- Whether the generated data was checked before use, what was checked, and how problems were handled once found.
- Whether a conclusion of "helps" or "doesn't help" survives follow-up questions: where the effect comes from, and whether it survives a rephrasing.
- Whether the budget and time were used methodically; whether the notebook is reproducible; whether the presentation makes clear "what we know and what we don't".

```python
INSTRUCTION = "Is the bracket string balanced? Answer with one word: balanced or unbalanced."


def classification_prompt(s: str, demos=(), instruction: str = INSTRUCTION) -> str:
    """The prompt for one string; demos is a sequence of (string, label) pairs shown as worked examples."""
    lines = [instruction, ""]
    if demos:
        lines += ["Examples:"] + [f"{d} -> {label}" for d, label in demos] + [""]
    return "\n".join(lines + [f"String: {s}"])


def generation_prompt(n: int) -> str:
    return (f"Write {n} new examples for the bracket task, one per line, "
            "in the form: <string> -> <balanced|unbalanced>")
```

### Part 1 — Score the two models

Without examples (zero-shot), evaluate both models on the 300 test strings using `classification_prompt(s)`.

1. Write a budgeted client wrapper: count calls, raise when the budget is exceeded, and save every call's (tag, model, prompt, raw reply).
2. Write a parser that extracts the label from a reply. Replies from which no label can be parsed count as wrong, but report how many there are separately.
3. Report both models' accuracy with 95% confidence intervals. Is the large model better? Give the difference and its interval.
4. Where are the errors concentrated? Back it up with data.

```py
class BudgetExceeded(RuntimeError): ...

class BudgetedClient:
    def __init__(self, api, budget: int): ...
    def complete(self, model: str, prompt: str, tag: str = "") -> str: ...   # raises BudgetExceeded once the budget is used up

def parse_label(reply: str) -> str | None: ...       # "balanced" / "unbalanced" / None when the reply gives neither
```

### Part 2 — Have a model generate data: check it first, then use it

Use `generation_prompt(n)` to have the large model generate new labeled examples; you decide how many calls to spend. Before using the data, check it; check the colleague's file the same way.

Someone on the team suggests: "The test set only has 300 strings, which is too small. Let's just use the generated data to compare the two models." Use data to show whether this suggestion works.

```py
def parse_examples(text: str) -> tuple[list[tuple[str, str]], list[str]]: ...   # (examples, rejected lines)
```

### Part 3 — Does the generated data help?

You can't fine-tune the models through the API, so the way to "use the data" is to put examples into the prompt as few-shot demonstrations, i.e. `classification_prompt(s, demos)`.

The question: **for the small model, does adding generated data on top of the 12 dev examples raise accuracy?** Compare at least these three: dev only; dev + data you generated yourself; dev + the colleague's file. Add any other controls you think are needed. Conclusions must come with intervals, and explain every unexpected result.

### Part 4 — Robustness and delivery

1. With the remaining calls, check whether Part 3's main conclusion depends on how the prompt is worded (the `instruction` parameter can be changed) and on the order of the examples.
2. Give the structure of the notebook and an outline of the 10-minute presentation.

### A stand-in for offline practice

The real take-home gives you an API key and model names. This page can't call a real API at run time, so it uses a **simulated API** instead: a deterministic Python class with the two models' "quirks" and the "quirks" of data generation built in, with randomness fixed by a seed. It is only a practice stand-in and does not represent the behavior of any real model. Your analysis code depends only on the single interface `complete(model, prompt) -> str`; to switch to a real API, you only swap out the client, and the rest of the code stays the same.

When practicing, copy the code in the collapsed section below and run it as is, **without reading it first**; reading it is the same as peeking at the answers. The stand-in only recognizes prompts built by the two functions above (`instruction` may be replaced with another sentence); any other prompt just gets the reply "I am not sure what you are asking." Real models have no such limitation.

<details>
<summary>Source of the simulated API and data (don't expand before solving)</summary>

```python
import math
import random
import re
from types import SimpleNamespace

_PAIR = {"(": ")", "[": "]"}
_OTHER = {"balanced": "unbalanced", "unbalanced": "balanced"}


def _balanced(s: str) -> bool:
    stack = []
    for ch in s:
        if ch in _PAIR:
            stack.append(_PAIR[ch])
        elif not stack or stack.pop() != ch:
            return False
    return not stack


def _kind(s: str) -> str:
    if _balanced(s):
        return "balanced"
    return "order" if s.count("(") == s.count(")") and s.count("[") == s.count("]") else "count"


def _random_balanced(pairs: int, rnd) -> str:
    out, stack = [], []
    while len(out) < 2 * pairs:
        if stack and (len(stack) == 2 * pairs - len(out) or rnd.random() < 0.5):
            out.append(stack.pop())
        else:
            opener = rnd.choice("([")
            out.append(opener)
            stack.append(_PAIR[opener])
    return "".join(out)


def _random_string(kind: str, pairs: int, rnd) -> str:
    while True:
        chars = list(_random_balanced(pairs, rnd))
        if kind == "order":                           # swap two brackets: the counts stay equal
            i, j = rnd.sample(range(len(chars)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        elif kind == "count":                         # drop or replace one bracket: the counts no longer match
            i = rnd.randrange(len(chars))
            if rnd.random() < 0.5:
                del chars[i]
            else:
                chars[i] = rnd.choice([c for c in "()[]" if c != chars[i]])
        s = "".join(chars)
        if _kind(s) == kind:
            return s


class SimulatedAPI:
    """Offline practice stand-in for two models behind one API: complete(model, prompt) -> reply text."""

    SKILL = {"small": 0.30, "large": 0.80}            # chance of really checking the nesting
    GAIN, TAU, TAU_NEG, COPY, FLIP, NO_ANSWER, SHARED = 0.65, 12.0, 5.0, 0.9, 0.04, 0.02, 0.7
    GEN_MIX, GEN_NOISE, GEN_NOISE_ORDER, GEN_JUNK = (0.62, 0.28, 0.10), 0.02, 0.72, 0.04
    REAL_MIX = (0.40, 0.20, 0.40)

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._seen: dict[tuple[str, str], int] = {}
        self._evidence: dict[tuple, int] = {}

    # ---- classification -------------------------------------------------------------------------------
    def _skill(self, model: str, demos: tuple) -> float:
        if demos not in self._evidence:               # only near-miss examples teach the model anything
            near = {(s, label) for s, label in demos if _kind(s) == "order"}
            good = sum(label == "unbalanced" for _, label in near)
            self._evidence[demos] = good - (len(near) - good)
        e, base = self._evidence[demos], self.SKILL[model]
        if e >= 0:
            return base + (1 - base) * self.GAIN * (1 - math.exp(-e / self.TAU))
        return base * math.exp(e / self.TAU_NEG)

    def _p_says(self, model: str, s: str, demos: tuple, label: str) -> float:
        """Probability that the parsed reply equals `label` (used only by the checks at the end of the page)."""
        truth = "balanced" if _balanced(s) else "unbalanced"
        lazy = "unbalanced" if _kind(s) == "count" else "balanced"
        skill = self._skill(model, demos)
        p = skill * (truth == label) + (1 - skill) * (lazy == label)
        shown = dict(reversed(demos)).get(s)
        if shown is not None:
            p = self.COPY * (shown == label) + (1 - self.COPY) * p
        return (1 - self.NO_ANSWER) * (p * (1 - self.FLIP) + (1 - p) * self.FLIP)

    def _classify(self, model: str, prompt: str, nth: int) -> str:
        s = re.search(r"^String: (\S+)\s*$", prompt, re.M).group(1)
        demos = tuple(re.findall(r"^(\S+) -> (balanced|unbalanced)$", prompt, re.M))
        call = random.Random(f"{self.seed}|{model}|{nth}|{prompt}")
        mix, u_call, u_copy, u_flip, style = (call.random() for _ in range(5))
        u = random.Random(f"{self.seed}|item|{s}").random() if mix < self.SHARED else u_call
        truth = "balanced" if _balanced(s) else "unbalanced"
        shown = dict(reversed(demos)).get(s)          # the same string among the examples: its first label
        if shown is not None and u_copy < self.COPY:
            answer = shown
        elif u < self._skill(model, demos):
            answer = truth
        else:                                         # lazy rule: only compare the bracket counts
            answer = "unbalanced" if _kind(s) == "count" else "balanced"
        if u_flip < self.FLIP:
            answer = _OTHER[answer]
        if style < self.NO_ANSWER:
            return "I cannot tell without more context."
        style = (style - self.NO_ANSWER) / (1 - self.NO_ANSWER)
        if style < 0.70:
            return answer
        if style < 0.80:
            return answer.capitalize() + "."
        if style < 0.94:
            return "The string is balanced." if answer == "balanced" else "The string is not balanced."
        return f"Answer: {answer}"

    # ---- generation -----------------------------------------------------------------------------------
    def _line(self, rnd, number: int) -> str:
        kind = rnd.choices(("balanced", "count", "order"), self.GEN_MIX)[0]
        s = _random_string(kind, rnd.randint(1, 4), rnd)
        label = "balanced" if kind == "balanced" else "unbalanced"
        if rnd.random() < (self.GEN_NOISE_ORDER if kind == "order" else self.GEN_NOISE):
            label = _OTHER[label]
        style = rnd.random()
        if style < self.GEN_JUNK:
            return rnd.choice(["Here are some more examples:", f"{s} ->", f"{s} (this one is {label})"])
        if style < 0.16:
            return f"{number}) {s} -> {label}"
        if style < 0.20:
            return f'"{s}" -> {label}'
        return f"{s} -> {label}"

    def complete(self, model: str, prompt: str) -> str:
        if model not in self.SKILL:
            raise ValueError(f"unknown model {model!r}")
        nth = self._seen.get((model, prompt), 0)      # the same prompt again draws fresh call-level noise
        self._seen[model, prompt] = nth + 1
        wanted = re.search(r"Write (\d+) new examples", prompt)
        if wanted:
            rnd = random.Random(f"{self.seed}|generate|{model}|{nth}|{prompt}")
            return "\n".join(self._line(rnd, i + 1) for i in range(int(wanted.group(1))))
        if re.search(r"^String: \S+\s*$", prompt, re.M):
            return self._classify(model, prompt, nth)
        return "I am not sure what you are asking."


def load_task(seed: int = 0):
    """The files of the take-home: dev (12 labelled strings), test (300), and the colleague's text file."""
    rnd = random.Random(f"task|{seed}")
    items, seen = [], set()
    while len(items) < 312:
        kind = rnd.choices(("balanced", "count", "order"), SimulatedAPI.REAL_MIX)[0]
        s = _random_string(kind, rnd.randint(3, 7), rnd)
        if s not in seen:
            seen.add(s)
            items.append((s, "balanced" if kind == "balanced" else "unbalanced"))
    dev, test = items[:12], items[12:]
    api, shown = SimulatedAPI(seed), rnd.sample(test, 150)
    lines = []
    for i in range(300):
        if rnd.random() < 0.5:
            lines.append("%s -> %s" % rnd.choice(shown))
        else:
            lines.append(api._line(rnd, i + 1))
    return SimpleNamespace(dev=dev, test=test, colleague_file="\n".join(lines),
                           _leaked={s for s, _ in shown if any(line.startswith(s + " ->") for line in lines)})
```

</details>

Usage:

```py
task = load_task()
client = BudgetedClient(SimulatedAPI(seed=0), 3000)
print(client.complete("small", classification_prompt("([)]")))
```

With the real API, the client looks roughly like this (official Anthropic Python SDK; this snippet is not run on this page). `MODEL_SMALL` and `MODEL_LARGE` are the two model names specified in the problem.

```py
import anthropic

class AnthropicAPI:
    def __init__(self, models: dict[str, str], max_tokens: int = 4096):
        self.client = anthropic.Anthropic()          # reads the API key from the environment
        self.models, self.max_tokens = models, max_tokens

    def complete(self, model: str, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.models[model], max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if msg.stop_reason == "max_tokens":          # cut off: treat as no answer
            return ""
        return "".join(block.text for block in msg.content if block.type == "text")

client = BudgetedClient(AnthropicAPI({"small": MODEL_SMALL, "large": MODEL_LARGE}), 3000)
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

This problem has several of the most common evaluation pitfalls built in. Below is a complete route through it; all numbers were produced by this page's code running against the stand-in.

### Set the time and budget first

The test set has only 300 strings, which determines how small a difference you can detect. With accuracy around 0.7, the 95% interval on 300 strings has a half-width of about ±0.05. When two approaches are compared on the same strings, the "re-run the baseline" condition later measures an interval half-width of about ±0.055, which translates to a standard error of about 0.028. To have an 80% chance of detecting a difference, it has to be at least $(1.96+0.84)\times 0.028 \approx 0.08$. So every approach has to be run on all 300 strings; you can't run only a subset to save calls.

| Time | What to do | Calls |
| --- | --- | --- |
| 0:00–0:30 | Read the problem, look at the data; ask a few questions by hand to see what replies look like; write the client wrapper, parser and logging | A few dozen manual probes at most |
| 0:30–1:15 | Part 1: zero-shot evaluation of both models; read the errors | 600 |
| 1:15–2:15 | Part 2: generate data, check the data; score both models on the generated strings | 12 + 388 |
| 2:15–3:30 | Part 3: six demonstration conditions, 300 strings each | 1,800 |
| 3:30–4:00 | Part 4: change the prompt wording and the example order | Whatever is left, about 200 (exploratory calls come out of this) |
| 4:00–5:00 | Tidy the notebook, write conclusions and limitations, prepare the presentation | 0 |

One of Part 3's six conditions re-runs "dev only" exactly as before. It costs 300 calls, but it is the noise floor of the whole experiment: how much the same setup differs between two runs is the direct answer to "how big a difference has to be before it's worth talking about".

### Part 1

**Approach.** Look at what the raw replies look like first, then write the parser. Then evaluate both models on the same 300 strings and compare them in a paired way: subtract the two models' correctness on each string and compute a t interval over the 300 differences. Finally, read the errors and find where they are concentrated.

- Parsing rule: if the reply contains `unbalanced` or `not balanced`, it is unbalanced; otherwise, if it contains `balanced`, it is balanced; if neither, return `None`. The order can't be reversed, because `unbalanced` contains `balanced`.
- For a single proportion, use the Wilson score interval.
- For "the difference between one group and another", use the stratified Agresti–Caffo interval (add one success and one failure to each group before computing).

```python
from collections import Counter

import numpy as np
from scipy import stats


class BudgetExceeded(RuntimeError):
    pass


class BudgetedClient:
    """Wraps any object with .complete(model, prompt) -> str: counts calls, enforces the budget, keeps raw replies."""

    def __init__(self, api, budget: int):
        self.api, self.budget, self.log = api, budget, []

    def complete(self, model: str, prompt: str, tag: str = "") -> str:
        if len(self.log) >= self.budget:
            raise BudgetExceeded(f"budget of {self.budget} calls is used up")
        reply = self.api.complete(model, prompt)
        self.log.append((tag, model, prompt, reply))
        return reply

    @property
    def used(self) -> int:
        return len(self.log)

    def spent(self) -> dict[str, int]:
        return dict(Counter(tag for tag, *_ in self.log))


def is_balanced(s: str) -> bool:
    stack = []
    for ch in s:
        if ch in "([":
            stack.append(")" if ch == "(" else "]")
        elif not stack or stack.pop() != ch:
            return False
    return not stack


def truth(s: str) -> str:
    return "balanced" if is_balanced(s) else "unbalanced"


def kind(s: str) -> str:
    """balanced / count (the bracket counts already differ) / order (the counts match, the nesting does not)."""
    if is_balanced(s):
        return "balanced"
    return "order" if s.count("(") == s.count(")") and s.count("[") == s.count("]") else "count"


def parse_label(reply: str) -> str | None:
    text = reply.lower()
    if "unbalanced" in text or "not balanced" in text:
        return "unbalanced"
    return "balanced" if "balanced" in text else None


def parse_label_naive(reply: str) -> str:
    """A first draft: no 'unbalanced' in the reply means 'balanced'."""
    return "unbalanced" if "unbalanced" in reply.lower() else "balanced"


def run(client, model: str, strings, tag: str, demos=(), instruction: str = INSTRUCTION) -> np.ndarray:
    """One call per string; returns the parsed labels (None where the reply could not be parsed)."""
    replies = [client.complete(model, classification_prompt(s, demos, instruction), tag) for s in strings]
    return np.array([parse_label(r) for r in replies], dtype=object)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, denom = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def paired_diff(base, other) -> tuple[float, float, float]:
    """Mean of other - base over the same items, with a t interval on the per-item differences."""
    d = np.asarray(other, dtype=float) - np.asarray(base, dtype=float)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


def diff_ci(strata, z: float = 1.96) -> tuple[float, float, float]:
    """Difference p2 - p1 between two independent groups, averaged over strata [(k1, n1, k2, n2), ...]."""
    w = np.array([n1 * n2 / (n1 + n2) for _, n1, _, n2 in strata], dtype=float)
    w /= w.sum()
    est = sum(wi * (k2 / n2 - k1 / n1) for wi, (k1, n1, k2, n2) in zip(w, strata))
    centre = var = 0.0
    for wi, (k1, n1, k2, n2) in zip(w, strata):
        p1, p2 = (k1 + 1) / (n1 + 2), (k2 + 1) / (n2 + 2)     # Agresti-Caffo: add one success and one failure
        centre += wi * (p2 - p1)
        var += wi * wi * (p1 * (1 - p1) / (n1 + 2) + p2 * (1 - p2) / (n2 + 2))
    return est, centre - z * np.sqrt(var), centre + z * np.sqrt(var)


def show(k, n) -> str:
    return "%3d/%3d = %.3f [%.3f, %.3f]" % ((k, n, k / n) + wilson(k, n))
```

```python
BUDGET = 3000
task = load_task(seed=0)
client = BudgetedClient(SimulatedAPI(seed=0), BUDGET)         # with a real API only this line changes

test_strings = [s for s, _ in task.test]
gold = np.array([label for _, label in task.test], dtype=object)
assert all(truth(s) == label for s, label in task.test + task.dev)      # the given labels agree with my checker
kinds = np.array([kind(s) for s in test_strings])
lengths = np.array([len(s) for s in test_strings])

pred = {m: run(client, m, test_strings, "zero-shot") for m in ("small", "large")}
hit = {m: pred[m] == gold for m in pred}
for m in pred:
    unparsed = int(sum(p is None for p in pred[m]))
    print(f"{m:5s}  all {show(int(hit[m].sum()), len(gold))}   unparsed {unparsed}")
    for kd in ("balanced", "count", "order"):
        print(f"       {kd:8s} {show(int(hit[m][kinds == kd].sum()), int((kinds == kd).sum()))}")
replies = Counter(reply for tag, _, _, reply in client.log)
print("reply shapes:", replies.most_common())
for m in pred:
    wrong = (pred[m] != gold) & np.array([p is not None for p in pred[m]])
    print(f"{m:5s}  {int(wrong.sum())} wrong answers, (kind, what it said):", Counter(zip(kinds[wrong].tolist(), pred[m][wrong])).most_common())
model_gap = paired_diff(hit["small"], hit["large"])
print("large - small on the same 300 strings: %+.3f [%+.3f, %+.3f]" % model_gap)
naive_hit = {m: int(sum(parse_label_naive(reply) == label for (_, model, _, reply), label
                        in zip([row for row in client.log if row[1] == m], gold))) for m in pred}
print("the first-draft parser would have reported: small %d/300, large %d/300" % (naive_hit["small"], naive_hit["large"]))

long = lengths > np.median(lengths)
length_effect = {}
for m in pred:                                                # control: within a kind, does length matter?
    strata = [(int(hit[m][(kinds == kd) & ~long].sum()), int(((kinds == kd) & ~long).sum()),
               int(hit[m][(kinds == kd) & long].sum()), int(((kinds == kd) & long).sum()))
              for kd in ("balanced", "count", "order")]
    length_effect[m] = diff_ci(strata)
    print(f"{m:5s}  long minus short strings, kind held fixed: %+.3f [%+.3f, %+.3f]" % length_effect[m])
```

```text
small  all 201/300 = 0.670 [0.615, 0.721]   unparsed 2
       balanced 107/115 = 0.930 [0.869, 0.964]
       count     54/ 55 = 0.982 [0.904, 0.997]
       order     40/130 = 0.308 [0.235, 0.392]
large  all 260/300 = 0.867 [0.824, 0.901]   unparsed 5
       balanced 109/115 = 0.948 [0.891, 0.976]
       count     49/ 55 = 0.891 [0.782, 0.949]
       order    102/130 = 0.785 [0.706, 0.847]
reply shapes: [('balanced', 234), ('unbalanced', 168), ('The string is balanced.', 50), ('The string is not balanced.', 45), ('Balanced.', 38), ('Unbalanced.', 28), ('Answer: unbalanced', 16), ('Answer: balanced', 14), ('I cannot tell without more context.', 7)]
small  97 wrong answers, (kind, what it said): [(('order', 'balanced'), 89), (('balanced', 'unbalanced'), 7), (('count', 'balanced'), 1)]
large  35 wrong answers, (kind, what it said): [(('order', 'balanced'), 25), (('balanced', 'unbalanced'), 5), (('count', 'balanced'), 5)]
large - small on the same 300 strings: +0.197 [+0.137, +0.256]
the first-draft parser would have reported: small 182/300, large 238/300
small  long minus short strings, kind held fixed: +0.039 [-0.049, +0.114]
large  long minus short strings, kind held fixed: +0.075 [-0.010, +0.142]
```

Reading this output:

- Replies come in nine shapes. One of them, `The string is not balanced.`, does not contain `unbalanced`. A parser that only checks "does it contain `unbalanced`" reads this kind of reply as balanced, and would also take `I cannot tell without more context.` as balanced. With such a parser, the two models would score 182/300 and 238/300 instead of 201 and 260.
- Small model 0.670 [0.615, 0.721], large model 0.867 [0.824, 0.901]. The paired difference is +0.197 [+0.137, +0.256], so the large model really is better.
- Reading the errors: the small model gets 97 wrong (plus 2 unparseable), and 89 of them call an unbalanced string balanced. Looking at those 89 strings, they have one thing in common: each kind of bracket has equal numbers of opening and closing brackets; only the order is wrong, as in `([)]`. It looks like the model only counted.
- So split the test set into three kinds: balanced (115 strings); wrong counts (count, 55 strings); right counts but wrong order (order, 130 strings). On the order kind the small model gets only 0.308 and the large model 0.785. Almost the entire gap between the two models is in this kind.
- Control: are the order strings simply longer? Comparing long and short strings within each kind (split at the median length): small model +0.039 [−0.049, +0.114], large model +0.075 [−0.010, +0.142]. Both intervals contain 0, and the point estimates are even positive (long strings do slightly better). So there is no evidence that length is the cause; what's hard is the "right counts but wrong order" kind. The large model's upper bound is +0.14, so you can't say length has no effect at all.

### Part 2

**Approach.** Before using generated data, answer five questions: can the format be parsed; how many duplicates are there; are the labels right; does the distribution resemble the test set; does it overlap with the test set. This task has a ready-made checker: `is_balanced` computes the correct label for any string. So whether the labels are right can be computed directly. What to do without a checker is covered in the follow-ups.

Be a little lenient when parsing generated lines: allow a number at the start of the line and quotes around the string. But reject every unrecognized line and count it; don't quietly "fix" it. When a string appears more than once, keep only the label from its first occurrence.

```python
EXAMPLE = re.compile(r'^\s*(?:\d+[.)]\s*)?"?([()\[\]]+)"?\s*->\s*(balanced|unbalanced)\s*$')


def parse_examples(text: str):
    """(examples, rejected lines). An example is (string, label); anything that does not fit the form is rejected."""
    examples, rejected = [], []
    for line in text.splitlines():
        m = EXAMPLE.match(line)
        if m:
            examples.append(m.groups())
        elif line.strip():
            rejected.append(line)
    return examples, rejected


def audit(name: str, examples) -> dict:
    """Everything that can be said about a set of generated examples without calling a model."""
    first = {}
    for s, label in examples:
        first.setdefault(s, label)
    pool = list(first.items())                                # one label per distinct string: the first one seen
    wrong = np.array([label != truth(s) for s, label in pool])
    kd = np.array([kind(s) for s, _ in pool])
    overlap = sorted(set(first) & set(test_strings))
    print(f"{name}: {len(examples)} examples, {len(pool)} distinct strings, "
          f"{len(set(examples)) - len(pool)} of them seen with both labels")
    print(f"  wrong labels  {show(int(wrong.sum()), len(pool))}")
    for k in ("balanced", "count", "order"):
        share_real = float((kinds == k).mean())
        print(f"  {k:8s} share {np.mean(kd == k):.2f} (test set {share_real:.2f})   "
              f"wrong labels {show(int(wrong[kd == k].sum()), int((kd == k).sum()))}")
    print(f"  mean length {np.mean([len(s) for s, _ in pool]):.1f} (test set {lengths.mean():.1f})   "
          f"strings that are also in the test set: {len(overlap)}")
    return {"pool": pool, "wrong": wrong, "kind": kd, "overlap": overlap, "examples": examples}


generated_text = "\n".join(client.complete("large", generation_prompt(30), "generate") for _ in range(12))
mine, mine_rejected = parse_examples(generated_text)
theirs, theirs_rejected = parse_examples(task.colleague_file)
print(f"my 12 calls: {len(generated_text.splitlines())} lines, {len(mine_rejected)} rejected, e.g. {mine_rejected[:2]}")
mine_audit = audit("mine", mine)
theirs_audit = audit("colleague", theirs)

longest_generated = max(len(s) for s, _ in mine)
long_theirs = [s for s, _ in theirs_audit["pool"] if len(s) > longest_generated]
print(f"longest string the generator wrote: {longest_generated}; colleague's strings longer than that: "
      f"{len(long_theirs)}, of which in the test set: {len(set(long_theirs) & set(test_strings))}")

pool = mine_audit["pool"]
pool_strings = [s for s, _ in pool]
pool_label = np.array([label for _, label in pool], dtype=object)
pool_truth = np.array([truth(s) for s in pool_strings], dtype=object)
# written down before the calls: Part 1's accuracy per kind, reweighted to the mix of kinds in my generated strings
share = {kd: float((mine_audit["kind"] == kd).mean()) for kd in ("balanced", "count", "order")}
predicted = {m: float(sum(share[kd] * hit[m][kinds == kd].mean() for kd in share)) for m in ("small", "large")}
pool_pred = {m: run(client, m, pool_strings, "generated-as-test") for m in ("small", "large")}
print("scored on my generated strings      small   large   large - small")
generated_gap = {}
for name, reference in (("generator's labels", pool_label), ("checker's labels", pool_truth)):
    acc = {m: float((pool_pred[m] == reference).mean()) for m in pool_pred}
    generated_gap[name] = paired_diff(pool_pred["small"] == reference, pool_pred["large"] == reference)
    print(f"  {name:20s}            {acc['small']:.3f}   {acc['large']:.3f}   %+.3f [%+.3f, %+.3f]" % generated_gap[name])
print("  the real test set               %.3f   %.3f   %+.3f [%+.3f, %+.3f]"
      % ((hit["small"].mean(), hit["large"].mean()) + model_gap))
print("  predicted from Part 1 by kind   %.3f   %.3f   %+.3f      (to compare with the checker's-labels row)"
      % (predicted["small"], predicted["large"], predicted["large"] - predicted["small"]))

flagged = (pool_pred["large"] != pool_label) & np.array([p is not None for p in pool_pred["large"]])
caught = int((flagged & mine_audit["wrong"]).sum())
print(f"label-free check: large disagrees with the generator's label on {int(flagged.sum())} strings; "
      f"{caught} of them are really mislabelled, out of {int(mine_audit['wrong'].sum())} mislabelled in total")
```

```text
my 12 calls: 360 lines, 17 rejected, e.g. ['[[]] (this one is balanced)', 'Here are some more examples:']
mine: 343 examples, 194 distinct strings, 2 of them seen with both labels
  wrong labels   23/194 = 0.119 [0.080, 0.172]
  balanced share 0.47 (test set 0.38)   wrong labels   1/ 91 = 0.011 [0.002, 0.060]
  count    share 0.37 (test set 0.18)   wrong labels   0/ 72 = 0.000 [0.000, 0.051]
  order    share 0.16 (test set 0.43)   wrong labels  22/ 31 = 0.710 [0.534, 0.839]
  mean length 6.1 (test set 10.0)   strings that are also in the test set: 26
colleague: 297 examples, 182 distinct strings, 1 of them seen with both labels
  wrong labels    9/182 = 0.049 [0.026, 0.091]
  balanced share 0.46 (test set 0.38)   wrong labels   1/ 83 = 0.012 [0.002, 0.065]
  count    share 0.26 (test set 0.18)   wrong labels   2/ 48 = 0.042 [0.012, 0.140]
  order    share 0.28 (test set 0.43)   wrong labels   6/ 51 = 0.118 [0.055, 0.234]
  mean length 8.2 (test set 10.0)   strings that are also in the test set: 103
longest string the generator wrote: 8; colleague's strings longer than that: 66, of which in the test set: 66
scored on my generated strings      small   large   large - small
  generator's labels              0.871   0.845   -0.026 [-0.086, +0.034]
  checker's labels                0.835   0.887   +0.052 [-0.009, +0.112]
  the real test set               0.670   0.867   +0.197 [+0.137, +0.256]
  predicted from Part 1 by kind   0.850   0.901   +0.051      (to compare with the checker's-labels row)
label-free check: large disagrees with the generator's label on 28 strings; 15 of them are really mislabelled, out of 23 mislabelled in total
```

The data I generated myself (12 calls, 30 lines each):

- 17 of the 360 lines had the wrong format and were rejected.
- The 343 examples contain only 194 distinct strings. The model repeats itself a lot, and 2 of the strings were even given opposite labels at different points.
- The label error rate is 23/194 = 0.119 [0.080, 0.172]. But almost all the errors are in the order kind: 22 of 31 wrong, 0.710 [0.534, 0.839]; the other two kinds have only 1 error in 163 strings. **The generator makes mistakes exactly on the kind the model couldn't handle in the first place.**
- The distribution differs from the test set: the order kind is only 0.16 of it (test set: 0.43); the average length is 6.1 (test set: 10.0). The generated data is shorter and easier.
- 26 strings coincide with the test set. All of them are at most 8 characters. There aren't many short strings to begin with (for example, there are only 40 balanced strings of length 6 in total), so collisions are normal. Whatever the cause, they have to be removed before being used as examples.

The colleague's file:

- Of its 182 distinct strings, 103 are in the test set.
- This is no coincidence. Of the 343 examples I had the generator write, the longest is only 8 characters; yet the colleague's file has 66 strings longer than that, and all 66 are in the test set. These lines were copied in from the test set, labels and all. That also explains why its label error rate is only 9/182, lower than my own generated data: the copied labels are correct.

**Does "comparing the two models on generated data" work?** First reweight Part 1's classification accuracy by the mix of kinds in the generated data and write down the prediction: small model 0.850, large model 0.901, a difference of about +0.05. Then actually run the queries:

- Scored against the generator's labels, the large model is slightly worse: −0.026 [−0.086, +0.034].
- Scored against the checker's correct labels, the difference is +0.052 [−0.009, +0.112], matching the advance prediction.
- On the real test set, the difference is +0.197 [+0.137, +0.256].

Two problems are stacked here. First, the generated data's distribution is easier, with too few order strings, so the gap between the two models is diluted. The prediction reweighted by kind already explains this part. Second, the generator's labels are wrong precisely on the order kind. The large model answers these strings correctly but is marked wrong by the wrong labels. Conclusion: it doesn't work. Evaluating on the generator's own data measures "how much do you agree with the generator", not "how often are you right".

The last line is a check that doesn't use the checker: there are 28 strings where the large model's answer disagrees with the generator's label, 15 of which really are mislabeled, out of 23 mislabeled in total. For tasks without a checker, this kind of "have another model double-check" approach finds most of the errors, but also flags some correct labels. Here there is a checker, so we simply relabel everything.

### Part 3

**Approach.** Every condition uses the small model and the same 300 test strings; each string is asked once under each condition, and each condition is compared in a paired way against "dev only". There are six conditions:

- Dev only (12 examples): the baseline.
- Dev only, run again: the noise floor (A/A control). Its interval should contain 0.
- Dev + generated data, used as is.
- Dev + generated data, cleaned: relabeled with the checker and with strings that coincide with the test set removed, leaving 168.
- Dev + 168 program-generated examples: a twenty-line generator, one third of each kind, containing no test strings. This is the "how well can you do without an LLM" baseline, with the same number of examples as the cleaned set.
- Dev + the colleague's file.

```python
def make_examples(n: int, rng, avoid) -> list[tuple[str, str]]:
    """Programmatic baseline: n distinct labelled strings, a third of each kind, none of them in `avoid`."""
    out, seen = [], set(avoid)
    while len(out) < n:
        want = ("balanced", "count", "order")[len(out) % 3]
        chars, stack = [], []
        pairs = int(rng.integers(3, 8))
        while len(chars) < 2 * pairs:                         # a random balanced string
            if stack and (len(stack) == 2 * pairs - len(chars) or rng.random() < 0.5):
                chars.append(stack.pop())
            else:
                chars.append("(["[int(rng.integers(2))])
                stack.append(")" if chars[-1] == "(" else "]")
        if want != "balanced":                                # swap two positions, or overwrite one
            i, j = (int(v) for v in rng.choice(len(chars), size=2, replace=False))
            if want == "order":
                chars[i], chars[j] = chars[j], chars[i]
            else:
                chars[i] = "()[]"[int(rng.integers(4))]
        s = "".join(chars)
        if kind(s) == want and s not in seen:
            seen.add(s)
            out.append((s, truth(s)))
    return out


rng = np.random.default_rng(0)
in_test = set(test_strings)
clean = [(s, truth(s)) for s in pool_strings if s not in in_test]             # relabelled, test strings removed
CLEANED, COLLEAGUE = "+ generated, cleaned", "+ colleague's file"
conditions = {
    "dev only": tuple(task.dev),
    "dev only (repeat)": tuple(task.dev),
    "+ generated, as is": tuple(task.dev + pool),
    CLEANED: tuple(task.dev + clean),
    "+ programmatic": tuple(task.dev + make_examples(len(clean), rng, in_test | {s for s, _ in task.dev})),
    COLLEAGUE: tuple(task.dev + theirs_audit["pool"]),
}
few = {name: run(client, "small", test_strings, name, demos) == gold for name, demos in conditions.items()}
order = kinds == "order"
effect = {}
print("demonstrations             examples  accuracy                       order kind   change vs dev only")
for name, demos in conditions.items():
    effect[name] = paired_diff(few["dev only"], few[name])
    change = "(baseline)" if name == "dev only" else "%+.3f [%+.3f, %+.3f]" % effect[name]
    print(f"{name:26s} {len(demos):4d}     {show(int(few[name].sum()), len(gold))}   "
          f"{few[name][order].mean():.3f}       {change}")
print("zero-shot (Part 1)            0     " + show(int(hit["small"].sum()), len(gold)))
versus_program = paired_diff(few["+ programmatic"], few[CLEANED])
print("cleaned generated minus programmatic: %+.3f [%+.3f, %+.3f]" % versus_program)

leaked = np.isin(test_strings, theirs_audit["overlap"])
leak_effect = {}
for name, mask in (("in the colleague's file", leaked), ("not in the file", ~leaked)):
    leak_effect[name] = paired_diff(few[CLEANED][mask], few[COLLEAGUE][mask])
    print(f"test strings {name:24s} n={int(mask.sum()):3d}  colleague {few[COLLEAGUE][mask].mean():.3f}  "
          f"cleaned {few[CLEANED][mask].mean():.3f}   diff %+.3f [%+.3f, %+.3f]" % leak_effect[name])
```

```text
demonstrations             examples  accuracy                       order kind   change vs dev only
dev only                     12     219/300 = 0.730 [0.677, 0.777]   0.492       (baseline)
dev only (repeat)            12     210/300 = 0.700 [0.646, 0.749]   0.400       -0.030 [-0.085, +0.025]
+ generated, as is          206     170/300 = 0.567 [0.510, 0.622]   0.100       -0.163 [-0.222, -0.105]
+ generated, cleaned        180     250/300 = 0.833 [0.787, 0.871]   0.685       +0.103 [+0.045, +0.161]
+ programmatic              180     244/300 = 0.813 [0.765, 0.853]   0.685       +0.083 [+0.023, +0.144]
+ colleague's file          194     266/300 = 0.887 [0.846, 0.918]   0.831       +0.157 [+0.101, +0.212]
zero-shot (Part 1)            0     201/300 = 0.670 [0.615, 0.721]
cleaned generated minus programmatic: +0.020 [-0.026, +0.066]
test strings in the colleague's file  n=103  colleague 0.903  cleaned 0.748   diff +0.155 [+0.054, +0.257]
test strings not in the file          n=197  colleague 0.878  cleaned 0.878   diff +0.000 [-0.051, +0.051]
```

- Noise floor: the same setup run twice differs by −0.030 [−0.085, +0.025]. The interval contains 0, which shows the paired interval is honest; it also shows that a difference like 0.03 isn't worth interpreting.
- Using the generated data as is actually lowers accuracy: −0.163 [−0.222, −0.105]. Accuracy on the order kind drops to 0.100. 22 of the 31 order examples are labeled "balanced", which amounts to demonstrating the wrong rule "just count". This isn't just bad luck on one run: in my scratch work I tried 40 seeds, computing only the expected effect built into the stand-in; using the data as is averaged −0.11, and the best run was only +0.004.
- The cleaned data helps: +0.103 [+0.045, +0.161].
- But the program-generated data also helps: +0.083 [+0.023, +0.144]. Comparing the two gives +0.020 [−0.026, +0.066], so you can't tell them apart. On this task, LLM-generated data, once cleaned, is no better than data generated by a small program. What really does the work is the checker.
- The colleague's file looks best: +0.157 [+0.101, +0.212]. Split the test strings into two groups by "is it in the colleague's file" and compare it against the cleaned data: on the 103 strings in the file, the colleague's file is ahead by +0.155 [+0.054, +0.257]; on the 197 strings not in the file, the two are exactly the same, +0.000 [−0.051, +0.051]. The entire extra effect comes from test strings whose "answer is written in the prompt". This is leakage, not data quality. The report should say so and exclude the colleague's file.
  This split actually doesn't have much statistical power. A leaked string can at most recover the part that "would otherwise have been answered wrong", and the cleaned examples already get the small model to over 80% correct. Across 40 seeds in my scratch work, the built-in boost ranges from +0.06 to +0.11, and with around 103 strings it often isn't significant; the +0.155 here is on the high side, partly luck. So the evidence of leakage is the overlap itself (103 overlapping strings, all 66 long strings in the test set); the split only estimates how much it inflates the score. Even if the split weren't significant, this comparison would already be invalid.
- Multiple comparisons: there are five comparisons against the baseline here. With a Bonferroni correction, each comparison uses a 99% interval. The cleaned data's 99% interval is [+0.027, +0.180], which still excludes 0.

One more thing belongs in the limitations: the conditions have different numbers of examples (the 206, 180, 180 and 194 in the table all include the 12 dev examples), so "number" and "source" are confounded. The cleaned data and the program-generated data have the same number, so that comparison is clean. The other comparisons have to be read with this caveat.

### Part 4

**Approach.** 200 calls remain. Randomly take 200 strings from the test set, use a different wording of the instruction, and at the same time shuffle the order of the cleaned examples; compare against the original prompt in a paired way on the same strings. Also check whether, with the new prompt, the improvement over "dev only" is still there. There aren't enough calls left to also re-run "dev only" with the new wording, so both sides are compared against "dev only" under the original prompt.

```python
ALT_INSTRUCTION = "Check the bracket string below. Reply with exactly one word, balanced or unbalanced."
left = BUDGET - client.used                                   # 200 calls: enough for a random part of the test set
subset = np.sort(rng.choice(len(test_strings), size=left, replace=False))
shuffled = tuple(conditions[CLEANED][i] for i in rng.permutation(len(conditions[CLEANED])))
alt = run(client, "small", [test_strings[i] for i in subset], "reworded + shuffled", shuffled, ALT_INSTRUCTION) == gold[subset]
wording = paired_diff(few[CLEANED][subset], alt)
gain_alt = paired_diff(few["dev only"][subset], alt)
gain_same = paired_diff(few["dev only"][subset], few[CLEANED][subset])
print(f"on {left} random test strings, small model, cleaned generated examples:")
print("  reworded instruction + shuffled examples, minus the original prompt: %+.3f [%+.3f, %+.3f]" % wording)
print("  gain over dev only, original prompt:  %+.3f [%+.3f, %+.3f]" % gain_same)
print("  gain over dev only, changed prompt:   %+.3f [%+.3f, %+.3f]" % gain_alt)
print("calls used:", client.used, client.spent())
```

```text
on 200 random test strings, small model, cleaned generated examples:
  reworded instruction + shuffled examples, minus the original prompt: +0.000 [-0.058, +0.058]
  gain over dev only, original prompt:  +0.115 [+0.043, +0.187]
  gain over dev only, changed prompt:   +0.115 [+0.048, +0.182]
calls used: 3000 {'zero-shot': 600, 'generate': 12, 'generated-as-test': 388, 'dev only': 300, 'dev only (repeat)': 300, '+ generated, as is': 300, '+ generated, cleaned': 300, '+ programmatic': 300, "+ colleague's file": 300, 'reworded + shuffled': 200}
```

After changing the wording and the order, the difference is +0.000 [−0.058, +0.058]; relative to "dev only" under the original prompt, the original and the new prompt both improve by +0.115, and neither interval contains 0. The conclusion does not depend on this particular wording. Limitations: only one new wording was tried, on only 200 strings, so differences smaller than ±0.06 can't be detected; "dev only" was not re-run with the new wording, so whether the new wording itself affects "dev only" can't be seen here.

**Checking against the ground truth built into the stand-in.** The real take-home has no such step. When practicing, you can use it to check your conclusions. The "built-in value" in the table below is the expected value computed from the stand-in's hidden rules on the same strings.

| Quantity | Built-in value | Estimate above |
| --- | --- | --- |
| Small model's accuracy on the order kind | 0.310 | 0.308, [0.235, 0.392] |
| Large model minus small model | +0.195 | +0.197, [+0.137, +0.256] |
| Effect of length itself | 0 | small +0.039, [−0.049, +0.114]; large +0.075, [−0.010, +0.142] |
| Generator's mislabeling probability on the order kind | 0.72 | 22/31 = 0.710, [0.534, 0.839] |
| Generator's mislabeling probability on the other two kinds | 0.02 | 1/163, [0.001, 0.034] |
| Comparing the two models with the generator's labels | −0.030 | −0.026, [−0.086, +0.034] |
| Effect of using the generated data as is | −0.156 | −0.163, [−0.222, −0.105] |
| Effect after cleaning | +0.096 | +0.103, [+0.045, +0.161] |
| Colleague's file over cleaned data: leaked strings | +0.086 | +0.155, [+0.054, +0.257] |
| Colleague's file over cleaned data: remaining strings | +0.007 | +0.000, [−0.051, +0.051] |
| Prompt wording and example order | 0 | +0.000, [−0.058, +0.058] |

This page computes 28 intervals in total, and all of them contain the built-in value. A single run can't tell you whether the interval method is right. In my scratch work I re-ran the whole analysis with 200 seeds: the average coverage of the first 26 intervals was 0.963, and the lowest was 0.92. That simulation takes over two minutes to run, so it isn't included on this page.

**Notebook structure.**

```text
1. Question and summary of conclusions: five lines at most, each conclusion with a number and an interval
2. Setup: client, budget ledger, parser; tests for the parser; counts of reply shapes
3. Data: the origin and look of dev / test / the colleague's file; the checker and its tests
4. Part 1: accuracy of the two models; paired difference; breakdown by kind; length control
5. Part 2: generation; the data-check table (format, duplicates, label errors, distribution, overlap); evidence of leakage in the colleague's file
6. Counterexample to "evaluate on generated data": the advance prediction, results under both sets of labels
7. Part 3: design and results of the six conditions; noise floor; the leakage split
8. Part 4: changing the wording and order
9. Limitations and next steps
Appendix: budget usage table; log file of every raw reply
```

**The 10-minute presentation.**

```text
0:00-1:00  One-line conclusion: cleaned generated data helps (+0.10) but is no better than program-generated data; used as is, it hurts; the colleague's data has leakage
1:00-2:30  How scoring works: the parser pitfall, intervals, paired comparison; large model +0.20, the whole gap is in the "right counts, wrong order" kind
2:30-4:30  What the generated data looks like: mislabels concentrated in the hard kind; an easier distribution; overlap with the test set
4:30-5:30  Why generated data can't be used for evaluation: results under both sets of labels
5:30-8:00  Does it help: the table of six conditions, the noise floor, the split of the colleague's file
8:00-9:00  Robustness: changing the wording and order doesn't affect the conclusion
9:00-10:00 What we don't know: only one task, 300 test strings, only the few-shot use; number and source not fully separated; what to do next
```

**Where people most often stumble.**

- Numbers without intervals; comparing two approaches on different strings, or not pairing.
- A parser that recognizes only one phrasing, so `not balanced` is read as balanced.
- Using generated data without checking it: duplicates, label errors, distribution shift, overlap with the test set.
- Computing only the overall label error rate (0.12) and missing that the errors all sit in the hardest kind (0.71).
- Evaluating the models on the generator's own data and labels and concluding "the two models are about the same".
- Seeing that the colleague's file works best and using it directly, without asking where the effect comes from.
- No noise floor, so a difference like 0.03 gets treated as a finding.
- No "without an LLM" baseline, so the credit for cleaning goes to the LLM.
- Spending the budget loosely early and running short later, with no calls left for robustness checks.

### Follow-ups

**This task has a checker. If it didn't (say, a natural-language task), how would you clean generated data?** The first step is to hand-label a random sample and estimate the label error rate, with an interval. Estimate it per kind too, because errors are often concentrated in the hard kind. Then filter with cheap methods: have another model double-check, or have the same model answer several times and keep only the consistent answers. Here, "have the large model double-check" found 15 of the 23 errors, but 13 of the 28 flagged strings were false alarms. After filtering, sample again to see how much the error rate dropped. The test set must be hand-checked and must not come from the generator.

**If program-generated data is just as good, why generate with an LLM at all?** On this task there really is no need. For tasks where a program can generate the inputs and another program can judge them, programs are cheaper and more controllable. The value of LLM generation is in settings with no generating program: the diversity of natural language, the kind of inputs real users write. But such settings usually have no checker either, so the cleaning methods from the previous question matter more. What this problem wants you to see is that the effect comes from "whether the labels are right", not "who wrote the data".

**Are more examples always better?** The comparisons here can't answer that, because the number and the source changed together. To answer it, fix the source (say, the cleaned data) and vary only the number: 12, 45, 90, 180 examples, each run on the same test strings, and plot accuracy against the number, with an interval at each point. With a real model you also have to do the accounting: more examples mean longer inputs on every call, so cost and latency rise, and past some number it may no longer pay off.

**How do you check for leakage in general?** Start with exact matching, comparing after normalizing case, whitespace and punctuation. For natural language you also need to check for near-duplicates: n-gram overlap or edit distance. Once found, split as done here: report the effect separately on overlapping and non-overlapping test samples. If the effect only shows up on the overlapping samples, it is leakage. For data of unknown origin, first find out how it was made; if you can't find out, don't use it.

**With real models, what changes?** In the code only the client line changes, but there are a few more things to do. Real models may be near perfect on strings this short, so run a small pilot first and lengthen the strings if needed to avoid a ceiling effect. There will be more reply shapes, so rewrite the parser against real replies. The quirks of the generated data won't necessarily match the stand-in's either, so run the whole check table, not just the items built into the stand-in. Hundreds of examples make every call's input longer; you can put the fixed examples at the front of the prompt and the test string at the end, and use the API's prompt caching to save money. But when Part 4 shuffles the example order, the cache is invalidated, so budget for that cost. Calls have latency and rate limits, so run them concurrently and handle retries.

<details>
<summary>Verification code (runnable)</summary>

```python
from itertools import product

A = SimulatedAPI                                              # the hidden constants, read only here
truth_api = SimulatedAPI(seed=0)


def planted(model, strings, labels, demos=()) -> float:
    """Expected share of parsed replies equal to `labels` under the stand-in's hidden rules."""
    return float(np.mean([truth_api._p_says(model, s, tuple(demos), lab) for s, lab in zip(strings, labels)]))


interval, planted_value = {}, {}


def record(name, lo, hi, value):
    interval[name], planted_value[name] = (lo, hi), value


for m in ("small", "large"):
    record(f"zero-shot accuracy, {m}", *wilson(int(hit[m].sum()), 300), planted(m, test_strings, gold))
    sub = kinds == "order"
    record(f"zero-shot accuracy on the order kind, {m}", *wilson(int(hit[m][sub].sum()), int(sub.sum())),
           planted(m, np.array(test_strings)[sub], gold[sub]))
    record(f"length control, {m}", *length_effect[m][1:], 0.0)
record("large minus small", *model_gap[1:], planted("large", test_strings, gold) - planted("small", test_strings, gold))

ex_kind = np.array([kind(s) for s, _ in mine])
lines_total = len(generated_text.splitlines())
record("generator: share of rejected lines", *wilson(len(mine_rejected), lines_total), A.GEN_JUNK)
record("generator: share of the order kind", *wilson(int((ex_kind == "order").sum()), len(mine)), A.GEN_MIX[2])
pool_order = mine_audit["kind"] == "order"
record("generator: wrong labels, order kind",
       *wilson(int(mine_audit["wrong"][pool_order].sum()), int(pool_order.sum())), A.GEN_NOISE_ORDER)
record("generator: wrong labels, other kinds",
       *wilson(int(mine_audit["wrong"][~pool_order].sum()), int((~pool_order).sum())), A.GEN_NOISE)
for name, reference in (("generator's labels", pool_label), ("checker's labels", pool_truth)):
    record(f"generated strings, {name}: large minus small", *generated_gap[name][1:],
           planted("large", pool_strings, reference) - planted("small", pool_strings, reference))

expected = {name: planted("small", test_strings, gold, demos) for name, demos in conditions.items()}
for name in conditions:
    record(f"accuracy with {name}", *wilson(int(few[name].sum()), 300), expected[name])
    if name != "dev only":
        record(f"change vs dev only: {name}", *effect[name][1:], expected[name] - expected["dev only"])
for name, mask in (("in the colleague's file", leaked), ("not in the file", ~leaked)):
    part = np.array(test_strings)[mask]
    record(f"colleague minus cleaned, test strings {name}", *leak_effect[name][1:],
           planted("small", part, gold[mask], conditions[COLLEAGUE]) - planted("small", part, gold[mask], conditions[CLEANED]))
sub_strings = np.array(test_strings)[subset]
record("changed prompt minus original prompt", *wording[1:], 0.0)
record("gain over dev only, changed prompt", *gain_alt[1:],
       planted("small", sub_strings, gold[subset], conditions[CLEANED]) - planted("small", sub_strings, gold[subset], task.dev))
misses = [name for name, (lo, hi) in interval.items() if not lo <= planted_value[name] <= hi]
print(f"{len(interval)} intervals compared with the planted values; misses: {misses}")
for name in ("zero-shot accuracy on the order kind, small", "large minus small", "generator: wrong labels, order kind",
             "generator: wrong labels, other kinds", "generated strings, generator's labels: large minus small",
             "change vs dev only: + generated, as is", "change vs dev only: + generated, cleaned",
             "colleague minus cleaned, test strings in the colleague's file",
             "colleague minus cleaned, test strings not in the file"):
    print(f"  {name:62s} planted {planted_value[name]:+.3f}   interval [%+.3f, %+.3f]" % interval[name])
assert len(interval) == 28 and not misses, misses

# the budget is enforced and every call is accounted for
assert client.used == BUDGET == 3000
assert client.spent() == {"zero-shot": 600, "generate": 12, "generated-as-test": 388, "dev only": 300,
                          "dev only (repeat)": 300, "+ generated, as is": 300, "+ generated, cleaned": 300,
                          "+ programmatic": 300, "+ colleague's file": 300, "reworded + shuffled": 200}
try:
    client.complete("small", classification_prompt("()"))
    raise AssertionError("the budget should be used up")
except BudgetExceeded:
    pass

# the prompt formats shown in the statement
assert classification_prompt("([)]") == (
    "Is the bracket string balanced? Answer with one word: balanced or unbalanced.\n\nString: ([)]")
assert classification_prompt("([)]", [("()", "balanced"), ("(]", "unbalanced")]) == (
    "Is the bracket string balanced? Answer with one word: balanced or unbalanced.\n\n"
    "Examples:\n() -> balanced\n(] -> unbalanced\n\nString: ([)]")
assert generation_prompt(30) == ("Write 30 new examples for the bracket task, one per line, "
                                 "in the form: <string> -> <balanced|unbalanced>")
spare = SimulatedAPI(seed=5)                                  # a separate copy: not in the budget
assert len(spare.complete("large", generation_prompt(7)).splitlines()) == 7
assert parse_label(spare.complete("small", classification_prompt("[()]", instruction=ALT_INSTRUCTION))) in ("balanced", "unbalanced", None)
assert spare.complete("small", "Hello") == "I am not sure what you are asking."

# the examples in the statement, and the checker against an independent implementation on every string up to length 8
assert [truth(s) for s in ("([])[]", "([)]", "(()", "")] == ["balanced", "unbalanced", "unbalanced", "balanced"]
assert [kind(s) for s in ("([])[]", "([)]", "(()", ")(")] == ["balanced", "order", "count", "order"]


def by_deletion(s: str) -> bool:
    while "()" in s or "[]" in s:
        s = s.replace("()", "").replace("[]", "")
    return s == ""


for n in range(9):
    for chars in product("()[]", repeat=n):
        s = "".join(chars)
        assert is_balanced(s) == by_deletion(s), s
        if kind(s) == "order":
            assert s.count("(") == s.count(")") and s.count("[") == s.count("]") and not by_deletion(s)
assert sum(by_deletion("".join(c)) for c in product("()[]", repeat=6)) == 40    # few short strings: collisions are easy
assert all(kind(s) == kd for s, kd in zip(test_strings, kinds)) and 5 <= lengths.min() and lengths.max() <= 14

# the reply parser on every reply shape the models produced, and the first-draft parser as a negative control
shapes = {"balanced": "balanced", "unbalanced": "unbalanced", "Balanced.": "balanced", "Unbalanced.": "unbalanced",
          "The string is balanced.": "balanced", "The string is not balanced.": "unbalanced",
          "Answer: balanced": "balanced", "Answer: unbalanced": "unbalanced", "I cannot tell without more context.": None}
assert set(replies) == set(shapes)
assert all(parse_label(r) == label for r, label in shapes.items())
assert parse_label_naive("The string is not balanced.") == "balanced"          # the bug the first draft had
assert parse_label_naive("I cannot tell without more context.") == "balanced"  # and an unparsable reply counted as an answer
assert (naive_hit["small"], naive_hit["large"]) == (182, 238)

# the examples parser on every line format the generator used
assert parse_examples('([]) -> balanced\n3) ([]) -> balanced\n"(]" -> unbalanced\n12. [] -> balanced') == (
    [("([])", "balanced"), ("([])", "balanced"), ("(]", "unbalanced"), ("[]", "balanced")], [])
junk = ["Here are some more examples:", "([]) ->", "[[]] (this one is balanced)"]
assert parse_examples("\n".join(junk) + "\n\n") == ([], junk)

# Part 1: the large model is better, all of the gap is on one kind of string, length is not the reason
assert [int(hit[m].sum()) for m in ("small", "large")] == [201, 260]
assert [int(sum(p is None for p in pred[m])) for m in ("small", "large")] == [2, 5]
assert "%+.3f [%+.3f, %+.3f]" % model_gap == "+0.197 [+0.137, +0.256]" and model_gap[1] > 0
for m in ("small", "large"):
    acc = {kd: hit[m][kinds == kd].mean() for kd in ("balanced", "count", "order")}
    assert acc["order"] < min(acc["balanced"], acc["count"]) - 0.1
    assert length_effect[m][1] < 0 < length_effect[m][2]
small_wrong = (pred["small"] != gold) & np.array([p is not None for p in pred["small"]])
assert int(small_wrong.sum()) == 97 and int((small_wrong & (kinds == "order") & (pred["small"] == "balanced")).sum()) == 89
assert [int((kinds == kd).sum()) for kd in ("balanced", "count", "order")] == [115, 55, 130]

# Part 2: the audit numbers, recomputed with separate code
distinct = {}
for s, label in mine:
    distinct.setdefault(s, set()).add(label)
assert (lines_total, len(mine_rejected), len(mine), len(distinct)) == (360, 17, 343, 194)
assert sum(len(v) == 2 for v in distinct.values()) == 2
assert sum(dict(pool)[s] != truth(s) for s in distinct) == 23
assert int(pool_order.sum()) == 31 and int(mine_audit["wrong"][pool_order].sum()) == 22
assert int((~pool_order).sum()) == 163 and int(mine_audit["wrong"][~pool_order].sum()) == 1
assert len(set(distinct) & set(test_strings)) == 26 and max(map(len, distinct)) == longest_generated == 8
assert all(len(s) <= 8 for s in set(distinct) & set(test_strings))   # the chance collisions are all short strings
their_distinct = {s for s, _ in theirs}
assert len(their_distinct) == 182 and len(their_distinct & set(test_strings)) == 103
assert task._leaked <= set(theirs_audit["overlap"])          # every planted leak is found ...
chance = set(theirs_audit["overlap"]) - task._leaked
assert all(len(s) <= 8 for s in chance)                       # ... and the rest are short chance collisions
assert len(long_theirs) == len(set(long_theirs) & set(test_strings)) == 66
assert "%+.3f [%+.3f, %+.3f]" % generated_gap["generator's labels"] == "-0.026 [-0.086, +0.034]"
assert "%+.3f [%+.3f, %+.3f]" % generated_gap["checker's labels"] == "+0.052 [-0.009, +0.112]"
assert generated_gap["generator's labels"][1] < 0 < generated_gap["generator's labels"][2]
assert abs((predicted["large"] - predicted["small"]) - generated_gap["checker's labels"][0]) < 0.01
assert (int(flagged.sum()), caught, int(mine_audit["wrong"].sum())) == (28, 15, 23)

# Part 3: noisy examples hurt, cleaned ones help as much as a program's, and the colleague's extra gain is leakage
fmt = {name: "%+.3f [%+.3f, %+.3f]" % effect[name] for name in conditions if name != "dev only"}
assert fmt == {"dev only (repeat)": "-0.030 [-0.085, +0.025]", "+ generated, as is": "-0.163 [-0.222, -0.105]",
               CLEANED: "+0.103 [+0.045, +0.161]", "+ programmatic": "+0.083 [+0.023, +0.144]",
               COLLEAGUE: "+0.157 [+0.101, +0.212]"}
assert effect["dev only (repeat)"][1] < 0 < effect["dev only (repeat)"][2]
se = (effect["dev only (repeat)"][2] - effect["dev only (repeat)"][1]) / 2 / stats.t.ppf(0.975, 299)
assert f"{se:.3f}" == "0.028" and f"{(1.96 + 0.84) * se:.2f}" == "0.08"   # smallest change detectable with 80% power
assert effect["+ generated, as is"][2] < 0 < effect[CLEANED][1] and effect["+ programmatic"][1] > 0
assert versus_program[1] < 0 < versus_program[2]
assert [len(conditions[name]) for name in conditions] == [12, 12, 206, 180, 180, 194]
assert f"{few['+ generated, as is'][order].mean():.3f}" == "0.100"
assert len(clean) == 168 and not set(s for s, _ in clean) & set(test_strings) and all(label == truth(s) for s, label in clean)
assert (int(leaked.sum()), int((~leaked).sum())) == (103, 197)
assert leak_effect["in the colleague's file"][1] > 0
assert leak_effect["not in the file"][1] < 0 < leak_effect["not in the file"][2]
# with five comparisons against the baseline, a Bonferroni interval (99%) for the cleaned examples still excludes 0
d = few[CLEANED].astype(float) - few["dev only"].astype(float)
half99 = stats.t.ppf(1 - 0.01 / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
assert d.mean() - half99 > 0 and "[%+.3f, %+.3f]" % (d.mean() - half99, d.mean() + half99) == "[+0.027, +0.180]"

# Part 4: a different instruction and example order changes nothing measurable
assert len(subset) == 200 and "%+.3f [%+.3f, %+.3f]" % wording == "+0.000 [-0.058, +0.058]"
assert "%+.3f %+.3f" % (gain_same[0], gain_alt[0]) == "+0.115 +0.115" and gain_same[1] > 0 and gain_alt[1] > 0

# the back-of-envelope numbers in the planning section
assert round(1.96 * np.sqrt(0.7 * 0.3 / 300), 2) == 0.05
assert "%.3f" % ((effect["dev only (repeat)"][2] - effect["dev only (repeat)"][1]) / 2) == "0.055"
print("all checks passed")
```

</details>

</details>
