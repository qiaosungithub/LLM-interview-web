# 小任务上评估 LLM，再用 LLM 造数据：生成的数据到底有没有用

## 题目

这是一道 5 小时的研究型 take-home。任务本身很小，但要回答的是做评估时最常见的两个问题：怎样可靠地给模型打分；"让 LLM 再多造一些数据"到底有没有用。

**小任务：括号配平。** 字符串只由 `(`、`)`、`[`、`]` 四种字符组成。从左往右读，每个右括号都要和它前面最近的、还没配对的左括号是同一种（`()` 或 `[]`），读完以后不能剩下没配对的左括号。满足这两条就叫配平（balanced），否则叫不配平（unbalanced）。空串算配平，但数据里没有空串。

```text
([])[]    balanced
([)]      unbalanced    ( 和 [ 各一对，数量都对；但读到 ) 时，最近的未配对左括号是 [
(()       unbalanced    多了一个 (
```

**你拿到的东西**

- 两个模型 `"small"` 和 `"large"`，都只能通过 `client.complete(model, prompt) -> str` 访问。每次调用是一段全新的对话，没有系统提示，也没有采样参数。同一个提示词问两次，回复可能不同。
- `load_task()` 返回三份数据：
  - `dev`：12 条带标签的串，`(string, label)` 的列表；
  - `test`：300 条带标签的测试串，长度在 5 到 14 之间；
  - `colleague_file`：一个 300 行的文本。同事说这是他上周"用 LLM 给这个任务生成的额外数据"，别的没说。
- `dev` 和 `test` 的标签都人工核对过，是对的。
- 下面两个构造提示词的函数。

**条件**

- 总共 3,000 次调用，两个模型共用，试探用的也算。超出后 client 直接报错。
- 5 小时，从打开题目算起。
- 这里的模型是研究对象，只能通过 client 调用。写代码时能不能用 AI 助手，以邀请信为准；没写就当作不能。
- 交付一个可以从头跑到尾的 notebook，和一次 10 分钟的讲解（讲完是问答）。

**评分看什么**

- 每个数字有没有置信区间（confidence interval）；比较是不是在同一批数据上做的；有没有对照（control）和基线（baseline）。
- 有没有读原始回复和错误本身，而不只是看准确率。
- 生成的数据在使用之前有没有检查，查了什么，查出问题之后怎么处理。
- "有用"或"没用"的结论能不能经得起追问：效果从哪里来，换个说法还在不在。
- 预算和时间用得是否有章法；notebook 能否复现；讲解有没有说清"知道什么、不知道什么"。

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

### Part 1 —— 给两个模型打分

不给示例（zero-shot），用 `classification_prompt(s)` 在 300 条测试串上评估两个模型。

1. 写一个带预算的 client 包装：给调用计数，超预算就报错，保存每一次调用的（标签，模型，提示词，原始回复）。
2. 写一个从回复里取出标签的解析函数。解析不出标签的回复算答错，但要单独报告有多少条。
3. 报告两个模型的准确率和 95% 置信区间。大模型是不是更好？给出差值和区间。
4. 错误集中在哪里？用数据说明。

```py
class BudgetExceeded(RuntimeError): ...

class BudgetedClient:
    def __init__(self, api, budget: int): ...
    def complete(self, model: str, prompt: str, tag: str = "") -> str: ...   # raises BudgetExceeded once the budget is used up

def parse_label(reply: str) -> str | None: ...       # "balanced" / "unbalanced" / None when the reply gives neither
```

### Part 2 —— 让模型造数据：先检查，再使用

用 `generation_prompt(n)` 让大模型生成新的带标签例子，调用多少次你自己定。在使用这些数据之前，先检查它们；同事的文件也一样要检查。

组里有人提议："测试集只有 300 条，太小了。干脆用生成的数据来比较两个模型。"用数据说明这个提议行不行。

```py
def parse_examples(text: str) -> tuple[list[tuple[str, str]], list[str]]: ...   # (examples, rejected lines)
```

### Part 3 —— 生成的数据有没有用

通过 API 不能微调模型，所以"用数据"的办法是把例子放进提示词当示例（few-shot demonstrations），也就是 `classification_prompt(s, demos)`。

问题是：**对小模型来说，在 12 条 dev 示例之外再加上生成的数据，准确率会不会更高？** 至少比较这三种做法：只用 dev；dev + 你自己生成的数据；dev + 同事的文件。你认为需要的其他对照都可以加。结论要带区间，并解释所有意外的结果。

### Part 4 —— 稳健性与交付

1. 用剩下的调用检查：Part 3 的主要结论是否依赖提示词的写法（`instruction` 参数可以换）和示例的顺序。
2. 给出 notebook 的结构和 10 分钟讲解的提纲。

### 离线练习用的替身

真实的 take-home 会给你一个 API key 和模型名。这一页在运行时没法调用真实 API，所以用一个**模拟 API**（simulated API）代替：一个确定性的 Python 类，里面埋了两个模型的"毛病"和生成数据时的"毛病"，随机性由种子固定。它只是练习用的替身，不代表任何真实模型的行为。你的分析代码只依赖 `complete(model, prompt) -> str` 这一个接口；换成真实 API 时只需要换掉 client，其余代码不用改。

练习时把下面折叠里的代码原样复制运行，**先不要读它**，读了就等于提前看了答案。替身只认得上面两个函数构造的提示词（`instruction` 可以换成别的句子），别的提示词只会得到一句"不知道你在问什么"。真实模型没有这个限制。

<details>
<summary>模拟 API 和数据的源码（做题前不要展开）</summary>

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

用法：

```py
task = load_task()
client = BudgetedClient(SimulatedAPI(seed=0), 3000)
print(client.complete("small", classification_prompt("([)]")))
```

真实 API 下的 client 大致是这样（官方 Anthropic Python SDK；这一段不在本页运行）。`MODEL_SMALL` 和 `MODEL_LARGE` 是题目里指定的两个模型名。

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

## 参考解答

<details>
<summary>展开参考解答</summary>

这道题里埋了几个做评估时最常见的坑。下面是一条完整的路线，所有数字都是本页代码用替身跑出来的。

### 先定时间和预算

测试集只有 300 条，这决定了能分辨多小的差别。准确率在 0.7 附近时，300 条的 95% 区间半宽约 ±0.05。两种做法在同一批串上比较时，后面的"重复跑一遍基线"测出来的区间半宽约 ±0.055，换算成标准误约 0.028。要有 80% 的把握测出一个差别，差别至少要有 $(1.96+0.84)\times 0.028 \approx 0.08$。所以每一种做法都要在全部 300 条上跑，不能为了省调用只跑一部分。

| 时间 | 做什么 | 调用 |
| --- | --- | --- |
| 0:00–0:30 | 读题，看数据；手动问几次，看回复长什么样；写 client 包装、解析器、日志 | 手动试探几十次以内 |
| 0:30–1:15 | Part 1：两个模型的 zero-shot 评估；读错误 | 600 |
| 1:15–2:15 | Part 2：生成数据，检查数据；在生成的串上给两个模型打分 | 12 + 388 |
| 2:15–3:30 | Part 3：六种示例条件，每种 300 条 | 1,800 |
| 3:30–4:00 | Part 4：换提示词写法和示例顺序 | 剩下的，约 200（试探用掉的从这里扣） |
| 4:00–5:00 | 整理 notebook，写结论和局限，准备讲解 | 0 |

Part 3 的六种条件里，有一种是把"只用 dev"原样再跑一遍。它花了 300 次调用，但它是整个实验的噪声底线：同样的设置跑两遍能差多少，就是"差别要大过多少才值得说"的直接答案。

### Part 1

**思路。** 先看原始回复长什么样，再写解析器。然后在同一批 300 条串上评估两个模型，用配对（paired）的办法比较：同一条串两个模型的对错相减，对 300 个差值做 t 区间。最后读错误，找出错误集中在哪里。

- 解析规则：回复里出现 `unbalanced` 或 `not balanced` 就是不配平；否则出现 `balanced` 就是配平；都没有返回 `None`。顺序不能反，因为 `unbalanced` 里也含有 `balanced`。
- 单个比例用 Wilson 区间（Wilson score interval）。
- "一组和另一组的差"用分层的 Agresti–Caffo 区间（每组先各加一次成功、一次失败再算）。

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

读这些输出：

- 回复有九种形状。其中 `The string is not balanced.` 里没有 `unbalanced`。如果解析器只看"有没有 `unbalanced`"，它会把这类回复读成配平，还会把 `I cannot tell without more context.` 也当成配平。用这样的解析器，两个模型的成绩会变成 182/300 和 238/300，而不是 201 和 260。
- 小模型 0.670 [0.615, 0.721]，大模型 0.867 [0.824, 0.901]。配对差值 +0.197 [+0.137, +0.256]，大模型确实更好。
- 读错误：小模型答错 97 条（另有 2 条解析不出），其中 89 条是把不配平的串说成了配平。看这 89 条串，它们有一个共同点：每种括号的左、右个数都相等，只是顺序不对，比如 `([)]`。看上去模型只数了个数。
- 于是把测试集分成三类：配平（115 条）；个数就不对（count，55 条）；个数对但顺序错（order，130 条）。小模型在 order 类上只有 0.308，大模型是 0.785。两个模型的差距几乎全在这一类上。
- 对照：order 类的串是不是只是更长？在每一类内部比较长串和短串（以长度中位数分开），小模型 +0.039 [−0.049, +0.114]，大模型 +0.075 [−0.010, +0.142]。两个区间都包含 0，点估计还是正的（长串反而略好）。所以没有证据说长度是原因，难的是"个数对但顺序错"这一类。大模型的上限是 +0.14，不能说长度完全没影响。

### Part 2

**思路。** 生成的数据在用之前要回答五个问题：格式能不能解析；有多少重复；标签对不对；分布和测试集像不像；有没有和测试集重叠。这个任务有一个现成的判题器（checker）：`is_balanced` 能给任何串算出正确标签。所以标签对不对可以直接算。没有判题器时怎么办，放在追问里。

解析生成的行时要宽容一点：允许行首有编号、串两边有引号。但不认识的行一律拒收，并且计数，不要悄悄"修好"。一条串出现多次时，只保留第一次的标签。

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

我自己生成的数据（12 次调用，每次 30 行）：

- 360 行里 17 行格式不对，被拒收。
- 343 条例子只有 194 条不同的串。模型会大量重复，其中 2 条串还前后给了相反的标签。
- 标签错误率 23/194 = 0.119 [0.080, 0.172]。但错误几乎全在 order 类：31 条里错了 22 条，0.710 [0.534, 0.839]；其他两类 163 条只错了 1 条。**生成器出错的地方，正是模型本来就不会的那一类。**
- 分布和测试集不一样：order 类只占 0.16（测试集 0.43）；平均长度 6.1（测试集 10.0）。生成的数据更短、更容易。
- 有 26 条串和测试集重合。这些串都不超过 8 个字符。短串本来就少，比如长度为 6 的配平串一共只有 40 个，撞上很正常。不管原因是什么，用作示例之前都要去掉。

同事的文件：

- 182 条不同的串里，有 103 条就在测试集里。
- 这不是巧合。我让生成器写的 343 条例子里，最长的只有 8 个字符；同事文件里却有 66 条比这更长，而且 66 条全都在测试集里。这些行是从测试集抄进来的，连标签一起。这也解释了为什么它的标签错误率只有 9/182，比我自己生成的低：抄来的标签是对的。

**"用生成的数据比较两个模型"行不行。** 先用 Part 1 的分类准确率按生成数据的类别比例重新加权，写下预测：小模型 0.850，大模型 0.901，差值约 +0.05。然后实际查询：

- 用生成器给的标签打分，大模型反而略差：−0.026 [−0.086, +0.034]。
- 用判题器的正确标签打分，差值是 +0.052 [−0.009, +0.112]，和事先的预测一致。
- 在真实测试集上，差值是 +0.197 [+0.137, +0.256]。

这里叠了两个问题。第一，生成数据的分布更容易，order 类太少，两个模型的差距被冲淡了。按类别重新加权的预测已经解释了这一部分。第二，生成器的标签恰好在 order 类上出错。大模型在这些串上答对了，却被错误的标签判成答错。结论：不行。用生成器自己的数据去评估，测到的是"像不像生成器"，不是"做得对不对"。

最后一行是一个不用判题器的检查：大模型的回答和生成器的标签不一致的串有 28 条，其中 15 条确实是标错的，而标错的一共 23 条。没有判题器的任务里，这类"让另一个模型复核"的办法能找出大部分错误，但也会误伤。这里有判题器，所以直接全部重新标注。

### Part 3

**思路。** 所有条件都用小模型、同一批 300 条测试串，每条串在每种条件下问一次，和"只用 dev"做配对比较。条件有六种：

- 只用 dev（12 条）：基线。
- 只用 dev，再跑一遍：噪声底线（A/A 对照）。它的区间应该包含 0。
- dev + 生成的数据，原样使用。
- dev + 生成的数据，清洗过：用判题器重新标注，去掉和测试集重合的串，剩 168 条。
- dev + 程序生成的 168 条：一个二十行的生成器，三类各占三分之一，不含测试串。这是"不用 LLM 能做到多好"的基线，条数和清洗过的那组一样。
- dev + 同事的文件。

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

- 噪声底线：同样的设置跑两遍，差 −0.030 [−0.085, +0.025]。区间包含 0，说明配对区间是诚实的；也说明 0.03 这样的差别不值得解读。
- 原样使用生成的数据，准确率反而下降：−0.163 [−0.222, −0.105]。order 类的准确率掉到 0.100。31 条 order 类示例里有 22 条标着"配平"，等于在示范"只数个数"这条错误规则。这不只是这一次运气不好：我在草稿里换了 40 个种子，只算替身里埋下的期望效果，原样使用平均是 −0.11，最好的一次也只有 +0.004。
- 清洗过的数据有用：+0.103 [+0.045, +0.161]。
- 但程序生成的数据也有用：+0.083 [+0.023, +0.144]。两者相比 +0.020 [−0.026, +0.066]，分不出高下。在这个任务上，LLM 生成的数据清洗之后，不比一个小程序生成的好。真正起作用的是判题器。
- 同事的文件看起来最好：+0.157 [+0.101, +0.212]。把测试串按"在不在同事文件里"分成两组再比较它和清洗过的数据：在文件里的 103 条，同事的文件高出 +0.155 [+0.054, +0.257]；不在文件里的 197 条，两者完全一样，+0.000 [−0.051, +0.051]。多出来的效果全部来自"答案就写在提示词里"的那些测试串。这是泄漏（leakage），不是数据质量。报告里应该说明这一点，并把同事的文件排除在外。
  这个拆分的统计功效其实不高。一条泄漏的串最多只能把"本来会答错"的那部分补上，而清洗过的示例已经能让小模型答对八成以上。草稿里换 40 个种子，埋下的抬高量在 +0.06 到 +0.11 之间，103 条左右的样本常常测不显著；这里的 +0.155 偏大，有运气成分。所以泄漏的证据是重合本身（103 条重合，66 条长串全在测试集里），拆分只是估计它把分数抬高了多少。就算拆分不显著，这个比较也已经不成立。
- 多重比较（multiple comparisons）：这里和基线比了五次。按 Bonferroni 校正，每个比较用 99% 区间。清洗过的数据的 99% 区间是 [+0.027, +0.180]，仍然不包含 0。

还有一点要写进局限：几种条件的示例条数不同（表里的 206、180、180、194 条都含 dev 的 12 条），"条数"和"来源"混在一起。清洗过的数据和程序生成的数据条数相同，所以这两者的比较是干净的。其余比较要带着这个限定来读。

### Part 4

**思路。** 剩下 200 次调用。从测试集里随机取 200 条，换一种 instruction 的写法，同时把清洗过的示例打乱顺序，和原来的提示词在同一批串上配对比较。同时看换了提示词之后，相对"只用 dev"的提升是否还在。剩下的调用不够再用新写法把"只用 dev"跑一遍，所以两边都和原提示词下的"只用 dev"比。

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

换写法、换顺序之后，差值是 +0.000 [−0.058, +0.058]；相对原提示词下的"只用 dev"，原提示词和新提示词的提升都是 +0.115，区间都不包含 0。结论不依赖这一处写法。局限：只试了一种新写法，只有 200 条，比 ±0.06 小的差别测不出来；"只用 dev"没有用新写法重跑，所以新写法本身对"只用 dev"有没有影响，这里看不出来。

**和替身里埋的真值对照。** 真实 take-home 没有这一步。练习时可以用它检查自己的结论。下表的"埋下的值"是替身的隐藏规则在同一批串上算出来的期望值。

| 量 | 埋下的值 | 上面的估计 |
| --- | --- | --- |
| 小模型在 order 类上的准确率 | 0.310 | 0.308，[0.235, 0.392] |
| 大模型减小模型 | +0.195 | +0.197，[+0.137, +0.256] |
| 长度本身的效应 | 0 | 小 +0.039，[−0.049, +0.114]；大 +0.075，[−0.010, +0.142] |
| 生成器在 order 类上标错的概率 | 0.72 | 22/31 = 0.710，[0.534, 0.839] |
| 生成器在其他两类上标错的概率 | 0.02 | 1/163，[0.001, 0.034] |
| 用生成器的标签比较两个模型 | −0.030 | −0.026，[−0.086, +0.034] |
| 原样使用生成数据的效果 | −0.156 | −0.163，[−0.222, −0.105] |
| 清洗后的效果 | +0.096 | +0.103，[+0.045, +0.161] |
| 同事文件比清洗数据多出的部分：泄漏的串 | +0.086 | +0.155，[+0.054, +0.257] |
| 同事文件比清洗数据多出的部分：其余的串 | +0.007 | +0.000，[−0.051, +0.051] |
| 提示词写法和示例顺序 | 0 | +0.000，[−0.058, +0.058] |

本页一共算了 28 个区间，全部包含埋下的值。只看一次运行说明不了区间方法对不对。我在草稿里换了 200 个种子，把整套分析重跑了一遍：前 26 个区间的平均覆盖率（coverage）是 0.963，最低的一个是 0.92。这个模拟要跑两分多钟，没有放进本页。

**notebook 的结构。**

```text
1. 问题和结论摘要：五行以内，每个结论带数字和区间
2. 设置：client、预算账本、解析器；解析器的测试；回复形状的计数
3. 数据：dev / test / 同事文件的来源和样子；判题器和它的测试
4. Part 1：两个模型的准确率；配对差值；按类别拆开；长度对照
5. Part 2：生成；数据检查表（格式、重复、标签错误、分布、重叠）；同事文件的泄漏证据
6. "用生成数据做评估"的反例：事先的预测、两种标签下的结果
7. Part 3：六种条件的设计和结果；噪声底线；泄漏的拆分
8. Part 4：换写法和顺序
9. 局限和下一步
附：预算使用表；全部原始回复的日志文件
```

**10 分钟讲解。**

```text
0:00-1:00  一句话结论：清洗过的生成数据有用（+0.10），但不比程序生成的好；原样用有害；同事的数据有泄漏
1:00-2:30  怎么打分：解析器的坑、区间、配对比较；大模型 +0.20，差距全在"个数对但顺序错"这一类
2:30-4:30  生成的数据长什么样：标错集中在难的那一类；分布更简单；和测试集重合
4:30-5:30  为什么不能用生成数据做评估：两种标签下的结果
5:30-8:00  有没有用：六种条件的表，噪声底线，同事文件的拆分
8:00-9:00  稳健性：换写法和顺序不影响结论
9:00-10:00 不知道什么：只有一个任务、300 条测试串、少样本这一种用法；条数和来源没完全分开；下一步做什么
```

**最容易栽的地方。**

- 数字不带区间；两种做法在不同的串上比较，或者不做配对。
- 解析器只认一种说法，`not balanced` 被读成配平。
- 生成的数据不检查就用：重复、标签错误、分布偏移、和测试集重合。
- 只算总的标签错误率（0.12），没发现错误全集中在最难的那一类（0.71）。
- 用生成器自己的数据和标签去评估模型，得出"两个模型差不多"。
- 看到同事的文件效果最好就直接用，没有追问效果从哪里来。
- 没有噪声底线，把 0.03 这样的差别当成发现。
- 没有"不用 LLM"的基线，把清洗的功劳记在 LLM 头上。
- 预算前松后紧，最后没有调用做稳健性检查。

### 追问

**这个任务有判题器。如果没有（比如自然语言任务），生成数据怎么清洗？** 第一步是抽一个随机样本人工标注，估计标签错误率，并且带区间。还要按类别分开估计，因为错误往往集中在难的那一类。然后用便宜的办法过滤：让另一个模型复核，或者让同一个模型多答几次，只保留答案一致的。这里"让大模型复核"找出了 23 个错误里的 15 个，但 28 条被标记的里有 13 条是误伤。过滤之后要再抽样，看错误率降了多少。测试集必须是人工核对过的，不能来自生成器。

**既然程序生成的一样好，为什么还要用 LLM 生成？** 在这个任务上确实没必要。有程序能生成、又有程序能判对错的任务，用程序更便宜、更可控。LLM 生成的价值在没有生成程序的场景：自然语言的多样性、真实用户会写的那种输入。但那种场景通常也没有判题器，所以上一个问题的清洗办法更重要。这道题想让你看到的是：效果来自"标签对不对"，而不是"数据是谁写的"。

**示例是不是越多越好？** 这里的比较没法回答这个问题，因为条数和来源一起变了。要回答它，就固定来源（比如清洗过的数据），只改条数：12、45、90、180 条，每种都在同一批测试串上跑，画出准确率随条数的曲线，每个点带区间。用真实模型时还要算账：示例越多，每次调用的输入越长，费用和延迟都会上升，可能在某个条数之后不再划算。

**泄漏在一般情况下怎么查？** 先做精确匹配，统一大小写、空白和标点之后再比。自然语言里还要查近似重复：n-gram 重叠或者编辑距离。查出来之后像这里一样拆分：在重叠和不重叠的测试样本上分别报告效果。如果效果只在重叠的样本上出现，就是泄漏。来源不明的数据，先问清楚它是怎么来的；问不清楚就不用。

**换成真实模型，哪些地方要变？** 代码只换 client 那一行，但还有几件事要做。真实模型在这么短的串上可能接近满分，要先做一个小规模试探，必要时把串加长，避免天花板效应。回复形状会更多，解析器要对着真实回复重写。生成数据的毛病也未必和替身一样，所以检查表要整套做，不能只查替身里埋的那几项。几百条示例会让每次调用的输入变长，可以把固定的示例放在提示词前面、测试串放在最后，用 API 的提示词缓存（prompt caching）省钱。但 Part 4 打乱示例顺序时，缓存会失效，要把这部分费用算进预算。调用有延迟和限流，要并发并处理重试。

<details>
<summary>验证代码（可运行）</summary>

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
