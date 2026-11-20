# 只靠 API 研究模型行为：多位数加法为什么会错

## 题目

这是一道 5 小时的研究型 take-home。你拿到一个语言模型的 API，别的什么都没有：看不到权重，拿不到 logits，也不能微调。

有人注意到：**这个模型做多位数加法时，数字越长越容易错。** 你的任务是弄清楚错误到底由什么决定，并给出一个能定量预测"某一道题做对的概率"的解释。题目范围是两个正整数相加，每个数最多 20 位。

**条件**

- 只能通过 `client.complete(prompt) -> str` 访问模型。每次调用是一段全新的对话，没有系统提示，也没有任何采样参数。同一个提示词问两次，回复可能不同。
- 总共 3,000 次调用，试探用的调用也算在内。超出后 client 直接报错。
- 5 小时，从打开题目算起。
- 这里的模型是研究对象，只能通过 client 调用。写代码时能不能用 AI 助手，以邀请信为准；没写就当作不能。
- 交付两样东西：一个可以从头跑到尾的 notebook，和一次 10 分钟的讲解（讲完是问答）。

**评分看什么**

- 结论有没有证据支撑：有没有对照（control），数字有没有置信区间（confidence interval），样本量是不是事先想过的。
- 有没有把同时变化的因素拆开，而不是看到一条下降曲线就下结论。
- 有没有在**没用来拟合的数据**上检验自己的假设；检验失败时怎么处理。
- 有没有读过原始回复和错误答案本身，而不只是看准确率。
- 结论对提示词写法是否稳健。
- 预算和时间用得是否有章法；notebook 能否复现；讲解有没有说清"知道什么、不知道什么"。

### Part 1 —— 搭好量具，确认现象

写三样东西：

1. 一个带预算的 client 包装：给调用计数，超预算就报错，并保存每一次调用的（标签，提示词，原始回复）。
2. 一个从回复里取出答案的解析函数。模型的回复格式不固定。解析不出数字的回复要单独计数，不能悄悄算成"答错"，也不能悄悄丢掉。
3. 一个预实验（pilot）：在 2、4、8、12、16、20 位这六档上各取 50 道随机加法题（两个加数位数相同，首位不为 0），报告每一档的准确率和 95% 置信区间。

```py
class BudgetExceeded(RuntimeError): ...

class BudgetedClient:
    def __init__(self, client, budget: int): ...
    def complete(self, prompt: str, tag: str = "") -> str: ...   # raises BudgetExceeded once the budget is used up

def parse_answer(reply: str) -> int | None: ...                  # None: the reply contains no number
```

### Part 2 —— 把混在一起的因素拆开

位数变多时，至少有两样东西在一起变：数字本身的长度，和竖式加法里进位（carry）的次数。一列的两个数字相加，再加上低位来的进位，结果达到 10，就算这一列有一次进位。例如：

```text
  4821
+ 3579
------
  8400      个位 1+9、十位 2+7(+1)、百位 8+5(+1) 都进位，千位 4+3(+1)=8 不进位：这道题有 3 次进位
```

设计一组实验把"长度"和"进位次数"这两个因素分开，各自给出效应的大小和区间。然后提出一个定量假设：给定一道题，模型做对的概率是多少。估计出假设里的参数，带区间。

### Part 3 —— 在没见过的数据上检验自己的假设

发出查询之前，先用 Part 2 的假设写下对一批新题的预测（预测值和区间）。新题要来自拟合时没有用过的题型。然后查询，对比。如果预测落空：读错误答案本身，找出假设漏掉了什么，修正假设，再找一批**新的**数据检验修正后的版本。

### Part 4 —— 稳健性与交付

1. 再换至少三种提示词写法，在**同一批题**上重复测量，说明哪些结论依赖写法，哪些不依赖。
2. 给出 notebook 的结构和 10 分钟讲解的提纲。

### 离线练习用的替身

真实的 take-home 会给你一个 API key 和一个模型名。这一页在运行时没法调用真实 API，所以用一个**模拟模型**（simulated model）代替：一个确定性的 Python 类，里面埋了一些做加法时的"毛病"，随机性由种子固定。它只是练习用的替身，不代表任何真实模型的行为。你的分析代码只依赖 `complete(prompt) -> str` 这一个接口；换成真实 API 时只需要换掉 client，其余代码一行都不用改。

练习时把下面折叠里的代码原样复制运行，**先不要读它**，读了就等于提前看了答案。它能看懂任何包含 `a + b` 形式算式的提示词。

<details>
<summary>模拟模型的源码（做题前不要展开）</summary>

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

用法：

```py
client = BudgetedClient(SimulatedModel(seed=0), 3000)
print(client.complete("What is 4821 + 3579?"))
```

真实 API 下的 client 大致是这样（官方 Anthropic Python SDK；这一段不在本页运行）。`MODEL` 是题目里指定的模型名。

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

## 参考解答

<details>
<summary>展开参考解答</summary>

这类题没有标准答案。评分的人看的是你怎么从"一条下降的曲线"走到"一个被检验过的解释"。下面是一条完整的路线，所有数字都是本页代码用替身跑出来的。

### 先定时间和预算

动手之前花 10 分钟把时间和调用次数分好。最常见的失败是前两个小时把预算花在一条越画越细的曲线上，后面没有调用可以做检验。

| 时间 | 做什么 | 调用 |
| --- | --- | --- |
| 0:00–0:30 | 读题；手动问几个问题，看回复长什么样；写 client 包装、解析器、日志 | 手动试探几十次以内 |
| 0:30–1:00 | 预实验：现象是否存在，回复有哪些格式，多少回复不给答案 | 300 |
| 1:00–2:15 | 对照实验：分开长度和进位；拟合第一个假设 | 960 |
| 2:15–3:15 | 写下预测，做留出检验；读错误答案；修正假设；再做一次留出检验 | 300 + 300 |
| 3:15–4:00 | 提示词写法的敏感性；复现意外结果 | 750 + 390 |
| 4:00–5:00 | 整理 notebook，写结论和局限，准备讲解 | 0 |

参考解答把 3,000 次全部记在账上。真做题时，手动试探和返工也要从这 3,000 次里出，最后那 390 次就是为这类事留的余量。

样本量怎么定：比例的 95% 区间半宽大约是 $1.96\sqrt{p(1-p)/n}$。按最坏的 $p=0.5$ 算，$n=50$ 时约 ±0.14，$n=80$ 时约 ±0.11，$n=250$ 时约 ±0.06。预实验只看趋势，每档 50 题就够。对照实验要分辨 0.1 左右的差别，所以每格 80 题，再按因素合并。提示词比较用同一批 250 题做配对。

### Part 1

**思路。** 先把量具做对。量具有三个零件：预算账本、解析器、题目生成器。哪一个有问题，后面的数字都不可信。

- 账本：每次调用都带一个标签，原始回复全部保存。之后的分析只读日志，这样 notebook 重跑时不需要再花调用。
- 解析器：先看回复长什么样，再写规则。办法是把回复里的数字都换成 `N`，统计有几种"形状"。这里的规则是"取回复里最后一个整数，允许千分位逗号"。取最后一个，是因为模型有时会把算式重抄一遍再给结果。
- 没有数字的回复返回 `None`，单独计数。
- 区间用 Wilson 区间（Wilson score interval）。样本小或者比例接近 0、1 时，它比"估计值 ± 1.96 个标准误"可靠。

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

预实验：六档位数，每档 50 道随机题。顺手把每道题的进位次数也算出来。`carry_profile` 把进位分成了两类，Part 3 才会用到；这里只用它们的和。

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

读这张表：

- 现象是真的。2 位题对了 47/50，20 位题只有 28/50，两个区间不重叠。
- 回复有四种形状：207 个只有数字，49 个是带千分位逗号的整句，34 个把算式重抄了一遍，10 个没有给答案。
- 表里中间那一列是"整个回复必须是一个整数"的严格解析器。它在 2 位题上只认 29/50。少掉的那 18 题不是算错，是格式不合它的意。用它的话，准确率被压在四到六成，下降的趋势也几乎看不出来（29 到 22）。
- 位数和进位次数在预实验里的相关系数是 0.88。这条曲线分不清是长度在起作用，还是进位在起作用。

### Part 2

**思路。** 要把两个因素分开，就不能用随机题，要自己造题：指定位数，再指定进位次数。做一个 3×4 的网格：位数取 8、14、20，进位次数取 0、2、4、6，每格 80 题。

造题的办法：先选好哪几列进位。进位的列让两个数字之和不小于 10，其余的列让和不超过 8。为什么不是"不超过 9"？因为和为 9 的列会被低位来的进位推到 10，进位次数就不受控制了。这样做有代价：网格里完全没有"和为 9"的列。先记下来。

效应的估计是分层的差值：在每一种进位次数下比较 20 位和 8 位，再把四层平均，这是长度的效应；在每一种位数下比较 6 次进位和 0 次进位，再把三层平均，这是进位的效应。区间用 Agresti–Caffo 的做法（每组先各加一次成功、一次失败再算），比例接近 1 时也稳。

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

- 进位次数固定时，20 位比 8 位只差 −0.033，区间 [−0.082, +0.019] 包含 0。这是一个"没有效应"的对照。
- "区间包含 0"不等于"没有效应"。准确的说法是：长度本身的影响最多 8 个百分点左右，很可能更小。
- 位数固定时，6 次进位比 0 次进位低 0.206，区间 [−0.259, −0.143]。同一个估计方法在进位上能测出效应，说明前一条"测不出长度效应"不是因为方法不灵。

**定量假设 H1。** 每一次进位都独立地以概率 $q$ 丢失；另外有一个和进位无关的小错误率 $r_0$。一道有 $c$ 次进位的题做对的概率是

$$P(\text{对}) = (1-r_0)\,(1-q)^{c}.$$

用极大似然（maximum likelihood）估计。令 $\theta = -\log(1-r) \ge 0$，对数似然对 $\theta$ 是凹的，用带边界的 L-BFGS 就能稳定求解。代码把相同进位数的题先合并再算，所以很快。

参数的区间用剖面似然（profile likelihood）：把一个参数固定在某个值上，其余参数重新拟合；对数似然比最优值低不到 1.92 的那些值组成 95% 区间（1.92 是 $\chi^2_1$ 的 95% 分位数 3.84 的一半）。

为什么不用更顺手的自助法（bootstrap）？我在草稿里用参数已知的模拟检查过覆盖率（coverage）：重复 200 次，bootstrap 百分位区间对 $r_0$ 的覆盖率只有 88% 到 89%，因为 $r_0$ 主要靠零进位那几格里少数几个错误来定，又贴着 0 这个边界。剖面似然区间重复 600 次，各参数的覆盖率在 94.7% 到 96% 之间。所以参数区间用剖面似然。bootstrap 只用来给下一步的预测区间提供参数抽样；预测区间重复 250 次，覆盖率在 95.6% 到 96.4% 之间。这些模拟要跑十几分钟，没有放进本页。

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

每次进位丢失的概率估计是 0.037，区间 [0.028, 0.045]。到这里，故事看上去很完整：长度无关，每次进位有 4% 左右的概率丢掉。

### Part 3

**思路。** 假设在拟合它的数据上表现好，不算证据。要让它去预测一批没见过的题，而且预测要在查询之前写下来。

留出数据选 300 道完全随机的 16 位题。网格里的题是特意构造的，随机题是"自然"的，题型不同。预测区间要同时包含两种不确定性：参数没估准，和新数据本身的抽样噪声。做法是对每一组 bootstrap 参数，按假设给的概率把每道题模拟一遍，取模拟结果的 2.5% 和 97.5% 分位数。

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

294 道题给了答案。H1 预测对 216 道，区间 [196, 236]；实际只对了 177 道，远在区间下方。**H1 被否定了。**

这时不要急着调参数。H1 在网格上没问题，在自然题上高估，说明自然题里有网格里没有的东西。答案藏在错误里，要去读错误答案。

加法有一个好用的性质：竖式里第 $k$ 列的进位如果丢了，答案正好比正确值少 $10^{k+1}$。所以只要"正确值 − 模型的答案"是 10 的幂，就能直接定位丢进位的那一列，再看那一列的两个数字之和是多少。

- 117 个错误里，90 个恰好是"丢了一次进位"。
- 其中 59 个丢在数字和不小于 10 的列，这一列自己产生进位（generated carry）。
- 另外 31 个丢在数字和等于 9 的列。这种列自己不进位，是被低位来的进位推过 10 的，它只是把进位传上去（propagated carry）。
- 传递进位只占这批题全部进位的 9.5%，却占了 90 个错误里的 31 个，也就是三分之一。

这正是网格的盲区：造题时为了控制进位次数，把和为 9 的列全排除了。

**修正后的假设 H2。** 两种进位各有各的丢失率。一道题有 $g$ 次产生进位、$p$ 次传递进位，做对的概率是

$$P(\text{对}) = (1-r_0)\,(1-q_g)^{g}\,(1-q_p)^{p}.$$

用网格题和自然题一起拟合：

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

传递进位的丢失率是 0.301，区间 [0.212, 0.392]；产生进位是 0.036，区间 [0.029, 0.044]。两个区间离得很远。

但 H2 是看了自然题的错误之后才提出来的，自然题不能再拿来当检验。需要一批新数据，并且要挑 H1 和 H2 预测差别大的题型，检验才有分辨力。这里用进位链（carry chain）：12 位的题，某一列产生进位，紧接着的 $L-1$ 列数字和都是 9，其余列不进位。取 $L = 3, 5, 7$，各 100 题。这么长的链在之前的数据里很少见。

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

三档的实际结果都落在 H2 的预测区间里，都远在 H1 的区间下方。链长为 7 时，H1 预测对 73 道左右，实际只对了 15 道。

结论的说法要注意：H2 通过了一次它本来可能失败的检验。这不等于"H2 是对的"，更不等于知道了模型内部是怎么算的。它只是一个在行为层面上站得住的描述。

### Part 4

**提示词敏感性。** 取自然题的前 250 道，换三种写法各问一遍：

- `terse`：换一种措辞，同样要求只给结果。
- `open`：只说"Compute a + b."，不要求只回数字。
- `grouped`：措辞和原来一样，但加数写成带千分位逗号的形式。

比较用配对（paired）的办法：同一道题在两种写法下的对错相减，对这 250 个差值做 t 区间。配对把"这道题本身难不难"消掉了，区间比两组独立样本窄。两种写法里只要有一种没给答案，这道题就不参与比较。

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

- `open` 和原写法没有差别：+0.004，区间 [−0.059, +0.067]。但如果用严格解析器，会得到 −0.336。原因是不要求"只回数字"时，模型更常回整句话。这是量具造出来的假效应，和模型的算术能力无关。
- `grouped` 是真效应，而且很大：+0.215，区间 [+0.152, +0.279]。所以结论要加一个限定：丢进位的概率取决于数字的书写格式。
- `terse` 是个意外：+0.072，区间 [+0.017, +0.127]，不包含 0。可是没有任何机制上的理由。同时做三个比较，至少出现一个假阳性的概率是 $1-0.95^3 \approx 14\%$，并不稀奇。

意外结果既不能挑着报，也不能悄悄删掉。正确的处理是在新题上复现一次。剩下的 390 次调用正好够 195 道新题、两种写法：

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

复现的结果是 +0.011，区间 [−0.067, +0.089]。报告里写"`terse` 没有稳定的效应"，并把两次的结果都留在 notebook 里。

**和替身里埋的真值对照。** 真实 take-home 没有这一步。练习时可以用它检查自己的结论：

| 量 | 埋下的值 | 上面的估计 |
| --- | --- | --- |
| 产生进位的丢失率 | 0.04 | 0.036，[0.029, 0.044] |
| 传递进位的丢失率 | 0.25 | 0.301，[0.212, 0.392] |
| 和进位无关的错误率 | 0.015 | 0.020，[0.007, 0.041] |
| 长度本身的效应 | 0 | −0.033，[−0.082, +0.019] |
| 措辞的效应 | 0 | `open` +0.004，[−0.059, +0.067]；`terse` 复现 +0.011，[−0.067, +0.089] |
| 千分位分组 | 两种丢失率都减半，在这批题上平均是 +0.164 | +0.215，[+0.152, +0.279] |
| 不给答案的比例 | 0.03 | 97/3000 |

**notebook 的结构。**

```text
1. 问题和结论摘要：五行以内，每个结论带数字和区间
2. 设置：client、预算账本、解析器；解析器的测试；各种回复形状的计数
3. 预实验：现象；位数和进位混在一起的证据
4. 对照实验：造题方法、网格表、两个效应的区间
5. 假设 H1 和它的参数
6. 留出检验一：事先写下的预测 -> 结果 -> 错误分析
7. 假设 H2；留出检验二（进位链）
8. 提示词敏感性；意外结果的复现
9. 局限和下一步
附：预算使用表；全部原始回复的日志文件
```

**10 分钟讲解。**

```text
0:00-1:00  一句话结论，配一张图：进位链长度 vs 准确率，叠上 H1 和 H2 两条预测线
1:00-3:00  怎么量的：解析规则、不给答案的回复、区间；严格解析器的反例
3:00-5:00  拆因素：网格表；长度的区间含 0，进位的区间不含 0
5:00-7:30  H1 预测失败 -> 读错误 -> H2 -> 新的检验通过
7:30-9:00  稳健性：措辞无关，分组有关，一个没能复现的意外
9:00-10:00 不知道什么：只是行为层面的描述；只测了加法、20 位以内、英文提示词；下一步做什么
```

**最容易栽的地方。**

- 只画一条"准确率随位数下降"的曲线就下结论。位数和进位是一起变的。
- 解析器只认一种格式，把格式问题算成算术错误；或者换了提示词之后格式变了，得出假的提示词效应。
- 数字不带区间；或者把"区间包含 0"说成"没有效应"。
- 假设只在拟合它的数据上检验；或者先看了数据再"预测"。
- 检验没有分辨力：新旧两个假设在检验题上的预测差不多。
- 意外的显著结果直接写进结论，或者悄悄删掉。
- 前半段把预算花光，没有给留出检验和复现留调用。
- 不读原始回复，不读错误答案。
- 把同一道题的多次回答当成独立样本。
- notebook 不能从头重跑；没有保存原始回复。

### 追问

**同一道题重复问 5 次，能不能当成 5 个独立样本？** 不能。同一道题的几次回答是相关的。在替身上拿 400 道题各问两次，两次对错的相关系数是 0.46。按设计效应（design effect）$1+(m-1)\rho$ 算，$m=5$、$\rho=0.46$ 时是 2.84，5 次重复只相当于 1.8 个独立样本。预算固定时，多问不同的题比重复问同一题划算。只有想估计"同一道题内部的波动"时才值得重复。

**API 没有 temperature 这类参数，回复又不确定，怎么保证可复现？** 能复现的是分析，不是模型的回复。保存每一次调用的原始回复，分析只读日志；题目生成器固定种子；notebook 重跑时回放日志，不重新调用。报告里写明模型名和调用日期。另外，不要给相同的提示词做缓存。缓存会把同一题的波动抹掉，重复测量就失去了意义。

**不给答案的回复怎么算？** 两种口径都报。一种把它们算成错，回答的是"用户拿到正确答案的概率"。一种只在给了答案的题上算，回答的是"它算得对不对"。更重要的是检查"不给答案"和题目难度有没有关系。这里 3,000 次调用里有 97 次没给答案，约 3%；网格里进位少的一半和进位多的一半各有 15 次。所以可以当成随机缺失，两种口径的结论一样。如果难题上不给答案的更多，只看给了答案的题就会高估。

**换成真实模型，哪些地方要变？** 代码只换 client 那一行，但分析不能照搬。回复的形状会更多：解题步骤、科学计数法、拒答。解析器要对着真实的原始回复重写。真实模型在这个长度上可能根本不出错，或者错法完全不同，假设必须从它自己的错误里来，不要先入为主地去找"传递进位"。调用有延迟和限流，要并发并处理重试。带思考（thinking）的模型要把 `max_tokens` 留够；被截断的回复（`stop_reason` 是 `max_tokens`）算格式失败，不算答错。

**比较了好几种提示词、好几个假设，多重比较（multiple comparisons）怎么处理？** 把比较分成两类。确认性的比较在看数据之前就定好，这里是长度效应、进位效应和两次留出预测，照常用 95% 区间。探索性的比较是做的过程中冒出来的，比如三种提示词写法。探索性的发现一律先在新数据上复现，再当结论。也可以直接收紧区间：三个比较各用 98.3% 的区间（Bonferroni 校正）。

<details>
<summary>验证代码（可运行）</summary>

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
