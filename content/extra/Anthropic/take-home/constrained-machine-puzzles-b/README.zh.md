# Take-home：受限指令集谜题（二）只有一条队列的机器

## 题目

> **这不是原题。** Anthropic 现在的性能 take-home 是一组在很小、限制很多的指令集上做的谜题，目标是用最少的指令解题。真实的机器和谜题没有公开，这台机器和下面 4 道题都是我们自己设计的。它和[第一套练习](#/anthropic/take-home/constrained-machine-puzzles-a)的机器风格完全不同：那一台靠寄存器和条件前缀，这一台没有寄存器，所有数据都在一条队列里。

你会拿到一台小机器的说明、一份检查器（checker），和 4 道互相独立的小题。每道题交一段这台机器的程序。检查器只告诉你两件事：程序是否有效，一共几条指令。目标是让 4 道题的指令总数尽量少。

### 机器

- 唯一的存储是一条先进先出的队列（FIFO queue），里面是 0..99 的整数，最多 16 个。没有寄存器。
- 开始时队列里是这道题的输入。程序停下时队列里剩下的数（从队首到队尾）就是输出。
- 指令从**队首**取数，把结果放到**队尾**。
- 算术都饱和（saturate）：加法超过 99 记 99；减法低于 0 记 0。后面把这种减法写成 `a ∸ b = max(a − b, 0)`（截断减法，monus）。
- 程序有 1–12 条指令，编号从 0 开始，从第 0 条开始顺序执行。跳转目标可以是 `0..n`，`n` 是程序长度，跳到 `n` 表示停机。
- **停机**有两种：执行到第 `n` 条（走完最后一条，或者跳到 `n`）；或者当前指令要取的数比队列里现有的多，这时这条指令**不执行**，直接停机（下文叫"饿死"，starvation）。
- 队列超过 16 个数算无效；每个测试超过 20,000 步也算无效。

| 指令 | 取几个数 | 效果（队列 `a b …`，左边是队首） |
| --- | --- | --- |
| `ADD` | 2 | 取出 `a`、`b`，放入 `min(a + b, 99)` |
| `SUB` | 2 | 取出 `a`、`b`，放入 `a ∸ b` |
| `DUP` | 1 | 取出 `a`，放入 `a`、`a` |
| `ROT` | 1 | 取出 `a`，放入 `a`（队首挪到队尾） |
| `DROP` | 1 | 取出 `a`，丢掉 |
| `INC` | 1 | 取出 `a`，放入 `min(a + 1, 99)` |
| `DEC` | 1 | 取出 `a`，放入 `a ∸ 1` |
| `ZERO` | 0 | 放入 `0` |
| `JMP k` | 0 | 下一步执行第 `k` 条 |
| `JZ k` | 1 | 取出 `a`；`a = 0` 就跳到第 `k` 条，否则顺序执行下一条 |

指令之间用换行或 `;` 分隔，`#` 后面是注释。

例子：求所有数之和的程序 `ADD; JMP 0`，输入 `3 4 5`。

```text
步  位置  指令    执行后的队列
          开始    3 4 5
 0   0   ADD     5 7
 1   1   JMP 0   5 7
 2   0   ADD     12
 3   1   JMP 0   12
 4   0   ADD     要取 2 个数，队列里只有 1 个：饿死，停机
输出：12
```

### 检查器

这就是你拿到的全部工具：

```py
"""Validity checker for the queue machine. This file is all the tooling you get."""
import random
from collections import deque

LIMIT = 99            # values are integers 0..LIMIT
CAP = 16              # the queue holds at most CAP values
MAX_LEN = 12          # a program has 1..MAX_LEN instructions
MAX_STEPS = 20_000    # per test; a program that runs longer is invalid
NEED = {"ADD": 2, "SUB": 2, "DUP": 1, "ROT": 1, "DROP": 1,     # values an instruction pops
        "INC": 1, "DEC": 1, "ZERO": 0, "JMP": 0, "JZ": 1}


def parse(source):
    """Instructions are separated by newlines or ';'. '#' starts a comment. Returns [(op, arg)]."""
    prog = []
    for line in source.replace(";", "\n").splitlines():
        parts = line.split("#")[0].split()
        if not parts:
            continue
        op = parts[0]
        if op not in NEED or len(parts) != (2 if op in ("JMP", "JZ") else 1):
            raise ValueError(f"bad instruction: {line.strip()!r}")
        prog.append((op, int(parts[1]) if len(parts) == 2 else None))
    if not 1 <= len(prog) <= MAX_LEN:
        raise ValueError(f"a program has 1..{MAX_LEN} instructions")
    if any(arg is not None and not 0 <= arg <= len(prog) for _, arg in prog):
        raise ValueError("jump target out of range")      # target len(prog) means "halt"
    return prog


def run(prog, values):
    """Final queue (front first), or None on overflow or after MAX_STEPS steps."""
    q = deque(values)
    pc = 0
    for _ in range(MAX_STEPS):
        if pc == len(prog):
            return list(q)                    # ran off the end (or jumped to len(prog))
        op, arg = prog[pc]
        if len(q) < NEED[op]:
            return list(q)                    # starved: halt without executing it
        pc += 1
        if op == "ADD":
            a = q.popleft(); b = q.popleft(); q.append(min(a + b, LIMIT))
        elif op == "SUB":
            a = q.popleft(); b = q.popleft(); q.append(max(a - b, 0))
        elif op == "DUP":
            a = q.popleft(); q.append(a); q.append(a)
        elif op == "ROT":
            q.append(q.popleft())
        elif op == "DROP":
            q.popleft()
        elif op == "INC":
            q.append(min(q.popleft() + 1, LIMIT))
        elif op == "DEC":
            q.append(max(q.popleft() - 1, 0))
        elif op == "ZERO":
            q.append(0)
        elif op == "JMP":
            pc = arg
        elif op == "JZ":
            if q.popleft() == 0:
                pc = arg
        if len(q) > CAP:
            return None                       # overflow
    return None


# ---- the four puzzles: a reference function (the spec) and the test inputs ----

PAIRS = [[a, b] for a in range(LIMIT + 1) for b in range(LIMIT + 1)]
PAIRS_98 = [[a, b] for a in range(LIMIT) for b in range(LIMIT)]       # Part 4: both inputs <= 98
_rng = random.Random(2026)
LISTS = [list(range(1, k + 1)) for k in range(1, CAP)] + [[LIMIT] * (CAP - 1)] + [
    [_rng.randint(1, LIMIT) for _ in range(_rng.randint(1, CAP - 1))] for _ in range(300)]

PUZZLES = {
    "q1": (lambda q: [max(q)], PAIRS),
    "q2": (lambda q: [abs(q[0] - q[1])], PAIRS),
    "q3": (lambda q: [len(q)], LISTS),
    "q4": (lambda q: [int(q[0] == q[1])], PAIRS_98),
}


def check(name, source):
    """(valid, instruction count). Valid: on every test the program halts with exactly the right queue."""
    prog = parse(source)
    spec, tests = PUZZLES[name]
    return all(run(prog, t) == spec(t) for t in tests), len(prog)


def report(submission):
    """submission: {puzzle name: program text}. Prints one line per puzzle, returns the total score."""
    total = 0
    for name in PUZZLES:
        valid, count = check(name, submission[name]) if name in submission else (False, 0)
        total += count if valid else 20
        print(f"{name}: {'valid' if valid else 'INVALID':7s} {count:2d} instructions")
    print(f"total score: {total}")
    return total
```

### 规则与计分

- 每道题交一段程序。**有效**指：在这道题的每个测试上都会停机，停机时队列**恰好**是要求的输出（多一个数、少一个数都不行），中途队列没有超过 16 个数。
- **分数 = 4 道题的指令条数之和**，越少越好。没交或无效的题记 20 分。
- 检查器不会告诉你哪个测试错了。没有调试器，需要工具就自己做：加 print，或者让 AI 编程助手写一个单步跟踪器。花多少时间做工具，本身就是考察的一部分。
- 时间 2 小时，可以使用 AI 助手。
- 判分只看给定的测试，但请按题目写明的输入范围来解题。

### Part 1 —— 较大值

输入是两个数 `a b`，都在 0..99。输出一个数：`max(a, b)`。

```text
3 8   ->  8
5 5   ->  5
99 0  ->  99
```

及格线 11 条，目标 3 条。

### Part 2 —— 差的绝对值

输入是两个数 `a b`，都在 0..99。输出一个数：`|a − b|`。

```text
3 8   ->  5
8 3   ->  5
0 99  ->  99
```

及格线 8 条，目标 6 条。

### Part 3 —— 数个数

输入是 1 到 15 个数，每个在 **1..99**（没有 0）。输出一个数：输入有几个数。

```text
4 4 9         ->  3
99            ->  1
7 1 2 3 99 5  ->  6
```

及格线 8 条，目标 6 条。

### Part 4 —— 相等

输入是两个数 `a b`，都在 **0..98**（这一题不会出现 99）。`a = b` 输出 `1`，否则输出 `0`。

```text
7 7   ->  1
7 2   ->  0
0 98  ->  0
```

及格线 11 条，目标 8 条。

4 道题的及格线合计 38 条，目标合计 23 条。

## 参考解答

<details>
<summary>展开参考解答</summary>

先给结论：

| 题 | 我们的第一版 | 参考解 | 关键想法 | 最优性 |
| --- | --- | --- | --- | --- |
| Part 1 较大值 | 11 | **3** | `max(a, b) = a + (b ∸ a)`，不用比较 | 穷举证明最优，且唯一 |
| Part 2 差的绝对值 | 7 | **6** | `(a ∸ b) + (b ∸ a)`，不用分支；按取用顺序摆好操作数 | 穷举证明最优（唯一性是离线确认的） |
| Part 3 数个数 | 7 | **6** | 先把每个数换成 0，再利用"个数 + 总和"不变 | 穷举证明最优 |
| Part 4 相等 | 11 | **8** | `(a + 1) ∸ b` 一次分出小于、等于、大于；程序开头兼作"输出 1" | 离线穷举证明最优 |

合计 23 条。这台机器上的技巧和第一套完全不同：那里是把状态藏进标志位和条件前缀，这里是**截断减法的恒等式**、**按队列顺序排操作数**、**用饿死当出口**。

### 先写一个单步跟踪器

和第一套一样，第一件事是写一个跟踪器（tracer），每一步打印位置、指令和执行后的整条队列。它和检查器分开写，验证代码里会拿两者互相对照。

```python
def trace_q(source, values, limit=200, show=True):
    """Step-by-step log of the queue machine, written separately from the starter.
    Returns (final queue or None, lines); None means overflow or step limit."""
    prog = []
    for raw in source.replace(";", "\n").splitlines():
        words = raw.split("#")[0].split()
        if words:
            prog.append((words[0], int(words[1]) if len(words) > 1 else None))
    pops = dict(ADD=2, SUB=2, DUP=1, ROT=1, DROP=1, INC=1, DEC=1, ZERO=0, JMP=0, JZ=1)
    q, pc, lines = list(values), 0, []
    show_q = lambda: " ".join(map(str, q)) or "(empty)"
    lines.append(f"            start    {show_q()}")
    for step in range(limit):
        if pc == len(prog):
            lines.append(f"{step:4d} {pc:3d}  end of program: halt")
            break
        op, arg = prog[pc]
        head = f"{step:4d} {pc:3d}  {op + ('' if arg is None else f' {arg}'):7s}"
        if len(q) < pops[op]:
            lines.append(f"{head}  needs {pops[op]}, queue has {len(q)}: halt")
            break
        pc += 1
        if op in ("ADD", "SUB"):
            a, b = q.pop(0), q.pop(0)
            q.append(min(a + b, 99) if op == "ADD" else max(a - b, 0))
        elif op == "DUP":
            q += [q.pop(0)] * 2
        elif op in ("ROT", "INC", "DEC"):
            a = q.pop(0)
            q.append(a if op == "ROT" else min(a + 1, 99) if op == "INC" else max(a - 1, 0))
        elif op == "DROP":
            q.pop(0)
        elif op == "ZERO":
            q.append(0)
        elif op == "JMP":
            pc = arg
        elif op == "JZ" and q.pop(0) == 0:
            pc = arg
        lines.append(f"{head}  {show_q()}")
        if len(q) > 16:
            lines.append("queue overflow: invalid")
            q = None
            break
    else:
        lines.append("step limit reached")
        q = None
    if show:
        print("\n".join(lines))
    return q, lines


def first_failure(name, source, limit=200):
    """Trace the first test of a puzzle that the program gets wrong (uses the starter's PUZZLES)."""
    spec, tests = PUZZLES[name]
    for values in tests:
        got, _ = trace_q(source, values, limit=MAX_STEPS, show=False)
        if got != spec(values):
            print(f"input {values}: want {spec(values)}, got {got}")
            trace_q(source, values, limit=limit)
            return values
    print("all tests pass")
```

在这台机器上，跟踪器比在第一台上更重要：队列里的数一直在转，靠脑子记"现在队首是谁"很快就会乱。

### Part 1 —— 3 条：截断减法代替比较

```text
DUP       # a b      ->  b a a
SUB       # b a a    ->  a (b ∸ a)
ADD       # a (b ∸ a) -> a + (b ∸ a) = max(a, b)
```

`b ∸ a` 在 `b > a` 时是 `b − a`，否则是 0。所以 `a + (b ∸ a)` 在 `b > a` 时是 `b`，否则是 `a`，正好是较大值，而且永远不会超过 99，不会被饱和截断。

第一版（11 条）是先比较再分支：

```text
DUP; DUP; ROT; SUB; ROT; ROT; JZ 9; DROP; JMP 11; ROT; DROP
```

先摆出 `a b b a`，算出 `a ∸ b` 并转到队首，`JZ` 判断它是不是 0，再分两路丢掉较小的那个。看到截断减法本身就带着"比较"之后，分支整个可以去掉。穷举搜索表明 2 条以内无解，3 条的解只有这一个。

### Part 2 —— 6 条：不分支，按取用顺序摆数

```text
DUP       # a b      ->  b a a
DUP       # b a a    ->  a a b b
ROT       # a a b b  ->  a b b a
SUB       # a b b a  ->  b a (a ∸ b)
SUB       # b a (a ∸ b) -> (a ∸ b) (b ∸ a)
ADD       # -> (a ∸ b) + (b ∸ a) = |a − b|
```

两个想法：

- **不分支**：`a ∸ b` 和 `b ∸ a` 至少有一个是 0，另一个就是 `|a − b|`，两个加起来就是答案。先比较再分支的写法既要留着 `a`、`b` 的副本，又要分两条路，很难写得短。
- **按取用顺序摆数**：队列是先进先出，算出来的结果排到队尾，等前面的数用完才轮到它。所以写程序时要先想好每条指令会按什么顺序取数，把操作数一次摆成 `a b b a`，后面的 `SUB; SUB; ADD` 就能一条接一条，不需要再搬。这和按层遍历一棵表达式树（level order）是一回事：同一层的运算先做，结果按顺序排在后面，供上一层使用。

第一版（7 条）多复制了一份，最后再扔掉：

```text
DUP; DUP; DUP; SUB; SUB; DROP; ADD
```

穷举搜索表明 5 条以内无解；6 条的解只有这一个（离线搜索列出了所有 6 条的解）。

### Part 3 —— 6 条：个数加总和不变

```text
ZERO      # phase 1: append a 0 ...
JZ 3      # ... and pop the front: a 0 at the front means every input is used up
JMP 0
INC       # phase 2: turn one 0 into a 1 at the back
ADD       # merge the two front values
JMP 3     # until ADD starves
```

队列机最难的就是"数东西"：没有寄存器放计数器，计数器放进队列里又会和输入混在一起。参考解分两段：

1. **把每个输入换成一个 0。** 每一圈放进一个 0、取出队首。输入都不是 0，所以前 `k` 圈 `JZ` 取出的都是输入，不跳转；`k` 圈之后队列里正好是 `k` 个 0。第 `k + 1` 圈 `JZ` 第一次取到 0，跳到第二段，此时队列里还是 `k` 个 0。**这些 0 既是替身，又是"输入已经用完"的记号**，不需要另放一个结束标记。
2. **把"个数"变成"数值"。** 看"队列长度 + 所有数之和"这个量：`INC` 让和加 1、长度不变；`ADD` 让长度减 1、和不变（这里的数都很小，不会被饱和截断）。所以每一圈开始时都有"长度 + 和 = k"。最后一圈开始时队列只剩 1 个数，按不变量它是 `k − 1`；`INC` 把它变成 `k`，接着 `ADD` 要取 2 个数，饿死，程序停机，输出正好是 `k`。

输入 `8 5 9 4` 的完整运行过程：

```text
            start    8 5 9 4
   0   0  ZERO     8 5 9 4 0
   1   1  JZ 3     5 9 4 0
   2   2  JMP 0    5 9 4 0
   3   0  ZERO     5 9 4 0 0
   4   1  JZ 3     9 4 0 0
   5   2  JMP 0    9 4 0 0
   6   0  ZERO     9 4 0 0 0
   7   1  JZ 3     4 0 0 0
   8   2  JMP 0    4 0 0 0
   9   0  ZERO     4 0 0 0 0
  10   1  JZ 3     0 0 0 0
  11   2  JMP 0    0 0 0 0
  12   0  ZERO     0 0 0 0 0
  13   1  JZ 3     0 0 0 0
  14   3  INC      0 0 0 1
  15   4  ADD      0 1 0
  16   5  JMP 3    0 1 0
  17   3  INC      1 0 1
  18   4  ADD      1 1
  19   5  JMP 3    1 1
  20   3  INC      1 2
  21   4  ADD      3
  22   5  JMP 3    3
  23   3  INC      4
  24   4  ADD      needs 2, queue has 1: halt
```

第一版（7 条）先单独放一个 0 当结束标记：

```text
ZERO; JZ 4; ZERO; JMP 1; INC; ADD; JMP 4
```

参考解发现替身 0 本身就能当结束标记，省掉一条。这道题真正难的是第二段：想到"长度 + 和"这个不变量之前，往往连一个能过的程序都写不出来。穷举搜索表明 5 条以内无解；6 条的解有 3 个，都是这个思路（离线搜索列出了全部）。

队列里最多同时有 `k + 1` 个数（第一段最后一圈放进 0 的时候），所以输入最多 15 个，正好不超过 16。

### Part 4 —— 8 条：一次减法分三类，开头兼作"输出 1"

```text
INC       # a b          ->  b (a+1)
ROT       # b (a+1)      ->  (a+1) b
SUB       #              ->  d = (a+1) ∸ b:  0 if a < b,  1 if a = b,  >= 2 if a > b
DUP       # d            ->  d d
JZ 1      # pops d. d = 0: jump to ROT with queue "0", ROT, then SUB starves: output 0
DEC       # d - 1
ZERO      # (d-1) 0
JZ 0      # pops d-1. a = b: jump to INC with queue "0": INC gives 1, ROT, SUB starves: output 1
          # a > b: fall off the end with queue "0": output 0
```

三个想法叠在一起：

- **`(a + 1) ∸ b` 一次分三类**：小于时是 0，等于时恰好是 1，大于时至少是 2。一次减法就把三种情况分开了，不用先算 `|a − b|`。
- **`ZERO` 放在 `JZ` 前面**：`JZ` 要取掉被测试的数，`ZERO` 事先在队尾留下一个 0，测试完队列里正好是答案需要的那个 0。
- **复用程序开头，用饿死当出口**：需要输出 1 时，跳回第 0 条，`INC` 把剩下的 0 变成 1，接着 `ROT` 原样转一下，`SUB` 要取 2 个数而队列只剩 1 个，饿死停机。输出 0 的那条路也一样，跳到第 1 条，在 `SUB` 处饿死。

`a = b = 7` 的运行过程：

```text
            start    7 7
   0   0  INC      7 8
   1   1  ROT      8 7
   2   2  SUB      1
   3   3  DUP      1 1
   4   4  JZ 1     1
   5   5  DEC      0
   6   6  ZERO     0 0
   7   7  JZ 0     0
   8   0  INC      1
   9   1  ROT      1
  10   2  SUB      needs 2, queue has 1: halt
```

这里用到了"输入不会是 99"：如果 `a = b = 99`，`INC` 被饱和截断，`d` 算出来是 0，程序会误判成小于（见追问）。

第一版（11 条）先用 Part 2 的 6 条算出 `|a − b|`，再分支：

```text
DUP; DUP; ROT; SUB; SUB; ADD; JZ 9; ZERO; JMP 11; ZERO; INC
```

页面里的穷举只跑到 5 条（无解）。6 条和 7 条是离线跑的（同一算法的 C 版，节点数和 Python 版逐个对上）：都无解，所以 8 条是最优。8 条的解一共 6 个，都是 `INC; ROT; SUB; DUP` 开头，区别只在 `DEC` 的位置和第一个 `JZ` 的目标（跳到 1、2 或 8 都能饿死或停机）。

### 下界：把短程序全部搜一遍

和第一套一样，搜索边运行边生成程序：某个位置的指令第一次被执行到时才去枚举；一旦停机时队列不对、或者队列超过 16 个数，就放弃这个分支。队列最多 16 个数、每个数只有 100 种取值，所以状态是有限的。停不下来的程序一定会重复某个状态 `(位置, 队列)`，用 Brent 的找环算法（cycle detection）精确判断。

为了快，搜索只用每道题的一小部分测试。这不影响下界：在全部测试上都对的程序，在其中一部分上当然也对。找到的解再拿全部测试复查。

```python
OPS_B = ("ADD", "SUB", "DUP", "ROT", "DROP", "INC", "DEC", "ZERO", "JMP", "JZ")
POPS = {"ADD": 2, "SUB": 2, "DUP": 1, "ROT": 1, "DROP": 1, "INC": 1, "DEC": 1, "ZERO": 0, "JMP": 0, "JZ": 1}


def search(tests, n, cap=16, limit=99):
    """Every program of exactly n instructions that halts with the expected queue on every test.

    Same idea as for machine A: a slot gets its instruction the first time execution reaches it, so
    branches of the program a test never visits are never enumerated. Non-termination is detected
    exactly (a repeated (pc, queue) state), so the result does not depend on any step limit.
    Returns (programs, nodes); slots no test ever reached are None."""
    code = [None] * n
    choices = [(op, None) for op in OPS_B if op not in ("JMP", "JZ")]
    choices += [(op, k) for op in ("JMP", "JZ") for k in range(n + 1)]
    found, nodes = [], [0]

    def start(t):
        if t == len(tests):
            found.append(list(code))
        else:
            go(t, 0, tuple(tests[t][0]), None, 0, 1)

    def go(t, pc, q, mark, steps, power):
        want = tests[t][1]
        while True:
            if pc == n:
                if list(q) == want:
                    start(t + 1)
                return
            if code[pc] is None:
                for choice in choices:
                    code[pc] = choice
                    nodes[0] += 1
                    go(t, pc, q, mark, steps, power)
                code[pc] = None
                return
            op, arg = code[pc]
            if len(q) < POPS[op]:                 # starved: halt
                if list(q) == want:
                    start(t + 1)
                return
            pc += 1
            if op == "ADD":
                q = q[2:] + (min(q[0] + q[1], limit),)
            elif op == "SUB":
                q = q[2:] + (max(q[0] - q[1], 0),)
            elif op == "DUP":
                q = q[1:] + (q[0], q[0])
            elif op == "ROT":
                q = q[1:] + q[:1]
            elif op == "DROP":
                q = q[1:]
            elif op == "INC":
                q = q[1:] + (min(q[0] + 1, limit),)
            elif op == "DEC":
                q = q[1:] + (max(q[0] - 1, 0),)
            elif op == "ZERO":
                q = q + (0,)
            elif op == "JMP":
                pc = arg
            else:                                 # JZ
                if q[0] == 0:
                    pc = arg
                q = q[1:]
            if len(q) > cap:                      # overflow: this partial program is invalid
                return
            state = (pc, q)
            if state == mark:                     # repeated state: runs forever
                return
            steps += 1
            if steps == power:
                mark, steps, power = state, 0, power * 2

    start(0)
    return found, nodes[0]


def as_source(program):
    return "; ".join((op or "ROT") + ("" if arg is None else f" {arg}") for op, arg in (c or (None, None) for c in program))
```

页面里复跑的部分（验证代码核对节点数）：

| 题 | n = 1 | n = 2 | n = 3 | n = 4 | n = 5 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| Part 1 | 12 | 168 | 2,032 | — | — | 3 条找到 1 个解 |
| Part 2 | 12 | 168 | 2,032 | 24,300 | 300,460 | 5 条以内无解 |
| Part 3 | 12 | 182 | 1,856 | 20,376 | 238,680 | 5 条以内无解 |
| Part 4 | 12 | 154 | 1,792 | 20,754 | 249,280 | 5 条以内无解 |

离线（C 版，同样的测试子集）的结果：Part 2 的 6 条访问 3,934,348 个节点，只有 1 个解；Part 3 的 6 条访问 3,032,766 个节点，有 3 个解；Part 4 的 6 条（3,188,570 个节点）和 7 条（43,806,960 个节点）无解，8 条（649,658,178 个节点）有 6 个解。所有找到的解都在全部测试上复查过，全部有效。

### 时间怎么分

- **0–10 分钟**：读题，写跟踪器，把求和的例子跑一遍，确认"从队首取、往队尾放"和"饿死就停机"的理解没错。
- **10–40 分钟**：Part 1、Part 2 是热身。截断减法的那条恒等式一旦想到，两题几分钟就能到最优，而且 Part 2 的结果直接给出 Part 4 的第一版。
- **40–80 分钟**：Part 3。先在纸上找"长度 + 和"这类不变量，比盲目试指令有效得多。
- **剩下的时间**：Part 4 从 11 条往下压。先想清楚每条路径停机时队列里应该剩什么，再看哪些指令可以复用、哪里可以让程序饿死。
- **搜索值不值得写**：Part 1–3 的最优解都在 6 条以内，Python 版搜索能直接证明，也能把解找出来。Part 4 的 8 条要访问六亿多个节点，纯 Python 不现实。

### 追问

- **如果 Part 4 的输入可以是 99？** 参考解在 `a = b = 99` 时出错：`INC` 得到的还是 99，`d = 0`，被当成小于，输出 0。我们找到的最好的、对 0..99 都正确的程序是 10 条：`DUP; DUP; ROT; SUB; SUB; ADD; ZERO; JZ 9; ADD; INC`。它先算 `|a − b|`；不相等时 `ADD` 只有 1 个数可取，饿死，留下的正好是 0。离线穷举表明这个版本 8 条以内无解；9 条有没有解，我们没有搜完，所以只能说"目前找到的最好是 10 条"。
- **较小值 `min(a, b)` 呢？** `min(a, b) = a ∸ (a ∸ b)`，5 条：`DUP; ROT; ROT; SUB; SUB`。穷举表明 4 条以内无解。比较大值多两条，是因为 `a` 要用两次，得先把它摆到合适的位置。
- **如果 Part 3 的输入里可以有 0？** 参考解的第一段靠"队首第一次出现 0"判断输入用完了，输入里有 0 就会提前结束。离线穷举表明这个版本 8 条以内无解，我们没有继续往下找。
- **搜索为什么能精确判断"停不下来"？** 状态是 `(位置, 队列)`，队列最多 16 个数、每个数 0..99，状态个数有限，所以一个停不下来的运行一定会重复某个状态；Brent 的算法在重复出现后的有限步内一定能发现。因此下界不依赖 20,000 步这个上限。
- **只在部分测试上搜，得到的下界算数吗？** 算。在全部测试上都正确的程序，在其中一部分上一定也正确；部分测试上 `n` 条无解，全部测试上更不可能有解。反过来，在部分测试上找到的解不一定对，所以每个解都要拿全部测试复查。

<details>
<summary>验证代码（可运行）</summary>

```python
# ---- verbatim copy of the checker given in the statement ----
"""Validity checker for the queue machine. This file is all the tooling you get."""
import random
from collections import deque

LIMIT = 99            # values are integers 0..LIMIT
CAP = 16              # the queue holds at most CAP values
MAX_LEN = 12          # a program has 1..MAX_LEN instructions
MAX_STEPS = 20_000    # per test; a program that runs longer is invalid
NEED = {"ADD": 2, "SUB": 2, "DUP": 1, "ROT": 1, "DROP": 1,     # values an instruction pops
        "INC": 1, "DEC": 1, "ZERO": 0, "JMP": 0, "JZ": 1}


def parse(source):
    """Instructions are separated by newlines or ';'. '#' starts a comment. Returns [(op, arg)]."""
    prog = []
    for line in source.replace(";", "\n").splitlines():
        parts = line.split("#")[0].split()
        if not parts:
            continue
        op = parts[0]
        if op not in NEED or len(parts) != (2 if op in ("JMP", "JZ") else 1):
            raise ValueError(f"bad instruction: {line.strip()!r}")
        prog.append((op, int(parts[1]) if len(parts) == 2 else None))
    if not 1 <= len(prog) <= MAX_LEN:
        raise ValueError(f"a program has 1..{MAX_LEN} instructions")
    if any(arg is not None and not 0 <= arg <= len(prog) for _, arg in prog):
        raise ValueError("jump target out of range")      # target len(prog) means "halt"
    return prog


def run(prog, values):
    """Final queue (front first), or None on overflow or after MAX_STEPS steps."""
    q = deque(values)
    pc = 0
    for _ in range(MAX_STEPS):
        if pc == len(prog):
            return list(q)                    # ran off the end (or jumped to len(prog))
        op, arg = prog[pc]
        if len(q) < NEED[op]:
            return list(q)                    # starved: halt without executing it
        pc += 1
        if op == "ADD":
            a = q.popleft(); b = q.popleft(); q.append(min(a + b, LIMIT))
        elif op == "SUB":
            a = q.popleft(); b = q.popleft(); q.append(max(a - b, 0))
        elif op == "DUP":
            a = q.popleft(); q.append(a); q.append(a)
        elif op == "ROT":
            q.append(q.popleft())
        elif op == "DROP":
            q.popleft()
        elif op == "INC":
            q.append(min(q.popleft() + 1, LIMIT))
        elif op == "DEC":
            q.append(max(q.popleft() - 1, 0))
        elif op == "ZERO":
            q.append(0)
        elif op == "JMP":
            pc = arg
        elif op == "JZ":
            if q.popleft() == 0:
                pc = arg
        if len(q) > CAP:
            return None                       # overflow
    return None


# ---- the four puzzles: a reference function (the spec) and the test inputs ----

PAIRS = [[a, b] for a in range(LIMIT + 1) for b in range(LIMIT + 1)]
PAIRS_98 = [[a, b] for a in range(LIMIT) for b in range(LIMIT)]       # Part 4: both inputs <= 98
_rng = random.Random(2026)
LISTS = [list(range(1, k + 1)) for k in range(1, CAP)] + [[LIMIT] * (CAP - 1)] + [
    [_rng.randint(1, LIMIT) for _ in range(_rng.randint(1, CAP - 1))] for _ in range(300)]

PUZZLES = {
    "q1": (lambda q: [max(q)], PAIRS),
    "q2": (lambda q: [abs(q[0] - q[1])], PAIRS),
    "q3": (lambda q: [len(q)], LISTS),
    "q4": (lambda q: [int(q[0] == q[1])], PAIRS_98),
}


def check(name, source):
    """(valid, instruction count). Valid: on every test the program halts with exactly the right queue."""
    prog = parse(source)
    spec, tests = PUZZLES[name]
    return all(run(prog, t) == spec(t) for t in tests), len(prog)


def report(submission):
    """submission: {puzzle name: program text}. Prints one line per puzzle, returns the total score."""
    total = 0
    for name in PUZZLES:
        valid, count = check(name, submission[name]) if name in submission else (False, 0)
        total += count if valid else 20
        print(f"{name}: {'valid' if valid else 'INVALID':7s} {count:2d} instructions")
    print(f"total score: {total}")
    return total


# ---- checks ----
import itertools
import random
import time

T0 = time.time()
rng = random.Random(11)

# ---- 1. the machine example in the statement ----
assert run(parse("ADD; JMP 0"), [3, 4, 5]) == [12]
got, lines = trace_q("ADD; JMP 0", [3, 4, 5], show=False)
assert got == [12] and len(lines) == 6 and "halt" in lines[-1]

# ---- 2. puzzle examples, against the spec and against independent rewrites ----
EXAMPLES = {'q1': [([3, 8], [8]), ([5, 5], [5]), ([99, 0], [99])], 'q2': [([3, 8], [5]), ([8, 3], [5]), ([0, 99], [99])], 'q3': [([4, 4, 9], [3]), ([99], [1]), ([7, 1, 2, 3, 99, 5], [6])], 'q4': [([7, 7], [1]), ([7, 2], [0]), ([0, 98], [0])]}
INDEPENDENT = {"q1": lambda q: [q[0] if q[0] >= q[1] else q[1]],
               "q2": lambda q: [max(q) - min(q)],
               "q3": lambda q: [sum(1 for _ in q)],
               "q4": lambda q: [1 if q[0] - q[1] == 0 else 0]}
for name, pairs in EXAMPLES.items():
    for values, want in pairs:
        assert PUZZLES[name][0](values) == want == INDEPENDENT[name](values), (name, values)
for name, (spec, tests) in PUZZLES.items():
    assert all(spec(t) == INDEPENDENT[name](t) for t in tests)
# the tests cover the promised domains: all pairs, every list length 1..15, no 0 and no 99 where promised
assert sorted(map(tuple, PUZZLES["q1"][1])) == [(a, b) for a in range(100) for b in range(100)]
assert sorted(map(tuple, PUZZLES["q4"][1])) == [(a, b) for a in range(99) for b in range(99)]
assert {len(t) for t in LISTS} == set(range(1, 16)) and all(1 <= x <= 99 for t in LISTS for x in t)

# ---- 3. reference solutions and first versions ----
REF = {"q1": "DUP; SUB; ADD",
       "q2": "DUP; DUP; ROT; SUB; SUB; ADD",
       "q3": "ZERO; JZ 3; JMP 0; INC; ADD; JMP 3",
       "q4": "INC; ROT; SUB; DUP; JZ 1; DEC; ZERO; JZ 0"}
FIRST = {"q1": "DUP; DUP; ROT; SUB; ROT; ROT; JZ 9; DROP; JMP 11; ROT; DROP",
         "q2": "DUP; DUP; DUP; SUB; SUB; DROP; ADD",
         "q3": "ZERO; JZ 4; ZERO; JMP 1; INC; ADD; JMP 4",
         "q4": "DUP; DUP; ROT; SUB; SUB; ADD; JZ 9; ZERO; JMP 11; ZERO; INC"}
for name, n in zip(PUZZLES, (3, 6, 6, 8)):
    assert check(name, REF[name]) == (True, n), name
for name, n in zip(PUZZLES, (11, 7, 7, 11)):
    assert check(name, FIRST[name]) == (True, n), name
assert report(REF) == 23 and report(FIRST) == 36 and report({}) == 80
assert all(run(parse(REF[name]), values) == want for name, pairs in EXAMPLES.items() for values, want in pairs)
assert 11 + 8 + 8 + 11 == 38                              # pass marks in the statement
# q1, q2 and q4 tests are the whole input domain, so "valid" already means "correct everywhere".
# q3 only looks at whether a value is 0, so its result depends only on the length; check every
# length with many random lists anyway, plus every list over {1, 2, 99} up to length 6.
for k in range(1, 16):
    for _ in range(40):
        values = [rng.randint(1, 99) for _ in range(k)]
        assert run(parse(REF["q3"]), values) == [k] == run(parse(FIRST["q3"]), values)
for k in range(1, 7):
    assert all(run(parse(REF["q3"]), list(t)) == [k] for t in itertools.product((1, 2, 99), repeat=k))

# ---- 4. the tracer agrees with the starter, and queue sizes stay within CAP ----


def random_program(n):
    words = []
    for _ in range(n):
        op = rng.choice(list(NEED))
        words.append(f"{op} {rng.randint(0, n)}" if op in ("JMP", "JZ") else op)
    return "; ".join(words)


saved, MAX_STEPS = MAX_STEPS, 400                         # run() reads the global MAX_STEPS at call time
for _ in range(500):
    src = random_program(rng.randint(1, 7))
    for _ in range(3):
        values = [rng.choice([0, 1, 2, 98, 99, rng.randint(0, 99)]) for _ in range(rng.randint(0, 6))]
        assert run(parse(src), values) == trace_q(src, values, limit=MAX_STEPS, show=False)[0], (src, values)
MAX_STEPS = saved
for name in PUZZLES:
    for src in (REF[name], FIRST[name]):
        for values in PUZZLES[name][1][::7]:
            assert trace_q(src, values, limit=MAX_STEPS, show=False)[0] == run(parse(src), values)


saved_cap, CAP = CAP, 15                                 # run() reads the global CAP too
assert run(parse(REF["q3"]), [5] * 15) is None and run(parse(REF["q3"]), [5] * 14) == [14]
CAP = saved_cap
assert run(parse(REF["q3"]), [5] * 15) == [15]            # so the longest queue is k + 1 = 16

# first_failure (from the tracer block) finds and traces the first failing test, or says all pass
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()) as buf:
    assert first_failure("q4", REF["q4"]) is None
    assert first_failure("q1", "DUP; SUB") == [0, 0]
printed = buf.getvalue().splitlines()
assert printed[0] == "all tests pass" and printed[1] == "input [0, 0]: want [0], got [0, 0]"

# ---- 5. traces quoted in the solution ----
assert trace_q(REF["q3"], [8, 5, 9, 4], show=False) == ([4], ['            start    8 5 9 4', '   0   0  ZERO     8 5 9 4 0', '   1   1  JZ 3     5 9 4 0', '   2   2  JMP 0    5 9 4 0', '   3   0  ZERO     5 9 4 0 0', '   4   1  JZ 3     9 4 0 0', '   5   2  JMP 0    9 4 0 0', '   6   0  ZERO     9 4 0 0 0', '   7   1  JZ 3     4 0 0 0', '   8   2  JMP 0    4 0 0 0', '   9   0  ZERO     4 0 0 0 0', '  10   1  JZ 3     0 0 0 0', '  11   2  JMP 0    0 0 0 0', '  12   0  ZERO     0 0 0 0 0', '  13   1  JZ 3     0 0 0 0', '  14   3  INC      0 0 0 1', '  15   4  ADD      0 1 0', '  16   5  JMP 3    0 1 0', '  17   3  INC      1 0 1', '  18   4  ADD      1 1', '  19   5  JMP 3    1 1', '  20   3  INC      1 2', '  21   4  ADD      3', '  22   5  JMP 3    3', '  23   3  INC      4', '  24   4  ADD      needs 2, queue has 1: halt'])
assert trace_q(REF["q4"], [7, 7], show=False) == ([1], ['            start    7 7', '   0   0  INC      7 8', '   1   1  ROT      8 7', '   2   2  SUB      1', '   3   3  DUP      1 1', '   4   4  JZ 1     1', '   5   5  DEC      0', '   6   6  ZERO     0 0', '   7   7  JZ 0     0', '   8   0  INC      1', '   9   1  ROT      1', '  10   2  SUB      needs 2, queue has 1: halt'])

# ---- 6. lower bounds by exhaustive search on small test subsets ----
SUBSETS = {
    "q1": [[a, b] for a in range(6) for b in range(6)] + [[99, 0], [0, 99], [99, 99]],
    "q2": [[a, b] for a in range(6) for b in range(6)] + [[99, 0], [0, 99], [99, 99], [50, 51]],
    "q3": [[1], [2, 3], [1, 1, 1], [5, 6, 7, 8], [9] * 7, list(range(1, 12)), [99] * 15, [1] * 15],
    "q4": [[a, b] for a in range(21) for b in range(21)] + [[98, 98], [0, 98], [98, 0], [97, 98], [98, 97]],
}
TOP = {"q1": 3, "q2": 5, "q3": 5, "q4": 5}
NODES = {'q1': [12, 168, 2032], 'q2': [12, 168, 2032, 24300, 300460], 'q3': [12, 182, 1856, 20376, 238680], 'q4': [12, 154, 1792, 20754, 249280]}
t = time.time()
found = {}
for name in PUZZLES:
    spec = PUZZLES[name][0]
    cases = [(values, spec(values)) for values in SUBSETS[name]]
    for n in range(1, TOP[name] + 1):
        programs, nodes = search(cases, n)
        assert nodes == NODES[name][n - 1], (name, n, nodes)
        if programs:
            found[name] = (n, [as_source(p) for p in programs])
            break
assert found == {"q1": (3, [REF["q1"]])}                  # q1: 3 is optimal and unique; q2-q4: nothing up to 5
print(f"search: {sum(sum(v) for v in NODES.values()):,} nodes in {time.time() - t:.1f}s")

# ---- 7. follow-ups ----
SAFE_EQ = "DUP; DUP; ROT; SUB; SUB; ADD; ZERO; JZ 9; ADD; INC"
everything = [[a, b] for a in range(100) for b in range(100)]
assert all(run(parse(SAFE_EQ), q) == [int(q[0] == q[1])] for q in everything) and len(parse(SAFE_EQ)) == 10
assert run(parse(REF["q4"]), [99, 99]) == [0]             # the saturation trap
MIN2 = "DUP; ROT; ROT; SUB; SUB"
assert all(run(parse(MIN2), q) == [min(q)] for q in everything)
min_cases = [(q, [min(q)]) for q in SUBSETS["q1"]]
assert all(not search(min_cases, n)[0] for n in range(1, 5)) and len(search(min_cases, 5)[0]) == 3
assert run(parse(REF["q3"]), [4, 0, 4]) != [3]            # Part 3 with zeros breaks the reference
print(f"all checks passed in {time.time() - T0:.1f}s")
```

</details>

</details>
