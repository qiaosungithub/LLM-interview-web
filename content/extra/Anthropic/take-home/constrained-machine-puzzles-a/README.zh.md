# Take-home：受限指令集谜题（一）带条件前缀的累加器机

## 题目

> **这不是原题。** Anthropic 现在的性能 take-home 是一组在很小、限制很多的指令集上做的谜题，目标是用最少的指令解题。真实的机器和谜题没有公开，这台机器和下面 5 道题都是我们自己设计的，只保留了题型的样子：指令少、寄存器少、状态要藏在标志位和程序位置里、不给调试工具。

你会拿到一台小机器的说明、一份检查器（checker），和 5 道互相独立的小题。每道题交一段这台机器的程序。检查器只告诉你两件事：程序是否有效，一共几条指令。目标是让 5 道题的指令总数尽量少。

### 机器

- 两个整数寄存器：`acc`（累加器，accumulator）和 `bak`（备用寄存器）。开始时都是 0。所有结果都饱和（saturate）到 −99..99：算出来超过 99 就记 99，低于 −99 就记 −99。
- 一个标志位 `F`（flag），开始时是 0。只有 `TZ` 和 `TP` 会改它。
- 程序有 1–12 条指令，编号从 0 开始，首尾相接成一个环（ring）：执行完最后一条，下一条是第 0 条。程序没有"结束"这一说。
- 每条指令前面可以加一个条件前缀（guard）：`+` 表示只在 F=1 时执行，`-` 表示只在 F=0 时执行，不加表示总是执行。条件不满足时这条指令被跳过：照样算一步，但什么都不做。被跳过的 `IN` 不读输入，也不会停机。
- 输入是一串整数，输出是一串整数。
- **唯一的停机方式**：执行 `IN` 时输入已经读完。停机时输出就是到此为止 `OUT` 过的数。
- 每个测试最多跑 20,000 步（被跳过的指令也算一步）。超过就算无效。

| 指令 | 效果 |
| --- | --- |
| `IN` | 读下一个输入放进 `acc`；输入读完了就停机 |
| `OUT` | 把 `acc` 追加到输出 |
| `INC` | `acc ← acc + 1`（饱和） |
| `NEG` | `acc ← −acc` |
| `ADD` | `acc ← acc + bak`（饱和） |
| `SAV` | `bak ← acc` |
| `SWP` | 交换 `acc` 和 `bak` |
| `TZ` | `F ← 1` 如果 `acc = 0`，否则 `F ← 0` |
| `TP` | `F ← 1` 如果 `acc > 0`，否则 `F ← 0` |
| `JMP k` | 下一步执行第 `k` 条（`0 ≤ k <` 程序长度） |

注意没有"减 1"，也没有比较两个寄存器的指令。指令之间用换行或 `;` 分隔，`#` 后面是注释。

例子：输出所有正数的程序 `IN; TP; +OUT`，输入 `3 -1 5`。

```text
步  位置  指令    执行后
 0   0   IN      acc=3
 1   1   TP      F=1
 2   2   +OUT    输出 3
 3   0   IN      acc=-1      （第 2 条之后回到第 0 条）
 4   1   TP      F=0
 5   2   +OUT    跳过（F=0）
 6   0   IN      acc=5
 7   1   TP      F=1
 8   2   +OUT    输出 5
 9   0   IN      输入读完，停机
输出：3 5，共 10 步
```

### 检查器

这就是你拿到的全部工具：

```py
"""Validity checker for the guarded accumulator machine. This file is all the tooling you get."""

LIMIT = 99            # acc and bak saturate to [-LIMIT, LIMIT]
MAX_LEN = 12          # a program has 1..MAX_LEN instructions
MAX_STEPS = 20_000    # per test; a program that runs longer is invalid
OPS = ("IN", "OUT", "INC", "NEG", "ADD", "SAV", "SWP", "TZ", "TP", "JMP")


def parse(source):
    """Instructions are separated by newlines or ';'. '#' starts a comment.
    A '+' or '-' in front of the mnemonic is the guard. Returns [(guard, op, arg)]."""
    prog = []
    for line in source.replace(";", "\n").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        guard = ""
        if line[0] in "+-":
            guard, line = line[0], line[1:].strip()
        parts = line.split()
        if not parts or parts[0] not in OPS or len(parts) != (2 if parts[0] == "JMP" else 1):
            raise ValueError(f"bad instruction: {line!r}")
        prog.append((guard, parts[0], int(parts[1]) if parts[0] == "JMP" else None))
    if not 1 <= len(prog) <= MAX_LEN:
        raise ValueError(f"a program has 1..{MAX_LEN} instructions")
    if any(op == "JMP" and not 0 <= arg < len(prog) for _, op, arg in prog):
        raise ValueError("jump target out of range")
    return prog


def run(prog, inputs):
    """Outputs of one run, or None if the program is still running after MAX_STEPS steps."""
    acc = bak = 0
    flag = False
    pc = pos = 0
    out = []
    for _ in range(MAX_STEPS):
        guard, op, arg = prog[pc]
        pc = (pc + 1) % len(prog)                 # the program is a ring
        if (guard == "+" and not flag) or (guard == "-" and flag):
            continue                              # skipped: one step, no effect
        if op == "IN":
            if pos == len(inputs):
                return out                        # the only way to halt
            acc = inputs[pos]
            pos += 1
        elif op == "OUT":
            out.append(acc)
        elif op == "INC":
            acc = min(acc + 1, LIMIT)
        elif op == "NEG":
            acc = -acc
        elif op == "ADD":
            acc = max(-LIMIT, min(LIMIT, acc + bak))
        elif op == "SAV":
            bak = acc
        elif op == "SWP":
            acc, bak = bak, acc
        elif op == "TZ":
            flag = acc == 0
        elif op == "TP":
            flag = acc > 0
        elif op == "JMP":
            pc = arg
    return None


# ---- the five puzzles: a reference function (the spec) and the test inputs ----

def after_zero(xs):
    return xs[xs.index(0) + 1:] if 0 in xs else []


def group_sums(xs):
    out, total = [], 0
    for x in xs:
        if x == 0:
            out.append(total)
            total = 0
        else:
            total += x
    return out


def mute(xs):
    out, muted = [], False
    for x in xs:
        if x == 0:
            muted = not muted
        elif not muted:
            out.append(x)
    return out


def count_down(xs):
    return [k for x in xs for k in range(x, 0, -1)]


def count_up(xs):
    return [k for x in xs for k in range(1, x + 1)]


PUZZLES = {
    "p1": (after_zero, [
        [5, 0, 3, 0, -2, 7], [0, 4], [1, 2, 3], [], [0], [0, 0],
        [-6, 9, 0, 0, 0, 5, -99, 99, 0, 1], [7, -7, 0, 12, 0, 0, -3]]),
    "p2": (group_sums, [
        [3, 4, 0, 5, 0], [0], [], [2, 2, 2], [0, 0, 7, 0],
        [-5, 9, -1, 0, 20, 20, 20, 0, 6], [10, -10, 0, 1, 0, 0, -8, -9, 0],
        [4, 0, -4, 0, 4, -4, 0, 15, 3, 0, 0, 1]]),
    "p3": (mute, [
        [3, 0, 5, 0, 7], [], [0], [0, 0], [1, 2, 3],
        [0, 4, 4, 0, 0, 9, 0, 6, 1], [8, 0, 0, 0, 2, 0, 99, 5, 0, 0, 3],
        [5, 5, 0, 1, 0, 0, 0, 2]]),
    "p4": (count_down, [
        [3], [], [0], [1], [2, 0, 4], [0, 0, 1, 5, 0], [20], [7, 1, 0, 2, 2]]),
    "p5": (count_up, [
        [3], [], [0], [1], [2, 0, 4], [0, 0, 1, 5, 0], [20], [7, 1, 0, 2, 2]]),
}


def check(name, source):
    """(valid, instruction count). Valid: on every test the program halts with exactly the right output."""
    prog = parse(source)
    spec, tests = PUZZLES[name]
    return all(run(prog, xs) == spec(xs) for xs in tests), len(prog)


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

- 每道题交一段程序。**有效**指：在这道题的每个测试上都会停机，而且输出和要求完全一样。
- **分数 = 5 道题的指令条数之和**，越少越好。没交或无效的题记 20 分。
- 检查器不会告诉你哪个测试错了、错在哪一步。没有调试器，也不能单步执行。需要工具就自己做：加 print，或者让 AI 编程助手几分钟写一个单步跟踪器。花多少时间做工具、做什么工具，本身就是考察的一部分。
- 时间 2 小时，可以使用 AI 助手。
- 判分只看给定的测试，但请按题目写明的输入范围来解题，不要专门凑测试。

### Part 1 —— 零之后

输入是一串整数，每个在 −99..99。输出第一个 0 **之后**的所有数，保持原顺序。第一个 0 本身不输出，后面再出现的 0 照常输出。没有 0 就什么都不输出。

```text
5 0 3 0 -2 7  ->  3 0 -2 7
1 2 3         ->  （空）
0 0           ->  0
-4 0          ->  （空）
```

及格线 6 条，目标 3 条。

### Part 2 —— 分组求和

输入是一串整数，0 是分隔符。每读到一个 0，就输出从上一个 0（或开头）到这个 0 之间那些数的和；中间没有数就输出 0。最后一个 0 之后剩下的数不输出。保证每一组从头加到任何位置的和（部分和，running sum）都在 −99..99 内。

```text
3 4 0 5 0  ->  7 5
0 0 7 0    ->  0 0 7
2 2 2      ->  （空）
-5 9 0 1   ->  4
```

及格线 8 条，目标 5 条。

### Part 3 —— 静音开关

输入是一串整数，每个在 **0..99**（没有负数）。开始时处于"放音"状态。每个 0 切换一次放音/静音，0 本身不输出；放音时其他数原样输出，静音时丢掉。

```text
3 0 5 0 7        ->  3 7
0 4 4 0 0 9 0 6  ->  6
0 0 8            ->  8
2 0              ->  2
```

及格线 7 条，目标 5 条。

### Part 4 —— 倒数

输入最多 8 个数，每个在 0..99。对每个数 `x` 依次输出 `x, x−1, …, 1`；`x = 0` 时什么都不输出。

```text
3 0 2  ->  3 2 1 2 1
1 1    ->  1 1
```

及格线 8 条，目标 6 条。

### Part 5 —— 正数

输入和 Part 4 一样。对每个数 `x` 依次输出 `1, 2, …, x`；`x = 0` 时什么都不输出。

```text
3 0 2  ->  1 2 3 1 2
1 1    ->  1 1
```

及格线 11 条，目标 7 条。

5 道题的及格线合计 40 条，目标合计 26 条。

## 参考解答

<details>
<summary>展开参考解答</summary>

先给结论：

| 题 | 及格线程序 | 参考解 | 关键想法 | 最优性 |
| --- | --- | --- | --- | --- |
| Part 1 零之后 | 6 | **3** | 带条件的测试只触发一次，当锁存器用 | 穷举证明最优，且唯一 |
| Part 2 分组求和 | 8 | **5** | 一条 `SWP` 同时完成存、取、清零 | 穷举证明最优 |
| Part 3 静音开关 | 7 | **5** | 标志位就是开关；`TZ` 和 `TP` 在非负数上互补 | 穷举证明最优 |
| Part 4 倒数 | 8 | **6** | 不用跳转，让程序环本身当内层循环；`NEG; INC; NEG` 就是减 1 | 穷举证明最优 |
| Part 5 正数 | 11 | **7** | 一条 `INC` 轮流给两个寄存器加 1，用正负号区分它们 | 离线穷举证明最优 |

合计 26 条。

### 先花 10 分钟写一个单步跟踪器

检查器只会说"无效"。一个程序无效时，你要知道的是：哪个测试错了，错在第几步，那一步之前寄存器和标志位是什么。所以第一件事是写一个跟踪器（tracer），每一步打印位置、指令、执行后的 `acc`、`bak`、`F`。它和检查器分开写（自己解析、自己执行），这样还能拿它和检查器互相对照，顺便确认自己对规则的理解没错。

```python
def trace(source, inputs, limit=200, show=True):
    """Step-by-step log of machine A, written separately from the starter (own parser, own loop).
    Returns (outputs, halted, lines)."""
    prog = []
    for raw in source.replace(";", "\n").splitlines():
        text = raw.split("#")[0].strip()
        if text:
            guard = text[0] if text[0] in "+-" else ""
            words = text[len(guard):].split()
            prog.append((guard, words[0], int(words[1]) if len(words) > 1 else None))
    clamp = lambda v: max(-99, min(99, v))
    acc = bak = flag = pc = 0
    rest, out, lines, halted = list(inputs), [], [], False
    for step in range(limit):
        guard, op, arg = prog[pc]
        head = f"{step:4d} {pc:3d}  {guard + op + ('' if arg is None else f' {arg}'):7s}"
        pc = (pc + 1) % len(prog)
        if guard and (guard == "+") != (flag == 1):
            lines.append(head + "  skipped")
            continue
        note = ""
        if op == "IN":
            if not rest:
                lines.append(head + "  no input left: halt")
                halted = True
                break
            acc = rest.pop(0)
        elif op == "OUT":
            out.append(acc)
            note = f"  -> out {acc}"
        elif op == "INC":
            acc = clamp(acc + 1)
        elif op == "NEG":
            acc = -acc
        elif op == "ADD":
            acc = clamp(acc + bak)
        elif op == "SAV":
            bak = acc
        elif op == "SWP":
            acc, bak = bak, acc
        elif op == "TZ":
            flag = int(acc == 0)
        elif op == "TP":
            flag = int(acc > 0)
        elif op == "JMP":
            pc = arg
        lines.append(f"{head}  acc={acc:3d} bak={bak:3d} F={flag}{note}")
    if show:
        print(" step  pc  instr    state after the step")
        print("\n".join(lines))
    return out, halted, lines


def first_failure(name, source, limit=200):
    """Trace the first test of a puzzle that the program gets wrong (uses the starter's PUZZLES)."""
    spec, tests = PUZZLES[name]
    for xs in tests:
        out, halted, _ = trace(source, xs, limit=MAX_STEPS, show=False)
        if not halted or out != spec(xs):
            print(f"input {xs}: want {spec(xs)}, got {out}{'' if halted else ' (no halt)'}")
            trace(source, xs, limit=limit)
            return xs
    print("all tests pass")
```

`trace("IN; +OUT; -TZ", [5, 0, 3])` 打印的就是 Part 1 参考解的运行过程：

```text
 step  pc  instr    state after the step
   0   0  IN       acc=  5 bak=  0 F=0
   1   1  +OUT     skipped
   2   2  -TZ      acc=  5 bak=  0 F=0
   3   0  IN       acc=  0 bak=  0 F=0
   4   1  +OUT     skipped
   5   2  -TZ      acc=  0 bak=  0 F=1
   6   0  IN       acc=  3 bak=  0 F=1
   7   1  +OUT     acc=  3 bak=  0 F=1  -> out 3
   8   2  -TZ      skipped
   9   0  IN       no input left: halt
```

这几十行就够用了，不值得再做图形界面。剩下的时间更值得花在两件事上：先给每道题交一个能过的版本，然后再一道一道压指令数。

### Part 1 —— 3 条：只触发一次的测试

```text
IN        # read x
+OUT      # after the first 0: echo
-TZ       # before the first 0: F = (x == 0)
```

开始时 F=0。在读到第一个 0 之前，每一圈是 `IN`、跳过 `+OUT`、`-TZ`。`-TZ` 一旦把 F 置成 1（刚读到的数是 0），从此 `-TZ` 永远被跳过，F 再也不会变回 0。于是之后每一圈都是 `IN`、`OUT`、跳过。带 `-` 前缀的测试就像一个只能拨一次的开关，也就是锁存器（latch）。之后再读到 0 也不影响，正好满足"后面的 0 照常输出"。

及格线程序用两个循环和跳转：

```text
IN; TZ; -JMP 0; IN; OUT; JMP 3
```

前半段找 0，后半段原样输出，状态放在"程序正在哪个循环里"。参考解把这个状态放进 F，两个循环合成一个，跳转全部省掉。穷举搜索表明 2 条以内无解，3 条的解只有这一个。

### Part 2 —— 5 条：一条 SWP 同时存、取、清零

```text
IN        # read x
TZ        # F = (x == 0)
-ADD      # x != 0: acc = x + running sum
SWP       # x != 0: bak = new sum   |  x == 0: acc = sum, bak = 0
+OUT      # x == 0: output the sum
```

不变式：每一圈开始时（`IN` 之前），`bak` 里是当前这一组的部分和。

- 读到 `x ≠ 0`：`ADD` 得到新的和，`SWP` 把它放进 `bak`。`acc` 里换回来的是旧的和，没用，但下一圈的 `IN` 马上会把它覆盖。
- 读到 `x = 0`：`ADD` 被跳过，`SWP` 把和换到 `acc` 准备输出，同时把刚读进来的 0 换进 `bak`。**作为分隔符的那个 0 正好就是下一组的初值**，不用另外清零。

一步步怎么想到的：

1. 及格线程序（8 条）用跳转分两条路：`IN; TZ; +JMP 6; ADD; SAV; JMP 0; SWP; OUT`。这里已经用到"分隔符的 0 当初值"了。
2. 把跳转换成条件前缀，得到 6 条：`IN; TZ; -ADD; -SAV; +SWP; +OUT`。
3. 观察 `x ≠ 0` 那条路上的 `SAV`：它和 `SWP` 对 `bak` 的效果一样（都是 `bak ← acc`），只是 `SWP` 还会弄脏 `acc`，而 `acc` 下一步就被覆盖。所以 `-SAV` 和 `+SWP` 可以合成一条不带条件的 `SWP`，得到 5 条。

穷举搜索表明 4 条以内无解。5 条能过测试的程序一共 4 个，其实是同一个程序：程序是环，从环上不同位置开始写就是不同的文本。环上有 5 个起点，少的那一个是从 `TZ` 开始：第一步 `TZ` 看到的是初始的 acc = 0，于是在读任何输入之前就先输出一个 0（`3 4 0 5 0` 会得到 `0 7 5`）。

### Part 3 —— 5 条：标志位就是开关

```text
-IN       # playing: read x
-TZ       # playing: F = (x == 0), i.e. a 0 switches to muted
+IN       # muted: read y (also right after the switch)
-OUT      # playing: echo x
+TP       # muted: F = (y > 0), i.e. a 0 switches back to playing
```

把状态放进 F：F=0 表示放音，F=1 表示静音。

- 放音时（F=0）执行的是 `-IN; -TZ; -OUT`：读一个数，不是 0 就输出。
- 静音时（F=1）执行的是 `+IN; +TP`：读一个数，丢掉。
- 切换靠的是一个巧合：在**非负数**上，`TZ`（是不是 0）和 `TP`（是不是正数）正好相反。放音时用 `TZ`，读到 0 得到 F=1；静音时用 `TP`，读到 0 得到 F=0。所以**两种状态下，读到 0 都会把 F 翻过来**，读到别的数 F 不变。
- 放音时读到 0，F 立刻变成 1，后面的 `+IN` 马上生效，直接接着读静音状态下的第一个数，中间不浪费指令。

及格线程序（7 条）把状态放在程序位置里，两个循环来回跳：

```text
IN; TZ; -OUT; -JMP 0; IN; TZ; -JMP 4
```

穷举搜索表明 4 条以内无解。5 条能过测试的程序有 8 个，其实只有两种写法：

- 这个程序本身，或者把相邻的 `+IN` 和 `-OUT` 对调（两条前缀相反，又都不改 F，一圈里恰好执行其中一条，先后无所谓），再换个起点，共 5 个。
- 镜像写法 `-TZ; +IN; +TP; +OUT; -IN` 和它的变体，共 3 个：F=1 表示放音，放音时用 `TP` 检测 0；F=0 表示静音，用 `TZ` 检测 0。开头的 `-TZ` 借初始的 acc = 0 把 F 置成 1。

这一题的输入范围特意写了"没有负数"：允许负数时，`TP` 会把负数当成 0，开关就会被负数误触发，参考解不再成立（见追问）。

### Part 4 —— 6 条：让程序环当内层循环

```text
-IN       # only when the previous count is finished
TP        # F = (acc > 0)
+OUT
NEG       # acc - 1 written as -(-acc + 1)
INC
NEG
```

两个想法：

- **没有减 1，就用 `NEG; INC; NEG`**：`−(−x + 1) = x − 1`。
- **不写循环**：程序本身就是一个环。`-IN` 只在 F=0 时读新数，于是只要还在倒数（F=1），每一圈就自动跳过 `-IN`，环本身就是内层循环。数到 0 时 `TP` 得到 F=0，这一圈不输出，`acc` 变成 −1（没关系），下一圈 `-IN` 读入新数把它覆盖。输入的 0 也走同一条路，什么都不输出。

及格线程序（8 条）是常规写法，外层读数、内层用跳转循环：

```text
IN; TP; -JMP 0; OUT; NEG; INC; NEG; JMP 1
```

参考解去掉了两条跳转。穷举搜索表明 5 条以内无解，所以 6 条是最优。

### Part 5 —— 7 条：一条 INC 轮流加两个寄存器

正数比倒数难：要同时记住"数到几了"（计数器）和"还要数几个"（剩余），两样东西都要加 1 或减 1，而机器只有一条 `INC`，还只能作用在 `acc` 上。

```text
TZ        # F = 1 exactly when the remaining count just reached 0
SWP       # every lap: switch which register INC works on
INC
+IN       # remaining is 0: read the next x ...
+NEG      # ... and store it as -x
TP        # positive = counter, negative (or 0) = remaining
+OUT      # the counter just went up: output it
```

关键是每一圈都执行一次 `SWP`，于是同一条 `INC` 这一圈加计数器，下一圈加剩余，轮流进行：

- 剩余存成**负数** `−r`，计数器是正数 `c`。给 `−r` 加 1 就是剩余减 1。
- 每圈 `INC` 之后用 `TP` 看刚加的是哪一个：正数说明刚加的是计数器，就输出；负数或 0 说明刚加的是剩余，不输出。
- 剩余加到 0 时，下一圈的 `TZ` 得到 F=1，`+IN` 读入下一个 `x`，`+NEG` 把它变成 `−x`。此时另一个寄存器里正好是刚用完的剩余 0，**它直接成了新的计数器初值**，不需要清零。
- 开始时 `acc = 0`，第一圈 `TZ` 就得到 F=1，所以读第一个数也走同一条路。

`trace(...)` 输入 `2 0 1` 的前 14 步：

```text
 step  pc  instr    state after the step
   0   0  TZ       acc=  0 bak=  0 F=1
   1   1  SWP      acc=  0 bak=  0 F=1
   2   2  INC      acc=  1 bak=  0 F=1
   3   3  +IN      acc=  2 bak=  0 F=1
   4   4  +NEG     acc= -2 bak=  0 F=1
   5   5  TP       acc= -2 bak=  0 F=0
   6   6  +OUT     skipped
   7   0  TZ       acc= -2 bak=  0 F=0
   8   1  SWP      acc=  0 bak= -2 F=0
   9   2  INC      acc=  1 bak= -2 F=0
  10   3  +IN      skipped
  11   4  +NEG     skipped
  12   5  TP       acc=  1 bak= -2 F=1
  13   6  +OUT     acc=  1 bak= -2 F=1  -> out 1
```

及格线程序（11 条）用一个计数器、一个剩余和三次 `SWP`：

```text
IN; NEG; TZ; +SWP; +JMP 0; INC; SWP; INC; OUT; SWP; JMP 2
```

页面里的穷举只跑到 5 条（无解）。6 条的搜索要访问约 685 万个节点，我们离线用同一算法跑过（C 版和 Python 版结果一致）：6 条无解，所以 7 条是最优。7 条能过测试的程序一共 28 个。

### 下界：把短程序全部搜一遍

每道题的"最优"不是靠感觉，而是靠穷举。直接枚举太多：长度为 `n` 的程序每个位置有 3 种前缀、`9 + n` 种指令（跳转要选目标），5 条的程序就有 `(3 × 14)^5`，约 1.3 亿个。下面的搜索边运行边生成程序：

- 某个位置的指令**第一次被执行到时**才去枚举它；某个位置的前缀在第一次以 F=0 或 F=1 到达时才去决定。没被执行到的位置不用枚举。
- 一旦输出了一个错的数，整个分支立刻放弃。大部分程序在头几步就被剪掉。
- 两次 `IN` 之间，如果状态 `(位置, acc, bak, F)` 重复出现，程序就永远停不下来，也放弃。用 Brent 的找环算法（cycle detection）判断，所以这个判断是精确的，下界和步数上限无关。
- 只在部分测试上搜得到的下界，对全部测试也成立：在全部测试上都对的程序，在其中一部分上当然也对。

```python
def search(tests, n):
    """Every program of exactly n instructions that passes tests = [(inputs, expected outputs), ...].

    The program is built while it runs. A slot gets its opcode the first time it executes, and its
    guard is narrowed the first time the slot is reached with each flag value. A partial program is
    abandoned at its first wrong output, so almost none of the (3 * (9 + n)) ** n full programs is ever completed.
    Returns (programs, nodes); a program is a list of (guard, op, arg), with None where a choice was
    never needed (any completion behaves the same on the tests)."""
    runs = [[None, None] for _ in range(n)]       # runs[i][f]: does slot i execute when the flag is f?
    code = [None] * n                             # (op, arg) once chosen
    choices = [(op, None) for op in ("IN", "OUT", "INC", "NEG", "ADD", "SAV", "SWP", "TZ", "TP")]
    choices += [("JMP", k) for k in range(n)]
    required = {"IN"} | ({"OUT"} if any(want for _, want in tests) else set())
    found, nodes = [], [0]

    def snapshot():
        guards = {(True, True): "", (False, True): "+", (True, False): "-"}
        return [(guards.get(tuple(r)), c and c[0], c and c[1]) for r, c in zip(runs, code)]

    def start(t):
        if t == len(tests):
            found.append(snapshot())
        else:
            go(t, 0, 0, 0, 0, 0, 0, None, 0, 1)

    def go(t, pc, acc, bak, f, taken, given, mark, steps, power):
        inputs, want = tests[t]
        while True:
            r = runs[pc]
            if r[f] is None:                      # first visit with this flag value: branch on the guard
                r[f] = True
                nodes[0] += 1
                go(t, pc, acc, bak, f, taken, given, mark, steps, power)
                if r[1 - f] is not False:         # skipped here, so it must run under the other value
                    other = r[1 - f]
                    r[f], r[1 - f] = False, True
                    nodes[0] += 1
                    go(t, pc, acc, bak, f, taken, given, mark, steps, power)
                    r[1 - f] = other
                r[f] = None
                return
            if r[f] and code[pc] is None:         # first execution: branch on the opcode
                for choice in choices:
                    code[pc] = choice
                    placed = {c[0] for c in code if c}
                    if len(required - placed) <= code.count(None):   # IN (and OUT) must still fit
                        nodes[0] += 1
                        go(t, pc, acc, bak, f, taken, given, mark, steps, power)
                code[pc] = None
                return
            nxt = pc + 1
            if r[f]:
                op, arg = code[pc]
                if op == "IN":
                    if taken == len(inputs):      # halt: right only if every expected output was given
                        if given == len(want):
                            start(t + 1)
                        return
                    acc = inputs[taken]
                    taken += 1
                    mark, steps, power = None, 0, 1
                elif op == "OUT":
                    if given == len(want) or want[given] != acc:
                        return
                    given += 1
                elif op == "INC":
                    acc = min(acc + 1, 99)
                elif op == "NEG":
                    acc = -acc
                elif op == "ADD":
                    acc = max(-99, min(99, acc + bak))
                elif op == "SAV":
                    bak = acc
                elif op == "SWP":
                    acc, bak = bak, acc
                elif op == "TZ":
                    f = 1 if acc == 0 else 0
                elif op == "TP":
                    f = 1 if acc > 0 else 0
                else:
                    nxt = arg
            pc = 0 if nxt == n else nxt
            state = (pc, acc, bak, f)
            if state == mark:                     # same state twice with no IN in between: never halts
                return
            steps += 1
            if steps == power:                    # Brent's cycle detection: move the mark, double the wait
                mark, steps, power = state, 0, power * 2

    start(0)
    return found, nodes[0]


def as_source(program):
    """A search result as program text (unreached slots become INC, undecided guards become none)."""
    return "; ".join((g or "") + (op or "INC") + ("" if arg is None else f" {arg}") for g, op, arg in program)
```

在页面里，5 道题都搜到 5 条，节点数如下（验证代码会复跑并核对）：

| 题 | n = 1 | n = 2 | n = 3 | n = 4 | n = 5 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| Part 1 | 2 | 11 | 284 | — | — | 3 条找到 1 个解 |
| Part 2 | 2 | 11 | 282 | 7,808 | 268,888 | 5 条找到 4 个解 |
| Part 3 | 2 | 11 | 366 | 13,620 | 541,323 | 5 条找到 8 个解 |
| Part 4 | 2 | 11 | 324 | 9,668 | 319,927 | 5 条以内无解 |
| Part 5 | 2 | 11 | 269 | 6,591 | 199,731 | 5 条以内无解 |

Part 4 和 Part 5 的 6 条、Part 5 的 7 条是离线跑的（同一算法的 C 版，节点数和 Python 版逐个对上）：Part 4 的 6 条访问 10,884,617 个节点，找到 234 个解；Part 5 的 6 条访问 6,854,750 个节点，无解；Part 5 的 7 条找到 28 个解。

测试只有 8 组，会不会有程序只是碰巧过了测试？我们把所有最短的、能过测试的程序（Part 1 的 1 个、Part 2 的 4 个、Part 3 的 8 个、Part 4 的 234 个、Part 5 的 28 个）都放到所有短输入上又跑了一遍（每个位置取几个小数，长度到 4–7），全都正确。前三题的这一步在页面里复跑，后两题是离线跑的。

### 时间怎么分

- **0–10 分钟**：读题，写跟踪器。顺手用跟踪器把题目里的例子跑一遍，确认对饱和、跳过、停机的理解没错。
- **10–30 分钟**：每道题先交一个及格线版本。这样不管后面卡在哪里，分数都有保底（40 分对比全部没做的 100 分）。
- **30–100 分钟**：一题一题压。顺序按"每分钟能省几条"排：Part 1 和 Part 3 的想法一旦看到就直接到最优；Part 5 最难，放最后。
- **要不要写搜索**：上面的搜索不到 100 行，有 AI 助手的话不难写。它能证明前四题的最优解，还能直接把前三题的解找出来（页面里的验证代码就复跑了这一步）。但 Part 5 的 7 条要访问两亿多个节点，纯 Python 不现实，靠想比靠搜快。比较好的判断是：先手工做到目标附近，再用搜索确认"已经不能再少"，然后停手去做下一题。

### 追问

- **测试只有 8 组，怎么确信程序是对的？** 两步：一是在所有短输入上穷举（每个位置取几个小数），二是在合法范围内随机生成长输入。验证代码对每个参考解都做了这两步。上面也提到，所有最短的、能过测试的程序我们都这样复查过。
- **如果 Part 3 允许负数？** 参考解会错：静音时读到负数，`+TP` 得到 F=0，被当成 0 切回放音。比如输入 `3 0 -1 5` 应该输出 `3`，参考解输出 `3 5`。离线穷举表明，允许负数时 6 条以内无解；及格线那个 7 条的两循环程序仍然正确，7 条能过测试的程序共 11 个。
- **如果给机器加一条 `DEC`（`acc ← acc − 1`）？** Part 4 能降到 4 条：`-IN; +OUT; +DEC; TP`。但 Part 5 不受影响，离线穷举表明加了 `DEC` 之后 6 条以内仍然无解。题目故意去掉 `DEC`，就是为了逼出 `NEG; INC; NEG` 和 Part 5 的轮流加 1。
- **20,000 步够不够？** 最慢的情况是 Part 4、Part 5 输入 8 个 99。验证代码算了参考解和及格线程序在这种输入上的步数，最多的是 Part 5 的参考解，具体数字由代码打印；都在上限以内。这也是 Part 4、Part 5 限制"最多 8 个数"的原因。
- **为什么被跳过的指令也要算一步？** 否则一个全部被跳过的环会在零步内无限转下去，"跑了多少步"就没有意义。按现在的规则，环里每转一圈至少走 1 步，步数上限总能截住死循环。

<details>
<summary>验证代码（可运行）</summary>

```python
# ---- verbatim copy of the checker given in the statement ----
"""Validity checker for the guarded accumulator machine. This file is all the tooling you get."""

LIMIT = 99            # acc and bak saturate to [-LIMIT, LIMIT]
MAX_LEN = 12          # a program has 1..MAX_LEN instructions
MAX_STEPS = 20_000    # per test; a program that runs longer is invalid
OPS = ("IN", "OUT", "INC", "NEG", "ADD", "SAV", "SWP", "TZ", "TP", "JMP")


def parse(source):
    """Instructions are separated by newlines or ';'. '#' starts a comment.
    A '+' or '-' in front of the mnemonic is the guard. Returns [(guard, op, arg)]."""
    prog = []
    for line in source.replace(";", "\n").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        guard = ""
        if line[0] in "+-":
            guard, line = line[0], line[1:].strip()
        parts = line.split()
        if not parts or parts[0] not in OPS or len(parts) != (2 if parts[0] == "JMP" else 1):
            raise ValueError(f"bad instruction: {line!r}")
        prog.append((guard, parts[0], int(parts[1]) if parts[0] == "JMP" else None))
    if not 1 <= len(prog) <= MAX_LEN:
        raise ValueError(f"a program has 1..{MAX_LEN} instructions")
    if any(op == "JMP" and not 0 <= arg < len(prog) for _, op, arg in prog):
        raise ValueError("jump target out of range")
    return prog


def run(prog, inputs):
    """Outputs of one run, or None if the program is still running after MAX_STEPS steps."""
    acc = bak = 0
    flag = False
    pc = pos = 0
    out = []
    for _ in range(MAX_STEPS):
        guard, op, arg = prog[pc]
        pc = (pc + 1) % len(prog)                 # the program is a ring
        if (guard == "+" and not flag) or (guard == "-" and flag):
            continue                              # skipped: one step, no effect
        if op == "IN":
            if pos == len(inputs):
                return out                        # the only way to halt
            acc = inputs[pos]
            pos += 1
        elif op == "OUT":
            out.append(acc)
        elif op == "INC":
            acc = min(acc + 1, LIMIT)
        elif op == "NEG":
            acc = -acc
        elif op == "ADD":
            acc = max(-LIMIT, min(LIMIT, acc + bak))
        elif op == "SAV":
            bak = acc
        elif op == "SWP":
            acc, bak = bak, acc
        elif op == "TZ":
            flag = acc == 0
        elif op == "TP":
            flag = acc > 0
        elif op == "JMP":
            pc = arg
    return None


# ---- the five puzzles: a reference function (the spec) and the test inputs ----

def after_zero(xs):
    return xs[xs.index(0) + 1:] if 0 in xs else []


def group_sums(xs):
    out, total = [], 0
    for x in xs:
        if x == 0:
            out.append(total)
            total = 0
        else:
            total += x
    return out


def mute(xs):
    out, muted = [], False
    for x in xs:
        if x == 0:
            muted = not muted
        elif not muted:
            out.append(x)
    return out


def count_down(xs):
    return [k for x in xs for k in range(x, 0, -1)]


def count_up(xs):
    return [k for x in xs for k in range(1, x + 1)]


PUZZLES = {
    "p1": (after_zero, [
        [5, 0, 3, 0, -2, 7], [0, 4], [1, 2, 3], [], [0], [0, 0],
        [-6, 9, 0, 0, 0, 5, -99, 99, 0, 1], [7, -7, 0, 12, 0, 0, -3]]),
    "p2": (group_sums, [
        [3, 4, 0, 5, 0], [0], [], [2, 2, 2], [0, 0, 7, 0],
        [-5, 9, -1, 0, 20, 20, 20, 0, 6], [10, -10, 0, 1, 0, 0, -8, -9, 0],
        [4, 0, -4, 0, 4, -4, 0, 15, 3, 0, 0, 1]]),
    "p3": (mute, [
        [3, 0, 5, 0, 7], [], [0], [0, 0], [1, 2, 3],
        [0, 4, 4, 0, 0, 9, 0, 6, 1], [8, 0, 0, 0, 2, 0, 99, 5, 0, 0, 3],
        [5, 5, 0, 1, 0, 0, 0, 2]]),
    "p4": (count_down, [
        [3], [], [0], [1], [2, 0, 4], [0, 0, 1, 5, 0], [20], [7, 1, 0, 2, 2]]),
    "p5": (count_up, [
        [3], [], [0], [1], [2, 0, 4], [0, 0, 1, 5, 0], [20], [7, 1, 0, 2, 2]]),
}


def check(name, source):
    """(valid, instruction count). Valid: on every test the program halts with exactly the right output."""
    prog = parse(source)
    spec, tests = PUZZLES[name]
    return all(run(prog, xs) == spec(xs) for xs in tests), len(prog)


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
rng = random.Random(7)

# ---- 1. the machine example in the statement, through both the starter and the tracer ----
assert run(parse("IN; TP; +OUT"), [3, -1, 5]) == [3, 5]
out, halted, lines = trace("IN; TP; +OUT", [3, -1, 5], show=False)
assert (out, halted, len(lines)) == ([3, 5], True, 10)          # "共 10 步"
assert lines[5].endswith("skipped") and lines[9].endswith("halt")

# ---- 2. puzzle examples, checked against the spec and against independent rewrites ----
EXAMPLES = {'p1': [([5, 0, 3, 0, -2, 7], [3, 0, -2, 7]), ([1, 2, 3], []), ([0, 0], [0]), ([-4, 0], [])], 'p2': [([3, 4, 0, 5, 0], [7, 5]), ([0, 0, 7, 0], [0, 0, 7]), ([2, 2, 2], []), ([-5, 9, 0, 1], [4])], 'p3': [([3, 0, 5, 0, 7], [3, 7]), ([0, 4, 4, 0, 0, 9, 0, 6], [6]), ([0, 0, 8], [8]), ([2, 0], [2])], 'p4': [([3, 0, 2], [3, 2, 1, 2, 1]), ([1, 1], [1, 1])], 'p5': [([3, 0, 2], [1, 2, 3, 1, 2]), ([1, 1], [1, 1])]}


def after_zero_2(xs):
    seen, out = False, []
    for x in xs:
        if seen:
            out.append(x)
        seen = seen or x == 0
    return out


def group_sums_2(xs):
    zeros = [i for i, x in enumerate(xs) if x == 0]
    starts = [0] + [i + 1 for i in zeros]
    return [sum(xs[a:b]) for a, b in zip(starts, zeros)]


def mute_2(xs):
    return [x for i, x in enumerate(xs) if x != 0 and xs[:i].count(0) % 2 == 0]


def count_down_2(xs):
    out = []
    for x in xs:
        while x > 0:
            out.append(x)
            x -= 1
    return out


def count_up_2(xs):
    return [k for x in xs for k in reversed(count_down_2([x]))]


INDEPENDENT = {"p1": after_zero_2, "p2": group_sums_2, "p3": mute_2, "p4": count_down_2, "p5": count_up_2}
for name, pairs in EXAMPLES.items():
    for xs, want in pairs:
        assert PUZZLES[name][0](xs) == want == INDEPENDENT[name](xs), (name, xs)


def random_input(name, length):
    """A random input inside the domain the statement promises for this puzzle."""
    if name == "p1":
        return [rng.choice([0, rng.randint(-99, 99)]) for _ in range(length)]
    if name == "p3":
        return [rng.choice([0, rng.randint(0, 99)]) for _ in range(length)]
    if name in ("p4", "p5"):
        return [rng.randint(0, 99) for _ in range(min(length, 8))]
    xs, total = [], 0                                   # p2: keep every running sum in -99..99
    for _ in range(length):
        x = 0 if rng.random() < 0.3 else rng.randint(-99 - total, 99 - total)
        total = 0 if x == 0 else total + x
        xs.append(x)
    return xs


for name in PUZZLES:                                    # the specs agree with the rewrites on random inputs
    for _ in range(300):
        xs = random_input(name, rng.randint(0, 12))
        assert PUZZLES[name][0](xs) == INDEPENDENT[name](xs)

# ---- 3. reference solutions and baselines: valid, counts and totals as stated ----
REF = {"p1": "IN; +OUT; -TZ",
       "p2": "IN; TZ; -ADD; SWP; +OUT",
       "p3": "-IN; -TZ; +IN; -OUT; +TP",
       "p4": "-IN; TP; +OUT; NEG; INC; NEG",
       "p5": "TZ; SWP; INC; +IN; +NEG; TP; +OUT"}
BASE = {"p1": "IN; TZ; -JMP 0; IN; OUT; JMP 3",
        "p2": "IN; TZ; +JMP 6; ADD; SAV; JMP 0; SWP; OUT",
        "p3": "IN; TZ; -OUT; -JMP 0; IN; TZ; -JMP 4",
        "p4": "IN; TP; -JMP 0; OUT; NEG; INC; NEG; JMP 1",
        "p5": "IN; NEG; TZ; +SWP; +JMP 0; INC; SWP; INC; OUT; SWP; JMP 2"}
MIDDLE_P2 = "IN; TZ; -ADD; -SAV; +SWP; +OUT"
for name, n in zip(PUZZLES, (3, 5, 5, 6, 7)):
    assert check(name, REF[name]) == (True, n), name
for name, n in zip(PUZZLES, (6, 8, 7, 8, 11)):
    assert check(name, BASE[name]) == (True, n), name
assert check("p2", MIDDLE_P2) == (True, 6)
assert report(REF) == 26 and report(BASE) == 40 and report({}) == 100
assert all(run(parse(REF[name]), xs) == want for name, pairs in EXAMPLES.items() for xs, want in pairs)

# ---- 4. the tracer agrees with the starter (random programs, shorter step limit for speed) ----
saved, MAX_STEPS = MAX_STEPS, 400                       # run() reads the global MAX_STEPS at call time
for _ in range(400):
    n = rng.randint(1, 6)
    src = "; ".join(rng.choice(["", "+", "-"]) + (f"JMP {rng.randrange(n)}" if (op := rng.choice(OPS)) == "JMP" else op)
                    for _ in range(n))
    for _ in range(3):
        xs = [rng.randint(-3, 3) for _ in range(rng.randint(0, 5))]
        got = run(parse(src), xs)
        out, halted, _ = trace(src, xs, limit=MAX_STEPS, show=False)
        assert (got is not None) == halted and (got is None or got == out), (src, xs)
MAX_STEPS = saved
for name in PUZZLES:
    for src in (REF[name], BASE[name]):
        for xs in PUZZLES[name][1]:
            assert trace(src, xs, limit=MAX_STEPS, show=False)[0] == run(parse(src), xs)

# ---- 5. reference solutions beyond the 8 tests: all short inputs, then long random ones ----
SHORT = {"p1": (range(-2, 3), 6), "p2": (range(-2, 3), 6), "p3": (range(0, 4), 7),
         "p4": (range(0, 5), 4), "p5": (range(0, 5), 4)}


def short_inputs(name):
    alphabet, longest = SHORT[name]
    for k in range(longest + 1):
        yield from (list(t) for t in itertools.product(alphabet, repeat=k))


def correct_everywhere(name, src, randoms=300):
    prog, spec = parse(src), PUZZLES[name][0]
    if any(run(prog, xs) != spec(xs) for xs in short_inputs(name)):
        return False
    edge = [[99, 0, 99, -99, 0, -99]] if name == "p1" else [[99] * 8, [0] * 8] if name in ("p4", "p5") else []
    tries = edge + [random_input(name, rng.randint(0, 20)) for _ in range(randoms)]
    return all(run(prog, xs) == spec(xs) for xs in tries)


for name in PUZZLES:
    assert correct_everywhere(name, REF[name]) and correct_everywhere(name, BASE[name], 50), name
print("references correct on", {name: sum(1 for _ in short_inputs(name)) for name in PUZZLES}, "short inputs each")

# first_failure (from the tracer block) finds and traces the first failing test, or says all pass
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()) as buf:
    assert first_failure("p2", "TZ; -ADD; SWP; +OUT; IN") == [3, 4, 0, 5, 0]
    assert first_failure("p2", REF["p2"]) is None
printed = buf.getvalue().splitlines()
assert printed[0] == "input [3, 4, 0, 5, 0]: want [7, 5], got [0, 7, 5]" and printed[-1] == "all tests pass"

# ---- 6. traces quoted in the solution ----
assert trace("IN; +OUT; -TZ", [5, 0, 3], show=False)[2] == ['   0   0  IN       acc=  5 bak=  0 F=0', '   1   1  +OUT     skipped', '   2   2  -TZ      acc=  5 bak=  0 F=0', '   3   0  IN       acc=  0 bak=  0 F=0', '   4   1  +OUT     skipped', '   5   2  -TZ      acc=  0 bak=  0 F=1', '   6   0  IN       acc=  3 bak=  0 F=1', '   7   1  +OUT     acc=  3 bak=  0 F=1  -> out 3', '   8   2  -TZ      skipped', '   9   0  IN       no input left: halt']
out, halted, lines = trace(REF["p5"], [2, 0, 1], show=False)
assert lines[:14] == ['   0   0  TZ       acc=  0 bak=  0 F=1', '   1   1  SWP      acc=  0 bak=  0 F=1', '   2   2  INC      acc=  1 bak=  0 F=1', '   3   3  +IN      acc=  2 bak=  0 F=1', '   4   4  +NEG     acc= -2 bak=  0 F=1', '   5   5  TP       acc= -2 bak=  0 F=0', '   6   6  +OUT     skipped', '   7   0  TZ       acc= -2 bak=  0 F=0', '   8   1  SWP      acc=  0 bak= -2 F=0', '   9   2  INC      acc=  1 bak= -2 F=0', '  10   3  +IN      skipped', '  11   4  +NEG     skipped', '  12   5  TP       acc=  1 bak= -2 F=1', '  13   6  +OUT     acc=  1 bak= -2 F=1  -> out 1'] and out == [1, 2, 1] and halted

# ---- 7. lower bounds by exhaustive search (every puzzle up to 5 instructions) ----
t = time.time()
TOP = {"p1": 3, "p2": 5, "p3": 5, "p4": 5, "p5": 5}
NODES = {'p1': [2, 11, 284], 'p2': [2, 11, 282, 7808, 268888], 'p3': [2, 11, 366, 13620, 541323], 'p4': [2, 11, 324, 9668, 319927], 'p5': [2, 11, 269, 6591, 199731]}
found = {}
for name in PUZZLES:
    spec, tests = PUZZLES[name]
    cases = [(xs, spec(xs)) for xs in tests]
    for n in range(1, TOP[name] + 1):
        programs, nodes = search(cases, n)
        assert nodes == NODES[name][n - 1], (name, n, nodes)
        if programs:
            found[name] = (n, [as_source(p) for p in programs])
            break
assert found["p1"] == (3, [REF["p1"]])                  # optimal and unique
assert found["p2"][0] == 5 and len(found["p2"][1]) == 4 and REF["p2"] in found["p2"][1]
assert found["p3"][0] == 5 and len(found["p3"][1]) == 8 and REF["p3"] in found["p3"][1]
assert "p4" not in found and "p5" not in found          # nothing up to 5: Part 4's 6 is optimal
for name in ("p1", "p2", "p3"):                          # every shortest program is really correct
    for src in found[name][1]:
        assert check(name, src)[0] and correct_everywhere(name, src, 50), src
# Part 3's eight programs: the reference ring, or it with the adjacent +IN / -OUT swapped (5 in all),
# and the mirrored ring -TZ; +IN; +TP; +OUT; -IN, or it with +OUT / -IN swapped (3 in all)
def ring_rotations(src):
    w = src.split("; ")
    return {"; ".join(w[i:] + w[:i]) for i in range(len(w))}


same = ring_rotations(REF["p3"]) | ring_rotations("-IN; -TZ; -OUT; +IN; +TP")
mirror = ring_rotations("-TZ; +IN; +TP; +OUT; -IN") | ring_rotations("-TZ; +IN; +TP; -IN; +OUT")
p3 = set(found["p3"][1])
assert len(p3 & same) == 5 and len(p3 & mirror) == 3 and p3 <= same | mirror
# Part 2's four programs are four of the five rotations of one ring; the one starting at TZ
# sees the initial acc = 0 and prints a 0 before reading anything
words = REF["p2"].split("; ")
rotations = ["; ".join(words[i:] + words[:i]) for i in range(5)]
assert sorted(found["p2"][1]) == sorted(r for r in rotations if not r.startswith("TZ"))
assert run(parse("TZ; -ADD; SWP; +OUT; IN"), [3, 4, 0, 5, 0]) == [0, 7, 5]
print(f"search: {sum(sum(v) for v in NODES.values()):,} nodes in {time.time() - t:.1f}s")
assert (3 * (9 + 5)) ** 5 == 130_691_232                # "约 1.3 亿"

# ---- 8. follow-ups ----
assert run(parse(REF["p3"]), [3, 0, -1, 5]) == [3, 5] and mute([3, 0, -1, 5]) == [3]
neg_inputs = [list(t) for k in range(6) for t in itertools.product((-1, 0, 2), repeat=k)]
assert all(run(parse(BASE["p3"]), xs) == mute(xs) for xs in neg_inputs)      # the 7 survives negatives


def run_with_dec(src, inputs):
    """Machine A plus DEC, only to check the 4-instruction Part 4 from the follow-up."""
    prog = [(line[0] if line[0] in "+-" else "", line.lstrip("+-")) for line in src.split("; ")]
    acc, flag, pc, rest, out = 0, False, 0, list(inputs), []
    for _ in range(MAX_STEPS):
        guard, op = prog[pc]
        pc = (pc + 1) % len(prog)
        if (guard == "+" and not flag) or (guard == "-" and flag):
            continue
        if op == "IN":
            if not rest:
                return out
            acc = rest.pop(0)
        elif op == "OUT":
            out.append(acc)
        elif op == "DEC":
            acc = max(acc - 1, -99)
        elif op == "TP":
            flag = acc > 0
    return None


assert all(run_with_dec("-IN; +OUT; +DEC; TP", xs) == count_down(xs)
           for xs in PUZZLES["p4"][1] + [list(t) for k in range(4) for t in itertools.product(range(6), repeat=k)])

steps = {(name, kind): len(trace(src[name], [99] * 8, limit=MAX_STEPS, show=False)[2])
         for name in ("p4", "p5") for kind, src in (("ref", REF), ("base", BASE))}
print("steps on eight 99s:", steps)
assert max(steps.values()) < MAX_STEPS and max(steps, key=steps.get) == ("p5", "ref")
print(f"all checks passed in {time.time() - T0:.1f}s")
```

</details>

</details>
