# Constrained-Machine Puzzles (A): A Guarded Accumulator Machine

## Problem

> **This is not the original problem.** Anthropic's current performance take-home is a set of puzzles on a tiny, heavily constrained instruction set, where the goal is to solve each puzzle in as few instructions as possible. The real machine and puzzles have not been published. This machine and the 5 puzzles below are our own design and keep only the shape of the format: few instructions, few registers, state that has to be hidden in the flag and the program position, and no debugging tools.

You get a description of a small machine, a checker, and 5 independent puzzles. For each puzzle you submit a program for this machine. The checker tells you only two things: whether the program is valid, and how many instructions it has. The goal is to make the total instruction count across the 5 puzzles as small as possible.

### The machine

- Two integer registers: `acc` (the accumulator) and `bak` (a backup register). Both start at 0. All results saturate to −99..99: anything above 99 is stored as 99, anything below −99 as −99.
- One flag `F`, which starts at 0. Only `TZ` and `TP` change it.
- A program has 1–12 instructions, numbered from 0, joined end to end into a ring: after the last instruction comes instruction 0. A program has no notion of "finishing".
- Any instruction can be prefixed with a guard: `+` means it executes only when F=1, `-` means it executes only when F=0, and no prefix means it always executes. When the guard isn't satisfied, the instruction is skipped: it still counts as a step but does nothing. A skipped `IN` reads no input and does not halt.
- The input is a sequence of integers, and so is the output.
- **The only way to halt**: executing `IN` when the input is exhausted. On halting, the output is whatever has been `OUT`-ed up to that point.
- Each test runs for at most 20,000 steps (skipped instructions count too). Going over makes the program invalid.

| Instruction | Effect |
| --- | --- |
| `IN` | Read the next input into `acc`; halt if the input is exhausted |
| `OUT` | Append `acc` to the output |
| `INC` | `acc ← acc + 1` (saturating) |
| `NEG` | `acc ← −acc` |
| `ADD` | `acc ← acc + bak` (saturating) |
| `SAV` | `bak ← acc` |
| `SWP` | Swap `acc` and `bak` |
| `TZ` | `F ← 1` if `acc = 0`, otherwise `F ← 0` |
| `TP` | `F ← 1` if `acc > 0`, otherwise `F ← 0` |
| `JMP k` | Execute instruction `k` next (`0 ≤ k <` program length) |

Note that there is no "decrement", and no instruction that compares the two registers. Instructions are separated by newlines or `;`, and `#` starts a comment.

Example: the program `IN; TP; +OUT`, which outputs all positive numbers, on input `3 -1 5`.

```text
step pos instr   after
 0   0   IN      acc=3
 1   1   TP      F=1
 2   2   +OUT    output 3
 3   0   IN      acc=-1      (after instruction 2 comes instruction 0 again)
 4   1   TP      F=0
 5   2   +OUT    skipped (F=0)
 6   0   IN      acc=5
 7   1   TP      F=1
 8   2   +OUT    output 5
 9   0   IN      input exhausted, halt
Output: 3 5, 10 steps in total
```

### The checker

This is all the tooling you get:

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

### Rules and scoring

- Submit one program per puzzle. **Valid** means: on every test for that puzzle it halts, and its output is exactly what is required.
- **Score = the sum of the instruction counts of the 5 programs**; lower is better. A puzzle that is missing or invalid scores 20.
- The checker won't tell you which test failed or at which step. There is no debugger and no single-stepping. If you need tools, build them yourself: add prints, or have an AI coding assistant write a single-step tracer in a few minutes. How much time you spend on tooling, and what tools you build, is itself part of what is being assessed.
- The time limit is 2 hours, and you may use an AI assistant.
- Scoring uses only the given tests, but solve each puzzle for the input range stated in it; don't tailor programs to the tests.

### Part 1 — After the zero

The input is a sequence of integers, each in −99..99. Output every number **after** the first 0, in the original order. The first 0 itself is not output; any later 0s are output as usual. If there is no 0, output nothing.

```text
5 0 3 0 -2 7  ->  3 0 -2 7
1 2 3         ->  (empty)
0 0           ->  0
-4 0          ->  (empty)
```

Pass line 6 instructions, target 3.

### Part 2 — Group sums

The input is a sequence of integers, with 0 as the separator. Each time a 0 is read, output the sum of the numbers between the previous 0 (or the start) and this 0; if there are none in between, output 0. Numbers left over after the last 0 are not output. Every running sum within a group, from its start to any position, is guaranteed to stay within −99..99.

```text
3 4 0 5 0  ->  7 5
0 0 7 0    ->  0 0 7
2 2 2      ->  (empty)
-5 9 0 1   ->  4
```

Pass line 8 instructions, target 5.

### Part 3 — Mute toggle

The input is a sequence of integers, each in **0..99** (no negatives). It starts in the "playing" state. Each 0 toggles between playing and muted, and the 0 itself is not output; while playing, other numbers are output unchanged, and while muted they are dropped.

```text
3 0 5 0 7        ->  3 7
0 4 4 0 0 9 0 6  ->  6
0 0 8            ->  8
2 0              ->  2
```

Pass line 7 instructions, target 5.

### Part 4 — Count down

The input has at most 8 numbers, each in 0..99. For each number `x`, output `x, x−1, …, 1` in turn; for `x = 0` output nothing.

```text
3 0 2  ->  3 2 1 2 1
1 1    ->  1 1
```

Pass line 8 instructions, target 6.

### Part 5 — Count up

The input is the same as in Part 4. For each number `x`, output `1, 2, …, x` in turn; for `x = 0` output nothing.

```text
3 0 2  ->  1 2 3 1 2
1 1    ->  1 1
```

Pass line 11 instructions, target 7.

The pass lines for the 5 puzzles add up to 40 instructions, and the targets add up to 26.

## Reference solution

<details>
<summary>Show the reference solution</summary>

The conclusions first:

| Puzzle | Pass-line program | Reference solution | Key idea | Optimality |
| --- | --- | --- | --- | --- |
| Part 1 After the zero | 6 | **3** | A guarded test that fires only once, used as a latch | Proved optimal by exhaustive search, and unique |
| Part 2 Group sums | 8 | **5** | One `SWP` stores, fetches and clears at the same time | Proved optimal by exhaustive search |
| Part 3 Mute toggle | 7 | **5** | The flag is the toggle; `TZ` and `TP` are complementary on non-negative numbers | Proved optimal by exhaustive search |
| Part 4 Count down | 8 | **6** | No jumps: the program ring itself is the inner loop; `NEG; INC; NEG` is a decrement | Proved optimal by exhaustive search |
| Part 5 Count up | 11 | **7** | One `INC` increments the two registers in turn, told apart by sign | Proved optimal by offline exhaustive search |

26 instructions in total.

### Spend the first 10 minutes writing a single-step tracer

The checker only says "invalid". When a program is invalid, what you need to know is: which test failed, at which step, and what the registers and flag were just before that step. So the first thing to do is write a tracer that prints, for every step, the position, the instruction, and `acc`, `bak` and `F` after it executes. Write it separately from the checker (its own parser, its own execution loop), so that you can also cross-check it against the checker and confirm that your understanding of the rules is right.

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

`trace("IN; +OUT; -TZ", [5, 0, 3])` prints the run of the Part 1 reference solution:

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

These few dozen lines are enough; a graphical interface isn't worth building. The remaining time is better spent on two things: first submit a passing version for every puzzle, then squeeze the instruction counts down one puzzle at a time.

### Part 1 — 3 instructions: a test that fires only once

```text
IN        # read x
+OUT      # after the first 0: echo
-TZ       # before the first 0: F = (x == 0)
```

F starts at 0. Until the first 0 is read, each lap is `IN`, skip `+OUT`, `-TZ`. Once `-TZ` sets F to 1 (the number just read is 0), `-TZ` is skipped forever after, and F never goes back to 0. So every lap after that is `IN`, `OUT`, skip. The `-`-guarded test acts like a switch that can be flipped only once, i.e. a latch. Reading another 0 later has no effect, which is exactly what "later 0s are output as usual" requires.

The pass-line program uses two loops and jumps:

```text
IN; TZ; -JMP 0; IN; OUT; JMP 3
```

The first half looks for the 0, the second half echoes the input, and the state is "which loop the program is currently in". The reference solution moves that state into F, merging the two loops into one and dropping every jump. Exhaustive search shows there is no solution with 2 or fewer instructions, and this is the only 3-instruction solution.

### Part 2 — 5 instructions: one SWP stores, fetches and clears

```text
IN        # read x
TZ        # F = (x == 0)
-ADD      # x != 0: acc = x + running sum
SWP       # x != 0: bak = new sum   |  x == 0: acc = sum, bak = 0
+OUT      # x == 0: output the sum
```

Invariant: at the start of every lap (before `IN`), `bak` holds the running sum of the current group.

- Reading `x ≠ 0`: `ADD` gives the new sum, and `SWP` puts it into `bak`. What comes back into `acc` is the old sum, which is useless, but the next lap's `IN` overwrites it right away.
- Reading `x = 0`: `ADD` is skipped, and `SWP` swaps the sum into `acc` ready for output, while swapping the 0 just read into `bak`. **The 0 that serves as the separator is exactly the initial value of the next group**, so no separate clearing is needed.

How to get there step by step:

1. The pass-line program (8 instructions) uses a jump to split into two paths: `IN; TZ; +JMP 6; ADD; SAV; JMP 0; SWP; OUT`. It already uses "the separator 0 as the initial value".
2. Replace the jumps with guards to get 6 instructions: `IN; TZ; -ADD; -SAV; +SWP; +OUT`.
3. Look at `SAV` on the `x ≠ 0` path: it has the same effect on `bak` as `SWP` (both do `bak ← acc`), except that `SWP` also clobbers `acc`, and `acc` is overwritten on the next step anyway. So `-SAV` and `+SWP` can be merged into one unguarded `SWP`, giving 5 instructions.

Exhaustive search shows there is no solution with 4 or fewer instructions. There are 4 programs with 5 instructions that pass the tests, and they are really the same program: the program is a ring, and starting to write it at a different position on the ring gives a different text. The ring has 5 starting points; the missing one starts at `TZ`: its first step, `TZ`, sees the initial acc = 0, so it outputs a 0 before reading any input (`3 4 0 5 0` gives `0 7 5`).

### Part 3 — 5 instructions: the flag is the toggle

```text
-IN       # playing: read x
-TZ       # playing: F = (x == 0), i.e. a 0 switches to muted
+IN       # muted: read y (also right after the switch)
-OUT      # playing: echo x
+TP       # muted: F = (y > 0), i.e. a 0 switches back to playing
```

Put the state into F: F=0 means playing, F=1 means muted.

- While playing (F=0), what executes is `-IN; -TZ; -OUT`: read a number, and output it unless it is 0.
- While muted (F=1), what executes is `+IN; +TP`: read a number and drop it.
- Toggling relies on a coincidence: on **non-negative numbers**, `TZ` (is it 0) and `TP` (is it positive) are exact opposites. While playing we use `TZ`, and reading a 0 gives F=1; while muted we use `TP`, and reading a 0 gives F=0. So **in both states, reading a 0 flips F**, and reading any other number leaves F unchanged.
- When a 0 is read while playing, F becomes 1 immediately, so the following `+IN` takes effect at once and reads the first number of the muted state directly, without wasting any instructions.

The pass-line program (7 instructions) keeps the state in the program position and jumps between two loops:

```text
IN; TZ; -OUT; -JMP 0; IN; TZ; -JMP 4
```

Exhaustive search shows there is no solution with 4 or fewer instructions. There are 8 programs with 5 instructions that pass the tests, but really only two ways of writing it:

- This program itself, or the same with the adjacent `+IN` and `-OUT` swapped (their guards are opposite and neither changes F, so exactly one of them executes on each lap and the order doesn't matter), each with different starting points: 5 programs in total.
- The mirror version `-TZ; +IN; +TP; +OUT; -IN` and its variants, 3 programs in total: F=1 means playing, and `TP` detects the 0 while playing; F=0 means muted, and `TZ` detects the 0. The leading `-TZ` uses the initial acc = 0 to set F to 1.

The input range for this puzzle deliberately says "no negatives": if negatives were allowed, `TP` would treat a negative number like 0, the toggle would be falsely triggered by negatives, and the reference solution would no longer work (see the follow-ups).

### Part 4 — 6 instructions: the program ring as the inner loop

```text
-IN       # only when the previous count is finished
TP        # F = (acc > 0)
+OUT
NEG       # acc - 1 written as -(-acc + 1)
INC
NEG
```

Two ideas:

- **There's no decrement, so use `NEG; INC; NEG`**: `−(−x + 1) = x − 1`.
- **Don't write a loop**: the program itself is a ring. `-IN` reads a new number only when F=0, so as long as the countdown is still going (F=1), each lap skips `-IN` automatically, and the ring itself is the inner loop. When the count reaches 0, `TP` gives F=0, nothing is output on that lap, and `acc` becomes −1 (which doesn't matter), and on the next lap `-IN` reads a new number over it. An input of 0 takes the same path and outputs nothing.

The pass-line program (8 instructions) is the conventional approach, with an outer loop reading numbers and an inner loop built from jumps:

```text
IN; TP; -JMP 0; OUT; NEG; INC; NEG; JMP 1
```

The reference solution removes both jumps. Exhaustive search shows there is no solution with 5 or fewer instructions, so 6 is optimal.

### Part 5 — 7 instructions: one INC increments two registers in turn

Counting up is harder than counting down: you have to remember both "how far we've counted" (the counter) and "how many are left" (the remainder) at once, and both need incrementing or decrementing, but the machine has only one `INC`, and it acts only on `acc`.

```text
TZ        # F = 1 exactly when the remaining count just reached 0
SWP       # every lap: switch which register INC works on
INC
+IN       # remaining is 0: read the next x ...
+NEG      # ... and store it as -x
TP        # positive = counter, negative (or 0) = remaining
+OUT      # the counter just went up: output it
```

The key is that `SWP` executes once on every lap, so the same `INC` increments the counter on one lap and the remainder on the next, taking turns:

- The remainder is stored as a **negative** number `−r`, and the counter is a positive `c`. Adding 1 to `−r` decrements the remainder.
- After each `INC`, `TP` checks which one was just incremented: positive means the counter was just incremented, so output it; negative or 0 means it was the remainder, so don't output.
- When the remainder reaches 0, the next lap's `TZ` gives F=1, `+IN` reads the next `x`, and `+NEG` turns it into `−x`. At that point the other register holds the remainder that just ran out, 0, and **it becomes the new counter's initial value directly**, with no clearing needed.
- At the start `acc = 0`, so the very first lap's `TZ` gives F=1, and reading the first number takes the same path.

The first 14 steps of `trace(...)` on input `2 0 1`:

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

The pass-line program (11 instructions) uses a counter, a remainder and three `SWP`s:

```text
IN; NEG; TZ; +SWP; +JMP 0; INC; SWP; INC; OUT; SWP; JMP 2
```

The exhaustive search on this page only goes up to 5 instructions (no solution). The search for 6 instructions visits about 6.85 million nodes; we ran it offline with the same algorithm (the C and Python versions agree): there is no 6-instruction solution, so 7 is optimal. There are 28 programs with 7 instructions that pass the tests.

### Lower bounds: search every short program

"Optimal" for each puzzle doesn't come from intuition but from exhaustive search. Enumerating directly is far too much: in a program of length `n`, each position has 3 guards and `9 + n` instructions (a jump has to pick a target), so programs of 5 instructions already number `(3 × 14)^5`, about 130 million. The search below generates programs while running them:

- The instruction at a position is enumerated only **when it is first executed**; the guard at a position is decided only when the position is first reached with F=0 or with F=1. Positions that are never executed don't need to be enumerated.
- As soon as a wrong number is output, the whole branch is abandoned. Most programs are pruned within the first few steps.
- If a state `(position, acc, bak, F)` repeats between two `IN`s, the program will never halt, so it is abandoned too. Brent's cycle detection algorithm decides this, so the check is exact, and the lower bound doesn't depend on the step limit.
- A lower bound found by searching on only some of the tests also holds for all of them: a program that is correct on all the tests is certainly correct on a subset of them.

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

On this page, all 5 puzzles are searched up to 5 instructions, with the following node counts (the check code reruns and verifies them):

| Puzzle | n = 1 | n = 2 | n = 3 | n = 4 | n = 5 | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Part 1 | 2 | 11 | 284 | — | — | 1 solution found at 3 instructions |
| Part 2 | 2 | 11 | 282 | 7,808 | 268,888 | 4 solutions found at 5 instructions |
| Part 3 | 2 | 11 | 366 | 13,620 | 541,323 | 8 solutions found at 5 instructions |
| Part 4 | 2 | 11 | 324 | 9,668 | 319,927 | No solution with 5 or fewer |
| Part 5 | 2 | 11 | 269 | 6,591 | 199,731 | No solution with 5 or fewer |

The 6-instruction searches for Parts 4 and 5 and the 7-instruction search for Part 5 were run offline (a C version of the same algorithm, whose node counts match the Python version one for one): the 6-instruction search for Part 4 visits 10,884,617 nodes and finds 234 solutions; the 6-instruction search for Part 5 visits 6,854,750 nodes and finds none; the 7-instruction search for Part 5 finds 28 solutions.

With only 8 tests, could some program pass them just by luck? We took every shortest program that passes the tests (1 for Part 1, 4 for Part 2, 8 for Part 3, 234 for Part 4, 28 for Part 5) and ran it again on all short inputs (a few small numbers at each position, lengths up to 4–7), and every one was correct. This step is rerun on the page for the first three puzzles; for the last two it was run offline.

### How to split the time

- **0–10 minutes**: read the problem and write the tracer. While you're at it, run the examples from the statement through the tracer to confirm your understanding of saturation, skipping and halting.
- **10–30 minutes**: submit a pass-line version for every puzzle first. That way, wherever you get stuck later, your score has a floor (40, compared with 100 for doing nothing).
- **30–100 minutes**: squeeze them one at a time. Order them by "instructions saved per minute": for Parts 1 and 3, once you see the idea you go straight to optimal; Part 5 is the hardest, so leave it for last.
- **Whether to write the search**: the search above is under 100 lines, and with an AI assistant it isn't hard to write. It can prove the optimal solutions for the first four puzzles, and can find the solutions for the first three directly (the check code on this page reruns that step). But 7 instructions for Part 5 means visiting over two hundred million nodes, which isn't realistic in pure Python, and thinking is faster than searching there. A good call is: get close to the target by hand first, then use the search to confirm "it can't get any shorter", then stop and move on to the next puzzle.

### Follow-ups

- **With only 8 tests, how can you be confident a program is correct?** Two steps: first, run exhaustively on all short inputs (a few small numbers at each position); second, generate long random inputs within the valid range. The check code does both for every reference solution. As mentioned above, we also rechecked every shortest program that passes the tests this way.
- **What if Part 3 allowed negative numbers?** The reference solution would be wrong: reading a negative number while muted, `+TP` gives F=0, which is treated like a 0 and switches back to playing. For example, input `3 0 -1 5` should output `3`, but the reference solution outputs `3 5`. Offline exhaustive search shows that with negatives allowed there is no solution with 6 or fewer instructions; the 7-instruction two-loop pass-line program is still correct, and there are 11 programs with 7 instructions that pass the tests.
- **What if the machine got a `DEC` instruction (`acc ← acc − 1`)?** Part 4 drops to 4 instructions: `-IN; +OUT; +DEC; TP`. But Part 5 is unaffected: offline exhaustive search shows that even with `DEC` there is still no solution with 6 or fewer instructions. `DEC` was deliberately left out to force `NEG; INC; NEG` and Part 5's alternating increment.
- **Are 20,000 steps enough?** The slowest case is Parts 4 and 5 with an input of eight 99s. The check code computes the step counts of the reference solutions and the pass-line programs on this input; the largest is Part 5's reference solution, and the code prints the exact numbers; all are within the limit. This is also why Parts 4 and 5 are limited to "at most 8 numbers".
- **Why does a skipped instruction count as a step?** Otherwise a ring in which everything is skipped would spin forever in zero steps, and "how many steps it ran" would be meaningless. Under the current rules, every lap around the ring takes at least 1 step, so the step limit always catches an infinite loop.

<details>
<summary>Checks (runnable)</summary>

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
assert (out, halted, len(lines)) == ([3, 5], True, 10)          # "10 steps in total"
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
assert (3 * (9 + 5)) ** 5 == 130_691_232                # "about 130 million"

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
