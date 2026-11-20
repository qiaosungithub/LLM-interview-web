# Constrained-Machine Puzzles (B): A Machine With Only a Queue

## Problem

> **This is not the original problem.** Anthropic's current performance take-home is a set of puzzles on a tiny, heavily constrained instruction set, where the goal is to solve each puzzle in as few instructions as possible. The real machine and puzzles have not been published; this machine and the 4 puzzles below are our own design. Its machine is completely different in style from the one in [the first set](#/anthropic/take-home/constrained-machine-puzzles-a): that one relies on registers and condition prefixes, while this one has no registers, and all data lives in a single queue.

You get a description of a small machine, a checker, and 4 independent small puzzles. For each puzzle you submit one program for this machine. The checker tells you only two things: whether the program is valid, and how many instructions it has. The goal is to make the total instruction count across the 4 puzzles as small as possible.

### The machine

- The only storage is a first-in, first-out queue (FIFO queue) holding integers in 0..99, at most 16 of them. There are no registers.
- At the start, the queue holds the puzzle's input. Whatever numbers remain in the queue when the program halts (from head to tail) are the output.
- Instructions take numbers from the **head** of the queue and put results at the **tail**.
- All arithmetic saturates: an addition above 99 gives 99; a subtraction below 0 gives 0. Below, this subtraction is written `a ∸ b = max(a − b, 0)` (truncated subtraction, monus).
- A program has 1–12 instructions, numbered from 0, executed in order starting from instruction 0. A jump target can be `0..n`, where `n` is the program length; jumping to `n` means halt.
- There are two ways to **halt**: reaching instruction `n` (finishing the last instruction, or jumping to `n`); or the current instruction needs more numbers than the queue holds, in which case that instruction is **not executed** and the machine halts immediately (called "starvation" below).
- More than 16 numbers in the queue is invalid; more than 20,000 steps on any test is also invalid.

| Instruction | Numbers taken | Effect (queue `a b …`, head on the left) |
| --- | --- | --- |
| `ADD` | 2 | Take `a` and `b`, put `min(a + b, 99)` |
| `SUB` | 2 | Take `a` and `b`, put `a ∸ b` |
| `DUP` | 1 | Take `a`, put `a`, `a` |
| `ROT` | 1 | Take `a`, put `a` (the head moves to the tail) |
| `DROP` | 1 | Take `a` and discard it |
| `INC` | 1 | Take `a`, put `min(a + 1, 99)` |
| `DEC` | 1 | Take `a`, put `a ∸ 1` |
| `ZERO` | 0 | Put `0` |
| `JMP k` | 0 | Execute instruction `k` next |
| `JZ k` | 1 | Take `a`; if `a = 0`, jump to instruction `k`, otherwise continue with the next instruction |

Instructions are separated by newlines or `;`, and anything after `#` is a comment.

Example: the program `ADD; JMP 0`, which sums all the numbers, on input `3 4 5`.

```text
step pc  instr   queue after execution
         start   3 4 5
 0   0   ADD     5 7
 1   1   JMP 0   5 7
 2   0   ADD     12
 3   1   JMP 0   12
 4   0   ADD     needs 2 numbers, the queue has only 1: starvation, halt
output: 12
```

### The checker

This is the only tool you get:

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

### Rules and scoring

- Submit one program per puzzle. **Valid** means: on every test of the puzzle it halts, the queue at halt is **exactly** the required output (one number too many or too few doesn't count), and the queue never exceeds 16 numbers along the way.
- **Score = the sum of the instruction counts of the 4 puzzles**; lower is better. A puzzle that is not submitted or is invalid counts as 20.
- The checker won't tell you which test failed. There's no debugger; if you need tools, build them yourself: add prints, or have an AI coding assistant write a single-step tracer. How much time you spend building tools is itself part of what's being assessed.
- The time limit is 2 hours, and you may use an AI assistant.
- Grading looks only at the given tests, but please solve each puzzle for the input range stated in it.

### Part 1 — The larger value

The input is two numbers `a b`, both in 0..99. Output one number: `max(a, b)`.

```text
3 8   ->  8
5 5   ->  5
99 0  ->  99
```

Passing bar 11 instructions, target 3.

### Part 2 — Absolute difference

The input is two numbers `a b`, both in 0..99. Output one number: `|a − b|`.

```text
3 8   ->  5
8 3   ->  5
0 99  ->  99
```

Passing bar 8 instructions, target 6.

### Part 3 — Counting

The input is 1 to 15 numbers, each in **1..99** (no 0). Output one number: how many numbers the input has.

```text
4 4 9         ->  3
99            ->  1
7 1 2 3 99 5  ->  6
```

Passing bar 8 instructions, target 6.

### Part 4 — Equality

The input is two numbers `a b`, both in **0..98** (99 never appears in this puzzle). If `a = b`, output `1`; otherwise output `0`.

```text
7 7   ->  1
7 2   ->  0
0 98  ->  0
```

Passing bar 11 instructions, target 8.

The passing bars of the 4 puzzles total 38 instructions, and the targets total 23.

## Reference solution

<details>
<summary>Show the reference solution</summary>

The bottom line first:

| Puzzle | Our first version | Reference solution | Key idea | Optimality |
| --- | --- | --- | --- | --- |
| Part 1 larger value | 11 | **3** | `max(a, b) = a + (b ∸ a)`, no comparison needed | Proved optimal by exhaustive search, and unique |
| Part 2 absolute difference | 7 | **6** | `(a ∸ b) + (b ∸ a)`, no branching; lay out operands in the order they're taken | Proved optimal by exhaustive search (uniqueness confirmed offline) |
| Part 3 counting | 7 | **6** | First replace each number with 0, then use the fact that "count + sum" is invariant | Proved optimal by exhaustive search |
| Part 4 equality | 11 | **8** | `(a + 1) ∸ b` separates less-than, equal and greater-than in one go; the start of the program doubles as "output 1" | Proved optimal by offline exhaustive search |

23 instructions in all. The tricks on this machine are completely different from the first set: there, state was hidden in flags and condition prefixes; here, it's **identities of truncated subtraction**, **ordering operands by queue order**, and **using starvation as the exit**.

### First, write a single-step tracer

As with the first set, the first thing to do is write a tracer that prints, at every step, the position, the instruction and the whole queue after execution. It's written separately from the checker, and the verification code cross-checks the two against each other.

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

On this machine the tracer matters even more than on the first one: the numbers in the queue keep rotating, and keeping "who's at the head right now" in your head quickly gets confusing.

### Part 1 — 3 instructions: truncated subtraction instead of comparison

```text
DUP       # a b      ->  b a a
SUB       # b a a    ->  a (b ∸ a)
ADD       # a (b ∸ a) -> a + (b ∸ a) = max(a, b)
```

`b ∸ a` is `b − a` when `b > a`, and 0 otherwise. So `a + (b ∸ a)` is `b` when `b > a` and `a` otherwise, which is exactly the larger value, and it never exceeds 99, so saturation never truncates it.

The first version (11 instructions) compared first and then branched:

```text
DUP; DUP; ROT; SUB; ROT; ROT; JZ 9; DROP; JMP 11; ROT; DROP
```

It lays out `a b b a`, computes `a ∸ b` and rotates it to the head, uses `JZ` to test whether it's 0, and then splits into two paths that drop the smaller one. Once you see that truncated subtraction already carries a "comparison", the branch can be removed entirely. Exhaustive search shows there's no solution with 2 instructions or fewer, and this is the only 3-instruction solution.

### Part 2 — 6 instructions: no branching, numbers laid out in the order they're taken

```text
DUP       # a b      ->  b a a
DUP       # b a a    ->  a a b b
ROT       # a a b b  ->  a b b a
SUB       # a b b a  ->  b a (a ∸ b)
SUB       # b a (a ∸ b) -> (a ∸ b) (b ∸ a)
ADD       # -> (a ∸ b) + (b ∸ a) = |a − b|
```

Two ideas:

- **No branching**: at least one of `a ∸ b` and `b ∸ a` is 0, and the other is `|a − b|`, so adding them gives the answer. Comparing first and then branching requires both keeping copies of `a` and `b` and splitting into two paths, which is hard to write short.
- **Lay out numbers in the order they're taken**: the queue is first-in, first-out, so a computed result goes to the tail and only gets its turn once the numbers ahead of it are used up. So when writing the program, first work out the order in which each instruction will take numbers, and lay out the operands as `a b b a` in one go; then the following `SUB; SUB; ADD` can run one after another with no more shuffling. This is the same thing as traversing an expression tree in level order: the operations at the same level go first, and their results line up behind, in order, for the level above.

The first version (7 instructions) made one extra copy and threw it away at the end:

```text
DUP; DUP; DUP; SUB; SUB; DROP; ADD
```

Exhaustive search shows there's no solution with 5 instructions or fewer; this is the only 6-instruction solution (an offline search listed all 6-instruction solutions).

### Part 3 — 6 instructions: count plus sum is invariant

```text
ZERO      # phase 1: append a 0 ...
JZ 3      # ... and pop the front: a 0 at the front means every input is used up
JMP 0
INC       # phase 2: turn one 0 into a 1 at the back
ADD       # merge the two front values
JMP 3     # until ADD starves
```

The hardest thing on a queue machine is "counting things": there's no register to hold a counter, and a counter put in the queue gets mixed up with the input. The reference solution has two phases:

1. **Replace each input with a 0.** Each loop puts in a 0 and takes out the head. None of the inputs is 0, so in the first `k` loops, what `JZ` takes out is always an input and it doesn't jump; after `k` loops the queue holds exactly `k` zeros. In loop `k + 1`, `JZ` takes out a 0 for the first time and jumps to the second phase, at which point the queue still holds `k` zeros. **These zeros are both stand-ins and the marker for "the input is used up"**, so no separate end marker is needed.
2. **Turn "count" into "value".** Look at the quantity "queue length + sum of all numbers": `INC` adds 1 to the sum and leaves the length unchanged; `ADD` reduces the length by 1 and leaves the sum unchanged (the numbers here are all small, so saturation never truncates them). So at the start of every loop, "length + sum = k" holds. At the start of the last loop the queue has only 1 number left, which by the invariant is `k − 1`; `INC` turns it into `k`, then `ADD` needs 2 numbers and starves, the program halts, and the output is exactly `k`.

The complete run on input `8 5 9 4`:

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

The first version (7 instructions) first put in a separate 0 as an end marker:

```text
ZERO; JZ 4; ZERO; JMP 1; INC; ADD; JMP 4
```

The reference solution realized that the stand-in zeros can serve as the end marker themselves, saving one instruction. What's really hard about this puzzle is the second phase: until you think of the "length + sum" invariant, you often can't write even one program that passes. Exhaustive search shows there's no solution with 5 instructions or fewer; there are 3 six-instruction solutions, all using this idea (the offline search listed all of them).

The queue holds at most `k + 1` numbers at once (when the last loop of the first phase puts in a 0), so with at most 15 inputs it stays within 16.

### Part 4 — 8 instructions: one subtraction splits three ways, and the start doubles as "output 1"

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

Three ideas stacked together:

- **`(a + 1) ∸ b` splits three ways at once**: it's 0 for less-than, exactly 1 for equal, and at least 2 for greater-than. One subtraction separates the three cases, without computing `|a − b|` first.
- **`ZERO` goes before `JZ`**: `JZ` has to consume the number being tested, so `ZERO` leaves a 0 at the tail beforehand, and after the test the queue holds exactly the 0 the answer needs.
- **Reuse the start of the program, with starvation as the exit**: when the output should be 1, jump back to instruction 0, where `INC` turns the remaining 0 into 1, then `ROT` rotates it unchanged, and `SUB` needs 2 numbers while the queue has only 1, so it starves and halts. The path that outputs 0 works the same way: it jumps to instruction 1 and starves at `SUB`.

The run for `a = b = 7`:

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

This relies on "the input is never 99": if `a = b = 99`, `INC` gets truncated by saturation, `d` comes out as 0, and the program wrongly decides it's less-than (see the follow-ups).

The first version (11 instructions) first computed `|a − b|` with Part 2's 6 instructions, then branched:

```text
DUP; DUP; ROT; SUB; SUB; ADD; JZ 9; ZERO; JMP 11; ZERO; INC
```

The exhaustive search on this page only goes up to 5 instructions (no solution). 6 and 7 instructions were run offline (a C version of the same algorithm, whose node counts matched the Python version one for one): neither has a solution, so 8 instructions is optimal. There are 6 eight-instruction solutions in all, all starting with `INC; ROT; SUB; DUP`; they differ only in the position of `DEC` and the target of the first `JZ` (jumping to 1, 2 or 8 all lead to starvation or halting).

### Lower bounds: search every short program

As with the first set, the search generates programs as it runs them: the instruction at a position is enumerated only when that position is first executed; as soon as the queue is wrong at halt, or exceeds 16 numbers, that branch is abandoned. The queue holds at most 16 numbers and each number has only 100 possible values, so the state space is finite. A program that doesn't halt must repeat some state `(position, queue)`, which Brent's cycle detection algorithm detects exactly.

For speed, the search uses only a small subset of each puzzle's tests. This doesn't affect the lower bound: a program that's correct on all the tests is certainly correct on a subset of them. Any solution found is then rechecked against all the tests.

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

The part rerun on this page (the verification code checks the node counts):

| Puzzle | n = 1 | n = 2 | n = 3 | n = 4 | n = 5 | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Part 1 | 12 | 168 | 2,032 | — | — | 1 solution found at 3 instructions |
| Part 2 | 12 | 168 | 2,032 | 24,300 | 300,460 | No solution with 5 or fewer |
| Part 3 | 12 | 182 | 1,856 | 20,376 | 238,680 | No solution with 5 or fewer |
| Part 4 | 12 | 154 | 1,792 | 20,754 | 249,280 | No solution with 5 or fewer |

Offline results (C version, same test subsets): Part 2 at 6 instructions visits 3,934,348 nodes and has only 1 solution; Part 3 at 6 instructions visits 3,032,766 nodes and has 3 solutions; Part 4 has no solution at 6 instructions (3,188,570 nodes) or 7 instructions (43,806,960 nodes), and 6 solutions at 8 instructions (649,658,178 nodes). Every solution found was rechecked against all the tests, and all are valid.

### How to split the time

- **0–10 minutes**: read the problem, write the tracer, run the summing example, and confirm you've understood "take from the head, put at the tail" and "starvation halts" correctly.
- **10–40 minutes**: Parts 1 and 2 are warm-ups. Once you think of the truncated subtraction identity, both reach optimal in a few minutes, and Part 2's result directly gives you a first version of Part 4.
- **40–80 minutes**: Part 3. Looking for an invariant like "length + sum" on paper first is far more effective than blindly trying instructions.
- **The remaining time**: push Part 4 down from 11 instructions. First work out what should be left in the queue when each path halts, then see which instructions can be reused and where the program can be made to starve.
- **Is writing the search worth it?** The optimal solutions to Parts 1–3 are all within 6 instructions, which the Python search can directly prove, and it can find the solutions too. Part 4's 8 instructions require visiting over six hundred million nodes, which isn't realistic in pure Python.

### Follow-ups

- **What if Part 4's input could be 99?** The reference solution is wrong when `a = b = 99`: `INC` still gives 99, `d = 0`, it's treated as less-than, and the output is 0. The best program we found that's correct for all of 0..99 has 10 instructions: `DUP; DUP; ROT; SUB; SUB; ADD; ZERO; JZ 9; ADD; INC`. It computes `|a − b|` first; when they're unequal, `ADD` has only 1 number to take, starves, and what's left is exactly 0. Offline exhaustive search shows this version has no solution with 8 instructions or fewer; we didn't finish searching whether 9 has a solution, so all we can say is "the best found so far is 10".
- **What about the smaller value `min(a, b)`?** `min(a, b) = a ∸ (a ∸ b)`, in 5 instructions: `DUP; ROT; ROT; SUB; SUB`. Exhaustive search shows no solution with 4 instructions or fewer. It takes two more than the larger value because `a` is used twice and has to be moved into the right position first.
- **What if Part 3's input could contain 0?** The first phase of the reference solution relies on "the first time a 0 appears at the head" to tell that the input is used up, so a 0 in the input would end it early. Offline exhaustive search shows this version has no solution with 8 instructions or fewer, and we didn't search further.
- **Why can the search decide "doesn't halt" exactly?** The state is `(position, queue)`, the queue holds at most 16 numbers, each in 0..99, so the number of states is finite, and a run that doesn't halt must repeat some state; Brent's algorithm is guaranteed to detect the repetition within a finite number of steps after it occurs. So the lower bounds don't depend on the 20,000-step limit.
- **Does a lower bound obtained by searching on only some of the tests count?** Yes. A program that's correct on all the tests is certainly correct on a subset of them; if there's no `n`-instruction solution on the subset, there can't be one on all the tests. Conversely, a solution found on the subset isn't necessarily correct, so every solution has to be rechecked against all the tests.

<details>
<summary>Verification code (runnable)</summary>

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
