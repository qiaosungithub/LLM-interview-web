# Official original: Anthropic's performance-engineering take-home (open-sourced older version)

## Problem

> **This is an original problem that Anthropic has officially made public.** On January 21, 2026, Anthropic
> published the take-home its performance engineering team had used, in the engineering blog post
> [Designing AI-resistant technical evaluations](https://www.anthropic.com/research/AI-resistant-technical-evaluations);
> the code is at https://github.com/anthropics/original_performance_takehome .
> It is no longer used for timed hiring. A local copy is at
> `/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome/`,
> identical word for word to the downloaded original; make your own copy before changing anything.

### Background

- Designed around November 2023 and in use from early 2024, it was taken by more than a thousand candidates.
  The machine simulates an accelerator similar to a TPU.
  The blog states explicitly that this take-home **allows the use of AI** (Anthropic's general rule is that
  take-homes don't allow AI by default unless stated otherwise).
- Version 1 had a 4-hour limit and started from fully serial code. The warm-up was multicore parallelism,
  after which candidates chose whether to do SIMD vectorization or VLIW instruction packing first.
  This version also had a planted bug that had to be debugged first.
- Around May 2025, a pre-release Claude Opus 4 did better within 4 hours than almost every candidate. That led
  to version 2: the time limit became 2 hours; the starting point moved to where Opus 4 started to struggle
  (the Readme says the starting code is 18532 cycles); the code was cleaner, new machine features were added,
  and multicore was removed.
- A pre-release Claude Opus 4.5 passed the bar in under an hour, then concluded it had hit a memory bandwidth
  bottleneck. The blog says most people would think the same, but that there are clever ways to get around this
  bottleneck by exploiting the problem's structure (without saying what they are). After being told the cycle
  count that was achievable, it found a way, and by the 2-hour mark it matched the best human result within
  2 hours (that candidate had also relied heavily on Claude 4). After that, the timed take-home switched to a
  different kind of problem (see "Relationship to other pages on this site" below).
- The public version, like version 1, starts from the slowest implementation (147734 cycles), but the
  instruction set and single-core design come from version 2, so its cycle counts are directly comparable with
  version 2.

Reference results given in the Readme. Note that these are all results of models doing the **2-hour version**,
starting from 18532 cycles:

| Cycles | Who, under what conditions |
|---:|---|
| 2164 | Claude Opus 4, running for many hours in a test-time compute harness |
| 1790 | Claude Opus 4.5, in a casual Claude Code session; roughly equal to the best human result within 2 hours |
| 1579 | Claude Opus 4.5, running for 2 hours in a harness |
| 1548 | Claude Sonnet 4.5, with far more than 2 hours of test-time compute |
| 1487 | Claude Opus 4.5, running for 11.5 hours in a harness |
| 1363 | Claude Opus 4.5, with an improved harness |
| ??? | The best human result ever is "substantially better"; no exact number has been published |

Public challenge: get below 1487 cycles (Claude's best result at the time of publication), and you can send
your code (ideally with a résumé) to performance-recruiting@anthropic.com. The Readme says this bar may change
after new models are released, and that there's no guarantee the Readme will be updated promptly.

The Readme also warns: every submission below 1300 received on the first day of publication was invalid; each
one was a language model modifying the tests.
One example: a model noticed that `problem.py` supports multiple cores, implemented multicore without getting
any speedup, and then "fixed" `N_CORES = 1`. Multicore is disabled on purpose.

### The machine

Everything is defined in `problem.py`. `tests/frozen_problem.py` is a frozen copy used for scoring, with the
same content as `problem.py`.

- **Word**: both memory and the scratch space consist of 32-bit unsigned integers. All arithmetic results are
  taken modulo $2^{32}$.
- **Memory**: a flat list of words. When the program ends, the result is read from memory.
- **Scratch space**: 1536 words per core (`SCRATCH_SIZE`), serving at once as registers, a constant area, and a
  manually managed cache.
  Almost every number in an instruction is a scratch address; the exceptions are the immediates of `const` and
  `add_imm`, the offset of `load_offset`, and jump targets.
- **Bundle**: a program is a sequence of instruction bundles. Each bundle is a dictionary
  `{engine name: [slot, ...]}`, and one bundle executes per cycle.
  Each engine can use at most this many slots per cycle:

| Engine | Slots per cycle | Purpose |
|---|---:|---|
| `alu` | 12 | Scalar arithmetic |
| `valu` | 6 | Vector arithmetic; a vector is 8 consecutive words in the scratch space (`VLEN = 8`) |
| `load` | 2 | Read from memory, or load a constant |
| `store` | 2 | Write to memory |
| `flow` | 1 | Select, jump, pause, etc. |
| `debug` | 64 | For debugging only; ignored when scoring |

- **Timing**: all slots in the same bundle read the state at the start of the cycle; writes to the scratch space
  and to memory only take effect at the end of the cycle.
  So slots in the same cycle don't affect each other, and results are available from the next cycle on. Apart
  from that there is **no latency of any kind**, loads included.
- **Cycle counting**: every executed bundle that contains a non-`debug` engine adds 1 to the cycle count. The
  program ends when `pc` runs off the end or a `halt` executes.
- **Single core**: `N_CORES = 1`. The simulator still contains multicore code, but this version deliberately
  uses only one core.

The instructions are as follows (`a`, `b`, `c`, `dest`, `cond` and `addr` are all scratch addresses). The `op`
of `alu` can be `+`, `-`, `*`, `//`, `cdiv`, `^`, `&`, bitwise OR, `<<`, `>>`, `%`, `<`, `==` (the last two
produce 0 or 1).

| Engine | Form | Meaning |
|---|---|---|
| `alu` | `(op, dest, a, b)` | `dest = a op b` |
| `valu` | `(op, dest, a, b)` | Apply the same `op` to each of the 8 positions |
| `valu` | `("vbroadcast", dest, a)` | Copy scalar `a` into all 8 positions |
| `valu` | `("multiply_add", dest, a, b, c)` | Per position, `dest = a * b + c` |
| `load` | `("load", dest, addr)` | `dest = mem[scratch[addr]]` |
| `load` | `("load_offset", dest, addr, k)` | Word `dest + k` `= mem[scratch[addr + k]]` |
| `load` | `("vload", dest, addr)` | Read 8 **consecutive** words starting at `mem[scratch[addr]]` |
| `load` | `("const", dest, v)` | `dest = v` (an immediate; also takes a load slot) |
| `store` | `("store", addr, a)` | `mem[scratch[addr]] = a` |
| `store` | `("vstore", addr, a)` | Write 8 words to consecutive locations starting at `mem[scratch[addr]]` |
| `flow` | `("select", dest, cond, a, b)` | `dest = a if cond != 0 else b` |
| `flow` | `("vselect", dest, cond, a, b)` | `select` per position |
| `flow` | `("add_imm", dest, a, imm)` | `dest = a + imm` |
| `flow` | `halt`, `pause`, `jump`, `cond_jump`, `cond_jump_rel`, `jump_indirect`, `coreid`, `trace_write` | Halt, pause, jumps, etc. |
| `debug` | `compare`, `vcompare`, `comment` | Compare against intermediate values of the reference implementation; disabled when scoring |

Pay special attention: **there is no gather load instruction**. `vload` can only read 8 consecutive words; to
read 8 non-adjacent addresses you need 8 `load`s, and `load` has only 2 slots per cycle.

### The kernel

Memory layout (`build_mem_image`):

| Location | Contents |
|---|---|
| `mem[0..6]` | `rounds, n_nodes, batch_size, forest_height, forest_values_p, inp_indices_p, inp_values_p` |
| Starting at `forest_values_p` (always 7) | The tree's `n_nodes` node values |
| Starting at `inp_indices_p` | `batch_size` current node indices, initially all 0 |
| Starting at `inp_values_p` | `batch_size` current values |

The total memory length is exactly `7 + n_nodes + 2 * batch_size` words, with no spare room after it
(`build_mem_image` originally leaves some blank space, but the final slice assignment truncates it).

The tree is a perfect binary tree of height `forest_height`, with $2^{h+1}-1$ nodes; the two children of node
`i` are `2i+1` and `2i+2`.
Node values and initial values are random integers in $[0, 2^{30})$. What needs to be computed (the meaning of
`reference_kernel2`):

```py
for r in range(rounds):
    for i in range(batch_size):
        idx = indices[i]
        val = myhash(values[i] ^ tree[idx])
        idx = 2 * idx + (1 if val % 2 == 0 else 2)   # even: left child, odd: right child
        idx = 0 if idx >= n_nodes else idx           # past a leaf: back to the root
        values[i] = val
        indices[i] = idx

HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]

def myhash(a):
    for op1, c1, op2, op3, c3 in HASH_STAGES:
        a = ((a op1 c1) op2 (a op3 c3)) % 2**32      # each inner result is also taken mod 2**32
    return a
```

What you write is `KernelBuilder.build_kernel(forest_height, n_nodes, batch_size, rounds)` in
`perf_takehome.py`: it generates a program and puts it in `self.instrs`. The program runs on the memory above,
and when it ends, the section starting at `inp_values_p` must match the reference implementation exactly.
These four parameters are ordinary Python integers at program generation time, so the program can be generated
for the specific sizes.
The starting code is written in scalar form with one slot per bundle, for a total of 147734 cycles.

### Scoring and rules

- **Scoring command**: `python tests/submission_tests.py`. The sizes are fixed at
  `forest_height=10, rounds=16, batch_size=256` (2047 nodes).
  The random numbers are not seeded.
- **Correctness**: 8 sets of inputs are generated at random and each is run once; only the final `values`
  section is compared. Writing back `indices` is optional.
- **Speed**: one more run reads `machine.cycle`; if the result is wrong, it counts as 2 × 147734. That is then
  compared against 8 thresholds, and must be **strictly less than** each:
  147734, 18532, 2164, 1790, 1579, 1548, 1487, 1363. The note in the test file says you don't need to pass all
  of them to pass the interview, that the number passed doesn't map linearly to skill level, and that the
  thresholds exist only to show a score on CodeSignal.
- **Rules**: your code goes in `perf_takehome.py` (mainly `build_kernel`), and **you may not change anything in
  `tests/`**.
  Multicore is disabled on purpose; don't touch `N_CORES`. Your result is the cycle count printed by
  `python tests/submission_tests.py`.
- **License**: the top of the starting code says you may modify and use it, but please don't publish or
  redistribute your solution, so that others can't find the answer by searching.
  So this page's reference solution is kept locally, for study only.

### Working on it locally

Requires Python 3.10 or newer (the code uses `match` statements); no third-party libraries are needed.

```bash
# 1. copy the pristine folder to a working folder; never edit the copy under materials/
cp -r /kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome ~/perf-takehome
cd ~/perf-takehome

# 2. while developing: fixed seed 123; also checks that `pause` lines up with the reference's `yield`
python perf_takehome.py Tests.test_kernel_cycles

# 3. the real score: which thresholds you pass
python tests/submission_tests.py

# 4. tests/ must be unchanged (this copy has no git history, so `git diff origin/main tests/` does not apply)
diff -r -x __pycache__ tests /kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome/tests && echo "tests/ unchanged"
```

The repository comes with debugging tools: `python perf_takehome.py Tests.test_kernel_trace` writes
`trace.json` (one record per slot, in Chrome's Trace Event format, viewable in Perfetto);
`python watch_trace.py` serves a page on local port 8000 that hot-reloads this file in Chrome; you can also drag
`trace.json` straight into https://ui.perfetto.dev/ .
The starting code also demonstrates using the `debug` engine's `compare` to check values in the scratch space
one by one against the reference implementation's intermediate results. (We haven't actually opened and used
either of these.)

### Suggested order of work

1. Understand `problem.py` and the starting code well enough to explain where 147734 comes from.
2. First get past 18532 (the starting point of version 2), then past 2164 and 1790. According to the Readme, the
   best human result within 2 hours is about 1790 (and that started from 18532).
3. If you want a challenge, aim for below 1487. If you enter the public challenge, what you submit should be
   your own code.

### Relationship to other pages on this site

- [Optimizing a compute kernel on a simulated vector machine](#/anthropic/take-home/performance-kernel-optimization):
  another exercise of the same kind on this site, with a separately designed machine and kernel. That machine
  has registers, each bundle holds far fewer instructions (2 scalar, 1 vector, 1 load, 1 store),
  loads have a 4-cycle latency, and the kernel is a floating-point multiply-add sum. Here there is no latency,
  the scratch space replaces registers, the kernel is a 32-bit integer hash plus tree reads where "the index
  depends on the data", and the thresholds are fixed cycle counts. Ideas such as packing, vectorization and
  finding the bottleneck carry over between the two.
- [The timed take-home in current use](#/anthropic/take-home/constrained-machine-puzzles-a): the blog says that
  after Opus 4.5, the timed take-home switched to a different kind of problem: solving a set of puzzles on a
  tiny, heavily constrained instruction set, optimizing for instruction count, with visualization and debugging
  tools deliberately not provided.
  It is not the same kind of task as this one.

## Reference solution

<details>
<summary>Show the reference solution</summary>

### Overview

We optimized our own copy in 9 stages. Each stage was measured with `tests/submission_tests.py` (sizes
10 / 16 / 256, `tests/` unchanged; in every stage all 9 tests printed the same cycle count, and the correctness
tests passed every time).

| Stage | What changed | Cycles | vs. start | Speed thresholds passed |
|---|---|---:|---:|---:|
| Start | Starting code | 147734 | 1.00× | 0 / 8 |
| 1 | Scalar; remove redundant operations | 97040 | 1.52× | 1 / 8 |
| 2 | Same operations, packed into the same cycle by dependency | 7690 | 19.21× | 2 / 8 |
| 3 | SIMD: 32 blocks of 8 elements each | 2084 | 70.89× | 3 / 8 |
| 4 | Tree levels 0–2 no longer read from memory | 1548 | 95.44× | 5 / 8 |
| 5 | Hand operations to the scheduler block by block | 1355 | 109.03× | 8 / 8 |
| 6 | Fewer vector operations per round | 1329 | 111.16× | 8 / 8 |
| 7 | Level 3 doesn't read memory either; node selection goes to the flow engine | 1107 | 133.45× | 8 / 8 |
| 8 | In the second pass, level 4 doesn't read memory for 20 of the blocks | 1064 | 138.85× | 8 / 8 |
| 9 | The XOR after a gather goes to the scalar ALU | 1040 | 142.05× | 8 / 8 |

Stage 4 is 1548, exactly equal to the 1548 threshold, so it doesn't count as passing (strictly less than is
required).

The full code is at `/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome-solution/perf_takehome_staged.py`.
It can directly replace `perf_takehome.py`; the environment variable `PERF_STAGE` selects the stage (default 9):

```bash
cp /kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome-solution/perf_takehome_staged.py ~/perf-takehome/perf_takehome.py
cd ~/perf-takehome
python tests/submission_tests.py                 # stage 9: 1040
PERF_STAGE=3 python tests/submission_tests.py    # stage 3: 2084
```

The code blocks below are all excerpted from this file (abridged, and simplified in a few places so they can be
shown on their own); they are for display only and are not run on this page.

### Method: compute each engine's lower bound at every step

At every step, first count how many operations the program has on each engine. The operation count divided by
the engine's slots per cycle is a lower bound on the cycle count.
For example, stage 9 has 1949 `load` operations, at most 2 per cycle, so it needs at least 975 cycles.

The largest of these lower bounds tells you which engine is the bottleneck; the gap between the measured value
and it tells you how much scheduling (placing operations into cycles) loses.
The next step is then chosen from three things: reduce the operations on the bottleneck engine; move operations
to idle engines; improve the scheduling.

| Stage | load | valu | alu | flow | Measured |
|---|---:|---:|---:|---:|---:|
| 1 | 2312 | — | 7680 | — | 97040 (one slot per cycle, equal to the total operation count) |
| 2 | 2312 | — | 7680 | — | 7690 |
| 3 | 2073 | 1336 | 3 | — | 2084 |
| 4 | 1306 | 1348 | 3 | — | 1548 |
| 5 | 1306 | 1348 | 3 | — | 1355 |
| 6 | 1307 | 1119 | 473 | — | 1329 |
| 7 | 1053 | 1067 | 431 | 704 | 1107 |
| 8 | 975 | 1053 | 432 | 1004 | 1064 |
| 9 | 975 | 1014 | 590 | 1004 | 1040 |

The table shows each engine's lower bound (operation count ÷ slot count, rounded up). `store` is small in every
stage and isn't listed.

### Infrastructure: virtual registers, the scheduler, scratch allocation

From stage 2 on, every program is generated by the same process:

1. **Straight-line code**: fully unroll all rounds and all elements for the given sizes, with no loops or jumps.
   That way the cycle count equals the number of bundles and doesn't depend on the input data.
2. **Virtual registers**: every computed value goes into a new virtual register, and each word is written only
   once.
   Then the only constraint left between operations is "a read must come after the write", and the scheduler
   can reorder freely.
3. **List scheduling**: process the operations in the given order, putting each into the earliest cycle that is
   no earlier than "the cycle after all its dependencies have been computed" and in which its engine still has a
   free slot. Then reverse the whole dependency graph and schedule it again (each operation as late as possible),
   then schedule forward again in the order of the reversed result,
   going back and forth like this 3 times and keeping the best result.
4. **Scratch allocation**: after scheduling, assign scratch addresses according to each virtual register's live
   range (from its first write to its last read).
   Because writes take effect only at the end of a cycle, a new value can take over an old value's address in the
   same cycle as the old value's last read.

```py
def place(order, deps, eng, reserved=()):
    """Serial list scheduling: each op goes into the earliest cycle that is later than all of
    deps[op] and still has a free slot on its engine."""
    slots = {name: Slots(cap) for name, cap in SLOT_LIMITS.items()}
    for name, c in reserved:
        slots[name].take(c)
    cyc = [-1] * len(eng)
    for i in order:
        lb = 0
        for p in deps[i]:
            if cyc[p] >= lb:
                lb = cyc[p] + 1
        c = slots[eng[i]].find(lb)          # first cycle >= lb with a free slot on this engine
        slots[eng[i]].take(c)
        cyc[i] = c
    return cyc


def schedule(pr, pack=True, passes=3):
    n = len(pr.eng)
    if not pack:
        return list(range(n))                     # stage 1: one slot per cycle, in emission order
    reserved = [("flow", 0)]                      # cycle 0 keeps its flow slot for the `pause`
    cyc = place(range(n), pr.preds, pr.eng, reserved)
    succs = [[] for _ in range(n)]
    for i, ps in enumerate(pr.preds):
        for p in ps:
            succs[p].append(i)
    best = cyc
    for _ in range(passes):
        # everything as late as possible on the reversed graph, then forward again in that order
        back = place(sorted(range(n), key=lambda i: (-cyc[i], -i)), succs, pr.eng)
        cyc = place(sorted(range(n), key=lambda i: (-back[i], i)), pr.preds, pr.eng, reserved)
        if max(cyc) < max(best):
            best = cyc
    return best
```

The `Prog` class records the operations: for each one it notes the engine, the instruction template (with
virtual registers as placeholders for scratch addresses), which registers it reads, which registers it writes,
and which operations it depends on. Constants and broadcasts are deduplicated: the same constant is loaded only
once, and the same scalar is broadcast only once.
`assemble` stitches the schedule into bundles and puts a `pause` in cycle 0 to match the reference
implementation's first `yield` (only the local development tests need it).

How much do the back-and-forth passes help? Below are measurements from our own small script: only the number
of passes changes, everything else stays the same, with one run on random input and the result checked.

| Stage | 0 passes | 1 pass | 3 passes (adopted) | 6 passes |
|---|---:|---:|---:|---:|
| 2 | 7944 | 7691 | 7690 | 7690 |
| 5 | 1473 | 1356 | 1355 | 1355 |
| 7 | 1178 | 1113 | 1107 | 1107 |
| 9 | 1140 | 1050 | 1040 | 1040 |

### Stage 1 — Remove redundant operations: 97040

The starting code uses 36 slots per (element, round): reading the index, the value and the node value (each
needing one addition to compute the address and one `load`), 1 XOR,
18 for the hash, 5 to compute the direction (`%`, `==`, `select`, `*`, `+`), 2 to check for going out of bounds,
and 4 to write back the index and the value.
36 × 4096 = 147456; add reading the memory header (14), loading constants (262) and two `pause`s (2), 278 cycles
in all, and you get 147734.

Two observations let most of the operations go:

- **Every element starts from the root**, so in round `r` all elements are on the same level, at depth
  `r % (forest_height + 1)`.
  Going out of bounds only happens after the leaf level, and then all elements return to the root together. So
  there's no need to check bounds per element.
- **Values and node addresses stay in the scratch space** throughout, with no memory reads and writes every
  round; indices don't need to be written back either (the tests don't check them).
  Maintain the node address `P + idx` directly (`P = forest_values_p = 7`); the next level's address is
  $P + (2\,\mathrm{idx} + 1 + \mathrm{bit}) = 2(P + \mathrm{idx}) + (1 - P) + \mathrm{bit}$, where bit is `val & 1`.

That leaves just 1 `load`, 1 XOR, 18 hash operations and 4 address operations per (element, round), 24
operations in all.

```py
def gen_scalar(pr, height, n_nodes, batch, rounds):
    P = 7                              # forest_values_p: the tree follows the 7-word header
    VP = P + n_nodes + batch           # inp_values_p
    one, two, fix = pr.const(1), pr.const(2), pr.const(1 - P)
    ptr = [pr.const(VP + i) for i in range(batch)]
    val = [pr.load(ptr[i]) for i in range(batch)]
    adr = [pr.const(P)] * batch        # address of the current node, P + idx
    for r in range(rounds):
        depth = r % (height + 1)       # every item starts at the root, so all items share one depth
        for i in range(batch):
            a = pr.alu("^", val[i], pr.load(adr[i]))
            for op1, c1, op2, op3, c3 in HASH_STAGES:
                a = pr.alu(op2, pr.alu(op1, a, pr.const(c1)), pr.alu(op3, a, pr.const(c3)))
            val[i] = a
            if r == rounds - 1:
                pr.store(ptr[i], a)
            elif depth == height:
                adr[i] = pr.const(P)   # all items are on leaves here, so all of them wrap to the root
            else:
                # P + (2*idx + 1 + bit)  =  2*(P + idx) + (1 - P) + bit
                t = pr.alu("+", pr.alu("*", adr[i], two), fix)
                adr[i] = pr.alu("+", t, pr.alu("&", a, one))
```

This stage still puts only one slot in each cycle, so the cycle count equals the total operation count: 97040.
It's only 1.52 times faster, because the real waste is that each cycle does only one thing.

### Stage 2 — Put independent operations in the same cycle: 7690

The operations are exactly the same as in stage 1, just handed to the scheduler above to pack. The 256 elements
are independent of each other, so there's plenty of parallelism, and the bottleneck becomes `alu`:
92160 scalar operations ÷ 12 = 7680. Measured: 7690, almost right at the lower bound.

At this point the scratch space uses 1294 words (out of 1536): 256 values, 256 addresses, plus many
intermediate values live at the same time. The scalar approach has little room left.

### Stage 3 — Vectorization: 2084

Split the 256 elements into 32 blocks of 8 and process a block at a time with vector operations. At the start,
each block's values are read with one `vload`, and at the end they're written back with one `vstore`.

Steps 0, 2 and 4 of the hash have the form `(a + c) + (a << k)`, which modulo $2^{32}$ equals
$a \cdot (1 + 2^k) + c$, exactly one `multiply_add`.
So one hash takes only 12 vector operations:

```py
def vhash(x):                                                # stage 3 version
    a = pr.madd(x, K(1 + (1 << 12)), K(c0))                  # (a + c0) + (a << 12)
    a = pr.valu("^", pr.valu("^", a, K(c1)), pr.valu(">>", a, K(19)))
    a = pr.madd(a, K(33), K(c2))                             # (a + c2) + (a << 5)
    a = pr.valu("^", pr.valu("+", a, K(c3)), pr.valu("<<", a, K(9)))
    a = pr.madd(a, K(9), K(c4))                              # (a + c4) + (a << 3)
    return pr.valu("^", pr.valu("^", a, K(c5)), pr.valu(">>", a, K(16)))
```

(`K(v)` is "the constant vector with value `v`": one `const` plus one `vbroadcast`, done only once in the whole
program.)

The node values are the hard part: the 8 elements are at different positions in the tree, and without a gather
you can only read them into the 8 positions of a vector register with 8 separate `load`s.
The address vector is updated with `multiply_add`: first compute `t2 = adr * 2 + (1 - P)`, then add `val & 1`.

```py
node = pr.reg(VLEN)
for lane in range(VLEN):
    pr.load_lane(node, lane, (st.adr, lane))       # one scalar load per lane
x = pr.valu("^", st.val, node)
t2 = pr.madd(st.adr, K(2), K(1 - P))               # next address = t2 + (val & 1)
```

Now the bottleneck becomes `load`: 4096 gathers at 2 per cycle means at least 2048 cycles. Measured: 2084.
This is the "memory bandwidth bottleneck" the blog talks about: no amount of saving on vector operations helps,
because the loads can't keep up.

### Stage 4 — Don't read the top few levels of the tree from memory: 1548

The key observation is again that "in a given round all elements are on the same level". Level 0 has only 1
node, level 1 has 2, and level 2 has 4.
These 7 node values can be read in one `vload` (nodes 0–7) at the start, and after that you only need to
"compute" which value to use from the directions already taken.

The direction is the lowest bit of each round's hash result. Let the values of the two level-1 nodes be
$v_1, v_2$ and the first direction be $b_0$; then the node value is
$v_1 + b_0 (v_2 - v_1)$, a single `multiply_add`. Level 2 has two directions, and the node value is a
multilinear polynomial in $b_0, b_1$:
three `multiply_add`s. Coefficients like the difference $v_2 - v_1$ depend only on the tree, so computing them
once on the scalar ALU and broadcasting them once is enough.

```py
def pick(depth, bits, pre):
    """bits[j] = (vector q, polarity p): the direction taken at depth j is q ^ p.
    The position inside the level is the directions read as a binary number, first one most significant."""
    leaves = [leaf((1 << depth) - 1 + o, pre) for o in range(1 << depth)]   # scalar tree values
    if depth == 0:
        return pr.bcast(leaves[0])

    def lin(vals, k):              # multilinear polynomial in bits[0..k-1] with scalar coefficients
        if k == 0:
            return pr.bcast(vals[0])
        q, p = bits[k - 1]         # peel the latest direction: even / odd entries
        base, other = (vals[1::2], vals[0::2]) if p else (vals[0::2], vals[1::2])
        diff = [pr.alu("-", o, b, cache=True) for o, b in zip(other, base)]
        return pr.madd(q, lin(diff, k - 1), lin(base, k - 1))

    return lin(leaves, depth)
```

(The polarity `p` is only ever 1 from stage 6 on; here, treat it as 0.)

At the first level that has to be read from memory (level 3), convert the recorded direction bits into an
address vector: the constant $P + 2^d - 1$ plus each direction bit times its weight
(`addr_from_bits`, using `+` and `multiply_add`).

Each pass down the tree (rounds 0–10 are the first pass, rounds 11–15 the second) has 3 rounds without a
gather, 6 of the 16 rounds in all, bringing gathers down to 2560:
the `load` lower bound is 1306 and the `valu` lower bound is 1348. Yet the measured value is 1548, far from the
lower bound.

Looking at the scheduled program shows the problem: there are no gathers at all in the first 153 cycles; and
between the first and last gather, there are still 104 cycles where the `load` slots aren't full.
The problem lies in the order in which operations are handed to the scheduler.

### Stage 5 — Hand operations to the scheduler block by block: 1355

The operations are exactly the same as in stage 4; only the order in which they're handed to the scheduler
changes: from "by round" (round 0's 32 blocks, round 1's 32 blocks, ..., i.e. the reference implementation's
loop order)
to "by block" (block 0's 16 rounds, block 1's 16 rounds, ...).

```py
steps = [(v, r) for v in range(n_blocks) for r in range(rounds)]   # block by block
if cfg["order"] == "round":
    steps.sort(key=lambda vr: (vr[1], vr[0]))                      # stages 3-4: round by round
for v, r in steps:
    step(v, r)
```

List scheduling is greedy: it packs each operation, in order, into the earliest cycle it fits. In round order,
the first things processed are the pure vector operations of the first 3 rounds of every block,
which fill up the early `valu` slots first, while the operations that would "unlock" the first gathers come
later. In block order, block 0 quickly reaches level 3 and starts gathering,
and later blocks naturally fill the gaps, which amounts to automatic software pipelining.

Result: gathering starts after the first 39 cycles; measured 1355, passing all 8 thresholds. The vectors
allocated in the scratch space also drop from 121 to 77.
Switching stage 3 to block order, on the other hand, changes nothing (we measured it: still 2084): it starts
gathering from cycle 2, and between then and the last gather there's only 1 cycle where `load` isn't full.

### Stage 6 — Fewer vector operations per round: 1329

Three rewrites:

1. **Merge hash steps 2 and 3.** Step 2 gives $b = 33a + c_2$; step 3 needs `(b + c3) ^ (b << 9)`.
   Both sides are "$a$ times a constant plus a constant": $33a + (c_2 + c_3)$ and
   $(33 \cdot 2^9) a + c_2 \cdot 2^9$. So two `multiply_add`s plus one XOR,
   3 operations instead of the original 4.
2. **Defer the final `^ c5`.** Step 5 is `(a ^ c5) ^ (a >> 16)`. If the next level's node value has been
   pre-read, skip the `^ c5` for now,
   giving `hash ^ c5`; also XOR the pre-read node values with `c5` in advance (scalar, once each), and in the next
   round
   `(hash ^ c5) ^ (node ^ c5) = hash ^ node`, which cancels out exactly. The cost is that the direction bit is
   flipped (`c5 = 0xB55A4F09` is odd),
   so every direction bit carries a polarity `p`, and selecting nodes and computing addresses interpret it as
   `q ^ p`.
3. **Move bit operations to the scalar ALU.** `val & 1` and "add the direction bit to the address" become one
   scalar operation per position (8 `alu` instead of 1 `valu`).
   `alu` has 12 slots per cycle and was almost idle until now.

```py
def vhash(x, skip_c5):
    a = pr.madd(x, K(1 + (1 << 12)), K(c0))
    a = pr.valu("^", pr.valu("^", a, K(c1)), pr.valu(">>", a, K(s1)))
    # stage 2 gives b = 33a + c2; stage 3 wants (b + c3) ^ (b << 9): both are a*const + const
    a = pr.valu("^", pr.madd(a, K(33), K(c2 + c3)), pr.madd(a, K(33 << s3), K(c2 << s3)))
    a = pr.madd(a, K(9), K(c4))
    if skip_c5:
        return pr.valu("^", a, pr.valu(">>", a, K(s5)))              # = hash ^ c5
    return pr.valu("^", pr.valu("^", a, K(c5)), pr.valu(">>", a, K(s5)))

def low_bit(val):                  # val & 1, one scalar ALU op per lane
    q = pr.reg(VLEN)
    for lane in range(VLEN):
        pr.alu_lane("&", q, lane, (val, lane), pr.const(1))
    return q

def leaf(i, pre):                  # tree value i as a scalar operand, pre-XORed with c5 if needed
    k = i // VLEN
    if k not in tree_blocks:
        tree_blocks[k] = pr.vload(pr.const(P + VLEN * k))
    s = (tree_blocks[k], i % VLEN)
    return pr.alu("^", s, pr.const(C5), cache=True) if pre else s
```

The `valu` lower bound drops from 1348 to 1119, the bottleneck goes back to `load` (1307), and the measured
value is 1329. This step on its own gains little: removing any one of the three rewrites
still leaves stage 6 at 1329–1332 (measured with our small script). Their effect only shows up later: removing
each of them from the stage 9 configuration
takes it from 1040 to 1107 (without merging steps 2 and 3), 1065 (without deferring `^ c5`), and 1130 (without
moving bit operations to the ALU).

### Stage 7 — Level 3 doesn't read memory either; node selection goes to the flow engine: 1107

Level 3 has 8 nodes; with a multilinear polynomial that takes 7 `multiply_add`s, and `valu` would become the
bottleneck (we measured it: doing that gives 1193, with a `valu` lower bound of 1185).

But the `flow` engine has been idle all along. It has only 1 slot per cycle, but one `vselect` makes a choice
for 8 positions at once.
Use a selection tree: first choose between the first half and the second half by the first direction, then
choose by the second direction, and so on. Level $d$ takes $2^d - 1$ `vselect`s.
Levels 1 and 2 are switched to this approach as well, to free up `valu`.

```py
cur = [pr.bcast(s) for s in leaves]
for q, p in bits:                                            # earliest direction first: halves
    half = len(cur) // 2
    lo, hi = cur[:half], cur[half:]
    cur = [pr.vselect(q, lo[i], hi[i]) if p else pr.vselect(q, hi[i], lo[i]) for i in range(half)]
return cur[0]
```

Each pass down the tree takes 1 + 3 + 7 = 11 `vselect`s; two passes and 32 blocks make 704 in all. Gathers are
left in only 8 of the 16 rounds, and the `load` lower bound is 1053. Measured: 1107.

Why not give level 4 to every block as well? That would add another 15 × 2 × 32 `vselect`s, making the `flow`
lower bound 1664; measured 1687, which is actually slower.

### Stage 8 — Use up the remaining flow slots: 1064

In stage 7, `flow` uses only 704 cycles, leaving about 400. Level 4 selection takes 15 `vselect`s per block, so
20 blocks is exactly 300, bringing `flow` to 1004 in all.
We use it in the second pass: in round 15 (level 4, which is also the last round), 20 blocks switch to
selection, saving 160 gathers and bringing the `load` lower bound down to 975.

```py
def preloaded(v, r):               # number of levels served from preloaded values, for block v in round r
    trip = 0 if r <= height else 1
    return min(height, top + (1 if v in deep[trip] else 0))

# deep = (0, 20): no extra blocks on the first trip, 20 blocks (spread evenly) on the second
deep = [{(i * n_blocks) // n for i in range(n)} for n in cfg.get("deep", (0, 0))]
```

Measured: 1064. We also tried other splits with our small script: all 32 blocks gives 1207 (`flow` lower bound
1184), 20 blocks in the first pass gives 1091,
and no level 4 selection at all (i.e. stage 7) gives 1107. Why the second pass works better, we didn't
investigate in detail.

### Stage 9 — The XOR after a gather goes to the scalar ALU: 1040

What a gather reads is a set of individual scalars to begin with. Rather than assembling them into a vector
first and then doing one vector XOR, XOR each one with the value at the corresponding position on the ALU as
soon as it's read.
Each (block, round) that needs a gather saves 1 `valu` and adds 8 `alu`: `valu` operations drop from 6318 to
6082, and `alu` rises from 5182 to 7070.

```py
x = pr.reg(VLEN)
for lane in range(VLEN):
    pr.alu_lane("^", x, lane, (st.val, lane), pr.load((st.adr, lane)))
```

Measured: 1040.

### Why we stopped at 1040

Now the `load` (975), `valu` (1014) and `flow` (1004) lower bounds are all close to the measured value. Looking
at the scheduled program:

- There are no gathers in the first 47 cycles. The first gather has to wait for block 0 to finish 4 rounds
  (levels 0–3 don't read memory), and these 4 rounds run one after another and can't be parallelized.
- From the first gather to the last there are 983 cycles, of which only 39 have `load` not full.
- After the last gather, it takes another 10 cycles to finish the final hash and write back.

So 1040 = 47 (start) + 983 (during gathers) + 10 (end). Going lower means either shortening the start or
reducing gathers further,
but reducing gathers needs more selection operations, and `flow` and `valu` are both nearly full.

The blog says Opus 4.5 found "clever ways to get around the bandwidth bottleneck by exploiting the problem's
structure", but doesn't say what they are. Stages 4, 7 and 8 are one way of exploiting the problem's structure
to get around gathers,
not necessarily the one the blog refers to.

### If indices must be written back

The tests don't check `indices`, so none of the versions above write them back. Setting `PERF_STORE_IDX=1`
(available in stages 3–9) also computes the indices in the last round and writes them back:
stage 9 then measures 1055 (`tests/submission_tests.py` all pass; we separately checked the written-back
indices on 10 sizes).

### Directions not taken (none implemented or measured)

- **Shorten the first 47 cycles**: for example, have the first few blocks pre-read one level fewer in the first
  pass and start gathering earlier, while later blocks pre-read more; or adjust scheduling priorities
  so block 0's critical path goes first.
- **Keep balancing the engines**: the `alu` lower bound is only 590, so it could take over some vector
  operations (1 `valu` is worth 8 `alu`);
  level 4 selection could also be done partly with `vselect` and partly with `multiply_add`, sharing the load
  between `flow` and `valu`.
- **A better scheduler**: critical-path priorities, random restarts, local search. The table above shows that
  scheduling quality matters a lot (stage 9: 1140 with 0 passes, 1040 with 3).
- **Use fewer load slots for constants**: about 25 `const`s currently take `load` slots; the rest could be
  derived on the ALU from a few constants.
- **Don't hard-code the layout**: currently `forest_values_p = 7` and the specific sizes are used directly.
  Reading them from the memory header would cost a few more cycles but be more general.

### Follow-ups

**Why fully unroll instead of using loops and jumps?**
The simulator doesn't limit program length; after unrolling, the cycle count is the number of bundles, and the
scheduler can reorder freely across rounds and blocks.
If program length were limited, you'd need loops with jumps (each jump takes a `flow` slot), plus manual
software pipelining.

**Is hard-coding `forest_values_p = 7` and the input sizes gaming the system?**
The rules only require not changing `tests/`. `build_kernel` receives the specific sizes, and the memory layout
is determined by the frozen `build_mem_image`, so this is allowed.
In an interview it's best to state this assumption up front, and to say what it would cost to read them from
the memory header instead. Our generator is also correct for other sizes:
each stage was run on 2 sets of random inputs at each of 10 sizes (height 1–10, rounds 1–23, batch size 8–256),
and the results all matched the reference implementation.
(The vector version requires the batch size to be a multiple of 8.)

**How do you show the result is trustworthy and the tests weren't changed?**
Compare `tests/` with the original (`diff -r`, or `git diff`), confirm that `N_CORES` and the like in
`problem.py` haven't been touched either, and then look only at
the cycle count printed by `python tests/submission_tests.py`. The machine used for scoring comes from the
frozen copy `tests/frozen_problem.py`, so changing the machine in `problem.py` doesn't change the score.

**Is the scratch space big enough?**
Stage 9 uses 56 words plus 149 vectors, 1248 words in all (out of 1536). If stage 9 still handed operations to
the scheduler in "by round" order,
too many vectors would be live at the same time: it would need 66 words plus 201 vectors, over the limit, and
the program couldn't even be generated. The order affects both speed and scratch usage.

**If loads had latency (as in the other exercise on this site), what would need to change?**
In the scheduler, "the cycle after the dependency" would become "some number of cycles after the dependency",
i.e. edges weighted by latency. Gathers would need to be issued earlier,
more blocks would be in flight at once, and pressure on the scratch space would be higher; the block order and
the back-and-forth passes would still apply.

**In the 2-hour timed version, how far counts as good?**
According to the Readme, the best human result within 2 hours is about 1790, and that started from 18532.
Getting packing, vectorization and not reading the top few levels from memory right
is already very strong; the later steps require repeatedly studying lower bounds and schedules, and are more
like untimed polishing.

<details>
<summary>Verification code (runnable)</summary>

The code below doesn't depend on the repository; it only checks the claims on this page that can be verified
in code: the hash rewrites, the "all elements on the same level" observation, the stage 1 address formula,
the two ways of selecting nodes, and the operation counts used in the text. The cycle counts themselves can
only be measured in the repository; see the commands above.

```python
import random

random.seed(2026)
M = 2**32
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]
OPS = {"+": lambda x, y: x + y, "^": lambda x, y: x ^ y,
       "<<": lambda x, y: x << y, ">>": lambda x, y: x >> y}
C = [h[1] for h in HASH_STAGES]
C5 = C[5]


def stage(a, k):
    op1, c1, op2, op3, c3 = HASH_STAGES[k]
    return OPS[op2](OPS[op1](a, c1) % M, OPS[op3](a, c3) % M) % M


def myhash(a):
    for k in range(6):
        a = stage(a, k)
    return a


def madd(a, b, c):                       # one lane of multiply_add
    return (a * b + c) % M


def vhash_lane(x, fuse23=True, skip_c5=False):
    """One lane of the reference solution's vector hash, using only valu operations."""
    a = madd(x, 1 + (1 << 12), C[0])
    a = (a ^ C[1]) ^ (a >> 19)
    if fuse23:
        a = madd(a, 33, C[2] + C[3]) ^ madd(a, 33 << 9, C[2] << 9)
    else:
        a = madd(a, 33, C[2])
        a = ((a + C[3]) % M) ^ ((a << 9) % M)
    a = madd(a, 9, C[4])
    return a ^ (a >> 16) if skip_c5 else (a ^ C5) ^ (a >> 16)


xs = [random.randrange(M) for _ in range(20000)] + [0, 1, M - 1, 2**31]

# Hash stages 0, 2, 4 are (a + c) + (a << k) = a * (1 + 2^k) + c: one multiply_add each.
for k in (0, 2, 4):
    shift = HASH_STAGES[k][4]
    assert all(stage(a, k) == madd(a, 1 + (1 << shift), C[k]) for a in xs)

# The vector hash (12 valu ops; 11 with stages 2 and 3 fused) equals myhash.
assert all(vhash_lane(x, fuse23=False) == myhash(x) for x in xs)
assert all(vhash_lane(x) == myhash(x) for x in xs)

# Deferring the final "^ C5": the result is hash ^ C5, and pre-XORing node values with C5 cancels it.
assert C5 % 2 == 1
for x in xs[:5000]:
    h, node = myhash(x), random.randrange(2**30)
    assert vhash_lane(x, skip_c5=True) == h ^ C5
    assert (h ^ C5) ^ (node ^ C5) == h ^ node
    assert ((h ^ C5) & 1) == 1 - (h & 1)          # C5 is odd: the direction bit comes out inverted


def reference(tree, values, rounds):
    """reference_kernel2 restated on Python lists; also records the depths after every round."""
    n, values, idx, depths = len(tree), list(values), [0] * len(values), []
    for _ in range(rounds):
        for i in range(len(values)):
            v = myhash(values[i] ^ tree[idx[i]])
            j = 2 * idx[i] + (1 if v % 2 == 0 else 2)
            idx[i] = 0 if j >= n else j
            values[i] = v
        depths.append({(k + 1).bit_length() - 1 for k in idx})
    return values, idx, depths


def stage1_form(tree, values, rounds, height):
    """The rewrite of stage 1: keep P + idx, no wrap test, no index write-back."""
    P = 7
    mem = [0] * P + list(tree)
    out = []
    for v0 in values:
        val, adr = v0, P
        for r in range(rounds):
            depth = r % (height + 1)
            val = myhash(val ^ mem[adr])
            if depth == height:
                adr = P                                  # every item is on a leaf: back to the root
            else:
                adr = 2 * adr + (1 - P) + (val & 1)      # P + (2*idx + 1 + bit)
        out.append(val)
    return out


for height, rounds, batch in [(10, 16, 64), (3, 9, 16), (1, 5, 8), (0, 3, 8), (4, 23, 8)]:
    tree = [random.randrange(2**30) for _ in range(2 ** (height + 1) - 1)]
    vals = [random.randrange(2**30) for _ in range(batch)]
    ref_vals, _, depths = reference(tree, vals, rounds)
    # all items share one depth: after round r it is (r + 1) % (height + 1)
    assert depths == [{(r + 1) % (height + 1)} for r in range(rounds)]
    assert stage1_form(tree, vals, rounds, height) == ref_vals


# Level picks. bits[j] = (q, p): the stored bit q, and polarity p (p = 1 means q is inverted).
def pick_madd(level_vals, bits):
    """Multilinear form: one multiply_add per inner node of the recursion, coefficients are scalars."""
    def lin(vals, k):
        if k == 0:
            return vals[0]
        q, p = bits[k - 1]
        base, other = (vals[1::2], vals[0::2]) if p else (vals[0::2], vals[1::2])
        diff = [(o - b) % M for o, b in zip(other, base)]
        return madd(q, lin(diff, k - 1), lin(base, k - 1))
    return lin(level_vals, len(bits))


def pick_vselect(level_vals, bits):
    """Selection tree on the flow engine: the earliest direction picks a half."""
    cur, n_sel = list(level_vals), 0
    for q, p in bits:
        half = len(cur) // 2
        lo, hi = cur[:half], cur[half:]
        cur = [(lo[i] if q else hi[i]) if p else (hi[i] if q else lo[i]) for i in range(half)]
        n_sel += half
    return cur[0], n_sel


def addr_from_bits(bits, depth, base):
    const = base + (1 << depth) - 1
    acc = const
    for j, (q, p) in enumerate(bits):
        w = 1 << (depth - 1 - j)
        acc += w * (1 - q if p else q)
    return acc


for depth in range(0, 6):
    level = [random.randrange(2**30) for _ in range(1 << depth)]
    for _ in range(200):
        dirs = [random.randrange(2) for _ in range(depth)]
        pol = [random.randrange(2) for _ in range(depth)]
        bits = [(d ^ p, p) for d, p in zip(dirs, pol)]
        offset = int("".join(map(str, dirs)) or "0", 2)
        assert pick_madd(level, bits) == level[offset]
        val, n_sel = pick_vselect(level, bits)
        assert val == level[offset] and n_sel == (1 << depth) - 1
        assert addr_from_bits(bits, depth, base=7) == 7 + (1 << depth) - 1 + offset

# Starter code: 36 non-debug slots per (item, round), one per cycle, plus setup.
per_item = {"load idx (add + load)": 2, "load val (add + load)": 2, "load node (add + load)": 2,
            "xor": 1, "hash 6 x 3": 18, "direction (% == select * +)": 5, "wrap (< select)": 2,
            "store idx (add + store)": 2, "store val (add + store)": 2}
assert sum(per_item.values()) == 36
consts = {0, 1, 2} | {h[1] for h in HASH_STAGES} | {h[4] for h in HASH_STAGES} | set(range(256))
setup = 7 * 2 + len(consts) + 2                   # header loads, const loads, two pauses
assert (len(consts), setup) == (262, 278)
assert 36 * 16 * 256 + setup == 147734

# Stage 1: per item 14 rounds x 24 + leaf round 20 + last round 21 (with the store).
assert 14 * 24 + 20 + 21 == 377
n_consts = 3 + 256 + 1 + 12                       # 1, 2, 1-P; 256 value addresses; P; hash constants
assert 377 * 256 + 256 + n_consts == 97040        # + 256 value loads
alu_ops = 256 * (16 * 19 + 14 * 4)
assert alu_ops == 92160 and -(-alu_ops // 12) == 7680   # stage 2 lower bound

# Stage 3: every (block, round) gathers 8 node values; two loads per cycle.
assert 32 * 16 * 8 == 4096 and 4096 // 2 == 2048
# Stage 4: rounds 0-2 and 11-13 need no gather.
assert sum(1 for r in range(16) if r % 11 >= 3) * 256 == 2560
# Stage 7: levels 1-3 by vselect on both trips; stage 8 adds 15 per block for 20 blocks.
assert 32 * 2 * (1 + 3 + 7) == 704 and 704 + 20 * 15 == 1004
# Stage 8/9 gathers: first trip rounds 4-10 for 32 blocks, second trip round 15 for 12 blocks.
gathers = 8 * (32 * 7 + 12 * 1)
assert gathers == 1888 and 6318 - 6082 == gathers // 8 and 7070 - 5182 == gathers
assert -(-1949 // 2) == 975 and -(-6082 // 6) == 1014
for cycles, speedup in [(97040, 1.52), (7690, 19.21), (2084, 70.89), (1548, 95.44), (1355, 109.03),
                        (1329, 111.16), (1107, 133.45), (1064, 138.85), (1040, 142.05)]:
    assert round(147734 / cycles, 2) == speedup
assert round(147734 / 18532, 2) == 7.97
print("all checks passed")
```

</details>

</details>
