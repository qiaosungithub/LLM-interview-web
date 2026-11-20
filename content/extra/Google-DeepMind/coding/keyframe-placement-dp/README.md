# Keyframe Placement: Minimum Cost Under Gap Limits, and Counting Optimal Plans

## Problem

A video has `n` frames, numbered `0` to `n - 1`. The encoder picks some of the frames to be keyframes (frames
that can be decoded on their own); every other frame stores only its difference from earlier frames. Making
frame `i` a keyframe costs an extra `c[i]` bytes. A choice must satisfy two rules:

- Frame `0` and frame `n - 1` must be keyframes: at the start there is nothing to reference, and the end must
  be a keyframe so the clip can be spliced directly onto the next one.
- The difference between the indices of two consecutive keyframes is called a gap. No gap may exceed `G`;
  otherwise scrubbing through the video means decoding too many frames in a row.

A plan is the set of keyframe indices, and its cost is the sum of `c[i]` over those frames.

All three parts share the same input ranges: `0 <= n <= 2000`, each `c[i]` is an integer with
`0 <= c[i] <= 10**9`, and `1 <= G <= 2000`; Parts 2 and 3 also have `1 <= H <= G`. Inputs are guaranteed to be
within these ranges, so no validation is needed. When `n = 0` (no frames) the only plan is the empty set, with
cost `0`; when `n = 1`, frame `0` is both the first and the last frame, and its cost is counted once. `G` may
be larger than `n`, in which case the gap limit has no effect.

### Part 1 — Minimum cost

```py
def min_cost(c: list[int], G: int) -> int: ...
```

Return the minimum cost of a valid plan. Since `G >= 1`, choosing every frame is always valid, so an answer
always exists. Enumerating plans one by one won't work: each of the `n - 2` inner frames can be chosen or not,
giving $2^{n-2}$ plans. It must finish within one second for `n = 2000`.

```text
c = [3, 1, 4, 1, 5], G = 2   ->  10    keyframes {0, 1, 3, 4}: 3 + 1 + 1 + 5
c = [3, 1, 4, 1, 5], G = 4   ->  8     keyframes {0, 4}, the gap is exactly 4
c = [3, 1, 4, 1, 5], G = 1   ->  14    every frame must be chosen
c = [7],             G = 3   ->  7
c = [],              G = 1   ->  0
```

### Part 2 — No two long gaps in a row

The player's buffer can absorb an occasional long gap, but not two in a row. You are now given one more
integer `H` (`1 <= H <= G`): a gap larger than `H` is called a long gap (it still may not exceed `G`). A new
rule is added: **two adjacent gaps may not both be long.** That is, for any three consecutive keyframes
`a < b < d`, `b - a` and `d - b` may not both be greater than `H`.

```py
def min_cost_limited(c: list[int], G: int, H: int) -> int: ...
```

Return the minimum cost under the new rule. A gap of `1` is never long, so an answer still always exists.

```text
c = [0, 2, 1, 3, 0], G = 2, H = 1   ->  3     {0, 1, 2, 4}; {0, 2, 4} costs only 1, but both of its gaps are 2
c = [3, 1, 4, 1, 5], G = 3, H = 2   ->  9     {0, 3, 4} or {0, 1, 4}
c = [3, 1, 4, 1, 5], G = 4, H = 2   ->  8     {0, 4}: there is only one gap, and a single long gap is allowed
c = [3, 1, 4, 1, 5], G = 2, H = 2   ->  10    with H = G there are no long gaps, same as Part 1
```

### Part 3 — How many optimal plans are there

The rate-control module wants to know how much freedom the encoder has. Under the rules of Part 2, find the
minimum cost and the number of plans whose cost is exactly that minimum. Two plans are different whenever
their keyframe sets differ. The number of plans can be very large, so return it modulo $10^9 + 7$; the cost is
not taken modulo anything.

```py
def count_optimal(c: list[int], G: int, H: int) -> tuple[int, int]: ...
# returns (minimum cost, number of plans with that cost modulo 1_000_000_007)
```

Build this by modifying your Part 2 code rather than writing a separate solution; it must finish within a few
seconds for `n = 2000`.

```text
c = [3, 1, 4, 1, 5], G = 3, H = 2   ->  (9, 2)    {0, 3, 4} and {0, 1, 4}
c = [0, 2, 1, 3, 0], G = 2, H = 1   ->  (3, 1)    only {0, 1, 2, 4}
c = [0, 0, 0, 0],    G = 3, H = 3   ->  (0, 4)    each of the two inner frames can be chosen or not
c = [0, 0, 0, 0],    G = 2, H = 1   ->  (0, 3)    {0, 1, 3}, {0, 2, 3}, {0, 1, 2, 3}
c = [],              G = 1, H = 1   ->  (0, 1)    only the empty plan
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two things are worth confirming before you start. First, whether both the first and last frames must be
chosen, and how `n = 1` is counted: both must be chosen, and when `n = 1` they are the same frame, counted
once. Second, whether a cost can be `0`: it can, so many plans may share the same cost, and Part 3 cannot
assume there is only one optimal plan.

### Part 1

**Finding the state.** Decide which frames are keyframes from left to right. Suppose frame `i` has been fixed
as a keyframe. Then which choices for the later frames are valid, and how much more they cost, depends only
on `i`: the next keyframe must fall between `i + 1` and `i + G`, regardless of which frames were chosen before
`i`. So each `i` needs just one number:

- `best[i]`: the minimum cost of a valid choice looking only at frames `0` through `i`, with frame `i` a
  keyframe.

**Recurrence.** For frame `i` (`i >= 1`), the previous keyframe `j` must satisfy `i - G <= j <= i - 1`;
enumerate it:

$$
\mathrm{best}[0] = c[0], \qquad
\mathrm{best}[i] = c[i] + \min_{\max(0,\, i-G) \,\le\, j \,<\, i} \mathrm{best}[j].
$$

The last frame must be a keyframe, so the answer is simply `best[n - 1]`.

**Why it is correct.** Induct on `i`. Any valid choice ending at `i`, with `i` removed, is a valid choice
ending at some `j`, and `j` lies in the window `[i - G, i - 1]`, so its cost is at least `c[i] + best[j]`.
Conversely, appending `i` to any valid choice ending at a `j` in the window adds a gap of at most `G`, so the
result is still valid. The two directions together give the equation above.

**Complexity.** There are `n` states, and each looks at no more than `G` predecessors: $O(nG)$ time and
$O(n)$ space. For `n = G = 2000` the upper bound is four million comparisons (the early frames have fewer than
`G` predecessors, so it is about two million in practice). The window minimum can also be brought down to
$O(n)$ in total with a monotonic deque; see the follow-ups.

```python
def min_cost(c, G):
    n = len(c)
    if n == 0:
        return 0
    best = [0] * n            # best[i]: cheapest valid choice on frames 0..i that ends with a keyframe at i
    best[0] = c[0]
    for i in range(1, n):
        best[i] = c[i] + min(best[max(0, i - G):i])      # previous keyframe j with i - G <= j < i
    return best[n - 1]
```

### Part 2

With the new rule, "which later choices are valid" no longer depends only on the position of the last
keyframe: if the last gap was long, the next gap must be short. So the state needs one more bit of
information, namely whether the last gap was long:

- `short[i]`: the minimum cost with frame `i` a keyframe and the last gap not long. For `i = 0` there is no gap
  yet, and that case also goes here.
- `long[i]`: the minimum cost with frame `i` a keyframe and the last gap long; infinity if there is no such
  choice.

Let the previous keyframe be `j`. When `i - j <= H` this step is a short gap, and `j` may be in either state;
when `H < i - j <= G` this step is a long gap, and `j` must be in the `short` state:

$$
\mathrm{short}[i] = c[i] + \min_{i-H \,\le\, j \,<\, i} \min(\mathrm{short}[j],\, \mathrm{long}[j]),
\qquad
\mathrm{long}[i] = c[i] + \min_{i-G \,\le\, j \,<\, i-H} \mathrm{short}[j].
$$

(Indices `j` below `0` don't exist and are simply skipped.) The answer is `min(short[n - 1], long[n - 1])`.
The correctness argument is the same as in Part 1, except that "valid choices ending at `i`" are split into
two classes by the type of the last gap.

**A common mistake** is to keep a single minimum cost per frame, and alongside it record "whether the choice
achieving that minimum ended with a long gap". Take the example from the statement,
`c = [0, 2, 1, 3, 0], G = 2, H = 1`: the cheapest way to reach frame `2` is `{0, 2}` (cost `1`, last step a
long gap), so `{0, 1, 2}` (cost `3`, last step a short gap) gets thrown away. But jumping from frame `2` to
frame `4` is another long gap, and only the discarded choice can be extended by it. The single-value version
ends up with `4`, while the correct answer is `3`. A choice that costs more but leaves fewer restrictions on
what follows cannot be discarded, and that is exactly why the state has to be split in two.

**Complexity.** The number of states doubles; for each `i`, the two windows together still cover at most `G`
predecessors. $O(nG)$ time, $O(n)$ space.

```python
INF = float("inf")


def min_cost_limited(c, G, H):
    n = len(c)
    if n == 0:
        return 0
    short = [INF] * n         # the last gap is not long (or there is no gap yet)
    long_ = [INF] * n         # the last gap is long
    short[0] = c[0]
    for i in range(1, n):
        near = max(0, i - H)  # j in [near, i) makes a short gap, j in [far, near) a long one
        far = max(0, i - G)
        short[i] = c[i] + min(min(short[near:i]), min(long_[near:i]))
        if far < near:
            long_[i] = c[i] + min(short[far:near])       # NOTE: short[j] only, a long gap may not follow a long gap
    return min(short[n - 1], long_[n - 1])
```

### Part 3

Change what each state stores from "the minimum cost" to a pair "(minimum cost, number of choices achieving
that cost)". The shape of the recurrence stays exactly the same; only two operations change:

- When taking the minimum over several sources: the one with the smaller cost wins and brings its count with
  it; when costs are equal, the counts are added.
- When appending frame `i`: add `c[i]` to the cost and leave the count unchanged.

**Why nothing is double-counted or missed.** A choice ending at `i` decomposes uniquely into "the previous
keyframe `j`, and whether it is in the `short` or `long` state at `j`", plus `i`. Different `j`, or the two
states at the same `j`, correspond to disjoint sets of choices. So adding up the counts of every source that
achieves the minimum cost counts each optimal choice exactly once. The two classes `short[i]` and `long[i]`
are also disjoint (the last gap is either long or not), and when they are merged at the end it is again
"smaller wins, equal adds".

Two details. Take the count modulo after every addition, but **never take the cost modulo anything**, because
it is used for comparisons. The cost can reach about $2000 \times 10^9 = 2 \times 10^{12}$, which is fine for
Python integers; in C++ or Java you need 64-bit integers.

**Complexity.** Same as Part 2: $O(nG)$ time, $O(n)$ space.

```python
MOD = 1_000_000_007


def _best(cost, ways, lo, hi):
    """Smallest cost at positions lo..hi-1 and how many choices reach it; (INF, 0) for an empty range."""
    best, total = INF, 0
    for j in range(lo, hi):
        if cost[j] < best:
            best, total = cost[j], ways[j]
        elif cost[j] == best:
            total += ways[j]
    return best, total % MOD


def _merge(a, b):
    """Combine two (cost, ways) alternatives that describe disjoint sets of choices."""
    if a[0] != b[0]:
        return a if a[0] < b[0] else b
    return a[0], (a[1] + b[1]) % MOD


def count_optimal(c, G, H):
    n = len(c)
    if n == 0:
        return 0, 1           # the empty plan
    s_cost, s_ways = [INF] * n, [0] * n       # same meaning as short[] in Part 2, plus a count
    l_cost, l_ways = [INF] * n, [0] * n       # same meaning as long_[]
    s_cost[0], s_ways[0] = c[0], 1
    for i in range(1, n):
        near = max(0, i - H)
        far = max(0, i - G)
        cost, ways = _merge(_best(s_cost, s_ways, near, i), _best(l_cost, l_ways, near, i))
        s_cost[i], s_ways[i] = c[i] + cost, ways
        cost, ways = _best(s_cost, s_ways, far, near)
        l_cost[i], l_ways[i] = c[i] + cost, ways         # stays (INF, 0) when no long gap can end here
    return _merge((s_cost[n - 1], s_ways[n - 1]), (l_cost[n - 1], l_ways[n - 1]))
```

### Follow-ups

- **When `n` and `G` both go up to $2 \times 10^5$, $O(nG)$ is too slow. What do you do?** The recurrence
  needs the minimum over a window that slides right as `i` grows. Use a monotonic deque: values increase from
  the front to the back; before pushing a new value, pop every value at the back that is not smaller than it,
  and pop the front once it falls out of the window, so the front is always the window minimum. Each index
  enters and leaves the deque once, so Part 1 is $O(n)$ in total. In Part 2 the two windows
  `[i - H, i - 1]` and `[i - G, i - H - 1]` also both slide right with `i`, so one deque each is enough. The
  check code includes both versions and runs them at `n = 200000`.
- **If the rule becomes "at most `R` long gaps in a row", how does the state change?** Replace the one bit
  in the state with a small counter: `f[i][r]` is the minimum cost with frame `i` a keyframe and exactly `r`
  long gaps in a row at the end, for `r` from `0` to `R`. A short gap resets `r` to zero; a long gap adds one
  to `r` and requires `r < R` beforehand. $O(nGR)$ time. Part 2 is the case `R = 1`. The principle of state
  design is always the same: of everything in the past, keep only the part that affects whether what follows
  is valid. The check code implements this general version.
- **To count all valid plans instead of just the optimal ones, do you have to rewrite anything?** No. Replace
  every cost with `0` and call `count_optimal`: every valid plan then costs `0`, so all of them are optimal,
  and the returned count is the total number of valid plans. For example, when `n >= 2` and `H = G >= n - 1`,
  each inner frame can be chosen or not, and the result is $2^{n-2}$.
- **How would you test this code?** On small inputs, compare against a brute force that "enumerates every
  subset of the inner frames and checks each rule in the statement", comparing both the cost and the count;
  the brute force is fast enough for `n` up to `10`. Write the edge cases separately: `n = 0, 1, 2`, `G = 1`
  (everything must be chosen), `H = G` (should match Part 1), and all-zero costs (the largest counts).
  Finally run once at `n = 2000` to check the timing, and verify the counts modulo the prime on inputs you
  can work out by hand (for example, with all-zero costs and `G = H = 2` the count is a Fibonacci number).

<details>
<summary>Checks (runnable)</summary>

```python
import itertools
import random
import time
from collections import deque

# ---- every example in the statement ----
assert min_cost([3, 1, 4, 1, 5], 2) == 10
assert min_cost([3, 1, 4, 1, 5], 4) == 8
assert min_cost([3, 1, 4, 1, 5], 1) == 14
assert min_cost([7], 3) == 7
assert min_cost([], 1) == 0

assert min_cost_limited([0, 2, 1, 3, 0], 2, 1) == 3
assert min_cost_limited([3, 1, 4, 1, 5], 3, 2) == 9
assert min_cost_limited([3, 1, 4, 1, 5], 4, 2) == 8
assert min_cost_limited([3, 1, 4, 1, 5], 2, 2) == 10

assert count_optimal([3, 1, 4, 1, 5], 3, 2) == (9, 2)
assert count_optimal([0, 2, 1, 3, 0], 2, 1) == (3, 1)
assert count_optimal([0, 0, 0, 0], 3, 3) == (0, 4)
assert count_optimal([0, 0, 0, 0], 2, 1) == (0, 3)
assert count_optimal([], 1, 1) == (0, 1)


# ---- independent brute force: every subset of the inner frames, checked against the rules as stated ----
def brute(c, G, H, R=1):
    """(min cost, number of plans with that cost); at most R long gaps in a row (the statement has R = 1)."""
    n = len(c)
    if n == 0:
        return 0, 1
    best, ways = None, 0
    inner = range(1, n - 1)
    for size in range(len(inner) + 1):
        for chosen in itertools.combinations(inner, size):
            keys = sorted({0, n - 1, *chosen})
            gaps = [b - a for a, b in zip(keys, keys[1:])]
            run = longest_run = 0
            for g in gaps:
                run = run + 1 if g > H else 0
                longest_run = max(longest_run, run)
            if any(g > G for g in gaps) or longest_run > R:
                continue
            cost = sum(c[k] for k in keys)
            if best is None or cost < best:
                best, ways = cost, 1
            elif cost == best:
                ways += 1
    return best, ways


def one_value_shortcut(c, G, H):
    """The mistake described in Part 2: one cost per frame plus a flag for how that one plan ended."""
    n = len(c)
    best = [INF] * n
    was_long = [False] * n
    best[0] = c[0]
    for i in range(1, n):
        for j in range(max(0, i - G), i):
            is_long = i - j > H
            if is_long and was_long[j]:
                continue
            cand = best[j] + c[i]
            if cand < best[i] or (cand == best[i] and was_long[i] and not is_long):
                best[i], was_long[i] = cand, is_long
    return best[n - 1]


# ---- follow-up 1: the same recurrences with monotonic deques, O(n) ----
class MinWindow:
    """Minimum of the pushed values whose index is still >= lo; indices arrive in increasing order."""

    def __init__(self):
        self.q = deque()

    def push(self, idx, val):
        while self.q and self.q[-1][1] >= val:
            self.q.pop()
        self.q.append((idx, val))

    def min_from(self, lo):
        while self.q and self.q[0][0] < lo:
            self.q.popleft()
        return self.q[0][1] if self.q else INF


def min_cost_fast(c, G):
    if not c:
        return 0
    win = MinWindow()
    best = c[0]
    win.push(0, best)
    for i in range(1, len(c)):
        best = c[i] + win.min_from(i - G)
        win.push(i, best)
    return best


def min_cost_limited_fast(c, G, H):
    n = len(c)
    if n == 0:
        return 0
    short, long_ = [INF] * n, [INF] * n
    short[0] = c[0]
    near, far = MinWindow(), MinWindow()      # near: min(short, long) on [i-H, i-1]; far: short on [i-G, i-H-1]
    for i in range(1, n):
        near.push(i - 1, min(short[i - 1], long_[i - 1]))
        if i - H - 1 >= 0:
            far.push(i - H - 1, short[i - H - 1])
        short[i] = c[i] + near.min_from(i - H)
        long_[i] = c[i] + far.min_from(i - G)
    return min(short[n - 1], long_[n - 1])


# ---- follow-up 2: at most R long gaps in a row ----
def min_cost_runs(c, G, H, R):
    n = len(c)
    if n == 0:
        return 0
    f = [[INF] * (R + 1) for _ in range(n)]   # f[i][r]: keyframe at i, exactly r long gaps at the end
    f[0][0] = c[0]
    for i in range(1, n):
        for j in range(max(0, i - G), i):
            if i - j <= H:
                f[i][0] = min(f[i][0], c[i] + min(f[j]))
            else:
                for r in range(R):
                    f[i][r + 1] = min(f[i][r + 1], c[i] + f[j][r])
    return min(f[n - 1])


# ---- the mistake from Part 2 really is a mistake on the statement's example ----
assert one_value_shortcut([0, 2, 1, 3, 0], 2, 1) == 4 and brute([0, 2, 1, 3, 0], 2, 1) == (3, 1)

# ---- edge cases ----
for G in (1, 2, 5):
    for H in range(1, G + 1):
        assert count_optimal([], G, H) == (0, 1) and min_cost_limited([], G, H) == 0
        assert count_optimal([9], G, H) == (9, 1) and min_cost_limited([9], G, H) == 9
        assert count_optimal([4, 6], G, H) == (10, 1) and min_cost([4, 6], G) == 10      # both ends are forced
assert min_cost([5] * 7, 1) == 35 and count_optimal([5] * 7, 1, 1) == (35, 1)             # G = 1: every frame
assert count_optimal([5] * 7, 3, 3) == (15, 1)        # all equal: fewest keyframes, {0, 3, 6} is the only way
assert count_optimal([5] * 7, 3, 2) == (20, 7)        # {0, 3, 6} is now forbidden; four keyframes: the gaps are
assert brute([5] * 7, 3, 2) == (20, 7)                # 2+2+2 or an ordering of 1+2+3, seven ways
assert count_optimal([5] * 6, 2000, 2000) == (10, 1)  # G larger than n: only the two ends
assert min_cost([0, 10 ** 9, 10 ** 9, 0], 2) == 10 ** 9

# ---- random comparison with the brute force ----
rng = random.Random(20261117)
saw_tie = saw_rule_binding = saw_shortcut_wrong = saw_long_used = 0
for _ in range(4000):
    n = rng.randint(0, 10)
    G = rng.randint(1, max(1, n + 1))
    H = rng.randint(1, G)
    top = rng.choice([0, 1, 3, 9, 10 ** 9])
    c = [rng.randint(0, top) for _ in range(n)]
    want_cost, want_ways = brute(c, G, H)
    assert min_cost_limited(c, G, H) == want_cost, (c, G, H)
    assert count_optimal(c, G, H) == (want_cost, want_ways % MOD), (c, G, H)
    assert min_cost(c, G) == brute(c, G, G)[0] == min_cost_limited(c, G, G), (c, G)
    assert min_cost_fast(c, G) == min_cost(c, G)
    assert min_cost_limited_fast(c, G, H) == want_cost, (c, G, H)
    R = rng.randint(1, 3)
    assert min_cost_runs(c, G, H, R) == brute(c, G, H, R)[0], (c, G, H, R)
    assert count_optimal([0] * n, G, H) == (0, brute([0] * n, G, H)[1])     # follow-up 3: all valid plans
    saw_tie += want_ways > 1
    saw_rule_binding += want_cost > min_cost(c, G)
    saw_long_used += want_cost < min_cost(c, H)          # cheaper than every plan without a long gap
    saw_shortcut_wrong += n >= 1 and one_value_shortcut(c, G, H) != want_cost
assert min(saw_tie, saw_rule_binding, saw_long_used, saw_shortcut_wrong) >= 20, \
    (saw_tie, saw_rule_binding, saw_long_used, saw_shortcut_wrong)

# ---- the largest size in the stated range, with answers that can be worked out by hand ----
N = 2000
big = [rng.randint(0, 10 ** 9) for _ in range(N)]
t0 = time.perf_counter()
a1 = min_cost(big, 2000)
a2 = min_cost_limited(big, 2000, 1000)
a3 = count_optimal(big, 2000, 1000)
zeros_free = count_optimal([0] * N, 2000, 2000)
zeros_fib = count_optimal([0] * N, 2, 2)
zeros_no_two = count_optimal([0] * N, 2, 1)
print(f"n = 2000, six calls: {time.perf_counter() - t0:.2f} s")      # printed, not asserted
assert a3[0] == a2 == min_cost_limited_fast(big, 2000, 1000) and a1 == min_cost_fast(big, 2000) and a1 <= a2
assert zeros_free == (0, pow(2, N - 2, MOD))                 # every inner frame is free

fib = [0, 1]                                  # gaps of 1 or 2 that add up to N - 1: Fibonacci number F(N)
while len(fib) <= N:
    fib.append(fib[-1] + fib[-2])
assert zeros_fib == (0, fib[N] % MOD)

t = [1, 1, 2]                                 # the same, but no two gaps of 2 next to each other
while len(t) <= N - 1:                        # t[m] = t[m-1] (ends with 1) + t[m-3] (ends with 1, 2)
    t.append(t[-1] + t[-3])
assert zeros_no_two == (0, t[N - 1] % MOD)
assert count_optimal([5] * 1999, 2, 2) == (5000, 1)          # 1998 = 999 gaps of 2, only one way
assert 2000 * 10 ** 9 > 2 ** 31 and 2000 * 10 ** 9 < 2 ** 63  # costs need 64 bits, and 64 bits are enough

# ---- follow-up 1 at n = 200000 ----
huge = [rng.randint(0, 10 ** 9) for _ in range(200_000)]
t0 = time.perf_counter()
h1 = min_cost_fast(huge, 150_000)
h2 = min_cost_limited_fast(huge, 150_000, 70_000)
print(f"n = 200000, deque versions: {time.perf_counter() - t0:.2f} s")
assert h1 <= h2 <= min_cost_fast(huge, 70_000)               # more freedom can only lower the cost
print("all checks passed")
```

</details>

</details>
