# Packing Training Jobs onto GPUs: Search Order, Pruning and Branch and Bound

## Problem

A machine has `k` identical GPUs, each with `cap` GiB of memory. There are `n` jobs to place on them; job `i`
needs `jobs[i]` GiB of memory. A job must sit entirely on one GPU and cannot be split; a GPU can hold several
jobs as long as their memory adds up to at most `cap`. An assignment is a list `assign` of length `n`, where
`assign[i]` is the number of the GPU job `i` is on (`0` to `k - 1`).

Input ranges: `0 <= n <= 14`, `1 <= k <= 14`, `1 <= jobs[i] <= 10**6`, `1 <= cap <= 10**7`, all integers. Inputs
are guaranteed to be in range; no validation needed. When `n = 0`, the empty list `[]` is a valid assignment.

**Node counting.** The interviewer will count how many nodes your search visits: each call of the recursive
function counts as one node, including the very first call and the call made once all jobs have been placed.
Every function takes an optional dict `stats` and, before returning, writes the node count to
`stats["nodes"]`. This number depends only on how the code searches, not on how fast the machine is, so below
it is used instead of running time to compare the effect of pruning.

### Part 1 — Find an assignment

```py
def pack(jobs: list[int], k: int, cap: int, stats: dict | None = None) -> list[int] | None: ...
```

Return the lexicographically smallest valid `assign` (compare `assign[0]` first, then `assign[1]` on a tie, and
so on); return `None` if there is no valid assignment. This part only needs to be correct; the interviewer
tests it only on inputs with `n <= 8`.

```text
jobs = [3, 5, 2, 4], k = 2, cap = 7  ->  [0, 1, 1, 0]    GPU 0 holds 3 + 4 = 7, GPU 1 holds 5 + 2 = 7
jobs = [3, 5, 2, 4], k = 3, cap = 5  ->  [0, 1, 0, 2]    3 + 2 = 5, 5, 4
jobs = [3, 5, 2, 4], k = 2, cap = 6  ->  None            total 14 > 2 × 6
jobs = [8],          k = 2, cap = 7  ->  None            a single job does not fit
jobs = [],           k = 3, cap = 1  ->  []
```

### Part 2 — Pruning

Now `n` can be up to `14`. Return any valid assignment (it no longer has to be the lexicographically smallest;
the interviewer only checks that it is valid), or `None` if there is none.

```py
def pack_fast(jobs: list[int], k: int, cap: int, stats: dict | None = None) -> list[int] | None: ...
```

The two inputs below each have 12 jobs and 4 GPUs. Counted with the Part 1 reference approach (jobs processed
in input order, GPUs tried from the smallest number up, check that a job fits before placing it, stop at the
first solution), the number of nodes visited is:

```text
A: jobs = [6, 9, 13, 15, 8, 9, 10, 15, 3, 3, 23, 30], k = 4, cap = 36
   feasible: total 144 = 4 × 36, e.g. {30, 6}, {23, 13}, {15, 15, 3, 3}, {10, 9, 9, 8}
   the Part 1 approach visits 66005 nodes

B: jobs = [31, 7, 13, 6, 30, 9, 9, 13, 31, 24, 40, 31], k = 4, cap = 61
   infeasible: total 244 = 4 × 61, so every GPU must be exactly full; but the only jobs that fit alongside 40
   are 13, 13, 9, 9, 7, 6, and no subset of them adds up to the 21 that 40's GPU still needs
   the Part 1 approach visits 350105 nodes
```

Requirements: on both A and B, visit at most one percent of the nodes above; inputs with `n = 14`, including
hard ones where the job sizes are all similar and they only just fail to fit, should finish in about a second.
Explain why each pruning never misses a solution, and why it saves so many nodes.

### Part 3 — Keep the busiest GPU as empty as possible

`cap` is no longer given. The peak of an assignment is the largest memory usage over all GPUs. Find an
assignment with the smallest peak.

```py
def min_peak(jobs: list[int], k: int, stats: dict | None = None) -> tuple[int, list[int]]: ...
# returns (smallest possible peak, one assignment that reaches it); any such assignment is accepted
```

When `n = 0`, return `(0, [])`. With `n = 14` it should finish in about a second; reuse the Part 2 code as much
as you can.

```text
jobs = [3, 3, 2, 2, 2], k = 2  ->  peak 6    {3, 3} and {2, 2, 2}
jobs = [3, 5, 2, 4],    k = 3  ->  peak 5    {5}, {3, 2}, {4}
A, k = 4                       ->  peak 36
B, k = 4                       ->  peak 62   total / 4 is 61, but Part 2 showed that 61 does not fit
jobs = [],              k = 2  ->  peak 0
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two things are worth confirming before you start. First, can a job be split across two GPUs? No; otherwise the
problem reduces to simple division. Second, are the GPUs identical? Yes. This decides whether the most
effective pruning in Part 2 can be used at all; once the problem changes to "each GPU has a different amount of
memory", that pruning has to change with it.

### Part 1

**Idea.** Decide where each job goes, one job at a time. For job `i`, try every GPU that can still hold it, in
increasing order of number: place it there and recurse on job `i + 1`; if the rest fails, take it back off
(undo) and try the next GPU. If every GPU has been tried without success, report failure to the level above.

**Invariant.** When `place(i)` is called, `loads[g]` is exactly the total memory of the jobs among `0..i-1`
placed on GPU `g`, and every `loads[g]` is at most `cap`. The undo line `loads[g] -= jobs[i]` is there so
that the invariant still holds on returning to the level above.

**Why the result is the lexicographically smallest assignment.** The search visits partial assignments in
lexicographic order: it first fixes the smallest possible value of `assign[0]`, under that the smallest
possible value of `assign[1]`, and so on. A branch cut by "does not fit" already has a GPU over `cap`, and
memory only grows as more jobs are added, so there is no valid assignment below such a branch. Therefore the
first complete assignment the search reaches is the lexicographically smallest valid one.

**Complexity.** Level `d` of the search tree has at most $k^d$ nodes and each node tries `k` GPUs, so the worst
case is $O(k^{n+1})$ time; space is the recursion depth plus two arrays, $O(n + k)$. That is fine for
`n <= 8`, but with `n = 14`, `k = 4` it is $4^{14} \approx 2.7 \times 10^8$, which is too slow.

```python
def pack(jobs, k, cap, stats=None):
    n = len(jobs)
    loads = [0] * k           # loads[g]: memory used on GPU g by jobs 0..i-1
    assign = [0] * n
    nodes = 0

    def place(i):
        nonlocal nodes
        nodes += 1
        if i == n:
            return True
        for g in range(k):    # smaller GPU numbers first, so the first full assignment is the smallest one
            if loads[g] + jobs[i] <= cap:
                loads[g] += jobs[i]
                assign[i] = g
                if place(i + 1):
                    return True
                loads[g] -= jobs[i]          # undo before trying the next GPU
        return False

    found = place(0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None
```

### Part 2

Part 1 is slow not because each node does a lot of work, but because it spends most of its time in subtrees
that are doomed to fail. The two prunings each attack one of the two causes.

**Pruning 1: place the large jobs first.** Sort the jobs by memory in decreasing order before searching. Large
jobs have the fewest GPUs they can go to; placing them first exposes conflicts close to the root, where a single
failure cuts away a whole large subtree. Searching in input order, the largest job in B, 40, is 11th in line, so
every arrangement of the first 10 jobs is enumerated first, and each one only discovers at level 11 that 40 does
not fit. Sorting changes only the order of the search, not what is searched, so no solution is missed. The idea
is usually called "most constrained variable first" (fail-first).

**Pruning 2: symmetry between GPUs.** The GPUs are identical. If two GPUs currently have the same load, then
whichever of them the current job goes into, the two subtrees below are mirror images: swapping the numbers of
the two GPUs turns every assignment in one subtree into an assignment in the other, and both are valid. So only
one GPU per load value needs to be tried. The most common case is empty GPUs: all empty GPUs are equivalent, so
try just one. Without this rule the first job has `k` equivalent choices, the second job another `k - 1`, and so
on, and the whole tree is pointlessly duplicated many times over.

**Correctness.** Suppose a valid assignment puts job `i` on GPU `g2`, while another GPU `g1` had the same load
as `g2` just before job `i` was placed. Swap `g1` and `g2` for job `i` and every job after it: the two GPUs'
final loads are exactly exchanged, both still at most `cap`, and the part before job `i` is unchanged. So
skipping `g2` cannot lose the conclusion "a solution exists".

In addition, before searching do two $O(n)$ checks: the total must not exceed `k * cap`, and the largest job
must not exceed `cap`. If either fails, return `None` immediately and record `0` nodes.

**Effect.** The table below gives the nodes each version visits on A and B from the statement (each entry is
asserted in the verification code):

| Version | A (feasible) | B (infeasible) |
| --- | ---: | ---: |
| Part 1 as is | 66005 | 350105 |
| Symmetry pruning only | 15491 | 13356 |
| Decreasing size only | 13 | 1577 |
| Both | 13 | 16 |

After sorting, A succeeds on the first dive: `13 = n + 1` nodes, with no backtracking at all. For B, proving
infeasibility, neither pruning is enough on its own; together they bring it down to 16 nodes. After sorting,
the first four jobs 40, 31, 31, 31 exceed 61 in any pair, so each takes its own GPU, and the symmetry pruning
leaves only one way to do that; next, 30 can only go with one of the 31s, and 24 only into one of the remaining
31s, and within a few steps the search hits a dead end.

**How much it saves depends on the input.** On the random inputs in the verification code (`n` from 9 to 11,
`k` 3 or 4, job sizes 5 to 40), for infeasible inputs with `cap` one less than the optimal peak, the median
node count shrinks by a few hundred times, and no input gets worse; for feasible inputs with `cap` exactly equal
to the optimal peak, the median shrinks only a few times, and some inputs actually take more nodes, because the
original order happened to run into a solution very early. Most of the benefit of pruning comes when the whole
tree must be searched: proving infeasibility or, as in Part 3, proving that no better solution exists.

**Complexity.** The worst case is still exponential: with `k = 2` this is the partition problem, which is
NP-complete, so do not expect a polynomial algorithm. An upper bound you can give: with the symmetry pruning,
distinct nodes at level `d` correspond to distinct ways of splitting the first `d` jobs into at most `k`
unlabeled groups (where two paths first diverge, the current job went into two GPUs with different loads, and
hence different sets of jobs), so level `d` has at most $\sum_{j \le k} S(d, j)$ nodes, where $S$ are the
Stirling numbers of the second kind; that is far less than $k^d$, and the `cap` constraint cuts most of it
away. Each node does $O(k)$ work, and space is $O(n + k)$.

```python
def pack_fast(jobs, k, cap, stats=None, sort_desc=True, skip_equal=True):
    """Any valid assignment or None. The two switches exist only so that the checks can measure each pruning."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i]) if sort_desc else list(range(n))
    loads = [0] * k
    assign = [0] * n
    nodes = 0

    def place(pos):
        nonlocal nodes
        nodes += 1
        if pos == n:
            return True
        i = order[pos]                       # the pos-th largest job
        tried = set()                        # loads already tried for this job at this node
        for g in range(k):
            if loads[g] + jobs[i] > cap:
                continue
            if skip_equal:
                if loads[g] in tried:        # NOTE: same load as a GPU tried before, so the subtree is a mirror image
                    continue
                tried.add(loads[g])
            loads[g] += jobs[i]
            assign[i] = g
            if place(pos + 1):
                return True
            loads[g] -= jobs[i]
        return False

    found = sum(jobs) <= k * cap and max(jobs, default=0) <= cap and place(0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None
```

### Part 3

This is an optimization problem. Search the same tree, but instead of stopping at the first solution, keep
track of the best solution so far (the incumbent) while searching, and use it to prune. This approach is called
branch and bound.

1. **Get a decent solution greedily first.** Take the jobs from largest to smallest and put each on the
   currently least loaded GPU (this greedy is called LPT, longest processing time first). Its peak is the
   initial `best`.
2. **Compute a lower bound.** The peak of any assignment is at least the largest job, and at least
   $\lceil \text{total} / k \rceil$. Call the larger of the two `lower`. If the greedy already reaches `lower`,
   return right away without searching.
3. **Prune during the search.** Keep the sorting and symmetry pruning from Part 2. After placing a job, if that
   GPU's load is already `>= best`, every assignment below this branch has a peak `>= best` and cannot be
   strictly better, so skip it. This amounts to running the Part 2 search with `cap = best - 1` at every
   moment, and as `best` keeps shrinking, the pruning gets more and more aggressive.
4. **Update at a leaf.** Reaching a leaf means every step on the way stayed `< best`, so this is a strictly
   better solution; record it. As soon as `best == lower`, nothing can do better, so end the whole search
   immediately.

**Correctness.** Every pruned branch has a peak `>= best` and cannot be a strictly better solution; the argument
for the symmetry pruning is the same as in Part 2 (swapping the numbers of two GPUs does not change the peak).
So when the search ends, `best` is the smallest peak.

**Effect.** Without the "prune at `>= best`" rule (keeping everything else, including the greedy initial value
and stopping at the lower bound), A takes 700662 nodes and B 624928; with it, 13 and 16 respectively. The
verification code also has an input where the greedy gives 69, the lower bound is 61 and the optimum is 62:
branch and bound visits 4424 nodes, versus 2412066 without the rule.

**Complexity.** The worst case is likewise exponential. At every node the search reaches, every GPU's load is
below the greedy peak, so the node count is at most that of the Part 2 search with `cap = greedy peak - 1`,
run over the whole tree without stopping at a solution. Sorting is $O(n \log n)$, the greedy $O(nk)$, each
node $O(k)$, and space $O(n + k)$. The verification code picks out the randomly generated `n = 14` inputs that
take the most nodes and prints their running times.

```python
def min_peak(jobs, k, stats=None, use_bound=True):
    """(smallest peak, an assignment reaching it). use_bound=False only exists so the checks can measure the bound."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i])
    loads = [0] * k
    assign = [0] * n
    for i in order:                          # greedy LPT: largest job onto the least loaded GPU
        g = loads.index(min(loads))
        loads[g] += jobs[i]
        assign[i] = g
    best, best_assign = max(loads), assign[:]
    lower = max(max(jobs, default=0), -(-sum(jobs) // k))
    loads = [0] * k
    nodes = 0

    def place(pos, peak):
        nonlocal best, best_assign, nodes
        nodes += 1
        if pos == n:
            if peak < best:                  # always true here: every step on the way stayed below best
                best, best_assign = peak, assign[:]
            return
        i = order[pos]
        tried = set()
        for g in range(k):
            new = loads[g] + jobs[i]
            if loads[g] in tried or (use_bound and new >= best):     # NOTE: >= because only a strictly better plan helps
                continue
            tried.add(loads[g])
            loads[g] = new
            assign[i] = g
            place(pos + 1, max(peak, new))
            loads[g] = new - jobs[i]
            if best == lower:                # nothing can beat the lower bound: unwind the whole search
                return

    if best > lower:
        place(0, 0)
    if stats is not None:
        stats["nodes"] = nodes               # search nodes only; the greedy pass is not counted
    return best, best_assign
```

### Follow-up questions

- **Why not just use the greedy?** LPT is not always optimal. Take `[3, 3, 2, 2, 2]`, `k = 2` from the
  statement: the greedy first separates the two 3s, then places the three 2s in turn, giving `{3, 2, 2}` and
  `{3, 2}` with peak 7, while the optimum is 6. Graham proved that LPT's peak is at most
  $\frac{4}{3} - \frac{1}{3k}$ times the optimum, and the verification code checks this bound on random
  inputs. The greedy is good as an initial value and a fallback; search only when you need the exact optimum.
- **Could you binary search on the peak, calling Part 2 each time?** Yes: the answer lies in
  `[lower, greedy peak]`; binary search over it, using `pack_fast` each time to decide whether everything fits,
  with a logarithmic number of calls. The downside is that every "does not fit" answer has to search the whole
  tree, which is exactly the most expensive kind of search; branch and bound keeps tightening `best` within a
  single search, so nothing already searched is repeated. The verification code cross-checks the binary-search
  version against `min_peak`.
- **When does backtracking turn into dynamic programming?** When the same state is reached again and again.
  After sorting, once the first `pos` jobs are placed, the remaining problem depends only on the multiset of GPU
  loads, not on which job is on which GPU. So you can keep the failed `(pos, sorted loads)` pairs in a set and
  return failure immediately when one comes up again; that is search with memoization, which is dynamic
  programming. Whether it helps depends on whether states repeat: with 14 jobs all of size 5, `k = 4`,
  `cap = 19`, the node count drops from 2035 to 61; when the job sizes are all different, load combinations
  almost never repeat, and memoization only costs memory. Another formulation is a DP over subsets of jobs:
  `f[j][mask]` is the smallest peak when placing the jobs in `mask` on `j` GPUs, enumerating the subset that
  goes on the last GPU, in $O(k \cdot 3^n)$ time. Its running time does not depend on luck, and the
  verification code uses it as an independent brute force.
- **If some jobs cannot share a GPU (for example, two jobs that both need exclusive use of some device), which
  prunings still hold?** Placing large jobs first still holds, since it only changes the search order; jobs with
  many conflicts should also be placed early. The symmetry pruning has to be tightened: two GPUs with the same
  load but different jobs on them conflict differently with later jobs, so they are no longer mirror images.
  Only completely empty GPUs are still equivalent, so the only rule you can keep is "try just one empty GPU".
- **What happens if `n` grows a bit, say 16 jobs of similar size?** It gets slow. The verification code has an
  input with 16 jobs, all between 1005 and 1049 in size, and `k = 5`; with `cap` one less than the optimal
  peak, `pack_fast` has to visit 3904649 nodes to prove infeasibility. You can add a "wasted space" pruning: if
  the space left on a GPU is smaller than the smallest job, that space can never be used; prune when the total
  waste exceeds `k * cap - total`. On this input the node count drops to 1043003, saving only about three
  quarters, and it is still exponential. At this scale the more realistic options are to switch to an integer
  programming solver, or to accept the greedy approximation.

<details>
<summary>Verification code (runnable)</summary>

```python
import itertools
import random
import time

A = [6, 9, 13, 15, 8, 9, 10, 15, 3, 3, 23, 30]
B = [31, 7, 13, 6, 30, 9, 9, 13, 31, 24, 40, 31]
C = [19, 19, 27, 27, 17, 16, 32, 34, 36, 28, 17, 29]


def is_valid(jobs, k, cap, assign):
    if assign is None or len(assign) != len(jobs) or any(not 0 <= g < k for g in assign):
        return False
    loads = [0] * k
    for size, g in zip(jobs, assign):
        loads[g] += size
    return max(loads, default=0) <= cap


def peak_of(jobs, k, assign):
    loads = [0] * k
    for size, g in zip(jobs, assign):
        loads[g] += size
    return max(loads, default=0)


def run(fn, *args, **kw):
    stats = {}
    out = fn(*args, stats=stats, **kw)
    return out, stats["nodes"]


# ---- independent brute forces ----
def brute_first(jobs, k, cap):
    """itertools.product yields assignments in lexicographic order, so the first valid one is the smallest."""
    for assign in itertools.product(range(k), repeat=len(jobs)):
        if is_valid(jobs, k, cap, list(assign)):
            return list(assign)
    return None


def brute_peak(jobs, k):
    return min(peak_of(jobs, k, a) for a in itertools.product(range(k), repeat=len(jobs)))


def subset_dp_peak(jobs, k):
    """f[mask] after j rounds: smallest peak placing the jobs in mask on j GPUs. O(k * 3^n)."""
    n = len(jobs)
    total = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        total[mask] = total[mask ^ low] + jobs[low.bit_length() - 1]
    f = total[:]
    for _ in range(k - 1):
        g = f[:]
        for mask in range(1 << n):
            sub = mask
            while sub:
                cand = max(f[mask ^ sub], total[sub])
                if cand < g[mask]:
                    g[mask] = cand
                sub = (sub - 1) & mask
        f = g
    return f[(1 << n) - 1]


# ---- every example in the statement ----
assert pack([3, 5, 2, 4], 2, 7) == [0, 1, 1, 0]
assert pack([3, 5, 2, 4], 3, 5) == [0, 1, 0, 2]
assert pack([3, 5, 2, 4], 2, 6) is None
assert pack([8], 2, 7) is None
assert pack([], 3, 1) == []
for jobs, k, cap in (([3, 5, 2, 4], 2, 7), ([3, 5, 2, 4], 3, 5), ([], 3, 1)):
    assert is_valid(jobs, k, cap, pack_fast(jobs, k, cap))
assert pack_fast([3, 5, 2, 4], 2, 6) is None and pack_fast([8], 2, 7) is None

assert run(pack, A, 4, 36)[1] == 66005 and is_valid(A, 4, 36, pack(A, 4, 36))
assert run(pack, B, 4, 61) == (None, 350105)
assert sum(A) == 4 * 36 and sum(B) == 4 * 61
A_groups = [[30, 6], [23, 13], [15, 15, 3, 3], [10, 9, 9, 8]]       # the solution quoted for A
assert sorted(A) == sorted(x for grp in A_groups for x in grp) and all(sum(grp) == 36 for grp in A_groups)
can_join_40 = sorted(x for x in B if x + 40 <= 61)
assert can_join_40 == [6, 7, 9, 9, 13, 13]
assert all(sum(c) != 21 for r in range(len(can_join_40) + 1) for c in itertools.combinations(can_join_40, r))
assert subset_dp_peak(B, 4) == 62                    # so cap = 61 really is impossible

# the table in Part 2: each pruning alone and both together
assert [run(pack_fast, A, 4, 36, sort_desc=s, skip_equal=e)[1] for s, e in ((0, 0), (0, 1), (1, 0), (1, 1))] == [66005, 15491, 13, 13]
assert [run(pack_fast, B, 4, 61, sort_desc=s, skip_equal=e)[1] for s, e in ((0, 0), (0, 1), (1, 0), (1, 1))] == [350105, 13356, 1577, 16]
assert is_valid(A, 4, 36, run(pack_fast, A, 4, 36)[0]) and run(pack_fast, B, 4, 61)[0] is None
assert run(pack_fast, A, 4, 36)[1] * 100 <= 66005 and run(pack_fast, B, 4, 61)[1] * 100 <= 350105

for jobs, k, want in (([3, 3, 2, 2, 2], 2, 6), ([3, 5, 2, 4], 3, 5), (A, 4, 36), (B, 4, 62), ([], 2, 0)):
    p, assign = min_peak(jobs, k)
    assert p == want and len(assign) == len(jobs) and peak_of(jobs, k, assign) == p, (jobs, k, p)
assert min_peak([], 2) == (0, [])

# the numbers quoted in Part 3
assert run(min_peak, A, 4)[1] == 13 and run(min_peak, A, 4, use_bound=False)[1] == 700662
assert run(min_peak, B, 4)[1] == 16 and run(min_peak, B, 4, use_bound=False)[1] == 624928
(c_peak, _), c_nodes = run(min_peak, C, 5)
(c_peak2, _), c_nodes_unbounded = run(min_peak, C, 5, use_bound=False)


def lpt(jobs, k):
    loads = [0] * k
    for size in sorted(jobs, reverse=True):
        loads[loads.index(min(loads))] += size
    return max(loads, default=0)


assert lpt(C, 5) == 69 and max(max(C), -(-sum(C) // 5)) == 61 and c_peak == c_peak2 == 62 == subset_dp_peak(C, 5)
assert c_nodes == 4424 and c_nodes_unbounded == 2412066
assert lpt([3, 3, 2, 2, 2], 2) == 7                  # the greedy is not optimal (follow-up 1)

# ---- edge cases ----
assert pack_fast([], 5, 3) == [] and run(pack_fast, [], 5, 3)[1] == 1
assert pack([7], 1, 7) == [0] and pack_fast([7], 1, 7) == [0] and min_peak([7], 1) == (7, [0])
assert min_peak([7], 4)[0] == 7 and min_peak([4, 4], 1)[0] == 8 and pack([4, 4], 1, 7) is None
assert min_peak([2, 9, 3], 14)[0] == 9               # more GPUs than jobs
assert run(pack_fast, [5] * 14, 4, 20)[0] is not None    # all equal: three or four jobs per GPU
assert min_peak([5] * 14, 4)[0] == 20
assert run(pack_fast, [5] * 14, 4, 19) == (None, 2035)   # at most 3 per GPU, 12 < 14; total 70 <= 76 passes the quick check
assert run(pack_fast, [10 ** 6] * 3, 2, 10 ** 7 - 1)[0] is not None and min_peak([10 ** 6] * 14, 14)[0] == 10 ** 6
assert run(pack_fast, [3, 3, 3], 2, 4) == (None, 0)      # the quick total check: 9 > 8, no search at all

# ---- random comparison with the brute forces ----
rng = random.Random(20261118)
saw_none = saw_some = saw_lex_differs = 0
for _ in range(3000):
    n = rng.randint(0, 6)
    k = rng.randint(1, 3)
    jobs = [rng.randint(1, rng.choice([2, 5, 12, 10 ** 6])) for _ in range(n)]
    best = brute_peak(jobs, k) if n else 0
    cap = max(1, best + rng.choice([-3, -1, 0, 0, 1, 4]))
    want = brute_first(jobs, k, cap)
    assert pack(jobs, k, cap) == want, (jobs, k, cap)
    fast = pack_fast(jobs, k, cap)
    assert (fast is None) == (want is None) and (fast is None or is_valid(jobs, k, cap, fast)), (jobs, k, cap)
    p, assign = min_peak(jobs, k)
    assert p == best and peak_of(jobs, k, assign) == p and len(assign) == n, (jobs, k)
    saw_none += want is None
    saw_some += want is not None
    saw_lex_differs += fast is not None and fast != want
assert min(saw_none, saw_some, saw_lex_differs) >= 100, (saw_none, saw_some, saw_lex_differs)

for _ in range(300):                                 # larger n against the subset DP
    n = rng.randint(7, 10)
    k = rng.randint(2, 5)
    jobs = [rng.randint(1, rng.choice([10, 50, 1000])) for _ in range(n)]
    best = subset_dp_peak(jobs, k)
    p, assign = min_peak(jobs, k)
    assert p == best and peak_of(jobs, k, assign) == best
    assert is_valid(jobs, k, best, pack_fast(jobs, k, best)) and pack_fast(jobs, k, best - 1) is None
    k_lpt = lpt(jobs, k)
    assert k_lpt * 3 * k <= best * (4 * k - 1)       # Graham's bound: LPT <= (4/3 - 1/(3k)) * optimum


# ---- node counts: how much the pruning saves, on random instances (no timing involved) ----
def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


rng = random.Random(8)
ratios = {True: [], False: []}
totals = {True: [0, 0], False: [0, 0]}
for _ in range(150):
    n = rng.randint(9, 11)
    k = rng.randint(3, 4)
    jobs = [rng.randint(5, 40) for _ in range(n)]
    best = min_peak(jobs, k)[0]
    for cap in (best - 1, best):
        (_, plain), (_, fast) = run(pack, jobs, k, cap), run(pack_fast, jobs, k, cap)
        ratios[cap == best].append(plain / max(fast, 1))     # fast is 0 when the quick total check already says no
        totals[cap == best][0] += plain
        totals[cap == best][1] += fast
infeasible, feasible = ratios[False], ratios[True]
print(f"infeasible (cap = best - 1): median {median(infeasible):.0f}x fewer nodes, total {totals[False][0]} -> {totals[False][1]}")
print(f"feasible   (cap = best):     median {median(feasible):.1f}x fewer nodes, total {totals[True][0]} -> {totals[True][1]}")
assert median(infeasible) >= 100 and min(infeasible) >= 1          # "a few hundred times", never worse
assert 2 <= median(feasible) <= 30 and min(feasible) < 1             # "a few times", and some instances get worse


# the Stirling-number bound on the nodes per level of the pruned search
def stirling_rows(n, k):
    S = [[0] * (k + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            S[i][j] = S[i - 1][j - 1] + j * S[i - 1][j]
    return [sum(row) for row in S]


assert sum(stirling_rows(12, 4)) == 934120 and sum(4 ** d for d in range(13)) == 22369621
for _ in range(200):
    n = rng.randint(6, 11)
    k = rng.randint(2, 5)
    jobs = [rng.randint(1, 30) for _ in range(n)]
    best = min_peak(jobs, k)[0]
    bound = sum(stirling_rows(n, k))
    assert run(pack_fast, jobs, k, best - 1)[1] <= bound and run(pack_fast, jobs, k, 10 ** 7)[1] <= bound


def full_tree_nodes(jobs, k, cap):
    """Nodes of the Part 2 tree (sorted order, one GPU per load value) when the search never stops early."""
    order = sorted(jobs, reverse=True)
    loads = [0] * k

    def count(pos):
        total = 1
        if pos < len(order):
            tried = set()
            for g in range(k):
                if loads[g] + order[pos] <= cap and loads[g] not in tried:
                    tried.add(loads[g])
                    loads[g] += order[pos]
                    total += count(pos + 1)
                    loads[g] -= order[pos]
        return total

    return count(0)


for _ in range(300):                                 # the bound on min_peak's nodes stated in Part 3
    n = rng.randint(1, 10)
    k = rng.randint(1, 5)
    jobs = [rng.randint(1, rng.choice([5, 30, 1000])) for _ in range(n)]
    assert run(min_peak, jobs, k)[1] <= full_tree_nodes(jobs, k, lpt(jobs, k) - 1)


# ---- follow-ups ----
def min_peak_bisect(jobs, k):
    lo, hi = max(max(jobs, default=0), -(-sum(jobs) // k)), lpt(jobs, k)
    while lo < hi:                                   # invariant: the answer lies in [lo, hi]
        mid = (lo + hi) // 2
        if pack_fast(jobs, k, mid) is not None:
            hi = mid
        else:
            lo = mid + 1
    return lo


def pack_memo(jobs, k, cap, stats=None):
    """pack_fast plus a set of dead states (pos, sorted loads): backtracking that remembers is a DP."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i])
    loads, assign, dead, nodes = [0] * k, [0] * n, set(), 0

    def place(pos):
        nonlocal nodes
        nodes += 1
        if pos == n:
            return True
        key = (pos, tuple(sorted(loads)))
        if key in dead:
            return False
        i, tried = order[pos], set()
        for g in range(k):
            if loads[g] + jobs[i] > cap or loads[g] in tried:
                continue
            tried.add(loads[g])
            loads[g] += jobs[i]
            assign[i] = g
            if place(pos + 1):
                return True
            loads[g] -= jobs[i]
        dead.add(key)
        return False

    found = sum(jobs) <= k * cap and max(jobs, default=0) <= cap and place(0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None


def pack_waste(jobs, k, cap, stats=None):
    """pack_fast plus the wasted-space bound from the last follow-up."""
    n = len(jobs)
    order = sorted(range(n), key=lambda i: -jobs[i])
    loads, assign, nodes = [0] * k, [0] * n, 0
    smallest, allowed = min(jobs, default=0), k * cap - sum(jobs)

    def place(pos, waste):
        nonlocal nodes
        nodes += 1
        if pos == n:
            return True
        i, tried = order[pos], set()
        for g in range(k):
            new = loads[g] + jobs[i]
            if new > cap or loads[g] in tried:
                continue
            tried.add(loads[g])
            lost = cap - new if cap - new < smallest else 0      # no remaining job fits in what is left
            if waste + lost > allowed:
                continue
            loads[g] = new
            assign[i] = g
            if place(pos + 1, waste + lost):
                return True
            loads[g] = new - jobs[i]
        return False

    found = allowed >= 0 and max(jobs, default=0) <= cap and place(0, 0)
    if stats is not None:
        stats["nodes"] = nodes
    return assign if found else None


for _ in range(1500):
    n = rng.randint(0, 10)
    k = rng.randint(1, 4)
    jobs = [rng.randint(1, rng.choice([2, 3, 10, 40])) for _ in range(n)]
    cap = rng.randint(1, 45)
    want_none = pack_fast(jobs, k, cap) is None
    for fn in (pack_memo, pack_waste):
        out, nodes = run(fn, jobs, k, cap)
        assert (out is None) == want_none and (out is None or is_valid(jobs, k, cap, out)), (fn.__name__, jobs, k, cap)
        assert nodes <= run(pack_fast, jobs, k, cap)[1]
    if n:
        assert min_peak_bisect(jobs, k) == min_peak(jobs, k)[0]

assert run(pack_memo, [5] * 14, 4, 19) == (None, 61)         # follow-up 3: 2035 -> 61 when states repeat

HARD = [1006, 1012, 1016, 1027, 1022, 1033, 1016, 1031, 1040, 1045, 1041, 1005, 1007, 1044, 1049, 1007]
hard_best = 4025
assert min(HARD) == 1005 and max(HARD) == 1049
assert is_valid(HARD, 5, hard_best, pack_fast(HARD, 5, hard_best))  # 4025 fits; 4024 does not (below)
t0 = time.perf_counter()
hard_out, hard_nodes = run(pack_fast, HARD, 5, hard_best - 1)
t1 = time.perf_counter()
waste_out, waste_nodes = run(pack_waste, HARD, 5, hard_best - 1)
t2 = time.perf_counter()
print(f"n = 16 near-equal, cap = best - 1: pack_fast {hard_nodes} nodes ({t1 - t0:.1f} s), "
      f"with the wasted-space bound {waste_nodes} nodes ({t2 - t1:.1f} s)")
assert hard_out is None and waste_out is None
assert hard_nodes == 3904649 and waste_nodes == 1043003

# ---- the largest size in the stated range (n = 14): the hardest inputs a random search found ----
WORST_PACK = ([130, 129, 126, 100, 127, 100, 128, 109, 149, 131, 123, 148, 125, 137], 4)
WORST_BB = ([66, 32, 48, 35, 26, 47, 64, 70, 67, 52, 39, 68, 36, 54], 6)
big_random = ([rng.randint(1, 10 ** 6) for _ in range(14)], 14)
for jobs, k in (WORST_PACK, WORST_BB, big_random, ([5] * 14, 4)):
    t0 = time.perf_counter()
    (best, assign), bb_nodes = run(min_peak, jobs, k)
    (yes, yes_nodes), (no, no_nodes) = run(pack_fast, jobs, k, best), run(pack_fast, jobs, k, best - 1)
    print(f"n = 14, k = {k}: peak {best}, min_peak {bb_nodes} nodes, pack_fast {yes_nodes} / {no_nodes} nodes, "
          f"{time.perf_counter() - t0:.2f} s")
    assert peak_of(jobs, k, assign) == best and is_valid(jobs, k, best, yes) and no is None
print("all checks passed")
```

</details>

</details>
