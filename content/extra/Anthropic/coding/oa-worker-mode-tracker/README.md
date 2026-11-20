# Online Assessment: Worker Mode Tracker

## Problem

Many workers each report integers (think of each worker as a data shard). Implement a `ModeTracker` class that
aggregates their data and can answer at any time "which value occurs most often globally". It must also be able
to remove a worker entirely: everything that worker ever reported has to be erased from the counts.

The problem has four levels, unlocked in order. Each level adds methods to the same `ModeTracker` class, and the
methods from earlier levels keep working.

Conventions that apply to all four levels:

- A worker id is a non-empty string and is case-sensitive.
- Reported values are integers, $-10^9 \le \text{value} \le 10^9$.
- In a single test there are at most $10^3$ method calls in total, and all `report` calls together report at most
  $10^4$ values.
- A failed call does not change any state.
- The examples of the four levels below are continuous: each level picks up from the state at the end of the
  previous one.

### Level 1 — Collect data from the workers

```py
class ModeTracker:
    def add_worker(self, worker_id: str) -> bool:
        """Registers a worker. False if a worker with this id is registered already."""

    def report(self, worker_id: str, values: list[int]) -> bool:
        """The worker reports these values (1 to 1000 of them). Every list element counts once,
        so a value that appears twice in the list is counted twice.
        False, and nothing is recorded, if the worker is not registered."""

    def get_count(self, value: int) -> int:
        """How many times `value` has been reported, summed over all registered workers.
        0 if nobody reported it."""

    def get_worker_count(self, worker_id: str, value: int) -> int | None:
        """How many times this worker has reported `value` (possibly 0).
        None if the worker is not registered."""
```

```text
t = ModeTracker()
t.add_worker("w1")                -> True
t.add_worker("w1")                -> False     # already registered
t.add_worker("w2")                -> True
t.report("w1", [5, 3, 5])         -> True
t.report("w2", [3, 3, 9])         -> True
t.report("w9", [1])               -> False     # w9 is not registered, so this report is not recorded
t.get_count(3)                    -> 3         # w1 reported it once, w2 twice
t.get_count(5)                    -> 2
t.get_count(1)                    -> 0
t.get_worker_count("w1", 5)       -> 2
t.get_worker_count("w2", 5)       -> 0         # registered, but never reported 5
t.get_worker_count("w9", 5)       -> None      # not registered
```

### Level 2 — Global mode

```py
    def get_mode(self) -> int | None:
        """The value with the highest global count. If several values share the highest count,
        the smallest of them. None if there is no data at all."""

    def top_values(self, k: int) -> list[str]:
        """The k values with the highest global counts, each formatted as "value(count)",
        ordered by count descending, then by value ascending. Fewer than k values: all of them."""
```

- Ties are compared as numbers, not strings: `2` comes before `10`, and `-4` comes before `2`.
- A value whose global count is 0 is not data and never appears in any result.
- `k >= 1`. With no data, `top_values` returns `[]`.
- The format is exactly Python's `f"{value}({count})"`; a negative number is written as `"-4(2)"`.

Continuing from the state after Level 1 (global counts: 3 occurs 3 times, 5 occurs 2 times, 9 occurs once):

```text
t.get_mode()                      -> 3
t.top_values(2)                   -> ["3(3)", "5(2)"]
t.top_values(10)                  -> ["3(3)", "5(2)", "9(1)"]
t.report("w1", [5])               -> True
t.get_mode()                      -> 3         # 3 and 5 both occur 3 times; take the smaller value
t.top_values(1)                   -> ["3(3)"]
```

### Level 3 — Remove a worker

```py
    def remove_worker(self, worker_id: str) -> int | None:
        """Unregisters the worker and erases everything it ever reported from all counts.
        Returns how many reported values were erased (0 for a worker that never reported).
        None if the worker is not registered."""
```

- After a removal, every query returns the same result as if this worker had never existed.
- The id can be passed to `add_worker` again later, which creates a brand-new worker with no data.
- Note that `0` and `None` are not the same thing: removing a worker that is registered but never reported anything
  returns `0`.

Continuing from the state above (w1 has reported 5, 3, 5, 5; w2 has reported 3, 3, 9):

```text
t.remove_worker("w2")             -> 3         # erase w2's 3 values: 3, 3, 9
t.get_count(3)                    -> 1
t.get_count(9)                    -> 0
t.get_mode()                      -> 5
t.top_values(10)                  -> ["5(3)", "3(1)"]     # the count of 9 is back to 0, so it no longer appears
t.remove_worker("w2")             -> None      # already gone
t.report("w2", [9])               -> False
t.add_worker("w2")                -> True      # registered again: a brand-new worker
t.get_worker_count("w2", 3)       -> 0
```

### Level 4 — Merge workers

(We added this level: the source only has points corresponding to the first three levels. "Merge" is one of the
common forms the fourth level of this kind of assessment takes.)

```py
    def merge_workers(self, target_id: str, source_id: str) -> bool:
        """Everything `source_id` has reported now counts as reported by `target_id`,
        and `source_id` is unregistered. False if either worker is not registered
        or the two ids are equal."""
```

- A merge does not change any global count; the results of `get_mode` and `top_values` stay the same.
- After a merge, `remove_worker(target_id)` erases the merged-in data as well.
- `source_id` can be registered again later, as a brand-new worker.

Continuing from the state above (w1 has reported 5, 3, 5, 5; w2 was just registered again and has no data):

```text
t.report("w2", [7, 7, 5])         -> True
t.add_worker("w3")                -> True
t.report("w3", [7])               -> True
t.merge_workers("w1", "w2")       -> True
t.get_worker_count("w1", 7)       -> 2         # merged in from w2
t.get_worker_count("w1", 5)       -> 4         # its own 3 + 1 merged in
t.get_worker_count("w2", 7)       -> None      # w2 is no longer a registered worker
t.get_count(7)                    -> 3         # global counts are not affected by the merge
t.get_mode()                      -> 5         # 5 occurs 4 times, 7 occurs 3 times
t.merge_workers("w1", "w2")       -> False     # w2 does not exist
t.merge_workers("w1", "w1")       -> False     # the two ids are equal
t.remove_worker("w1")             -> 7         # its own 4 values + the 3 merged in
t.get_mode()                      -> 7         # only the single 7 reported by w3 is left
t.top_values(5)                   -> ["7(1)"]
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

The key to the whole problem is Level 3: to be able to remove a worker, you have to remember what **each worker**
reported on its own. Storing only the global counts is not enough, because at removal time you would not know how
much to subtract from each value. So from Level 1 on, store two layers:

- `by_worker[worker_id]`: this worker's count table, value → count.
- `total`: the global count table, value → count.

The invariant to maintain: for every value, `total[value]` equals the sum of `by_worker[w][value]` over all
registered workers; values with count 0 are not kept in `total`. Global counts are modified only in the single
function `_add`, and every later level goes through it to change them.

Below, each level adds its features in a subclass, which makes it easy to show the solution in stages. In the real
assessment you would simply add methods to the same class.

### Level 1

For each value in the list, `report` adds 1 to the worker's own count and 1 to the global count.
`get_worker_count` looks it up directly in the `Counter`: a `Counter` returns 0 for a missing key, and a lookup
does not create an extra entry.

Complexity: `report` is $O(m)$, where $m$ is the number of values in this report; every other operation is
$O(1)$ on average.

```python
from collections import Counter


class TrackerL1:
    def __init__(self):
        self.by_worker = {}                  # worker id -> Counter: value -> how often this worker reported it
        self.total = {}                      # value -> global count; values with count 0 are not kept

    def add_worker(self, worker_id):
        if worker_id in self.by_worker:
            return False
        self.by_worker[worker_id] = Counter()
        return True

    def report(self, worker_id, values):
        mine = self.by_worker.get(worker_id)
        if mine is None:
            return False
        for value in values:
            mine[value] += 1
            self._add(value, 1)
        return True

    def get_count(self, value):
        return self.total.get(value, 0)

    def get_worker_count(self, worker_id, value):
        mine = self.by_worker.get(worker_id)
        return None if mine is None else mine[value]         # a Counter answers 0 for a missing key

    def _add(self, value, delta):
        """The only place that changes a global count."""
        count = self.total.get(value, 0) + delta
        if count:
            self.total[value] = count
        else:
            del self.total[value]                            # back to 0: the value is gone
```

### Level 2

Compute directly on `total`. The mode is "highest count, then smallest value", written as taking the minimum of
the key `(-count, value)`. `top_values` sorts by the same key. Since values with count 0 are not in `total`,
neither method needs any extra filtering.

With $V$ distinct values currently present: `get_mode` is one pass, $O(V)$, and `top_values` sorts, $O(V \log V)$.
At the problem's scale (at most $10^3$ calls, at most $10^4$ distinct values), all calls together cost at worst
$10^7$ to $10^8$ basic operations (the sort in `top_values` also multiplies by $\log V$), which is good enough,
and this way of writing it is the hardest to get wrong. With more data it would not hold up; a faster version is in
the follow-ups below.

```python
class TrackerL2(TrackerL1):
    def get_mode(self):
        if not self.total:
            return None
        return min(self.total, key=lambda value: (-self.total[value], value))    # highest count, then smallest value

    def top_values(self, k):
        ranked = sorted(self.total.items(), key=lambda item: (-item[1], item[0]))
        return [f"{value}({count})" for value, count in ranked[:k]]
```

### Level 3

Take out this worker's whole count table, and for each value in it, subtract the count it contributed from the
global count. A value that drops to 0 is deleted from `total` by `_add`, so it no longer appears in
`top_values`. The return value is the sum of all counts in this table.

Because `get_mode` recomputes from `total` every time, the mode is automatically correct after a removal. If you
had cached the "current mode" in Level 2, you would need to be careful at this level: a removal can make the old
mode drop, and the cache has no information about the "runner-up".

Complexity: $O(d)$, where $d$ is the number of distinct values this worker reported.

```python
class TrackerL3(TrackerL2):
    def remove_worker(self, worker_id):
        mine = self.by_worker.pop(worker_id, None)
        if mine is None:
            return None
        for value, count in mine.items():
            self._add(value, -count)                         # take back exactly what this worker put in
        return sum(mine.values())
```

### Level 4

Add `source`'s count table into `target`'s count table, then remove `source` from the registry. Global counts
don't need to change: this data was counted globally before, and it still is. `Counter.update` adds counts; a
plain `dict`'s `update` overwrites, and using it by mistake loses data.

Complexity: $O(d)$, where $d$ is the number of distinct values `source` reported.

```python
class TrackerL4(TrackerL3):
    def merge_workers(self, target_id, source_id):
        if target_id == source_id or target_id not in self.by_worker or source_id not in self.by_worker:
            return False
        self.by_worker[target_id].update(self.by_worker.pop(source_id))      # Counter.update adds the counts
        return True                                                          # global counts do not change


ModeTracker = TrackerL4
```

### Follow-ups

**There is a lot of data, and the mode is queried after every report, so `get_mode` can't scan the whole table each
time. What do you do?**

With reports only, it is easy: remember the current mode, and after each increment compare against it. The hard
part is removal: one removal can lower the counts of many values at once, the old mode may no longer be first, and
then you need to know "who is next". The verification code below includes a version that only remembers "the
current best", and it gives wrong answers after a removal.

One approach that is very short to write is a **heap with lazy deletion**. The heap holds `(-count, value)`.
Whenever a value's count changes (up or down), push its new count onto the heap; don't look for or delete the old
entry, just leave it in the heap as stale. On a query, look at the top of the heap: if the count recorded at the top
differs from that value's current true count, the entry is stale, so pop it and look at the next one; the first
top that is not stale is the answer.

Why this is correct: every value that still has data pushed a "current count" entry at its most recent change, so
every value has one valid entry in the heap. Among the valid entries, the one with the smallest key is the value with
the highest count and, among those, the smallest value. Stale entries are popped only when they reach the top, and
don't affect the result.

One more thing to guard against: if stale entries stay buried deep in the heap, the heap keeps growing. So when the
heap size exceeds "twice the number of live values plus some slack", rebuild a heap containing only valid entries
directly from `total`.

Complexity ($V$ is the current number of distinct values): the heap size is always $O(V)$, so pushing an entry
for each count change is $O(\log V)$. Each entry is popped at most once, and the cost of popping can be charged
to the push. A rebuild is $O(V)$, and at least $V$ count changes happen between two rebuilds, so amortized it is
$O(1)$ per change. Thus `get_mode` is amortized $O(\log V)$, `report` is $O(m \log V)$, and `remove_worker` is
$O(d \log V)$. `top_values` still sorts.

```python
import heapq


class FastModeTracker(TrackerL4):
    """get_mode in amortised O(log V): a max-heap of (count, value) with lazy deletion."""

    SLACK = 64                               # rebuild once the heap holds more than 2 * live values + SLACK entries

    def __init__(self):
        super().__init__()
        self.heap = []                       # (-count, value); some entries may be out of date

    def _add(self, value, delta):
        super()._add(value, delta)
        if value in self.total:
            heapq.heappush(self.heap, (-self.total[value], value))           # after every change, up or down
        if len(self.heap) > 2 * len(self.total) + self.SLACK:                # mostly stale: rebuild from scratch
            self.heap = [(-count, v) for v, count in self.total.items()]
            heapq.heapify(self.heap)

    def get_mode(self):
        while self.heap and self.total.get(self.heap[0][1], 0) != -self.heap[0][0]:
            heapq.heappop(self.heap)                                         # this count is no longer true
        return self.heap[0][1] if self.heap else None
```

**Would buckets of the form "count → set of values" work?** Yes. Also keep a "current maximum count": a report
raises it by at most 1; after a removal, if the top bucket is empty, step down one level at a time, and the number of
steps down can't exceed the number of steps that earlier reports raised it, so amortized it is $O(1)$. The trouble is
that ties must take the smallest value in the bucket: the buckets have to be ordered, or each bucket needs its own
heap. Python's standard library has no sorted set, so the lazy-deletion heap is shorter to write here.

**Can memory be saved?** Not much, if the counts must be exact: to support removal, every worker's count table has
to be kept, and memory is the sum over workers of their number of distinct values. If approximate counts are
enough, each worker can maintain a Count-Min sketch. It is a fixed-size table of counters, and two tables that use
the same set of hash functions can be added cell by cell, and also subtracted cell by cell, so "remove a worker"
just means subtracting its table from the total table. However, it can only answer "roughly how many times did a
given value occur"; which values are the candidate heavy hitters has to be tracked separately. Summaries like
Misra–Gries that keep only a few candidate values can be merged, but their standard form does not support
subtraction, so they don't help here.

**In a real distributed environment, reports may be resent. How do you avoid double counting?** An increment like
"add 1" is not idempotent: resending the same batch once counts it one extra time. Two common approaches: one,
each worker numbers its batches with increasing sequence numbers, the aggregator remembers which sequence number it
has processed up to for each worker, and discards any sequence number it has already seen; two, each worker reports
its **cumulative** count table instead of an increment, and the aggregator replaces that worker's old table with the
new one (updating the global counts by the difference), so the result is the same no matter how many times a message
is resent, at the cost of larger messages. In this problem, the structure of "one separate table per worker"
supports the second approach directly. Also, if the data stays on the shards and the global mode can only be found
by passing messages, that is a different problem; see [Distributed Mode, Median and Sort](#/anthropic/coding/distributed-mode-median).

**How do you make `top_values(k)` faster?** For a single call, `heapq.nsmallest(k, total.items(), key=...)` is
$O(V \log k)$, cheaper than a full sort. If it is called very often, you need a structure kept sorted by
`(-count, value)` that supports insertion and deletion (a balanced tree, or the third-party
`sortedcontainers.SortedList`), updated once per count change, so taking the first $k$ needs no further sorting.

<details>
<summary>Checks (runnable)</summary>

```python
import random

# --- 1. the statement's examples, level by level, on one tracker (both implementations) ---
for cls in (ModeTracker, FastModeTracker):
    t = cls()
    assert t.get_mode() is None and t.top_values(3) == [] and t.get_count(5) == 0
    # Level 1
    assert t.add_worker("w1") is True
    assert t.add_worker("w1") is False
    assert t.add_worker("w2") is True
    assert t.report("w1", [5, 3, 5]) is True
    assert t.report("w2", [3, 3, 9]) is True
    assert t.report("w9", [1]) is False
    assert (t.get_count(3), t.get_count(5), t.get_count(1)) == (3, 2, 0)
    assert t.get_worker_count("w1", 5) == 2
    assert t.get_worker_count("w2", 5) == 0
    assert t.get_worker_count("w9", 5) is None
    # Level 2
    assert t.get_mode() == 3
    assert t.top_values(2) == ["3(3)", "5(2)"]
    assert t.top_values(10) == ["3(3)", "5(2)", "9(1)"]
    assert t.report("w1", [5]) is True
    assert t.get_mode() == 3                                 # 3 and 5 both have count 3: the smaller value wins
    assert t.top_values(1) == ["3(3)"]
    # Level 3
    assert t.remove_worker("w2") == 3
    assert (t.get_count(3), t.get_count(9)) == (1, 0)
    assert t.get_mode() == 5
    assert t.top_values(10) == ["5(3)", "3(1)"]
    assert t.remove_worker("w2") is None
    assert t.report("w2", [9]) is False
    assert t.add_worker("w2") is True
    assert t.get_worker_count("w2", 3) == 0
    assert t.remove_worker("w2") == 0                        # registered but silent: 0, not None
    assert t.add_worker("w2") is True
    # Level 4
    assert t.report("w2", [7, 7, 5]) is True
    assert t.add_worker("w3") is True
    assert t.report("w3", [7]) is True
    assert t.merge_workers("w1", "w2") is True
    assert (t.get_worker_count("w1", 7), t.get_worker_count("w1", 5)) == (2, 4)
    assert t.get_worker_count("w2", 7) is None
    assert t.get_count(7) == 3
    assert t.get_mode() == 5
    assert t.merge_workers("w1", "w2") is False              # w2 is gone
    assert t.merge_workers("w1", "w1") is False
    assert t.merge_workers("w9", "w1") is False
    assert t.remove_worker("w1") == 7                        # 4 reports of its own + 3 merged in from w2
    assert t.get_mode() == 7
    assert t.top_values(5) == ["7(1)"]
    assert t.remove_worker("w3") == 1
    assert t.get_mode() is None and t.top_values(5) == []

# --- 2. details of the stated rules ---
for cls in (ModeTracker, FastModeTracker):
    t = cls()
    t.add_worker("a")
    t.report("a", [-4, -4, 10, 2, 10])
    assert t.get_mode() == -4                                # ties are broken by the number, not by its text
    assert t.top_values(3) == ["-4(2)", "10(2)", "2(1)"]     # as text "10" < "2", as numbers 2 < 10
    assert t.get_worker_count("a", 777) == 0
    assert 777 not in t.by_worker["a"]                       # asking did not create an entry
    t.add_worker("b")
    t.report("b", [2, 2])                                    # 2 now has count 3
    assert t.get_mode() == 2
    assert t.merge_workers("b", "a") is True                 # merging changes no global count
    assert t.top_values(9) == ["2(3)", "-4(2)", "10(2)"]
    assert t.add_worker("a") is True                         # the id is free again and starts empty
    assert t.get_worker_count("a", 2) == 0 and t.get_worker_count("b", 2) == 3


# --- 3. random sequences against a model that keeps the raw reports and recomputes everything ---
class LogTracker:
    def __init__(self):
        self.workers = []                # registered ids
        self.log = []                    # [worker id, value], one item per reported value

    def add_worker(self, w):
        if w in self.workers:
            return False
        self.workers.append(w)
        return True

    def report(self, w, values):
        if w not in self.workers:
            return False
        self.log.extend([w, v] for v in values)
        return True

    def get_count(self, value):
        return len([1 for _, v in self.log if v == value])

    def get_worker_count(self, w, value):
        if w not in self.workers:
            return None
        return len([1 for who, v in self.log if who == w and v == value])

    def ranking(self):
        values = sorted({v for _, v in self.log})
        values.sort(key=self.get_count, reverse=True)        # stable: equal counts keep ascending values
        return values

    def get_mode(self):
        ranking = self.ranking()
        return ranking[0] if ranking else None

    def top_values(self, k):
        return ["%d(%d)" % (v, self.get_count(v)) for v in self.ranking()[:k]]

    def remove_worker(self, w):
        if w not in self.workers:
            return None
        erased = len([1 for who, _ in self.log if who == w])
        self.log = [item for item in self.log if item[0] != w]
        self.workers.remove(w)
        return erased

    def merge_workers(self, target, source):
        if target == source or target not in self.workers or source not in self.workers:
            return False
        for item in self.log:
            if item[0] == source:
                item[0] = target
        self.workers.remove(source)
        return True


class EagerRebuildTracker(FastModeTracker):
    SLACK = 0                            # rebuild as often as possible, to exercise that path


WORKERS = ["w1", "w2", "w3", "w4"]
OPS = (["add_worker"] * 3 + ["report"] * 8 + ["get_count"] + ["get_worker_count"] + ["get_mode"] * 4
       + ["top_values"] * 2 + ["remove_worker"] * 2 + ["merge_workers"])


def random_call(rng):
    op = rng.choice(OPS)
    if op == "report":
        return op, (rng.choice(WORKERS), [rng.randint(-3, 5) for _ in range(rng.randint(1, 6))])
    if op == "get_count":
        return op, (rng.randint(-3, 5),)
    if op == "get_worker_count":
        return op, (rng.choice(WORKERS), rng.randint(-3, 5))
    if op == "get_mode":
        return op, ()
    if op == "top_values":
        return op, (rng.randint(1, 10),)
    if op == "merge_workers":
        return op, (rng.choice(WORKERS), rng.choice(WORKERS))
    return op, (rng.choice(WORKERS),)


rng = random.Random(71)
seen = {}
mode_changed_by_removal = 0
for _ in range(600):
    trackers = [LogTracker(), ModeTracker(), FastModeTracker(), EagerRebuildTracker()]
    for _ in range(rng.randint(1, 100)):
        op, args = random_call(rng)
        mode_before = trackers[0].get_mode()
        results = [getattr(t, op)(*args) for t in trackers]
        assert results[1:] == results[:1] * 3, (op, args, results)
        seen[op, type(results[0]).__name__, bool(results[0])] = True
        if op == "remove_worker" and results[0] and trackers[0].get_mode() != mode_before:
            mode_changed_by_removal += 1
        modes = [t.get_mode() for t in trackers]
        assert modes[1:] == modes[:1] * 3, (op, args, modes)
        for fast in trackers[2:]:
            assert len(fast.heap) <= 2 * len(fast.total) + fast.SLACK + 1    # the heap stays close to the live values
            assert all(count > 0 for count in fast.total.values())
for case in [("add_worker", "bool", True), ("add_worker", "bool", False), ("report", "bool", True),
             ("report", "bool", False), ("get_worker_count", "NoneType", False), ("get_worker_count", "int", True),
             ("get_mode", "NoneType", False), ("remove_worker", "NoneType", False), ("remove_worker", "int", True),
             ("remove_worker", "int", False), ("merge_workers", "bool", True), ("merge_workers", "bool", False)]:
    assert case in seen, case                                # every kind of answer really occurred
assert mode_changed_by_removal >= 100, mode_changed_by_removal


# --- 4. negative control: "just remember the current best" survives reports but not removals ---
class RememberBestTracker(TrackerL4):
    def __init__(self):
        super().__init__()
        self.best = None

    def _add(self, value, delta):
        super()._add(value, delta)
        if value in self.total and (self.best is None or
                                    (-self.total[value], value) < (-self.total.get(self.best, 0), self.best)):
            self.best = value

    def get_mode(self):
        return self.best


for cls, after_reports, after_removal in ((RememberBestTracker, 1, 1), (ModeTracker, 1, 2), (FastModeTracker, 1, 2)):
    t = cls()
    t.add_worker("a"), t.add_worker("b")
    t.report("a", [1, 1, 1]), t.report("b", [2, 2])
    assert t.get_mode() == after_reports
    t.remove_worker("a")                                     # the true mode is now 2
    assert t.get_mode() == after_removal, cls.__name__

# --- 5. the lazy heap: a count that goes down right after a rebuild must still be found ---
t = FastModeTracker()
for name in "abcd":
    t.add_worker(name)
t.report("a", [1, 1, 1]), t.report("b", [1, 1]), t.report("d", [2, 2, 2, 2])
heap_before = t.heap
t.report("c", [3] * 200)
t.remove_worker("c")
assert t.heap is not heap_before                             # a rebuild replaced the list: only current counts are in it
assert t.get_mode() == 1                                     # 1 has count 5, 2 has count 4
t.remove_worker("a")                                         # the count of 1 drops from 5 to 2
assert t.get_mode() == 2
t.remove_worker("d")                                         # 2 disappears
assert t.get_mode() == 1 and t.top_values(3) == ["1(2)"]     # needs the entry pushed when the count went down

# --- 6. the lazy heap under a long run: stale entries on top, counts that return to an old value, rebuilds ---
t, slow = FastModeTracker(), ModeTracker()
for tracker in (t, slow):
    tracker.add_worker("keep")
    tracker.report("keep", [0])
rebuilds = 0
for round_no in range(300):
    name = f"tmp{round_no % 7}"
    batch = [rng.randint(1, 40) for _ in range(rng.randint(1, 30))]
    heap_before = t.heap
    for tracker in (t, slow):
        tracker.add_worker(name)
        tracker.report(name, batch)
    assert t.get_mode() == slow.get_mode()
    if round_no % 3 == 2:
        for tracker in (t, slow):
            for k in range(7):
                tracker.remove_worker(f"tmp{k}")
        assert t.get_mode() == slow.get_mode() == 0          # only the single report of "keep" is left
        assert len(t.heap) <= 2 * len(t.total) + 65
    rebuilds += t.heap is not heap_before                    # a rebuild replaces the list object
assert rebuilds >= 50, rebuilds
print("all checks passed")
```

</details>

</details>
