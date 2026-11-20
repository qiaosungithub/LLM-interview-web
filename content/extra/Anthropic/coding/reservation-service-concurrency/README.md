# Reservation Service with Concurrency Constraints

## Problem

Implement an in-memory reservation service. The service manages a fixed set of *resources* (meeting rooms, GPUs,
and so on). Resource names are strings, given all at once in the constructor; none are added or removed later. Time
is an integer (think of it as "minute number"). A booking occupies a **half-open** interval `[start, end)` on one
resource.

The following rules apply to all three parts:

- Intervals `[a, b)` and `[c, d)` **overlap** if and only if `a < d` and `c < b`. Touching end to end does not
  count as overlapping: `[9, 11)` and `[11, 12)` can coexist.
- On the same resource, no two active bookings may overlap. Different resources are independent.
- `0 <= start < end <= 10**9`. Raise `ValueError` when `start >= end`, and `KeyError` when the resource name does
  not exist. A call that raises must not change any state. If both errors apply, raising either one is fine.
- A successful booking returns a booking id. Ids start at 1 and increase by 1 with each success; failed calls do
  not consume an id.
- `owner` is just a label and plays no part in any decision; anyone can cancel a booking by its id.

### Part 1 — Single-threaded version

```py
class ReservationService:
    def __init__(self, resources: list[str]): ...

    def reserve(self, resource: str, start: int, end: int, owner: str) -> int | None:
        """Book [start, end) on `resource`. Returns the new booking id,
        or None if it overlaps a booking that is already there."""

    def cancel(self, booking_id: int) -> bool:
        """True if the booking existed and is now cancelled.
        False if the id was never issued or is already cancelled."""

    def query(self, resource: str, start: int, end: int) -> list[tuple[int, int, int, str]]:
        """The bookings on `resource` that overlap [start, end),
        as (start, end, booking_id, owner), sorted by start."""
```

A resource may hold tens of thousands of bookings, so when `reserve` checks for a conflict it must not scan all of
that resource's bookings; it should make only $O(\log n)$ comparisons ($n$ is the number of active bookings on that
resource).

```text
svc = ReservationService(["room-a", "room-b"])
svc.reserve("room-a", 9, 11, "ann")     -> 1
svc.reserve("room-a", 11, 12, "bob")    -> 2       # [9,11) and [11,12) only touch end to end
svc.reserve("room-a", 10, 13, "cat")    -> None    # overlaps both 1 and 2
svc.reserve("room-b", 10, 13, "cat")    -> 3       # a different resource; the previous failure used no id
svc.query("room-a", 0, 24)              -> [(9, 11, 1, "ann"), (11, 12, 2, "bob")]
svc.query("room-a", 11, 12)             -> [(11, 12, 2, "bob")]
svc.query("room-a", 12, 24)             -> []
svc.cancel(1)                           -> True
svc.cancel(1)                           -> False   # already cancelled
svc.cancel(99)                          -> False   # an id that was never issued
svc.reserve("room-a", 10, 11, "cat")    -> 4
svc.query("room-a", 0, 24)              -> [(10, 11, 4, "cat"), (11, 12, 2, "bob")]
svc.reserve("room-c", 1, 2, "dan")      -> KeyError
svc.reserve("room-a", 5, 5, "dan")      -> ValueError
```

### Part 2 — Many threads calling at once

Now many threads call `reserve`, `cancel` and `query` on the same object concurrently. The service as a whole must
behave as if these calls ran one at a time in some order, and a call that starts after another call has returned
must come after it and see all of its effects (this is called linearizable). Specifically:

1. **No double bookings.** At any moment, the active bookings on one resource are pairwise non-overlapping. When
   several threads race for the same slot, exactly one gets an id and the rest get `None`.
2. **Ids are unique and have no gaps.** If there were $K$ successes in total, the ids issued are exactly $1..K$.
3. **A cancel takes effect exactly once.** When several threads `cancel` the same id at once, exactly one returns
   `True`. If a thread's `cancel` returns `False` because "someone else already cancelled it," then a `reserve`
   that starts after it returns must see the slot as free.
4. **A query is a snapshot.** `query` returns a state that really existed at some moment, and the returned list is
   not affected by later operations.
5. **Different resources don't wait on each other.** You may not wrap the whole of `reserve` in one global lock:
   later on the critical section may contain slow operations, such as writing to a database, and someone booking
   room A should not queue behind someone booking room B. A small lock that protects only very short operations,
   such as the id counter, is exempt from this rule.
6. No lock may be left held when an exception is raised.

In a real interview there are no ready-made tests. Write a test yourself that proves that the Part 1 code, used
as is from multiple threads, really does book the same slot to two people. It must **reproduce on every run**,
not by luck; then use the same test to show that the new version does not have this problem.

### Part 3 — Booking several resources at once

A meeting needs one room plus two GPUs, and can't happen if any one is missing. Add:

```py
    def reserve_all(self, requests: list[tuple[str, int, int]], owner: str) -> int | None:
        """Book every (resource, start, end) in `requests`, or none of them.
        One booking id covers all of them."""
```

- **All or nothing.** If any leg conflicts with an existing booking, return `None` and book nothing. Other threads
  must not see a "half-booked" intermediate state, nor be blocked by one.
- The same resource may appear more than once in one request (e.g. the same room in the morning and the
  afternoon). If two legs within the request overlap each other, the whole request fails and returns `None`.
- `requests` has 1 to 16 legs; raise `ValueError` if it is empty. If any leg has an invalid resource name or
  interval, raise as per the rules above and book nothing.
- The whole request uses a single id. Each of its legs carries that id in `query`; `cancel(id)` cancels all legs
  at once and still takes effect exactly once.
- `reserve(r, s, e, owner)` behaves exactly like `reserve_all([(r, s, e)], owner)`.
- **No deadlock.** If one thread calls `reserve_all([("a", …), ("b", …)], …)` while another simultaneously calls
  `reserve_all([("b", …), ("a", …)], …)`, both calls must return.
- All the Part 2 requirements still hold. Rule 5 becomes: two requests whose resources don't intersect must not
  wait on each other.

```text
svc = ReservationService(["gpu-0", "gpu-1", "room-a"])
svc.reserve("gpu-1", 5, 8, "ann")                               -> 1
svc.reserve_all([("gpu-0", 0, 10), ("gpu-1", 0, 5)], "bob")     -> 2      # on gpu-1, [0,5) touches [5,8)
svc.reserve_all([("room-a", 0, 4), ("gpu-1", 7, 9)], "cat")     -> None   # conflicts with 1 on gpu-1
svc.query("room-a", 0, 100)                                     -> []     # so room-a was not booked either
svc.reserve_all([("room-a", 0, 4), ("room-a", 3, 6)], "cat")    -> None   # its own two legs overlap
svc.reserve_all([("room-a", 0, 4), ("room-a", 4, 6)], "cat")    -> 3
svc.query("room-a", 0, 100)                                     -> [(0, 4, 3, "cat"), (4, 6, 3, "cat")]
svc.cancel(2)                                                   -> True
svc.query("gpu-0", 0, 100)                                      -> []
svc.query("gpu-1", 0, 100)                                      -> [(5, 8, 1, "ann")]
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

The three parts are written as three classes: `ReservationService` (Part 1), `ConcurrentReservationService`
(Part 2), and `AtomicReservationService` (Part 3). They are kept separate so that the Part 1 class can be taken
unchanged into a multithreaded setting to show how it fails. In the interview, just modify one class step by step.

### Part 1

Each resource keeps a list sorted by `start`, whose elements are `(start, end, booking_id, owner)`. The intervals
in the list are pairwise disjoint, so sorted by `start` means `end` is increasing too. Both operations rely on
this:

- **Checking whether `[start, end)` is free.** Any interval that could overlap it must start before `end`; those
  intervals form a prefix `cal[:i]` of the list, with `i` found by binary search. The last interval in the prefix
  has the largest `end`. If its `end <= start`, no interval in the prefix reaches into `[start, end)`, so it is
  free; otherwise that interval itself overlaps `[start, end)`. So only the single interval `cal[i-1]` needs
  checking.
- **Querying the intervals that overlap `[start, end)`.** "Starts before `end`" is a prefix and "ends after
  `start`" is a suffix; two binary searches give their intersection, a contiguous slice.

The binary search uses a small trick of tuple comparison: `(end,)` is smaller than any longer tuple starting with
`end`, so `bisect_left(cal, (end,))` is exactly "the first position with `start >= end`."

Complexity: the conflict check takes $O(\log n)$ comparisons. Insertion and deletion locate their position by
binary search, but a Python list has to shift the elements after it, so the worst case is $O(n)$ (a single memory
move, still fast when $n$ is in the tens of thousands; strict $O(\log n)$ needs a balanced tree or a structure like
`sortedcontainers`). A query is $O(\log n + k)$, where $k$ is the number of results returned.

The code deliberately splits "check" and "write" into two steps (`_commit`, `_uncommit`). The gap between these
two steps is exactly what Part 2 has to protect, and it is also where the tests attack.

```python
import bisect
import threading
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Booking:
    id: int
    owner: str
    legs: tuple                  # ((resource, start, end), ...)
    active: bool = True          # guarded by the locks of the resources in legs


def check_interval(start, end):
    if start >= end:
        raise ValueError(f"empty interval [{start}, {end})")


def is_free(cal, start, end):
    """cal: sorted, pairwise disjoint [(start, end, booking_id, owner)]. True iff [start, end) hits none."""
    i = bisect.bisect_left(cal, (end,))          # cal[:i] are exactly the intervals that start before `end`
    return i == 0 or cal[i - 1][1] <= start      # disjoint + sorted: the last of them also ends last


def overlapping(cal, start, end):
    """The intervals of cal that overlap [start, end), as a new list (still sorted by start)."""
    lo = bisect.bisect_right(cal, start, key=lambda item: item[1])   # first interval that ends after `start`
    hi = bisect.bisect_left(cal, (end,))                              # first interval that starts at/after `end`
    return cal[lo:hi]


def remove_leg(cal, start, end, booking_id):
    i = bisect.bisect_left(cal, (start, end, booking_id))
    assert cal[i][:3] == (start, end, booking_id)
    del cal[i]


class ReservationService:
    """Part 1: correct for one caller at a time. No locks."""

    def __init__(self, resources):
        self._cal = {name: [] for name in resources}     # resource -> sorted [(start, end, booking_id, owner)]
        self._bookings = {}                              # booking_id -> Booking, active ones only
        self._next_id = 1

    def reserve(self, resource, start, end, owner):
        cal = self._cal[resource]                        # KeyError for an unknown resource
        check_interval(start, end)
        if not is_free(cal, start, end):                 # check ...
            return None
        return self._commit(owner, ((resource, start, end),))    # ... then act

    def _commit(self, owner, legs):
        booking = Booking(self._next_id, owner, legs)
        self._next_id += 1
        self._bookings[booking.id] = booking
        for resource, start, end in legs:
            bisect.insort(self._cal[resource], (start, end, booking.id, owner))
        return booking.id

    def cancel(self, booking_id):
        booking = self._bookings.get(booking_id)
        if booking is None:                              # check ...
            return False
        self._uncommit(booking)                          # ... then act
        return True

    def _uncommit(self, booking):
        del self._bookings[booking.id]
        for resource, start, end in booking.legs:
            remove_leg(self._cal[resource], start, end, booking.id)

    def query(self, resource, start, end):
        cal = self._cal[resource]
        check_interval(start, end)
        return overlapping(cal, start, end)
```

### Part 2

**Why the Part 1 code is wrong under multiple threads.** `reserve` is check-then-act. Thread A checks and finds the
slot free, but before it writes, thread B checks too and also finds it free, so both write and the same slot is
booked to two people. `cancel` is the same: both threads see "the booking is still there" and both delete it.
`self._next_id += 1` is three steps, "read, add, write," so two threads may read the same value. CPython's global
interpreter lock (GIL) only guarantees that a single bytecode instruction is not interrupted, not a sequence of
operations, so it doesn't help. It's just that this gap is usually only a few microseconds wide, and a casually
written stress test will probably never hit it, which is exactly what makes this kind of bug dangerous.

**What must be atomic, and which lock protects it:**

| Shared state | Protected by |
| --- | --- |
| A resource's booking list `_cal[r]`; "check free + insert" as a whole | That resource's lock `_locks[r]` |
| `Booking.active`; "check still active + delete" as a whole | The locks of the resources the booking involves |
| The id counter `_next_id`, the id index `_bookings` | The small lock `_meta` |

- **One lock per resource.** Operations on different resources take different locks and don't wait on each
  other. All locks are created at once in the constructor: if they were "created on first use," creating the lock
  would itself be another check-then-act.
- **`_meta` is always taken last.** `_meta` is only ever taken while already holding resource locks; while holding
  `_meta` no other lock is ever taken, and it is released after just a few assignments. With a fixed locking
  order, there is no deadlock.
- **`cancel` has to "look up, then lock, then confirm."** With only an id you don't know which resource to lock, so
  first look up the booking under `_meta`, release `_meta`, then take the resource lock. There is a gap between
  the lookup and the lock, and another thread may already have cancelled it, so after taking the resource lock you
  must check `booking.active` once more.
- **Why not just `pop` under `_meta` to decide whose cancel succeeds?** That would still give "exactly one
  `True`," but it allows this ordering: thread A's `pop` succeeds, but it hasn't yet removed the interval from the
  list; thread B's `cancel` gets `False` and returns; B then `reserve`s the same slot and gets `None`, because the
  interval is still there. B was just told "it's already cancelled," yet can't book the slot, which violates
  rule 3. Putting the "is it still active" check and the removal of the interval under the same resource lock
  avoids this problem.
- **`query` must take the resource lock too**, otherwise it might binary-search a list that is halfway through
  being modified. It returns a slice, i.e. a new list holding immutable tuples, so whatever others do after the
  lock is released doesn't affect it.
- Argument checks happen before taking locks; locks are always taken with `with`, so they are released when an
  exception is raised.

**How to test it.** The race window is too narrow, so widen it in the test. Write a subclass that overrides
`_commit` and pauses just before the actual write. The pause is a `threading.Barrier(8)`: only after all 8 threads
have passed the check and are all waiting there are they released together. So it is not "possibly" wrong; every
time, all 8 threads get an id. Do the same with the locked version: only one thread can be in the critical section
at a time, so the 8 can never all gather; the first thread times out and carries on, and the other 7 then see the
slot taken and get `None`. That is exactly what the verification code below does. Rule 5 (different resources
don't wait on each other) is tested the same way: 6 threads each book a different resource, and they are required
to all be paused inside their critical sections at the same time.

```python
class ConcurrentReservationService:
    """Part 2: one lock per resource; a small inner lock for the id counter and the id index."""

    def __init__(self, resources):
        self._cal = {name: [] for name in resources}
        self._locks = {name: threading.Lock() for name in self._cal}    # all created here, never lazily
        self._bookings = {}
        self._next_id = 1
        self._meta = threading.Lock()        # guards _bookings and _next_id; always the innermost lock

    def reserve(self, resource, start, end, owner):
        lock = self._locks[resource]                     # KeyError before any lock is taken
        check_interval(start, end)
        with lock:                                       # check and insert are one atomic step
            if not is_free(self._cal[resource], start, end):
                return None
            return self._commit(owner, ((resource, start, end),))

    def _commit(self, owner, legs):
        """Caller holds the lock of every resource in legs."""
        with self._meta:
            booking = Booking(self._next_id, owner, legs)
            self._next_id += 1
            self._bookings[booking.id] = booking
        for resource, start, end in legs:
            bisect.insort(self._cal[resource], (start, end, booking.id, owner))
        return booking.id

    def cancel(self, booking_id):
        with self._meta:
            booking = self._bookings.get(booking_id)
        if booking is None:
            return False
        with self._locks[booking.legs[0][0]]:
            if not booking.active:                       # another cancel got in between the lookup and the lock
                return False
            self._uncommit(booking)
        return True

    def _uncommit(self, booking):
        """Caller holds the lock of every resource in booking.legs."""
        booking.active = False
        for resource, start, end in booking.legs:
            remove_leg(self._cal[resource], start, end, booking.id)
        with self._meta:
            del self._bookings[booking.id]

    def query(self, resource, start, end):
        lock = self._locks[resource]
        check_interval(start, end)
        with lock:
            return overlapping(self._cal[resource], start, end)      # a slice: a private copy of immutable tuples
```

### Part 3

**First, think through how it would deadlock.** A request involving several resources has to hold all their locks
at once to make "check all + write all" atomic. If locks are taken in the order the request lists them, thread 1
takes `a` and waits for `b`, thread 2 takes `b` and waits for `a`, and neither ever lets go.

**The fix: everyone takes locks in the same global order.** Here, sorted by resource name. Then every lock a
thread waits for comes after all the locks it already holds. A deadlock needs a cycle of threads that closes on
itself; walking around that cycle, the lock names would have to keep increasing and yet return to the start, which
is impossible.

Three details:

- **Deduplicate.** The same resource can appear twice in a request, and `threading.Lock` is not reentrant: a
  thread that takes it a second time deadlocks itself. Hence `sorted(set(...))`.
- **Check everything first, then write everything.** After acquiring all the locks, first confirm every leg is
  free, then confirm the legs within the request don't overlap each other (sort the legs and compare only
  adjacent pairs), and only start writing once everything passes. On failure nothing has been written yet, so no
  rollback is needed. The alternative is to insert as you check and undo on failure; as long as all the locks are
  held throughout, others still can't see the intermediate state, but there is an extra undo path that is easy to
  get wrong.
- **Release.** In `try/finally`, release only the locks actually acquired. Release order doesn't affect
  correctness.

`cancel` also has to take all the resource locks the booking involves, in the same order. `reserve`, as in
Part 2, involves only one lock.

For a request with $m$ legs: sorting is $O(m \log m)$, the check is $O(m \log n)$ comparisons, and insertion is
$O(n)$ per leg in the worst case. When two requests' resources don't intersect, they take completely different
locks and don't wait on each other.

```python
class AtomicReservationService(ConcurrentReservationService):
    """Part 3: all-or-nothing bookings across several resources, locks taken in one global order."""

    def _lock_order(self, resources):
        return sorted(set(resources))        # set(): Lock is not reentrant; sorted(): the same order for everyone

    @contextmanager
    def _locked(self, resources):
        held = []
        try:
            for name in self._lock_order(resources):
                self._locks[name].acquire()
                held.append(name)
            yield
        finally:
            for name in reversed(held):
                self._locks[name].release()

    def reserve_all(self, requests, owner):
        legs = tuple((resource, start, end) for resource, start, end in requests)
        if not legs:
            raise ValueError("empty request")
        for resource, start, end in legs:                # validate everything before taking any lock
            self._locks[resource]
            check_interval(start, end)
        with self._locked([resource for resource, _, _ in legs]):
            if not self._all_free(legs):
                return None                              # nothing has been written yet: no rollback needed
            return self._commit(owner, legs)

    def _all_free(self, legs):
        if not all(is_free(self._cal[resource], start, end) for resource, start, end in legs):
            return False
        ordered = sorted(legs)                           # legs of one request must not hit each other either
        return all(a[0] != b[0] or a[2] <= b[1] for a, b in zip(ordered, ordered[1:]))

    def cancel(self, booking_id):
        with self._meta:
            booking = self._bookings.get(booking_id)
        if booking is None:
            return False
        with self._locked([resource for resource, _, _ in booking.legs]):
            if not booking.active:
                return False
            self._uncommit(booking)
        return True
```

### Follow-up questions

**With the GIL, why do we still need locks?** The GIL ensures "only one thread executes bytecode at any moment,"
but the interpreter switches threads between bytecode instructions (by default it asks the running thread to yield
every 5 milliseconds). Between the check and the write there are dozens of bytecode instructions and several
function calls, and a switch landing in the middle is a bug. Single operations like `list.append` and `dict.pop`
happen to be atomic in CPython, but relying on that is neither portable nor robust to code changes; and in
free-threaded Python without the GIL (optional since 3.13), even that "happens to be" needs re-examining.

**Would a single global lock work?** Yes, and it is the hardest to get wrong; Part 3 wouldn't need to think about
locking order either. The cost is that every operation queues up. For purely in-memory operations the critical
section is only a few microseconds, and the GIL already serializes CPU work, so splitting into one lock per
resource buys almost no throughput. Splitting locks really pays off when the critical section contains I/O
(writing logs, writing to a database), or with a truly parallel runtime. In the interview, state this trade-off
first, then split the locks as the problem requires. When reads far outnumber writes, `query` can even go
lock-free: the writer builds a new immutable list each time and swaps the reference as a whole, and readers use
whichever version they got.

**How do you test concurrency bugs? What does a passing test tell you?** Use two kinds of test together. One is
deterministic: insert pause points where you suspect a gap, and use a `Barrier` to force the worst interleaving,
as above, so it reproduces every time. The other is a stress test: many threads do random operations, and
afterwards you check the invariants (no overlaps on any resource, ids are exactly $1..K$, each id is successfully
cancelled at most once, the final state equals exactly "issued minus cancelled"); you can also record the order of
every call and check whether two overlapping bookings ever existed at the same time. During stress tests, add a
very short `sleep` inside the critical section, or call `sys.setswitchinterval(1e-6)`, and thread switches become
much more frequent. But a passing test only says "the interleavings we tried didn't fail." So you still need the
table above, "which lock protects each piece of shared state," to make the argument; tests are just for catching
holes in the argument.

**What if the service runs as several processes, on several machines?** An in-process lock only works within one
process; the invariant has to be enforced by the storage everyone shares. Common approaches:

- *Database constraints.* PostgreSQL's exclusion constraint can directly express "time ranges on the same resource
  don't overlap": `EXCLUDE USING gist (resource WITH =, during WITH &&)`, where `during` is a range type (such as
  `tstzrange`, which is half-open by default), and using `=` on a plain scalar column requires the `btree_gist`
  extension. An insert that violates the constraint raises an error, which the application translates to "already
  taken." On a database without such constraints, you can cut time into fixed-length slots and put a unique index
  on `(resource, slot)`.
- *Pessimistic locking.* In one transaction, first lock the resource's row in the resources table with
  `SELECT … FOR UPDATE`, then check for conflicts, then insert. Locking the resource row is the safest: in
  PostgreSQL, locking only the existing booking rows you found does not stop someone else from inserting a new
  one. When booking several resources at once, lock the rows in a fixed order, for the same reason as in Part 3;
  when the database detects a deadlock it rolls back one of the transactions, and the application must be ready to
  retry.
- *Optimistic concurrency.* Put a version number on the resource row. Read version `v`, check for conflicts, then
  in the same transaction insert the booking and run `UPDATE … SET version = v + 1 WHERE id = ? AND version = v`.
  If zero rows are affected, someone got there first: roll back, re-read, and retry. Fast when conflicts are rare;
  on a hot resource it retries over and over.
- Just "first `SELECT` to check for conflicts, then `INSERT`" is wrong under the default isolation level, exactly
  the same race as in Part 1. PostgreSQL's `SERIALIZABLE` level can detect this conflict and end one of the
  transactions with a serialization failure (SQLSTATE 40001), which the application must retry; its
  `REPEATABLE READ` (snapshot isolation) does not prevent it.
- A distributed lock using something like Redis also works, but such locks come with leases: if the holder stalls
  briefly, the lock may already have expired without it knowing. So the write side still needs a database
  constraint, or a check on a monotonically increasing fencing token.

**Can a client retrying after a timeout cause problems?** Yes. The request actually succeeded, but the response was
lost. When the client retries, it finds the slot taken (by its own previous attempt), gets `None`, concludes it
didn't get the booking, and leaves behind a booking nobody claims. The fix is an idempotency key: the client
generates a unique key for each logical request and sends the same key on retries; the server writes "key →
result" in the same transaction as the booking itself, with a unique constraint on the key. When it sees the same
key again it doesn't re-execute, it just returns the stored result. Three more things to handle: two requests with
the same key arriving at once (the unique constraint makes the later one wait or fail, then read the existing
result); the same key with different parameters (reject); and how long to keep keys.

**What about "hold the slot first, confirm within N seconds, otherwise release automatically"?** Give bookings a
status and an expiry time: `hold(...)` creates a provisional booking with `expires_at`, and `confirm(id)` turns it
into a confirmed one. The key is to make every decision under the resource lock: when checking whether a slot is
free, treat expired holds as nonexistent and delete them while you're there; `confirm` checks "not yet expired"
and changes the status under the same lock. That way you never get "`confirm` returned success, but the slot has
already gone to someone else." Judge expiry with a monotonic clock (`time.monotonic`), and make the clock an
injectable parameter so tests can use a fake clock instead of actually calling `sleep`. In a database this is
usually a status column plus `expires_at`, with a cleanup job marking expired records as released, and the
exclusion constraint applying only to unreleased rows (add a `WHERE` clause to the constraint).

<details>
<summary>Verification code (runnable)</summary>

```python
import itertools
import random
import threading
import time

HORIZON = 10 ** 9


def expect_raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{args} did not raise {exc.__name__}")


def run_threads(targets, timeout=60.0):
    """Start one daemon thread per callable, wait for all; fail loudly instead of hanging."""
    threads = [threading.Thread(target=t, daemon=True) for t in targets]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "threads are stuck"


# ---------- 1. the statement's examples (one caller) ----------
for cls in (ReservationService, ConcurrentReservationService, AtomicReservationService):
    svc = cls(["room-a", "room-b"])
    assert svc.reserve("room-a", 9, 11, "ann") == 1
    assert svc.reserve("room-a", 11, 12, "bob") == 2            # touching is not overlapping
    assert svc.reserve("room-a", 10, 13, "cat") is None
    assert svc.reserve("room-b", 10, 13, "cat") == 3            # the failed call did not use up an id
    assert svc.query("room-a", 0, 24) == [(9, 11, 1, "ann"), (11, 12, 2, "bob")]
    assert svc.query("room-a", 11, 12) == [(11, 12, 2, "bob")]
    assert svc.query("room-a", 12, 24) == []
    assert svc.cancel(1) is True
    assert svc.cancel(1) is False
    assert svc.cancel(99) is False
    assert svc.reserve("room-a", 10, 11, "cat") == 4
    assert svc.query("room-a", 0, 24) == [(10, 11, 4, "cat"), (11, 12, 2, "bob")]
    expect_raises(KeyError, svc.reserve, "room-c", 1, 2, "dan")
    expect_raises(ValueError, svc.reserve, "room-a", 5, 5, "dan")
    expect_raises(ValueError, svc.query, "room-a", 7, 3)
    expect_raises(KeyError, svc.query, "room-c", 1, 2)
    assert svc.reserve("room-a", 12, 13, "dan") == 5            # errors used no id and left no lock held

svc = AtomicReservationService(["gpu-0", "gpu-1", "room-a"])
assert svc.reserve("gpu-1", 5, 8, "ann") == 1
assert svc.reserve_all([("gpu-0", 0, 10), ("gpu-1", 0, 5)], "bob") == 2
assert svc.reserve_all([("room-a", 0, 4), ("gpu-1", 7, 9)], "cat") is None
assert svc.query("room-a", 0, 100) == []                        # all or nothing
assert svc.reserve_all([("room-a", 0, 4), ("room-a", 3, 6)], "cat") is None
assert svc.reserve_all([("room-a", 0, 4), ("room-a", 4, 6)], "cat") == 3
assert svc.query("room-a", 0, 100) == [(0, 4, 3, "cat"), (4, 6, 3, "cat")]
assert svc.cancel(2) is True
assert svc.query("gpu-0", 0, 100) == []
assert svc.query("gpu-1", 0, 100) == [(5, 8, 1, "ann")]
assert svc.cancel(3) is True and svc.query("room-a", 0, 100) == []
expect_raises(ValueError, svc.reserve_all, [], "dan")
expect_raises(KeyError, svc.reserve_all, [("gpu-0", 20, 30), ("nope", 0, 1)], "dan")
expect_raises(ValueError, svc.reserve_all, [("gpu-0", 20, 30), ("gpu-1", 9, 9)], "dan")
assert svc.query("gpu-0", 0, 100) == []                         # the failed calls booked nothing
assert svc.reserve_all([("gpu-0", 20, 30)], "dan") == 4         # ... and left every lock free


# ---------- 2. one caller, random operations, against a model that only scans lists ----------
class ScanModel:
    def __init__(self, resources):
        self.resources = set(resources)
        self.active = {}                       # id -> (owner, legs)
        self.issued = 0

    def reserve_all(self, requests, owner):
        legs = list(requests)
        taken = [leg for _, booked in self.active.values() for leg in booked]
        for i, (r, s, e) in enumerate(legs):
            if any(r == r2 and s < e2 and s2 < e for r2, s2, e2 in taken + legs[:i]):
                return None
        self.issued += 1
        self.active[self.issued] = (owner, legs)
        return self.issued

    def cancel(self, booking_id):
        return self.active.pop(booking_id, None) is not None

    def query(self, resource, start, end):
        return sorted((s, e, bid, owner) for bid, (owner, legs) in self.active.items()
                      for r, s, e in legs if r == resource and s < end and start < e)


def random_leg(rng, names, span=30):
    start = rng.randrange(span)
    return rng.choice(names), start, start + rng.randint(1, 6)


for cls in (ReservationService, ConcurrentReservationService, AtomicReservationService):
    rng = random.Random(2026)
    for _ in range(150):
        names = ["r%d" % i for i in range(rng.randint(1, 3))]
        svc, model = cls(names), ScanModel(names)
        for _ in range(80):
            p = rng.random()
            if p < 0.45:
                leg = random_leg(rng, names)
                assert svc.reserve(*leg, "u") == model.reserve_all([leg], "u")
            elif p < 0.60 and cls is AtomicReservationService:
                legs = [random_leg(rng, names) for _ in range(rng.randint(1, 3))]
                assert svc.reserve_all(legs, "v") == model.reserve_all(legs, "v")
            elif p < 0.80:
                bid = rng.randint(1, model.issued + 2)
                assert svc.cancel(bid) == model.cancel(bid)
            else:
                r, s, e = random_leg(rng, names)
                assert svc.query(r, s, e) == model.query(r, s, e)
        for r in names:
            assert svc.query(r, 0, HORIZON) == model.query(r, 0, HORIZON)
        assert sorted(svc._bookings) == sorted(model.active)


# ---------- 3. the race, made deterministic ----------
def paused(cls, pause):
    """cls with pause() called in the gap between a check and the write that follows it."""
    class Paused(cls):
        def _commit(self, owner, legs):          # reserve: after "is it free?", before the insert
            pause()
            return super()._commit(owner, legs)

        def _uncommit(self, booking):            # cancel: after "is it still there?", before the removal
            pause()
            return super()._uncommit(booking)
    return Paused


def meeting_point(n, wait):
    """pause() that blocks until n threads are in it together; gives up after `wait` seconds.
    Returns (pause, gave_up) where gave_up collects one item per thread that gave up."""
    gate, gave_up = threading.Barrier(n), []

    def pause():
        try:
            gate.wait(timeout=wait)
        except threading.BrokenBarrierError:
            gave_up.append(1)
    return pause, gave_up


def same_slot_race(cls, wait, n=8):
    """n threads book the same slot; each is held after its check until all n have passed theirs."""
    pause, _ = meeting_point(n, wait)
    svc = paused(cls, pause)(["room"])
    results = []                                 # list.append is atomic in CPython; good enough for a harness
    run_threads([lambda i=i: results.append(svc.reserve("room", 10, 20, f"user{i}")) for i in range(n)])
    return results, svc.query("room", 0, HORIZON)


# Part 1 code called from 8 threads. Nothing stops all 8 from passing the check, so the gate fills at once
# (the 30 s is only a safety net) and every caller is told the slot is theirs.
results, cal = same_slot_race(ReservationService, wait=30.0)
assert results.count(None) == 0 and len(cal) == 8
assert all(item[:2] == (10, 20) for item in cal)
print("no locks:", len(cal), "bookings of room [10, 20) at the same time")

# With the lock only one thread can be between check and insert, so the gate can never fill: the first
# thread gives up after 0.3 s, books, and the other seven then see the slot taken.
for cls in (ConcurrentReservationService, AtomicReservationService):
    results, cal = same_slot_race(cls, wait=0.3)
    assert sorted(results, key=str) == [1] + [None] * 7 and len(cal) == 1


# ---------- 4. different resources really do not wait for each other ----------
def together_in_critical_section(cls, wait, n=6):
    """n threads book n different resources; each pauses inside its critical section until all n are inside.
    Returns how many gave up waiting (0 means all n critical sections were open at the same moment)."""
    pause, gave_up = meeting_point(n, wait)
    svc = paused(cls, pause)([f"r{i}" for i in range(n)])
    results = []
    run_threads([lambda i=i: results.append(svc.reserve(f"r{i}", 0, 1, "u")) for i in range(n)])
    assert sorted(results) == list(range(1, n + 1))
    return len(gave_up)


class GlobalLockService(ConcurrentReservationService):
    """Negative control: every resource shares one lock."""
    def __init__(self, resources):
        super().__init__(resources)
        one = threading.Lock()
        self._locks = {name: one for name in self._cal}


assert together_in_critical_section(ConcurrentReservationService, wait=30.0) == 0
assert together_in_critical_section(AtomicReservationService, wait=30.0) == 0
assert together_in_critical_section(GlobalLockService, wait=0.3) == 6      # the test can tell the difference


# ---------- 5. one booking, many cancels: exactly one True ----------
class NoLock:
    def acquire(self):
        return True

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class UnsynchronisedService(AtomicReservationService):
    """The Part 3 reference with every lock replaced by a no-op: the deliberately broken version."""
    def __init__(self, resources):
        super().__init__(resources)
        self._locks = {name: NoLock() for name in self._cal}
        self._meta = NoLock()


def cancel_race(cls, wait, n=8):
    """n threads cancel the same booking; each is held after its "still there?" check until all n passed."""
    pause, _ = meeting_point(n, wait)
    svc = cls(["a", "b"])                        # book first, without any pause ...
    bid = svc.reserve("a", 0, 5, "u")
    svc.__class__ = paused(cls, pause)           # ... then switch the pauses on for the cancels
    answers = []

    def cancel_once():
        try:
            answers.append(svc.cancel(bid))
        except Exception as exc:
            answers.append(type(exc).__name__)

    run_threads([cancel_once] * n)
    return answers


for cls in (ReservationService, UnsynchronisedService):         # no locks: all 8 pass the check, then all 8 act
    answers = cancel_race(cls, wait=30.0)
    assert answers.count(True) + answers.count(False) < 8 or answers.count(True) > 1, answers
    print("no locks, 8 cancels of one booking:", sorted(answers, key=str))
for cls in (ConcurrentReservationService, AtomicReservationService):
    assert sorted(cancel_race(cls, wait=0.3)) == [False] * 7 + [True]

for cls in (ConcurrentReservationService, AtomicReservationService):       # and without any pause, many times
    for _ in range(30):
        svc = cls(["a", "b"])
        bid = svc.reserve("a", 0, 5, "u")
        if cls is AtomicReservationService:
            bid = svc.reserve_all([("a", 5, 9), ("b", 0, 9)], "u")
        start, answers = threading.Barrier(8), []

        def cancel_once():
            start.wait()
            answers.append(svc.cancel(bid))

        run_threads([cancel_once] * 8)
        assert sorted(answers) == [False] * 7 + [True]
        assert svc.query("b", 0, 99) == [] and bid not in svc._bookings


# ---------- 6. two callers, two resources, opposite order ----------
class ProbeLock:
    """Stands in for a Lock. Gives up after `timeout` seconds instead of waiting forever, and
    calls after_acquire() each time it has been taken."""
    def __init__(self, timeout, after_acquire):
        self._lock, self._timeout, self._after = threading.Lock(), timeout, after_acquire

    def acquire(self):
        if not self._lock.acquire(timeout=self._timeout):
            raise TimeoutError("still waiting for a lock")
        self._after()

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()


class RequestOrderService(AtomicReservationService):
    """The tempting version: take the locks in the order the request lists them."""
    def _lock_order(self, resources):
        return list(dict.fromkeys(resources))


def opposite_orders(cls, lock_timeout, gate_timeout):
    """Thread 1 books (a, b), thread 2 books (b, a), different hours. After its first lock each thread
    waits at a gate until the other one also holds a lock (or until gate_timeout)."""
    gate, first = threading.Barrier(2), threading.local()

    def after_acquire():
        if getattr(first, "done", False):
            return
        first.done = True
        try:
            gate.wait(timeout=gate_timeout)
        except threading.BrokenBarrierError:
            pass

    svc = cls(["a", "b"])
    svc._locks = {name: ProbeLock(lock_timeout, after_acquire) for name in svc._locks}
    outcome = []

    def book(legs, owner):
        try:
            outcome.append(svc.reserve_all(legs, owner))
        except TimeoutError:
            outcome.append("deadlock")

    run_threads([lambda: book([("a", 0, 1), ("b", 0, 1)], "t1"), lambda: book([("b", 5, 6), ("a", 5, 6)], "t2")])
    return outcome


# With the gate both threads hold their first lock before either asks for its second one, so neither
# request can be granted until somebody gives up: the timeout below is not a guess about speed.
outcome = opposite_orders(RequestOrderService, lock_timeout=1.0, gate_timeout=10.0)
assert "deadlock" in outcome, outcome
print("request-order locking:", outcome)
# Sorted order: both want "a" first, so they can never each hold one lock. The gate times out, nobody is stuck.
outcome = opposite_orders(AtomicReservationService, lock_timeout=30.0, gate_timeout=0.2)
assert sorted(outcome) == [1, 2], outcome


# ---------- 7. the storm ----------
class Ticker:
    """A logical clock shared by all threads."""
    def __init__(self):
        self._lock, self._now = threading.Lock(), 0

    def tick(self):
        with self._lock:
            self._now += 1
            return self._now


def storm(svc, n_threads=12, n_ops=250, seed=0, span=120):
    """Random reserve / reserve_all / cancel / query from n_threads threads. Returns a dict of problem counts."""
    names = sorted(svc._cal)
    ticker, pool_lock = Ticker(), threading.Lock()
    pool = []                                    # ids anyone may try to cancel
    logs = [dict(issued=[], cancelled=[], errors=[], bad_snapshots=0) for _ in range(n_threads)]

    def worker(k):
        rng, log = random.Random(seed * 1000 + k), logs[k]
        for _ in range(n_ops):
            p = rng.random()
            try:
                if p < 0.40:
                    legs = [random_leg(rng, names, span)]
                    bid = svc.reserve(*legs[0], f"t{k}")
                elif p < 0.60:
                    legs = [random_leg(rng, names, span) for _ in range(rng.randint(2, 3))]
                    bid = svc.reserve_all(legs, f"t{k}")
                elif p < 0.90:
                    with pool_lock:
                        victim = rng.choice(pool) if pool else 0
                    before = ticker.tick()                       # taken before the call starts
                    if svc.cancel(victim):
                        log["cancelled"].append((victim, before))
                    continue
                else:
                    r, s, e = random_leg(rng, names, span)
                    snap = svc.query(r, 0, HORIZON)
                    if any(a[1] > b[0] for a, b in zip(snap, snap[1:])) or snap != sorted(snap):
                        log["bad_snapshots"] += 1
                    continue
                if bid is not None:
                    log["issued"].append((bid, legs, ticker.tick()))     # tick taken after the call returned
                    with pool_lock:
                        pool.append(bid)
            except Exception as exc:                             # only the broken version gets here
                log["errors"].append(repr(exc))

    run_threads([lambda k=k: worker(k) for k in range(n_threads)], timeout=120.0)

    issued = [x for log in logs for x in log["issued"]]
    cancelled = [x for log in logs for x in log["cancelled"]]
    ids = sorted(bid for bid, _, _ in issued)
    cancelled_ids = [bid for bid, _ in cancelled]
    problems = dict(
        errors=sum(len(log["errors"]) for log in logs),
        bad_snapshots=sum(log["bad_snapshots"] for log in logs),
        ids_not_1_to_k=int(ids != list(range(1, len(ids) + 1))),
        cancelled_twice=len(cancelled_ids) - len(set(cancelled_ids)),
    )
    # (a) what is left must be exactly: everything issued and not cancelled
    gone = set(cancelled_ids)
    expected = {name: [] for name in names}
    for bid, legs, _ in issued:
        if bid not in gone:
            for r, s, e in legs:
                expected[r].append((s, e, bid))
    final = {name: [item[:3] for item in svc.query(name, 0, HORIZON)] for name in names}
    problems["final_state_wrong"] = sum(final[name] != sorted(expected[name]) for name in names)
    problems["final_overlaps"] = sum(1 for name in names for a, b in itertools.combinations(final[name], 2)
                                     if a[0] < b[1] and b[0] < a[1])
    problems["index_wrong"] = int(sorted(svc._bookings) != sorted(set(ids) - gone))
    # (b) double bookings that were cancelled again before the end: booking X certainly existed from the
    # tick after its reserve returned until the tick before its successful cancel started. Two bookings
    # that overlap on a resource must never have certainly existed at the same tick.
    end_tick = dict(cancelled)
    alive = [(r, s, e, born, end_tick.get(bid, float("inf"))) for bid, legs, born in issued for r, s, e in legs]
    by_resource = {name: [x for x in alive if x[0] == name] for name in names}
    problems["double_booked_pairs"] = sum(
        1 for name in names for a, b in itertools.combinations(by_resource[name], 2)
        if a[1] < b[2] and b[1] < a[2] and a[3] < b[4] and b[3] < a[4])
    return problems, len(ids), len(cancelled_ids)


def yield_a_little():
    time.sleep(0.0002)       # lets other threads run in the middle of the critical section


RESOURCES = ["r0", "r1", "r2", "r3"]
for seed in range(3):
    problems, n_issued, n_cancelled = storm(paused(AtomicReservationService, yield_a_little)(RESOURCES), seed=seed)
    assert not any(problems.values()), problems
    assert n_issued > 200 and n_cancelled > 100            # the storm really did book and cancel a lot
    print("storm, reference:", n_issued, "bookings,", n_cancelled, "cancels, no problems")

problems, n_issued, n_cancelled = storm(paused(UnsynchronisedService, yield_a_little)(RESOURCES), seed=0)
assert problems["double_booked_pairs"] > 0, problems       # the same storm catches the version without locks
print("storm, no locks:", problems)
print("all checks passed")
```

</details>

</details>
