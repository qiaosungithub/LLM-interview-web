# Hash Map from Scratch

## Problem

You have just solved a problem using `dict`, and the interviewer follows up: now you may not use `dict` or `set`;
implement a hash map yourself. You may use `list`, the built-in `hash()`, and `==`. Collisions must be handled with
open addressing and linear probing.

So that the examples can be worked out by hand, the table layout is fixed as follows:

- The table is an array of `capacity` slots. `capacity` is always a power of two, starting at 8.
- The home slot of a key `key` is `hash(key) & (capacity - 1)`. If that slot is taken by another key, look at the
  next slot, wrapping around to slot 0 after the end, until you find the key or an empty slot.
- Two keys `a` and `b` count as the same key if and only if `a is b or a == b` (just like `dict`).
- Growth rule: before inserting a **new** key, if `3 * (occupied slots + 1) > 2 * capacity`, first double the
  capacity and reinsert all keys into the new table in the order of the old table's slots 0, 1, 2, …, and only then
  insert the new key. In other words the load factor never exceeds 2/3: 8 slots hold at most 5 keys, and before
  the 6th the table grows to 16.

### Part 1 — Insert, look up, grow

```py
class HashMap:
    def __init__(self) -> None: ...

    def put(self, key, value) -> None:
        """Insert key, or overwrite its value if it is already present."""

    def get(self, key, default=None):
        """The value stored for key, or default if key is absent."""

    def __contains__(self, key) -> bool: ...

    def __len__(self) -> int:
        """Number of keys stored."""
```

```text
m = HashMap()
m.put("a", 1); m.put("b", 2); m.put("a", 3)
len(m)            -> 2
m.get("a")        -> 3
m.get("zzz")      -> None
m.get("zzz", 0)   -> 0
"b" in m          -> True

m.put(1, "int"); m.put(1.0, "float")     # 1 == 1.0 and hash(1) == hash(1.0): the same key
len(m)            -> 3
m.get(1)          -> "float"
m.get(True)       -> "float"             # True == 1, same reason
```

Also answer: what do `put` and `get` cost, in expectation and in the worst case? A single growth step is $O(n)$, so
why is `put` still amortised $O(1)$?

### Part 2 — Delete

```py
    def delete(self, key) -> bool:
        """Remove key if it is present. Returns whether it was present."""
```

First answer: why can't you simply clear the slot of the deleted key? Give a counterexample of three or four
operations. Then implement deletion correctly, using either tombstones or backward-shift deletion. Requirements:

1. After any sequence of operations, the results match `dict`.
2. The table must not keep growing because of repeated "insert a new key, delete an old key": if the number of keys
   present at the same time never exceeds $k$, the capacity never exceeds $\max(8, 4k)$.
3. No operation may loop forever because "the table has no empty slot left".

Once deletion exists, you may adjust Part 1's growth rule as needed, as long as the three requirements above hold
and the rule still matches Part 1's when no key has ever been deleted.

```text
m = HashMap()
for k in (0, 8, 16): m.put(k, str(k))    # a small integer hashes to itself: all three keys have home slot 0
m.delete(8)       -> True
m.get(16)         -> "16"
m.delete(8)       -> False
len(m)            -> 2
m.put(8, "again")
m.get(8)          -> "again"
len(m)            -> 3
```

### Part 3 — Iteration

```py
    def items(self):
        """Iterates over (key, value) for every stored key, in slot order."""

    def keys(self): ...

    def __iter__(self):
        """Same as keys()."""
```

The iteration order is defined as slot order, from slot 0 to slot `capacity - 1`, with each key appearing exactly
once. If a key is added or deleted during iteration, the next request for an element from the iterator must raise
`RuntimeError`; merely overwriting the value of an existing key does not count.

```text
m = HashMap()
for k in (9, 1, 2): m.put(k, None)
list(m)           -> [9, 1, 2]           # 9's home slot is 9 & 7 = 1; 1's home slot is also 1, taken, so it lands in slot 2; 2 lands in slot 3
for k in (3, 4): m.put(k, None)
list(m)           -> [9, 1, 2, 3, 4]
m.put(5, None)                           # the 6th key: grow to 16 first, then insert
list(m)           -> [1, 2, 3, 4, 5, 9]
```

Also answer: how does this order relate to insertion order? What does one full iteration cost? (Once deletion
exists, the exact order depends on the deletion scheme you chose; the problem does not specify it.)

### Part 4 — Custom hash and equality

Some keys can't use the built-in `hash` directly: a `list` is unhashable; or you may want string keys to be
case-insensitive. Make the constructor accept two optional arguments:

```py
class HashMap:
    def __init__(self, hash_fn=None, eq_fn=None) -> None:
        """hash_fn(key) -> int replaces hash(key); eq_fn(a, b) -> bool replaces a == b.
        The caller guarantees: eq_fn(a, b) implies hash_fn(a) == hash_fn(b)."""
```

Requirement: each call to `put`, `get`, `delete`, or `in` calls `hash_fn` exactly once on the key passed in, and
growing the table calls it not even once (hashing a large array can be expensive).

```text
m = HashMap(hash_fn=lambda s: hash(s.lower()), eq_fn=lambda a, b: a.lower() == b.lower())
m.put("Key", 1); m.put("KEY", 2)
len(m)            -> 1
m.get("key")      -> 2
list(m)           -> ["Key"]             # an overwrite only replaces the value; the key object stored first is kept

m = HashMap(hash_fn=lambda xs: hash(tuple(xs)))
m.put([1, 2], "a")
m.get([1, 2])     -> "a"
[2, 1] in m       -> False
```

Then answer two more questions:

- If a `list` used as a key is modified in place after it was inserted, what happens?
- When the keys are `i << 40` (`i = 0, 1, …, 1999`), what does the default `hash` turn this table into? Write a
  `hash_fn` that fixes it.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Below, inheritance strings the four parts together, and each part shows only its new code: `LinearProbingMap`
(Part 1) → `TombstoneMap` (Part 2) → `IterableMap` (Part 3) → `HashMap` (Part 4). In an interview you would just
change the same class step by step.

### Part 1

The table is stored in three lists of equal length: keys, values, and each key's hash. An empty slot is marked with
a dedicated sentinel object `_EMPTY`; you can't use `None`, because `None` can itself be a key.

Every operation comes down to one internal function, `_find(key, h)`: walk forward from the home slot; on meeting
this key, return "found, in slot i"; on meeting an empty slot, return "absent, it belongs in slot i". It relies on
one invariant:

> Between a key's home slot and the slot it actually occupies, there is no empty slot.

So as soon as a lookup hits an empty slot, it can conclude the key is absent. An insertion always goes into the
first empty slot on the probe path, which preserves the invariant. As long as the table has at least one empty slot,
`_find` is guaranteed to stop; the growth rule keeps the load factor at most 2/3, so there is always an empty slot.

Storing the hashes has two benefits: growing doesn't need to recompute them, and probing can compare hashes first
(an integer comparison, very cheap) and only call `==` when the hashes match. When reinserting keys into the new
table during growth, and when placing the new key after growth, we already know the key is not in the table, so we
only need the first empty slot (`_free_slot`) and no key comparison at all.

```python
_EMPTY = object()   # a slot that has never held a key: every probe sequence stops here


class LinearProbingMap:
    """Part 1: open addressing with linear probing. Grows by doubling; no deletion yet."""

    def __init__(self) -> None:
        self._allocate(8)

    def _allocate(self, capacity: int) -> None:
        self._capacity = capacity                 # always a power of two
        self._hashes = [0] * capacity             # hash of the key in each slot (unused if no key)
        self._keys = [_EMPTY] * capacity
        self._values = [None] * capacity
        self._size = 0                            # number of keys stored

    def _hash(self, key) -> int:
        return hash(key)

    def _same(self, a, b) -> bool:
        return a is b or a == b

    def _find(self, key, h: int):
        """(slot, True) if key is stored in slot; otherwise (slot, False), where slot is the
        empty slot that ended the probe sequence -- which is where key would be inserted."""
        mask = self._capacity - 1
        i = h & mask                              # NOTE: for a negative h, & still lands in [0, mask]
        while True:
            k = self._keys[i]
            if k is _EMPTY:
                return i, False
            if self._hashes[i] == h and self._same(k, key):   # compare the stored hash first
                return i, True
            i = (i + 1) & mask

    def _entries(self):
        """(hash, key, value) of every stored key, in slot order."""
        for i in range(self._capacity):
            if self._keys[i] is not _EMPTY:
                yield self._hashes[i], self._keys[i], self._values[i]

    def _free_slot(self, h: int) -> int:
        """The first empty slot on the probe sequence of hash h, for a key known to be absent."""
        mask = self._capacity - 1
        i = h & mask
        while self._keys[i] is not _EMPTY:
            i = (i + 1) & mask
        return i

    def _rebuild(self, capacity: int) -> None:
        entries = list(self._entries())
        self._allocate(capacity)
        for h, key, value in entries:
            i = self._free_slot(h)                # the keys are distinct: no comparison needed
            self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size = len(entries)

    def put(self, key, value) -> None:
        h = self._hash(key)
        i, found = self._find(key, h)
        if found:
            self._values[i] = value               # overwrite: the stored key object is kept
            return
        if 3 * (self._size + 1) > 2 * self._capacity:
            self._rebuild(2 * self._capacity)
            i = self._free_slot(h)                # NOTE: the slot found before the rebuild is stale
        self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size += 1

    def get(self, key, default=None):
        i, found = self._find(key, self._hash(key))
        return self._values[i] if found else default

    def __contains__(self, key) -> bool:
        return self._find(key, self._hash(key))[1]

    def __len__(self) -> int:
        return self._size
```

**Cost.** The cost of an operation is the number of slots probed. Let the load factor be $\alpha$.

- Expected: if the hashes behave like uniform random values, the classic result for linear probing (Knuth) is that
  a lookup of a present key probes about $\frac{1}{2}\left(1 + \frac{1}{1-\alpha}\right)$ slots on average, and a
  lookup of an absent key or an insertion about $\frac{1}{2}\left(1 + \frac{1}{(1-\alpha)^2}\right)$ slots. At
  $\alpha = 2/3$ these are 2 and 5, so the cost is expected $O(1)$. This is also why the load-factor cap can't be
  set too high: at $\alpha = 0.9$ the second formula gives 50.5. These are averages over large tables; a
  measurement on any single table fluctuates, and the smaller the table, the bigger the fluctuation.
- Worst case: $O(n)$. If all keys' hashes collide, they form one run, and every lookup walks the whole run. For $n$
  keys with identical hashes, inserting them one after another alone takes exactly $n(n-1)/2$ key comparisons. This
  is determined by the hash function and the keys, and the table can't defend against it by itself (Part 4 shows an
  example).

So the accurate statement is "expected $O(1)$, worst case $O(n)$", not "$O(1)$".

**Why growth is amortised $O(1)$.** With insertions only, the capacity goes 8, 16, 32, …. Growing from capacity $C$
to $2C$ scans $C$ old slots and allocates $2C$ new ones, $3C$ in total. All growth steps together cost
$3(8 + 16 + \dots + C_{\text{final}}/2) < 3C_{\text{final}}$. And when the last growth happened, the number of
keys already exceeded 2/3 of the old capacity, i.e. $n > C_{\text{final}}/3$, so the total is less than $9n$:
constant per insertion. Reinserting each key also involves probing; the new table's load factor is below 1/2, so
that part is expected $O(1)$. Altogether, `put` is **amortised expected** $O(1)$; a single `put` is still $O(n)$
in the worst case (exactly the one that triggers growth).

### Part 2

**Why you can't simply clear the slot.** Take the example from the statement: 0, 8, and 16 all have home slot 0,
and land in slots 0, 1, 2 in turn. If deleting 8 just marks slot 1 as empty, then look up 16: start at slot 0, slot 0
holds key 0, no match; slot 1 is empty, so conclude that 16 is absent. But 16 is right there in slot 2. The reason is
that clearing broke Part 1's invariant: an empty slot appeared in the middle of 16's probe path. If you then call
`put(16, …)`, another 16 is placed in slot 1, and the same key appears twice in the table.

**Tombstones.** Instead of marking the deleted slot empty, put a tombstone marker `_TOMB` in it. A lookup treats a
tombstone as "something is here, keep going"; inserting a new key may reuse the first tombstone met on the path. The
invariant becomes: between the home slot and the key's slot there is no `_EMPTY` (tombstones are allowed).

Tombstones bring two problems that must be handled, matching requirements 3 and 2 of the problem:

- Tombstones also block the way. If the load rule counts only live keys, then after repeated insertions and
  deletions the table may be all keys and tombstones without a single `_EMPTY`, and looking up an absent key loops
  forever. So **tombstones count as occupied in the load rule**: rebuild when
  `3 * (keys + tombstones + 1) > 2 * capacity`. An insertion that reuses a tombstone consumes no `_EMPTY`, so it
  needs no check.
- A rebuild discards all tombstones, so the new capacity should be **sized by the live keys only**: the smallest
  power of two that is at least `2 * (keys + 1)` (at least 8). With no tombstones this is exactly doubling,
  consistent with Part 1; with many tombstones, the capacity stays the same or even shrinks, which amounts to a
  cleanup in place.

```python
_TOMB = object()    # a slot whose key was deleted: probe sequences walk past it


class TombstoneMap(LinearProbingMap):
    """Part 2: deletion with tombstones."""

    def _allocate(self, capacity: int) -> None:
        super()._allocate(capacity)
        self._tombs = 0                           # slots holding _TOMB

    def _find(self, key, h: int):
        """(slot, True) if key is stored in slot; otherwise (slot, False), where slot is where key
        should be inserted: the first tombstone passed, or else the empty slot that ended the search."""
        mask = self._capacity - 1
        i = h & mask
        first_tomb = -1
        while True:
            k = self._keys[i]
            if k is _EMPTY:
                return (i if first_tomb < 0 else first_tomb), False
            if k is _TOMB:
                if first_tomb < 0:
                    first_tomb = i                # NOTE: keep walking -- the key may still be further on
            elif self._hashes[i] == h and self._same(k, key):
                return i, True
            i = (i + 1) & mask

    def _entries(self):
        for i in range(self._capacity):
            k = self._keys[i]
            if k is not _EMPTY and k is not _TOMB:
                yield self._hashes[i], k, self._values[i]

    def put(self, key, value) -> None:
        h = self._hash(key)
        i, found = self._find(key, h)
        if found:
            self._values[i] = value
            return
        if self._keys[i] is _TOMB:
            self._tombs -= 1                      # reusing a tombstone uses up no empty slot
        elif 3 * (self._size + self._tombs + 1) > 2 * self._capacity:   # NOTE: tombstones count
            capacity = 8
            while capacity < 2 * (self._size + 1):                      # NOTE: sized by live keys only
                capacity *= 2
            self._rebuild(capacity)               # drops every tombstone
            i = self._free_slot(h)
        self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size += 1

    def delete(self, key) -> bool:
        i, found = self._find(key, self._hash(key))
        if not found:
            return False
        self._keys[i] = _TOMB                     # NOTE: not _EMPTY -- other keys may sit beyond this slot
        self._values[i] = None                    # drop the reference to the value
        self._size -= 1
        self._tombs += 1
        return True
```

**Why this sizing is enough.**

- Capacity bound (requirement 2): on a rebuild, the new capacity is the smallest power of two that is at least
  `2 * (keys + 1)`. When the number of keys present at once never exceeds $k$, `keys + 1` is at most $k$, and the
  smallest power of two that is at least $2k$ is less than $4k$. So the capacity never exceeds $\max(8, 4k)$.
- Amortisation: charge a rebuild as "old capacity + new capacity" slots. Once the new table is built and the key that
  triggered it is placed, occupancy is at most half; the next rebuild happens only when occupancy exceeds 2/3.
  Occupancy grows only when "a new key goes into an `_EMPTY`" (a deletion turns a key into a tombstone, and reusing a
  tombstone turns a tombstone into a key; neither changes occupancy). So before a table of capacity $C$ is replaced,
  it absorbs at least $\lfloor 2C/3 \rfloor - C/2 \ge C/8$ such insertions. The rebuild that replaces it costs at
  most $C + 2C = 3C$. Each insertion of a new key is therefore charged at most $3C / (C/8) = 24$ slots. As in
  Part 1, probing during reinsertion is expected $O(1)$, so `put` is still amortised expected $O(1)$; `delete` and
  `get` never trigger a rebuild and are expected $O(1)$.
- Sizing the new capacity to "just under 2/3" would cause trouble: when the number of keys sits right at the
  boundary, each delete followed by an insert could trigger yet another rebuild, and the amortisation would fail.
  Leaving the headroom from 1/2 to 2/3 is what gives the $C/8$ above.

With tombstones, a lookup walks over keys and tombstones together, so probe length depends on the load factor of
"keys + tombstones". That is exactly the quantity the load rule limits.

**The alternative: backward-shift deletion.** Leave no tombstone. After deleting a key, look at the run of
consecutive non-empty slots that follows: if a key, once moved into the hole, would still satisfy "no empty slot
between its home slot and its slot", move it there; the hole moves to where it was, and continue forward until an
empty slot. The condition: the key in slot `j` can move to hole `i` if and only if its home slot is not in the
circular interval `(i, j]`. This way the table never has tombstones, and Part 1's `put` and load rule need not change
by a single line; the cost is that deletion moves elements, and that circular-interval condition is very easy to get
wrong.

```python
class ShiftDeleteMap(LinearProbingMap):
    """Part 2, the alternative: backward-shift deletion. No tombstones, so put() is unchanged."""

    def delete(self, key) -> bool:
        i, found = self._find(key, self._hash(key))
        if not found:
            return False
        mask = self._capacity - 1
        j = i                                     # i is the hole; j scans the rest of the run
        while True:
            j = (j + 1) & mask
            if self._keys[j] is _EMPTY:
                break
            home = self._hashes[j] & mask
            # distances measured backwards from j, modulo capacity: home is outside (i, j]
            # exactly when home is at least as far back as the hole is
            if ((j - home) & mask) >= ((j - i) & mask):
                self._hashes[i], self._keys[i], self._values[i] = \
                    self._hashes[j], self._keys[j], self._values[j]
                i = j                             # the hole moves to j
        self._keys[i] = _EMPTY
        self._values[i] = None
        self._size -= 1
        return True
```

It also satisfies requirement 2: the capacity doubles only when `3 * (keys + 1) > 2 * capacity`, at which point
`capacity < 1.5k`, and after doubling it is less than $3k$. How to choose: with few deletions, or if you want the
simplest `put`, use backward shift; with many deletions, and if you want `delete` itself to be as cheap as possible,
use tombstones and let rebuilds clean up in bulk.

### Part 3

Walk the slots in order, skipping empty slots and tombstones. To detect "the table changed during iteration",
maintain a version number: increment it whenever a key is added or deleted. The iterator records the version number
when it is created, and checks it every time it is asked for the next element. Overwriting the value of an existing
key doesn't change the version number and never triggers a rebuild, so it is allowed.

```python
class IterableMap(TombstoneMap):
    """Part 3: iteration in slot order; adding or removing a key during iteration is detected."""

    def __init__(self) -> None:
        super().__init__()
        self._version = 0                         # bumped whenever a key is added or removed

    def put(self, key, value) -> None:
        before = self._size
        super().put(key, value)
        if self._size != before:
            self._version += 1

    def delete(self, key) -> bool:
        removed = super().delete(key)
        if removed:
            self._version += 1
        return removed

    def items(self):
        return self._walk(self._version, self._keys, self._values)

    def _walk(self, version: int, keys: list, values: list):
        if self._version != version:
            raise RuntimeError("HashMap changed size during iteration")
        for i in range(len(keys)):
            k = keys[i]
            if k is not _EMPTY and k is not _TOMB:
                yield k, values[i]
                if self._version != version:      # checked when the caller asks for the next item
                    raise RuntimeError("HashMap changed size during iteration")

    def keys(self):
        return (k for k, _ in self.items())

    def __iter__(self):
        return self.keys()
```

**How the order relates to insertion order: it doesn't.** The order is determined by the low bits of the hashes and
the current capacity. In the example, 9 is inserted first; at capacity 8 it is in slot 1 and comes first; after
growing to 16 it is in slot 9 and comes last. So a single growth step can completely reshuffle the order. With string
keys, Python uses a different hash seed on each start, so two runs of the same program may iterate in different
orders. Callers must not rely on this order.

**Cost:** one full iteration is $O(\text{capacity})$, not $O(n)$: empty slots and tombstones also have to be looked
at. Under Part 2's rules, the capacity is at most $\max(8, 4k)$, where $k$ is the maximum number of keys ever present
at once, not the current number of keys. Insert a million keys and then delete down to ten, and an iteration still
scans the whole big table.

### Part 4

Hashing and equality are wrapped in the two methods `_hash` and `_same`, and all of the earlier code uses keys only
through them, so here we only need to replace those two methods.

```python
class HashMap(IterableMap):
    """Part 4: caller-supplied hash and equality."""

    def __init__(self, hash_fn=None, eq_fn=None) -> None:
        self._hash_fn = hash if hash_fn is None else hash_fn
        self._eq_fn = eq_fn
        super().__init__()

    def _hash(self, key) -> int:
        return self._hash_fn(key)

    def _same(self, a, b) -> bool:
        if a is b:
            return True
        return a == b if self._eq_fn is None else self._eq_fn(a, b)
```

"Each operation calls `hash_fn` exactly once, and growth never calls it" follows from the structure set up in Part 1:
each public method computes the hash once at the start and then passes it to `_find`; rebuilds use the stored hashes.
The contract "if `eq_fn(a, b)` is true then `hash_fn(a) == hash_fn(b)`" must be guaranteed by the caller: if two
equal keys hash differently, they start from different home slots, can't find each other, and "duplicate" keys
appear in the table.

**A key modified in place.** The table stores the hash computed back then and a reference to the object. After the
`list` is modified: looking up with the new contents computes a hash different from the stored one, so it isn't
found; looking up with the old contents matches the hash, but `==` fails, so it isn't found either. The entry is
still in the table, and both `len` and iteration see it; it just can never be looked up or deleted again. This is why
Python doesn't allow a `list` as a `dict` key; if you provide your own `hash_fn`, you have to guarantee yourself that
keys don't change after they are inserted.

**Keys that all collide.** A small integer's `hash` is the integer itself, and the home slot takes only the low bits
of the hash. The low 40 bits of `i << 40` are all 0, so all 2000 keys have home slot 0 and form one run: the $j$-th
key takes $j$ probes, 1000.5 on average, and the whole table degenerates into a linear search. The fix is to mix the
bits of the hash first so the high bits also affect the low bits, for example with splitmix64's finalizing mix
function:

```python
_MASK64 = (1 << 64) - 1


def mix64(key) -> int:
    """hash(key) pushed through the splitmix64 finaliser, so that high input bits reach the low bits."""
    x = hash(key) & _MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)
```

With `HashMap(hash_fn=mix64)` and the same 2000 keys, the average probe count is back to about 1.5. Note that this
only handles keys that "happen to have a pattern". `mix64` is a public, fixed function, and someone who knows it can
still construct a batch of keys that all collide; to defend against malicious input, the hash has to mix in a random
seed the other side doesn't know.

### Follow-ups

**Q: The `put` that triggers growth is $O(n)$. Can you avoid that stall?** You can do incremental resizing: keep the
old table and the new table at the same time, and on every subsequent operation move a fixed number of slots from the
old table to the new one; lookups check both tables, insertions go only into the new table; once the old table is
empty, drop it. The migration rate must ensure the new table doesn't fill up again before the old one is emptied.
That makes the migration work per operation constant, but the code is much more complex and lookups are noticeably
slower. One more point to make clear: in Python, merely allocating the new table with `[_EMPTY] * capacity` is
$O(\text{capacity})$, so unless the new table is also allocated in chunks on demand, the worst-case cost of a single
operation hasn't really dropped to $O(1)$.

**Q: CPython's `dict` also uses open addressing. How does it preserve insertion order?** It splits the data into two
layers: a compact entry array, appended to in insertion order (hash, key, value); and a sparse index table whose
slots store only indices into the entry array. Probing happens on the index table, and iteration walks the entry
array directly, so the order is insertion order. A deletion leaves a gap in the entry array, which is compacted on
the next rebuild. Also, its probe sequence is not linear: it uses the high bits of the hash as a perturbation, which
reduces the clustering described below.

**Q: How do you choose between linear probing and chaining?** Linear probing stores data contiguously, so probing is
a sequential memory read, which is cache-friendly, and there is no per-node allocation; the costs are that the load
factor must stay clearly below 1, deletion is awkward, and there is primary clustering: the longer a run is, the more
likely a new key lands in it and makes it longer, so it is more sensitive to hash function quality. Chaining makes
deletion simple, allows a load factor above 1, and degrades more gracefully with a mediocre hash; the costs are
pointer chasing and extra memory.

**Q: After you delete everything from the table, does the capacity shrink back?** In the implementation above,
`delete` never rebuilds, so with deletions only the capacity stays the same and the tombstones stay too; only the next
`put` that needs a new empty slot triggers a rebuild, which then shrinks the table according to the number of live
keys. To make deletion shrink the table promptly, add a rule to `delete`: rebuild when the number of keys drops below
1/8 of the capacity. There must be a gap between the shrink threshold and the growth threshold; otherwise alternating
inserts and deletes at the boundary trigger a rebuild on every operation.

**Q: Why is the capacity a power of two?** Then the modulo can be written as a single bitwise AND, and growing only
looks at one more bit. The cost is that only the low bits of the hash are used, and if the low bits are poorly
distributed you get the situation in Part 4. The other option is a prime capacity with `%`, which involves all the
bits, but the modulo is slower. With powers of two, putting the mixing into the hash function is the more common
approach.

<details>
<summary>Checks (runnable)</summary>

```python
import math
import random

# ---- Part 1 examples ----
for cls in (LinearProbingMap, TombstoneMap, ShiftDeleteMap, IterableMap, HashMap):
    m = cls()
    m.put("a", 1); m.put("b", 2); m.put("a", 3)
    assert len(m) == 2 and m.get("a") == 3
    assert m.get("zzz") is None and m.get("zzz", 0) == 0
    assert "b" in m and "zzz" not in m
    m.put(1, "int"); m.put(1.0, "float")
    assert len(m) == 3 and m.get(1) == "float" and m.get(True) == "float"
    # 8 slots hold at most 5 keys; the 6th distinct key doubles the table first
    m = cls()
    for k in range(5):
        m.put(k, k)
    assert m._capacity == 8
    m.put(0, "overwrite")                            # not a new key: no growth
    assert m._capacity == 8
    m.put(5, 5)
    assert m._capacity == 16 and len(m) == 6
    # None is an ordinary key and an ordinary value
    m.put(None, None)
    assert None in m and m.get(None, "default") is None and len(m) == 7

# ---- Part 2 example, and the naive deletion it rules out ----
for cls in (TombstoneMap, ShiftDeleteMap, HashMap):
    m = cls()
    for k in (0, 8, 16):
        m.put(k, str(k))
    assert m._keys[:3] == [0, 8, 16]                 # all three start at slot 0
    assert m.delete(8) is True
    assert m.get(16) == "16"
    assert m.delete(8) is False
    assert len(m) == 2
    m.put(8, "again")
    assert m.get(8) == "again" and len(m) == 3


class NaiveDeleteMap(LinearProbingMap):
    """What Part 2 warns against: deletion that just empties the slot."""

    def delete(self, key) -> bool:
        i, found = self._find(key, self._hash(key))
        if not found:
            return False
        self._keys[i] = _EMPTY
        self._values[i] = None
        self._size -= 1
        return True


m = NaiveDeleteMap()
for k in (0, 8, 16):
    m.put(k, str(k))
assert m.delete(8)
assert m.get(16) is None                             # lost: the probe stops at the emptied slot 1
m.put(16, "dup")
assert [k for k in m._keys if k is not _EMPTY].count(16) == 2   # and now the key is stored twice


# ---- random operation sequences against dict, invariants checked after every operation ----
def check_invariants(m) -> None:
    cap = m._capacity
    assert cap >= 8 and cap & (cap - 1) == 0
    live = sum(1 for k in m._keys if k is not _EMPTY and k is not _TOMB)
    tombs = sum(1 for k in m._keys if k is _TOMB)
    assert live == m._size == len(m)
    assert tombs == getattr(m, "_tombs", 0)
    assert 3 * (live + tombs) <= 2 * cap             # so at least one slot is always empty
    for i, k in enumerate(m._keys):
        if k is not _EMPTY and k is not _TOMB:
            h = m._hashes[i]
            assert h == m._hash(k)
            j = h & (cap - 1)
            while j != i:                            # no empty slot between home and actual slot
                assert m._keys[j] is not _EMPTY
                j = (j + 1) & (cap - 1)


def run_against_dict(cls, rng, steps, pool, check=True):
    """Returns the number of answers that differ from dict's."""
    m, model, wrong, peak = cls(), {}, 0, 0
    for _ in range(steps):
        op, key = rng.random(), rng.choice(pool)
        if op < 0.45:
            value = rng.randint(0, 999)
            m.put(key, value)
            model[key] = value
        elif op < 0.75:
            wrong += m.delete(key) != (key in model)
            model.pop(key, None)
        else:
            wrong += m.get(key, "missing") != model.get(key, "missing")
            wrong += (key in m) != (key in model)
        wrong += len(m) != len(model)
        peak = max(peak, len(model))
        if check:
            check_invariants(m)
            assert m._capacity <= max(8, 4 * peak)   # requirement 2 of Part 2
    wrong += sum(m.get(k, "missing") != v for k, v in model.items())
    return wrong


def random_pool(rng, size):
    makers = (
        lambda: rng.randint(-50, 50),
        lambda: rng.randint(0, 20) * 64,             # many keys sharing a home slot
        lambda: str(rng.randint(0, 30)),
        lambda: (rng.randint(0, 3), "t"),
        lambda: float(rng.randint(0, 5)),            # equal to some of the ints
        lambda: None,
        lambda: True,
    )
    return [rng.choice(makers)() for _ in range(size)]


rng = random.Random(20261117)
for cls in (TombstoneMap, ShiftDeleteMap, HashMap):
    for trial in range(120):
        pool = random_pool(rng, rng.choice([3, 10, 40, 150]))
        assert run_against_dict(cls, rng, 300, pool) == 0, cls.__name__

# the same workload exposes the naive deletion: most runs give wrong answers
broken_runs = sum(
    run_against_dict(NaiveDeleteMap, rng, 300, random_pool(rng, 40), check=False) > 0 for _ in range(50))
assert broken_runs >= 40, broken_runs


# ---- requirement 3: tombstones must count towards the load ----
class TombstoneBlindMap(TombstoneMap):
    """Load rule that counts live keys only."""

    def put(self, key, value) -> None:
        h = self._hash(key)
        i, found = self._find(key, h)
        if found:
            self._values[i] = value
            return
        if self._keys[i] is _TOMB:
            self._tombs -= 1
        elif 3 * (self._size + 1) > 2 * self._capacity:
            self._rebuild(2 * self._capacity)
            i = self._free_slot(h)
        self._hashes[i], self._keys[i], self._values[i] = h, key, value
        self._size += 1


def churn_until_full(cls, steps):
    """Keep 3 keys alive: insert a fresh key, delete the oldest. Returns the step at which no
    empty slot is left (the next lookup of an absent key would never return), or None."""
    m = cls()
    for k in range(3):
        m.put(k, k)
    for step in range(3, steps):
        m.put(step, step)
        assert m.delete(step - 3)
        if not any(k is _EMPTY for k in m._keys):
            return step
    return None


assert churn_until_full(TombstoneBlindMap, 1000) is not None
assert churn_until_full(TombstoneMap, 1000) is None
assert churn_until_full(ShiftDeleteMap, 1000) is None


# ---- delete-heavy workloads: bounded capacity, bounded rebuild work ----
class Tally:
    """Mixin that records the cost of every rebuild as old capacity + new capacity."""
    rebuild_slots = 0

    def _rebuild(self, capacity: int) -> None:
        type(self).rebuild_slots += self._capacity + capacity
        super()._rebuild(capacity)


for base in (TombstoneMap, ShiftDeleteMap):
    for window in (1, 5, 10, 100, 1000):
        cls = type("Tallied" + base.__name__, (Tally, base), {})
        m = cls()
        new_keys = 0
        for step in range(30 * window + 200):        # a sliding window: `window` to `window + 1` live keys
            m.put(step, step)
            new_keys += 1
            if step >= window:
                assert m.delete(step - window)
            assert m._capacity <= max(8, 4 * (window + 1))
        assert len(m) == window
        assert all(m.get(k) == k for k in range(step - window + 1, step + 1))
        assert step - window not in m
        assert cls.rebuild_slots <= 24 * new_keys, (base.__name__, window, cls.rebuild_slots, new_keys)

# pure growth: total rebuild work is below 9n, as computed in Part 1
for n in (6, 100, 1000, 5461, 5462, 20000):
    cls = type("TalliedGrow", (Tally, LinearProbingMap), {})
    m = cls()
    for k in range(n):
        m.put(k, k)
    assert cls.rebuild_slots < 9 * n, (n, cls.rebuild_slots)
    assert 3 * n <= 2 * m._capacity and (m._capacity == 8 or 3 * n > m._capacity)

# fill a table, delete everything, refill: tombstones are reused or cleaned up, nothing is lost
m = TombstoneMap()
for k in range(1000):
    m.put(k, k)
grown = m._capacity
for k in range(1000):
    assert m.delete(k)
assert len(m) == 0 and m._capacity == grown and m._tombs == 1000   # deletes alone never rebuild
assert all(k not in m for k in range(0, 2000, 7))                  # misses still terminate
m.put("x", 1)
m.put(123456, 2)
for k in range(5000, 5700):
    m.put(k, k)
assert len(m) == 702 and m.get("x") == 1 and m._capacity <= grown
check_invariants(m)


# ---- probe lengths ----
def probe_count(m, key) -> int:
    """Slots that a lookup of key inspects (the walk of _find, written out a second time)."""
    h = m._hash(key)
    mask = m._capacity - 1
    i, count = h & mask, 1
    while True:
        k = m._keys[i]
        if k is _EMPTY:
            return count
        if k is not _TOMB and m._hashes[i] == h and m._same(k, key):
            return count
        i = (i + 1) & mask
        count += 1


def mean_probes(m, keys) -> float:
    return sum(probe_count(m, k) for k in keys) / len(keys)


def knuth_hit(load):
    return 0.5 * (1 + 1 / (1 - load))


def knuth_miss(load):
    return 0.5 * (1 + 1 / (1 - load) ** 2)


assert abs(knuth_hit(2 / 3) - 2.0) < 1e-9 and abs(knuth_miss(2 / 3) - 5.0) < 1e-9
assert abs(knuth_miss(0.9) - 50.5) < 1e-9

def miss_probes_by_home(m) -> list:
    """For every possible home slot, the slots inspected by a lookup that fails from there."""
    cap = m._capacity
    cost, run = [0] * cap, None
    for i in range(2 * cap - 1, -1, -1):             # backwards and twice around, for the wrap-around
        if m._keys[i % cap] is _EMPTY:
            run = 0
        elif run is not None:
            run += 1
        if i < cap:
            cost[i] = run + 1
    return cost


capacity = 65536
full = (2 * capacity) // 3                           # the most keys this capacity ever holds
sample = rng.sample(range(1 << 60), full + 2000)     # random ints: hash(k) == k, uniform low bits
keys, absent = sample[:full], sample[full:]
m = HashMap()
for k in keys[:capacity // 2]:
    m.put(k, None)
assert m._capacity == capacity                       # load exactly 1/2
cost = miss_probes_by_home(m)
hit, miss = mean_probes(m, keys[:capacity // 2]), sum(cost) / capacity
assert abs(hit - knuth_hit(0.5)) < 0.05 and abs(miss - knuth_miss(0.5)) < 0.15, (hit, miss)
for k in keys[capacity // 2:]:
    m.put(k, None)
assert m._capacity == capacity and len(m) == full    # load 2/3: the highest the rule allows
cost = miss_probes_by_home(m)
assert all(probe_count(m, k) == cost[k & (capacity - 1)] for k in absent)
hit, miss = mean_probes(m, keys), sum(cost) / capacity
assert abs(hit - 2.0) < 0.1 and abs(miss - 5.0) < 0.4, (hit, miss)
print(f"capacity {capacity}, load 2/3: mean probes hit {hit:.2f} (theory 2), miss {miss:.2f} (theory 5)")

# keys that all collide under the default hash, and the fix from Part 4
colliding = [i << 40 for i in range(2000)]
assert all(hash(k) == k for k in colliding)
m = HashMap()
for k in colliding:
    m.put(k, None)
assert mean_probes(m, colliding) == 1000.5           # one run: the j-th key costs j probes
m = HashMap(hash_fn=mix64)
for k in colliding:
    m.put(k, None)
assert all(k in m for k in colliding) and len(m) == 2000
fixed = mean_probes(m, colliding)
assert fixed < 1.8, fixed
print(f"keys i << 40: mean hit probes 1000.5 with hash(), {fixed:.2f} with mix64")


# a hash function that maps everything to one value: n distinct keys cost exactly n(n-1)/2 comparisons
class Key:
    eq_calls = 0

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return 42

    def __eq__(self, other):
        Key.eq_calls += 1
        return isinstance(other, Key) and self.name == other.name


for cls in (TombstoneMap, ShiftDeleteMap, HashMap):
    n = 300
    keys = [Key(i) for i in range(n)]
    Key.eq_calls = 0
    m = cls()
    for i, k in enumerate(keys):
        m.put(k, i)
    assert Key.eq_calls == n * (n - 1) // 2          # the i-th insert is compared with all i earlier keys
    assert all(m.get(Key(i)) == i for i in range(n)) and Key("nope") not in m
    for i in range(0, n, 2):
        assert m.delete(Key(i))
    assert all((Key(i) in m) == (i % 2 == 1) for i in range(n)) and len(m) == n // 2
    check_invariants(m)


# ---- Part 3 ----
m = HashMap()
for k in (9, 1, 2):
    m.put(k, None)
assert list(m) == [9, 1, 2] and m._keys[1:4] == [9, 1, 2]
for k in (3, 4):
    m.put(k, None)
assert list(m) == [9, 1, 2, 3, 4] and m._capacity == 8
m.put(5, None)
assert list(m) == [1, 2, 3, 4, 5, 9] and m._capacity == 16
assert list(m.keys()) == list(m) and list(m.items()) == [(k, None) for k in list(m)]
assert list(HashMap()) == []

def expect_runtime_error(iterator) -> None:
    try:
        next(iterator)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


for change in (lambda m: m.put("new", 0), lambda m: m.delete(2)):
    for consumed in (0, 1, 5):                       # before the first item, in the middle, after the last
        for view in (iter, HashMap.keys, HashMap.items):
            m = HashMap()
            for k in range(5):
                m.put(k, k)
            it = view(m)
            for _ in range(consumed):
                next(it)
            change(m)
            expect_runtime_error(it)

m = HashMap()
for k in range(5):
    m.put(k, k)
for k, v in m.items():
    m.put(k, v * 10)                                 # overwriting values while iterating is allowed
assert sorted(m.items()) == [(k, k * 10) for k in range(5)]

for trial in range(200):                             # after any history: every key exactly once
    m, model = HashMap(), {}
    pool = random_pool(rng, 30)
    for _ in range(200):
        key = rng.choice(pool)
        if rng.random() < 0.6:
            m.put(key, 1)
            model[key] = 1
        else:
            m.delete(key)
            model.pop(key, None)
    seen = list(m)
    assert len(seen) == len(model) and all(k in model for k in seen)
    assert all(sum(1 for s in seen if s is k or s == k) == 1 for k in seen)


# ---- Part 4 ----
m = HashMap(hash_fn=lambda s: hash(s.lower()), eq_fn=lambda a, b: a.lower() == b.lower())
m.put("Key", 1); m.put("KEY", 2)
assert len(m) == 1 and m.get("key") == 2 and list(m) == ["Key"]

m = HashMap(hash_fn=lambda xs: hash(tuple(xs)))
m.put([1, 2], "a")
assert m.get([1, 2]) == "a" and [1, 2] in m and [2, 1] not in m

# overwriting keeps the key object that was stored first
m = HashMap()
m.put(1, "a"); m.put(1.0, "b"); m.put(True, "c")
assert list(m.items()) == [(1, "c")] and type(list(m)[0]) is int

# NaN behaves as in dict: the same object is the same key, another NaN is not
nan = float("nan")
m = HashMap()
m.put(nan, 1)
assert m.get(nan) == 1 and float("nan") not in m and {nan: 1}.get(nan) == 1

# hash_fn is called once per public operation and never during a rebuild
calls = []


def counting_hash(key):
    calls.append(key)
    return hash(key)


m = HashMap(hash_fn=counting_hash)
for k in range(100):                                 # several rebuilds happen on the way
    m.put(k, k)
assert m._capacity == 256 and len(calls) == 100
m.put(7, "x"); m.get(7); m.get(-1); 7 in m; m.delete(7); m.delete(7)
assert len(calls) == 106

# a key mutated after insertion can no longer be found, but is still stored
xs = [1, 2]
m = HashMap(hash_fn=lambda v: hash(tuple(v)))
m.put(xs, "a")
xs.append(3)
assert [1, 2, 3] not in m and xs not in m and [1, 2] not in m
assert len(m) == 1 and list(m) == [xs] and m.delete(xs) is False

# random operations with list keys under a custom hash, against a dict keyed by tuples
for trial in range(60):
    m, model = HashMap(hash_fn=lambda v: hash(tuple(v)) & 7), {}   # only 8 hash values: long runs
    for _ in range(300):
        key = [rng.randint(0, 3) for _ in range(rng.randint(0, 3))]
        op = rng.random()
        if op < 0.45:
            m.put(key, op)
            model[tuple(key)] = op
        elif op < 0.75:
            assert m.delete(key) == (tuple(key) in model)
            model.pop(tuple(key), None)
        else:
            assert m.get(key, "missing") == model.get(tuple(key), "missing")
        check_invariants(m)
    assert sorted(map(tuple, m)) == sorted(model)

print("all checks passed")
```

</details>

</details>
