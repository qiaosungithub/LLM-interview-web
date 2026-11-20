# Online Assessment: In-Memory Database (TTL, Conditional Writes, Backup and Historical Reads)

## Problem

Implement the four levels below in order: the tests for a level only run once every test of the previous
level passes. Each level keeps adding methods to the same `InMemoryDatabase` class rather than starting a
new class.

This is a different set of levels for the "in-memory database" problem, not the same one as
[Online Assessment: In-Memory Database](#/anthropic/coding/oa-in-memory-database). The main differences: in
Level 3 the timestamp is the **last** argument, in Level 4 it is the **first** argument; timestamps are
strictly increasing; there are conditional writes (compare-and-set) and conditional deletes
(compare-and-delete); `backup` returns a count; and Level 4 adds `get_when`, which reads the value at some
moment in the past. Don't mix the rules of the two pages; when working on this page, only look at this
page's statement.

General conventions:

- The database maps a `key` to a *record*, and a record maps a `field` (field name) to a `value`. A record
  with no fields is indistinguishable from a record that doesn't exist.
- `key`, `field`, `value`, `expected_value`, `new_value` and `prefix` are all non-empty, case-sensitive
  strings that never contain the three characters `(`, `)` and `,`.
- Every method returns a string. When there is nothing to return, it is the empty string `""`; success or
  failure is reported as lowercase `"true"` / `"false"`, not as booleans.

### Level 1 — Basic operations and conditional writes

```py
class InMemoryDatabase:
    def set(self, key: str, field: str, value: str) -> str:
        """Sets field of record key to value, creating the record or the field if needed, and
        overwriting the old value if there is one. Returns ""."""

    def get(self, key: str, field: str) -> str:
        """Returns the value of the field, or "" if the record or the field does not exist."""

    def delete(self, key: str, field: str) -> str:
        """Removes the field. Returns "true" if it existed, "false" otherwise."""

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        """If the field exists and its value equals expected_value, changes the value to
        new_value and returns "true". Otherwise changes nothing and returns "false"; in
        particular it never creates a field."""

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        """If the field exists and its value equals expected_value, removes the field and
        returns "true". Otherwise changes nothing and returns "false"."""
```

Example:

```text
set("u1", "name", "Ann")                        # -> ""
set("u1", "city", "Rome")                       # -> ""
get("u1", "name")                               # -> "Ann"
get("u1", "age")                                # -> ""        no such field
get("u2", "name")                               # -> ""        no such record
compare_and_set("u1", "name", "Bob", "Cat")     # -> "false"   the current value is "Ann", not "Bob"
compare_and_set("u1", "name", "Ann", "Cat")     # -> "true"
get("u1", "name")                               # -> "Cat"
compare_and_set("u1", "age", "30", "31")        # -> "false"   the field doesn't exist; a conditional write never creates a field
compare_and_delete("u1", "city", "Paris")       # -> "false"
compare_and_delete("u1", "city", "Rome")        # -> "true"
get("u1", "city")                               # -> ""
set("u1", "name", "Dan")                        # -> ""        overwrite
delete("u1", "name")                            # -> "true"
delete("u1", "name")                            # -> "false"   already deleted
delete("u9", "name")                            # -> "false"
```

### Level 2 — Scanning

```py
class InMemoryDatabase:
    def scan(self, key: str) -> str:
        """Returns all fields of the record as "<field1>(<value1>), <field2>(<value2>), ...",
        sorted by field name ascending (ordinary string comparison, so uppercase letters come
        before lowercase ones) and joined by a comma and one space. Returns "" if the record
        does not exist or has no fields."""

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        """The same as scan, but only the fields whose name starts with prefix."""
```

Example:

```text
set("u1", "name", "Ann"); set("u1", "nick", "an"); set("u1", "age", "30"); set("u1", "Zip", "00100")
scan("u1")                                      # -> "Zip(00100), age(30), name(Ann), nick(an)"   uppercase letters sort before lowercase
scan_by_prefix("u1", "n")                       # -> "name(Ann), nick(an)"
scan_by_prefix("u1", "na")                      # -> "name(Ann)"
scan_by_prefix("u1", "name")                    # -> "name(Ann)"    a field name counts as starting with itself
scan_by_prefix("u1", "z")                       # -> ""             case-sensitive: no field starts with lowercase z
scan("u2")                                      # -> ""
delete("u1", "age")                             # -> "true"
scan("u1")                                      # -> "Zip(00100), name(Ann), nick(an)"
```

### Level 3 — Timestamps and TTL

Every method from the first two levels gets a timestamped version, named with an `_at` suffix. `timestamp`
is the **last** argument; in the two methods with a TTL (time to live), `ttl` comes after `timestamp`. Tests
may pass `timestamp=`, `ttl=` and `prefix=` as keyword arguments, so these three parameter names must match
the ones below.

```py
class InMemoryDatabase:
    def set_at(self, key: str, field: str, value: str, timestamp: int) -> str: ...
    def set_at_with_ttl(self, key: str, field: str, value: str, timestamp: int, ttl: int) -> str: ...
    def get_at(self, key: str, field: str, timestamp: int) -> str: ...
    def delete_at(self, key: str, field: str, timestamp: int) -> str: ...
    def scan_at(self, key: str, timestamp: int) -> str: ...
    def scan_by_prefix_at(self, key: str, prefix: str, timestamp: int) -> str: ...
    def compare_and_set_at(self, key: str, field: str, expected_value: str, new_value: str,
                           timestamp: int) -> str: ...
    def compare_and_set_at_with_ttl(self, key: str, field: str, expected_value: str, new_value: str,
                                    timestamp: int, ttl: int) -> str: ...
    def compare_and_delete_at(self, key: str, field: str, expected_value: str, timestamp: int) -> str: ...
```

Return values follow the same conventions as the corresponding methods of the first two levels. Rules:

- `timestamp` and `ttl` are positive integers. On a given instance, the `timestamp` of timestamped calls
  (including the Level 4 methods) is strictly increasing and never repeats.
- A field written by `set_at_with_ttl` is valid during `[timestamp, timestamp + ttl)`: at the moment
  `timestamp + ttl` it has already expired. A field written by `set_at` never expires.
- When an existing field is overwritten, its old expiry time is discarded and recomputed from this write:
  overwritten by `set_at`, the field never expires; overwritten by `set_at_with_ttl`, the clock restarts from
  this call's `timestamp`.
- For the timestamped methods, an expired field simply doesn't exist: `get_at` returns `""`, `delete_at`
  returns `"false"`, both scans skip it, and conditional writes and conditional deletes return `"false"`.
- When a conditional write succeeds, the write itself is exactly the corresponding set:
  `compare_and_set_at` behaves like `set_at` (the new value never expires, even if the old value had a TTL),
  and `compare_and_set_at_with_ttl` behaves like `set_at_with_ttl`.
- The methods of the first two levels must keep working. Within one test, data is **written** with only one
  family of methods: either all untimestamped or all timestamped. Untimestamped reads (`get`, `scan`,
  `scan_by_prefix`) may appear after timestamped writes; in that case they ignore expiry: a field currently in
  the database, i.e. one that hasn't been overwritten, successfully deleted, or replaced by a Level 4
  `restore`, is returned as usual even if it has expired.

Example:

```text
set_at("u1", "name", "Ann", 1)                                   # -> ""
set_at_with_ttl("u1", "token", "t1", 2, 10)                      # -> ""        valid during [2, 12)
get_at("u1", "token", 11)                                        # -> "t1"
get_at("u1", "token", 12)                                        # -> ""        exactly 12: already expired
delete_at("u1", "token", 13)                                     # -> "false"   an expired field counts as nonexistent
set_at_with_ttl("u1", "token", "t2", 14, 5)                      # -> ""        [14, 19)
set_at_with_ttl("u1", "token", "t3", 16, 10)                     # -> ""        overwrite and restart the clock: [16, 26)
scan_at("u1", 19)                                                # -> "name(Ann), token(t3)"
compare_and_set_at("u1", "token", "t3", "t4", 20)                # -> "true"    written like set_at: t4 never expires
get_at("u1", "token", 30)                                        # -> "t4"
compare_and_set_at_with_ttl("u1", "name", "Ann", "Bob", 31, 4)   # -> "true"    name is now valid during [31, 35)
scan_by_prefix_at("u1", "n", 34)                                 # -> "name(Bob)"
compare_and_delete_at("u1", "name", "Bob", 35)                   # -> "false"   at time 35 it has already expired
scan_at("u1", 36)                                                # -> "token(t4)"
compare_and_set_at("u1", "name", "Bob", "Cat", 37)               # -> "false"   a conditional write can't revive an expired field
get("u1", "name")                                                # -> "Bob"     untimestamped reads ignore expiry
```

### Level 4 — Backup, restore and historical reads

In the three methods of this level, `timestamp` is the **first** argument. They all belong to the
"timestamped" family.

```py
class InMemoryDatabase:
    def backup(self, timestamp: int) -> str:
        """Saves the state of the database at timestamp: every field that has not expired,
        and, for each field with a TTL, its remaining lifetime (its expiry time minus
        timestamp). Returns the number of records that have at least one unexpired field at
        timestamp, as a string, e.g. "2"."""

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        """Replaces the whole database with the latest backup whose timestamp is less than or
        equal to timestamp_to_restore (such a backup is guaranteed to exist;
        timestamp_to_restore < timestamp). A restored field that had a remaining lifetime r in
        the backup now expires at timestamp + r; a field that had no TTL still has none.
        Returns ""."""

    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> str:
        """Returns what get_at(key, field, at_timestamp) would have returned at the moment
        at_timestamp, i.e. the value the field had then, or "" if it did not exist or had
        expired by then. 1 <= at_timestamp <= timestamp."""
```

Rules:

- Backups are never deleted and are not affected by later operations; the same backup can be restored many
  times. `restore` itself does not create a new backup.
- After a `restore`, the database holds exactly the fields in that backup: fields written after the backup
  disappear, and fields deleted or expired after the backup come back.
- `get_when` looks at the database as of the moment `at_timestamp`: only calls whose timestamp is at most
  `at_timestamp` take effect (including any `restore` before it), and expiry is judged at `at_timestamp`.
  Writes, deletes and `restore`s that happen later don't change the past: a `restore` changes the state from
  its own `timestamp` onward.
- When `at_timestamp` is exactly equal to some call's timestamp, that call has already taken effect.

Example:

```text
set_at("a", "x", "1", 1)                        # -> ""
set_at_with_ttl("a", "y", "2", 2, 10)           # -> ""       valid during [2, 12)
set_at_with_ttl("b", "z", "3", 3, 4)            # -> ""       valid during [3, 7)
backup(5)                                       # -> "2"      a and b both have unexpired fields; right now y has 7 left, z has 2 left
delete_at("a", "x", 6)                          # -> "true"
backup(8)                                       # -> "1"      b's only field expired at 7; right now y has 4 left
set_at("c", "w", "9", 9)                        # -> ""
restore(20, 7)                                  # -> ""       the latest backup at or before 7 is the one from time 5
scan_at("a", 21)                                # -> "x(1), y(2)"   y's new expiry time is 20 + 7 = 27
get_at("b", "z", 22)                            # -> ""       z's new expiry time is 20 + 2 = 22
get_at("c", "w", 23)                            # -> ""       c was written after the backup, so it doesn't exist after the restore
get_at("a", "y", 26)                            # -> "2"
get_at("a", "y", 27)                            # -> ""
backup(28)                                      # -> "1"      only a (field x) is left
get_when(30, "a", "x", 5)                       # -> "1"
get_when(31, "a", "x", 6)                       # -> ""       the delete at time 6 has already taken effect
get_when(32, "a", "x", 19)                      # -> ""
get_when(33, "a", "x", 20)                      # -> "1"      the restore at time 20 brought it back
get_when(34, "a", "y", 11)                      # -> "2"
get_when(35, "a", "y", 12)                      # -> ""       its original validity ended at 12
get_when(36, "a", "y", 26)                      # -> "2"      after the restore it is valid during [20, 27)
get_when(37, "b", "z", 21)                      # -> "3"      after the restore it is valid during [20, 22)
get_when(38, "c", "w", 10)                      # -> "9"      it still existed at time 10; restore doesn't rewrite the past
get_when(39, "c", "w", 20)                      # -> ""
get_when(40, "b", "z", 2)                       # -> ""       not written yet at that time
restore(41, 8)                                  # -> ""       the backup at time 8 only has a's y, with 4 left at that time
scan_at("a", 44)                                # -> "y(2)"
scan_at("a", 45)                                # -> ""       41 + 4 = 45
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Three points are worth confirming up front, because reasonable designs can differ on them: once TTL is
introduced, how the untimestamped methods of the first two levels handle expiry (here: they ignore expiry);
when a conditional write succeeds, whether the old value's TTL is discarded or kept (here: discarded; the
write itself is exactly the corresponding set); and when you ask after a `restore` about a moment before that
`restore`, whether you see the value that was really there at the time or the value in the backup (here: the
value really there at the time). In a real assessment, the platform's examples are the authority on all of
these; for anything the examples don't cover, choose the approach that changes the least.

### Time allocation

90 minutes for four levels, light at the start and heavy at the end. A workable budget, with the remainder
kept as slack:

| Level | Budget | Main work |
| --- | --- | --- |
| Level 1 | under 10 minutes | Five methods, two to four lines each |
| Level 2 | 5 to 10 minutes | One filtered sort-and-join |
| Level 3 | about 25 minutes | Store an extra expiry time; nine `_at` methods; turn the old methods into forwarders |
| Level 4 | about 30 minutes | Backup and restore; add a history log to writes and deletes |

A few practices that don't depend on the specific problem:

- Before starting each level, read its signatures and examples from top to bottom, especially the return
  values and argument order. In this set everything returned is a string (`"true"`, not `True`; the backup
  count is `"2"`, not `2`), and the timestamp is last in Level 3 but first in Level 4.
- Run the platform's tests after every method you write; don't let them pile up.
- Build new methods out of old ones as much as possible. A conditional write is just "read once, compare,
  then call set"; that way, in Level 3, the TTL rules only need to be implemented once, in the read and the
  write, and the conditional writes come out right automatically.
- You can't see the later levels in advance, so don't design ahead; but each time a level is added, first
  work out "how many entry points modify the data". Here, from Level 3 on, every write goes through `_write`
  and every delete goes through `_erase`, so the Level 4 history log only needs one extra line in each of
  those two places.
- An unfinished last level still earns points: [the interview process](#/anthropic/process) records
  candidates advancing to the next round with 750/1000 and 800/1000. In Level 4, write `backup` and `restore`
  first and leave `get_when` for last: it requires changing `_write` and `_erase` first, and breaking them
  would take down tests that already passed.
- Don't chase optimal complexity. Sorting at scan time and searching backwards linearly in `get_when` are the
  fastest to write; the statement makes no performance demands.

### Level 1

The whole database is a `dict[str, dict[str, str]]`: the outer level keyed by `key`, the inner level by
field name. `get` chains two `.get` calls, handling "the record doesn't exist" and "the field doesn't exist"
in a single line.

Both conditional writes and conditional deletes call `get` first. The problem guarantees that values are
non-empty strings, so the `""` that `get` returns for a missing field never equals any `expected_value`:
"the field doesn't exist" and "the value doesn't match" are both caught by the same comparison, with no
separate check needed.

```python
class InMemoryDatabase:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}          # key -> {field: value}

    def set(self, key: str, field: str, value: str) -> str:
        self._data.setdefault(key, {})[field] = value
        return ""

    def get(self, key: str, field: str) -> str:
        return self._data.get(key, {}).get(field, "")

    def delete(self, key: str, field: str) -> str:
        if field not in self._data.get(key, {}):
            return "false"
        del self._data[key][field]
        return "true"

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        if self.get(key, field) != expected_value:        # NOTE: a missing field reads as "", never equal
            return "false"
        self.set(key, field, new_value)
        return "true"

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        if self.get(key, field) != expected_value:
            return "false"
        return self.delete(key, field)
```

All five methods are $O(1)$ (counting string hashing and comparison as constant). The empty record left
behind after all its fields are deleted doesn't need cleaning up: `get` and `scan` on an empty record give
the same results as on a nonexistent record.

### Level 2

`scan` is just `scan_by_prefix` with an empty prefix, so only one is written. Sort the field names first,
then filter, and join with `", ".join`, so there are no extra separators at either end and an empty record
naturally yields `""`.

```python
class InMemoryDatabase(InMemoryDatabase):
    def scan(self, key: str) -> str:
        return self.scan_by_prefix(key, "")

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        record = self._data.get(key, {})
        return ", ".join(f"{field}({record[field]})" for field in sorted(record) if field.startswith(prefix))
```

Reopening the class and defining it as a subclass of itself is just how this page shows the code level by
level; in the assessment, add the methods directly to the original class. For a record with $m$ fields, one
scan is $O(m \log m)$.

### Level 3

The storage changes in only one place: the inner dictionary's value goes from `value` to
`(value, expires_at)`. A field that never expires records `expires_at` as `math.inf`, so "is it valid" is
always the same comparison `timestamp < expires_at`, with no branch for "no TTL"; `timestamp + math.inf` is
still `math.inf`, so computing the expiry time in `_write` needs no branch either.

Expiry is checked *lazily* (lazy expiration): expired fields are never actively deleted, only compared at
read time. The validity interval is the half-open `[timestamp, timestamp + ttl)`, so the comparison is a
strict `<`: when `timestamp` equals `expires_at`, the field has already expired.

All the rules live in just three places:

- `get_at` is the single point read that decides "is this field here right now"; `delete_at` and the three
  conditional methods all call it first, so "expired means nonexistent" automatically holds for them.
- `scan_by_prefix_at` filters out expired fields itself.
- `_write` is the only write entry point and replaces `(value, expires_at)` as a whole every time, so "the
  old expiry time is discarded on overwrite" automatically holds.

The untimestamped methods of the first two levels become one-line forwarders: equivalent to calling the
corresponding `_at` method at time $0$. Real timestamps are all positive integers and expiry times are at
least $2$, so seen from time $0$ no field has expired, which is exactly the required "ignore expiry"; a `set`
at time $0$ writes an expiry time of `math.inf`, i.e. it never expires.

```python
import math


class InMemoryDatabase(InMemoryDatabase):
    # self._data now maps key -> {field: (value, expires_at)}; expires_at is math.inf without a TTL

    def _write(self, key: str, field: str, value: str, timestamp: int, ttl: float = math.inf) -> None:
        self._data.setdefault(key, {})[field] = (value, timestamp + ttl)

    def _erase(self, key: str, field: str, timestamp: int) -> None:
        del self._data[key][field]

    def set_at(self, key: str, field: str, value: str, timestamp: int) -> str:
        self._write(key, field, value, timestamp)
        return ""

    def set_at_with_ttl(self, key: str, field: str, value: str, timestamp: int, ttl: int) -> str:
        self._write(key, field, value, timestamp, ttl)
        return ""

    def get_at(self, key: str, field: str, timestamp: int) -> str:
        value, expires_at = self._data.get(key, {}).get(field, ("", math.inf))
        return value if timestamp < expires_at else ""    # NOTE: strict -- at expires_at the field is already gone

    def delete_at(self, key: str, field: str, timestamp: int) -> str:
        if self.get_at(key, field, timestamp) == "":
            return "false"
        self._erase(key, field, timestamp)
        return "true"

    def scan_at(self, key: str, timestamp: int) -> str:
        return self.scan_by_prefix_at(key, "", timestamp)

    def scan_by_prefix_at(self, key: str, prefix: str, timestamp: int) -> str:
        record = self._data.get(key, {})
        return ", ".join(f"{field}({record[field][0]})" for field in sorted(record)
                         if field.startswith(prefix) and timestamp < record[field][1])

    def compare_and_set_at(self, key: str, field: str, expected_value: str, new_value: str,
                           timestamp: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return "false"
        self._write(key, field, new_value, timestamp)     # NOTE: exactly a set_at -- the old TTL is dropped
        return "true"

    def compare_and_set_at_with_ttl(self, key: str, field: str, expected_value: str, new_value: str,
                                    timestamp: int, ttl: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return "false"
        self._write(key, field, new_value, timestamp, ttl)
        return "true"

    def compare_and_delete_at(self, key: str, field: str, expected_value: str, timestamp: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return "false"
        self._erase(key, field, timestamp)
        return "true"

    # Levels 1-2: the same operations at time 0, before every real timestamp, when nothing has expired yet
    def set(self, key: str, field: str, value: str) -> str:
        return self.set_at(key, field, value, 0)

    def get(self, key: str, field: str) -> str:
        return self.get_at(key, field, 0)

    def delete(self, key: str, field: str) -> str:
        return self.delete_at(key, field, 0)

    def scan(self, key: str) -> str:
        return self.scan_at(key, 0)

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        return self.scan_by_prefix_at(key, prefix, 0)

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        return self.compare_and_set_at(key, field, expected_value, new_value, 0)

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        return self.compare_and_delete_at(key, field, expected_value, 0)
```

In the assessment, what you actually do at this level is: rename each of the seven Level 1–2 methods with an
`_at` suffix, add the `timestamp` parameter, switch the reads and writes to the form above, then add seven
one-line forwarders. `_erase`'s `timestamp` parameter isn't used at this level; it's only needed for the
Level 4 history log, and in the assessment there's still time to add it when you get to Level 4.

Point reads and writes are still $O(1)$, and a scan is still $O(m \log m)$, where $m$ includes expired fields
still left in the dictionary. The cost of lazy expiration is that expired fields keep occupying memory until
they're overwritten or replaced by a `restore`; see "Follow-ups".

### Level 4

**Backup.** Go over all the data, keep only the fields that haven't expired at this moment, and replace each
field's `expires_at` with its *remaining lifetime* `expires_at - timestamp` (for a field that never expires
this is `math.inf - timestamp`, still `math.inf`). Only records with at least one valid field are included,
so the count to return is just the length of this snapshot. The snapshot is a newly built dictionary whose
tuples are immutable, so later writes can't affect it.

**Restore.** Search from newest to oldest for the first backup whose timestamp is at most
`timestamp_to_restore`. Then delete all current fields through `_erase`, and write the snapshot's fields back
one by one through `_write`, passing the remaining lifetime as `ttl`: the expiry time `_write` computes is
then exactly `timestamp + remaining lifetime`. We don't simply replace `self._data` wholesale because the
history log below needs to know what happened to each field at this moment.

**Historical reads.** Keep an append-only list of versions for each `(key, field)`: each write appends
`(timestamp, value, expires_at)`, and each delete appends `(timestamp, "", math.inf)`. `_write` and `_erase`
are the only two entry points that modify data, so adding one line to each of them records every change,
`restore` included. `get_when` searches backwards for the first version whose timestamp is at most
`at_timestamp`, which is the version in effect at that time; then it uses `at_timestamp` to check whether that
version had already expired. A delete is recorded as a version with value `""`, so reading it naturally
gives `""`.

`restore` records, for one field and at the same timestamp, first a delete and then a write; searching
backwards, the first one you hit is the write recorded later, so the result is correct.

```python
class InMemoryDatabase(InMemoryDatabase):
    def __init__(self) -> None:
        super().__init__()
        self._backups: list = []      # (timestamp, {key: {field: (value, remaining)}}), oldest first
        self._history: dict = {}      # (key, field) -> [(timestamp, value, expires_at)]; value "" = deleted

    def _write(self, key: str, field: str, value: str, timestamp: int, ttl: float = math.inf) -> None:
        super()._write(key, field, value, timestamp, ttl)
        self._history.setdefault((key, field), []).append((timestamp, value, timestamp + ttl))

    def _erase(self, key: str, field: str, timestamp: int) -> None:
        super()._erase(key, field, timestamp)
        self._history[(key, field)].append((timestamp, "", math.inf))

    def backup(self, timestamp: int) -> str:
        snapshot = {}
        for key, record in self._data.items():
            alive = {field: (value, expires_at - timestamp)
                     for field, (value, expires_at) in record.items() if timestamp < expires_at}
            if alive:                                     # NOTE: a record whose fields have all expired is not counted
                snapshot[key] = alive
        self._backups.append((timestamp, snapshot))
        return str(len(snapshot))

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        snapshot = next(saved for when, saved in reversed(self._backups) if when <= timestamp_to_restore)
        for key, record in self._data.items():
            for field in list(record):                    # list(): _erase removes from record while we loop
                self._erase(key, field, timestamp)
        for key, record in snapshot.items():
            for field, (value, remaining) in record.items():
                self._write(key, field, value, timestamp, remaining)     # NOTE: expires at timestamp + remaining
        return ""

    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> str:
        for when, value, expires_at in reversed(self._history.get((key, field), [])):
            if when <= at_timestamp:
                return value if at_timestamp < expires_at else ""
        return ""
```

Let $F$ be the number of fields currently in the database (including expired fields not yet cleaned up).
`backup` is $O(F)$, and each backup takes $O(F)$ space. `restore` is $O(B + F + S)$: $B$ is the number of
backups and $S$ is the number of fields in that snapshot. `get_when` is $O(v)$, where $v$ is the number of
versions of this field. The history log grows by one entry per write or delete, and by $O(F + S)$ entries per
`restore`.

### Follow-ups

- **Can `get_when` be faster?** The version list is sorted by non-decreasing timestamp, so you can binary
  search: from Python 3.10 on you can write `bisect_right(versions, at_timestamp, key=lambda v: v[0]) - 1`;
  on older versions keep a separate parallel list of just the timestamps. When there are two entries at the
  same timestamp, `bisect_right` also lands on the later one. That makes it $O(\log v)$.
- **If the rule were "a conditional write keeps the old value's TTL", what would change?** Only the write
  line of `compare_and_set_at`: take out the old `expires_at` first, and pass `ttl = expires_at - timestamp`
  when writing. Because the expiry logic lives only in `get_at` and `_write`, no other method needs to change.
  In the assessment, look for an example line that "does a conditional write on a field with a TTL and then
  reads it again" to tell which rule applies.
- **Lazy expiration leaves expired fields occupying memory forever; how would you clean them up?** Add a
  min-heap ordered by expiry time, with elements `(expires_at, key, field)`. At the start of every timestamped
  call, pop all elements that are due; when popping, check whether this field's expiry time in the dictionary
  still equals the value recorded in the heap, because it may have been overwritten or deleted since. This
  isn't done here: in this problem, untimestamped reads must be able to see "expired" fields, and active
  cleanup would break that rule.
- **When backups are very frequent and the data is large, a full copy each time is too expensive; what then?**
  This page's implementation actually already has an alternative: the history log. A backup can just record a
  timestamp; on restore, read each field's version as of the backup moment from the history log, get the
  remaining lifetime as `expires_at - backup timestamp`, and write it back. The cost is a slower restore, and
  you need an index of "which fields exist under this record".
- **When asking about an earlier moment after a `restore`, why not return the value in the backup?** Both
  definitions make sense. The definition here is "what calling `get_at` at that time would have returned", so
  history is only ever appended to, never changed, which in the implementation is an append-only list. If the
  problem wanted the other one (after a restore, history also reverts to how it was at backup time), you would
  have to store the history log in the backup as well and swap it back wholesale on restore.
- **What if timestamps were only non-decreasing rather than strictly increasing?** Multiple calls at the same
  timestamp take effect in call order, timestamps repeat in the version list, and searching backwards still
  finds the last one to take effect. What you'd need to settle separately is how two backups at the same
  timestamp count (the later one replaces the earlier, or both are kept), which depends on the statement.

<details>
<summary>Verification code (runnable)</summary>

```python
import random
from collections import Counter

# The page builds the class level by level; pick the intermediate versions back out so that the code shown
# for Levels 1-2 (later overridden by one-line forwards) is tested as written.
_final, _level3, _level2, _level1 = InMemoryDatabase.__mro__[:4]
assert _level1.get is not _final.get and "compare_and_set" in vars(_level1) and "scan" in vars(_level2)
assert "set_at" in vars(_level3) and "backup" in vars(_final)

# ---- Level 1: the statement's example, on every version of the class ----
for cls in (_level1, _level2, _level3, _final):
    db = cls()
    assert db.set("u1", "name", "Ann") == ""
    assert db.set("u1", "city", "Rome") == ""
    assert db.get("u1", "name") == "Ann"
    assert db.get("u1", "age") == ""
    assert db.get("u2", "name") == ""
    assert db.compare_and_set("u1", "name", "Bob", "Cat") == "false"
    assert db.compare_and_set("u1", "name", "Ann", "Cat") == "true"
    assert db.get("u1", "name") == "Cat"
    assert db.compare_and_set("u1", "age", "30", "31") == "false"
    assert db.compare_and_delete("u1", "city", "Paris") == "false"
    assert db.compare_and_delete("u1", "city", "Rome") == "true"
    assert db.get("u1", "city") == ""
    assert db.set("u1", "name", "Dan") == ""
    assert db.delete("u1", "name") == "true"
    assert db.delete("u1", "name") == "false"
    assert db.delete("u9", "name") == "false"

# ---- Level 2: the statement's example ----
for cls in (_level2, _level3, _final):
    db = cls()
    for field, value in (("name", "Ann"), ("nick", "an"), ("age", "30"), ("Zip", "00100")):
        assert db.set("u1", field, value) == ""
    assert db.scan("u1") == "Zip(00100), age(30), name(Ann), nick(an)"
    assert db.scan_by_prefix("u1", "n") == "name(Ann), nick(an)"
    assert db.scan_by_prefix("u1", "na") == "name(Ann)"
    assert db.scan_by_prefix("u1", "name") == "name(Ann)"
    assert db.scan_by_prefix("u1", "z") == ""
    assert db.scan("u2") == ""
    assert db.delete("u1", "age") == "true"
    assert db.scan("u1") == "Zip(00100), name(Ann), nick(an)"

# ---- Level 3: the statement's example ----
for cls in (_level3, _final):
    db = cls()
    assert db.set_at("u1", "name", "Ann", 1) == ""
    assert db.set_at_with_ttl("u1", "token", "t1", 2, 10) == ""
    assert db.get_at("u1", "token", 11) == "t1"
    assert db.get_at("u1", "token", 12) == ""
    assert db.delete_at("u1", "token", 13) == "false"
    assert db.set_at_with_ttl("u1", "token", "t2", 14, 5) == ""
    assert db.set_at_with_ttl("u1", "token", "t3", 16, 10) == ""
    assert db.scan_at("u1", 19) == "name(Ann), token(t3)"
    assert db.compare_and_set_at("u1", "token", "t3", "t4", 20) == "true"
    assert db.get_at("u1", "token", 30) == "t4"
    assert db.compare_and_set_at_with_ttl("u1", "name", "Ann", "Bob", 31, 4) == "true"
    assert db.scan_by_prefix_at("u1", "n", 34) == "name(Bob)"
    assert db.compare_and_delete_at("u1", "name", "Bob", 35) == "false"
    assert db.scan_at("u1", 36) == "token(t4)"
    assert db.compare_and_set_at("u1", "name", "Bob", "Cat", 37) == "false"
    assert db.get("u1", "name") == "Bob"

# ---- Level 4: the statement's example ----
db = InMemoryDatabase()
assert db.set_at("a", "x", "1", 1) == ""
assert db.set_at_with_ttl("a", "y", "2", 2, 10) == ""
assert db.set_at_with_ttl("b", "z", "3", 3, 4) == ""
assert db.backup(5) == "2"
assert db.delete_at("a", "x", 6) == "true"
assert db.backup(8) == "1"
assert db.set_at("c", "w", "9", 9) == ""
assert db.restore(20, 7) == ""
assert db.scan_at("a", 21) == "x(1), y(2)"
assert db.get_at("b", "z", 22) == ""
assert db.get_at("c", "w", 23) == ""
assert db.get_at("a", "y", 26) == "2"
assert db.get_at("a", "y", 27) == ""
assert db.backup(28) == "1"
assert db.get_when(30, "a", "x", 5) == "1"
assert db.get_when(31, "a", "x", 6) == ""
assert db.get_when(32, "a", "x", 19) == ""
assert db.get_when(33, "a", "x", 20) == "1"
assert db.get_when(34, "a", "y", 11) == "2"
assert db.get_when(35, "a", "y", 12) == ""
assert db.get_when(36, "a", "y", 26) == "2"
assert db.get_when(37, "b", "z", 21) == "3"
assert db.get_when(38, "c", "w", 10) == "9"
assert db.get_when(39, "c", "w", 20) == ""
assert db.get_when(40, "b", "z", 2) == ""
assert db.restore(41, 8) == ""
assert db.scan_at("a", 44) == "y(2)"
assert db.scan_at("a", 45) == ""

# ---- the practice set's own example tables, transcribed. Its Level 3-4 tables write timestamps and ttl
#      as strings ("1", "9"); its tests pass integers, and so do these. ----
for cls in (_level1, _final):                                 # its Level 1 table
    db = cls()
    assert db.set("A", "B", "E") == ""
    assert db.set("A", "C", "F") == ""
    assert db.get("A", "B") == "E"
    assert db.get("A", "D") == ""
    assert db.delete("A", "B") == "true"
    assert db.delete("A", "D") == "false"

for cls in (_level2, _final):                                 # its Level 2 table
    db = cls()
    assert db.set("A", "BC", "E") == ""
    assert db.set("A", "BD", "F") == ""
    assert db.set("A", "C", "G") == ""
    assert db.scan_by_prefix("A", "B") == "BC(E), BD(F)"
    assert db.scan("A") == "BC(E), BD(F), C(G)"
    assert db.scan_by_prefix("B", "B") == ""

db = InMemoryDatabase()                                       # its first Level 3 table
assert db.set_at_with_ttl("A", "BC", "E", 1, 9) == ""
assert db.set_at_with_ttl("A", "BC", "E", 5, 10) == ""
assert db.set_at("A", "BD", "F", 6) == ""
assert db.scan_by_prefix_at("A", "B", 14) == "BC(E), BD(F)"
assert db.scan_by_prefix_at("A", "B", 15) == "BD(F)"

db = InMemoryDatabase()                                       # its second Level 3 table
assert db.set_at("A", "B", "C", 1) == ""
assert db.set_at_with_ttl("X", "Y", "Z", 2, 15) == ""
assert db.get_at("X", "Y", 3) == "Z"
assert db.set_at_with_ttl("A", "D", "E", 4, 10) == ""
assert db.scan_at("A", 13) == "B(C), D(E)"
assert db.scan_at("X", 16) == "Y(Z)"
assert db.scan_at("X", 17) == ""
assert db.delete_at("X", "Y", 20) == "false"

db = InMemoryDatabase()                                       # its Level 4 table
assert db.set_at_with_ttl("A", "B", "C", 1, 10) == ""
assert db.backup(3) == "1"
assert db.set_at("A", "D", "E", 4) == ""
assert db.backup(5) == "1"
assert db.delete_at("A", "B", 8) == "true"
assert db.backup(9) == "1"
assert db.restore(10, 7) == ""                                # the backup at 5; B had 6 left, so it now ends at 16
assert db.set_at("B", "C", "D", 11) == ""
assert db.backup(12) == "2"
assert db.scan_at("A", 15) == "B(C), D(E)"
assert db.scan_at("A", 16) == "D(E)"
assert db.scan_at("B", 17) == "C(D)"

# ---- the practice set's unit tests, transcribed (keyword arguments as in the source) ----
for cls in (_level1, _final):
    db = cls()
    assert db.set("user1", "name", "Alice") == ""
    assert db.set("user1", "age", "30") == ""
    assert db.get("user1", "name") == "Alice"
    assert db.get("user1", "age") == "30"

    db = cls()
    assert db.set("user1", "name", "Alice") == ""
    assert db.set("user1", "name", "Bob") == ""
    assert db.get("user1", "name") == "Bob"

    db = cls()
    assert db.get("user1", "field") == ""
    assert db.set("user1", "name", "Alice") == ""
    assert db.get("user1", "non_existent") == ""

    db = cls()
    assert db.set("user1", "name", "Alice") == ""
    assert db.delete("user1", "name") == "true"
    assert db.get("user1", "name") == ""
    assert db.delete("user1", "name") == "false"
    assert db.delete("non_existent", "field") == "false"

for cls in (_level2, _final):
    db = cls()
    for field, value in (("name", "Alice"), ("age", "30"), ("city", "NY"), ("abc", "123")):
        assert db.set("user1", field, value) == ""
    assert db.scan("user1") == "abc(123), age(30), city(NY), name(Alice)"
    assert db.scan("non_existent") == ""
    assert db.scan_by_prefix("user1", "a") == "abc(123), age(30)"
    assert db.scan_by_prefix("user1", "n") == "name(Alice)"
    assert db.scan_by_prefix("user1", "xyz") == ""

db = InMemoryDatabase()
assert db.set_at("user1", "name", "Alice", timestamp=100) == ""
assert db.set_at("user1", "age", "30", timestamp=101) == ""
assert db.get_at("user1", "name", timestamp=102) == "Alice"
assert db.get_at("user1", "age", timestamp=103) == "30"

db = InMemoryDatabase()
assert db.get_at("user2", "name", timestamp=100) == ""
assert db.get_at("user1", "non_existent", timestamp=101) == ""

db = InMemoryDatabase()
assert db.set_at_with_ttl("user1", "name", "Alice", timestamp=100, ttl=10) == ""
assert db.get_at("user1", "name", timestamp=105) == "Alice"
assert db.get_at("user1", "name", timestamp=110) == ""
assert db.get_at("user1", "name", timestamp=115) == ""

db = InMemoryDatabase()                                       # set_at over a TTL field: no expiry any more
assert db.set_at_with_ttl("user1", "name", "Alice", timestamp=100, ttl=10) == ""
assert db.set_at("user1", "name", "Bob", timestamp=105) == ""
assert db.get_at("user1", "name", timestamp=110) == "Bob"
assert db.get_at("user1", "name", timestamp=140) == "Bob"

db = InMemoryDatabase()                                       # set_at_with_ttl over a TTL field: [106, 116)
assert db.set_at_with_ttl("user1", "name", "Alice", timestamp=100, ttl=10) == ""
assert db.get_at("user1", "name", timestamp=105) == "Alice"
assert db.set_at_with_ttl("user1", "name", "Bob", timestamp=106, ttl=10) == ""
assert db.get_at("user1", "name", timestamp=110) == "Bob"
assert db.get_at("user1", "name", timestamp=117) == ""


def _user1(cls, with_nationality=False):
    db = cls()
    assert db.set_at_with_ttl("user1", "name", "Alice", timestamp=100, ttl=10) == ""     # [100, 110)
    assert db.set_at_with_ttl("user1", "age", "30", timestamp=101, ttl=5) == ""          # [101, 106)
    assert db.set_at_with_ttl("user1", "city", "NY", timestamp=102, ttl=15) == ""        # [102, 117)
    if with_nationality:
        assert db.set_at_with_ttl("user1", "nationality", "free_country", timestamp=103, ttl=5) == ""   # [103, 108)
    return db


for cls in (_level3, _final):
    db = _user1(cls)
    assert db.scan_at("user1", timestamp=105) == "age(30), city(NY), name(Alice)"
    assert db.scan_at("user1", timestamp=106) == "city(NY), name(Alice)"
    assert db.scan_at("user1", timestamp=110) == "city(NY)"
    assert db.scan_at("user1", timestamp=116) == "city(NY)"
    assert db.scan_at("user1", timestamp=117) == ""

    db = _user1(cls, with_nationality=True)
    assert db.scan_by_prefix_at("user1", prefix="a", timestamp=105) == "age(30)"
    assert db.scan_by_prefix_at("user1", prefix="a", timestamp=106) == ""
    assert db.scan_by_prefix_at("user1", prefix="n", timestamp=107) == "name(Alice), nationality(free_country)"
    assert db.scan_by_prefix_at("user1", prefix="n", timestamp=109) == "name(Alice)"

    # Three source tests call the plain reads right after timestamped writes, although the source's own
    # Level 3 text says a test never mixes the two families. They are what fixes the rule "a plain read
    # does not look at expiry".
    db = _user1(cls)
    assert db.get("user1", "name") == "Alice" and db.get("user1", "age") == "30" and db.get("user1", "city") == "NY"
    db = _user1(cls)
    assert db.scan("user1") == "age(30), city(NY), name(Alice)"
    db = _user1(cls, with_nationality=True)
    assert db.scan_by_prefix("user1", prefix="a") == "age(30)"
    assert db.scan_by_prefix("user1", prefix="n") == "name(Alice), nationality(free_country)"

db = InMemoryDatabase()
assert db.set_at_with_ttl("A", "B", "C", timestamp=1, ttl=10) == ""
assert db.backup(3) == "1"

db = InMemoryDatabase()
db.set_at_with_ttl("A", "B", "C", 1, 10)                      # expires at 11
assert db.backup(12) == "0"

# ---- edge cases beyond the source ----
db = InMemoryDatabase()                                       # every kind of call exactly at the expiry time
db.set_at_with_ttl("k", "f", "v", 10, 5)                      # [10, 15)
assert db.get_at("k", "f", 14) == "v"
db.set_at_with_ttl("k", "g", "v", 15, 1)                      # [15, 16)
assert db.scan_at("k", 16) == ""
db.set_at_with_ttl("k", "f", "v", 20, 5)                      # [20, 25)
assert db.compare_and_set_at("k", "f", "v", "w", 25) == "false"
db.set_at_with_ttl("k", "f", "v", 30, 5)                      # [30, 35)
assert db.compare_and_set_at_with_ttl("k", "f", "v", "w", 35, 9) == "false"
db.set_at_with_ttl("k", "f", "v", 40, 5)                      # [40, 45)
assert db.compare_and_delete_at("k", "f", "v", 45) == "false"
db.set_at_with_ttl("k", "f", "v", 50, 5)                      # [50, 55)
assert db.delete_at("k", "f", 55) == "false"
db.set_at_with_ttl("k", "f", "v", 60, 5)                      # [60, 65)
assert db.backup(65) == "0"
assert db.get_when(66, "k", "f", 64) == "v" and db.get_when(67, "k", "f", 65) == ""
assert [db.get_when(68, "k", "f", t) for t in (9, 10, 14, 15, 19, 20, 24, 25)] == ["", "v", "v", "", "", "v", "v", ""]

db = InMemoryDatabase()                                       # a conditional write resets the lifetime either way
db.set_at_with_ttl("k", "f", "v", 1, 10)                      # [1, 11)
assert db.compare_and_set_at_with_ttl("k", "f", "v", "w", 5, 3) == "true"     # now [5, 8)
assert db.get_at("k", "f", 7) == "w" and db.get_at("k", "f", 8) == ""
db.set_at_with_ttl("k", "f", "v", 9, 2)                       # [9, 11)
assert db.compare_and_set_at("k", "f", "v", "w", 10) == "true"                # permanent
assert db.get_at("k", "f", 1000) == "w"
assert db.compare_and_set_at("k", "f", "w", "w", 1001) == "true"              # same value: still succeeds
assert db.compare_and_delete_at("k", "f", "v", 1002) == "false"
assert db.compare_and_delete_at("k", "f", "w", 1003) == "true"
assert db.compare_and_delete_at("k", "f", "w", 1004) == "false"
assert db.compare_and_set_at("nokey", "f", "w", "x", 1005) == "false" and db.scan_at("nokey", 1006) == ""

db = InMemoryDatabase()                                       # backup counts records, and only live ones
db.set_at("a", "p", "1", 1)
db.set_at_with_ttl("a", "q", "2", 2, 3)                       # [2, 5)
db.set_at_with_ttl("b", "r", "3", 3, 3)                       # [3, 6)
db.set_at("c", "s", "4", 4)
assert db.backup(5) == "3"                                    # a (one live field of two), b, c
assert db.backup(6) == "2"                                    # b has expired
assert db.delete_at("c", "s", 7) == "true"
assert db.backup(8) == "1"                                    # c is an empty record now
assert db.restore(9, 5) == ""                                 # exactly the timestamp of a backup: that backup
assert db.scan_at("a", 10) == "p(1)" and db.get_at("b", "r", 10) == ""     # r had 1 left at 5: [9, 10)
assert db.restore(12, 5) == ""                                # the same backup again: lifetimes restart again
assert db.get_at("b", "r", 12) == "3" and db.get_at("b", "r", 13) == ""
assert db.backup(14) == "2"                                   # a and c
assert db.restore(15, 7) == ""                                # the backup at 6, not the one at 8
assert db.scan_at("c", 16) == "s(4)" and db.get_at("b", "r", 16) == ""
assert db.restore(17, 16) == ""                               # a backup made after an earlier restore
assert db.scan_at("a", 18) == "p(1)" and db.scan_at("c", 18) == "s(4)"
assert [db.get_when(19, "b", "r", t) for t in (2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15)] == \
    ["", "3", "3", "", "", "3", "", "", "3", "", ""]
assert [db.get_when(20, "c", "s", t) for t in (3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 17)] == \
    ["", "4", "4", "", "", "4", "4", "4", "4", "4", "4"]

db = InMemoryDatabase()                                       # the plain methods alone, on the finished class
assert db.set("k", "f", "v") == "" and db.compare_and_set("k", "f", "v", "w") == "true"
assert db.scan("k") == "f(w)" and db.compare_and_delete("k", "f", "v") == "false"
assert db.compare_and_delete("k", "f", "w") == "true" and db.scan("k") == "" and db.delete("k", "f") == "false"


# ---- an independent model: a flat list of rows, None for "never expires", and get_when answered by
#      replaying every earlier call into an empty database ----
class _SlowDatabase:
    CHANGES = ("set_at", "set_at_with_ttl", "delete_at", "compare_and_set_at", "compare_and_set_at_with_ttl",
               "compare_and_delete_at", "backup", "restore", "set", "delete", "compare_and_set",
               "compare_and_delete")

    def __init__(self):
        self.rows = []               # [key, field, value, end]; end is None for a field without TTL
        self.saved = []              # (timestamp, [(key, field, value, time left or None)])
        self.journal = []            # (when, name, args) of every call that can change something
        self.seen = Counter()        # how often the awkward situations came up

    def apply(self, when, name, *args):
        if name in self.CHANGES:
            self.journal.append((when, name, args))
        return getattr(self, name)(*args)

    def _row(self, key, field, timestamp):
        """The row of this field if it is alive at timestamp; timestamp None means "do not look at expiry"."""
        for row in self.rows:
            if row[0] == key and row[1] == field:
                self.seen["call exactly at an expiry"] += row[3] == timestamp
                if timestamp is None or row[3] is None or timestamp < row[3]:
                    return row
        return None

    def _store(self, key, field, value, end):
        self.rows = [row for row in self.rows if (row[0], row[1]) != (key, field)]
        self.rows.append([key, field, value, end])
        return ""

    def _scan(self, key, prefix, timestamp):
        found = [(row[1], row[2]) for row in self.rows
                 if row[0] == key and row[1][:len(prefix)] == prefix
                 and (timestamp is None or row[3] is None or timestamp < row[3])]
        self.seen["call exactly at an expiry"] += any(row[0] == key and row[3] == timestamp for row in self.rows)
        return ", ".join("%s(%s)" % pair for pair in sorted(found))

    def _conditional(self, key, field, expected, timestamp, then):
        row = self._row(key, field, timestamp)
        stale = [r for r in self.rows if (r[0], r[1], r[2]) == (key, field, expected)]
        self.seen["conditional call on an expired field with the expected value"] += row is None and bool(stale)
        if row is None or row[2] != expected:
            return "false"
        then(row)
        return "true"

    # timestamped methods
    def set_at(self, key, field, value, timestamp):
        return self._store(key, field, value, None)

    def set_at_with_ttl(self, key, field, value, timestamp, ttl):
        return self._store(key, field, value, timestamp + ttl)

    def get_at(self, key, field, timestamp):
        row = self._row(key, field, timestamp)
        return "" if row is None else row[2]

    def delete_at(self, key, field, timestamp):
        row = self._row(key, field, timestamp)
        if row is None:
            return "false"
        self.rows.remove(row)
        return "true"

    def scan_at(self, key, timestamp):
        return self._scan(key, "", timestamp)

    def scan_by_prefix_at(self, key, prefix, timestamp):
        return self._scan(key, prefix, timestamp)

    def compare_and_set_at(self, key, field, expected, new, timestamp):
        return self._conditional(key, field, expected, timestamp, lambda row: row.__setitem__(slice(2, 4), [new, None]))

    def compare_and_set_at_with_ttl(self, key, field, expected, new, timestamp, ttl):
        return self._conditional(key, field, expected, timestamp,
                                 lambda row: row.__setitem__(slice(2, 4), [new, timestamp + ttl]))

    def compare_and_delete_at(self, key, field, expected, timestamp):
        return self._conditional(key, field, expected, timestamp, self.rows.remove)

    # plain methods: no clock at all
    def set(self, key, field, value):
        return self._store(key, field, value, None)

    def get(self, key, field):
        row = self._row(key, field, None)
        return "" if row is None else row[2]

    def delete(self, key, field):
        row = self._row(key, field, None)
        if row is None:
            return "false"
        self.rows.remove(row)
        return "true"

    def scan(self, key):
        return self._scan(key, "", None)

    def scan_by_prefix(self, key, prefix):
        return self._scan(key, prefix, None)

    def compare_and_set(self, key, field, expected, new):
        return self._conditional(key, field, expected, None, lambda row: row.__setitem__(slice(2, 4), [new, None]))

    def compare_and_delete(self, key, field, expected):
        return self._conditional(key, field, expected, None, self.rows.remove)

    # Level 4
    def backup(self, timestamp):
        alive = [(k, f, v, None if end is None else end - timestamp)
                 for k, f, v, end in self.rows if end is None or timestamp < end]
        self.saved.append((timestamp, alive))
        return str(len({row[0] for row in alive}))

    def restore(self, timestamp, timestamp_to_restore):
        chosen = None
        for when, rows in self.saved:
            if when <= timestamp_to_restore:
                chosen = rows
        self.rows = [[k, f, v, None if left is None else timestamp + left] for k, f, v, left in chosen]
        self.seen["restored field with time left"] += sum(left is not None for _, _, _, left in chosen)
        return ""

    def get_when(self, timestamp, key, field, at_timestamp):
        replica = _SlowDatabase()
        for when, name, args in self.journal:
            if when <= at_timestamp:
                getattr(replica, name)(*args)
        restores = [when for when, name, _ in self.journal if name == "restore"]
        self.seen["get_when for a moment before a later restore"] += any(at_timestamp < when for when in restores)
        self.seen["get_when for a moment after a restore"] += any(when <= at_timestamp for when in restores)
        return replica.get_at(key, field, at_timestamp)


_PLAIN_READS = ["get", "scan", "scan_by_prefix"]
_PLAIN_WRITES = ["set", "set", "delete", "compare_and_set", "compare_and_delete"]
_TIMED = ["set_at", "set_at", "set_at_with_ttl", "set_at_with_ttl", "set_at_with_ttl", "set_at_with_ttl",
          "get_at", "get_at", "delete_at", "scan_at", "scan_by_prefix_at", "compare_and_set_at",
          "compare_and_set_at_with_ttl", "compare_and_delete_at", "backup", "restore", "restore",
          "get_when", "get_when", "get_when"]


def _random_db_run(db_class, seed, mode, steps=110):
    """Drive db_class and the model with the same random calls. Returns the model, or None on a mismatch.
    mode: "plain" (Level 1-2 methods only), "timed" (timestamped methods only), "mixed" (timed + plain reads)."""
    rng = random.Random(seed)
    fast, slow = db_class(), _SlowDatabase()
    keys, fields, values = ["k", "kk", "m"], ["a", "ab", "abc", "b", "ba", "B"], ["x", "y", "z"]
    now, marks = 0, []
    for _ in range(steps):
        key, field, value = rng.choice(keys), rng.choice(fields), rng.choice(values)
        prefix = rng.choice(["a", "ab", "b", "B", "c"])
        if mode == "plain":
            name = rng.choice(_PLAIN_READS + _PLAIN_WRITES)
        else:
            ends = [(row, row[3] + d) for row in slow.rows if row[3] is not None
                    for d in (-1, 0, 0, 1) if row[3] + d > now]
            if ends and rng.random() < 0.4:          # just before, exactly at, or just after some field expires,
                row, now = rng.choice(ends)          # usually calling on that very field
                key, field = (row[0], row[1]) if rng.random() < 0.7 else (key, field)
            else:
                now += rng.randint(1, 3)
            marks.append(now)
            name = rng.choice(_TIMED + (_PLAIN_READS if mode == "mixed" else []))
            if name == "restore" and not slow.saved:
                name = "backup"
        held = [row[2] for row in slow.rows if (row[0], row[1]) == (key, field)]
        expected = held[0] if held and rng.random() < 0.6 else rng.choice(values)
        if name in ("set", "get", "delete"):
            args = (key, field) + ((value,) if name == "set" else ())
        elif name in ("scan", "scan_by_prefix"):
            args = (key,) + ((prefix,) if name == "scan_by_prefix" else ())
        elif name == "compare_and_set":
            args = (key, field, expected, value)
        elif name == "compare_and_delete":
            args = (key, field, expected)
        elif name in ("set_at", "set_at_with_ttl"):
            args = (key, field, value, now) + ((rng.randint(1, 9),) if name == "set_at_with_ttl" else ())
        elif name in ("get_at", "delete_at"):
            args = (key, field, now)
        elif name == "scan_at":
            args = (key, now)
        elif name == "scan_by_prefix_at":
            args = (key, prefix, now)
        elif name in ("compare_and_set_at", "compare_and_set_at_with_ttl"):
            args = (key, field, expected, value, now) + ((rng.randint(1, 9),) if name.endswith("ttl") else ())
        elif name == "compare_and_delete_at":
            args = (key, field, expected, now)
        elif name == "backup":
            args = (now,)
        elif name == "restore":
            args = (now, min(now - 1, rng.choice(slow.saved)[0] + rng.choice([0, 0, 1, 2])))
        else:
            near = [t + d for t in marks for d in (-1, 0, 1) if 1 <= t + d <= now]
            args = (now, key, field, rng.choice(near))
        if getattr(fast, name)(*args) != slow.apply(now, name, *args):
            return None
        marks.extend(row[3] for row in slow.rows if row[3] is not None and name in ("set_at_with_ttl", "restore",
                                                                                    "compare_and_set_at_with_ttl"))
    return slow


for cls in (_level2, _level3, _final):
    assert all(_random_db_run(cls, seed, "plain") is not None for seed in range(100))
assert all(_random_db_run(_final, seed, "mixed") is not None for seed in range(200))
runs = [_random_db_run(_final, seed, "timed") for seed in range(500)]
assert all(run is not None for run in runs)
seen = sum((run.seen for run in runs), Counter())
# the random calls really reach the awkward situations, each of them hundreds of times
for situation in ("call exactly at an expiry", "conditional call on an expired field with the expected value",
                  "restored field with time left", "get_when for a moment before a later restore",
                  "get_when for a moment after a restore"):
    assert seen[situation] > 300, (situation, seen)


# ---- the comparison has teeth: each of these mistakes is caught by the same random calls ----
class _AliveAtExpiry(InMemoryDatabase):          # "<=" instead of "<": a field still readable at its expiry time
    def get_at(self, key, field, timestamp):
        value, expires_at = self._data.get(key, {}).get(field, ("", math.inf))
        return value if timestamp <= expires_at else ""


class _KeepsOldDeadline(InMemoryDatabase):       # restore puts back the original expiry time
    def restore(self, timestamp, timestamp_to_restore):
        made = max(when for when, _ in self._backups if when <= timestamp_to_restore)
        super().restore(timestamp, timestamp_to_restore)
        for key, record in self._data.items():
            for field, (value, expires_at) in list(record.items()):
                self._write(key, field, value, timestamp, expires_at - timestamp - (timestamp - made))
        return ""


class _StrictlyBefore(InMemoryDatabase):         # restore ignores a backup made exactly at timestamp_to_restore
    def restore(self, timestamp, timestamp_to_restore):
        if self._backups[0][0] < timestamp_to_restore:
            timestamp_to_restore -= 1
        return super().restore(timestamp, timestamp_to_restore)


class _CountsDeadRecords(InMemoryDatabase):      # backup counts records whose fields have all expired
    def backup(self, timestamp):
        super().backup(timestamp)
        return str(sum(1 for record in self._data.values() if record))


class _ConditionalKeepsTtl(InMemoryDatabase):    # a conditional write keeps the old expiry time
    def compare_and_set_at(self, key, field, expected_value, new_value, timestamp):
        if self.get_at(key, field, timestamp) != expected_value:
            return "false"
        self._write(key, field, new_value, timestamp, self._data[key][field][1] - timestamp)
        return "true"


class _ConditionalRevives(InMemoryDatabase):     # a conditional write compares the stored value, expired or not
    def compare_and_set_at(self, key, field, expected_value, new_value, timestamp):
        if self._data.get(key, {}).get(field, ("",))[0] != expected_value:
            return "false"
        self._write(key, field, new_value, timestamp)
        return "true"


class _RestoreRewritesPast(InMemoryDatabase):    # restore leaves no trace in the history
    def restore(self, timestamp, timestamp_to_restore):
        lengths = {name: len(versions) for name, versions in self._history.items()}
        super().restore(timestamp, timestamp_to_restore)
        for name, versions in self._history.items():
            del versions[lengths.get(name, 0):]
        return ""


for wrong in (_AliveAtExpiry, _KeepsOldDeadline, _StrictlyBefore, _CountsDeadRecords, _ConditionalKeepsTtl,
              _ConditionalRevives, _RestoreRewritesPast):
    assert any(_random_db_run(wrong, seed, "timed") is None for seed in range(60)), wrong.__name__
```

</details>

</details>
