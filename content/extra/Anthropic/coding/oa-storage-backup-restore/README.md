# Online Assessment: Cloud Storage with Delete, N Largest, Backup and Restore

## Problem

Implement an in-memory cloud storage system. The system only records each file's name and size, not its contents.

This problem is the same kind as [Online Assessment: Cloud File Storage](#/anthropic/coding/oa-cloud-storage) on this
site, but what circulates here is a different set of operations: the interface is string commands, files can be
deleted, you can get the n largest files under a prefix, and restoring a backup "restores as many as it can". The
rules of the two pages differ; don't mix them up.

Every operation goes through one function:

```py
def solution(queries: list[list[str]]) -> list[str]:
    """Run the queries in order on one storage that starts empty.
    Returns one string per query."""
```

Each command is a list of strings: the first element is the operation name, the rest are its arguments. **All
arguments are strings**, and numbers are written in decimal (for example `"10"`). Each command returns a string:
success or failure is `"true"` / `"false"`, numbers are converted to decimal strings, and "no result" is the empty
string `""`. What each operation returns exactly is described one by one below.

The problem has four levels, unlocked in order. Each level adds operations to the same system; the operations of
earlier levels keep working, and their tests keep running too.

Conventions that apply to all four levels:

- File names, user ids, and prefixes are non-empty strings without whitespace, and are case-sensitive. The file
  names in the examples look like paths (`/dir/file.txt`), but the system has no directories; a name is just an
  ordinary string.
- At any moment, a name corresponds to at most one file, no matter who owns it.
- `size`, `capacity`, and `n` are positive integers no greater than $10^9$. A single call has at most $10^4$
  commands.
- The input is guaranteed to be valid: the operation name is always one of those listed below, the number of
  arguments is right, and wherever a number is expected there is a number.
- A failed command does not change any state.
- Each level's example starts from an empty system.

### Level 1 — Add, get, delete

- `ADD_FILE <name> <size>`: add a file. Fails if a file with the same name already exists. Returns `"true"` on
  success and `"false"` on failure.
- `GET_FILE_SIZE <name>`: return the file's size; returns `""` if the file does not exist.
- `DELETE_FILE <name>`: delete the file and return the size of the deleted file; returns `""` if the file does not
  exist. After a deletion, the name can be used again.

```text
["ADD_FILE", "/dir1/dir2/file.txt", "10"]    -> "true"
["ADD_FILE", "/dir1/dir2/file.txt", "5"]     -> "false"   # a file with this name exists; the size is still 10
["GET_FILE_SIZE", "/dir1/dir2/file.txt"]     -> "10"
["DELETE_FILE", "/not-existing.file"]        -> ""
["DELETE_FILE", "/dir1/dir2/file.txt"]       -> "10"
["GET_FILE_SIZE", "/not-existing.file"]      -> ""
```

### Level 2 — The n largest files under a prefix

- `GET_N_LARGEST <prefix> <n>`: among the files whose names start with `prefix`, take the `n` largest and return
  `"<name1>(<size1>), <name2>(<size2>), ..."`.

Rules:

- "Starts with `prefix`" is a plain string prefix (`str.startswith`), unrelated to directory levels: the prefix
  `/dir/file` matches `/dir/file2` and `/dir/file1.txt`.
- Sort by size, largest first; for equal sizes, sort by name in ascending lexicographic order (ordinary string
  comparison).
- Items are separated by one comma and one space, `", "`.
- If fewer than `n` files match, return all of them. If none match, return `""`.

```text
["ADD_FILE", "/dir/file1.txt", "5"]          -> "true"
["ADD_FILE", "/dir/file2", "20"]             -> "true"
["ADD_FILE", "/dir/deeper/file3.mov", "9"]   -> "true"
["GET_N_LARGEST", "/dir", "2"]               -> "/dir/file2(20), /dir/deeper/file3.mov(9)"
["GET_N_LARGEST", "/dir/file", "3"]          -> "/dir/file2(20), /dir/file1.txt(5)"     # only two match
["GET_N_LARGEST", "/another_dir", "2"]       -> ""
["ADD_FILE", "/big_file.mp4", "20"]          -> "true"
["GET_N_LARGEST", "/", "2"]                  -> "/big_file.mp4(20), /dir/file2(20)"     # sizes tie, so sort by name
```

### Level 3 — Users and capacity

All users share one file system (names are still globally unique), but each user has a capacity limit: the total
size of the files the user owns may not exceed it. A user's **remaining capacity** = capacity limit − total size of
the files it currently owns.

- `ADD_USER <userId> <capacity>`: create a user. Fails if the user already exists. Returns `"true"` on success and
  `"false"` on failure.
- `ADD_FILE_BY <userId> <name> <size>`: add a file like `ADD_FILE`, but the file is owned by `userId`. Fails in any
  of these cases: the user does not exist; a file with the same name already exists (no matter who owns it); `size`
  is larger than the user's remaining capacity. On success returns the user's remaining capacity after the addition;
  on failure returns `""`. Using up the capacity exactly is allowed, and then it returns `"0"`.
- `MERGE_USER <userId1> <userId2>`: merge `userId2` into `userId1`. All of `userId2`'s files become owned by
  `userId1`; `userId2`'s capacity limit is added to `userId1`'s capacity limit (so the two remaining capacities also
  add up); then `userId2` is deleted. Returns `userId1`'s remaining capacity after the merge. If either user does not
  exist, or the two arguments are equal, it fails and returns `""`. The deleted user id can later be created again
  with `ADD_USER`, giving a brand-new, empty user.

This level also changes the meaning of the operations from the first two levels:

- Level 1's `ADD_FILE` now counts as performed by the special user `"admin"`. `"admin"` exists from the start and
  has unlimited capacity. The input guarantees that `"admin"` never appears as the `userId` of any operation that
  takes a `userId` argument (including the two Level 4 operations).
- `DELETE_FILE` can delete any file, no matter who owns it. Deleting a user's file increases that user's remaining
  capacity accordingly.
- `GET_FILE_SIZE` and `GET_N_LARGEST` still see all files, no matter who owns them.

```text
["ADD_USER", "user1", "200"]                       -> "true"
["ADD_USER", "user1", "100"]                       -> "false"   # already exists; capacity is still 200
["ADD_FILE_BY", "user1", "/dir/file.med", "50"]    -> "150"
["ADD_FILE_BY", "user1", "/big.blob", "140"]       -> "10"
["ADD_FILE_BY", "user1", "/file-small", "20"]      -> ""        # only 10 left
["ADD_FILE", "/dir/admin_file", "300"]             -> "true"    # admin, unlimited capacity
["ADD_USER", "user2", "110"]                       -> "true"
["ADD_FILE_BY", "user2", "/dir/file.med", "45"]    -> ""        # a file with this name exists (owned by user1)
["ADD_FILE_BY", "user2", "/new_file", "50"]        -> "60"
["MERGE_USER", "user1", "user2"]                   -> "70"      # limit 200 + 110 = 310, used 190 + 50 = 240
```

### Level 4 — Backup and restore

- `BACKUP_USER <userId>`: record the names and sizes of all files `userId` currently owns. The backup is stored
  elsewhere, and no later operation on files changes it. Only the latest backup is kept per user; a new one replaces
  the old one. Returns the number of files in the backup (can be `"0"`); returns `""` if the user does not exist.
- `RESTORE_USER <userId>`: restore `userId`'s files to what they were in the latest backup. Returns the number of
  files successfully restored; returns `""` if the user does not exist.

`RESTORE_USER` runs in these three steps:

1. Delete all files `userId` currently owns.
2. For each (name, size) in the backup: if no file with that name exists now, recreate it with the size from the
   backup, owned by `userId`, and count it as one successful restore. If the name is now a file owned by someone
   else (another user or `"admin"`), skip it and leave the other file untouched.
3. Return the number of files created in step 2.

Additional rules:

- If the user has never made a backup, treat it as "the backup is empty": all of its files are deleted, and it
  returns `"0"`.
- A restore does not consume the backup. The same backup can be restored many times.
- A restore does not change the user's capacity limit. After a restore, remaining capacity = capacity limit − total
  size of the restored files.
- A file that is owned by `userId` both at backup time and now is deleted in step 1, recreated in step 2 with the
  size from the backup, and also counts toward the returned number.
- `MERGE_USER` does not change `userId1`'s backup: that backup still contains only the files `userId1` itself owned
  when it made the backup. `userId2`'s backup is deleted along with `userId2`.

```text
["ADD_USER", "user", "100"]                        -> "true"
["ADD_FILE_BY", "user", "/dir/file1", "50"]        -> "50"
["ADD_FILE_BY", "user", "/file2.txt", "30"]        -> "20"
["RESTORE_USER", "user"]                           -> "0"      # no backup: both files are deleted
["ADD_FILE_BY", "user", "/file3.mp4", "60"]        -> "40"
["ADD_FILE_BY", "user", "/file4.txt", "10"]        -> "30"
["BACKUP_USER", "user"]                            -> "2"
["DELETE_FILE", "/file3.mp4"]                      -> "60"
["DELETE_FILE", "/file4.txt"]                      -> "10"
["ADD_FILE", "/file3.mp4", "140"]                  -> "true"   # admin takes this name
["ADD_FILE_BY", "user", "/dir/file5.new", "20"]    -> "80"
["RESTORE_USER", "user"]                           -> "1"      # file5.new deleted; file4.txt restored; file3.mp4 skipped
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Settle two things first, and all four levels get easier:

- **Strings are handled only at the outermost layer.** The class inside uses normal Python types: numbers are
  `int`, failure is `False` or `None`. `solution` converts arguments to numbers and then converts the results to the
  strings the problem asks for. That way every method is easy to write and easy to test, and the output format
  appears in only one place.
- **Adding and removing a file each have exactly one entry point** (`_create`, `_remove`). When Level 3 adds "who
  owns it" to files, only these two functions need to change, instead of patching bookkeeping code everywhere.

Below, each level adds its features in a subclass, which makes it easy to show the solution in stages. In the real
assessment you would simply change the same class.

### Level 1

One dictionary, name → size. Each of the three operations is a single hash lookup, $O(1)$ on average (plus the
time to hash the name).

```python
class StorageL1:
    def __init__(self):
        self.size = {}                       # file name -> size

    def add_file(self, name, size):
        if name in self.size:
            return False
        self._create(name, size)
        return True

    def get_file_size(self, name):
        return self.size.get(name)           # None if there is no such file

    def delete_file(self, name):
        if name not in self.size:
            return None
        return self._remove(name)

    # The only two places that add or remove a file. Later levels extend them.
    def _create(self, name, size):
        self.size[name] = size

    def _remove(self, name):
        return self.size.pop(name)
```

### Level 2

Scan all files once, keep those whose names start with `prefix`, sort, and take the first `n`. The sort key is
`(-size, name)`: negating the size turns "size descending, name ascending" into a single ordinary ascending sort.

With $F$ files in total and $M$ matching: filtering takes $F$ prefix comparisons, and sorting takes $O(M \log M)$
comparisons. At the scale of $10^4$ commands that is plenty.

```python
class StorageL2(StorageL1):
    def get_n_largest(self, prefix, n):
        matches = [(name, size) for name, size in self.size.items() if name.startswith(prefix)]
        matches.sort(key=lambda item: (-item[1], item[0]))       # size descending, then name ascending
        return matches[:n]
```

### Level 3

Each file additionally records an owner; each user records three things: the capacity limit `limit`, the bytes
used `used`, and the set of file names it owns `names`. `"admin"` is not put into these three tables: it has no
limit and needs no bookkeeping, so the checks simply route around it.

The invariant to maintain: for each user, `used` equals the total size of the files in `names`, and
`used <= limit`. Every addition and removal goes through `_create` and `_remove`, and `used` and `names` are updated
in those two places, so the invariant can't be missed. `DELETE_FILE` calls the same `_remove`, so "deleting a file
gives the space back to its owner" needs no separate code.

`MERGE_USER`: the files change owner, `used` adds up, and `limit` adds up. The statement's "remaining capacities also
add up" and "limits add up" are the same thing: remaining = limit − used; both terms add up, so the difference adds
up too. In the example, $10 + 60 = 70$.

Complexity: `ADD_USER` and `ADD_FILE_BY` are $O(1)$ on average; `MERGE_USER` is $O(k)$, where $k$ is the number of
files `userId2` owns.

```python
ADMIN = "admin"


class StorageL3(StorageL2):
    def __init__(self):
        super().__init__()
        self.owner = {}                      # file name -> user id, or ADMIN
        self.limit = {}                      # user id -> capacity limit    (ADMIN is never a key of these three)
        self.used = {}                       # user id -> total size of the files it owns
        self.names = {}                      # user id -> set of the file names it owns

    def _create(self, name, size, owner=ADMIN):
        super()._create(name, size)
        self.owner[name] = owner
        if owner != ADMIN:
            self.names[owner].add(name)
            self.used[owner] += size

    def _remove(self, name):
        owner = self.owner.pop(name)
        size = super()._remove(name)
        if owner != ADMIN:                   # deleting a file gives its bytes back to the owner
            self.names[owner].remove(name)
            self.used[owner] -= size
        return size

    def remaining(self, user):
        return self.limit[user] - self.used[user]

    def add_user(self, user, capacity):
        if user in self.limit:
            return False
        self.limit[user], self.used[user], self.names[user] = capacity, 0, set()
        return True

    def add_file_by(self, user, name, size):
        if user not in self.limit or name in self.size or size > self.remaining(user):
            return None
        self._create(name, size, user)
        return self.remaining(user)

    def merge_user(self, target, source):
        if target == source or target not in self.limit or source not in self.limit:
            return None
        moved = self.names.pop(source)
        for name in moved:
            self.owner[name] = target
        self.names[target] |= moved
        self.used[target] += self.used.pop(source)
        self.limit[target] += self.limit.pop(source)             # limits add up, so remaining capacities add up
        return self.remaining(target)
```

### Level 4

A backup is a dictionary: name → size. It must be a **copy** of the state at that moment. If you only store a
reference to `names[user]`, later additions and removals change the "backup" along with it.

The restore follows the statement's three steps: first delete all of the user's current files, then check the names
in the backup one by one and recreate the ones that aren't taken. Two points are worth thinking through here:

- **How do you tell that "the name is taken by someone else"?** After step 1, the user no longer owns any file. So
  in step 2, if a name is still in the `size` table, it must belong to someone else, and there is no need to compare
  owners.
- **Can a restore exceed the capacity?** No, so the code does no capacity check, just leaves an assertion. The
  reasoning: at the moment of the backup, the total size of the files in it did not exceed the limit at that time;
  the limit can only grow (through merges), never shrink; what is restored is a subset of the backup, and the user's
  existing files have all been deleted. So after the restore, the space used does not exceed the current limit.

On a merge, delete `userId2`'s backup as well. If you skip this step, a user later created under the same name will
"inherit" a backup that doesn't belong to it.

Complexity: `BACKUP_USER` is $O(k)$, where $k$ is the user's current number of files; `RESTORE_USER` is
$O(k + b)$, where $b$ is the number of files in the backup.

```python
class StorageL4(StorageL3):
    def __init__(self):
        super().__init__()
        self.backup = {}                     # user id -> {file name: size}, the latest backup only

    def backup_user(self, user):
        if user not in self.limit:
            return None
        self.backup[user] = {name: self.size[name] for name in self.names[user]}    # a copy, not a reference
        return len(self.backup[user])

    def restore_user(self, user):
        if user not in self.limit:
            return None
        for name in list(self.names[user]):                      # 1. drop everything the user owns now
            self._remove(name)
        restored = 0
        for name, size in self.backup.get(user, {}).items():     # 2. bring back what is not taken by someone else
            if name not in self.size:
                self._create(name, size, user)
                restored += 1
        assert self.used[user] <= self.limit[user]               # cannot fail: see the argument in the text
        return restored

    def merge_user(self, target, source):
        remaining = super().merge_user(target, source)
        if remaining is not None:
            self.backup.pop(source, None)                        # the account is gone, and its backup with it
        return remaining
```

Finally, the string layer. Note that `as_text` must check for `bool` before handling numbers: in Python `True` is
also an `int`, and if the order is reversed `"true"` becomes `"True"`.

```python
def as_text(value):
    """The string protocol of the assessment, applied in one place."""
    if value is None:
        return ""
    if isinstance(value, bool):              # must come before int: True is an int in Python
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(f"{name}({size})" for name, size in value)
    return str(value)


def solution(queries):
    storage = StorageL4()
    handlers = {
        "ADD_FILE": lambda name, size: storage.add_file(name, int(size)),
        "GET_FILE_SIZE": storage.get_file_size,
        "DELETE_FILE": storage.delete_file,
        "GET_N_LARGEST": lambda prefix, n: storage.get_n_largest(prefix, int(n)),
        "ADD_USER": lambda user, capacity: storage.add_user(user, int(capacity)),
        "ADD_FILE_BY": lambda user, name, size: storage.add_file_by(user, name, int(size)),
        "MERGE_USER": storage.merge_user,
        "BACKUP_USER": storage.backup_user,
        "RESTORE_USER": storage.restore_user,
    }
    return [as_text(handlers[op](*args)) for op, *args in queries]
```

### Follow-ups

**There are many files and `GET_N_LARGEST` is called very often. How do you speed it up?** Two places can change.
One is finding the matching files: put all names in a sorted structure; names sharing a prefix form one contiguous
run in lexicographic order, so binary-search for the start of that run, then walk forward until the first name that
doesn't start with `prefix`. That costs $O(\log F + M)$ comparisons instead of scanning all $F$. If you keep the
sorted names in a Python list, insertions and deletions have to shift elements, which is $O(F)$; to make everything
fast you need a balanced tree, or a trie. The other is taking the first `n`: `heapq.nsmallest(n, matches, key=...)`
is $O(M \log n)$, which beats a full sort when $n$ is much smaller than $M$.

**What if the problem were changed to "if even one file can't be restored, the whole restore fails and nothing
changes"?** The other cloud storage page on this site has exactly that rule. The way to write it is check first,
then act: first confirm that every name in the backup is either unused now or owned by the user, and only once all
of them pass start deleting and creating. This page's rule is "skip what can't be restored", which needs no
up-front check, but the return value becomes "how many were restored". The test cases of the two rules don't carry
over to each other, so read the statement carefully to see which one it is.

**Why does the backup store only names and sizes, and not the capacity limit?** Because the statement says a restore
does not change the capacity. The other page on this site has the opposite rule: the backup stores the capacity
limit too, and a restore sets the limit back as well. Whichever rule applies, when writing the restore you should ask
"can the restore exceed the capacity?" and be able to give the reason. This page's reason is in Level 4: the limit
only grows, never shrinks, and what is restored is only a subset of the backup.

**After a merge, what happens when `userId1` restores its backup?** All files taken over from `userId2` are deleted.
The backup contains only `userId1`'s own files from back then, and step 1 of the restore deletes all the files it
owns **now**. This follows directly from the rules, and it is also one of the easiest cases to get wrong: some people
merge `userId2`'s backup into `userId1`'s backup during the merge, and the statement explicitly says not to.

**Four levels in 90 minutes: how do you budget the time?** A few minutes each for the first two levels, saving the
time for the last two. After passing each level, run that level's example; entering Level 3 is where you are most
likely to break earlier things, because the meanings of `ADD_FILE` and `DELETE_FILE` change, so after the change
rerun the examples of Levels 1 and 2. Keep the return-value formats (`"true"`, `""`, `", "`) in one function so
nothing is missed when they change.

<details>
<summary>Checks (runnable)</summary>

```python
import random

# --- 1. the example tables of the four levels (each starts from an empty storage) ---
LEVEL1 = [
    (["ADD_FILE", "/dir1/dir2/file.txt", "10"], "true"),
    (["ADD_FILE", "/dir1/dir2/file.txt", "5"], "false"),
    (["GET_FILE_SIZE", "/dir1/dir2/file.txt"], "10"),
    (["DELETE_FILE", "/not-existing.file"], ""),
    (["DELETE_FILE", "/dir1/dir2/file.txt"], "10"),
    (["GET_FILE_SIZE", "/not-existing.file"], ""),
]
LEVEL2 = [
    (["ADD_FILE", "/dir/file1.txt", "5"], "true"),
    (["ADD_FILE", "/dir/file2", "20"], "true"),
    (["ADD_FILE", "/dir/deeper/file3.mov", "9"], "true"),
    (["GET_N_LARGEST", "/dir", "2"], "/dir/file2(20), /dir/deeper/file3.mov(9)"),
    (["GET_N_LARGEST", "/dir/file", "3"], "/dir/file2(20), /dir/file1.txt(5)"),
    (["GET_N_LARGEST", "/another_dir", "2"], ""),
    (["ADD_FILE", "/big_file.mp4", "20"], "true"),
    (["GET_N_LARGEST", "/", "2"], "/big_file.mp4(20), /dir/file2(20)"),
]
LEVEL3 = [
    (["ADD_USER", "user1", "200"], "true"),
    (["ADD_USER", "user1", "100"], "false"),
    (["ADD_FILE_BY", "user1", "/dir/file.med", "50"], "150"),
    (["ADD_FILE_BY", "user1", "/big.blob", "140"], "10"),
    (["ADD_FILE_BY", "user1", "/file-small", "20"], ""),
    (["ADD_FILE", "/dir/admin_file", "300"], "true"),
    (["ADD_USER", "user2", "110"], "true"),
    (["ADD_FILE_BY", "user2", "/dir/file.med", "45"], ""),
    (["ADD_FILE_BY", "user2", "/new_file", "50"], "60"),
    (["MERGE_USER", "user1", "user2"], "70"),
]
LEVEL4 = [
    (["ADD_USER", "user", "100"], "true"),
    (["ADD_FILE_BY", "user", "/dir/file1", "50"], "50"),
    (["ADD_FILE_BY", "user", "/file2.txt", "30"], "20"),
    (["RESTORE_USER", "user"], "0"),
    (["ADD_FILE_BY", "user", "/file3.mp4", "60"], "40"),
    (["ADD_FILE_BY", "user", "/file4.txt", "10"], "30"),
    (["BACKUP_USER", "user"], "2"),
    (["DELETE_FILE", "/file3.mp4"], "60"),
    (["DELETE_FILE", "/file4.txt"], "10"),
    (["ADD_FILE", "/file3.mp4", "140"], "true"),
    (["ADD_FILE_BY", "user", "/dir/file5.new", "20"], "80"),
    (["RESTORE_USER", "user"], "1"),
]


def check_table(table, run):
    got = run([query for query, _ in table])
    want = [answer for _, answer in table]
    assert got == want, (got, want)


for table in (LEVEL1, LEVEL2, LEVEL3, LEVEL4):
    check_table(table, solution)

# What the Level 4 table leaves behind: file3 is admin's 140-byte file, file4 is back, file5 is gone.
assert solution([q for q, _ in LEVEL4] + [["GET_N_LARGEST", "/", "9"], ["ADD_FILE_BY", "user", "/x", "90"]])[-2:] == [
    "/file3.mp4(140), /file4.txt(10)", "0"]

# Without the ADD_FILE row nobody has taken "/file3.mp4", and the stated rules restore both files.
without_admin_row = [row for row in LEVEL4 if row[0][0] != "ADD_FILE"]
assert solution([q for q, _ in without_admin_row])[-1] == "2"

# --- 2. edge cases of the stated rules ---
def run(*queries):
    return solution([q.split() for q in queries])


# Level 1-2: the name is free again after a delete; n larger than the number of matches; plain string prefixes
assert run("ADD_FILE /a 5", "DELETE_FILE /a", "ADD_FILE /a 7", "GET_FILE_SIZE /a") == ["true", "5", "true", "7"]
assert run("ADD_FILE /ab 5", "ADD_FILE /a/b 5", "ADD_FILE /b 9", "GET_N_LARGEST /a 10", "GET_N_LARGEST /a 1",
           "GET_N_LARGEST /ab/ 1") == ["true", "true", "true", "/a/b(5), /ab(5)", "/a/b(5)", ""]
assert run("ADD_FILE A 1", "ADD_FILE a 1", "ADD_FILE B 1", "GET_N_LARGEST A 5", "GET_N_LARGEST a 5") == [
    "true", "true", "true", "A(1)", "a(1)"]                      # case sensitive
assert run("ADD_FILE f10 3", "ADD_FILE f9 3", "ADD_FILE f100 3", "GET_N_LARGEST f 3")[-1] == "f10(3), f100(3), f9(3)"

# Level 3: filling the capacity exactly is allowed; unknown users; deleting gives the bytes back to the owner
assert run("ADD_USER u 10", "ADD_FILE_BY u /a 10", "ADD_FILE_BY u /b 1", "ADD_FILE_BY nobody /b 1",
           "DELETE_FILE /a", "ADD_FILE_BY u /b 4", "GET_FILE_SIZE /b") == ["true", "0", "", "", "10", "6", "4"]
# a name is unique across all owners, admin included
assert run("ADD_FILE /a 1", "ADD_USER u 10", "ADD_FILE_BY u /a 1", "ADD_FILE_BY u /b 1", "ADD_FILE /b 1") == [
    "true", "true", "", "9", "false"]
# merge: failures change nothing; limits add up; the merged id is free again and starts empty
assert run("ADD_USER a 10", "ADD_USER b 20", "MERGE_USER a a", "MERGE_USER a c", "MERGE_USER c a",
           "ADD_FILE_BY a /a 7", "ADD_FILE_BY b /b 5", "MERGE_USER a b", "ADD_FILE_BY b /c 1",
           "ADD_USER b 3", "ADD_FILE_BY b /c 3", "DELETE_FILE /b", "ADD_FILE_BY a /d 23", "ADD_FILE_BY a /e 1") == [
    "true", "true", "", "", "", "3", "15", "18", "", "true", "0", "5", "0", ""]

# Level 4
# backup of an unknown user, of no files; a restore does not use the backup up; a newer backup replaces the older
assert run("BACKUP_USER u", "RESTORE_USER u", "ADD_USER u 10", "BACKUP_USER u", "ADD_FILE_BY u /a 4",
           "RESTORE_USER u", "GET_FILE_SIZE /a", "ADD_FILE_BY u /a 4", "BACKUP_USER u", "DELETE_FILE /a",
           "RESTORE_USER u", "DELETE_FILE /a", "RESTORE_USER u", "GET_FILE_SIZE /a") == [
    "", "", "true", "0", "6", "0", "", "6", "1", "4", "1", "4", "1", "4"]
# a file the user owns both at backup time and now is restored with the backed-up size, and it counts
assert run("ADD_USER u 10", "ADD_FILE_BY u /a 4", "BACKUP_USER u", "DELETE_FILE /a", "ADD_FILE_BY u /a 9",
           "RESTORE_USER u", "GET_FILE_SIZE /a", "ADD_FILE_BY u /b 6") == ["true", "6", "1", "4", "1", "1", "4", "0"]
# names taken by another user or by admin are skipped and left alone
assert run("ADD_USER u 10", "ADD_USER v 10", "ADD_FILE_BY u /a 1", "ADD_FILE_BY u /b 2", "ADD_FILE_BY u /c 3",
           "BACKUP_USER u", "DELETE_FILE /a", "DELETE_FILE /b", "ADD_FILE_BY v /a 7", "ADD_FILE /b 99",
           "RESTORE_USER u", "GET_N_LARGEST / 5", "ADD_FILE_BY u /d 7", "ADD_FILE_BY v /e 3") == [
    "true", "true", "9", "7", "4", "3", "1", "2", "3", "true", "1", "/b(99), /a(7), /c(3)", "0", "0"]
# merge keeps the target's backup as it was and throws the source's backup away
assert run("ADD_USER a 10", "ADD_USER b 10", "ADD_FILE_BY a /a 1", "ADD_FILE_BY b /b 2", "BACKUP_USER a",
           "BACKUP_USER b", "MERGE_USER a b", "RESTORE_USER a", "GET_FILE_SIZE /b", "GET_FILE_SIZE /a",
           "ADD_USER b 5", "RESTORE_USER b", "ADD_FILE_BY a /c 19") == [
    "true", "true", "9", "8", "1", "1", "17", "1", "", "1", "true", "0", "0"]
# after a merge the target can own a name that its own backup also lists: its files go first, then the backup returns
assert run("ADD_USER a 10", "ADD_USER b 10", "ADD_FILE_BY a /f 3", "BACKUP_USER a", "DELETE_FILE /f",
           "ADD_FILE_BY b /f 8", "MERGE_USER a b", "RESTORE_USER a", "GET_FILE_SIZE /f") == [
    "true", "true", "7", "1", "3", "2", "12", "1", "3"]


# --- 3. random sequences against an independent, deliberately naive model ---
class SlowStorage:
    """Flat lists, no bookkeeping: every answer is recomputed by scanning."""

    def __init__(self):
        self.rows = []          # [name, size, owner]
        self.users = []         # [user, limit]
        self.snaps = []         # [user, [(name, size), ...]]

    def row(self, name):
        hits = [r for r in self.rows if r[0] == name]
        return hits[0] if hits else None

    def user(self, user):
        hits = [u for u in self.users if u[0] == user]
        return hits[0] if hits else None

    def left(self, user):
        return self.user(user)[1] - sum(r[1] for r in self.rows if r[2] == user)

    def run(self, query):
        op, args = query[0], query[1:]
        if op == "ADD_FILE":
            if self.row(args[0]):
                return "false"
            self.rows.append([args[0], int(args[1]), "admin"])
            return "true"
        if op == "GET_FILE_SIZE":
            r = self.row(args[0])
            return str(r[1]) if r else ""
        if op == "DELETE_FILE":
            r = self.row(args[0])
            if not r:
                return ""
            self.rows.remove(r)
            return str(r[1])
        if op == "GET_N_LARGEST":
            hits = sorted(r[0] for r in self.rows if r[0][:len(args[0])] == args[0])
            hits.sort(key=lambda name: self.row(name)[1], reverse=True)          # stable: names stay ascending
            return ", ".join("%s(%d)" % (name, self.row(name)[1]) for name in hits[:int(args[1])])
        if op == "ADD_USER":
            if self.user(args[0]):
                return "false"
            self.users.append([args[0], int(args[1])])
            return "true"
        if op == "ADD_FILE_BY":
            user, name, size = args[0], args[1], int(args[2])
            if not self.user(user) or self.row(name) or self.left(user) < size:
                return ""
            self.rows.append([name, size, user])
            return str(self.left(user))
        if op == "MERGE_USER":
            a, b = self.user(args[0]), self.user(args[1])
            if not a or not b or a is b:
                return ""
            for r in self.rows:
                if r[2] == b[0]:
                    r[2] = a[0]
            a[1] += b[1]
            self.users.remove(b)
            self.snaps = [s for s in self.snaps if s[0] != b[0]]
            return str(self.left(a[0]))
        if op == "BACKUP_USER":
            if not self.user(args[0]):
                return ""
            self.snaps = [s for s in self.snaps if s[0] != args[0]]
            self.snaps.append([args[0], [(r[0], r[1]) for r in self.rows if r[2] == args[0]]])
            return str(len(self.snaps[-1][1]))
        if op == "RESTORE_USER":
            if not self.user(args[0]):
                return ""
            self.rows = [r for r in self.rows if r[2] != args[0]]
            count = 0
            for snap_user, files in self.snaps:
                if snap_user == args[0]:
                    for name, size in files:
                        if not self.row(name):
                            self.rows.append([name, size, args[0]])
                            count += 1
            assert self.left(args[0]) >= 0
            return str(count)
        raise AssertionError(op)


NAMES = ["/a", "/a/b", "/a/c", "/ab", "/b", "/b/a", "/b/ab", "/c"]
PREFIXES = ["/", "/a", "/a/", "/b", "/b/a", "/c", "/zz"]
USERS = ["u1", "u2", "u3", "u4"]
WEIGHTED_OPS = (["ADD_FILE"] * 3 + ["GET_FILE_SIZE"] + ["DELETE_FILE"] * 5 + ["GET_N_LARGEST"] * 2 + ["ADD_USER"] * 2
                + ["ADD_FILE_BY"] * 8 + ["MERGE_USER"] + ["BACKUP_USER"] * 2 + ["RESTORE_USER"] * 2)


def random_query(rng):
    op = rng.choice(WEIGHTED_OPS)
    if op == "ADD_FILE":
        return [op, rng.choice(NAMES), str(rng.randint(1, 40))]
    if op in ("GET_FILE_SIZE", "DELETE_FILE"):
        return [op, rng.choice(NAMES)]
    if op == "GET_N_LARGEST":
        return [op, rng.choice(PREFIXES), str(rng.randint(1, 5))]
    if op == "ADD_USER":
        return [op, rng.choice(USERS), str(rng.randint(30, 120))]
    if op == "ADD_FILE_BY":
        return [op, rng.choice(USERS), rng.choice(NAMES), str(rng.randint(1, 40))]
    if op == "MERGE_USER":
        return [op, rng.choice(USERS), rng.choice(USERS)]
    return [op, rng.choice(USERS)]


rng = random.Random(2026)
seen = {}                                                    # (operation, kind of answer) -> how often
skipping_restores = 0
for _ in range(2000):
    queries = [random_query(rng) for _ in range(rng.randint(1, 150))]
    slow = SlowStorage()
    want = []
    for query in queries:
        op = query[0]
        saved = [len(s[1]) for s in slow.snaps if s[0] == query[1]] if op == "RESTORE_USER" else []
        answer = slow.run(query)
        want.append(answer)
        kind = "empty" if answer == "" else "false" if answer == "false" else "zero" if answer == "0" else "value"
        seen[op, kind] = seen.get((op, kind), 0) + 1
        if saved and int(answer) < saved[0]:
            skipping_restores += 1                           # a backed-up name was taken by someone else
    got = solution(queries)
    assert got == want, (queries, got, want)

for case in [("ADD_FILE", "value"), ("ADD_FILE", "false"), ("GET_FILE_SIZE", "value"), ("GET_FILE_SIZE", "empty"),
             ("DELETE_FILE", "value"), ("DELETE_FILE", "empty"), ("GET_N_LARGEST", "value"), ("GET_N_LARGEST", "empty"),
             ("ADD_USER", "value"), ("ADD_USER", "false"), ("ADD_FILE_BY", "value"), ("ADD_FILE_BY", "empty"),
             ("ADD_FILE_BY", "zero"), ("MERGE_USER", "value"), ("MERGE_USER", "empty"), ("BACKUP_USER", "value"),
             ("BACKUP_USER", "zero"), ("BACKUP_USER", "empty"), ("RESTORE_USER", "value"), ("RESTORE_USER", "zero"),
             ("RESTORE_USER", "empty")]:
    assert seen.get(case, 0) >= 20, (case, seen.get(case, 0))    # every branch was really exercised
assert skipping_restores >= 50, skipping_restores

# --- 4. the levels are layered: an earlier level's class still passes its own table ---
def run_on(cls):
    def run_table(queries):
        storage, out = cls(), []
        for op, *args in queries:
            method = getattr(storage, op.lower())
            out.append(as_text(method(*[int(a) if a.isdigit() else a for a in args])))
        return out
    return run_table


check_table(LEVEL1, run_on(StorageL1))
check_table(LEVEL2, run_on(StorageL2))
check_table(LEVEL3, run_on(StorageL3))
print("random sequences agree with the naive model;", skipping_restores, "restores had to skip a taken name")
print("all checks passed")
```

</details>

</details>
