"""A literal model of the statement, used for the expected values of the random tests, and the random
operation-sequence generator. Every query re-scans what it needs; nothing is indexed or cached.

An account is an object, so a file's owner and a backup belong to one account, not to a user id: a merge
removes the source object (its backup goes with it), and a later add_user of that id makes a new object."""

COMPRESSED_SUFFIX = ".cmp"


class _User:
    def __init__(self, capacity):
        self.capacity = capacity
        self.backup = None          # (capacity, {name: size}) from the last backup_user, or None


class CloudStorage:
    def __init__(self):
        self.files = {}             # name -> [size, owner _User or None]
        self.users = {}             # user id -> its live _User

    def used(self, u):
        return sum(size for size, owner in self.files.values() if owner is u)

    def remaining(self, uid):
        u = self.users[uid]
        return u.capacity - self.used(u)

    def owned(self, uid):
        u = self.users[uid]
        return {name: size for name, (size, owner) in self.files.items() if owner is u}

    # Level 1
    def add_file(self, name, size):
        if name in self.files:
            return False
        self.files[name] = [size, None]
        return True

    def copy_file(self, source, destination):
        if source not in self.files or destination in self.files:
            return False
        size, owner = self.files[source]
        if owner is not None and size > owner.capacity - self.used(owner):
            return False
        self.files[destination] = [size, owner]
        return True

    def get_file_size(self, name):
        rec = self.files.get(name)
        return None if rec is None else rec[0]

    # Level 2
    def find_files(self, prefix, suffix):
        hits = [(name, rec[0]) for name, rec in self.files.items()
                if name.startswith(prefix) and name.endswith(suffix)]
        hits.sort(key=lambda h: (-h[1], h[0]))
        return [f"{name}({size})" for name, size in hits]

    # Level 3
    def add_user(self, uid, capacity):
        if uid in self.users:
            return False
        self.users[uid] = _User(capacity)
        return True

    def add_file_by(self, uid, name, size):
        if uid not in self.users or name in self.files or size > self.remaining(uid):
            return None
        self.files[name] = [size, self.users[uid]]
        return self.remaining(uid)

    def merge_user(self, target, source):
        if target == source or target not in self.users or source not in self.users:
            return None
        t, s = self.users[target], self.users[source]
        for rec in self.files.values():
            if rec[1] is s:
                rec[1] = t
        t.capacity += s.capacity
        del self.users[source]
        return self.remaining(target)

    # Level 4
    def compress_file(self, uid, name):
        u, rec = self.users.get(uid), self.files.get(name)
        if u is None or rec is None or rec[1] is not u:
            return None
        new = name + COMPRESSED_SUFFIX
        if new in self.files:
            return None
        del self.files[name]
        self.files[new] = [(rec[0] + 1) // 2, u]
        return new

    def decompress_file(self, uid, name):
        u, rec = self.users.get(uid), self.files.get(name)
        if u is None or rec is None or rec[1] is not u or not name.endswith(COMPRESSED_SUFFIX):
            return None
        new = name[: -len(COMPRESSED_SUFFIX)]
        if new in self.files or rec[0] > self.remaining(uid):
            return None
        del self.files[name]
        self.files[new] = [rec[0] * 2, u]
        return new

    def backup_user(self, uid):
        if uid not in self.users:
            return False
        u = self.users[uid]
        u.backup = (u.capacity, self.owned(uid))
        return True

    def restore_user(self, uid):
        u = self.users.get(uid)
        if u is None or u.backup is None:
            return None
        capacity, snap = u.backup
        now = self.owned(uid)
        create = [name for name in snap if name not in now]
        if any(name in self.files for name in create):
            return None
        for name in now:
            if name not in snap:
                del self.files[name]
        for name in create:
            self.files[name] = [snap[name], u]
        u.capacity = capacity
        return self.remaining(uid)


# ---------------------------------------------------------------- random operation sequences

KINDS = {   # operation -> weight; each level also uses every earlier level's operations
    1: {"add_file": 4, "copy_file": 4, "get_file_size": 3},
    2: {"add_file": 4, "copy_file": 3, "get_file_size": 1, "find_files": 4},
    3: {"add_file": 2, "copy_file": 4, "get_file_size": 1, "find_files": 1, "add_user": 2, "add_file_by": 5,
        "merge_user": 2},
    4: {"add_file": 2, "copy_file": 3, "get_file_size": 2, "find_files": 1, "add_user": 2, "add_file_by": 4,
        "merge_user": 2, "compress_file": 4, "decompress_file": 4, "backup_user": 3, "restore_user": 4},
}

BASES = ["a", "b", "c", "ab", "ba", "x.txt", "a.txt", "B"]
SIZES = [1, 1, 2, 3, 4, 5, 7, 8, 10, 13, 20]


def random_ops(r, level, n_ops, users=("u", "v", "w")):
    """A sequence of (method, args) calls for one CloudStorage, drawn while a model follows along so that the
    draws hit the interesting cases often: free and taken names, sizes equal to the remaining capacity, names
    with one or two COMPRESSED_SUFFIX, owned files of the right and the wrong user, the id a merge just freed."""
    m = CloudStorage()
    kinds, weights = zip(*KINDS[level].items())
    pool = list(BASES)
    if level == 4:
        pool += [b + COMPRESSED_SUFFIX for b in BASES[:3]] + ["a" + COMPRESSED_SUFFIX * 2]
    gone = None                 # the id the last merge removed
    ops = []

    def name(taken=None):
        """A file name; taken=True prefers an existing one, False a free one."""
        live = sorted(m.files)
        if taken is True and live and r.random() < 0.85:
            return r.choice(live)
        if taken is False and r.random() < 0.85:
            free = [n for n in pool if n not in m.files]
            if free:
                return r.choice(free)
        return r.choice(pool + live)

    def user(live_bias=0.85):
        if m.users and r.random() < live_bias:
            return r.choice(sorted(m.users))
        return r.choice(users)

    def owned_by(uid):
        return sorted(m.owned(uid)) if uid in m.users else []

    def size_for(uid):
        if uid in m.users and r.random() < 0.5:
            rem = m.remaining(uid)
            if rem > 0:
                return r.choice([rem, rem, rem + 1, r.randint(1, rem)])
            return r.choice(SIZES)
        return r.choice(SIZES)

    for _ in range(n_ops):
        if level >= 3 and not m.users and r.random() < 0.7:
            kind = "add_user"
        else:
            kind = r.choices(kinds, weights)[0]
        if kind == "add_file":
            args = (name(taken=r.random() < 0.15), r.choice(SIZES))
        elif kind == "copy_file":
            src = name(taken=True)
            dst = src if r.random() < 0.05 else name(taken=r.random() < 0.2)
            args = (src, dst)
        elif kind == "get_file_size":
            args = (name(taken=r.random() < 0.8),)
        elif kind == "find_files":
            sample = name(taken=True)
            cut = r.randint(0, len(sample))
            side = r.random()
            prefix = sample[:cut] if side < 0.6 else r.choice(["", "", "a", "b", "x"])
            suffix = sample[cut:] if side >= 0.3 else r.choice(["", "", "b", COMPRESSED_SUFFIX, ".txt"])
            if r.random() < 0.15:                       # overlapping sides on one short name
                prefix, suffix = sample, sample[r.randint(0, len(sample)):]
            args = (prefix, suffix)
        elif kind == "add_user":
            uid = gone if gone is not None and gone not in m.users and r.random() < 0.6 else user(0.2)
            args = (uid, r.choice([5, 10, 10, 20, 30, 50]))
        elif kind == "add_file_by":
            uid = user()
            args = (uid, name(taken=r.random() < 0.15), size_for(uid))
        elif kind == "merge_user":
            target = user(0.95)
            others = [u for u in sorted(m.users) if u != target]
            source = target if r.random() < 0.08 else r.choice(others) if others and r.random() < 0.7 else user(0.95)
            args = (target, source)
            if target != source and target in m.users and source in m.users:
                gone = source
        elif kind in ("compress_file", "decompress_file"):
            uid = user()
            mine = owned_by(uid)
            if kind == "decompress_file":
                cmp = [n for n in mine if n.endswith(COMPRESSED_SUFFIX)]
                mine = cmp if cmp and r.random() < 0.85 else mine
            args = (uid, r.choice(mine) if mine and r.random() < 0.8 else name(taken=True))
        elif kind == "backup_user":
            args = (user(),)
        else:   # restore_user
            args = (gone if gone is not None and r.random() < 0.2 else user(),)
        ops.append((kind, args))
        getattr(m, kind)(*args)
    return ops
