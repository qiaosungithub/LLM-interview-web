"""A literal model of the statement, used for the expected values of the random tests, and the random
operation-sequence generator.

The tree is a flat dict from a path's components (a tuple) to DIR or to a (content, owner) pair. A user's
used total is recomputed from the files whenever it is needed, and every changing call builds the new state
first and keeps it only if the statement's conditions hold, so a failed call cannot leave anything behind."""

DIR = "dir"


def norm(path):
    """The components of a valid path as a tuple (() for the root), or None for an invalid path."""
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    comps = tuple(c for c in path.split("/") if c)
    if any(c in (".", "..") for c in comps):
        return None
    return comps


def same_chain(a, b):
    """True if one component tuple is a prefix of the other (equal counts)."""
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def is_file(v):
    return isinstance(v, tuple)


class FileSystem:
    def __init__(self):
        self.nodes = {(): DIR}
        self.caps = {}              # registered user id -> capacity
        self.snaps = {}             # snapshot id -> (nodes, caps); both hold immutable values only

    def blocked(self, p):
        """Some proper ancestor of p exists as a file."""
        return any(is_file(self.nodes.get(p[:i])) for i in range(1, len(p)))

    def subtree(self, p):
        return {q: v for q, v in self.nodes.items() if q[:len(p)] == p}

    @staticmethod
    def used(nodes, uid):
        return sum(len(v[0]) for v in nodes.values() if is_file(v) and v[1] == uid)

    @staticmethod
    def add_parents(nodes, p):
        for i in range(1, len(p)):
            nodes.setdefault(p[:i], DIR)

    # Level 1
    def mkdir(self, path):
        p = norm(path)
        if p is None or is_file(self.nodes.get(p)) or self.blocked(p):
            return False
        self.add_parents(self.nodes, p)
        self.nodes.setdefault(p, DIR)
        return True

    def write(self, path, content, owner="admin"):
        p = norm(path)
        if owner != "admin" and owner not in self.caps:
            return False
        if not p or self.blocked(p):
            return False
        old = self.nodes.get(p)
        if old == DIR or (is_file(old) and old[1] != owner):
            return False
        new = dict(self.nodes)
        self.add_parents(new, p)
        new[p] = (content, owner)
        if owner != "admin" and self.used(new, owner) > self.caps[owner]:
            return False
        self.nodes = new
        return True

    def read(self, path):
        p = norm(path)
        v = None if p is None else self.nodes.get(p)
        return v[0] if is_file(v) else None

    def ls(self, path):
        p = norm(path)
        if p is None or self.nodes.get(p) != DIR:
            return None
        return sorted(q[-1] for q in self.nodes if len(q) == len(p) + 1 and q[:len(p)] == p)

    # Level 2
    def size(self, path):
        p = norm(path)
        if p is None or p not in self.nodes:
            return None
        return sum(len(v[0]) for v in self.subtree(p).values() if is_file(v))

    def rm(self, path):
        p = norm(path)
        if not p or p not in self.nodes:
            return False
        for q in self.subtree(p):
            del self.nodes[q]
        return True

    def _transfer(self, src, dst, keep):
        s, d = norm(src), norm(dst)
        if s is None or d is None or s not in self.nodes or same_chain(s, d) or self.blocked(d):
            return False
        here, there = self.nodes[s], self.nodes.get(d)
        if there is not None:
            if is_file(here) != is_file(there):
                return False
            if is_file(here) and here[1] != there[1]:
                return False
        moving = self.subtree(s)
        new = {q: v for q, v in self.nodes.items() if q[:len(d)] != d and (keep or q[:len(s)] != s)}
        self.add_parents(new, d)
        for q, v in moving.items():
            new[d + q[len(s):]] = v
        if keep:
            for uid in {v[1] for v in moving.values() if is_file(v)} - {"admin"}:
                if self.used(new, uid) > self.caps[uid]:
                    return False
        self.nodes = new
        return True

    def mv(self, src, dst):
        return self._transfer(src, dst, keep=False)

    def cp(self, src, dst):
        return self._transfer(src, dst, keep=True)

    # Level 3
    def add_user(self, user_id, capacity):
        if user_id == "admin" or user_id in self.caps or capacity < 0:
            return False
        self.caps[user_id] = capacity
        return True

    def update_capacity(self, user_id, capacity):
        if user_id not in self.caps or capacity < 0:
            return None
        self.caps[user_id] = capacity
        mine = sorted((p for p, v in self.nodes.items() if is_file(v) and v[1] == user_id),
                      key=lambda p: (-len(self.nodes[p][0]), "/" + "/".join(p)))
        deleted = 0
        for p in mine:
            if self.used(self.nodes, user_id) <= capacity:
                break
            del self.nodes[p]
            deleted += 1
        return deleted

    # Level 4
    def snapshot(self, snapshot_id):
        if snapshot_id in self.snaps:
            return False
        self.snaps[snapshot_id] = (dict(self.nodes), dict(self.caps))
        return True

    def restore(self, snapshot_id):
        if snapshot_id not in self.snaps:
            return False
        nodes, caps = self.snaps[snapshot_id]
        self.nodes, self.caps = dict(nodes), dict(caps)
        return True

    # not part of the API
    def ambiguous(self, method, args):
        """True for an mv/cp of a directory onto an existing directory where some file inside the old
        destination sits at the same relative path as an incoming file with a different owner. The statement
        does not say whether that owner check applies inside a replaced directory, so no test does this."""
        if method not in ("mv", "cp") or len(args) != 2:
            return False
        s, d = norm(args[0]), norm(args[1])
        if s is None or d is None or same_chain(s, d) or self.nodes.get(s) != DIR or self.nodes.get(d) != DIR:
            return False
        for q, v in self.subtree(s).items():
            w = self.nodes.get(d + q[len(s):])
            if is_file(v) and is_file(w) and v[1] != w[1]:
                return True
        return False


# ---------------------------------------------------------------- random operation sequences

KINDS = {   # operation -> weight; each level also uses every earlier level's operations
    1: {"mkdir": 3, "write": 5, "read": 3, "ls": 3},
    2: {"mkdir": 2, "write": 4, "read": 2, "ls": 2, "mv": 4, "cp": 4, "rm": 3, "size": 3},
    3: {"mkdir": 1, "write": 6, "read": 1, "ls": 1, "mv": 3, "cp": 4, "rm": 2, "size": 2, "add_user": 2,
        "update_capacity": 3},
    4: {"mkdir": 1, "write": 5, "read": 2, "ls": 1, "mv": 3, "cp": 3, "rm": 2, "size": 2, "add_user": 2,
        "update_capacity": 2, "snapshot": 3, "restore": 3},
}

COMPS = ["a", "b", "c"]
RARE = ["a.b", "B", "é", "..."]
INVALID = ["", "a", "a/b", "/.", "/..", "/a/./b", "/a/..", "/../a", "./a", "/b/../c"]
CONTENTS = ["", "x", "hi", "abc", "hello", "0123456789", "日本語"]
USERS = ("ann", "bob", "cy")
SNAPS = ("s1", "s2", "s3")


def random_ops(r, level, n_ops):
    """A sequence of (method, args, kwargs) calls for one FileSystem, drawn while a model follows along so
    that the draws hit the interesting cases often: existing and new paths, paths under a file, destinations
    of the same and of the other kind, ancestors and descendants of the source, unnormalized and invalid
    spellings, contents that exactly fill or overflow a quota, overwrites by the owner and by someone else."""
    m = FileSystem()
    kinds, weights = zip(*KINDS[level].items())
    ops = []

    def live(kind=None):
        return sorted(p for p, v in m.nodes.items() if p and (kind is None or is_file(v) == (kind == "file")))

    def comp():
        return r.choice(RARE) if r.random() < 0.07 else r.choice(COMPS)

    def fresh(avoid=None):
        for _ in range(4):
            dirs = [p for p in live("dir") if len(p) < 3 and (avoid is None or not same_chain(p, avoid))]
            base = r.choice(dirs) if dirs and r.random() < 0.6 else ()
            p = base + tuple(comp() for _ in range(r.randint(1, 3 - len(base))))
            if avoid is None or not same_chain(p, avoid):
                break
        return p

    def pick(bias, kind=None):
        if r.random() < 0.03:
            return ()
        pool = live(kind) or live()
        if pool and r.random() < bias:
            return r.choice(pool)
        x = r.random()
        files = live("file")
        if files and x < 0.08:
            return r.choice(files) + (comp(),)              # under a file
        return fresh()

    def spell(p):
        x = r.random()
        if x < 0.04:
            return r.choice(INVALID)
        s = "/" + "/".join(p)
        if p and x < 0.10:
            return s + "/"
        if p and x < 0.15:
            i = r.randint(0, len(p) - 1)
            return "/" + "/".join(p[:i]) + ("/" if i else "") + "/" + "/".join(p[i:])
        return s

    def owner_for(p):
        v = m.nodes.get(p)
        if is_file(v) and r.random() < 0.7:
            return v[1]
        if r.random() < 0.06:
            return "zed"
        return r.choice(["admin"] + sorted(m.caps)) if m.caps else "admin"

    def content_for(p, owner):
        if owner in m.caps and r.random() < 0.45:
            v = m.nodes.get(p)
            old = len(v[0]) if is_file(v) and v[1] == owner else 0
            room = m.caps[owner] - m.used(m.nodes, owner) + old
            k = r.choice([room, room, room + 1, r.randint(0, max(room, 0))])
            return "x" * max(0, min(k, 40))
        return r.choice(CONTENTS)

    def dst_for(s):
        x = r.random()
        same = [p for p in live("file" if is_file(m.nodes.get(s)) else "dir") if not same_chain(p, s)]
        other = [p for p in live("dir" if is_file(m.nodes.get(s)) else "file") if not same_chain(p, s)]
        if x < 0.35 or s not in m.nodes:
            return fresh(avoid=s)
        if x < 0.62 and same:
            return r.choice(same)
        if x < 0.72 and other:
            return r.choice(other)
        if x < 0.82 and s:
            return s[:r.randint(0, len(s))] if r.random() < 0.5 else s + (comp(),)
        if x < 0.90:
            files = live("file")
            if files:
                return r.choice(files) + (comp(),)
        return pick(0.5)

    while len(ops) < n_ops:
        if level >= 3 and not m.caps and r.random() < 0.5:
            kind = "add_user"
        elif len(m.nodes) < 4 and r.random() < 0.5:
            kind = "write"
        else:
            kind = r.choices(kinds, weights)[0]
        kwargs = {}
        if kind == "mkdir":
            args = (spell(pick(0.3, "dir")),)
        elif kind == "write":
            p = pick(0.4, "file")
            if level >= 3:
                owner = owner_for(p)
                if owner != "admin" or r.random() < 0.3:
                    kwargs = {"owner": owner}
                args = (spell(p), content_for(p, owner))
            else:
                args = (spell(p), r.choice(CONTENTS))
        elif kind in ("read", "size"):
            args = (spell(pick(0.75)),)
        elif kind == "ls":
            args = (spell(() if r.random() < 0.25 else pick(0.6, "dir")),)
        elif kind == "rm":
            args = (spell(pick(0.8)),)
        elif kind in ("mv", "cp"):
            s = pick(0.9)
            args = (spell(s), spell(dst_for(s)))
            if m.ambiguous(kind, args):
                continue
        elif kind == "add_user":
            free = [u for u in USERS if u not in m.caps]
            uid = r.choice(free) if free and r.random() < 0.8 else r.choice(USERS + ("admin",))
            args = (uid, r.choice([0, 5, 10, 10, 20, 30, -1] if r.random() < 0.1 else [5, 10, 10, 20, 30]))
        elif kind == "update_capacity":
            uid = r.choice(sorted(m.caps)) if m.caps and r.random() < 0.85 else r.choice(USERS + ("admin",))
            used = m.used(m.nodes, uid)
            args = (uid, r.choice([0, used // 2, used - 1, used - 1, used, used + 5, r.randint(0, 30), -1]))
            if args[1] < 0 and r.random() < 0.8:
                args = (uid, 0)
        elif kind == "snapshot":
            free = [s for s in SNAPS if s not in m.snaps]
            args = (r.choice(free) if free and r.random() < 0.75 else r.choice(SNAPS),)
        else:   # restore
            taken = sorted(m.snaps)
            args = (r.choice(taken) if taken and r.random() < 0.85 else r.choice(SNAPS + ("nope",)),)
        ops.append((kind, args, kwargs))
        getattr(m, kind)(*args, **kwargs)
    return ops
