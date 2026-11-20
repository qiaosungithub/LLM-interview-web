"""A literal model of the statement, used for the expected answers of the random tests, and the random query
generator. Files are one flat dict name -> (size, owner); a user's used space is recomputed by summing every time,
and nothing is indexed."""

ADMIN = "admin"


class Storage:
    def __init__(self):
        self.files = {}         # name -> (size, owner); owner is ADMIN for ADD_FILE
        self.limits = {}        # user -> capacity
        self.backups = {}       # user -> [(name, size), ...], the latest backup only

    def left(self, user):
        return self.limits[user] - sum(size for size, owner in self.files.values() if owner == user)

    def run(self, query):
        op, args = query[0], query[1:]
        return getattr(self, op.lower())(*args)

    # Level 1
    def add_file(self, name, size):
        if name in self.files:
            return "false"
        self.files[name] = (int(size), ADMIN)
        return "true"

    def get_file_size(self, name):
        return str(self.files[name][0]) if name in self.files else ""

    def delete_file(self, name):
        if name not in self.files:
            return ""
        return str(self.files.pop(name)[0])

    # Level 2
    def get_n_largest(self, prefix, n):
        found = [(name, size) for name, (size, _) in self.files.items() if name[:len(prefix)] == prefix]
        found.sort(key=lambda item: item[0])
        found.sort(key=lambda item: item[1], reverse=True)          # stable: equal sizes keep names ascending
        return ", ".join(f"{name}({size})" for name, size in found[:int(n)])

    # Level 3
    def add_user(self, user, capacity):
        if user in self.limits:
            return "false"
        self.limits[user] = int(capacity)
        return "true"

    def add_file_by(self, user, name, size):
        if user not in self.limits or name in self.files or int(size) > self.left(user):
            return ""
        self.files[name] = (int(size), user)
        return str(self.left(user))

    def merge_user(self, user1, user2):
        if user1 == user2 or user1 not in self.limits or user2 not in self.limits:
            return ""
        for name, (size, owner) in list(self.files.items()):
            if owner == user2:
                self.files[name] = (size, user1)
        self.limits[user1] += self.limits.pop(user2)
        self.backups.pop(user2, None)
        return str(self.left(user1))

    # Level 4
    def backup_user(self, user):
        if user not in self.limits:
            return ""
        self.backups[user] = [(name, size) for name, (size, owner) in self.files.items() if owner == user]
        return str(len(self.backups[user]))

    def restore_user(self, user):
        if user not in self.limits:
            return ""
        self.files = {name: row for name, row in self.files.items() if row[1] != user}
        count = 0
        for name, size in self.backups.get(user, []):
            if name not in self.files:
                self.files[name] = (size, user)
                count += 1
        assert self.left(user) >= 0
        return str(count)


def solution(queries):
    storage = Storage()
    return [storage.run(q) for q in queries]


# ---------------------------------------------------------------- random query sequences

NAMES = ["/a", "/a/b", "/a/c", "/ab", "/a.txt", "/b", "/b/a", "/B", "/c1", "/c10"]
PREFIXES = ["/", "/a", "/a/", "/ab", "/b", "/B", "/c1", "/c", "/zz", "/a/b"]
USERS = ["u1", "u2", "u3", "U1"]
SIZES = [1, 2, 3, 5, 9, 10, 10, 20, 100]

KINDS = {   # operation -> weight; each level also uses every earlier level's operations
    1: {"ADD_FILE": 4, "GET_FILE_SIZE": 2, "DELETE_FILE": 3},
    2: {"ADD_FILE": 4, "GET_FILE_SIZE": 1, "DELETE_FILE": 2, "GET_N_LARGEST": 4},
    3: {"ADD_FILE": 2, "GET_FILE_SIZE": 1, "DELETE_FILE": 3, "GET_N_LARGEST": 2, "ADD_USER": 2,
        "ADD_FILE_BY": 7, "MERGE_USER": 2},
    4: {"ADD_FILE": 2, "GET_FILE_SIZE": 1, "DELETE_FILE": 3, "GET_N_LARGEST": 1, "ADD_USER": 2,
        "ADD_FILE_BY": 6, "MERGE_USER": 2, "BACKUP_USER": 3, "RESTORE_USER": 4},
}


def random_queries(r, level, n_queries):
    """A list of queries for one storage, drawn while a model follows along so that the draws hit the interesting
    cases often: sizes equal to a user's remaining capacity and one more, names already taken by admin or by
    another user, equal sizes for the tie rule, merges of users that own files or backups, merged ids created
    again, and restores whose backed-up names were deleted or taken since."""
    m = Storage()
    kinds, weights = zip(*KINDS[level].items())
    out = []

    def taken_or_free(p_taken):
        if m.files and r.random() < p_taken:
            return r.choice(sorted(m.files))
        free = [name for name in NAMES if name not in m.files]
        return r.choice(free or NAMES)

    def user(p_known=0.85):
        if m.limits and r.random() < p_known:
            return r.choice(sorted(m.limits))
        return r.choice(USERS)

    for _ in range(n_queries):
        kind = r.choices(kinds, weights)[0]
        if kind in ("ADD_FILE_BY", "MERGE_USER", "BACKUP_USER", "RESTORE_USER") and not m.limits \
                and r.random() < 0.8:
            kind = "ADD_USER"
        if kind == "ADD_FILE":
            q = [kind, taken_or_free(0.25), str(r.choice(SIZES))]
        elif kind in ("GET_FILE_SIZE", "DELETE_FILE"):
            q = [kind, taken_or_free(0.75)]
        elif kind == "GET_N_LARGEST":
            q = [kind, r.choice(PREFIXES), str(r.choice([1, 1, 2, 2, 3, 4, 6, 1000000000]))]
        elif kind == "ADD_USER":
            q = [kind, user(0.2), str(r.choice([5, 10, 20, 30, 50, 100]))]
        elif kind == "ADD_FILE_BY":
            u = user()
            x = r.random()
            if u in m.limits and x < 0.3:
                size = max(m.left(u), 1)
            elif u in m.limits and x < 0.4:
                size = m.left(u) + 1
            else:
                size = r.choice(SIZES)
            q = [kind, u, taken_or_free(0.3), str(size)]
        elif kind == "MERGE_USER":
            owners = sorted({owner for _, owner in m.files.values() if owner != ADMIN} | set(m.backups))
            u2 = r.choice(owners) if owners and r.random() < 0.6 else user()
            u1 = u2 if r.random() < 0.08 else user()
            q = [kind, u1, u2]
        elif kind == "BACKUP_USER":
            q = [kind, user()]
        else:   # RESTORE_USER
            q = [kind, r.choice(sorted(m.backups)) if m.backups and r.random() < 0.7 else user()]
        out.append(q)
        m.run(q)
    return out
