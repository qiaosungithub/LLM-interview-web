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


class StorageL2(StorageL1):
    def get_n_largest(self, prefix, n):
        matches = [(name, size) for name, size in self.size.items() if name.startswith(prefix)]
        matches.sort(key=lambda item: (-item[1], item[0]))       # size descending, then name ascending
        return matches[:n]


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
