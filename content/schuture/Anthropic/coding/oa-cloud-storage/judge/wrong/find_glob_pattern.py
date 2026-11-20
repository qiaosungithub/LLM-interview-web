# expect: WA
# find_files matches the glob prefix + '*' + suffix, so a short name cannot satisfy two overlapping sides.
class CloudStorage:
    def __init__(self):
        self._files: dict[str, int] = {}

    def add_file(self, name: str, size: int) -> bool:
        if name in self._files:
            return False
        self._files[name] = size
        return True

    def copy_file(self, source: str, destination: str) -> bool:
        if source not in self._files or destination in self._files:
            return False
        self._files[destination] = self._files[source]
        return True

    def get_file_size(self, name: str) -> int | None:
        return self._files.get(name)


def find_files(self, prefix: str, suffix: str) -> list[str]:
    import fnmatch
    matches = [(name, size) for name, size in self._files.items()
               if fnmatch.fnmatchcase(name, prefix + "*" + suffix)]
    matches.sort(key=lambda item: (-item[1], item[0]))    # NOTE: size descending, then name ascending
    return [f"{name}({size})" for name, size in matches]


CloudStorage.find_files = find_files


_init_l2 = CloudStorage.__init__


def __init__(self):
    _init_l2(self)
    self._owner: dict[str, str] = {}      # file name -> owning user id (absent: unowned)
    self._capacity: dict[str, int] = {}   # user id -> total capacity
    self._used: dict[str, int] = {}       # user id -> bytes currently charged to it


def _remaining(self, user_id: str) -> int:
    return self._capacity[user_id] - self._used[user_id]


def add_user(self, user_id: str, capacity: int) -> bool:
    if user_id in self._capacity:
        return False
    self._capacity[user_id] = capacity
    self._used[user_id] = 0
    return True


def add_file_by(self, user_id: str, name: str, size: int) -> int | None:
    if user_id not in self._capacity or name in self._files:
        return None
    if size > self._remaining(user_id):        # NOTE: capacity checked before any mutation
        return None
    self._files[name] = size
    self._owner[name] = user_id
    self._used[user_id] += size
    return self._remaining(user_id)


def copy_file(self, source: str, destination: str) -> bool:
    if source not in self._files or destination in self._files:
        return False
    size = self._files[source]
    owner = self._owner.get(source)
    if owner is not None and size > self._remaining(owner):    # NOTE: a copy counts against SOURCE's owner
        return False
    self._files[destination] = size
    if owner is not None:
        self._owner[destination] = owner
        self._used[owner] += size
    return True


def merge_user(self, target: str, source: str) -> int | None:
    if target == source or target not in self._capacity or source not in self._capacity:
        return None                              # NOTE: forbids a self-merge that would double one's own capacity
    for name, owner in self._owner.items():
        if owner == source:
            self._owner[name] = target
    self._capacity[target] += self._capacity[source]
    self._used[target] += self._used[source]
    del self._capacity[source]
    del self._used[source]
    return self._remaining(target)


CloudStorage.__init__ = __init__
CloudStorage._remaining = _remaining
CloudStorage.add_user = add_user
CloudStorage.add_file_by = add_file_by
CloudStorage.copy_file = copy_file
CloudStorage.merge_user = merge_user


COMPRESSED_SUFFIX = ".cmp"

_init_l3 = CloudStorage.__init__


def __init__(self):
    _init_l3(self)
    self._backups: dict[str, tuple[int, dict[str, int]]] = {}    # user id -> (capacity, {name: size})


_merge_l3 = CloudStorage.merge_user


def merge_user(self, target: str, source: str) -> int | None:
    remaining = _merge_l3(self, target, source)
    if remaining is not None:
        self._backups.pop(source, None)          # NOTE: source's account is gone, so its backup goes too --
    return remaining                             # a later add_user(source) must start with no backup


def compress_file(self, user_id: str, name: str) -> str | None:
    if user_id not in self._capacity or self._owner.get(name) != user_id:   # NOTE: also covers "no such file"
        return None
    new_name = name + COMPRESSED_SUFFIX
    if new_name in self._files:
        return None                              # NOTE: compressed-name collision
    size = self._files.pop(name)
    del self._owner[name]
    self._used[user_id] -= size
    new_size = -(-size // 2)                     # NOTE: ceiling division -- never rounds size 1 down to 0
    self._files[new_name] = new_size
    self._owner[new_name] = user_id
    self._used[user_id] += new_size              # NOTE: never exceeds capacity: new_size <= size always
    return new_name


def decompress_file(self, user_id: str, name: str) -> str | None:
    if user_id not in self._capacity or self._owner.get(name) != user_id:
        return None
    if not name.endswith(COMPRESSED_SUFFIX):
        return None
    new_name = name[:-len(COMPRESSED_SUFFIX)]
    if new_name in self._files:
        return None
    size = self._files[name]
    if size > self._remaining(user_id):          # NOTE: name itself still counts here -- see the level text
        return None
    del self._files[name]
    del self._owner[name]
    self._used[user_id] -= size
    new_size = size * 2
    self._files[new_name] = new_size
    self._owner[new_name] = user_id
    self._used[user_id] += new_size
    return new_name


def backup_user(self, user_id: str) -> bool:
    if user_id not in self._capacity:
        return False
    owned = {name: size for name, size in self._files.items() if self._owner.get(name) == user_id}
    self._backups[user_id] = (self._capacity[user_id], owned)     # NOTE: replaces any earlier backup
    return True


def restore_user(self, user_id: str) -> int | None:
    if user_id not in self._capacity or user_id not in self._backups:
        return None
    snap_capacity, snap_files = self._backups[user_id]
    current = {name for name, owner in self._owner.items() if owner == user_id}
    to_delete = current - snap_files.keys()       # owned now, not in the snapshot
    to_create = snap_files.keys() - current        # in the snapshot, not owned now
    if any(name in self._files for name in to_create):    # NOTE: every collision checked before any mutation
        return None
    for name in to_delete:
        del self._files[name]
        del self._owner[name]
    for name in to_create:
        self._files[name] = snap_files[name]
        self._owner[name] = user_id
    self._capacity[user_id] = snap_capacity
    # NOTE: recomputed from what user_id actually owns now, NOT sum(snap_files.values()) -- a name kept
    # untouched because it was owned at backup time and still is now is not re-verified against its
    # backed-up size, so the snapshot's own total can be stale (see the level text).
    self._used[user_id] = sum(size for name, size in self._files.items() if self._owner.get(name) == user_id)
    return self._remaining(user_id)


CloudStorage.__init__ = __init__
CloudStorage.merge_user = merge_user
CloudStorage.compress_file = compress_file
CloudStorage.decompress_file = decompress_file
CloudStorage.backup_user = backup_user
CloudStorage.restore_user = restore_user
