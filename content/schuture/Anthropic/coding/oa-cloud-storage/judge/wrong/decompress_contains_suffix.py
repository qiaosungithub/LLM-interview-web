# expect: WA
# decompress_file accepts any name that contains '.cmp' and removes its last occurrence, not only a trailing one.
from dataclasses import dataclass, field

COMPRESSED_SUFFIX = ".cmp"


@dataclass
class Account:
    capacity: int
    names: set = field(default_factory=set)
    used: int = 0
    snapshot: tuple | None = None       # (capacity, {name: size})

    @property
    def free(self):
        return self.capacity - self.used


class CloudStorage:
    def __init__(self):
        self.size = {}          # name -> size
        self.owner = {}         # name -> Account (absent: unowned)
        self.accounts = {}      # uid -> Account

    # ---- internal helpers
    def _put(self, name, size, acct=None):
        self.size[name] = size
        if acct is not None:
            self.owner[name] = acct
            acct.names.add(name)
            acct.used += size

    def _drop(self, name):
        size = self.size.pop(name)
        acct = self.owner.pop(name, None)
        if acct is not None:
            acct.names.discard(name)
            acct.used -= size
        return size

    # ---- level 1
    def add_file(self, name, size):
        if name in self.size:
            return False
        self._put(name, size)
        return True

    def copy_file(self, source, destination):
        if source not in self.size or destination in self.size:
            return False
        sz = self.size[source]
        acct = self.owner.get(source)
        if acct is not None and acct.free < sz:
            return False
        self._put(destination, sz, acct)
        return True

    def get_file_size(self, name):
        return self.size.get(name)

    # ---- level 2
    def find_files(self, prefix, suffix):
        hits = sorted((n for n in self.size if n.startswith(prefix) and n.endswith(suffix)),
                      key=lambda n: (-self.size[n], n))
        return ["%s(%d)" % (n, self.size[n]) for n in hits]

    # ---- level 3
    def add_user(self, user_id, capacity):
        if user_id in self.accounts:
            return False
        self.accounts[user_id] = Account(capacity)
        return True

    def add_file_by(self, user_id, name, size):
        acct = self.accounts.get(user_id)
        if acct is None or name in self.size or size > acct.free:
            return None
        self._put(name, size, acct)
        return acct.free

    def merge_user(self, target, source):
        if target == source:
            return None
        t, s = self.accounts.get(target), self.accounts.get(source)
        if t is None or s is None:
            return None
        for n in list(s.names):
            self.owner[n] = t
            t.names.add(n)
        t.used += s.used
        t.capacity += s.capacity
        del self.accounts[source]
        return t.free

    # ---- level 4
    def _mine(self, user_id, name):
        acct = self.accounts.get(user_id)
        if acct is None or self.owner.get(name) is not acct:
            return None
        return acct

    def compress_file(self, user_id, name):
        acct = self._mine(user_id, name)
        new = name + COMPRESSED_SUFFIX
        if acct is None or new in self.size:
            return None
        sz = self._drop(name)
        self._put(new, (sz + 1) // 2, acct)
        return new

    def decompress_file(self, user_id, name):
        acct = self._mine(user_id, name)
        if acct is None or COMPRESSED_SUFFIX not in name:
            return None
        new = "".join(name.rsplit(COMPRESSED_SUFFIX, 1))
        if new in self.size or self.size[name] > acct.free:
            return None
        sz = self._drop(name)
        self._put(new, 2 * sz, acct)
        return new

    def backup_user(self, user_id):
        acct = self.accounts.get(user_id)
        if acct is None:
            return False
        acct.snapshot = (acct.capacity, {n: self.size[n] for n in acct.names})
        return True

    def restore_user(self, user_id):
        acct = self.accounts.get(user_id)
        if acct is None or acct.snapshot is None:
            return None
        cap, files = acct.snapshot
        create = [n for n in files if n not in acct.names]
        if any(n in self.size for n in create):
            return None
        for n in [n for n in acct.names if n not in files]:
            self._drop(n)
        for n in create:
            self._put(n, files[n], acct)
        acct.capacity = cap
        return acct.free
