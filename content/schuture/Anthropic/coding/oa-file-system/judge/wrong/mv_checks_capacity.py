# expect: WA
# mv runs cp's capacity check too, so a user at full capacity cannot move its own files.
from __future__ import annotations
import copy


class _File:
    __slots__ = ("content", "owner")

    def __init__(self, content: str, owner: str):
        self.content = content
        self.owner = owner


class _Dir:
    __slots__ = ("children",)

    def __init__(self):
        self.children: dict[str, "_Dir | _File"] = {}


def _split(path: str) -> list[str] | None:
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    parts = [seg for seg in path.split("/") if seg != ""]   # NOTE: collapses repeated "/" and a trailing "/"
    if any(seg in (".", "..") for seg in parts):
        return None
    return parts


class FileSystem:
    def __init__(self):
        self._root = _Dir()
        self._users: dict[str, dict] = {}        # user_id -> {"capacity": int, "used": int} -- Level 3

    def _precheck(self, comps: list[str]):
        node = self._root
        for c in comps:
            if not isinstance(node, _Dir):
                return "blocked", None
            nxt = node.children.get(c)
            if nxt is None:
                return "ok", None           # NOTE: nothing under an absent directory can exist either
            node = nxt
        return "ok", node

    def _materialize(self, comps: list[str]) -> "_Dir":
        node = self._root
        for c in comps:
            if c not in node.children:
                node.children[c] = _Dir()
            node = node.children[c]
        return node

    def _resolve(self, comps: list[str]):
        status, node = self._precheck(comps)
        return node if status == "ok" else None

    def mkdir(self, path: str) -> bool:
        comps = _split(path)
        if comps is None:
            return False
        status, existing = self._precheck(comps)
        if status == "blocked" or isinstance(existing, _File):
            return False
        self._materialize(comps)
        return True

    def write(self, path: str, content: str) -> bool:
        comps = _split(path)
        if comps is None or not comps:          # NOTE: not comps -- "/" itself is never a file
            return False
        status, existing = self._precheck(comps)
        if status == "blocked" or isinstance(existing, _Dir):
            return False
        parent = self._materialize(comps[:-1])
        parent.children[comps[-1]] = _File(content, "admin")
        return True

    def read(self, path: str) -> str | None:
        comps = _split(path)
        node = self._resolve(comps) if comps is not None else None
        return node.content if isinstance(node, _File) else None

    def ls(self, path: str) -> list[str] | None:
        comps = _split(path)
        node = self._resolve(comps) if comps is not None else None
        return sorted(node.children) if isinstance(node, _Dir) else None



def _deepcopy(node):
    if isinstance(node, _File):
        return _File(node.content, node.owner)
    new = _Dir()
    for k, v in node.children.items():
        new.children[k] = _deepcopy(v)
    return new


def _on_same_chain(a: list[str], b: list[str]) -> bool:
    """True if a is a prefix of b, or b is a prefix of a (equal counts as both)."""
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return longer[:len(shorter)] == shorter


def _size_of(node) -> int:
    if isinstance(node, _File):
        return len(node.content)
    return sum(_size_of(c) for c in node.children.values())


def size(self, path: str) -> int | None:
    comps = _split(path)
    node = self._resolve(comps) if comps is not None else None
    return None if node is None else _size_of(node)


def rm(self, path: str) -> bool:
    comps = _split(path)
    if comps is None or not comps:
        return False
    parent = self._resolve(comps[:-1])
    if not isinstance(parent, _Dir) or comps[-1] not in parent.children:
        return False
    del parent.children[comps[-1]]
    return True


def _move_or_copy(self, src: str, dst: str, remove_src: bool) -> bool:
    src_comps, dst_comps = _split(src), _split(dst)
    if src_comps is None or not src_comps or dst_comps is None or not dst_comps:
        return False
    src_node = self._resolve(src_comps)
    if src_node is None or _on_same_chain(src_comps, dst_comps):
        return False                        # NOTE: covers dst == src, dst inside src, AND src inside dst
    status, existing = self._precheck(dst_comps)
    if status == "blocked" or (existing is not None and type(existing) is not type(src_node)):
        return False                        # NOTE: file/directory type mismatch at an existing destination
    new_node = src_node if remove_src else _deepcopy(src_node)
    parent = self._materialize(dst_comps[:-1])   # -- validated: safe to mutate from here on --
    if remove_src:
        src_parent = self._resolve(src_comps[:-1])
        del src_parent.children[src_comps[-1]]
    parent.children[dst_comps[-1]] = new_node
    return True


def mv(self, src: str, dst: str) -> bool:
    return self._move_or_copy(src, dst, remove_src=True)


def cp(self, src: str, dst: str) -> bool:
    return self._move_or_copy(src, dst, remove_src=False)


FileSystem.size = size
FileSystem.rm = rm
FileSystem._move_or_copy = _move_or_copy
FileSystem.mv = mv
FileSystem.cp = cp



def _files_by_owner(node) -> dict:
    totals: dict[str, int] = {}
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, _File):
            totals[n.owner] = totals.get(n.owner, 0) + len(n.content)
        else:
            stack.extend(n.children.values())
    return totals


def add_user(self, user_id: str, capacity: int) -> bool:
    if user_id == "admin" or user_id in self._users or capacity < 0:
        return False
    self._users[user_id] = {"capacity": capacity, "used": 0}
    return True


def write(self, path: str, content: str, owner: str = "admin") -> bool:
    if owner != "admin" and owner not in self._users:
        return False
    comps = _split(path)
    if comps is None or not comps:
        return False
    status, existing = self._precheck(comps)
    if status == "blocked" or isinstance(existing, _Dir):
        return False
    if isinstance(existing, _File) and existing.owner != owner:
        return False                        # NOTE: an existing file can only be overwritten by its own owner
    old_size = len(existing.content) if isinstance(existing, _File) else 0
    if owner != "admin":
        u = self._users[owner]
        if u["used"] - old_size + len(content) > u["capacity"]:   # NOTE: old_size cancels when owner
            return False                                          # overwrites its own file, so shrinking never fails
    parent = self._materialize(comps[:-1])    # -- validated: safe to mutate from here on --
    if owner != "admin":
        self._users[owner]["used"] += len(content) - old_size
    parent.children[comps[-1]] = _File(content, owner)
    return True


def _move_or_copy(self, src: str, dst: str, remove_src: bool) -> bool:
    src_comps, dst_comps = _split(src), _split(dst)
    if src_comps is None or not src_comps or dst_comps is None or not dst_comps:
        return False
    src_node = self._resolve(src_comps)
    if src_node is None or _on_same_chain(src_comps, dst_comps):
        return False
    status, existing = self._precheck(dst_comps)
    if status == "blocked" or (existing is not None and type(existing) is not type(src_node)):
        return False
    if isinstance(existing, _File) and existing.owner != src_node.owner:
        return False                        # NOTE: same rule as write -- owner must match to overwrite a file
    freed = _files_by_owner(existing) if existing is not None else {}
    added = _files_by_owner(src_node)       # checked for mv as well as cp
    for owner, amount in added.items():
        if owner == "admin":
            continue
        u = self._users[owner]
        if u["used"] - freed.get(owner, 0) + amount > u["capacity"]:
            return False
    parent = self._materialize(dst_comps[:-1])   # -- validated: safe to mutate from here on --
    for owner, amount in freed.items():
        if owner != "admin":
            self._users[owner]["used"] -= amount
    if remove_src:
        new_node = src_node
        src_parent = self._resolve(src_comps[:-1])
        del src_parent.children[src_comps[-1]]
    else:
        new_node = _deepcopy(src_node)
        for owner, amount in _files_by_owner(src_node).items():
            if owner != "admin":
                self._users[owner]["used"] += amount
    parent.children[dst_comps[-1]] = new_node
    return True


def rm(self, path: str) -> bool:
    comps = _split(path)
    if comps is None or not comps:
        return False
    parent = self._resolve(comps[:-1])
    if not isinstance(parent, _Dir) or comps[-1] not in parent.children:
        return False
    for owner, amount in _files_by_owner(parent.children[comps[-1]]).items():
        if owner != "admin":                # NOTE: rm must free capacity too, not just write/mv/cp/eviction
            self._users[owner]["used"] -= amount
    del parent.children[comps[-1]]
    return True


def update_capacity(self, user_id: str, capacity: int) -> int | None:
    if user_id not in self._users or capacity < 0:
        return None
    u = self._users[user_id]
    u["capacity"] = capacity
    if u["used"] <= capacity:
        return 0
    victims = []                            # (size, full path, parent dir, key) for every file user_id owns
    def collect(node, prefix):
        for name, child in node.children.items():
            path = f"{prefix}/{name}"
            if isinstance(child, _File):
                if child.owner == user_id:
                    victims.append((len(child.content), path, node, name))
            else:
                collect(child, path)
    collect(self._root, "")
    victims.sort(key=lambda v: (-v[0], v[1]))   # largest first; ties by path ascending
    deleted = 0
    for sz, _, parent, name in victims:
        if u["used"] <= capacity:
            break
        del parent.children[name]
        u["used"] -= sz
        deleted += 1
    return deleted


FileSystem.add_user = add_user
FileSystem.write = write
FileSystem.rm = rm
FileSystem._move_or_copy = _move_or_copy
FileSystem.update_capacity = update_capacity



def snapshot(self, snapshot_id: str) -> bool:
    if not hasattr(self, "_snapshots"):
        self._snapshots = {}
    if snapshot_id in self._snapshots:
        return False
    self._snapshots[snapshot_id] = (_deepcopy(self._root), copy.deepcopy(self._users))
    return True


def restore(self, snapshot_id: str) -> bool:
    if not hasattr(self, "_snapshots") or snapshot_id not in self._snapshots:
        return False
    root, users = self._snapshots[snapshot_id]
    self._root = _deepcopy(root)            # NOTE: copy out -- restoring twice must not share state either
    self._users = copy.deepcopy(users)
    return True


FileSystem.snapshot = snapshot
FileSystem.restore = restore
