# expect: WA
# cp gives every copied file the default owner 'admin' instead of keeping each file's own owner.
def _parts(path):
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    out = []
    for seg in path.split("/"):
        if seg == "":
            continue
        if seg in (".", ".."):
            return None
        out.append(seg)
    return out


def _to_admin(n):
    if isinstance(n, dict):
        return {k: _to_admin(v) for k, v in n.items()}
    return (n[0], 'admin')


def _clone(d):
    return {k: (_clone(v) if isinstance(v, dict) else v) for k, v in d.items()}


def _owner_totals(node, acc=None):
    acc = {} if acc is None else acc
    if isinstance(node, dict):
        for child in node.values():
            _owner_totals(child, acc)
    else:
        acc[node[1]] = acc.get(node[1], 0) + len(node[0])
    return acc


class FileSystem:
    def __init__(self):
        self.root = {}
        self.cap = {}
        self.used = {}
        self.snaps = {}

    # ---- helpers
    def _look(self, parts):
        """('blocked', None) if a proper ancestor is a file, else ('ok', node or None)."""
        node = self.root
        for seg in parts:
            if not isinstance(node, dict):
                return "blocked", None
            if seg not in node:
                return "ok", None
            node = node[seg]
        return "ok", node

    def _dir_for(self, parts):
        node = self.root
        for seg in parts:
            node = node.setdefault(seg, {})
        return node

    def _charge(self, totals, sign):
        for who, amt in totals.items():
            if who != "admin":
                self.used[who] += sign * amt

    # ---- Level 1
    def mkdir(self, path):
        p = _parts(path)
        if p is None:
            return False
        st, node = self._look(p)
        if st == "blocked" or (node is not None and not isinstance(node, dict)):
            return False
        self._dir_for(p)
        return True

    def write(self, path, content, owner="admin"):
        if owner != "admin" and owner not in self.cap:
            return False
        p = _parts(path)
        if not p:
            return False
        st, node = self._look(p)
        if st == "blocked" or isinstance(node, dict):
            return False
        if node is not None and node[1] != owner:
            return False
        old = len(node[0]) if node is not None else 0
        if owner != "admin" and self.used[owner] - old + len(content) > self.cap[owner]:
            return False
        self._dir_for(p[:-1])[p[-1]] = (content, owner)
        if owner != "admin":
            self.used[owner] += len(content) - old
        return True

    def read(self, path):
        p = _parts(path)
        if p is None:
            return None
        st, node = self._look(p)
        return node[0] if isinstance(node, tuple) else None

    def ls(self, path):
        p = _parts(path)
        if p is None:
            return None
        st, node = self._look(p)
        return sorted(node.keys()) if isinstance(node, dict) else None

    # ---- Level 2
    def size(self, path):
        p = _parts(path)
        if p is None:
            return None
        st, node = self._look(p)
        if node is None:
            return None
        return sum(_owner_totals(node).values())

    def rm(self, path):
        p = _parts(path)
        if not p:
            return False
        st, node = self._look(p)
        if node is None:
            return False
        self._charge(_owner_totals(node), -1)
        _, parent = self._look(p[:-1])
        del parent[p[-1]]
        return True

    def _transfer(self, src, dst, keep):
        s, d = _parts(src), _parts(dst)
        if s is None or d is None:
            return False
        n = min(len(s), len(d))
        if s[:n] == d[:n]:
            return False
        _, here = self._look(s)
        if here is None:
            return False
        st, there = self._look(d)
        if st == "blocked":
            return False
        if there is not None:
            if isinstance(here, dict) != isinstance(there, dict):
                return False
            if not isinstance(here, dict) and here[1] != there[1]:
                return False
        freed = _owner_totals(there) if there is not None else {}
        added = _owner_totals(here)
        if False:
            for who, amt in added.items():
                if who != "admin" and self.used[who] - freed.get(who, 0) + amt > self.cap[who]:
                    return False
        # commit
        self._charge(freed, -1)
        if keep:
            moving = _to_admin(here)
        else:
            moving = here
            _, sp = self._look(s[:-1])
            del sp[s[-1]]
        self._dir_for(d[:-1])[d[-1]] = moving
        return True

    def mv(self, src, dst):
        return self._transfer(src, dst, False)

    def cp(self, src, dst):
        return self._transfer(src, dst, True)

    # ---- Level 3
    def add_user(self, user_id, capacity):
        if user_id == "admin" or user_id in self.cap or capacity < 0:
            return False
        self.cap[user_id] = capacity
        self.used[user_id] = 0
        return True

    def update_capacity(self, user_id, capacity):
        if user_id not in self.cap or capacity < 0:
            return None
        self.cap[user_id] = capacity
        if self.used[user_id] <= capacity:
            return 0
        found = []
        stack = [("", self.root)]
        while stack:
            prefix, d = stack.pop()
            for name, child in d.items():
                full = prefix + "/" + name
                if isinstance(child, dict):
                    stack.append((full, child))
                elif child[1] == user_id:
                    found.append((-len(child[0]), full, d, name))
        found.sort(key=lambda t: (t[0], t[1]))
        count = 0
        for neg, _, d, name in found:
            if self.used[user_id] <= capacity:
                break
            del d[name]
            self.used[user_id] += neg
            count += 1
        return count

    # ---- Level 4
    def snapshot(self, snapshot_id):
        if snapshot_id in self.snaps:
            return False
        self.snaps[snapshot_id] = (_clone(self.root), dict(self.cap), dict(self.used))
        return True

    def restore(self, snapshot_id):
        if snapshot_id not in self.snaps:
            return False
        root, cap, used = self.snaps[snapshot_id]
        self.root, self.cap, self.used = _clone(root), dict(cap), dict(used)
        return True
