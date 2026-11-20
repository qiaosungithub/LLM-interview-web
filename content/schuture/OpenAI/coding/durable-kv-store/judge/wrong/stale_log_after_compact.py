# expect: WA
# compact() moves the log to a new file but the same instance keeps appending to the old, deleted name.
import re
import zlib

_ESC = {"\\": "\\\\", "\n": "\\n", "=": "\\e"}
_UNESC = {"\\": "\\", "n": "\n", "e": "="}
_ESC_RE = re.compile(r"[\\\n=]")
_UNESC_RE = re.compile(r"\\(.)", re.S)


def _esc(s):
    return _ESC_RE.sub(lambda m: _ESC[m.group(0)], s)


def _unesc(s):
    return _UNESC_RE.sub(lambda m: _UNESC[m.group(1)], s)


def dump(d):
    text = "".join(f"{_esc(k)}={_esc(v)}\n" for k, v in d.items())
    return text.encode("utf-8", "surrogatepass")


def parse(raw):
    out = {}
    text = raw.decode("utf-8", "surrogatepass")
    for line in text.split("\n")[:-1]:
        k, v = line.split("=", 1)
        out[_unesc(k)] = _unesc(v)
    return out


class KVStore:
    FILE = "store.txt"

    def __init__(self, fs):
        self.fs = fs
        self.d = {}

    def put(self, key, value):
        self.d[key] = value

    def get(self, key):
        return self.d.get(key)

    def save(self):
        self.fs.write(self.FILE, dump(self.d))

    def load(self):
        raw = self.fs.read(self.FILE)
        self.d = {} if raw is None else parse(raw)


class ChunkedKVStore:
    POINTER = "CURRENT"

    def __init__(self, fs):
        self.fs = fs
        self.d = {}

    def put(self, key, value):
        self.d[key] = value

    def get(self, key):
        return self.d.get(key)

    @staticmethod
    def _gen_of(name):
        if name.startswith("part."):
            try:
                return int(name.split(".")[1])
            except ValueError:
                return None
        return None

    def save(self):
        cap = self.fs.max_file_size
        if cap is not None and cap < 16:
            raise ValueError("max_file_size too small")
        blob = dump(self.d)
        size = cap if cap is not None else 1 << 16
        pieces = [blob[i:i + size] for i in range(0, len(blob), size)]
        gens = [g for g in map(self._gen_of, self.fs.list()) if g is not None]
        cur = self.fs.read(self.POINTER)
        if cur is not None:
            gens.append(int(cur.split()[0]))
        gen = max(gens, default=0) + 1
        mine = set()
        for i, p in enumerate(pieces):
            name = f"part.{gen}.{i}"
            self.fs.write(name, p)
            mine.add(name)
        self.fs.write(self.POINTER, f"{gen} {len(pieces)}".encode())
        mine.add(self.POINTER)
        for name in self.fs.list():
            if name not in mine:
                self.fs.delete(name)

    def load(self):
        cur = self.fs.read(self.POINTER)
        if cur is None:
            self.d = {}
            return
        gen, n = map(int, cur.split())
        self.d = parse(b"".join(self.fs.read(f"part.{gen}.{i}") for i in range(n)))


PUT, DEL = 1, 2


def _record(op, key, value=""):
    k = key.encode("utf-8", "surrogatepass")
    v = value.encode("utf-8", "surrogatepass")
    body = bytes([op]) + len(k).to_bytes(4, "big") + len(v).to_bytes(4, "big") + k + v
    return body + zlib.crc32(body).to_bytes(4, "big")


def replay(data, d):
    """Applies every valid record of data to d; returns the offset just past the last valid one."""
    pos, n = 0, len(data)
    while pos + 13 <= n:
        op = data[pos]
        if op not in (PUT, DEL):
            break
        kl = int.from_bytes(data[pos + 1:pos + 5], "big")
        vl = int.from_bytes(data[pos + 5:pos + 9], "big")
        end = pos + 9 + kl + vl
        if end + 4 > n or (op == DEL and vl):
            break
        if zlib.crc32(data[pos:end]) != int.from_bytes(data[end:end + 4], "big"):
            break
        try:
            key = data[pos + 9:pos + 9 + kl].decode("utf-8", "surrogatepass")
            val = data[pos + 9 + kl:end].decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:
            break
        if op == PUT:
            d[key] = val
        else:
            d.pop(key, None)
        pos = end + 4
    return pos


class LogKVStore:
    LOG = "journal"

    def __init__(self, fs):
        self.fs = fs
        self.d = {}

    def put(self, key, value):
        self.fs.append(self.LOG, _record(PUT, key, value))
        self.d[key] = value

    def delete(self, key):
        self.fs.append(self.LOG, _record(DEL, key))
        self.d.pop(key, None)

    def get(self, key):
        return self.d.get(key)

    def load(self):
        data = self.fs.read(self.LOG)
        d = {}
        if data is not None:
            good = replay(data, d)
            if good != len(data):
                self.fs.write(self.LOG, data[:good])
        self.d = d

    def compact(self):
        self.fs.write(self.LOG, b"".join(_record(PUT, k, v) for k, v in self.d.items()))


# ---- wrong: compact() writes the live pairs to a new log file and deletes the old one, but this instance keeps
# appending to the old name afterwards
class LogKVStore(LogKVStore):
    def __init__(self, fs):
        super().__init__(fs)
        self.name = "journal.0"

    def put(self, key, value):
        self.fs.append(self.name, _record(PUT, key, value))
        self.d[key] = value

    def delete(self, key):
        self.fs.append(self.name, _record(DEL, key))
        self.d.pop(key, None)

    def load(self):
        names = sorted((int(n.split(".")[1]), n) for n in self.fs.list() if n.startswith("journal."))
        self.name = names[-1][1] if names else "journal.0"
        data = self.fs.read(self.name)
        d = {}
        if data is not None:
            good = replay(data, d)
            if good != len(data):
                self.fs.write(self.name, data[:good])
        self.d = d

    def compact(self):
        n = int(self.name.split(".")[1]) + 1
        new = f"journal.{n}"
        self.fs.write(new, b"".join(_record(PUT, k, v) for k, v in self.d.items()))
        self.fs.delete(self.name)
        # forgot: self.name = new
