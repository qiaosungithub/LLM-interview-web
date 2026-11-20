# expect: WA
# Values and expiries live in two dicts; the plain set (kept from Level 1) writes only the value, so it leaves an earlier TTL in force.
import bisect


class InMemoryDatabase:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, str]] = {}
        self.expiry: dict[str, dict[str, int]] = {}
        self.clock = 0
        self.backups: dict[int, list[tuple[str, str, str, int | None]]] = {}

    # ---- time
    def _plain(self) -> int:
        ts = self.clock
        self.clock += 1
        return ts

    def _at(self, ts: int) -> int:
        self.clock = max(self.clock, ts)
        return ts

    def _alive(self, key: str, field: str, ts: int) -> bool:
        if field not in self.values.get(key, {}):
            return False
        exp = self.expiry.get(key, {}).get(field)
        return exp is None or ts < exp

    def _get(self, ts, key, field):
        return self.values[key][field] if self._alive(key, field, ts) else ""

    def _delete(self, ts, key, field):
        if not self._alive(key, field, ts):
            return "false"
        del self.values[key][field]
        self.expiry.get(key, {}).pop(field, None)
        return "true"

    def _scan(self, ts, key, prefix):
        names = sorted(f for f in self.values.get(key, {}) if f.startswith(prefix) and self._alive(key, f, ts))
        return ", ".join(f"{f}({self.values[key][f]})" for f in names)

    # ---- Level 1 / 2
    def set(self, key: str, field: str, value: str) -> str:
        self._plain()
        self.values.setdefault(key, {})[field] = value       # BUG: an old expiry for this field is left in place
        return ""

    def get(self, key: str, field: str) -> str:
        return self._get(self._plain(), key, field)

    def delete(self, key: str, field: str) -> str:
        return self._delete(self._plain(), key, field)

    def scan(self, key: str) -> str:
        return self._scan(self._plain(), key, "")

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        return self._scan(self._plain(), key, prefix)

    # ---- Level 3
    def set_at(self, timestamp: int, key: str, field: str, value: str) -> str:
        self._at(timestamp)
        self.values.setdefault(key, {})[field] = value
        self.expiry.get(key, {}).pop(field, None)
        return ""

    def set_at_with_ttl(self, timestamp: int, key: str, field: str, value: str, ttl: int) -> str:
        self._at(timestamp)
        self.values.setdefault(key, {})[field] = value
        self.expiry.setdefault(key, {})[field] = timestamp + ttl
        return ""

    def get_at(self, timestamp: int, key: str, field: str) -> str:
        return self._get(self._at(timestamp), key, field)

    def delete_at(self, timestamp: int, key: str, field: str) -> str:
        return self._delete(self._at(timestamp), key, field)

    def scan_at(self, timestamp: int, key: str) -> str:
        return self._scan(self._at(timestamp), key, "")

    def scan_by_prefix_at(self, timestamp: int, key: str, prefix: str) -> str:
        return self._scan(self._at(timestamp), key, prefix)

    # ---- Level 4
    def backup(self, timestamp: int) -> str:
        self._at(timestamp)
        snap = []
        for key, fields in self.values.items():
            for field, value in fields.items():
                if self._alive(key, field, timestamp):
                    exp = self.expiry.get(key, {}).get(field)
                    snap.append((key, field, value, None if exp is None else exp - timestamp))
        self.backups[timestamp] = snap
        return ""

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        self._at(timestamp)
        times = sorted(self.backups)
        best = times[bisect.bisect_right(times, timestamp_to_restore) - 1]
        self.values, self.expiry = {}, {}
        for key, field, value, left in self.backups[best]:
            self.values.setdefault(key, {})[field] = value
            if left is not None:
                self.expiry.setdefault(key, {})[field] = timestamp + left
        return ""
