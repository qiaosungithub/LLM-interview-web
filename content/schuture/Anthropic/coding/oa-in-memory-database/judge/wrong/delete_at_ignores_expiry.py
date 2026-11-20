# expect: WA
# Lazy expiry: get and scan skip expired fields, but delete only checks that the field is still stored, so it
# returns "true" for a field whose TTL has run out.
class InMemoryDatabase:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, tuple[str, int | None]]] = {}
        self.clock = 0
        self.snapshots: dict[int, dict[str, dict[str, tuple[str, int | None]]]] = {}

    def _now(self, timestamp=None) -> int:
        if timestamp is None:
            timestamp = self.clock
            self.clock += 1
        else:
            self.clock = max(self.clock, timestamp)
        return timestamp

    @staticmethod
    def _visible(entry, ts) -> bool:
        return entry is not None and (entry[1] is None or ts < entry[1])

    def _write(self, ts, key, field, value, ttl):
        self.data.setdefault(key, {})[field] = (value, None if ttl is None else ts + ttl)
        return ""

    def _read(self, ts, key, field):
        entry = self.data.get(key, {}).get(field)
        return entry[0] if self._visible(entry, ts) else ""

    def _remove(self, ts, key, field):
        fields = self.data.get(key, {})
        if field in fields:                      # BUG: an expired field still counts as present
            del fields[field]
            return "true"
        return "false"

    def _list(self, ts, key, prefix):
        rows = sorted((f, e[0]) for f, e in self.data.get(key, {}).items()
                      if f.startswith(prefix) and self._visible(e, ts))
        return ", ".join(f"{f}({v})" for f, v in rows)

    def set(self, key, field, value):
        return self._write(self._now(), key, field, value, None)

    def get(self, key, field):
        return self._read(self._now(), key, field)

    def delete(self, key, field):
        return self._remove(self._now(), key, field)

    def scan(self, key):
        return self._list(self._now(), key, "")

    def scan_by_prefix(self, key, prefix):
        return self._list(self._now(), key, prefix)

    def set_at(self, timestamp, key, field, value):
        return self._write(self._now(timestamp), key, field, value, None)

    def set_at_with_ttl(self, timestamp, key, field, value, ttl):
        return self._write(self._now(timestamp), key, field, value, ttl)

    def get_at(self, timestamp, key, field):
        return self._read(self._now(timestamp), key, field)

    def delete_at(self, timestamp, key, field):
        return self._remove(self._now(timestamp), key, field)

    def scan_at(self, timestamp, key):
        return self._list(self._now(timestamp), key, "")

    def scan_by_prefix_at(self, timestamp, key, prefix):
        return self._list(self._now(timestamp), key, prefix)

    def backup(self, timestamp):
        ts = self._now(timestamp)
        snap = {}
        for key, fields in self.data.items():
            for f, e in fields.items():
                if self._visible(e, ts):
                    snap.setdefault(key, {})[f] = (e[0], None if e[1] is None else e[1] - ts)
        self.snapshots[ts] = snap
        return ""

    def restore(self, timestamp, timestamp_to_restore):
        ts = self._now(timestamp)
        best = max(t for t in self.snapshots if t <= timestamp_to_restore)
        self.data = {key: {f: (v, None if d is None else ts + d) for f, (v, d) in fields.items()}
                     for key, fields in self.snapshots[best].items()}
        return ""
