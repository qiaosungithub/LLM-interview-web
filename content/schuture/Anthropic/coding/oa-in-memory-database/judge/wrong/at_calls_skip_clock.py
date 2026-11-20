# expect: WA
# The _at methods leave the clock alone, so plain calls keep counting from their own calls only.
class InMemoryDatabase:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, str]] = {}

    def set(self, key: str, field: str, value: str) -> str:
        self._records.setdefault(key, {})[field] = value
        return ""

    def get(self, key: str, field: str) -> str:
        return self._records.get(key, {}).get(field, "")

    def delete(self, key: str, field: str) -> str:
        fields = self._records.get(key, {})
        if field in fields:
            del fields[field]
            return "true"
        return "false"

def _format_fields(fields: dict[str, str], names) -> str:
    return ", ".join(f"{name}({fields[name]})" for name in names)


class InMemoryDatabase(InMemoryDatabase):
    def scan(self, key: str) -> str:
        fields = self._records.get(key, {})
        return _format_fields(fields, sorted(fields))

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        fields = self._records.get(key, {})
        return _format_fields(fields, sorted(f for f in fields if f.startswith(prefix)))

class InMemoryDatabase(InMemoryDatabase):
    def __init__(self) -> None:
        # _records[key][field] = (value, expiry); expiry is None (permanent) or the first
        # timestamp at which the field is gone.
        self._records: dict[str, dict[str, tuple[str, int | None]]] = {}
        self._clock = 0

    def _tick(self) -> int:
        timestamp = self._clock
        self._clock += 1
        return timestamp

    def _bump(self, timestamp: int) -> None:
        return None                                 # explicit timestamps do not touch the call counter

    @staticmethod
    def _alive(entry: tuple[str, int | None] | None, timestamp: int) -> bool:
        return entry is not None and (entry[1] is None or timestamp < entry[1])   # NOTE: strict "<": t+ttl is dead

    def set(self, key: str, field: str, value: str) -> str:
        return self.set_at(self._tick(), key, field, value)

    def get(self, key: str, field: str) -> str:
        return self.get_at(self._tick(), key, field)

    def delete(self, key: str, field: str) -> str:
        return self.delete_at(self._tick(), key, field)

    def scan(self, key: str) -> str:
        return self.scan_at(self._tick(), key)

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        return self.scan_by_prefix_at(self._tick(), key, prefix)

    def set_at(self, timestamp: int, key: str, field: str, value: str) -> str:
        self._bump(timestamp)
        self._records.setdefault(key, {})[field] = (value, None)   # NOTE: plain-setting always clears a TTL
        return ""

    def set_at_with_ttl(self, timestamp: int, key: str, field: str, value: str, ttl: int) -> str:
        self._bump(timestamp)
        self._records.setdefault(key, {})[field] = (value, timestamp + ttl)   # NOTE: a fresh window, not extended
        return ""

    def get_at(self, timestamp: int, key: str, field: str) -> str:
        self._bump(timestamp)
        entry = self._records.get(key, {}).get(field)
        return entry[0] if self._alive(entry, timestamp) else ""

    def delete_at(self, timestamp: int, key: str, field: str) -> str:
        self._bump(timestamp)
        fields = self._records.get(key, {})
        if self._alive(fields.get(field), timestamp):
            del fields[field]
            return "true"
        return "false"                                            # NOTE: an already-expired field is "false"

    def _live_fields(self, key: str, timestamp: int) -> dict[str, str]:
        fields = self._records.get(key, {})
        return {name: value for name, (value, expiry) in fields.items()
                if self._alive((value, expiry), timestamp)}

    def scan_at(self, timestamp: int, key: str) -> str:
        self._bump(timestamp)
        live = self._live_fields(key, timestamp)
        return _format_fields(live, sorted(live))

    def scan_by_prefix_at(self, timestamp: int, key: str, prefix: str) -> str:
        self._bump(timestamp)
        live = self._live_fields(key, timestamp)
        return _format_fields(live, sorted(f for f in live if f.startswith(prefix)))

import copy


class InMemoryDatabase(InMemoryDatabase):
    def __init__(self) -> None:
        super().__init__()
        self._backups: dict[int, dict[str, dict[str, tuple[str, int | None]]]] = {}

    def backup(self, timestamp: int) -> str:
        self._bump(timestamp)
        snapshot: dict[str, dict[str, tuple[str, int | None]]] = {}
        for key, fields in self._records.items():
            live = {name: (value, None if expiry is None else expiry - timestamp)
                    for name, (value, expiry) in fields.items() if self._alive((value, expiry), timestamp)}
            if live:
                snapshot[key] = live
        self._backups[timestamp] = snapshot         # NOTE: a second backup at this timestamp replaces the first
        return ""

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        self._bump(timestamp)
        backup_ts = max(ts for ts in self._backups if ts <= timestamp_to_restore)
        restored: dict[str, dict[str, tuple[str, int | None]]] = {}
        for key, fields in self._backups[backup_ts].items():
            restored[key] = {name: (value, None if delta is None else timestamp + delta)   # NOTE: delta, not the
                              for name, (value, delta) in fields.items()}                   #  backup's own clock
        self._records = copy.deepcopy(restored)      # NOTE: deep copy -- later writes must not mutate the backup
        return ""
