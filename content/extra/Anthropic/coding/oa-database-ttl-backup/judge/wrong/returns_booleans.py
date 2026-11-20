# expect: WA
# delete and the conditional methods return Python True/False instead of "true"/"false"
class InMemoryDatabase:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}          # key -> {field: value}

    def set(self, key: str, field: str, value: str) -> str:
        self._data.setdefault(key, {})[field] = value
        return ""

    def get(self, key: str, field: str) -> str:
        return self._data.get(key, {}).get(field, "")

    def delete(self, key: str, field: str) -> str:
        if field not in self._data.get(key, {}):
            return "false"
        del self._data[key][field]
        return "true"

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        if self.get(key, field) != expected_value:        # NOTE: a missing field reads as "", never equal
            return "false"
        self.set(key, field, new_value)
        return "true"

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        if self.get(key, field) != expected_value:
            return "false"
        return self.delete(key, field)


class InMemoryDatabase(InMemoryDatabase):
    def scan(self, key: str) -> str:
        return self.scan_by_prefix(key, "")

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        record = self._data.get(key, {})
        return ", ".join(f"{field}({record[field]})" for field in sorted(record) if field.startswith(prefix))


import math


class InMemoryDatabase(InMemoryDatabase):
    # self._data now maps key -> {field: (value, expires_at)}; expires_at is math.inf without a TTL

    def _write(self, key: str, field: str, value: str, timestamp: int, ttl: float = math.inf) -> None:
        self._data.setdefault(key, {})[field] = (value, timestamp + ttl)

    def _erase(self, key: str, field: str, timestamp: int) -> None:
        del self._data[key][field]

    def set_at(self, key: str, field: str, value: str, timestamp: int) -> str:
        self._write(key, field, value, timestamp)
        return ""

    def set_at_with_ttl(self, key: str, field: str, value: str, timestamp: int, ttl: int) -> str:
        self._write(key, field, value, timestamp, ttl)
        return ""

    def get_at(self, key: str, field: str, timestamp: int) -> str:
        value, expires_at = self._data.get(key, {}).get(field, ("", math.inf))
        return value if timestamp < expires_at else ""    # NOTE: strict -- at expires_at the field is already gone

    def delete_at(self, key: str, field: str, timestamp: int) -> str:
        if self.get_at(key, field, timestamp) == "":
            return False
        self._erase(key, field, timestamp)
        return True

    def scan_at(self, key: str, timestamp: int) -> str:
        return self.scan_by_prefix_at(key, "", timestamp)

    def scan_by_prefix_at(self, key: str, prefix: str, timestamp: int) -> str:
        record = self._data.get(key, {})
        return ", ".join(f"{field}({record[field][0]})" for field in sorted(record)
                         if field.startswith(prefix) and timestamp < record[field][1])

    def compare_and_set_at(self, key: str, field: str, expected_value: str, new_value: str,
                           timestamp: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return False
        self._write(key, field, new_value, timestamp)
        return True

    def compare_and_set_at_with_ttl(self, key: str, field: str, expected_value: str, new_value: str,
                                    timestamp: int, ttl: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return False
        self._write(key, field, new_value, timestamp, ttl)
        return True

    def compare_and_delete_at(self, key: str, field: str, expected_value: str, timestamp: int) -> str:
        if self.get_at(key, field, timestamp) != expected_value:
            return False
        self._erase(key, field, timestamp)
        return True

    # Levels 1-2: the same operations at time 0, before every real timestamp, when nothing has expired yet
    def set(self, key: str, field: str, value: str) -> str:
        return self.set_at(key, field, value, 0)

    def get(self, key: str, field: str) -> str:
        return self.get_at(key, field, 0)

    def delete(self, key: str, field: str) -> str:
        return self.delete_at(key, field, 0)

    def scan(self, key: str) -> str:
        return self.scan_at(key, 0)

    def scan_by_prefix(self, key: str, prefix: str) -> str:
        return self.scan_by_prefix_at(key, prefix, 0)

    def compare_and_set(self, key: str, field: str, expected_value: str, new_value: str) -> str:
        return self.compare_and_set_at(key, field, expected_value, new_value, 0)

    def compare_and_delete(self, key: str, field: str, expected_value: str) -> str:
        return self.compare_and_delete_at(key, field, expected_value, 0)


class InMemoryDatabase(InMemoryDatabase):
    def __init__(self) -> None:
        super().__init__()
        self._backups: list = []      # (timestamp, {key: {field: (value, remaining)}}), oldest first
        self._history: dict = {}      # (key, field) -> [(timestamp, value, expires_at)]; value "" = deleted

    def _write(self, key: str, field: str, value: str, timestamp: int, ttl: float = math.inf) -> None:
        super()._write(key, field, value, timestamp, ttl)
        self._history.setdefault((key, field), []).append((timestamp, value, timestamp + ttl))

    def _erase(self, key: str, field: str, timestamp: int) -> None:
        super()._erase(key, field, timestamp)
        self._history[(key, field)].append((timestamp, "", math.inf))

    def backup(self, timestamp: int) -> str:
        snapshot = {}
        for key, record in self._data.items():
            alive = {field: (value, expires_at - timestamp)
                     for field, (value, expires_at) in record.items() if timestamp < expires_at}
            if alive:                                     # NOTE: a record whose fields have all expired is not counted
                snapshot[key] = alive
        self._backups.append((timestamp, snapshot))
        return str(len(snapshot))

    def restore(self, timestamp: int, timestamp_to_restore: int) -> str:
        snapshot = next(saved for when, saved in reversed(self._backups) if when <= timestamp_to_restore)
        for key, record in self._data.items():
            for field in list(record):                    # list(): _erase removes from record while we loop
                self._erase(key, field, timestamp)
        for key, record in snapshot.items():
            for field, (value, remaining) in record.items():
                self._write(key, field, value, timestamp, remaining)     # NOTE: expires at timestamp + remaining
        return ""

    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> str:
        for when, value, expires_at in reversed(self._history.get((key, field), [])):
            if when <= at_timestamp:
                return value if at_timestamp < expires_at else ""
        return ""
