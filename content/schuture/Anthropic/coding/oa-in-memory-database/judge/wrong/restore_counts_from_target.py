# expect: WA
# restore restarts each remaining TTL from timestamp_to_restore instead of from the restore's own timestamp.
import bisect


class InMemoryDatabase:
    def __init__(self):
        self.now = 0
        self.store = {}            # (key, field) -> [value, expires_at or None]
        self.snap_times = []       # sorted backup timestamps
        self.snaps = {}            # ts -> tuple of (key, field, value, remaining or None)

    # ---- time handling
    def _enter(self, ts):
        self.now = ts
        self._purge(ts)
        return ts

    def _plain(self):
        ts = self.now
        self._purge(ts)
        self.now = ts + 1
        return ts

    def _purge(self, ts):
        dead = [kf for kf, (_, exp) in self.store.items() if exp is not None and exp <= ts]
        for kf in dead:
            del self.store[kf]

    # ---- core, given an already-entered timestamp
    def _set(self, key, field, value, exp):
        self.store[(key, field)] = [value, exp]
        return ""

    def _get(self, key, field):
        e = self.store.get((key, field))
        return e[0] if e else ""

    def _delete(self, key, field):
        return "true" if self.store.pop((key, field), None) is not None else "false"

    def _scan(self, key, prefix):
        items = sorted((f, v) for (k, f), (v, _) in self.store.items() if k == key and f[:len(prefix)] == prefix)
        return ", ".join("%s(%s)" % (f, v) for f, v in items)

    # ---- level 1/2
    def set(self, key, field, value):
        self._plain()
        return self._set(key, field, value, None)

    def get(self, key, field):
        self._plain()
        return self._get(key, field)

    def delete(self, key, field):
        self._plain()
        return self._delete(key, field)

    def scan(self, key):
        self._plain()
        return self._scan(key, "")

    def scan_by_prefix(self, key, prefix):
        self._plain()
        return self._scan(key, prefix)

    # ---- level 3
    def set_at(self, timestamp, key, field, value):
        self._enter(timestamp)
        return self._set(key, field, value, None)

    def set_at_with_ttl(self, timestamp, key, field, value, ttl):
        self._enter(timestamp)
        return self._set(key, field, value, timestamp + ttl)

    def get_at(self, timestamp, key, field):
        self._enter(timestamp)
        return self._get(key, field)

    def delete_at(self, timestamp, key, field):
        self._enter(timestamp)
        return self._delete(key, field)

    def scan_at(self, timestamp, key):
        self._enter(timestamp)
        return self._scan(key, "")

    def scan_by_prefix_at(self, timestamp, key, prefix):
        self._enter(timestamp)
        return self._scan(key, prefix)

    # ---- level 4
    def backup(self, timestamp):
        self._enter(timestamp)
        frozen = tuple((k, f, v, None if exp is None else exp - timestamp) for (k, f), (v, exp) in self.store.items())
        if timestamp not in self.snaps:
            bisect.insort(self.snap_times, timestamp)
        self.snaps[timestamp] = frozen
        return ""

    def restore(self, timestamp, timestamp_to_restore):
        self._enter(timestamp)
        i = bisect.bisect_right(self.snap_times, timestamp_to_restore) - 1
        frozen = self.snaps[self.snap_times[i]]
        self.store = {(k, f): [v, None if rem is None else timestamp_to_restore + rem] for k, f, v, rem in frozen}
        return ""
