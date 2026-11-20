# expect: WA
# Lazy heap that pushes a new entry only when a count goes up, so after a rebuild a count that drops is lost.
from collections import Counter
import heapq


class TrackerL1:
    def __init__(self):
        self.by_worker = {}                  # worker id -> Counter: value -> how often this worker reported it
        self.total = {}                      # value -> global count; values with count 0 are not kept

    def add_worker(self, worker_id):
        if worker_id in self.by_worker:
            return False
        self.by_worker[worker_id] = Counter()
        return True

    def report(self, worker_id, values):
        mine = self.by_worker.get(worker_id)
        if mine is None:
            return False
        for value in values:
            mine[value] += 1
            self._add(value, 1)
        return True

    def get_count(self, value):
        return self.total.get(value, 0)

    def get_worker_count(self, worker_id, value):
        mine = self.by_worker.get(worker_id)
        return None if mine is None else mine[value]         # a Counter answers 0 for a missing key

    def _add(self, value, delta):
        """The only place that changes a global count."""
        count = self.total.get(value, 0) + delta
        if count:
            self.total[value] = count
        else:
            del self.total[value]                            # back to 0: the value is gone
        h = self.__dict__.setdefault("heap", [])
        if delta > 0:
            heapq.heappush(h, (-count, value))
        if len(h) > 2 * len(self.total) + 64:
            self.heap = [(-c, v) for v, c in self.total.items()]
            heapq.heapify(self.heap)


class TrackerL2(TrackerL1):
    def get_mode(self):
        h = self.__dict__.setdefault("heap", [])
        while h and self.total.get(h[0][1], 0) != -h[0][0]:
            heapq.heappop(h)
        return h[0][1] if h else None

    def top_values(self, k):
        ranked = sorted(self.total.items(), key=lambda item: (-item[1], item[0]))
        return [f"{value}({count})" for value, count in ranked[:k]]


class TrackerL3(TrackerL2):
    def remove_worker(self, worker_id):
        mine = self.by_worker.pop(worker_id, None)
        if mine is None:
            return None
        for value, count in mine.items():
            self._add(value, -count)                         # take back exactly what this worker put in
        return sum(mine.values())


class TrackerL4(TrackerL3):
    def merge_workers(self, target_id, source_id):
        if target_id == source_id or target_id not in self.by_worker or source_id not in self.by_worker:
            return False
        self.by_worker[target_id].update(self.by_worker.pop(source_id))      # Counter.update adds the counts
        return True                                                          # global counts do not change


ModeTracker = TrackerL4
