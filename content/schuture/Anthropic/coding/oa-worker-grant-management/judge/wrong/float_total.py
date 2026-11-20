# expect: WA
# calculate_pay totals in a float (pay starts at 0.0), which is no longer exact once the total passes 2**53.
class _Worker:
    __slots__ = ("position", "rate", "pending", "opened", "sessions", "total")

    def __init__(self, position, rate):
        self.position, self.rate = position, rate
        self.pending = None         # (position, rate) waiting for the next clock_in
        self.opened = None          # (start, rate) of the open session
        self.sessions = []          # closed sessions: (start, end, rate)
        self.total = 0


class Workforce:
    def __init__(self):
        self._workers = {}
        self._double = []           # raw [start, end) periods, as given
        self._merged = []           # their union, sorted and disjoint (rebuilt when stale)
        self._stale = False

    # Level 1
    def add_worker(self, worker_id, position, hourly_rate):
        if worker_id in self._workers:
            return False
        self._workers[worker_id] = _Worker(position, hourly_rate)
        return True

    def clock_in(self, worker_id, timestamp):
        w = self._workers.get(worker_id)
        if w is None or w.opened is not None:
            return False
        if w.pending is not None:
            w.position, w.rate = w.pending
            w.pending = None
        w.opened = (timestamp, w.rate)
        return True

    def clock_out(self, worker_id, timestamp):
        w = self._workers.get(worker_id)
        if w is None or w.opened is None:
            return False
        start, rate = w.opened
        w.opened = None
        w.sessions.append((start, timestamp, rate))
        w.total += timestamp - start
        return True

    # Level 2
    def get_total_work_time(self, worker_id):
        w = self._workers.get(worker_id)
        return None if w is None else w.total

    def top_k_workers(self, k):
        ranked = sorted(self._workers.items(), key=lambda kv: (-kv[1].total, kv[0]))
        return [wid for wid, _ in ranked[:k]]

    # Level 3
    def set_promotion(self, worker_id, new_position, new_hourly_rate):
        w = self._workers.get(worker_id)
        if w is None:
            return False
        w.pending = (new_position, new_hourly_rate)
        return True

    def calculate_pay(self, worker_id, start, end):
        w = self._workers.get(worker_id)
        if w is None:
            return None
        pieces = list(w.sessions)
        if w.opened is not None:
            pieces.append((w.opened[0], end, w.opened[1]))
        pay = 0.0
        for s, e, rate in pieces:
            a, b = max(s, start), min(e, end)
            if a < b:
                pay += rate * ((b - a) + self._doubled(a, b))
        return pay

    # Level 4
    def set_double_pay(self, start, end):
        if start < end:
            self._double.append((start, end))
            self._stale = True

    def _doubled(self, a, b):
        if self._stale:
            self._merged = []
            for s, e in sorted(self._double):
                if self._merged and s <= self._merged[-1][1]:
                    if e > self._merged[-1][1]:
                        self._merged[-1][1] = e
                else:
                    self._merged.append([s, e])
            self._stale = False
        return sum(max(0, min(b, e) - max(a, s)) for s, e in self._merged)
