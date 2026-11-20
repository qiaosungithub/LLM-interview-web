# expect: WA
# The doubled part is measured on the whole session instead of on its overlap with [start, end), so double pay outside the window is counted.
class _Worker:
    def __init__(self, position: str, hourly_rate: int):
        self.position = position
        self.hourly_rate = hourly_rate
        self.open_clock_in: int | None = None
        self.closed_sessions: list[tuple[int, int, int]] = []   # (start, end, rate), rate fixed at close
        self.pending_promotion: tuple[str, int] | None = None


class Workforce:
    def __init__(self):
        self._workers: dict[str, _Worker] = {}

    def add_worker(self, worker_id: str, position: str, hourly_rate: int) -> bool:
        if worker_id in self._workers:
            return False
        self._workers[worker_id] = _Worker(position, hourly_rate)
        return True

    def clock_in(self, worker_id: str, timestamp: int) -> bool:
        worker = self._workers.get(worker_id)
        if worker is None or worker.open_clock_in is not None:
            return False
        if worker.pending_promotion is not None:
            worker.position, worker.hourly_rate = worker.pending_promotion
            worker.pending_promotion = None
        worker.open_clock_in = timestamp
        return True

    def clock_out(self, worker_id: str, timestamp: int) -> bool:
        worker = self._workers.get(worker_id)
        if worker is None or worker.open_clock_in is None:
            return False
        worker.closed_sessions.append((worker.open_clock_in, timestamp, worker.hourly_rate))
        worker.open_clock_in = None
        return True

def _closed_time(worker: _Worker) -> int:
    return sum(end - start for start, end, _ in worker.closed_sessions)


def get_total_work_time(self, worker_id: str) -> int | None:
    worker = self._workers.get(worker_id)
    return None if worker is None else _closed_time(worker)


def top_k_workers(self, k: int) -> list[str]:
    ranked = sorted(self._workers.items(), key=lambda item: (-_closed_time(item[1]), item[0]))
    return [worker_id for worker_id, _ in ranked[:k]]


Workforce.get_total_work_time = get_total_work_time
Workforce.top_k_workers = top_k_workers

def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Length of the overlap between half-open [a_start, a_end) and [b_start, b_end)."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _sessions_as_of(worker: _Worker, query_end: int) -> list[tuple[int, int, int]]:
    """Every closed session, plus the open one (if any) clamped to end at query_end."""
    sessions = list(worker.closed_sessions)
    if worker.open_clock_in is not None:
        sessions.append((worker.open_clock_in, query_end, worker.hourly_rate))
    return sessions


def set_promotion(self, worker_id: str, new_position: str, new_hourly_rate: int) -> bool:
    worker = self._workers.get(worker_id)
    if worker is None:
        return False
    worker.pending_promotion = (new_position, new_hourly_rate)
    return True


def calculate_pay(self, worker_id: str, start: int, end: int) -> int | None:
    worker = self._workers.get(worker_id)
    if worker is None:
        return None
    return sum(_overlap(s, e, start, end) * rate for s, e, rate in _sessions_as_of(worker, end))


Workforce.set_promotion = set_promotion
Workforce.calculate_pay = calculate_pay

def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def set_double_pay(self, start: int, end: int) -> None:
    if not hasattr(self, "_bonus_periods"):
        self._bonus_periods: list[tuple[int, int]] = []
    self._bonus_periods = _merge_intervals(self._bonus_periods + [(start, end)])


def calculate_pay(self, worker_id: str, start: int, end: int) -> int | None:
    worker = self._workers.get(worker_id)
    if worker is None:
        return None
    bonus_periods = getattr(self, "_bonus_periods", [])
    total = 0
    for s, e, rate in _sessions_as_of(worker, end):
        seg_start, seg_end = max(s, start), min(e, end)
        if seg_start >= seg_end:
            continue
        doubled = sum(_overlap(s, e, bs, be) for bs, be in bonus_periods)
        total += (seg_end - seg_start - doubled) * rate + doubled * rate * 2
    return total


Workforce.set_double_pay = set_double_pay
Workforce.calculate_pay = calculate_pay
