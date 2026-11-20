"""A literal model of the statement, used for the expected values of the random tests, and the random
operation-sequence generator.

The model stores each closed session as (start, end, rate) with the rate the session's own clock_in
used, and answers calculate_pay tick by tick: for every integer tick t of [start, end) it finds the
session containing t (the open one runs to the query's end) and pays its rate, twice that when any
double-pay period contains t. No interval arithmetic, no merging."""


class _Worker:
    def __init__(self, rate):
        self.rate = rate
        self.pending = None         # rate of the pending promotion
        self.open = None            # (clock_in timestamp, rate) of the open session
        self.closed = []            # (start, end, rate)


class Workforce:
    def __init__(self):
        self.workers = {}
        self.periods = []

    # Level 1
    def add_worker(self, wid, position, rate):
        if wid in self.workers:
            return False
        self.workers[wid] = _Worker(rate)
        return True

    def clock_in(self, wid, ts):
        w = self.workers.get(wid)
        if w is None or w.open is not None:
            return False
        if w.pending is not None:
            w.rate, w.pending = w.pending, None
        w.open = (ts, w.rate)
        return True

    def clock_out(self, wid, ts):
        w = self.workers.get(wid)
        if w is None or w.open is None:
            return False
        w.closed.append((w.open[0], ts, w.open[1]))
        w.open = None
        return True

    # Level 2
    def get_total_work_time(self, wid):
        w = self.workers.get(wid)
        return None if w is None else sum(e - s for s, e, _ in w.closed)

    def top_k_workers(self, k):
        return sorted(self.workers, key=lambda wid: (-self.get_total_work_time(wid), wid))[:k]

    # Level 3
    def set_promotion(self, wid, position, rate):
        w = self.workers.get(wid)
        if w is None:
            return False
        w.pending = rate
        return True

    def calculate_pay(self, wid, start, end):
        w = self.workers.get(wid)
        if w is None:
            return None
        sessions = list(w.closed) + ([(w.open[0], end, w.open[1])] if w.open else [])
        total = 0
        for t in range(start, end):
            rates = [rate for s, e, rate in sessions if s <= t < e]
            if rates:
                double = any(s <= t < e for s, e in self.periods)
                total += rates[0] * (2 if double else 1)
        return total

    # Level 4
    def set_double_pay(self, start, end):
        self.periods.append((start, end))
        return None


# ---------------------------------------------------------------- random operation sequences

KINDS = {   # operation -> weight
    1: {"add_worker": 3, "clock_in": 5, "clock_out": 5},
    2: {"add_worker": 3, "clock_in": 5, "clock_out": 5, "get_total_work_time": 3, "top_k_workers": 3},
    3: {"add_worker": 2, "clock_in": 5, "clock_out": 5, "get_total_work_time": 1, "top_k_workers": 1,
        "set_promotion": 4, "calculate_pay": 5},
    4: {"add_worker": 2, "clock_in": 5, "clock_out": 4, "get_total_work_time": 1, "top_k_workers": 1,
        "set_promotion": 3, "calculate_pay": 6, "set_double_pay": 4},
}
IDS = ["alice", "al", "bo", "cy"]       # "al" < "alice": ties are broken by plain string order
GHOST = "dee"                           # never registered
POSITIONS = ["Engineer", "Designer", "Manager", "工程师"]


def random_ops(r, level, n_ops):
    """A sequence of (method, args) calls. clock_in/clock_out timestamps never decrease. The draws, made
    while the model follows along, often hit the interesting cases: repeated clock_in and clock_out,
    zero-length and back-to-back sessions, promotions set while a session is open or replaced before use,
    pay windows that start or end exactly on a session or double-pay boundary, and overlapping, nested,
    touching and empty double-pay periods."""
    m = Workforce()
    kinds, weights = zip(*KINDS[level].items())
    clock = r.randint(0, 5)
    ops = []

    def registered(p=0.85):
        return r.choice(list(m.workers)) if m.workers and r.random() < p else r.choice(IDS + [GHOST])

    def pick(want_open, p):
        """A registered worker with (or without) an open session, with probability p when there is one."""
        pool = [wid for wid, w in m.workers.items() if (w.open is not None) == want_open]
        return r.choice(pool) if pool and r.random() < p else registered()

    def points(wid):
        w = m.workers.get(wid)
        pts = [0, clock, clock + 5]
        if w is not None:
            pts += [x for s, e, _ in w.closed for x in (s, e)] + ([w.open[0]] if w.open else [])
        pts += [x for s, e in m.periods for x in (s, e)]
        return pts

    for _ in range(n_ops):
        kind = r.choices(kinds, weights)[0]
        if not m.workers and r.random() < 0.7:
            kind = "add_worker"
        if kind == "add_worker":
            fresh = [wid for wid in IDS if wid not in m.workers]
            wid = r.choice(fresh) if fresh and r.random() < 0.8 else r.choice(IDS)
            args = (wid, r.choice(POSITIONS), r.randint(1, 9))
        elif kind == "clock_in":
            clock += r.choice([0, 0, 1, 2, 3, 5])
            args = (pick(False, 0.75), clock)
        elif kind == "clock_out":
            clock += r.choice([0, 1, 2, 3, 5, 8])
            args = (pick(True, 0.8), clock)
        elif kind == "get_total_work_time":
            args = (registered(),)
        elif kind == "top_k_workers":
            args = (r.randint(1, len(IDS) + 1),)
        elif kind == "set_promotion":
            args = (pick(True, 0.4), r.choice(POSITIONS), r.randint(1, 20))
        elif kind == "calculate_pay":
            wid = registered(0.9)
            pts = points(wid)
            a, b = sorted(max(0, r.choice(pts) + r.choice([0, 0, -1, 1, r.randint(-6, 6)])) for _ in range(2))
            args = (wid, a, b)
        else:   # set_double_pay
            s = max(0, r.choice(points(registered()) + [r.randint(0, clock + 5)]) + r.choice([0, -2, 3]))
            args = (s, s + r.choice([0, 1, 2, 4, 6, 10, 20]))
        ops.append((kind, args))
        getattr(m, kind)(*args)
    return ops
