"""A literal model of the statement, used for the expected answers of the random tests, and the random call
generator. Every registered worker keeps the raw list of the values it reported (merged lists included); counts
are recounted from those lists on every query, and nothing is cached."""


class ModeTracker:
    def __init__(self):
        self.reports = {}           # worker id -> every value it reported, one item per reported value

    # Level 1
    def add_worker(self, worker_id):
        if worker_id in self.reports:
            return False
        self.reports[worker_id] = []
        return True

    def report(self, worker_id, values):
        if worker_id not in self.reports:
            return False
        self.reports[worker_id].extend(values)
        return True

    def get_count(self, value):
        return sum(values.count(value) for values in self.reports.values())

    def get_worker_count(self, worker_id, value):
        if worker_id not in self.reports:
            return None
        return self.reports[worker_id].count(value)

    # Level 2
    def _ranking(self):
        values = sorted({v for reported in self.reports.values() for v in reported})
        values.sort(key=self.get_count, reverse=True)       # stable: equal counts keep ascending values
        return values

    def get_mode(self):
        ranking = self._ranking()
        return ranking[0] if ranking else None

    def top_values(self, k):
        return ["%d(%d)" % (v, self.get_count(v)) for v in self._ranking()[:k]]

    # Level 3
    def remove_worker(self, worker_id):
        if worker_id not in self.reports:
            return None
        return len(self.reports.pop(worker_id))

    # Level 4
    def merge_workers(self, target_id, source_id):
        if target_id == source_id or target_id not in self.reports or source_id not in self.reports:
            return False
        self.reports[target_id].extend(self.reports.pop(source_id))
        return True


# ---------------------------------------------------------------- random call sequences

WORKERS = ["w1", "w2", "w3", "W1"]
VALUES = [-4, -1, 0, 1, 2, 2, 3, 3, 9, 10, 10, 1000000000, -1000000000]

KINDS = {   # method -> weight; each level also uses every earlier level's methods
    1: {"add_worker": 3, "report": 6, "get_count": 3, "get_worker_count": 3},
    2: {"add_worker": 2, "report": 6, "get_count": 1, "get_worker_count": 1, "get_mode": 4, "top_values": 3},
    3: {"add_worker": 3, "report": 6, "get_count": 1, "get_worker_count": 2, "get_mode": 4, "top_values": 3,
        "remove_worker": 3},
    4: {"add_worker": 3, "report": 6, "get_count": 1, "get_worker_count": 2, "get_mode": 3, "top_values": 2,
        "remove_worker": 2, "merge_workers": 3},
}


def random_ops(r, level, n_calls):
    """A list of (method, args) for one tracker, drawn while a model follows along so that the draws hit the
    interesting cases often: values already reported (ties, counts that drop to 0), duplicates inside one list,
    known and unknown worker ids, ids removed or merged away and registered again, and merges of an id with
    itself."""
    m = ModeTracker()
    kinds, weights = zip(*KINDS[level].items())
    out = []

    def worker(p_known=0.85):
        if m.reports and r.random() < p_known:
            return r.choice(sorted(m.reports))
        return r.choice(WORKERS)

    def value():
        seen = sorted({v for reported in m.reports.values() for v in reported})
        if seen and r.random() < 0.5:
            return r.choice(seen)
        return r.choice(VALUES)

    for _ in range(n_calls):
        kind = r.choices(kinds, weights)[0]
        if kind in ("report", "remove_worker", "merge_workers") and not m.reports and r.random() < 0.8:
            kind = "add_worker"
        if kind == "add_worker":
            args = (worker(0.2),)
        elif kind == "report":
            pool = [value() for _ in range(r.randint(1, 3))]
            length = r.choice([1, 1, 2, 3, 4, 6, 30]) if r.random() < 0.97 else 200
            args = (worker(), [r.choice(pool) for _ in range(length)])
        elif kind == "get_count":
            args = (value(),)
        elif kind == "get_worker_count":
            args = (worker(), value())
        elif kind == "get_mode":
            args = ()
        elif kind == "top_values":
            args = (r.choice([1, 1, 2, 3, 5, 100]),)
        elif kind == "remove_worker":
            args = (worker(),)
        else:   # merge_workers
            source = worker()
            args = (source, source) if r.random() < 0.08 else (worker(), source)
        out.append((kind, args))
        getattr(m, kind)(*args)             # report() copies the list's items, so args stays as drawn
    return out
