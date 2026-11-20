"""A literal model of the statement, used for the expected values of the random tests, and the random
operation-sequence generator.

The model keeps only the log of successful add/update/complete calls, in call order. The state at any
instant is rebuilt from scratch by replaying the calls with timestamp <= that instant, and Level 3's
expiry rule is applied at the instant asked about. Nothing is indexed or cached."""


class TaskManager:
    def __init__(self):
        self.log = []       # successful mutating calls in call order: (timestamp, kind, task_id, data)
        self.added = 0

    def state(self, at):
        """task_id -> record, as the calls with timestamp <= at left it."""
        tasks = {}
        for ts, kind, tid, data in self.log:
            if ts > at:
                continue
            if kind == "add":
                tasks[tid] = dict(data, done=False)
            elif kind == "update":
                tasks[tid]["title"], tasks[tid]["priority"] = data
            else:
                tasks[tid]["done"] = True
        return tasks

    @staticmethod
    def active(rec, at):
        return not rec["done"] and (rec["ttl"] is None or at < rec["created_at"] + rec["ttl"])

    def find(self, at, user, tid):
        rec = self.state(at).get(tid)
        return rec if rec is not None and rec["user"] == user and self.active(rec, at) else None

    @staticmethod
    def view(tid, rec):
        return (tid, rec["title"], rec["priority"], rec["created_at"])

    def add_task(self, timestamp, user, title, priority, ttl=None):
        self.added += 1
        tid = f"task-{self.added}"
        self.log.append((timestamp, "add", tid, {"user": user, "title": title, "priority": priority,
                                                  "created_at": timestamp, "ttl": ttl}))
        return tid

    def update_task(self, timestamp, user, task_id, title, priority):
        if self.find(timestamp, user, task_id) is None:
            return False
        self.log.append((timestamp, "update", task_id, (title, priority)))
        return True

    def complete_task(self, timestamp, user, task_id):
        if self.find(timestamp, user, task_id) is None:
            return False
        self.log.append((timestamp, "complete", task_id, None))
        return True

    def get_task(self, timestamp, user, task_id):
        rec = self.find(timestamp, user, task_id)
        return None if rec is None else self.view(task_id, rec)

    def get_task_list(self, at_timestamp, user, min_priority=None):
        tasks = self.state(at_timestamp)
        hits = [tid for tid, rec in tasks.items()
                if rec["user"] == user and self.active(rec, at_timestamp)
                and (min_priority is None or rec["priority"] >= min_priority)]
        # tasks is in creation order (the order of the "add" entries), and sorted() is stable
        hits = sorted(hits, key=lambda tid: -tasks[tid]["priority"])
        return [self.view(tid, tasks[tid]) for tid in hits]


# ---------------------------------------------------------------- random operation sequences

KINDS = {   # operation -> weight
    1: {"add_task": 4, "update_task": 4, "complete_task": 3, "get_task": 4},
    2: {"add_task": 5, "update_task": 4, "complete_task": 2, "get_task": 1, "get_task_list": 5},
    3: {"add_task": 5, "update_task": 3, "complete_task": 2, "get_task": 3, "get_task_list": 4},
    4: {"add_task": 5, "update_task": 4, "complete_task": 2, "get_task": 1, "get_task_list": 7},
}
USERS = ["alice", "bob", "carol"]
TITLES = ["write report", "review PR", "file expenses", "renew badge", "", "写周报"]


def random_ops(r, level, n_ops):
    """A sequence of (method, args, kwargs) calls. Timestamps follow the statement's guarantees: every call
    is at or after every earlier one, except Level 4's get_task_list, which may also ask about the past.
    The draws often hit the interesting cases: equal priorities, equal timestamps, wrong owners, the
    exact instant a task expires or was changed, and instants before a user's first task."""
    m = TaskManager()
    kinds, weights = zip(*KINDS[level].items())
    clock = r.randint(0, 3)
    instants = [clock]          # timestamps of the calls so far, for past queries
    ops = []

    def owner(tid):
        return next((d["user"] for ts, k, t, d in m.log if k == "add" and t == tid), None)

    def pick_tid():
        p = r.random()
        if m.added and p < 0.85:
            return f"task-{r.randint(1, m.added)}"
        return r.choice(["task-0", f"task-{m.added + 1}", "task-99"])

    def pick_user(tid):
        u = owner(tid)
        return u if u is not None and r.random() < 0.75 else r.choice(USERS)

    def deadlines():
        return [d["created_at"] + d["ttl"] for ts, k, t, d in m.log if k == "add" and d["ttl"] is not None]

    def advance():
        """A timestamp at or after every call so far, often the exact instant some task expires."""
        nonlocal clock
        later = [d for d in deadlines() if d >= clock]
        if level >= 3 and later and r.random() < 0.3:
            clock = r.choice(later) + r.choice([0, 0, -1, 1]) if r.random() < 0.8 else r.choice(later)
            clock = max(clock, instants[-1])
        else:
            clock += r.choice([0, 0, 1, 1, 2, 3])
        instants.append(clock)
        return clock

    for _ in range(n_ops):
        kind = r.choices(kinds, weights)[0]
        if m.added == 0 and r.random() < 0.8:
            kind = "add_task"
        args, kwargs = None, {}
        if kind == "add_task":
            args = (advance(), r.choice(USERS), r.choice(TITLES), r.randint(-1, 5))
            if level >= 3 and r.random() < 0.55:
                kwargs = {"ttl": r.choice([1, 1, 2, 3, 4, 6, None])}
        elif kind == "update_task":
            tid = pick_tid()
            args = (advance(), pick_user(tid), tid, r.choice(TITLES), r.randint(-1, 5))
        elif kind == "complete_task":
            tid = pick_tid()
            args = (advance(), pick_user(tid), tid)
        elif kind == "get_task":
            tid = pick_tid()
            args = (advance(), pick_user(tid), tid)
        else:   # get_task_list
            if level == 4 and r.random() < 0.65:
                choices = instants + [d for d in deadlines() if d <= clock] + [instants[0] - 1]
                at = r.choice(choices) + r.choice([0, 0, 0, -1, 1])
                at = max(0, min(at, clock))         # the past or the present; never ahead of the clock
            else:
                at = advance()
            args = (at, pick_user(f"task-{r.randint(1, m.added)}") if m.added else r.choice(USERS))
            if r.random() < 0.45:
                kwargs = {"min_priority": r.randint(-1, 5)}
        ops.append((kind, args, kwargs))
        getattr(m, kind)(*args, **kwargs)
    return ops
