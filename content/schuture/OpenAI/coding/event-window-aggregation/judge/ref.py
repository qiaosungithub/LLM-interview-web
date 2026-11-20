"""Models of the three trackers written straight from the statement (they keep every event, so they are slow and
use unbounded memory, which is fine for a model), and generators of valid call sequences.

A sequence is a list of (method, args). The models have the same method names as the statement's classes."""
from bisect import bisect_left, bisect_right

NEG_INF = float("-inf")


class CounterModel:
    """Part 1. Timestamps arrive in non-decreasing order, so each chat's list stays sorted and bisect counts
    exactly the timestamps in [T - window, T]."""

    def __init__(self, window):
        self.window = window
        self.T = None
        self.times = {}                 # (user_id, chat_id) -> every timestamp recorded, in order

    def record_event(self, user_id, chat_id, timestamp):
        self.T = timestamp if self.T is None else max(self.T, timestamp)
        self.times.setdefault((user_id, chat_id), []).append(timestamp)

    def recent_count(self, user_id, chat_id):
        times = self.times.get((user_id, chat_id), [])
        if self.T is None:
            return 0
        return bisect_right(times, self.T) - bisect_left(times, self.T - self.window)


class ActivityModel:
    """Part 2. With calls in timestamp order, a chat is active iff the last event recorded for it is a ping inside
    [T - window, T]: a later close cancels every earlier ping, and an earlier ping is no newer than the last one."""

    def __init__(self, window):
        self.window = window
        self.T = None
        self.last = {}                  # user_id -> {chat_id: (event_type, timestamp) of its last recorded event}

    def record(self, user_id, chat_id, event_type, timestamp):
        self.T = timestamp if self.T is None else max(self.T, timestamp)
        self.last.setdefault(user_id, {})[chat_id] = (event_type, timestamp)

    def active_sessions(self, user_id):
        if self.T is None:
            return 0
        return sum(1 for kind, ts in self.last.get(user_id, {}).values()
                   if kind == "ping" and ts >= self.T - self.window)


class AsOfModel:
    """Part 3, rule 2 applied to every event ever recorded. Rule 3 says the records a tracker drops cannot change
    any answer, so keeping them gives the same answers."""

    def __init__(self, window):
        self.window = window
        self.chats = {}                 # user_id -> {chat_id: [last_ping, last_close]}

    def record(self, user_id, chat_id, event_type, timestamp):
        pair = self.chats.setdefault(user_id, {}).setdefault(chat_id, [NEG_INF, NEG_INF])
        i = 0 if event_type == "ping" else 1
        pair[i] = max(pair[i], timestamp)

    def active_sessions(self, user_id, now):
        return sum(1 for last_ping, last_close in self.chats.get(user_id, {}).values()
                   if last_ping >= now - self.window and last_close <= last_ping)


MODELS = {"RecentEventCounter": CounterModel, "SessionActivityTracker": ActivityModel,
          "OutOfOrderSessionTracker": AsOfModel}
QUERY = {"RecentEventCounter": "recent_count", "SessionActivityTracker": "active_sessions",
         "OutOfOrderSessionTracker": "active_sessions"}


# ---------------------------------------------------------------- valid random sequences

def _jump(r, window):
    """A timestamp step: often 0 (ties), sometimes landing exactly on or just past the window's edge."""
    return r.choice([0, 0, 1, 1, 2, max(0, window - 1), window, window + 1, 3 * window])


def p1_ops(r, steps, window, users, chats, start=0):
    ops, ts = [], start
    for _ in range(steps):
        if r.random() < 0.5:
            ts += _jump(r, window)
            ops.append(("record_event", (r.choice(users), r.choice(chats), ts)))
        else:
            ops.append(("recent_count", (r.choice(users), r.choice(chats))))
    return ops


def p2_ops(r, steps, window, users, chats, start=0):
    ops, ts = [], start
    for _ in range(steps):
        if r.random() < 0.6:
            ts += _jump(r, window)
            kind = "close" if r.random() < 0.35 else "ping"
            ops.append(("record", (r.choice(users), r.choice(chats), kind, ts)))
        else:
            ops.append(("active_sessions", (r.choice(users),)))
    return ops


def p3_ops(r, steps, window, users, chats, start=0):
    """Records may be late (some older than C - window, so dropped), on time, or ahead of the last `now`; every
    `now` is at least C, the largest `now` or timestamp so far, as rule 1 requires."""
    ops, clock = [], start
    for _ in range(steps):
        if r.random() < 0.6:
            ts = max(0, clock + r.randint(-(window + 3), 3))
            kind = "close" if r.random() < 0.4 else "ping"
            ops.append(("record", (r.choice(users), r.choice(chats), kind, ts)))
            clock = max(clock, ts)
        else:
            clock += r.choice([0, 0, 1, 2, window, window + 1])
            ops.append(("active_sessions", (r.choice(users), clock)))
    return ops
