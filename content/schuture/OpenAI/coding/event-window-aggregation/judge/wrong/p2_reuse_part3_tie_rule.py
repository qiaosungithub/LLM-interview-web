# expect: WA
# Part 2 reuses the Part 3 tracker with now = T, so a same-timestamp tie goes to ping instead of to the later call.
from collections import deque


class RecentEventCounter:
    def __init__(self, window):
        self.window = window
        self._max_ts = float('-inf')   # T: the largest timestamp recorded so far
        self._order = deque()          # (timestamp, key), in arrival order == timestamp order here
        self._per_key = {}             # key -> deque of timestamps, only while one is still in the window

    def record_event(self, user_id, chat_id, timestamp):
        key = (user_id, chat_id)
        self._max_ts = max(self._max_ts, timestamp)
        self._order.append((timestamp, key))
        self._per_key.setdefault(key, deque()).append(timestamp)
        threshold = self._max_ts - self.window
        while self._order and self._order[0][0] < threshold:      # NOTE: '<' -- the window's left edge is inclusive
            ts, k = self._order.popleft()
            bucket = self._per_key[k]
            bucket.popleft()               # always the front of this key's own deque too -- both share one order
            if not bucket:
                del self._per_key[k]       # NOTE: this delete is the whole memory guarantee -- skip it and it leaks

    def recent_count(self, user_id, chat_id):
        return len(self._per_key.get((user_id, chat_id), ()))

import heapq

NEG_INF = float('-inf')


class OutOfOrderSessionTracker:
    def __init__(self, window):
        self.window = window
        self._clock = NEG_INF   # C: the largest `now` or `timestamp` seen so far
        self._chat = {}         # (user_id, chat_id) -> (last_ping, last_close, counted)
        self._active_count = {}
        self._heap = []         # (forget_at, user_id, chat_id, last_ping, last_close); may go stale

    def record(self, user_id, chat_id, event_type, timestamp):
        if timestamp < self._clock - self.window:
            return                    # NOTE: rule 3 -- drop it before it costs a heap push
        key = (user_id, chat_id)
        last_ping, last_close, counted = self._chat.get(key, (NEG_INF, NEG_INF, False))
        if event_type == 'ping':
            last_ping = max(last_ping, timestamp)
        else:
            last_close = max(last_close, timestamp)
        ping_dominant = last_close <= last_ping   # NOTE: ties go to ping, and unlike Part 2 on the two
        if ping_dominant != counted:              # timestamps alone, never on the order they arrived in
            counted = ping_dominant
            self._active_count[user_id] = self._active_count.get(user_id, 0) + (1 if counted else -1)
            if self._active_count[user_id] == 0:
                del self._active_count[user_id]
        self._chat[key] = (last_ping, last_close, counted)
        forget_at = (last_ping if ping_dominant else last_close) + self.window
        heapq.heappush(self._heap, (forget_at, user_id, chat_id, last_ping, last_close))
        self._advance(timestamp)      # NOTE: ingest evicts too -- without this a query-free stream leaks

    def _advance(self, t):
        self._clock = max(self._clock, t)
        while self._heap and self._heap[0][0] < self._clock:
            _, user_id, chat_id, ping_v, close_v = heapq.heappop(self._heap)
            key = (user_id, chat_id)
            cur = self._chat.get(key)
            if cur is None or cur[0] != ping_v or cur[1] != close_v:
                continue                                 # stale: a later record() already replaced this
            if cur[2]:                                   # was still counted -- its ping just aged out
                self._active_count[user_id] -= 1
                if self._active_count[user_id] == 0:
                    del self._active_count[user_id]
            del self._chat[key]                          # safe either way, per the derivation above

    def active_sessions(self, user_id, now):
        self._advance(now)
        return self._active_count.get(user_id, 0)


class SessionActivityTracker:
    def __init__(self, window):
        self._inner = OutOfOrderSessionTracker(window)
        self._max_ts = NEG_INF

    def record(self, user_id, chat_id, event_type, timestamp):
        self._max_ts = max(self._max_ts, timestamp)
        self._inner.record(user_id, chat_id, event_type, timestamp)

    def active_sessions(self, user_id):
        return self._inner.active_sessions(user_id, self._max_ts)   # in order, T plays the role of now

