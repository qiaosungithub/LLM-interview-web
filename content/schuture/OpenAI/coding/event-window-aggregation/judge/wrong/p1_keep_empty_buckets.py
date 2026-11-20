# expect: WA
# Part 1 never deletes a chat whose deque has emptied, so state grows with every chat ever seen.
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
            bucket.popleft()               # an empty deque just reads as count 0

    def recent_count(self, user_id, chat_id):
        return len(self._per_key.get((user_id, chat_id), ()))

class SessionActivityTracker:
    def __init__(self, window):
        self.window = window
        self._max_ts = float('-inf')
        self._last_ping = {}         # (user_id, chat_id) -> timestamp; present iff currently active
        self._ping_queue = deque()   # (timestamp, user_id, chat_id), in arrival order
        self._active_count = {}      # user_id -> number of active chats; absent means 0

    def record(self, user_id, chat_id, event_type, timestamp):
        self._max_ts = max(self._max_ts, timestamp)
        key = (user_id, chat_id)
        if event_type == 'ping':
            if key not in self._last_ping:                    # NOTE: only a *new* activation bumps the count
                self._active_count[user_id] = self._active_count.get(user_id, 0) + 1
            self._last_ping[key] = timestamp
            self._ping_queue.append((timestamp, user_id, chat_id))
        else:  # 'close'
            if self._last_ping.pop(key, None) is not None:   # NOTE: in order, a close always supersedes
                self._active_count[user_id] -= 1               # a ping already recorded for this key
                if self._active_count[user_id] == 0:
                    del self._active_count[user_id]           # NOTE: the invariant Part 2 asks for
        threshold = self._max_ts - self.window
        while self._ping_queue and self._ping_queue[0][0] < threshold:
            ts, uid, cid = self._ping_queue.popleft()
            k = (uid, cid)
            if self._last_ping.get(k) == ts:      # NOTE: else stale -- a later ping or close replaced it
                del self._last_ping[k]
                self._active_count[uid] -= 1
                if self._active_count[uid] == 0:
                    del self._active_count[uid]

    def active_sessions(self, user_id):
        return self._active_count.get(user_id, 0)

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

