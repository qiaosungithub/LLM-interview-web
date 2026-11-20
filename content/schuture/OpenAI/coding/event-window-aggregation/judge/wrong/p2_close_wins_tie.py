# expect: WA
# Part 2 compares the last ping and close timestamps, so a close wins a same-timestamp tie even when the ping was recorded later.
import heapq
from collections import deque


class RecentEventCounter:
    def __init__(self, window):
        self.w = window
        self.T = None
        self.times = deque()
        self.keys = deque()
        self.cnt = {}

    def record_event(self, user_id, chat_id, timestamp):
        k = (user_id, chat_id)
        self.T = timestamp if self.T is None or timestamp > self.T else self.T
        self.times.append(timestamp)
        self.keys.append(k)
        self.cnt[k] = self.cnt.get(k, 0) + 1
        lo = self.T - self.w
        while self.times[0] < lo:
            self.times.popleft()
            old = self.keys.popleft()
            c = self.cnt[old] - 1
            if c:
                self.cnt[old] = c
            else:
                del self.cnt[old]

    def recent_count(self, user_id, chat_id):
        return self.cnt.get((user_id, chat_id), 0)


class SessionActivityTracker:
    def __init__(self, window):
        self.w = window
        self.T = None
        self.lp = {}            # (u, c) -> last ping timestamp
        self.lc = {}            # (u, c) -> last close timestamp
        self.q = deque()        # (ts, (u, c)) for every event

    def record(self, user_id, chat_id, event_type, timestamp):
        k = (user_id, chat_id)
        self.T = timestamp if self.T is None else max(self.T, timestamp)
        (self.lp if event_type == "ping" else self.lc)[k] = timestamp
        self.q.append((timestamp, k))
        while self.q[0][0] < self.T - self.w:
            ts, o = self.q.popleft()
            if self.lp.get(o) == ts:
                del self.lp[o]
            if self.lc.get(o) == ts:
                del self.lc[o]

    def active_sessions(self, user_id):
        return sum(1 for (u, c), p in self.lp.items() if u == user_id and self.lc.get((u, c), -1) < p)


class OutOfOrderSessionTracker:
    """Kept chats always have max(last_ping, last_close) >= C - window, and every query sets C = now, so a kept
    chat is active exactly when last_close <= last_ping (with a ping seen)."""

    NONE = float("-inf")

    def __init__(self, window):
        self.w = window
        self.C = None
        self.state = {}         # (u, c) -> [last_ping, last_close]
        self.heap = []          # (latest ts of the chat when pushed, u, c)
        self.good = {}          # u -> number of kept chats with last_close <= last_ping

    @staticmethod
    def _ok(st):
        return st[0] != OutOfOrderSessionTracker.NONE and st[1] <= st[0]

    def _bump(self, u, d):
        n = self.good.get(u, 0) + d
        if n:
            self.good[u] = n
        else:
            self.good.pop(u, None)

    def _advance(self, t):
        if self.C is None or t > self.C:
            self.C = t
        lo = self.C - self.w
        h = self.heap
        while h and h[0][0] < lo:
            ts, u, c = heapq.heappop(h)
            st = self.state.get((u, c))
            if st is not None and max(st) == ts:
                del self.state[(u, c)]
                if self._ok(st):
                    self._bump(u, -1)

    def record(self, user_id, chat_id, event_type, timestamp):
        self._advance(timestamp)
        if timestamp < self.C - self.w:
            return
        k = (user_id, chat_id)
        st = self.state.get(k)
        if st is None:
            st = self.state[k] = [self.NONE, self.NONE]
            before, latest = False, self.NONE
        else:
            before, latest = self._ok(st), max(st)
        i = 0 if event_type == "ping" else 1
        if timestamp > st[i]:
            st[i] = timestamp
        after = self._ok(st)
        if after != before:
            self._bump(user_id, 1 if after else -1)
        if max(st) > latest:
            heapq.heappush(self.heap, (max(st), user_id, chat_id))

    def active_sessions(self, user_id, now):
        self._advance(now)
        return self.good.get(user_id, 0)
