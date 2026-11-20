# expect: WA
# Part 1 measures each chat's window from that chat's own latest timestamp, not from T over all chats.
import heapq
from collections import deque


class RecentEventCounter:
    def __init__(self, window):
        self.w = window
        self.per = {}           # (u, c) -> deque of timestamps within that chat's own window

    def record_event(self, user_id, chat_id, timestamp):
        k = (user_id, chat_id)
        d = self.per.setdefault(k, deque())
        d.append(timestamp)
        while d[0] < timestamp - self.w:
            d.popleft()

    def recent_count(self, user_id, chat_id):
        return len(self.per.get((user_id, chat_id), ()))


class SessionActivityTracker:
    """Each live ping gets a serial number; the chat remembers the serial of its current active ping."""

    def __init__(self, window):
        self.w = window
        self.T = None
        self.serial = 0
        self.live = {}          # (u, c) -> serial of the ping that makes it active
        self.queue = deque()    # (ts, (u, c), serial)
        self.active = {}        # u -> count > 0

    def _dec(self, u):
        n = self.active[u] - 1
        if n:
            self.active[u] = n
        else:
            del self.active[u]

    def record(self, user_id, chat_id, event_type, timestamp):
        k = (user_id, chat_id)
        self.T = timestamp if self.T is None or timestamp > self.T else self.T
        if event_type == "ping":
            self.serial += 1
            if k not in self.live:
                self.active[user_id] = self.active.get(user_id, 0) + 1
            self.live[k] = self.serial
            self.queue.append((timestamp, k, self.serial))
        else:
            if self.live.pop(k, None) is not None:
                self._dec(user_id)
        lo = self.T - self.w
        q = self.queue
        while q and q[0][0] < lo:
            _, old, s = q.popleft()
            if self.live.get(old) == s:
                del self.live[old]
                self._dec(old[0])

    def active_sessions(self, user_id):
        return self.active.get(user_id, 0)


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
