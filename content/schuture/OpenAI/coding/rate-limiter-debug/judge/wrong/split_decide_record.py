# expect: WA
# The decision and the record each run under the lock, but in two separate `with` blocks: between them another thread can decide on the same, not yet updated, windows.
import threading
import time
from bisect import bisect_left
from collections import defaultdict, deque
from functools import wraps


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class RateLimiter:
    def __init__(self, minute_rule, hour_rule, day_rule, clock=time.time, checkpoint=lambda: None):
        self.minute_rule = minute_rule
        self.hour_rule = hour_rule
        self.day_rule = day_rule
        self.clock = clock
        self.checkpoint = checkpoint
        self.windows = defaultdict(lambda: (deque(), deque(), deque()))
        self.lock = threading.Lock()

    def _fits(self, window, now, max_requests, period):
        while window and now - window[0] > period:
            window.popleft()
        return len(window) < max_requests

    def should_allow_request(self, user_id):
        now = self.clock()
        rules = (self.minute_rule, self.hour_rule, self.day_rule)
        with self.lock:
            windows = self.windows[user_id]
            ok = all([self._fits(win, now, *rule) for win, rule in zip(windows, rules)])
            self.checkpoint()
        if not ok:
            return False
        with self.lock:
            for win in windows:
                win.append(now)
        return True


def rate_limited(limiter):
    def decorator(view):
        @wraps(view)
        def wrapped(user_id, *args, **kwargs):
            if not limiter.should_allow_request(user_id):
                return {"status": 429, "body": {"error": "rate limit exceeded"}}
            return view(user_id, *args, **kwargs)
        return wrapped
    return decorator
