# expect: WA
# Bug 4 'fixed' with a Lock created inside should_allow_request: every call gets its own lock, which excludes nobody.
import time
import threading
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
        self.windows = defaultdict(lambda: {"minute": deque(), "hour": deque(), "day": deque()})

    def _fits(self, window, now, max_requests, period):
        while window and now - window[0] > period:
            window.popleft()
        return len(window) < max_requests

    def should_allow_request(self, user_id):
        with threading.Lock():
            now = self.clock()
            w = self.windows[user_id]
            fits_minute = self._fits(w["minute"], now, *self.minute_rule)
            fits_hour = self._fits(w["hour"], now, *self.hour_rule)
            fits_day = self._fits(w["day"], now, *self.day_rule)
            self.checkpoint()
            if not (fits_minute and fits_hour and fits_day):
                return False
            w["minute"].append(now)
            w["hour"].append(now)
            w["day"].append(now)
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


if __name__ == "__main__":
    limiter = RateLimiter(minute_rule=(5, 60), hour_rule=(20, 3600), day_rule=(50, 86400))

    @rate_limited(limiter)
    def post_message(user_id, text):
        return {"status": 200, "body": {"echo": text}}

    for _ in range(3):
        assert post_message("alice", "hi")["status"] == 200
    print("smoke test passed")
