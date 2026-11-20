# expect: WA
# Bugs 1, 2, 4 fixed, bug 3 left: the minute and hour rules still share one list and each records as soon as it passes, so an admitted request counts twice.
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
        self.recent = defaultdict(deque)    # user_id -> timestamps
        self.days = defaultdict(deque)      # user_id -> timestamps of the last day
        self._lock = threading.Lock()

    def _within_window(self, user_id, now, max_requests, period):
        timestamps = self.recent[user_id]
        while timestamps and now - timestamps[0] > period:
            timestamps.popleft()
        would_pass = len(timestamps) < max_requests
        self.checkpoint()
        if not would_pass:
            return False
        timestamps.append(now)
        return True

    def should_allow_request(self, user_id):
        with self._lock:
            now = self.clock()
            if not self._within_window(user_id, now, *self.minute_rule):
                return False
            if not self._within_window(user_id, now, *self.hour_rule):
                return False
            max_requests, period = self.day_rule
            days = self.days[user_id]
            while days and now - days[0] > period:
                days.popleft()
            if len(days) >= max_requests:
                return False
            days.append(now)
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
