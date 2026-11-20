# expect: WA
# Bugs 2-4 fixed, bug 1 left: the eviction loop rebinds a local name (timestamps = timestamps[1:]) and the record goes to that name, so once a rule has an expired entry its records land on a throwaway copy and it stops counting.
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
        self.windows = defaultdict(lambda: {"minute": [], "hour": [], "day": []})
        self._lock = threading.Lock()

    def _evict(self, timestamps, now, period):
        while timestamps and now - timestamps[0] > period:
            timestamps = timestamps[1:]
        return timestamps

    def should_allow_request(self, user_id):
        with self._lock:
            now = self.clock()
            w = self.windows[user_id]
            rules = {"minute": self.minute_rule, "hour": self.hour_rule, "day": self.day_rule}
            kept = {name: self._evict(w[name], now, rules[name][1]) for name in rules}
            fits = all(len(kept[name]) < rules[name][0] for name in rules)
            self.checkpoint()
            if not fits:
                return False
            for timestamps in kept.values():
                timestamps.append(now)
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
