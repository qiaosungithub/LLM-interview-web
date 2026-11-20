# expect: WA
# Wrong for Part 1: self-follow is stored as an edge instead of being a no-op.
import bisect
from collections import Counter


class SocialNetwork:
    def __init__(self):
        self._users = set()
        self._following = {}

    def add_user(self, user_id):
        if user_id in self._users:
            raise ValueError(f"user already exists: {user_id!r}")
        self._users.add(user_id)
        self._following[user_id] = set()

    def follow(self, follower, followee):
        if follower not in self._users or followee not in self._users:
            raise ValueError("unknown user")
        self._following[follower].add(followee)   # NOTE: set.add is idempotent -> duplicate follow is a no-op too

    def create_snapshot(self):
        return Snapshot(self._users, self._following)


class Snapshot:
    def __init__(self, users, following):
        self._users = set(users)
        # NOTE: copy EACH inner set. following.copy() alone would copy the outer dict but keep the
        # same inner set objects, so a later SocialNetwork.follow() would still mutate this snapshot.
        self._following = {u: set(fs) for u, fs in following.items()}
        self._followers = {u: set() for u in self._users}   # reverse index, used starting get_followers below
        for u, followees in self._following.items():
            for v in followees:
                self._followers[v].add(u)

    def _check(self, user_id):
        if user_id not in self._users:
            raise ValueError(f"unknown user: {user_id!r}")

    def is_following(self, follower, followee):
        self._check(follower)
        self._check(followee)
        return followee in self._following[follower]


def get_following(self, user_id):
    self._check(user_id)
    return sorted(self._following[user_id])          # NOTE: sets have no reliable order; sort for determinism


def get_followers(self, user_id):
    self._check(user_id)
    return sorted(self._followers[user_id])


Snapshot.get_following = get_following      # attach both methods to the Snapshot class of Part 1
Snapshot.get_followers = get_followers


def recommend(self, user_id, k):
    self._check(user_id)
    direct = self._following[user_id]
    counts = Counter()
    for f in direct:
        for c in self._following[f]:
            if c != user_id and c not in direct:   # NOTE: exclude the user itself and already-followed accounts
                counts[c] += 1
    ranked = sorted(counts, key=lambda c: (-counts[c], c))   # NOTE: ties broken by user id ascending
    return ranked[:k]           # NOTE: k > len(ranked) just returns every candidate, no error or padding


Snapshot.recommend = recommend              # attach to the Snapshot class of Part 1


class FollowTimeline:
    def __init__(self):
        self._events = {}  # (follower, followee) -> toggle timestamps, always sorted by construction

    def _following_at(self, key, t):
        events = self._events.get(key)
        if not events:
            return False
        # NOTE: binary search over this one pair's own event list; the alternative -- keeping a
        # full graph snapshot for every past instant and deep-copying one on each event -- would
        # cost O(n + m) per event instead of the O(1) amortized appends follow()/unfollow() do
        i = bisect.bisect_right(events, t)   # NOTE: bisect_right, so a stop/start AT t is already in effect
        return i % 2 == 1                    # NOTE: relies on the log alternating start/stop, first event a start

    def follow(self, follower, followee, t):
        if follower == followee:
            return
        key = (follower, followee)
        if self._following_at(key, t):       # already following -> duplicate call, no-op
            return
        self._events.setdefault(key, []).append(t)

    def unfollow(self, follower, followee, t):
        if follower == followee:
            return
        key = (follower, followee)
        if not self._following_at(key, t):   # not following -> duplicate call, no-op
            return
        self._events[key].append(t)

    def is_following(self, follower, followee, t):
        return self._following_at((follower, followee), t)
