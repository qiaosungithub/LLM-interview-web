# expect: WA
# Wrong for Part 1/2: a snapshot copies the follow edges but keeps a reference to the live user set, so a user
# registered after the snapshot counts as one of its users (False / [] instead of ValueError).
from bisect import bisect_right
from heapq import nsmallest


class SocialNetwork:
    def __init__(self):
        self._order = []                 # users in registration order
        self._users = set()
        self._out = {}                   # user -> frozenset of followees (replaced, never mutated)

    def add_user(self, user_id):
        if user_id in self._out:
            raise ValueError("duplicate user %r" % (user_id,))
        self._order.append(user_id)
        self._users.add(user_id)
        self._out[user_id] = frozenset()

    def follow(self, follower, followee):
        for u in (follower, followee):
            if u not in self._out:
                raise ValueError("unknown user %r" % (u,))
        if follower != followee and followee not in self._out[follower]:
            self._out[follower] = self._out[follower] | {followee}

    def create_snapshot(self):
        # frozensets are immutable and replaced on write, so a shallow dict copy is a true freeze
        return Snapshot(dict(self._out), self._users)


class Snapshot:
    def __init__(self, out, users):
        self._out = out
        self._users = users
        self._in = None

    def _need(self, *ids):
        bad = [u for u in ids if u not in self._users]
        if bad:
            raise ValueError("unknown user %r" % (bad[0],))

    def is_following(self, follower, followee):
        self._need(follower, followee)
        return followee in self._out.get(follower, ())

    def get_following(self, user_id):
        self._need(user_id)
        return sorted(self._out.get(user_id, ()))

    def get_followers(self, user_id):
        self._need(user_id)
        if self._in is None:
            rev = {u: [] for u in self._out}
            for a, bs in self._out.items():
                for b in bs:
                    rev[b].append(a)
            self._in = rev
        return sorted(self._in.get(user_id, ()))

    def recommend(self, user_id, k):
        self._need(user_id)
        mine = self._out.get(user_id, frozenset())
        score = {}
        for f in mine:
            for c in self._out[f]:
                if c == user_id or c in mine:
                    continue
                score[c] = score.get(c, 0) + 1
        if k <= 0:
            return []
        return nsmallest(k, score, key=lambda c: (-score[c], c))


class FollowTimeline:
    def __init__(self):
        self._times = {}    # pair -> [t, ...] where the state changed
        self._states = {}   # pair -> [state after that change, ...]

    def _set(self, pair, t, on):
        if pair[0] == pair[1]:
            return
        times = self._times.setdefault(pair, [])
        states = self._states.setdefault(pair, [])
        current = states[-1] if states else False
        if current == on:
            return
        if times and times[-1] == t:      # several calls at one t: the last one decides
            times.pop()
            states.pop()
            if (states[-1] if states else False) == on:
                return
        times.append(t)
        states.append(on)

    def follow(self, follower, followee, t):
        self._set((follower, followee), t, True)

    def unfollow(self, follower, followee, t):
        self._set((follower, followee), t, False)

    def is_following(self, follower, followee, t):
        times = self._times.get((follower, followee))
        if not times:
            return False
        i = bisect_right(times, t)
        return self._states[(follower, followee)][i - 1] if i else False
