"""Plain models of the statement, used to compute expected answers, plus random operation generators.

An operation is a tuple (name, obj, method, args): it calls obj.method(*args), or method(*args) when obj is None
(a constructor), and stores the result under name when name is not None.
"""


class SocialNetwork:
    def __init__(self):
        self.users, self.edges = [], set()

    def add_user(self, user_id):
        if user_id in self.users:
            raise ValueError(user_id)
        self.users.append(user_id)

    def follow(self, follower, followee):
        if follower not in self.users or followee not in self.users:
            raise ValueError("unknown user")
        if follower != followee:
            self.edges.add((follower, followee))

    def create_snapshot(self):
        return Snapshot(self.users, self.edges)


class Snapshot:
    """Holds a copy of the users and of the (follower, followee) pairs; every query scans them."""

    def __init__(self, users, edges):
        self.users, self.edges = tuple(users), frozenset(edges)

    def _known(self, *ids):
        for u in ids:
            if u not in self.users:
                raise ValueError(u)

    def is_following(self, follower, followee):
        self._known(follower, followee)
        return (follower, followee) in self.edges

    def get_following(self, user_id):
        self._known(user_id)
        return sorted(b for a, b in self.edges if a == user_id)

    def get_followers(self, user_id):
        self._known(user_id)
        return sorted(a for a, b in self.edges if b == user_id)

    def scores(self, user_id):
        """candidate -> number of user_id's followees that follow it."""
        direct = {b for a, b in self.edges if a == user_id}
        out = {}
        for c in self.users:
            if c == user_id or c in direct:
                continue
            score = sum(1 for f in direct if (f, c) in self.edges)
            if score:
                out[c] = score
        return out

    def recommend(self, user_id, k):
        self._known(user_id)
        scores = self.scores(user_id)
        return sorted(scores, key=lambda c: (-scores[c], c))[:k]


class FollowTimeline:
    """Keeps every call; the state of a pair at t is set by the last call (in call order) on it with time <= t."""

    def __init__(self):
        self.log = []

    def follow(self, follower, followee, t):
        if follower != followee:
            self.log.append((follower, followee, t, True))

    def unfollow(self, follower, followee, t):
        if follower != followee:
            self.log.append((follower, followee, t, False))

    def is_following(self, follower, followee, t):
        state = False
        for a, b, when, on in self.log:
            if (a, b) == (follower, followee) and when <= t:
                state = on
        return state


CLASSES = {"SocialNetwork": SocialNetwork, "FollowTimeline": FollowTimeline}
QUERIES = {1: ["is_following"], 2: ["is_following", "get_following", "get_followers"],
           3: ["is_following", "get_following", "get_followers", "recommend"]}


def random_network_ops(r, part, length, names, ghost="zed"):
    """A random session on one SocialNetwork: registrations (some repeated), follows (some self, repeated or with an
    unregistered id), snapshots, and queries of the given part on any snapshot taken so far."""
    ops = [("net", None, "SocialNetwork", ())]
    users, snaps = [], []
    for _ in range(length):
        x = r.random()
        if x < 0.18 or len(users) < 2:
            pool = [u for u in names if u not in users]
            u = r.choice(pool) if pool and r.random() < 0.85 else r.choice(names)
            ops.append((None, "net", "add_user", (u,)))
            if u not in users:
                users.append(u)
        elif x < 0.55:
            a = r.choice(users) if r.random() < 0.95 else ghost
            b = r.choice(users) if r.random() < 0.95 else ghost
            if r.random() < 0.08:
                b = a
            if a == b == ghost:     # the statement does not say whether an unknown self-follow raises or is a no-op
                b = r.choice(users)
            ops.append((None, "net", "follow", (a, b)))
        elif x < 0.65 or not snaps:
            name = f"s{len(snaps) + 1}"
            snaps.append(name)
            ops.append((name, "net", "create_snapshot", ()))
        else:
            s = r.choice(snaps)
            method = r.choice(QUERIES[part])
            pick = lambda: r.choice(names) if r.random() < 0.15 else r.choice(users)
            if method == "is_following":
                args = (pick(), pick())
            elif method == "recommend":
                args = (pick(), r.choice([0, 1, 2, 3, 3, 5, 10]))
            else:
                args = (pick(),)
            ops.append((None, s, method, args))
    return ops


def random_timeline_ops(r, length, names, start=0):
    """follow/unfollow with non-decreasing t (often repeated), queries at any t."""
    ops = [("tl", None, "FollowTimeline", ())]
    t = start
    for _ in range(length):
        a, b = r.choice(names), r.choice(names)
        x = r.random()
        if x < 0.55:
            t += r.choice([0, 0, 1, 1, 2, 5])
            ops.append((None, "tl", "follow" if r.random() < 0.55 else "unfollow", (a, b, t)))
        else:
            ops.append((None, "tl", "is_following", (a, b, r.randint(start - 2, t + 3))))
    return ops
