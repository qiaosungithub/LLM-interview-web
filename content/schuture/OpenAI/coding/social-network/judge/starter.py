class SocialNetwork:
    def __init__(self):
        """Part 1: the live, mutable set of users and follow edges."""

    def add_user(self, user_id: str) -> None:
        """Registers a new user. Raises ValueError if user_id already exists."""
        raise NotImplementedError

    def follow(self, follower: str, followee: str) -> None:
        """follower starts following followee. Raises ValueError if follower or followee is not a registered user."""
        raise NotImplementedError

    def create_snapshot(self) -> "Snapshot":
        """Freezes the current users and follow edges into a Snapshot."""
        raise NotImplementedError


class Snapshot:
    def is_following(self, follower: str, followee: str) -> bool:
        """Raises ValueError if follower or followee was not a user of this snapshot."""
        raise NotImplementedError

    def get_following(self, user_id: str) -> list[str]:
        """Part 2: users user_id follows, as of this snapshot, sorted by user id ascending."""
        raise NotImplementedError

    def get_followers(self, user_id: str) -> list[str]:
        """Part 2: users who follow user_id, as of this snapshot, sorted by user id ascending."""
        raise NotImplementedError

    def recommend(self, user_id: str, k: int) -> list[str]:
        """Part 3: top-k two-hop recommendations, score descending, ties by user id ascending."""
        raise NotImplementedError


class FollowTimeline:
    def __init__(self):
        """Part 4: follow edges that can be queried at any past time."""

    def follow(self, follower: str, followee: str, t: int) -> None:
        """follower starts following followee at time t."""
        raise NotImplementedError

    def unfollow(self, follower: str, followee: str, t: int) -> None:
        """follower stops following followee at time t."""
        raise NotImplementedError

    def is_following(self, follower: str, followee: str, t: int) -> bool:
        """Whether follower was following followee at time t."""
        raise NotImplementedError
