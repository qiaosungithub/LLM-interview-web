class RecentEventCounter:
    def __init__(self, window: int):
        """Part 1: a tracker constructed once with a fixed positive integer window."""

    def record_event(self, user_id: str, chat_id: str, timestamp: int) -> None:
        """Logs one event for the chat (user_id, chat_id); timestamps arrive in non-decreasing order."""
        raise NotImplementedError

    def recent_count(self, user_id: str, chat_id: str) -> int:
        """How many events logged for that chat have a timestamp in [T - window, T], where T is the largest
        timestamp passed to record_event so far, over every user and chat."""
        raise NotImplementedError


class SessionActivityTracker:
    def __init__(self, window: int):
        """Part 2: a tracker constructed once with a fixed positive integer window."""

    def record(self, user_id: str, chat_id: str, event_type: str, timestamp: int) -> None:
        """Logs one "ping" or "close" event for the chat (user_id, chat_id); timestamps arrive in non-decreasing
        order."""
        raise NotImplementedError

    def active_sessions(self, user_id: str) -> int:
        """The number of that user's chats that are currently active."""
        raise NotImplementedError


class OutOfOrderSessionTracker:
    def __init__(self, window: int):
        """Part 3: a tracker constructed once with a fixed positive integer window."""

    def record(self, user_id: str, chat_id: str, event_type: str, timestamp: int) -> None:
        """Logs one "ping" or "close" event for the chat (user_id, chat_id); calls may come in any timestamp
        order."""
        raise NotImplementedError

    def active_sessions(self, user_id: str, now: int) -> int:
        """The number of that user's chats that are active as of now, per the statement's three rules."""
        raise NotImplementedError
