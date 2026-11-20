from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    title: str
    priority: int
    created_at: int


class TaskManager:
    def __init__(self) -> None:
        """Creates an empty task manager."""

    def add_task(self, timestamp: int, user: str, title: str, priority: int) -> str:
        """Level 1: creates a task owned by user at timestamp and returns its id ("task-1", "task-2", ...).
        (Level 3 adds a last parameter ttl: int | None = None.)"""
        raise NotImplementedError

    def update_task(self, timestamp: int, user: str, task_id: str, title: str, priority: int) -> bool:
        """Level 1: replaces task_id's title and priority; False, changing nothing, if it does not exist, is not user's, or is not active."""
        raise NotImplementedError

    def complete_task(self, timestamp: int, user: str, task_id: str) -> bool:
        """Level 1: marks task_id completed; False, changing nothing, if it does not exist, is not user's, or is already not active."""
        raise NotImplementedError

    def get_task(self, timestamp: int, user: str, task_id: str) -> Task | None:
        """Level 1: task_id as a Task if it exists, is owned by user and is active; None otherwise."""
        raise NotImplementedError

    def get_task_list(self, at_timestamp: int, user: str, min_priority: int | None = None) -> list[Task]:
        """Level 2: user's active tasks with priority >= min_priority, by priority descending, ties in creation order."""
        raise NotImplementedError
