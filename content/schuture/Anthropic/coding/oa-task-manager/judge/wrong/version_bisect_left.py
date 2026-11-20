# expect: WA
# get_task_list finds the version with bisect_left, so an update made exactly at at_timestamp is missed (and at the creation instant the index -1 wraps to the newest version).
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    title: str
    priority: int
    created_at: int


class TaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}   # task_id -> mutable record
        self._next_id = 1

    def add_task(self, timestamp: int, user: str, title: str, priority: int) -> str:
        task_id = f"task-{self._next_id}"
        self._next_id += 1
        self._tasks[task_id] = {"user": user, "title": title, "priority": priority,
                                 "created_at": timestamp, "completed": False}
        return task_id

    def _active_record(self, user: str, task_id: str) -> dict | None:
        rec = self._tasks.get(task_id)
        if rec is None or rec["user"] != user or rec["completed"]:
            return None
        return rec

    def update_task(self, timestamp: int, user: str, task_id: str, title: str, priority: int) -> bool:
        rec = self._active_record(user, task_id)
        if rec is None:
            return False
        rec["title"], rec["priority"] = title, priority
        return True

    def complete_task(self, timestamp: int, user: str, task_id: str) -> bool:
        rec = self._active_record(user, task_id)
        if rec is None:
            return False
        rec["completed"] = True
        return True

    def get_task(self, timestamp: int, user: str, task_id: str) -> Task | None:
        rec = self._active_record(user, task_id)
        return None if rec is None else Task(task_id, rec["title"], rec["priority"], rec["created_at"])


def _creation_seq(task_id: str) -> int:
    return int(task_id.rsplit("-", 1)[1])


class TaskManager(TaskManager):
    def get_task_list(self, at_timestamp: int, user: str, min_priority: int | None = None) -> list[Task]:
        matches = [(task_id, rec) for task_id, rec in self._tasks.items()
                   if rec["user"] == user and not rec["completed"]
                   and (min_priority is None or rec["priority"] >= min_priority)]
        matches.sort(key=lambda pair: (-pair[1]["priority"], _creation_seq(pair[0])))
        return [Task(task_id, rec["title"], rec["priority"], rec["created_at"]) for task_id, rec in matches]


class TaskManager(TaskManager):
    def add_task(self, timestamp: int, user: str, title: str, priority: int, ttl: int | None = None) -> str:
        task_id = super().add_task(timestamp, user, title, priority)
        self._tasks[task_id]["ttl"] = ttl
        return task_id

    @staticmethod
    def _alive(rec: dict, at_timestamp: int) -> bool:
        if rec["completed"]:
            return False
        ttl = rec.get("ttl")
        return ttl is None or at_timestamp < rec["created_at"] + ttl

    def _record(self, user: str, task_id: str, at_timestamp: int) -> dict | None:
        rec = self._tasks.get(task_id)
        if rec is None or rec["user"] != user or not self._alive(rec, at_timestamp):
            return None
        return rec

    def update_task(self, timestamp: int, user: str, task_id: str, title: str, priority: int) -> bool:
        rec = self._record(user, task_id, timestamp)
        if rec is None:
            return False
        rec["title"], rec["priority"] = title, priority
        return True

    def complete_task(self, timestamp: int, user: str, task_id: str) -> bool:
        rec = self._record(user, task_id, timestamp)
        if rec is None:
            return False
        rec["completed"] = True
        return True

    def get_task(self, timestamp: int, user: str, task_id: str) -> Task | None:
        rec = self._record(user, task_id, timestamp)
        return None if rec is None else Task(task_id, rec["title"], rec["priority"], rec["created_at"])

    def get_task_list(self, at_timestamp: int, user: str, min_priority: int | None = None) -> list[Task]:
        matches = [(task_id, rec) for task_id, rec in self._tasks.items()
                   if rec["user"] == user and self._alive(rec, at_timestamp)
                   and (min_priority is None or rec["priority"] >= min_priority)]
        matches.sort(key=lambda pair: (-pair[1]["priority"], _creation_seq(pair[0])))
        return [Task(task_id, rec["title"], rec["priority"], rec["created_at"]) for task_id, rec in matches]


from bisect import bisect_left, bisect_right


@dataclass
class _History:
    user: str
    ttl: int | None
    created_at: int
    timestamps: list[int]     # version timestamps, non-decreasing, bisected against at_timestamp
    titles: list[str]         # titles[i] / priorities[i] held from timestamps[i] up to the next entry
    priorities: list[int]
    completed_at: int | None = None


class TaskManager(TaskManager):
    def __init__(self) -> None:
        self._history: dict[str, _History] = {}
        self._by_user: dict[str, list[str]] = {}
        self._next_id = 1

    def add_task(self, timestamp: int, user: str, title: str, priority: int, ttl: int | None = None) -> str:
        task_id = f"task-{self._next_id}"
        self._next_id += 1
        self._history[task_id] = _History(user, ttl, timestamp, [timestamp], [title], [priority])
        self._by_user.setdefault(user, []).append(task_id)
        return task_id

    def _version_at(self, hist: _History, at_timestamp: int) -> tuple[str, int] | None:
        i = bisect_right(hist.timestamps, at_timestamp) - 1
        if i < 0:
            return None
        return hist.titles[i], hist.priorities[i]

    def _alive_at(self, hist: _History, at_timestamp: int) -> bool:
        if self._version_at(hist, at_timestamp) is None:
            return False
        if hist.completed_at is not None and hist.completed_at <= at_timestamp:
            return False
        return hist.ttl is None or at_timestamp < hist.created_at + hist.ttl

    def _record(self, user: str, task_id: str, at_timestamp: int) -> _History | None:
        hist = self._history.get(task_id)
        if hist is None or hist.user != user or not self._alive_at(hist, at_timestamp):
            return None
        return hist

    def update_task(self, timestamp: int, user: str, task_id: str, title: str, priority: int) -> bool:
        hist = self._record(user, task_id, timestamp)
        if hist is None:
            return False
        hist.timestamps.append(timestamp)
        hist.titles.append(title)
        hist.priorities.append(priority)
        return True

    def complete_task(self, timestamp: int, user: str, task_id: str) -> bool:
        hist = self._record(user, task_id, timestamp)
        if hist is None:
            return False
        hist.completed_at = timestamp
        return True

    def get_task(self, timestamp: int, user: str, task_id: str) -> Task | None:
        hist = self._record(user, task_id, timestamp)
        if hist is None:
            return None
        title, priority = self._version_at(hist, timestamp)
        return Task(task_id, title, priority, hist.created_at)

    def get_task_list(self, at_timestamp: int, user: str, min_priority: int | None = None) -> list[Task]:
        out = []
        for task_id in self._by_user.get(user, []):
            hist = self._history[task_id]
            if not self._alive_at(hist, at_timestamp):
                continue
            i = bisect_left(hist.timestamps, at_timestamp) - 1
            title, priority = hist.titles[i], hist.priorities[i]
            if min_priority is not None and priority < min_priority:
                continue
            out.append((priority, _creation_seq(task_id), task_id, title))
        out.sort(key=lambda x: (-x[0], x[1]))
        return [Task(task_id, title, priority, self._history[task_id].created_at)
                for priority, _, task_id, title in out]
