class Workforce:
    def __init__(self) -> None:
        """Creates an empty workforce."""

    def add_worker(self, worker_id: str, position: str, hourly_rate: int) -> bool:
        """Level 1: registers a new worker; False, without effect, if worker_id is already registered."""
        raise NotImplementedError

    def clock_in(self, worker_id: str, timestamp: int) -> bool:
        """Level 1: opens a session at timestamp; False, without effect, if not registered or a session is already open."""
        raise NotImplementedError

    def clock_out(self, worker_id: str, timestamp: int) -> bool:
        """Level 1: closes the open session as [start, timestamp); False, without effect, if not registered or no session is open."""
        raise NotImplementedError

    def get_total_work_time(self, worker_id: str) -> int | None:
        """Level 2: sum of the durations of worker_id's closed sessions; None if not registered."""
        raise NotImplementedError

    def top_k_workers(self, k: int) -> list[str]:
        """Level 2: up to k worker_ids by total work time descending, ties by worker_id ascending."""
        raise NotImplementedError

    def set_promotion(self, worker_id: str, new_position: str, new_hourly_rate: int) -> bool:
        """Level 3: schedules a new position and rate, effective at worker_id's next clock_in; False if not registered."""
        raise NotImplementedError

    def calculate_pay(self, worker_id: str, start: int, end: int) -> int | None:
        """Level 3: worker_id's pay earned during [start, end); None if not registered."""
        raise NotImplementedError

    def set_double_pay(self, start: int, end: int) -> None:
        """Level 4: marks every timestamp in [start, end) as double pay for every worker."""
        raise NotImplementedError
