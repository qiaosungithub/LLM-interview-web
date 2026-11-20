from typing import List, Optional, Tuple

Assignment = Tuple[int, int, int]  # (task, model, human)


def build_basic_schedule(t: int, m: int, h: int, k: int) -> Optional[List[Assignment]]:
    """Part 1: None for invalid dimensions or k > t, [] if k == 0, else h * k assignments with coverage and uniqueness."""
    raise NotImplementedError


def build_balanced_schedule(t: int, m: int, h: int, k: int) -> Optional[List[Assignment]]:
    """Part 2: same contract as build_basic_schedule, and also balanced by task at every prefix."""
    raise NotImplementedError


def build_doubly_balanced_schedule(t: int, m: int, h: int, k: int) -> Optional[List[Assignment]]:
    """Part 3: same contract as build_balanced_schedule; a greedy rule also attempts to balance by human."""
    raise NotImplementedError
