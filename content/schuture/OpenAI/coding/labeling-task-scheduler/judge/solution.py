from typing import List, Optional, Tuple

Assignment = Tuple[int, int, int]  # (task, model, human)


def build_basic_schedule(t, m, h, k):
    if t <= 0 or m <= 0 or h <= 0:
        return None
    if k == 0:
        return []
    if k > t:
        return None
    schedule = []
    for r in range(k):
        for u in range(h):
            schedule.append(((u + r) % t, 0, u))   # NOTE: model is unconstrained in Part 1; 0 is simplest
    return schedule


def build_balanced_schedule(t, m, h, k):
    if t <= 0 or m <= 0 or h <= 0:
        return None
    if k == 0:
        return []
    if k > t:
        return None
    task_seen = [0] * t
    schedule = []
    for r in range(k):
        for u in range(h):
            task = (u + r) % t
            model = task_seen[task] % m          # NOTE: task_seen[task] is x's occurrence count so far
            task_seen[task] += 1
            schedule.append((task, model, u))
    return schedule


def build_doubly_balanced_schedule(t, m, h, k):
    if t <= 0 or m <= 0 or h <= 0:
        return None
    if k == 0:
        return []
    if k > t:
        return None
    task_count = [[0] * m for _ in range(t)]
    human_count = [[0] * m for _ in range(h)]
    schedule = []
    for r in range(k):
        for u in range(h):
            task = (u + r) % t
            # NOTE: argmin over the task's own counts first; the human's counts only break ties,
            # so they can lose every tie and the human axis stays unprotected (see the text above).
            model = min(range(m), key=lambda i: (task_count[task][i], human_count[u][i], i))
            task_count[task][model] += 1
            human_count[u][model] += 1
            schedule.append((task, model, u))
    return schedule
