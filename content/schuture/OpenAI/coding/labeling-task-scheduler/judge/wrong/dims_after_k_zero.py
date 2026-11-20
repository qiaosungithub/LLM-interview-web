# expect: WA
# Wrong: tests k == 0 before the dimensions, so (t, m, h) with a non-positive entry and k == 0 returns [] instead of None.
def _round_robin(t, m, h, k, pick):
    if k == 0:
        return []
    if t <= 0 or m <= 0 or h <= 0 or k > t:
        return None
    task_count = [[0] * m for _ in range(t)]
    human_count = [[0] * m for _ in range(h)]
    schedule = []
    for r in range(k):
        for u in range(h):
            task = (u + r) % t
            model = pick(task_count[task], human_count[u], m)
            task_count[task][model] += 1
            human_count[u][model] += 1
            schedule.append((task, model, u))
    return schedule


def build_basic_schedule(t, m, h, k):
    return _round_robin(t, m, h, k, lambda tc, hc, m: 0)


def build_balanced_schedule(t, m, h, k):
    return _round_robin(t, m, h, k, lambda tc, hc, m: sum(tc) % m)


def build_doubly_balanced_schedule(t, m, h, k):
    return _round_robin(t, m, h, k, lambda tc, hc, m: min(range(m), key=lambda i: (tc[i], hc[i], i)))
