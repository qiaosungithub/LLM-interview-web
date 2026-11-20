# expect: WA
# Wrong for Part 2: within round r a task's occurrences take models r, r+1, ... (mod m), restarting each round
# instead of continuing the task's rotation, so a task seen twice in one round drifts out of balance.
def build_basic_schedule(t, m, h, k):
    if t <= 0 or m <= 0 or h <= 0:
        return None
    if k == 0:
        return []
    if k > t:
        return None
    return [((u + r) % t, 0, u) for r in range(k) for u in range(h)]


def build_balanced_schedule(t, m, h, k):
    if t <= 0 or m <= 0 or h <= 0:
        return None
    if k == 0:
        return []
    if k > t:
        return None
    schedule = []
    for r in range(k):
        in_round = [0] * t
        for u in range(h):
            task = (u + r) % t
            schedule.append((task, (r + in_round[task]) % m, u))
            in_round[task] += 1
    return schedule


def build_doubly_balanced_schedule(t, m, h, k):
    if t <= 0 or m <= 0 or h <= 0:
        return None
    if k == 0:
        return []
    if k > t:
        return None
    tc = [[0] * m for _ in range(t)]
    hc = [[0] * m for _ in range(h)]
    schedule = []
    for r in range(k):
        for u in range(h):
            task = (u + r) % t
            model = min(range(m), key=lambda i: (tc[task][i], hc[u][i], i))
            tc[task][model] += 1
            hc[u][model] += 1
            schedule.append((task, model, u))
    return schedule
