# expect: WA
# Wrong for Part 3: the page's follow-up -- minimising the human's counts first and the task's counts only on ties
# protects the human axis but breaks balance by task (first at t, m, h, k = 2, 3, 6, 2).
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
    seen, schedule = [0] * t, []
    for r in range(k):
        for u in range(h):
            task = (u + r) % t
            schedule.append((task, seen[task] % m, u))
            seen[task] += 1
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
            model = min(range(m), key=lambda i: (hc[u][i], tc[task][i], i))
            tc[task][model] += 1
            hc[u][model] += 1
            schedule.append((task, model, u))
    return schedule
