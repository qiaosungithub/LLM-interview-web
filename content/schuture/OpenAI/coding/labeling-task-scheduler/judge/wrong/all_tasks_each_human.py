# expect: WA
# Wrong: gives every human all t tasks (h * t assignments), which covers everyone but is not the minimal h * k.
def _all(t, m, h, k):
    if t <= 0 or m <= 0 or h <= 0:
        return None
    if k == 0:
        return []
    if k > t:
        return None
    seen, schedule = [0] * t, []
    for r in range(t):
        for u in range(h):
            task = (u + r) % t
            schedule.append((task, seen[task] % m, u))
            seen[task] += 1
    return schedule


def build_basic_schedule(t, m, h, k):
    return _all(t, m, h, k)


def build_balanced_schedule(t, m, h, k):
    return _all(t, m, h, k)


def build_doubly_balanced_schedule(t, m, h, k):
    return _all(t, m, h, k)
