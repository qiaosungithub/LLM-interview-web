# expect: WA
# Wrong: the j-th assignment gets task j % t and human j % h; when gcd(t, h) > 1 a human sees the same task twice.
def _flat(t, m, h, k, balanced):
    if t <= 0 or m <= 0 or h <= 0:
        return None
    if k == 0:
        return []
    if k > t:
        return None
    seen, schedule = [0] * t, []
    for j in range(h * k):
        task, human = j % t, j % h
        model = seen[task] % m if balanced else 0
        seen[task] += 1
        schedule.append((task, model, human))
    return schedule


def build_basic_schedule(t, m, h, k):
    return _flat(t, m, h, k, False)


def build_balanced_schedule(t, m, h, k):
    return _flat(t, m, h, k, True)


def build_doubly_balanced_schedule(t, m, h, k):
    return _flat(t, m, h, k, True)
