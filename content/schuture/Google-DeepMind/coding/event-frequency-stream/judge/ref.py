"""Independent restatements of the three parts, used to compute expected answers for the tests."""
import heapq
from collections import Counter, deque


# ---------------------------------------------------------------- Part 1

def most_frequent(events):
    counts = Counter(m for m, _ in events)
    first = {}
    for i, (m, _) in enumerate(events):
        first.setdefault(m, i)
    top = max(counts.values())
    winner = min((m for m in counts if counts[m] == top), key=first.__getitem__)
    return winner, [t for m, t in events if m == winner]


# ---------------------------------------------------------------- Part 2

def window_brute(n, events):
    """Recount the held events from scratch after every arrival. Every count change gets the next number
    (an arrival's own eviction first); ties at the maximum go to the highest number."""
    window, last, step, out = deque(), {}, 0, []
    for m, t in events:
        if len(window) == n:
            old, _ = window.popleft()
            step += 1
            last[old] = step
        window.append((m, t))
        step += 1
        last[m] = step
        counts = Counter(x for x, _ in window)
        top = max(counts.values())
        out.append((max((x for x in counts if counts[x] == top), key=last.__getitem__), top))
    return out


def window_fast(n, events):
    """The same rule with a heap of (-count, -change number, message); stale entries are skipped when read.
    O(log n) per arrival, for the big inputs where window_brute is too slow."""
    window, count, last, heap, step, out = deque(), {}, {}, [], 0, []

    def change(m, d):
        nonlocal step
        step += 1
        c = count.get(m, 0) + d
        count[m], last[m] = c, step
        if c:
            heapq.heappush(heap, (-c, -step, m))

    for m, _t in events:
        if len(window) == n:
            change(window.popleft(), -1)
        window.append(m)
        change(m, 1)
        while True:
            c, s, x = heap[0]
            if count[x] == -c and last[x] == -s:
                out.append((x, -c))
                break
            heapq.heappop(heap)
    return out


def random_events(r, length, alphabet, max_gap=2):
    """Messages from `alphabet` with non-decreasing timestamps (repeats allowed)."""
    t, out = r.randint(-3, 5), []
    for _ in range(length):
        t += r.randint(0, max_gap)
        out.append((r.choice(alphabet), t))
    return out


# ---------------------------------------------------------------- Part 3

class SpaceSaving:
    """Literal Space-Saving. A monitored message gains 1 per occurrence; a new message takes a free counter at 1;
    when all k are taken it evicts the lowest counter (ties: the one that has held its value longest, i.e. reached
    it earliest) and starts at the evicted count plus this occurrence, as in the statement's trace
    ("d inherits a's count 1, plus this occurrence")."""

    def __init__(self, k):
        self.k, self.count, self.since, self.clock = k, {}, {}, 0

    def add(self, m):
        self.clock += 1
        if m in self.count:
            self.count[m] += 1
        elif len(self.count) < self.k:
            self.count[m] = 1
        else:
            victim = min(self.count, key=lambda x: (self.count[x], self.since[x]))
            inherited = self.count.pop(victim)
            del self.since[victim]
            self.count[m] = inherited + 1
        self.since[m] = self.clock

    def top(self, m):
        return sorted(self.count.items(), key=lambda kv: (-kv[1], kv[0]))[:m]


def zipf_stream(r, length, n_messages):
    """A skewed stream: message i has weight 1 / (i + 1), so a few messages dominate."""
    names = [f"m{i}" for i in range(n_messages)]
    return r.choices(names, weights=[1 / (i + 1) for i in range(n_messages)], k=length)
