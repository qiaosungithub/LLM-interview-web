"""Helpers for tests.py: the statement's Sample type, plain restatements of the three rules (frames found as
maximal runs along the common-prefix chain; suffixes compared by slicing), random inputs and display helpers."""
from typing import NamedTuple


class Sample(NamedTuple):
    ts: float
    stack: list[str]


# ---------------------------------------------------------------- frames as runs

def runs(samples):
    """Every frame as (depth, name, first_index, leave_index or None): a frame lives at one position from the
    sample where the chain of common prefixes first reaches it until the first sample where it is gone."""
    out, live, prev = [], {}, []
    for i, sample in enumerate(samples):
        cur = sample.stack
        p = 0
        while p < min(len(prev), len(cur)) and prev[p] == cur[p]:
            p += 1
        for d in range(p, len(prev)):
            name, first = live.pop(d)
            out.append((d, name, first, i))
        for d in range(p, len(cur)):
            live[d] = (cur[d], i)
        prev = cur
    out.extend((d, name, first, None) for d, (name, first) in live.items())
    return out


def _ordered(tagged):
    """tagged: (group, kind, ts, name, depth). Within a group, ends innermost first, then starts outermost first."""
    tagged.sort(key=lambda t: (t[0], t[1] == "start", -t[4] if t[1] == "end" else t[4]))
    return [t[1:] for t in tagged]


def events(samples, end_time):
    """Part 1. Group i holds the changes seen at sample i; group len(samples) is the closing batch at end_time."""
    n, tagged = len(samples), []
    for d, name, first, leave in runs(samples):
        tagged.append((first, "start", samples[first].ts, name, d))
        if leave is None:
            tagged.append((n, "end", end_time, name, d))
        else:
            tagged.append((leave, "end", samples[leave].ts, name, d))
    return _ordered(tagged)


def event_index(samples, k):
    """Part 2 as (index, event) pairs: index is the 0-based sample that settles the event. A run of at least k
    samples starts at its k-th sample; it ends at the k-th sample counted from the first sample without it, if
    the input gets that far."""
    n, tagged = len(samples), []
    for d, name, first, leave in runs(samples):
        if (n if leave is None else leave) - first < k:
            continue
        tagged.append((first + k - 1, "start", samples[first + k - 1].ts, name, d))
        if leave is not None and leave + k - 1 < n:
            tagged.append((leave + k - 1, "end", samples[leave + k - 1].ts, name, d))
    tagged.sort(key=lambda t: (t[0], t[1] == "start", -t[4] if t[1] == "end" else t[4]))
    return [(t[0], t[1:]) for t in tagged]


def debounced(samples, k):
    """Part 2."""
    return [e for _, e in event_index(samples, k)]


def match(traces, frames):
    """Part 3 by slicing."""
    m = len(frames)
    ids = sorted(tid for tid, fs in traces if len(fs) >= m and fs[len(fs) - m:] == frames)
    return ids or "UNKNOWN"


# ---------------------------------------------------------------- random inputs

TS_STEPS = [0.5, 1, 1, 1.5, 2, 3]


def walk_samples(r, n, names, start=0.0, noise=0.0, hold=1):
    """n samples of a random push/pop walk (recursion is common with few names), now and then jumping to an
    unrelated stack. With noise > 0 some samples show a one-off extra or replaced frame; hold > 1 keeps each state
    for a random number of samples, so debounced frames get confirmed. Timestamps strictly increase."""
    stack, out, ts, left = [], [], start, 0
    for _ in range(n):
        if left == 0:
            if r.random() < 0.1:
                stack = [r.choice(names) for _ in range(r.randint(0, 4))]
            else:
                for _ in range(r.randint(1, 2)):
                    if stack and r.random() < 0.5:
                        stack.pop()
                    elif len(stack) < 7:
                        stack.append(r.choice(names))
            left = r.randint(1, hold)
        left -= 1
        shown = list(stack)
        if r.random() < noise:
            if shown and r.random() < 0.5:
                shown[r.randrange(len(shown))] = r.choice(names)
            else:
                shown.insert(r.randint(0, len(shown)), r.choice(names))
        ts += r.choice(TS_STEPS)
        out.append(Sample(ts, shown))
    return out


def random_traces(r, count, names, max_len=6):
    """(trace_id, frames) pairs in random order; some traces repeat another's frames or extend another at the
    front, so one query often matches several traces."""
    ids = r.sample(range(1, 10 * count + 10), count)
    out = []
    for i in ids:
        if out and r.random() < 0.25:
            frames = list(r.choice(out)[1])
            if r.random() < 0.6:
                frames = [r.choice(names) for _ in range(r.randint(1, 2))] + frames
        else:
            frames = [r.choice(names) for _ in range(r.randint(1, max_len))]
        out.append((f"{r.choice(['t', 'job-', 'err-'])}{i}", frames))
    r.shuffle(out)
    return out


def random_query(r, traces, names):
    if traces and r.random() < 0.7:
        source = r.choice(traces)[1]
        if r.random() < 0.2:
            source = [r.choice(names)] + source          # one frame longer than that trace
    else:
        source = [r.choice(names) for _ in range(r.randint(1, 6))]
    return list(source[len(source) - r.randint(1, len(source)):])


# ---------------------------------------------------------------- display

def show_ts(ts):
    return repr(ts)


def show_samples(samples):
    return "[" + ", ".join(f"Sample({show_ts(s.ts)}, {s.stack!r})" for s in samples) + "]"


def show_event(e):
    kind, ts, name, depth = e
    return f"Event({kind!r}, {show_ts(ts)}, {name!r}, {depth!r})"


def show_events(evs):
    return "[" + ", ".join(show_event(e) for e in evs) + "]"
