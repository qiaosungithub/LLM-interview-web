"""Helpers for tests.py: a brute force that replays the rules as stated, a one-pass sweep for big inputs, and
random call sequences.

A call is a tuple:  ("add", credit_id, amount, timestamp, expiration) | ("sub", amount, timestamp) | ("ask", t).
"""
import heapq


def when(op):
    return op[3] if op[0] == "add" else op[2]


def balance(ops, t):
    """get_balance(t) after the add/sub calls in ops (any order): replay those with timestamp <= t in timestamp
    order, one grant at a time, no heap. Each subtract scans the grants valid at its instant, soonest expiry first."""
    grants, debt = [], 0                    # [first valid instant, last valid instant, remaining]
    for op in sorted((op for op in ops if op[0] != "ask" and when(op) <= t), key=when):
        if op[0] == "add":
            _, _cid, amount, ts, expiration = op
            repaid = min(debt, amount)
            debt -= repaid
            grants.append([ts, ts + expiration, amount - repaid])
        else:
            _, need, ts = op
            for grant in sorted((g for g in grants if g[0] <= ts <= g[1]), key=lambda g: g[1]):
                take = min(need, grant[2])
                grant[2] -= take
                need -= take
            debt += need
    v = sum(g[2] for g in grants if g[0] <= t <= g[1]) - debt
    return None if v < 0 else v


def sweep(events, queries):
    """Answers for sorted queries over events (any order) in one pass: a heap of grants by expiry and the sum of
    what they hold, events applied in timestamp order up to each query. Used where balance() per query would be
    too slow; checked against balance() on random inputs while writing the judge."""
    events = sorted(events, key=when)
    heap, total, debt, k, out = [], 0, 0, 0, []
    for t in queries:
        while k < len(events) and when(events[k]) <= t:
            op = events[k]
            k += 1
            if op[0] == "add":
                _, _cid, amount, ts, expiration = op
                repaid = min(debt, amount)
                debt -= repaid
                if amount > repaid:
                    heapq.heappush(heap, [ts + expiration, amount - repaid])
                    total += amount - repaid
            else:
                _, need, ts = op
                while heap and heap[0][0] < ts:
                    total -= heapq.heappop(heap)[1]
                while need and heap:
                    take = min(need, heap[0][1])
                    heap[0][1] -= take
                    total -= take
                    need -= take
                    if heap[0][1] == 0:
                        heapq.heappop(heap)
                debt += need
        while heap and heap[0][0] < t:
            total -= heapq.heappop(heap)[1]
        v = total - debt
        out.append(None if v < 0 else v)
    return out


def random_events(r, n, ts_max, amount_max=20):
    """n add/sub calls with distinct timestamps in [0, ts_max], in timestamp order."""
    ops = []
    for i, ts in enumerate(sorted(r.sample(range(ts_max + 1), n))):
        if r.random() < 0.55:
            ops.append(("add", f"g{i}", r.randint(1, amount_max), ts, r.choice([0, 1, 4, 4, r.randint(0, 15)])))
        else:
            ops.append(("sub", r.choice([1, 2, 3, 5, 8, 30]), ts))
    return ops


def in_order_calls(r, events, ts_max):
    """Part 1: the events in timestamp order with queries in between, every query timestamp non-decreasing.
    A query may share its timestamp with the next event and come before it."""
    calls, now = [], 0
    for op in events:
        if r.random() < 0.6:
            now = r.randint(now, when(op))
            calls.append(("ask", now))
        calls.append(op)
        now = when(op)
        if r.random() < 0.7:
            calls.append(("ask", now))
    for t in sorted(r.sample(range(now, ts_max + 18), 4)):
        calls.append(("ask", t))
    return calls


def any_order_calls(r, events, ts_max):
    """Parts 2 and 3: the events shuffled, queries in between (some repeated), then a sweep in random order."""
    events = list(events)
    r.shuffle(events)
    calls, asked = [], []
    for op in events:
        calls.append(op)
        for _ in range(r.randint(0, 3)):
            t = r.choice(asked) if asked and r.random() < 0.3 else r.randint(0, ts_max + 17)
            asked.append(t)
            calls.append(("ask", t))
    for t in r.sample(range(ts_max + 18), min(ts_max + 18, 25)):
        calls.append(("ask", t))
    return calls


def ahead_calls(r, events, ts_max):
    """Part 3's fast pattern: events arrive shuffled in blocks, each block ahead of every query made so far;
    queries are non-decreasing and repeat now and then."""
    events = sorted(events, key=when)
    calls, now, i = [], 0, 0
    while i < len(events):
        block = events[i:i + r.randint(1, 4)]
        i += len(block)
        block_end = when(block[-1])
        r.shuffle(block)
        calls += block
        nxt = when(events[i]) - 1 if i < len(events) else ts_max + 17
        for _ in range(r.randint(0, 3)):
            if nxt >= max(now, block_end):
                now = r.randint(max(now, block_end), nxt)
                calls.append(("ask", now))
    calls.append(("ask", now))
    calls.append(("ask", ts_max + 17))
    return calls
