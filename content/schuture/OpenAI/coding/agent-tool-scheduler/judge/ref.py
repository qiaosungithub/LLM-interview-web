"""Helpers for tests.py: the statement's ToolCall, a brute force taken straight from the rules, a fast reference for
the big inputs, and random instance generators."""
import heapq
from typing import NamedTuple, Sequence


class ToolCall(NamedTuple):
    tool_id: int
    duration: int
    deps: Sequence[int] = ()

    def __repr__(self):         # the short form, so a shown call can be pasted into 自定义运行
        if self.deps:
            return f"ToolCall({self.tool_id}, {self.duration}, deps={tuple(self.deps)!r})"
        return f"ToolCall({self.tool_id}, {self.duration})"


class Cycle(Exception):
    """Raised by the references when some call can never start."""


def brute(agents, capacity):
    """Every agent's calls share `capacity` slots. Straight from the rules: at each finish time release every call
    that finishes then, rescan all calls for unstarted ones whose dependencies have all finished, sort those by
    (agent_id, tool_id) and fill the free slots. Nothing can change between two finish times, so jumping from one
    to the next is exact. Raises Cycle when calls remain but nothing is running."""
    flat = [(a, c) for a, calls in enumerate(agents) for c in calls]
    dur = {(a, c.tool_id): c.duration for a, c in flat}
    finish = {}                 # (agent_id, tool_id) -> start + duration, for every started call
    starts = []
    t = 0
    while True:
        running = sum(1 for f in finish.values() if f > t)
        ready = sorted((a, c.tool_id) for a, c in flat
                       if (a, c.tool_id) not in finish and all(finish.get((a, d), t + 1) <= t for d in c.deps))
        for key in ready[:capacity - running]:
            starts.append((t,) + key)
            finish[key] = t + dur[key]
        if len(starts) == len(flat):
            return starts, max(finish.values(), default=0)
        later = [f for f in finish.values() if f > t]
        if not later:
            raise Cycle()
        t = min(later)


def independent(agents, capacities):
    """Part 2: each agent alone on its own capacity, merged by (start_time, agent_id, tool_id)."""
    starts, span = [], 0
    for a, (calls, cap) in enumerate(zip(agents, capacities)):
        one, one_span = brute([calls], cap)
        starts += [(st, a, tid) for st, _, tid in one]
        span = max(span, one_span)
    return sorted(starts), span


def fast(agents, capacity):
    """O((N + E) log N) version of brute() for the big inputs (cross-checked against brute() while writing the
    judge). Call (a, t) gets the number base[a] + t, so plain integers already order like (agent_id, tool_id)."""
    base, keys = [], []
    for a, calls in enumerate(agents):
        base.append(len(keys))
        keys.extend((a, t) for t in range(len(calls)))
    n = len(keys)
    dur, waiting, after = [0] * n, [0] * n, [[] for _ in range(n)]
    for a, calls in enumerate(agents):
        b = base[a]
        for c in calls:
            u = b + c.tool_id
            dur[u] = c.duration
            waiting[u] = len(c.deps)
            for d in c.deps:
                after[b + d].append(u)
    ready = [u for u in range(n) if waiting[u] == 0]      # increasing, so already a heap
    running, starts = [], []
    free, t, span = capacity, 0, 0
    while True:
        while free and ready:
            u = heapq.heappop(ready)
            starts.append((t,) + keys[u])
            free -= 1
            heapq.heappush(running, (t + dur[u], u))
            span = max(span, t + dur[u])
        if len(starts) == n:
            return starts, span
        if not running:
            raise Cycle()
        t = running[0][0]
        while running and running[0][0] == t:
            u = heapq.heappop(running)[1]
            free += 1
            for v in after[u]:
                waiting[v] -= 1
                if not waiting[v]:
                    heapq.heappush(ready, v)


def random_agent(r, n, max_dur, max_deps=2, back_edges=0):
    """n calls with tool ids 0..n-1 in shuffled list positions. Dependencies point to lower ids, so the agent is
    acyclic unless `back_edges` edges from a lower id to a higher one are added (they may close a cycle)."""
    deps = [set(r.sample(range(t), r.randint(0, min(max_deps, t)))) for t in range(n)]
    for _ in range(back_edges if n >= 2 else 0):
        lo, hi = sorted(r.sample(range(n), 2))
        deps[lo].add(hi)
    calls = [ToolCall(t, r.randint(1, max_dur), tuple(r.sample(sorted(d), len(d)))) for t, d in enumerate(deps)]
    r.shuffle(calls)
    return calls


def random_agents(r, n_agents, max_calls, max_dur, back_edge_p=0.0, min_calls=0):
    agents = []
    for _ in range(n_agents):
        n = r.randint(min_calls, max_calls)
        agents.append(random_agent(r, n, max_dur, r.randint(1, 3), 1 if r.random() < back_edge_p else 0))
    return agents


def big_many_agents(r, n_agents=100_000):
    """Two calls per agent (the second waits for the first half of the time), durations up to 10**9."""
    agents = []
    for _ in range(n_agents):
        calls = [ToolCall(0, r.randint(1, 10 ** 9)), ToolCall(1, r.randint(1, 10 ** 9), (0,) if r.random() < 0.5 else ())]
        if r.random() < 0.5:
            calls.reverse()
        agents.append(calls)
    return agents


def big_dense(r, n_agents=2000, per_agent=100, max_edges=500_000):
    """n_agents * per_agent calls, two or three dependencies each on lower ids, at most max_edges edges in all."""
    agents, edges = [], 0
    for _ in range(n_agents):
        calls = []
        for t in range(per_agent):
            k = min(t, r.choice((2, 3)))
            d = tuple(r.sample(range(t), k))
            edges += k
            calls.append(ToolCall(t, r.randint(1, 10 ** 9), d))
        r.shuffle(calls)
        agents.append(calls)
    assert edges <= max_edges, edges
    return agents
