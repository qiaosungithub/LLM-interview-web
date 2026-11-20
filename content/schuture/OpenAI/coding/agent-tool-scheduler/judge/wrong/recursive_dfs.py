# expect: RE
# Checks for cycles with a recursive DFS first; a 50000-call dependency chain recurses too deep.
import heapq
from typing import List, NamedTuple, Sequence, Tuple


class ToolCall(NamedTuple):
    tool_id: int
    duration: int
    deps: Sequence[int] = ()


class CycleError(Exception):
    pass


FINISH, START = 0, 1


def _check_cycles(groups):
    for a, calls in groups:
        deps = {c.tool_id: c.deps for c in calls}
        state = {}
        def visit(t):
            if state.get(t) == 1:
                raise CycleError()
            if state.get(t) == 2:
                return
            state[t] = 1
            for d in deps[t]:
                visit(d)
            state[t] = 2
        for t in deps:
            visit(t)


def _simulate(groups, capacity):
    _check_cycles(groups)
    """groups: list of (agent_id, calls). One event heap of (time, kind, agent_id, tool_id): all FINISH events at a
    time come before the START attempt at that time; a START event at t means 'fill free slots now'."""
    dur, indeg, children = {}, {}, {}
    total = 0
    for a, calls in groups:
        for c in calls:
            k = (a, c.tool_id)
            dur[k] = c.duration
            indeg[k] = len(c.deps)
            children.setdefault(k, [])
            total += 1
        for c in calls:
            for d in c.deps:
                children[(a, d)].append((a, c.tool_id))
    ready = [k for k, v in indeg.items() if v == 0]
    heapq.heapify(ready)
    events = [(0, START, -1, -1)]
    free = capacity
    out, span = [], 0
    while events:
        t, kind, a, tid = heapq.heappop(events)
        if kind == FINISH:
            free += 1
            for ch in children[(a, tid)]:
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    heapq.heappush(ready, ch)
            if not events or events[0][0] != t or events[0][1] != FINISH:
                heapq.heappush(events, (t, START, -1, -1))
            continue
        while free and ready:
            k = heapq.heappop(ready)
            out.append((t, k[0], k[1]))
            free -= 1
            end = t + dur[k]
            if end > span:
                span = end
            heapq.heappush(events, (end, FINISH, k[0], k[1]))
    if len(out) != total:
        raise CycleError()
    return out, span


def schedule_agent(calls: List[ToolCall], capacity: int) -> Tuple[List[Tuple[int, int, int]], int]:
    return _simulate([(0, calls)], capacity)


def schedule_agents_independently(agents, capacities):
    allout, span = [], 0
    for a, (calls, cap) in enumerate(zip(agents, capacities)):
        out, s = _simulate([(a, calls)], cap)
        allout.extend(out)
        span = max(span, s)
    allout.sort()
    return allout, span


def schedule_agents_global(agents, capacity):
    return _simulate(list(enumerate(agents)), capacity)
