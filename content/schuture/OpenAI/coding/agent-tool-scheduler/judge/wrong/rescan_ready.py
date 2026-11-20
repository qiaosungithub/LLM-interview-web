# expect: TLE
# Jumps between finish times but rescans every call for readiness at each one: O(N^2) at N = 2 * 10**5.
import heapq
from typing import List, Tuple, NamedTuple, Sequence


class ToolCall(NamedTuple):
    tool_id: int
    duration: int
    deps: Sequence[int] = ()


class CycleError(Exception):
    """Some tool calls can never run: their dependencies form a cycle."""


# ---------------------------------------------------------------- Part 1

def _event_driven_schedule(tasks, capacity):
    n = len(tasks)
    pos = {(agent_id, call.tool_id): i for i, (agent_id, call) in enumerate(tasks)}
    deps = [[pos[(agent_id, d)] for d in call.deps] for agent_id, call in tasks]
    order = sorted(range(n), key=lambda i: (tasks[i][0], tasks[i][1].tool_id))
    finish = [None] * n
    running: List[int] = []
    starts: List[Tuple[int, int, int]] = []
    now, makespan = 0, 0
    while len(starts) < n:
        free = capacity - len(running)
        for i in order:
            if free == 0:
                break
            if finish[i] is None and all(finish[j] is not None and finish[j] <= now for j in deps[i]):
                agent_id, call = tasks[i]
                starts.append((now, agent_id, call.tool_id))
                finish[i] = now + call.duration
                makespan = max(makespan, finish[i])
                heapq.heappush(running, finish[i])
                free -= 1
        if len(starts) == n:
            break
        if not running:
            raise CycleError("some tool calls can never become ready")
        now = running[0]
        while running and running[0] == now:
            heapq.heappop(running)
    return starts, makespan


def schedule_agent(calls: List[ToolCall], capacity: int):
    return _event_driven_schedule([(0, c) for c in calls], capacity)


# ---------------------------------------------------------------- Part 2

def schedule_agents_independently(agents: List[List[ToolCall]], capacities: List[int]):
    all_starts: List[Tuple[int, int, int]] = []
    makespan = 0
    for agent_id, (calls, capacity) in enumerate(zip(agents, capacities)):
        starts, agent_makespan = _event_driven_schedule([(agent_id, c) for c in calls], capacity)
        all_starts.extend(starts)
        makespan = max(makespan, agent_makespan)
    all_starts.sort()                  # NOTE: needed here; each agent's own starts were already sorted
    return all_starts, makespan


# ---------------------------------------------------------------- Part 3

def schedule_agents_global(agents: List[List[ToolCall]], capacity: int):
    tasks = [(agent_id, call) for agent_id, calls in enumerate(agents) for call in calls]
    return _event_driven_schedule(tasks, capacity)
