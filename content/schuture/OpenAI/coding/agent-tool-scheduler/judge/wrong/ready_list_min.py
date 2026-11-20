# expect: TLE
# Keeps the ready calls in a plain list and takes min() of it for every start: O(N * ready) under contention.
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
    indegree = [0] * n
    dependents: List[List[int]] = [[] for _ in range(n)]
    for i, (agent_id, call) in enumerate(tasks):
        for dep in call.deps:
            dependents[pos[(agent_id, dep)]].append(i)
            indegree[i] += 1
    ready = [(agent_id, call.tool_id, i) for i, (agent_id, call) in enumerate(tasks) if indegree[i] == 0]
    running: List[Tuple[int, int]] = []
    starts: List[Tuple[int, int, int]] = []
    free, now, makespan = capacity, 0, 0
    while len(starts) < n:
        while free > 0 and ready:
            best = min(ready)
            ready.remove(best)
            agent_id, tool_id, i = best
            starts.append((now, agent_id, tool_id))
            free -= 1
            finish = now + tasks[i][1].duration
            makespan = max(makespan, finish)
            heapq.heappush(running, (finish, i))
        if len(starts) == n:
            break
        if not running:
            raise CycleError("some tool calls can never become ready")
        now = running[0][0]
        while running and running[0][0] == now:
            _, i = heapq.heappop(running)
            free += 1
            for j in dependents[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    ready.append((tasks[j][0], tasks[j][1].tool_id, j))
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
