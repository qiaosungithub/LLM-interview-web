from typing import List, NamedTuple, Sequence, Tuple


class ToolCall(NamedTuple):
    tool_id: int
    duration: int              # 1 .. 10**9
    deps: Sequence[int] = ()   # other tool_id's belonging to the SAME agent, all required first


class CycleError(Exception):
    """Some tool calls can never run: their dependencies form a cycle."""


def schedule_agent(calls: List[ToolCall], capacity: int) -> Tuple[List[Tuple[int, int, int]], int]:
    """Every triple in the returned list has agent_id == 0. Returns ([], 0) for an empty `calls`."""
    raise NotImplementedError


def schedule_agents_independently(
    agents: List[List[ToolCall]], capacities: List[int]
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Schedule agents[a] as schedule_agent would on its own under capacities[a]; return every start and the largest per-agent makespan."""
    raise NotImplementedError


def schedule_agents_global(
    agents: List[List[ToolCall]], capacity: int
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Same `agents` shape as schedule_agents_independently, but one `capacity` shared by all of them."""
    raise NotImplementedError
