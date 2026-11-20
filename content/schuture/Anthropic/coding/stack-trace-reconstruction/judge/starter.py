from collections.abc import Iterable, Iterator
from typing import Literal, NamedTuple


class Sample(NamedTuple):
    ts: float
    stack: list[str]   # outermost (root) first, innermost (currently executing) last


class Event(NamedTuple):
    kind: Literal["start", "end"]
    ts: float
    name: str
    depth: int   # 0-based position in the stack


def convert_samples_to_events(samples: list[Sample], end_time: float) -> list[Event]:
    """Turn samples sorted by ts into start/end events, closing what is still active at end_time."""
    raise NotImplementedError


def stream_debounced_events(samples: Iterable[Sample], k: int) -> Iterator[Event]:
    """Read samples once and yield events for frames that held their position for k consecutive samples."""
    raise NotImplementedError


class TraceIndex:
    def __init__(self, traces: list[tuple[str, list[str]]]) -> None:
        """Index the known (trace_id, frames) traces, frames listed from entry point to error point."""
        raise NotImplementedError

    def match_suffix(self, frames: list[str]) -> list[str] | str:
        """Sorted ids of the traces that end with frames, or "UNKNOWN" if none does."""
        raise NotImplementedError
