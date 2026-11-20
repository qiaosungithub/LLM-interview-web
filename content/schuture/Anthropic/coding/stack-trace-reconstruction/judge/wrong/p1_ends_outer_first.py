# expect: WA
# ends the frames that left from the split point inward instead of innermost first
from collections.abc import Iterable, Iterator
from typing import Literal, NamedTuple


class Sample(NamedTuple):
    ts: float
    stack: list[str]   # outermost (root) first, innermost (currently executing) last


class Event(NamedTuple):
    kind: Literal["start", "end"]
    ts: float
    name: str
    depth: int   # 0-based position in the stack -- the same name at two depths is two different frames


def _common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def convert_samples_to_events(samples: list[Sample], end_time: float) -> list[Event]:
    if not samples:
        return []
    events: list[Event] = []
    first = samples[0]
    for depth, name in enumerate(first.stack):
        events.append(Event("start", first.ts, name, depth))   # NOTE: only the very first sample gets this;
    prev = first.stack                                          # Part 2 has no such special case
    for sample in samples[1:]:
        lcp = _common_prefix_len(prev, sample.stack)
        for depth in range(lcp, len(prev)):                   # the frames that left, from the split point inward
            events.append(Event("end", sample.ts, prev[depth], depth))
        for depth in range(lcp, len(sample.stack)):             # outermost first: a caller starts before its callee
            events.append(Event("start", sample.ts, sample.stack[depth], depth))
        prev = sample.stack
    for depth in range(len(prev) - 1, -1, -1):                  # innermost first, same rule as any other close
        events.append(Event("end", end_time, prev[depth], depth))
    return events


def stream_debounced_events(samples: Iterable[Sample], k: int) -> Iterator[Event]:
    prev: list[str] = []
    streak: list[int] = []           # streak[d]: consecutive samples (ending at prev) position d has held prev[d]
    open_len = 0                     # positions [0, open_len) are confirmed -- their start has already been emitted
    closing: dict[int, list] = {}    # depth -> [name, consecutive samples absent since] for a confirmed frame just vacated

    for sample in samples:
        cur = sample.stack
        lcp = _common_prefix_len(prev, cur)

        for entry in closing.values():
            entry[1] += 1                                             # every pending closure ages by one sample
        for depth in range(len(prev) - 1, lcp - 1, -1):                # positions leaving the chain, innermost first
            if depth < open_len:
                closing[depth] = [prev[depth], 1]                      # NOTE: overwrites -- no stale entry can coexist here
        open_len = min(open_len, lcp)
        for depth in sorted((d for d, entry in closing.items() if entry[1] >= k), reverse=True):
            name, _count = closing.pop(depth)
            yield Event("end", sample.ts, name, depth)

        streak = [streak[d] + 1 if d < lcp else 1 for d in range(len(cur))]
        while open_len < len(cur) and streak[open_len] >= k:            # NOTE: streak is non-increasing in depth,
            yield Event("start", sample.ts, cur[open_len], open_len)    # so this never skips a confirmable position
            open_len += 1
        prev = cur


class TraceIndex:
    """Indexes a fixed set of complete stack traces for exact-suffix lookup."""

    def __init__(self, traces: list[tuple[str, list[str]]]) -> None:
        self._root: dict = {}
        for trace_id, frames in traces:
            node = self._root
            for frame in reversed(frames):                      # innermost (error point) first
                node = node.setdefault(frame, {})
                node.setdefault("$", []).append(trace_id)        # NOTE: every node on the path, not just the last
        self._sort_ids(self._root)

    def _sort_ids(self, node: dict) -> None:
        if "$" in node:
            node["$"].sort()
        for key, child in node.items():
            if key != "$":
                self._sort_ids(child)

    def match_suffix(self, frames: list[str]) -> list[str] | str:
        node = self._root
        for frame in reversed(frames):
            if frame not in node:
                return "UNKNOWN"          # NOTE: no known trace even has this many matching trailing frames
            node = node[frame]
        matches = node.get("$", [])
        return list(matches) if matches else "UNKNOWN"
