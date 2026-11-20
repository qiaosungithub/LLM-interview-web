# expect: WA
# debounces each depth on its own by name, ignoring the common-prefix chain
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
        for depth in range(len(prev) - 1, lcp - 1, -1):        # innermost first: a callee returns before its caller
            events.append(Event("end", sample.ts, prev[depth], depth))
        for depth in range(lcp, len(sample.stack)):             # outermost first: a caller starts before its callee
            events.append(Event("start", sample.ts, sample.stack[depth], depth))
        prev = sample.stack
    for depth in range(len(prev) - 1, -1, -1):                  # innermost first, same rule as any other close
        events.append(Event("end", end_time, prev[depth], depth))
    return events


def stream_debounced_events(samples: Iterable[Sample], k: int) -> Iterator[Event]:
    # one debouncer per depth: a depth keeps its frame for as long as the same name is seen there
    held: list[list] = []            # held[d] = [name, consecutive samples, confirmed]
    closing: dict[int, list] = {}    # depth -> [name, consecutive samples absent]

    for sample in samples:
        cur = sample.stack
        for entry in closing.values():
            entry[1] += 1
        for depth in range(len(held) - 1, -1, -1):
            if (depth >= len(cur) or cur[depth] != held[depth][0]) and held[depth][2]:
                closing[depth] = [held[depth][0], 1]
        for depth in sorted((d for d, entry in closing.items() if entry[1] >= k), reverse=True):
            name, _count = closing.pop(depth)
            yield Event("end", sample.ts, name, depth)

        held = [[name, held[d][1] + 1, held[d][2]] if d < len(held) and held[d][0] == name else [name, 1, False]
                for d, name in enumerate(cur)]
        for depth, entry in enumerate(held):
            if not entry[2] and entry[1] >= k and (depth == 0 or held[depth - 1][2]):
                entry[2] = True
                yield Event("start", sample.ts, entry[0], depth)


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
