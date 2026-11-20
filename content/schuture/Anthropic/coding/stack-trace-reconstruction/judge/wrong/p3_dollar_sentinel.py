# expect: RE
# keeps each trie node's ids under the string key "$", which collides with a frame named "$".
from collections import deque
from collections.abc import Iterable, Iterator
from itertools import takewhile
from typing import Literal, NamedTuple


class Sample(NamedTuple):
    ts: float
    stack: list[str]


class Event(NamedTuple):
    kind: Literal["start", "end"]
    ts: float
    name: str
    depth: int


def _lcp(a, b):
    return sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(a, b)))


def convert_samples_to_events(samples: list[Sample], end_time: float) -> list[Event]:
    out = []
    before = []
    for s in samples:
        p = _lcp(before, s.stack)
        out += [Event("end", s.ts, before[d], d) for d in reversed(range(p, len(before)))]
        out += [Event("start", s.ts, s.stack[d], d) for d in range(p, len(s.stack))]
        before = list(s.stack)
    if samples:
        out += [Event("end", end_time, before[d], d) for d in reversed(range(len(before)))]
    return out


def stream_debounced_events(samples: Iterable[Sample], k: int) -> Iterator[Event]:
    names = []      # current stack
    entered = []    # sample index at which the frame at each depth entered
    confirmed = []  # whether it has been reported
    closing = deque()  # (index of the sample that reports the end, depth, name), FIFO
    i = -1
    for s in samples:
        i += 1
        stack = list(s.stack)
        p = _lcp(names, stack)
        for d in range(len(names) - 1, p - 1, -1):
            if confirmed[d]:
                closing.append((i + k - 1, d, names[d]))
        del names[p:], entered[p:], confirmed[p:]
        for d in range(p, len(stack)):
            names.append(stack[d])
            entered.append(i)
            confirmed.append(False)
        while closing and closing[0][0] == i:
            _, d, name = closing.popleft()
            yield Event("end", s.ts, name, d)
        for d in range(len(names)):
            if not confirmed[d] and i - entered[d] + 1 == k:
                confirmed[d] = True
                yield Event("start", s.ts, names[d], d)


class _Node:
    __slots__ = ("kids", "ids")

    def __init__(self):
        self.kids = {}
        self.ids = []


class TraceIndex:
    def __init__(self, traces: list[tuple[str, list[str]]]) -> None:
        self._root = {}
        for trace_id, frames in traces:
            node = self._root
            for frame in reversed(frames):
                node = node.setdefault(frame, {})
                node.setdefault("$", []).append(trace_id)
        self._sort(self._root)

    def _sort(self, node):
        if "$" in node:
            node["$"].sort()
        for key, child in node.items():
            if key != "$":
                self._sort(child)

    def match_suffix(self, frames: list[str]) -> list[str] | str:
        node = self._root
        for frame in reversed(frames):
            if frame not in node:
                return "UNKNOWN"
            node = node[frame]
        return list(node.get("$", [])) or "UNKNOWN"
