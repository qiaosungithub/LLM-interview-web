# expect: WA
# Rejects a state given as a list, so a state that went through JSON is refused with TypeError.
from abc import ABC, abstractmethod


class ResumableIterator(ABC):
    @abstractmethod
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        ...

    @abstractmethod
    def get_state(self):
        ...

    @abstractmethod
    def set_state(self, state) -> None:
        ...


def _check_index_tuple(state, length):
    # NOTE: bool is a subclass of int, so isinstance(True, int) is True
    if (not isinstance(state, tuple) or len(state) != length
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in state)):
        raise TypeError(f"expected a {length}-tuple of ints, got {state!r}")
    return state


class ResumableListIterator(ResumableIterator):
    def __init__(self, items):
        self.items = items
        self._pos = 0                                 # index of the next element to produce

    def __iter__(self):
        return self

    def __next__(self):
        pos = self._pos
        if pos == len(self.items):                    # the exhausted state; set_state never goes past it
            raise StopIteration
        self._pos = pos + 1
        return self.items[pos]

    def get_state(self):
        return (self._pos,)

    def set_state(self, state):
        (pos,) = _check_index_tuple(state, 1)
        if not 0 <= pos <= len(self.items):          # NOTE: '<=': pos == len(items) is exhausted
            raise ValueError(f"{state!r} is not a canonical position")
        self._pos = pos


class Resumable2DIterator(ResumableIterator):
    def __init__(self, items):
        self.items = items
        self._outer = 0
        self._inner = 0
        self._skip_empty()

    def _skip_empty(self):
        while self._outer < len(self.items) and self._inner >= len(self.items[self._outer]):
            self._outer += 1
            self._inner = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._outer >= len(self.items):
            raise StopIteration
        item = self.items[self._outer][self._inner]
        self._inner += 1
        self._skip_empty()              # NOTE: right away, not before producing the next item
        return item

    def get_state(self):
        return (self._outer, self._inner)

    def set_state(self, state):
        outer, inner = _check_index_tuple(state, 2)
        exhausted = (outer, inner) == (len(self.items), 0)
        inside = 0 <= outer < len(self.items) and 0 <= inner < len(self.items[outer])
        if not (exhausted or inside):   # NOTE: strict '<': inner == len(row) is not canonical
            raise ValueError(f"{state!r} is not a canonical position")
        self._outer, self._inner = outer, inner


class ResumableNestedIterator(ResumableIterator):
    def __init__(self, items, depth):
        self.items = items
        self.depth = depth
        self._stack = [(items, 0)]
        self._advance()

    def _advance(self):
        while self._stack:
            container, idx = self._stack[-1]
            if idx >= len(container):
                self._stack.pop()
                if self._stack:
                    parent, pidx = self._stack[-1]
                    self._stack[-1] = (parent, pidx + 1)
                continue
            if len(self._stack) < self.depth:
                self._stack.append((container[idx], 0))
            else:
                return                   # top of stack points at the next item

    def __iter__(self):
        return self

    def __next__(self):
        if not self._stack:
            raise StopIteration
        container, idx = self._stack[-1]
        item = container[idx]
        self._stack[-1] = (container, idx + 1)
        self._advance()
        return item

    def _exhausted_state(self):
        return (len(self.items),) + (0,) * (self.depth - 1)

    def get_state(self):
        if not self._stack:              # NOTE: the empty stack alone would give (), not a d-tuple
            return self._exhausted_state()
        return tuple(idx for _, idx in self._stack)

    def set_state(self, state):
        indices = _check_index_tuple(state, self.depth)
        if indices == self._exhausted_state():
            self._stack = []
            return
        stack, container = [], self.items
        for idx in indices:
            if not 0 <= idx < len(container):    # NOTE: the same strict '<' at every level
                raise ValueError(f"{state!r} is not a canonical position")
            stack.append((container, idx))
            container = container[idx]
        self._stack = stack
