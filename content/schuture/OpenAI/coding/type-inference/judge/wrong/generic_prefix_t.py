# expect: WA
# Treats an atomic name as a generic only when it starts with "T", so names like K or value are compared as primitives.
from typing import List, Union

BASE = frozenset(["int", "float", "str", "bool", "char"])


class ArityError(Exception):
    pass


class TypeMismatchError(Exception):
    pass


class GenericConflictError(Exception):
    pass


class Node:
    def __init__(self, value: Union[str, List["Node"]]) -> None:
        self.value = value if isinstance(value, str) else tuple(value)

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return self.value
        return "[%s]" % ",".join(map(str, self.value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return freeze(self) == freeze(other)

    __hash__ = None


class Function:
    def __init__(self, parameters: List[Node], return_type: Node) -> None:
        self.parameters = list(parameters)
        self.return_type = return_type

    def __str__(self) -> str:
        return "(%s) -> %s" % (",".join(map(str, self.parameters)), self.return_type)


def freeze(n):
    """Node -> nested python tuples / str (immutable, hashable)."""
    if isinstance(n.value, str):
        return n.value
    return tuple(freeze(c) for c in n.value)


def thaw(t):
    return Node(t) if isinstance(t, str) else Node([thaw(c) for c in t])


def get_return_type(parameters: List[Node], function: Function) -> Node:
    if len(parameters) != len(function.parameters):
        raise ArityError(f"expected {len(function.parameters)} arguments, got {len(parameters)}")
    env = {}
    stack = [(freeze(d), freeze(a)) for d, a in zip(function.parameters, parameters)]
    stack.reverse()
    while stack:
        d, a = stack.pop()
        if isinstance(d, str) and d not in BASE and d.startswith("T"):
            if d in env and env[d] != a:
                raise GenericConflictError(d)
            env.setdefault(d, a)
        elif isinstance(d, str) or isinstance(a, str) or len(d) != len(a):
            if d != a:
                raise TypeMismatchError(f"{d} vs {a}")
        else:
            stack.extend(reversed(list(zip(d, a))))

    def sub(t):
        if isinstance(t, str):
            return env[t] if t not in BASE else t
        return tuple(sub(c) for c in t)

    return thaw(sub(freeze(function.return_type)))
