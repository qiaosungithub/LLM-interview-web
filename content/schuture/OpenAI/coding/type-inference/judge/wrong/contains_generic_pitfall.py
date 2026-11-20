# expect: WA
# Asks whether the declared node contains a generic instead of whether it is one (the page's pitfall), so [T1,int] is bound whole and never checked.
from typing import List, Union


PRIMITIVES = {"int", "float", "str", "bool", "char"}


class Node:
    def __init__(self, value: Union[str, List["Node"]]) -> None:
        if isinstance(value, str):
            self.name, self.children = value, None
        else:
            self.name, self.children = None, list(value)   # NOTE: [] is a legal empty tuple, distinct from None

    def __str__(self) -> str:
        if self.children is None:
            return self.name
        return "[" + ",".join(str(c) for c in self.children) + "]"

    def __eq__(self, other) -> bool:
        return isinstance(other, Node) and str(self) == str(other)   # structural equality, via the string form

    def clone(self) -> "Node":
        if self.children is None:
            return Node(self.name)
        return Node([c.clone() for c in self.children])


class Function:
    def __init__(self, parameters: List[Node], return_type: Node) -> None:
        self.parameters, self.return_type = list(parameters), return_type

    def __str__(self) -> str:
        return "(" + ",".join(str(p) for p in self.parameters) + ") -> " + str(self.return_type)


class TypeInferenceError(Exception):
    pass


class ArityError(TypeInferenceError):
    pass


class TypeMismatchError(TypeInferenceError):
    pass


class GenericConflictError(TypeInferenceError):
    pass


def is_tuple(node: Node) -> bool:
    return node.children is not None


def is_generic_base(node: Node) -> bool:
    # NOTE: asks about `node` itself, never about a generic somewhere inside it -- see contains_generic below.
    return not is_tuple(node) and node.name not in PRIMITIVES


def contains_generic(node: Node) -> bool:
    if is_tuple(node):
        return any(contains_generic(c) for c in node.children)
    return is_generic_base(node)


def bind_generics(expected: Node, actual: Node, bindings: dict) -> None:
    if contains_generic(expected):                                    # 1. a generic: bind it, or check the old binding
        if expected.name in bindings:
            if bindings[expected.name] != actual:
                raise GenericConflictError(
                    f"{expected.name} is bound to two different types: {bindings[expected.name]} and {actual}")
        else:
            bindings[expected.name] = actual.clone()                  # NOTE: clone so `bindings` never aliases `actual`
        return
    if expected == actual:                                           # 2. both concrete, and already identical
        return
    if is_tuple(expected) and is_tuple(actual):                      # 3. neither is a generic: recurse into tuples
        if len(expected.children) != len(actual.children):
            raise TypeMismatchError(f"expected {expected} but received {actual}")
        for e, a in zip(expected.children, actual.children):
            bind_generics(e, a, bindings)
        return
    raise TypeMismatchError(f"expected {expected} but received {actual}")   # 4. concrete and different


def substitute_generics(node: Node, bindings: dict) -> Node:
    if not contains_generic(node):
        return node.clone()
    if is_generic_base(node):
        return bindings[node.name].clone()
    return Node([substitute_generics(c, bindings) for c in node.children])


def get_return_type(parameters: List[Node], function: Function) -> Node:
    if len(parameters) != len(function.parameters):
        raise ArityError(f"expected {len(function.parameters)} arguments, got {len(parameters)}")
    bindings: dict = {}
    for expected, actual in zip(function.parameters, parameters):
        bind_generics(expected, actual, bindings)
    return substitute_generics(function.return_type, bindings)
