from typing import List, Union


class Node:
    def __init__(self, value: Union[str, List["Node"]]) -> None:
        """An atomic node (value is its name) or a tuple node (value is the list of its element types)."""
        raise NotImplementedError

    def __str__(self) -> str:
        """The name of an atomic node, or "[t1,t2,...]" for a tuple node."""
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        """Structural equality: equal iff the string forms are equal."""
        raise NotImplementedError


class Function:
    def __init__(self, parameters: List[Node], return_type: Node) -> None:
        """A function with these parameter types and this return type."""
        raise NotImplementedError

    def __str__(self) -> str:
        """"(p1,p2,...) -> returnType"."""
        raise NotImplementedError


# Part 2
class ArityError(Exception):
    """The call has the wrong number of arguments."""


class TypeMismatchError(Exception):
    """Two types differ at a position with no generic involved."""


class GenericConflictError(Exception):
    """One generic would have to bind to two structurally different types."""


def get_return_type(parameters: List[Node], function: Function) -> Node:
    """Unifies function.parameters with the concrete parameters and returns the substituted return type."""
    raise NotImplementedError
