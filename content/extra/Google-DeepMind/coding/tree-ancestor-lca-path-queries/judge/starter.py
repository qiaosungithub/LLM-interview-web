class RootedTree:
    def __init__(self, parent: list[int], weight: list[int] | None = None) -> None:
        """Builds the tree from its parent array (ValueError if it is not a rooted tree); weight[v] is the weight of the edge v - parent[v]."""
        raise NotImplementedError

    def kth_ancestor(self, v: int, k: int) -> int:
        """The node k edges above v, or -1 if there is none."""
        raise NotImplementedError

    def lca(self, u: int, v: int) -> int:
        """Lowest common ancestor of u and v."""
        raise NotImplementedError

    def distance(self, u: int, v: int) -> int:
        """Number of edges on the path between u and v."""
        raise NotImplementedError

    def path_max(self, u: int, v: int) -> int | None:
        """Largest edge weight on the path between u and v, or None when u == v."""
        raise NotImplementedError
