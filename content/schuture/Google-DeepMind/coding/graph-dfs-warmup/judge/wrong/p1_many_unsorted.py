# expect: WA
# Part 1 count_above_many binary-searches the array without sorting it first.

def above_threshold(arr: list[int], threshold: int) -> tuple[int, list[int]]:
    indices = [i for i, x in enumerate(arr) if x > threshold]
    return len(indices), indices


from bisect import bisect_right


def count_above_many(arr: list[int], thresholds: list[int]) -> list[int]:
    sorted_arr = arr
    n = len(sorted_arr)
    # NOTE: bisect_right, not bisect_left -- bisect_left's insertion point sits before every element
    # equal to t, so n - bisect_left(sorted_arr, t) would also count those equal elements as "above"
    return [n - bisect_right(sorted_arr, t) for t in thresholds]


def build_graph(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    neighbours: dict[str, set[str]] = {}
    for u, v in edges:
        neighbours.setdefault(u, set())
        neighbours.setdefault(v, set())
        if u != v:
            neighbours[u].add(v)
            neighbours[v].add(u)
        # NOTE: u == v (a self-loop) falls through to here -- u is registered as a node above, but
        # gets no neighbour; a repeated edge adds nothing new, since neighbours are kept in a set
    return {node: sorted(adj) for node, adj in neighbours.items()}


def dfs_order_recursive(edges: list[tuple[str, str]], start: str) -> list[str]:
    graph = build_graph(edges)
    visited = {start}
    order: list[str] = []

    def visit(node: str) -> None:
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                visit(neighbour)   # NOTE: one Python call frame stays open per node on this branch

    visit(start)
    return order


def dfs_order(edges: list[tuple[str, str]], start: str) -> list[str]:
    graph = build_graph(edges)
    visited: set[str] = set()
    order: list[str] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue           # NOTE: a node is pushed once by each neighbour expanded before it is
                               # visited; every pop of it after the first is skipped here
        visited.add(node)      # NOTE: mark at pop time, not push time. Marking at push time turns edges
                               # A-B, A-C, B-D, B-E, C-D from A into A, B, D, E, C (C, marked when A
                               # pushed it, is skipped from D), not the DFS preorder A, B, D, C, E.
        order.append(node)
        for neighbour in reversed(graph.get(node, [])):
            if neighbour not in visited:
                stack.append(neighbour)
    return order


def components(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph = build_graph(edges)
    visited: set[str] = set()
    result: list[list[str]] = []
    for node in sorted(graph):
        if node in visited:
            continue            # NOTE: sorted() -- the first unvisited node here is always the
                                # smallest node of a component nothing has reached yet
        order: list[str] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            order.append(cur)
            for neighbour in reversed(graph.get(cur, [])):
                if neighbour not in visited:
                    stack.append(neighbour)
        result.append(order)
    return result


def has_cycle(edges: list[tuple[str, str]]) -> bool:
    graph = build_graph(edges)
    visited: set[str] = set()
    for node in graph:
        if node in visited:
            continue
        visited.add(node)
        stack = [(node, None)]              # (current node, the node we arrived from)
        while stack:
            cur, parent = stack.pop()
            for neighbour in graph[cur]:
                if neighbour == parent:
                    continue                # NOTE: skip only the edge back to where we came from --
                                            # build_graph already merged any repeat of it away
                if neighbour in visited:
                    return True             # a visited neighbour that is not the parent closes a cycle
                visited.add(neighbour)      # NOTE: marked at discovery (push) time -- see above
                stack.append((neighbour, cur))
    return False
