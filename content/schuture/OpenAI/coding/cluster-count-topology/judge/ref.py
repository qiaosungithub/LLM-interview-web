"""The statement's two networks, tree generators, and the true count and topology of a tree.

Network and UnreliableNetwork behave exactly as in the statement (same queue, same random draws, so a seed gives
the same run as the copy in answer.py). The only addition: once `edges` is set, a send between two machines that
are not parent and child is recorded in `bad_send` and dropped, so the test can report it."""
import random


class Network:
    def __init__(self, seed=0):
        self.nodes = {}
        self._queue = []                 # list of (to_id, from_id, message), not yet delivered
        self._rng = random.Random(seed)
        self.edges = None                # judge only: allowed (from_id, to_id) pairs, None = unchecked
        self.bad_send = None             # judge only: the first forbidden send, as (from_id, to_id)

    def register(self, node):
        self.nodes[node.node_id] = node

    def _allowed(self, from_id, to_id):
        if self.edges is None or (from_id, to_id) in self.edges:
            return True
        if self.bad_send is None:
            self.bad_send = (from_id, to_id)
        return False

    def send_async_message(self, from_id, to_id, message):
        if self._allowed(from_id, to_id):
            self._queue.append((to_id, from_id, message))

    def run(self, max_rounds=10_000):
        rounds = 0
        while self._queue and rounds < max_rounds:
            batch, self._queue = self._queue, []
            self._rng.shuffle(batch)
            for to_id, from_id, message in batch:
                self.nodes[to_id].receive_message(from_id, message)
            rounds += 1
        return rounds


class UnreliableNetwork(Network):
    def __init__(self, drop_prob=0.0, duplicate_prob=0.0, seed=0):
        super().__init__(seed=seed)
        self.drop_prob = drop_prob
        self.duplicate_prob = duplicate_prob

    def send_async_message(self, from_id, to_id, message):
        if not self._allowed(from_id, to_id):
            return                       # judge only: checked before any random draw
        if self._rng.random() < self.drop_prob:
            return
        super().send_async_message(from_id, to_id, message)
        if self._rng.random() < self.duplicate_prob:
            super().send_async_message(from_id, to_id, message)

    def run(self, max_rounds=10_000, idle_rounds_to_stop=30):
        idle, rounds = 0, 0
        while rounds < max_rounds:
            if self._queue:
                batch, self._queue = self._queue, []
                self._rng.shuffle(batch)
                for to_id, from_id, message in batch:
                    self.nodes[to_id].receive_message(from_id, message)
                idle = 0
            else:
                idle += 1
            for node in self.nodes.values():
                node.on_tick()
            rounds += 1
            if idle >= idle_rounds_to_stop and not self._queue:
                break
        return rounds


# ---------------------------------------------------------------- trees: {node_id: [child ids]}, root listed first

def parents(children):
    return {c: p for p, kids in children.items() for c in kids}


def build(cls, children, network):
    """One cls instance per machine, created in the order of `children` (the order matters for on_tick), with
    `network` told which sends are allowed."""
    parent = parents(children)
    network.edges = {(p, c) for c, p in parent.items()} | {(c, p) for c, p in parent.items()}
    return {x: cls(x, parent.get(x), list(kids), network) for x, kids in children.items()}


def true_count(children, node=None):
    node = next(iter(children)) if node is None else node
    return 1 + sum(true_count(children, c) for c in children[node])


def true_topology(children, node=None):
    node = next(iter(children)) if node is None else node
    return (node, [true_topology(children, c) for c in children[node]])


def random_tree(r, n, reach=None):
    """n machines with distinct random string ids ("7", "42", "815": string order differs from number order and
    from the order children are listed in). Each new machine hangs under one of the `reach` most recent ones, so
    shapes range from a chain (reach 1) to a bushy random tree; by default `reach` is drawn too."""
    ids = [str(x) for x in r.sample(range(max(1000, 3 * n)), n)]
    reach = reach or r.choice([1, 2, 3, n])
    children = {x: [] for x in ids}
    for i in range(1, n):
        children[ids[r.randrange(max(0, i - reach), i)]].append(ids[i])
    return children


def chain(n):
    ids = [f"c{i}" for i in range(n)]
    return {x: ids[i + 1:i + 2] for i, x in enumerate(ids)}


def star(n):
    return {"hub": [f"s{i}" for i in range(1, n)]} | {f"s{i}": [] for i in range(1, n)}
