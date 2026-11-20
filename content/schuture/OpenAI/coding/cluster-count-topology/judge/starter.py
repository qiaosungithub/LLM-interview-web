import random


class Network:
    """A minimal asynchronous network. send_async_message only enqueues a message; nothing is
    delivered synchronously. run() repeatedly takes every message currently queued, shuffles it
    with the network's own random generator (delivery order has nothing to do with send order),
    and calls receive_message on each target in that shuffled order. A handler invoked while a
    round is being delivered may itself call send_async_message; those new messages join the queue
    for the NEXT round, never the one being delivered, so a message can never be processed before
    it is sent. run() stops once the queue is empty, or after max_rounds as a safety net against a
    handler that keeps sending forever."""

    def __init__(self, seed=0):
        self.nodes = {}
        self._queue = []                 # list of (to_id, from_id, message), not yet delivered
        self._rng = random.Random(seed)  # NOTE: seeded, so a run is reproducible for a fixed seed

    def register(self, node):
        self.nodes[node.node_id] = node

    def send_async_message(self, from_id, to_id, message):
        self._queue.append((to_id, from_id, message))

    def run(self, max_rounds=10_000):
        rounds = 0
        while self._queue and rounds < max_rounds:
            batch, self._queue = self._queue, []       # NOTE: freeze this round's batch first, so
            self._rng.shuffle(batch)                    #       sends made while delivering it land
            for to_id, from_id, message in batch:       #       in self._queue for the NEXT round
                self.nodes[to_id].receive_message(from_id, message)
            rounds += 1
        return rounds


class Node:
    """Given skeleton, subclassed for every part. node_id is this machine's id, parent_id its
    parent's id (None only for the root), children_ids the ids of its children, and network the
    Network instance to send through."""

    def __init__(self, node_id, parent_id, children_ids, network):
        self.node_id = node_id
        self.parent_id = parent_id
        self.children_ids = list(children_ids)
        self.network = network
        network.register(self)

    def send(self, to_id, message):
        self.network.send_async_message(self.node_id, to_id, message)

    def receive_message(self, from_id, message):
        raise NotImplementedError

    def on_tick(self):
        """Called once per round by UnreliableNetwork.run() (Part 3); unused before that, and the
        base Network above never calls it."""


class UnreliableNetwork(Network):
    """A message is independently dropped with probability drop_prob (never delivered at all) and,
    if not dropped, duplicated with probability duplicate_prob (a second, unrelated copy is queued,
    so receive_message runs twice for that one send). After every round -- including one where the
    queue happened to be empty -- on_tick() is called once on every registered node; this is the
    model's only notion of time, and it is what a node uses to notice that a child has been silent
    for too long and resend to it. run() stops once the queue has been empty for
    idle_rounds_to_stop consecutive rounds, or after max_rounds, whichever comes first."""

    def __init__(self, drop_prob=0.0, duplicate_prob=0.0, seed=0):
        super().__init__(seed=seed)
        self.drop_prob = drop_prob
        self.duplicate_prob = duplicate_prob

    def send_async_message(self, from_id, to_id, message):
        if self._rng.random() < self.drop_prob:
            return                                        # NOTE: silently discarded, never queued
        super().send_async_message(from_id, to_id, message)
        if self._rng.random() < self.duplicate_prob:
            super().send_async_message(from_id, to_id, message)   # a second, independent delivery

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
                node.on_tick()                            # NOTE: ticks even on an idle round, so a
            rounds += 1                                    #       pending timeout still fires
            if idle >= idle_rounds_to_stop and not self._queue:
                break
        return rounds


# ---------------------------------------------------------------- your answer

class CountingNode(Node):
    def __init__(self, node_id, parent_id, children_ids, network):
        super().__init__(node_id, parent_id, children_ids, network)

    def receive_message(self, from_id: str | None, message) -> None:
        """After Network.run() returns, self.result on the ROOT holds the number of machines in the whole cluster."""
        raise NotImplementedError


class TopologyNode(Node):
    def __init__(self, node_id, parent_id, children_ids, network):
        super().__init__(node_id, parent_id, children_ids, network)

    def receive_message(self, from_id: str | None, message) -> None:
        """After Network.run() returns, self.result on the ROOT holds (node_id, [child topologies])."""
        raise NotImplementedError


class RobustNode(Node):
    def __init__(self, node_id, parent_id, children_ids, network):
        super().__init__(node_id, parent_id, children_ids, network)

    def receive_message(self, from_id: str | None, message) -> None:
        """Parts 1-2 over an UnreliableNetwork; the root stores each finished query in self.results[request_id]."""
        raise NotImplementedError

    def on_tick(self) -> None:
        """Called once per round by UnreliableNetwork.run(); resend to any child pending too long."""
        raise NotImplementedError
