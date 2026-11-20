# expect: WA
# Handles duplicates but never resends: on_tick does nothing, so a single lost message leaves the query unfinished.
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


QUERIES = {}   # query_type -> (leaf_value(node), combine(node, child_values in children_ids order))


class QueryNode(Node):
    """Broadcasts a request down the tree and aggregates the answers back up."""

    def __init__(self, node_id, parent_id, children_ids, network):
        super().__init__(node_id, parent_id, children_ids, network)
        self.result = None     # set on the ROOT only, once the query has completed
        self._pending = {}     # query_type -> set of children that have not answered yet
        self._answers = {}     # query_type -> {child_id: value}  NOTE: per child, not a running total

    def _respond(self, query_type, value):
        if self.parent_id is None:
            self.result = value
        else:
            self.send(self.parent_id, ("response", query_type, value))

    def receive_message(self, from_id, message):
        kind = message[0]
        if kind == "request":             # from the parent, or from outside for the root
            _, query_type = message
            leaf_value, _combine = QUERIES[query_type]
            if not self.children_ids:
                self._respond(query_type, leaf_value(self))     # a leaf answers immediately
            else:
                self._pending[query_type] = set(self.children_ids)
                self._answers[query_type] = {}
                for child in self.children_ids:
                    self.send(child, ("request", query_type))
        elif kind == "response":          # from one child
            _, query_type, value = message
            pending = self._pending.get(query_type, ())
            if from_id not in pending:
                return                    # NOTE: not a child we are still waiting on
            self._answers[query_type][from_id] = value
            pending.discard(from_id)
            if not pending:               # every child has now answered
                _, combine = QUERIES[query_type]
                ordered = [self._answers[query_type][c] for c in self.children_ids]
                self._respond(query_type, combine(self, ordered))


QUERIES["count"] = (
    lambda node: 1,
    lambda node, child_counts: 1 + sum(child_counts),
)
CountingNode = QueryNode      # Part 1: start it with ("request", "count")


QUERIES["topology"] = (
    lambda node: (node.node_id, []),
    lambda node, child_topologies: (node.node_id, child_topologies),
)
TopologyNode = QueryNode      # Part 2: the same class, started with ("request", "topology")


class RobustNode(Node):
    """QueryNode's protocol over the same QUERIES table, with all state kept per request_id."""

    TIMEOUT = 3   # ticks to wait for a child's answer before asking that child again

    def __init__(self, node_id, parent_id, children_ids, network):
        super().__init__(node_id, parent_id, children_ids, network)
        self.results = {}   # request_id -> final value, on the ROOT only
        self._state = {}    # request_id -> see _new_state; kept forever, it is also the cache
        self._tick = 0

    def _new_state(self, query_type):
        return {
            "query_type": query_type,
            "pending": set(self.children_ids),
            "answers": {},   # NOTE: keyed by child_id, never `+=`
            "done": False,
            "value": None,
            "last_sent": {c: self._tick for c in self.children_ids},
        }

    def receive_message(self, from_id, message):
        kind = message[0]
        if kind == "request":
            _, query_type, request_id = message
            state = self._state.get(request_id)
            if state is not None:
                if state["done"]:
                    self._reply(request_id)   # NOTE: cached replay, nothing is recomputed
                return                        # NOTE: still in progress -> ignore, do not restart
            leaf_value, _combine = QUERIES[query_type]
            self._state[request_id] = self._new_state(query_type)
            if not self.children_ids:
                self._finish(request_id, leaf_value(self))
            else:
                for child in self.children_ids:
                    self.send(child, ("request", query_type, request_id))
        elif kind == "response":
            _, request_id, value = message
            state = self._state.get(request_id)
            if state is None or state["done"] or from_id not in state["pending"]:
                return   # NOTE: a repeated or late answer: this child is already recorded
            state["answers"][from_id] = value
            state["pending"].discard(from_id)
            if not state["pending"]:
                _, combine = QUERIES[state["query_type"]]
                ordered = [state["answers"][c] for c in self.children_ids]
                self._finish(request_id, combine(self, ordered))

    def _finish(self, request_id, value):
        state = self._state[request_id]
        state["done"], state["value"] = True, value
        self._reply(request_id)

    def _reply(self, request_id):
        state = self._state[request_id]
        if self.parent_id is None:
            self.results[request_id] = state["value"]
        else:
            self.send(self.parent_id, ("response", request_id, state["value"]))

    def on_tick(self):
        self._tick += 1
        return
        for request_id, state in self._state.items():
            if state["done"]:
                continue
            for child in self.children_ids:   # NOTE: not the set: its order changes from run to run
                if child in state["pending"] and self._tick - state["last_sent"][child] >= self.TIMEOUT:
                    self.send(child, ("request", state["query_type"], request_id))
                    state["last_sent"][child] = self._tick
