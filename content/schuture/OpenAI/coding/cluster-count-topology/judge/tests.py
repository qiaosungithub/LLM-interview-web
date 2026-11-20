from collections import Counter

from judgelib import calling, case, expect, expect_true, rng
import ref

EXAMPLE = {"A": ["B", "C", "D"], "B": ["E", "F"], "C": [], "D": ["G"], "E": [], "F": [], "G": ["H"], "H": []}
EXAMPLE_TOPOLOGY = ("A", [("B", [("E", []), ("F", [])]), ("C", []), ("D", [("G", [("H", [])])])])
CLASS = {"count": "CountingNode", "topology": "TopologyNode"}


# ---------------------------------------------------------------- how a run is shown and checked

def _setup(children, net_expr, cls_name):
    """Code that builds the same cluster in 自定义运行 (machines created in the same order, so the run is the same)."""
    tree = repr(children)
    if len(tree) > 600:
        tree = f"...   # {len(children)} 台机器的树，太长，这里不列出"
    return (f"children = {tree}\n"
            "parent = {c: p for p, kids in children.items() for c in kids}\n"
            f"net = {net_expr}\n"
            f"nodes = {{x: {cls_name}(x, parent.get(x), kids, net) for x, kids in children.items()}}\n")


def _check_sends(net, code):
    if net.bad_send is not None:
        frm, to = net.bad_send
        if to is None:
            why = "根节点没有父节点（parent_id 是 None），它算出的结果应存进自己的属性，而不是发给父节点"
        else:
            why = "每台机器只能给自己的父节点或某个子节点发消息"
        expect_true(False, f"机器 {frm!r} 给 {to!r} 发了消息：{why}", call=code)


def _attr(root, name, code):
    expect_true(hasattr(root, name), f"根节点没有 {name} 属性：查询完成后，结果应保存在根节点的 {name} 里", call=code)
    return getattr(root, name)


def _norm(t):
    """(id, [subtrees]) with lists and tuples accepted at either level."""
    if isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[1], (list, tuple)):
        return (t[0], [_norm(c) for c in t[1]])
    return t


def _topology_hint(got, want):
    """A sentence about the first machine whose children come back in another order, or ''."""
    if not (isinstance(got, tuple) and got[0] == want[0] and len(got[1]) == len(want[1])):
        return ""
    gids = [c[0] if isinstance(c, tuple) else c for c in got[1]]
    wids = [c[0] for c in want[1]]
    if gids != wids:
        if sorted(map(repr, gids)) == sorted(map(repr, wids)):
            return (f"机器 {want[0]!r} 的子树顺序不对：应按它的 children_ids 排列 {wids}，得到的顺序是 {gids}"
                    "（与应答到达的先后、id 的大小都无关）")
        return ""
    for g, w in zip(got[1], want[1]):
        if g != w:
            return _topology_hint(g, w)
    return ""


def _check_value(kind, got, want, call, missing=""):
    if got is None and missing:
        expect_true(False, missing, call=call)
    if kind == "topology":
        got = _norm(got)
        expect(got, want, call=call, msg=_topology_hint(got, want) if got != want else "")
    else:
        expect(got, want, call=call, msg=f"机器数不对：这棵树共有 {want} 台机器" if got != want else "")


# ---------------------------------------------------------------- Parts 1-2: one query on the reliable Network

def run_once(sol, kind, children, seed, want=None):
    if want is None:
        want = ref.true_count(children) if kind == "count" else ref.true_topology(children)
    root, start = next(iter(children)), ("request", kind)
    code = (_setup(children, f"Network(seed={seed})", CLASS[kind])
            + f"nodes[{root!r}].receive_message(None, {start!r})\nnet.run()\n")
    calling(code)
    net = ref.Network(seed=seed)
    nodes = ref.build(getattr(sol, CLASS[kind]), children, net)
    nodes[root].receive_message(None, start)
    net.run()
    _check_sends(net, code)
    got = _attr(nodes[root], "result", code)
    _check_value(kind, got, want, code + f"print(nodes[{root!r}].result)",
                 missing="run() 返回后根节点的 result 还是 None：查询没有完成")


def random_trees(r, sizes):
    return [ref.random_tree(r, n) for n in sizes]


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目例子：8 台机器")
def p1_example(sol):
    for seed in range(10):              # delivery order changes with the seed; the count does not
        run_once(sol, "count", EXAMPLE, seed, want=8)


@case("Part 1", title="只有根节点一台机器")
def p1_single_machine(sol):
    run_once(sol, "count", {"solo": []}, 0)


@case("Part 1", title="两台机器")
def p1_two_machines(sol):
    for seed in range(3):
        run_once(sol, "count", {"top": ["bottom"], "bottom": []}, seed)


@case("Part 1", title="星形：应答到达顺序任意")
def p1_star(sol):
    for seed in range(20):
        run_once(sol, "count", ref.star(9), seed)


@case("Part 1", title="链：每层只有一个子节点")
def p1_chain(sol):
    for n in (3, 10, 60):
        run_once(sol, "count", ref.chain(n), n)


@case("Part 1", title="随机树")
def p1_random(sol):
    r = rng()
    for children in random_trees(r, [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40] * 12):
        run_once(sol, "count", children, r.randrange(1000))


@case("Part 1", title="2000 台机器")
def p1_big(sol):
    r = rng()
    for reach in (3, 2000):
        run_once(sol, "count", ref.random_tree(r, 2000, reach), r.randrange(1000))


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目例子：8 台机器的拓扑")
def p2_example(sol):
    for seed in range(10):
        run_once(sol, "topology", EXAMPLE, seed, want=EXAMPLE_TOPOLOGY)


@case("Part 2", title="只有根节点一台机器")
def p2_single_machine(sol):
    run_once(sol, "topology", {"solo": []}, 0)


# children_ids deliberately out of id order, and the deepest subtree listed first, so neither arrival order nor
# sorting by id gives the children_ids order
ORDER_TREE = {"root": ["m", "z", "b", "a"], "m": ["q", "c"], "q": ["k"], "k": [], "c": [], "z": [], "b": ["y"],
              "y": [], "a": []}


@case("Part 2", title="子树顺序按 children_ids")
def p2_children_order(sol):
    for seed in range(10):
        run_once(sol, "topology", ORDER_TREE, seed)


@case("Part 2", title="星形与链")
def p2_star_chain(sol):
    for seed in range(5):
        run_once(sol, "topology", ref.star(9), seed)
    for n in (2, 10, 60):
        run_once(sol, "topology", ref.chain(n), n)


@case("Part 2", title="随机树")
def p2_random(sol):
    r = rng()
    for children in random_trees(r, [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40] * 12):
        run_once(sol, "topology", children, r.randrange(1000))


@case("Part 2", title="2000 台机器")
def p2_big(sol):
    r = rng()
    for reach in (3, 2000):
        run_once(sol, "topology", ref.random_tree(r, 2000, reach), r.randrange(1000))


# ---------------------------------------------------------------- Part 3: whole runs on UnreliableNetwork

# Only the statement's own example (0.2/0.2, seeds 0-999, the default run()) pins how soon a machine must retry.
# The other whole runs check the results, so they use a looser stop rule: an answer whose retry interval passes the
# example is not failed because a harder setting (drop 0.3, more seeds) happened to have a long silent stretch.
LOOSE = 100


def run_robust(sol, children, seed, drop, dup, queries, idle=30, repeat_start=False, wants=None):
    """Starts every (kind, request_id) in `queries` on the root, runs the network once, checks each result."""
    if wants is None:
        wants = {rid: ref.true_count(children) if kind == "count" else ref.true_topology(children)
                 for kind, rid in queries}
    root = next(iter(children))
    starts = [("request", kind, rid) for kind, rid in queries]
    if repeat_start:
        starts = [s for s in starts for _ in range(2)]
    run = "net.run()" if idle == 30 else f"net.run(idle_rounds_to_stop={idle})"
    code = (_setup(children, f"UnreliableNetwork(drop_prob={drop}, duplicate_prob={dup}, seed={seed})",
                   "RobustNode")
            + "".join(f"nodes[{root!r}].receive_message(None, {s!r})\n" for s in starts) + run + "\n")
    calling(code)
    net = ref.UnreliableNetwork(drop_prob=drop, duplicate_prob=dup, seed=seed)
    nodes = ref.build(sol.RobustNode, children, net)
    for s in starts:
        nodes[root].receive_message(None, s)
    net.run(idle_rounds_to_stop=idle)
    _check_sends(net, code)
    check_results(nodes[root], queries, wants, code, f"nodes[{root!r}]")
    return net, nodes


def check_results(root, queries, wants, code, expr):
    """`code` reproduces the run; `expr` is the root in it, e.g. "nodes['A']"."""
    results = _attr(root, "results", code + f"print({expr}.results)")
    for kind, rid in queries:
        expect_true(rid in results, f"run() 返回时根节点的 results 里还没有 {rid!r}：查询没有完成",
                    call=code + f"print({expr}.results)", got=results)
        _check_value(kind, results[rid], wants[rid], code + f"print({expr}.results[{rid!r}])")


@case("Part 3", sample=True, title="题目例子：种子 0 到 999")
def p3_example(sol):
    for seed in range(1000):
        run_robust(sol, EXAMPLE, seed, 0.2, 0.2, [("count", "q1")], wants={"q1": 8})


# ---------------------------------------------------------------- Part 3: the four rules, message by message

TREE4 = {"0": ["1"], "1": ["2", "3"], "2": [], "3": []}
R1 = ("request", "count", "r1")
WAIT = 29       # UnreliableNetwork.run() stops after 30 silent rounds, so a retry has to come before that
HELPERS = '''
def take(frm, to):                # remove the first queued message frm -> to (it is lost) and return it
    i = next(i for i, m in enumerate(net._queue) if (m[1], m[0]) == (frm, to))
    return net._queue.pop(i)

def deliver(frm, to, copies=1):   # deliver the first queued message frm -> to, `copies` times
    to, frm, message = take(frm, to)
    for _ in range(copies):
        nodes[to].receive_message(frm, message)

def sent():                       # (from, to) of every queued message
    return [(m[1], m[0]) for m in net._queue]

def drain():                      # deliver everything queued, and whatever that sends, oldest first, no ticks
    for _ in range(50):
        batch, net._queue = net._queue, []
        for to, frm, message in batch:
            nodes[to].receive_message(frm, message)

'''
PARENT4 = ref.parents(TREE4)
QUIET, LAST = 50, 30    # ticks after everything is done; none of the last 30 may send (run() needs 30 silent rounds)


class Script:
    """The tree 0 -> 1 -> {2, 3} on a plain Network that delivers nothing by itself: the test picks which message
    arrives, how many copies, which one is lost, and when a machine ticks. Messages are told apart only by
    sender and receiver, never by content (its format is the answer's own)."""

    def __init__(self, sol):
        self.net = ref.Network()
        self.nodes = ref.build(sol.RobustNode, TREE4, self.net)
        self.lines = []

    def code(self):
        steps = list(self.lines)
        if steps:
            steps[-1] += "   # <- 这一步"
        return (_setup(TREE4, "Network()", "RobustNode") + HELPERS + "\n".join(steps)
                + "\nprint(sent())   # 队列里每条消息的 (发送方, 接收方)")

    def _step(self, line):
        self.lines.append(line)
        calling(self.code())

    def _sends(self):
        return [(m[1], m[0]) for m in self.net._queue]

    def _new(self, action):
        n = len(self.net._queue)
        action()
        _check_sends(self.net, self.code())
        return [(m[1], m[0]) for m in self.net._queue[n:]]

    def _pop(self, frm, to):
        i = next((i for i, m in enumerate(self.net._queue) if (m[1], m[0]) == (frm, to)), None)
        assert i is not None, f"no queued message {frm} -> {to}"
        return self.net._queue.pop(i)

    def start(self):
        self._step(f"nodes['0'].receive_message(None, {R1!r})")
        return self._new(lambda: self.nodes["0"].receive_message(None, R1))

    def deliver(self, frm, to, copies=1):
        """The new sends after each copy."""
        self._step(f"deliver({frm!r}, {to!r}" + (f", copies={copies})" if copies > 1 else ")"))
        _, _, message = self._pop(frm, to)
        return [self._new(lambda: self.nodes[to].receive_message(frm, message)) for _ in range(copies)]

    def lose(self, frm, to):
        """Every queued message frm -> to is lost."""
        k = self._sends().count((frm, to))
        self._step(f"for _ in range({k}): take({frm!r}, {to!r})   # 丢失" if k > 1 else f"take({frm!r}, {to!r})   # 丢失")
        for _ in range(k):
            self._pop(frm, to)

    def tick(self, node, until):
        """Ticks `node` alone until it sends `until` (at most WAIT ticks); every send made meanwhile."""
        self._step(f"for _ in range({WAIT}): nodes[{node!r}].on_tick()")
        sent = []
        for k in range(1, WAIT + 1):
            sent += self._new(self.nodes[node].on_tick)
            if until in sent:
                break
        self.lines[-1] = f"for _ in range({k}): nodes[{node!r}].on_tick()"
        return sent

    def tick_all(self, times):
        """Every node ticks `times` times; the sends as (tick number from 1, (from, to))."""
        self._step(f"for _ in range({times}):\n    for node in nodes.values(): node.on_tick()")
        sent = []
        for k in range(1, times + 1):
            for node in self.nodes.values():
                sent += [(k, p) for p in self._new(node.on_tick)]
        return sent

    def drain(self):
        """Delivers every queued message, and whatever those deliveries send, oldest first, without ticking.
        For an answer whose machines send only requests and responses the queue is already empty here."""
        self._step("drain()   # 投递队列里剩下的所有消息（以及由此新发出的消息），不调用 on_tick")
        for _ in range(50):
            if not self.net._queue:
                return
            batch, self.net._queue = self.net._queue, []
            for to, frm, message in batch:
                self.nodes[to].receive_message(frm, message)
            _check_sends(self.net, self.code())
        self.check(not self.net._queue, "查询已经完成，但把队列里的消息逐轮投递 50 轮之后，机器之间还在来回发消息、"
                                        f"队列始终不空：{Counter(self._sends())}")

    def check(self, cond, msg, got=None):
        expect_true(cond, msg, call=self.code(), got=got)


def stage_in_progress(s):
    s.check(("0", "1") in s.start(), "根节点收到启动请求后，应在这次 receive_message 里立即把请求转发给子节点 '1'"
                                     "（题目：除超时重发外，各步都在 receive_message 里当场完成，on_tick 只用来重发）")
    new = s.start()
    s.check(not new, "根节点还在处理 'r1'，又收到同一个启动请求时必须忽略它（不转发、不应答、不重新开始汇聚）；"
                     f"你的代码又发出了这些消息（发送方, 接收方）：{new}")
    first, second = s.deliver("0", "1", copies=2)
    s.check(("1", "2") in first and ("1", "3") in first, "机器 '1' 收到请求后，应把请求转发给全部子节点 '2'、'3'")
    s.check(not second, "机器 '1' 还在处理 'r1'，又收到同一个请求（网络重复投递）时必须忽略它（不转发、不应答、不重新开始汇聚）；"
                        f"你的代码又发出了这些消息（发送方, 接收方）：{second}")


def stage_duplicate_answer(s):
    first, second = s.deliver("1", "2", copies=2)
    s.check(("2", "1") in first, "叶子 '2' 收到请求后应立即应答父节点 '1'")
    s.check(("2", "1") in second, "叶子 '2' 已经算完 'r1'，又收到同一个请求时必须把缓存的答案重发给父节点 '1'")
    new = s.deliver("2", "1")[0] + s.deliver("2", "1")[0]
    s.check(("1", "0") not in new, "机器 '1' 只收到了子节点 '2' 的应答（同一份答案的两次投递），还在等 '3'，"
                                   "此时不能向父节点汇报：同一个子节点对同一个 request_id 的两次应答不能都计入")


def stage_retry_one_child(s):
    s.lose("1", "3")
    for attempt in ("请求", "重发的请求"):
        sent = s.tick("1", until=("1", "3"))
        s.check(("1", "2") not in sent, "子节点 '2' 已经应答过了；超时重发只应发给还没有应答的子节点 '3'，不能向全部子节点重发")
        s.check(("1", "3") in sent, f"发给子节点 '3' 的{attempt}丢了，机器 '1' 等了 {WAIT} 个 tick 还没有向 '3' 重发"
                                    "（UnreliableNetwork.run() 连续 30 轮没有消息就会停止，重发必须远早于此）")
        if attempt == "请求":
            s.lose("1", "3")
    s.check(("3", "1") in s.deliver("1", "3")[0], "叶子 '3' 收到请求后应立即应答父节点 '1'")
    s.check(("1", "0") in s.deliver("3", "1")[0], "机器 '1' 已经收齐 '2' 和 '3' 的应答，应向父节点 '0' 汇报")


def stage_cached_replay(s):
    s.lose("1", "0")
    for lost in ("机器 '1' 的应答丢了", "机器 '1' 重发的应答也丢了"):
        sent = s.tick("0", until=("0", "1"))
        s.check(("0", "1") in sent, f"{lost}，根节点等了 {WAIT} 个 tick 还没有向 '1' 重发请求"
                                    "（UnreliableNetwork.run() 连续 30 轮没有消息就会停止，重发必须远早于此）")
        new = s.deliver("0", "1")[0]
        s.check(("1", "2") not in new and ("1", "3") not in new,
                "机器 '1' 已经算完 'r1'，又收到同一个请求时应直接重发缓存的答案，不能重新询问子节点 '2'、'3'")
        s.check(("1", "0") in new, "机器 '1' 已经算完 'r1'，每次收到同一个请求都必须把缓存的答案重发给父节点 '0'"
                                   "（只重发一次不够：那一份也可能丢）")
        if lost == "机器 '1' 的应答丢了":
            s.lose("1", "0")
    s.deliver("1", "0", copies=2)
    check_results(s.nodes["0"], [("count", "r1")], {"r1": 4}, s.code() + "\n", "nodes['0']")


def stage_done(s):
    s.drain()
    sent = s.tick_all(QUIET)
    down = [p for _, p in sent if PARENT4.get(p[1]) == p[0]]
    s.check(not down, "查询已经完成，每个子节点都已应答过；之后的 tick 不应再向子节点重发请求，"
                      f"你的代码发出了这些消息（发送方, 接收方）：{Counter(down)}")
    late = [p for k, p in sent if k > QUIET - LAST]
    s.check(not late, f"查询已经完成、所有消息都已投递，之后 {QUIET} 个 tick 里的最后 {LAST} 个仍有机器在发消息："
                      f"{Counter(late)}。这样 UnreliableNetwork.run() 永远等不到连续 30 轮空队列，只能跑到 max_rounds")
    new = s.start()
    s.check(("0", "1") not in new, "根节点已经算完 'r1'，又收到同一个启动请求时应直接使用缓存的答案，不能重新询问子节点")
    check_results(s.nodes["0"], [("count", "r1")], {"r1": 4}, s.code() + "\n", "nodes['0']")


STAGES = [stage_in_progress, stage_duplicate_answer, stage_retry_one_child, stage_cached_replay, stage_done]


def script(sol, upto):
    s = Script(sol)
    for stage in STAGES[:upto]:
        stage(s)


@case("Part 3", title="处理中收到重复请求：忽略")
def p3_rule_ignore_in_progress(sol):
    script(sol, 1)


@case("Part 3", title="同一子节点的两次应答只计一次")
def p3_rule_count_child_once(sol):
    script(sol, 2)


@case("Part 3", title="超时只向没应答的子节点重发")
def p3_rule_retry_one_child(sol):
    script(sol, 3)


@case("Part 3", title="算完后重发缓存的答案")
def p3_rule_cached_replay(sol):
    script(sol, 4)


@case("Part 3", title="完成后不再重发")
def p3_rule_quiet_when_done(sol):
    script(sol, 5)


# ---------------------------------------------------------------- Part 3: more whole runs

@case("Part 3", cpu=4, title="例子再换 2000 个种子")
def p3_example_more_seeds(sol):
    for seed in range(1000, 3000):
        run_robust(sol, EXAMPLE, seed, 0.2, 0.2, [("count", "q1")], idle=LOOSE)


@case("Part 3", title="只有根节点一台机器")
def p3_single_machine(sol):
    for seed in range(5):
        run_robust(sol, {"solo": []}, seed, 0.3, 0.3, [("count", "c"), ("topology", "t")])


@case("Part 3", title="不丢不重：与 Part 1、2 相同")
def p3_no_faults(sol):
    r = rng()
    for i, children in enumerate(random_trees(r, [2, 3, 5, 8, 13, 21, 34] * 3) + [ref.chain(101)]):
        run_robust(sol, children, r.randrange(1000), 0.0, 0.0, [("count" if i % 2 else "topology", f"q{i}")])


@case("Part 3", title="只有重复、没有丢失")
def p3_duplicates_only(sol):
    r = rng()
    for i, children in enumerate(random_trees(r, [2, 3, 4, 6, 9, 14, 20] * 6)):
        run_robust(sol, children, r.randrange(1000), 0.0, 0.5, [("count" if i % 2 else "topology", f"d{i}")])


@case("Part 3", title="随机树：计数")
def p3_random_count(sol):
    r = rng()
    for i, children in enumerate(random_trees(r, [2, 3, 4, 6, 9, 14, 20] * 10)):
        drop, dup = r.choice([(0.1, 0.1), (0.2, 0.3), (0.3, 0.2), (0.3, 0.0), (0.2, 0.2)])
        run_robust(sol, children, r.randrange(1000), drop, dup, [("count", f"c{i}")], idle=LOOSE,
                   repeat_start=i % 4 == 0)


@case("Part 3", title="随机树：拓扑")
def p3_random_topology(sol):
    r = rng()
    for i, children in enumerate(random_trees(r, [2, 3, 4, 6, 9, 14, 20] * 10)):
        drop, dup = r.choice([(0.1, 0.1), (0.2, 0.3), (0.3, 0.2), (0.3, 0.0), (0.2, 0.2)])
        run_robust(sol, children, r.randrange(1000), drop, dup, [("topology", f"t{i}")], idle=LOOSE,
                   repeat_start=i % 4 == 0)


@case("Part 3", title="两个查询同时进行")
def p3_two_queries(sol):
    for seed in range(50):
        run_robust(sol, EXAMPLE, seed, 0.3, 0.3, [("count", "c"), ("topology", "t")], idle=LOOSE)


@case("Part 3", title="两个同类查询同时进行")
def p3_same_kind_at_once(sol):
    """Two count queries and two topology queries in flight together: state must be kept per request_id."""
    for seed in range(20):
        run_robust(sol, EXAMPLE, seed, 0.2, 0.2,
                   [("count", "c1"), ("count", "c2"), ("topology", "t1"), ("topology", "t2")], idle=LOOSE)


@case("Part 3", title="同一批机器上先后两次查询")
def p3_one_after_another(sol):
    for seed in range(30):
        net, nodes = run_robust(sol, EXAMPLE, seed, 0.2, 0.2, [("count", "first")], idle=LOOSE)
        code = (_setup(EXAMPLE, f"UnreliableNetwork(drop_prob=0.2, duplicate_prob=0.2, seed={seed})", "RobustNode")
                + f"nodes['A'].receive_message(None, ('request', 'count', 'first'))\nnet.run(idle_rounds_to_stop={LOOSE})\n"
                + f"nodes['A'].receive_message(None, ('request', 'topology', 'second'))\nnet.run(idle_rounds_to_stop={LOOSE})\n")
        calling(code)
        nodes["A"].receive_message(None, ("request", "topology", "second"))
        net.run(idle_rounds_to_stop=LOOSE)
        _check_sends(net, code)
        check_results(nodes["A"], [("count", "first"), ("topology", "second")],
                      {"first": 8, "second": EXAMPLE_TOPOLOGY}, code, "nodes['A']")


@case("Part 3", cpu=6, title="300 台机器，丢失与重复各 0.5")
def p3_big(sol):
    r = rng()
    for i in range(3):
        children = ref.random_tree(r, 300)
        run_robust(sol, children, r.randrange(1000), 0.5, 0.5, [("topology", f"big{i}")], idle=200)
