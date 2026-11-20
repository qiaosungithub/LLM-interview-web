import ast
import time

from judgelib import Mismatch, calling, case, expect, rng
import ref

SHOW = 18                        # a replay lists at most this many calls
MUTATORS = {"add_user", "follow", "unfollow"}


class _Raise:
    def __repr__(self):
        return "ValueError"


RAISE = _Raise()                 # the step must raise ValueError
SKIP = object()                  # the step's return value is not checked (it must not raise)


# ---------------------------------------------------------------- how a sequence is shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str) and '"' not in x and "\\" not in x and x.isprintable():
        return f'"{x}"'
    if isinstance(x, (list, tuple)):
        inner = ", ".join(_lit(v) for v in x)
        return f"[{inner}]" if isinstance(x, list) else f"({inner}{',' if len(x) == 1 else ''})"
    return repr(x)


def _code(op):
    name, obj, method, args = op
    call = f"{method}({', '.join(_lit(a) for a in args)})"
    if obj is not None:
        call = f"{obj}.{call}"
    return f"{name} = {call}" if name else call


def _replay(ops, wants, k, mark="这一步", show=SHOW):
    """ops[:k + 1] as code to paste into 自定义运行; wants[i] says what step i returned (or that it raised)."""
    entries = []
    for i in range(k + 1):
        op, want, code = ops[i], wants[i], _code(ops[i])
        if i == k:
            note = f"# <- {mark}" + ("：应当抛出 ValueError" if want is RAISE else "")
            entries.append([(f"print({code})" if want not in (SKIP, RAISE) else code, note)])
        elif want is RAISE:
            entries.append([("try: " + code, ""), ("except ValueError: pass", "# 抛出 ValueError")])
        elif want is SKIP:
            entries.append([(code, "")])
        else:
            entries.append([(f"print({code})", f"# -> {_lit(want)}")])
    head = [e for i, e in enumerate(entries) if ops[i][1] is None]          # constructors are always kept
    body = [e for i, e in enumerate(entries) if ops[i][1] is not None]
    skipped = max(0, len(body) - show)
    rows = [row for e in head for row in e]
    if skipped:
        rows.append((f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）", ""))
    rows += [row for e in body[skipped:] for row in e]
    width = max(len(code) for code, note in rows if note) + 3 if any(note for _, note in rows) else 0
    return "\n".join(code.ljust(width) + note if note else code for code, note in rows)


# ---------------------------------------------------------------- running a sequence

def _same(got, want):
    if isinstance(want, list) and isinstance(got, tuple):
        got = list(got)
    try:
        return bool(got == want)
    except Exception:
        return False


def _model_step(model, op):
    name, obj, method, args = op
    fn = ref.CLASSES[method] if obj is None else getattr(model[obj], method)
    try:
        out = fn(*args)
    except ValueError:
        return RAISE
    if name:
        model[name] = out
        return SKIP
    return SKIP if method in MUTATORS else out


def _outcome(sol, ops, wants=None, show=SHOW):
    """Runs ops on fresh objects of the answer, checking each step as soon as it returns. The expected results come
    from `wants`, or from ref when wants is None. Returns None when every step is right, else what went wrong."""
    env, model, seen = {}, {}, []
    for k, op in enumerate(ops):
        name, obj, method, args = op
        want = _model_step(model, op) if wants is None else wants[k]
        seen.append(want)
        target = sol if obj is None else env[obj]
        fn = getattr(target, method, None)
        if not callable(fn):
            where = "答案" if obj is None else type(target).__name__
            raise Mismatch(f"{where} 没有 {method}（名字要和题目一致）", call=_replay(ops, seen, k, show=show), show=False)
        calling(_replay(ops, seen, k, show=show))           # shown if the answer raises or hangs in this call
        try:
            got = fn(*args)
        except Exception as e:
            if want is RAISE and isinstance(e, ValueError):
                continue
            return {"k": k, "kind": "wrong_exc" if want is RAISE else "exc", "exc": e, "wants": seen}
        if want is RAISE:
            return {"k": k, "kind": "no_raise", "got": got, "wants": seen}
        if name:
            env[name] = got
        elif want is not SKIP and not _same(got, want):
            return {"k": k, "kind": "value", "got": got, "want": want, "wants": seen}
    return None


def _report(ops, bad, msg, show=SHOW):
    k, kind = bad["k"], bad["kind"]
    if kind == "exc":
        calling(_replay(ops, bad["wants"], k, mark="这一步抛出异常", show=show))
        raise bad["exc"]
    call = _replay(ops, bad["wants"], k, show=show)
    if kind == "wrong_exc":
        e = bad["exc"]
        raise Mismatch(f"{msg}应当抛出 ValueError，实际抛出的是 {type(e).__name__}: {e}", call=call, show=False)
    if kind == "no_raise":
        raise Mismatch(f"{msg}应当抛出 ValueError，却正常返回了", call=call, got=bad["got"])
    expect(bad["got"], bad["want"], call=call, msg=msg)


# ---------------------------------------------------------------- hand-written sequences

def _parse(script):
    """Lines such as  s1 = net.create_snapshot()  or  s1.is_following("a", "b") -> True   # why
    become ops, wants and the reasons shown on failure."""
    ops, wants, whys = [], [], []
    for line in script.strip().splitlines():
        if not line.strip():
            continue
        call, arrow, rest = line.strip().partition("->")
        if arrow:
            expr, _, why = rest.partition("#")
            expr = expr.strip()
            want = RAISE if expr == "ValueError" else ast.literal_eval(expr)
        else:
            call, _, why = call.partition("#")
            want = SKIP
        stmt = ast.parse(call.strip()).body[0]
        name = stmt.targets[0].id if isinstance(stmt, ast.Assign) else None
        node = stmt.value
        obj = node.func.value.id if isinstance(node.func, ast.Attribute) else None
        method = node.func.attr if obj else node.func.id
        ops.append((name, obj, method, tuple(ast.literal_eval(a) for a in node.args)))
        wants.append(want)
        whys.append(why.strip())
    return ops, wants, whys


def run(sol, script):
    ops, wants, whys = _parse(script)
    bad = _outcome(sol, ops, wants, show=40)        # hand-written sequences are short enough to show whole
    if bad is not None:
        why = whys[bad["k"]]
        _report(ops, bad, f"{why}。" if why and bad["kind"] != "value" else why, show=40)


# ---------------------------------------------------------------- random sequences against ref

def _valid(ops):
    made = set()
    for name, obj, _, _ in ops:
        if obj is not None and obj not in made:
            return False
        if name:
            made.add(name)
    return True


def _shrink(sol, ops, bad, budget=0.5):
    """Drops calls that the failure does not need, so the replay shown is short."""
    ops = ops[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 1, False                   # ops[0] is the constructor
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            out = _outcome(sol, cand) if _valid(cand) else None
            if out is not None and out["kind"] == bad["kind"] and type(out.get("exc")) is type(bad.get("exc")):
                ops, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, bad


def random_check(sol, make_ops, sequences):
    r = rng()
    for _ in range(sequences):
        ops = make_ops(r)
        bad = _outcome(sol, ops)
        if bad is None:
            continue
        ops, bad = _shrink(sol, ops, bad)
        _report(ops, bad, f"随机操作序列的第 {bad['k']} 步不对（已自动删去与出错无关的调用）")


NAMES = ["amy", "ben", "cat", "dan", "eve", "fay", "gus"]

EXAMPLE_S2 = """
    net = SocialNetwork()
    net.add_user("alice")
    net.add_user("bob")
    net.add_user("carol")
    net.add_user("dave")
    net.follow("alice", "bob")
    net.follow("alice", "carol")
    s1 = net.create_snapshot()
    net.follow("bob", "dave")
    net.add_user("erin")
    net.add_user("frank")
    net.follow("bob", "erin")
    net.follow("carol", "dave")
    net.follow("carol", "frank")
    net.follow("carol", "bob")
    net.follow("carol", "alice")
    s2 = net.create_snapshot()
"""


# ---------------------------------------------------------------- Part 1
@case("Part 1", sample=True, title="题目例子：快照 s1 与之后的 follow")
def p1_example(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("alice")
        net.add_user("bob")
        net.add_user("carol")
        net.add_user("dave")
        net.follow("alice", "bob")
        net.follow("alice", "carol")
        s1 = net.create_snapshot()
        net.follow("bob", "dave")
        s1.is_following("alice", "carol") -> True
        s1.is_following("bob", "dave")    -> False    # s1 创建时这条边还不存在
        s2 = net.create_snapshot()
        s2.is_following("bob", "dave")    -> True     # 在这次 follow 之后取的快照能看到它
    """)


@case("Part 1", sample=True, title="题目例子：重复关注、关注自己")
def p1_example_noops(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("alice")
        net.add_user("bob")
        net.add_user("carol")
        net.add_user("dave")
        net.follow("alice", "alice")
        net.follow("alice", "bob")
        net.follow("alice", "bob")
        net.follow("alice", "carol")
        net.follow("alice", "alice")
        s1 = net.create_snapshot()
        net.follow("alice", "bob")
        net.follow("bob", "dave")
        net.follow("alice", "alice")
        s1.is_following("alice", "carol") -> True
        s1.is_following("bob", "dave")    -> False    # s1 创建时这条边还不存在
        s2 = net.create_snapshot()
        s2.is_following("bob", "dave")    -> True
        s2.is_following("alice", "bob")   -> True
    """)


@case("Part 1", title="关注自己是空操作")
def p1_self_follow(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("a")
        net.add_user("b")
        net.follow("a", "a")
        net.follow("a", "b")
        net.follow("b", "b")
        s1 = net.create_snapshot()
        s1.is_following("a", "a") -> False    # follower == followee 是空操作
        s1.is_following("b", "b") -> False
        s1.is_following("a", "b") -> True
        s1.is_following("b", "a") -> False    # 关注是有方向的
    """)


@case("Part 1", title="重复注册抛 ValueError")
def p1_add_user_twice(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("alice")
        net.add_user("bob")
        net.follow("alice", "bob")
        net.add_user("alice") -> ValueError    # alice 已经注册过
        net.add_user("bob")   -> ValueError
        s1 = net.create_snapshot()
        s1.is_following("alice", "bob") -> True     # 失败的 add_user 不能清掉 alice 已有的关注
    """)


@case("Part 1", title="follow 未注册的用户抛 ValueError")
def p1_follow_unknown(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("alice")
        net.follow("alice", "ghost") -> ValueError    # ghost 没有注册
        net.follow("ghost", "alice") -> ValueError
        net.follow("ghost", "nobody") -> ValueError
        net.add_user("ghost")
        net.follow("alice", "ghost")
        s1 = net.create_snapshot()
        s1.is_following("alice", "ghost") -> True
        s1.is_following("ghost", "alice") -> False    # 前面那次失败的 follow 不能留下这条边
    """)


@case("Part 1", title="快照里没有的用户抛 ValueError")
def p1_snapshot_unknown(sol):
    run(sol, """
        net = SocialNetwork()
        s0 = net.create_snapshot()
        s0.is_following("alice", "bob")   -> ValueError    # 这个快照里还没有任何用户
        net.add_user("alice")
        net.add_user("bob")
        net.follow("alice", "bob")
        s1 = net.create_snapshot()
        s1.is_following("alice", "carol") -> ValueError    # carol 不是用户
        s1.is_following("carol", "alice") -> ValueError
        net.add_user("carol")
        net.follow("carol", "alice")
        s1.is_following("carol", "alice") -> ValueError    # carol 是在 s1 之后才注册的，不是 s1 的用户
        s1.is_following("alice", "carol") -> ValueError
        s0.is_following("alice", "bob")   -> ValueError    # s0 创建时还没有 alice 和 bob
        s1.is_following("alice", "bob")   -> True
        s2 = net.create_snapshot()
        s2.is_following("carol", "alice") -> True
    """)


@case("Part 1", title="多个快照互不影响")
def p1_isolation(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("a")
        net.add_user("b")
        net.add_user("c")
        s1 = net.create_snapshot()
        net.follow("a", "b")
        s2 = net.create_snapshot()
        net.follow("b", "c")
        net.follow("c", "a")
        s3 = net.create_snapshot()
        net.add_user("d")
        net.follow("d", "a")
        net.follow("a", "c")
        s1.is_following("a", "b") -> False         # 已经返回的快照不能被之后的 follow 改变
        s2.is_following("a", "b") -> True
        s2.is_following("b", "c") -> False
        s2.is_following("a", "c") -> False
        s3.is_following("c", "a") -> True
        s3.is_following("a", "c") -> False
        s3.is_following("d", "a") -> ValueError    # d 是在 s3 之后才注册的
        s4 = net.create_snapshot()
        s4.is_following("a", "c") -> True
        s4.is_following("d", "a") -> True
        net.follow("b", "a")
        s4.is_following("b", "a") -> False
        s3.is_following("b", "a") -> False
        s1.is_following("c", "a") -> False
    """)


@case("Part 1", title="随机操作序列（对照参考实现）")
def p1_random(sol):
    random_check(sol, lambda r: ref.random_network_ops(r, 1, r.randint(10, 45), NAMES[:r.randint(3, 6)]), 60)


# ---------------------------------------------------------------- Part 2
@case("Part 2", sample=True, title="题目例子 s2 的关注与粉丝")
def p2_example(sol):
    run(sol, EXAMPLE_S2 + """
        s2.get_following("carol") -> ["alice", "bob", "dave", "frank"]
        s2.get_followers("dave")  -> ["bob", "carol"]
    """)


@case("Part 2", title="没有关注、没有粉丝：空列表")
def p2_empty(sol):
    run(sol, EXAMPLE_S2 + """
        s2.get_following("dave")  -> []
        s2.get_followers("erin")  -> ["bob"]
        s2.get_following("erin")  -> []
        s1.get_followers("dave")  -> []        # s1 创建时还没有人关注 dave
        s1.get_following("bob")   -> []
        s1.get_followers("alice") -> []
        s2.get_followers("alice") -> ["carol"]
    """)


@case("Part 2", title="重复关注不重复出现；没有自己")
def p2_noops(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("zoe")
        net.add_user("max")
        net.add_user("ann")
        net.follow("zoe", "max")
        net.follow("zoe", "zoe")
        net.follow("zoe", "max")
        net.follow("zoe", "ann")
        net.follow("ann", "max")
        net.follow("zoe", "ann")
        s1 = net.create_snapshot()
        s1.get_following("zoe") -> ["ann", "max"]    # 重复的 follow 和关注自己都是空操作；按 id 升序
        s1.get_followers("max") -> ["ann", "zoe"]
        s1.get_followers("zoe") -> []
        s1.get_following("max") -> []
    """)


@case("Part 2", title="未知用户抛 ValueError")
def p2_unknown(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("a")
        s1 = net.create_snapshot()
        s1.get_following("b") -> ValueError    # b 不是这个快照的用户
        s1.get_followers("b") -> ValueError
        net.add_user("b")
        net.follow("b", "a")
        s1.get_following("b") -> ValueError    # b 是在 s1 之后才注册的
        s1.get_followers("a") -> []
        s2 = net.create_snapshot()
        s2.get_followers("a") -> ["b"]
    """)


@case("Part 2", title="旧快照的列表不随之后的 follow 变化")
def p2_isolation(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("d")
        net.add_user("c")
        net.add_user("b")
        net.add_user("a")
        net.follow("a", "b")
        s1 = net.create_snapshot()
        net.follow("d", "b")
        net.follow("c", "b")
        net.follow("b", "a")
        net.follow("a", "d")
        s1.get_followers("b") -> ["a"]         # 粉丝列表也要冻结在 s1 创建的那一刻
        s1.get_following("b") -> []
        s1.get_followers("a") -> []
        s1.get_following("a") -> ["b"]
        s2 = net.create_snapshot()
        s2.get_followers("b") -> ["a", "c", "d"]
        s2.get_following("a") -> ["b", "d"]
        net.follow("c", "a")
        s2.get_followers("a") -> ["b"]
        s1.get_followers("b") -> ["a"]
    """)


@case("Part 2", title="随机操作序列（对照参考实现）")
def p2_random(sol):
    random_check(sol, lambda r: ref.random_network_ops(r, 2, r.randint(10, 45), NAMES[:r.randint(3, 7)]), 60)


# ---------------------------------------------------------------- Part 3
@case("Part 3", sample=True, title="题目例子 recommend(alice, 2/5)")
def p3_example(sol):
    run(sol, EXAMPLE_S2 + """
        s2.recommend("alice", 2) -> ["dave", "erin"]
        s2.recommend("alice", 5) -> ["dave", "erin", "frank"]
    """)


@case("Part 3", title="k = 0、k = 1、k 超过候选人数")
def p3_k(sol):
    run(sol, EXAMPLE_S2 + """
        s2.recommend("alice", 0)   -> []
        s2.recommend("alice", 1)   -> ["dave"]
        s2.recommend("alice", 3)   -> ["dave", "erin", "frank"]
        s2.recommend("alice", 100) -> ["dave", "erin", "frank"]   # 候选人不足 k 个时有几个返回几个
    """)


@case("Part 3", title="排除自己和已经关注的人")
def p3_exclusions(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("u")
        net.add_user("a")
        net.add_user("b")
        net.add_user("c")
        net.add_user("x")
        net.follow("u", "a")
        net.follow("u", "b")
        net.follow("u", "c")
        net.follow("a", "b")
        net.follow("a", "u")
        net.follow("a", "x")
        net.follow("b", "c")
        net.follow("b", "u")
        net.follow("b", "x")
        net.follow("c", "a")
        net.follow("c", "u")
        s1 = net.create_snapshot()
        s1.recommend("u", 10) -> ["x"]    # u 自己（得分 3）和 u 已经关注的 a、b、c 都不是候选人
        s1.recommend("a", 10) -> ["c"]    # a 关注 b、u、x：c 经 b、u 到达；a 自己经 u 到达，要排除
        s1.recommend("x", 10) -> []       # x 没有关注任何人
    """)


@case("Part 3", title="得分只数 user_id 的关注对象")
def p3_score(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("u")
        net.add_user("a")
        net.add_user("b")
        net.add_user("p")
        net.add_user("q")
        net.add_user("f1")
        net.add_user("f2")
        net.add_user("f3")
        net.follow("u", "a")
        net.follow("u", "b")
        net.follow("a", "p")
        net.follow("a", "q")
        net.follow("b", "q")
        net.follow("f1", "p")
        net.follow("f2", "p")
        net.follow("f3", "p")
        s1 = net.create_snapshot()
        s1.recommend("u", 2) -> ["q", "p"]   # q 经 a、b 到达（得分 2），p 只经 a 到达（得分 1）；f1、f2、f3 不是 u 关注的人，不算
        s1.recommend("u", 1) -> ["q"]
        s1.recommend("f1", 5) -> []          # f1 关注的 p 没有关注任何人
    """)


@case("Part 3", title="得分并列时按 id 升序")
def p3_ties(sol):
    run(sol, """
        net = SocialNetwork()
        net.add_user("u")
        net.add_user("c")
        net.add_user("b")
        net.add_user("a")
        net.add_user("z")
        net.add_user("y")
        net.add_user("n")
        net.add_user("m")
        net.add_user("k")
        net.follow("u", "c")
        net.follow("u", "b")
        net.follow("u", "a")
        net.follow("a", "m")
        net.follow("a", "k")
        net.follow("a", "z")
        net.follow("a", "y")
        net.follow("b", "z")
        net.follow("b", "k")
        net.follow("b", "n")
        net.follow("c", "k")
        net.follow("c", "y")
        s1 = net.create_snapshot()
        s1.recommend("u", 5) -> ["k", "y", "z", "m", "n"]   # k 得 3 分；y、z 各 2 分；m、n 各 1 分
        s1.recommend("u", 2) -> ["k", "y"]
        s1.recommend("u", 4) -> ["k", "y", "z", "m"]
    """)


@case("Part 3", title="recommend 不改变快照")
def p3_no_side_effects(sol):
    run(sol, EXAMPLE_S2 + """
        s2.recommend("alice", 5)          -> ["dave", "erin", "frank"]
        s2.is_following("alice", "alice") -> False    # recommend 之后快照里的关注关系不能变
        s2.get_following("alice")         -> ["bob", "carol"]
        s2.recommend("alice", 5)          -> ["dave", "erin", "frank"]
        s2.recommend("carol", 5)          -> ["erin"]
        s2.get_following("carol")         -> ["alice", "bob", "dave", "frank"]
        s2.get_followers("carol")         -> ["alice"]
        s2.recommend("carol", 5)          -> ["erin"]
    """)


@case("Part 3", title="在旧快照上推荐；未知用户抛 ValueError")
def p3_old_snapshot(sol):
    run(sol, EXAMPLE_S2 + """
        s1.recommend("alice", 5) -> []                # s1 创建时 bob、carol 还没有关注任何人
        s1.recommend("erin", 1)  -> ValueError        # erin 是在 s1 之后才注册的
        s2.recommend("ghost", 1) -> ValueError
        net.follow("bob", "frank")
        net.follow("carol", "erin")
        s2.recommend("alice", 5) -> ["dave", "erin", "frank"]
        s3 = net.create_snapshot()
        s3.recommend("alice", 5) -> ["dave", "erin", "frank"]
        s3.recommend("alice", 1) -> ["dave"]          # dave、erin、frank 都得 2 分，按 id 升序
    """)


@case("Part 3", title="随机操作序列（对照参考实现）")
def p3_random(sol):
    random_check(sol, lambda r: ref.random_network_ops(r, 3, r.randint(10, 50), NAMES[:r.randint(3, 7)]), 60)


@case("Part 3", title="150 个用户的随机图：逐个推荐")
def p3_larger(sol):
    r = rng()
    users = [f"user{i:03d}" for i in range(150)]
    net, model = sol.SocialNetwork(), ref.SocialNetwork()
    calling("net = SocialNetwork()\n# 注册 150 个用户 user000 … user149")
    for u in r.sample(users, len(users)):
        net.add_user(u)
        model.add_user(u)
    calling("# 150 个用户之间随机加入约 1500 条关注")
    for _ in range(1500):
        a, b = r.choice(users), r.choice(users[: r.choice([20, 150])])
        net.follow(a, b)
        model.follow(a, b)
    calling("# 150 个用户、约 1500 条关注之后\ns = net.create_snapshot()")
    snap, want_snap = net.create_snapshot(), model.create_snapshot()
    for u in users:
        k = r.choice([1, 3, 10])
        want = want_snap.recommend(u, k)
        call = f"s.recommend(\"{u}\", {k})   # s：150 个用户、约 1500 条随机关注的快照"
        calling(call)
        got = snap.recommend(u, k)
        if not _same(got, want):
            scores = want_snap.scores(u)
            extra = [c for c in (list(got) if isinstance(got, (list, tuple)) else [])[:k] if c in scores]
            hint = "；".join(f"{c} 得 {scores[c]} 分" for c in dict.fromkeys(list(want) + extra))
            expect(got, want, call=call, msg=f"推荐结果不对（按题目的得分：{hint}）")


# ---------------------------------------------------------------- Part 4
@case("Part 4", sample=True, title="题目例子：10 关注、20 取关、30 再关注")
def p4_example(sol):
    run(sol, """
        tl = FollowTimeline()
        tl.follow("x", "y", 10)
        tl.unfollow("x", "y", 20)
        tl.follow("x", "y", 30)
        tl.is_following("x", "y", 5)    -> False
        tl.is_following("x", "y", 9)    -> False
        tl.is_following("x", "y", 10)   -> True     # 10 <= t < 20 时在关注
        tl.is_following("x", "y", 15)   -> True
        tl.is_following("x", "y", 19)   -> True
        tl.is_following("x", "y", 20)   -> False    # 20 <= t < 30 时没有关注
        tl.is_following("x", "y", 29)   -> False
        tl.is_following("x", "y", 30)   -> True     # t >= 30 时在关注
        tl.is_following("x", "y", 1000) -> True
    """)


@case("Part 4", title="没有记录的一对用户、反方向、自己")
def p4_unknown(sol):
    run(sol, """
        tl = FollowTimeline()
        tl.is_following("x", "y", 0)  -> False    # 还没有任何记录
        tl.follow("x", "y", 3)
        tl.follow("m", "m", 4)
        tl.is_following("y", "x", 9)  -> False    # 关注是有方向的
        tl.is_following("x", "z", 9)  -> False
        tl.is_following("q", "r", 9)  -> False
        tl.is_following("m", "m", 9)  -> False    # 关注自己是空操作
        tl.unfollow("m", "m", 5)
        tl.is_following("m", "m", 9)  -> False
        tl.is_following("x", "y", 9)  -> True
    """)


@case("Part 4", title="重复状态的调用是空操作")
def p4_redundant(sol):
    run(sol, """
        tl = FollowTimeline()
        tl.follow("a", "b", 1)
        tl.unfollow("c", "d", 2)        # 从来没关注过：空操作
        tl.follow("a", "b", 3)          # 已经在关注：空操作
        tl.follow("c", "d", 4)
        tl.unfollow("a", "b", 5)
        tl.unfollow("a", "b", 6)        # 已经取关：空操作
        tl.follow("a", "b", 8)
        tl.follow("c", "d", 8)          # 已经在关注：空操作
        tl.is_following("a", "b", 2)   -> True
        tl.is_following("a", "b", 3)   -> True    # 3 时刻的 follow 是空操作，不改变状态
        tl.is_following("a", "b", 4)   -> True
        tl.is_following("a", "b", 5)   -> False
        tl.is_following("a", "b", 6)   -> False
        tl.is_following("a", "b", 7)   -> False
        tl.is_following("a", "b", 8)   -> True
        tl.is_following("c", "d", 2)   -> False   # 2 时刻的 unfollow 是空操作
        tl.is_following("c", "d", 3)   -> False
        tl.is_following("c", "d", 4)   -> True
        tl.is_following("c", "d", 100) -> True
    """)


@case("Part 4", title="同一时刻多次调用：后调用的决定")
def p4_same_t(sol):
    run(sol, """
        tl = FollowTimeline()
        tl.follow("m", "n", 3)
        tl.unfollow("m", "n", 3)
        tl.follow("m", "n", 3)
        tl.follow("r", "s", 7)
        tl.unfollow("r", "s", 9)
        tl.follow("r", "s", 9)
        tl.follow("p", "q", 50)
        tl.unfollow("p", "q", 50)
        tl.is_following("p", "q", 50) -> False    # 同一时刻先 follow 后 unfollow：后调用的 unfollow 决定
        tl.is_following("p", "q", 49) -> False
        tl.is_following("p", "q", 51) -> False
        tl.is_following("r", "s", 9)  -> True     # 同一时刻先 unfollow 后 follow
        tl.is_following("r", "s", 8)  -> True
        tl.is_following("m", "n", 3)  -> True     # follow、unfollow、follow 都在 3 时刻
        tl.is_following("m", "n", 2)  -> False
    """)


@case("Part 4", title="写入之后查询更早的时刻")
def p4_past_queries(sol):
    run(sol, """
        tl = FollowTimeline()
        tl.follow("a", "b", 10)
        tl.is_following("a", "b", 10) -> True
        tl.follow("a", "c", 20)
        tl.unfollow("a", "b", 30)
        tl.is_following("a", "b", 25) -> True
        tl.is_following("a", "b", 30) -> False
        tl.is_following("a", "c", 15) -> False
        tl.follow("a", "b", 40)
        tl.is_following("a", "b", 35) -> False
        tl.is_following("a", "b", 12) -> True
        tl.is_following("a", "b", 9)  -> False
        tl.is_following("b", "a", 12) -> False
        tl.unfollow("a", "c", 40)
        tl.is_following("a", "c", 39) -> True
        tl.is_following("a", "c", 40) -> False
    """)


@case("Part 4", title="负数时刻和很大的时刻")
def p4_time_range(sol):
    run(sol, """
        tl = FollowTimeline()
        tl.follow("a", "b", -5)
        tl.unfollow("a", "b", 0)
        tl.follow("a", "b", 1000000000000)
        tl.is_following("a", "b", -6)            -> False
        tl.is_following("a", "b", -5)            -> True
        tl.is_following("a", "b", -1)            -> True
        tl.is_following("a", "b", 0)             -> False
        tl.is_following("a", "b", 999999999999)  -> False
        tl.is_following("a", "b", 1000000000000) -> True
    """)


@case("Part 4", title="随机操作序列（对照参考实现）")
def p4_random(sol):
    def make(r):
        names = ["x", "y", "z", "w"][: r.randint(2, 4)]
        return ref.random_timeline_ops(r, r.randint(10, 60), names, start=r.choice([0, 0, -10, 1000]))
    random_check(sol, make, 80)
