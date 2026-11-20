"""Tests for the weighted data batcher. The registry is ref.Registry, the statement's DataRegistry with examples
"name-0", "name-1", ... Parts 1 and 2 have one right answer (ref.exact_batches). Part 3's deterministic rule is the
answer's own choice, so it is checked for per-dataset order and for |count_k(n) - n w_k / W| <= K + W at every
choice; the stochastic rule is checked for reproducibility, exact resumption and its proportions."""
import json
from collections import Counter

from judgelib import Mismatch, calling, case, expect, expect_true, expect_valid, rng
import ref

IMPORTS = ("import json\n"
           'from ref import Registry   # 题目的 DataRegistry：数据集 name 的样本依次是 "name-0"、"name-1"、…，读完回绕'
           "（评测用的就是它）")


class Replay:
    """The operations so far as code, shown with every failure."""

    def __init__(self):
        self.lines = []

    def text(self):
        lines = self.lines
        if len("\n".join(lines)) > 2400 and len(lines) > 10:
            lines = lines[:3] + [f"# …（省略了中间 {len(lines) - 9} 行，所以不能直接粘贴重现）"] + lines[-6:]
        return IMPORTS + "\n" + "\n".join(lines)

    def run(self, line, fn):
        self.lines.append(line)
        calling(self.text())
        return fn()


def _kw(kw):
    return "".join(f", {k}={v!r}" for k, v in kw.items())


def build(sol, rep, var, lengths, weights, bs, reg=None, reg_var=None, **kw):
    """`var = DataBatcher(Registry(lengths), weights, bs, **kw)`; returns (batcher, registry)."""
    if reg is None:
        reg, reg_code = ref.Registry(lengths), f"Registry({lengths!r})"
    else:
        reg_code = reg_var
    b = rep.run(f"{var} = DataBatcher({reg_code}, {weights!r}, {bs}{_kw(kw)})",
                lambda: sol.DataBatcher(reg, dict(weights), bs, **kw))
    return b, reg


def as_batch(got, call):
    if isinstance(got, (list, tuple)):
        return list(got)
    raise Mismatch(f"next_batch() 应返回 list，实际返回了 {type(got).__name__}", call, got=got)


def next_batch(rep, b, var):
    got = rep.run(f"{var}.next_batch()", b.next_batch)
    return as_batch(got, rep.text())


def take_state(rep, b, var, svar):
    """`svar = json.loads(json.dumps(var.state_dict()))`: the state as a checkpoint file would hold it."""
    call = rep.run(f"{svar} = json.loads(json.dumps({var}.state_dict()))", lambda: None)
    state = b.state_dict()
    if not isinstance(state, dict):
        raise Mismatch(f"state_dict() 应返回 dict，实际返回了 {type(state).__name__}", call, got=state)
    try:
        text = json.dumps(state)
    except (TypeError, ValueError) as e:
        raise Mismatch(f"state_dict() 的结果不能直接 JSON 序列化（{type(e).__name__}: {e}）", call, got=state) from None
    return json.loads(text)


def load(rep, b, var, svar, state):
    rep.run(f"{var}.load_state_dict({svar})", lambda: b.load_state_dict(ref.roundtrip(state)))


def run_batches(rep, b, var, count, check):
    """Calls var.next_batch() `count` times; check(i, batch) raises on a wrong batch. The replay shows the run as
    one list comprehension, so it can be pasted."""
    base = len(rep.lines)
    for i in range(count):
        rep.lines[base:] = [f"batches = [{var}.next_batch() for _ in range({i + 1})]",
                            f"batches[-1]   # 第 {i} 个 batch（从 0 数）"]
        call = calling(rep.text())
        check(i, as_batch(b.next_batch(), call))
    rep.lines[base:] = [f"for _ in range({count}): {var}.next_batch()"]


def skip_batches(rep, b, var, count):
    """count batches whose content another check covers; returns them."""
    if count == 0:
        return []
    out = rep.run(f"for _ in range({count}): {var}.next_batch()", lambda: [b.next_batch() for _ in range(count)])
    return [as_batch(x, rep.text()) for x in out]


def run_pairs(rep, b1, v1, b2, v2, count, why, check=None):
    """Calls v1.next_batch() and v2.next_batch() in turn, `count` times; each pair must be equal (why(i) explains
    otherwise). check(batch), when given, validates v1's batch."""
    base = len(rep.lines)
    for i in range(count):
        rep.lines[base:] = [f"for _ in range({i + 1}): x, y = {v1}.next_batch(), {v2}.next_batch()",
                            f"x == y   # 第 {i} 对 batch（从 0 数），应为 True"]
        call = calling(rep.text())
        x = as_batch(b1.next_batch(), call)
        y = as_batch(b2.next_batch(), call)
        if check is not None:
            expect_valid(x, check, call=call)
        if x != y:
            raise Mismatch(why(i), call, x, y)
    rep.lines[base:] = [f"for _ in range({count}): {v1}.next_batch(), {v2}.next_batch()"]


# ---------------------------------------------------------------- exact batches (Parts 1 and 2)

def _batch_msg(got, want, after_load=None):
    """Where an exact batch goes wrong, in words. after_load names the state just loaded, if any."""
    try:
        cg = Counter(p[0] if p else None for p in map(ref.parse, got))
        cw = Counter(ref.parse(x)[0] for x in want)
    except TypeError:
        return ""
    if None in cg:
        bad = next(x for x in got if ref.parse(x) is None)
        return f"{bad!r} 不是从 registry 取出的样本"
    head = f"batch 长度应为 {len(want)}，实际 {len(got)}；" if len(got) != len(want) else ""
    for name in sorted(set(cw) | set(cg)):
        if cg[name] != cw[name]:
            return head + f"{name} 应有 {cw[name]} 个样本（batch_size·w/W），实际 {cg[name]} 个"
    if Counter(got) == Counter(want):
        order = " < ".join(sorted(cw))
        return f"样本都对，但顺序不对：数据集按名字的字母升序排列（{order}），每个数据集的样本连成一段"
    i = next(j for j in range(len(want)) if got[j] != want[j])
    name = ref.parse(want[i])[0]
    msg = f"第 {i} 个样本应是 {want[i]!r}，实际是 {got[i]!r}"
    if ref.parse(got[i])[0] == name:
        msg += (f"：load_state_dict({after_load}) 之后，{name} 要从 {after_load} 记录的位置取，不能接着原来的迭代器往下取"
                if after_load else f"：{name} 要接着它上一次取到的位置往下取")
    return msg


def check_batch(got, want, call=None, after_load=None):
    if got != want:
        raise Mismatch(_batch_msg(got, want, after_load), call, want, got)


def exact_run(sol, rep, var, lengths, weights, bs, count):
    b, reg = build(sol, rep, var, lengths, weights, bs)
    want, _ = ref.exact_batches(lengths, weights, bs, count)
    run_batches(rep, b, var, count, lambda i, got: check_batch(got, want[i]))
    return b, reg


# ---------------------------------------------------------------- Part 1

EX_W = {"code": 2, "math": 1, "web": 1}
EX_LEN = {"code": 97, "math": 97, "web": 97}


@case("Part 1", sample=True, title="题目例子")
def p1_example(sol):
    rep = Replay()
    b, _ = build(sol, rep, "b", EX_LEN, EX_W, 8)
    got = next_batch(rep, b, "b")
    check_batch(got, ["code-0", "code-1", "code-2", "code-3", "math-0", "math-1", "web-0", "web-1"])


@case("Part 1", title="连续多个 batch 接着往下取")
def p1_consecutive(sol):
    exact_run(sol, Replay(), "b", EX_LEN, EX_W, 8, 5)
    exact_run(sol, Replay(), "b", {"code": 6, "math": 5, "web": 3}, EX_W, 12, 6)


@case("Part 1", title="按名字字母序，不按插入顺序或权重")
def p1_alphabetical(sol):
    weights = {"web": 1, "books": 3, "math": 2, "arxiv": 2, "codex": 1, "code": 3}
    exact_run(sol, Replay(), "b", {name: 50 for name in weights}, weights, 24, 3)
    exact_run(sol, Replay(), "b", {"wiki": 40, "web": 40}, {"wiki": 3, "web": 1}, 8, 2)


@case("Part 1", title="只有一个数据集")
def p1_single_dataset(sol):
    exact_run(sol, Replay(), "b", {"math": 4}, {"math": 3}, 6, 3)
    exact_run(sol, Replay(), "b", {"qa": 10}, {"qa": 1}, 1, 4)


@case("Part 1", title="数据集比配额短：在 batch 内回绕")
def p1_short_datasets(sol):
    exact_run(sol, Replay(), "b", {"tiny": 2, "big": 50}, {"tiny": 1, "big": 1}, 10, 2)
    exact_run(sol, Replay(), "b", {"arxiv": 1, "books": 1}, {"arxiv": 2, "books": 1}, 6, 3)


@case("Part 1", title="随机配置")
def p1_random(sol):
    r = rng()
    for _ in range(150):
        lengths, weights, bs = ref.random_config(r, divisible=True)
        exact_run(sol, Replay(), "b", lengths, weights, bs, r.randint(1, 5))


# ---------------------------------------------------------------- Part 2

EX_STATE = {"offsets": {"code": 4, "math": 2, "web": 2}}
EX_BATCH2 = ["code-4", "code-5", "code-6", "code-7", "math-2", "math-3", "web-2", "web-3"]


@case("Part 2", sample=True, title="题目例子")
def p2_example(sol):
    rep = Replay()
    b, _ = build(sol, rep, "b", EX_LEN, EX_W, 8)
    next_batch(rep, b, "b")
    state = rep.run("b.state_dict()", b.state_dict)
    expect_valid(state, lambda s: None if isinstance(s, dict) and s.get("offsets") == EX_STATE["offsets"] else
                 '题目例子里此时 state_dict() 是 {"offsets": {"code": 4, "math": 2, "web": 2}}'
                 "（例子里的 b2.load_state_dict 也直接传入这个字典）", want=EX_STATE)
    check_batch(next_batch(rep, b, "b"), EX_BATCH2)
    b2, _ = build(sol, rep, "b2", EX_LEN, EX_W, 8)
    rep.run(f"b2.load_state_dict({EX_STATE!r})", lambda: b2.load_state_dict(ref.roundtrip(EX_STATE)))
    check_batch(next_batch(rep, b2, "b2"), EX_BATCH2)


def resume_everywhere(sol, lengths, weights, bs, count):
    """Pause after every batch k, save through JSON, load into a fresh batcher, and finish the run."""
    want, _ = ref.exact_batches(lengths, weights, bs, count)
    for k in range(count + 1):
        rep = Replay()
        b, _ = build(sol, rep, "b", lengths, weights, bs)
        for i, got in enumerate(skip_batches(rep, b, "b", k)):
            check_batch(got, want[i])
        state = take_state(rep, b, "b", "state")
        b2, _ = build(sol, rep, "b2", lengths, weights, bs)
        load(rep, b2, "b2", "state", state)
        run_batches(rep, b2, "b2", count - k, lambda i, got: check_batch(got, want[k + i]))


@case("Part 2", title="状态能 JSON 序列化，新对象接着跑")
def p2_json_roundtrip(sol):
    resume_everywhere(sol, {"news": 5, "books": 9, "qa": 4}, {"news": 1, "books": 2, "qa": 3}, 12, 3)


@case("Part 2", title="在每个 batch 边界保存并恢复")
def p2_every_boundary(sol):
    resume_everywhere(sol, EX_LEN, EX_W, 8, 6)
    resume_everywhere(sol, {"wiki": 3, "chat": 1, "papers": 7}, {"wiki": 1, "chat": 1, "papers": 2}, 4, 6)
    r = rng()
    for _ in range(8):
        lengths, weights, bs = ref.random_config(r, divisible=True, max_k=4)
        resume_everywhere(sol, lengths, weights, bs, r.randint(1, 4))


@case("Part 2", title="恢复时用 offset 直接定位，不重读")
def p2_no_replay(sol):
    lengths, weights, bs, warm = {"code": 1000, "math": 300, "web": 7}, {"code": 5, "math": 2, "web": 1}, 40, 600
    want, after = ref.exact_batches(lengths, weights, bs, warm + 1)
    rep = Replay()
    b, _ = build(sol, rep, "b", lengths, weights, bs)
    skip_batches(rep, b, "b", warm)
    state = take_state(rep, b, "b", "state")
    reg2 = ref.Registry(lengths)
    rep.lines.append(f"reg2 = Registry({lengths!r})")
    b2, _ = build(sol, rep, "b2", lengths, weights, bs, reg=reg2, reg_var="reg2")
    load(rep, b2, "b2", "state", state)
    got = next_batch(rep, b2, "b2")
    saved = after[warm - 1]
    rep.run("reg2.lowest   # Registry 记录的：每个数据集交出过的最小位置（不取模）", lambda: None)
    for name in sorted(reg2.lowest):
        expect_true(reg2.lowest[name] >= saved[name],
                    f"恢复后 {name} 从第 {reg2.lowest[name]} 个位置开始读（保存时已经读到第 {saved[name]} 个）：恢复不能把"
                    f"已经产出过的样本重新迭代一遍，要用 get_iterator(name, offset) 直接定位到上次结束的地方")
    check_batch(got, want[warm])


@case("Part 2", title="同一个对象回到之前的状态")
def p2_rewind_same_object(sol):
    lengths, weights, bs = {"code": 9, "math": 4, "web": 5}, EX_W, 8
    want, _ = ref.exact_batches(lengths, weights, bs, 6)
    rep = Replay()
    b, _ = build(sol, rep, "b", lengths, weights, bs)
    check_batch(next_batch(rep, b, "b"), want[0])
    state = take_state(rep, b, "b", "s1")
    check_batch(next_batch(rep, b, "b"), want[1])
    check_batch(next_batch(rep, b, "b"), want[2])
    load(rep, b, "b", "s1", state)
    check_batch(next_batch(rep, b, "b"), want[1], call=rep.text() + "   # 回到 s1 之后，应重新得到第 1 个 batch（从 0 数）",
                after_load="s1")
    check_batch(next_batch(rep, b, "b"), want[2])
    # forward: a state taken from another batcher that is further ahead
    other, _ = build(sol, rep, "other", lengths, weights, bs)
    skip_batches(rep, other, "other", 5)
    ahead = take_state(rep, other, "other", "s5")
    load(rep, b, "b", "s5", ahead)
    check_batch(next_batch(rep, b, "b"), want[5], after_load="s5")


@case("Part 2", title="保存的状态不随原对象前进而改变")
def p2_state_snapshot(sol):
    lengths, weights, bs = EX_LEN, EX_W, 8
    want, _ = ref.exact_batches(lengths, weights, bs, 4)
    rep = Replay()
    b, _ = build(sol, rep, "b", lengths, weights, bs)
    check_batch(next_batch(rep, b, "b"), want[0])
    state = rep.run("s = b.state_dict()", b.state_dict)
    try:
        taken = json.dumps(state, sort_keys=True)
    except (TypeError, ValueError) as e:
        raise Mismatch(f"state_dict() 的结果不能直接 JSON 序列化（{type(e).__name__}: {e}）", got=state) from None
    check_batch(next_batch(rep, b, "b"), want[1])
    check_batch(next_batch(rep, b, "b"), want[2])
    b2, _ = build(sol, rep, "b2", lengths, weights, bs)
    changed = json.dumps(state, sort_keys=True) != taken
    rep.run("b2.load_state_dict(s)", lambda: b2.load_state_dict(json.loads(json.dumps(state))))
    got = next_batch(rep, b2, "b2")
    if got != want[1] and changed:
        raise Mismatch("b2 没有从 s 保存时的位置接着取：s 在 b 继续调用 next_batch() 之后被改动了（例如 state_dict() 直接"
                       "返回了内部的字典）。保存的状态应记录取出那一刻的位置", want=want[1], got=got)
    check_batch(got, want[1])


@case("Part 2", title="状态大小不随 batch 数增长")
def p2_state_size(sol):
    rep = Replay()
    b, _ = build(sol, rep, "b", EX_LEN, EX_W, 8)
    skip_batches(rep, b, "b", 1)
    small = len(json.dumps(take_state(rep, b, "b", "s1")))
    skip_batches(rep, b, "b", 3000)
    large = len(json.dumps(take_state(rep, b, "b", "s3001")))
    rep.run("len(json.dumps(s1)), len(json.dumps(s3001))", lambda: None)
    expect_true(large - small <= 16 * len(EX_W) + 64,
                f"1 个 batch 之后状态的 JSON 有 {small} 个字符，3001 个 batch 之后有 {large} 个：状态的大小应只由数据集个数 K "
                f"决定，不能随已经产出的 batch 数增长")


# ---------------------------------------------------------------- Part 3

def det_check(tracker, bs):
    def check(batch):
        if len(batch) != bs:
            return f"应返回 batch_size = {bs} 个样本，实际 {len(batch)} 个"
        return tracker.feed(batch)
    return check


def checked_run(rep, b, var, count, tracker, bs):
    """count batches, each checked with tracker (order, and the bound when the tracker has one); returns them."""
    out = []

    def check(i, batch):
        expect_valid(batch, det_check(tracker, bs))
        out.append(batch)
    run_batches(rep, b, var, count, check)
    return out


EX3 = ["code-0", "math-0", "web-0", "code-1", "code-2", "math-1"]


@case("Part 3", sample=True, title="题目例子（规则可以不同）")
def p3_example(sol):
    rep = Replay()
    b, _ = build(sol, rep, "b", EX_LEN, EX_W, 6, allocation="deterministic")
    got = next_batch(rep, b, "b")
    tracker = ref.Tracker(EX_LEN, EX_W, ref.det_bound(EX_W))
    expect_valid(got, det_check(tracker, 6), want=EX3)


@case("Part 3", title="能整除时退回 Part 1 的规则")
def p3_exact_fallback(sol):
    r = rng()
    configs = [(EX_LEN, EX_W, 8), ({"arxiv": 5, "books": 2, "code": 3}, {"arxiv": 1, "books": 2, "code": 1}, 4),
               ({"web": 6, "chat": 2}, {"web": 2, "chat": 3}, 15)]
    for lengths, weights, bs in configs:
        want, _ = ref.exact_batches(lengths, weights, bs, 3)
        for kw in ({"allocation": "deterministic"}, {"allocation": "stochastic", "seed": r.randrange(10 ** 6)}):
            rep = Replay()
            b, _ = build(sol, rep, "b", lengths, weights, bs, **kw)
            def check(i, got):
                if got != want[i]:
                    raise Mismatch(f"batch_size = {bs} 能被 W = {sum(weights.values())} 整除，不论 allocation 取什么都要退回"
                                   f" Part 1 的精确规则：{_batch_msg(got, want[i])}", want=want[i], got=got)
            run_batches(rep, b, "b", 3, check)


@case("Part 3", title="默认 deterministic，且与 seed 无关")
def p3_default_args(sol):
    lengths, weights, bs = {"code": 11, "math": 7, "web": 5}, EX_W, 6
    rep = Replay()
    base, _ = build(sol, rep, "b", lengths, weights, bs)
    first = checked_run(rep, base, "b", 5, ref.Tracker(lengths, weights, ref.det_bound(weights)), bs)
    for kw in ({"allocation": "deterministic"}, {"allocation": "deterministic", "seed": 12345}):
        rep = Replay()
        b, _ = build(sol, rep, "b", lengths, weights, bs, **kw)

        def same(i, got, kw=kw):
            expect(got, first[i], msg=f"和 DataBatcher(registry, weights, {bs})（不传 allocation 和 seed）的第 {i} 个 batch "
                                      f"不同：allocation 默认是 \"deterministic\"，确定性分配不引入任何随机性，seed 只在 "
                                      f"stochastic 时使用")
        run_batches(rep, b, "b", 5, same)


DET_CONFIGS = [
    ({"code": 7, "math": 5, "web": 3}, EX_W, 6, 500),
    ({"chat": 5, "code": 5, "math": 5, "web": 5}, {"chat": 1, "code": 1, "math": 1, "web": 1}, 2, 1000),
    ({"arxiv": 13, "books": 8, "code": 6}, {"arxiv": 5, "books": 3, "code": 2}, 7, 400),
    ({name: 20 for name in ["arxiv", "books", "chat", "code", "codex", "forum", "math", "news", "wiki"]},
     {"arxiv": 1, "books": 1, "chat": 1, "code": 1, "codex": 1, "forum": 1, "math": 1, "news": 1, "wiki": 8}, 9, 300),
    ({"qa": 4, "web": 9}, {"qa": 1, "web": 6}, 3, 600),
    ({"math": 5}, {"math": 3}, 7, 50),
]


@case("Part 3", title="确定性：长时间运行偏差有界")
def p3_det_long(sol):
    for lengths, weights, bs, count in DET_CONFIGS:
        rep = Replay()
        b, _ = build(sol, rep, "b", lengths, weights, bs, allocation="deterministic")
        checked_run(rep, b, "b", count, ref.Tracker(lengths, weights, ref.det_bound(weights)), bs)


@case("Part 3", title="确定性：batch 内每个前缀也有界")
def p3_det_within_batch(sol):
    for lengths, weights, bs, count in [({"arxiv": 50, "books": 30, "code": 40}, {"arxiv": 3, "books": 2, "code": 2}, 101, 20),
                                        ({"math": 9, "web": 9}, {"math": 1, "web": 1}, 41, 10),
                                        ({"qa": 40, "web": 60}, {"qa": 1, "web": 3}, 201, 4)]:
        rep = Replay()
        b, _ = build(sol, rep, "b", lengths, weights, bs, allocation="deterministic")
        checked_run(rep, b, "b", count, ref.Tracker(lengths, weights, ref.det_bound(weights)), bs)


def resume_matches(sol, lengths, weights, bs, count, kw, resume_kw, why):
    """A checked run of `count` batches, then a pause after every batch k: save b through JSON, load into a fresh
    batcher b2 built with resume_kw, and from then on b and b2 must return the same batches."""
    rep = Replay()
    b, _ = build(sol, rep, "b", lengths, weights, bs, **kw)
    bound = ref.det_bound(weights) if kw.get("allocation", "deterministic") == "deterministic" else None
    checked_run(rep, b, "b", count, ref.Tracker(lengths, weights, bound), bs)
    for k in range(count + 1):
        rep = Replay()
        b, _ = build(sol, rep, "b", lengths, weights, bs, **kw)
        skip_batches(rep, b, "b", k)
        state = take_state(rep, b, "b", "state")
        b2, _ = build(sol, rep, "b2", lengths, weights, bs, **resume_kw)
        load(rep, b2, "b2", "state", state)
        run_pairs(rep, b, "b", b2, "b2", count - k,
                  lambda i, k=k: f"在第 {k} 个 batch 之后保存 b 的状态、加载到新对象 b2，之后两者的第 {i} 个 batch 应完全"
                                 f"相同（want 是 b 的，got 是 b2 的）：{why}")


@case("Part 3", title="确定性：保存后恢复，结果不变")
def p3_det_resume(sol):
    why = "state_dict() 要记录续跑所需的全部信息（包括分配规则用到的计数）"
    kw, other = {"allocation": "deterministic"}, {"allocation": "deterministic", "seed": 99}
    resume_matches(sol, {"code": 7, "math": 5, "web": 3}, EX_W, 6, 6, kw, other, why)
    resume_matches(sol, {"arxiv": 4, "books": 9, "code": 2}, {"arxiv": 5, "books": 3, "code": 2}, 7, 5, kw, other, why)


@case("Part 3", title="确定性：随机配置")
def p3_det_random(sol):
    r = rng()
    for _ in range(60):
        lengths, weights, bs = ref.random_config(r, divisible=False)
        count = r.randint(1, 12)
        rep = Replay()
        b, _ = build(sol, rep, "b", lengths, weights, bs, allocation="deterministic")
        checked_run(rep, b, "b", count, ref.Tracker(lengths, weights, ref.det_bound(weights)), bs)
    for _ in range(6):
        lengths, weights, bs = ref.random_config(r, divisible=False, max_k=4)
        resume_matches(sol, lengths, weights, bs, r.randint(1, 4), {"allocation": "deterministic"},
                       {"allocation": "deterministic"}, "state_dict() 要记录续跑所需的全部信息（包括分配规则用到的计数）")


STOCH_WHY = "state_dict() 要带上随机数生成器当前的状态（例如 random.Random.getstate()），只存 seed 或只存偏移量都不够"


@case("Part 3", title="随机：同一 seed 可复现，互不干扰")
def p3_stoch_same_seed(sol):
    seed = rng().randrange(10 ** 6)
    lengths, weights, bs, count = {"arxiv": 9, "books": 4, "code": 6}, {"arxiv": 3, "books": 1, "code": 2}, 5, 30
    rep = Replay()
    b1, _ = build(sol, rep, "b1", lengths, weights, bs, allocation="stochastic", seed=seed)
    b2, _ = build(sol, rep, "b2", lengths, weights, bs, allocation="stochastic", seed=seed)
    valid, first = det_check(ref.Tracker(lengths, weights), bs), []

    def check(batch):
        first.append(batch)
        return valid(batch)
    run_pairs(rep, b1, "b1", b2, "b2", count,
              lambda i: f"两个用同一个 seed 构造的 batcher 交替调用，第 {i} 个 batch 不同：每个 batcher 要有自己的 "
                        f"random.Random(seed) 实例，不能共用全局的 random，也不能不传 seed", check=check)
    rep = Replay()
    b3, _ = build(sol, rep, "b3", lengths, weights, bs, allocation="stochastic", seed=seed + 1)
    other = skip_batches(rep, b3, "b3", count)
    expect_true(other != first, f"seed 不同（{seed} 和 {seed + 1}）的两个 batcher 抽出了完全相同的 {count} 个 batch："
                                f"抽取应来自 random.Random(seed)")


@case("Part 3", title="随机：保存后精确续接")
def p3_stoch_resume(sol):
    r = rng()
    for lengths, weights, bs in [({"arxiv": 13, "books": 8, "code": 6}, {"arxiv": 5, "books": 3, "code": 2}, 7),
                                 ({"math": 4, "web": 3}, {"math": 1, "web": 1}, 3)]:
        seed = r.randrange(10 ** 6)
        resume_matches(sol, lengths, weights, bs, 6, {"allocation": "stochastic", "seed": seed},
                       {"allocation": "stochastic", "seed": seed + 12345}, STOCH_WHY)


@case("Part 3", title="随机：同一对象回到之前的状态")
def p3_stoch_rewind(sol):
    seed = rng().randrange(10 ** 6)
    lengths, weights, bs = {"arxiv": 13, "books": 8, "code": 6}, {"arxiv": 5, "books": 3, "code": 2}, 7
    rep = Replay()
    b, _ = build(sol, rep, "b", lengths, weights, bs, allocation="stochastic", seed=seed)
    skip_batches(rep, b, "b", 2)
    state = take_state(rep, b, "b", "s2")
    later = skip_batches(rep, b, "b", 3)
    load(rep, b, "b", "s2", state)
    for i in range(3):
        got = next_batch(rep, b, "b")
        expect(got, later[i], msg=f"load_state_dict(s2) 之后的第 {i} 个 batch 和保存后原来的第 {i} 个不同：同一个对象加载"
                                  f"状态时也要恢复随机数生成器的状态和各数据集的位置")


@case("Part 3", title="随机：抽样比例符合权重")
def p3_stoch_distribution(sol):
    r = rng()
    configs = [({"arxiv": 7, "books": 5, "code": 9}, {"arxiv": 1, "books": 1, "code": 2}, 7, 600),
               ({"math": 11, "web": 6, "qa": 4}, {"math": 5, "web": 3, "qa": 2}, 3, 1400),
               ({name: 8 for name in ["chat", "code", "forum", "news", "papers", "wiki"]},
                {"chat": 1, "code": 2, "forum": 3, "news": 4, "papers": 5, "wiki": 6}, 5, 1000)]
    for lengths, weights, bs, count in configs:
        seed = r.randrange(10 ** 6)
        rep = Replay()
        b, _ = build(sol, rep, "b", lengths, weights, bs, allocation="stochastic", seed=seed)
        tracker = ref.Tracker(lengths, weights)
        checked_run(rep, b, "b", count, tracker, bs)
        rep.lines[1:] = [f"batches = [b.next_batch() for _ in range({count})]",
                         "from collections import Counter",
                         'Counter(x.rpartition("-")[0] for batch in batches for x in batch)   # 各数据集被选中的次数']
        calling(rep.text())
        stat = ref.chi_square(tracker.counts, weights)
        n, total = tracker.n, sum(weights.values())
        expected = {k: round(n * w / total, 1) for k, w in sorted(weights.items())}
        expect_true(stat <= 40,
                    f"按权重 {weights} 抽了 {n} 次，各数据集被选中 {dict(sorted(tracker.counts.items()))} 次，期望约 "
                    f"{expected}；卡方统计量 {stat:.1f}，远超正常范围（自由度 {len(weights) - 1}，阈值 40）。[0, W) 内的整数"
                    f" x 落在前缀和区间 [P(k-1), P(k)) 时应选 d_k")


@case("Part 3", title="随机：随机配置")
def p3_stoch_random(sol):
    r = rng()
    for _ in range(40):
        lengths, weights, bs = ref.random_config(r, divisible=False)
        seed, count = r.randrange(10 ** 6), r.randint(1, 10)
        rep = Replay()
        b, _ = build(sol, rep, "b", lengths, weights, bs, allocation="stochastic", seed=seed)
        checked_run(rep, b, "b", count, ref.Tracker(lengths, weights), bs)
    for _ in range(6):
        lengths, weights, bs = ref.random_config(r, divisible=False, max_k=4)
        seed = r.randrange(10 ** 6)
        resume_matches(sol, lengths, weights, bs, r.randint(1, 4), {"allocation": "stochastic", "seed": seed},
                       {"allocation": "stochastic", "seed": seed + 1}, STOCH_WHY)


@case("Part 3", title="状态大小不随 batch 数增长")
def p3_state_size(sol):
    seed = rng().randrange(10 ** 6)
    for kw in ({"allocation": "deterministic"}, {"allocation": "stochastic", "seed": seed}):
        rep = Replay()
        b, _ = build(sol, rep, "b", EX_LEN, EX_W, 6, **kw)
        skip_batches(rep, b, "b", 1)
        small = len(json.dumps(take_state(rep, b, "b", "s1")))
        skip_batches(rep, b, "b", 2000)
        large = len(json.dumps(take_state(rep, b, "b", "s2001")))
        rep.run("len(json.dumps(s1)), len(json.dumps(s2001))", lambda: None)
        expect_true(abs(large - small) <= 16 * len(EX_W) + 300,
                    f"1 个 batch 之后状态的 JSON 有 {small} 个字符，2001 个 batch 之后有 {large} 个：状态的大小应只由数据集个数"
                    f" K 决定（随机分配再加上固定大小的生成器状态），不能随已经产出的 batch 数增长")
