import ast
import inspect
import re
import time
from enum import Enum

from judgelib import Mismatch, calling, case, expect_true, rng
import ref

SHOW = 60           # a replay lists at most this many lines (every sequence these tests make fits)
BIG = 10**9         # what a Level 1 test passes for token_budget / max_seqs when the answer's Engine requires them


# ---------------------------------------------------------------- what is looked at, and how it is shown

ALIASES = {         # short probe names used in the scripts below -> the expression the replay prints
    "free": "eng.allocator.num_free_or_evictable()",
    "refs": "[b.ref_count for b in eng.allocator.blocks.values()]",
    "cached": "[i for i, b in eng.allocator.blocks.items() if b.ref_count == 0 and b.content_hash is not None]",
}
ATTR = {"gen": "generated_token_ids", "status": "status", "blocks": "block_ids"}
_compiled = {}


def _expr(probe):
    """The Python expression (on `eng`) for a probe such as "free" or "sequences[0].block_ids"."""
    return ALIASES.get(probe, "eng." + probe)


def _read(eng, probe):
    code = _compiled.get(probe)
    if code is None:
        code = _compiled[probe] = compile(_expr(probe), "<probe>", "eval")
    return eval(code, {"eng": eng, "block_hash": ref.block_hash})


def _what(probe):
    """How a probe is named in a message."""
    m = re.fullmatch(r"sequences\[(\d+)\]\.(\w+)", probe)
    if m:
        rid, attr = m.groups()
        return {"generated_token_ids": f"请求 {rid} 生成的 token（generated_token_ids）",
                "status": f"请求 {rid} 的状态（status）",
                "block_ids": f"请求 {rid} 的块（block_ids）"}.get(attr, probe)
    m = re.fullmatch(r"allocator\.blocks\[(\d+)\]\.ref_count", probe)
    if m:
        return f"块 {m.group(1)} 的 ref_count"
    m = re.fullmatch(r"allocator\.blocks\[(\d+)\]\.content_hash == .*", probe)
    if m:
        return f"块 {m.group(1)} 的内容哈希"
    return {"free": "空闲或可驱逐的块数（allocator.num_free_or_evictable()）",
            "refs": "各个块的 ref_count",
            "cached": "处于缓存态（ref_count 为 0 且仍带着内容哈希）的块"}.get(probe, probe)


class _Status:
    """An expected SeqStatus member, by name (the answer's SeqStatus class is its own)."""

    def __init__(self, name):
        self.name = name


class _Shown(str):
    """A value already written as the user should read it (shown without quotes)."""

    def __repr__(self):
        return str(self)


def _show(x):
    if isinstance(x, _Status):
        return _Shown(f"SeqStatus.{x.name}")
    if isinstance(x, Enum):
        return _Shown(f"{type(x).__name__}.{x.name}")
    if x is None:
        return _Shown("None")
    return x


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _problem(got, want):
    """None when got is want; otherwise why not ("" for a plain wrong value, a sentence for a wrong type)."""
    if isinstance(want, _Status):
        if not isinstance(got, Enum):
            return f"类型不对：状态应该是 SeqStatus 的成员，这里得到 {type(got).__name__}"
        return None if got.name == want.name else ""
    if isinstance(want, bool):
        if not isinstance(got, bool):
            return f"类型不对：应该是 bool，这里得到 {type(got).__name__}"
        return None if got == want else ""
    if isinstance(want, int):
        if not _is_int(got):
            return f"类型不对：应该是 int，这里得到 {type(got).__name__}"
        return None if got == want else ""
    if isinstance(want, list):
        if not isinstance(got, list):
            return f"类型不对：应该是 list，这里得到 {type(got).__name__}"
        if not all(_is_int(x) for x in got):
            return "类型不对：列表里应该都是 int"
        return None if got == want else ""
    return None if got is want else ""


def _verdict(what, problem):
    """'<what>不对', or '<what>：<type problem>' when _problem named one."""
    if problem:
        return f"{what}：{problem}"
    return what + (" " if what[-1:].isascii() and what[-1:].isalnum() else "") + "不对"


def _fail(problem, what, why, call, want, got):
    """Fails the case for a value that _problem rejected (problem is what _problem said)."""
    msg = _verdict(what, problem) + (f"。{why}" if why and not problem else "")
    raise Mismatch(msg, call, _show(want), _show(got))


# ---------------------------------------------------------------- the replay

def _add_code(prompt, max_tokens):
    return f"eng.add_request({prompt!r}, max_tokens={max_tokens})"


def _replay(head, done, last, mark="# <- 这一步"):
    """Code to paste into 自定义运行: the engine, the calls so far (done: ("add", prompt, max_tokens, id) and
    ("step",)), and `last`, the failing line."""
    lines = []
    for op in done:
        if op[0] == "add":
            lines.append([_add_code(op[1], op[2]), f"# -> {op[3]}", 0])
        elif lines and lines[-1][2]:
            lines[-1][2] += 1
        else:
            lines.append(["", "", 1])
    rows = [(f"for _ in range({k}): eng.step()" if k > 1 else "eng.step()", "") if k else (code, note)
            for code, note, k in lines] + [(last, mark)]
    skipped = max(0, len(rows) - SHOW)
    rows = rows[skipped:]
    width = max(len(code) for code, note in rows if note) + 3
    out = [head] + ([f"# …（省略了前面 {skipped} 行，所以不能直接粘贴重现）"] if skipped else [])
    return "\n".join(out + [(code.ljust(width) + note) if note else code for code, note in rows])


def _engine(sol, config):
    """A fresh Engine(**config) and the replay's first line. A Level 1 test passes only num_blocks; when the
    answer's Engine requires token_budget / max_seqs, they get BIG."""
    cls = getattr(sol, "Engine", None)
    expect_true(isinstance(cls, type), "没有找到 Engine 类（类名要和题目一致）")
    kwargs = dict(config)
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        params = {}
    for name in ("token_budget", "max_seqs"):
        if name not in kwargs and name in params and params[name].default is inspect.Parameter.empty:
            kwargs[name] = BIG
    head = "eng = Engine(" + ", ".join(f"{k}={'10**9' if v == BIG else v}" for k, v in kwargs.items()) + ")"
    calling(f"{head}   # <- 这一步")
    try:
        eng = cls(**kwargs)
    except Exception:
        calling(f"{head}   # <- 这一步抛出异常")
        raise
    expect_true(hasattr(getattr(eng, "allocator", None), "num_free_or_evictable"),
                "Engine 没有属性 allocator（题目要求把引擎的 BlockAllocator 放在 eng.allocator 里）")
    expect_true(hasattr(eng, "sequences"),
                "Engine 没有属性 sequences（题目要求 eng.sequences[request_id] 是这个请求的 Sequence）")
    return eng, head


def _guarded(fn, head, done, last):
    """fn(); if it raises, the replay marks `last` as the line that raised."""
    try:
        return fn()
    except Exception:
        calling(_replay(head, done, last, mark="# <- 这一步抛出异常"))
        raise


# ---------------------------------------------------------------- hand-written sequences

def _value(text):
    """A literal such as [0, 1], 5, True or SeqStatus.RUNNING (the scripts below are the only input)."""
    text = text.strip()
    if text.startswith("SeqStatus."):
        return _Status(text[len("SeqStatus."):])
    return ast.literal_eval(text)


_SEQ = re.compile(r"seq\s+(\d+)\s+([A-Z]+)\s+(\[[^\]]*\])\s+(\[[^\]]*\])")


def _parse(script):
    """The first line is Engine(...); then lines like
        add_request([7], max_tokens=2) -> 0      # why
        step()
        sequences[0].block_ids -> [0, 1]        # why
        free -> 5                                (also refs, cached; see ALIASES)
        seq 0  RUNNING  [0, 1]  [15]            (status, block_ids and generated_token_ids of request 0)
    become (config, ops) with ops ("add", prompt, max_tokens, id, why), ("step",), ("probe", text, want, why)."""
    lines = [line.strip() for line in script.strip().splitlines() if line.strip()]
    node = ast.parse(lines[0], mode="eval").body
    config = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    ops = []
    for line in lines[1:]:
        body, _, why = line.partition("#")
        body, why = body.strip(), why.strip()
        m = _SEQ.fullmatch(body)
        if body == "step()":
            ops.append(("step",))
        elif body.startswith("add_request("):
            call, _, want = body.partition("->")
            node = ast.parse(call.strip(), mode="eval").body
            ops.append(("add", ast.literal_eval(node.args[0]), ast.literal_eval(node.keywords[0].value),
                        int(want), why))
        elif m:
            rid, status, blocks, gen = m.groups()
            ops += [("probe", f"sequences[{rid}].generated_token_ids", _value(gen), why),
                    ("probe", f"sequences[{rid}].status", _Status(status), why),
                    ("probe", f"sequences[{rid}].block_ids", _value(blocks), why)]
        else:
            expr, arrow, want = body.rpartition("->")
            assert arrow, f"missing '->' in {line!r}"
            ops.append(("probe", expr.strip(), _value(want), why))
    return config, ops


def run(sol, script):
    """Runs the script on a fresh Engine, checking each value as soon as it can be looked at."""
    config, ops = _parse(script)
    eng, head = _engine(sol, config)
    done = []
    for op in ops:
        if op[0] == "add":
            _, prompt, max_tokens, want, why = op
            code = _add_code(prompt, max_tokens)
            fn = getattr(eng, "add_request", None)
            expect_true(callable(fn), "Engine 没有 add_request 方法（方法名要和题目一致）")
            calling(_replay(head, done, code))
            got = _guarded(lambda: fn(list(prompt), max_tokens), head, done, code)
            problem = _problem(got, want)
            if problem is not None:
                _fail(problem, "add_request 返回的值", why, _replay(head, done, f"print({code})"), want, got)
            done.append(("add", prompt, max_tokens, want))
        elif op[0] == "step":
            fn = getattr(eng, "step", None)
            expect_true(callable(fn), "Engine 没有 step 方法（方法名要和题目一致）")
            calling(_replay(head, done, "eng.step()"))
            _guarded(fn, head, done, "eng.step()")
            done.append(("step",))
        else:
            _, probe, want, why = op
            code = f"print({_expr(probe)})"
            calling(_replay(head, done, code))
            got = _guarded(lambda: _read(eng, probe), head, done, code)
            problem = _problem(got, want)
            if problem is not None:
                _fail(problem, _what(probe), why, _replay(head, done, code), want, got)


# ---------------------------------------------------------------- random workloads against ref

def _keys(n, level):
    """What is compared after every step, in the order it is checked (ref.Model.observe's probes)."""
    keys = [(kind, rid) for kind in ("gen", "status", "blocks") for rid in range(n)] + [("free",)]
    return keys + ([("refs",), ("cached",)] if level >= 4 else [])


def _probe(key):
    return f"sequences[{key[1]}].{ATTR[key[0]]}" if key[0] in ATTR else key[0]


def _outcome(sol, level, config, ops, want):
    """Runs ops on a fresh answer; None if everything agrees with want (ref.expected's list), else where the
    first difference is: {"k": index of the op, "probe": what was looked at or None, "exc" or "got"/"want"}."""
    eng, _ = _engine(sol, config)
    n = 0
    for k, (op, w) in enumerate(zip(ops, want)):
        if op[0] == "add":
            try:
                got = eng.add_request(list(op[1]), op[2])
            except Exception as e:
                return {"k": k, "probe": None, "exc": e}
            if _problem(got, w) is not None:
                return {"k": k, "probe": None, "exc": None, "got": got, "want": w}
            n += 1
            continue
        try:
            eng.step()
        except Exception as e:
            return {"k": k, "probe": None, "exc": e}
        seen = dict(w)
        for key in _keys(n, level):
            probe = _probe(key)
            try:
                got = _read(eng, probe)
            except Exception as e:
                return {"k": k, "probe": probe, "exc": e}
            expect = _Status(seen[key]) if key[0] == "status" else seen[key]
            if _problem(got, expect) is not None:
                return {"k": k, "probe": probe, "exc": None, "got": got, "want": expect}
    return None


def _shrink(sol, level, config, ops, want, bad, budget=0.6):
    """Drops requests and steps that the failure does not need, keeping only fair workloads
    (ref.expected), so the replay shown is short."""
    ops, want = ops[: bad["k"] + 1], want[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(ops) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(ops) and time.process_time() < stop:
            cand = ops[:i] + ops[i + size:]
            fair = ref.expected(level, config, cand) if cand else None
            out = _outcome(sol, level, config, cand, fair) if fair is not None else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                ops, want, bad, progress = cand[: out["k"] + 1], fair[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return ops, want, bad


def random_check(sol, level, workloads):
    r = rng()
    for _ in range(workloads):
        config, ops, want = ref.random_workload(r, level)
        bad = _outcome(sol, level, config, ops, want)
        if bad is None:
            continue
        ops, want, bad = _shrink(sol, level, config, ops, want, bad)
        _, head = _engine(sol, config)
        k, op = bad["k"], ops[bad["k"]]
        done = [("add", o[1], o[2], w) if o[0] == "add" else o for o, w in zip(ops[:k], want[:k])]
        if op[0] == "add":
            code = _add_code(op[1], op[2])
            if bad["exc"] is not None:
                calling(_replay(head, done, code, mark="# <- 这一步抛出异常"))
                raise bad["exc"]
            _fail(_problem(bad["got"], bad["want"]), "随机工作负载：add_request 返回的值", "",
                  _replay(head, done, f"print({code})"), bad["want"], bad["got"])
        if bad["probe"] is None:        # the step itself raised
            calling(_replay(head, done, "eng.step()", mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        code = f"print({_expr(bad['probe'])})"
        done.append(("step",))
        if bad["exc"] is not None:
            calling(_replay(head, done, code, mark="# <- 这一步抛出异常"))
            raise bad["exc"]
        steps = sum(o[0] == "step" for o in ops[: k + 1])
        problem = _problem(bad["got"], bad["want"])
        raise Mismatch(f"随机工作负载：第 {steps} 次 step() 之后，{_verdict(_what(bad['probe']), problem)}"
                       "（已自动删去与出错无关的请求和 step）",
                       _replay(head, done, code), _show(bad["want"]), _show(bad["got"]))


# ---------------------------------------------------------------- the statement's examples

EXAMPLE_L1 = """
    Engine(num_blocks=10)
    add_request([2, 9, 4, 7, 1], max_tokens=3)   -> 0     # 5 个 token 的提示
    add_request([5, 3, 8], max_tokens=3)         -> 1     # 3 个 token 的提示
    step()
    sequences[0].block_ids              -> [0, 1]         # ceil((5 + 1) / 4) = 2 个块
    sequences[0].generated_token_ids    -> [15]
    sequences[1].block_ids              -> [2]            # ceil((3 + 1) / 4) = 1 个块
    sequences[1].generated_token_ids    -> [3]
"""

EXAMPLE_L2 = """
    Engine(num_blocks=20, token_budget=6, max_seqs=2)
    add_request([1, 2, 3, 4], max_tokens=5)  -> 0
    add_request([5, 6, 7], max_tokens=5)     -> 1
    add_request([8, 9], max_tokens=5)        -> 2
    step()
    sequences[0].generated_token_ids -> [14]           # A 被准入 (0+4<=6)
    sequences[1].generated_token_ids -> []             # B 放不下 (4+3=7>6)，仍在等待
    step()
    sequences[0].generated_token_ids -> [14, 13]
    sequences[1].generated_token_ids -> [4]            # B 现在放得下 (1+3<=6)
    sequences[2].generated_token_ids -> []             # 运行数达到 max_seqs=2：C 仍在等待
    step()
    sequences[0].generated_token_ids -> [14, 13, 0]
    sequences[1].generated_token_ids -> [4, 2]
    sequences[2].generated_token_ids -> []             # 运行数已经是 max_seqs=2
"""

EXAMPLE_L3 = """
    Engine(num_blocks=8, token_budget=20, max_seqs=5)
    add_request([2, 9, 4, 7, 1], max_tokens=2)  -> 0
    add_request([0, 3], max_tokens=5)           -> 1
    add_request([6, 6, 6], max_tokens=4)        -> 2
    step()
    seq 0  RUNNING   [0, 1]  [15]
    seq 1  FINISHED  []      [16]       # 预填充期间就遇到 EOS：从未进入 RUNNING
    seq 2  RUNNING   [2]     [11]
    free -> 5                           # 8 - 2(A) - 1(C) = 5
    step()
    seq 0  FINISHED  []      [15, 9]    # 达到 max_tokens=2：释放 [0, 1]
    seq 2  RUNNING   [2, 1]  [11, 8]    # token 数达到 4：需要第二个块
    free -> 6
    step()
    step()
    seq 2  FINISHED  []      [11, 8, 3, 6]
    free -> 8                           # 每个块都已归还给分配器
"""

EXAMPLE_L4 = """
    Engine(num_blocks=4, token_budget=40, max_seqs=5)
    add_request([5, 5, 5, 5, 2], max_tokens=2)  -> 0
    add_request([5, 5, 5, 5, 9], max_tokens=4)  -> 1
    step()
    seq 0  RUNNING   [0, 1]  [12]
    seq 1  RUNNING   [0, 2]  [2]        # 块 0 是 A 的块，被复用，而不是复制
    allocator.blocks[0].ref_count -> 2
    step()
    seq 0  FINISHED  []      [12, 4]
    seq 1  RUNNING   [0, 2]  [2, 10]
    allocator.blocks[0].ref_count -> 1  # 仍被 B 单独持有
    step()
    step()
    seq 1  FINISHED  []      [2, 10, 12, 4]
    allocator.blocks[0].ref_count -> 0
    allocator.blocks[0].content_hash == block_hash(0, [5, 5, 5, 5]) -> True     # 块 0 是缓存态，仍保存着 [5, 5, 5, 5]
    add_request([1, 1, 1, 1, 1, 1, 1, 1], max_tokens=1)  -> 2
    step()
    seq 2  FINISHED  []      [12]
    allocator.blocks[0].content_hash == block_hash(0, [5, 5, 5, 5]) -> False    # 只有 2 个空闲块：缓存中的块 0 被驱逐
    add_request([5, 5, 5, 5, 7], max_tokens=1)  -> 3
    step()
    seq 3  FINISHED  []      [0]
"""

EXAMPLE_L5 = """
    Engine(num_blocks=3, token_budget=30, max_seqs=5)
    add_request([6, 6], max_tokens=5)        -> 0
    add_request([2, 9, 4, 7], max_tokens=5)  -> 1
    step()
    step()
    seq 0  RUNNING   [0]     [6, 11]
    seq 1  RUNNING   [1, 2]  [11, 8]
    free -> 0                           # 3 个块全部在用
    step()
    seq 0  FINISHED  []      [6, 11, 8, 3, 6]     # 最后一次被重新准入时达到 max_tokens=5
    seq 1  WAITING   []      [11, 8, 3, 6]        # 在距 max_tokens=5 还差一个 token 时被抢占
    step()
    seq 1  FINISHED  []      [11, 8, 3, 6, 11]
    free -> 3                           # 每个块都空闲或处于缓存态——没有泄漏
"""


# ---------------------------------------------------------------- Level 1
@case("Level 1", sample=True, title="题目例子")
def l1_example(sol):
    run(sol, EXAMPLE_L1)


@case("Level 1", title="块数是 ceil((len + 1) / 4)，预填充后是 RUNNING")
def l1_block_count(sol):
    run(sol, """
        Engine(num_blocks=12)
        add_request([7], max_tokens=4)                        -> 0
        add_request([1, 2, 3], max_tokens=4)                  -> 1
        add_request([4, 4, 4, 4], max_tokens=4)               -> 2
        add_request([3, 1, 4, 1, 5, 9, 2], max_tokens=4)      -> 3
        add_request([8, 6, 7, 5, 3, 0, 9, 1], max_tokens=4)   -> 4
        step()
        seq 0  RUNNING  [0]        [8]      # 1 + 1 个 token：1 个块
        seq 1  RUNNING  [1]        [5]      # 3 + 1 = 4 个 token：正好 1 个块
        seq 2  RUNNING  [2, 3]     [7]      # 4 + 1 = 5 个 token：2 个块
        seq 3  RUNNING  [4, 5]     [0]      # 7 + 1 = 8 个 token：2 个块
        seq 4  RUNNING  [6, 7, 8]  [13]     # 8 + 1 = 9 个 token：3 个块
        free -> 3
    """)


@case("Level 1", title="step() 之前请求一直在等待")
def l1_waiting_until_step(sol):
    run(sol, """
        Engine(num_blocks=4)
        step()                              # 还没有任何请求
        free -> 4
        add_request([5, 3, 8], max_tokens=3)    -> 0
        seq 0  WAITING  []  []              # 只入队，还没有预填充
        free -> 4
        add_request([7], max_tokens=2)          -> 1
        step()
        seq 0  RUNNING  [0]  [3]
        seq 1  RUNNING  [1]  [8]
        free -> 2
    """)


@case("Level 1", cpu=2, title="随机工作负载（对照参考实现）")
def l1_random(sol):
    random_check(sol, 1, workloads=400)


# ---------------------------------------------------------------- Level 2
@case("Level 2", sample=True, title="题目例子")
def l2_example(sol):
    run(sol, EXAMPLE_L2)


@case("Level 2", title="解码前先为新 token 预留块")
def l2_new_block_before_token(sol):
    run(sol, """
        Engine(num_blocks=10, token_budget=10, max_seqs=4)
        add_request([7], max_tokens=10)     -> 0
        step()
        seq 0  RUNNING  [0]     [8]
        step()
        seq 0  RUNNING  [0]     [8, 3]
        step()
        seq 0  RUNNING  [0]     [8, 3, 6]   # 4 个 token：块 0 正好装满
        free -> 9
        step()
        seq 0  RUNNING  [0, 1]  [8, 3, 6, 11]   # 第 5 个 token 之前先要一个新块
        free -> 8
    """)


@case("Level 2", title="按准入顺序解码")
def l2_decode_order(sol):
    run(sol, """
        Engine(num_blocks=10, token_budget=10, max_seqs=4)
        add_request([7], max_tokens=10)     -> 0
        add_request([5], max_tokens=10)     -> 1
        step()
        step()
        step()
        seq 0  RUNNING  [0]     [8, 3, 6]
        seq 1  RUNNING  [1]     [6, 11, 8]
        step()
        seq 0  RUNNING  [0, 2]  [8, 3, 6, 11]   # 先解码 A，它先拿到新块 2
        seq 1  RUNNING  [1, 3]  [6, 11, 8, 3]
        free -> 6
    """)


@case("Level 2", title="每解码一个序列也计入预算")
def l2_budget_counts_decode(sol):
    run(sol, """
        Engine(num_blocks=20, token_budget=5, max_seqs=3)
        add_request([7], max_tokens=10)                 -> 0
        add_request([2, 9, 4, 7, 3], max_tokens=10)     -> 1
        step()
        seq 0  RUNNING  [0]  [8]
        seq 1  WAITING  []   []             # 1 + 5 > 5
        step()
        seq 0  RUNNING  [0]  [8, 3]
        seq 1  WAITING  []   []             # 解码 A 记 1 个 token：1 + 5 > 5
        step()
        seq 0  RUNNING  [0]  [8, 3, 6]
        seq 1  WAITING  []   []
        free -> 19
    """)


@case("Level 2", title="正好用满预算也可以准入")
def l2_budget_exact(sol):
    run(sol, """
        Engine(num_blocks=20, token_budget=6, max_seqs=3)
        add_request([7], max_tokens=10)                 -> 0
        step()
        add_request([2, 9, 4, 7, 3], max_tokens=10)     -> 1
        step()
        seq 0  RUNNING  [0]     [8, 3]
        seq 1  RUNNING  [1, 2]  [0]         # 1 + 5 = 6 <= 6
        step()
        seq 0  RUNNING  [0]     [8, 3, 6]
        seq 1  RUNNING  [1, 2]  [0, 1]
        free -> 17
    """)


@case("Level 2", title="放不下的请求挡住后面的请求")
def l2_fifo(sol):
    run(sol, """
        Engine(num_blocks=20, token_budget=6, max_seqs=3)
        add_request([1, 2, 3, 4], max_tokens=10)        -> 0
        add_request([2, 9, 4, 7, 3], max_tokens=10)     -> 1
        add_request([7], max_tokens=10)                 -> 2
        step()
        seq 0  RUNNING  [0, 1]  [14]
        seq 1  WAITING  []      []          # 4 + 5 > 6
        seq 2  WAITING  []      []          # 4 + 1 <= 6，但不能越过 B 提前准入
        step()
        seq 0  RUNNING  [0, 1]  [14, 13]
        seq 1  RUNNING  [2, 3]  [0]         # 1 + 5 = 6
        seq 2  WAITING  []      []          # 6 + 1 > 6
        step()
        seq 0  RUNNING  [0, 1]  [14, 13, 0]
        seq 1  RUNNING  [2, 3]  [0, 1]
        seq 2  RUNNING  [4]     [8]         # 2 + 1 <= 6
        free -> 15
    """)


@case("Level 2", title="运行数达到 max_seqs 就停止准入")
def l2_max_seqs(sol):
    run(sol, """
        Engine(num_blocks=20, token_budget=20, max_seqs=2)
        add_request([7], max_tokens=10)     -> 0
        add_request([5], max_tokens=10)     -> 1
        add_request([6], max_tokens=10)     -> 2
        step()
        seq 0  RUNNING  [0]  [8]
        seq 1  RUNNING  [1]  [6]
        seq 2  WAITING  []   []             # 运行数已经是 max_seqs=2
        step()
        seq 0  RUNNING  [0]  [8, 3]
        seq 1  RUNNING  [1]  [6, 11]
        seq 2  WAITING  []   []
        free -> 18
    """)


@case("Level 2", cpu=2, title="随机工作负载（对照参考实现）")
def l2_random(sol):
    random_check(sol, 2, workloads=400)


# ---------------------------------------------------------------- Level 3
@case("Level 3", sample=True, title="题目例子")
def l3_example(sol):
    run(sol, EXAMPLE_L3)


@case("Level 3", title="预填充生成的 token 就可能让序列停止")
def l3_stop_in_prefill(sol):
    run(sol, """
        Engine(num_blocks=4, token_budget=20, max_seqs=2)
        add_request([7], max_tokens=1)          -> 0
        add_request([0, 3], max_tokens=5)       -> 1
        add_request([5, 3, 8], max_tokens=3)    -> 2
        step()
        seq 0  FINISHED  []   [8]           # 预填充就达到 max_tokens=1
        seq 1  FINISHED  []   [16]          # 预填充就生成了 EOS
        seq 2  RUNNING   [0]  [3]           # A、B 已停止，不占 max_seqs 的名额
        free -> 3
        step()
        seq 0  FINISHED  []      [8]
        seq 1  FINISHED  []      [16]
        seq 2  RUNNING   [0, 1]  [3, 6]
        free -> 2
    """)


@case("Level 3", title="解码出 EOS 就停止并释放")
def l3_eos_in_decode(sol):
    run(sol, """
        Engine(num_blocks=4, token_budget=20, max_seqs=2)
        add_request([1, 2, 3], max_tokens=10)   -> 0
        step()
        step()
        step()
        seq 0  RUNNING   [0, 1]  [5, 15, 9]
        free -> 2
        step()
        seq 0  FINISHED  []      [5, 15, 9, 16]     # EOS：block_ids 变为空
        free -> 4
        step()
        seq 0  FINISHED  []      [5, 15, 9, 16]     # 已停止的序列不会再被解码
        free -> 4
    """)


@case("Level 3", title="同一步里停止的序列马上让出名额和块")
def l3_finished_frees_slot(sol):
    run(sol, """
        Engine(num_blocks=10, token_budget=20, max_seqs=1)
        add_request([7], max_tokens=2)          -> 0
        add_request([5, 3, 8], max_tokens=2)    -> 1
        step()
        seq 0  RUNNING   [0]  [8]
        seq 1  WAITING   []   []
        step()
        seq 0  FINISHED  []   [8, 3]        # 解码时停止，块 0 立即归还
        seq 1  RUNNING   [0]  [3]           # 同一步的准入阶段：名额已经空出来
        step()
        seq 1  FINISHED  []   [3, 6]
        free -> 10
    """)


@case("Level 3", title="解码后停止的序列仍记一个预算 token")
def l3_budget_after_finish(sol):
    run(sol, """
        Engine(num_blocks=10, token_budget=4, max_seqs=4)
        add_request([7], max_tokens=2)              -> 0
        add_request([4, 4, 4, 4], max_tokens=2)     -> 1
        step()
        seq 0  RUNNING   [0]     [8]
        seq 1  WAITING   []      []         # 1 + 4 > 4
        step()
        seq 0  FINISHED  []      [8, 3]
        seq 1  WAITING   []      []         # A 这一步被解码过，记 1 个 token：1 + 4 > 4
        step()
        seq 1  RUNNING   [0, 1]  [7]        # 0 + 4 <= 4
        step()
        seq 1  FINISHED  []      [7, 7]
        free -> 10
    """)


@case("Level 3", cpu=2, title="随机工作负载（对照参考实现）")
def l3_random(sol):
    random_check(sol, 3, workloads=400)


# ---------------------------------------------------------------- Level 4
@case("Level 4", sample=True, title="题目例子")
def l4_example(sol):
    run(sol, EXAMPLE_L4)


@case("Level 4", title="共享前缀：命中只增加一次引用计数")
def l4_shared_prefix(sol):
    run(sol, """
        Engine(num_blocks=8, token_budget=40, max_seqs=4)
        add_request([1, 1, 1, 1, 2, 2, 2, 2, 3], max_tokens=2)   -> 0
        add_request([1, 1, 1, 1, 2, 2, 2, 2, 5], max_tokens=2)   -> 1
        step()
        seq 0  RUNNING   [0, 1, 2]  [5]
        seq 1  RUNNING   [0, 1, 3]  [7]     # 前两个满块命中 A 刚注册的块 0、1
        refs -> [2, 2, 1, 1, 0, 0, 0, 0]
        free -> 4
        step()
        seq 0  FINISHED  []  [5, 15]
        seq 1  FINISHED  []  [7, 7]
        refs -> [0, 0, 0, 0, 0, 0, 0, 0]
        cached -> [0, 1]
        add_request([1, 1, 1, 1, 2, 2, 2, 2, 7], max_tokens=2)   -> 2
        step()
        seq 2  RUNNING   [0, 1, 3]  [9]     # 命中缓存态的块 0、1
        refs -> [1, 1, 0, 1, 0, 0, 0, 0]
        cached -> []
        free -> 5
    """)


@case("Level 4", title="token 数是 4 的倍数时，最后一个满块也要查")
def l4_lookup_all_full_blocks(sol):
    run(sol, """
        Engine(num_blocks=8, token_budget=40, max_seqs=4)
        add_request([1, 1, 1, 1, 2, 2, 2, 2, 3], max_tokens=2)   -> 0
        add_request([1, 1, 1, 1, 2, 2, 2, 2], max_tokens=2)      -> 1
        step()
        seq 0  RUNNING   [0, 1, 2]  [5]
        seq 1  RUNNING   [0, 1, 3]  [10]    # 两个满块都命中，再为新 token 分配块 3
        refs -> [2, 2, 1, 1, 0, 0, 0, 0]
        free -> 4
    """)


@case("Level 4", title="在第一个未命中处停止查找")
def l4_stop_at_first_miss(sol):
    run(sol, """
        Engine(num_blocks=4, token_budget=40, max_seqs=4)
        add_request([1, 1, 1, 1, 2, 2, 2, 2, 3], max_tokens=1)   -> 0
        step()
        cached -> [0, 1]
        add_request([4], max_tokens=2)      -> 1
        add_request([4], max_tokens=2)      -> 2
        add_request([4], max_tokens=2)      -> 3
        step()
        seq 3  RUNNING   [0]  [5]           # 空闲块不够：驱逐最久未用的缓存块 0
        cached -> [1]
        step()
        seq 1  FINISHED  []  [5, 15]
        seq 2  FINISHED  []  [5, 15]
        seq 3  FINISHED  []  [5, 15]
        add_request([1, 1, 1, 1, 2, 2, 2, 2, 3], max_tokens=2)   -> 4
        step()
        seq 4  RUNNING   [0, 3, 2]  [5]     # 第 0 个满块未命中：停止查找，块 1 虽然还在缓存里也不复用
        cached -> [1]
        refs -> [1, 0, 1, 1]
    """)


@case("Level 4", title="预填充生成的 token 填满的块也要注册")
def l4_register_after_prefill(sol):
    run(sol, """
        Engine(num_blocks=6, token_budget=40, max_seqs=4)
        add_request([5, 3, 8], max_tokens=3)        -> 0
        step()
        seq 0  RUNNING   [0]     [3]        # [5, 3, 8, 3]：块 0 被预填充生成的 token 填满
        add_request([5, 3, 8, 3, 1], max_tokens=2)  -> 1
        step()
        seq 0  RUNNING   [0, 1]  [3, 6]
        seq 1  RUNNING   [0, 2]  [6]        # 命中块 0
        refs -> [2, 1, 1, 0, 0, 0]
        step()
        seq 0  FINISHED  []  [3, 6, 11]
        seq 1  FINISHED  []  [6, 11]
        cached -> [0]
    """)


@case("Level 4", title="解码填满的块也要注册")
def l4_register_on_decode(sol):
    run(sol, """
        Engine(num_blocks=6, token_budget=40, max_seqs=4)
        add_request([7], max_tokens=5)      -> 0
        step()
        step()
        step()
        seq 0  RUNNING   [0]     [8, 3, 6]  # [7, 8, 3, 6]：块 0 在解码时被填满
        add_request([7, 8, 3, 6, 11], max_tokens=2)     -> 1
        step()
        seq 0  RUNNING   [0, 1]  [8, 3, 6, 11]
        seq 1  RUNNING   [0, 2]  [8]        # 命中块 0
        refs -> [2, 1, 1, 0, 0, 0]
        step()
        seq 0  FINISHED  []  [8, 3, 6, 11, 8]
        seq 1  FINISHED  []  [8, 3]
        cached -> [0]
        add_request([7, 8, 3, 6, 1], max_tokens=2)      -> 2
        step()
        seq 2  RUNNING   [0, 2]  [15]       # 命中缓存态的块 0
        refs -> [1, 0, 1, 0, 0, 0]
    """)


@case("Level 4", title="解码填满的第二个块按链式哈希注册")
def l4_register_chained_hash(sol):
    run(sol, """
        Engine(num_blocks=6, token_budget=40, max_seqs=4)
        add_request([7], max_tokens=10)     -> 0
        step()
        step()
        step()
        step()
        step()
        step()
        step()
        step()
        seq 0  RUNNING   [0, 1, 2]  [8, 3, 6, 11, 8, 3, 6, 11]     # 块 1 装的是第 4..7 个 token，在解码时填满
        add_request([7, 8, 3, 6, 11, 8, 3, 6, 11], max_tokens=2)   -> 1
        step()
        seq 1  RUNNING   [0, 1, 3]  [8]     # 两个满块都命中：块 1 的内容哈希要从块 0 的哈希链式算起
        refs -> [2, 2, 1, 1, 0, 0]
        step()
        seq 0  FINISHED  []  [8, 3, 6, 11, 8, 3, 6, 11, 8, 3]
        seq 1  FINISHED  []  [8, 3]
        cached -> [0, 1]
    """)


@case("Level 4", cpu=2, title="随机工作负载（对照参考实现）")
def l4_random(sol):
    random_check(sol, 4, workloads=300)


# ---------------------------------------------------------------- Level 5
@case("Level 5", sample=True, title="题目例子")
def l5_example(sol):
    run(sol, EXAMPLE_L5)


@case("Level 5", title="抢占准入时间最晚的序列")
def l5_victim_latest_admitted(sol):
    run(sol, """
        Engine(num_blocks=3, token_budget=9, max_seqs=3)
        add_request([7], max_tokens=2)                  -> 0
        add_request([6], max_tokens=8)                  -> 1
        add_request([3, 1, 4, 1, 5, 9], max_tokens=3)   -> 2
        step()
        seq 0  RUNNING   [0]     [8]
        seq 1  WAITING   []      [7]        # B 拿到块 1 后，被晚准入的 C 抢占
        seq 2  RUNNING   [1, 2]  [4]
        free -> 0
        step()
        seq 0  FINISHED  []      [8, 3]
        seq 1  RUNNING   [0]     [7, 7]     # 用 A 归还的块 0 重新准入
        seq 2  RUNNING   [1, 2]  [4, 2]
        step()
        seq 1  RUNNING   [0]     [7, 7, 7]
        seq 2  FINISHED  []      [4, 2, 10]
        free -> 2
        refs -> [1, 0, 0]
        cached -> [1, 2]
    """)


@case("Level 5", title="解码阶段被抢占的序列仍记一个预算 token")
def l5_preempted_still_charged(sol):
    run(sol, """
        Engine(num_blocks=2, token_budget=5, max_seqs=2)
        add_request([1, 2], max_tokens=3)       -> 0
        step()
        add_request([5, 3, 8], max_tokens=2)    -> 1
        step()
        seq 0  RUNNING   [0]  [10, 12]
        seq 1  RUNNING   [1]  [3]
        free -> 0
        step()
        seq 0  FINISHED  []   [10, 12, 4]   # A 要第 2 个块：抢占 B，然后停止
        seq 1  WAITING   []   [3]           # B 被抢占也记 1 个 token：2 + 4 > 5，这一步不能重新准入
        free -> 2
        cached -> [0]
        step()
        seq 1  FINISHED  []   [3, 6]
    """)


@case("Level 5", title="被抢占的序列回到队列最前面")
def l5_preempt_to_front(sol):
    run(sol, """
        Engine(num_blocks=2, token_budget=5, max_seqs=2)
        add_request([1, 2], max_tokens=3)       -> 0
        step()
        add_request([5, 3, 8], max_tokens=2)    -> 1
        add_request([7], max_tokens=2)          -> 2
        step()
        seq 1  RUNNING   [1]  [3]
        seq 2  WAITING   []   []
        step()
        seq 0  FINISHED  []   [10, 12, 4]
        seq 1  WAITING   []   [3]           # 被抢占：放在 C 前面
        seq 2  WAITING   []   []            # B 在队首放不下 (2 + 4 > 5)，C 不能越过它
        step()
        seq 1  FINISHED  []   [3, 6]
        seq 2  RUNNING   [1]  [8]
        step()
        seq 2  FINISHED  []   [8, 3]
    """)


@case("Level 5", title="重新准入按当前的 len(token_ids) 计预算")
def l5_readmit_counts_tokens(sol):
    run(sol, """
        Engine(num_blocks=2, token_budget=10, max_seqs=2)
        add_request([6], max_tokens=9)              -> 0
        step()
        step()
        step()
        add_request([2, 9, 4, 7, 1], max_tokens=2)  -> 1
        step()
        seq 0  WAITING   []      [7, 7, 7, 7]   # 解码要第 2 个块：被晚准入的 B 抢占
        seq 1  RUNNING   [0, 1]  [15]
        step()
        seq 0  RUNNING   [0, 1]  [7, 7, 7, 7, 7]    # 1 + 5 <= 10：按 5 个 token 计，不是提示的 1 个
        seq 1  FINISHED  []      [15, 9]
        free -> 0
    """)


@case("Level 5", title="重新准入时查找 token_ids 的全部满块")
def l5_lookup_whole_tokens(sol):
    run(sol, """
        Engine(num_blocks=3, token_budget=11, max_seqs=2)
        add_request([6], max_tokens=10)     -> 0
        step()
        step()
        step()
        step()
        step()
        step()
        step()
        seq 0  RUNNING   [0, 1]  [7, 7, 7, 7, 7, 7, 7]
        free -> 1
        add_request([5, 3], max_tokens=1)   -> 1
        step()
        seq 0  WAITING   []  [7, 7, 7, 7, 7, 7, 7, 7]   # 被 B 抢占；它的两个满块进入缓存态
        seq 1  FINISHED  []  [8]
        cached -> [0, 1]
        step()
        seq 0  RUNNING   [0, 1, 2]  [7, 7, 7, 7, 7, 7, 7, 7, 7]    # 两个满块都命中（第二个块里全是生成的 token）
        refs -> [1, 1, 1]
        step()
        seq 0  FINISHED  []  [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        cached -> [0, 1]
    """)


@case("Level 5", title="抢占自己时，预留照常完成并归还")
def l5_self_preempt(sol):
    run(sol, """
        Engine(num_blocks=2, token_budget=5, max_seqs=2)
        add_request([7], max_tokens=4)          -> 0
        add_request([5, 3, 8], max_tokens=2)    -> 1
        step()
        seq 0  RUNNING   [0]  [8]
        seq 1  RUNNING   [1]  [3]           # [5, 3, 8, 3]：块 1 被填满并注册
        step()
        seq 0  RUNNING   [0]  [8, 3]
        seq 1  WAITING   []   [3]           # B 要块，自己是准入最晚的：抢占自己，这次解码没有 token
        free -> 1
        refs -> [1, 0]
        cached -> []                        # 预留照常完成：驱逐了缓存态的块 1，拿到后随即归还
        step()
        seq 0  WAITING   []   [8, 3, 6]
        seq 1  FINISHED  []   [3, 6]
        step()
        seq 0  FINISHED  []   [8, 3, 6, 11]
        free -> 2
    """)


@case("Level 5", cpu=2, title="随机工作负载（对照参考实现）")
def l5_random(sol):
    random_check(sol, 5, workloads=300)
