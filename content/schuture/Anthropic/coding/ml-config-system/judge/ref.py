"""Helpers for the config-system tests: a judge-side stand-in for DELETE, a strict comparison, a printer that writes
configs as pasteable Python, the statement's rules restated, schemas and random generators.
In 自定义运行 the schemas can be imported, e.g. `from ref import ExperimentConfig`."""
import dataclasses
import math
import re
import typing
from collections import Counter
from types import MappingProxyType


# ---------------------------------------------------------------- DELETE, printing, strict comparison

class _Delete:
    """Stands for the answer's DELETE in the judge's data; inject() swaps in the answer's own sentinel."""

    def __repr__(self):
        return "DELETE"


D = _Delete()


def inject(value, delete):
    """A fresh copy of `value` with every D replaced by `delete` (the answer's DELETE)."""
    if value is D:
        return delete
    if isinstance(value, dict):
        return {k: inject(v, delete) for k, v in value.items()}
    if isinstance(value, list):
        return [inject(v, delete) for v in value]
    return value


def clone(value):
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone(v) for v in value]
    return value


def src(value):
    """Python source for a config value: D prints as DELETE, nan and inf as float("...")."""
    if value is D:
        return "DELETE"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{src(k)}: {src(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(src(v) for v in value) + "]"
    if isinstance(value, float) and not math.isfinite(value):
        return f'float("{value}")'
    return repr(value)


def store_src(store, name="store"):
    return "\n".join([f"{name} = {{"] + [f"    {k!r}: {src(v)}," for k, v in store.items()] + ["}"])


def kind(x):
    for name, t in (("bool", bool), ("int", int), ("float", float), ("str", str), ("dict", dict), ("list", list)):
        if isinstance(x, t):
            return name
    return "None" if x is None else type(x).__name__


def _short(x, limit=80):
    try:
        text = src(x) if kind(x) in ("bool", "int", "float", "str", "dict", "list", "None") else repr(x)
    except Exception:
        text = f"<{type(x).__name__}>"
    return text if len(text) <= limit else text[:limit] + "…"


def _join(path, key):
    return f"{path}.{key}" if path else str(key)


def diff(got, want, delete=None, path="", root="返回值"):
    """None when got equals want strictly (types and the sign of 0.0 included, dict order ignored), else a Chinese
    sentence naming the first difference. A D in `want` must be the answer's `delete` in `got`."""
    where = path or root
    if want is D:
        return None if got is delete else f"{where} 应当是 DELETE，得到 {_short(got)}"
    kw, kg = kind(want), kind(got)
    if kw != kg:
        return f"{where} 应当是 {_short(want)}（{kw}），得到 {_short(got)}（{kg}）"
    if kw == "dict":
        for k in want:
            if k not in got:
                return f"缺少键 {_join(path, k)}（应当是 {_short(want[k])}）"
        for k in got:
            if k not in want:
                if delete is not None and got[k] is delete:
                    return f"多出了键 {_join(path, k)}，值为 DELETE：DELETE 应当让这个键从结果里消失"
                return f"多出了键 {_join(path, k)}（值为 {_short(got[k])}）"
        for k in want:
            m = diff(got[k], want[k], delete, _join(path, k), root)
            if m:
                return m
        return None
    if kw == "list":
        if len(got) != len(want):
            return f"{where} 应当是有 {len(want)} 个元素的列表 {_short(want)}，得到 {len(got)} 个元素 {_short(got)}"
        for i, (g, w) in enumerate(zip(got, want)):
            m = diff(g, w, delete, _join(path, i), root)
            if m:
                return m
        return None
    if kw == "float":
        if got == want and math.copysign(1.0, got) == math.copysign(1.0, want):
            return None
        return f"{where} 应当是 {want!r}，得到 {got!r}"
    return None if got == want else f"{where} 应当是 {_short(want)}，得到 {_short(got)}"


def containers(value, out=None):
    """id -> object for every dict and list inside value (value included)."""
    out = {} if out is None else out
    if isinstance(value, (dict, list)):
        out[id(value)] = value
        for v in (value.values() if isinstance(value, dict) else value):
            containers(v, out)
    return out


def shared_with(result, sources):
    """(path, source label) of the first dict or list in result that is the very object found in a source."""
    owner = {}
    for label, value in sources.items():
        for i in containers(value):
            owner.setdefault(i, label)

    def walk(x, path):
        if isinstance(x, (dict, list)):
            if id(x) in owner:
                return path or "返回值本身", owner[id(x)]
            for k, v in (x.items() if isinstance(x, dict) else enumerate(x)):
                hit = walk(v, _join(path, k))
                if hit:
                    return hit
        return None
    return walk(result, "")


TOKEN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def cycle_problem(message, cycle):
    """None when `message` lists the cycle's names in order, as in "a -> b -> a" (any starting point, either
    direction, any separators; repeating the first name at the end is optional, the statement only shows it in an
    example), else a Chinese sentence."""
    tokens = TOKEN.findall(message)
    n = len(cycle)
    for seq in (list(cycle), list(reversed(cycle))):
        for i in range(n):
            rot = seq[i:] + seq[:i]
            for j in range(len(tokens) - n + 1):
                if tokens[j:j + n] == rot:
                    return None
    example = " -> ".join(list(cycle) + [cycle[0]])
    return f"错误信息里应当按顺序列出环经过的名字，例如 {example!r}；实际的信息是 {message!r}"


# ---------------------------------------------------------------- Part 1, restated

class NotFound(Exception):
    pass


class Cycle(Exception):
    pass


def merge(base, overlay):
    """deep_merge restated from the statement, on judge data (D is DELETE)."""
    out = {}
    for k in list(base) + [k for k in overlay if k not in base]:
        if k not in overlay:
            out[k] = clone(base[k])
        elif overlay[k] is D:
            continue
        elif isinstance(overlay[k], dict):
            b = base.get(k)
            out[k] = merge(b if isinstance(b, dict) else {}, overlay[k])
        else:
            out[k] = clone(overlay[k])
    return out


def resolve(store, name, path=()):
    """resolve_bases restated, without a memo; raises NotFound or Cycle."""
    if name not in store:
        raise NotFound(name)
    if name in path:
        raise Cycle(list(path[path.index(name):]))
    merged = {}
    for b in store[name].get("_base_", []):
        merged = merge(merged, resolve(store, b, path + (name,)))
    return merge(merged, {k: v for k, v in store[name].items() if k != "_base_"})


def reachable(store, name, seen=None):
    seen = set() if seen is None else seen
    if name in seen or name not in store:
        return seen
    seen.add(name)
    for b in store[name].get("_base_", []):
        reachable(store, b, seen)
    return seen


LEAVES = ["trainer.epochs", "trainer.batch_size", "trainer.lr", "trainer", "optimizer.name", "optimizer.momentum",
          "optimizer.betas", "data.path", "data.aug.flip", "data.aug", "tags", "seed"]


def rand_scalar(r):
    return r.choice([r.randint(-5, 100), round(r.uniform(0, 1), 3), r.choice(["sgd", "adam", "s3://b/x"]),
                     r.choice([True, False]), None])


def rand_value(r):
    if r.random() < 0.15:
        return [rand_scalar(r) for _ in range(r.randint(0, 3))]
    return rand_scalar(r)


def set_path(d, dotted, value):
    node, parts = d, dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def rand_own(r, deletable=True):
    own = {}
    for _ in range(r.randint(0, 4)):
        set_path(own, r.choice(LEAVES), D if deletable and r.random() < 0.25 else rand_value(r))
    return own


def rand_store(r, n, prefix="c"):
    """An acyclic store c0..c{n-1}; each config's bases come from the earlier names (sometimes one listed twice)."""
    names = [f"{prefix}{i}" for i in range(n)]
    store = {}
    for i, name in enumerate(names):
        k = min(i, r.choice([0, 0, 1, 1, 2, 3]))
        own = rand_own(r)
        if k and r.random() < 0.1:
            store[name] = {"_base_": r.choices(names[:i], k=k + 1), **own}
        elif k or r.random() < 0.1:
            store[name] = {"_base_": r.sample(names[:i], k), **own}
        else:
            store[name] = own
    return store, names


def rand_ring_store(r):
    """A store whose only cycle is a ring r0 -> r1 -> ... -> r0, reached from the returned start name."""
    store, clean = rand_store(r, r.randint(0, 4))
    k = r.randint(1, 4)
    ring = [f"r{i}" for i in range(k)]
    for i, nm in enumerate(ring):
        bases = r.sample(clean, min(len(clean), r.randint(0, 1))) + [ring[(i + 1) % k]]
        r.shuffle(bases)
        store[nm] = {"_base_": bases, **rand_own(r)}
    chain = [f"p{i}" for i in range(r.randint(0, 2))] + [r.choice(ring)]
    for i, nm in enumerate(chain[:-1]):
        store[nm] = {"_base_": [chain[i + 1]], **rand_own(r)}
    names = list(store)
    r.shuffle(names)
    return {nm: store[nm] for nm in names}, chain[0], ring


def rand_missing_store(r):
    """An acyclic store where one config lists a name the store lacks; returns (store, a name that reaches it)."""
    store, names = rand_store(r, r.randint(1, 6))
    if r.random() < 0.15:
        return store, "absent"
    victim = r.choice(names)
    bases = list(store[victim].get("_base_", []))
    bases.insert(r.randint(0, len(bases)), "ghost")
    store[victim] = {"_base_": bases, **{k: v for k, v in store[victim].items() if k != "_base_"}}
    return store, r.choice([n for n in names if victim in reachable(store, n)])


# ---------------------------------------------------------------- Part 2, overrides restated

class OverrideKey(Exception):
    pass


def apply_ops(config, ops):
    """apply_overrides restated on parsed overrides (create, path segments, value); raises OverrideKey."""
    out = clone(config)
    for create, path, value in ops:
        node = out
        for seg in path[:-1]:
            if seg not in node:
                if not create:
                    raise OverrideKey(".".join(path))
                node[seg] = {}
            elif not isinstance(node[seg], dict):
                raise OverrideKey(".".join(path))
            node = node[seg]
        if path[-1] not in node and not create:
            raise OverrideKey(".".join(path))
        node[path[-1]] = clone(value)
    return out


WORDS = ["trainer", "lr", "epochs", "optimizer", "name", "data", "path", "aug", "flip", "seed", "tags", "x", "_y", "k2"]


def rand_config(r, depth=0, scalar=rand_scalar):
    cfg = {}
    for w in r.sample(WORDS, r.randint(1 if depth == 0 else 0, 4)):
        x = r.random()
        if depth < 2 and x < 0.35:
            cfg[w] = rand_config(r, depth + 1, scalar)
        elif x < 0.5:
            cfg[w] = [scalar(r) for _ in range(r.randint(0, 3))]
        else:
            cfg[w] = scalar(r)
    return cfg


def key_paths(cfg, prefix=()):
    for k, v in cfg.items():
        yield prefix + (k,), v
        if isinstance(v, dict):
            yield from key_paths(v, prefix + (k,))


def get_path(cfg, path):
    for seg in path:
        cfg = cfg[seg]
    return cfg


FLOAT_TEXTS = ["1.5", "-0.25", "2.0", "0.001", "1e3", "-2e-2", "2.50", "1.0e2", "-0.0", "10.125", "7e0"]
STR_CHARS = "abcXYZ019-_./:,=+$"


def rand_override_value(r, in_list=False):
    """(text, value): an override value as typed on a command line, and what it means."""
    x = r.random()
    if not in_list and x < 0.15:
        items = [rand_override_value(r, True) for _ in range(r.randint(0, 3))]
        ws = lambda: r.choice(["", "", " ", "  "])   # noqa: E731
        text = "[" + ws()
        for i, (t, _) in enumerate(items):
            text += (ws() + "," + ws() if i else "") + t
        return text + ws() + "]", [v for _, v in items]
    x = r.random()
    if x < 0.12:
        return r.choice([("null", None), ("true", True), ("false", False)])
    if x < 0.42:
        n = r.randint(-300, 300)
        if r.random() < 0.2:
            return ("-" if n < 0 else "") + "0" * r.randint(1, 2) + str(abs(n)), n
        return str(n), n
    if x < 0.65:
        t = r.choice(FLOAT_TEXTS)
        return t, float(t)
    s = "".join(r.choice(STR_CHARS) for _ in range(r.randint(0, 6)))
    q = r.choice("'\"")
    return q + s + q, s


def rand_ops(r, cfg):
    """Random overrides for cfg: (texts, ops). Mostly valid; may end with one that must raise OverrideKeyError."""
    cur, texts, ops = clone(cfg), [], []
    for _ in range(r.randint(1, 5)):
        ps = list(key_paths(cur))
        x = r.random()
        if ps and x < 0.45:                                        # an existing key
            path, create = r.choice(ps)[0], r.random() < 0.3
        elif ps and x < 0.55:                                      # below a scalar or list: fails, '+' or not
            cands = [p for p, v in ps if not isinstance(v, dict)]
            if not cands:
                continue
            path = r.choice(cands) + tuple(r.choice(WORDS) for _ in range(r.randint(1, 2)))
            create = r.random() < 0.5
        else:                                                      # a new key, maybe several levels deep
            base = r.choice([()] + [p for p, v in ps if isinstance(v, dict)])
            new = [w for w in WORDS if w not in get_path(cur, base)]
            if not new:
                continue
            path = base + (r.choice(new),) + tuple(r.choice(WORDS) for _ in range(r.choice([0, 0, 1, 2])))
            create = r.random() < 0.8
        text, value = rand_override_value(r)
        ops.append((create, list(path), value))
        texts.append(("+" if create else "") + ".".join(path) + "=" + text)
        try:
            cur = apply_ops(cur, ops[-1:])
        except OverrideKey:
            break
    return texts, ops


# ---------------------------------------------------------------- Part 2, interpolation generators

EMBEDDABLE = (bool, int, float, str)


def text_of(v):
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def _lit_scalar(r):
    return r.choice([r.randint(-20, 300), r.choice([0.5, 0.001, 2.0, -1.25, 1e-07]), r.choice([True, False]),
                     r.choice(["x", "adam", "a.b", "costs $5", ""]), None])


def _literal(r):
    x = r.random()
    if x < 0.15:
        v = [_lit_scalar(r) for _ in range(r.randint(0, 3))]
    elif x < 0.3:
        v = {k: _lit_scalar(r) for k in r.sample(["a", "b", "c"], r.randint(0, 3))}
    else:
        v = _lit_scalar(r)
    return v, clone(v), True


def _targets(nodes):
    out = []
    for path, val, raw, _ in nodes:
        out.append((path, val))
        if isinstance(raw, dict):
            out += [(f"{path}.{k}", val[k]) for k in raw]
    return out


def _embedded(r, emb):
    k = r.randint(1, 3)
    parts = [r.choice(["", "run-", "-", "/", "bs", "_", "lr="]) for _ in range(k + 1)]
    if k == 1 and not (parts[0] or parts[1]):
        parts[0] = "v"
    raw = val = parts[0]
    for i in range(k):
        p, v = r.choice(emb)
        raw += "${" + p + "}" + parts[i + 1]
        val += text_of(v) + parts[i + 1]
    return raw, val


def _node(r, nodes):
    targets = _targets(nodes)
    emb = [(p, v) for p, v in targets if isinstance(v, EMBEDDABLE)]
    x = r.random()
    if not targets or x < 0.3:
        return _literal(r)
    if x < 0.55:
        p, v = r.choice(targets)
        return "${" + p + "}", clone(v), False
    if x < 0.8 and emb:
        return (*_embedded(r, emb), False)
    items = []
    for _ in range(r.randint(1, 3)):
        y = r.random()
        if y < 0.4:
            p, v = r.choice(targets)
            items.append(("${" + p + "}", clone(v)))
        elif y < 0.7 and emb:
            items.append(_embedded(r, emb))
        else:
            s = _lit_scalar(r)
            items.append((s, s))
    if r.random() < 0.5:
        return [a for a, _ in items], [b for _, b in items], False
    keys = ["a", "b", "c"]
    return ({k: a for k, (a, _) in zip(keys, items)}, {k: b for k, (_, b) in zip(keys, items)}, False)


def _bad(r, nodes):
    """A raw value whose resolution must raise InterpolationError: a missing path or an embedded non-scalar."""
    opts = ["${nope}", "x-${nope.k}", "${g.zz}"]
    for path, val, raw, literal in nodes:
        if literal and not isinstance(raw, dict):
            opts.append("${" + path + ".k}")
        if isinstance(raw, dict):
            opts.append("${" + path + ".zz}")
    opts += [f"v-${{{p}}}" for p, v in _targets(nodes) if not isinstance(v, EMBEDDABLE)]
    raw = r.choice(opts)
    return ["ok", raw] if r.random() < 0.25 else raw


def rand_interp(r, bad=False):
    """(config, resolved config) with the resolution predicted while it is built; with bad=True one value cannot be
    resolved and the second item is None."""
    cfg, want, nodes = {}, {}, []
    n = r.randint(1, 7)
    bad_at = r.randrange(n) if bad else -1
    for i in range(n):
        group = r.choice(["", "", "g", "h.k"])
        path = (group + "." if group else "") + f"n{i}"
        if i == bad_at:
            set_path(cfg, path, _bad(r, nodes))
            continue
        raw, val, literal = _node(r, nodes)
        set_path(cfg, path, raw)
        set_path(want, path, val)
        nodes.append((path, val, raw, literal))
    return cfg, (None if bad else want)


def rand_interp_ring(r):
    """(config, ring paths): the ring m0 -> m1 -> ... -> m0 is the config's only cycle."""
    cfg, _ = rand_interp(r)
    k = r.randint(1, 4)
    ring = []
    for i in range(k):
        group = r.choice(["", "", "q", "q.w"])
        ring.append((group + "." if group else "") + f"m{i}")
    for i, p in enumerate(ring):
        nxt = ring[(i + 1) % k]
        set_path(cfg, p, "${" + nxt + "}" if r.random() < 0.5 else f"a-${{{nxt}}}")
    if r.random() < 0.5:
        set_path(cfg, "pre", "${" + r.choice(ring) + "}")
    return cfg, ring


# ---------------------------------------------------------------- Part 3: schemas and a validator restated

@dataclasses.dataclass
class OptimizerConfig:
    name: str
    momentum: float = 0.0


@dataclasses.dataclass
class TrainerConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float = 0.0
    warmup_steps: int = 0


@dataclasses.dataclass
class ExperimentConfig:
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    run_name: str
    eval_batch_size: int
    tags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Prims:
    i: int
    f: float
    b: bool
    s: str


@dataclasses.dataclass
class Lists:
    ints: list[int] = dataclasses.field(default_factory=list)
    floats: list[float] = dataclasses.field(default_factory=list)
    flags: list[bool] = dataclasses.field(default_factory=list)
    names: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class LayerConfig:
    hidden_size: int
    activation: str = "relu"


@dataclasses.dataclass
class ModelConfig:
    layers: list[LayerConfig]
    dropout: float = 0.0


@dataclasses.dataclass
class AugConfig:
    flip: bool = False
    crop: int = 0
    scales: list[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DataConfig:
    path: str
    aug: AugConfig
    shards: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RunConfig:
    data: DataConfig
    model: ModelConfig
    seeds: list[int]
    lr: float
    name: str = "run"
    stages: list[DataConfig] = dataclasses.field(default_factory=list)


MISSING_MSG = "required field is missing"
UNEXPECTED_MSG = "unexpected field, not declared in the schema"
KIND_ZH = {"missing": "缺少必需字段", "unexpected": "未声明的键", "type": "类型不符"}


def _is_schema(tp):
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _required_field(f):
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def validate(config, schema, path=""):
    """validate_config restated: a list of (path, kind), kind being missing / unexpected / type."""
    if not isinstance(config, dict):
        return [(path or "<root>", "type")]
    hints = typing.get_type_hints(schema)
    declared = {f.name: f for f in dataclasses.fields(schema)}
    out = []
    for name, f in declared.items():
        full = _join(path, name)
        if name not in config:
            if _required_field(f):
                out.append((full, "missing"))
            continue
        out += _check(config[name], hints[name], full)
    out += [(_join(path, k), "unexpected") for k in config if k not in declared]
    return out


def _check(value, tp, path):
    if _is_schema(tp):
        return validate(value, tp, path)
    if typing.get_origin(tp) is list:
        if not isinstance(value, list):
            return [(path, "type")]
        return [e for i, item in enumerate(value) for e in _check(item, typing.get_args(tp)[0], f"{path}.{i}")]
    ok = {int: isinstance(value, int) and not isinstance(value, bool),
          float: isinstance(value, (int, float)) and not isinstance(value, bool),
          bool: isinstance(value, bool), str: isinstance(value, str)}[tp]
    return [] if ok else [(path, "type")]


def classify(message):
    return {MISSING_MSG: "missing", UNEXPECTED_MSG: "unexpected"}.get(message, "type")


class ShowErrors:
    """How the expected errors are shown: the two fixed messages in full, type mismatches as a placeholder."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __repr__(self):
        items = [f"FieldError({p!r}, {MISSING_MSG!r})" if k == "missing" else
                 f"FieldError({p!r}, {UNEXPECTED_MSG!r})" if k == "unexpected" else
                 f"FieldError({p!r}, <说明类型不符的任意文字>)" for p, k in self.pairs]
        return "[" + ", ".join(items) + "]"


def errors_problem(got, want):
    """None when the FieldErrors in got match the expected (path, kind) pairs as a multiset, else a Chinese sentence."""
    try:
        items = list(got)
    except TypeError:
        return f"validate_config 应当返回 FieldError 的列表，得到 {type(got).__name__}"
    pairs = []
    for e in items:
        path, message = getattr(e, "path", None), getattr(e, "message", None)
        if not isinstance(path, str) or not isinstance(message, str):
            return f"列表里的每一项都应当是 FieldError(path: str, message: str)，得到 {e!r}"
        pairs.append((path, classify(message)))
    lack, extra = Counter(want) - Counter(pairs), Counter(pairs) - Counter(want)
    if not lack and not extra:
        return None
    desc = lambda c: "、".join(f"{p}（{KIND_ZH[k]}）" for p, k in sorted(c.elements()))   # noqa: E731
    parts = ([f"缺少这些错误：{desc(lack)}"] if lack else []) + ([f"多出了这些错误：{desc(extra)}"] if extra else [])
    return "；".join(parts) + "。（按 path 和错误种类比较，不比较顺序，也不比较类型不符时的 message 文字）"


def sample_value(r, tp):
    if _is_schema(tp):
        hints = typing.get_type_hints(tp)
        return {f.name: sample_value(r, hints[f.name]) for f in dataclasses.fields(tp)
                if _required_field(f) or r.random() < 0.7}
    if typing.get_origin(tp) is list:
        return [sample_value(r, typing.get_args(tp)[0]) for _ in range(r.randint(0, 3))]
    return {int: lambda: r.randint(0, 50), float: lambda: r.choice([round(r.uniform(0, 10), 2), r.randint(0, 5)]),
            bool: lambda: r.choice([True, False]), str: lambda: r.choice(["relu", "run-a", ""])}[tp]()


WRONG_TYPE_POOL = [3, 2.5, "text", True, False, None, [1, 2], {"x": 1}]


def corrupt(r, config, schema):
    """A copy of a valid config with keys removed, keys added and values of the wrong type, at any depth."""
    config = clone(config)
    hints = typing.get_type_hints(schema)
    if config and r.random() < 0.3:
        del config[r.choice(list(config))]
    if r.random() < 0.3:
        config[f"extra_{r.randint(0, 999)}"] = 1
    for _ in range(r.randint(0, 2)):
        if not config:
            break
        key = r.choice(list(config))
        tp = hints.get(key)
        if _is_schema(tp) and isinstance(config[key], dict) and r.random() < 0.7:
            config[key] = corrupt(r, config[key], tp)
        elif typing.get_origin(tp) is list and isinstance(config[key], list) and config[key] and r.random() < 0.8:
            i, elem = r.randrange(len(config[key])), typing.get_args(tp)[0]
            item = config[key][i]
            config[key][i] = (corrupt(r, item, elem) if _is_schema(elem) and isinstance(item, dict)
                              and r.random() < 0.7 else r.choice(WRONG_TYPE_POOL))
        elif tp is not None:
            config[key] = r.choice(WRONG_TYPE_POOL)
    return config


# ---------------------------------------------------------------- Part 4

def hash_scalar(r):
    return r.choice([r.randint(-3, 3), r.choice([0.0, -0.0, 2.0, 0.1, 1e-07, 1e22, -2.5]), 10 ** 20,
                     r.choice([True, False]), None, r.choice(["", "a", "é", "数据", "2", "None", "true"])])


def shuffled(r, value):
    if isinstance(value, dict):
        items = [(k, shuffled(r, v)) for k, v in value.items()]
        r.shuffle(items)
        return dict(items)
    if isinstance(value, list):
        return [shuffled(r, v) for v in value]
    return value


def _changed(r, v):
    if isinstance(v, bool):
        return r.choice([not v, int(v)])
    if isinstance(v, int):
        return r.choice([v + 1, float(v), str(v)])
    if isinstance(v, float):
        opts = [v + 1.0 if abs(v) < 1e15 else v * 2, str(v),
                -v if v != 0 else (0.0 if math.copysign(1.0, v) < 0 else -0.0)]
        if v.is_integer() and abs(v) < 1e15:
            opts.append(int(v))
        return r.choice(opts)
    if isinstance(v, str):
        return r.choice([v + "x", None])
    if v is None:
        return r.choice([0, "None", False])
    if isinstance(v, list):
        return v + [1] if not v or r.random() < 0.5 else v[:-1]
    return {**v, "extra": 0}


def mutated(r, cfg):
    """A copy of cfg with one value (a leaf, a list element or an empty container) changed."""
    out = clone(cfg)
    slots = []

    def walk(node):
        for k, v in (node.items() if isinstance(node, dict) else enumerate(node)):
            if isinstance(v, (dict, list)) and v and r.random() < 0.8:
                walk(v)
            else:
                slots.append((node, k))
    walk(out)
    node, k = r.choice(slots)
    node[k] = _changed(r, node[k])
    return out


def canon(v):
    """A key that tells configs apart exactly as config_hash must: by type, value and float sign, not key order."""
    if isinstance(v, dict):
        return ("dict", tuple(sorted((k, canon(x)) for k, x in v.items())))
    if isinstance(v, list):
        return ("list", tuple(canon(x) for x in v))
    return (kind(v), repr(v))


def frozen_problem(got, want, path=""):
    """None when got is want frozen as the statement says (dict -> MappingProxyType, list -> tuple, scalars as they
    are), else a Chinese sentence."""
    where = path or "返回值"
    if isinstance(want, dict):
        if not isinstance(got, MappingProxyType):
            return f"{where} 应当是 types.MappingProxyType，得到 {type(got).__name__}"
        if set(got.keys()) != set(want):
            return f"{where} 的键应当是 {sorted(want)}，得到 {sorted(got.keys())}"
        for k in want:
            m = frozen_problem(got[k], want[k], _join(path, k))
            if m:
                return m
        return None
    if isinstance(want, list):
        if not isinstance(got, tuple):
            return f"{where} 应当是 tuple，得到 {type(got).__name__}"
        if len(got) != len(want):
            return f"{where} 应当有 {len(want)} 个元素，得到 {len(got)} 个"
        for i, (g, w) in enumerate(zip(got, want)):
            m = frozen_problem(g, w, _join(path, i))
            if m:
                return m
        return None
    return diff(got, want, path=path, root="返回值")


def index_src(path):
    """frozen["a"][0] style access for a tuple of keys and indexes."""
    return "".join(f"[{k!r}]" for k in path)
