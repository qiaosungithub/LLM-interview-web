"""Helpers for tests.py: the statement's DataRegistry (instrumented), the Part 1 block rule restated, random
configurations, and checkers for the parts whose answer is not unique (Part 3)."""
import json

NAME_POOL = ["arxiv", "books", "chat", "code", "codex", "forum", "math", "news", "papers", "qa", "web", "wiki"]


class Registry:
    """The statement's DataRegistry. Dataset `name` holds lengths[name] examples "name-0", "name-1", ...;
    get_iterator(name, offset) opens it directly at `offset` and wraps around forever.

    For the tests it also records every get_iterator call in `opened` and, per dataset, the lowest absolute position
    (offset + steps taken, before wrapping) of any example it handed out, in `lowest`."""

    def __init__(self, lengths: dict[str, int]) -> None:
        self.lengths = dict(lengths)
        self.opened: list[tuple[str, int]] = []
        self.lowest: dict[str, int] = {}

    def get_iterator(self, name: str, offset: int = 0):
        if name not in self.lengths:
            raise KeyError(f"registry 里没有名为 {name!r} 的数据集")
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise TypeError(f"get_iterator 的 offset 应是 int，收到 {type(offset).__name__} {offset!r}")
        self.opened.append((name, offset))
        return self._walk(name, offset)

    def _walk(self, name, pos):
        size = self.lengths[name]
        while True:
            if pos < self.lowest.get(name, pos + 1):
                self.lowest[name] = pos
            yield f"{name}-{pos % size}"
            pos += 1


def parse(item):
    """("code", 7) for "code-7", or None for anything that is not an example of this registry."""
    if not isinstance(item, str) or "-" not in item:
        return None
    name, _, idx = item.rpartition("-")
    return (name, int(idx)) if idx.isdigit() and idx.isascii() else None


# ---------------------------------------------------------------- Part 1 and 2

def exact_batches(lengths, weights, batch_size, count, offsets=None):
    """Part 1's rule: `count` batches from `offsets` (default all 0). Returns (batches, offsets after each batch)."""
    names = sorted(weights)
    total = sum(weights.values())
    assert batch_size % total == 0
    pos = dict(offsets) if offsets else {name: 0 for name in names}
    batches, after = [], []
    for _ in range(count):
        batch = []
        for name in names:
            quota = batch_size * weights[name] // total
            batch += [f"{name}-{(pos[name] + j) % lengths[name]}" for j in range(quota)]
            pos[name] += quota
        batches.append(batch)
        after.append(dict(pos))
    return batches, after


def random_config(r, divisible, max_k=6, max_w=6):
    """(lengths, weights, batch_size): weights in random insertion order; lengths sometimes shorter than a quota."""
    k = r.randint(1, max_k)
    names = r.sample(NAME_POOL, k)
    weights = {name: r.randint(1, max_w) for name in names}
    if not divisible and sum(weights.values()) == 1:
        weights[names[0]] = 2                            # W = 1 divides every batch_size
    total = sum(weights.values())
    if divisible:
        batch_size = total * r.randint(1, 4)
    else:
        batch_size = r.choice([n for n in range(1, 3 * total + 4) if n % total] or [1])
    lengths = {name: r.choice([1, 2, 3, r.randint(4, 12), r.randint(13, 60)]) for name in names}
    return lengths, weights, batch_size


# ---------------------------------------------------------------- Part 3

class Tracker:
    """Follows a run of non-divisible batches choice by choice. Each example must be the next one of its dataset
    (continuing where that dataset stopped), and, for "deterministic", every prefix n of the run must keep
    |count_k(n) - n * w_k / W| <= bound for every dataset. Counts are kept as exact integers."""

    def __init__(self, lengths, weights, bound=None):
        self.lengths, self.weights = lengths, weights
        self.names = sorted(weights)
        self.total = sum(weights.values())
        self.bound = bound
        self.counts = {name: 0 for name in self.names}
        self.n = 0

    def copy(self):
        other = Tracker(self.lengths, self.weights, self.bound)
        other.counts, other.n = dict(self.counts), self.n
        return other

    def feed(self, batch):
        """None if the batch is acceptable (and the tracker moves past it), else a Chinese sentence."""
        for slot, item in enumerate(batch):
            got = parse(item)
            if got is None or got[0] not in self.counts:
                return f"第 {slot} 个样本 {item!r} 不是从 registry 取出的样本"
            name, idx = got
            want = self.counts[name] % self.lengths[name]
            if idx != want:
                return (f"第 {slot} 个样本是 {item!r}，但 {name} 的下一个样本应是 {name}-{want}：每个数据集内部要按它自己"
                        f"的迭代器顺序取，接着上一次取到的位置")
            self.counts[name] += 1
            self.n += 1
            if self.bound is not None:
                for k in self.names:
                    gap = abs(self.counts[k] * self.total - self.n * self.weights[k])     # W * |count - ideal|
                    if gap > self.bound * self.total:
                        ideal = self.n * self.weights[k] / self.total
                        return (f"从构造算起第 {self.n} 次选择之后（这个 batch 的第 {slot} 个槽位），{k} 被选了 "
                                f"{self.counts[k]} 次，理想份额 n·w/W = {self.n}·{self.weights[k]}/{self.total} = "
                                f"{ideal:.4g}，偏差 {gap / self.total:.4g} 超过了 K + W = {self.bound}：偏差没有被一个"
                                f"与 n、batch_size 都无关的常数限制住（要对每个 n 成立，不只是 batch 的边界）")
        return None


def det_bound(weights):
    """The deviation the deterministic tests allow: K + W. The page proves 1 + K for its rule; any rule whose
    deviation is bounded by a constant of this size or smaller passes."""
    return len(weights) + sum(weights.values())


def chi_square(counts, weights):
    n, total = sum(counts.values()), sum(weights.values())
    return sum((counts[k] - n * w / total) ** 2 / (n * w / total) for k, w in weights.items())


def roundtrip(state):
    return json.loads(json.dumps(state))
