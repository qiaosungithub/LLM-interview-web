"""Helpers for tests.py: the None / [] / schedule contract and a prefix-by-prefix verifier of the rules."""
from numbers import Integral


def kind(t, m, h, k):
    """What the statement's contract asks for: "none", "empty" or "schedule"."""
    if t <= 0 or m <= 0 or h <= 0:
        return "none"
    if k == 0:
        return "empty"
    if k > t:
        return "none"
    return "schedule"


def one_schedule(t, m, h, k):
    """One correct answer for display: round r gives human u task (u + r) % t, each task rotates its models."""
    seen, out = [0] * t, []
    for r in range(k):
        for u in range(h):
            x = (u + r) % t
            out.append((x, seen[x] % m, u))
            seen[x] += 1
    return out


def normalize(got):
    """The answer as a list of 3-tuples, or a Chinese sentence saying why it is not a schedule."""
    if isinstance(got, (str, bytes, dict)) or not hasattr(got, "__iter__"):
        return f"返回值应当是分配的列表，得到的是 {type(got).__name__}"
    out = []
    for p, a in enumerate(got):
        try:
            a = tuple(a)
        except TypeError:
            return f"第 {p} 个分配（下标从 0 开始）不是 (task, model, human) 三元组：{a!r:.80}"
        if len(a) != 3 or not all(isinstance(v, Integral) and not isinstance(v, bool) for v in a):
            return f"第 {p} 个分配（下标从 0 开始）不是 3 个整数组成的 (task, model, human)：{a!r:.80}"
        out.append(tuple(int(v) for v in a))
    return out


def problem(s, t, m, h, k, task_balance):
    """None if s (a list of 3-tuples) obeys every rule; otherwise a Chinese sentence naming the first violation."""
    if len(s) != h * k:
        return (f"分配数应当恰好是 h·k = {h * k}（每个标注员恰好 k 个，这是满足覆盖的最少数量），"
                f"得到 {len(s)} 个")
    first = {}
    counts = [[0] * m for _ in range(t)]
    per_human = [0] * h
    for p, (x, i, u) in enumerate(s):
        if not (0 <= x < t and 0 <= i < m and 0 <= u < h):
            return (f"第 {p} 个分配（下标从 0 开始）{(x, i, u)} 的编号越界："
                    f"任务应在 0..{t - 1}，模型应在 0..{m - 1}，标注员应在 0..{h - 1}")
        if (x, u) in first:
            return (f"违反唯一：第 {first[(x, u)]} 个和第 {p} 个分配（下标从 0 开始）"
                    f"都是标注员 {u} 做任务 {x}")
        first[(x, u)] = p
        per_human[u] += 1
        counts[x][i] += 1
        if task_balance and max(counts[x]) - min(counts[x]) > 1:
            return (f"不是按任务均衡：长度 {p + 1} 的前缀里，任务 {x} 在各模型上的使用次数是 "
                    f"{counts[x]}，最大值和最小值相差超过 1")
    for u in range(h):
        if per_human[u] < k:
            return f"违反覆盖：标注员 {u} 只有 {per_human[u]} 个分配，少于 k = {k}"
    return None


def human_imbalance(s, m, h):
    """(prefix length, human, counts) of the first prefix that is not balanced by human, or None."""
    counts = [[0] * m for _ in range(h)]
    for p, (_x, i, u) in enumerate(s):
        counts[u][i] += 1
        if max(counts[u]) - min(counts[u]) > 1:
            return p + 1, u, list(counts[u])
    return None
