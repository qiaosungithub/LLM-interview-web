import math

from judgelib import Mismatch, calling, case, expect, expect_valid, info, rng
import ref

P = ref.P
EMPTY7 = [[] for _ in range(7)]


# ---------------------------------------------------------------- helpers

def call_text(fname: str, shards: list, desc: str | None = None) -> str:
    text = f"Cluster().run({fname}, {shards!r})"
    if desc is None and len(text) <= 600:
        return text
    sizes = [len(s) for s in shards]
    return (f"Cluster().run({fname}, shards)  # shards：{desc or '随机整数'}；各分片长度 {sizes}，"
            f"共 {sum(sizes)} 个元素")


def run(sol, fname: str, shards: list, desc: str | None = None):
    """Runs the answer's fname on a fresh cluster; returns (results, cluster, call text)."""
    call = calling(call_text(fname, shards, desc))
    results, cluster = ref.run(getattr(sol, fname), shards)
    return results, cluster, call


def flat(shards: list) -> list:
    return [v for s in shards for v in s]


def split_random(r, values: list) -> list:
    """Scatters values over the ten shards at random."""
    shards = [[] for _ in range(P)]
    for v in values:
        shards[r.randrange(P)].append(v)
    return shards


def sizes_around(r, n: int, spread: float = 0.3) -> list:
    weights = [r.uniform(1 - spread, 1 + spread) for _ in range(P)]
    sizes = [int(n * w / sum(weights)) for w in weights]
    sizes[0] += n - sum(sizes)
    return sizes


def random_shards(r, kind: str, n: int) -> list:
    """Mixed shapes for the correctness sweeps."""
    if kind == "small_range":
        lo = r.randint(-5, 5)
        return split_random(r, [r.randint(lo, lo + r.choice([1, 3, 6])) for _ in range(n)])
    if kind == "wide":
        return split_random(r, [r.randint(-10**6, 10**6) for _ in range(n)])
    if kind == "huge":
        return split_random(r, [r.randint(-10**15, 10**15) for _ in range(n)])
    if kind == "dup_heavy":
        return split_random(r, [7 if r.random() < 0.8 else r.randint(0, 12) for _ in range(n)])
    if kind == "one_shard":
        shards = [[] for _ in range(P)]
        shards[r.randrange(P)] = [r.randint(0, 20) for _ in range(n)]
        return shards
    if kind == "skewed_sizes":
        big = r.randrange(P)
        shards = [[] for _ in range(P)]
        for _ in range(n):
            shards[big if r.random() < 0.8 else r.randrange(P)].append(r.randint(0, 50))
        return shards
    raise ValueError(kind)


KINDS = ["small_range", "wide", "huge", "dup_heavy", "one_shard", "skewed_sizes"]


def sweep(r, count: int, max_n: int):
    for i in range(count):
        kind = KINDS[i % len(KINDS)]
        n = r.randint(1, max_n)
        shards = random_shards(r, kind, n)
        if not flat(shards):
            shards[r.randrange(P)].append(r.randint(-3, 3))
        yield kind, shards


# ---------------------------------------------------------------- Part 1

def check_mode(sol, shards, desc=None):
    values = flat(shards)
    want = ref.mode_of(values)
    results, cluster, call = run(sol, "global_mode", shards, desc)
    got, msg = results[0], ""
    if got is None:
        msg = "rank 0 返回了 None"
    elif got != want and values.count(got) == values.count(want):
        msg = f"{got!r} 和 {want!r} 都出现了 {values.count(want)} 次，并列最多时应当返回较小的值"
    expect(got, want, call, msg=msg)
    return cluster


def mode_budget(shards) -> tuple:
    h = ref.histogram_bytes(shards)
    return h, 4 * h / P + 2000


def check_mode_traffic(sol, shards, desc, why=""):
    h, budget = mode_budget(shards)
    cluster = check_mode(sol, shards, desc)
    worst = max(range(P), key=lambda r: cluster.received[r])
    info(f"每个 rank 接收的字节数：{cluster.received}；上限 {budget:.0f}（H = {h}）")
    if cluster.received[worst] > budget:
        raise Mismatch(
            f"rank {worst} 接收了 {cluster.received[worst]} 字节，超过 O(D/P + P) 的上限 {budget:.0f}。"
            f"把十个分片的本地直方图（(value, count) 列表）各 pickle 一次共 H = {h} 字节，"
            f"这里允许每个 rank 接收不超过 4·H/P + 2000 字节。{why}",
            want=f"每个 rank ≤ {budget:.0f}", got=cluster.received)


@case("Part 1", sample=True, title="题目例子：并列取较小值")
def p1_statement_example(sol):
    shards = [[4, 1, 4], [2, 4, 2], [1, 1]] + EMPTY7
    assert ref.mode_of(flat(shards)) == 1          # the statement's answer
    check_mode(sol, shards)


@case("Part 1", title="并列最高时取最小的值")
def p1_ties(sol):
    fixed = [
        [[5, 3], [3, 5]] + [[] for _ in range(8)],
        [[-2, 7, 7, -2, 100, 100]] + [[] for _ in range(9)],
        [[] for _ in range(9)] + [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]],
        [[40], [30], [20], [10], [0], [-10], [-20], [-30], [-40], [-50]],
        [[2, 2, 1], [1, 3], [3, 3, 2, 1]] + EMPTY7,
    ]
    for shards in fixed:
        check_mode(sol, shards)
    r = rng("ties")
    for _ in range(30):
        top = r.randint(2, 5)
        k = r.randint(top, top + 6)
        values = r.sample(range(-60, 60), k)
        best = r.randint(2, 6)
        counts = [best] * top + [r.randint(1, best - 1) for _ in range(k - top)]
        bag = [v for v, c in zip(values, counts) for _ in range(c)]
        r.shuffle(bag)
        check_mode(sol, split_random(r, bag))


@case("Part 1", title="空分片、单个元素、跨分片累加")
def p1_edges(sol):
    check_mode(sol, [[] for _ in range(9)] + [[42]])
    check_mode(sol, [[-7]] + [[] for _ in range(9)])
    check_mode(sol, [[5] * 6] + [[5] * 3 for _ in range(9)])
    check_mode(sol, [[-3, -3, -1000000000000], [-1000000000000]] + [[] for _ in range(8)])
    # 7 appears once in every shard (10 in total); 3 is the local mode of shard 0 with 9 copies
    check_mode(sol, [[3] * 9 + [7]] + [[7, r] for r in range(1, 10)])
    # every shard's local mode differs from the global one
    check_mode(sol, [[r, r, 50] for r in range(10)])


@case("Part 1", title="随机数据（多种形状）")
def p1_random(sol):
    r = rng("p1")
    for kind, shards in sweep(r, 48, 120):
        check_mode(sol, shards, None if len(repr(shards)) <= 560 else f"随机整数（{kind}）")


@case("Part 1", title="接收量 O(D/P + P)：取值分散")
def p1_traffic_wide(sol):
    r = rng("wide")
    shards = [[r.randrange(10**6) for _ in range(900)] for _ in range(P)]
    check_mode_traffic(sol, shards, "0..999999 的随机整数，每个分片 900 个")


@case("Part 1", title="接收量：取值全是 10 的倍数")
def p1_traffic_multiples(sol):
    r = rng("mul10")
    shards = [[10 * r.randrange(3000) for _ in range(900)] for _ in range(P)]
    check_mode_traffic(sol, shards, "全是 10 的倍数（0..29990），每个分片 900 个",
                       why="取值全是 10 的倍数时，value % 10 或内置 hash()（对小整数就是它本身）会把所有取值"
                           "分给同一个 rank；需要一个不受取值规律影响的哈希。")


@case("Part 1", title="接收量：取值集中，另有两个极端值")
def p1_traffic_outliers(sol):
    r = rng("outliers")
    shards = [[r.randrange(10000) for _ in range(900)] for _ in range(P)]
    a, b = r.sample(range(P), 2)
    shards[a].append(-10**12)
    shards[b].append(10**12)
    check_mode_traffic(sol, shards, f"0..9999 的随机整数，每个分片 900 个；另在 rank {a} 加一个 -10¹²、rank {b} 加一个 10¹²",
                       why="把 [最小值, 最大值] 等分成 10 段来分配所有者时，两个极端值会让其余所有取值落进同一段；"
                           "所有者要由不受取值分布影响的哈希决定。")


@case("Part 1", title="接收量：分片内大量重复")
def p1_traffic_duplicates(sol):
    r = rng("dups")
    shards = []
    for _ in range(P):
        values = r.sample(range(100000, 200000), 200)
        shard = [v for v in values for _ in range(40)]
        r.shuffle(shard)
        shards.append(shard)
    check_mode_traffic(sol, shards, "每个分片 200 个不同取值、各重复 40 次",
                       why="D 只数（rank, 不同取值）组合；按元素而不是按本地计数发送，流量会随重复次数增长。")


# ---------------------------------------------------------------- Part 2

def check_median(sol, shards, desc=None):
    values = sorted(flat(shards))
    want = ref.median_of(values)
    results, cluster, call = run(sol, "global_median", shards, desc)
    n = len(values)
    msg = "rank 0 返回了 None" if results[0] is None else \
        f"N = {n}，中位数是排序后下标 (N-1)//2 = {(n - 1) // 2} 的元素（N 为偶数时取较小的那个）"
    expect(results[0], want, call, msg=msg)
    return cluster


def check_median_traffic(sol, shards, desc):
    values = flat(shards)
    span = max(values) - min(values) + 1
    budget = 100 * P * (math.log2(span) + 1)
    cluster = check_median(sol, shards, desc)
    info(f"N = {len(values)}，R = {span}：所有 rank 共发送 {cluster.bytes_sent} 字节（{cluster.messages} 条消息），"
         f"上限 {budget:.0f}")
    if cluster.bytes_sent > budget:
        raise Mismatch(
            f"所有 rank 共发送 {cluster.bytes_sent} 字节，超过 O(P·(log₂R + 1)) 的上限 {budget:.0f}"
            f"（N = {len(values)}，R = {span}，这里允许 100·P·(log₂R + 1) 字节）。发送量不能随 N 增长。",
            want=f"总量 ≤ {budget:.0f}", got=f"总量 {cluster.bytes_sent}，各 rank 发送 {cluster.sent}")


@case("Part 2", sample=True, title="题目例子：N 为偶数取下中位数")
def p2_statement_example(sol):
    shards = [[10, 1], [7, 3], [15, 6], [8, 2]] + [[] for _ in range(6)]
    assert ref.median_of(flat(shards)) == 6        # the statement's answer
    check_median(sol, shards)


@case("Part 2", title="奇偶、重复、负数、单个元素")
def p2_edges(sol):
    cases = [
        [[] for _ in range(9)] + [[42]],
        [[3], [1]] + [[] for _ in range(8)],                     # N = 2: the smaller one
        [[5, 5, 5], [5, 5]] + [[] for _ in range(8)],
        [[1, 9], [9, 9], [1, 1]] + EMPTY7,
        [[-5, -1, -3], [-2], [-4, -6]] + EMPTY7,
        [[0, 100], [50, 50, 50, 50], [1, 99]] + EMPTY7,
        [[v] for v in (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)],
        [[v] for v in (9, 8, 7, 6, 5, 4, 3, 2, 1)] + [[]],
        [[-10**12, 10**12, 7]] + [[] for _ in range(9)],
    ]
    for shards in cases:
        check_median(sol, shards)


@case("Part 2", title="随机数据（多种形状）")
def p2_random(sol):
    r = rng("p2")
    for kind, shards in sweep(r, 48, 120):
        check_median(sol, shards, None if len(repr(shards)) <= 560 else f"随机整数（{kind}）")


@case("Part 2", title="发送总量不随 N 增长")
def p2_traffic_large_n(sol):
    r = rng("bign")
    for n in (200, 100000):
        values = [r.randrange(1000) for _ in range(n)]
        sizes = sizes_around(r, n)
        shards, pos = [], 0
        for k in sizes:
            shards.append(values[pos:pos + k])
            pos += k
        check_median_traffic(sol, shards, f"0..999 的随机整数，N = {n}")


@case("Part 2", title="发送总量：取值范围很大")
def p2_traffic_wide_range(sol):
    r = rng("widerange")
    shards = [[r.randrange(2**31) for _ in range(2000)] for _ in range(P)]
    check_median_traffic(sol, shards, "0..2³¹-1 的随机整数，每个分片 2000 个")
    shards = [[r.randint(-10**12, 10**12) for _ in range(30)] for _ in range(P)]
    check_median_traffic(sol, shards, "±10¹² 的随机整数，每个分片 30 个")


# ---------------------------------------------------------------- Part 3

def check_sort(sol, shards, desc=None):
    want_flat = sorted(flat(shards))
    results, cluster, call = run(sol, "sample_sort", shards, desc)
    n = len(want_flat)
    legal = [want_flat[i * n // P:(i + 1) * n // P] for i in range(P)]
    expect_valid(results, lambda res: ref.sort_problem(res, want_flat), call,
                 want=legal if n <= 60 else None)
    return cluster


def check_sort_traffic(sol, shards, desc, why=""):
    w = ref.data_bytes(shards)
    budget = 4 * w / P + 2000
    cluster = check_sort(sol, shards, desc)
    info(f"每个 rank 发送：{cluster.sent}；接收：{cluster.received}；上限 {budget:.0f}（W = {w}）")
    for label, counts in (("接收", cluster.received), ("发送", cluster.sent)):
        worst = max(range(P), key=lambda r: counts[r])
        if counts[worst] > budget:
            raise Mismatch(
                f"rank {worst} {label}了 {counts[worst]} 字节，超过上限 {budget:.0f}。整个数据集 pickle 成一个列表"
                f"是 W = {w} 字节，N/P 个元素约 W/P = {w / P:.0f} 字节；这里允许每个 rank 发送和接收各不超过 "
                f"4·W/P + 2000 字节。{why}",
                want=f"每个 rank ≤ {budget:.0f}", got=f"{label}：{counts}")


@case("Part 3", sample=True, title="题目例子")
def p3_statement_example(sol):
    shards = [[5, 1], [9, 2], [4]] + EMPTY7
    want_flat = [1, 2, 4, 5, 9]
    results, cluster, call = run(sol, "sample_sort", shards)
    expect_valid(results, lambda res: ref.sort_problem(res, want_flat), call,
                 want=[[1, 2], [4, 5], [9], [], [], [], [], [], [], []])


@case("Part 3", title="极小数据、重复、负数")
def p3_edges(sol):
    r = rng("tiny")
    for n in range(1, 16):
        check_sort(sol, split_random(r, [r.randint(-2, 2) for _ in range(n)]))
    check_sort(sol, [[] for _ in range(9)] + [[3, 1, 2]])
    check_sort(sol, [[7] * 5 for _ in range(P)])
    check_sort(sol, [[-10**15, 10**15, 0], [10**15, -10**15]] + [[] for _ in range(8)])


@case("Part 3", title="随机数据（多种形状）")
def p3_random(sol):
    r = rng("p3")
    for kind, shards in sweep(r, 36, 300):
        check_sort(sol, shards, None if len(repr(shards)) <= 560 else f"随机整数（{kind}）")


@case("Part 3", title="倾斜数据只查正确性")
def p3_exempt(sol):
    r = rng("exempt")
    big = [r.randrange(10**6) for _ in range(9000)]
    rest = [[r.randrange(10**6) for _ in range(110)] for _ in range(9)]
    check_sort(sol, [big] + rest, "rank 0 一个分片占 90%")
    shards = [[7 if r.random() < 0.85 else r.randrange(10**6) for _ in range(1000)] for _ in range(P)]
    check_sort(sol, shards, "85% 的元素都是 7")
    check_sort(sol, [[42] * 1000 for _ in range(P)], "全部是 42")


@case("Part 3", title="流量：均匀随机取值")
def p3_traffic_uniform(sol):
    r = rng("uniform")
    shards = [[r.randrange(10**6) for _ in range(k)] for k in sizes_around(r, 40000)]
    check_sort_traffic(sol, shards, "0..999999 的随机整数，N = 40000")


@case("Part 3", title="流量：每个分片本身已排好序")
def p3_traffic_sorted_shards(sol):
    r = rng("sorted")
    shards = [sorted(r.randrange(10**6) for _ in range(k)) for k in sizes_around(r, 40000)]
    check_sort_traffic(sol, shards, "0..999999 的随机整数，每个分片已按升序排好，N = 40000",
                       why="分片本身有序时，只取分片开头几个元素当样本，分割点会全部偏向小值。")


@case("Part 3", title="流量：各分片的取值落在互不重叠的区间")
def p3_traffic_range_shards(sol):
    r = rng("blocks")
    blocks = list(range(P))
    r.shuffle(blocks)
    shards = [[blocks[i] * 10**5 + r.randrange(10**5) for _ in range(k)]
              for i, k in enumerate(sizes_around(r, 40000))]
    check_sort_traffic(sol, shards, f"rank i 的取值都在 [b_i·10⁵, (b_i+1)·10⁵) 里，b = {blocks}，N = 40000",
                       why="各分片的取值落在不同区间时，只用某一个 rank 的数据选分割点，其余数据会挤到同一个 rank；"
                           "样本要来自所有分片。")


@case("Part 3", title="流量：取值分布不均匀（指数分布）")
def p3_traffic_exponential(sol):
    r = rng("expo")
    shards = [[int(r.expovariate(1 / 1000)) for _ in range(k)] for k in sizes_around(r, 40000)]
    check_sort_traffic(sol, shards, "均值 1000 的指数分布取整，N = 40000",
                       why="取值不是均匀分布时，把 [最小值, 最大值] 等分成 10 段会让小值那一段装下大部分数据；"
                           "分割点要按数据的分位数选。")
