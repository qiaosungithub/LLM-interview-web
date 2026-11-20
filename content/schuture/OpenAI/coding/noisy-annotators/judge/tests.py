import numpy as np

from judgelib import Mismatch, case, expect_close, expect_true, rng
import ref

TOY = np.array([[0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1], [0, -1, 1, 1]])
TOY_SRC = ("toy = np.array([[0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1], "
           "[0, -1, 1, 1]])")
R_TOY = np.array([4 / 7, 4 / 6, 4 / 7, 2 / 7])          # annotator_reliability(toy, 2), exactly
R_TOY_SRC = "r = np.array([4 / 7, 4 / 6, 4 / 7, 2 / 7])   # annotator_reliability(toy, 2) 的精确值"

# The seeds the page's own checks use. Measured offline: on each, the reference flags exactly annotators 4 and 5
# with either standard deviation (every score at least 0.0014 from both thresholds, far above rounding), and the
# page's weights keep every row labelled.
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def npgen(salt=""):
    return np.random.default_rng(rng(salt).getrandbits(64))


def code(a):
    """An array as code the user can paste, or a description when it is long."""
    a = np.asarray(a)
    text = f"np.array({a.tolist()!r})"
    return text if len(text) <= 600 else f"<形状 {a.shape} 的数组>"


def get_pair(result, n, call):
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise Mismatch("aggregate_labels 应当返回 (labels, has_label) 两个数组", call, "(labels, has_label)", result)
    la, ha = np.asarray(result[0]), np.asarray(result[1])
    if la.shape != (n,) or ha.shape != (n,):
        raise Mismatch(f"labels 和 has_label 的形状都应当是 ({n},)，得到 {la.shape} 和 {ha.shape}", call, None, result)
    if ha.dtype != bool:
        if np.issubdtype(ha.dtype, np.integer) and set(np.unique(ha).tolist()) <= {0, 1}:
            ha = ha.astype(bool)
        else:
            raise Mismatch(f"has_label 应当是布尔数组，得到 dtype {ha.dtype}", call, None, result)
    if not np.issubdtype(la.dtype, np.number) or not np.all(np.isfinite(la.astype(float))):
        raise Mismatch("labels 应当是整数类别下标", call, None, result)
    return la, ha


def totals(row, K, w):
    t = [0.0] * K
    for j, a in enumerate(row):
        if a != -1:
            t[a] += 1.0 if w is None else float(w[j])
    return t


def check_agg(sol, ann, K, w, call, rows=None):
    """aggregate_labels(ann, K[, weights=w]) against ref.aggregate. labels are compared only where a label exists."""
    want_l, want_h = ref.aggregate(ann, K, w)
    if w is None:
        got = sol.aggregate_labels(ann.copy(), K)
    else:
        got = sol.aggregate_labels(ann.copy(), K, weights=np.array(w, dtype=float))
    la, ha = get_pair(got, len(ann), call)
    shown = ([l if h else "任意" for l, h in zip(want_l, want_h)], want_h)
    for i in range(len(ann)):
        row = ann[i].tolist()
        t = totals(row, K, w)
        if bool(ha[i]) != want_h[i]:
            if want_h[i]:
                msg = f"第 {i} 行 {row} 的总权重是 {sum(t):g} > 0，has_label 应当是 True"
            elif all(a == -1 for a in row):
                msg = f"第 {i} 行 {row} 没有人标注，has_label 应当是 False"
            else:
                msg = f"第 {i} 行 {row}：标注过它的标注者权重都是 0，has_label 应当是 False"
            raise Mismatch(msg, call, shown, got)
    for i in range(len(ann)):
        if want_h[i] and int(la[i]) != want_l[i]:
            row = ann[i].tolist()
            t = totals(row, K, w)
            msg = f"第 {i} 行 {row}：各类别的总权重是 {[round(v, 6) for v in t]}，应当预测 {want_l[i]}，得到 {int(la[i])}"
            if t.count(max(t)) > 1:
                msg += "（并列时取较小的类别下标）"
            if -1 in row and int(la[i]) == K - 1:
                msg += "；-1 表示没有标注，不能当作类别下标（负下标会落到最后一个类别上）"
            raise Mismatch(msg, call, shown, got)


def random_matrix(g, n, A, K, p_missing):
    ann = g.integers(0, K, (n, A))
    ann[g.random((n, A)) < p_missing] = -1
    return ann


def dyadic_weights(g, A):
    return g.integers(0, 9, A) / 4.0           # 0, 0.25, ..., 2: sums are exact, so ties are exact


# ============================================================================== Part 1

@case("Part 1", sample=True, title="例子：完整矩阵上的多数投票")
def p1_example_aggregate(sol):
    call = f"{TOY_SRC}\naggregate_labels(toy, 2)"
    la, ha = get_pair(sol.aggregate_labels(TOY.copy(), 2), 7, call)
    expect_true(ha.all(), "每一行都有人标注，has_label 应当全是 True", call, ha)
    if la.tolist() != [0, 1, 0, 0, 0, 1, 1]:
        raise Mismatch("labels 不对（第 2 行 2 比 2 并列，取较小的类别 0）", call, [0, 1, 0, 0, 0, 1, 1], la)


@case("Part 1", sample=True, title="例子：可靠度约 [0.57, 0.67, 0.57, 0.29]")
def p1_example_reliability(sol):
    call = f"{TOY_SRC}\nannotator_reliability(toy, 2)"
    expect_close(sol.annotator_reliability(TOY.copy(), 2), np.array([0.57, 0.67, 0.57, 0.29]), rtol=0, atol=0.005,
                 call=call, msg="可靠度应当约为 [0.57, 0.67, 0.57, 0.29]（题目保留两位小数）")


@case("Part 1", sample=True, title="例子：只标记标注者 3")
def p1_example_flag(sol):
    call = f"{R_TOY_SRC}\nflag_bad_annotators(r)"
    got = np.asarray(sol.flag_bad_annotators(R_TOY.copy()))
    if got.shape != (4,) or got.tolist() != [False, False, False, True]:
        raise Mismatch("只有标注者 3 的分数比均值低出一个标准差以上", call, np.array([False, False, False, True]), got)


@case("Part 1", title="-1 是缺失，不是类别")
def p1_aggregate_missing(sol):
    ann = np.array([[-1, -1, 0], [2, -1, -1], [-1, -1, -1], [0, 1, -1], [1, -1, -1], [-1, 0, -1]])
    check_agg(sol, ann, 3, None, f"aggregate_labels({code(ann)}, 3)")
    ann = np.array([[0, -1, -1, -1], [-1, -1, 0, -1], [0, 1, -1, -1], [-1, -1, -1, -1]])
    check_agg(sol, ann, 2, None, f"aggregate_labels({code(ann)}, 2)")


@case("Part 1", title="并列时取较小的类别")
def p1_aggregate_ties(sol):
    ann = np.array([[1, 0, -1, -1], [2, 1, 0, -1], [2, 2, 1, 1], [1, 2, -1, -1], [0, 0, 2, 2], [2, 1, 2, 1]])
    check_agg(sol, ann, 3, None, f"aggregate_labels({code(ann)}, 3)")
    w = np.array([0.5, 0.25, 0.25, 1.0])
    ann = np.array([[0, 1, 1, -1], [2, -1, -1, 1], [-1, 2, 1, -1], [3, 2, 1, 0], [1, 3, 3, -1]])
    check_agg(sol, ann, 4, w, f"aggregate_labels({code(ann)}, 4, weights={code(w)})")


@case("Part 1", title="权重：0 权重、重票胜过多票")
def p1_aggregate_weights(sol):
    w = np.array([0.0, 1.0, 2.0, 0.0])
    ann = np.array([[0, -1, -1, 1], [0, 1, -1, -1], [1, 1, 0, -1], [-1, -1, -1, 2], [2, 0, -1, 2], [1, -1, 2, 1]])
    check_agg(sol, ann, 3, w, f"aggregate_labels({code(ann)}, 3, weights={code(w)})")
    z = np.zeros(4)
    check_agg(sol, ann, 3, z, f"aggregate_labels({code(ann)}, 3, weights={code(z)})")
    one = np.ones(4)
    check_agg(sol, ann, 3, one, f"aggregate_labels({code(ann)}, 3, weights={code(one)})")


@case("Part 1", title="随机小矩阵（含权重）")
def p1_aggregate_random(sol):
    g = npgen()
    for t in range(60):
        n, A, K = int(g.integers(1, 26)), int(g.integers(1, 8)), int(g.integers(2, 6))
        ann = random_matrix(g, n, A, K, float(g.choice([0.0, 0.3, 0.6])))
        if t % 2:
            w = dyadic_weights(g, A)
            check_agg(sol, ann, K, w, f"aggregate_labels({code(ann)}, {K}, weights={code(w)})")
        else:
            check_agg(sol, ann, K, None, f"aggregate_labels({code(ann)}, {K})")


def check_reliability(sol, ann, K, call):
    want = ref.reliability(ann, K)
    got = sol.annotator_reliability(ann.copy(), K)
    expect_close(got, np.array(want), rtol=0, atol=1e-12, call=call)


def own_vote_reliability(ann, K):
    """The page's pitfall: j's own vote is inside its reference majority."""
    out = []
    for j in range(ann.shape[1]):
        agree = total = 0
        for row in ann.tolist():
            if row[j] == -1 or sum(a != -1 for a in row) < 2:
                continue
            counts = [row.count(c) for c in range(K)]
            total += 1
            agree += row[j] == counts.index(max(counts))
        out.append(agree / total)
    return np.array(out)


@case("Part 1", title="参照多数不含自己那一票")
def p1_reliability_own_vote(sol):
    # Every row has exactly two annotators, so the reference "majority" is simply the other annotator's label.
    ann = np.array([[0, 1, -1], [1, -1, 0], [-1, 0, 1], [0, 0, -1], [1, -1, 1], [-1, 1, 0], [1, 0, -1], [0, -1, 1]])
    call = f"annotator_reliability({code(ann)}, 2)"
    want = np.array(ref.reliability(ann, 2))
    got = sol.annotator_reliability(ann.copy(), 2)
    if np.shape(got) == want.shape and np.allclose(got, own_vote_reliability(ann, 2), rtol=0, atol=1e-12):
        raise Mismatch("把标注者 j 自己的票也算进了它的参照多数：应当只和其余标注者的聚合结果比较", call, want, got)
    expect_close(got, want, rtol=0, atol=1e-12, call=call)


@case("Part 1", title="没有其他人标注的样本不计入")
def p1_reliability_skip(sol):
    ann = np.array([[0, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 0, 0], [-1, -1, -1], [-1, 1, -1], [0, 1, 1],
                    [-1, -1, 1], [2, -1, 2]])
    check_reliability(sol, ann, 3, f"annotator_reliability({code(ann)}, 3)")


@case("Part 1", title="随机小矩阵上的可靠度")
def p1_reliability_random(sol):
    g = npgen()
    done = 0
    while done < 40:
        n, A, K = int(g.integers(3, 30)), int(g.integers(2, 7)), int(g.integers(2, 5))
        ann = random_matrix(g, n, A, K, float(g.choice([0.0, 0.3, 0.5])))
        if None in ref.reliability(ann, K):         # an annotator with no sample that counts: score not specified
            continue
        check_reliability(sol, ann, K, f"annotator_reliability({code(ann)}, {K})")
        done += 1


def check_flags(sol, r, call):
    pop, smp, gap = ref.flag_readings(r)
    # All-equal scores have standard deviation exactly 0 and nobody is "more than" 0 below the mean.
    assert pop == smp and (gap > 1e-9 or len(set(r)) == 1), "judge: flag case depends on which standard deviation"
    got = np.asarray(sol.flag_bad_annotators(np.array(r, dtype=float)))
    if got.shape != (len(r),) or [bool(v) for v in got] != pop:
        r = np.asarray(r, dtype=float)
        msg = (f"均值 {r.mean():.4g}，标准差 {r.std():.4g}，阈值约 {r.mean() - r.std():.4g}：分数低于“均值 − 一个标准差”的"
               "标注者才被标记")
        raise Mismatch(msg, call, np.array(pop), got)


@case("Part 1", title="标记规则：均值减一个标准差")
def p1_flag_cases(sol):
    for r in ([0.6, 0.62, 0.64, 0.66, 0.68, 0.7],
              [0.5, 0.5, 0.5, 0.5, 0.5, 0.95],
              [0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
              [0.95, 0.9, 0.92, 0.7, 0.93, 0.94],
              [0.3, 0.32, 0.31, 0.05, 0.33, 0.29],
              [0.9, 0.1, 0.8]):
        check_flags(sol, r, f"flag_bad_annotators(np.array({r}))")


@case("Part 1", title="随机分数向量的标记")
def p1_flag_random(sol):
    g = npgen()
    done = 0
    while done < 60:
        A = int(g.integers(3, 11))
        r = g.random(A) if done % 2 else np.clip(0.8 + 0.05 * g.standard_normal(A) - 0.4 * (g.random(A) < 0.25), 0, 1)
        r = [round(float(v), 4) for v in r]
        pop, smp, gap = ref.flag_readings(r)
        if pop != smp or gap <= 1e-6:              # depends on population vs sample standard deviation
            continue
        check_flags(sol, r, f"flag_bad_annotators(np.array({r}))")
        done += 1


@case("Part 1", title="题目数据：可靠度和标记")
def p1_statement_data(sol):
    seed = rng().choice(SEEDS)
    ann = ref.make_dataset(seed)[4]
    call = f"annotations = make_dataset({seed})[4]\nannotator_reliability(annotations, 3)"
    want = np.array(ref.reliability(ann, 3))
    expect_close(sol.annotator_reliability(ann.copy(), 3), want, rtol=0, atol=1e-12, call=call)
    call = f"flag_bad_annotators(np.array({want.tolist()}))   # make_dataset({seed}) 上的可靠度"
    check_flags(sol, want.tolist(), call)


# ============================================================================== Part 2

@case("Part 2", sample=True, title="例子：丢掉标注者 3 后重新聚合")
def p2_example(sol):
    call = f"{TOY_SRC}\naggregate_labels(toy[:, :3], 2)   # 丢弃标注者 3 那一列"
    la, ha = get_pair(sol.aggregate_labels(TOY[:, :3].copy(), 2), 7, call)
    expect_true(ha.all(), "每一行都还有人标注，has_label 应当全是 True", call, ha)
    if la.tolist() != [0, 1, 0, 0, 0, 1, 0]:
        raise Mismatch("最后一个样本只剩 0 和 1 各一票，并列取 0", call, [0, 1, 0, 0, 0, 1, 0], la)


def check_fit(sol, X_train, labels, has, X_test, y_test, call, why=""):
    want = ref.fit_and_score(X_train, labels, has, X_test, y_test)
    got = sol.fit_and_score(X_train.copy(), labels.copy(), has.copy(), X_test.copy(), y_test.copy())
    # lbfgs stops at a tolerance, so the same fit reached with rows or arrays laid out differently can end at a
    # slightly different optimum and flip a test point or two near the boundary: allow 3 of the 1000 points.
    expect_close(got, want, rtol=0, atol=0.0031, call=call,
                 msg=f"测试准确率应当约为 {want:.3f}（允许相差 3 个测试点），得到 {float(np.asarray(got)):.3f}{why}")


@case("Part 2", title="过滤后的标签：训练并打分")
def p2_fit_filtered(sol):
    seed = rng().choice(SEEDS)
    X_train, _, X_test, y_test, ann = ref.make_dataset(seed)
    lab, has = ref.aggregate(ann[:, :4], 3)
    labels = np.array([l if h else 0 for l, h in zip(lab, has)])
    has = np.array(has)
    call = (f"X_train, _, X_test, y_test, annotations = make_dataset({seed})\n"
            "labels, has_label = aggregate_labels(annotations[:, :4], 3)   # 丢弃标注者 4、5\n"
            "fit_and_score(X_train, labels, has_label, X_test, y_test)")
    check_fit(sol, X_train, labels, has, X_test, y_test, call)


@case("Part 2", title="has_label 为 False 的行不参与训练")
def p2_fit_ignores_unlabeled(sol):
    g = npgen()
    seed = int(g.choice(SEEDS))
    X_train, y_true, X_test, y_test, _ = ref.make_dataset(seed)
    mask_seed = int(g.integers(1000))
    has = np.random.default_rng(mask_seed).random(len(y_true)) < 0.4
    labels = np.where(has, y_true, (y_true + 1) % 3)
    call = (f"X_train, y_true, X_test, y_test, _ = make_dataset({seed})\n"
            f"has_label = np.random.default_rng({mask_seed}).random({len(y_true)}) < 0.4   # 约 40% 为 True\n"
            "labels = np.where(has_label, y_true, (y_true + 1) % 3)   # 没有标签的行放了错误的类别\n"
            "fit_and_score(X_train, labels, has_label, X_test, y_test)")
    check_fit(sol, X_train, labels, has, X_test, y_test, call,
              why="；只能用 has_label 为 True 的行训练，其余行的 labels 是任意值")


# ============================================================================== Part 3

def get_weights(sol, r, K, call):
    got = sol.reliability_weights(np.array(r, dtype=float), K)
    w = np.asarray(got)
    if w.shape != (len(r),):
        raise Mismatch(f"权重的形状应当是 ({len(r)},)，得到 {w.shape}", call, None, got)
    try:
        w = w.astype(float)
    except (TypeError, ValueError):
        raise Mismatch("权重应当是浮点数组", call, None, got) from None
    # +inf at r = 1 is the log-odds limit and still >= 0, so only nan and negative weights are rejected.
    bad = np.isnan(w) | (w < 0)
    if bad.any():
        i = int(np.argmax(bad))
        raise Mismatch(f"r = {float(r[i]):.6g} 时权重是 {w[i]}：每个权重都必须是非负数", call,
                       ref.log_odds_weights(r, K), got)
    return w


@case("Part 3", sample=True, title="例子：权重约 [0.29, 0.69, 0.29, 0.0]")
def p3_example(sol):
    call = f"{R_TOY_SRC}\nreliability_weights(r, 2)"
    w = get_weights(sol, R_TOY, 2, call)
    expect_close(w, np.array([0.29, 0.69, 0.29, 0.0]), rtol=0, atol=0.005, call=call,
                 msg="权重应当约为 [0.29, 0.69, 0.29, 0.0]（题目保留两位小数）")
    call = f"{TOY_SRC}\naggregate_labels(toy, 2, weights=np.array([0.29, 0.69, 0.29, 0.0]))"
    la, ha = get_pair(sol.aggregate_labels(TOY.copy(), 2, weights=np.array([0.29, 0.69, 0.29, 0.0])), 7, call)
    expect_true(ha.all(), "加权投票不丢弃任何一行：has_label 应当全是 True", call, ha)
    expect_true(int(la[-1]) == 0, f"最后一个样本的预测应当仍然是 0，得到 {int(la[-1])}", call, la)


@case("Part 3", title="权重非负（含 r = 0 和 1）")
def p3_weights_valid(sol):
    g = npgen()
    for K in (2, 3, 5):
        r = [0.0, 1.0, 0.5, 1 / K, 0.999, 0.001] + [round(float(v), 4) for v in g.random(6)]
        get_weights(sol, r, K, f"reliability_weights(np.array({r}), {K})")


@case("Part 3", title="越可靠的标注者权重不更低")
def p3_weights_monotone(sol):
    g = npgen()
    for K in (2, 3, 4):
        r = sorted(set([i / 40 for i in range(41)] + [round(float(v), 4) for v in g.random(20)]))
        call = f"reliability_weights(np.array({r}), {K})"
        w = get_weights(sol, r, K, call)
        for i in range(len(r) - 1):
            # Same precision as the example (0.005): a floor such as np.where(w > 0, w, 1e-3) keeps below-chance
            # annotators a hair above the log-odds just past chance (about 1e-16 there) and is still a fair answer.
            if w[i + 1] < w[i] - 0.005:
                raise Mismatch(f"r 从 {r[i]} 增加到 {r[i + 1]} 时，权重从 {w[i]:.6g} 降到了 {w[i + 1]:.6g}："
                               "更可靠的标注者不应得到更低的权重", call, None, w)


@case("Part 3", title="题目数据：只降权，不丢样本")
def p3_statement_data(sol):
    seed = rng().choice(SEEDS)
    ann = ref.make_dataset(seed)[4]
    r = ref.reliability(ann, 3)
    call = (f"annotations = make_dataset({seed})[4]\n"
            f"w = reliability_weights(np.array({[round(v, 4) for v in r]}), 3)   # annotator_reliability 的结果\n"
            "aggregate_labels(annotations, 3, weights=w)")
    w = get_weights(sol, r, 3, call)
    if w[4:].max() >= w[:4].min():
        raise Mismatch(f"标注者 4、5 的可靠度最低（{r[4]:.3f}、{r[5]:.3f}），权重却不低于其余标注者：权重应当随可靠度"
                       "降低", call, ref.log_odds_weights(r, 3), w)
    _, has = ref.aggregate(ann, 3, w)
    lost = len(has) - sum(has)
    if lost:
        raise Mismatch(f"用这些权重聚合后有 {lost} 个样本没有标签（它们只被权重为 0 的标注者标过）：加权投票应当只降低"
                       "不可靠标注者的权重，而不丢弃他们标注过的样本", call, ref.log_odds_weights(r, 3), w)
