import numpy as np

from judgelib import Mismatch, case, expect_true, info, rng
import ref

NAMES = ref.NAMES
LN3 = float(np.log(3))


def npgen(salt=""):
    return np.random.default_rng(rng(salt).getrandbits(64))


def cp(params):
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in params.items()}


def close_or_fail(got, want, label, call, rtol, atol, hint=""):
    """got must have want's shape, be finite, and match within atol + rtol * |want|; the message names the entry."""
    try:
        a = np.asarray(got, dtype=np.float64)
    except Exception:
        raise Mismatch(f"{label} 不是数值数组：{type(got).__name__}", call, None, got) from None
    want = np.asarray(want, dtype=np.float64)
    shown = float(want) if want.ndim == 0 else want
    if a.shape != want.shape:
        raise Mismatch(f"{label} 的形状应当是 {want.shape}，得到 {a.shape}", call, shown, got)
    if not np.all(np.isfinite(a)):
        idx = tuple(int(i) for i in np.argwhere(~np.isfinite(a))[0])
        at = f"位置 {idx} " if a.ndim else ""
        raise Mismatch(f"{label} 里有 nan 或 inf（{at}是 {a[idx] if a.ndim else float(a)}）{hint}", call, shown, got)
    err = np.abs(a - want)
    bad = err > atol + rtol * np.abs(want)
    if bad.any():
        idx = tuple(int(i) for i in np.unravel_index(np.argmax(np.where(bad, err, -1.0)), err.shape)) if a.ndim else ()
        at = f"位置 {idx} " if a.ndim else ""
        raise Mismatch(f"{label} 不对：{at}期望 {float(want[idx]):.10g}，得到 {float(a[idx]):.10g}"
                       f"（允许误差 atol={atol:g}, rtol={rtol:g}）{hint}", call, shown, got)


def get_probs(out, n, call):
    if not isinstance(out, dict) or "probs" not in out:
        raise Mismatch('forward 应当返回带 "probs" 键的 dict', call, None, out)
    p = out["probs"]
    if np.shape(p) != (n, 3):
        raise Mismatch(f"probs 的形状应当是 ({n}, 3)，得到 {np.shape(p)}", call, None, p)
    return np.asarray(p, dtype=np.float64)


def unpack(result, call):
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise Mismatch("loss_and_grads 应当返回 (loss, grads) 两个值", call, "(loss, grads)", result)
    loss, grads = result
    if np.ndim(loss) != 0:
        raise Mismatch(f"loss 应当是一个标量，得到形状 {np.shape(loss)}", call, None, loss)
    if not isinstance(grads, dict) or any(k not in grads for k in NAMES):
        raise Mismatch("grads 应当是带 W1、b1、W2、b2 四个键的 dict", call, None, grads)
    return float(loss), grads


# A random network whose probabilities are far from uniform: the statement's init with W2 scaled up.
def scaled_params(seed, factor):
    p = ref.init_params(seed)
    p["W2"] = p["W2"] * factor
    return p


def scaled_setup(seed, factor):
    return f'params = init_params({seed}); params["W2"] = params["W2"] * {factor}'


# ============================================================================== Part 1

@case("Part 1", sample=True, title="题目例子：train(seed=0) ≥ 0.9")
def p1_example(sol):
    call = "train(seed=0)"
    out = sol.train(seed=0)
    if not isinstance(out, dict) or "train_losses" not in out or "val_accuracy" not in out:
        raise Mismatch('train 应当返回带 "train_losses" 和 "val_accuracy" 的 dict', call, None, out)
    losses = list(out["train_losses"])
    expect_true(len(losses) == ref.EPOCHS, f"train_losses 应当有 {ref.EPOCHS} 个数（每个 epoch 一个），得到 {len(losses)} 个",
                call, losses)
    expect_true(all(np.isfinite(float(v)) for v in losses), "train_losses 里有 nan 或 inf", call, losses)
    acc = float(out["val_accuracy"])
    expect_true(acc >= 0.9, f"验证集准确率 {acc:.3f}：六个 bug 都修好后应当至少 0.9", call, acc)


def check_eval_probs(got, want, call, hint=""):
    """Eval-time probabilities against the reference, naming the likely bug."""
    sums = got.sum(axis=1)
    if np.all(np.isfinite(sums)) and np.abs(sums - 1).max() > 1e-9:
        i = int(np.argmax(np.abs(sums - 1)))
        raise Mismatch(f"第 {i} 行概率之和是 {sums[i]:.6g}，不是 1：softmax 应当对每个样本的 3 个类别归一化"
                       "（max 和 sum 沿 axis=1）", call, want, got)
    close_or_fail(got, want, "probs", call, rtol=1e-9, atol=1e-12, hint=hint)


@case("Part 1", title="评估时不做 dropout")
def p1_eval_no_dropout(sol):
    g = npgen()
    ps, ds = int(g.integers(1000)), int(g.integers(1000))
    params = scaled_params(ps, 100)
    X = ref.make_dataset(ds)[2][:50]
    setup = f"{scaled_setup(ps, 100)}\nX = make_dataset({ds})[2][:50]\n"
    want = ref.probs(params, X)
    outs = []
    for k in (1, 2):
        call = setup + f"forward(params, X, training=False, rng=np.random.default_rng({k}))"
        outs.append(get_probs(sol.forward(cp(params), X.copy(), training=False, rng=np.random.default_rng(k)), 50, call))
    if not np.array_equal(outs[0], outs[1]):
        raise Mismatch("同样的参数和输入，两次 training=False 的结果不同：评估时不应随机丢弃隐藏单元",
                       setup + "forward(params, X, training=False, rng=np.random.default_rng(1))  # 与 rng=…(2) 的结果比较",
                       outs[0], outs[1])
    check_eval_probs(outs[0], want, call, hint="；评估时 h_drop 应当就是 h：既不丢弃，也不缩放")


@case("Part 1", title="softmax 按行（类别）归一化")
def p1_softmax_rows(sol):
    g = npgen()
    for n in (7, 1, 40):
        ps, ds = int(g.integers(1000)), int(g.integers(1000))
        start = int(g.integers(0, 600 - n))
        params = scaled_params(ps, 100)
        X = ref.make_dataset(ds)[0][start:start + n]
        call = (f"{scaled_setup(ps, 100)}\nX = make_dataset({ds})[0][{start}:{start + n}]\n"
                f"forward(params, X, training=False, rng=np.random.default_rng(0))")
        want = ref.probs(params, X)
        got = get_probs(sol.forward(cp(params), X.copy(), training=False, rng=np.random.default_rng(0)), n, call)
        check_eval_probs(got, want, call)


@case("Part 1", title="logit 上千时 softmax 不溢出")
def p1_softmax_large(sol):
    # W1 = eye(2, 32), W2 = eye(32, 3): the logits are (relu(x0), relu(x1), 0), so a few rows reach 1000.
    params = dict(W1=np.eye(2, 32), b1=np.zeros(32), W2=np.eye(32, 3), b2=np.zeros(3))
    X = np.array([[800.0, 801.0], [1000.0, 999.5], [3.0, 2.0], [0.0, 0.0], [-5.0, 710.0]])
    call = ("params = dict(W1=np.eye(2, 32), b1=np.zeros(32), W2=np.eye(32, 3), b2=np.zeros(3))\n"
            "X = np.array([[800.0, 801.0], [1000.0, 999.5], [3.0, 2.0], [0.0, 0.0], [-5.0, 710.0]])\n"
            "forward(params, X, training=False, rng=np.random.default_rng(0))")
    want = ref.probs(params, X)
    got = get_probs(sol.forward(cp(params), X.copy(), training=False, rng=np.random.default_rng(0)), len(X), call)
    close_or_fail(got, want, "probs", call, rtol=1e-9, atol=1e-12,
                  hint="；exp 之前要先减去每行 logits 的最大值，否则 e^800 会溢出")


@case("Part 1", title="训练时 dropout：p=0.2、放大 1/(1-p)")
def p1_train_dropout(sol):
    # W1 = 0, b1 = 1 make h = 1 for every unit; W2 routes unit j into logit 0 and unit j + 1 into logit 1 (logit 2
    # stays 0), so log(p0 / p2) and log(p1 / p2) are exactly the two units' values after dropout.
    gen = npgen()
    n = 500
    X = np.zeros((n, 2))
    vals = []
    for j in range(0, 32, 2):
        W2 = np.zeros((32, 3))
        W2[j, 0] = W2[j + 1, 1] = 1.0
        params = dict(W1=np.zeros((2, 32)), b1=np.ones(32), W2=W2, b2=np.zeros(3))
        call = (f"W2 = np.zeros((32, 3)); W2[{j}, 0] = W2[{j + 1}, 1] = 1.0\n"
                "params = dict(W1=np.zeros((2, 32)), b1=np.ones(32), W2=W2, b2=np.zeros(3))  # h 全是 1\n"
                f"probs = forward(params, np.zeros(({n}, 2)), training=True, rng=np.random.default_rng(0))[\"probs\"]\n"
                f"np.log(probs[:, :2] / probs[:, 2:])   # 第 {j}、{j + 1} 个隐藏单元 dropout 之后的值")
        p = get_probs(sol.forward(cp(params), X.copy(), training=True, rng=gen), n, call)
        with np.errstate(all="ignore"):
            hd = np.log(p[:, :2] / p[:, 2:])
        ok = np.isclose(hd, 0.0, atol=1e-6) | np.isclose(hd, 1.25, atol=1e-6)
        if not ok.all():
            r, c = (int(i) for i in np.argwhere(~ok)[0])
            raise Mismatch(f"h = 1 时，dropout 之后的隐藏单元只能是 0 或 1/(1-p) = 1.25，第 {r} 行第 {j + c} 个单元是 "
                           f"{hd[r, c]:.6g}", call, None, hd)
        zero = np.isclose(hd, 0.0, atol=1e-6)
        for c in (0, 1):
            f = zero[:, c].mean()
            if not 0.1 <= f <= 0.3:
                raise Mismatch(f"第 {j + c} 个隐藏单元在 {n} 行里被置零的比例是 {f:.3f}，应当约为 p = 0.2"
                               "（训练时每个单元、每个样本独立地以概率 0.2 丢弃）", call, None, hd)
        vals.append(zero)
    zero = np.concatenate(vals)
    both = float((zero[:, 0] & zero[:, 1]).mean())
    # independent drops: P(both) = 0.04, standard error 0.0022 over 8000 rows
    expect_true(abs(both - 0.04) <= 0.015,
                f"同一行的两个隐藏单元同时被置零的比例是 {both:.3f}，独立丢弃时应当约为 0.2² = 0.04", call)


@case("Part 1", title="损失和梯度（含 1/N）")
def p1_loss_grads(sol):
    g = npgen()
    for n in (5, 1, 17, 32):
        for _ in range(100):
            ps, ds = int(g.integers(1000)), int(g.integers(1000))
            idx = sorted(int(i) for i in g.choice(600, n, replace=False))
            labels = [int(c) for c in g.integers(0, 3, n)]
            params = scaled_params(ps, 30)
            X = ref.make_dataset(ds)[0][idx]
            y = np.array(labels)
            # keep every label probability >= 1e-3, so the page's log(p + 1e-12) is exact to 1e-9
            if ref.probs(params, X)[np.arange(n), y].min() >= 1e-3:
                break
        call = (f"{scaled_setup(ps, 30)}\nX = make_dataset({ds})[0][{idx}]\ny = np.array({labels})\n"
                "loss, grads = loss_and_grads(params, X, y, training=False, rng=np.random.default_rng(0))")
        want_loss, want = ref.loss_and_grads(params, X, y)
        loss, grads = unpack(sol.loss_and_grads(cp(params), X.copy(), y.copy(), training=False,
                                                rng=np.random.default_rng(0)), call)
        close_or_fail(loss, want_loss, "loss", call, rtol=1e-7, atol=1e-9)
        if n > 1 and all(np.shape(grads[k]) == want[k].shape and np.allclose(grads[k], n * want[k], rtol=1e-6, atol=1e-12)
                         for k in NAMES):
            raise Mismatch(f"每个梯度都正好是正确值的 N = {n} 倍：损失是 batch 上的平均，梯度也要除以 N", call,
                           want, grads)
        for k in NAMES:
            close_or_fail(grads[k], want[k], f"grads['{k}']", call, rtol=1e-6, atol=1e-10)


@case("Part 1", title="accuracy_score 手算例子")
def p1_accuracy(sol):
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [0.3, 0.4, 0.3], [0.5, 0.25, 0.25]])
    y = np.array([0, 1, 1, 1, 2])
    call = ("accuracy_score(np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [0.3, 0.4, 0.3], "
            "[0.5, 0.25, 0.25]]), np.array([0, 1, 1, 1, 2]))   # 预测 0,2,1,1,0，对 3 个")
    got = sol.accuracy_score(P.copy(), y.copy())
    hint = "；0.36 是 N×N 两两比较的平均：argmax 的结果形状应当是 (N,)，不能带 keepdims" if np.isclose(
        np.asarray(got, dtype=float), 0.36) else ""
    close_or_fail(got, 0.6, "准确率", call, rtol=0, atol=1e-12, hint=hint)
    y2 = np.array([2, 0, 1, 1])
    call = "accuracy_score(np.eye(3)[[2, 0, 1, 1]], np.array([2, 0, 1, 1]))"
    close_or_fail(sol.accuracy_score(np.eye(3)[y2], y2.copy()), 1.0, "准确率", call, rtol=0, atol=1e-12)


def spy_train(sol, seed):
    """Runs sol.train(seed) with make_dataset and loss_and_grads wrapped; returns the training split and, for every
    training-mode loss_and_grads call, copies of (params, X batch, y batch, grads)."""
    real_md, real_lg = sol.make_dataset, sol.loss_and_grads
    data, calls = [], []

    def make_dataset(s):
        out = real_md(s)
        if not data:
            data.append((np.array(out[0], dtype=np.float64, copy=True), np.array(out[1], copy=True)))
        return out

    def loss_and_grads(params, X, y, training, rng):
        result = real_lg(params, X, y, training=training, rng=rng)
        if training:
            calls.append((cp(params), np.array(X, dtype=np.float64, copy=True), np.array(y, copy=True),
                          cp(result[1])))
        return result

    sol.make_dataset, sol.loss_and_grads = make_dataset, loss_and_grads
    try:
        sol.train(seed=seed)
    finally:
        sol.make_dataset, sol.loss_and_grads = real_md, real_lg
    call = f"train(seed={seed})   # 判题程序记录每次 loss_and_grads(..., training=True, ...) 的参数和小批量"
    expect_true(bool(data), "train 没有调用 make_dataset(seed)，无法检查训练过程", call)
    expect_true(bool(calls), "train 没有调用 loss_and_grads(..., training=True, ...)，无法检查训练过程", call)
    return data[0], calls, call


@case("Part 1", title="洗牌后 X 与 y 仍一一对应")
def p1_shuffle(sol):
    seed = int(rng().randrange(1000))
    (X_train, y_train), calls, call = spy_train(sol, seed)
    N = len(X_train)
    index = {row.tobytes(): i for i, row in enumerate(X_train)}
    per_epoch = -(-N // ref.BATCH_SIZE)
    expect_true(len(calls) == ref.EPOCHS * per_epoch,
                f"应当有 {ref.EPOCHS} 个 epoch × {per_epoch} 个小批量 = {ref.EPOCHS * per_epoch} 次梯度计算，"
                f"实际 {len(calls)} 次", call)
    prev = None
    for e in range(ref.EPOCHS):
        order = []
        for b in range(per_epoch):
            _, Xb, yb, _ = calls[e * per_epoch + b]
            size = min(ref.BATCH_SIZE, N - b * ref.BATCH_SIZE)
            expect_true(len(Xb) == size and len(yb) == size,
                        f"第 {e + 1} 个 epoch 的第 {b + 1} 个小批量应当有 {size} 个样本，得到 {len(Xb)} 个", call)
            for i in range(size):
                k = index.get(Xb[i].tobytes())
                expect_true(k is not None, f"第 {e + 1} 个 epoch 的第 {b + 1} 个小批量里第 {i} 行不是训练集里的样本", call)
                if int(yb[i]) != int(y_train[k]):
                    raise Mismatch(f"第 {e + 1} 个 epoch 的第 {b + 1} 个小批量：Xb 第 {i} 行是训练样本 {k}，它的标签是 "
                                   f"{int(y_train[k])}，yb 里却是 {int(yb[i])}：X 和 y 要用同一个排列洗牌", call,
                                   show=False)
                order.append(k)
        expect_true(sorted(order) == list(range(N)),
                    f"第 {e + 1} 个 epoch 没有把 {N} 个训练样本各用一次", call)
        expect_true(order != list(range(N)), f"第 {e + 1} 个 epoch 按原始顺序遍历了训练集：每个 epoch 开始时要重新洗牌",
                    call)
        expect_true(order != prev, f"第 {e + 1} 个 epoch 的顺序和上一个 epoch 完全一样：每个 epoch 开始时要重新洗牌", call)
        prev = order


@case("Part 1", title="动量：速度只在训练前清零一次")
def p1_momentum(sol):
    seed = int(rng().randrange(1000))
    _, calls, call = spy_train(sol, seed)
    lr, mu = ref.LR, ref.MOMENTUM
    per_epoch = -(-ref.N_PER_CLASS_TRAIN * 3 // ref.BATCH_SIZE)
    # theta_{t+1} - theta_t = mu (theta_t - theta_{t-1}) - lr g_t, with theta_0 - theta_{-1} = 0. Rounding of the
    # parameters (|theta| < 10) puts about 1e-14 into each side; a reset velocity is off by mu |v|, 1e-5 or more.
    for t in range(len(calls) - 1):
        th, g, nxt = calls[t][0], calls[t][3], calls[t + 1][0]
        prev = calls[t - 1][0] if t else th
        for k in NAMES:
            step = nxt[k] - th[k]
            want = mu * (th[k] - prev[k]) - lr * g[k]
            err = np.abs(step - want)
            if err.max() > 1e-9:
                reset = np.abs(step + lr * g[k]).max() <= 1e-9
                e, b = divmod(t, per_epoch)
                where = f"第 {e + 1} 个 epoch 的第 {b + 1} 步" if len(calls) == ref.EPOCHS * per_epoch else f"第 {t + 1} 步"
                why = "：这一步的更新是 -η∇L，说明速度在这一步之前被清零了，v 应当只在训练开始前初始化一次" if reset and t else ""
                raise Mismatch(f"{where}参数 {k} 的更新不符合 v ← μv − η∇L、θ ← θ + v（μ = 0.9、η = 0.15）{why}；"
                               f"最大偏差 {err.max():.3g}", call, show=False)


# ============================================================================== Part 3

PIPES = {
    "fixed": "修好的流水线",
    "bug2": "只留 bug 2（softmax 沿 batch 轴归一化）的流水线",
    "bug3": "只留 bug 3（梯度漏了 1/N）的流水线",
    "bug5": "只留 bug 5（评估时也做 dropout）的流水线",
    "bug6": "只留 bug 6（accuracy_score 里 argmax 带 keepdims）的流水线",
    "bad_init": "初始化尺度过大（W2 的标准差是 10）、其余正确的流水线",
    "zero_grad": "反向传播返回全 0 梯度、其余正确的流水线",
    "no_relu_gate": "反向传播漏乘 ReLU 导数、其余正确的流水线",
    "tiny_no_relu_gate": "W2 初始化极小（标准差 1e-6）、反向传播漏乘 ReLU 导数的流水线",
    "train_drop_all": "训练模式（training=True）下把隐藏单元全部丢弃、关闭 dropout 时完全正确的流水线",
}

WHY = {
    ("bug2", "initial_loss_near_ln_C"): "沿 batch 轴归一化时，整个数据集上的初始损失约为 ln N，远离 ln 3",
    ("bug2", "overfits_tiny_batch"): "沿 batch 轴归一化时，8 个样本里至少有 3 个同类，它们分这一类的总概率 1，损失降不到 0.05",
    ("bug2", "gradients_match_finite_differences"): "这条流水线的梯度公式不是它自己损失的梯度",
    ("bug2", "eval_invariant_to_batching"): "单独评估一行时沿 batch 轴归一化会得到全 1，和整批评估不同",
    ("bug3", "gradients_match_finite_differences"): "解析梯度是中心差分的 6 倍",
    ("bug5", "eval_invariant_to_batching"): "每次调用 forward 都会随机丢弃隐藏单元，单独评估和整批评估的结果不同",
    ("bug6", "accuracy_metric_correct_on_labels"): "传入真实标签的 one-hot 时，这个 accuracy_score 返回的不是 1.0",
    ("bad_init", "initial_loss_near_ln_C"): "这样的初始化让初始损失离 ln 3 超过 1",
    ("zero_grad", "overfits_tiny_batch"): "梯度全 0 时参数不动，损失一直在 ln 3 附近",
    ("zero_grad", "gradients_match_finite_differences"): "解析梯度全是 0，中心差分不是",
    ("no_relu_gate", "gradients_match_finite_differences"): "W1、b1 的解析梯度和中心差分对不上",
    ("tiny_no_relu_gate", "gradients_match_finite_differences"):
        "W1、b1 的梯度只有约 1e-7，解析值和中心差分的绝对误差不到 1e-5，但相对误差超过 0.5；题目要求的是相对误差",
}
for _k in ("initial_loss_near_ln_C", "overfits_tiny_batch", "gradients_match_finite_differences",
           "eval_invariant_to_batching"):
    WHY[("train_drop_all", _k)] = ("前四项检查都要关闭 dropout（training=False）；这条流水线只有训练模式是坏的，"
                                   "关闭 dropout 时它和修好的流水线完全一样")

# Expected results per pipeline, in ref.KEYS order; None marks a check whose outcome the statement leaves open for
# that pipeline (bug 3's or a bad init's larger steps may still fit 8 points; bug 5's finite differences match when
# the answer reuses a fixed rng for every call), which is not checked.
T, F, U = True, False, None
EXPECT = {
    "fixed": (T, T, T, T, T),
    "bug2": (F, F, F, F, T),
    "bug3": (T, U, F, T, T),
    "bug5": (T, U, U, F, T),
    "bug6": (T, T, T, T, F),
    "bad_init": (F, U, U, T, T),
    "zero_grad": (T, F, F, T, T),
    "no_relu_gate": (T, U, F, T, T),
    "tiny_no_relu_gate": (T, U, F, T, T),
    "train_drop_all": (T, T, T, T, T),
}


# Training splits for the Part 3 cases, by make_dataset seed. "sorted" is the split as make_dataset returns it (class
# 0 first); "mixed" is the same split interleaved 0, 1, 2, 0, 1, 2, ... whose first 8 rows keep points of different
# classes at least 0.5 apart (ref.separated). Each seed was vetted offline: at least 8 of 10 statement-style initial
# draws pass ref.good_init on it, so ref.make_init finds a well-conditioned init for any seed in a draw or two. On
# some other splits most draws give a gradient entry so small that its central difference is off by more than 1e-5
# relative, and a correct check would fail there.
SPLITS = {
    "sorted": (
        6, 8, 9, 14, 15, 16, 17, 18, 19, 24, 43, 48, 51, 53, 55, 56, 58, 59, 64, 67, 69, 71, 72, 80, 84, 87, 92, 96,
        99, 101, 102, 111, 112, 116, 120, 121, 122, 125, 128, 132, 137, 138, 141, 149, 151, 155, 158, 161, 163, 164,
        168, 180, 188, 190, 191, 192, 193, 195, 196, 202, 203, 204, 210, 214, 216, 218, 219, 223, 225, 235, 237,
        242, 243, 244, 246, 249, 252, 254, 256, 257, 259),
    "mixed": (
        1, 6, 18, 20, 21, 24, 27, 33, 37, 40, 47, 48, 50, 53, 56, 59, 69, 70, 72, 74, 76, 83, 84, 85, 89, 90, 91,
        92, 99, 101, 102, 109, 120, 121, 122, 123, 126, 128, 137, 140, 141, 149, 151, 152, 158, 164, 171, 175, 186,
        190, 191, 193, 195, 199, 203, 207, 208, 214, 217, 218, 227, 234, 236, 239, 244, 246, 247, 248, 249, 250,
        252, 256),
}


def dataset(source, r):
    ds = r.choice(SPLITS[source])
    X, y = ref.make_dataset(ds)[:2]
    if source == "mixed":
        X, y = ref.interleave(X, y)
        return X, y, f"X, y = make_dataset({ds})[:2] 的训练集按类别交错重排（类别 0, 1, 2, 0, 1, 2, …）"
    return X, y, f"X, y = make_dataset({ds})[:2]   # 训练集，按类别排好（前 200 个都是类别 0）"


def run_checks(sol, kind, source, r):
    if r == "statement":
        X, y = ref.make_dataset(0)[:2]
        data_desc, seed = "X, y = make_dataset(0)[:2]   # 题目的训练集", 0
    else:
        X, y, data_desc = dataset(source, r)
        seed = r.randrange(1000)
    pieces = ref.pipeline(kind, X, y)
    call = (f"{data_desc}\nsanity_checks(init_params, forward, loss_and_grads, accuracy_score, X, y, 3, seed={seed})\n"
            f"# 传入的四个函数是判题程序的：{PIPES[kind]}")
    got = sol.sanity_checks(*pieces, X.copy(), y.copy(), 3, seed=seed)
    if not isinstance(got, dict):
        raise Mismatch(f"sanity_checks 应当返回 dict，得到 {type(got).__name__}", call, None, got)
    missing = [k for k in ref.KEYS if k not in got]
    expect_true(not missing, f"返回的 dict 缺少键 {missing}", call, got)
    for k in ref.KEYS:
        if not isinstance(got[k], (bool, np.bool_)):
            raise Mismatch(f"{k} 的值应当是 bool，得到 {type(got[k]).__name__}", call, None, got)
    want = dict(zip(ref.KEYS, EXPECT[kind]))
    shown = {k: ("不检查" if v is None else v) for k, v in want.items()}
    for k in ref.KEYS:
        if want[k] is not None and bool(got[k]) != want[k]:
            why = WHY.get((kind, k), "这条流水线在这一项上是正确的")
            raise Mismatch(f"{k} 应当是 {want[k]}，得到 {bool(got[k])}：{why}", call, shown, got)


@case("Part 3", sample=True, title="题目数据、修好的流水线：全部 True")
def p3_example(sol):
    run_checks(sol, "fixed", "sorted", "statement")


@case("Part 3", title="类别交错的数据：全部 True")
def p3_fixed_mixed(sol):
    run_checks(sol, "fixed", "mixed", rng())


@case("Part 3", title="有梯度恰为 0 的项（死 ReLU）")
def p3_fixed_dead_units(sol):
    # On a class-sorted training split the first 6 rows are close together, so some hidden units are inactive on
    # all of them and their gradients and finite differences are both exactly 0: that is a match, not a failure.
    run_checks(sol, "fixed", "sorted", rng())


@case("Part 3", title="bug 2：softmax 沿 batch 轴")
def p3_bug2(sol):
    run_checks(sol, "bug2", "mixed", rng())


@case("Part 3", title="bug 3：梯度漏了 1/N")
def p3_bug3(sol):
    run_checks(sol, "bug3", "mixed", rng())


@case("Part 3", title="bug 5：评估时也做 dropout")
def p3_bug5(sol):
    run_checks(sol, "bug5", "sorted", rng())


@case("Part 3", title="bug 6：accuracy 带 keepdims")
def p3_bug6(sol):
    run_checks(sol, "bug6", "mixed", rng())


@case("Part 3", title="初始化尺度过大")
def p3_bad_init(sol):
    run_checks(sol, "bad_init", "mixed", rng())


@case("Part 3", title="反向传播返回全 0")
def p3_zero_grad(sol):
    run_checks(sol, "zero_grad", "sorted", rng())


@case("Part 3", title="反向传播漏了 ReLU 导数")
def p3_no_relu_gate(sol):
    run_checks(sol, "no_relu_gate", "mixed", rng())


@case("Part 3", title="梯度很小时看相对误差")
def p3_tiny_no_relu_gate(sol):
    run_checks(sol, "tiny_no_relu_gate", "mixed", rng())


@case("Part 3", title="检查时要关闭 dropout")
def p3_train_mode_broken(sol):
    run_checks(sol, "train_drop_all", "mixed", rng())
