import copy
import math
from collections.abc import Mapping

import torch

from judgelib import Mismatch, case, expect, expect_close, expect_true, info, rng
import ref

torch.set_num_threads(1)        # the tensors are tiny: a second thread only adds CPU time

F64 = torch.float64

# A training run is a deterministic function of its seeds, but two correct loops may round the loss differently
# (a float32 running total, or .sum() * (1 / n)): that changes a history by about 2e-7 relative, and even a loss scaled
# by (1 + 2^-22) on every step moves it by at most 7e-7 on the runs drawn with seeds 0 and 1. The smallest real
# mistake judged here (an unweighted epoch average with a short last batch) moves it by 5e-3 or more.
HIST_RTOL, HIST_ATOL = 1e-4, 1e-9
SAME_MODEL_RTOL = 1e-5          # one evaluation of the same model, rounded in a different order: ~1e-7

X_TRAIN, Y_TRAIN = ref.make_data(1000, seed=0)
X_VAL, Y_VAL = ref.make_data(1000, seed=1000)
DEFAULT = ref.train(ref.make_model(0), X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)      # the statement's run, built at import

# (n_train, n_val, batch_size, epochs, lr)
UNEVEN = [(230, 200, 50, 3, 0.1), (40, 30, 7, 5, 0.1), (1000, 300, 128, 2, 0.1)]
EVEN = [(96, 64, 16, 4, 0.05), (320, 100, 64, 3, 0.08), (200, 100, 50, 3, 0.1)]


def seeds(r, n):
    return [r.randrange(2 ** 31) for _ in range(n)]


def setup(r, cfg, fn="train"):
    """Data, the model seed, keyword arguments and the call text for one configuration."""
    nt, nv, bs, ep, lr = cfg
    ds, vs, ms, ts = seeds(r, 4)
    Xt, yt = ref.make_data(nt, ds)
    Xv, yv = ref.make_data(nv, vs)
    kw = dict(lr=lr, epochs=ep, batch_size=bs, seed=ts)
    call = (f"X_tr, y_tr = make_data({nt}, {ds}); X_va, y_va = make_data({nv}, {vs}); "
            f"{fn}(make_model({ms}), X_tr, y_tr, X_va, y_va, lr={lr}, epochs={ep}, batch_size={bs}, seed={ts})")
    return (Xt, yt, Xv, yv), ms, kw, call


def copies(data):
    return tuple(t.clone() for t in data)


def values(h, key, call):
    """history[key] as a list of floats; any sequence of numbers or 0-d tensors is accepted."""
    expect_true(isinstance(h, Mapping), f"应当返回一个 dict，得到的是 {type(h).__name__}", call, got=h)
    if key not in h:
        raise Mismatch(f'返回的 dict 里没有 "{key}" 这一项', call, want=f'"{key}" 键', got=sorted(map(str, h)))
    try:
        return [float(v) for v in h[key]]
    except (TypeError, ValueError, RuntimeError):
        raise Mismatch(f'history["{key}"] 应当是一串数', call, got=h[key]) from None


def compare_history(h, want, keys, call, what="参考解答不同"):
    """Each list in keys must have want's length and match it entry by entry, within HIST_RTOL."""
    for key, wkey in keys:
        got, w = values(h, key, call), want[wkey]
        if len(got) != len(w):
            raise Mismatch(f'history["{key}"] 的长度应当是 {len(w)}，得到 {len(got)}', call, want=len(w), got=len(got))
        for i, (a, b) in enumerate(zip(got, w)):
            if not abs(a - b) <= HIST_RTOL * abs(b) + HIST_ATOL:      # also false for nan
                extra = "：出现了 nan / inf，训练发散了" if not math.isfinite(a) else ""
                raise Mismatch(f'history["{key}"][{i}] 和{what}{extra}', call, want=b, got=a)


def trained_in_place(model, Xv, yv, h, key, call):
    """The last validation entry is the evaluation of the model passed in, after training."""
    last = values(h, key, call)[-1]
    actual = ref.evaluate(model, Xv, yv)
    expect_true(abs(last - actual) <= SAME_MODEL_RTOL * abs(actual),
                f'history["{key}"][-1] 应当是训练完的 model 在验证集上 eval 模式下的 MSE（{actual:.6g}），'
                f"得到 {last:.6g}", call, got=last)


# ---------------------------------------------------------------- Part 1
P1_CALL = "train(make_model(0), X_train, y_train, X_val, y_val)"


@case("Part 1", sample=True, cpu=5, title="题目的训练：make_model(0)，40 个 epoch")
def p1_example(sol):
    data = (X_TRAIN.clone(), Y_TRAIN.clone(), X_VAL.clone(), Y_VAL.clone())
    h = sol.train(ref.make_model(0), *data)
    compare_history(h, DEFAULT, [("train", "train"), ("val", "val"), ("train_eval", "train_eval")], P1_CALL)
    info(f'val[-1] = {values(h, "val", P1_CALL)[-1]:.4f}，train[-1] = {values(h, "train", P1_CALL)[-1]:.4f}，'
         f'train_eval[-1] = {values(h, "train_eval", P1_CALL)[-1]:.4f}')


def evaluate_once(sol, model, X, y):
    """Call the answer's evaluate, recording (model.training, dropout.training, grad enabled) at each forward."""
    seen = []
    hook = model.register_forward_hook(lambda m, i, o: seen.append((m.training, m[2].training, torch.is_grad_enabled())))
    try:
        got = sol.evaluate(model, X.clone(), y.clone())
    finally:
        hook.remove()
    return got, seen


@case("Part 1", title="evaluate：eval 模式、不建图、值正确")
def p1_evaluate_value(sol):
    r = rng()
    for n, mode in [(50, "train"), (1, "train"), (300, "eval"), (1000, "train")]:
        ms, ds = seeds(r, 2)
        X, y = ref.make_data(n, ds)
        model = ref.make_model(ms)
        model.train(mode == "train")
        want = ref.evaluate(copy.deepcopy(model), X, y)
        call = (f"model = make_model({ms}); model.{mode}(); X, y = make_data({n}, {ds}); evaluate(model, X, y)")
        got, seen = evaluate_once(sol, model, X, y)
        if any(m or d for m, d, _ in seen):
            raise Mismatch("evaluate 做前向计算时模型还在训练模式，Dropout 是开着的", call, show=False)
        if any(grad for _, _, grad in seen):
            raise Mismatch("evaluate 做前向计算时仍在记录计算图（应当在 torch.no_grad() 下算）", call, show=False)
        try:
            value = float(got)
        except (TypeError, ValueError, RuntimeError):
            raise Mismatch("evaluate 应当返回一个数（均方误差）", call, want=want, got=got) from None
        expect_close(value, want, rtol=SAME_MODEL_RTOL, atol=0, call=call,
                     msg="均方误差不对（对样本取平均，没有 1/2）")


@case("Part 1", title="evaluate 之后模型模式不变")
def p1_evaluate_mode(sol):
    r = rng()
    X, y = ref.make_data(64, seeds(r, 1)[0])
    for ms, mode in zip(seeds(r, 2), ["train", "eval"]):
        model = ref.make_model(ms)
        model.train(mode == "train")
        call = f"model = make_model({ms}); model.{mode}(); evaluate(model, X, y); model.training"
        sol.evaluate(model, X.clone(), y.clone())
        bad = list(dict.fromkeys(type(m).__name__ for m in model.modules() if m.training != (mode == "train")))
        if mode == "train":
            expect_true(not bad, "调用前 model 是训练模式，evaluate 之后应当仍是训练模式（每个子模块都是），"
                                 f"现在 {', '.join(bad)} 是 eval 模式", call)
        else:
            expect_true(not bad, "调用前 model 是 eval 模式，evaluate 之后应当仍是 eval 模式（每个子模块都是），"
                                 f"现在 {', '.join(bad)} 是训练模式", call)


@case("Part 1", cpu=3, title="最后一个 batch 不满：按 batch 大小加权")
def p1_uneven_batches(sol):
    r = rng()
    for cfg in UNEVEN:
        data, ms, kw, call = setup(r, cfg)
        want = ref.train(ref.make_model(ms), *data, **kw)
        h = sol.train(ref.make_model(ms), *copies(data), **kw)
        compare_history(h, want, [("val", "val"), ("train_eval", "train_eval")], call)
        got = values(h, "train", call)
        if len(got) == len(want["train"]) and not all(abs(a - b) <= HIST_RTOL * abs(b) for a, b in zip(got, want["train"])):
            raise Mismatch(f"{data[0].shape[0]} 个样本、batch_size={kw['batch_size']}，最后一个 batch 不满："
                           f'history["train"] 应当是各 batch 损失按 batch 大小加权的平均', call, want=want["train"], got=got)
        compare_history(h, want, [("train", "train")], call)


@case("Part 1", cpu=3, title="其他学习率、batch 大小和 seed")
def p1_configs(sol):
    r = rng()
    for cfg in EVEN:
        data, ms, kw, call = setup(r, cfg)
        want = ref.train(ref.make_model(ms), *data, **kw)
        model = ref.make_model(ms)
        h = sol.train(model, *copies(data), **kw)
        compare_history(h, want, [("train", "train"), ("val", "val"), ("train_eval", "train_eval")], call)
        trained_in_place(model, data[2], data[3], h, "val", call)


@case("Part 1", title="epochs=0：只有训练前的一次评估")
def p1_zero_epochs(sol):
    r = rng()
    data, ms, kw, call = setup(r, (100, 50, 50, 0, 0.1))
    want = ref.train(ref.make_model(ms), *data, **kw)
    h = sol.train(ref.make_model(ms), *copies(data), **kw)
    compare_history(h, want, [("train", "train"), ("val", "val"), ("train_eval", "train_eval")], call)


# ---------------------------------------------------------------- Part 2
P2_CALL = "train_colleague(make_model(0), X_train, y_train, X_val, y_val)"
P2_KEYS = [("train", "train"), ("val", "val")]
P2_WHAT = " Part 1 的 train(...) 不同（修好后应当逐项相同）"


def colleague_want(want):
    return {"train": want["train"], "val": want["val"][1:]}


@case("Part 2", sample=True, cpu=5, title="修好之后与 Part 1 逐项相同")
def p2_example(sol):
    data = (X_TRAIN.clone(), Y_TRAIN.clone(), X_VAL.clone(), Y_VAL.clone())
    h = sol.train_colleague(ref.make_model(0), *data)
    compare_history(h, colleague_want(DEFAULT), P2_KEYS, P2_CALL, P2_WHAT)


@case("Part 2", title="每一步只用当前 batch 的梯度")
def p2_one_epoch(sol):
    r = rng()
    data, ms, kw, call = setup(r, (150, 100, 50, 1, 0.1), "train_colleague")
    # first one epoch of a single full batch (one update), then the same data in three batches of 50
    kw1, call1 = dict(kw, batch_size=150), call.replace("batch_size=50", "batch_size=150")
    want1 = ref.train(ref.make_model(ms), *data, **kw1)
    compare_history(sol.train_colleague(ref.make_model(ms), *copies(data), **kw1), colleague_want(want1), P2_KEYS,
                    call1, P2_WHAT)
    want = ref.train(ref.make_model(ms), *data, **kw)
    h = sol.train_colleague(ref.make_model(ms), *copies(data), **kw)
    got = values(h, "train", call)
    if len(got) == 1 and not abs(got[0] - want["train"][0]) <= HIST_RTOL * want["train"][0]:
        raise Mismatch(f"只做 1 次更新时结果正确（{call1.split('; ')[-1]}），分成 3 个 batch 后训练损失就和 Part 1 的 "
                       f"train(...) 不同：从第 2 步起，更新用的梯度不只是当前 batch 的梯度", call,
                       want=want["train"], got=got)
    compare_history(h, colleague_want(want), P2_KEYS, call, P2_WHAT)


@case("Part 2", cpu=3, title="模型不应只学到 y 的均值")
def p2_not_constant(sol):
    r = rng()
    ms, ts = seeds(r, 2)
    call = f"train_colleague(make_model({ms}), X_train, y_train, X_val, y_val, epochs=5, seed={ts})"
    want_model, model = ref.make_model(ms), ref.make_model(ms)
    ref.train(want_model, X_TRAIN, Y_TRAIN, X_VAL, Y_VAL, epochs=5, seed=ts)
    sol.train_colleague(model, X_TRAIN.clone(), Y_TRAIN.clone(), X_VAL.clone(), Y_VAL.clone(), epochs=5, seed=ts)
    with torch.no_grad():
        spread = copy.deepcopy(model).eval()(X_VAL).std().item()
        want_spread = want_model.eval()(X_VAL).std().item()
    # the fixed loop trains the same model as Part 1, so the spreads agree; the (B, 1) - (B,) loss gives ~0.04
    if not math.isfinite(spread):
        raise Mismatch("训练 5 个 epoch 后，模型在验证集上的输出出现了 nan / inf：训练发散了", call, show=False)
    expect_true(spread >= 0.2 * want_spread,
                f"训练 5 个 epoch 后，模型在验证集上的输出几乎是常数（标准差 {spread:.3g}，"
                f"y_val 的标准差是 {Y_VAL.std().item():.3g}，修好后应当约为 {want_spread:.3g}）", call)


@case("Part 2", title="验证损失按 eval 模式计算")
def p2_val_eval_mode(sol):
    r = rng()
    for cfg in [(200, 100, 50, 3, 0.1), (120, 80, 40, 2, 0.1)]:
        data, ms, kw, call = setup(r, cfg, "train_colleague")
        model = ref.make_model(ms)
        h = sol.train_colleague(model, *copies(data), **kw)
        trained_in_place(model, data[2], data[3], h, "val", call)


@case("Part 2", cpu=3, title="验证之后回到训练模式")
def p2_back_to_train(sol):
    r = rng()
    data, ms, kw, call = setup(r, (230, 200, 50, 3, 0.1), "train_colleague")
    want = ref.train(ref.make_model(ms), *data, **kw)
    h = sol.train_colleague(ref.make_model(ms), *copies(data), **kw)
    got = values(h, "train", call)
    if len(got) == 3 and abs(got[0] - want["train"][0]) <= HIST_RTOL * want["train"][0]:
        for e in (1, 2):
            if not abs(got[e] - want["train"][e]) <= HIST_RTOL * want["train"][e]:
                raise Mismatch(f'第 1 个 epoch 的训练损失正确，第 {e + 1} 个 epoch 起和 Part 1 的 train(...) 不同：'
                               f"验证之后的训练不对", call, want=want["train"], got=got)
    compare_history(h, colleague_want(want), P2_KEYS, call, P2_WHAT)


@case("Part 2", cpu=4, title="其他学习率、batch 大小和 seed")
def p2_configs(sol):
    r = rng()
    for cfg in [UNEVEN[1], EVEN[0], EVEN[1]]:
        data, ms, kw, call = setup(r, cfg, "train_colleague")
        want = ref.train(ref.make_model(ms), *data, **kw)
        h = sol.train_colleague(ref.make_model(ms), *copies(data), **kw)
        compare_history(h, colleague_want(want), P2_KEYS, call, P2_WHAT)


# ---------------------------------------------------------------- Part 3
X64, Y64 = X_TRAIN.double(), Y_TRAIN.double()
THETA0 = torch.randn(9, generator=torch.Generator().manual_seed(0), dtype=F64)
P3_DATA = "X = X_train.double(); y = y_train.double(); "
_XT = torch.cat([X64, torch.ones(1000, 1, dtype=F64)], dim=1)
H1 = torch.linalg.eigvalsh(2.0 / 1000 * _XT.T @ _XT)[-1].item()     # largest Hessian eigenvalue, ~2.453


def as_tensor(x, call, what):
    if isinstance(x, torch.Tensor):
        return x.detach().to(F64)
    try:
        return torch.as_tensor(x, dtype=F64)
    except (TypeError, ValueError, RuntimeError):
        raise Mismatch(f"{what} 应当返回一个 tensor", call, got=x) from None


def thetas(sol, X, y, lr, steps, call):
    got = as_tensor(sol.gd_linear(X.clone(), y.clone(), lr, steps), call, "gd_linear")
    expect(tuple(got.shape), (steps + 1, 9), call=call,
           msg=f"形状应当是 (steps + 1, 9) = ({steps + 1}, 9)：θ_0 到 θ_steps 各一行")
    return got


def check_rows(got, want, call):
    """Row t must equal the hand recurrence to 1e-10 of its own size: rounding over 100 steps stays near 1e-14."""
    for t in range(want.shape[0]):
        err = (got[t] - want[t]).abs().max().item()
        if not err <= 1e-10 * (1 + want[t].abs().max().item()):
            if t == 0 and want.shape[0] > 1 and bool((got == got[-1]).all()) and not bool((got[-1] == 0).all()):
                msg = ("每一行都相同，等于最后的 θ：保存的是参数本身（和参数共享内存），后面的更新把前面的行也改了，"
                       "保存时要 .clone()")
            elif t == 0:
                msg = "θ_0 应当全是 0：weight 和 bias 都初始化为 0"
            elif torch.allclose(got[t].roll(-1), want[t], rtol=1e-9, atol=1e-12):
                msg = f"θ_{t} 的数都对，但顺序应当是 (w_0, …, w_7, b)：bias 在最后"
            else:
                msg = f"θ_{t} 应当等于 θ_{t - 1} − lr · ∇L(θ_{t - 1})（L 是整个集合上的均方误差）"
            raise Mismatch(msg, call, want=want[t].tolist(), got=got[t].tolist())


@case("Part 3", sample=True, title="η = 0.9 · 2/h_1 跑 100 步：L ≈ 0.2893")
def p3_example_gd(sol):
    call = P3_DATA + "gd_linear(X, y, 0.9 * 2 / 2.453, 100)"
    got = thetas(sol, X64, Y64, 0.9 * 2 / 2.453, 100, call)
    if not bool((got[0] == 0).all()) and bool((got == got[-1]).all()):
        raise Mismatch("每一行都相同，等于最后的 θ：保存的是参数本身（和参数共享内存），后面的更新把前面的行也改了，"
                       "保存时要 .clone()", call, got=got[0].tolist())
    expect_true(bool((got[0] == 0).all()), "θ_0 应当全是 0：weight 和 bias 都初始化为 0", call, got=got[0].tolist())
    loss = ref.linear_loss(got[-1], X64, Y64).item()
    rolled = ref.linear_loss(got[-1].roll(-1), X64, Y64).item()
    hint = "。把每行的第一个数当作 bias 时 L 就对了：顺序应当是 (w_0, …, w_7, b)，bias 在最后" if abs(rolled - 0.2893) <= 5e-5 else ""
    # every factor |1 - lr h_i| is at most 0.8, so after 100 steps the excess loss is below 1e-15
    expect_true(abs(loss - 0.2893) <= 5e-5, f"100 步之后 L(θ) 应当约为 0.2893（题目给的 L(theta_hat)），"
                                           f"得到 {loss:.6g}{hint}", call, got=loss)


@case("Part 3", sample=True, title="由 linear_grad 拼出 H：h_1 ≈ 2.453")
def p3_example_h1(sol):
    call = (P3_DATA + "g0 = linear_grad(torch.zeros(9, dtype=torch.float64), X, y); "
            "H = torch.stack([linear_grad(e, X, y) - g0 for e in torch.eye(9, dtype=torch.float64)], 1)")
    g0 = as_tensor(sol.linear_grad(torch.zeros(9, dtype=F64), X64.clone(), Y64.clone()), call, "linear_grad")
    expect(tuple(g0.shape), (9,), call=call, msg="linear_grad 的形状应当是 (9,)")
    cols = [as_tensor(sol.linear_grad(e.clone(), X64.clone(), Y64.clone()), call, "linear_grad") - g0
            for e in torch.eye(9, dtype=F64)]
    H = torch.stack(cols, 1)
    H = (H + H.T) / 2                     # the gradient is affine in theta, so these columns are exactly H
    h1 = torch.linalg.eigvalsh(H)[-1].item()
    hint = "（只有一半：L 没有 1/2，求导多出因子 2）" if abs(h1 - 2.453 / 2) < 5e-3 else ""
    expect_true(abs(h1 - 2.453) <= 5e-4, f"H 的最大特征值应当约为 2.453，得到 {h1:.6g}{hint}", call, got=h1)


@case("Part 3", sample=True, title="theta0 处中心差分与 autograd 一致")
def p3_example_fd(sol):
    call = (P3_DATA + "theta0 = torch.randn(9, generator=torch.Generator().manual_seed(0), dtype=torch.float64); "
            "fd_grad(lambda t: linear_loss(t, X, y), theta0, 1e-6)")
    want = ref.autograd_grad(THETA0, X64, Y64)
    raw = sol.fd_grad(lambda t: ref.linear_loss(t, X64, Y64), THETA0.clone(), 1e-6)
    got = as_tensor(raw, call, "fd_grad")
    expect(tuple(got.shape), (9,), call=call, msg="fd_grad 的形状应当和 theta 一样，是 (9,)")
    err = ((got - want).norm() / want.norm()).item()
    # float64 round-off amplified by 1 / eps: about 1.5 u / eps = 2e-10 relative here; L is quadratic, so
    # central differences have no truncation error at all
    hint = "（fd_grad 返回的是 float32，精度不够）" if getattr(raw, "dtype", None) == torch.float32 else ""
    expect_true(err <= 1e-8, f"和 autograd 的相对误差是 {err:.3g}，float64、eps=1e-6 的中心差分应当在 1e-10 量级"
                             f"（这里要求不超过 1e-8）{hint}", call, got=got.tolist())


@case("Part 3", title="linear_grad 与 autograd 一致")
def p3_grad(sol):
    r = rng()
    for n in (1, 3, 40, 200):
        s = seeds(r, 1)[0]
        g = torch.Generator().manual_seed(s)
        X = torch.randn(n, 8, generator=g, dtype=F64) * 2 + torch.randn(8, generator=g, dtype=F64)
        y = torch.randn(n, generator=g, dtype=F64) + 1.5
        theta = torch.randn(9, generator=g, dtype=F64)
        call = (f"g = torch.Generator().manual_seed({s}); X = torch.randn({n}, 8, generator=g, dtype=torch.float64) * 2 "
                f"+ torch.randn(8, generator=g, dtype=torch.float64); y = torch.randn({n}, generator=g, "
                f"dtype=torch.float64) + 1.5; theta = torch.randn(9, generator=g, dtype=torch.float64); "
                f"linear_grad(theta, X, y)")
        want = ref.autograd_grad(theta, X, y)
        got = as_tensor(sol.linear_grad(theta.clone(), X.clone(), y.clone()), call, "linear_grad")
        if got.shape == want.shape and torch.allclose(got, want / 2, rtol=1e-9, atol=1e-12):
            raise Mismatch("梯度只有正确值的一半：L 没有 1/2，求导会多出因子 2", call, want=want.tolist(), got=got.tolist())
        # both sides are the same sums in a different order: they agree to ~1e-15 relative
        expect_close(got, want, rtol=1e-9, atol=1e-12, call=call, msg="和 autograd 算出的梯度不一致")


@case("Part 3", title="fd_grad 是中心差分（非二次函数）")
def p3_fd_central(sol):
    r = rng()
    f = lambda t: (t ** 3).sum() + t[0] * t[-1]
    for k, eps in [(1, 0.5), (4, 0.25), (9, 1e-3)]:
        s = seeds(r, 1)[0]
        theta = torch.randn(k, generator=torch.Generator().manual_seed(s), dtype=F64)
        call = (f"theta = torch.randn({k}, generator=torch.Generator().manual_seed({s}), dtype=torch.float64); "
                f"fd_grad(lambda t: (t ** 3).sum() + t[0] * t[-1], theta, {eps})")
        want = ref.central_differences(f, theta, eps)
        got = as_tensor(sol.fd_grad(f, theta.clone(), eps), call, "fd_grad")
        # f is cubic, so the central difference differs from f' by eps^2 per entry: only the exact formula passes;
        # round-off is below 1e-12 for these eps
        expect_close(got, want, rtol=1e-9, atol=1e-9, call=call,
                     msg="第 i 项应当是 (f(θ + eps·e_i) − f(θ − eps·e_i)) / (2·eps)")


@case("Part 3", title="gd_linear 的前几步")
def p3_gd_steps(sol):
    r = rng()
    for n, steps in [(30, 0), (30, 1), (5, 2), (60, 6)]:
        s = seeds(r, 1)[0]
        lr = round(r.uniform(0.05, 0.5), 3)
        g = torch.Generator().manual_seed(s)
        X = torch.randn(n, 8, generator=g, dtype=F64) + 0.5
        y = torch.randn(n, generator=g, dtype=F64) - 1.0
        call = (f"g = torch.Generator().manual_seed({s}); X = torch.randn({n}, 8, generator=g, dtype=torch.float64) "
                f"+ 0.5; y = torch.randn({n}, generator=g, dtype=torch.float64) - 1.0; gd_linear(X, y, {lr}, {steps})")
        want = ref.gd_recurrence(X, y, lr, steps)
        check_rows(thetas(sol, X, y, lr, steps, call), want, call)


@case("Part 3", title="阈值附近：c = 0.9、1.0、1.1")
def p3_gd_threshold(sol):
    for c in (0.9, 1.0, 1.1):
        lr = c * 2 / H1
        call = P3_DATA + f"gd_linear(X, y, {c} * 2 / {H1!r}, 100)"
        want = ref.gd_recurrence(X64, Y64, lr, 100)
        check_rows(thetas(sol, X64, Y64, lr, 100, call), want, call)
