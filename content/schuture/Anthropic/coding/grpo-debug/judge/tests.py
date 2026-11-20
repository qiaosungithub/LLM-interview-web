import contextlib
import copy

import torch
import torch.nn.functional as F

from judgelib import Mismatch, case, expect_close, expect_true, info, rng
import ref

torch.set_num_threads(1)        # the tensors are tiny: a second thread only adds CPU time

V, F64 = ref.V, ref.F64
CLIP = ref.CLIP_EPS

# A_i in float32 is within ~1e-7 relative of the float64 formula. Where eps_adv enters the denominator (std + eps,
# max(std, eps), or a 0 written for an all-equal group) moves a non-constant group by eps / std <= 4e-6 relative
# for the groups used here (binary rewards, G <= 16, so std >= 1/4): the statement says eps_adv barely matters
# there. Dividing by G instead of G - 1 moves A by a factor sqrt(G / (G - 1)): 3% at G = 16, 15% at G = 4.
ADV_RTOL, ADV_ATOL = 1e-5, 1e-6
# The statement prints its example to 4 decimals: A = (0.8660, -0.8660, ...), exactly 0.5 / (0.57735 + 1e-6) = 0.86602.
EXAMPLE_ATOL = 1e-4

# dL/dlogits computed in float32 is within ~1e-6 of its largest entry from the float64 formula; a 1e-5 relative
# change of A_i (above) moves it by at most that much. Leaving logp_old pinned to the current policy changes the
# second inner step by ~1e-2 of the largest entry.
GRAD_RTOL, GRAD_ATOL = 1e-4, 1e-9
NEAR_CLIP = 1e-4                # a sample whose rho is this close to 1 +- eps may round to either side of the clip

# Rollouts from constant logits SAMPLE_BIAS: the chi-square statistic of 6400 draws (9 degrees of freedom) exceeds
# 60 with probability 1e-9 for a correct sampler. Sampling in proportion to the logits shifted by their minimum, to
# |logits|, relu(logits), sigmoid(logits), softmax(logits / 2), or uniformly gives 1000 or more.
SAMPLE_BIAS = [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, 2.0, -2.0, 0.25]
CHI2_LIMIT = 60.0

SCRIPT = ("torch.manual_seed(0); policy = Policy(); ref_policy = copy.deepcopy(policy); "
          "train(policy, ref_policy); evaluate(policy, 2000, seed=999)")


def seeds(r, n=2):
    return [r.randrange(2 ** 31) for _ in range(n)]


def frozen_copy(policy):
    """pi_ref: the statement's network with policy's current weights, excluded from the optimizer."""
    ref_policy = ref.Policy()
    ref_policy.load_state_dict(policy.state_dict())
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    return ref_policy


@contextlib.contextmanager
def watch_reward(sol, record):
    """Record each call train makes to the answer's reward(a, b, answer); restore reward afterwards."""
    original = sol.reward

    def reward(a, b, answer):
        record(a, b, answer)
        return original(a, b, answer)

    sol.reward = reward
    try:
        yield
    finally:
        sol.reward = original


class Watched(ref.Policy):
    """The statement's network; records each forward call and the gradient that reaches its logits."""

    def __init__(self):
        super().__init__()
        self.events = []

    def forward(self, a, b):
        out = super().forward(a, b)
        rec = {"kind": "forward", "a": a.detach().clone(), "b": b.detach().clone(), "out": out.detach().clone(),
               "grad": None}
        if out.requires_grad:
            out.register_hook(lambda g, rec=rec: rec.__setitem__("grad", g.detach().clone()))
        self.events.append(rec)
        return out


def fmt(xs):
    return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"


# ---------------------------------------------------------------- the fixed script
@case("Part 1", sample=True, cpu=10, title="python grpo_train.py：留出集准确率超过 0.5")
def p1_script(sol):
    torch.manual_seed(0)
    policy = sol.Policy()
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    sol.train(policy, ref_policy)
    expect_true(all(torch.isfinite(p).all() for p in policy.parameters()), "训练结束后策略的参数里有 nan / inf", SCRIPT)
    accuracy = ref.evaluate(policy, 2000, seed=999)
    info(f"留出集贪心准确率 {accuracy:.3f}（随机水平 0.1）")
    expect_true(accuracy > 0.5, f"留出集贪心准确率是 {accuracy:.3f}，没有超过 0.5", SCRIPT)


# ---------------------------------------------------------------- group_advantages
def check_adv(sol, r, call, want=None, rtol=ADV_RTOL, atol=ADV_ATOL):
    want = ref.advantages(r) if want is None else want
    got = sol.group_advantages(r.clone())
    expect_true(isinstance(got, torch.Tensor), f"应当返回 tensor，得到的是 {type(got).__name__}", call, got=got)
    expect_true(tuple(got.shape) == tuple(r.shape), f"形状应当是 {tuple(r.shape)}，得到 {tuple(got.shape)}", call,
                got=tuple(got.shape))
    got = got.detach().to(F64)
    if not torch.isfinite(got).all():
        constant = (r == r[:, :1]).all(dim=1)
        if (~torch.isfinite(got)).any(dim=1)[constant].any():
            raise Mismatch("组内奖励全部相同时 std 恰好是 0，(rewards - mean) / std 是 0/0 = nan；"
                           "这样一组的优势应当是 0", call, want=want, got=got)
        raise Mismatch("结果里有 nan / inf", call, want=want, got=got)
    if (not torch.allclose(got, want, rtol=rtol, atol=atol)
            and torch.allclose(got, ref.advantages(r, population=True), rtol=ADV_RTOL, atol=ADV_ATOL)):
        raise Mismatch("std 应当除以 G - 1（题目的公式），现在像是除以 G", call, want=want, got=got)
    expect_close(got, want, rtol=rtol, atol=atol, call=call,
                 msg="和题目的公式 A_i = (r_i - mean) / (std + ε_adv) 不一致")
    return got


@case("Part 1", sample=True, title="题目例子：奖励 (1, 0, 1, 0) 的优势")
def p1_adv_example(sol):
    got = check_adv(sol, torch.tensor([[1.0, 0.0, 1.0, 0.0]]), "group_advantages(torch.tensor([[1., 0., 1., 0.]]))",
                    want=torch.tensor([[0.8660, -0.8660, 0.8660, -0.8660]], dtype=F64), rtol=0.0, atol=EXAMPLE_ATOL)
    info(f"A = {fmt(got[0].tolist())}")


@case("Part 1", title="奖励全部相同的组：优势是 0，不是 nan")
def p1_adv_constant(sol):
    check_adv(sol, torch.tensor([[0.0, 0.0, 0.0, 0.0]]), "group_advantages(torch.tensor([[0., 0., 0., 0.]]))")
    rows = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
    check_adv(sol, torch.tensor(rows), f"group_advantages(torch.tensor({rows}))")
    r = torch.zeros(3, 16)
    r[1, 5], r[2] = 1.0, 1.0
    check_adv(sol, r, "r = torch.zeros(3, 16); r[1, 5] = 1; r[2] = 1; group_advantages(r)")


@case("Part 1", title="随机奖励：各种组大小")
def p1_adv_random(sol):
    r = rng()
    for P, G, p in [(1, 2, 0.5), (5, 3, 0.4), (7, 8, 0.2), (100, 16, 0.1), (100, 16, 0.6)]:
        (s,) = seeds(r, 1)
        rewards = (torch.rand(P, G, generator=torch.Generator().manual_seed(s)) < p).float()
        check_adv(sol, rewards, f"r = (torch.rand({P}, {G}, generator=torch.Generator().manual_seed({s})) < {p}).float(); "
                                f"group_advantages(r)")


# ---------------------------------------------------------------- sampling the rollouts
@case("Part 1", title="rollout 的回答从 softmax(logits) 中采样")
def p1_sampling(sol):
    r = rng()
    s, ts = seeds(r)
    P, G = 400, 16
    n = P * G
    torch.manual_seed(s)
    policy = ref.Policy()
    with torch.no_grad():
        policy.net[2].weight.zero_()                      # every prompt gets the logits SAMPLE_BIAS
        policy.net[2].bias.copy_(torch.tensor(SAMPLE_BIAS))
    ref_policy = frozen_copy(policy)
    call = (f"torch.manual_seed({s}); policy = Policy(); policy.net[2].weight.data.zero_(); "
            f"policy.net[2].bias.data = torch.tensor({SAMPLE_BIAS}); ref_policy = copy.deepcopy(policy); "
            f"train(policy, ref_policy, steps=1, batch_size={P}, group_size={G}, inner_steps=1, seed={ts})")
    seen = []
    with watch_reward(sol, lambda a, b, answer: seen.append(answer.detach().clone())):
        sol.train(policy, ref_policy, steps=1, batch_size=P, group_size=G, inner_steps=1, seed=ts)
    expect_true(len(seen) == 1, f"steps=1 时 train 应当调用一次 reward(a, b, answer)，现在调用了 {len(seen)} 次", call)
    actions = seen[0].reshape(-1)
    expect_true(actions.numel() == n, f"应当采样 {P} × {G} = {n} 个回答，传给 reward 的有 {actions.numel()} 个", call)
    actions = actions.long()
    expect_true(bool(((actions >= 0) & (actions < V)).all()), "回答应当是 0 到 9 的 token", call)
    counts = torch.bincount(actions, minlength=V).to(F64)
    p = torch.softmax(torch.tensor(SAMPLE_BIAS, dtype=F64), dim=0)
    if counts.max() == n:
        raise Mismatch(f"{n} 个回答全是同一个 token：rollout 应当从 π_θold 中随机采样，而不是取 argmax", call, show=False)
    stat = ((counts - n * p) ** 2 / (n * p)).sum().item()
    expect_true(stat < CHI2_LIMIT,
                f"每个 prompt 的 logits 都是 {SAMPLE_BIAS}，回答的分布应当是 softmax(logits) = {fmt(p.tolist())}，"
                f"实际频率是 {fmt((counts / n).tolist())}（卡方 {stat:.0f}；正确采样时超过 {CHI2_LIMIT:g} 的概率约 1e-9）",
                call)


# ---------------------------------------------------------------- the inner steps
INNER = [(20, 8, 2, 3, 0.02), (12, 4, 3, 2, 0.05)]       # batch_size P, group_size G, steps, inner_steps, lr


def logp_of(logits, actions):
    return F.log_softmax(logits.to(F64), dim=-1).gather(1, actions[:, None]).squeeze(1)


def check_inner(sol, events, ref_policy, P, G, steps, inner, call):
    """Every backward pass through the logits must carry dL/dlogits of the statement's L, step by step."""
    outer = []
    for e in events:
        if e["kind"] == "reward":
            outer.append((e, []))
        elif e["grad"] is not None and outer:
            outer[-1][1].append(e)
    expect_true(len(outer) == steps,
                f"steps={steps}：train 应当做 {steps} 个外层步，每步调用一次 reward(...)，现在调用了 {len(outer)} 次", call)
    for k, (rew, inner_calls) in enumerate(outer, 1):
        expect_true(len(inner_calls) == inner, f"第 {k} 个外层步应当做 {inner} 个内层步（对 L(θ) 反向传播 {inner} 次），"
                                               f"现在是 {len(inner_calls)} 次", call)
        a, b, actions = rew["a"], rew["b"], rew["answer"].reshape(-1).long()
        old = ref.Policy()
        old.load_state_dict(rew["params"])                 # theta_old: the policy when this step's rollouts were drawn
        with torch.no_grad():
            logp_old = logp_of(old(a, b), actions)
            logp_ref_all = F.log_softmax(ref_policy(a, b).to(F64), dim=-1)
        rewards = ref.reward(a, b, actions).view(P, G)
        adv = ref.advantages(rewards).reshape(-1)
        for j, f in enumerate(inner_calls, 1):
            where = f"第 {k} 个外层步的第 {j} 个内层步"
            expect_true(torch.equal(f["a"], a) and torch.equal(f["b"], b),
                        f"{where}：应当对本外层步采样的同一批 P·G 个 (a, b) 计算 L(θ)", call)
            got = f["grad"].to(F64)
            if not torch.isfinite(got).all():
                hint = ""
                with torch.no_grad():
                    if not torch.isfinite(torch.as_tensor(sol.group_advantages(rewards.clone()))).all():
                        hint = "：group_advantages 对奖励全部相同的组返回了 nan"
                raise Mismatch(f"{where}，L(θ) 对 logits 的梯度里有 nan / inf{hint}", call, show=False)
            want, ratio = ref.loss_grad(f["out"], actions, adv, logp_old, logp_ref_all)
            keep = ((ratio - (1 - CLIP)).abs() > NEAR_CLIP) & ((ratio - (1 + CLIP)).abs() > NEAR_CLIP)

            def off(w):
                return (got - w)[keep].abs().max().item() if keep.any() else 0.0

            scale = want.abs().max().item()
            if off(want) <= GRAD_RTOL * scale + GRAD_ATOL:
                continue
            guesses = [
                (logp_of(f["out"], actions),
                 "梯度等于 ρ_i ≡ 1 时的梯度：logp_old 在内层循环里被换成了当前策略的值。ρ_i 里的 log π_θold 应当一直是"
                 "本外层步采样时的值，只有第 1 个内层步它才和当前策略相同"),
                (logp_of(ref_policy(a, b).detach(), actions),
                 "ρ_i 里用的像是冻结的 π_ref 的对数概率。θ_old 是本外层步采样时的策略，每个外层步都换一份；π_ref 是训练开始前"
                 "的那份，只出现在 KL 项里"),
            ]
            for guess, msg in guesses:
                g, _ = ref.loss_grad(f["out"], actions, adv, guess, logp_ref_all)
                if off(g) <= GRAD_RTOL * scale + GRAD_ATOL:
                    raise Mismatch(f"{where}：{msg}", call, show=False)
            g, _ = ref.loss_grad(f["out"], actions, adv, logp_old, logp_ref_all, exp=False)
            if off(g) <= GRAD_RTOL * scale + GRAD_ATOL:
                raise Mismatch(f"{where}：ρ_i 应当是 exp(log π_θ − log π_θold)，现在像是没有取 exp", call, show=False)
            raise Mismatch(f"{where}：L(θ) 对 logits 的梯度和题目的公式不一致（最大偏差 {off(want):.3g}，"
                           f"梯度最大的分量 {scale:.3g}）", call, show=False)


@case("Part 1", title="每个内层步的梯度：ρ_i 用本外层步采样时的 log π_θold")
def p1_inner_steps(sol):
    r = rng()
    for P, G, steps, inner, lr in INNER:
        s, ts = seeds(r)
        call = (f"torch.manual_seed({s}); policy = Policy(); ref_policy = copy.deepcopy(policy); "
                f"train(policy, ref_policy, steps={steps}, batch_size={P}, group_size={G}, inner_steps={inner}, "
                f"lr={lr}, seed={ts})")
        torch.manual_seed(s)
        policy = Watched()
        ref_policy = frozen_copy(policy)

        def on_reward(a, b, answer):
            policy.events.append({"kind": "reward", "a": a.detach().clone(), "b": b.detach().clone(),
                                  "answer": answer.detach().clone(),
                                  "params": {k: v.detach().clone() for k, v in policy.state_dict().items()}})

        with watch_reward(sol, on_reward):
            sol.train(policy, ref_policy, steps=steps, batch_size=P, group_size=G, inner_steps=inner, lr=lr, seed=ts)
        check_inner(sol, policy.events, ref_policy, P, G, steps, inner, call)
