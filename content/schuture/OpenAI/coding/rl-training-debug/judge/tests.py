import contextlib
import io

import torch
from torch.distributions import Categorical

from judgelib import Mismatch, case, expect, expect_close, expect_true, info, rng
import ref

torch.set_num_threads(1)        # the tensors are tiny: a second thread only adds CPU time

HORIZON, START = ref.HORIZON, ref.START

LOSS_TOL = dict(rtol=1e-5, atol=1e-6)       # a float32 loss of size about 1 against the float64 reference
GRAD_RTOL = 1e-4                            # float32 gradients summed over 32 positions, against float64
RET_TOL = 1e-5                              # at most 20 float32 roundings of numbers below 2 (each <= 1.2e-7)
CHI2_LIMIT = 44.84                          # chi-square with 3 degrees of freedom exceeds this with probability 1e-9


def seeds(r, n=2):
    return [r.randrange(2 ** 31) for _ in range(n)]


def tt(x, digits=4):
    """A tensor as pasteable code."""
    values = x.tolist()
    if x.is_floating_point():
        values = [[round(v, digits) for v in row] for row in values] if x.dim() == 2 else [round(v, digits) for v in values]
    return f"torch.tensor({values})"


def fmt(xs):
    return "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"


def last10(history, key):
    return sum(float(h[key]) for h in history[-10:]) / 10


# ---------------------------------------------------------------- the statement's run_tests, one test at a time

def check_test1(sol, model, call):
    _, actions, _, _ = sol.rollout(model, 64)
    if bool((actions == actions[:, :1]).all()):
        raise Mismatch("测试 1 没有通过：一次采样里 64 份环境副本走出了完全相同的轨迹", call, show=False)


TEST2_CALL = ("torch.manual_seed(7); probe = PolicyValue(); states, taken = torch.full((1, 8), 0), "
              "torch.zeros(1, 8, dtype=torch.long); returns = torch.full((1, 8), 5.0); "
              "opt = torch.optim.SGD(probe.parameters(), lr=0.1); loss, _ = surrogate(probe, states, taken, returns); "
              "opt.zero_grad(); loss.backward(); opt.step()")


def check_test2(sol, call):
    torch.manual_seed(7)
    probe = sol.PolicyValue()
    states, taken = torch.full((1, 8), START), torch.zeros(1, 8, dtype=torch.long)
    returns = torch.full((1, 8), 5.0)       # far above anything the untrained value head predicts
    with torch.no_grad():
        before = Categorical(logits=probe(states)[0]).log_prob(taken).mean().item()
    opt = torch.optim.SGD(probe.parameters(), lr=0.1)
    loss, _ = sol.surrogate(probe, states, taken, returns)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        after = Categorical(logits=probe(states)[0]).log_prob(taken).mean().item()
    info(f"测试 2：所采取动作的 log π 从 {before:.3f} 变成 {after:.3f}")
    expect_true(after > before, f"测试 2 没有通过：回报 5.0 远高于基线，走一步梯度之后所采取动作的 log π 却从 {before:.3f} "
                                f"变成 {after:.3f}，变得更不可能了", call)


TEST3_REWARDS = [[-0.02], [-0.02], [1.0], [-0.02]]
TEST3_DONES = [[False], [False], [True], [False]]
TEST3_CALL = (f"compute_returns(torch.tensor({TEST3_REWARDS}), torch.tensor({TEST3_DONES})).flatten()")


def returns_hint(got, rewards, dones):
    """Names the mistake when the answer's returns match a known wrong recursion."""
    r, d = rewards.tolist(), dones.tolist()
    g = got.detach().double()
    for wrong, text in ((ref.discounted_past, "这是第 t 步之前（含这一步）收集到的奖励的折扣和，而剩余回报是这一步及以后的奖励的折扣和："
                                              "递推 G_t = r_t + γ·G_{t+1} 要从最后一步倒着展开"),
                        (ref.uncut_reward_to_go, "剩余回报跨过了回合的结尾：done 的那一步之后已经是下一个回合，它的奖励不能算进来")):
        if torch.allclose(g, torch.tensor(wrong(r, d), dtype=torch.float64), rtol=0, atol=RET_TOL):
            return text
    return ""


def check_returns(sol, rewards, dones, call, prefix=""):
    T, B = rewards.shape
    want = torch.tensor(ref.reward_to_go(rewards.tolist(), dones.tolist()), dtype=torch.float64)
    got = sol.compute_returns(rewards.clone(), dones.clone())
    expect_true(isinstance(got, torch.Tensor), f"compute_returns 应当返回 tensor，得到的是 {type(got).__name__}", call, got=got)
    expect(tuple(got.shape), (T, B), call=call, msg=f"返回值的形状应当和 rewards 一样是 {(T, B)}，得到 {tuple(got.shape)}")
    if torch.allclose(got.detach().double(), want, rtol=0, atol=RET_TOL):
        return
    hint = returns_hint(got, rewards, dones)
    if hint:
        b = int((got.detach().double() - want).abs().amax(dim=0).argmax())
        where = f"（下面是第 {b} 份副本的 {T} 步）" if B > 1 else ""
        raise Mismatch(prefix + hint + where, call, [round(v, 4) for v in want[:, b].tolist()],
                       [round(v, 4) for v in got[:, b].tolist()])
    expect_close(got.detach(), want, rtol=0, atol=RET_TOL, call=call)


# ---------------------------------------------------------------- Part 1
SCRIPT = "model, history = train(); run_tests(model, history)"


def check_training(sol, model, history, call):
    check_test1(sol, model, call)
    bias, reward = last10(history, "advantage"), last10(history, "reward")
    expect_true(abs(bias) < 0.1, f"测试 4 没有通过：最后十轮的平均优势是 {bias:+.3f}，应当在 0 的 ±0.1 以内（基线有偏差）", call)
    expect_true(reward > 1.5, f"测试 5 没有通过：最后十轮里每次采样平均只收集到 {reward:+.3f} 的奖励，要求超过 1.5", call)


@case("Part 1", sample=True, cpu=12, title="python gridworld_pg.py：五项测试都通过")
def p1_script(sol):
    model, history = sol.train()
    info(f"最后十轮：每次采样平均奖励 {last10(history, 'reward'):+.3f}，平均优势 {last10(history, 'advantage'):+.3f}，"
         f"value loss {last10(history, 'value_loss'):.3f}")
    check_test1(sol, model, SCRIPT)
    check_test2(sol, SCRIPT)
    rewards, dones = torch.tensor(TEST3_REWARDS), torch.tensor(TEST3_DONES)
    got = sol.compute_returns(rewards, dones).flatten().tolist()
    expect_true(all(abs(a - b) < 5e-5 for a, b in zip(got, [0.9015, 0.95, 1.0, -0.02])) and len(got) == 4,
                f"测试 3 没有通过：剩余回报是 {[round(x, 4) for x in got]}，应当是 [0.9015, 0.95, 1.0, -0.02]", SCRIPT)
    check_training(sol, model, history, SCRIPT)


@case("Part 1", sample=True, title="测试 2：受奖励的动作变得更可能")
def p1_test2(sol):
    check_test2(sol, TEST2_CALL)


@case("Part 1", sample=True, title="测试 3：手写四步数据的剩余回报")
def p1_test3(sol):
    check_returns(sol, torch.tensor(TEST3_REWARDS), torch.tensor(TEST3_DONES), TEST3_CALL, prefix="测试 3 没有通过：")


PROBS = [0.1, 0.4, 0.2, 0.3]
FIXED = "policy = lambda pos: (torch.tensor([0.1, 0.4, 0.2, 0.3]).log().expand(*pos.shape, 4), torch.zeros(pos.shape))"


def fixed_policy(pos):
    return torch.tensor(PROBS).log().expand(*pos.shape, 4), torch.zeros(pos.shape)


@case("Part 1", title="rollout 的动作按策略的概率抽样")
def p1_rollout(sol):
    s, n = seeds(rng(), 1)[0], 2000
    call = f"{FIXED}; torch.manual_seed({s}); states, actions, rewards, dones = rollout(policy, {n})"
    torch.manual_seed(s)
    out = sol.rollout(fixed_policy, n)
    expect_true(isinstance(out, (tuple, list)) and len(out) == 4,
                "rollout 应当返回 (states, actions, rewards, dones) 四个 tensor", call, got=type(out).__name__)
    for name, x in zip(("states", "actions", "rewards", "dones"), out):
        expect_true(isinstance(x, torch.Tensor), f"{name} 应当是 tensor，得到的是 {type(x).__name__}", call)
        expect(tuple(x.shape), (HORIZON, n), call=call,
               msg=f"{name} 的形状应当是 (HORIZON, n_envs) = {(HORIZON, n)}，得到 {tuple(x.shape)}")
    states, actions, rewards, dones = (x.tolist() for x in out)
    expect_true(all(a in (0, 1, 2, 3) for row in actions for a in row), "actions 里有不在 0..3 之内的动作", call)
    for t in range(HORIZON):
        for b in range(n):
            pos, reward, done = ref.step(states[t][b], actions[t][b])
            nxt = states[t + 1][b] if t + 1 < HORIZON else pos
            if (nxt, done) != (pos, bool(dones[t][b])) or abs(rewards[t][b] - reward) > 1e-6 or (t == 0 and states[0][b] != START):
                raise Mismatch(f"第 {t} 步、第 {b} 份副本的记录与环境对不上：states[t] 应当是选动作时所在的格子，"
                               f"rewards[t]、dones[t] 是这一步 env_step 的结果，第 0 步从格子 {START} 出发", call, show=False)
    counts = torch.bincount(out[1].flatten().long(), minlength=4).double()
    total = counts.sum().item()
    expected = torch.tensor(PROBS, dtype=torch.float64) * total
    stat = ((counts - expected) ** 2 / expected).sum().item()
    freq = (counts / total).tolist()
    info(f"{int(total)} 个动作的频率 {fmt(freq)}（策略的概率 [0.1, 0.4, 0.2, 0.3]），χ² = {stat:.1f}")
    if stat >= CHI2_LIMIT:
        why = ("每一步都选了同一个动作：argmax 让行为变成确定性的，64 份副本于是走同一条轨迹" if (counts > 0).sum() == 1
               else "动作要从策略 π 本身抽样")
        raise Mismatch(f"策略在每个格子的动作概率都是 [0.1, 0.4, 0.2, 0.3]，rollout 采到的 {int(total)} 个动作的频率却是 "
                       f"{fmt(freq)}（χ² = {stat:.0f}，按 π 抽样时超过 {CHI2_LIMIT:g} 的概率不到 10⁻⁹）：{why}", call, show=False)


@case("Part 1", title="随机轨迹上的折扣剩余回报")
def p1_returns_random(sol):
    r = rng()
    for T, B, probs in [(10, 3, [0.0, 0.5, 0.0, 0.5]), (20, 64, [0.05, 0.45, 0.05, 0.45]), (20, 16, [0.1, 0.4, 0.1, 0.4]),
                        (20, 1, [0.0, 0.5, 0.0, 0.5]), (1, 4, [0.25] * 4), (12, 5, [0.25] * 4)]:
        rewards, dones = ref.walk(r, T, B, probs)
        call = f"compute_returns({tt(rewards)}, {tt(dones)})"
        if len(call) > 600:
            call = f"compute_returns(rewards, dones)  # rewards, dones: <{B} 份环境副本各走 {T} 步的一次随机采样>"
        check_returns(sol, rewards, dones, call, prefix=f"{T} 步、{B} 份副本：")


def surrogate_batch(r, T=4, B=8):
    g = torch.Generator().manual_seed(r.randrange(2 ** 31))
    states = torch.randint(0, 25, (T, B), generator=g)
    actions = torch.randint(0, 4, (T, B), generator=g)
    returns = torch.round(torch.empty(T, B).uniform_(-0.5, 1.7, generator=g), decimals=2)
    return states, actions, returns


def run_surrogate(sol, s, states, actions, returns):
    """The answer's surrogate on a fresh network: (loss value, {name: grad}, call). The reference is computed first."""
    torch.manual_seed(s)
    model = ref.PolicyValue()
    variants = {(flip, detach): ref.surrogate_grads(model, states, actions, returns, flip=flip, detach=detach)
                for flip in (False, True) for detach in (True, False)}
    call = (f"torch.manual_seed({s}); model = PolicyValue(); states = {tt(states)}; actions = {tt(actions)}; "
            f"returns = {tt(returns, 2)}; loss, _ = surrogate(model, states, actions, returns); loss.backward()")
    out = sol.surrogate(model, states.clone(), actions.clone(), returns.clone())
    expect_true(isinstance(out, (tuple, list)) and len(out) == 2, "surrogate 应当返回 (loss, stats)", call, got=out)
    loss = out[0]
    expect_true(isinstance(loss, torch.Tensor) and loss.dim() == 0 and loss.requires_grad,
                "loss 应当是一个能 backward() 的标量 tensor", call, got=loss)
    loss.backward()
    return loss.item(), {name: p.grad for name, p in model.named_parameters()}, variants, call


def close_grad(got, want):
    if got is None:
        return want.abs().max().item() == 0
    scale = want.abs().max().item()
    return torch.allclose(got.double(), want, rtol=GRAD_RTOL, atol=GRAD_RTOL * scale)


def check_grad(name, got, variants, call, hints):
    want = variants[False, True][1][name]
    if close_grad(got, want):
        return
    if got is None or got.abs().max().item() == 0:
        raise Mismatch(f"model.{name} 没有拿到梯度。" + hints.get("none", ""), call, show=False)
    for key, text in hints.items():
        if key != "none" and close_grad(got, variants[key][1][name]):
            raise Mismatch(f"model.{name} 的梯度不对：{text}", call, show=False)
    scale = want.abs().max().item()
    expect_close(got.detach(), want, rtol=GRAD_RTOL, atol=GRAD_RTOL * scale, call=call,
                 msg=f"model.{name} 的梯度与题目的 ℒ 不一致（最大误差 {(got.double() - want).abs().max().item():.3g}，"
                     f"梯度本身最大 {scale:.3g}）")


FLIPPED = ("这正是把优势写成 V(s) − G 时得到的结果。优势是回报减基线 G − V(s)；写反之后对 ℒ 做梯度下降就是在降低期望回报，"
           "比预测做得好的动作反而被压得更不可能")
ATTACHED = ("里面混进了策略项的梯度：policy_loss 里的优势没有 detach()。优势在 ∇log π · A 里只是一个系数，不 detach 的话策略项会把"
            "基线推到回报之上一个 −log π 的位置，平均优势停在 −H 附近（测试 4）")


@case("Part 1", title="损失值和策略头的梯度符合 ℒ")
def p1_policy_grad(sol):
    r = rng()
    for s in seeds(r, 3):
        states, actions, returns = surrogate_batch(r)
        got, grads, variants, call = run_surrogate(sol, s, states, actions, returns)
        want, flipped = variants[False, True][0], variants[True, True][0]
        if abs(got - want) > LOSS_TOL["atol"] + LOSS_TOL["rtol"] * abs(want) and \
                abs(got - flipped) <= LOSS_TOL["atol"] + LOSS_TOL["rtol"] * abs(flipped):
            raise Mismatch(f"损失值与题目的 ℒ 不一致：期望 {want:.6f}，得到 {got:.6f}。{FLIPPED}", call, show=False)
        expect_close(got, want, **LOSS_TOL, call=call, msg=f"损失值与题目的 ℒ 不一致：期望 {want:.6f}，得到 {got:.6f}")
        for name in ("policy.bias", "policy.weight"):
            check_grad(name, grads[name], variants, call, {(True, True): FLIPPED, (True, False): FLIPPED})


@case("Part 1", title="策略项不能训练基线（价值头的梯度）")
def p1_value_grad(sol):
    r = rng()
    for s in seeds(r, 3):
        states, actions, returns = surrogate_batch(r)
        _, grads, variants, call = run_surrogate(sol, s, states, actions, returns)
        hints = {"none": "value_loss 必须训练基线：如果在计算优势的地方就把 V(s) detach 掉，value_loss 也跟着没有梯度了",
                 (False, False): ATTACHED, (True, False): ATTACHED}
        for name in ("value.bias", "value.weight"):
            check_grad(name, grads[name], variants, call, hints)
        check_grad("trunk.0.weight", grads["trunk.0.weight"], variants, call, {**hints, (True, True): FLIPPED})


@case("Part 1", cpu=15, title="换两个种子训练：测试 1、4、5 照样通过")
def p1_more_seeds(sol):
    for s in seeds(rng(), 2):
        call = f"model, history = train(seed={s}); run_tests(model, history)"
        with contextlib.redirect_stdout(io.StringIO()):
            model, history = sol.train(seed=s)
        info(f"seed={s}：最后十轮每次采样平均奖励 {last10(history, 'reward'):+.3f}，平均优势 {last10(history, 'advantage'):+.3f}")
        check_training(sol, model, history, call)
