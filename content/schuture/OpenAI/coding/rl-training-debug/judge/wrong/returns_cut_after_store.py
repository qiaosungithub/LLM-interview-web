# expect: WA
# Wrong: compute_returns loops backwards but cuts the chain after storing returns[t], so a terminal step still adds the next episode.
import torch
import torch.nn as nn
from torch.distributions import Categorical

GRID, START, GOAL = 5, 0, 24                # states 0..24 of a 5x5 grid, row-major; 0 is top-left
HORIZON, STEP_COST, GAMMA = 20, 0.02, 0.97
VALUE_COEF, ENT_COEF = 0.5, 0.02


def env_step(pos, action):                  # pos, action: (B,) long
    """One step in every environment. Returns the next position, the reward and the done flag."""
    row, col = pos // GRID, pos % GRID
    row = (row + (action == 1).long() - (action == 0).long()).clamp(0, GRID - 1)   # 0 = up, 1 = down
    col = (col + (action == 3).long() - (action == 2).long()).clamp(0, GRID - 1)   # 2 = left, 3 = right
    moved = row * GRID + col
    done = moved == GOAL
    reward = torch.where(done, 1.0, -STEP_COST)
    return torch.where(done, torch.full_like(moved, START), moved), reward, done


class PolicyValue(nn.Module):
    def __init__(self, n_states=GRID * GRID, n_actions=4, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Embedding(n_states, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, pos):                 # pos: (...,) long -> logits (..., n_actions), values (...,)
        hidden = self.trunk(pos)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


@torch.no_grad()
def rollout(model, n_envs):
    """HORIZON steps in n_envs environments. Every returned tensor has shape (HORIZON, n_envs)."""
    pos = torch.full((n_envs,), START)
    states, actions, rewards, dones = [], [], [], []
    for _ in range(HORIZON):
        logits, _ = model(pos)
        action = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
        states.append(pos)
        actions.append(action)
        pos, reward, done = env_step(pos, action)
        rewards.append(reward)
        dones.append(done)
    return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(dones)


def compute_returns(rewards, dones):        # rewards, dones: (T, B) -> (T, B)
    """returns[t] is the discounted reward-to-go of step t inside its own episode."""
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + GAMMA * running
        returns[t] = running
        running = running * (~dones[t]).float()
    return returns


def surrogate(model, states, actions, returns):
    """The loss whose gradient is the policy gradient, plus the baseline and entropy terms."""
    logits, values = model(states)
    dist = Categorical(logits=logits)
    advantages = -(values - returns)
    policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
    value_loss = advantages.pow(2).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + VALUE_COEF * value_loss - ENT_COEF * entropy
    return loss, {"value_loss": value_loss.item(), "entropy": entropy.item(),
                  "advantage": advantages.mean().item()}


def train(iters=120, n_envs=64, lr=0.02, seed=0):
    torch.manual_seed(seed)
    model = PolicyValue()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for it in range(iters):
        states, actions, rewards, dones = rollout(model, n_envs)
        returns = compute_returns(rewards, dones)
        loss, stats = surrogate(model, states, actions, returns)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append({"reward": rewards.sum(0).mean().item(), **stats})
        if it % 20 == 0 or it == iters - 1:
            h = history[-1]
            print(f"iter {it:3d}  reward {h['reward']:+.3f}  entropy {h['entropy']:.3f}  "
                  f"value loss {h['value_loss']:.3f}  mean adv {h['advantage']:+.3f}")
    return model, history


def run_tests(model, history):
    _, actions, _, _ = rollout(model, 64)
    assert not (actions == actions[:, :1]).all(), "test 1: every environment ran the same trajectory"

    torch.manual_seed(7)
    probe = PolicyValue()
    states, taken = torch.full((1, 8), START), torch.zeros(1, 8, dtype=torch.long)
    returns = torch.full((1, 8), 5.0)       # far above anything the untrained value head predicts
    with torch.no_grad():
        before = Categorical(logits=probe(states)[0]).log_prob(taken).mean().item()
    opt = torch.optim.SGD(probe.parameters(), lr=0.1)
    loss, _ = surrogate(probe, states, taken, returns)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        after = Categorical(logits=probe(states)[0]).log_prob(taken).mean().item()
    assert after > before, f"test 2: log pi of a rewarded action went {before:.3f} -> {after:.3f}"

    rewards = torch.tensor([[-0.02], [-0.02], [1.0], [-0.02]])
    dones = torch.tensor([[False], [False], [True], [False]])
    got = [round(x, 4) for x in compute_returns(rewards, dones).flatten().tolist()]
    assert got == [0.9015, 0.95, 1.0, -0.02], f"test 3: reward-to-go is {got}"

    bias = sum(h["advantage"] for h in history[-10:]) / 10
    assert abs(bias) < 0.1, f"test 4: the baseline is off by {bias:+.3f} on average"

    mean_reward = sum(h["reward"] for h in history[-10:]) / 10
    assert mean_reward > 1.5, f"test 5: the agent collects only {mean_reward:+.3f} per rollout"


if __name__ == "__main__":
    model, history = train()
    run_tests(model, history)
    print("all tests passed")
