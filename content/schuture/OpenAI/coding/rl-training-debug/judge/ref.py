"""Independent restatement of the problem, used by tests.py for expected values and inputs.

The environment is rewritten one step at a time in plain Python, the reward-to-go is a direct discounted sum over the
rest of each episode (no recursion), and the loss is the statement's formula evaluated in float64 with the advantage
used as a coefficient.
"""
import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical

GRID, START, GOAL = 5, 0, 24
HORIZON, STEP_COST, GAMMA = 20, 0.02, 0.97
VALUE_COEF, ENT_COEF = 0.5, 0.02


def step(pos, action):
    """One step of one environment: (next position, reward, done)."""
    row, col = divmod(pos, GRID)
    row = min(max(row + (action == 1) - (action == 0), 0), GRID - 1)
    col = min(max(col + (action == 3) - (action == 2), 0), GRID - 1)
    moved = row * GRID + col
    if moved == GOAL:
        return START, 1.0, True
    return moved, -STEP_COST, False


class PolicyValue(nn.Module):
    """The statement's network, unchanged, so that torch.manual_seed(s); PolicyValue() gives the same weights."""

    def __init__(self, n_states=GRID * GRID, n_actions=4, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Embedding(n_states, hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, pos):
        hidden = self.trunk(pos)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


# ---------------------------------------------------------------- reward-to-go

def reward_to_go(rewards, dones):
    """returns[t][b] = sum over k = t .. end of GAMMA^(k - t) * rewards[k][b], where end is the first k >= t with
    dones[k][b] (or the last step). Nested lists in, nested lists of floats out."""
    T, B = len(rewards), len(rewards[0])
    out = [[0.0] * B for _ in range(T)]
    for b in range(B):
        for t in range(T):
            total, k = 0.0, t
            while True:
                total += GAMMA ** (k - t) * float(rewards[k][b])
                if dones[k][b] or k == T - 1:
                    break
                k += 1
            out[t][b] = total
    return out


def discounted_past(rewards, dones):
    """What the forward loop of the given file computes: the discounted sum of the rewards up to and including step t,
    cut where an earlier step ended an episode."""
    T, B = len(rewards), len(rewards[0])
    out = [[0.0] * B for _ in range(T)]
    for b in range(B):
        running = 0.0
        for t in range(T):
            running = float(rewards[t][b]) + GAMMA * running * (not dones[t][b])
            out[t][b] = running
    return out


def uncut_reward_to_go(rewards, dones):
    """The reward-to-go if episode ends are ignored: the sum runs on to the end of the rollout."""
    return reward_to_go(rewards, [[False] * len(rewards[0]) for _ in rewards])


def walk(r, T, B, probs):
    """T steps of B environments with actions drawn from probs by the random.Random r.
    Returns (rewards float32 (T, B), dones bool (T, B))."""
    pos = [START] * B
    rewards, dones = [], []
    for _ in range(T):
        row_r, row_d = [], []
        for b in range(B):
            pos[b], reward, done = step(pos[b], r.choices(range(4), weights=probs)[0])
            row_r.append(reward)
            row_d.append(done)
        rewards.append(row_r)
        dones.append(row_d)
    return torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones)


# ---------------------------------------------------------------- the loss

def surrogate_grads(model, states, actions, returns, flip=False, detach=True):
    """The statement's loss L on a float64 copy of model: (loss, {parameter name: gradient}).
    flip=True uses V - G as the advantage and detach=False lets the policy term reach the baseline: the two bugs of
    surrogate, used only to name a mistake in a message."""
    m = copy.deepcopy(model).double()
    logits, values = m(states)
    dist = Categorical(logits=logits)
    advantages = returns.double() - values
    if flip:
        advantages = -advantages
    coef = advantages.detach() if detach else advantages
    loss = (-(dist.log_prob(actions) * coef).mean() + VALUE_COEF * advantages.pow(2).mean()
            - ENT_COEF * dist.entropy().mean())
    loss.backward()
    return loss.item(), {name: p.grad.clone() for name, p in m.named_parameters()}
