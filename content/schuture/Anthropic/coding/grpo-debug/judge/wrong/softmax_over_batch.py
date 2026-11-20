# expect: RE
# Bug 1 fixed with softmax over the batch instead of the tokens: wrong samples, and later a row of zeros crashes torch.multinomial.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

V = 10                          # digits 0-9: both the two prompt digits and the possible answer tokens
D_MODEL = 16
HIDDEN = 32
CLIP_EPS = 0.2                   # epsilon: the clip range is [1 - CLIP_EPS, 1 + CLIP_EPS]
KL_COEF = 0.01                   # beta: weight of the KL penalty against the frozen reference policy
ADV_EPS = 1e-6                   # epsilon_adv: the advantage's standard-deviation denominator


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_a = nn.Embedding(V, D_MODEL)
        self.embed_b = nn.Embedding(V, D_MODEL)
        self.net = nn.Sequential(nn.Linear(D_MODEL, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, V))

    def forward(self, a, b):                   # a, b: (N,) digit ids -> logits (N, V)
        return self.net(self.embed_a(a) + self.embed_b(b))


def sample_prompts(n, rng):                     # n random (a, b) prompts, each digit uniform in [0, V)
    a = torch.randint(0, V, (n,), generator=rng)
    b = torch.randint(0, V, (n,), generator=rng)
    return a, b


def reward(a, b, answer):                       # verifiable reward: 1.0 iff answer == (a + b) mod 10
    return (answer == (a + b) % V).float()


def group_advantages(rewards):                   # rewards: (P, G) -> advantages (P, G)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + ADV_EPS)     # bug 2: an all-equal group has std == 0


def train(policy, ref_policy, steps=200, batch_size=100, group_size=16, inner_steps=4, lr=0.02, seed=0):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)
    for step in range(steps):
        a, b = sample_prompts(batch_size, rng)
        a_rep, b_rep = a.repeat_interleave(group_size), b.repeat_interleave(group_size)
        with torch.no_grad():
            logits_old = policy(a_rep, b_rep)
            probs_old = F.softmax(logits_old, dim=0)                     # bug 1: multinomial wants probabilities
            actions = torch.multinomial(probs_old, 1, generator=rng).squeeze(1)
            logp_old = F.log_softmax(logits_old, dim=-1).gather(1, actions[:, None]).squeeze(1)
            logp_ref_all = F.log_softmax(ref_policy(a_rep, b_rep), dim=-1)
        r = reward(a_rep, b_rep, actions).view(batch_size, group_size)
        adv = group_advantages(r).reshape(-1)
        for _ in range(inner_steps):
            logits_new = policy(a_rep, b_rep)
            logp_new_all = F.log_softmax(logits_new, dim=-1)
            logp_new = logp_new_all.gather(1, actions[:, None]).squeeze(1)
            ratio = torch.exp(logp_new - logp_old)     # bug 3: logp_old stays the rollout's value
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            pg_loss = -torch.min(ratio * adv, clipped * adv).mean()
            probs_new_all = logp_new_all.exp()
            kl = (probs_new_all * (logp_new_all - logp_ref_all)).sum(dim=-1).mean()
            loss = pg_loss + KL_COEF * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy


@torch.no_grad()
def evaluate(policy, n, seed):                   # greedy accuracy on n freshly sampled held-out prompts
    rng = torch.Generator().manual_seed(seed)
    a, b = sample_prompts(n, rng)
    pred = policy(a, b).argmax(dim=-1)
    return reward(a, b, pred).mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    policy = Policy()
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    train(policy, ref_policy)
    accuracy = evaluate(policy, 2000, seed=999)
    print(f"held-out accuracy: {accuracy:.3f}")
    assert accuracy > 0.5, "the policy did not learn to add single digits"
    print("cleared the reward threshold")
