"""Helpers for tests.py: the statement's network, prompts, reward and evaluation, plus the formulas of A_i and L."""
import torch
import torch.nn as nn
import torch.nn.functional as F

V = 10
D_MODEL = 16
HIDDEN = 32
CLIP_EPS = 0.2
KL_COEF = 0.01
ADV_EPS = 1e-6
F64 = torch.float64


class Policy(nn.Module):
    """The statement's network, unchanged."""

    def __init__(self):
        super().__init__()
        self.embed_a = nn.Embedding(V, D_MODEL)
        self.embed_b = nn.Embedding(V, D_MODEL)
        self.net = nn.Sequential(nn.Linear(D_MODEL, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, V))

    def forward(self, a, b):
        return self.net(self.embed_a(a) + self.embed_b(b))


def sample_prompts(n, rng):
    a = torch.randint(0, V, (n,), generator=rng)
    b = torch.randint(0, V, (n,), generator=rng)
    return a, b


def reward(a, b, answer):
    return (answer == (a + b) % V).float()


@torch.no_grad()
def evaluate(policy, n, seed):
    """The statement's greedy held-out accuracy."""
    rng = torch.Generator().manual_seed(seed)
    a, b = sample_prompts(n, rng)
    pred = policy(a, b).argmax(dim=-1)
    return reward(a, b, pred).mean().item()


def advantages(rewards, population=False):
    """A_i = (r_i - mean) / (std + eps_adv) per row, in float64; std divides by G - 1 (population=True: by G)."""
    r = rewards.to(F64)
    std = r.std(dim=1, keepdim=True, unbiased=not population)
    return (r - r.mean(dim=1, keepdim=True)) / (std + ADV_EPS)


def loss_grad(logits, actions, adv, logp_old, logp_ref_all, exp=True):
    """dL/dlogits of the statement's L for one inner step, in float64, and each sample's ratio rho_i.

    logits: (N, V) of the current policy; actions: (N,); adv: (N,); logp_old: (N,) log pi_old(o_i);
    logp_ref_all: (N, V) log pi_ref. Row i of the gradient depends on sample i only.
    exp=False: rho_i = log pi - log pi_old, the follow-up's mistake."""
    z = logits.to(F64).requires_grad_(True)
    lp_all = F.log_softmax(z, dim=-1)
    lp = lp_all.gather(1, actions[:, None]).squeeze(1)
    ratio = torch.exp(lp - logp_old.to(F64)) if exp else lp - logp_old.to(F64)
    adv = adv.to(F64)
    surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv)
    kl = (lp_all.exp() * (lp_all - logp_ref_all.to(F64))).sum(dim=-1).mean()
    (g,) = torch.autograd.grad(-surrogate.mean() + KL_COEF * kl, z)
    return g, ratio.detach()
