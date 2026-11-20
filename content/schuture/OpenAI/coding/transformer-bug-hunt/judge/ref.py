"""Judge-side helpers for tests.py: the data, the fixed TinyGPT and training loop (without the KV cache), and checks
that only use an answer through the interface the statement fixes."""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIGITS, SEP = 6, 10
VOCAB, MAX_LEN = 11, 2 * N_DIGITS + 1


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.register_buffer("visible", torch.ones(max_len, max_len).tril().bool())

    def split_heads(self, x):
        return x.unflatten(-1, (self.n_head, self.d_head)).transpose(1, 2)

    def forward(self, x):
        T = x.shape[1]
        q, k, v = self.split_heads(self.w_q(x)), self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))
        scores = q @ k.transpose(-1, -2) / self.d_head ** 0.5
        scores = scores.masked_fill(~self.visible[:T, :T], float("-inf"))
        return self.w_o((scores.softmax(dim=-1) @ v).transpose(1, 2).flatten(-2))


class Block(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, max_len)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """A correct model, handed to the answer's train() so that its training loop is judged on its own."""

    def __init__(self, vocab=VOCAB, d_model=64, n_head=4, n_layer=1, max_len=MAX_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_head, max_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb[:, :idx.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def make_batch(batch_size, rng):
    digits = torch.randint(0, 10, (batch_size, N_DIGITS), generator=rng)
    sep = torch.full((batch_size, 1), SEP)
    return torch.cat([digits, sep, digits.flip(1)], dim=1)


def train(model, steps=400, batch_size=64, lr=3e-3):
    """The statement's train() with the missing backward() put back, and without the prints."""
    rng = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        seq = make_batch(batch_size, rng)
        logits = model(seq[:, :-1])
        loss = F.cross_entropy(logits[:, -N_DIGITS:].reshape(-1, VOCAB), seq[:, -N_DIGITS:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


@torch.no_grad()
def generate(model, prompt, n_new):
    """Greedy decoding with full forward passes only (the statement's original generate)."""
    seq = prompt
    for _ in range(n_new):
        seq = torch.cat([seq, model(seq)[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    return seq


@torch.no_grad()
def greedy_problem(model, prompt, out, n_new, tol=1e-4):
    """None if `out` is a greedy decoding of `prompt` under the model's own full forward pass (a token whose logit is
    within `tol` of the maximum also counts, so near ties are not held against the answer); else what is wrong."""
    B, P = prompt.shape
    if not isinstance(out, torch.Tensor):
        return f"generate 应当返回一个 tensor，得到的是 {type(out).__name__}"
    if tuple(out.shape) != (B, P + n_new):
        return f"返回的形状应当是 {(B, P + n_new)}（提示长 {P}，再生成 {n_new} 个），得到 {tuple(out.shape)}"
    if not torch.equal(out[:, :P].long(), prompt):
        return "返回序列的开头不是原来的提示"
    out = out.long()
    logits = model(out[:, :-1])                 # logits[:, t] predicts out[:, t + 1]
    for i in range(n_new):
        lg = logits[:, P - 1 + i]
        chosen = out[:, P + i]
        gap = lg.max(dim=-1).values - lg.gather(1, chosen[:, None]).squeeze(1)
        bad = (gap > tol).nonzero().flatten()
        if len(bad):
            r = int(bad[0])
            return (f"第 {r} 行（下标从 0 开始）生成的第 {i + 1} 个 token 是 {int(chosen[r])}，但对整条序列做前向时，"
                    f"这个位置 logits 最大的是 {int(lg[r].argmax())}，贪心解码应当选它")
    return None
