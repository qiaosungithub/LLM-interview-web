# expect: WA
# Wrong for Part 3: adds positional rows 0..T-1 to every chunk instead of rows past..past+T-1.
import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIGITS, SEP = 6, 10                       # tokens 0..9 are digits, token 10 is the separator
VOCAB, MAX_LEN = 11, 2 * N_DIGITS + 1


class KVCache:                              # Part 3: the given class, with update() filled in
    """Keys and values of all positions processed so far, one entry per layer."""

    def __init__(self, n_layer: int):
        self.k = [None] * n_layer       # per layer: (B, n_head, T_past, d_head), or None while empty
        self.v = [None] * n_layer

    @property
    def length(self) -> int:            # T_past
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: (B, n_head, T_new, d_head), the new positions of this layer. Appends them to the cache
        and returns the keys and values of all T_past + T_new positions."""
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)        # NOTE: dim 2 is time in (B, n_head, T, d_head)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head, self.d_head = n_head, d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)                          # bug 1: was nn.Linear(self.d_head, d_model)
        # visible[i, j] is True when the query at position i may attend to the key at position j
        self.register_buffer("visible", torch.ones(max_len, max_len).tril().bool())

    def split_heads(self, x):               # (B, T, d_model) -> (B, n_head, T, d_head)
        return x.unflatten(-1, (self.n_head, self.d_head)).transpose(1, 2)

    def forward(self, x, cache=None, layer=0):
        T = x.shape[1]                                                  # number of new positions
        q, k, v = self.split_heads(self.w_q(x)), self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))
        if cache is not None:
            k, v = cache.update(layer, k, v)                            # k, v now cover positions 0 .. past + T - 1
        past = k.shape[2] - T
        scores = q @ k.transpose(-1, -2) / self.d_head ** 0.5           # (B, n_head, T, past + T)
        # NOTE: new query i sits at absolute position past + i, so take rows past.. of the mask, not rows 0..
        scores = scores.masked_fill(~self.visible[past:past + T, :past + T], float("-inf"))     # bug 3: was -1e-9
        y = scores.softmax(dim=-1) @ v                                  # (B, n_head, T, d_head)
        return self.w_o(y.transpose(1, 2).flatten(-2))


class Block(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, max_len)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, cache=None, layer=0):
        x = x + self.attn(self.ln1(x), cache, layer)                    # pass-through only
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, d_model=64, n_head=4, n_layer=1, max_len=MAX_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)    # bug 4: was randn(1, 1, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, max_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx, cache=None):     # idx: (B, T) token ids -> logits (B, T, vocab)
        T = idx.shape[1]
        past = cache.length if cache is not None else 0                 # NOTE: read before the blocks update it
        x = self.tok_emb(idx) + self.pos_emb[:, :T]
        for layer, block in enumerate(self.blocks):
            x = block(x, cache, layer)
        return self.head(self.ln_f(x))


def make_batch(batch_size, rng):            # each row: N_DIGITS random digits, SEP, the same digits reversed
    digits = torch.randint(0, 10, (batch_size, N_DIGITS), generator=rng)
    sep = torch.full((batch_size, 1), SEP)
    return torch.cat([digits, sep, digits.flip(1)], dim=1)


def train(model, steps=400, batch_size=64, lr=3e-3):
    rng = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        seq = make_batch(batch_size, rng)
        logits = model(seq[:, :-1])                                     # logits[:, t] predicts seq[:, t + 1]
        answer_logits, answer = logits[:, -N_DIGITS:], seq[:, -N_DIGITS:]   # only the reversed digits are scored
        loss = F.cross_entropy(answer_logits.reshape(-1, VOCAB), answer.reshape(-1))
        opt.zero_grad()
        loss.backward()                                                 # bug 2: this line was missing
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:3d}  loss {loss.item():.4f}")
    return loss.item()


@torch.no_grad()
def generate(model, prompt, n_new, use_cache=False):
    cache = KVCache(len(model.blocks)) if use_cache else None
    seq, new = prompt, prompt                                           # the first call processes the whole prompt
    for _ in range(n_new):
        logits = model(new, cache) if use_cache else model(seq)
        new = logits[:, -1].argmax(dim=-1, keepdim=True)                # NOTE: later calls feed only this one token
        seq = torch.cat([seq, new], dim=1)
    return seq


@torch.no_grad()
def run_tests(model, final_loss):
    seq = make_batch(500, torch.Generator().manual_seed(1))
    a, b = seq[:, :-1].clone(), seq[:, :-1].clone()
    b[:, -1] = (b[:, -1] + 1) % 10                                      # a and b differ in the last token only
    early_a, early_b = model(a)[:, :-1], model(b)[:, :-1]               # logits at all positions before the change
    assert torch.allclose(early_a, early_b, atol=1e-5), "test 1: a later token changed earlier logits"
    assert final_loss < 0.05, "test 2: the loss did not converge"
    completed = generate(model, seq[:, :N_DIGITS + 1], N_DIGITS)
    accuracy = (completed == seq).all(dim=1).float().mean().item()
    assert accuracy > 0.99, f"test 3: only {accuracy:.1%} of the generated answers are right"


# Part 2: the parity classifier (a wrapper that leaves TinyGPT unchanged)
class ParityClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.body = TinyGPT(**kwargs)
        d_model = self.body.head.in_features
        self.body.head = nn.Identity()              # drop the vocabulary head: body(idx) now returns hidden states
        self.cls = nn.Linear(d_model, 2)            # class 0 = even, class 1 = odd

    def forward(self, idx):                         # idx: (B, T) digits -> logits (B, 2)
        hidden = self.body(idx)                     # (B, T, d_model)
        return self.cls(hidden.mean(dim=1))         # NOTE: average over time (dim=1), not over features (dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyGPT()
    final_loss = train(model)
    run_tests(model, final_loss)
    print("all tests passed")
