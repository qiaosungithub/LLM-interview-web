"""Reference pieces for tests.py.

Part 1 compares the answer's forward / loss_and_grads / accuracy_score with plain reference formulas below.
Part 3 hands the answer's sanity_checks the judge's own pipelines: a correct one and versions with one known bug,
built by `pipeline()`. Their forward pass sums in a fixed order with elementwise operations only, so a row's
probabilities are bit-identical whether it is evaluated alone or inside a batch (an answer may compare them with
==), and their init_params redraws until the network is well conditioned for the gradient check (see `make_init`).
"""
import numpy as np

HIDDEN = 32
DROPOUT_P = 0.2
N_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.15
MOMENTUM = 0.9
N_PER_CLASS_TRAIN = 200
N_PER_CLASS_VAL = 100
CENTERS = np.array([[0.0, 2.0], [-1.7320508, -1.0], [1.7320508, -1.0]])
BLOB_STD = 0.9
NAMES = ("W1", "b1", "W2", "b2")


def make_dataset(seed):
    """The statement's make_dataset, unchanged."""
    rng = np.random.default_rng(seed)

    def blobs(n_per_class, rng):
        X = np.concatenate([CENTERS[c] + BLOB_STD * rng.standard_normal((n_per_class, 2)) for c in range(N_CLASSES)])
        y = np.repeat(np.arange(N_CLASSES), n_per_class)
        return X, y

    X_train, y_train = blobs(N_PER_CLASS_TRAIN, rng)
    X_val, y_val = blobs(N_PER_CLASS_VAL, rng)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, y_train, (X_val - mean) / std, y_val


def init_params(seed):
    """The statement's init_params, unchanged."""
    rng = np.random.default_rng(seed)
    return dict(
        W1=rng.standard_normal((2, HIDDEN)) * np.sqrt(2.0 / 2),
        b1=np.zeros(HIDDEN),
        W2=rng.standard_normal((HIDDEN, N_CLASSES)) * 0.01,
        b2=np.zeros(N_CLASSES),
    )


def interleave(X, y):
    """Reorder a class-sorted dataset as class 0, 1, 2, 0, 1, 2, ... so every prefix of 2 or more mixes labels."""
    idx = np.stack([np.flatnonzero(y == c) for c in range(N_CLASSES)], axis=1).reshape(-1)
    return X[idx].copy(), y[idx].copy()


def separated(X, y, n=8, gap=0.5):
    """True when, among the first n rows, points of different classes are at least `gap` apart."""
    d = np.linalg.norm(X[:n, None, :] - X[None, :n, :], axis=2)
    return bool(d[y[:n, None] != y[None, :n]].min(initial=np.inf) >= gap)


# ---------------------------------------------------------------- Part 1 reference formulas (dropout off)

def probs(params, X):
    """Softmax over classes of the eval-time logits, shifted by the row max."""
    h = np.maximum(X @ params["W1"] + params["b1"], 0.0)
    logits = h @ params["W2"] + params["b2"]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grads(params, X, y):
    """Mean cross-entropy (exact log-sum-exp) and its gradient, dropout off."""
    N = X.shape[0]
    z1 = X @ params["W1"] + params["b1"]
    h = np.maximum(z1, 0.0)
    logits = h @ params["W2"] + params["b2"]
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=1, keepdims=True)
    loss = float(np.mean(np.log(s[:, 0]) + m[:, 0] - logits[np.arange(N), y]))
    d = e / s
    d[np.arange(N), y] -= 1.0
    d /= N
    dh = d @ params["W2"].T
    dz1 = dh * (z1 > 0)
    return loss, dict(W1=X.T @ dz1, b1=dz1.sum(axis=0), W2=h.T @ d, b2=d.sum(axis=0))


# ---------------------------------------------------------------- Part 3 pipelines

_FALLBACK = np.random.default_rng(12345)    # used only if a buggy pipeline is handed rng=None


def _gen(rng):
    return rng if isinstance(rng, np.random.Generator) else _FALLBACK


def _logits_fixed_order(params, X, mask=None):
    """z1, h_drop and logits computed with elementwise operations in a fixed order: each row's result does not
    depend on the other rows of the batch."""
    X = np.asarray(X, dtype=np.float64)
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X[:, 0:1] * W1[0] + X[:, 1:2] * W1[1] + b1
    h = np.maximum(z1, 0.0)
    hd = h if mask is None else h * mask
    # b2 + hd[:, 0] W2[0] + hd[:, 1] W2[1] + ..., added left to right (an accumulate is sequential by definition)
    terms = np.concatenate([np.broadcast_to(b2, (X.shape[0], 1, W2.shape[1])), hd[:, :, None] * W2[None]], axis=1)
    return z1, hd, np.add.accumulate(terms, axis=1)[:, -1, :]


def _softmax_rows(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = np.add.accumulate(e, axis=1)[:, -1:]
    return e / s, m, s


def _draw_mask(rng, shape, drop_all=False):
    if drop_all:
        return np.zeros(shape)
    keep = 1.0 - DROPOUT_P
    return (_gen(rng).random(shape) < keep) / keep


def _backward(params, X, y, z1, hd, mask, d, scale_by_n=True, relu_gate=True):
    N = X.shape[0]
    d = d.copy()
    d[np.arange(N), y] -= 1.0
    if scale_by_n:
        d /= N
    dh = d @ params["W2"].T
    if mask is not None:
        dh = dh * mask
    dz1 = dh * (z1 > 0) if relu_gate else dh
    return dict(W1=np.asarray(X, dtype=np.float64).T @ dz1, b1=dz1.sum(axis=0), W2=hd.T @ d, b2=d.sum(axis=0))


def _loss_pieces(params, X, y, mask=None):
    """The fixed-order pipeline's mean cross-entropy (exact log-sum-exp) with what backprop needs."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    z1, hd, logits = _logits_fixed_order(params, X, mask)
    pr, m, s = _softmax_rows(logits)
    loss = float(np.mean(np.log(s[:, 0]) + m[:, 0] - logits[np.arange(X.shape[0]), y]))
    return loss, z1, hd, pr


def _fd_worst_rel(params, X, y, eps=1e-5):
    """Largest |numeric - analytic| / max(|numeric|, |analytic|) over every parameter entry of the fixed
    pipeline on (X, y), central differences of step eps, counting entries where both are exactly 0 as 0."""
    _, z1, hd, pr = _loss_pieces(params, X, y)
    g = _backward(params, X, y, z1, hd, None, pr)
    p = {k: v.copy() for k, v in params.items()}
    worst = 0.0
    for k in NAMES:
        arr, flat_g = p[k].reshape(-1), g[k].reshape(-1)
        for i in range(arr.size):
            orig = arr[i]
            arr[i] = orig + eps
            lp = _loss_pieces(p, X, y)[0]
            arr[i] = orig - eps
            lm = _loss_pieces(p, X, y)[0]
            arr[i] = orig
            num, ana = (lp - lm) / (2 * eps), flat_g[i]
            if num != ana:
                worst = max(worst, abs(num - ana) / max(abs(num), abs(ana)))
    return worst


def _overfit_final(params, X, y):
    """The loss after 500 steps of plain gradient descent (learning rate 0.5) on (X, y), fixed pipeline."""
    p = {k: v.copy() for k, v in params.items()}
    for _ in range(500):
        _, z1, hd, pr = _loss_pieces(p, X, y)
        g = _backward(p, X, y, z1, hd, None, pr)
        p = {k: p[k] - 0.5 * g[k] for k in p}
    return _loss_pieces(p, X, y)[0]


def good_init(p, X, y, bad=False):
    """The acceptance rule of make_init below."""
    if bad:
        return abs(_loss_pieces(p, X, y)[0] - np.log(N_CLASSES)) > 1.0
    if np.abs(X[:8, 0:1] * p["W1"][0] + X[:8, 1:2] * p["W1"][1] + p["b1"]).min() < 1e-4:
        return False
    if abs(_loss_pieces(p, X, y)[0] - np.log(N_CLASSES)) > 0.03:
        return False
    return _fd_worst_rel(p, X[:6], y[:6]) <= 1e-6 and _overfit_final(p, X[:8], y[:8]) < 0.01


def make_init(X, y, bad=False, tiny=False):
    """init_params(seed) for the given dataset: draws W1 ~ N(0, 1), W2 ~ N(0, 0.01^2), zero biases from
    default_rng(seed) as the statement does, then redraws from the same generator until the draw is well
    conditioned, so that no seed an answer might pick makes a correct check fail by numerical accident. With the
    statement's own init, about 1 seed in 50 gives some gradient entry so small (below 1e-7) that its central
    difference has a relative error above 1e-5 although the gradient is right. A draw is kept when
      - every pre-activation on the first 8 rows is at least 1e-4 from the ReLU kink (a step of 1e-5 times
        |x| <= 4 never crosses it, so a dead unit's finite differences are exactly 0, like its gradient),
      - on the first 6 rows every analytic gradient entry matches its central difference (step 1e-5, computed
        exactly as an answer would, through this pipeline's loss) within relative error 1e-6, ten times below the
        statement's bound, whichever of |numeric|, |analytic|, their max or their sum is the denominator,
      - the loss on the whole dataset is within 0.03 of ln 3 (the check allows 0.1),
      - 500 steps of gradient descent (learning rate 0.5) on the first 8 rows, done exactly as an answer would,
        end below 0.01 (the check allows 0.05).
    tests.py only uses training splits on which at least 8 of 10 raw draws are kept, so a seed costs a draw or two
    of about 60 ms each. With bad=True, W2 is drawn 1000 times larger and the draw
    is kept only if the initial loss is more than 1 away from ln 3. With tiny=True, W2 is drawn 10^4 times smaller
    and the first draw is kept: the initial loss is then within 1e-5 of ln 3 and the W1, b1 gradients are about
    1e-7, so a backward pass that forgets the ReLU derivative is off by less than 1e-5 in absolute terms but by more
    than 0.5 in relative terms (checked offline on every "mixed" split in tests.py)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    cache = {}

    def init(seed):
        key = int(seed)
        if key not in cache:
            rng = np.random.default_rng(key)
            for _ in range(1000):
                p = dict(W1=rng.standard_normal((2, HIDDEN)) * np.sqrt(2.0 / 2), b1=np.zeros(HIDDEN),
                         W2=rng.standard_normal((HIDDEN, N_CLASSES)) * (10.0 if bad else 1e-6 if tiny else 0.01),
                         b2=np.zeros(N_CLASSES))
                if tiny or good_init(p, X, y, bad):
                    break
            else:
                raise RuntimeError("judge: no acceptable initialisation found")
            cache[key] = p
        return {k: v.copy() for k, v in cache[key].items()}

    return init


def pipeline(kind, X, y):
    """(init_params, forward, loss_and_grads, accuracy_score) for the given dataset. kind is "fixed", a single
    bug of the statement left in ("bug2", "bug3", "bug5", "bug6"), "bad_init" (W2 1000x too large), "zero_grad"
    (the backward pass returns zeros), "no_relu_gate" (the backward pass forgets the ReLU derivative),
    "tiny_no_relu_gate" (the same with W2 drawn 10^4 times smaller, so the wrong gradients are tiny in absolute
    terms) or "train_drop_all" (training mode drops every hidden unit; with dropout off it is the fixed pipeline)."""
    init = make_init(X, y, bad=(kind == "bad_init"), tiny=(kind == "tiny_no_relu_gate"))

    def forward(params, X, training, rng):
        X = np.asarray(X, dtype=np.float64)
        if kind == "bug2":
            # bug 2: the softmax max and sum run over the batch axis
            h = np.maximum(X @ params["W1"] + params["b1"], 0.0)
            mask = _draw_mask(rng, h.shape) if training else np.ones_like(h)
            logits = (h * mask) @ params["W2"] + params["b2"]
            e = np.exp(logits - logits.max(axis=0, keepdims=True))
            p = e / e.sum(axis=0, keepdims=True)
            return {"probs": p, "cache": dict(X=X, z1=X @ params["W1"] + params["b1"], mask=mask, h_drop=h * mask)}
        n = X.shape[0]
        if kind == "bug5" or training:
            # bug 5 draws the mask whatever `training` says
            mask = _draw_mask(rng, (n, HIDDEN), drop_all=(kind == "train_drop_all"))
        else:
            mask = None
        z1, hd, logits = _logits_fixed_order(params, X, mask)
        p, _, _ = _softmax_rows(logits)
        return {"probs": p, "cache": dict(X=X, z1=z1, mask=np.ones((n, HIDDEN)) if mask is None else mask, h_drop=hd)}

    def loss_and_grads(params, X, y, training, rng):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = X.shape[0]
        if kind == "bug2":
            out = forward(params, X, training, rng)
            pr, c = out["probs"], out["cache"]
            loss = float(np.mean(-np.log(pr[np.arange(n), y] + 1e-12)))
            return loss, _backward(params, X, y, c["z1"], c["h_drop"], c["mask"], pr)
        mask = _draw_mask(rng, (n, HIDDEN), drop_all=(kind == "train_drop_all")) if (kind == "bug5" or training) else None
        loss, z1, hd, pr = _loss_pieces(params, X, y, mask)
        grads = _backward(params, X, y, z1, hd, mask, pr, scale_by_n=(kind != "bug3"),
                          relu_gate=(kind not in ("no_relu_gate", "tiny_no_relu_gate")))
        if kind == "zero_grad":
            grads = {k: np.zeros_like(v) for k, v in grads.items()}
        return loss, grads

    def accuracy_score(probs, y):
        if kind == "bug6":
            pred = np.argmax(probs, axis=1, keepdims=True)
        else:
            pred = np.argmax(probs, axis=1)
        return float(np.mean(pred == y))

    return init, forward, loss_and_grads, accuracy_score


# The checks a correct sanity_checks must report for each pipeline: True / False, or None where the statement
# leaves the outcome open for that bug (see tests.py for why).
KEYS = ("initial_loss_near_ln_C", "overfits_tiny_batch", "gradients_match_finite_differences",
        "eval_invariant_to_batching", "accuracy_metric_correct_on_labels")
