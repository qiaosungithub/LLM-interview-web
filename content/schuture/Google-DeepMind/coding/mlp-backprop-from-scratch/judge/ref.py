"""Independent restatement of the problem, used by tests.py to compute expected values and build inputs."""
import numpy as np

NAMES = ("W1", "b1", "W2", "b2")


def copy_params(params):
    return {k: np.array(params[k], dtype=np.float64, copy=True) for k in NAMES}


def forward(params, X):
    Z1 = X @ params["W1"] + params["b1"]
    H = np.maximum(Z1, 0.0)
    S = H @ params["W2"] + params["b2"]
    return Z1, H, S


def log_softmax(S):
    shifted = S - S.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def forward_backward(params, X, y):
    """Exact mean softmax cross-entropy (via log-sum-exp) and its gradients."""
    B = X.shape[0]
    Z1, H, S = forward(params, X)
    logp = log_softmax(S)
    rows = np.arange(B)
    loss = float(-logp[rows, y].mean())
    dS = np.exp(logp)
    dS[rows, y] -= 1.0
    dS /= B
    dZ1 = (dS @ params["W2"].T) * (Z1 > 0)
    return loss, {"W1": X.T @ dZ1, "b1": dZ1.sum(axis=0), "W2": H.T @ dS, "b2": dS.sum(axis=0)}


def label_probs(params, X, y):
    _, _, S = forward(params, X)
    return np.exp(log_softmax(S)[np.arange(X.shape[0]), y])


def grad_check(fb, params, X, y, eps):
    """The statement's definition, word for word, against the gradient function fb."""
    params = copy_params(params)
    _, analytic = fb(params, X, y)
    worst = 0.0
    for name in NAMES:
        arr = params[name]
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            plus, _ = fb(params, X, y)
            arr[idx] = original - eps
            minus, _ = fb(params, X, y)
            arr[idx] = original
            g_n = (plus - minus) / (2 * eps)
            g_a = analytic[name][idx]
            worst = max(worst, abs(g_a - g_n) / max(abs(g_a) + abs(g_n), 1e-12))
    return worst


def numeric_grads(fb, params, X, y, eps=1e-5):
    """Central differences of the loss reported by fb, entry by entry, on a private copy of params."""
    params = copy_params(params)
    out = {}
    for name in NAMES:
        arr = params[name]
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            original = arr[idx]
            arr[idx] = original + eps
            plus = float(fb(params, X, y)[0])
            arr[idx] = original - eps
            minus = float(fb(params, X, y)[0])
            arr[idx] = original
            g[idx] = (plus - minus) / (2 * eps)
        out[name] = g
    return out


def with_bug(name, idx, factor):
    """forward_backward whose analytic gradient has entry `idx` of grads[name] multiplied by `factor`
    (idx None: the whole array). The loss is left exact."""
    def fb(params, X, y):
        loss, grads = forward_backward(params, X, y)
        g = grads[name].copy()
        if idx is None:
            g = g * factor
        else:
            g[idx] = g[idx] * factor
        grads[name] = g
        return loss, grads
    return fb


def train_same_example(params, x, label, steps, lr, momentum):
    """train() on a data set whose rows are all the same example: every mini-batch, however it is drawn,
    has the gradient of that one example, so the result does not depend on the random draws."""
    params = copy_params(params)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    X, y = np.array([x], dtype=np.float64), np.array([label])
    for _ in range(steps):
        _, grads = forward_backward(params, X, y)
        for k in NAMES:
            velocity[k] = momentum * velocity[k] - lr * grads[k]
            params[k] = params[k] + velocity[k]
    return params


def accuracy(params, X, y):
    _, _, S = forward(params, X)
    return float(np.mean(np.argmax(S, axis=1) == y))


def make_spirals(n_per_class, n_turns, noise, seed):
    rng = np.random.default_rng(seed)
    T = n_turns * 2 * np.pi
    t = T * np.sqrt(rng.random(n_per_class))
    r = t / T
    arm = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    X = np.concatenate([arm, -arm]) + rng.normal(scale=noise, size=(2 * n_per_class, 2))
    return X, np.repeat([0, 1], n_per_class)


def he_init(d, m, C, seed):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(scale=np.sqrt(2.0 / d), size=(d, m)), "b1": np.zeros(m),
            "W2": rng.normal(scale=np.sqrt(2.0 / m), size=(m, C)), "b2": np.zeros(C)}


# ---------------------------------------------------------------- inputs

def random_net(gen, d, m, C, B, scale=0.5, margin=0.05, min_prob=1e-2, min_grad=0.0, live=False, shift=0.0):
    """A random network and batch with entries rounded to 2 decimals (so they print exactly), redrawn until
    every pre-activation is at least `margin` away from the ReLU kink, every label has probability at least
    `min_prob`, every gradient entry is exactly 0 or at least `min_grad` in size, and (live=True) every
    hidden unit is active on some example. `shift` is added to b2 afterwards (softmax ignores it)."""
    for _ in range(10_000):
        p = {"W1": gen.normal(scale=scale, size=(d, m)), "b1": gen.normal(scale=scale, size=m),
             "W2": gen.normal(scale=scale, size=(m, C)), "b2": gen.normal(scale=scale, size=C)}
        p = {k: np.round(v, 2) + 0.0 for k, v in p.items()}
        X = np.round(gen.normal(size=(B, d)), 2) + 0.0
        y = gen.integers(0, C, size=B)
        Z1, _, _ = forward(p, X)
        if np.abs(Z1).min() < margin or label_probs(p, X, y).min() < min_prob:
            continue
        if live and not (Z1 > 0).any(axis=0).all():
            continue
        if min_grad:
            _, g = forward_backward(p, X, y)
            if any(((np.abs(v) < min_grad) & (v != 0)).any() for v in g.values()):
                continue
        p["b2"] = p["b2"] + shift
        return p, X, y
    raise RuntimeError("random_net: no draw met the conditions")


def literal(a):
    a = np.asarray(a)
    return f"np.array({a.tolist()!r})"


def setup_code(params, X, y):
    """Python that rebuilds these inputs, for the 'call' shown with a failure (pasteable into 自定义运行)."""
    body = ",\n          ".join(f'"{k}": {literal(params[k])}' for k in NAMES)
    return f"params = {{{body}}}\nX = {literal(X)}\ny = {literal(y)}"


def bug_code(name, idx, factor):
    """Python that redefines forward_backward the way with_bug() does, for the 'call' of a Part 2 failure."""
    where = f"[{', '.join(map(str, idx))}]" if idx is not None else ""
    op = f'g["{name}"]{where} = g["{name}"]{where} * {factor!r}' if idx is not None else f'g["{name}"] = g["{name}"] * {factor!r}'
    return ("_correct_fb = forward_backward          # 你的 forward_backward（假定它是对的）\n"
            "def forward_backward(params, X, y):     # 损失不变，只把一项解析梯度改错\n"
            "    loss, g = _correct_fb(params, X, y)\n"
            f'    g = dict(g); g["{name}"] = np.array(g["{name}"], dtype=float)\n'
            f"    {op}\n"
            "    return loss, g")
