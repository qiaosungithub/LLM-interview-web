# expect: WA
# Wrong: bias gradients are not summed over the batch axis, so db1 is (B, m) and db2 is (B, C).
import numpy as np


def _forward(params: dict[str, np.ndarray], X: np.ndarray) -> dict[str, np.ndarray]:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = X @ W1 + b1
    H = np.maximum(Z1, 0.0)
    S = H @ W2 + b2
    row_max = S.max(axis=1, keepdims=True)          # NOTE: subtract the row max before exp -- keeps every
    exp = np.exp(S - row_max)                        #       exponent <= 0 (no overflow); softmax is unchanged,
    probs = exp / exp.sum(axis=1, keepdims=True)      #       since a per-row constant cancels in the ratio
    return {"Z1": Z1, "H": H, "S": S, "probs": probs}


def forward_backward(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    B = X.shape[0]
    cache = _forward(params, X)
    probs, H, Z1 = cache["probs"], cache["H"], cache["Z1"]
    loss = float(np.mean(-np.log(probs[np.arange(B), y] + 1e-12)))   # NOTE: +1e-12 only guards log(0);
                                                                        #       it does not appear in dS below
    onehot = np.zeros_like(probs)
    onehot[np.arange(B), y] = 1.0
    dS = (probs - onehot) / B          # NOTE: divide by B exactly once, here -- every gradient below inherits
                                        #       this factor through the chain rule; do not divide by B again
    dW2 = H.T @ dS
    db2 = dS
    dH = dS @ params["W2"].T
    dZ1 = dH * (Z1 > 0)                # NOTE: the gate uses the PRE-activation Z1, not H -- using dH directly,
                                        #       without this factor, silently makes ReLU the identity here
    dW1 = X.T @ dZ1
    db1 = dZ1                          # NOTE: bias gradients sum over the batch axis, giving shape (m,), not (B, m)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def grad_check(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    for name, arr in params.items():
        if arr.dtype != np.float64:                            # NOTE: float32 rounding error swamps the
            raise ValueError(f"grad_check requires float64 parameters, got {arr.dtype} for {name!r}")  # eps**2 truncation term derived above -- see the checks for a direct demonstration
    if np.any(np.abs(_forward(params, X)["Z1"]) < eps):          # NOTE: a pre-activation within eps of the
        raise ValueError("a hidden pre-activation lies within eps of the ReLU kink; redraw the inputs")  # kink makes the two-sided difference straddle it -- redraw rather than report a misleading error
    _, analytic = forward_backward(params, X, y)
    worst = 0.0
    for name, arr in params.items():
        grad_a = analytic[name]
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = arr[idx]
            arr[idx] = original + eps
            loss_plus, _ = forward_backward(params, X, y)
            arr[idx] = original - eps
            loss_minus, _ = forward_backward(params, X, y)
            arr[idx] = original                                  # NOTE: restore before moving to the next entry
            grad_n = (loss_plus - loss_minus) / (2 * eps)
            denom = max(abs(grad_a[idx]) + abs(grad_n), 1e-12)
            worst = max(worst, abs(grad_a[idx] - grad_n) / denom)
    return worst


def he_init(d: int, m: int, C: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(scale=np.sqrt(2.0 / d), size=(d, m)),      # fan_in = d
        "b1": np.zeros(m),
        "W2": rng.normal(scale=np.sqrt(2.0 / m), size=(m, C)),      # fan_in = m
        "b2": np.zeros(C),
    }


def zero_init(d: int, m: int, C: int) -> dict[str, np.ndarray]:
    return {"W1": np.zeros((d, m)), "b1": np.zeros(m), "W2": np.zeros((m, C)), "b2": np.zeros(C)}


def make_spirals(n_per_class: int, n_turns: float, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.random(n_per_class)) * n_turns * 2 * np.pi    # sqrt: points evenly spaced along the arm, not crowded at the centre
    r = t / (n_turns * 2 * np.pi)                                   # r in [0, 1], grows with the angle
    arm0 = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    arm1 = np.stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)], axis=1)   # the same arm, rotated by pi
    X = np.concatenate([arm0, arm1]) + rng.normal(scale=noise, size=(2 * n_per_class, 2))
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y


def train(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, *, steps: int, batch_size: int,
          lr: float, momentum: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = {k: v.copy() for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    N = X.shape[0]
    for _ in range(steps):
        idx = rng.integers(0, N, size=batch_size)     # NOTE: with replacement -- the simplest correct mini-batching
        _, grads = forward_backward(params, X[idx], y[idx])
        for k in params:
            velocity[k] = momentum * velocity[k] - lr * grads[k]
            params[k] = params[k] + velocity[k]
    return params


def accuracy(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(_forward(params, X)["probs"], axis=1)
    return float(np.mean(pred == y))
