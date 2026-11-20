# expect: WA
# Gradient check compares the absolute error |numeric - analytic| with 1e-5 instead of the relative error.
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


def make_dataset(seed: int):
    rng = np.random.default_rng(seed)

    def blobs(n_per_class, rng):
        X = np.concatenate([CENTERS[c] + BLOB_STD * rng.standard_normal((n_per_class, 2)) for c in range(N_CLASSES)])
        y = np.repeat(np.arange(N_CLASSES), n_per_class)
        return X, y

    X_train, y_train = blobs(N_PER_CLASS_TRAIN, rng)
    X_val, y_val = blobs(N_PER_CLASS_VAL, rng)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, y_train, (X_val - mean) / std, y_val


def init_params(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        W1=rng.standard_normal((2, HIDDEN)) * np.sqrt(2.0 / 2),
        b1=np.zeros(HIDDEN),
        W2=rng.standard_normal((HIDDEN, N_CLASSES)) * 0.01,
        b2=np.zeros(N_CLASSES),
    )


def forward(params, X, training, rng):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)
    keep_prob = 1.0 - DROPOUT_P
    if training:
        mask = rng.binomial(1, keep_prob, size=h.shape) / keep_prob   #VARIANT_MASK
    else:
        mask = np.ones_like(h)
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    m = logits.max(axis=1, keepdims=True)                              #VARIANT_SOFTMAX
    lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    probs = np.exp(logits - lse)
    cache = dict(X=X, z1=z1, mask=mask, h_drop=h_drop)
    return {"probs": probs, "cache": cache}


def loss_and_grads(params, X, y, training, rng):
    out = forward(params, X, training=training, rng=rng)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = (probs - onehot) * (1.0 / N)
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score(probs, y):
    pred = probs.argmax(axis=-1)
    return float((pred == np.asarray(y).reshape(-1)).mean())


def sgd_momentum_step(params, velocity, grads, lr, momentum):
    new_v = {k: momentum * velocity[k] - lr * grads[k] for k in params}
    new_p = {k: params[k] + new_v[k] for k in params}
    return new_p, new_v


def train(seed: int = 0) -> dict:
    X_train, y_train, X_val, y_val = make_dataset(seed)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1000)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    N = len(X_train)
    train_losses = []
    for epoch in range(EPOCHS):
        idx = np.arange(N)                                             #VARIANT_SHUFFLE
        rng.shuffle(idx)
        X_shuf, y_shuf = X_train[idx], y_train[idx]
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            loss, grads = loss_and_grads(params, Xb, yb, training=True, rng=rng)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
        print(f"epoch {epoch + 1:2d}  training loss {train_losses[-1]:.4f}")
    val_accuracy = accuracy_score(forward(params, X_val, training=False, rng=rng)["probs"], y_val)
    print(f"validation accuracy: {val_accuracy:.3f}")
    return {"train_losses": train_losses, "val_accuracy": val_accuracy}


def sanity_checks(init_params, forward, loss_and_grads, accuracy_score, X, y, n_classes, seed=0):
    g = np.random.default_rng(seed)
    out = {}
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    loss0, _ = loss_and_grads(init_params(seed), X, y, False, g)
    out["initial_loss_near_ln_C"] = bool(abs(loss0 - np.log(n_classes)) < 0.1)

    P = init_params(seed + 11)
    X8, y8 = X[:8], y[:8]
    for _ in range(500):
        _, gr = loss_and_grads(P, X8, y8, False, g)
        for k in P:
            P[k] = P[k] - 0.5 * gr[k]
    lf, _ = loss_and_grads(P, X8, y8, False, g)
    out["overfits_tiny_batch"] = bool(lf < 0.05)

    Q = {k: np.array(v, dtype=np.float64) for k, v in init_params(seed + 22).items()}
    X6, y6 = X[:6], y[:6]
    _, ana = loss_and_grads(Q, X6, y6, False, g)
    ok = True
    for k in Q:
        flat = Q[k].reshape(-1)
        a = np.asarray(ana[k], dtype=np.float64).reshape(-1)
        for i in range(flat.size):
            o = flat[i]
            flat[i] = o + 1e-5
            lp, _ = loss_and_grads(Q, X6, y6, False, g)
            flat[i] = o - 1e-5
            lm, _ = loss_and_grads(Q, X6, y6, False, g)
            flat[i] = o
            num = (lp - lm) / 2e-5
            if abs(num - a[i]) >= 1e-5:
                ok = False
    out["gradients_match_finite_differences"] = ok

    R = init_params(seed + 33)
    full = np.asarray(forward(R, X6, False, g)["probs"])
    single = np.empty_like(full)
    for i in g.permutation(6):
        single[i] = np.asarray(forward(R, X6[i:i + 1], False, g)["probs"])[0]
    out["eval_invariant_to_batching"] = bool(np.max(np.abs(full - single)) <= 1e-10)

    oh = np.eye(n_classes)[y]
    out["accuracy_metric_correct_on_labels"] = bool(accuracy_score(oh, y) == 1.0)
    return out


if __name__ == "__main__":
    train()
