# expect: WA
# Fixes bug 4 only half way: the velocity is reset at the start of every epoch.
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


def forward(params: dict, X: np.ndarray, training: bool, rng: np.random.Generator) -> dict:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)
    if training:
        keep_prob = 1.0 - DROPOUT_P
        mask = (rng.random(h.shape) < keep_prob) / keep_prob   # NOTE: inverted dropout, drawn only while training
    else:
        mask = np.ones_like(h)                                  # NOTE: identity at eval -- no masking, no rescaling
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    shifted = logits - logits.max(axis=1, keepdims=True)        # NOTE: max over classes (axis=1), not the batch
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)                 # NOTE: likewise sum over classes -- rows sum to 1
    cache = dict(X=X, z1=z1, mask=mask, h_drop=h_drop)
    return {"probs": probs, "cache": cache}


def loss_and_grads(params: dict, X: np.ndarray, y: np.ndarray, training: bool, rng: np.random.Generator):
    out = forward(params, X, training=training, rng=rng)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = (probs - onehot) / N       # NOTE: /N mirrors the mean in `loss` -- omitting it scales every
                                          #       gradient (and so the effective learning rate) by N
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score(probs: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)      # NOTE: no keepdims -- shape (N,), matching y, so pred == y is elementwise
    return float(np.mean(pred == y))


def sgd_momentum_step(params, velocity, grads, lr, momentum):
    new_v = {k: momentum * velocity[k] - lr * grads[k] for k in params}
    new_p = {k: params[k] + new_v[k] for k in params}
    return new_p, new_v


def train(seed: int = 0) -> dict:
    X_train, y_train, X_val, y_val = make_dataset(seed)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1000)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}   # NOTE: created once, never reset inside the loop
    N = len(X_train)
    train_losses = []
    for epoch in range(EPOCHS):
        velocity = {k: np.zeros_like(v) for k, v in params.items()}
        perm = rng.permutation(N)
        X_shuf, y_shuf = X_train[perm], y_train[perm]              # NOTE: one shared permutation for X and y
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            loss, grads = loss_and_grads(params, Xb, yb, training=True, rng=rng)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
    val_accuracy = accuracy_score(forward(params, X_val, training=False, rng=rng)["probs"], y_val)
    return {"train_losses": train_losses, "val_accuracy": val_accuracy}


def sanity_checks(init_params, forward, loss_and_grads, accuracy_score, X, y, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    results = {}

    # a freshly initialised net should be no more sure of anything than a uniform guess over the classes
    params = init_params(seed)
    loss0, _ = loss_and_grads(params, X, y, training=False, rng=rng)
    results["initial_loss_near_ln_C"] = bool(abs(loss0 - np.log(n_classes)) < 0.1)

    # backprop should be able to memorise a handful of examples almost exactly
    Xb, yb = X[:8], y[:8]
    p = init_params(seed + 1)
    for _ in range(500):
        _, grads = loss_and_grads(p, Xb, yb, training=False, rng=rng)
        p = {k: p[k] - 0.5 * grads[k] for k in p}
    final_loss, _ = loss_and_grads(p, Xb, yb, training=False, rng=rng)
    results["overfits_tiny_batch"] = bool(final_loss < 0.05)

    # the analytic gradient must match a central finite difference, parameter by parameter
    p64 = {k: v.astype(np.float64) for k, v in init_params(seed + 2).items()}
    Xg, yg = X[:6].astype(np.float64), y[:6]
    _, analytic = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
    eps = 1e-5
    worst_rel_err = 0.0
    for name, arr in p64.items():
        grad = analytic[name]
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            loss_plus, _ = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
            arr[idx] = orig - eps
            loss_minus, _ = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
            arr[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            denom = max(abs(numeric), abs(grad[idx]), 1e-8)
            worst_rel_err = max(worst_rel_err, abs(numeric - grad[idx]) / denom)
    results["gradients_match_finite_differences"] = bool(worst_rel_err < 1e-5)

    # a prediction must not depend on which other rows happen to share its batch
    params_eval = init_params(seed + 3)
    X_sub = X[:6]
    probs_batch = forward(params_eval, X_sub, training=False, rng=rng)["probs"]
    order = np.random.default_rng(seed + 4).permutation(len(X_sub))
    probs_singleton = np.zeros_like(probs_batch)
    for i in order:
        probs_singleton[i] = forward(params_eval, X_sub[i:i + 1], training=False, rng=rng)["probs"][0]
    results["eval_invariant_to_batching"] = bool(np.allclose(probs_batch, probs_singleton, atol=1e-8))

    # the metric itself must report perfect accuracy when handed perfect predictions
    onehot = np.zeros((len(y), n_classes))
    onehot[np.arange(len(y)), y] = 1.0
    results["accuracy_metric_correct_on_labels"] = bool(accuracy_score(onehot, y) == 1.0)

    return results
