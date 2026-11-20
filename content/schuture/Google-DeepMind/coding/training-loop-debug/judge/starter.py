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
    """Three Gaussian blobs in 2-D, one per class. Returns standardised (X_train, y_train, X_val, y_val)."""
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
    """X: (N, 2). Returns {"probs": (N, 3), "cache": ...}. training selects training- versus eval-time behaviour."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)                                       # ReLU
    keep_prob = 1.0 - DROPOUT_P
    mask = (rng.random(h.shape) < keep_prob) / keep_prob           # inverted dropout
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)
    cache = dict(X=X, z1=z1, mask=mask, h_drop=h_drop)
    return {"probs": probs, "cache": cache}


def loss_and_grads(params: dict, X: np.ndarray, y: np.ndarray, training: bool, rng: np.random.Generator):
    """Mean softmax cross-entropy over the batch, and its gradient with respect to every parameter."""
    out = forward(params, X, training=training, rng=rng)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = probs - onehot
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score(probs: np.ndarray, y: np.ndarray) -> float:
    """probs: (N, 3) predicted class probabilities. y: (N,) true labels. Returns the fraction correct."""
    pred = np.argmax(probs, axis=1, keepdims=True)
    return float(np.mean(pred == y))


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
        X_shuf = X_train[rng.permutation(N)]
        y_shuf = y_train[rng.permutation(N)]
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            velocity = {k: np.zeros_like(v) for k, v in params.items()}
            loss, grads = loss_and_grads(params, Xb, yb, training=True, rng=rng)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
        print(f"epoch {epoch + 1:2d}  training loss {train_losses[-1]:.4f}")
    val_accuracy = accuracy_score(forward(params, X_val, training=False, rng=rng)["probs"], y_val)
    print(f"validation accuracy: {val_accuracy:.3f}")
    return {"train_losses": train_losses, "val_accuracy": val_accuracy}


# Part 3
def sanity_checks(init_params, forward, loss_and_grads, accuracy_score,
                   X: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 0) -> dict:
    """Runs the five checks above against the given pipeline pieces and dataset."""
    raise NotImplementedError


if __name__ == "__main__":
    train()
