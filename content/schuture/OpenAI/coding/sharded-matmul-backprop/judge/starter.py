import numpy as np


# Part 2: the sharded linear layers. all_gather and all_reduce serve both parts; they are defined once, in the
# Part 3 file below, which is where the statement gives them.

def column_parallel_forward(X, W_shards: list) -> list:
    """X: (B, d_in), a full copy on every device. W_shards[k]: (d_in, d_out_k).
    Returns Y_shards[k]: (B, d_out_k), device k's own shard of Y (not gathered)."""
    raise NotImplementedError


def column_parallel_backward(X, W_shards: list, dY_shards: list) -> tuple:
    """dY_shards[k]: (B, d_out_k), dL/dY restricted to device k's columns.
    Returns (dW_shards, dX_shards); dX_shards[k]: (B, d_in), the full dL/dX, identical on every device."""
    raise NotImplementedError


def row_parallel_forward(X_shards: list, W_shards: list) -> list:
    """X_shards[k]: (B, d_in_k). W_shards[k]: (d_in_k, d_out).
    Returns Y_shards[k]: (B, d_out), the full Y, identical on every device."""
    raise NotImplementedError


def row_parallel_backward(X_shards: list, W_shards: list, dY) -> tuple:
    """dY: (B, d_out), identical on every device. Returns (dW_shards, dX_shards)."""
    raise NotImplementedError


# Part 3: the given two-layer sharded MLP; find and fix its 5 bugs.

import numpy as np

B, D_IN, D_HIDDEN, D_OUT, N = 5, 4, 7, 3, 3   # NOTE: D_HIDDEN is not divisible by N


def all_gather(shards, axis):
    full = np.concatenate(shards, axis=axis)
    return [full for _ in shards]


def all_reduce(shards):
    total = sum(shards)
    return [total for _ in shards]


def shard_columns(A, n):
    return np.array_split(A, n, axis=1)


def shard_rows(A, n):
    chunk = A.shape[0] // n
    return [A[i * chunk:(i + 1) * chunk] for i in range(n)]


def layer1_forward(X, W1_shards):
    Z1_shards = [X @ W1k for W1k in W1_shards]
    H_shards = [np.tanh(Z1k) for Z1k in Z1_shards]
    return Z1_shards, H_shards


def layer1_backward(X, W1_shards, Z1_shards, H_shards, dH_shards):
    dZ1_shards = [dHk * (1 - Z1k ** 2) for dHk, Z1k in zip(dH_shards, Z1_shards)]
    dW1_shards = [dZ1k.T @ X for dZ1k in dZ1_shards]
    dX_shards = [dZ1k @ W1k.T for dZ1k, W1k in zip(dZ1_shards, W1_shards)]
    return dW1_shards, dX_shards


def layer2_forward(H_shards, W2_shards):
    Y_partial = [Hk @ W2k for Hk, W2k in zip(H_shards, W2_shards)]
    return Y_partial


def layer2_backward(H_shards, W2_shards, dY):
    dW2_shards = [Hk.T @ dY for Hk in H_shards]
    dH_shards = [dY @ W2k.T for W2k in W2_shards]
    return dW2_shards, dH_shards


def mse_loss_and_grad(Y, target):
    diff = Y - target
    loss = np.mean(diff ** 2)
    dY = 2 * diff / diff.size
    return loss, dY


def check_gradients(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((B, D_IN))
    W1 = rng.standard_normal((D_IN, D_HIDDEN)) * 0.5
    W2 = rng.standard_normal((D_HIDDEN, D_OUT)) * 0.5
    target = rng.standard_normal((B, D_OUT))
    W1_shards = shard_columns(W1, N)
    W2_shards = shard_rows(W2, N)

    Z1_shards, H_shards = layer1_forward(X, W1_shards)
    Y_shards = layer2_forward(H_shards, W2_shards)
    Y = Y_shards[0]
    loss, dY = mse_loss_and_grad(Y, target)
    dW2_shards, dH_shards = layer2_backward(H_shards, W2_shards, dY)
    dW1_shards, dX_shards = layer1_backward(X, W1_shards, Z1_shards, H_shards, dH_shards)

    W1_full = all_gather(W1_shards, axis=1)[0]
    W2_full = all_gather(W2_shards, axis=0)[0]
    Z1 = X @ W1_full
    H = np.tanh(Z1)
    Y_ref = H @ W2_full
    loss_ref, dY_ref = mse_loss_and_grad(Y_ref, target)
    dW2_ref = H.T @ dY_ref
    dH_ref = dY_ref @ W2_full.T
    dZ1_ref = dH_ref * (1 - H ** 2)
    dW1_ref = X.T @ dZ1_ref
    dX_ref = dZ1_ref @ W1_full.T

    return {
        "Y": np.abs(Y - Y_ref).max(),
        "dW1": np.abs(np.concatenate(dW1_shards, axis=1) - dW1_ref).max(),
        "dW2": np.abs(np.concatenate(dW2_shards, axis=0) - dW2_ref).max(),
        "dX": np.abs(dX_shards[0] - dX_ref).max(),
    }


if __name__ == "__main__":
    print(check_gradients())
