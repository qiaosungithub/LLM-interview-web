# expect: WA
# Wrong for Part 3: layer2_forward all-reduces only device 0's entry and leaves the other devices with their partial products; check_gradients reads Y_shards[0] and passes.
import numpy as np


def column_parallel_forward(X, W_shards: list) -> list:
    return [np.matmul(X, Wk) for Wk in W_shards]


def column_parallel_backward(X, W_shards: list, dY_shards: list) -> tuple:
    dW = [np.matmul(X.T, g) for g in dY_shards]
    contributions = [np.matmul(g, Wk.T) for g, Wk in zip(dY_shards, W_shards)]
    return dW, all_reduce(contributions)


def row_parallel_forward(X_shards: list, W_shards: list) -> list:
    return all_reduce([np.matmul(x, w) for x, w in zip(X_shards, W_shards)])


def row_parallel_backward(X_shards: list, W_shards: list, dY) -> tuple:
    return [x.T @ dY for x in X_shards], [dY @ w.T for w in W_shards]


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
    q, r = divmod(A.shape[0], n)
    out, start = [], 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        out.append(A[start:start + size])
        start += size
    return out


def layer1_forward(X, W1_shards):
    Z1_shards = [X @ W1k for W1k in W1_shards]
    H_shards = [np.tanh(Z1k) for Z1k in Z1_shards]
    return Z1_shards, H_shards


def layer1_backward(X, W1_shards, Z1_shards, H_shards, dH_shards):
    dZ1_shards = [dHk * (1 - np.tanh(Z1k) ** 2) for dHk, Z1k in zip(dH_shards, Z1_shards)]
    dW1_shards = [X.T @ dZ1k for dZ1k in dZ1_shards]
    dX_shards = all_reduce([dZ1k @ W1k.T for dZ1k, W1k in zip(dZ1_shards, W1_shards)])
    return dW1_shards, dX_shards


def layer2_forward(H_shards, W2_shards):
    Y_partial = [Hk @ W2k for Hk, W2k in zip(H_shards, W2_shards)]
    return all_reduce(Y_partial)[:1] + Y_partial[1:]


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
