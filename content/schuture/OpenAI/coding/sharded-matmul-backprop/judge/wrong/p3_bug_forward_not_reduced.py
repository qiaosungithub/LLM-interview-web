# expect: WA
# Wrong for Part 3: layer2_forward still returns the partial products, so Y and every gradient are wrong.
import numpy as np


# Part 2 (the page's reference answer). all_gather and all_reduce are defined once, in the Part 3 file below; the
# page writes the same two functions in both parts.

def column_parallel_forward(X, W_shards):
    return [X @ Wk for Wk in W_shards]              # NOTE: X is replicated; no communication here


def column_parallel_backward(X, W_shards, dY_shards):
    dW_shards = [X.T @ dYk for dYk in dY_shards]                       # local
    dX_partial = [dYk @ Wk.T for dYk, Wk in zip(dY_shards, W_shards)]
    return dW_shards, all_reduce(dX_partial)                           # NOTE: sum over devices


def row_parallel_forward(X_shards, W_shards):
    Y_partial = [Xk @ Wk for Xk, Wk in zip(X_shards, W_shards)]
    return all_reduce(Y_partial)                     # NOTE: forward itself needs a reduction


def row_parallel_backward(X_shards, W_shards, dY):
    dW_shards = [Xk.T @ dY for Xk in X_shards]        # local
    dX_shards = [dY @ Wk.T for Wk in W_shards]        # local
    return dW_shards, dX_shards


# Part 3: the given two-layer sharded MLP; the page's fixed file.

import numpy as np

B, D_IN, D_HIDDEN, D_OUT, N = 5, 4, 7, 3, 3   # hidden width not divisible by the shard count


def all_gather(shards, axis):
    full = np.concatenate(shards, axis=axis)
    return [full for _ in shards]


def all_reduce(shards):
    total = sum(shards)
    return [total for _ in shards]


def shard_columns(A, n):
    return np.array_split(A, n, axis=1)


def shard_rows(A, n):
    return np.array_split(A, n, axis=0)          # bug 4 fix: same split function as shard_columns


def layer1_forward(X, W1_shards):
    Z1_shards = [X @ W1k for W1k in W1_shards]
    H_shards = [np.tanh(Z1k) for Z1k in Z1_shards]
    return Z1_shards, H_shards


def layer1_backward(X, W1_shards, Z1_shards, H_shards, dH_shards):
    dZ1_shards = [dHk * (1 - Hk ** 2) for dHk, Hk in zip(dH_shards, H_shards)]   # bug 1 fix: use H, not Z1
    dW1_shards = [X.T @ dZ1k for dZ1k in dZ1_shards]                            # bug 2 fix: X.T @ dZ1k
    dX_partial = [dZ1k @ W1k.T for dZ1k, W1k in zip(dZ1_shards, W1_shards)]
    dX_shards = all_reduce(dX_partial)      # bug 5 fix: X is replicated, so its gradient is summed
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

    W1_full = all_gather(W1_shards, axis=1)[0]   # W1 was split by columns, so it is gathered along axis=1
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
