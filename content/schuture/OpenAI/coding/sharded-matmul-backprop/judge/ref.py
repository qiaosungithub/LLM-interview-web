"""Independent restatement of the problem, used by tests.py for expected values and inputs.

Shards are cut by hand from the statement's rule (contiguous slices; the first L % n shards get one extra entry), and
every expected value is the unsharded float64 product it should equal: Y = X W, dW = X^T dY, dX = dY W^T, and for the
Part 3 MLP the chain Z1 = X W1, H = tanh(Z1), Y = H W2 with a mean-squared-error loss.

Inputs come from np.random.default_rng(seed) by a fixed recipe, and each recipe has a one-line text form, so that a
failing call can be pasted into 自定义运行 as it is.
"""
import numpy as np


def sizes(length, n):
    """Shard sizes along an axis of the given length: the remainder goes one apiece to the first shards."""
    return [length // n + (1 if k < length % n else 0) for k in range(n)]


def split(A, n, axis):
    """A cut into n contiguous shards along axis, by the statement's rule."""
    out, start = [], 0
    for size in sizes(A.shape[axis], n):
        index = [slice(None)] * A.ndim
        index[axis] = slice(start, start + size)
        out.append(A[tuple(index)].copy())
        start += size
    return out


def abs_product(A, B):
    """max over entries of |A| @ |B|: the size of the terms in A @ B, which bounds its rounding error."""
    return float((np.abs(A) @ np.abs(B)).max())


# ---------------------------------------------------------------- Part 2 inputs

def linear_inputs(seed, B, d_in, d_out, a):
    """X (B, d_in), W (d_in, d_out) and the upstream gradient G (B, d_out), uniform in [-a, a]."""
    g = np.random.default_rng(seed)
    X = g.uniform(-a, a, (B, d_in))
    W = g.uniform(-a, a, (d_in, d_out))
    G = g.uniform(-a, a, (B, d_out))
    return X, W, G


def linear_recipe(seed, B, d_in, d_out, a):
    return (f"g = np.random.default_rng({seed}); X = g.uniform(-{a}, {a}, ({B}, {d_in})); "
            f"W = g.uniform(-{a}, {a}, ({d_in}, {d_out})); G = g.uniform(-{a}, {a}, ({B}, {d_out}))")


def full_input(seed, shape, a):
    """One array uniform in [-a, a], to be cut into shards for all_gather."""
    return np.random.default_rng(seed).uniform(-a, a, shape)


def full_recipe(seed, shape, a):
    return f"g = np.random.default_rng({seed}); full = g.uniform(-{a}, {a}, {shape})"


def summand_inputs(seed, n, shape, a):
    """n arrays of one shape, uniform in [-a, a], for all_reduce."""
    g = np.random.default_rng(seed)
    return [g.uniform(-a, a, shape) for _ in range(n)]


def summand_recipe(seed, n, shape, a):
    return f"g = np.random.default_rng({seed}); shards = [g.uniform(-{a}, {a}, {shape}) for _ in range({n})]"


# ---------------------------------------------------------------- Part 3

def mlp_inputs(seed, B, d_in, d_hidden, d_out):
    """Inputs drawn the way check_gradients draws them, then an upstream gradient dH for layer1_backward."""
    g = np.random.default_rng(seed)
    X = g.standard_normal((B, d_in))
    W1 = g.standard_normal((d_in, d_hidden)) * 0.5
    W2 = g.standard_normal((d_hidden, d_out)) * 0.5
    target = g.standard_normal((B, d_out))
    dH = g.standard_normal((B, d_hidden))
    return X, W1, W2, target, dH


def mlp_recipe(seed, B, d_in, d_hidden, d_out):
    return (f"g = np.random.default_rng({seed}); X = g.standard_normal(({B}, {d_in})); "
            f"W1 = g.standard_normal(({d_in}, {d_hidden})) * 0.5; W2 = g.standard_normal(({d_hidden}, {d_out})) * 0.5; "
            f"target = g.standard_normal(({B}, {d_out})); dH = g.standard_normal(({B}, {d_hidden}))")


def mlp(X, W1, W2, target):
    """The unsharded two-layer MLP: a dict of every intermediate and gradient."""
    Z1 = X @ W1
    H = np.tanh(Z1)
    Y = H @ W2
    diff = Y - target
    dY = 2 * diff / diff.size
    dZ1 = (dY @ W2.T) * (1 - H ** 2)
    return {"Z1": Z1, "H": H, "Y": Y, "loss": float(np.mean(diff ** 2)), "dY": dY, "dW2": H.T @ dY,
            "dW1": X.T @ dZ1, "dX": dZ1 @ W1.T}


def layer1_grads(X, W1, dH, derivative_from_z=False):
    """(dW1, dX) of H = tanh(X W1) for the upstream gradient dH. derivative_from_z=True uses 1 - Z1^2 in place of
    1 - H^2, the activation mistake the page discusses; it is used only to name that mistake in a message."""
    Z1 = X @ W1
    dZ1 = dH * (1 - (Z1 if derivative_from_z else np.tanh(Z1)) ** 2)
    return X.T @ dZ1, dZ1 @ W1.T
