"""Helpers for tests.py: the prefix products as a plain loop, their gradient by autograd, and the variants that the
tests use to say what a wrong answer seems to do. Everything here works on float64 tensors."""
import torch


def prefix(W):
    """P[i] = W[0] @ ... @ W[i], one product at a time."""
    out = [W[0]]
    for i in range(1, W.shape[0]):
        out.append(out[-1] @ W[i])
    return torch.stack(out)


def grad(W, G):
    """dL/dW for L = sum(prefix(W) * G), by autograd on the loop above."""
    Wv = W.detach().clone().requires_grad_(True)
    (g,) = torch.autograd.grad((prefix(Wv) * G).sum(), Wv)
    return g


# ---------------------------------------------------------------- forward variants
def prefix_reversed(W):
    """Q[i] = W[i] @ ... @ W[0]: each new matrix multiplied on the left."""
    out = [W[0]]
    for i in range(1, W.shape[0]):
        out.append(W[i] @ out[-1])
    return torch.stack(out)


def scan_rounds(W, rounds):
    """The Hillis-Steele forward pass stopped after `rounds` rounds."""
    X, stride = W, 1
    for _ in range(rounds):
        Y = X.clone()
        Y[stride:] = X[:-stride] @ X[stride:]
        X, stride = Y, 2 * stride
    return X


def n_rounds(N):
    """ceil(log2 N): the number of strides 1, 2, 4, ... below N."""
    return (N - 1).bit_length()


# ---------------------------------------------------------------- Part 3 variants
def backward_no_accumulation(W, P, G):
    """dW[i] = P[i-1]^T G[i]: only the direct use of P[i] by the loss."""
    dW = G.clone()
    dW[1:] = P[:-1].transpose(1, 2) @ G[1:]
    return dW


def backward_w0_wraps(W, P, G):
    """The right answer except dW[0] = P[-1]^T A_0 (P[i - 1] with i = 0 is the last prefix product)."""
    dW = grad(W, G)
    dW[0] = P[-1].T @ dW[0]
    return dW


def backward_no_transpose(W, P, G):
    """A_i = G_i + A_{i+1} W_{i+1} (W not transposed)."""
    N = W.shape[0]
    dW = torch.empty_like(W)
    carry = torch.zeros_like(W[0])
    for i in reversed(range(N)):
        carry = carry + G[i]
        dW[i] = carry if i == 0 else P[i - 1].T @ carry
        carry = carry @ W[i]
    return dW


# ---------------------------------------------------------------- Part 4 backward variants
def scan_backward_variant(W, G, assign=False, final_x=False):
    """The scan's backward pass with a mistake. assign=True: the three contributions are written with = instead of
    += (each one overwrites the earlier ones where they overlap). final_x=True: every round uses the final P instead
    of the round's input, as when the saved tensors were later overwritten in place."""
    N = W.shape[0]
    tape, X, stride = [], W, 1
    while stride < N:
        Y = X.clone()
        Y[stride:] = X[:-stride] @ X[stride:]
        tape.append((stride, X))
        X, stride = Y, 2 * stride
    P = X
    H = G
    for stride, X in reversed(tape):
        if final_x:
            X = P
        dX = torch.zeros_like(H)
        if assign:
            dX[:stride] = H[:stride]
            dX[stride:] = X[:-stride].transpose(1, 2) @ H[stride:]
            dX[:-stride] = H[stride:] @ X[stride:].transpose(1, 2)
        else:
            dX[:stride] += H[:stride]
            dX[stride:] += X[:-stride].transpose(1, 2) @ H[stride:]
            dX[:-stride] += H[stride:] @ X[stride:].transpose(1, 2)
        H = dX
    return H
