# expect: WA
# Part 4's backward writes the three contributions with = instead of +=, so later ones overwrite earlier ones.
import torch


def prefix_products_inplace(W):
    P = torch.empty_like(W)
    P[0] = W[0]
    for i in range(1, W.shape[0]):
        P[i] = P[i - 1] @ W[i]      # NOTE: P[i - 1] is a view of P, and this line writes into P
    return P


def prefix_products(W):
    products = [W[0]]
    for i in range(1, W.shape[0]):
        products.append(products[-1] @ W[i])    # every product is a new tensor; nothing is overwritten
    return torch.stack(products)                # stack allocates a new (N, D, D) tensor and is differentiable


def prefix_products_backward(W, P, G):
    dW = torch.empty_like(W)
    carry = torch.zeros_like(W[0])                          # holds A_i
    for i in reversed(range(W.shape[0])):
        carry = carry + G[i]                                # NOTE: a sum, because P[i] feeds the loss AND P[i+1]
        dW[i] = carry if i == 0 else P[i - 1].T @ carry     # NOTE: i == 0 has no left factor
        carry = carry @ W[i].T                              # the part of A_i that flows on to P[i-1]
    return dW


def scan_forward(W):
    """Returns (P, tape). tape[k] = (stride, input of round k), kept for the backward pass."""
    X, tape, stride = W, [], 1
    while stride < W.shape[0]:
        Y = X.clone()                               # NOTE: read from X, write to Y; never update X in place
        Y[stride:] = X[:-stride] @ X[stride:]       # NOTE: lower indices on the LEFT, matmul does not commute
        tape.append((stride, X))                    #       (diagonal test matrices would hide a swapped order)
        X, stride = Y, 2 * stride
    return X, tape


def scan_backward(tape, G):
    H = G
    for stride, X in reversed(tape):
        dX = torch.zeros_like(H)
        dX[:stride] = H[:stride]
        dX[stride:] = X[:-stride].transpose(1, 2) @ H[stride:]
        dX[:-stride] = H[stride:] @ X[stride:].transpose(1, 2)
        H = dX                                                      # NOTE: all three lines are +=, most entries
    return H                                                        #       are used twice in one round
