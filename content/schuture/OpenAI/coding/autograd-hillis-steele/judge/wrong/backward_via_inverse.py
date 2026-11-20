# expect: RE
# Part 3 vectorizes A_i = (sum_{j>=i} G_j P_j^T) P_i^{-T} with inverses; fails on a singular W_i.
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
    # W_{i+1} ... W_j = P_i^{-1} P_j, so A_i = sum_{j >= i} G_j (W_{i+1} ... W_j)^T = (sum_{j >= i} G_j P_j^T) P_i^{-T}
    S = torch.flip(torch.cumsum(torch.flip(G @ P.transpose(1, 2), [0]), 0), [0])
    A = S @ torch.linalg.inv(P).transpose(1, 2)
    dW = A.clone()
    dW[1:] = P[:-1].transpose(1, 2) @ A[1:]
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
        dX[:stride] += H[:stride]                                   # copied entries
        dX[stride:] += X[:-stride].transpose(1, 2) @ H[stride:]     # X_j as the right factor of Y_j
        dX[:-stride] += H[stride:] @ X[stride:].transpose(1, 2)     # X_j as the left factor of Y_{j+s}
        H = dX                                                      # NOTE: all three lines are +=, most entries
    return H                                                        #       are used twice in one round
