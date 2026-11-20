# expect: WA
# Part 2 silences the in-place error with detach(), which cuts every W[i], i >= 1, out of the graph.
import torch


def prefix_products_inplace(W):
    P = torch.empty_like(W)
    P[0] = W[0]
    for i in range(1, W.shape[0]):
        P[i] = P[i - 1] @ W[i]      # NOTE: P[i - 1] is a view of P, and this line writes into P
    return P


def prefix_products(W):
    P = torch.empty_like(W)
    P[0] = W[0]
    for i in range(1, W.shape[0]):
        P[i] = (P[i - 1] @ W[i]).detach()       # detached: autograd no longer needs the saved P[i - 1]
    return P


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
        dX[:stride] += H[:stride]                                   # copied entries
        dX[stride:] += X[:-stride].transpose(1, 2) @ H[stride:]     # X_j as the right factor of Y_j
        dX[:-stride] += H[stride:] @ X[stride:].transpose(1, 2)     # X_j as the left factor of Y_{j+s}
        H = dX                                                      # NOTE: all three lines are +=, most entries
    return H                                                        #       are used twice in one round
