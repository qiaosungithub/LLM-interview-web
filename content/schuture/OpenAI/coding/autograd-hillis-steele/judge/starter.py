import torch


# Part 1
def prefix_products_inplace(W: torch.Tensor) -> torch.Tensor:
    """W: float tensor of shape (N, D, D). Returns P of shape (N, D, D) with P[i] = W[0] @ ... @ W[i]."""
    raise NotImplementedError


# Part 2
def prefix_products(W: torch.Tensor) -> torch.Tensor:
    """Same result as prefix_products_inplace, without any in-place operation, so that autograd can differentiate it."""
    raise NotImplementedError


# Part 3
def prefix_products_backward(W: torch.Tensor, P: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """W, P, G: shape (N, D, D). P is the forward output and G[i] = dL/dP[i].
    Returns dW of shape (N, D, D) with dW[i] = dL/dW[i]."""
    raise NotImplementedError


# Part 4
def scan_forward(W: torch.Tensor):
    """W: shape (N, D, D). Returns (P, tape): P equals prefix_products(W); tape is whatever scan_backward needs."""
    raise NotImplementedError


def scan_backward(tape, G: torch.Tensor) -> torch.Tensor:
    """tape: the second value returned by scan_forward(W). G[i] = dL/dP[i]. Returns dW with dW[i] = dL/dW[i]."""
    raise NotImplementedError
