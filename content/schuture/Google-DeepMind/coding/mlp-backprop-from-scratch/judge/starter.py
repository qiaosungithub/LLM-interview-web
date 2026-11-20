import numpy as np


def forward_backward(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    """Part 1: returns (L, grads), the mean softmax cross-entropy and dL/dW1, dL/db1, dL/dW2, dL/db2 (same shapes as params)."""
    raise NotImplementedError


def grad_check(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """Part 2: largest relative error between forward_backward's analytic gradient and a central finite difference."""
    raise NotImplementedError


def make_spirals(n_per_class: int, n_turns: float, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Part 3: two interleaved spiral arms (class 0 and the same arm rotated by pi), returns (X, y)."""
    raise NotImplementedError


def he_init(d: int, m: int, C: int, seed: int) -> dict[str, np.ndarray]:
    """W1, W2 entries drawn i.i.d. from N(0, 2 / fan_in) (fan_in = d for W1, m for W2); b1, b2 zero."""
    raise NotImplementedError


def zero_init(d: int, m: int, C: int) -> dict[str, np.ndarray]:
    """Every entry of every one of the four parameter arrays is exactly 0.0."""
    raise NotImplementedError


def train(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray, *, steps: int, batch_size: int,
          lr: float, momentum: float, seed: int) -> dict[str, np.ndarray]:
    """Mini-batch SGD with momentum (v <- momentum * v - lr * grad, theta <- theta + v); returns the final params."""
    raise NotImplementedError
