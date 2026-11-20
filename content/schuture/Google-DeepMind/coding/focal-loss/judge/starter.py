import numpy as np


def focal_loss_and_grad(z: np.ndarray, y: np.ndarray, alpha: float = 0.25,
                         gamma: float = 2.0) -> tuple[float, np.ndarray]:
    """z, y: 1-D arrays of the same shape (N,), one logit and one label (0.0 or 1.0) per example.
    Returns (mean_loss, grad), where mean_loss is the batch mean of FL and grad has shape (N,),
    grad[i] = d(mean_loss)/dz[i]. Finite for every finite z, including |z| up to 100."""
    raise NotImplementedError
