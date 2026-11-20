import numpy as np


# Part 1

def predict_1nn(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Shapes: X_train (n, d) float, y_train (n,) int, X_query (m, d) float.
    Returns the predicted labels, shape (m,)."""
    raise NotImplementedError


# Part 2

def affine_layer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Builds the layer from X_train of shape (n, d). Returns W of shape (d, n) and b of shape (n,)."""
    raise NotImplementedError


def predict_1nn_affine(X_train: np.ndarray, y_train: np.ndarray,
                       X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass. Returns (probs, labels): probs[j] = softmax(logits[j]), shape (m, n),
    and labels[j] = y_train[argmax(probs[j])], shape (m,)."""
    raise NotImplementedError
