# expect: WA
# Part 2: softmax shifted by the maximum of the whole matrix instead of each row's; rows far below it underflow to 0 / 0.
import numpy as np


# Part 1

def nearest_indices(X_train, X_query):
    sq_train = (X_train ** 2).sum(axis=1)                           # (n,)    |x_i|^2
    sq_query = (X_query ** 2).sum(axis=1, keepdims=True)            # (m, 1)  |q_j|^2
    # NOTE: keepdims=True. With shape (m,) the sum below raises "operands could not be broadcast together"
    #       if m != n, and if m == n it silently adds |q_j|^2 to COLUMN j, which changes the argmin.
    inner = X_query @ X_train.T                                     # (m, d) @ (d, n) -> (m, n)
    sqdist = sq_query + sq_train - 2.0 * inner                      # (m, 1) + (n,) - (m, n) -> (m, n)
    # NOTE: rounding can leave entries such as -7e-15. Harmless for argmin; clip at 0 before any sqrt.
    # NOTE: axis=1 runs over the training points, and argmin returns the FIRST minimum, which is the tie rule.
    return np.argmin(sqdist, axis=1)                                # (m,)


def predict_1nn(X_train, y_train, X_query):
    return y_train[nearest_indices(X_train, X_query)]               # integer-array indexing: (n,)[(m,)] -> (m,)


# Part 2

def affine_layer(X_train):
    # NOTE: W is the TRANSPOSE of W_1, column i is 2 x_i. Without .T, X_query @ W raises a shape error
    #       if d != n and is silently wrong if d == n.
    W = 2.0 * X_train.T                                 # (d, n)
    b = -(X_train ** 2).sum(axis=1)                     # (n,)
    return W, b


def softmax(Z):
    # NOTE: the logits grow like |x|^2. np.exp(1000.0) is inf and inf / inf is nan, so shift every row to max 0.
    E = np.exp(Z - Z.max())                             # one shift for the whole (m, n) matrix
    # NOTE: axis=1, one distribution per query. axis=0 runs without an error and changes the argmax.
    return E / E.sum(axis=1, keepdims=True)             # (m, n) / (m, 1) -> (m, n)


def predict_1nn_affine(X_train, y_train, X_query):
    W, b = affine_layer(X_train)
    logits = X_query @ W + b                            # (m, d) @ (d, n) + (n,) -> (m, n)
    # NOTE: b of shape (n,) lines up with the LAST axis of (m, n), so it is added to every row.
    probs = softmax(logits)                             # (m, n), every row sums to 1
    nearest = np.argmax(probs, axis=1)                  # (m,)
    # NOTE: in floating point the softmax can turn two distinct logits into a tie: softmax([[-1e-17, 0.]])
    #       is [[0.5, 0.5]], and the argmax moves from 1 to 0. np.argmax(logits, axis=1) avoids this.
    return probs, y_train[nearest]
