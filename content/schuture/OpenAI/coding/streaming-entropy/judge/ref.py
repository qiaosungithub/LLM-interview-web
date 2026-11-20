"""Reference entropy for tests.py."""
import math

import numpy as np


def entropy(x):
    """H(softmax(x)) as log s - u / s around the maximum m, with s = sum e^(x - m) and u = sum e^(x - m) (x - m).
    Both sums are correctly rounded (math.fsum), so the result is good to a few ulps for every input the tests use
    (spreads far below the float64 maximum, so x - m never overflows)."""
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    d = x - m
    e = np.exp(d)
    s = math.fsum(e.tolist())
    u = math.fsum((e * d).tolist())
    return math.log(s) - u / s
