"""Helpers for tests.py: bitonic list generators, strictly increasing integer functions written as lambda text
(so a failing call can be pasted back), and exact roots for Part 3(b) computed with 60-digit decimals."""
import math
from decimal import Decimal, localcontext


# ---------------------------------------------------------------- Parts 1 and 3(a)

def random_bitonic(r, n, min_step):
    """A bitonic list of length n with a uniformly random peak index (both ends included). Steps away from the peak
    are min_step..min_step + 4: min_step = 1 gives a strictly bitonic list, min_step = 0 allows plateaus."""
    p = r.randrange(n)
    peak = r.randint(-30, 60)
    left, v = [], peak
    for _ in range(p):
        v -= r.randint(min_step, min_step + 4)
        left.append(v)
    right, v = [], peak
    for _ in range(n - 1 - p):
        v -= r.randint(min_step, min_step + 4)
        right.append(v)
    return left[::-1] + [peak] + right


def all_strict_bitonic(n):
    """Every strictly bitonic arrangement of 0..n-1: the values left of the peak n-1 are any subset, ascending."""
    out = []
    for mask in range(1 << (n - 1)):
        left = [v for v in range(n - 1) if mask >> v & 1]
        right = [v for v in range(n - 2, -1, -1) if not mask >> v & 1]
        out.append(left + [n - 1] + right)
    return out


def is_weak_bitonic(a):
    k = 0
    while k + 1 < len(a) and a[k] <= a[k + 1]:
        k += 1
    while k + 1 < len(a) and a[k] >= a[k + 1]:
        k += 1
    return k == len(a) - 1


# ---------------------------------------------------------------- Part 2

def _affine(a, b, term="x"):
    head = term if a == 1 else f"{a} * {term}"
    return head if b == 0 else f"{head} + {b}" if b > 0 else f"{head} - {-b}"


def random_increasing_text(r):
    """The text of a lambda that is strictly increasing on the positive integers, with random coefficients."""
    kind = r.randrange(6)
    if kind == 0:
        return "lambda x: " + _affine(r.randint(1, 6), r.randint(-200, 200))
    if kind == 1:
        return "lambda x: " + _affine(1, r.randint(-500, 500), "x * x")
    if kind == 2:
        return "lambda x: " + _affine(1, -r.randint(0, 1000), "x ** 3")
    if kind == 3:
        return "lambda x: " + _affine(1, -r.randint(0, 100), "2 ** x")
    if kind == 4:
        return f"lambda x: x // {r.randint(2, 5)} + " + _affine(1, -r.randint(0, 300))
    a = r.randint(1, 3)                                  # a x^2 - b x rises on x >= 1 when 3a - b >= 1
    return f"lambda x: {a} * x * x - {r.randint(0, 3 * a - 1)} * x"


def floor_brute(fn, y):
    """The largest positive x with fn(x) <= y, or None, by scanning x = 1, 2, 3, ..."""
    if fn(1) > y:
        return None
    x = 1
    while fn(x + 1) <= y:
        x += 1
    return x


# ---------------------------------------------------------------- Part 3(b)

# For each function the user's g is written as: its decimal twin, used to find the exact root.
DECIMAL_TWINS = {
    "lambda x: x": lambda x: x,
    "lambda x: x ** 3": lambda x: x ** 3,
    "lambda x: 2 * x + 1": lambda x: 2 * x + 1,
    "lambda x: x / 1e6": lambda x: x / Decimal(10 ** 6),
    "lambda x: 1e6 * x": lambda x: Decimal(10 ** 6) * x,
    "lambda x: x ** 5 + x": lambda x: x ** 5 + x,
    "math.exp": lambda x: x.exp(),
    "math.log": lambda x: x.ln(),
}


def decimal_twin(text):
    if text in DECIMAL_TWINS:
        return DECIMAL_TWINS[text]
    # random cubic "lambda x: x ** 3 + A * x - B"-style texts: evaluated as decimals by the same expression
    body = text.split(":", 1)[1]
    return lambda x: eval(body, {"x": x, "Decimal": Decimal})


def exact_root(text, y, lo, hi):
    """(x*, g'(x*)) for g(x*) == y on [lo, hi], with g evaluated exactly to 60 digits (bisection, 220 halvings)."""
    g = decimal_twin(text)
    with localcontext() as ctx:
        ctx.prec = 60
        target = Decimal(y)
        a, b = Decimal(lo), Decimal(hi)
        for _ in range(220):
            m = (a + b) / 2
            if g(m) < target:
                a = m
            else:
                b = m
        root = (a + b) / 2
        h = Decimal(10) ** -25 * max(Decimal(1), abs(root))
        slope = (g(root + h) - g(root - h)) / (2 * h)
    return root, float(slope)


def tolerance(root, y, slope, eps):
    """eps plus what floating point can resolve near the root: a couple of ulps of x*, and the x-distance over which
    g's own rounding (about one ulp of y) can flip the comparison g(x) < y."""
    return eps + 2 * math.ulp(float(root)) + 4 * math.ulp(float(y)) / slope
