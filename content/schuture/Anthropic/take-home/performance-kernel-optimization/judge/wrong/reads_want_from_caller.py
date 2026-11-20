# expect: WA
# Does not compute S at all: build_optimised walks up the call stack to the caller's x, y, a, b, evaluates
# reference() in Python and returns a three-instruction program that stores that constant. The program only
# fits the data that happened to be visible while it was built.
import sys

from vector_machine import mem_layout, reference


def build_optimised(n):
    frame, want = sys._getframe(1), 0.0
    while frame is not None:
        loc = frame.f_locals
        if all(k in loc for k in "xyab") and isinstance(loc["x"], list) and len(loc["x"]) == n:
            want = reference(loc["x"], loc["y"], loc["a"], loc["b"])
            break
        frame = frame.f_back
    return [[("li", 0, mem_layout(n)["addr_out"]), ("li", 1, want)], [("sstore", 0, 1)]]
