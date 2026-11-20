# The statement's code (VectorMachine, reference, mem_layout, make_memory, run_program, build_baseline, VLEN, ...)
# is the module vector_machine here, copied verbatim from the page, so a custom run can call run_program directly.
from vector_machine import VLEN, VectorMachine, mem_layout, reference, run_program  # noqa: F401


def build_baseline(n):
    S_ADDR_A, S_ADDR_B, S_PTR_X, S_PTR_Y, S_ACC, S_TMP_AB, S_TMP_XY, S_PROD = range(8)
    L = mem_layout(n)
    prog = [
        [("li", S_ADDR_A, L["addr_a"])],
        [("li", S_ADDR_B, L["addr_b"])],
        [("li", S_PTR_X, L["base_x"])],
        [("li", S_PTR_Y, L["base_y"])],
        [("li", S_ACC, 0.0)],
    ]
    for _ in range(n):
        prog += [
            [("sload", S_TMP_AB, S_ADDR_A)],        # re-read a
            [("sload", S_TMP_XY, S_PTR_X)],          # x[i]
            [("mul", S_PROD, S_TMP_AB, S_TMP_XY)],    # a * x[i]
            [("sload", S_TMP_AB, S_ADDR_B)],         # re-read b
            [("add", S_PROD, S_PROD, S_TMP_AB)],      # a * x[i] + b
            [("sload", S_TMP_XY, S_PTR_Y)],           # y[i]
            [("mul", S_PROD, S_PROD, S_TMP_XY)],       # (a * x[i] + b) * y[i]
            [("add", S_ACC, S_ACC, S_PROD)],          # accumulate
            [("addi", S_PTR_X, S_PTR_X, 1)],
            [("addi", S_PTR_Y, S_PTR_Y, 1)],
        ]
    prog += [[("li", S_TMP_AB, L["addr_out"])], [("sstore", S_TMP_AB, S_ACC)]]
    return prog


def build_optimised(n):
    """Same contract as build_baseline: return a program that, run through run_program, writes
    reference(x, y, a, b) to mem[2n + 2] for every n, in fewer cycles."""
    raise NotImplementedError
