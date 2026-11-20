# expect: WA
# Step 3 (two chunks per pass) made general with Step 4's scalar tail for the last n % 8 elements: when n // 8 is
# odd, the unpaired last whole chunk is in no pass, and the tail loop starts at it, so it sums the wrong elements.
from vector_machine import VLEN, mem_layout


V_A, V_B, V_ACC = 0, 1, 2      # persistent vector registers: broadcast a, broadcast b, running sum
P_X, P_Y = 0, 1                # persistent scalar registers: pointers into x and y


def _setup(L):
    a_val, b_val, zero = 4, 5, 6     # scratch, free again once broadcast
    return [
        [("li", P_X, L["base_x"]), ("li", P_Y, L["base_y"])],
        [("li", 2, L["addr_a"]), ("li", 3, L["addr_b"])],
        [("sload", a_val, 2), ("li", zero, 0.0)],
        [("sload", b_val, 3)],
        [("vbcast", V_A, a_val)],
        [("vbcast", V_B, b_val)],
        [("vbcast", V_ACC, zero)],
    ]


def _finish(L, scalar_acc_reg):
    out_addr = 6
    return [[("li", out_addr, L["addr_out"])], [("sstore", out_addr, scalar_acc_reg)]]


def build_optimised(n):
    L = mem_layout(n)
    V_X0, V_Y0, V_X1, V_Y1 = 3, 4, 5, 6
    prog = _setup(L)
    for _ in range(n // (2 * VLEN)):
        prog += [
            [("vload", V_X0, P_X), ("addi", P_X, P_X, VLEN)],
            [("vload", V_Y0, P_Y), ("addi", P_Y, P_Y, VLEN)],
            [("vload", V_X1, P_X), ("addi", P_X, P_X, VLEN)],
            [("vload", V_Y1, P_Y), ("addi", P_Y, P_Y, VLEN)],
            [("vmadd", V_X0, V_A, V_X0, V_B)],
            [("vmadd", V_ACC, V_X0, V_Y0, V_ACC)],
            [("vmadd", V_X1, V_A, V_X1, V_B)],
            [("vmadd", V_ACC, V_X1, V_Y1, V_ACC)],
        ]
    tail = n % VLEN
    scalar_acc = 2
    prog.append([("vrsum", scalar_acc, V_ACC)])
    if tail:
        a_val, b_val, addr_a, addr_b = 3, 4, 5, 6
        prog.append([("li", addr_a, L["addr_a"]), ("li", addr_b, L["addr_b"])])
        prog.append([("sload", a_val, addr_a)])
        prog.append([("sload", b_val, addr_b)])
        prod, tmp = 5, 6         # reuse addr_a's and addr_b's registers once a_val/b_val are loaded
        for _ in range(tail):
            prog += [
                [("sload", prod, P_X)],
                [("mul", prod, a_val, prod)],
                [("add", prod, prod, b_val)],
                [("sload", tmp, P_Y)],
                [("mul", prod, prod, tmp)],
                [("add", scalar_acc, scalar_acc, prod)],
                [("addi", P_X, P_X, 1), ("addi", P_Y, P_Y, 1)],
            ]
    prog += _finish(L, scalar_acc)
    return prog
