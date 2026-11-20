# expect: WA
# Step 4 without the scalar tail: like Steps 1 to 3 it assumes n is a multiple of VLEN, so the last n % 8
# elements are never added.
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


def build_step2(n):
    assert n % VLEN == 0
    L = mem_layout(n)
    V_X, V_Y = 3, 4
    prog = _setup(L)
    for _ in range(n // VLEN):
        prog += [
            [("vload", V_X, P_X)],
            [("vload", V_Y, P_Y)],
            [("vmadd", V_X, V_A, V_X, V_B)],          # V_X <- a * x + b
            [("vmadd", V_ACC, V_X, V_Y, V_ACC)],       # V_ACC <- (a * x + b) * y + V_ACC
            [("addi", P_X, P_X, VLEN), ("addi", P_Y, P_Y, VLEN)],
        ]
    prog.append([("vrsum", 4, V_ACC)])
    prog += _finish(L, 4)
    return prog


def build_step4(n):
    L = mem_layout(n)
    num_chunks, tail = n // VLEN, n % VLEN
    SLOTS = [(3, 4), (5, 6)]        # two rotating (v_x, v_y) register pairs
    V_TMP = 7
    prog = _setup(L)

    if num_chunks == 1:
        vx, vy = SLOTS[0]
        prog.append([("vload", vx, P_X), ("addi", P_X, P_X, VLEN)])
        prog.append([("vload", vy, P_Y), ("addi", P_Y, P_Y, VLEN)])
        prog.append([("vmadd", V_TMP, V_A, vx, V_B)])
        prog.append([("vmadd", V_ACC, V_TMP, vy, V_ACC)])
    elif num_chunks >= 2:
        for k in (0, 1):                                    # prologue: fill both slots
            vx, vy = SLOTS[k]
            prog.append([("vload", vx, P_X), ("addi", P_X, P_X, VLEN)])
            prog.append([("vload", vy, P_Y), ("addi", P_Y, P_Y, VLEN)])
        for i in range(num_chunks - 2):                     # steady state: compute i, prefetch i + 2
            vx, vy = SLOTS[i % 2]
            prog.append([("vmadd", V_TMP, V_A, vx, V_B), ("vload", vx, P_X), ("addi", P_X, P_X, VLEN)])
            prog.append([("vmadd", V_ACC, V_TMP, vy, V_ACC), ("vload", vy, P_Y), ("addi", P_Y, P_Y, VLEN)])
        for i in range(max(num_chunks - 2, 0), num_chunks):  # epilogue: drain the last two chunks
            vx, vy = SLOTS[i % 2]
            prog.append([("vmadd", V_TMP, V_A, vx, V_B)])
            prog.append([("vmadd", V_ACC, V_TMP, vy, V_ACC)])

    scalar_acc = 2
    prog.append([("vrsum", scalar_acc, V_ACC)])
    prog += _finish(L, scalar_acc)
    return prog


build_optimised = build_step4
