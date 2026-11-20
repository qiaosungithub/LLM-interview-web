"""The statement's own code, copied verbatim from the code blocks of README.md (identical in README.zh.md):
the simulator ("The machine"), reference, mem_layout / make_memory / run_program, and build_baseline
("The kernel to optimise"). Only this docstring is added. tests.py runs every answer through run_program."""

VLEN = 8
NUM_SREG = 8
NUM_VREG = 8
LOAD_LATENCY = 4
UNIT_WIDTH = {"scalar": 2, "vector": 1, "load": 1, "store": 1}

# Every opcode belongs to exactly one issue unit.
OP_UNIT = {
    "li": "scalar", "mov": "scalar", "add": "scalar", "addi": "scalar", "mul": "scalar",
    "sload": "load", "vload": "load",
    "sstore": "store", "vstore": "store",
    "vadd": "vector", "vmul": "vector", "vmadd": "vector", "vbcast": "vector", "vrsum": "vector",
}


def _operand_regs(instr):
    """The (reads, writes) of one instruction, each a list of ("s" | "v", index)."""
    op, *args = instr
    if op == "li": return [], [("s", args[0])]
    if op == "mov": return [("s", args[1])], [("s", args[0])]
    if op in ("add", "mul"): return [("s", args[1]), ("s", args[2])], [("s", args[0])]
    if op == "addi": return [("s", args[1])], [("s", args[0])]
    if op == "sload": return [("s", args[1])], [("s", args[0])]
    if op == "vload": return [("s", args[1])], [("v", args[0])]
    if op == "sstore": return [("s", args[0]), ("s", args[1])], []
    if op == "vstore": return [("s", args[0]), ("v", args[1])], []
    if op in ("vadd", "vmul"): return [("v", args[1]), ("v", args[2])], [("v", args[0])]
    if op == "vmadd": return [("v", args[1]), ("v", args[2]), ("v", args[3])], [("v", args[0])]
    if op == "vbcast": return [("s", args[1])], [("v", args[0])]
    if op == "vrsum": return [("v", args[1])], [("s", args[0])]
    raise ValueError(f"unknown opcode {op!r}")


class VectorMachine:
    """An in-order vector processor. A program is a list of bundles; a bundle is a list of
    instructions issued in the same cycle, at most UNIT_WIDTH[unit] per unit. See the machine
    description above for the read/write and stall rules."""

    def __init__(self, mem):
        self.s = [0] * NUM_SREG
        self.v = [[0.0] * VLEN for _ in range(NUM_VREG)]
        self.mem = mem
        self.cycles = 0

    def _eval(self, instr):
        """The writes this instruction produces, computed from the register file as it stood
        before the enclosing bundle. Returns a list of (file, index, value); a store's effect on
        memory is applied immediately, since nothing here re-reads a location before its own
        prior write to it has retired."""
        op, *args = instr
        s, v, mem = self.s, self.v, self.mem
        if op == "li": return [("s", args[0], args[1])]
        if op == "mov": return [("s", args[0], s[args[1]])]
        if op == "add": return [("s", args[0], s[args[1]] + s[args[2]])]
        if op == "mul": return [("s", args[0], s[args[1]] * s[args[2]])]
        if op == "addi": return [("s", args[0], s[args[1]] + args[2])]
        if op == "sload":
            addr = s[args[1]]
            if not 0 <= addr < len(mem):
                raise IndexError(f"sload: address {addr} out of range")
            return [("s", args[0], mem[addr])]
        if op == "vload":
            base = s[args[1]]
            if base < 0 or base + VLEN > len(mem):
                raise IndexError(f"vload: range [{base}, {base + VLEN}) out of range")
            return [("v", args[0], mem[base:base + VLEN])]
        if op == "sstore":
            addr = s[args[0]]
            if not 0 <= addr < len(mem):
                raise IndexError(f"sstore: address {addr} out of range")
            mem[addr] = s[args[1]]
            return []
        if op == "vstore":
            base = s[args[0]]
            if base < 0 or base + VLEN > len(mem):
                raise IndexError(f"vstore: range [{base}, {base + VLEN}) out of range")
            mem[base:base + VLEN] = v[args[1]]
            return []
        if op == "vadd": return [("v", args[0], [p + q for p, q in zip(v[args[1]], v[args[2]])])]
        if op == "vmul": return [("v", args[0], [p * q for p, q in zip(v[args[1]], v[args[2]])])]
        if op == "vmadd":
            return [("v", args[0], [p * q + r for p, q, r in zip(v[args[1]], v[args[2]], v[args[3]])])]
        if op == "vbcast": return [("v", args[0], [s[args[1]]] * VLEN)]
        if op == "vrsum": return [("s", args[0], sum(v[args[1]]))]
        raise ValueError(f"unknown opcode {op!r}")

    def run(self, program):
        """Executes `program` (a list of bundles) and returns its cycle count."""
        ready = {("s", i): 0 for i in range(NUM_SREG)}
        ready.update({("v", i): 0 for i in range(NUM_VREG)})
        cycle = 0
        for bnum, bundle in enumerate(program):
            used = {"scalar": 0, "vector": 0, "load": 0, "store": 0}
            reads = []
            for instr in bundle:
                unit = OP_UNIT[instr[0]]
                used[unit] += 1
                if used[unit] > UNIT_WIDTH[unit]:
                    raise ValueError(f"bundle {bnum}: more than {UNIT_WIDTH[unit]} {unit} op(s): {bundle}")
                r, _w = _operand_regs(instr)
                reads += r
            issue = max([cycle] + [ready[reg] for reg in reads])   # stall until every source is ready
            pending, seen = [], set()
            for instr in bundle:
                unit = OP_UNIT[instr[0]]
                for file, idx, val in self._eval(instr):
                    if (file, idx) in seen:
                        raise ValueError(f"bundle {bnum}: two writes to {file}{idx}")
                    seen.add((file, idx))
                    latency = LOAD_LATENCY if unit == "load" else 1
                    pending.append((file, idx, val, issue + latency))
            for file, idx, val, ready_at in pending:
                (self.s if file == "s" else self.v)[idx] = val
                ready[(file, idx)] = ready_at
            cycle = issue + 1
        self.cycles = cycle
        return cycle


def reference(x, y, a, b):
    """The exact quantity the kernel must produce."""
    return sum((a * xi + b) * yi for xi, yi in zip(x, y))


def mem_layout(n):
    """Where each input and the output live, for arrays of length n."""
    return dict(base_x=0, base_y=n, addr_a=2 * n, addr_b=2 * n + 1, addr_out=2 * n + 2)


def make_memory(x, y, a, b):
    n = len(x)
    assert len(y) == n
    return list(x) + list(y) + [float(a), float(b), 0.0]


def run_program(build, n, x, y, a, b):
    """Builds a program for this n, runs it on a fresh VectorMachine, and returns (S, cycles)."""
    machine = VectorMachine(make_memory(x, y, a, b))
    cycles = machine.run(build(n))
    return machine.mem[mem_layout(n)["addr_out"]], cycles


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
