# expect: WA
# The reference kernel with the wrap period off by one: an item reaches the leaves at depth `height` and wraps to
# the root after height + 1 rounds, not height, so from round 10 on every item reads the wrong tree level.
"""
Staged study solution for Anthropic's original performance take-home.

Keep this file local. The header of the original perf_takehome.py allows modifying and using the
code but asks people not to publish or redistribute solutions.

This is a drop-in replacement for perf_takehome.py in a copy of the repository:

    cp perf_takehome_staged.py <copy of the repository>/perf_takehome.py
    cd <copy of the repository>
    python tests/submission_tests.py                 # stage 9
    PERF_STAGE=3 python tests/submission_tests.py    # any earlier stage

It implements nine stages; the environment variable PERF_STAGE picks one (default: 9, the last).
The submission tests compare only the final values, so the final indices are not written back
unless PERF_STORE_IDX=1 is set (stages 3 to 9; measured: stage 9 then takes 1055 cycles).
Cycle counts below were measured with tests/submission_tests.py (forest_height=10, rounds=16,
batch_size=256); the unmodified starter code measures 147734.

    stage  cycles  what changes
      1     97040  scalar, one slot per cycle, redundant work removed
      2      7690  the same operations, packed into cycles by a dependency-aware scheduler
      3      2084  SIMD across the batch: 32 blocks of 8 items, node values gathered with scalar loads
      4      1548  tree levels 0-2 need no loads (node picked arithmetically from 7 preloaded values)
      5      1355  operations handed to the scheduler block by block instead of round by round
      6      1329  fewer vector operations per round (hash rewrites, bit/address work on the scalar ALU)
      7      1107  level 3 needs no loads either; all level picks move to the idle flow engine (vselect)
      8      1064  20 of the 32 blocks also pick level 4 on the flow engine on the second trip down
      9      1040  the XOR with a gathered node value moves to the scalar ALU

How the program is built: the kernel is written as straight-line code over *virtual* registers
(every word is written once), a scheduler assigns each operation to a cycle subject to the
per-engine slot limits, and only then are virtual registers mapped to scratch addresses.
There are no jumps, so the cycle count is simply the number of bundles.
"""

import heapq
import os
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

STAGE = int(os.environ.get("PERF_STAGE", "9"))
STORE_IDX = os.environ.get("PERF_STORE_IDX", "0") == "1"
M32 = 2**32

STAGES = {
    1: dict(form="scalar", pack=False),
    2: dict(form="scalar"),
    3: dict(form="vector", order="round", top=0, sel="madd"),
    4: dict(form="vector", order="round", top=3, sel="madd"),
    5: dict(form="vector", order="block", top=3, sel="madd"),
    6: dict(form="vector", order="block", top=3, sel="madd", fuse23=True, defer_c5=True, alu_bits=True),
    7: dict(form="vector", order="block", top=4, sel="flow", fuse23=True, defer_c5=True, alu_bits=True),
    8: dict(form="vector", order="block", top=4, sel="flow", fuse23=True, defer_c5=True, alu_bits=True,
            deep=(0, 20)),
    9: dict(form="vector", order="block", top=4, sel="flow", fuse23=True, defer_c5=True, alu_bits=True,
            deep=(0, 20), alu_xor=True),
}


# ---------------------------------------------------------------- straight-line IR

class R:
    """An operand that is a scratch address: virtual register `v`, word offset `off`."""

    __slots__ = ("v", "off")

    def __init__(self, v, off=0):
        self.v = v
        self.off = off


class Prog:
    """Straight-line code over virtual registers. A register is 1 word or VLEN words, and each
    word is written exactly once, so the only ordering rule is: read after the defining write.

    A scalar operand is a pair (register, lane); lane is 0 for a 1-word register, and (v, k)
    names word k of a vector register. A vector operand is just the register number."""

    def __init__(self):
        self.width = []     # register -> 1 or VLEN
        self.eng = []       # op -> engine name
        self.tmpl = []      # op -> slot tuple whose scratch operands are still R(...)
        self.rd = []        # op -> registers read
        self.wr = []        # op -> registers written
        self.preds = []     # op -> set of ops whose results it reads
        self.defop = {}     # (register, lane or None for "all words") -> defining op
        self._cache = {}

    def reg(self, width=1):
        self.width.append(width)
        return len(self.width) - 1

    def _defs(self, v, lane):
        if (v, None) in self.defop:
            return {self.defop[(v, None)]}
        if lane is None:
            return {self.defop[(v, k)] for k in range(self.width[v])}
        return {self.defop[(v, lane)]}

    def emit(self, eng, tmpl, rd, wr):
        """rd / wr: lists of (register, lane); lane None means the whole register."""
        i = len(self.eng)
        ps = set()
        for v, lane in rd:
            ps |= self._defs(v, None if self.width[v] == 1 else lane)
        for v, lane in wr:
            assert (v, None) not in self.defop and (v, lane) not in self.defop, "single assignment"
            self.defop[(v, lane)] = i
        self.eng.append(eng)
        self.tmpl.append(tmpl)
        self.rd.append([v for v, _ in rd])
        self.wr.append([v for v, _ in wr])
        self.preds.append(ps)
        return i

    # -- scalar operations
    def const(self, val):
        key = ("const", val % M32)
        if key not in self._cache:
            d = self.reg()
            self.emit("load", ("const", R(d), val % M32), [], [(d, None)])
            self._cache[key] = (d, 0)
        return self._cache[key]

    def alu(self, op, a, b, cache=False):
        key = ("alu", op, a, b)
        if cache and key in self._cache:
            return self._cache[key]
        d = self.reg()
        self.emit("alu", (op, R(d), R(*a), R(*b)), [a, b], [(d, None)])
        if cache:
            self._cache[key] = (d, 0)
        return (d, 0)

    def alu_lane(self, op, dest, lane, a, b):
        """One scalar ALU slot that writes word `lane` of the vector register `dest`."""
        self.emit("alu", (op, R(dest, lane), R(*a), R(*b)), [a, b], [(dest, lane)])

    def load(self, addr):
        d = self.reg()
        self.emit("load", ("load", R(d), R(*addr)), [addr], [(d, None)])
        return (d, 0)

    def load_lane(self, dest, lane, addr):
        self.emit("load", ("load", R(dest, lane), R(*addr)), [addr], [(dest, lane)])

    def store(self, addr, src):
        self.emit("store", ("store", R(*addr), R(*src)), [addr, src], [])

    # -- vector operations
    def valu(self, op, a, b):
        d = self.reg(VLEN)
        self.emit("valu", (op, R(d), R(a), R(b)), [(a, None), (b, None)], [(d, None)])
        return d

    def madd(self, a, b, c):
        d = self.reg(VLEN)
        self.emit("valu", ("multiply_add", R(d), R(a), R(b), R(c)),
                  [(a, None), (b, None), (c, None)], [(d, None)])
        return d

    def bcast(self, s):
        """Broadcast a scalar operand to a vector. Cached: each scalar is broadcast once."""
        key = ("bcast", s)
        if key not in self._cache:
            d = self.reg(VLEN)
            self.emit("valu", ("vbroadcast", R(d), R(*s)), [s], [(d, None)])
            self._cache[key] = d
        return self._cache[key]

    def kvec(self, val):
        return self.bcast(self.const(val))

    def vload(self, addr):
        d = self.reg(VLEN)
        self.emit("load", ("vload", R(d), R(*addr)), [addr], [(d, None)])
        return d

    def vstore(self, addr, src):
        self.emit("store", ("vstore", R(*addr), R(src)), [addr, (src, None)], [])

    def vselect(self, cond, a, b):
        d = self.reg(VLEN)
        self.emit("flow", ("vselect", R(d), R(cond), R(a), R(b)),
                  [(cond, None), (a, None), (b, None)], [(d, None)])
        return d


# ---------------------------------------------------------------- scheduling

class Slots:
    """Slot counter of one engine, with 'first cycle >= c that still has a free slot'
    (full cycles are skipped with a union-find style pointer)."""

    def __init__(self, cap):
        self.cap = cap
        self.use = []
        self.nxt = []

    def _grow(self, c):
        while len(self.use) <= c:
            self.nxt.append(len(self.use))
            self.use.append(0)

    def find(self, c):
        self._grow(c)
        r = c
        while self.nxt[r] != r:
            r = self.nxt[r]
            self._grow(r)
        while self.nxt[c] != c:
            self.nxt[c], c = r, self.nxt[c]
        return r

    def take(self, c):
        self._grow(c + 1)
        self.use[c] += 1
        if self.use[c] == self.cap:
            self.nxt[c] = c + 1


def place(order, deps, eng, reserved=()):
    """Serial list scheduling. Take the operations in `order` (every operation must come after
    the ones in deps[op]) and put each into the earliest cycle that is later than all of deps[op]
    and still has a free slot on its engine. "Later" because a result written in cycle c is
    visible from cycle c + 1 on."""
    slots = {name: Slots(cap) for name, cap in SLOT_LIMITS.items()}
    for name, c in reserved:
        slots[name].take(c)
    cyc = [-1] * len(eng)
    for i in order:
        lb = 0
        for p in deps[i]:
            if cyc[p] >= lb:
                lb = cyc[p] + 1
        c = slots[eng[i]].find(lb)
        slots[eng[i]].take(c)
        cyc[i] = c
    return cyc


def schedule(pr, pack=True, passes=3):
    n = len(pr.eng)
    if not pack:
        return list(range(n))                     # one slot per cycle, in emission order
    reserved = [("flow", 0)]                      # cycle 0 keeps its flow slot for the `pause`
    cyc = place(range(n), pr.preds, pr.eng, reserved)
    succs = [[] for _ in range(n)]
    for i, ps in enumerate(pr.preds):
        for p in ps:
            succs[p].append(i)
    best = cyc
    for _ in range(passes):
        # Schedule the reversed graph (everything as late as possible), then go forward again
        # in the order that the late schedule suggests. Keep the best forward schedule seen.
        back = place(sorted(range(n), key=lambda i: (-cyc[i], -i)), succs, pr.eng)
        cyc = place(sorted(range(n), key=lambda i: (-back[i], i)), pr.preds, pr.eng, reserved)
        if max(cyc) < max(best):
            best = cyc
    return best


def allocate(pr, cyc):
    """Map virtual registers to scratch addresses (linear scan over live ranges). A register is
    live from its first write to its last read. Writes land at the end of a cycle, so a new
    register may take over an address in the very cycle in which the old owner is read last."""
    first, last = {}, {}
    for i, c in enumerate(cyc):
        for v in pr.wr[i]:
            first[v] = min(first.get(v, c), c)
        for v in pr.rd[i]:
            last[v] = max(last.get(v, c), c)
    phys, counts = {}, {}
    for width in (1, VLEN):
        active, free, n_slots = [], [], 0
        for v in sorted((v for v in first if pr.width[v] == width), key=lambda v: first[v]):
            while active and active[0][0] <= first[v]:
                free.append(heapq.heappop(active)[1])
            if free:
                slot = free.pop()
            else:
                slot, n_slots = n_slots, n_slots + 1
            phys[v] = slot
            heapq.heappush(active, (max(last.get(v, 0), first[v] + 1), slot))
        counts[width] = n_slots
    n_scalar = counts[1]
    assert n_scalar + VLEN * counts[VLEN] <= SCRATCH_SIZE, (
        f"out of scratch space: {n_scalar} words + {counts[VLEN]} vectors")
    return {v: (s if pr.width[v] == 1 else n_scalar + VLEN * s) for v, s in phys.items()}, counts


def assemble(pr, cyc, phys):
    bundles = [dict() for _ in range(max(cyc) + 1)]
    for i, c in enumerate(cyc):
        slot = tuple(phys[x.v] + x.off if isinstance(x, R) else x for x in pr.tmpl[i])
        bundles[c].setdefault(pr.eng[i], []).append(slot)
    if "flow" not in bundles[0]:
        # Matches the first `yield` of reference_kernel2 in the local harness (nothing has been
        # stored yet at this point). The submission tests run with pauses disabled.
        bundles[0]["flow"] = [("pause",)]
    return [b for b in bundles if b]


# ---------------------------------------------------------------- the kernel, scalar form (stages 1 and 2)

def gen_scalar(pr, height, n_nodes, batch, rounds):
    """One item at a time, as in the starter code, but values and node addresses stay in scratch
    for the whole run: no per-round load/store of idx and val, indices are never stored, and
    there is no wrap test."""
    P = 7                              # forest_values_p: build_mem_image puts the tree right after the 7-word header
    VP = P + n_nodes + batch           # inp_values_p
    one, two, fix = pr.const(1), pr.const(2), pr.const(1 - P)
    ptr = [pr.const(VP + i) for i in range(batch)]
    val = [pr.load(ptr[i]) for i in range(batch)]
    adr = [pr.const(P)] * batch        # address of the current node, P + idx
    for r in range(rounds):
        depth = r % (height + 1)       # every item starts at the root, so all items share one depth
        for i in range(batch):
            a = pr.alu("^", val[i], pr.load(adr[i]))
            for op1, c1, op2, op3, c3 in HASH_STAGES:
                a = pr.alu(op2, pr.alu(op1, a, pr.const(c1)), pr.alu(op3, a, pr.const(c3)))
            val[i] = a
            if r == rounds - 1:
                pr.store(ptr[i], a)
            elif depth == height:
                adr[i] = pr.const(P)   # all items are on leaves here, so all of them wrap to the root
            else:
                # P + (2*idx + 1 + bit)  =  2*(P + idx) + (1 - P) + bit
                t = pr.alu("+", pr.alu("*", adr[i], two), fix)
                adr[i] = pr.alu("+", t, pr.alu("&", a, one))


# ---------------------------------------------------------------- the kernel, vector form (stages 3 to 9)

def gen_vector(pr, cfg, height, n_nodes, batch, rounds):
    assert batch % VLEN == 0
    P = 7                              # forest_values_p (fixed by build_mem_image)
    VP = P + n_nodes + batch           # inp_values_p
    n_blocks = batch // VLEN
    top = cfg.get("top", 0)            # tree levels 0..top-1 are served from preloaded node values
    # deep = (n0, n1): this many blocks (spread evenly) use one more preloaded level on the
    # first / second trip down the tree
    deep = [{(i * n_blocks) // n for i in range(n)} for n in cfg.get("deep", (0, 0))]
    fuse23, defer_c5 = cfg.get("fuse23", False), cfg.get("defer_c5", False)
    alu_bits, alu_xor = cfg.get("alu_bits", False), cfg.get("alu_xor", False)
    C5 = HASH_STAGES[5][1]
    K = pr.kvec

    def preloaded(v, r):               # number of preloaded levels for block v in round r
        trip = 0 if r <= height else 1
        return min(height, top + (1 if v in deep[trip] else 0))

    ptrs = {}

    def ptr(v, base=VP):               # address of the v-th block of 8 values (scalar adds, two chains)
        if (v, base) not in ptrs:
            if v == 0:
                ptrs[v, base] = pr.const(base)
            elif v == 1:
                ptrs[v, base] = pr.alu("+", ptr(0, base), pr.const(VLEN))
            else:
                ptrs[v, base] = pr.alu("+", ptr(v - 2, base), pr.const(2 * VLEN))
        return ptrs[v, base]

    tree_blocks = {}

    def leaf(i, pre):                  # scalar operand holding tree value i (already XORed with C5 if `pre`)
        k = i // VLEN
        if k not in tree_blocks:
            tree_blocks[k] = pr.vload(pr.const(P + VLEN * k))
        s = (tree_blocks[k], i % VLEN)
        return pr.alu("^", s, pr.const(C5), cache=True) if pre else s

    def vhash(x, skip_c5):
        c0, c1, c2, c3, c4, c5 = (h[1] for h in HASH_STAGES)
        s1, s3, s5 = HASH_STAGES[1][4], HASH_STAGES[3][4], HASH_STAGES[5][4]
        a = pr.madd(x, K(1 + (1 << 12)), K(c0))                          # (a + c0) + (a << 12)
        a = pr.valu("^", pr.valu("^", a, K(c1)), pr.valu(">>", a, K(s1)))
        if fuse23:
            # stage 2 gives b = 33a + c2; stage 3 wants (b + c3) ^ (b << 9): both are a*const + const
            a = pr.valu("^", pr.madd(a, K(33), K(c2 + c3)), pr.madd(a, K(33 << s3), K(c2 << s3)))
        else:
            a = pr.madd(a, K(33), K(c2))                                 # (a + c2) + (a << 5)
            a = pr.valu("^", pr.valu("+", a, K(c3)), pr.valu("<<", a, K(s3)))
        a = pr.madd(a, K(9), K(c4))                                      # (a + c4) + (a << 3)
        if skip_c5:
            return pr.valu("^", a, pr.valu(">>", a, K(s5)))              # = hash ^ c5
        return pr.valu("^", pr.valu("^", a, K(c5)), pr.valu(">>", a, K(s5)))

    def low_bit(val):                  # val & 1 for the 8 lanes
        if not alu_bits:
            return pr.valu("&", val, K(1))
        q = pr.reg(VLEN)
        for lane in range(VLEN):
            pr.alu_lane("&", q, lane, (val, lane), pr.const(1))
        return q

    def pick(depth, bits, pre):
        """Node value for a preloaded level. bits[j] = (vector q, polarity p): the direction taken
        at depth j is q ^ p. The position inside the level is the directions read as a binary
        number, first one most significant."""
        leaves = [leaf((1 << depth) - 1 + o, pre) for o in range(1 << depth)]
        if depth == 0:
            return pr.bcast(leaves[0])
        if cfg["sel"] == "flow":
            cur = [pr.bcast(s) for s in leaves]
            for q, p in bits:                                            # earliest direction first: halves
                half = len(cur) // 2
                lo, hi = cur[:half], cur[half:]
                cur = [pr.vselect(q, lo[i], hi[i]) if p else pr.vselect(q, hi[i], lo[i]) for i in range(half)]
            return cur[0]

        def lin(vals, k):              # multilinear polynomial in bits[0..k-1] with scalar coefficients
            if k == 0:
                return pr.bcast(vals[0])
            q, p = bits[k - 1]         # peel the latest direction: even / odd entries
            base, other = (vals[1::2], vals[0::2]) if p else (vals[0::2], vals[1::2])
            diff = [pr.alu("-", o, b, cache=True) for o, b in zip(other, base)]
            return pr.madd(q, lin(diff, k - 1), lin(base, k - 1))

        return lin(leaves, depth)

    def addr_from_bits(bits, depth, base=P):
        """Vector of node addresses base + idx at `depth`, from the directions taken so far."""
        const = base + (1 << depth) - 1
        terms = []
        for j, (q, p) in enumerate(bits):
            w = 1 << (depth - 1 - j)
            if p:                      # direction is 1 - q
                const, w = const + w, -w
            terms.append((q, w))
        acc = K(const)
        for q, w in terms:
            if w == 1:
                acc = pr.valu("+", acc, q)
            elif w == -1:
                acc = pr.valu("-", acc, q)
            else:
                acc = pr.madd(q, K(w), acc)
        return acc

    class Block:
        pass

    state = [Block() for _ in range(n_blocks)]

    def step(v, r):
        """Round r for block v (8 items)."""
        st = state[v]
        depth = r % height
        nd = (depth + 1) % height
        last = r == rounds - 1
        pre_now, pre_next = preloaded(v, r), preloaded(v, r + 1)
        if r == 0:
            st.val, st.flip, st.bits = pr.vload(ptr(v)), False, []
            st.adr = K(P) if pre_now == 0 else None
        t2 = None
        if depth < pre_now:
            x = pr.valu("^", st.val, pick(depth, st.bits, st.flip))
        else:
            # gather: one scalar load per lane, at most two per cycle
            if alu_xor:
                x = pr.reg(VLEN)
                for lane in range(VLEN):
                    pr.alu_lane("^", x, lane, (st.val, lane), pr.load((st.adr, lane)))
            else:
                node = pr.reg(VLEN)
                for lane in range(VLEN):
                    pr.load_lane(node, lane, (st.adr, lane))
                x = pr.valu("^", st.val, node)
            if not last and nd > depth:
                t2 = pr.madd(st.adr, K(2), K(1 - P))                     # next address is t2 + bit
        skip = defer_c5 and not last and nd < pre_next
        val = vhash(x, skip)
        st.val, st.flip = val, skip
        if last:
            pr.vstore(ptr(v), val)
            if cfg.get("store_idx"):                                     # optional: final indices too
                if nd == 0:
                    idx = K(0)
                elif depth < pre_now:
                    idx = addr_from_bits(st.bits + [(low_bit(val), 0)], nd, base=0)
                else:                                                    # 2*(P + idx) + (1 - 2P) + bit
                    idx = pr.valu("+", pr.madd(st.adr, K(2), K(1 - 2 * P)), low_bit(val))
                pr.vstore(ptr(v, P + n_nodes), idx)
        elif nd == 0:                                                    # leaves: everything wraps to the root
            st.bits, st.adr = [], (K(P) if pre_next == 0 else None)
        elif nd < pre_next:
            st.bits = st.bits + [(low_bit(val), int(skip))]
        elif depth < pre_now:                                            # first gathered level
            st.adr, st.bits = addr_from_bits(st.bits + [(low_bit(val), int(skip))], nd), []
        elif alu_bits:
            adr = pr.reg(VLEN)
            for lane in range(VLEN):
                pr.alu_lane("+", adr, lane, (t2, lane), pr.alu("&", (val, lane), pr.const(1)))
            st.adr = adr
        else:
            st.adr = pr.valu("+", t2, pr.valu("&", val, K(1)))

    steps = [(v, r) for v in range(n_blocks) for r in range(rounds)]
    if cfg["order"] == "round":
        steps.sort(key=lambda vr: (vr[1], vr[0]))                        # the order of the reference loops
    for v, r in steps:
        step(v, r)


# ---------------------------------------------------------------- the builder the tests call

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.stats = {}

    def debug_info(self):
        return DebugInfo(scratch_map={})

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int, cfg=None):
        cfg = dict(STAGES[STAGE], store_idx=STORE_IDX) if cfg is None else cfg
        assert n_nodes == 2 ** (forest_height + 1) - 1
        pr = Prog()
        if cfg["form"] == "scalar":
            gen_scalar(pr, forest_height, n_nodes, batch_size, rounds)
        else:
            gen_vector(pr, cfg, forest_height, n_nodes, batch_size, rounds)
        cyc = schedule(pr, cfg.get("pack", True))
        phys, counts = allocate(pr, cyc)
        self.instrs = assemble(pr, cyc, phys)
        ops = {}
        for e in pr.eng:
            ops[e] = ops.get(e, 0) + 1
        # ops[e] / SLOT_LIMITS[e] is a lower bound on the cycle count for this set of operations
        self.stats = dict(cycles=len(self.instrs), ops=ops, scratch=counts)


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
