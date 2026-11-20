"""answer.py defines build_optimised(n), which returns a program (a list of bundles) for arrays of length n. Each
program runs through the page's own run_program on the page's own VectorMachine (vector_machine.py, the statement's
code copied verbatim); the value it leaves in mem[2n + 2] must match reference(x, y, a, b) within the tolerance of
the page's checks. Cycle counts, the speedup over build_baseline and the lower bound are reported with info(), never
graded."""
import numbers
import random

from judgelib import Mismatch, case, info, rng
import vector_machine
from vector_machine import VLEN, build_baseline, mem_layout, reference, run_program   # bound before the answer loads

# the page's quoted cycle counts for build_step4, the version it submits ("The lower bound" table)
STEP4 = {64: 32, 1024: 272}
BASELINE = {n: run_program(build_baseline, n, [0.0] * n, [0.0] * n, 0.0, 0.0)[1] for n in STEP4}   # 1223, 19463


# ---------------------------------------------------------------- the machine must stay as it is
def _snapshot():
    return {"vector_machine": dict(vars(vector_machine)),
            "VectorMachine": dict(vars(vector_machine.VectorMachine))}


def _contents():
    return {"UNIT_WIDTH": dict(vector_machine.UNIT_WIDTH), "OP_UNIT": dict(vector_machine.OP_UNIT)}


ORIGINAL, CONTENTS = _snapshot(), _contents()


def check_untouched():
    """The statement says to speed the kernel up "without changing the processor itself". The answer shares the
    process with the simulator, so it could patch it at run time; refuse to score if it did."""
    now, contents = _snapshot(), _contents()
    changed = [f"{where}.{k}" for where, names in ORIGINAL.items()
               for k in names.keys() | now[where].keys()
               if k != "__warningregistry__" and names.get(k) is not now[where].get(k)]
    changed += [name for name, value in CONTENTS.items() if contents[name] != value]
    if changed:
        raise Mismatch(f"答案在运行时改动了模拟器：{'、'.join(sorted(changed)[:5])}。题目要求只改内核、不改处理器本身，"
                       "这样跑出来的结果不算数", show=False)


# ---------------------------------------------------------------- one run
def from_answer(exc):
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "/work/answer.py":
            return True
        tb = tb.tb_next
    return False


def rejected(e):
    """Why the page's simulator refused the program, in Chinese."""
    text = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    if isinstance(e, ValueError) and "more than" in str(e):
        why = "一个指令束里每类单元最多：标量 ALU 2 条、向量 ALU 1 条、load 1 条、store 1 条"
    elif isinstance(e, ValueError) and "two writes" in str(e):
        why = "同一个指令束里不能有两条指令写同一个寄存器"
    elif isinstance(e, IndexError) and "out of range" in str(e):
        why = "访存越界：内存只有 2n + 3 个字（x、y、a、b 和输出），vload/vstore 一次访问连续 VLEN = 8 个字"
    elif isinstance(e, KeyError) and isinstance(e.args[0], tuple):
        why = "寄存器编号越界：只有标量寄存器 s0–s7 和向量寄存器 v0–v7"
    elif isinstance(e, KeyError) or "unknown opcode" in str(e):
        why = "指令名不在指令表里（指令名要小写，例如 vmadd）"
    else:
        why = '程序的格式不对：程序是指令束的列表，指令束是指令元组的列表，例如 [[("li", 0, 5)], [("sload", 1, 0)]]'
    return f"{why}（{text}）"


def data(n, seed, lo=-5.0, hi=5.0):
    r = random.Random(seed)
    x = [r.uniform(lo, hi) for _ in range(n)]
    y = [r.uniform(lo, hi) for _ in range(n)]
    code = f"r = random.Random({seed}); x = [r.uniform({lo}, {hi}) for _ in range({n})]; y = [r.uniform({lo}, {hi}) for _ in range({n})]"
    return x, y, code


def shown_call(n, x, y, a, b, code=None):
    shown = f"run_program(build_optimised, {n}, {x!r}, {y!r}, {a!r}, {b!r})"
    if len(shown) > 600 and code:
        shown = f"import random; {code}; run_program(build_optimised, {n}, x, y, {a!r}, {b!r})"
    elif len(shown) > 600:
        shown = f"run_program(build_optimised, {n}, x, y, {a!r}, {b!r})  # x、y 各 {n} 个数"
    return shown


def run(sol, n, x, y, a, b, code=None, want=None):
    """One run_program(build_optimised, n, x, y, a, b); returns the cycle count. `code` is Python that rebuilds x and
    y, shown in the call when the lists are too long to print. The same program is then run once more on data drawn
    after it was built: build_optimised gets only n, so the program must not depend on the data (an answer could
    otherwise read x, y or want from the caller's frame and store a constant)."""
    want = reference(x, y, a, b) if want is None else want
    prog = sol.build_optimised(n)          # errors here are the answer's own, shown with its traceback
    check_untouched()
    cycles = run_once(prog, n, x, y, a, b, want, shown_call(n, x, y, a, b, code))
    r = rng(f"again {n}")
    x2, y2, code2 = data(n, r.getrandbits(32))
    a2, b2 = round(r.uniform(-4, 4), 3), round(r.uniform(-4, 4), 3)
    run_once(prog, n, x2, y2, a2, b2, reference(x2, y2, a2, b2), shown_call(n, x2, y2, a2, b2, code2),
             again="同一个程序换一组数据（在 build_optimised 返回之后才生成）就不对了：build_optimised(n) 只拿到 n，"
                   "返回的程序要对任意 x、y、a、b 都写出 S。")
    return cycles


def run_once(prog, n, x, y, a, b, want, shown, again=""):
    try:
        got, cycles = run_program(lambda _n: prog, n, list(x), list(y), a, b)
    except Exception as e:      # noqa: BLE001 - TimeUp is a BaseException and passes through
        if from_answer(e):
            raise
        raise Mismatch(f"题目的模拟器拒绝了 build_optimised({n}) 返回的程序：{rejected(e)}", shown, show=False) from None
    addr = mem_layout(n)["addr_out"]
    if not isinstance(got, numbers.Real):
        raise Mismatch(f"mem[2n + 2]（这里是 mem[{addr}]）里应当是一个数，得到的是 {type(got).__name__}", shown, want, got)
    tol = 1e-6 * max(1.0, abs(want))      # the page's own check: abs(got - want) < 1e-6 * max(1.0, abs(want))
    if not abs(got - want) < tol:
        msg = again + f"程序结束时 mem[2n + 2]（这里是 mem[{addr}]）和 reference(x, y, a, b) 不一致，超出题目检查用的容差 {tol:.3g}"
        if not any(instr[0] in ("sstore", "vstore") for bundle in prog for instr in bundle):
            msg += "。程序里没有一条 sstore/vstore：结果没有写回 mem[2n + 2]，那里还是初始值 0.0"
        raise Mismatch(msg, shown, want, got)
    return cycles


def report(n, cycles):
    base, step4, bound = BASELINE[n], STEP4[n], 2 * (n // VLEN)
    info(f"n = {n}：周期数 {cycles:,}，build_baseline {base:,}，加速比 {base / cycles:.2f}×" if cycles else
         f"n = {n}：周期数 {cycles}")
    info(f"下界 2 × (n // 8) = {bound}（load 和向量 ALU 各每周期 1 条）；参考解答 Step 4 是 {step4} 个周期，"
         f"加速比 {base / step4:.2f}×。周期数只报告，不影响评测结果")


# ---------------------------------------------------------------- cases
@case("Part 1", sample=True, title="题目的例子：n = 64，种子 0")
def worked_example(sol):
    # "All five versions below are measured on the same worked example: 64 elements, drawn once with a fixed seed"
    rng0 = random.Random(0)
    x = [rng0.uniform(-3, 3) for _ in range(64)]
    y = [rng0.uniform(-3, 3) for _ in range(64)]
    code = "rng = random.Random(0); x = [rng.uniform(-3, 3) for _ in range(64)]; y = [rng.uniform(-3, 3) for _ in range(64)]"
    report(64, run(sol, 64, x, y, 1.7, -0.4, code, want=49.60332079481874))


@case("Part 1", title="n = 0 到 40：不满一个向量和零头")
def small_sizes(sol):
    r = rng()
    for n in range(41):
        x, y, code = data(n, r.getrandbits(32))
        run(sol, n, x, y, round(r.uniform(-4, 4), 3), round(r.uniform(-4, 4), 3), code)


@case("Part 1", title="较大的 n，不是 8 的倍数")
def tails(sol):
    r = rng()
    for n in (63, 65, 71, 100, 127, 129, 199, 255, 257, 1000, 1023):
        x, y, code = data(n, r.getrandbits(32))
        run(sol, n, x, y, round(r.uniform(-4, 4), 3), round(r.uniform(-4, 4), 3), code)


@case("Part 1", title="随机规模 300 组（同题目的随机检查）")
def random_sweep(sol):
    # sizes drawn the way the page's "random correctness sweep" draws them
    r = rng()
    for _ in range(300):
        n = r.choice([0, 1, 2, 3, 7, 8, 9, 15, 16, 17]) if r.random() < 0.6 else r.randint(0, 90)
        x, y, code = data(n, r.getrandbits(32))
        run(sol, n, x, y, r.uniform(-4, 4), r.uniform(-4, 4), code)


@case("Part 1", title="a、b 取 0 或 1，x、y 取整数")
def special_values(sol):
    for n in (5, 8, 24, 37):
        xs, ys = [float(i % 7 - 3) for i in range(n)], [float(2 - i % 5) for i in range(n)]
        for a, b in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 2.0)):
            run(sol, n, xs, ys, a, b)
        run(sol, n, [0.0] * n, ys, 2.5, -1.5)


@case("Part 1", title="n = 1024：周期数、加速比和下界")
def large(sol):
    x, y, code = data(1024, rng().getrandbits(32))
    report(1024, run(sol, 1024, x, y, 1.1, -0.2, code))
