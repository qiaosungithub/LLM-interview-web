"""answer.py is the take-home's perf_takehome.py: its KernelBuilder.build_kernel(forest_height, n_nodes, batch_size,
rounds) puts a program in self.instrs. The program is run by the official harness, a byte-identical copy of
tests/submission_tests.py, on the frozen machine of tests/frozen_problem.py (copied here too, see judge.yaml), at the
scored size 10 / 16 / 256. Correctness is checked first; the cycle count and the eight official thresholds are
reported with info(), never graded."""
import random
import sys
import types

from judgelib import Mismatch, case, info, rng

# submission_tests.py runs `from perf_takehome import KernelBuilder` when it is imported. Import it now, before the
# answer is loaded, with a placeholder in its place, so the harness binds the frozen machine before any answer code
# runs; each case then gives the harness, under that name, a KernelBuilder that builds the answer's kernel (Built).
sys.modules["perf_takehome"] = types.SimpleNamespace(KernelBuilder=None)
import frozen_problem          # noqa: E402
import submission_tests        # noqa: E402
del sys.modules["perf_takehome"]

HEIGHT, ROUNDS, BATCH = 10, 16, 256                      # do_kernel_test(10, 16, 256) in tests/submission_tests.py
N_NODES = 2 ** (HEIGHT + 1) - 1
BASELINE = submission_tests.BASELINE                     # 147734, the starter code
# the SpeedTests of tests/submission_tests.py, in file order: each passes when cycles() is strictly below its number
THRESHOLDS = [147734, 18532, 2164, 1790, 1579, 1548, 1487, 1363]
WHO = {147734: "起始代码", 18532: "第 2 版的起点", 2164: "Opus 4 跑了很多小时",
       1790: "Opus 4.5 随手一次会话，约等于人类 2 小时的最好成绩", 1579: "Opus 4.5 在框架里跑 2 小时",
       1548: "Sonnet 4.5 远多于 2 小时", 1487: "Opus 4.5 在框架里跑 11.5 小时，公开挑战的门槛",
       1363: "Opus 4.5 改进后的框架"}


# ---------------------------------------------------------------- the scoring code must stay as it is
def _snapshot():
    fp = frozen_problem
    return {
        "frozen_problem": dict(vars(fp)), "frozen_problem.Machine": dict(vars(fp.Machine)),
        "frozen_problem.Tree": dict(vars(fp.Tree)), "frozen_problem.Input": dict(vars(fp.Input)),
        "submission_tests": {k: v for k, v in vars(submission_tests).items() if k != "KernelBuilder"},
        "random": {"seed": random.seed, "randint": random.randint},
    }


def _contents():
    return {"frozen_problem.SLOT_LIMITS": dict(frozen_problem.SLOT_LIMITS),
            "frozen_problem.HASH_STAGES": list(frozen_problem.HASH_STAGES)}


ORIGINAL, CONTENTS = _snapshot(), _contents()


def check_untouched():
    """The statement forbids changing anything under tests/. The answer is loaded into the same process as the
    harness, so it could patch the frozen machine at run time; refuse to score if it did."""
    now, contents = _snapshot(), _contents()
    changed = [f"{where}.{k}" for where, names in ORIGINAL.items()
               for k in names.keys() | now[where].keys()
               if k != "__warningregistry__" and names.get(k) is not now[where].get(k)]
    changed += [name for name, value in CONTENTS.items() if contents[name] != value]
    if changed:
        raise Mismatch(f"答案在运行时改动了计分程序：{'、'.join(sorted(changed)[:5])}。题目规定不能改 tests/ 里的任何东西"
                       "（计分用的冻结模拟器 tests/frozen_problem.py 和 tests/submission_tests.py），这样跑出来的结果不算数",
                       show=False)


# ---------------------------------------------------------------- one run of the official harness
def plain(instrs):
    """A copy of the program made only of lists, dicts, tuples, str and int, so reading it runs no answer code."""
    def value(x):
        return str(x) if isinstance(x, str) else int(x) if isinstance(x, int) else x
    try:
        return [{str(engine): [tuple(value(x) for x in slot) for slot in slots] for engine, slots in bundle.items()}
                for bundle in instrs]
    except Exception as e:      # noqa: BLE001
        raise Mismatch(f"build_kernel 之后 self.instrs 应当是指令束的列表，每个指令束是 {{引擎名: [槽位元组, …]}} 的字典"
                       f"（{type(e).__name__}: {e}）", "kb.instrs", show=False) from None


class Built:
    """What the harness's kernel_builder makes and caches: the answer's KernelBuilder, built once, with its program
    and debug info copied out. do_kernel_test reads kb.instrs and calls kb.debug_info() after it has generated the
    inputs; on the answer's own object that would run answer code while `mem` sits in the harness's frame, and such
    code could rewrite the program, or the inputs, to fit (see wrong/debug_info_*.py). The frozen machine only reads
    the program, and the debug info only for traces, so for an honest answer nothing changes."""

    def __init__(self, answer_class):
        self.answer_class = answer_class

    def build_kernel(self, *args):
        kb = self.answer_class()
        kb.build_kernel(*args)
        self.instrs = plain(kb.instrs)
        self.info = kb.debug_info()

    def debug_info(self):
        return self.info


def kernel(sol):
    """The answer's kernel, built the way the harness builds it (and cached there): errors in build_kernel or
    debug_info come out as the answer's own traceback."""
    submission_tests.KernelBuilder = lambda: Built(sol.KernelBuilder)
    return submission_tests.kernel_builder(HEIGHT, N_NODES, BATCH, ROUNDS)


def from_answer(exc):
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "/work/answer.py":
            return True
        tb = tb.tb_next
    return False


def official_run(sol, seed):
    """submission_tests.do_kernel_test(10, 16, 256) on the inputs that follow random.seed(seed) (the harness itself
    does not seed); returns the cycle count. Wrong values, or anything the frozen machine rejects, is a wrong answer."""
    kb = kernel(sol)
    check_untouched()
    random.seed(seed)
    try:
        return submission_tests.do_kernel_test(HEIGHT, ROUNDS, BATCH)
    except Exception as e:      # noqa: BLE001 - TimeUp is a BaseException and passes through
        if from_answer(e):
            raise
        raise Mismatch(diagnose(seed, kb, e), f"random.seed({seed}); do_kernel_test({HEIGHT}, {ROUNDS}, {BATCH})"
                       "（tests/submission_tests.py）", show=False) from None


def diagnose(seed, kb, err):
    """Run the same inputs again, step by step as do_kernel_test does, and say what went wrong."""
    fp = frozen_problem
    random.seed(seed)
    forest = fp.Tree.generate(HEIGHT)
    inp = fp.Input.generate(forest, BATCH, ROUNDS)
    mem = fp.build_mem_image(forest, inp)
    machine = fp.Machine(mem, kb.instrs, kb.debug_info(), n_cores=fp.N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    where = {}
    step = machine.step

    def traced(instr, core):    # on this instance only: remember which bundle is running (a jump moves pc)
        where["k"], where["bundle"] = core.pc - 1, instr
        return step(instr, core)

    machine.step = traced
    try:
        machine.run()
    except Exception as e:      # noqa: BLE001
        k, bundle = where.get("k", machine.cores[0].pc), where.get("bundle")
        error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        text = f"冻结的模拟器执行 instrs[{k}] 时出错（之前已经跑完 {machine.cycle} 个周期）：{error}"
        if isinstance(e, IndexError):
            text += "（地址越界：scratch 只有 SCRATCH_SIZE 个字，内存也有边界；也可能是读到了还没写好的地址值，" \
                    "同一个指令束里的槽位读的是这个周期开始时的值）"
        if isinstance(bundle, dict):
            over = [f"{name} 引擎放了 {len(slots)} 个槽位，每个周期最多 {fp.SLOT_LIMITS[name]} 个"
                    for name, slots in bundle.items()
                    if name in fp.SLOT_LIMITS and hasattr(slots, "__len__") and len(slots) > fp.SLOT_LIMITS[name]]
            if over:
                text += "。这个指令束里 " + "，".join(over)
            shown = repr(bundle)
            text += f"。这个指令束是 {shown[:300]}{' …' if len(shown) > 300 else ''}"
        return text
    for ref_mem in fp.reference_kernel2(mem):
        pass
    p = ref_mem[6]
    want, got = ref_mem[p:p + BATCH], machine.mem[p:p + BATCH]
    if got == want:
        return f"官方测试报错（{type(err).__name__}: {err}），但同样的输入重跑一遍结果是对的：程序的行为是不是不确定？"
    bad = [i for i in range(BATCH) if got[i] != want[i]]
    text = (f"Incorrect output values：程序结束时 inp_values_p（mem[{p}]）开始的 {BATCH} 个值里有 {len(bad)} 个和参考实现"
            f"不一致，第一个是第 {bad[0]} 个：期望 {want[bad[0]]}，得到 {got[bad[0]]}")
    if got == inp.values:
        text += "。这一段和输入完全一样：程序没有把结果写回内存"
    return text


def report(cycles):
    speedup = f"，快 {BASELINE / cycles:.2f} 倍" if cycles else ""
    info(f"周期数 {cycles:,}（起始代码 {BASELINE:,}{speedup}）")
    passed = [t for t in THRESHOLDS if cycles < t]
    info(f"严格低于 {len(passed)}/8 个官方门槛：" + "，".join(f"{t} {'✓' if cycles < t else '✗'}" for t in THRESHOLDS))
    nxt = next((t for t in THRESHOLDS if cycles >= t), None)
    if nxt is not None:
        info(f"下一个门槛 {nxt}（{WHO[nxt]}）；门槛只是参考，不影响这里的评测结果")


# ---------------------------------------------------------------- cases
@case("Part 1", sample=True, cpu=10, title="固定种子 123：完整规模跑一次")
def fixed_seed(sol):
    # random.seed(123) gives the same inputs as `python perf_takehome.py Tests.test_kernel_cycles`
    report(official_run(sol, 123))


@case("Part 1", cpu=40, title="官方正确性测试：8 组随机输入")
def correctness(sol):
    r = rng("correctness")
    counts = [official_run(sol, r.getrandbits(32)) for _ in range(8)]
    same = len(set(counts)) == 1
    info(f"8 次的周期数都是 {counts[0]:,}" if same else f"8 次的周期数从 {min(counts):,} 到 {max(counts):,}")


@case("Part 1", cpu=10, title="官方计时：周期数和 8 个门槛")
def timed(sol):
    # the harness's cycles() runs do_kernel_test once more and scores that run; a wrong result there would count as
    # 2 * BASELINE, which official_run reports as a wrong answer instead
    report(official_run(sol, rng("timed").getrandbits(32)))
