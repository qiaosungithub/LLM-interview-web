"""Tests for the corrected checkpointing.py of Part 2.

Part 1 (the ranked review), Part 3 (the user's own tests) and the prose Part 2 asks for (why this serialisation, the
trust assumption, how the leftover indices are distributed) are not judged. What is judged is the behaviour Part 2
requires, through the module's public API only:

- Which files make up a checkpoint is never read from their names, since the fix may change the format: a step owns
  the regular files its save_checkpoint call created and no later call changed.
- A "kill" is a forked child running save_checkpoint under RLIMIT_FSIZE = k with SIGXFSZ wired straight to _exit, so
  the process dies the moment any file it writes reaches k bytes and no except/finally clause runs -- what a SIGKILL
  or a power cut partway through the write does. The limits are fractions of the size an uninterrupted save writes.
  The limit is also armed only when the save opens its 2nd, 3rd, ... file under d for writing, so a kill lands inside
  every file, including a small one written after a big one ("any moment" in the statement).
- States hold only numpy arrays and int/float scalars, which every reasonable serialisation keeps exactly.
"""
import ctypes
import hashlib
import inspect
import os
import resource
import shutil
import signal
import stat
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping

import numpy as np

from judgelib import Mismatch, calling, case, expect_true, rng, short


# ---------------------------------------------------------------- states

def small_state(step):
    return {"w": np.arange(3.0) + step, "loss": 1.0 / step}


SMALL_CODE = "{'w': np.arange(3.0) + step, 'loss': 1.0 / step}"
BIG = 4096      # a checkpoint of ~33 KB, so a kill can land well inside the write


def big_state(step):
    return {"w": np.arange(float(BIG)) + step, "epoch": step}


def big_code(step):
    return f"{{'w': np.arange({BIG}.0) + {step}, 'epoch': {step}}}"


def code_of(v):
    if isinstance(v, np.ndarray):
        dt = "" if v.dtype == np.float64 else f", dtype=np.{v.dtype}"
        return f"np.array({v.tolist()!r}{dt})"
    return repr(v)


def state_code(state):
    return "{" + ", ".join(f"{k!r}: {code_of(v)}" for k, v in state.items()) + "}"


def copy_state(state):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}


def same_state(got, want):
    """None when got equals the saved state (arrays by shape and value, scalars by ==), else a Chinese reason."""
    if got is None:
        return "返回了 None"
    if not isinstance(got, Mapping):
        return f"应当返回一个 dict，得到 {type(got).__name__}"
    if set(got) != set(want):
        return f"键应当是 {sorted(want)}，得到 {sorted(map(str, got))}"
    for k, w in want.items():
        g = got[k]
        try:
            if isinstance(w, np.ndarray):
                a = np.asarray(g)
                ok = a.shape == w.shape and bool(np.array_equal(a, w))
            else:
                ok = np.ndim(g) == 0 and bool(g == w)
        except Exception:       # noqa: BLE001 - a value that cannot even be compared is simply wrong
            ok = False
        if not ok:
            return f"键 {k!r} 应当是 {short(w, 300)}，得到 {short(g, 300)}"
    return None


def err(e):
    text = f"{type(e).__name__}: {e}"
    return text if len(text) <= 300 else text[:300] + "…"


# ---------------------------------------------------------------- the directory

def snapshot(d):
    """{relative path: (inode, size, sha1)} for every regular file under d."""
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.lstat(p)
                if not stat.S_ISREG(st.st_mode):
                    continue
                with open(p, "rb") as fh:
                    digest = hashlib.sha1(fh.read()).hexdigest()
            except OSError:
                continue
            out[os.path.relpath(p, d)] = (st.st_ino, st.st_size, digest)
    return out


def listing(d):
    snap = snapshot(d)
    return "，".join(f"{p}（{snap[p][1]} 字节）" for p in sorted(snap)) or "（空）"


class Run:
    """A checkpoint directory driven through the answer's API, with a replay of the calls for messages."""

    def __init__(self, sol):
        self.sol, self.d = sol, tempfile.mkdtemp()
        self.lines = ["d = tempfile.mkdtemp()"]
        self.owned, self.states = {}, {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.d, ignore_errors=True)

    def replay(self):
        lines = self.lines if len(self.lines) <= 14 else ["# ……（前面省略）"] + self.lines[-13:]
        return "\n".join(lines)

    def step(self, line):
        self.lines.append(line)
        calling(self.replay())

    def save(self, step, state, code=None, metadata=None, line=True):
        if line:
            kw = "" if metadata is None else f", metadata={metadata!r}"
            self.step(f"save_checkpoint(d, {step}, {code or state_code(state)}{kw})")
        before = snapshot(self.d)
        if metadata is None:
            self.sol.save_checkpoint(self.d, step, copy_state(state))
        else:
            self.sol.save_checkpoint(self.d, step, copy_state(state), metadata=dict(metadata))
        after = snapshot(self.d)
        new = {p: sig for p, sig in after.items() if p not in before}
        expect_true(new, f"save_checkpoint(d, {step}, …) 之后目录里没有出现新文件：每次保存都应当写出这一步的 checkpoint 文件",
                    call=self.replay())
        for files in self.owned.values():   # a file a later save rewrites is shared bookkeeping, not one checkpoint's
            for p in [p for p, sig in files.items() if after.get(p) != sig]:
                del files[p]
        self.owned[step], self.states[step] = new, state

    def save_all(self, steps, pause=0.0):
        loop = f"for step in {steps!r}:\n    save_checkpoint(d, step, {SMALL_CODE})"
        if pause:
            loop = f"import time\n{loop}\n    time.sleep({pause})   # 每个文件的修改时间都不同"
        self.lines.append(loop)
        for s in steps:
            calling(self.replay() + f"\n# 正在保存 step = {s}")
            self.save(s, small_state(s), line=False)
            time.sleep(pause)

    def load(self):
        """(True, value) or (False, exception) from load_latest(d)."""
        self.step("load_latest(d)")
        try:
            return True, self.sol.load_latest(self.d)
        except Exception as e:      # noqa: BLE001 - whether raising is acceptable depends on the case
            return False, e

    def which(self, got):
        return next((s for s, st in self.states.items() if got is not None and same_state(got, st) is None), None)

    def outcome(self, ok, got):
        if not ok:
            return f"抛出了 {err(got)}"
        if got is None:
            return "返回了 None"
        s = self.which(got)
        return f"返回了第 {s} 步保存的 state" if s is not None else f"返回了 {short(got, 300)}"

    def expect_load(self, step, hint=""):
        ok, got = self.load()
        want = self.states[step]
        if not ok:
            raise Mismatch(f"load_latest(d) 应当返回第 {step} 步保存的 state，却抛出了 {err(got)}", self.replay(), want,
                           err(got))
        reason = same_state(got, want)
        if reason:
            other = self.which(got)
            if other is not None:
                reason = f"返回的却是第 {other} 步保存的 state"
            raise Mismatch(f"load_latest(d) 应当返回第 {step} 步保存的 state：{reason}{hint}", self.replay(), want, got)

    def retain(self, keep, check_files=True):
        self.step(f"apply_retention(d, keep={keep})")
        self.sol.apply_retention(self.d, keep=keep)
        if not check_files:
            return
        now = snapshot(self.d)
        steps = sorted(self.owned)
        kept, dropped = steps[-keep:], steps[:-keep]
        names = "、".join(map(str, kept))
        for s in kept:
            gone = sorted(p for p in self.owned[s] if p not in now)
            if gone:
                raise Mismatch(f"apply_retention(d, keep={keep}) 应当保留最新的 {keep} 个 checkpoint（第 {names} 步），"
                               f"却删掉了第 {s} 步写出的 {', '.join(gone)}。目录里现在有：{listing(self.d)}",
                               self.replay(), show=False)
        for s in dropped:
            left = sorted(p for p in self.owned[s] if p in now)
            if left:
                raise Mismatch(f"apply_retention(d, keep={keep}) 应当只保留最新的 {keep} 个 checkpoint（第 {names} 步），"
                               f"第 {s} 步写出的 {', '.join(left)} 却还在", self.replay(), show=False)
            del self.owned[s]

    def corrupt(self, step, how):
        """Truncates every file step `step` wrote: to half its length ("half") or to nothing ("empty")."""
        files = sorted(self.owned[step])
        for p in files:
            full = os.path.join(self.d, p)
            os.truncate(full, os.path.getsize(full) // 2 if how == "half" else 0)
        done = "截掉了后一半" if how == "half" else "清空了"
        self.lines.append(f"# 把第 {step} 步写出的 {', '.join(files)} {done}")


# ---------------------------------------------------------------- a process killed partway through a save

_LIBC = ctypes.CDLL(None)
_LIBC.signal.restype = ctypes.c_void_p
_LIBC.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
_EXIT = ctypes.cast(_LIBC._exit, ctypes.c_void_p)
KILLED = int(signal.SIGXFSZ)        # the child's exit status when the size limit stopped it


def _on_write_open(d, action):
    """An audit hook calling action(n) when the process opens its n-th file under d for writing (n = 1, 2, ...)."""
    root, count = os.path.join(os.path.realpath(d), ""), [0]

    def hook(event, args):
        try:
            if event != "open" or isinstance(args[0], int) or not (args[2] & (os.O_WRONLY | os.O_RDWR)):
                return
            if os.path.realpath(os.fsdecode(args[0])).startswith(root):
                count[0] += 1
                action(count[0])
        except Exception:       # noqa: BLE001 - never disturb the answer's own open()
            pass
    return hook


def killed_save(sol, d, step, state, limit, nth=1):
    """Runs save_checkpoint(d, step, state) in a forked child that dies, running nothing more, as soon as any file it
    writes reaches `limit` bytes -- counting from the start, or (nth > 1) from the moment it opens its nth file under d
    for writing. Returns the child's exit code: KILLED, 0 (finished under the limit), 3 (raised); with limit None, no
    kill and 10 + the number of files it opened under d for writing."""
    pid = os.fork()
    if pid == 0:
        code = 4
        try:
            _LIBC.signal(signal.SIGXFSZ, _EXIT)     # the kernel's SIGXFSZ goes straight to _exit(25)
            opened = [0]
            if limit is None:       # no kill: count the files the save opens for writing
                sys.addaudithook(_on_write_open(d, lambda n: opened.__setitem__(0, n)))
            elif nth == 1:
                resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
            else:
                sys.addaudithook(_on_write_open(d, lambda n: n == nth and resource.setrlimit(
                    resource.RLIMIT_FSIZE, (limit, limit))))
            code = 3
            sol.save_checkpoint(d, step, copy_state(state))
            code = 0 if limit is not None else 10 + min(opened[0], MAX_FILES)
        finally:
            os._exit(code)
    try:
        _, status = os.waitpid(pid, 0)
        pid = None
        code = os.waitstatus_to_exitcode(status)
    finally:
        if pid is not None:     # the case ran out of time while waiting
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except OSError:
                pass
    if code == 4:
        raise RuntimeError("the judge could not set up the simulated kill")
    return code


MAX_FILES = 6     # kills armed from the 1st .. 6th file a save opens for writing


def kill_points(sol, prior, step):
    """(nth, limit) pairs for killed_save: byte limits inside the largest file an uninterrupted save of big_state(step)
    writes after `prior` steps, armed from the start and from each later file the save opens for writing."""
    with Run(sol) as run:
        for s in prior:
            run.save(s, big_state(s), big_code(s))
        run.save(step, big_state(step), big_code(step))
        size = max(sig[1] for sig in run.owned[step].values())
    with Run(sol) as run:
        for s in prior:
            run.save(s, big_state(s), big_code(s))
        files = max(1, killed_save(sol, run.d, step, big_state(step), None) - 10)
    return [(nth, k) for nth in range(1, files + 1) for k in sorted({0, 1, size // 4, size // 2, 3 * size // 4})]


def at(nth, k):
    where = "" if nth == 1 else f"打开第 {nth} 个要写入的文件之后，"
    return where + ("刚要写入第一个字节时" if k == 0 else f"某个文件写到 {k} 字节时")


def kill_line(step, nth, k):
    where = "" if nth == 1 else f"它在 d 里打开第 {nth} 个要写入的文件之后，"
    return (f"# 子进程调用 save_checkpoint(d, {step}, {big_code(step)})，{where}"
            f"{'任何文件刚要写入第一个字节时' if k == 0 else f'任何文件写到 {k} 字节时'}进程立刻被杀（不运行任何清理代码）")


NO_KILL = "评测没能在保存中途结束进程：在子进程里调用 save_checkpoint 时，它每次都没写文件就结束了，或者抛出了异常"


# ---------------------------------------------------------------- Part 2: checkpointing

@case("Part 2", sample=True, title="题目例子：保存、读取、只留 1 个")
def p2_example(sol):
    with Run(sol) as run:
        run.save(1, {"loss": 0.9}, "{'loss': 0.9}")
        run.save(2, {"loss": 0.7}, "{'loss': 0.7}")
        run.expect_load(2)
        run.retain(1)
        run.expect_load(2)


@case("Part 2", title="数组和数值存进去再原样读出")
def p2_round_trip(sol):
    g = np.random.default_rng(rng().getrandbits(64))
    with Run(sol) as run:
        ok, got = run.load()
        expect_true(ok and got is None, f"目录里还没有任何 checkpoint 时，load_latest(d) 应当返回 None，却{run.outcome(ok, got)}",
                    call=run.replay())
        for i, step in enumerate((1, 2, 3, 5)):
            state = {"w": g.normal(size=(3, 2)).round(4), "b": g.normal(size=4).astype(np.float32),
                     "counts": g.integers(0, 1000, size=5), "loss": round(float(g.random()), 6), "epoch": step}
            run.save(step, state, metadata={"note": "run-a", "lr": 0.01} if i % 2 == 0 else None)
            run.expect_load(step)


@case("Part 2", title="步数跨过 9→10、999→1000 时")
def p2_step_order(sol):
    for steps in (list(range(1, 11)), [7, 99, 100, 998, 1000, 1001]):
        with Run(sol) as run:
            run.save_all(steps)
            run.expect_load(steps[-1])
            for keep in (3, 1):
                run.retain(keep)
                run.expect_load(steps[-1])


@case("Part 2", title="后保存的 step 反而更小时")
def p2_step_not_save_time(sol):
    # Sorting by modification time agrees with sorting by step whenever steps only grow, so save them out of order.
    with Run(sol) as run:
        run.save_all([30, 10, 20], pause=0.02)
        hint = "。题目要求按 step 排序 checkpoint：最新的是 step 最大的，不是最后保存（修改时间最晚）的"
        run.expect_load(30, hint)
        for keep in (2, 1):
            run.retain(keep)
            run.expect_load(30, hint)


@case("Part 2", title="第一次保存时进程被杀")
def p2_crash_first_save(sol):
    checked = 0
    for nth, k in kill_points(sol, [], 1):
        with Run(sol) as run:
            run.states[1] = big_state(1)
            run.step(kill_line(1, nth, k))
            if killed_save(sol, run.d, 1, big_state(1), k, nth) in (0, 3):
                continue        # the limit never stopped it
            checked += 1
            ok, got = run.load()
            if not (ok and (got is None or run.which(got) == 1)):
                raise Mismatch(f"保存第一个 checkpoint 时，进程在{at(nth, k)}被杀。调用前目录里没有 checkpoint，"
                               f"load_latest(d) 应当返回 None（新的 checkpoint 已经完整写好时返回第 1 步的 state 也算对），"
                               f"却{run.outcome(ok, got)}（被杀时留下的文件被当成了 checkpoint）。目录里现在有：{listing(run.d)}",
                               run.replay(), show=False)
    expect_true(checked, NO_KILL)


@case("Part 2", title="已有 checkpoint 时保存中途被杀")
def p2_crash_later_save(sol):
    checked = 0
    for nth, k in kill_points(sol, [1, 2], 3):
        with Run(sol) as run:
            run.save(1, big_state(1), big_code(1))
            run.save(2, big_state(2), big_code(2))
            run.states[3] = big_state(3)
            run.step(kill_line(3, nth, k))
            if killed_save(sol, run.d, 3, big_state(3), k, nth) in (0, 3):
                continue
            checked += 1
            ok, got = run.load()
            now = run.which(got) if ok else None
            if now not in (2, 3):
                raise Mismatch(f"保存第 3 步时，进程在{at(nth, k)}被杀。第 1、2 步的 checkpoint 应当原封不动，"
                               f"load_latest(d) 应当返回第 2 步保存的 state（新的 checkpoint 已经完整写好时返回第 3 步的也算对），"
                               f"却{run.outcome(ok, got)}。目录里现在有：{listing(run.d)}", run.replay(), show=False)
            run.retain(1, check_files=False)
            ok, got = run.load()
            if not ok or run.which(got) != now:
                raise Mismatch(f"保存第 3 步时进程在{at(nth, k)}被杀，之后 load_latest 返回第 {now} 步的 state；"
                               f"再调用 apply_retention(d, keep=1) 只保留最新的一个 checkpoint，load_latest(d) 应当仍返回"
                               f"第 {now} 步的 state，却{run.outcome(ok, got)}（被杀时留下的文件被当成了最新的 checkpoint，"
                               f"完好的那个被删掉了）。目录里现在有：{listing(run.d)}", run.replay(), show=False)
    expect_true(checked, NO_KILL)


@case("Part 2", title="最新的 checkpoint 读不出来")
def p2_newest_unreadable(sol):
    for how in ("half", "empty"):
        with Run(sol) as run:
            run.save_all([1, 2, 3])
            for bad, fallback in ((3, 2), (2, 1)):
                run.corrupt(bad, how)
                ok, got = run.load()
                if ok and run.which(got) != fallback:
                    damaged = "第 3 步" if bad == 3 else "第 3、2 步"
                    raise Mismatch(f"{damaged}的 checkpoint 文件坏了之后，load_latest(d) 可以回退到最新的完好 checkpoint"
                                   f"（第 {fallback} 步），也可以抛出异常报错，却{run.outcome(ok, got)}。"
                                   f"None 只表示目录里从来没有保存过 checkpoint", run.replay(), show=False)


@case("Part 2", title="所有 checkpoint 都读不出来")
def p2_all_unreadable(sol):
    with Run(sol) as run:
        ok, got = run.load()
        expect_true(ok and got is None, f"目录里从来没有保存过 checkpoint 时，load_latest(d) 应当返回 None，"
                                        f"却{run.outcome(ok, got)}", call=run.replay())
    for steps, how in (([1], "half"), ([1, 2], "empty")):
        with Run(sol) as run:
            run.save_all(steps)
            for s in steps:
                run.corrupt(s, how)
            ok, got = run.load()
            expect_true(not (ok and got is None),
                        "目录里的 checkpoint 全都读不出来时，load_latest(d) 返回了 None：这和“从来没有保存过 checkpoint”"
                        "没法区分，训练会悄悄从头开始。题目接受大声报错（抛出异常）或有说明的回退，但不能不声不响",
                        call=run.replay())


@case("Part 2", title="metadata 不能用可变默认值")
def p2_mutable_default(sol):
    call = "inspect.signature(save_checkpoint).parameters['metadata'].default"
    try:
        params = inspect.signature(sol.save_checkpoint).parameters
    except (TypeError, ValueError):
        params = {}
    p = params.get("metadata")
    expect_true(p is not None, "save_checkpoint 应当保留参数 metadata：题目只允许改动缺陷逼着要改的签名部分", call=call)
    expect_true(p.default is not inspect.Parameter.empty,
                "metadata 应当仍是可选参数：题目的用法 save_checkpoint(run_dir, step, state) 不传它", call=call)
    expect_true(not isinstance(p.default, (dict, list, set, bytearray)),
                f"metadata 的默认值仍是可变的 {type(p.default).__name__}（{p.default!r}）：所有不传 metadata 的调用共用同一个"
                f"对象。题目要求去掉这个可变默认参数（例如改成 None），即使函数里先复制一份再改也不行", call=call)
    fn = sol.save_checkpoint
    before = repr((getattr(fn, "__defaults__", None), getattr(fn, "__kwdefaults__", None)))
    with Run(sol) as run:
        run.save_all([1, 2, 3])
    after = repr((getattr(fn, "__defaults__", None), getattr(fn, "__kwdefaults__", None)))
    expect_true(after == before, f"不传 metadata 调用三次 save_checkpoint 之后，它的默认参数从 {before} 变成了 {after}",
                call=run.replay())


# ---------------------------------------------------------------- Part 2: ShardedSampler

def index_list(got, call):
    try:
        items = list(got)
    except TypeError:
        raise Mismatch("indices() 应当返回一个下标列表", call, None, got) from None
    if not all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) for i in items):
        raise Mismatch("indices() 应当返回整数下标组成的列表", call, None, got)
    return [int(i) for i in items]


def indices(sol, n, rank, ws, seed, epoch):
    call = calling(f"ShardedSampler({n}, {rank}, {ws}, {seed}, {epoch}).indices()")
    return index_list(sol.ShardedSampler(n, rank, ws, seed, epoch).indices(), call)


SAMPLER_CONFIGS = [(n, ws, seed, epoch) for seed, epoch in ((0, 0), (7, 3))
                   for n in (1, 2, 5, 16, 17, 23, 40, 101) for ws in (1, 2, 3, 4, 7, 8)]


@case("Part 2", title="各 worker 的分片恰好覆盖一次")
def p2_sampler_cover(sol):
    for n, ws, seed, epoch in SAMPLER_CONFIGS:
        parts = [indices(sol, n, r, ws, seed, epoch) for r in range(ws)]
        counts = Counter(i for part in parts for i in part)
        if sorted(counts.elements()) != list(range(n)):
            missing = [i for i in range(n) if i not in counts]
            dup = sorted(i for i, c in counts.items() if c > 1)
            bad = sorted(i for i in counts if not 0 <= i < n)
            reason = "；".join(x for x in (f"缺了 {missing}" if missing else "", f"重复了 {dup}" if dup else "",
                                           f"超出范围的 {bad}" if bad else "") if x)
            raise Mismatch(f"num_examples={n}、world_size={ws} 时，{ws} 个 worker 的 indices() 合起来应当恰好覆盖 "
                           f"range({n}) 的每个下标一次：{reason}",
                           f"[ShardedSampler({n}, r, {ws}, {seed}, {epoch}).indices() for r in range({ws})]", None, parts)


@case("Part 2", title="indices() 多次调用结果相同")
def p2_sampler_repeatable(sol):
    for n, ws, seed, epoch in ((23, 4, 0, 0), (40, 3, 7, 3), (17, 7, 1, 2)):
        for r in range(ws):
            call = calling(f"s = ShardedSampler({n}, {r}, {ws}, {seed}, {epoch})\n[s.indices(), s.indices()]")
            s = sol.ShardedSampler(n, r, ws, seed, epoch)
            first, second = index_list(s.indices(), call), index_list(s.indices(), call)
            expect_true(first == second, "同一个对象第二次调用 indices() 得到了不同的列表：indices() 返回的是由 seed 和 epoch "
                                         "算出的那一次 shuffle 里分给这个 worker 的部分，参数不变就应当每次相同",
                        call=call, got=[first, second])
            again = indices(sol, n, r, ws, seed, epoch)
            expect_true(again == first, "用同样的参数新建一个 ShardedSampler，indices() 得到了不同的列表",
                        call=f"[ShardedSampler({n}, {r}, {ws}, {seed}, {epoch}).indices() for _ in range(2)]",
                        got=[first, again])
