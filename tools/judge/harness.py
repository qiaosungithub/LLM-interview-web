"""Runs inside the judge sandbox (started by run.py, never directly by the server).

Loads the answer (/work/answer.py) as the module `sol`, runs the cases registered by /judge/tests.py, and writes
one JSON object per line to the result channel (the original stdout). Everything the answer prints is captured
per case; anything written straight to file descriptors 1 and 2 is discarded.

Events, in order:  {"type": "plan", "cases": [...]}  then either {"type": "ce" | "load", ...} or, per case,
{"type": "start", "i": n} and {"type": "case", ...};  finally {"type": "done"}.  In scratch mode the answer runs
as __main__ followed by /work/scratch.py, reported as one {"type": "scratch", ...} event.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import io
import json
import os
import resource
import re
import signal
import sys
import threading
import time
import traceback
import types

USER_FILES = {"/work/answer.py": "answer.py", "/work/scratch.py": "scratch.py"}
TEXT_LIMIT = 16000


class TimeUp(BaseException):
    """Raised by the timer; a BaseException so that `except Exception` in the answer does not swallow it."""


class Timer:
    def __init__(self):
        self.armed = False
        self.fired = ""
        signal.signal(signal.SIGPROF, self._on)
        signal.signal(signal.SIGALRM, self._on)

    def _on(self, signum, frame):
        if not self.armed:
            return
        self.fired = self.fired or ("cpu" if signum == signal.SIGPROF else "wall")
        raise TimeUp()

    def arm(self, cpu: float, wall: float):
        self.fired = ""
        self.armed = True
        # the interval re-fires the signal, so an answer that catches TimeUp with a bare `except:` is stopped again
        signal.setitimer(signal.ITIMER_PROF, cpu, 0.25)
        signal.setitimer(signal.ITIMER_REAL, wall, 0.25)

    def disarm(self):
        self.armed = False
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.setitimer(signal.ITIMER_REAL, 0)


class Capture(io.TextIOBase):
    """sys.stdout / sys.stderr during a case: keeps the first TEXT_LIMIT characters, counts the rest."""

    def __init__(self):
        self.parts: list[str] = []
        self.size = 0
        self.dropped = 0

    def write(self, s):
        if not isinstance(s, str):
            raise TypeError(f"write() argument must be str, not {type(s).__name__}")
        room = TEXT_LIMIT - self.size
        if room > 0:
            self.parts.append(s[:room])
            self.size += min(len(s), room)
        if len(s) > room:
            self.dropped += len(s) - max(room, 0)
        return len(s)

    def writable(self):
        return True

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"

    def text(self) -> str:
        out = "".join(self.parts)
        if self.dropped:
            out += f"\n…（还有 {self.dropped} 个字符的输出没有显示）"
        return out


def cut(text, limit=3000):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f" …（共 {len(text)} 个字符，已截断）"


def user_trace(exc: BaseException) -> tuple[str, list[int]]:
    """The traceback restricted to the answer's own frames, and the answer.py line numbers in it."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename in USER_FILES]
    lines = [f.lineno for f in frames if f.filename == "/work/answer.py"]
    out = []
    if frames:
        out.append("Traceback（只保留你的代码）:")
        for f in frames:
            out.append(f'  File "{USER_FILES[f.filename]}", line {f.lineno}, in {f.name}')
            if f.line:
                out.append(f"    {f.line}")
    out.append("".join(traceback.format_exception_only(type(exc), exc)).strip())
    return cut("\n".join(out), 6000), lines


def test_line(exc: BaseException) -> str:
    """The line of the test that raised, e.g. the failing `assert`."""
    for f in reversed(traceback.extract_tb(exc.__traceback__)):
        if f.filename.startswith("/judge/"):
            return f"tests.py 第 {f.lineno} 行：{f.line}" if f.line else ""
    return ""


def has_user_frame(exc: BaseException) -> bool:
    return any(f.filename in USER_FILES for f in traceback.extract_tb(exc.__traceback__))


SIGNATURE = re.compile(r"positional arguments? but|unexpected keyword argument|missing \d+ required|takes no arguments")


def classify(exc: BaseException, sol, judgelib) -> dict:
    """Turn an exception raised by a case into a verdict and the details shown for it."""
    if isinstance(exc, judgelib.Mismatch):
        return {"verdict": "WA", "msg": cut(exc.msg), "call": cut(exc.call), "want": exc.want, "got": exc.got}
    if isinstance(exc, MemoryError):
        trace, lines = user_trace(exc)
        return {"verdict": "RE", "msg": "内存不够：用量超过了上限", "trace": trace, "lines": lines}
    if isinstance(exc, SystemExit):
        return {"verdict": "RE", "msg": f"代码调用了 exit()/sys.exit()（参数 {exc.code!r}）"}
    if isinstance(exc, AttributeError) and getattr(exc, "obj", None) is sol:
        return {"verdict": "RE", "msg": f"answer.py 里没有定义 `{exc.name}`（名字要和题目或初始代码一致）"}
    if has_user_frame(exc):
        trace, lines = user_trace(exc)
        msg = ("递归太深（超过 20000 层）" if isinstance(exc, RecursionError)
               else "还没有实现（抛出了 NotImplementedError）" if isinstance(exc, NotImplementedError) else "")
        return {"verdict": "RE", "msg": msg, "trace": trace, "lines": lines}
    # raised in the test itself: a plain assert, or the returned value could not even be checked
    where = test_line(exc)
    text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    if isinstance(exc, TypeError) and SIGNATURE.search(str(exc)):
        # the call never reached the answer: its function or method takes other parameters than the statement's
        return {"verdict": "RE", "msg": cut(f"参数和题目要求的不一致（{text}）"),
                "call": cut(judgelib._current.get("call") or where)}
    if isinstance(exc, AssertionError):
        return {"verdict": "WA", "msg": cut(str(exc) or "检查没有通过"), "call": cut(where)}
    return {"verdict": "WA", "msg": cut(f"检查返回值时出错：{text}"), "call": cut(where)}


def load_module(path: str, name: str, timer: Timer, cpu: float, extra: str | None = None) -> dict:
    """Compile and run the answer (and, in scratch mode, the extra code after it) under a time limit."""
    out = {"cpu": 0.0, "wall": 0.0}
    sources = [(path, open(path, encoding="utf-8").read())]
    if extra:
        sources.append((extra, open(extra, encoding="utf-8").read()))
    codes = []
    for file, src in sources:
        try:
            codes.append(compile(src, file, "exec"))
        except SyntaxError as e:
            label = USER_FILES.get(file, file)
            lines = [e.lineno] if e.lineno and file == "/work/answer.py" else []
            text = f'  File "{label}", line {e.lineno}\n    {(e.text or "").rstrip()}\n{type(e).__name__}: {e.msg}'
            return {"verdict": "CE", "trace": text, "lines": lines, "msg": f"{label} 第 {e.lineno} 行有语法错误"}
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    cap = Capture()
    sys.stdout = sys.stderr = cap
    t0, c0 = time.perf_counter(), time.process_time()
    verdict, detail = "OK", {}
    try:
        try:
            timer.arm(cpu, cpu * 3 + 2)
            for code in codes:
                exec(code, mod.__dict__)
        finally:
            timer.disarm()
    except TimeUp:
        timer.disarm()
        verdict, detail = "TLE", {"msg": f"超过 {cpu:g} 秒" + ("（可能死锁或在等待）" if timer.fired == "wall" else " CPU 时间")}
    except BaseException as e:      # noqa: BLE001 - any failure of the answer is reported, not raised
        trace, lines = user_trace(e)
        verdict = "RE"
        detail = {"trace": trace, "lines": lines}
        if isinstance(e, SystemExit):
            verdict = "OK" if e.code in (None, 0) else "RE"
            detail = {} if verdict == "OK" else {"msg": f"exit({e.code!r})"}
    out.update(verdict=verdict, module=mod, out=cap.text(), cpu=round(time.process_time() - c0, 3),
               wall=round(time.perf_counter() - t0, 3), **detail)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="/judge")
    ap.add_argument("--code", default="/work/answer.py")
    ap.add_argument("--mode", choices=["samples", "submit", "scratch", "list"], default="submit")
    ap.add_argument("--scratch", default="/work/scratch.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--case-cpu", type=float, default=2.0)
    ap.add_argument("--load-cpu", type=float, default=10.0)
    ap.add_argument("--nproc", type=int, default=0)
    args = ap.parse_args()

    if args.nproc:      # counted inside this sandbox only (threads count too)
        resource.setrlimit(resource.RLIMIT_NPROC, (args.nproc, args.nproc))
    # a new thread's stack defaults to RLIMIT_STACK (256 MB, kept for deep recursion on the main thread), which would
    # exhaust the address-space limit after ~25 threads; concurrency tests need a hundred or more
    import threading
    threading.stack_size(16 * 1024 * 1024)
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)       # an oversized write fails with an error instead of killing us
    chan = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)

    def emit(obj):
        chan.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        chan.flush()

    try:
        # The start method is left unset: fork is the default on Linux (Python 3.11), and an answer may still call
        # multiprocessing.set_start_method("spawn") itself. A spawned or forkserver worker is a new interpreter that
        # imports the answer's module by name to unpickle its functions, so "sol" must be importable as a file:
        # write a stub that runs the answer. It goes last on sys.path, so a judge's own sol.py (one that also
        # installs its counters in the worker) takes precedence.
        import tempfile
        stub_dir = tempfile.mkdtemp(prefix="sol_")
        with open(os.path.join(stub_dir, "sol.py"), "w", encoding="utf-8") as f:
            f.write(f"_p = {args.code!r}\n__file__ = _p\nwith open(_p, encoding='utf-8') as _f:\n"
                    f"    exec(compile(_f.read(), _p, 'exec'), globals())\n")
        sys.path.append(stub_dir)
        sys.setrecursionlimit(20000)
        sys.path.insert(0, args.judge)          # tests may import helpers kept next to them (ref.py, ...)
        import judgelib
        judgelib.SEED = args.seed
        spec = importlib.util.spec_from_file_location("tests", os.path.join(args.judge, "tests.py"))
        tests = importlib.util.module_from_spec(spec)
        sys.modules["tests"] = tests
        spec.loader.exec_module(tests)
    except BaseException as e:      # noqa: BLE001
        emit({"type": "je", "msg": "评测程序载入测试时出错", "trace": cut(traceback.format_exc(), 6000)})
        emit({"type": "done"})
        return

    timer = Timer()
    cases = list(judgelib.CASES)
    numbered, per_part = [], {}
    for i, c in enumerate(cases, 1):
        per_part[c.part] = per_part.get(c.part, 0) + 1
        numbered.append((i, per_part[c.part], c))
    if args.mode == "samples":
        numbered = [x for x in numbered if x[2].sample]
    if args.mode != "scratch":
        emit({"type": "plan", "cases": [{"i": i, "k": k, "part": c.part, "name": c.name, "title": c.title,
                                         "sample": c.sample} for i, k, c in numbered]})
    if args.mode == "list":
        emit({"type": "done"})
        return

    if args.mode == "scratch":
        sys.argv = ["answer.py"]        # as `python answer.py`: unittest.main() or argparse in the answer must not see ours
        r = load_module(args.code, "__main__", timer, args.load_cpu, extra=args.scratch)
        main_mod = sys.modules.get("__main__")
        r.pop("module", None)
        sys.stdout = sys.stderr = io.StringIO()
        emit({"type": "scratch", **r})
        emit({"type": "done"})
        del main_mod
        return

    loaded = load_module(args.code, "sol", timer, args.load_cpu)
    sol = loaded.pop("module", None)
    if loaded["verdict"] == "CE":
        emit({"type": "ce", **loaded})
        emit({"type": "done"})
        return
    if loaded["verdict"] != "OK":
        emit({"type": "load", **loaded})
        emit({"type": "done"})
        return
    if loaded["out"]:
        emit({"type": "loadout", "out": loaded["out"]})

    for i, k, c in numbered:
        emit({"type": "start", "i": i})
        cpu = c.cpu or args.case_cpu
        wall = cpu * 3 + 2
        cap = Capture()
        sys.stdout = sys.stderr = cap
        judgelib._current.update(case=c, info=[], call=None)
        threads_before = threading.active_count()
        gc.collect()
        t0, c0 = time.perf_counter(), time.process_time()
        result = {"verdict": "AC"}
        try:
            try:
                timer.arm(cpu, wall)
                c.fn(sol)
            finally:
                timer.disarm()
        except TimeUp:
            timer.disarm()
            result = {"verdict": "TLE", "msg": f"超过 {cpu:g} 秒 CPU 时间" if timer.fired == "cpu"
                      else f"超过 {wall:g} 秒还没结束（可能死锁，或者在等待永远不会发生的事）"}
        except BaseException as e:  # noqa: BLE001
            result = classify(e, sol, judgelib)
        if result["verdict"] in ("RE", "TLE") and not result.get("call") and judgelib._current.get("call"):
            result["call"] = cut(judgelib._current["call"])     # what was running when it failed (calling())
        used_cpu, used_wall = time.process_time() - c0, time.perf_counter() - t0
        sys.stdout = sys.stderr = io.StringIO()
        emit({"type": "case", "i": i, **result, "cpu": round(used_cpu, 3), "wall": round(used_wall, 3),
              "limit": cpu, "out": cap.text(), "info": [cut(x, 500) for x in judgelib._current["info"]][:20]})
        if result["verdict"] == "TLE" and threading.active_count() > threads_before:
            emit({"type": "stop", "msg": "超时的测试留下了仍在运行的线程，后面的测试没有运行"})
            break
    emit({"type": "done"})


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)     # threads the answer left running must not keep the sandbox alive
