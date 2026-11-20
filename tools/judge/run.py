#!/usr/bin/env python3
"""Run an answer against one problem's tests inside a bubblewrap sandbox. Used by server.py and judge_check.py.

    .venv/bin/python tools/judge/run.py <problem folder or its judge/ folder> <answer.py> [--mode samples|submit|scratch|list]
                                        [--scratch extra.py] [--seed N] [--json]

The sandbox has no network, sees only the system libraries, the project's Python, this folder (/harness), the
problem's judge folder (/judge, read-only), the answer (/work, read-only) and an empty writable /tmp. Limits: wall
time for the whole run (judge.yaml `timeout`, default 20 s, at most 120 s), CPU time per case (`case_cpu`), address
space (`memory_gb`, default 8), 64 MB per file and 512 MB in /tmp, `nproc` processes and threads, nice 10, two
threads for numeric libraries. Whatever happens inside, the result is a dict (see `judge`).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULTS = {"timeout": 20, "case_cpu": 2.0, "load_cpu": 10.0, "memory_gb": 8, "nproc": 256}
MAX_TIMEOUT = 120
TMP_BYTES = 512 * 1024 * 1024
TMP_FILES = 5000
FILE_BYTES = 64 * 1024 * 1024
CHANNEL_BYTES = 8 * 1024 * 1024
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
VERDICT_ZH = {"AC": "通过", "WA": "答案错误", "RE": "运行出错", "TLE": "超时", "CE": "语法错误", "JE": "评测程序出错",
              "OK": "运行完成", "skip": "未运行"}


def judge_dir_of(folder: Path) -> Path:
    folder = Path(folder)
    return folder if folder.name == "judge" else folder / "judge"


def config(judge_dir: Path) -> dict:
    cfg = dict(DEFAULTS)
    path = Path(judge_dir) / "judge.yaml"
    if path.exists():
        import yaml
        cfg.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    cfg["timeout"] = min(float(cfg["timeout"]), MAX_TIMEOUT)
    return cfg


def sandbox_cmd(judge_dir: Path, job: Path, mode: str, seed: int, cfg: dict) -> list[str]:
    py_home = ROOT / ".python"
    cmd = ["nice", "-n", "10", "bwrap",
           "--ro-bind", "/usr", "/usr",
           "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
           "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
           "--ro-bind-try", "/etc/localtime", "/etc/localtime",
           "--ro-bind", str(py_home), str(py_home),
           "--ro-bind", str(ROOT / ".venv"), str(ROOT / ".venv"),
           "--ro-bind", str(HERE), "/harness",
           "--ro-bind", str(judge_dir), "/judge"]
    # the full reference answer and the known-wrong answers are for judge_check.py only; tests must not use them
    if (judge_dir / "solution.py").exists():
        cmd += ["--ro-bind", str(job / "empty"), "/judge/solution.py"]
    if (judge_dir / "wrong").is_dir():
        cmd += ["--tmpfs", "/judge/wrong", "--remount-ro", "/judge/wrong"]
    cmd += ["--ro-bind", str(job / "work"), "/work",
           "--bind", str(job / "tmp"), "/tmp",
           "--proc", "/proc", "--dev", "/dev",
           "--unshare-all", "--die-with-parent", "--new-session",
           "--chdir", "/tmp", "--clearenv"]
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp", "LANG": "C.UTF-8", "PYTHONUTF8": "1",
           "PYTHONHASHSEED": str(seed), "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp",
           # glibc gives each thread its own 64 MB malloc arena (up to 8 per core, and this host has 240 cores), which
           # under the address-space limit would cap a program at ~100 threads
           "MALLOC_ARENA_MAX": "2"}
    env.update({v: "2" for v in THREAD_VARS})
    for k, v in env.items():
        cmd += ["--setenv", k, v]
    cmd += [str(PYTHON), "-s", "/harness/harness.py", "--mode", mode, "--seed", str(seed),
            "--case-cpu", str(cfg["case_cpu"]), "--load-cpu", str(cfg["load_cpu"]), "--nproc", str(int(cfg["nproc"]))]
    return cmd


def _limits(cfg: dict):
    mem = int(float(cfg["memory_gb"]) * 1024 ** 3)
    cpu = int(cfg["timeout"] * 3) + 10      # whole-process CPU seconds: a backstop behind the wall-clock limit

    def apply():
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_STACK, (256 * 1024 * 1024, 256 * 1024 * 1024))
    return apply


def _tmp_usage(path: Path) -> tuple[int, int]:
    total = count = 0
    for base, dirs, files in os.walk(path):
        for name in dirs + files:
            count += 1
            try:
                total += os.lstat(os.path.join(base, name)).st_size
            except OSError:
                pass
            if count > TMP_FILES:
                return total, count
    return total, count


def _remove(path: Path) -> None:
    def retry(func, p, _exc):
        try:
            os.chmod(os.path.dirname(p), stat.S_IRWXU)
            os.chmod(p, stat.S_IRWXU)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=retry)


def judge(judge_dir: Path, code: str, mode: str = "submit", scratch: str = "", seed: int = 0) -> dict:
    """Run `code` against the tests in `judge_dir`.

    Returns {"mode", "verdict", "verdict_zh", "passed", "total", "parts": [{"name", "passed", "total"}],
    "cases": [...], "first": i of the first failed case or None, "cpu", "wall", "msg", "trace", "lines", "out"}.
    In scratch mode "cases" is empty and "out" holds what the run printed.
    """
    judge_dir = Path(judge_dir).resolve()
    cfg = config(judge_dir)
    job = Path(tempfile.mkdtemp(prefix="interview_judge_"))
    try:
        (job / "work").mkdir()
        (job / "tmp").mkdir()
        (job / "empty").touch()
        (job / "work" / "answer.py").write_text(code, encoding="utf-8")
        (job / "work" / "scratch.py").write_text(scratch or "", encoding="utf-8")
        return _run(judge_dir, job, mode, seed, cfg)
    finally:
        _remove(job)


def _run(judge_dir: Path, job: Path, mode: str, seed: int, cfg: dict) -> dict:
    started = time.monotonic()
    proc = subprocess.Popen(sandbox_cmd(judge_dir, job, mode, seed, cfg), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                            preexec_fn=_limits(cfg))
    chunks: list[bytes] = []
    err: list[bytes] = []

    def pump(stream, into, cap):
        size = 0
        for block in iter(lambda: stream.read(65536), b""):
            if size < cap:
                into.append(block[: cap - size])
            size += len(block)

    readers = [threading.Thread(target=pump, args=(proc.stdout, chunks, CHANNEL_BYTES), daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, err, 65536), daemon=True)]
    for t in readers:
        t.start()
    killed = ""
    last_check = 0.0
    while proc.poll() is None:
        now = time.monotonic()
        if now - started > cfg["timeout"]:
            killed = "timeout"
        elif now - last_check > 0.5:
            last_check = now
            size, count = _tmp_usage(job / "tmp")
            if size > TMP_BYTES or count > TMP_FILES:
                killed = "disk"
        if killed:
            try:
                os.killpg(proc.pid, signal.SIGKILL)     # --die-with-parent then takes the whole sandbox down
            except ProcessLookupError:
                pass
            break
        time.sleep(0.05)
    proc.wait()
    for t in readers:
        t.join(timeout=5)
    events = []
    for line in b"".join(chunks).decode("utf-8", "replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return assemble(events, mode, killed, proc.returncode, cfg, round(time.monotonic() - started, 2),
                    b"".join(err).decode("utf-8", "replace"))


def _death(killed: str, returncode: int, cfg: dict, stderr: str) -> dict:
    """Why a sandbox stopped without finishing."""
    if killed == "timeout":
        return {"verdict": "TLE", "msg": f"整个评测超过 {cfg['timeout']:g} 秒，被强制结束"}
    if killed == "disk":
        return {"verdict": "RE", "msg": f"写入 /tmp 的文件超过 {TMP_BYTES >> 20} MB 或 {TMP_FILES} 个，被强制结束"}
    if returncode < 0:
        sig = -returncode
        name = {signal.SIGSEGV: "段错误（常见原因：递归太深）", signal.SIGXCPU: "CPU 时间总量超限",
                signal.SIGKILL: "进程被系统结束（可能内存不够）", signal.SIGABRT: "进程中止（abort）"}.get(sig, f"信号 {sig}")
        return {"verdict": "TLE" if sig == signal.SIGXCPU else "RE", "msg": f"进程崩溃：{name}"}
    if "bwrap:" in stderr:
        return {"verdict": "JE", "msg": "沙箱没能启动", "trace": stderr[-2000:]}
    return {"verdict": "RE", "msg": f"进程意外退出（退出码 {returncode}）"}


def assemble(events: list[dict], mode: str, killed: str, returncode: int, cfg: dict, wall: float, stderr: str) -> dict:
    res = {"mode": mode, "verdict": None, "passed": 0, "total": 0, "parts": [], "cases": [], "first": None,
           "cpu": 0.0, "wall": wall, "msg": "", "trace": "", "lines": [], "out": "", "timeout": cfg["timeout"]}
    done = any(e.get("type") == "done" for e in events)
    cases: dict[int, dict] = {}
    running = None
    for e in events:
        kind = e.pop("type", "")
        if kind == "plan":
            for c in e["cases"]:
                cases[c["i"]] = {**c, "verdict": None}
        elif kind == "start":
            running = e["i"]
        elif kind == "case":
            cases.setdefault(e["i"], {}).update(e)
            running = None
        elif kind in ("ce", "load", "je"):
            res.update({k: v for k, v in e.items() if k in ("msg", "trace", "lines", "out", "cpu")})
            res["verdict"] = {"ce": "CE", "je": "JE"}.get(kind, e.get("verdict"))
            res["stage"] = "load"
        elif kind == "loadout":
            res["out"] = e.get("out", "")
        elif kind == "scratch":
            res.update({k: v for k, v in e.items() if k in ("verdict", "msg", "trace", "lines", "out", "cpu")})
        elif kind == "stop":
            res["msg"] = e.get("msg", "")
    if not done:
        death = _death(killed, returncode, cfg, stderr)
        if mode == "scratch" or not cases:
            if res["verdict"] is None:
                res.update(death)
                res["stage"] = "load"
        elif running is not None and cases.get(running, {}).get("verdict") is None:
            cases[running].update(death)
        elif res["verdict"] is None and not any(c.get("verdict") for c in cases.values()):
            res.update(death)
            res["stage"] = "load"
    if mode == "scratch":
        res["verdict"] = res["verdict"] or "OK"
        res["verdict_zh"] = VERDICT_ZH.get(res["verdict"], res["verdict"])
        return res
    ordered = [cases[i] for i in sorted(cases)]
    for c in ordered:
        if c.get("verdict") is None:
            c["verdict"] = "skip"
    parts: dict[str, dict] = {}
    for c in ordered:
        p = parts.setdefault(c["part"], {"name": c["part"], "passed": 0, "total": 0, "verdict": "AC"})
        p["total"] += 1
        if c["verdict"] == "AC":
            p["passed"] += 1
        elif p["verdict"] == "AC":
            p["verdict"] = c["verdict"]
    res["cases"], res["parts"] = ordered, list(parts.values())
    res["passed"] = sum(c["verdict"] == "AC" for c in ordered)
    res["total"] = len(ordered)
    res["cpu"] = round(sum(c.get("cpu") or 0 for c in ordered) + (res.get("cpu") or 0), 3)
    if res["verdict"] is None:
        failed = [c for c in ordered if c["verdict"] != "AC"]
        if failed:
            res["first"] = failed[0]["i"]
            res["verdict"] = failed[0]["verdict"] if failed[0]["verdict"] != "skip" else "RE"
        elif ordered:
            res["verdict"] = "AC"
        else:
            res["verdict"], res["msg"] = "JE", "这道题没有可运行的测试"
    res["verdict_zh"] = VERDICT_ZH.get(res["verdict"], res["verdict"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("problem", help="the problem folder (content/.../<slug>) or its judge/ folder")
    ap.add_argument("answer", help="the file to judge")
    ap.add_argument("--mode", default="submit", choices=["samples", "submit", "scratch", "list"])
    ap.add_argument("--scratch", help="code run after the answer in scratch mode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the whole result as JSON")
    args = ap.parse_args()
    code = Path(args.answer).read_text(encoding="utf-8")
    scratch = Path(args.scratch).read_text(encoding="utf-8") if args.scratch else ""
    r = judge(judge_dir_of(Path(args.problem)), code, args.mode, scratch, args.seed)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        print(f"{r['verdict']} {r['verdict_zh']}  {r['passed']}/{r['total']}  cpu {r['cpu']}s wall {r['wall']}s")
        for p in r["parts"]:
            print(f"  {p['name']}: {p['passed']}/{p['total']}")
        for c in r["cases"]:
            if c["verdict"] != "AC":
                print(f"  #{c['i']} {c['part']} {c['name']}: {c['verdict']} {c.get('msg') or ''}")
        if r.get("msg") or r.get("trace"):
            print(r.get("msg", ""), r.get("trace", ""), sep="\n")
        if r["mode"] == "scratch":
            print(r.get("out", ""))
    return 0 if r["verdict"] in ("AC", "OK") else 1


if __name__ == "__main__":
    sys.exit(main())
