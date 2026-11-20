#!/usr/bin/env python3
"""Run the ```python blocks of one or more topic pages, top to bottom, each page in a fresh interpreter.

    .venv/bin/python tools/run_page.py content/schuture/Anthropic/coding/tokenizer            # README.md + README.zh.md
    .venv/bin/python tools/run_page.py content/schuture/Anthropic/coding/tokenizer/README.zh.md
    .venv/bin/python tools/run_page.py --all                                                   # every page, both languages
    .venv/bin/python tools/run_page.py --all --json tools/run_all.json

A page is a script: its solution blocks followed by the collapsed "runnable checks" block, which asserts the
solution against the statement's examples. Only ```python fences run; ```py fences are illustrative.
Exit status is 0 only when every requested page ran to completion.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
FENCE_RE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

CHILD = r"""
import multiprocessing, sys, types, traceback
if "fork" in multiprocessing.get_all_start_methods():
    multiprocessing.set_start_method("fork", force=True)
blocks = __import__("json").load(open(sys.argv[1], encoding="utf-8"))
module = types.ModuleType("__snippets__")
sys.modules["__snippets__"] = module
ns = module.__dict__
for i, block in enumerate(blocks, 1):
    try:
        exec(compile(block, f"{sys.argv[2]} block #{i}", "exec"), ns)
    except BaseException:
        traceback.print_exc()
        print(f"FAILED at block #{i} of {len(blocks)}", file=sys.stderr)
        sys.exit(1)
print(f"OK {len(blocks)} block(s)")
"""


def blocks_of(page: Path) -> list[str]:
    return [b for b in FENCE_RE.findall(page.read_text(encoding="utf-8")) if b.strip()]


def run_page(page: Path, timeout: int, hash_seed: str | None = None) -> dict:
    blocks = blocks_of(page)
    rel = str(page.relative_to(ROOT)) if page.is_relative_to(ROOT) else str(page)
    if not blocks:
        return {"page": rel, "status": "no-code", "blocks": 0, "seconds": 0.0, "tail": ""}
    scratch = Path(tempfile.mkdtemp(prefix="run_page_", dir=os.environ.get("RUN_PAGE_TMP")))
    blocks_file = scratch / "blocks.json"
    blocks_file.write_text(json.dumps(blocks), encoding="utf-8")
    child = scratch / "child.py"
    child.write_text(CHILD, encoding="utf-8")
    env = dict(os.environ)
    for var in THREAD_VARS:      # this host has 240 cores and many users; unbounded BLAS/torch pools thrash
        env.setdefault(var, "2")
    env.setdefault("MPLBACKEND", "Agg")      # pages that plot must not need a display
    if hash_seed is not None:
        env["PYTHONHASHSEED"] = hash_seed
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, str(child), str(blocks_file), rel], cwd=scratch, env=env,
                              capture_output=True, text=True, timeout=timeout)
        status = "ok" if proc.returncode == 0 else "fail"
        tail = (proc.stdout[-1500:] + proc.stderr[-3000:]).strip()
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        tail = f"timed out after {timeout}s\n" + ((e.stderr or b"").decode("utf-8", "replace")[-1500:] if e.stderr else "")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"page": rel, "status": status, "blocks": len(blocks), "seconds": round(time.time() - t0, 1), "tail": tail}


def expand(arg: str) -> list[Path]:
    p = Path(arg)
    if not p.is_absolute():
        p = (Path.cwd() / p) if (Path.cwd() / p).exists() else (ROOT / p)
    if p.is_dir():
        return [q for q in (p / "README.md", p / "README.zh.md") if q.exists()]
    return [p]


def all_pages() -> list[Path]:
    pages = []
    for meta in sorted(CONTENT.rglob("meta.yaml")):
        for name in ("README.md", "README.zh.md"):
            if (meta.parent / name).exists():
                pages.append(meta.parent / name)
    return pages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pages", nargs="*", help="topic folders or README files")
    ap.add_argument("--all", action="store_true", help="every topic page under content/")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--hash-seeds", action="store_true", help="run each page under PYTHONHASHSEED=0 and 1")
    ap.add_argument("--json", help="write the full result list to this file")
    ap.add_argument("--quiet", action="store_true", help="print only failures and the final count")
    args = ap.parse_args()
    pages = all_pages() if args.all else [q for a in args.pages for q in expand(a)]
    if not pages:
        ap.error("give at least one page or --all")
    seeds = ["0", "1"] if args.hash_seeds else [None]
    jobs = [(p, s) for p in pages for s in seeds]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for (page, seed), res in zip(jobs, pool.map(lambda j: run_page(j[0], args.timeout, j[1]), jobs)):
            if seed is not None:
                res["hash_seed"] = seed
            results.append(res)
            mark = {"ok": "PASS", "no-code": "----", "fail": "FAIL", "timeout": "TIME"}[res["status"]]
            if not args.quiet or res["status"] in ("fail", "timeout"):
                suffix = f" (PYTHONHASHSEED={seed})" if seed is not None else ""
                print(f"{mark} {res['page']}{suffix}  [{res['blocks']} blocks, {res['seconds']}s]")
                if res["status"] in ("fail", "timeout"):
                    print("    " + res["tail"].replace("\n", "\n    "))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if all(r["status"] in ("ok", "no-code") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
