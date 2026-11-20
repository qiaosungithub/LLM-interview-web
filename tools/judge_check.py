#!/usr/bin/env python3
"""Check every problem's judge/ folder by running it, the way the site will.

    .venv/bin/python tools/judge_check.py                      # every problem that has a judge/ folder
    .venv/bin/python tools/judge_check.py longest-match-tokenizer oa-bank-system   # by slug, id or folder
    .venv/bin/python tools/judge_check.py --seeds 0,1,2 --jobs 4

For each judge/ folder:
  - solution.py (the page's reference answer) passes every test, under each seed, and passes the samples;
  - every wrong/*.py fails; a first line `# expect: TLE` (or WA, RE) also pins the verdict;
  - starter.py fails cleanly (WA or RE inside a test), so it neither passes nor breaks the judge;
  - the cases the site lists (read from tests.py without running it) are the cases the harness runs;
  - the reference answer uses at most half of any case's CPU limit and of the run's time limit, so a correct
    answer is not timed out on a busy machine.
Exit status 1 if anything fails. Warnings (tight limits) do not fail the check.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "judge"))

import build  # noqa: E402
import run  # noqa: E402

TIGHT = 0.5


def judge_dirs(selectors: list[str]) -> list[tuple[str, Path]]:
    found = []
    for _source, folder, company in build.topic_dirs():
        if (folder / "judge").is_dir():
            found.append((f"{company['key']}/{folder.parts[-2]}/{folder.parts[-1]}", folder / "judge"))
    if not selectors:
        return found
    out = []
    for sel in selectors:
        sel = sel.rstrip("/")
        hits = [(i, d) for i, d in found if sel in (i, i.split("/")[-1], str(d), str(d.parent))
                or Path(sel).resolve() in (d.resolve(), d.parent.resolve())]
        if not hits:
            raise SystemExit(f"no judge/ folder matches {sel!r}")
        out += [h for h in hits if h not in out]
    return out


def expected_verdict(path: Path) -> str | None:
    first = path.read_text(encoding="utf-8").split("\n", 1)[0]
    m = re.match(r"#\s*expect:\s*(WA|RE|TLE)\b", first)
    return m.group(1) if m else None


def where(r: dict) -> str:
    if r.get("stage") == "load":
        return f"{r['verdict']} while loading: {(r.get('msg') or '')[:120]} {(r.get('trace') or '')[-300:]}"
    c = next((c for c in r.get("cases", []) if c["i"] == r.get("first")), None)
    if not c:
        return f"{r['verdict']}: {(r.get('msg') or '')[:200]}"
    detail = c.get("msg") or (c.get("trace") or "")[-300:]
    return f"{r['verdict']} at #{c['i']} {c['part']} {c['name']}: {detail[:300]}"


def check(item_id: str, jdir: Path, seeds: list[int]) -> tuple[list[str], list[str], list[str]]:
    """Returns (failures, warnings, notes) for one judge/ folder."""
    fails, warns, notes = [], [], []
    cfg = run.config(jdir)
    solution = jdir / "solution.py"
    if not solution.is_file():
        return [f"no solution.py"], warns, notes
    code = solution.read_text(encoding="utf-8")

    for seed in seeds:
        r = run.judge(jdir, code, "submit", seed=seed)
        if r["verdict"] != "AC":
            fails.append(f"solution.py, seed {seed}: {where(r)}")
            continue
        notes.append(f"solution seed {seed}: {r['passed']}/{r['total']} in {r['wall']:.1f}s wall (limit {cfg['timeout']:g}s)")
        if r["wall"] > TIGHT * cfg["timeout"]:
            warns.append(f"solution.py takes {r['wall']:.1f}s of the {cfg['timeout']:g}s run limit (seed {seed})")
        for c in r["cases"]:
            if c["cpu"] > TIGHT * c["limit"]:
                warns.append(f"#{c['i']} {c['name']}: solution uses {c['cpu']:.2f}s of its {c['limit']:g}s CPU limit")
        if seed == seeds[0]:
            # the site lists cases from the source; the harness must run exactly those, in that order
            listed = [(c["name"], c["part"], c["sample"]) for c in build.judge_cases(jdir / "tests.py", item_id)]
            ran = [(c["name"], c["part"], c["sample"]) for c in r["cases"]]
            if listed != ran:
                fails.append(f"cases read from tests.py {listed[:4]}… differ from the cases run {ran[:4]}…")
            parts = []
            for c in r["cases"]:
                if c["part"] not in parts:
                    parts.append(c["part"])
                elif parts[-1] != c["part"]:
                    fails.append(f"part {c['part']!r} is split: keep each part's cases together")
                    break
            for p in parts:
                own = [c for c in r["cases"] if c["part"] == p]
                if not any(c["sample"] for c in own):
                    warns.append(f"{p} has no sample case")
                if any(c["sample"] for c in own) and not own[0]["sample"]:
                    warns.append(f"{p}: put the sample cases first")

    r = run.judge(jdir, code, "samples", seed=seeds[0])
    if r["verdict"] != "AC":
        fails.append(f"solution.py, samples: {where(r)}")

    wrong = sorted((jdir / "wrong").glob("*.py")) if (jdir / "wrong").is_dir() else []
    if len(wrong) < 2:
        fails.append(f"wrong/ has {len(wrong)} solutions; need at least 2")
    for w in wrong:
        r = run.judge(jdir, w.read_text(encoding="utf-8"), "submit", seed=seeds[0])
        want = expected_verdict(w)
        if r["verdict"] == "AC":
            fails.append(f"wrong/{w.name} PASSES every test; the tests miss its bug")
        elif r["verdict"] in ("JE", "CE"):
            fails.append(f"wrong/{w.name}: {where(r)}")
        elif want and r["verdict"] != want:
            fails.append(f"wrong/{w.name}: expected {want}, got {where(r)}")
        else:
            notes.append(f"wrong/{w.name}: {where(r)[:110]}")

    starter = jdir / "starter.py"
    if not starter.is_file():
        fails.append("no starter.py")
    else:
        r = run.judge(jdir, starter.read_text(encoding="utf-8"), "submit", seed=seeds[0])
        if cfg.get("special") == "perf" and r["verdict"] == "AC":
            notes.append("starter.py passes: allowed for special: perf (the official baseline is correct, only slow)")
        elif r["verdict"] not in ("WA", "RE") or r.get("stage") == "load":
            fails.append(f"starter.py should fail inside a test (WA or RE), got {where(r)}")
    return fails, warns, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", help="slugs, ids or folders; default: all")
    ap.add_argument("--seeds", default="0,1", help="comma-separated seeds for the reference run (default 0,1)")
    ap.add_argument("--jobs", type=int, default=2, help="problems checked at once (default 2)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print what each wrong solution got")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    targets = judge_dirs(args.problems)
    bad = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(check, i, d, seeds): i for i, d in targets}
        for fut in cf.as_completed(futures):
            item_id = futures[fut]
            try:
                fails, warns, notes = fut.result()
            except Exception as e:      # noqa: BLE001
                fails, warns, notes = [f"checker crashed: {type(e).__name__}: {e}"], [], []
            bad += bool(fails)
            print(f"{'FAIL' if fails else 'ok  '} {item_id}")
            for f in fails:
                print(f"     FAIL {f}")
            for w in warns:
                print(f"     warn {w}")
            if args.verbose:
                for n in notes:
                    print(f"     {n}")
    print(f"{len(targets)} judge folder(s), {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
