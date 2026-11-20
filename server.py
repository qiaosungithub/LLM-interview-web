#!/usr/bin/env python3
"""Local server for the interview-prep site: static files from site/ plus a small API that keeps the user's
answers as plain files, so they can be read, run and graded outside the browser.

    python3 server.py                 # http://127.0.0.1:8771
    python3 server.py --port 9000

Files it writes (nothing else is ever written):
    answers/<company>/<category>/<slug>/answer.py | notes.md | answer.md | qNN.md   the answers themselves
    answers/state.json                                                              per-problem status marks
    answers/.history/<same path>/<file>.<timestamp>                                 older versions of an answer
    answers/<same path>/submissions.jsonl, submissions/NNNN.py | NNNN.json         judged submissions (code, result)
Files it reads for the "批改" tab, written later by whoever grades:
    reviews/<company>/<category>/<slug>/review.md | qNN.md

Judging (problems with a judge/ folder) runs the code in a sandbox through tools/judge/run.py, at most
JUDGE_WORKERS at a time; a run is started with POST /api/judge and its result polled with GET /api/judge?job=,
so a long run never holds a request open (a tunnel cuts requests off after about 100 seconds).

Apart from the judge (which reads judge.yaml with PyYAML) only the standard library is used. The server listens on the loopback interface only; requests whose Host or
Origin is not a loopback name are refused, so a web page elsewhere cannot read or change the answers.

Reaching it through a tunnel (for example a Cloudflare quick tunnel pointed at the port):

    python3 server.py --key-file .access_key

Requests that arrive through a proxy (a non-loopback Host, or Cloudflare / X-Forwarded-For headers) then need the
key: open  https://<tunnel>/?key=<key>  once; the server stores the key in a cookie for that site and redirects to
the same page without it. Without the cookie every request, the static pages included, gets 401. Writes through
the tunnel must also come from the tunnel's own origin. If the key file does not exist it is created with a new
random key, readable only by its owner. Loopback requests (a forwarded port) work as before, without a key.
"""
from __future__ import annotations

import argparse
import hmac
import importlib.util
import json
import mimetypes
import os
import re
import secrets
import shutil
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from http.cookies import CookieError, SimpleCookie
from urllib.parse import parse_qs, urlencode, urlsplit

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
ANSWERS = ROOT / "answers"
REVIEWS = ROOT / "reviews"
HISTORY = ANSWERS / ".history"
STATE = ANSWERS / "state.json"
MAX_BODY = 4 * 1024 * 1024
SNAPSHOT_EVERY = 600            # seconds between history snapshots of one file while it is being edited
LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}
QUESTION_FILE = re.compile(r"q(\d{2,3})\.md")
STATUS_VALUES = {"", "done", "review", "mastered"}
PROXY_HEADERS = ("Cf-Connecting-Ip", "Cf-Ray", "Cdn-Loop", "X-Forwarded-For", "Forwarded")
COOKIE = "interview_prep_key"
ACCESS_KEY: str | None = None   # set by --key-file; None means proxied requests are always refused
LOCKED_PAGE = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
               "<title>需要密钥</title><body style='font:16px/1.6 system-ui,sans-serif;max-width:32em;margin:3em auto;"
               "padding:0 16px'><h1 style='font-size:1.3em'>这个网站需要密钥</h1>"
               "<p>请用带 <code>?key=</code> 的完整链接打开一次，之后这个浏览器会记住密钥。</p>").encode("utf-8")

JUDGE_WORKERS = 2               # sandboxed runs at the same time; more requests wait in line
JUDGE_QUEUE = 8                 # runs waiting or running, beyond which new ones are refused
JUDGE_MODES = {"samples", "submit", "scratch"}
JOB_KEEP = 3600                 # seconds a finished run's result stays available for polling

lock = threading.Lock()
_index_cache: dict = {"mtime": None, "items": {}, "judge": {}}
judge_slots = threading.BoundedSemaphore(JUDGE_WORKERS)
jobs: dict = {}
jobs_lock = threading.Lock()
_judge_module = None


def judge_runner():
    """tools/judge/run.py, loaded on first use."""
    global _judge_module
    if _judge_module is None:
        spec = importlib.util.spec_from_file_location("judge_run", ROOT / "tools" / "judge" / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _judge_module = module
    return _judge_module


def items() -> dict:
    """Problem id -> {file name allowed for that problem}, reloaded when the build rewrites index.json."""
    path = SITE / "data" / "index.json"
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return {}
    if _index_cache["mtime"] != mtime:
        index = json.loads(path.read_text(encoding="utf-8"))
        table, judged = {}, {}
        for item in index["items"]:
            files = {p["key"]: p["file"] for p in item.get("parts", [])}
            for n in range(1, int(item.get("nq") or 0) + 1):
                files[f"q{n:02d}"] = f"q{n:02d}.md"
            table[item["id"]] = files
            if item.get("judge"):
                judged[item["id"]] = item["judge"]
        _index_cache.update(mtime=mtime, items=table, judge=judged)
    return _index_cache["items"]


def judge_of(item_id: str) -> dict | None:
    items()
    return _index_cache["judge"].get(item_id)


def answer_path(item_id: str, part: str) -> Path | None:
    files = items().get(item_id)
    if not files or part not in files:
        return None
    return ANSWERS / item_id / files[part]


def read_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def snapshot(path: Path, force: bool) -> None:
    """Keep the version about to be replaced, at most once per SNAPSHOT_EVERY seconds unless forced."""
    if not path.exists() or path.stat().st_size == 0:
        return
    folder = HISTORY / path.parent.relative_to(ANSWERS)
    previous = sorted(folder.glob(path.name + ".*")) if folder.is_dir() else []
    if previous and not force and time.time() - previous[-1].stat().st_mtime < SNAPSHOT_EVERY:
        return
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, folder / f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}")


def submissions(item_id: str) -> list[dict]:
    path = ANSWERS / item_id / "submissions.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def record_submission(item_id: str, code: str, result: dict) -> int:
    """Keep a submitted program and its result; returns its number (1, 2, ...)."""
    with lock:
        folder = ANSWERS / item_id
        n = max([s.get("n", 0) for s in submissions(item_id)], default=0) + 1
        write_atomic(folder / "submissions" / f"{n:04d}.py", code)
        write_atomic(folder / "submissions" / f"{n:04d}.json", json.dumps(result, ensure_ascii=False))
        summary = {"n": n, "t": int(time.time()), "verdict": result.get("verdict"), "passed": result.get("passed", 0),
                   "total": result.get("total", 0), "first": result.get("first"), "cpu": result.get("cpu"),
                   "wall": result.get("wall"), "lines": code.count("\n") + 1,
                   "parts": [{k: p.get(k) for k in ("name", "passed", "total")} for p in result.get("parts", [])]}
        with open(folder / "submissions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return n


def judge_stats(item_id: str) -> dict | None:
    subs = submissions(item_id)
    if not subs:
        return None
    return {"n": len(subs), "ac": sum(s.get("verdict") == "AC" for s in subs), "last": subs[-1].get("verdict"),
            "best": max(s.get("passed", 0) for s in subs), "total": subs[-1].get("total", 0)}


def run_job(job: dict, judge_dir: Path, code: str, scratch: str) -> None:
    with judge_slots:
        job["status"], job["started"] = "running", time.time()
        try:
            result = judge_runner().judge(judge_dir, code, job["mode"], scratch)
        except Exception as e:      # noqa: BLE001 - a judge failure is reported to the page, not raised
            result = {"mode": job["mode"], "verdict": "JE", "verdict_zh": "评测程序出错", "msg": f"{type(e).__name__}: {e}",
                      "cases": [], "parts": [], "passed": 0, "total": 0}
            sys.stderr.write(f"judge error for {job['item']}: {e!r}\n")
    if job["mode"] == "submit":
        result["n"] = record_submission(job["item"], code, result)
    job.update(status="done", result=result, finished=time.time())


def start_job(item_id: str, mode: str, code: str, scratch: str) -> tuple[dict | None, str]:
    cfg = judge_of(item_id)
    with jobs_lock:
        now = time.time()
        for key in [k for k, j in jobs.items() if j["status"] == "done" and now - j["finished"] > JOB_KEEP]:
            del jobs[key]
        if sum(j["status"] != "done" for j in jobs.values()) >= JUDGE_QUEUE:
            return None, "评测排队的太多了，请等前面的跑完再提交"
        job = {"id": secrets.token_hex(8), "item": item_id, "mode": mode, "status": "queued", "created": now}
        jobs[job["id"]] = job
    threading.Thread(target=run_job, args=(job, ROOT / cfg["dir"], code, scratch), daemon=True).start()
    return job, ""


def overview() -> dict:
    answers: dict = {}
    for item_id, files in items().items():
        folder = ANSWERS / item_id
        if not folder.is_dir():
            continue
        found = {}
        for part, name in files.items():
            p = folder / name
            if p.is_file() and p.stat().st_size:
                found[part] = {"mtime": int(p.stat().st_mtime), "size": p.stat().st_size}
        if found:
            answers[item_id] = found
    reviews: dict = {}
    if REVIEWS.is_dir():
        for item_id in items():
            folder = REVIEWS / item_id
            if folder.is_dir():
                names = sorted(p.name for p in folder.glob("*.md"))
                if names:
                    reviews[item_id] = names
    judged = {}
    for item_id in _index_cache["judge"]:
        stats = judge_stats(item_id)
        if stats:
            judged[item_id] = stats
    return {"answers": answers, "status": read_state(), "reviews": reviews, "judge": judged, "root": str(ROOT)}


class Handler(BaseHTTPRequestHandler):
    server_version = "InterviewPrep/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # one short line per request on stderr; the access key is never logged
        line = re.sub(r"([?&]key=)[^&\s\"]*", r"\1***", fmt % args)
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), line))

    # ---- helpers
    def send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.close_if_error(status)
        self.end_headers()
        self.wfile.write(body)

    def close_if_error(self, status):
        # a refused request may leave its body unread; closing keeps those bytes out of the next request on the
        # same connection (a tunnel reuses connections, so they would otherwise corrupt someone else's request)
        if status >= 400:
            self.send_header("Connection", "close")
            self.close_connection = True

    def fail(self, status, message):
        self.send_json({"error": message}, status)

    def local_request(self) -> bool:
        """A request from this machine, e.g. through a forwarded port; anything relayed by a proxy is not local."""
        raw = self.headers.get("Host") or ""
        host = raw.split("]")[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
        if host not in LOOPBACK or any(h in self.headers for h in PROXY_HEADERS):
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "null":
            if (urlsplit(origin).hostname or "") not in {"127.0.0.1", "localhost", "::1"}:
                return False
        return True

    def has_key_cookie(self) -> bool:
        if ACCESS_KEY is None:
            return False
        try:
            morsel = SimpleCookie(self.headers.get("Cookie") or "").get(COOKIE)
        except CookieError:
            return False
        return morsel is not None and hmac.compare_digest(morsel.value.encode(), ACCESS_KEY.encode())

    def admit(self, write: bool) -> bool:
        """True if the request may go on; otherwise the refusal has already been sent."""
        if self.local_request():
            return True
        if ACCESS_KEY is None:
            self.fail(HTTPStatus.FORBIDDEN, "local requests only")
            return False
        if not self.has_key_cookie():
            self.send_locked()
            return False
        origin = self.headers.get("Origin")
        # browsers send Origin on every write; it must be this same site, so no other page can write answers
        if (write and not origin) or (origin and urlsplit(origin).netloc != (self.headers.get("Host") or "")):
            self.fail(HTTPStatus.FORBIDDEN, "cross-origin request refused")
            return False
        return True

    def send_locked(self):
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(LOCKED_PAGE)))
        self.send_header("Cache-Control", "no-store")
        self.close_if_error(HTTPStatus.UNAUTHORIZED)
        self.end_headers()
        self.wfile.write(LOCKED_PAGE)

    def accept_key(self, url, query):
        """GET /any/path?key=K: store K in a cookie for this site, then show the same page without the key."""
        given = query.pop("key", [""])[0]
        if ACCESS_KEY is None or not hmac.compare_digest(given.encode(), ACCESS_KEY.encode()):
            return self.send_locked()
        rest = urlencode(query, doseq=True)
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", (url.path or "/") + (f"?{rest}" if rest else ""))
        self.send_header("Set-Cookie", f"{COOKIE}={ACCESS_KEY}; Max-Age=31536000; Path=/; HttpOnly; Secure; SameSite=Lax")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def read_json(self):
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            self.fail(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send application/json")
            return None
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.fail(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too large")
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.fail(HTTPStatus.BAD_REQUEST, "body is not JSON")
            return None

    # ---- routes
    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if "key" in query and not self.local_request():
            return self.accept_key(url, query)
        if not self.admit(write=False):
            return
        if url.path == "/api/state":
            return self.send_json(overview())
        if url.path == "/api/answer":
            path = answer_path(query.get("id", [""])[0], query.get("part", [""])[0])
            if path is None:
                return self.fail(HTTPStatus.NOT_FOUND, "unknown problem or part")
            if path.is_file():
                return self.send_json({"content": path.read_text(encoding="utf-8"), "mtime": int(path.stat().st_mtime),
                                       "path": str(path.relative_to(ROOT))})
            return self.send_json({"content": "", "mtime": 0, "path": str(path.relative_to(ROOT))})
        if url.path == "/api/review":
            item_id = query.get("id", [""])[0]
            if item_id not in items():
                return self.fail(HTTPStatus.NOT_FOUND, "unknown problem")
            folder = REVIEWS / item_id
            files = {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))} if folder.is_dir() else {}
            return self.send_json({"files": files, "path": str(folder.relative_to(ROOT))})
        if url.path == "/api/judge":
            job = jobs.get(query.get("job", [""])[0])
            if job is None:
                return self.fail(HTTPStatus.NOT_FOUND, "unknown run (the server may have restarted)")
            waiting = sum(1 for j in list(jobs.values()) if j["status"] == "queued" and j["created"] < job["created"])
            return self.send_json({"status": job["status"], "ahead": waiting, "result": job.get("result")})
        if url.path == "/api/submissions":
            item_id = query.get("id", [""])[0]
            if not judge_of(item_id):
                return self.fail(HTTPStatus.NOT_FOUND, "unknown problem")
            return self.send_json({"items": submissions(item_id), "stats": judge_stats(item_id)})
        if url.path == "/api/submission":
            item_id, n = query.get("id", [""])[0], query.get("n", [""])[0]
            if not judge_of(item_id) or not n.isdigit():
                return self.fail(HTTPStatus.NOT_FOUND, "unknown submission")
            base = ANSWERS / item_id / "submissions" / f"{int(n):04d}"
            if not base.with_suffix(".py").is_file():
                return self.fail(HTTPStatus.NOT_FOUND, "unknown submission")
            result = json.loads(base.with_suffix(".json").read_text(encoding="utf-8")) if base.with_suffix(".json").is_file() else None
            return self.send_json({"code": base.with_suffix(".py").read_text(encoding="utf-8"), "result": result})
        if url.path.startswith("/api/"):
            return self.fail(HTTPStatus.NOT_FOUND, "no such endpoint")
        return self.send_static(url.path)

    def do_PUT(self):
        if not self.admit(write=True):
            return
        url = urlsplit(self.path)
        body = self.read_json()
        if body is None:
            return
        if url.path == "/api/answer":
            path = answer_path(str(body.get("id", "")), str(body.get("part", "")))
            content = body.get("content")
            if path is None or not isinstance(content, str):
                return self.fail(HTTPStatus.BAD_REQUEST, "unknown problem or part")
            with lock:
                old = path.read_text(encoding="utf-8") if path.is_file() else ""
                if content == old:
                    pass
                elif content.strip() == "":
                    snapshot(path, force=True)          # clearing an answer always keeps the old text
                    if path.exists():
                        path.unlink()
                else:
                    snapshot(path, force=bool(body.get("keep")) or len(content) < len(old) // 2)
                    write_atomic(path, content if content.endswith("\n") else content + "\n")
            mtime = int(path.stat().st_mtime) if path.exists() else 0
            return self.send_json({"ok": True, "mtime": mtime, "path": str(path.relative_to(ROOT))})
        if url.path == "/api/status":
            key, value = str(body.get("key", "")), body.get("value", "")
            item_id, _, part = key.partition("#")
            if item_id not in items() or (part and part not in items()[item_id]) or value not in STATUS_VALUES:
                return self.fail(HTTPStatus.BAD_REQUEST, "unknown problem or status")
            with lock:
                state = read_state()
                if value:
                    state[key] = {"status": value, "updated": int(time.time())}
                else:
                    state.pop(key, None)
                write_atomic(STATE, json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            return self.send_json({"ok": True})
        if url.path == "/api/judge":
            item_id, mode = str(body.get("id", "")), str(body.get("mode", ""))
            code, scratch = body.get("code"), body.get("scratch", "")
            if not judge_of(item_id) or mode not in JUDGE_MODES or not isinstance(code, str) or not isinstance(scratch, str):
                return self.fail(HTTPStatus.BAD_REQUEST, "unknown problem, mode or code")
            job, why = start_job(item_id, mode, code, scratch)
            if job is None:
                return self.fail(HTTPStatus.TOO_MANY_REQUESTS, why)
            return self.send_json({"job": job["id"]})
        return self.fail(HTTPStatus.NOT_FOUND, "no such endpoint")

    do_POST = do_PUT        # fetch(..., {keepalive: true}) on page close arrives the same way

    def send_static(self, url_path: str):
        rel = url_path.lstrip("/") or "index.html"
        target = (SITE / rel).resolve()
        if target.is_dir():
            target = target / "index.html"
        if SITE.resolve() not in target.parents or not target.is_file():
            return self.fail(HTTPStatus.NOT_FOUND, "not found")
        data = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # vendored libraries never change between builds; everything else must show a rebuild immediately.
        # Through a tunnel, "private" keeps the proxy from caching pages for visitors without the key.
        cache = "max-age=86400" if rel.startswith("vendor/") else "no-cache"
        self.send_header("Cache-Control", cache if self.local_request() else "private, " + cache)
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--host", default="127.0.0.1", help="loopback address to listen on")
    ap.add_argument("--key-file", help="allow access through a tunnel with the key in this file (created if missing)")
    args = ap.parse_args()
    if not (SITE / "data" / "index.json").exists():
        sys.exit("site/data/index.json is missing: run `.venv/bin/python build.py` first")
    global ACCESS_KEY
    if args.key_file:
        key_path = Path(args.key_file)
        if not key_path.exists():
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_urlsafe(24) + "\n")
        if key_path.stat().st_mode & 0o077:
            sys.exit(f"{key_path} is readable by other users: chmod 600 it first")
        ACCESS_KEY = key_path.read_text(encoding="utf-8").strip()
        if len(ACCESS_KEY) < 16 or not re.fullmatch(r"[A-Za-z0-9_-]+", ACCESS_KEY):
            sys.exit(f"{key_path}: the key must be at least 16 characters of A-Z a-z 0-9 _ -")
        print(f"tunnel access enabled; key in {key_path.resolve()}", flush=True)
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("font/woff2", ".woff2")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"interview prep: http://{args.host}:{args.port}/   (answers in {ANSWERS})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
