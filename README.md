# Interview prep site (Anthropic / OpenAI / Google DeepMind)

A local study site: one page per interview problem, grouped by company and by round type, with the reference
answer hidden until asked for, an editor that saves answers as plain files, and a record of what was checked
and changed on every page. Code problems also have an online judge (run the examples, submit against every test,
run your own lines, keep a submission history); answers and submissions stay on this machine.

## Repository and deployed source

GitHub repository: [qiaosungithub/LLM-interview-web](https://github.com/qiaosungithub/LLM-interview-web)
(branch `main`).

This directory is the source of the currently running site:
`/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep`.
On 2026-11-20 the Cloudflare tunnel address was
`https://melissa-controlling-excluded-requests.trycloudflare.com/`.
The tunnel forwards to this directory's `server.py` on `127.0.0.1:8771`;
the server serves `site/`. Its homepage and front-end script were compared
byte-for-byte with the public endpoint. The initial Git snapshot includes the
current 185 topics, 97 judge configurations, source notes and deployed assets.

The Python server and judge code, `build.py`, `content/`, `materials/`,
`tools/`, the Node package lock, and all of `site/` are versioned. Both authored
front-end assets and generated content are retained so a checkout contains the
currently deployable pages. After a content change, rebuild and commit the
corresponding `site/data/` changes together with the source.

Answers, submissions, grading, the access key, logs, local Python installations,
virtual environments, `node_modules/` and `tools/out/` are excluded. They remain
on this machine; a Git clone is not a backup of personal answers. Keep separate
backups of `answers/` and `reviews/`. Never force-add `.access_key` or include a
key-bearing URL in a commit.

## Set up a fresh checkout

The existing deployment uses Python 3.11.18, Node 24.22.2, and bubblewrap 0.6.1
on Linux. `requirements.txt` records the installed Python package versions
(excluding packaging tools); `tools/node/package-lock.json` records the browser
and rendering dependencies. The judge expects both `.python/` and `.venv/`
inside this directory, so use a project-local Python installation:

```bash
# Requires uv, Node/npm, and bubblewrap (bwrap) installed on the new Linux host.
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
uv python install 3.11.18
uv venv --python 3.11.18 .venv
uv pip install --python .venv/bin/python -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
npm ci --prefix tools/node
mkdir -p logs answers reviews
.venv/bin/python build.py --check
# Use the bundled site/ immediately, or run build.py without --check to rebuild.
.venv/bin/python server.py --port 8771 --key-file .access_key
```

The server generates a new private key when `.access_key` is absent. Configure
the tunnel separately using the commands below. Its address and key are runtime
state, not repository configuration. The dependency versions above were read
from the running environment; installing them on another host was not tested
as part of repository initialization.

## Run it

Two tmux sessions, both started from this folder; logs go to `logs/`:

```bash
# the site, on loopback; tunnel visitors need the key in .access_key (owner-only, never commit or print it)
tmux new-session -d -s interview-prep -c "$PWD" 'exec .venv/bin/python server.py --port 8771 --key-file .access_key >> logs/server.log 2>&1'
# a Cloudflare quick tunnel to it; its random https://….trycloudflare.com address is in logs/tunnel.log
tmux new-session -d -s interview_tunnel -c "$PWD" "$HOME/.local/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8771 2>&1 | tee -a logs/tunnel.log"
```

The user opens `https://<tunnel>/?key=<key>` once per browser; the server keeps the key in a cookie and drops it
from the address. Restarting the tunnel gives a new address, so tell the user the new link; restarting only the
server keeps it. A forwarded port (`http://127.0.0.1:8771`) works without the key. Without `--key-file` the server
refuses anything that arrives through a proxy. GPT-6-astra's earlier site (`../interview_workbench_local/`, same
port) was stopped when this one was deployed; it can still be started there with `./start.sh --port <other>`.

After changing anything under `content/` or `site/assets/`, rebuild (the server picks up the new files without a
restart):

```bash
export PATH=$HOME/.local/bin:$PATH    # node, used to pre-render maths and code highlighting
.venv/bin/python build.py             # --check validates without writing
```

## Layout

| Path | What it is |
| --- | --- |
| `content/schuture/<Company>/<category>/<slug>/` | The collected problem notes (`README.zh.md` is what the site shows, `README.md` is the English original, `meta.yaml`, `audit.json`). Audited and corrected in place; the untouched originals are in `../anthropic_interviews_2026-11-10/Schuture-interview-notes-2026-10-05/`. |
| `content/extra/<Company>/<category>/<slug>/` | Problems written for this site: question types the reports describe without a concrete problem, the official Anthropic performance take-home, and the Anthropic safety-philosophy review. |
| `content/site/` | Per-company pages (`rules.*`: AI rules and interview environment; `selection.*`: what was kept, added, left out), the audit page intro, AI-allowed labels (`ai_overrides.yaml`), home-page cards. |
| `materials/` | Third-party code the pages refer to (the official take-home repository and a staged solution). |
| `site/` | Build output plus the front end (`site/assets/`). |
| `answers/` | The user's saved answers and judge submissions (`<id>/submissions.jsonl`, `<id>/submissions/<n>.py`). **User data: never edit or delete.** |
| `reviews/` | Grading written by an assistant on request; shown in each problem's "批改" tab. |
| `tools/` | `run_page.py` runs every `python` block of a page; `page_ns.py` loads a page's code for experiments; `judge/` is the judge (below); `judge_check.py` checks every problem's tests; the briefs below. |
| `.access_key`, `logs/` | The tunnel access key (secret, mode 600) and the server and tunnel logs. |

## The judge

A problem with a `judge/` folder (`starter.py`, `tests.py`, optional `ref.py`, `solution.py`, `wrong/*.py`,
`judge.yaml`) gets a test panel under its editor. The server queues at most two runs at a time
(`POST /api/judge`, then poll `GET /api/judge?job=`) and starts each in a bubblewrap sandbox with no network,
nothing of this machine but the Python install and that problem's `judge/` folder (minus `solution.py` and
`wrong/`), and per-case CPU limits (`tools/judge/run.py`, which runs `tools/judge/harness.py` inside it). Submissions
are recorded under `answers/<id>/`. Try a file by hand:

```bash
.venv/bin/python tools/judge/run.py <problem folder> some_answer.py --mode samples   # or submit, scratch, list
.venv/bin/python tools/judge_check.py [slug …]   # reference passes, wrong answers fail, starter fails cleanly
```

`run.py` and the harness are loaded by the server on the first judge request, so restart the `interview-prep`
session after changing them (the tunnel address stays the same). How to write a problem's tests is in
`tools/JUDGE_BRIEF.md`. Problems without a single right output (code review, code reading, estimates, research
take-homes) have no judge and are graded on request as before.

## Briefs for assistants

- `tools/GRADING_BRIEF.md` — how to grade saved answers when the user asks ("批改…").
- `tools/AUDIT_BRIEF.md` — how a page is audited: real errors only, verified by running code, fixed minimally
  or relaxed; writes `audit.json`.
- `tools/AUTHOR_BRIEF.md` — the page format and the bar for adding a new problem.
- `tools/JUDGE_BRIEF.md` — how to write a problem's judge (tests, starter code, reference and wrong answers).

Check all pages still run: `.venv/bin/python tools/run_page.py --all --quiet` (add `--hash-seeds` to vary
`PYTHONHASHSEED`). Check every judge: `.venv/bin/python tools/judge_check.py --jobs 4`.
