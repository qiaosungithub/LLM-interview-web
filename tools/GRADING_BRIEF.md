# Grading brief: how to grade the user's saved answers

The site never grades anything. The user writes answers in the browser, the local server saves them as plain
files, and later the user asks an assistant to grade them ("批改我新写的答案", or "批改 anthropic/coding/oa-bank-system").
This file says where everything is and what a useful review looks like.

Project root: `/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep`. Use absolute paths.

## Where things are

A problem id is `<company>/<category>/<slug>`, with company keys `anthropic`, `openai`, `deepmind`.

| What | Where |
| --- | --- |
| The user's answer | `answers/<id>/` — `answer.py` + `notes.md` (coding, take-home), `answer.md` (system design), `qNN.md` one file per question (quiz, discussion, behavioral, safety) |
| Status marks | `answers/state.json` (written by the site; do not edit) |
| Older versions of an answer | `answers/.history/<id>/<file>.<timestamp>` |
| The problem and its reference answer | `content/schuture/<CompanyDir>/<category>/<slug>/README.zh.md`, or `content/extra/...` for problems we wrote; `<CompanyDir>` is `Anthropic`, `OpenAI`, `Google-DeepMind` |
| What the audit found on that page | `audit.json` next to the README |
| Your review (you write this) | `reviews/<id>/review.md` for coding, take-home and system design; `reviews/<id>/qNN.md` for the matching question file |

To find what needs grading: every answer file under `answers/` (ignore `.history` and `state.json`) that has no
review file, or whose modification time is newer than its review file. `GET http://127.0.0.1:8771/api/state`
returns the same information as JSON if the server is running.

**Never modify anything under `answers/`.** It is the user's work. Write only under `reviews/`.

## How to grade

1. Read the statement in `README.zh.md` first, then the user's answer, and only then the reference solution.
   The reference is one good answer, not the only one: a different correct approach is correct.
2. **Code answers: run them.** Do not judge code by reading it. Write a scratch test file under
   `tools/out/scratch/grading/` that imports the user's `answer.py` by path and replays (a) every worked example
   in the statement and (b) the checks from the page's `验证代码（可运行）` block, adapted to the user's names.
   Add a brute-force comparison on random inputs when the page has one. Use
   `/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/.venv/bin/python`, a timeout, and
   `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` (the machine is shared). `tools/page_ns.py` loads a page's reference
   code into a namespace if you want to compare outputs directly.
3. Check the stated complexity and the edge cases the statement names (empty input, ties, error behaviour).
4. For system design: check the capacity arithmetic, whether the data model can answer the queries the API
   promises, and whether each hard sub-problem in the reference was noticed at all. Do not penalise a
   different but defensible architecture.
5. For quiz questions: the answer is right or wrong; say which, and show the missing step.
6. For behavioral, discussion and safety questions there is no key. Judge whether the answer is specific
   (a real situation, the user's own action, a result), whether it answers the question asked, whether a
   claim about a company or a document is accurate (check it against the page's notes and sources), and what
   follow-up question it invites. Do not rewrite the answer into a model essay; point at what to change.

## What the review file contains

Write it in plain Chinese, short sentences; the site shows it under the problem's "批改" tab. Keep code,
identifiers and quoted English terms as they are.

```markdown
**结论：** 一句话——对 / 基本对 / 有明显问题，以及最要紧的一点。

## 实际跑出来的结果        ← 代码题才有：跑了哪些测试，过了几个，哪个没过（给出输入、期望、实际）

## 问题                    ← 按严重程度排，每条指到具体的行或句子，并说怎么改

## 做得好的地方            ← 具体的，不要客套

## 面试里可能被追问
```

Say plainly what you did not check (for example, a part that needs a real model call). Date the review at the
top (`批改于 YYYY-MM-DD，对应答案文件的修改时间 …`), so a later edit of the answer is not mistaken for graded.
