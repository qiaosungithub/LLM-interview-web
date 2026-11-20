# Statement brief: rewrite a system-design problem statement so it reads like a real interview prompt

The user read the statement of `schuture/Anthropic/system-design/inference-api-batching` and said it
"doesn't read like something a person wrote": one wall of long sentences that crammed scale numbers,
scope notes and five deep-dive requirements together, and a Chinese version translated sentence by
sentence. That page has been rewritten; it is the model for this brief. The user wants every other
system-design statement rewritten the same way, **in English on both the English and the Chinese
page** (the Chinese page keeps its Chinese reference solution; only the statement becomes English).

## What to change, and what never to change

- Change **only the statement**: in `README.md` the text between `## Problem` and `## Reference
  solution`; in `README.zh.md` the text between `## 题目` and `## 参考解答`. Put the **same English
  text** in both. A folder with only `README.zh.md` (the `extra/` pages) gets the English statement
  in `README.zh.md`; do not create a `README.md`.
- Never touch the reference solution, `meta.yaml`, the headings, or the meta table at the top.
- **Keep every fact the reference solution relies on.** Before writing, read the whole reference
  solution (English, and the Chinese one too for zh-only pages) and list every given it uses: each
  number, limit, interface, definition, assumption, traffic mix, failure mode, follow-up scenario.
  Every one of them must still be in your statement, with the same value. Do not invent new numbers
  or requirements. If the solution refers to a statement label (`part (e)`, `(c)`, `Q3`), keep that
  label on the same item.
- Material the candidate is meant to read — a design document to review, a code snippet, a schema,
  a log excerpt — is part of the problem, not prose to tidy. Keep it verbatim; in particular never
  remove or "fix" a flaw that was planted for the candidate to find. Only the prompt around it gets
  rewritten.
- Keep links (to other pages, `#/...` routes, external sources) and source citations.

## How the statement should read

The model is the rewritten inference-api-batching statement:
`content/schuture/Anthropic/system-design/inference-api-batching/README.md` (section `## Problem`).
The original it replaced is in
`tools/out/statement-rewrite-backup-20261120/schuture/Anthropic/system-design/inference-api-batching/README.md`.
Read both before you start.

- Open with **one sentence** saying what to design, the way an interviewer would say it.
- Then short blocks with a bold lead-in, chosen to fit the problem, for example: **What you have.**
  (given components, interfaces) · **Requirements.** · **Scale.** or **Traffic at peak.** ·
  **Design:** (the numbered deliverables) · **Topics the interviewer will dig into.** (the deep dives,
  each a bullet with a bold label and one or two plain questions) · **Out of scope:**.
- One fact per bullet. Short sentences. Plain words; define a term only where a candidate could
  misread it. Code identifiers in backticks.
- Say requirements as questions or plain asks ("When do you send a batch: when it is full, or after a
  timeout?"), not as a list of noun phrases joined by dashes and semicolons.
- If the original says the interviewer gives some numbers only when asked, keep that framing ("If you
  ask, the interviewer gives these numbers:").
- Cut padding: repeated definitions, restating the same constraint twice, meta-commentary about what
  the page is. Most statements end up a half to two thirds of their original length; a statement that
  carries reading material stays longer. There is no hard limit, but readability beats completeness of
  wording, never completeness of facts.
- The statement renders as GitHub-flavoured markdown with KaTeX (`$...$`); keep any math valid.

## Process for each folder

1. Read the current statement (both files) and the whole reference solution. Write down the givens
   list (keep it in your notes; it goes into your report).
2. Write the new English statement. Check it against the givens list item by item.
3. Replace the statement in both files (only between the two headings; keep one blank line after the
   heading and before the next heading). Confirm with `diff` that the two files now carry the same
   statement and that nothing outside the statement changed (compare against the backup under
   `tools/out/statement-rewrite-backup-20261120/`).
4. Record the change in the folder's `audit.json` with
   `python3 <scratchpad>/record_rewrite.py <folder>` (path given in your task), which appends a
   standard issue entry. Do not hand-edit `audit.json`.
5. Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_page.py <folder>` from the project root
   (it must still pass or report no-code). Do not run `build.py` without `--check`; the lead runs the
   real build once at the end. `.venv/bin/python build.py --check` is fine.

## Report

Per folder: old and new statement word count; the givens list, marking anything you deliberately left
out and why (should be nothing the solution uses); anything in the original you found wrong or
unclear and did not change (statement facts are not yours to fix: report them instead).
