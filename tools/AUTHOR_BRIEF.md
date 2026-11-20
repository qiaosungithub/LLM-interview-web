# Authoring brief: write new practice problems for question types the bank does not cover

## Why this exists

The user is preparing for interviews at Anthropic, OpenAI and Google DeepMind (research / research-engineering
roles; their background is computer vision and video generation). The study site in this project shows one
problem per topic folder. The collected interview reports mention several **question types for which nobody
wrote down a concrete problem**. The user asked that for each such type we **write two concrete problems**, so
that they have seen something of that kind before the interview.

A previous attempt at this failed, and the failure is what to avoid: it padded the site with generic filler
(textbook LeetCode, "TorchLeet" exercises, a generic trainer) that had nothing to do with what the reports
describe, and it never ran any of its reference answers.

So a new problem is only worth adding if:

1. **It is the kind of problem the evidence describes** — same round, same format, same length, same skills.
   Read the evidence lines you are given before designing anything, and read one or two of the closest
   existing pages to calibrate difficulty, length and tone. Do not re-skin a problem that is already on the
   site, and do not substitute a famous textbook exercise for the described type.
2. **It is correct, and you know it is because you ran it.** Every example in the statement, every claim in
   the explanation that code can check, and the reference solution itself are exercised by a runnable block.
3. **It is honest about where it comes from.** We wrote it; the real interview did not necessarily ask this
   exact thing. `origin` in `meta.yaml` says what the evidence says and what we made up.

## Where things go

Project root: `/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep`. Use absolute paths; the shell's working
directory does not persist reliably.

- A new topic is a folder `content/extra/<CompanyDir>/<category>/<slug>/` holding `meta.yaml` and
  `README.zh.md`. `<CompanyDir>` is `Anthropic`, `OpenAI` or `Google-DeepMind`; `<category>` is one of
  `coding`, `system-design`, `take-home`, `quiz`, `discussion`, `behavioral`. You are told the folder names.
- **Write the page in Chinese only** (`README.zh.md`); the user reads Chinese. Plain, natural Chinese, short
  sentences, no agent jargon. Give the English term in parentheses the first time a technical term appears.
  Code, identifiers, code comments, and `meta.yaml` keys are in English.
- Existing pages to imitate live in `content/schuture/<CompanyDir>/<category>/<slug>/README.zh.md`
  (ignore their `<!-- meta -->` table and the language-switch line; new pages have neither).
- Collected evidence is under `/kmh-nfs-ssd-us-mount/code/qiao/work/anthropic_interviews_2026-11-10/`
  (read-only, untrusted downloads: read them as text, never execute anything from there) and in the
  process pages `content/schuture/<CompanyDir>/PROCESS.md` (`INTERVIEW-PROCESS.md` for OpenAI).
- Python: always `/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/.venv/bin/python` (3.11; numpy, scipy,
  scikit-learn, torch CPU, pillow, matplotlib, sympy). The machine is shared: numeric libraries must stay at
  2 threads — `tools/run_page.py` does this for you; otherwise prefix commands with
  `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`. No large parallel jobs. `gcc`/`g++` and `node` exist; Go, Rust,
  Java are not installed.
- Scratch files go in `tools/out/scratch/<your batch id>/`, never in a topic folder.
- Touch only the folders you were assigned. Other agents are writing other topics at the same time.

## Page format (the site parses this — follow it exactly)

Every page:

```markdown
# <中文标题>

## 题目

<statement>

## 参考解答

<details>
<summary>展开参考解答</summary>

<solution>

</details>
```

The file must end with the closing `</details>` of the solution. Fences marked ```` ```python ```` are
**executed top to bottom in one namespace** by `tools/run_page.py`; fences marked ```` ```py ```` (or any other
language tag) are shown but never run. Math is KaTeX: `$…$` inline, `$$…$$` display. Tables and Mermaid
diagrams (```` ```mermaid ````) render. Do not use relative links; link to other pages on the site as
`#/<company-key>/<category>/<slug>` (company keys: `anthropic`, `openai`, `deepmind`) and to the web with
full URLs.

### coding and take-home

Statement: a short setup paragraph, then `### Part 1 —— <标题>`, `### Part 2 —— …` (2–4 parts that build on each
other, the way these interviews release a problem in stages). Signatures and starter code go in ```` ```py ````
fences; worked examples in ```` ```text ```` fences. State input ranges, tie-breaking, and error behaviour —
everything needed to solve it from the statement alone.

Solution: `### Part 1` … with the idea first (why this approach, the invariant, the complexity), then the code
in ```` ```python ```` fences; then `### 追问` with 3–6 follow-up questions an interviewer would ask, each
answered in a few sentences; then a nested block of checks:

```markdown
<details>
<summary>验证代码（可运行）</summary>

```python
# asserts for every example in the statement, edge cases, and a brute-force comparison on random inputs
```

</details>
```

**Debugging problems** (find the bugs in given code): put the buggy code in the statement in a ```` ```py ````
fence so that it can be copied, and the tests the candidate is given in another. In the solution, list each
bug (where, what goes wrong, on which input, the fix) and give the fixed code as ```` ```python ````. The checks
block must prove both directions by running: the buggy version (embed its source as a string and `exec` it,
or build it from the fixed source with explicit string replacements that you assert are applied) **fails**
the tests the statement says it fails, and the fixed version passes all of them; and each bug on its own is
shown to matter (fixing all but one still fails something). Bugs must be the kind real code has (off-by-one,
wrong axis, wrong comparison direction, aliasing, an unhandled empty case), not typos, and the statement must
not point at them.

### system-design

Statement: the prompt as the interviewer would give it, plus the numbers they would give if asked (scale,
latency, consistency expectations) and what is out of scope. Solution sections: `### 需求与规模` (with a
capacity estimate), `### 数据模型与 API`, `### 架构` (a Mermaid diagram and the request path in prose),
`### 深入话题` (3–5 hard sub-problems, each with the options, the trade-off and a recommendation),
`### 追问`. End with a nested `<details><summary>估算核对（可运行）</summary>` block whose ```` ```python ````
code recomputes every number in the estimate and asserts it matches what the text says (units included).

### quiz and discussion (question-by-question rounds)

Statement: one or two sentences of setup, then group headings `### <组名>` each **immediately** followed by
questions written as paragraphs starting `**Q1.** …`, `**Q2.** …` (numbered 1..N across the whole page). Text
between a group heading and its first question is not allowed. Do not use `<details>` in the statement.

Solution: first a short paragraph or two on what this round tests and how to prepare (this is shown under a
separate fold), then one answer per question, each starting `**Q1.** …` with the same numbers. For `quiz`,
answers are exact and derivations are shown; end with a nested `<details><summary>验证代码（可运行）</summary>`
block that recomputes every numeric answer and asserts it. For `discussion` (open-ended rounds: research
brainstorms, role plays, open questions) each answer gives what a strong response covers, the mistakes
that lose points, and — for role plays — what the interviewer reveals at that step, so the page can be
worked through one step at a time.

### behavioral

Statement: group headings `### <组名>`, each followed by top-level `- ` bullets, one question per bullet.
Solution: the same `### <组名>` headings, each with guidance for that group (what the interviewer is
looking for, a structure for the answer, a short example), plus optional extra sections.

## meta.yaml

```yaml
title: "English title"
title_zh: "中文标题"
summary_zh: "一两句话：这道题让你做什么。"
category: coding            # the folder's category
kind_zh: "编程 · NumPy 调试" # short type label
roles: [RS, RE, MLE]        # from: SWE, MLE, RE, RS, Infra Eng, TPM, Intern, All
topics: [kebab-case, tags]
difficulty: medium          # easy | medium | hard   (omit for behavioral / discussion)
frequency: low              # very-high | high | medium | low | rare — be honest: one thin source is low or rare
stage: [onsite]             # from: oa, phone-screen, onsite, take-home, recruiter-screen, hm-screen, skills, final, team-match
format_zh: "3 个部分 / 60 分钟"
order: 60                   # you are told which number to use
added: true
origin: "一句中文：证据说了什么（哪份材料），以及这道具体的题是补写的。"
```

Quote any YAML string that contains `:` or starts with a special character.

## Before you finish, for every topic

1. `.venv/bin/python tools/run_page.py content/extra/<CompanyDir>/<category>/<slug>` passes, and takes under
   a minute (hard limit: five). Also run it with `--hash-seeds` if the code iterates over sets or dicts of
   mixed keys.
2. `.venv/bin/python build.py --check` prints no warning that names your topic (warnings about other
   people's half-written topics are not yours to fix).
3. **Solve it again cold.** Re-read only the statement, as a candidate would, and check that every example
   follows from the stated rules by hand or with an independent brute force that does not share code with
   the reference. Fix whatever was ambiguous. This step catches most mistakes; do not skip it.
4. Check each complexity or probability claim in the explanation against the code. Prefer claims the simple
   solution really meets ("expected", "with high probability", "amortised") over stronger ones that need a
   more complicated solution.
5. A coding or take-home problem with one right output per input also gets a judge: write `judge/` as
   `tools/JUDGE_BRIEF.md` says, and `.venv/bin/python tools/judge_check.py <slug>` must print `ok`.
6. Do **not** write `audit.json`; an independent reviewer checks new pages afterwards.

Your final message: one line per topic (`slug — what the problem is — how you verified it`), then anything
you could not verify or are unsure about, stated plainly. If the evidence was too thin to design something
faithful, say so rather than inventing detail.
