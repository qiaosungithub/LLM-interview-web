# Google DeepMind: which problems are included, and why

**In short: all 37 topics come from the Google DeepMind interview notes compiled by Schuture, none of them removed; for the few types of problem that the interview process mentions but the notes have no concrete problem for, problems of the same type were added: two each for coding and take-homes, and one page each for Q&A and discussion (with several questions on a page).** Every page was re-checked; see the [audit report](#/audit) for what was changed.

## What is included

| Category | Count | Notes |
| --- | ---: | --- |
| Coding | 17 | Graph traversal, generators and weighted sampling, k-d trees, fence painting, bitwise encoding and decoding, code review, plus several ML coding problems (attention, backpropagation, debugging a training loop, and more) |
| System design | 8 | ML system design: recommendations, molecular reaction prediction, data-center failure prediction, the Gemini app, retrieval augmentation and agent platforms, and more |
| Knowledge questions | 7 | Verbally answered questions on ML, LLMs, math, CS fundamentals, and code reading; each page is split into one card per question |
| Behavioral and culture | 5 | Recruiter screen, team lead round, research deep dive, Googleyness, product judgment for product managers; likewise split into one card per question |

## Which types were added, and on what basis

Problems are added in only one situation: the [interview process](#/deepmind/process) mentions a type of problem, but the notes have not a single concrete problem of that type. The basis is always the wording on the process page:

| What the process page says | What the notes already had | What was added |
| --- | --- | --- |
| Coding rounds have included "obscure data structure and trees questions" | k-d trees and a snapshot array, but no general tree-query problem | One tree problem and one obscure-data-structure problem |
| Coding rounds for software engineering roles are described as "dynamic programming and backtracking" | Neither of these | One dynamic programming problem and one backtracking problem |
| A few tracks have a take-home: one research scientist got "a take home with a timer (3 hours)"; a public code repository shows someone was asked to build a small working app | No take-home | One timed research assignment and one small app prototype |
| The early quiz and the internship screen both tested reinforcement learning | Only "RL for LLMs" (PPO, GRPO, and so on), no classic reinforcement learning | One page of classic reinforcement learning Q&A |
| The hiring manager round includes open-ended technical discussion: one report was asked "how to evaluate a multi-task system across domains" and "how to improve a model with no additional data", and another was asked about the limitations of current models and the directions they wanted to work on | Only generic questions on the behavioral pages | One page of hiring-manager technical-judgment discussion (built mainly around the first two questions; the follow-ups were written by us) |
| Program-management roles have a hiring manager round testing "stakeholder management"; a strategy-and-operations candidate gave a presentation on a recent DeepMind achievement | Nothing | One page each |

Most of this evidence comes from just one or two reports, so the added problems are all marked "Low" or "Rare" frequency on their pages. Their purpose is to let you see what this type of problem looks like; it does not mean the interview will necessarily include them. See the list at the end of this page for each problem.

## What is not included

The other materials at hand (generic ML exercises, mock assessment question banks) were all gathered with Anthropic in mind and have nothing to do with Google DeepMind's interviews, so none of them were included here.

## Things to keep in mind

- **The problems are write-ups, not originals.** The notes' author rewrote the problem statements and added reference solutions based on multiple interview reports.
- **Teams differ a lot.** Officially, all that is said is that the skills interviews take "two or three further calls"; what exactly is tested is whatever the materials from your recruiter say. As for whether math is tested separately, the process page suggests asking your recruiter directly.
- **The "Frequency" column is the notes author's own count**, which we have no way to verify independently.
