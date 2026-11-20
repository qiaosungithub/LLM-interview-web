# Anthropic: which problems are included, and why

**In short: the core is the 38 topics compiled by Schuture, none of them removed; on top of that sits the original performance-optimization assignment that Anthropic published; for each type of problem that interview reports mention only by type, without leaving a concrete problem, two problems were written; and where a report left only a one-sentence prompt, that sentence was expanded into a complete problem.** Mock question banks and generic practice found online were not included. Every page was re-checked; see the [audit report](#/audit) for what was changed.

## What is included

| Source | Count | Notes |
| --- | ---: | --- |
| Schuture's notes: coding | 25 | Includes 9 online assessments (business systems that add requirements level by level), onsite coding, and ML debugging |
| Schuture's notes: system design | 6 | Inference APIs and batching, model weight distribution, chat, telemetry data, and more |
| Schuture's notes: take-home | 3 | Two ML research assignments and one practice problem of the same kind as the performance-optimization assignment |
| Schuture's notes: behavioral and culture | 4 | Culture and AI safety, hiring manager, project deep dive, recruiter screen; each page is split into one card per question |
| Officially published original | 1 | [The original performance-optimization take-home](#/anthropic/take-home/original-performance-takehome), from a repository Anthropic published itself, included as is |
| Safety philosophy review | 7 | Compiled from the conversation you saved, placed in a separate "Safety philosophy review" section (安全理念复习); these are not interview problems |

## Added problems: when we wrote one

Only in two situations:

- An interview report or official article clearly says a certain type of problem was asked, but no concrete problem of that type can be found anywhere in the materials at hand. In that case we wrote two problems of that type (for Q&A and discussion types, one page with several questions), so that you have at least seen what such a problem looks like.
- An interview report left only a one-sentence prompt, such as "design a banking app" or "for each element of an array, count how many smaller elements are to its right". In that case we expanded the sentence into a complete problem, one page per sentence; details such as scale and constraints are assumptions we added.

Added problems follow three rules:

- **The basis is stated.** At the top of each problem page, and in the list at the end of this page, we note which material mentioned this type of problem and roughly what it said.
- **They do not pose as originals.** Each page carries an "Added problem" tag (补写的题). What you meet in the interview will be the same type of problem, not the same problem.
- **They were actually run.** All code in the reference solutions was executed, and the conclusions come from running it, not from memory.

The added problems fall roughly into these groups; see the list at the end of this page for details:

- **The Fellows program's second online assessment**: debugging a given tree-model implementation (fix the crash, fix the accuracy). Two reposted accounts describe the problem, but neither includes code.
- **The current take-home described in an official article**: solving puzzles on a small machine with very few instructions, with no debugging tools provided. The article describes only the format; the problems were not published.
- **Several open-ended rounds for research roles**: research brainstorming, black-box probing of a model, evaluations and synthetic data, and a small open-topic research assignment. The interview reports give only the round names and a rough description.
- **"Research fundamentals" coding and multi-language code reading**: the reports say these were asked, but give no problems.
- **Onsite coding problems that survive only as one sentence**: for example "implement a booking service with concurrency constraints", "figure out why a streaming tokenizer system got slow", "for each element of an array, count how many smaller elements are to its right".
- **Variants of the last two online-assessment levels**: cashback and account merging for the banking system, TTL and backup/restore for the database, backup/restore for storage. The online assessments in the notes do not cover these variants, but they come up again and again in public mock question banks.
- **System design and role-play**: harmful-output detection, conversation context management, a banking app, API rate limiting, and a role-play of debugging a network outage.

## What is not included, and why

The previous version of the site had 1,748 entries, 1,085 of which were basic exercises bulk-imported from a practice repository. This version keeps only content directly relevant to the interviews; none of the following was included:

| Material | What it is | Why it is not included |
| --- | --- | --- |
| Fellowship Trainer (public code repository) | Mock exams and basic exercises newly written by the author, over a thousand entries | The repository itself says these are not real interview problems. Basic exercises (Python syntax, small quizzes) are not the same thing as interview problems, and including them would bury the real problems; its level-by-level projects are the same type as the online assessments in the notes, which we already have |
| TorchLeet (public code repository) | 68 PyTorch exercises | The company tags on its problems are guesses: the repository's own source notes say "0 reported, 136 inferred" |
| "Anthropic CodeSignal test" (public code repository) | A four-level mock problem about a file-hosting system | The repository itself says it is a mock problem; it is the same type as the existing [cloud storage assessment](#/anthropic/coding/oa-cloud-storage) |
| LibreSignal (public code repository) | Mock online assessments: banking, database, storage | The author's interview experience was at a different company, so these are not original Anthropic problems. We did not copy them; we only rewrote, as added problems, the few last-two-level variants that the notes do not cover |
| A crawler write-up, an image-pipeline exercise, a weighted-batching write-up | Three problem statements rewritten by third parties | They are the same problems as three already in the notes; we used them for cross-checking and did not include duplicates |
| A preparation guide for agent interviews | Explanations and sample code | It is a tutorial, not a problem |
| Index pages of a few paid question banks | Problem names only | The details are paywalled, so nothing was downloaded. We used these indexes to cross-check and confirm that no problem type was missing |

## Things to keep in mind

- **The problems in the notes are write-ups, not originals.** The author rewrote the problem statements and added reference solutions based on multiple interview reports. Only the officially published performance-optimization assignment is verbatim.
- **The "Frequency" column is the notes author's own count**, which we have no way to verify independently. Added problems are all marked "Low" or "Rare", because the evidence is usually just one or two reports.
- **Reposts and paid sites are of low reliability.** The statement of the tree-model debugging assessment comes from sites that sell interview assistance, and several of them are likely reposting the same report; so it is used only as evidence that "this type of problem exists", and the specific details were written fresh this time.
