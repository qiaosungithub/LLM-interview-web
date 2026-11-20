# Anthropic: can you use AI, and what is the interview environment

**In short: no AI by default, unless a round explicitly tells you it is allowed.** Every coding round takes place in an environment where code can run, with the interviewer watching you write and run it; but apart from the online assessment, you usually are not given ready-made tests.

Every point below is labeled with its source: **Official** means Anthropic's own public writing (the originals were reopened and checked on 2026-11-17); **Candidate reports** means claims that several people made in the collected interview reports, with no official text to check them against.

## Can you use AI in each stage

| Stage | Can you use AI | Source |
| --- | --- | --- |
| Preparing for interviews | Yes | Official |
| Writing your resume and answering application questions | Write the first draft yourself, then polish it with Claude; a few job postings ask you not to use AI anywhere in the application, so go by the posting for the role you are applying to | Official |
| Online assessment (CodeSignal) | No | Official default rule |
| Coding rounds in phone screens and onsites | No; looking up documentation and searching are allowed | Official |
| The "AI-assisted coding" round (review PRs and improve one) | Yes, with the Claude Code provided in the environment | Candidate reports |
| System design, design-doc review | No | Official default rule |
| Take-home | No by default; the performance-engineering one explicitly allows it | Official |
| Culture, hiring manager, project deep dive | Conversation rounds; no tools involved | — |

The official wording ([Guidance on candidates' AI usage](https://www.anthropic.com/candidate-ai-guidance)):

- Live interviews: "This is all you–no AI assistance unless we indicate otherwise."
- Take-homes: "Complete these without Claude unless we indicate otherwise." and "We'll be clear when AI is allowed (example: "You may use Claude for this coding challenge")."
- Explicitly listed as not allowed: "Having Claude write your code when we haven't explicitly allowed it".
- Preparation: "Use Claude to research Anthropic, practice your answers, and prepare questions for us."

Two exceptions:

- **The performance-engineering take-home allows AI.** The official engineering blog says the candidate instructions require take-homes to be done without AI by default, but "For this take-home, we explicitly indicate otherwise"; the reasoning is that AI has a hard time solving long problems like this on its own, so candidates can use AI the way they would at work while still showing their own skill. ([Engineering blog](https://www.anthropic.com/engineering/AI-resistant-technical-evaluations))
- **The "AI-assisted coding" round allows Claude Code.** This comes from candidate reports, with no official text: since mid-2026, some phone screens and onsites take place in a browser environment resembling VS Code plus GitHub, with Claude Code available; the task is to review several open PRs in an unfamiliar repository and then pick one to improve.

The tag next to each problem on this site follows this table: almost all are marked "AI not allowed by default" (默认禁用 AI); only [Reviewing PRs with an AI assistant](#/anthropic/coding/agentic-pr-review) and [the performance-engineering take-home](#/anthropic/take-home/performance-kernel-optimization) are marked "AI allowed" (可以用 AI).

## Does the interviewer run your code, or just read it

**In coding rounds, the code always runs.** No round has you write on a whiteboard or in plain text just for the interviewer to read.

| Stage | Where you write | Does the code run | Are tests provided |
| --- | --- | --- | --- |
| Online assessment | CodeSignal, 90 minutes, one problem with four levels | Yes | **Yes**: the platform has built-in tests, and you must pass all tests for one level to unlock the next; partial credit counts |
| Coding rounds in phone screens and onsites | Google Meet screen share + CodeSignal or Colab, usually 55 minutes | Yes | **Usually not**: the editor may contain just an empty `main()`, and you write the file-reading helpers and test data yourself |
| The "AI-assisted coding" round | A browser environment resembling VS Code + GitHub, with a terminal | Yes | Whatever the repository already has |
| Rounds that call a model (building an agent loop, writing prompts) | A notebook with the model-calling functions already provided | Yes | No |
| System design | A shared Google Doc, or a whiteboard you share yourself (some people used Excalidraw or Figma) | No runnable code | — |
| Design-doc review | Read a prepared design doc and discuss it verbally | No code, no diagrams either | — |
| Take-home | Your own machine | Yes | The performance-engineering one comes with a simulator and scoring tests; for the research ones you design your own experiments |

"Google Meet", "CodeSignal or Colab" and "you can look things up" are official statements ([careers page](https://www.anthropic.com/careers)):

- "All interviews are conducted over Google Meet"
- "For technical roles, we use live coding tools like Colab and CodeSignal."
- "You can look things up—just be comfortable with basic syntax and standard libraries so it doesn't eat up your time."

All other details (55 minutes, screen sharing, no tests, the empty `main()`, which tools design rounds use) come from candidate reports. The standard online assessment does not require a camera; the Fellows program's assessment is camera-proctored and does not allow web searches.

As for language, Python is the safest choice. Some candidates used Java or TypeScript after agreeing on it with their recruiter beforehand, and some rounds allow only Python.

## What this means for practice

- **Online assessment problems (titles starting with "Online Assessment")**: work through the levels in order, and do not look at the next level until the current one is correct; time yourself to 90 minutes. The real environment runs the tests for you, so when practicing, run the examples in the problem statement yourself as tests.
- **Coding problems from phone screens and onsites**: the real environment gives you no tests, so **writing a few test cases yourself and running them** is part of what you are practicing. This site stores your answer as `answer.py`, which you can run directly on your machine.
- **The two problems that allow AI**: practice them with AI, the way the round allows; what matters is whether you can take responsibility for every line the assistant writes.
- Every other problem: turn off AI assistants and autocomplete while practicing.

## What we could not verify

- The specific rules of the "AI-assisted coding" round come only from candidates' accounts; there is no official page.
- Whether web searches are allowed in the online assessment: only the Fellows version has a clear report (not allowed); we found no clear statement for the standard version.
- Take-home content varies across teams; whether AI is allowed is determined by the instructions for the assignment you receive.
