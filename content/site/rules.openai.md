# OpenAI: can you use AI, and what is the interview environment

**In short: no AI coding assistants during interviews.** You write code in CoderPad and run it, and the problem is released one part at a time; sometimes tests are provided, sometimes not.

**A note on sources.** Everything on this page comes from **candidate reports** (a compilation of the collected interview reports). OpenAI has a public [interview guide](https://openai.com/interview-guide/), but on 2026-11-17 it could not be opened from this machine (the server refused access), so not a single sentence of official wording could be checked. The specific rules for each round are whatever your coordinator's instructions say; how detailed those instructions are varies from person to person, with some very thorough and some giving nothing at all.

## Can you use AI in each stage

| Stage | Can you use AI | Source |
| --- | --- | --- |
| Preparing for interviews | Yes | — |
| Coding rounds, ML coding rounds | No; at least one round also required turning off the editor's built-in autocomplete | Candidate reports |
| System design | No | Candidate reports |
| Technical deep dive, behavioral, hiring manager | Conversation rounds; no tools involved | — |

What you can look up: one coordinator's instructions explicitly said you may consult the standard library documentation during the interview.

Some third-party interview-prep sites say OpenAI is piloting coding rounds that allow AI, while others say AI is never allowed; the claims contradict each other, and nobody in the collected interview reports mentions such a round, so this site marks everything as "not allowed".

## Does the interviewer run your code, or just read it

**In coding rounds, the code runs.** But there are also cases where no tests are given and you have to walk through your code verbally; expect to see both.

| Stage | Where you write | Does the code run | Are tests provided |
| --- | --- | --- | --- |
| Coding round (60 minutes, sometimes 75) | Mostly CoderPad + Google Meet | Yes; output and logs appear below the editor pane, which is easy to miss while debugging | **Sometimes yes, sometimes no** (see below) |
| ML coding round (60 to 75 minutes) | Same as above | Yes | Same as above |
| System design (60 minutes) | A whiteboard such as Excalidraw + video | No runnable code | — |
| Technical deep dive | Your own slides | — | — |

On tests and how problems are given, three points come up again and again in the reports:

- **The problem is released one part at a time.** At first you only get part 1, and you see the next part only after the current one passes; your code carries forward the whole way. Going fast early buys time for the later parts; tearing up your design halfway through is very costly.
- **Both test situations happen.** Some problems come with tests (visible or hidden) that your code must pass; others give you nothing, and you write your own tests or walk through the code verbally.
- **The language is usually up to you**, and Python is the most common; people who used verbose languages said they ran short on time.

## What this means for practice

- The OpenAI coding problems on this site are all split into several parts. **Finish a part and get it running correctly before looking at the next one**, and time yourself: in the real interview you cannot see the later parts.
- Turn off AI assistants **and autocomplete** while practicing. This site's editor has no autocomplete anyway.
- For each part, write two or three test cases yourself and run them; also practice walking through the code verbally without running it.
- For system design problems, spend an hour talking through the whole problem first, then read the reference solution.

## What we could not verify

- The official interview guide's wording (it could not be opened).
- Whether there is a pilot round that allows AI.
- Whether "turn off autocomplete" is a requirement of particular rounds or a general one: the reports only say "at least one round".
