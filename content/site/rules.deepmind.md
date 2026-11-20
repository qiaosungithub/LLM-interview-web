# Google DeepMind: can you use AI, and what is the interview environment

**In short: unless told otherwise, no AI in live interviews or interview tasks.** Coding rounds take place in a shared editor that can run code; the ML and math knowledge questions are answered verbally.

Every point below is labeled with its source: **Official** means Google DeepMind's [interview guide PDF](https://storage.googleapis.com/deepmind-media/DeepMind.com/Assets/Docs/interviewing-at-google-deepmind.pdf) (re-downloaded and checked against the original on 2026-11-17); **Candidate reports** means claims from the collected interview reports.

## Can you use AI in each stage

| Stage | Can you use AI | Source |
| --- | --- | --- |
| Preparing for interviews | Yes | Official |
| Coding rounds, ML coding or debugging rounds | No | Official |
| Verbal ML and math knowledge questions | No | Official |
| ML system design | No | Official |
| Interview tasks (if assigned) | No | Official |
| Behavioral, hiring manager | Conversation rounds; no tools involved | — |

The official wording is a single paragraph, under the question "Can I use AI during the interview process?" in the interview guide:

> You're free to use AI to help you prepare for interviews. But – unless told otherwise – please don't use any AI tools during live interviews, or during interview tasks.

"unless told otherwise" leaves the door open. The collected materials mention that in December 2025, GDM's head of recruiting said in an interview that there is a pilot for coding assessments: candidates use an AI system of their choice, and the assessment looks at how they use it. Nobody in the interview reports has encountered it so far, and that interview was not rechecked this time, so this site marks everything as "not allowed".

## Does the interviewer run your code, or just read it

**In coding rounds, the code runs, and sometimes it must pass test cases the interviewer has prepared.** Knowledge-question rounds involve no code.

| Stage | Format | Does the code run | Are tests provided |
| --- | --- | --- | --- |
| Coding round (about an hour) | Video + a shared editor such as CoderPad | Yes | **Sometimes**: you have to pass the interviewer's test cases (one candidate passed three out of four); some problems instead ask you to write your own unit tests and handle edge cases such as `None` |
| ML coding or debugging round (some roles) | Same as above | Yes | No clear reports |
| ML and math knowledge questions (about an hour) | Video, **answered verbally** | No code | — |
| ML system design (varies by team) | Video, verbal discussion; some are rapid-fire Q&A with the interviewer setting the pace | No code | — |
| Behavioral, hiring manager | Video conversation | — | — |

The official guide says only that most interviews are virtual, held over Google Meet, and require screen sharing; "the shared editor can run code", "something like CoderPad", "you have to pass test cases" and "knowledge questions are verbal" all come from candidate reports.

The official guide does not publish the details of each round; it says that once invited you will receive "detailed, role-specific information to help you prepare at every stage", so the specific format is whatever the materials from your recruiter say.

## What this means for practice

- **Coding problems**: once you finish, write your own test cases and run them, especially edge cases; in the interview you may be asked to write unit tests on the spot. Turn off AI assistants while practicing.
- **Knowledge questions**: the real interview is verbal. This site splits each question onto its own card; we suggest **saying your answer out loud first, then typing up the key points** to save, rather than writing a long essay straight away. For derivations, you can work on paper and then record the conclusion.
- **Design problems**: you need to be able to explain clearly even without a whiteboard; when practicing, walk through it verbally in the order problem, data, model, evaluation, launch.

## What we could not verify

- Whether the AI-assisted coding assessment pilot is already in use (only one interview mentions it).
- Whether ML coding or debugging rounds provide tests.
- Whether the knowledge-question round allows writing formulas in a shared doc: the reports only say it is verbal.
