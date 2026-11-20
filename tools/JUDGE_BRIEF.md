# Writing a problem's judge

The site judges code problems like an online judge. The user writes `answer.py` in the browser. **运行样例** runs the
statement's examples, **提交** runs every test, and **自定义运行** runs the answer followed by a few lines of their own.
The judge runs on the server in a bubblewrap sandbox: `tools/judge/run.py` starts it, and `tools/judge/harness.py`
runs the tests inside it. A failing test shows the call, the expected value, what came back, the answer's prints and a
traceback whose line numbers jump to the editor. This brief says how to write the judge for one problem. Follow it
exactly. Finished examples, read the first one before starting:
- `content/schuture/Anthropic/coding/longest-match-tokenizer/judge/`: functions, samples, edge cases, a TLE case.
- `content/extra/Google-DeepMind/coding/keyframe-placement-dp/judge/`: a DP checked against a brute force in
  `ref.py`, big inputs built at import time, a message saying which half of a tuple answer is wrong.
- `content/extra/Anthropic/take-home/constrained-machine-puzzles-a/judge/`: a puzzle judge that runs the page's own
  checker and reports scores with `info()`.
- `content/schuture/Anthropic/coding/oa-bank-system/judge/`: a "levels" class API. Scripted operation sequences are
  checked after every call, with the replay passed to `calling()`. Random sequences are compared with `ref.py` and
  shrunk to a short replay when they fail.
- `content/schuture/OpenAI/coding/transformer-bug-hunt/judge/`: a PyTorch debugging problem. The starter is the
  buggy file; each bug has a test aimed at its symptom; the judge trains the answer's model itself.
- `content/schuture/Google-DeepMind/coding/mlp-backprop-from-scratch/judge/`: a NumPy problem. Gradients are compared
  with an independent reference and with central differences; training randomness is handled by retrying with fresh
  seeds; a statistical test (KS) checks a sampler.
- `content/extra/Anthropic/coding/reservation-service-concurrency/judge/`: a thread-safety problem. `ref.py` passes
  int/str subclasses whose comparisons pause the armed threads, so races and lock-order deadlocks are forced
  deterministically (barriers, hold-inside, lockstep); jittered stress rounds and invariant checks back them up.

## What goes in `judge/`

```
<problem folder>/judge/
  judge.yaml      limits and notes (all keys optional)
  starter.py      what the editor shows before the user has written anything
  tests.py        the tests, written with tools/judge/judgelib.py
  ref.py          optional helpers for tests.py: a reference implementation, random input generators
  solution.py     the page's reference answer as one file; it must pass everything
  wrong/*.py      at least two plausible wrong answers; each must fail
```

Inside the sandbox, `judge/` is mounted read-only at `/judge` and is on `sys.path`. So `tests.py` can `import ref`,
and the user's code could too, which is acceptable for a practice site. `solution.py` and `wrong/` are hidden from
the sandbox, so tests must never import them.

### judge.yaml

| key | default | meaning |
| --- | --- | --- |
| `timeout` | 20 | Wall-clock seconds for the whole run, 1 to 120. When it expires the sandbox is killed and the results so far are kept. |
| `case_cpu` | 2 | CPU seconds per case; one case can override it with `@case(..., cpu=N)`. |
| `load_cpu` | 10 | CPU seconds to import the answer. Raise it for torch: `import torch` costs about 3 to 6 s. |
| `memory_gb` | 8 | Address-space limit. torch needs about 6; leave the default unless a test needs more. |
| `nproc` | 256 | Processes plus threads inside the sandbox. |
| `special` | none | A short tag for unusual judges: `concurrency`, `puzzle`, `perf`, `debug`. |
| `note` | none | One or two Chinese sentences shown above the results, e.g. what passing does not prove. |

Set `timeout` to cover a wrong answer that times out on every case: the sum of the case limits, plus about 5 s.
Never go past 120. In the sandbox the CPU timer counts every thread of the process.

## starter.py

- Copy the statement's signatures exactly: function and class names, parameter names, type hints. Under each,
  write a one-line English docstring taken from the statement, then `raise NotImplementedError`.
- Classes get `__init__` and every method the statement names.
- A debugging problem ("find and fix the bug") starts from the buggy code exactly as the statement gives it.
- Never give hints beyond the statement. Never leave helper code that solves part of the problem.
- The file must compile and must fail inside a test with WA or RE. It must not fail while loading.
  `judge_check.py` checks this. One exception: with `special: perf`, the starter may be an official baseline that
  is correct but slow, and then it passes.
- 自定义运行 runs the answer as `python answer.py` would (`__name__ == "__main__"`, `sys.argv == ["answer.py"]`).

## tests.py

```python
from judgelib import calling, case, expect, expect_close, expect_raises, expect_valid, expect_true, info, rng, show_call
import ref

@case("Part 1", sample=True, title="题目例子 abcxbcdy")
def p1_example(sol):
    expect(sol.tokenize("abcxbcdy", dict(VOCAB), 0), [3, 0, 5, 0], call='tokenize("abcxbcdy", vocab, 0)')
```

A case receives the answer module as `sol` and fails by raising. Every `expect*` raises `Mismatch`, and a plain
`assert` also counts as a wrong answer. Rules:

1. **Structure**
   - Use `@case` on top-level functions only. Never pass `name=`. The part name is a string literal using the
     statement's own names ("Part 1", "Level 2"); a single-part problem uses "Part 1".
   - Cases run in file order. Keep each part's cases together and put its sample cases first.
   - `title` is a short Chinese label (at most about 20 characters) that says what the case checks, e.g.
     "空输入", "并列时任选一个都算对", "10⁵ 个元素：要求 O(n log n)".
2. **Samples** (`sample=True`)
   - Write each example in the statement as its own sample case, at least one per part.
   - Hard-code the input and expected output from the statement; do not compute them from `ref`.
   - If an example in the statement is wrong, fix the statement. Never make the test agree with a wrong example.
3. **Every expectation must follow from the statement**
   - Never test behaviour the statement leaves open: ordering of ties, order of a returned set, which of several
     optimal answers, exact error message text, float formatting.
   - Accept equivalent forms the statement does not make matter: a 2-element list where it says tuple, a numpy
     integer where it says int, any iterable where it says "return the items". Normalize `got` before comparing.
   - When more than one answer is correct, check with `expect_valid(got, checker, want=one_correct_answer)`. The
     checker returns None or True when the answer is acceptable, or a Chinese sentence saying what is wrong.
   - Floats: use `expect_close` with tolerances the statement or the maths justifies. It checks the shape first
     and reports a nan or inf in the answer by position, so no separate finiteness check is needed.
   - Exceptions: use `expect_raises` with the exception type the statement names, and nothing narrower.
   - If the statement is silent on something a fair test needs, see "When the statement is unclear" below.
4. **What the user sees**
   - Each `call=` shows how to reproduce the failure: valid Python they could paste into 自定义运行 where
     practical. Use `show_call("fn", *args)` to build it; it prints each argument in full when it fits in 600
     characters.
   - For a big input, describe it ("tokenize(<100000 个字符的文本>, …)") rather than dumping it.
   - For a stateful API (classes, "levels" problems), drive a sequence of operations and check after each one.
     Before each operation, pass the replay so far (the operations as code, at most the last 10 or so, the current
     one marked) to `calling(text)`. If the answer then raises or runs out of time, the result shows that replay;
     `expect*()` without `call=` uses it too.
   - Messages are in Chinese.
5. **Inputs**
   - Pass copies of mutable inputs (`dict(vocab)`, `list(xs)`), and compute the expected value before calling the
     answer. That way an answer that mutates its input cannot corrupt the check.
6. **Random tests**
   - Use `r = rng()` (or `rng("salt")`) only, never `random` directly. `judge_check.py` runs two seeds, and every
     seed must pass for the reference.
   - Compare against `ref`, a straightforward implementation that is obviously right. A brute force is ideal.
7. **Edge cases**
   - Cover what the statement's constraints allow: empty input, one element, duplicates, negative numbers, the
     largest sizes, unicode if the input is text.
   - Do not invent inputs the constraints rule out.
8. **Complexity.** When the statement states a time bound or input sizes that rule out the naive method:
   - Add a case at a size where the reference uses at most 1/5 of the case's CPU limit and the naive method uses
     more than 3 times it. Measure both; do not guess.
   - Set `cpu=` on that case.
   - Put the naive but correct answer in `wrong/` with `# expect: TLE`.
   - The CPU timer covers the whole case, including the test's own work (building inputs, running `ref`). Keep that
     work small next to the limit, or build big inputs at import time with `rng("salt")`. Import time is not
     charged to the answer.
9. **Printing problems.** When the answer must print, capture the output with
   `contextlib.redirect_stdout(io.StringIO())` and compare lines.
10. **Files.** Each run gets an empty writable `/tmp`, shared by all cases of that run. Use `tempfile.mkdtemp()` per
    case. There is no network and nothing outside `/judge`, `/tmp` and the Python install.
11. **Special judges**
    - `concurrency`: repeat each scenario many times, with many threads and random sleeps where the statement allows.
      The wall-clock limit catches deadlocks. Set `note` to say a pass does not prove there is no race condition.
    - `puzzle` and `perf`: run the user's program with the page's own interpreter or simulator, and check
      correctness first. Report the score with `info("指令数 12（已知最短 9）")`; info lines show even when a case
      passes. Fail only on what the statement requires (a hard limit). Report scores, do not grade them.
    - torch: import torch at the top of `tests.py` (loading the tests is not charged to the answer). Raise `load_cpu`
      and `timeout`. Keep tensors small; the sandbox has two threads and no GPU.
    - processes: the start method is left at the Linux default (fork), and the answer may pick spawn or forkserver.
      Such a worker imports the answer as module `sol`; the harness puts a stub `sol.py` last on `sys.path` that runs
      `/work/answer.py`. If a forked worker inherits counters from `ref.py` that a spawned one would not, ship a
      `judge/sol.py` that imports `ref` first and then runs the answer, as image-processing-pipeline does, and pass
      shared state through environment variables or files in `/tmp`, not module globals.

## solution.py

This is the page's reference answer (from the 参考解答 tab) as one file. Join the parts if the page builds the
answer up part by part, and add the imports. Do not change the logic. If it fails a test, either the test is wrong
or the page is. Never weaken a test so that a wrong reference passes; fix the page instead (see below).

## wrong/

- Write at least two wrong answers, ideally three to five. The first line is `# expect: WA` (or `RE` or `TLE`);
  the second line is an English comment saying what is wrong.
- Prefer the mistakes the page itself discusses (pitfalls, follow-ups, audit notes), then the classic mistakes for
  this kind of problem: off-by-one errors, missing an edge case, ignoring ties, mutating the input, a slow method
  when there is a complexity requirement.
- Each must be a plausible attempt that passes the samples or most of them. Do not write a trivial `return None`.
- Each must be caught by a test that targets its mistake, not only by a sample.

## Problems that get no judge, or a partial one

Judge only what has one right output, or a checker that accepts every right output. If after reading the page you
find the problem has none (a code review, an open design question, an estimate, prose answers), write no `judge/`
and say why in your report. If only some parts qualify, judge those parts and say in `note` which parts are not
judged (for example, "只评测修复后的代码；找出问题的书面部分不评测"). A judge that needs something the sandbox
lacks (network, a GPU, files outside the problem folder) is reported, not faked.

## When the statement is unclear

Writing tests is where a vague statement shows. In order of preference:
1. Make the tests independent of the vague point: use a checker that accepts every reasonable reading.
2. If that is impossible, make the smallest edit to the statement in `README.zh.md` and the same edit in
   `README.md` (English) when it exists. Edit the reference answer only if it was wrong.
3. Do not edit `audit.json`. List every statement or solution edit in your final report as where / problem /
   change, so it can be recorded.

## Checking your work

```
cd /kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep
.venv/bin/python tools/judge_check.py <slug> -v          # must print "ok"; read the warnings too
.venv/bin/python tools/judge/run.py <problem folder> judge/starter.py --mode samples   # how a first run reads
.venv/bin/python build.py --check                        # must report 0 problems for your folders
```

Run your own scripts with `PYTHONDONTWRITEBYTECODE=1`, and delete any `judge/__pycache__` (also under `wrong/`)
before you finish.

`judge_check.py` checks the following:
- `solution.py` passes under two seeds, and passes the samples.
- Every `wrong/*.py` fails, with its `# expect:` verdict.
- `starter.py` fails cleanly.
- The cases listed on the site match the cases that run.
- The reference uses at most half of each limit.

It prints a warning when a limit is tight. A warning means the limit should be raised, or the test's own work made
smaller.

Do not touch `answers/` (the user's data), `server.py`, `site/`, `tools/judge/`, or other problems' folders. Do not
start or restart any server.
