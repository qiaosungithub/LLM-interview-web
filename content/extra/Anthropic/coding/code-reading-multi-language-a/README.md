# Code reading in several languages (A): C, JavaScript, SQL

## Problem

This round involves no writing code, only reading it. The interviewer pastes three snippets into a shared
document one after another, each in a different language, with about ten minutes per snippet. The code can't
be run; you talk through it as you read. The three snippets come from the same fictional video-model training
project and are independent of each other.

Answer each snippet using the following five questions:

1. What is this code trying to do?
2. For the given input, what does it actually output? (Be specific, down to every line and every number.)
3. Where are the bugs, and where are the risks?
4. How would you fix it?
5. If the author were here, what would you ask them?

Once you've answered the five questions, the interviewer usually follows up on your answers with a few more
questions, such as what happens with a different input or on a different platform.
These follow-ups and their answers are in the reference solution; answer the five questions yourself first
before looking.

Environment: C is compiled with gcc on x86-64 Linux, where `int` is 32 bits; JavaScript runs on Node.js (16 or
newer); SQL runs on SQLite. The timers in the JavaScript snippet fire in order from shortest to longest delay,
and nothing else slows them down.

### Part 1 — C: statistics for one grayscale frame

```c
/* frame_stats.c -- summary statistics for one 8-bit grayscale frame. */
#include <stdio.h>

typedef struct {
    int mean;        /* average pixel value, 0..255 */
    int bright_pct;  /* percentage of pixels >= threshold, 0..100 */
    int contrast;    /* brightest pixel minus darkest pixel */
} FrameStats;

FrameStats frame_stats(const char *pix, int width, int height, int threshold) {
    FrameStats s = {0, 0, 0};
    int n = width * height;
    int sum = 0;
    int bright = 0;
    char lo = pix[0];
    char hi = pix[0];

    for (int i = 0; i < n; i++) {
        sum += pix[i];
        if (pix[i] >= threshold) bright++;
        if (pix[i] < lo) lo = pix[i];
        if (pix[i] > hi) hi = pix[i];
    }
    s.mean = sum / n;
    s.bright_pct = bright / n * 100;
    s.contrast = hi - lo;
    return s;
}

int main(void) {
    unsigned char frame[6] = {10, 20, 30, 200, 250, 90};   /* 3 x 2 pixels */
    FrameStats s = frame_stats((const char *)frame, 3, 2, 128);
    printf("mean=%d bright=%d%% contrast=%d\n", s.mean, s.bright_pct, s.contrast);
    return 0;
}
```

### Part 2 — JavaScript: fetching thumbnails in bulk, with retries on failure

```js
// thumbs.js -- fetch thumbnails for a list of frame ids, retrying failures.
const LATENCY_MS = { 1: 30, 2: 10, 3: 20 };       // fake network latency per frame id

function fetchThumb(id) {                         // stand-in for the real HTTP call
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (id === 3) reject(new Error("503 for frame " + id));
      else resolve("thumb-" + id);
    }, LATENCY_MS[id]);
  });
}

// Returns the thumbnails for `ids`. opts.retries = extra attempts per frame (default 2).
async function fetchAll(ids, opts) {
  const retries = opts.retries || 2;
  const thumbs = [];
  ids.forEach(async (id) => {
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        thumbs.push(await fetchThumb(id));
        return;
      } catch (err) {
        console.log(`frame ${id}: attempt ${attempt} failed`);
      }
    }
  });
  console.log(`fetched ${thumbs.length} of ${ids.length}`);
  return thumbs;
}

fetchAll([1, 2, 3], { retries: 0 }).then((thumbs) => {
  console.log("done:", thumbs);
  setTimeout(() => console.log("300 ms later:", thumbs), 300);
});
```

### Part 3 — SQL: the weekly experiment report

Schema and data:

```sql
-- Experiment tracking. One row per training run.
CREATE TABLE runs (
  id        INTEGER PRIMARY KEY,
  model     TEXT    NOT NULL,
  owner     TEXT,               -- NULL for runs launched by the nightly scheduler
  status    TEXT    NOT NULL,   -- 'done' | 'failed' | 'running'
  val_loss  REAL,               -- NULL until the first evaluation has finished
  gpu_hours INTEGER NOT NULL
);
CREATE TABLE retired_models (model TEXT);   -- models the weekly report should skip

INSERT INTO runs (id, model, owner, status, val_loss, gpu_hours) VALUES
  (1, 'vit-b',  'alice',    'done',    0.42, 10),
  (2, 'vit-b',  'bob',      'done',    0.38, 12),
  (3, 'vit-b',  'bob',      'failed',  NULL,  3),
  (4, 'vit-b',  NULL,       'failed',  NULL,  2),
  (5, 'vit-l',  'alice',    'done',    0.31, 40),
  (6, 'vit-l',  'test-bot', 'done',    0.90,  1),
  (7, 'vit-l',  NULL,       'running', NULL,  9),
  (8, 'vit-h',  'carol',    'running', NULL, 20),
  (9, 'resnet', 'bob',      'done',    0.55,  5);
INSERT INTO retired_models (model) VALUES ('resnet'), (NULL);
```

The query that generates the weekly report:

```sql
-- Weekly report: one row per model, best (lowest) average loss first.
-- Skip retired models and runs launched by test-bot.
SELECT model,
       COUNT(*)                                                        AS runs,
       SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) / COUNT(*) * 100 AS done_pct,
       ROUND(AVG(val_loss), 3)                                         AS avg_loss,
       SUM(gpu_hours) / COUNT(*)                                       AS gpu_h_per_run
FROM runs
WHERE owner != 'test-bot'
  AND model NOT IN (SELECT model FROM retired_models)
GROUP BY model
ORDER BY avg_loss;
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Each of the three snippets has one kind of semantics that "changes when you change languages": in C it's
integer types (whether `char` is signed, integer division, overflow); in JavaScript it's asynchronous
execution order and "falsy" values; in SQL it's the three-valued logic of `NULL`.
When reading, first state the intent in one sentence, then work through the sample input line by line, and
only then talk about risks and fixes.
Every "observed output" below was actually produced on this machine: gcc 11.4.0 (x86-64 Linux), Node
v24.22.2, SQLite 3.53.1.

### Part 1

**What it's trying to do.** Compute three numbers for one 8-bit grayscale frame: the mean brightness; the
percentage of pixels whose brightness is at or above a threshold; and the difference between the brightest
and darkest pixels.

**Actual output.** The sample's 6 pixels are 10, 20, 30, 200, 250, 90, with threshold 128.
The correct answer would be `mean=100` (600 ÷ 6), `bright=33%` (2 out of 6), `contrast=240` (250 − 10). The
observed output is:

```text
mean=14 bright=0% contrast=146
```

All three numbers are wrong. Taking them one at a time:

- `pix` has type `const char *`. On x86-64 Linux, gcc's `char` is signed, with range −128 to 127.
  The bytes 200 and 250 are read as −56 and −6. So the sum is 10 + 20 + 30 − 56 − 6 + 90 = 88, and 88 ÷ 6
  truncates to 14.
- `pix[i] >= threshold`: the largest value read is only 90, so none is at or above 128, and `bright` is 0.
- `lo` and `hi` are also `char`: the minimum is −56 and the maximum is 90, a difference of 146.

**The interviewer follows up: is this code's output the same on every platform?** No. Whether `char` is
signed is left by the C standard to each implementation (implementation-defined).
On ARM, Linux defaults to unsigned. There's no ARM machine here, so gcc's `-funsigned-char` option is used to
simulate it, and the observed output is:

```text
mean=100 bright=0% contrast=240
```

`mean` and `contrast` are now right, but `bright` is still 0. The reason is `bright / n * 100`: the integer
division happens first, 2 ÷ 6 gives 0, and multiplying by 100 is still 0.
This expression only gives 100 when every pixel is bright; in every other case it's 0.

**The interviewer follows up: what happens if the input is an 8K frame (7680×4320), or an empty frame
(`width * height` equals 0)?**
Both inputs lead to undefined behavior; see items 3 and 4 of the issue list below.

**Issue list.**

1. **The signedness of `char`.** The same source gives different results on two platforms. `-Wall -Wextra`
   produces not a single warning (nor does adding `-Wconversion`).
   Fix: always use `uint8_t` (or `unsigned char`) for byte data.
2. **Dividing before multiplying.** `bright / n * 100` should be written `bright * 100 / n`.
3. **`int sum` can overflow.** Each pixel is at most 255 and `int` tops out at 2147483647, so more than
   8421504 pixels can overflow.
   The sum of an all-white 4K frame (3840×2160) is 2115072000, which just fits; an 8K frame (33177600
   pixels) doesn't.
   Signed integer overflow is undefined behavior (UB) in C: the standard doesn't specify what happens.
   Observed: given an 8K frame with every pixel 127, UBSan reports `signed integer overflow`, and the program
   goes on to print `mean=-2`.
   That −2 is only what happened this time; it can't be taken as a rule. `int n = width * height` is the
   same: with a larger width and height, it overflows too.
   Fix: use `size_t` for counts and `uint64_t` for the sum.
   Incidentally, on a platform where `char` is signed, the all-white 8K frame doesn't overflow at all, because
   every 255 is read as −1, and the output is `mean=-1 bright=0% contrast=0`.
4. **Empty frame.** When `n` is 0, `sum / n` is a division by zero, which is also UB. Even before that,
   `pix[0]` has already read a byte that isn't part of the frame.
   Observed: UBSan reports `division by zero`. Without a sanitizer, the same program compiled with `-O0` is
   killed by a SIGFPE signal, while compiled with `-O2` it prints `mean=0 bright=0% contrast=0` normally and
   returns 0.
   The same UB behaves two different ways just by changing the optimization level. Fix: check for empty input
   at the start, and document in a comment what an empty frame returns.
5. **Minor issues.** Results are rounded down, which the comment doesn't say; `width` and `height` can be
   passed negative values; `threshold` isn't range-checked.

**The fixed version.**

```c
/* frame_stats.c -- summary statistics for one 8-bit grayscale frame (fixed). */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    int mean;        /* average pixel value, 0..255, rounded down */
    int bright_pct;  /* percentage of pixels >= threshold, 0..100, rounded down */
    int contrast;    /* brightest pixel minus darkest pixel */
} FrameStats;

/* An empty frame returns all zeros. */
FrameStats frame_stats(const uint8_t *pix, size_t width, size_t height, int threshold) {
    FrameStats s = {0, 0, 0};
    size_t n = width * height;
    if (n == 0) return s;
    uint64_t sum = 0;
    size_t bright = 0;
    uint8_t lo = pix[0];
    uint8_t hi = pix[0];

    for (size_t i = 0; i < n; i++) {
        sum += pix[i];
        if (pix[i] >= threshold) bright++;
        if (pix[i] < lo) lo = pix[i];
        if (pix[i] > hi) hi = pix[i];
    }
    s.mean = (int)(sum / n);
    s.bright_pct = (int)(bright * 100 / n);
    s.contrast = hi - lo;
    return s;
}

int main(void) {
    uint8_t frame[6] = {10, 20, 30, 200, 250, 90};   /* 3 x 2 pixels */
    FrameStats s = frame_stats(frame, 3, 2, 128);
    printf("mean=%d bright=%d%% contrast=%d\n", s.mean, s.bright_pct, s.contrast);
    return 0;
}
```

Observed output (the same when compiled with either `-fsigned-char` or `-funsigned-char`):

```text
mean=100 bright=33% contrast=240
```

For an all-white 8K frame the fixed version outputs `mean=255 bright=100% contrast=0`, for an empty frame it
outputs all zeros, and neither ASan nor UBSan reports anything.

**Questions for the author.**

- What type is the caller's buffer? Why does this take `const char *`? (Most likely a `char` buffer used for
  reading a file was passed straight in.)
- Which platforms does this code need to run on? Are there any ARM dev boards or phones?
- How big is the largest frame? Could a buffer with several frames concatenated be passed in at once?
- Is an empty frame a valid input? What should it return?
- Should the mean and the percentage have decimals? Round down or round to nearest?
- Are there test images with known answers, such as all black, all white, half black and half white?

### Part 2

**What it's trying to do.** Fetch thumbnails for a set of frame ids concurrently, retry the failures a few
times, and finally return an array of thumbnails.

**Actual output.** The observed output (the same over many runs):

```text
fetched 0 of 3
done: []
frame 3: attempt 0 failed
frame 3: attempt 1 failed
frame 3: attempt 2 failed
300 ms later: [ 'thumb-2', 'thumb-1' ]
```

Walking through it in time order:

1. `ids.forEach(async (id) => {...})`: `forEach` calls the callback three times in turn. Each callback is an
   `async` function, which hands control back as soon as it reaches its first `await` and returns a Promise.
   `forEach` ignores return values, so nobody waits on these three Promises.
2. So `forEach` returns immediately, before a single request has completed. It prints `fetched 0 of 3` and
   returns an empty array.
3. The caller's `.then` runs and prints `done: []`.
4. At 10 ms: frame 2 succeeds, and `thumb-2` is `push`ed into the array that has already been returned.
5. At 20 ms: frame 3 fails for the first time and prints `frame 3: attempt 0 failed`. The caller passed
   `retries: 0`, meaning no retries.
   But `0 || 2` evaluates to 2, because 0 is falsy in JavaScript. So a second request goes out immediately.
6. At 30 ms: frame 1 succeeds, and `thumb-1` is `push`ed.
7. At 40 ms and 60 ms: frame 3's two retries each wait 20 ms and both fail, printing the `attempt 1` and
   `attempt 2` lines.
   When the loop ends, this callback leaves nothing behind and reports no error.
8. At 300 ms: the caller looks at the same array again, which now holds `[ 'thumb-2', 'thumb-1' ]`. The order
   is completion order, not the order of `ids`; frame 3 left no trace at all.

(A run with a timestamp added after each line of output showed the times 10, 21, 30, 41, 60 ms, matching the
reasoning above.)

**Issue list.**

1. **`forEach` with an `async` callback, and nobody waits.** The function returns before all the requests
   complete.
   Fix: use `ids.map(...)` to get an array of Promises, then `await Promise.all(...)` or
   `Promise.allSettled(...)`.
2. **`opts.retries || 2` treats 0 as "not passed".** Fix: use `??`, which only takes the default when the
   left side is `null` or `undefined`.
3. **Results are in completion order.** The caller has no way of knowing which image belongs to which frame.
   Fixing only item 1 still returns `[ 'thumb-2', 'thumb-1' ]`.
   Fix: place results by their position in `ids`; the results of `map` are already in that order.
4. **Failures are swallowed.** Once the retries are used up, it neither throws nor leaves a placeholder. Fix:
   pick a convention. Below, `null` is used as a placeholder; another convention is to have the whole
   `fetchAll` throw.
5. **`opts` has no default.** See the first follow-up below.
6. **Risks.** There's no wait between retries, so a service that is currently returning 503 gets hit three
   times in a row; all requests go out at once, with no concurrency limit when there are many ids.

**The interviewer follows up: what happens if the caller writes `fetchAll([1, 2, 3])` without the second
argument?** `opts` is `undefined`, and reading `opts.retries` throws a `TypeError`.
The exception happens inside an `async` function, so it becomes a rejected Promise. The caller only wrote
`.then`, with no `.catch`, so nobody handles the rejection.
Observed: Node prints `TypeError: Cannot read properties of undefined (reading 'retries')`, the process exits
with exit code 1, and the `done` line isn't printed.

**The interviewer follows up: if the requests were sent one after another (the next one only after the
previous finishes) instead of all at once, what would be different?**
Use a `for (const id of ids)` loop and `await` inside the loop body. The result array is the same; the time
taken is different.
Observed: the three requests have delays of 30, 10 and 20 ms; one after another took about 60 ms (the sum of
the three), all at once took about 30 ms (the largest of the three).
The upside of one after another is less load on the service and naturally correct ordering; the downside is
that it's slow.

**The fixed version.**

```js
// thumbs.js -- fetch thumbnails for a list of frame ids, retrying failures (fixed).
const LATENCY_MS = { 1: 30, 2: 10, 3: 20 };       // fake network latency per frame id

function fetchThumb(id) {                         // stand-in for the real HTTP call
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (id === 3) reject(new Error("503 for frame " + id));
      else resolve("thumb-" + id);
    }, LATENCY_MS[id]);
  });
}

async function fetchWithRetry(id, retries) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fetchThumb(id);
    } catch (err) {
      lastErr = err;
      console.log(`frame ${id}: attempt ${attempt} failed`);
    }
  }
  throw lastErr;
}

// Returns one entry per id, in the order of `ids`; null marks a frame that failed every attempt.
// opts.retries = extra attempts per frame (default 2; 0 means "try once").
async function fetchAll(ids, opts = {}) {
  const retries = opts.retries ?? 2;
  const settled = await Promise.allSettled(ids.map((id) => fetchWithRetry(id, retries)));
  const thumbs = settled.map((r) => (r.status === "fulfilled" ? r.value : null));
  console.log(`fetched ${thumbs.filter((t) => t !== null).length} of ${ids.length}`);
  return thumbs;
}

fetchAll([1, 2, 3], { retries: 0 }).then((thumbs) => {
  console.log("done:", thumbs);
  setTimeout(() => console.log("300 ms later:", thumbs), 300);
});
```

Observed output:

```text
frame 3: attempt 0 failed
fetched 2 of 3
done: [ 'thumb-1', 'thumb-2', null ]
300 ms later: [ 'thumb-1', 'thumb-2', null ]
```

**Questions for the author.**

- Does the caller need the results to line up one-to-one with `ids`? Should a failed frame get a placeholder,
  throw, or can it be dropped?
- Is `retries: 0` a valid use? (If it is, you can't use `||`.)
- What's the maximum number of ids in one call? Does the other service have any concurrency or rate-limit
  requirements?
- Should there be a wait between retries? Which errors are worth retrying? (A 503 is worth it, a 404 isn't.)
- Is there a test covering the case "one frame keeps failing"?

### Part 3

**What it's trying to do.** Summarize a week of training runs by model: the number of runs, the fraction
completed, the mean validation loss, and the average GPU hours per run.
Skip retired models and runs launched by test-bot. The lowest mean loss comes first.

**What it actually returns.** Zero rows.

The reason is that `retired_models` has a `NULL` row. `x NOT IN ('resnet', NULL)` is equivalent to
`x != 'resnet' AND x != NULL`.
A comparison with `NULL` is neither true nor false, but `NULL` (unknown). SQL's logic has three values: true,
false and unknown; this is called three-valued logic.
For `vit-b`, the whole condition is "true AND unknown", which is unknown. For `resnet`, the condition is
false. `WHERE` keeps only rows where the condition is true, so not a single row is left.

**The interviewer follows up: if the `retired_models` table had only the `'resnet'` row, what would the query
return?** The observed result:

| model | runs | done_pct | avg_loss | gpu_h_per_run |
|---|---|---|---|---|
| vit-h | 1 | 0 | NULL | 20 |
| vit-l | 1 | 100 | 0.31 | 40 |
| vit-b | 3 | 0 | 0.4 | 8 |

Now the query returns something, but nearly every column has a problem.

**The interviewer follows up: runs 4 and 7 were launched by the nightly scheduler, and their `owner` is
`NULL`. Do they show up in the weekly report?**
No; see item 2 of the issue list below. The comment only says to skip test-bot, so they should have been
there.

**Issue list.**

1. **`NOT IN` meets `NULL`.** If the subquery contains even one `NULL`, `NOT IN` is never true.
   Fix: use `NOT EXISTS`. You should also add a `NOT NULL` constraint to `retired_models.model`.
2. **`owner != 'test-bot'` also filters out `NULL`.** `NULL != 'test-bot'` is unknown, not true.
   That's why runs 4 and 7 disappear: `vit-b` actually has 4 runs but the table shows 3; `vit-l` should have 2
   after removing test-bot, but the table shows 1.
   Fix: `(owner IS NULL OR owner != 'test-bot')`.
3. **`done_pct` can only be 0 or 100.** In SQLite an integer divided by an integer is still an integer.
   `vit-b`'s 2 ÷ 3 gives 0, and multiplying by 100 is still 0.
   Fix: multiply by `100.0` first, making the arithmetic floating point.
4. **`gpu_h_per_run` is truncated.** Integer division again: `vit-b`'s 25 ÷ 3 is truncated to 8. Fix: multiply
   by `1.0` first.
5. **`AVG` skips `NULL`.** `vit-b`'s 0.4 is the mean of the two runs that have been evaluated, not of all runs.
   That calculation is reasonable in itself, but someone reading the report can't tell how many runs were
   averaged. Fix: add a `COUNT(val_loss)` column, which counts only non-`NULL` values.
6. **Ordering.** `vit-h` has no results at all and its `avg_loss` is `NULL`, yet it sits in the "best"
   position. SQLite puts `NULL` first when sorting in ascending order.
   Fix: `ORDER BY avg_loss IS NULL, avg_loss`, sorting first by "has a value or not", then by the value.

**The fixed version.**

```sql
-- Weekly report: one row per model, best (lowest) average loss first; models with no loss yet go last.
-- Skip retired models and runs launched by test-bot (scheduler runs, owner IS NULL, are kept).
SELECT r.model,
       COUNT(*)                                                                        AS runs,
       ROUND(100.0 * SUM(CASE WHEN r.status = 'done' THEN 1 ELSE 0 END) / COUNT(*), 1) AS done_pct,
       COUNT(r.val_loss)                                                               AS evaluated,
       ROUND(AVG(r.val_loss), 3)                                                       AS avg_loss,
       ROUND(1.0 * SUM(r.gpu_hours) / COUNT(*), 2)                                     AS gpu_h_per_run
FROM runs AS r
WHERE (r.owner IS NULL OR r.owner != 'test-bot')
  AND NOT EXISTS (SELECT 1 FROM retired_models AS m WHERE m.model = r.model)
GROUP BY r.model
ORDER BY avg_loss IS NULL, avg_loss;
```

The observed result (the same whether or not `retired_models` contains that `NULL` row):

| model | runs | done_pct | evaluated | avg_loss | gpu_h_per_run |
|---|---|---|---|---|---|
| vit-l | 2 | 50.0 | 1 | 0.31 | 24.5 |
| vit-b | 4 | 50.0 | 2 | 0.4 | 6.75 |
| vit-h | 1 | 0.0 | 0 | NULL | 20.0 |

**Questions for the author.**

- Why does `retired_models.model` allow `NULL`? How did that row get in?
- Should scheduler runs whose `owner` is `NULL` count in the weekly report? (The comment only says to skip
  test-bot.)
- Should the denominator of `done_pct` include runs that are still in progress?
- `avg_loss` only averages the evaluated runs, and failed runs don't count. Could a model with many failures
  look better because of this?
- What kind of database will this query ultimately run on? (See the follow-up below.)

### Follow-ups

**1. What are the rules for "integer divided by integer" in these languages?**
C's integer division truncates toward zero: observed, `-7 / 2` is −3 and `-7 % 2` is −1.
JavaScript numbers are floating point by default: `7 / 2` is 3.5, and if you want an integer you have to write
`Math.trunc` yourself.
SQLite's integer division also truncates toward zero: `-7 / 2` is −3 and `2 / 3` is 0.
Python's `//` rounds down (floor): `-7 // 2` is −4 and `-7 % 2` is 1.
So when copying a formula from one language to another, negative numbers and "divide before multiplying" are
the two places most likely to go wrong.

**2. What does "no value" look like in each of these languages?**
C's `int` has no "empty" state; you can only agree on a special value, or pass a separate flag.
JavaScript has both `null` and `undefined`, plus a set of falsy values: `0`, `""`, `NaN`, `false`. `||` looks
at truthiness, while `??` only checks for `null` or `undefined`.
What's easy to get wrong: an empty array `[]`, an empty object `{}` and the string `"0"` are all truthy
(verified).
SQL's `NULL` means "unknown": comparing it with anything gives unknown, even `NULL = NULL`; testing for it
requires `IS NULL`; aggregate functions skip it.

**3. Why doesn't the compiler give a single warning for the C snippet? What should catch it instead?**
Because every line is legal C: `char` being signed is the implementation's choice, and dividing `int`s is a
well-defined operation.
Warnings can't detect "the intent doesn't match the type". There are three ways to catch it:
use types with a clear meaning (`uint8_t`); write tests with known answers, and include pixels greater than
127 in the test images;
run the tests with `-fsanitize=address,undefined`. Note that a sanitizer can only report on inputs the tests
actually exercise; the 6 pixels in the sample won't trigger any report.

**4. What is the fixed JavaScript version still missing before it can be used in production?**
Three things. First, a concurrency limit: right now all requests go out at once, and with thousands of ids it
would overwhelm the other side or get rate-limited, so you should cap how many are in flight at once.
Second, a wait before retrying, ideally doubling each time plus a bit of randomness.
Third, distinguishing kinds of errors: only a server being temporarily unavailable is worth retrying; a bad
argument gives the same result however many times you retry.

**5. What would be different if this SQL ran on a different database?**
This one wasn't run on this page; it's based on each database's documentation, and in the interview you
should say which one you're assuming.
Where `NULL` sorts differs between databases: SQLite puts `NULL` first in ascending order, while PostgreSQL
puts it last in ascending order by default.
Integer division also differs: SQLite and PostgreSQL truncate, while MySQL's `/` returns a decimal.
How `NOT IN` behaves with `NULL` is specified by the SQL standard and is the same everywhere.
So in the fix, prefer forms that don't depend on these differences: multiply by `100.0` explicitly, and state
explicitly which end `NULL` sorts to.

<details>
<summary>Verification code (runnable)</summary>

The code below puts every piece of source code from the problem and the solution verbatim into strings,
writes them to a temporary directory, and runs them with `gcc`, `node` and Python's built-in `sqlite3`.
If the machine has no `gcc` or `node`, the corresponding part prints a one-line note and is skipped.
For undefined behavior, only the sanitizer reports are asserted; behavior under a normal compile is checked
only on gcc 11, x86-64.

```python
import os
import platform
import random
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="code_reading_a_"))
random.seed(0)

def sub(src, old, new):
    """Replace exactly one occurrence, so a variant can never silently stay equal to its parent."""
    assert src.count(old) == 1, (old, src.count(old))
    return src.replace(old, new)

# A sanitizer binary refuses to start when another allocator is preloaded, so LD_PRELOAD is dropped.
RUN_ENV = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
RUN_ENV["ASAN_OPTIONS"] = "detect_leaks=0"

def run(cmd):
    p = subprocess.run([str(c) for c in cmd], cwd=WORK, env=RUN_ENV, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr

# ------------------------------------------------------------------ Part 1: C
C_BUGGY = r'''/* frame_stats.c -- summary statistics for one 8-bit grayscale frame. */
#include <stdio.h>

typedef struct {
    int mean;        /* average pixel value, 0..255 */
    int bright_pct;  /* percentage of pixels >= threshold, 0..100 */
    int contrast;    /* brightest pixel minus darkest pixel */
} FrameStats;

FrameStats frame_stats(const char *pix, int width, int height, int threshold) {
    FrameStats s = {0, 0, 0};
    int n = width * height;
    int sum = 0;
    int bright = 0;
    char lo = pix[0];
    char hi = pix[0];

    for (int i = 0; i < n; i++) {
        sum += pix[i];
        if (pix[i] >= threshold) bright++;
        if (pix[i] < lo) lo = pix[i];
        if (pix[i] > hi) hi = pix[i];
    }
    s.mean = sum / n;
    s.bright_pct = bright / n * 100;
    s.contrast = hi - lo;
    return s;
}

int main(void) {
    unsigned char frame[6] = {10, 20, 30, 200, 250, 90};   /* 3 x 2 pixels */
    FrameStats s = frame_stats((const char *)frame, 3, 2, 128);
    printf("mean=%d bright=%d%% contrast=%d\n", s.mean, s.bright_pct, s.contrast);
    return 0;
}
'''

C_FIXED = r'''/* frame_stats.c -- summary statistics for one 8-bit grayscale frame (fixed). */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    int mean;        /* average pixel value, 0..255, rounded down */
    int bright_pct;  /* percentage of pixels >= threshold, 0..100, rounded down */
    int contrast;    /* brightest pixel minus darkest pixel */
} FrameStats;

/* An empty frame returns all zeros. */
FrameStats frame_stats(const uint8_t *pix, size_t width, size_t height, int threshold) {
    FrameStats s = {0, 0, 0};
    size_t n = width * height;
    if (n == 0) return s;
    uint64_t sum = 0;
    size_t bright = 0;
    uint8_t lo = pix[0];
    uint8_t hi = pix[0];

    for (size_t i = 0; i < n; i++) {
        sum += pix[i];
        if (pix[i] >= threshold) bright++;
        if (pix[i] < lo) lo = pix[i];
        if (pix[i] > hi) hi = pix[i];
    }
    s.mean = (int)(sum / n);
    s.bright_pct = (int)(bright * 100 / n);
    s.contrast = hi - lo;
    return s;
}

int main(void) {
    uint8_t frame[6] = {10, 20, 30, 200, 250, 90};   /* 3 x 2 pixels */
    FrameStats s = frame_stats(frame, 3, 2, 128);
    printf("mean=%d bright=%d%% contrast=%d\n", s.mean, s.bright_pct, s.contrast);
    return 0;
}
'''

C_DRIVER_MAIN = r'''#include <stdlib.h>
#include <string.h>

/* test driver:  prog THRESHOLD WIDTH HEIGHT PIXEL...            explicit pixels, 0..255
                 prog THRESHOLD WIDTH HEIGHT fill VALUE SPARE    WIDTH*HEIGHT bytes of VALUE, plus SPARE unused bytes */
int main(int argc, char **argv) {
    int threshold = atoi(argv[1]);
    size_t w = strtoul(argv[2], NULL, 10), h = strtoul(argv[3], NULL, 10), n = w * h;
    unsigned char *frame;
    if (argc == 7 && strcmp(argv[4], "fill") == 0) {
        frame = malloc(n + strtoul(argv[6], NULL, 10));
        if (frame == NULL) return 2;
        memset(frame, atoi(argv[5]), n);
    } else {
        frame = malloc(n);
        if (frame == NULL) return 2;
        for (size_t i = 0; i < n; i++) frame[i] = (unsigned char)atoi(argv[4 + i]);
    }
    FrameStats s = frame_stats((const void *)frame, w, h, threshold);
    printf("mean=%d bright=%d%% contrast=%d\n", s.mean, s.bright_pct, s.contrast);
    free(frame);
    return 0;
}
'''

C_EMPTY_MAIN = r'''int main(void) {                                   /* an empty frame; the buffer still has one readable byte */
    unsigned char spare[1] = {7};
    FrameStats s = frame_stats((const void *)spare, 0, 0, 128);
    printf("mean=%d bright=%d%% contrast=%d\n", s.mean, s.bright_pct, s.contrast);
    return 0;
}
'''

def with_main(src, main):
    head, _ = src.split("\nint main(void) {")          # exactly one main, or this unpacking fails
    return head + "\n" + main

def gcc(name, src, *flags):
    (WORK / f"{name}.c").write_text(src)
    rc, _, err = run(["gcc", "-std=c11", "-Wall", "-Wextra", *flags, "-o", name, f"{name}.c"])
    assert rc == 0, err
    return err                                         # the compiler's warnings; "" when there are none

def exe(name, *args):
    return run([WORK / name, *args])

def c_div(a, b):                                       # C truncates toward zero; Python's // rounds down
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q

def predict_buggy(pixels, threshold, char_is_signed):
    """What the statement's C code computes, worked out without the C code (n > 0, no overflow)."""
    vals = [p - 256 if char_is_signed and p >= 128 else p for p in pixels]
    bright = sum(v >= threshold for v in vals)
    return "mean=%d bright=%d%% contrast=%d\n" % (
        c_div(sum(vals), len(vals)), c_div(bright, len(vals)) * 100, max(vals) - min(vals))

def predict_correct(pixels, threshold):
    n = len(pixels)
    return "mean=%d bright=%d%% contrast=%d\n" % (
        sum(pixels) // n, sum(p >= threshold for p in pixels) * 100 // n, max(pixels) - min(pixels))

SAMPLE = [10, 20, 30, 200, 250, 90]
BUGGY_SIGNED = "mean=14 bright=0% contrast=146\n"
BUGGY_UNSIGNED = "mean=100 bright=0% contrast=240\n"
C_WANT = "mean=100 bright=33% contrast=240\n"
assert predict_buggy(SAMPLE, 128, True) == BUGGY_SIGNED
assert predict_buggy(SAMPLE, 128, False) == BUGGY_UNSIGNED
assert predict_correct(SAMPLE, 128) == C_WANT
assert 10 + 20 + 30 - 56 - 6 + 90 == 88 and c_div(88, 6) == 14 and 90 - (-56) == 146
assert c_div(-7, 2) == -3 and -7 // 2 == -4 and -7 % 2 == 1
assert 2147483647 // 255 == 8421504 and 3840 * 2160 * 255 == 2115072000 <= 2**31 - 1 < 7680 * 4320 * 255
assert 7680 * 4320 == 33177600 and 7680 * 4320 * 127 > 2**31 - 1

def check_c():
    version = run(["gcc", "-dumpfullversion"])[1].strip()
    reference = version.startswith("11.") and platform.machine() == "x86_64"   # the toolchain the page was written on

    # 1. the two programs exactly as printed on the page, under both conventions for plain char
    for flag, buggy_out in (("-fsigned-char", BUGGY_SIGNED), ("-funsigned-char", BUGGY_UNSIGNED)):
        warnings = gcc("buggy", C_BUGGY, flag)
        assert exe("buggy") == (0, buggy_out, ""), (flag, exe("buggy"))
        if reference:
            assert warnings == "" and gcc("buggy", C_BUGGY, flag, "-Wconversion") == ""
        assert gcc("fixed", C_FIXED, flag, "-Wconversion") == ""
        assert exe("fixed") == (0, C_WANT, ""), (flag, exe("fixed"))
    gcc("buggy", C_BUGGY)
    default_out = exe("buggy")[1]
    assert default_out in (BUGGY_SIGNED, BUGGY_UNSIGNED)
    print(f"Part 1 (C): gcc {version} on {platform.machine()}, plain char is",
          "signed" if default_out == BUGGY_SIGNED else "unsigned", "by default")
    if platform.machine() == "x86_64":
        assert default_out == BUGGY_SIGNED

    # 2. random frames through a test driver, compared with the predictions above
    buggy_drv, fixed_drv = with_main(C_BUGGY, C_DRIVER_MAIN), with_main(C_FIXED, C_DRIVER_MAIN)
    gcc("buggy_s", buggy_drv, "-fsigned-char")
    gcc("buggy_u", buggy_drv, "-funsigned-char")
    gcc("fixed_s", fixed_drv, "-fsigned-char")
    gcc("fixed_u", fixed_drv, "-funsigned-char")
    for _ in range(50):
        w, h = random.randint(1, 5), random.randint(1, 4)
        pixels = [random.choice([0, 1, 127, 128, 255, random.randint(0, 255)]) for _ in range(w * h)]
        threshold = random.choice([0, 1, 100, 128, 255])
        args = [threshold, w, h, *pixels]
        assert exe("buggy_s", *args) == (0, predict_buggy(pixels, threshold, True), ""), args
        assert exe("buggy_u", *args) == (0, predict_buggy(pixels, threshold, False), ""), args
        assert exe("fixed_s", *args) == exe("fixed_u", *args) == (0, predict_correct(pixels, threshold), ""), args
    assert exe("fixed_s", 128, 3, 2, *SAMPLE)[1] == C_WANT

    # 3. every fix matters on its own: undo one fix at a time in the fixed source
    no_uint8 = sub(sub(sub(fixed_drv, "const uint8_t *pix", "const char *pix"),
                       "uint8_t lo = pix[0];", "char lo = pix[0];"), "uint8_t hi = pix[0];", "char hi = pix[0];")
    no_mul_first = sub(fixed_drv, "bright * 100 / n", "bright / n * 100")
    no_wide_sum = sub(fixed_drv, "uint64_t sum = 0;", "int sum = 0;")
    no_guard = sub(fixed_drv, "    if (n == 0) return s;\n", "")
    gcc("no_uint8", no_uint8, "-fsigned-char")
    assert exe("no_uint8", 128, 3, 2, *SAMPLE)[1] == BUGGY_SIGNED != C_WANT
    gcc("no_mul_first", no_mul_first)
    assert exe("no_mul_first", 128, 3, 2, *SAMPLE)[1] == BUGGY_UNSIGNED != C_WANT

    # 4. undefined behaviour, as reported by the sanitizers
    san = ("-g", "-O1", "-fsanitize=address,undefined")
    (WORK / "probe.c").write_text("int main(void) { return 0; }\n")
    if run(["gcc", *san, "-o", "probe", "probe.c"])[0] != 0 or exe("probe")[0] != 0:
        print("Part 1 (C): sanitizer checks SKIPPED, -fsanitize=address,undefined does not work here")
    else:
        gcc("san_buggy", buggy_drv, "-fsigned-char", *san)
        gcc("san_fixed", fixed_drv, "-fsigned-char", *san)
        gcc("san_no_wide_sum", no_wide_sum, *san)
        gcc("san_no_guard", no_guard, *san)
        big = (128, 7680, 4320)                                                  # threshold, width, height of an 8K frame
        rc, out, err = exe("san_buggy", *big, "fill", 127, 0)
        assert "signed integer overflow" in err and "cannot be represented in type 'int'" in err, err
        print("  buggy, 8K frame of 127s  ->", out.strip(), "|", err.strip().splitlines()[0].split(": ", 1)[1])
        if reference:
            assert out == "mean=-2 bright=0% contrast=0\n"
        assert exe("san_buggy", *big, "fill", 255, 0) == (0, "mean=-1 bright=0% contrast=0\n", "")   # every 255 is read as -1
        rc, out, err = exe("san_buggy", 128, 0, 0, "fill", 0, 1)                 # empty frame (one spare byte, so pix[0] is readable)
        assert rc != 0 and "division by zero" in err, err
        assert exe("san_fixed", *big, "fill", 255, 0) == (0, "mean=255 bright=100% contrast=0\n", "")
        assert exe("san_fixed", *big, "fill", 127, 0) == (0, "mean=127 bright=0% contrast=0\n", "")
        assert exe("san_fixed", 128, 0, 0, "fill", 0, 0) == (0, "mean=0 bright=0% contrast=0\n", "")
        assert exe("san_fixed", 128, 3, 2, *SAMPLE) == (0, C_WANT, "")
        rc, out, err = exe("san_no_wide_sum", *big, "fill", 255, 0)              # all fixes except the 64-bit sum
        assert "signed integer overflow" in err, err
        assert exe("san_no_wide_sum", 128, 3840, 2160, "fill", 255, 0) == (0, "mean=255 bright=100% contrast=0\n", "")
        rc, out, err = exe("san_no_guard", 128, 0, 0, "fill", 0, 1)              # all fixes except the empty-frame guard
        assert "division by zero" in err, err

    # 5. the same division by zero in plain builds: observed behaviour, not something the standard promises
    seen = {}
    for opt in ("-O0", "-O2"):
        gcc("empty", with_main(C_BUGGY, C_EMPTY_MAIN), opt)
        rc, out, _ = exe("empty")
        seen[opt] = (rc, out)
        print(f"  buggy, empty frame, plain {opt} build -> returncode {rc}, stdout {out!r}")
    if reference:
        assert seen["-O0"] == (-8, "")                                           # killed by SIGFPE
        assert seen["-O2"] == (0, "mean=0 bright=0% contrast=0\n")

    # 6. integer division in C
    (WORK / "div.c").write_text('#include <stdio.h>\nint main(void) { int a = -7, b = 2; printf("%d %d\\n", a / b, a % b); return 0; }\n')
    assert run(["gcc", "-std=c11", "-o", "div", "div.c"])[0] == 0 and exe("div")[1] == "-3 -1\n"
    print("Part 1 (C): ok")

if shutil.which("gcc"):
    check_c()
else:
    print("Part 1 (C): SKIPPED, gcc not found")

# ------------------------------------------------------------------ Part 2: JavaScript
JS_BUGGY = r'''// thumbs.js -- fetch thumbnails for a list of frame ids, retrying failures.
const LATENCY_MS = { 1: 30, 2: 10, 3: 20 };       // fake network latency per frame id

function fetchThumb(id) {                         // stand-in for the real HTTP call
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (id === 3) reject(new Error("503 for frame " + id));
      else resolve("thumb-" + id);
    }, LATENCY_MS[id]);
  });
}

// Returns the thumbnails for `ids`. opts.retries = extra attempts per frame (default 2).
async function fetchAll(ids, opts) {
  const retries = opts.retries || 2;
  const thumbs = [];
  ids.forEach(async (id) => {
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        thumbs.push(await fetchThumb(id));
        return;
      } catch (err) {
        console.log(`frame ${id}: attempt ${attempt} failed`);
      }
    }
  });
  console.log(`fetched ${thumbs.length} of ${ids.length}`);
  return thumbs;
}

fetchAll([1, 2, 3], { retries: 0 }).then((thumbs) => {
  console.log("done:", thumbs);
  setTimeout(() => console.log("300 ms later:", thumbs), 300);
});
'''

JS_FIXED = r'''// thumbs.js -- fetch thumbnails for a list of frame ids, retrying failures (fixed).
const LATENCY_MS = { 1: 30, 2: 10, 3: 20 };       // fake network latency per frame id

function fetchThumb(id) {                         // stand-in for the real HTTP call
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (id === 3) reject(new Error("503 for frame " + id));
      else resolve("thumb-" + id);
    }, LATENCY_MS[id]);
  });
}

async function fetchWithRetry(id, retries) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fetchThumb(id);
    } catch (err) {
      lastErr = err;
      console.log(`frame ${id}: attempt ${attempt} failed`);
    }
  }
  throw lastErr;
}

// Returns one entry per id, in the order of `ids`; null marks a frame that failed every attempt.
// opts.retries = extra attempts per frame (default 2; 0 means "try once").
async function fetchAll(ids, opts = {}) {
  const retries = opts.retries ?? 2;
  const settled = await Promise.allSettled(ids.map((id) => fetchWithRetry(id, retries)));
  const thumbs = settled.map((r) => (r.status === "fulfilled" ? r.value : null));
  console.log(`fetched ${thumbs.filter((t) => t !== null).length} of ${ids.length}`);
  return thumbs;
}

fetchAll([1, 2, 3], { retries: 0 }).then((thumbs) => {
  console.log("done:", thumbs);
  setTimeout(() => console.log("300 ms later:", thumbs), 300);
});
'''

JS_BUGGY_OUT = '''fetched 0 of 3
done: []
frame 3: attempt 0 failed
frame 3: attempt 1 failed
frame 3: attempt 2 failed
300 ms later: [ 'thumb-2', 'thumb-1' ]
'''

JS_FIXED_OUT = '''frame 3: attempt 0 failed
fetched 2 of 3
done: [ 'thumb-1', 'thumb-2', null ]
300 ms later: [ 'thumb-1', 'thumb-2', null ]
'''

JS_FACTS = r'''console.log(0 || 2, 0 ?? 2, "" || "x", null ?? 2, undefined ?? 2);
console.log(Boolean([]), Boolean({}), Boolean(""), Boolean(0), Boolean("0"), Boolean(NaN));
console.log(7 / 2, Math.trunc(-7 / 2), -7 % 2);
setTimeout(() => console.log("timeout"), 0);
Promise.resolve().then(() => console.log("microtask"));
console.log("sync");
'''

JS_ONE_BY_ONE = r'''const LATENCY_MS = { 1: 30, 2: 10, 3: 20 };
function fetchThumb(id) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (id === 3) reject(new Error("503 for frame " + id));
      else resolve("thumb-" + id);
    }, LATENCY_MS[id]);
  });
}
async function oneByOne(ids) {                       // sequential: the next request starts after the previous one ends
  const thumbs = [];
  for (const id of ids) {
    try { thumbs.push(await fetchThumb(id)); } catch (err) { thumbs.push(null); }
  }
  return thumbs;
}
async function allAtOnce(ids) {                      // concurrent: all requests start immediately
  const settled = await Promise.allSettled(ids.map((id) => fetchThumb(id)));
  return settled.map((r) => (r.status === "fulfilled" ? r.value : null));
}
(async () => {
  let t0 = Date.now();
  const a = await oneByOne([1, 2, 3]);
  const seqMs = Date.now() - t0;
  t0 = Date.now();
  const b = await allAtOnce([1, 2, 3]);
  const conMs = Date.now() - t0;
  console.log(JSON.stringify({ a, b, seqMs, conMs }));
})();
'''

def node(name, src):
    (WORK / f"{name}.js").write_text(src)
    return run(["node", f"{name}.js"])

def check_js():
    version = run(["node", "-p", "process.versions.node"])[1].strip()
    if int(version.split(".")[0]) < 16:
        print(f"Part 2 (JavaScript): SKIPPED, node {version} is older than 16")
        return
    assert node("buggy", JS_BUGGY) == (0, JS_BUGGY_OUT, "")
    assert node("fixed", JS_FIXED) == (0, JS_FIXED_OUT, "")

    call = "fetchAll([1, 2, 3], { retries: 0 })"
    fail3 = "".join(f"frame 3: attempt {k} failed\n" for k in range(3))
    in_order = "[ 'thumb-1', 'thumb-2', null ]"
    by_finish = "[ 'thumb-2', 'thumb-1' ]"

    # the timeline in the text: the same code with a timestamp added to each event
    timed = sub(JS_BUGGY, "const LATENCY_MS", "const T0 = Date.now();\nconst LATENCY_MS")
    timed = sub(timed, "console.log(`frame ${id}: attempt ${attempt} failed`);",
                "console.log(`frame ${id}: attempt ${attempt} failed @`, Date.now() - T0);")
    timed = sub(timed, "thumbs.push(await fetchThumb(id));",
                "thumbs.push(await fetchThumb(id)); console.log(`pushed ${id} @`, Date.now() - T0);")
    rc, out, err = node("timed", timed)
    events = [line.split(" @ ") for line in out.splitlines() if " @ " in line]
    assert rc == 0 and [e for e, _ in events] == [
        "pushed 2", "frame 3: attempt 0 failed", "pushed 1", "frame 3: attempt 1 failed", "frame 3: attempt 2 failed"], out
    times = [int(t) for _, t in events]
    # timers never fire early; how late they fire depends on machine load, so only the lower side is checked
    assert all(t >= want - 2 for t, want in zip(times, [10, 20, 30, 40, 60])), times
    assert times == sorted(times)

    # the statement's code called without opts: the TypeError becomes an unhandled rejection
    rc, out, err = node("buggy_no_opts", sub(JS_BUGGY, call, "fetchAll([1, 2, 3])"))
    assert rc == 1 and out == "" and "TypeError" in err and "retries" in err, (rc, out, err)

    # fix only "nobody waits": complete, but still in completion order, and retries: 0 still retries twice
    waited = sub(sub(JS_BUGGY, "  ids.forEach(async (id) => {", "  await Promise.all(ids.map(async (id) => {"),
                 "    }\n  });\n  console.log(`fetched", "    }\n  }));\n  console.log(`fetched")
    assert node("waited", waited) == (0, fail3 + f"fetched 2 of 3\ndone: {by_finish}\n300 ms later: {by_finish}\n", "")

    # fix everything except the waiting
    not_waited = sub(sub(JS_BUGGY, "opts.retries || 2", "opts.retries ?? 2"),
                     "async function fetchAll(ids, opts) {", "async function fetchAll(ids, opts = {}) {")
    assert node("not_waited", not_waited) == (
        0, f"fetched 0 of 3\ndone: []\nframe 3: attempt 0 failed\n300 ms later: {by_finish}\n", "")

    # fix everything except the ordering / the placeholder for failures
    not_ordered = sub(sub(waited, "opts.retries || 2", "opts.retries ?? 2"),
                      "async function fetchAll(ids, opts) {", "async function fetchAll(ids, opts = {}) {")
    assert node("not_ordered", not_ordered) == (
        0, f"frame 3: attempt 0 failed\nfetched 2 of 3\ndone: {by_finish}\n300 ms later: {by_finish}\n", "")

    # fixed version with one fix undone at a time
    with_or = sub(JS_FIXED, "opts.retries ?? 2", "opts.retries || 2")
    assert node("with_or", with_or) == (0, fail3 + f"fetched 2 of 3\ndone: {in_order}\n300 ms later: {in_order}\n", "")
    no_default = sub(sub(JS_FIXED, "async function fetchAll(ids, opts = {}) {", "async function fetchAll(ids, opts) {"),
                     call, "fetchAll([1, 2, 3])")
    rc, out, err = node("no_default", no_default)
    assert rc == 1 and "TypeError" in err, (rc, out, err)
    # the fixed version without opts uses the default of 2 extra attempts
    assert node("fixed_no_opts", sub(JS_FIXED, call, "fetchAll([1, 2, 3])")) == (
        0, fail3 + f"fetched 2 of 3\ndone: {in_order}\n300 ms later: {in_order}\n", "")

    # facts quoted in the text
    assert node("facts", JS_FACTS) == (0, "2 0 x 2 2\ntrue true false false true false\n3.5 -3 -1\nsync\nmicrotask\ntimeout\n", "")
    rc, out, err = node("one_by_one", JS_ONE_BY_ONE)
    timing = __import__("json").loads(out)
    assert timing["a"] == timing["b"] == ["thumb-1", "thumb-2", None]
    # lower bounds only: one by one cannot finish before 30+10+20 ms, all at once not before 30 ms
    assert timing["seqMs"] >= 55 and timing["conMs"] >= 25, timing
    print(f"Part 2 (JavaScript): ok on node {version}; one by one {timing['seqMs']} ms, all at once {timing['conMs']} ms")

if shutil.which("node"):
    check_js()
else:
    print("Part 2 (JavaScript): SKIPPED, node not found")

# ------------------------------------------------------------------ Part 3: SQL (SQLite, in-process)
SQL_SCHEMA = r'''-- Experiment tracking. One row per training run.
CREATE TABLE runs (
  id        INTEGER PRIMARY KEY,
  model     TEXT    NOT NULL,
  owner     TEXT,               -- NULL for runs launched by the nightly scheduler
  status    TEXT    NOT NULL,   -- 'done' | 'failed' | 'running'
  val_loss  REAL,               -- NULL until the first evaluation has finished
  gpu_hours INTEGER NOT NULL
);
CREATE TABLE retired_models (model TEXT);   -- models the weekly report should skip

INSERT INTO runs (id, model, owner, status, val_loss, gpu_hours) VALUES
  (1, 'vit-b',  'alice',    'done',    0.42, 10),
  (2, 'vit-b',  'bob',      'done',    0.38, 12),
  (3, 'vit-b',  'bob',      'failed',  NULL,  3),
  (4, 'vit-b',  NULL,       'failed',  NULL,  2),
  (5, 'vit-l',  'alice',    'done',    0.31, 40),
  (6, 'vit-l',  'test-bot', 'done',    0.90,  1),
  (7, 'vit-l',  NULL,       'running', NULL,  9),
  (8, 'vit-h',  'carol',    'running', NULL, 20),
  (9, 'resnet', 'bob',      'done',    0.55,  5);
INSERT INTO retired_models (model) VALUES ('resnet'), (NULL);
'''

SQL_BUGGY = r'''-- Weekly report: one row per model, best (lowest) average loss first.
-- Skip retired models and runs launched by test-bot.
SELECT model,
       COUNT(*)                                                        AS runs,
       SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) / COUNT(*) * 100 AS done_pct,
       ROUND(AVG(val_loss), 3)                                         AS avg_loss,
       SUM(gpu_hours) / COUNT(*)                                       AS gpu_h_per_run
FROM runs
WHERE owner != 'test-bot'
  AND model NOT IN (SELECT model FROM retired_models)
GROUP BY model
ORDER BY avg_loss;
'''

SQL_FIXED = r'''-- Weekly report: one row per model, best (lowest) average loss first; models with no loss yet go last.
-- Skip retired models and runs launched by test-bot (scheduler runs, owner IS NULL, are kept).
SELECT r.model,
       COUNT(*)                                                                        AS runs,
       ROUND(100.0 * SUM(CASE WHEN r.status = 'done' THEN 1 ELSE 0 END) / COUNT(*), 1) AS done_pct,
       COUNT(r.val_loss)                                                               AS evaluated,
       ROUND(AVG(r.val_loss), 3)                                                       AS avg_loss,
       ROUND(1.0 * SUM(r.gpu_hours) / COUNT(*), 2)                                     AS gpu_h_per_run
FROM runs AS r
WHERE (r.owner IS NULL OR r.owner != 'test-bot')
  AND NOT EXISTS (SELECT 1 FROM retired_models AS m WHERE m.model = r.model)
GROUP BY r.model
ORDER BY avg_loss IS NULL, avg_loss;
'''

def query(sql, only_resnet_retired=False):
    db = sqlite3.connect(":memory:")
    db.executescript(SQL_SCHEMA)
    if only_resnet_retired:
        db.execute("DELETE FROM retired_models WHERE model IS NULL")
        assert db.execute("SELECT model FROM retired_models").fetchall() == [("resnet",)]
    rows = db.execute(sql).fetchall()
    db.close()
    return rows

def value(expr):
    return query("SELECT " + expr)[0][0]

def expected_report():
    """The intended report, recomputed in Python from the raw rows (no SQL aggregation involved)."""
    runs = query("SELECT id, model, owner, status, val_loss, gpu_hours FROM runs")
    report = []
    for model in sorted({r[1] for r in runs} - {"resnet"}):
        mine = [r for r in runs if r[1] == model and r[2] != "test-bot"]       # in Python, None != "test-bot" is True
        losses = [r[4] for r in mine if r[4] is not None]
        report.append((model, len(mine), round(100 * sum(r[3] == "done" for r in mine) / len(mine), 1), len(losses),
                       round(sum(losses) / len(losses), 3) if losses else None,
                       round(sum(r[5] for r in mine) / len(mine), 2)))
    return sorted(report, key=lambda row: (row[4] is None, row[4] or 0.0))

FIXED_ROWS = [("vit-l", 2, 50.0, 1, 0.31, 24.5), ("vit-b", 4, 50.0, 2, 0.4, 6.75), ("vit-h", 1, 0.0, 0, None, 20.0)]

assert query(SQL_BUGGY) == []
assert query(SQL_BUGGY, only_resnet_retired=True) == [("vit-h", 1, 0, None, 20), ("vit-l", 1, 100, 0.31, 40), ("vit-b", 3, 0, 0.4, 8)]
assert query(SQL_FIXED) == query(SQL_FIXED, only_resnet_retired=True) == FIXED_ROWS == expected_report()

# every fix matters on its own: undo one fix at a time in the fixed query
undo = {
    "not exists": ("NOT EXISTS (SELECT 1 FROM retired_models AS m WHERE m.model = r.model)",
                   "r.model NOT IN (SELECT model FROM retired_models)"),
    "owner is null": ("(r.owner IS NULL OR r.owner != 'test-bot')", "r.owner != 'test-bot'"),
    "done_pct": ("ROUND(100.0 * SUM(CASE WHEN r.status = 'done' THEN 1 ELSE 0 END) / COUNT(*), 1)",
                 "SUM(CASE WHEN r.status = 'done' THEN 1 ELSE 0 END) / COUNT(*) * 100"),
    "gpu hours": ("ROUND(1.0 * SUM(r.gpu_hours) / COUNT(*), 2)", "SUM(r.gpu_hours) / COUNT(*)"),
    "null order": ("ORDER BY avg_loss IS NULL, avg_loss", "ORDER BY avg_loss"),
}
partly = {name: query(sub(SQL_FIXED, old, new)) for name, (old, new) in undo.items()}
assert all(rows != FIXED_ROWS for rows in partly.values())
assert partly["not exists"] == []
assert [row[:2] for row in partly["owner is null"]] == [("vit-l", 1), ("vit-b", 3), ("vit-h", 1)]
assert [row[2] for row in partly["done_pct"]] == [0, 0, 0]
assert [row[5] for row in partly["gpu hours"]] == [24, 6, 20]
assert [row[0] for row in partly["null order"]] == ["vit-h", "vit-l", "vit-b"]

# facts quoted in the text
assert value("NULL = NULL") is None and value("NULL != 'test-bot'") is None
assert value("'vit-b' NOT IN ('resnet', NULL)") is None and value("'resnet' NOT IN ('resnet', NULL)") == 0
assert value("2 / 3") == 0 and value("2 / 3 * 100") == 0 and value("25 / 3") == 8 and value("-7 / 2") == -3
assert query("SELECT COUNT(*), COUNT(val_loss), AVG(val_loss) FROM runs WHERE model = 'vit-b'") == [(4, 2, 0.4)]
assert query("SELECT COUNT(val_loss), AVG(val_loss) FROM runs WHERE model = 'vit-h'") == [(0, None)]
assert [r[0] for r in query("SELECT id FROM runs WHERE owner != 'test-bot'")] == [1, 2, 3, 5, 8, 9]   # 4 and 7 are gone
assert query("SELECT val_loss FROM runs ORDER BY val_loss LIMIT 1") == [(None,)]                      # NULLs first in ASC
print(f"Part 3 (SQL): ok on SQLite {sqlite3.sqlite_version}")
shutil.rmtree(WORK, ignore_errors=True)
```

</details>

</details>
