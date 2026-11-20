# Code reading in several languages (B): C++, Python, Bash

## Problem

In this round you don't write code; you only read it. The interviewer pastes three snippets into a shared doc one after another, each in a different language, with about ten minutes per snippet.
The code can't be run; you talk through it as you read. All three snippets come from the same fictional video-model training project and are independent of each other.

Answer the same five questions for each snippet:

1. What is this code trying to do?
2. For the given input, what does it actually output? (Be specific, down to every line and every number.)
3. Where are the bugs, and where are the risks?
4. How would you fix it?
5. If the author were here, what would you ask them?

Once you've answered the five questions, the interviewer will usually follow up on your answers with a few more questions, such as what happens with a different input or on a different platform.
These follow-ups and their answers are in the reference solution; answer the five questions on your own before looking.

Environment: C++ is compiled with g++ (C++17, Linux); Python is 3.x; Bash is 4.4 or newer, with GNU `ls`, `rm` and so on.

### Part 1 — C++: per-frame update of a multi-object tracker

```cpp
// tracker.cpp -- bookkeeping for a multi-object tracker, called once per video frame.
#include <algorithm>
#include <iostream>
#include <vector>

struct Track {
    int id;
    int misses;   // consecutive frames in which no detection matched this track
};

// 1. Reset `misses` for every matched track, increment it for the others.
// 2. Drop tracks that have been missed more than `max_misses` frames in a row.
// 3. Open one new track per unmatched detection.
void update(std::vector<Track>& tracks, const std::vector<int>& matched_ids,
            int unmatched_detections, int max_misses) {
    static int next_id = 100;

    for (auto t : tracks) {
        bool hit = std::find(matched_ids.begin(), matched_ids.end(), t.id) != matched_ids.end();
        t.misses = hit ? 0 : t.misses + 1;
    }
    for (auto it = tracks.begin(); it != tracks.end(); ++it) {
        if (it->misses > max_misses) {
            tracks.erase(it);
        }
    }
    for (int i = 0; i < unmatched_detections; i++) {
        tracks.push_back(Track{next_id++, 0});
    }
}

int main() {
    std::vector<Track> tracks = {{1, 0}, {2, 2}, {3, 2}, {4, 0}};
    update(tracks, {1, 4}, 1, 2);   // detections matched tracks 1 and 4; one detection matched nothing
    for (const Track& t : tracks) std::cout << t.id << ":" << t.misses << " ";
    std::cout << "\n";
    return 0;
}
```

### Part 2 — Python: bucketing clips by length and padding them

```py
# bucketing.py -- group variable-length clips into length buckets and pad each bucket.

def make_buckets(clips, edges, stats={}):
    """clips: list of clips, each a list of frame ids.
    edges: ascending bucket sizes, e.g. [4, 8, 16].
    A clip goes into the first bucket whose edge >= its length and is padded with -1 up to
    that edge. Clips longer than the last edge are dropped and counted in stats["dropped"].
    Returns (buckets, stats); buckets[i] holds the padded clips for edges[i]."""
    for clip in clips:
        if len(clip) > edges[-1]:
            clips.remove(clip)
            stats["dropped"] = stats.get("dropped", 0) + 1

    padders = []
    for size in edges:
        padders.append(lambda clip: clip + [-1] * (size - len(clip)))

    buckets = [[]] * len(edges)
    for clip in clips:
        for i, edge in enumerate(edges):
            if len(clip) <= edge:
                buckets[i].append(clip)
                break

    padded = [[pad(clip) for clip in bucket] for pad, bucket in zip(padders, buckets)]
    return padded, stats


clips = [[7] * 3, [1] * 20, [2] * 30, [5] * 6, [9] * 9]
buckets, stats = make_buckets(clips, [4, 8, 16])
print([[len(c) for c in b] for b in buckets], stats, len(clips))

_, stats2 = make_buckets([[3] * 2, [4] * 99], [4, 8, 16])
print(stats2, stats)
```

### Part 3 — Bash: pruning old checkpoints

```bash
#!/bin/bash
# prune_ckpts.sh -- keep the newest KEEP checkpoints of one run, delete the older ones.
# usage: CKPT_ROOT=/data/ckpts prune_ckpts.sh <run name> [KEEP]
# Set DRY_RUN=1 to only print what would be deleted.

RUN=$1
KEEP=${2:-3}
ROOT=$CKPT_ROOT/$RUN

cd $ROOT
count=$(ls | grep ckpt | wc -l)
echo "found $count checkpoints in $ROOT"

if [ $count -gt $KEEP ]; then
    # newest first, skip the first KEEP, delete the rest
    for f in $(ls -t | grep ckpt | tail -n +$((KEEP + 1))); do
        if [ $DRY_RUN ]; then
            echo "would delete $f"
        else
            echo "deleting $f"
            rm -rf $f
        fi
    done
fi

echo "$(date +%F) kept $KEEP" >> $ROOT/prune.log
echo "done"
```

The directory on disk looks like this (the modification times of the files are on the right):

```text
/data/ckpts/
├── exp1/
│   ├── ckpt_0100.pt         11-01 10:00
│   ├── ckpt_0200.pt         11-01 11:00
│   ├── ckpt_0300.pt         11-01 12:00
│   ├── ckpt_0300 copy.pt    11-01 12:30      the file name contains a space
│   ├── ckpt_0400.pt         11-01 13:00
│   ├── ckpt_0500.pt         11-01 14:00
│   └── train.log            11-01 14:05
└── exp 2/                                    the directory name contains a space
    └── ckpt_0100.pt         11-02 09:00
```

The three commands below are independent of each other; each starts from the directory state above. Apart from what is written in the command, `DRY_RUN` is not set in the environment.
The script itself lives at `/home/me/prune_ckpts.sh` and is executable.
For each command, say what it prints (including errors), what its exit code is, and which files are left on disk.

```text
Command 1 (current directory is /home/me):
    CKPT_ROOT=/data/ckpts ./prune_ckpts.sh exp1 2

Command 2 (current directory is /data/ckpts/exp1):
    CKPT_ROOT=/data/ckpts /home/me/prune_ckpts.sh "exp 2" 1

Command 3 (current directory is /home/me):
    DRY_RUN=0 CKPT_ROOT=/data/ckpts ./prune_ckpts.sh exp1 2
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

Each of the three snippets has a kind of semantics that “changes when you switch languages”: in C++ it is copying by value and iterator invalidation;
in Python it is aliasing, because “names are just references”, and late binding in closures; in Bash it is unquoted variables being split into several words, and the script carrying on after a command fails.
When reading, first state the intent in one sentence, then work through the sample input line by line, and only then discuss risks and fixes.
Every “observed output” below was actually produced on this machine: g++ 11.4.0 (x86-64 Linux, libstdc++), Python 3.11.18, GNU bash 5.1.16.

### Part 1

**What it's trying to do.** Called once per frame: matched tracks reset `misses` to zero, unmatched ones increment it by 1;
tracks that have been missing for more than `max_misses` consecutive frames are deleted; each unmatched detection box starts a new track.

**Actual output.** By intent, the `misses` of tracks 2 and 3 go from 2 to 3, exceeding the limit of 2, so they should be deleted; the correct output is `1:0 4:0 100:0`. The observed output is:

```text
1:0 2:2 3:2 4:0 100:0
```

Apart from the new track 100, nothing changed. The cause is `for (auto t : tracks)`: `auto t` copies by value, so `t` is a copy of the element,
`t.misses = ...` modifies the copy, and the copy is destroyed at the end of each iteration.
Since `misses` never changed, no track exceeds the limit, and the `erase` in the second loop never runs.
So the program as given has no undefined behavior (UB); it just doesn't update anything. `-Wall -Wextra` gives no warnings at all.

**Interviewer follow-up: if you change the first loop to `for (auto& t : tracks)` and leave everything else alone, what is the output? Is the program correct now?**
No. The first loop is now right, the `misses` of tracks 2 and 3 become 3, the second loop starts actually deleting things, and the second bug shows up.
`tracks.erase(it)` invalidates `it` and every iterator after it (iterator invalidation). The following `++it` uses an iterator that has already been invalidated, which is UB.
Observed behavior (identical with a plain build, `-O2`, and ASan plus UBSan):

```text
1:0 3:3 4:0 100:0
```

Track 3 was not deleted. Here is what happens: after track 2 is deleted, the following elements shift forward by one, so track 3 moves into the position `it` points to; then `++it` immediately skips past it.
No crash, no sanitizer report; the result is just silently wrong. This is what was observed this time; the standard guarantees nothing.
With libstdc++'s debug mode (compile with `-D_GLIBCXX_DEBUG`), the program aborts and reports `attempt to increment a singular iterator`.

If the deleted element happens to be the last one, it gets worse. Change the input to `{{1, 0}, {2, 2}}` and match only track 1:
after track 2 is deleted, `end()` moves back by one, `++it` steps past `end()`, `it != tracks.end()` stays true forever, and the loop reads on past the end of the buffer.
Observed: ASan reports `heap-buffer-overflow`; without sanitizers the program crashes (segmentation fault).

**Interviewer follow-up: one process handles two video streams, each with its own tracker, each calling `update`. What ids do the new tracks get?**
`static int next_id` is a static variable inside the function, so there is only one for the whole process. The two trackers share the same counter.
Observed: two empty trackers each call `update` once and each start one new track; the ids they get are 100 and 101, rather than each starting from 100.

**List of problems.**

1. **`for (auto t : tracks)` copies by value.** Fix: `Track& t`.
2. **Continuing to use the old iterator after `erase` inside the loop.** Fix: use the erase-remove idiom (first `std::remove_if` moves the elements to keep to the front, then a single `erase` removes the tail).
   You can also write `it = tracks.erase(it)` and only `++it` when nothing was deleted; in C++20 you can simply write `std::erase_if(tracks, ...)`. All three give the same result in the checks below.
3. **The `static` counter inside the function.** It is shared by multiple trackers; it can't be reset between tests; and when two threads call `update` at the same time, `next_id++` is a data race.
   Fix: have the tracker own the counter, and have the caller pass it in.
4. **Efficiency.** Each track does a linear search over `matched_ids`, so T tracks and M matches cost O(T·M).
   With a few dozen tracks it doesn't matter; with thousands, put `matched_ids` into a `std::unordered_set` first.
   Also, `erase`-ing one element at a time inside the loop shifts all the following elements every time, O(T²) in the worst case; erase-remove shifts them only once, O(T).

**The fixed version.**

```cpp
// tracker.cpp -- bookkeeping for a multi-object tracker, called once per video frame (fixed).
#include <algorithm>
#include <iostream>
#include <vector>

struct Track {
    int id;
    int misses;   // consecutive frames in which no detection matched this track
};

// 1. Reset `misses` for every matched track, increment it for the others.
// 2. Drop tracks that have been missed more than `max_misses` frames in a row.
// 3. Open one new track per unmatched detection, numbering them from `next_id`.
void update(std::vector<Track>& tracks, const std::vector<int>& matched_ids,
            int unmatched_detections, int max_misses, int& next_id) {
    for (Track& t : tracks) {                      // reference: modify the element, not a copy
        bool hit = std::find(matched_ids.begin(), matched_ids.end(), t.id) != matched_ids.end();
        t.misses = hit ? 0 : t.misses + 1;
    }
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [max_misses](const Track& t) { return t.misses > max_misses; }),
                 tracks.end());                    // erase-remove: no iterator outlives the erase
    for (int i = 0; i < unmatched_detections; i++) {
        tracks.push_back(Track{next_id++, 0});
    }
}

int main() {
    std::vector<Track> tracks = {{1, 0}, {2, 2}, {3, 2}, {4, 0}};
    int next_id = 100;              // owned by the tracker instance, not a function-level static
    update(tracks, {1, 4}, 1, 2, next_id);
    for (const Track& t : tracks) std::cout << t.id << ":" << t.misses << " ";
    std::cout << "\n";
    return 0;
}
```

Observed output (identical with a plain build, ASan plus UBSan, and `-D_GLIBCXX_DEBUG`):

```text
1:0 4:0 100:0
```

**Questions for the author.**

- Is there a test covering the case where “a track has been missing for longer than the limit and gets deleted”? (The code in the problem never deletes any track, which means this path has never been tested.)
- Will there ever be more than one tracker in a process? Will it be called from multiple threads?
- Should ids be globally unique, or should each video stream number its tracks from the start?
- How many tracks are there typically? That decides whether the linear search needs replacing.
- Does “more than `max_misses`” mean strictly greater? Both the comment and the code say greater than; confirm that this is intended.

### Part 2

**What it's trying to do.** Put clips of varying length into a few buckets by length: each clip goes into the first bucket that can hold it and is padded with −1 up to the bucket's length;
clips longer than the largest bucket are dropped and counted in `stats["dropped"]`.

**Actual output.** The 5 clips have lengths 3, 20, 30, 6, 9, and the buckets have lengths 4, 8, 16.
By intent, 20 and 30 are dropped, 3 goes into bucket 4, 6 into bucket 8, and 9 into bucket 16. The correct output should be:

```text
[[4], [8], [16]] {'dropped': 2} 5
{'dropped': 1} {'dropped': 2}
```

The observed output is:

```text
[[16, 16, 16], [16, 16, 16], [16, 16, 16]] {'dropped': 1} 4
{'dropped': 2} {'dropped': 2}
```

There are four independent bugs in here.

1. **Deleting while iterating.** `for clip in clips` advances an index internally. When it reaches index 1 (length 20), that clip is deleted, and the following elements shift forward by one;
   the index then moves on to 2, which now holds the clip of length 6. The clip of length 30 is skipped and never checked.
   So only 1 `dropped` is counted, and the clip of length 30 stays in the list. At the bucketing step, no bucket can hold it, and the loop finishes without a `break`, so it simply vanishes: it neither goes into a bucket nor gets counted.
   On top of that, the `clips` passed in by the caller has been modified, going from 5 clips to 4.
2. **`[[]] * len(edges)`.** This gives three references to the same list, not three lists. Putting a clip into `buckets[0]` adds one to all three buckets. That is why each bucket has 3 clips.
3. **Late binding in closures.** The three `lambda`s remember the variable `size` itself, not its value at creation time.
   By the time `pad` is actually called, the loop has long finished, and `size` is stuck at its last value, 16. So every clip is padded to 16.
4. **Mutable default argument.** The dict `stats={}` is created only once, when `def` runs, and every later call that doesn't pass `stats` uses that same dict.
   The second call's count continues from 1 up to 2; the `stats` returned by the first call is the same object, so it becomes 2 as well. That is where the second line of output comes from.

**Interviewer follow-up: after the first call, does the caller's `clips` still hold the original 5 clips?**
No. Only 4 are left, with lengths 3, 30, 6, 9. See item 1 above for why.

**Interviewer follow-up: if you change `edges` in both calls to a single bucket, `[16]`, what are the two lines of output? Which bugs become invisible?** The observed output is:

```text
[[16, 16, 16]] {'dropped': 1} 4
{'dropped': 2} {'dropped': 2}
```

The correct output is:

```text
[[16, 16, 16]] {'dropped': 2} 5
{'dropped': 1} {'dropped': 2}
```

The bucket contents happen to be right. With only one bucket, `[[]] * 1` has no second name to share the list with, and the last value of `size` is also its only value.
Bugs 2 and 3 both become invisible; only the counts and the caller's list are wrong. So if the tests only ever use one bucket, these two bugs will never be caught.

**The fixed version.**

```python
def make_buckets(clips, edges, stats=None):
    """clips: list of clips, each a list of frame ids.
    edges: ascending bucket sizes, e.g. [4, 8, 16].
    A clip goes into the first bucket whose edge >= its length and is padded with -1 up to
    that edge. Clips longer than the last edge are dropped and counted in stats["dropped"].
    Returns (buckets, stats); buckets[i] holds the padded clips for edges[i].
    `clips` itself is not modified."""
    if stats is None:
        stats = {}                                    # a fresh dict per call
    kept = [clip for clip in clips if len(clip) <= edges[-1]]      # build a new list, do not remove in place
    stats["dropped"] = stats.get("dropped", 0) + len(clips) - len(kept)

    buckets = [[] for _ in edges]                     # one distinct list per bucket
    for clip in kept:
        for i, edge in enumerate(edges):
            if len(clip) <= edge:
                buckets[i].append(clip + [-1] * (edge - len(clip)))   # pad here: no closures needed
                break
    return buckets, stats


clips = [[7] * 3, [1] * 20, [2] * 30, [5] * 6, [9] * 9]
buckets, stats = make_buckets(clips, [4, 8, 16])
print([[len(c) for c in b] for b in buckets], stats, len(clips))

_, stats2 = make_buckets([[3] * 2, [4] * 99], [4, 8, 16])
print(stats2, stats)
```

When this code runs on this page, it prints exactly the two lines of “correct output” above.

**Questions for the author.**

- Is this function allowed to modify the `clips` passed in by the caller? (The original implementation does, and the docs don't say so.)
- Should clips that are too long be dropped, truncated, or split into pieces? For video data, always dropping the longest samples skews the data distribution.
- Is the `stats` parameter meant to let the caller accumulate across multiple calls? If so, the caller should be required to pass it in explicitly.
- Is `edges` guaranteed to be ascending and non-empty?
- Could padding with −1 collide with real frame ids?

### Part 3

**What it's trying to do.** Keep the newest KEEP checkpoints of a run, delete the older ones, and write a line to `prune.log`. If `DRY_RUN` is set, only print; don't delete.

**Command 1.** Observed output (exit code 0):

```text
found 6 checkpoints in /data/ckpts/exp1
deleting ckpt_0300
deleting copy.pt
deleting ckpt_0300.pt
deleting ckpt_0200.pt
deleting ckpt_0100.pt
done
```

What's left in `exp1` is `ckpt_0300 copy.pt`, `ckpt_0400.pt`, `ckpt_0500.pt`, `train.log`, plus the newly written `prune.log`. It was asked to keep 2 and actually kept 3.
The cause: the result of `$(ls -t | ...)` is unquoted, so the shell splits it into separate words on whitespace (word splitting).
`ckpt_0300 copy.pt` is split into two names, `ckpt_0300` and `copy.pt`. Neither file exists, and the `-f` in `rm -rf` keeps it silent about missing files.
The script printed “deleting” but didn't actually delete anything. If the directory happened to contain a file called `copy.pt`, that is the file that would be deleted (verified).

**Command 2.** Observed output (exit code 0; the two lines starting with `/home/me/prune_ckpts.sh:` are printed by bash to standard error):

```text
/home/me/prune_ckpts.sh: line 10: cd: too many arguments
found 6 checkpoints in /data/ckpts/exp 2
deleting ckpt_0400.pt
deleting ckpt_0300
deleting copy.pt
deleting ckpt_0300.pt
deleting ckpt_0200.pt
deleting ckpt_0100.pt
/home/me/prune_ckpts.sh: line 26: $ROOT/prune.log: ambiguous redirect
done
```

This is the most serious bug. `cd $ROOT` expands to `cd /data/ckpts/exp 2`; `cd` receives two arguments and fails.
The script doesn't check whether `cd` succeeded and carries on working in the current directory, which is `exp1`:
`exp1`'s checkpoints are deleted down to just the newest one, `ckpt_0500.pt` (plus the file with the space in its name), while not a single file in `exp 2` is touched.
Yet what it prints is `in /data/ckpts/exp 2`. The final logging line fails for the same lack of quoting, the script still prints `done`, and the exit code is 0.
Whoever called it, or the cron job running it, sees nothing wrong.

**Command 3.** Observed output (exit code 0):

```text
found 6 checkpoints in /data/ckpts/exp1
would delete ckpt_0300
would delete copy.pt
would delete ckpt_0300.pt
would delete ckpt_0200.pt
would delete ckpt_0100.pt
done
```

Not a single file was deleted. `[ $DRY_RUN ]` tests “is this string non-empty”. `0` is a non-empty string, so the condition is true.
There is no such thing as “0 is false” in the shell. Whoever wrote `DRY_RUN=0` meant “turn off the dry run” and got a dry run instead.
Also, the dry run still wrote a `kept 2` line to `prune.log`.

**Interviewer follow-up: what happens if the run name is mistyped (say `exp3`, a directory that doesn't exist), or left out entirely?** Both were actually run.

- Mistyped run name: `cd` reports `No such file or directory`, and after that it's the same as command 2: it deletes in the current directory.
  Observed: running `CKPT_ROOT=/data/ckpts /home/me/prune_ckpts.sh exp3 1` inside `exp1` deletes `exp1` down to just `ckpt_0300 copy.pt`, `ckpt_0500.pt` and `train.log`. The exit code is still 0.
- Run name left out: `RUN` is empty, `ROOT` becomes `/data/ckpts/`, `cd` succeeds, and the script starts pruning the parent directory of all runs.
  `grep ckpt` matches anything with `ckpt` in its name, directories included.
  Observed: put four directories, `ckpt_archive_a` through `ckpt_archive_d` (a is the oldest), under `/data/ckpts` and run the script with no arguments.
  KEEP takes its default of 3, the script prints `found 4 checkpoints in /data/ckpts/`, and the oldest, `ckpt_archive_a`, is deleted wholesale by `rm -rf`, along with the files inside it. Exit code 0.
  If `CKPT_ROOT` isn't set either, `ROOT` is `/` (running that assignment line on its own does give `/`). The whole script was never run in this situation; the consequence is inferred: it would look in the root directory for things with `ckpt` in their names and delete them.

**List of problems.**

1. **`cd $ROOT` is unquoted and its result is not checked.** Fix: add quotes; put `set -euo pipefail` at the top so the script stops immediately when any command fails.
2. **Word splitting in `for f in $(ls ...)` and `rm -rf $f`.** Fix: read the file names into an array, and write `"$f"` when using them.
3. **`grep ckpt` is too broad, and `rm -rf` is too aggressive.** Fix: match only `ckpt_*.pt`; delete files with `rm -f --`, without `-r`. If a required argument is missing, report an error and exit right away.
4. **`[ $DRY_RUN ]` tests for non-empty.** Fix: compare explicitly with `1`.
5. **The redirect for the log line is unquoted.** The observed error is `ambiguous redirect`.
6. **The exit code is always 0.** When `KEEP` is passed as `abc`, `[` reports `integer expression expected`, the script deletes nothing, still prints `done`, and returns 0.
7. **“Newest” is determined by modification time.** Once files have been copied, synced or restored from backup, the modification time no longer reflects training order. Ask the author about this one.

**The fixed version.**

```bash
#!/bin/bash
# prune_ckpts.sh -- keep the newest KEEP checkpoints of one run, delete the older ones (fixed).
# usage: CKPT_ROOT=/data/ckpts prune_ckpts.sh <run name> [KEEP]
# Set DRY_RUN=1 to only print what would be deleted (anything else, or unset, means delete).
set -euo pipefail                       # stop on a failed command or an unset variable

run=${1:?usage: prune_ckpts.sh <run name> [KEEP]}
keep=${2:-3}
dry_run=${DRY_RUN:-0}
root="${CKPT_ROOT:?CKPT_ROOT is not set}/$run"

if ! [[ $keep =~ ^[0-9]+$ ]]; then
    echo "KEEP must be a non-negative integer, got '$keep'" >&2
    exit 2
fi

cd -- "$root"                           # quoted; under set -e a failed cd ends the script here

# Only files named ckpt_*.pt, newest first by mtime, one name per line (names may contain spaces).
mapfile -t ckpts < <(ls -td -- ckpt_*.pt 2>/dev/null || true)
echo "found ${#ckpts[@]} checkpoints in $root"

for f in "${ckpts[@]:keep}"; do         # everything after the first $keep entries
    if [[ $dry_run == 1 ]]; then
        echo "would delete $f"
    else
        echo "deleting $f"
        rm -f -- "$f"                   # plain files only: no -r
    fi
done

if [[ $dry_run != 1 ]]; then
    echo "$(date +%F) kept $keep" >> "$root/prune.log"
fi
echo "done"
```

Observed output. Command 1 (what's left in `exp1` is `ckpt_0400.pt`, `ckpt_0500.pt`, `train.log` and `prune.log`):

```text
found 6 checkpoints in /data/ckpts/exp1
deleting ckpt_0300 copy.pt
deleting ckpt_0300.pt
deleting ckpt_0200.pt
deleting ckpt_0100.pt
done
```

Command 2 (`exp1` is untouched, and `exp 2` gains a `prune.log`):

```text
found 1 checkpoints in /data/ckpts/exp 2
done
```

Command 3's output is the same as command 1's, and the files really are deleted; with `DRY_RUN=1` it prints four `would delete ...` lines, deletes no files, and writes no log.
With a mistyped run name, the script stops at the `cd` line with exit code 1; with the run name left out, the exit code is 1; with `KEEP` passed as `abc`, the exit code is 2. In none of these three cases are any files deleted.

The fixed version still parses the output of `ls`, so it still breaks on file names containing newlines. To fix that for good, sort by the step number in the file name, or switch to Python.

**Questions for the author.**

- Is a checkpoint a single file or a directory? Was `rm -rf` used originally because there really are directory-style checkpoints, or was it just written out of habit?
- Should “newest” be by modification time, or by the step number in the file name?
- Are there files that must always be kept, such as `ckpt_best.pt`?
- Who runs this script, and from which directory? If it runs as a cron job, who finds out when it fails?
- Can run names contain spaces or other special characters?
- Before deleting, should it confirm that no training process is currently writing to this directory?

### Follow-ups

**1. “Copy or reference”: what is the default in each of the three languages?**
C++ copies by default: `auto t`, passing by value and assignment all make a copy; to modify the original object you have to write `&`.
Python never copies automatically: variables and parameters are just names for objects, which is why you get things like three names pointing at the same list in `[[]] * 3`, or a function modifying the caller's list; if you want a copy, you have to make it yourself.
In Bash everything is a string, and assignment copies the string; the trap isn't copying but whether a variable gets re-split into words when it's used.
When you read code and see “something was modified”, first ask: was it the original or a copy?

**2. How does deleting while iterating behave in each of the three languages, and what is the safe way to do it in each?**
A Python list doesn't raise an error; it just silently skips elements: deleting the 2s from `[1, 2, 2, 3, 2]` while iterating gives `[1, 3, 2]`. A dict, on the other hand, raises `RuntimeError` outright. The safe way is to build a new list.
For a C++ `vector` it's undefined behavior: it may skip elements or run out of bounds. The safe way is erase-remove, or continuing with the return value of `erase`.
In Bash, first collect the names to delete into an array, then delete them all together.

**3. Does a closure remember the variable or the value at the time?**
Python remembers the variable: three `lambda: i` created in a loop return `[2, 2, 2]` when called; only by writing `lambda i=i: i`, which stores the current value in a default argument, do you get `[0, 1, 2]`.
JavaScript depends on how the variable is declared: the three closures in `for (var i ...)` share one `i`, giving `[ 3, 3, 3 ]`; with `let`, each iteration has its own `i`, giving `[ 0, 1, 2 ]`.
A C++ lambda has to spell it out in the square brackets: `[=]` makes a copy, `[&]` captures a reference. If it captures a reference and the variable has already been destroyed, that's a dangling reference.

**4. Why didn't ASan catch the C++ iterator bug? Are there similar traps?**
ASan checks “does the memory this access touches belong to you”. After `erase`, that memory still belongs to the `vector`, so accessing it is legal; what's wrong is only that, logically, that position now holds a different element.
So no sanitizer report doesn't mean no problem; bugs like this need libstdc++'s debug mode, or tests with known answers.
A similar trap is `push_back`: when capacity runs out, the `vector` moves to a bigger block of memory, and every reference, pointer and iterator obtained earlier is invalidated.
Observed: take a reference to `tracks.front()`, then `push_back`, then set `misses` to 0 through that reference. ASan reports `heap-use-after-free`; with a plain build the program finishes normally, the printed `misses` is still the original 5, and the write is lost.

**5. Should this Bash script just be rewritten in Python?**
It depends on whether it will keep growing. At a dozen or so lines, deleting only file names with a fixed format, it can stay in Bash, provided it does three things: `set -euo pipefail`, quote every variable, and don't parse the output of `ls` (the fixed version above only does the first two; see the note there for why).
Once it needs to sort by step number, keep the best checkpoint, total up sizes, or have tests, switch to Python: file names are ordinary strings that don't get split; numbers are numbers; and errors raise exceptions instead of silently carrying on.
Also, static analysis tools like ShellCheck can point out unquoted variables directly. It isn't installed on this machine, so it wasn't run here.

<details>
<summary>Checks (runnable)</summary>

The code below puts every piece of source code from the problem and the solution verbatim into strings, writes them to a temporary directory, and then runs them with `g++`, `bash` and Python.
If `g++` or `bash` is not available on the machine, the corresponding part prints a one-line note and is skipped.
The buggy Bash script deletes files in the “current directory”, so it is always run inside the temporary directory, and `CKPT_ROOT` always points to the temporary directory too.
For undefined behavior, assertions are made only on the reports from the sanitizers and libstdc++'s debug mode; behavior under a plain build is only printed, and is checked only on g++ 11 on x86-64.

```python
import contextlib
import io
import itertools
import os
import platform
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="code_reading_b_"))
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

# ------------------------------------------------------------------ Part 1: C++
CPP_BUGGY = r'''// tracker.cpp -- bookkeeping for a multi-object tracker, called once per video frame.
#include <algorithm>
#include <iostream>
#include <vector>

struct Track {
    int id;
    int misses;   // consecutive frames in which no detection matched this track
};

// 1. Reset `misses` for every matched track, increment it for the others.
// 2. Drop tracks that have been missed more than `max_misses` frames in a row.
// 3. Open one new track per unmatched detection.
void update(std::vector<Track>& tracks, const std::vector<int>& matched_ids,
            int unmatched_detections, int max_misses) {
    static int next_id = 100;

    for (auto t : tracks) {
        bool hit = std::find(matched_ids.begin(), matched_ids.end(), t.id) != matched_ids.end();
        t.misses = hit ? 0 : t.misses + 1;
    }
    for (auto it = tracks.begin(); it != tracks.end(); ++it) {
        if (it->misses > max_misses) {
            tracks.erase(it);
        }
    }
    for (int i = 0; i < unmatched_detections; i++) {
        tracks.push_back(Track{next_id++, 0});
    }
}

int main() {
    std::vector<Track> tracks = {{1, 0}, {2, 2}, {3, 2}, {4, 0}};
    update(tracks, {1, 4}, 1, 2);   // detections matched tracks 1 and 4; one detection matched nothing
    for (const Track& t : tracks) std::cout << t.id << ":" << t.misses << " ";
    std::cout << "\n";
    return 0;
}
'''

CPP_FIXED = r'''// tracker.cpp -- bookkeeping for a multi-object tracker, called once per video frame (fixed).
#include <algorithm>
#include <iostream>
#include <vector>

struct Track {
    int id;
    int misses;   // consecutive frames in which no detection matched this track
};

// 1. Reset `misses` for every matched track, increment it for the others.
// 2. Drop tracks that have been missed more than `max_misses` frames in a row.
// 3. Open one new track per unmatched detection, numbering them from `next_id`.
void update(std::vector<Track>& tracks, const std::vector<int>& matched_ids,
            int unmatched_detections, int max_misses, int& next_id) {
    for (Track& t : tracks) {                      // reference: modify the element, not a copy
        bool hit = std::find(matched_ids.begin(), matched_ids.end(), t.id) != matched_ids.end();
        t.misses = hit ? 0 : t.misses + 1;
    }
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [max_misses](const Track& t) { return t.misses > max_misses; }),
                 tracks.end());                    // erase-remove: no iterator outlives the erase
    for (int i = 0; i < unmatched_detections; i++) {
        tracks.push_back(Track{next_id++, 0});
    }
}

int main() {
    std::vector<Track> tracks = {{1, 0}, {2, 2}, {3, 2}, {4, 0}};
    int next_id = 100;              // owned by the tracker instance, not a function-level static
    update(tracks, {1, 4}, 1, 2, next_id);
    for (const Track& t : tracks) std::cout << t.id << ":" << t.misses << " ";
    std::cout << "\n";
    return 0;
}
'''

CPP_DRIVER_MAIN = r'''#include <cstdlib>

// test driver:  prog MAX_MISSES UNMATCHED N  id misses (N pairs)  matched_id...
int main(int argc, char** argv) {
    int max_misses = std::atoi(argv[1]), unmatched = std::atoi(argv[2]), n = std::atoi(argv[3]);
    std::vector<Track> tracks;
    for (int i = 0; i < n; i++) tracks.push_back(Track{std::atoi(argv[4 + 2 * i]), std::atoi(argv[5 + 2 * i])});
    std::vector<int> matched;
    for (int i = 4 + 2 * n; i < argc; i++) matched.push_back(std::atoi(argv[i]));
    int next_id = 100;
    update(tracks, matched, unmatched, max_misses, next_id);
    for (const Track& t : tracks) std::cout << t.id << ":" << t.misses << " ";
    std::cout << "\n";
    return 0;
}
'''

CPP_TWO_TRACKERS_STATIC = r'''int main() {
    std::vector<Track> cam_a, cam_b;            // two independent trackers, e.g. two video streams
    update(cam_a, {}, 1, 2);
    update(cam_b, {}, 1, 2);
    std::cout << cam_a[0].id << " " << cam_b[0].id << "\n";
    return 0;
}
'''

CPP_TWO_TRACKERS_OWNED = r'''int main() {
    std::vector<Track> cam_a, cam_b;            // two independent trackers, each with its own counter
    int next_a = 100, next_b = 100;
    update(cam_a, {}, 1, 2, next_a);
    update(cam_b, {}, 1, 2, next_b);
    std::cout << cam_a[0].id << " " << cam_b[0].id << "\n";
    return 0;
}
'''

CPP_TAIL_MAIN = r'''int main() {
    std::vector<Track> tracks = {{1, 0}, {2, 2}};
    update(tracks, {1}, 0, 2);                  // track 2 is missed a third time, and it is the last element
    for (const Track& t : tracks) std::cout << t.id << ":" << t.misses << " ";
    std::cout << "\n";
    return 0;
}
'''

CPP_DANGLING = r'''#include <iostream>
#include <vector>

struct Track { int id; int misses; };

int main() {
    std::vector<Track> tracks = {{1, 5}, {2, 0}};     // size 2, capacity 2
    Track& oldest = tracks.front();                   // a reference into the vector's buffer
    tracks.push_back(Track{3, 0});                    // no room left: the buffer is reallocated
    oldest.misses = 0;                                // writes into the old, freed buffer
    std::cout << tracks.front().misses << "\n";
    return 0;
}
'''

BUGGY_OUT = "1:0 2:2 3:2 4:0 100:0 \n"
WANT_OUT = "1:0 4:0 100:0 \n"
SKIPPED_OUT = "1:0 3:3 4:0 100:0 \n"          # what the auto&-only version was seen to print (UB)

OLD_LOOP = """    for (auto it = tracks.begin(); it != tracks.end(); ++it) {
        if (it->misses > max_misses) {
            tracks.erase(it);
        }
    }
"""
ERASE_REMOVE = """    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                                [max_misses](const Track& t) { return t.misses > max_misses; }),
                 tracks.end());
"""
ERASE_REMOVE_IN_FIXED = ERASE_REMOVE.rstrip("\n") + "                    // erase-remove: no iterator outlives the erase\n"
ERASE_IN_LOOP = """    for (auto it = tracks.begin(); it != tracks.end(); ) {
        if (it->misses > max_misses) it = tracks.erase(it);   // erase returns the iterator after the removed element
        else ++it;
    }
"""
ERASE_IF = "    std::erase_if(tracks, [max_misses](const Track& t) { return t.misses > max_misses; });\n"

def fix_ref(src):
    return sub(src, "for (auto t : tracks) {", "for (auto& t : tracks) {")

def fix_erase(src):
    return sub(src, OLD_LOOP, ERASE_REMOVE)

def fix_static(src):
    src = sub(src, "int unmatched_detections, int max_misses) {\n    static int next_id = 100;\n\n",
              "int unmatched_detections, int max_misses, int& next_id) {\n")
    return sub(src, "    update(tracks, {1, 4}, 1, 2);", "    int next_id = 100;\n    update(tracks, {1, 4}, 1, 2, next_id);")

def with_main(src, main):
    head, _ = src.split("\nint main() {")              # exactly one main, or this unpacking fails
    return head + "\n" + main

def gxx(name, src, *flags, std="c++17"):
    (WORK / f"{name}.cpp").write_text(src)
    rc, _, err = run(["g++", f"-std={std}", "-Wall", "-Wextra", *flags, "-o", name, f"{name}.cpp"])
    assert rc == 0, err
    return err                                         # the compiler's warnings; "" when there are none

def exe(name, *args):
    return run([WORK / name, *args])

def predict_update(tracks, matched, unmatched, max_misses, next_id=100):
    """The intended behaviour of update(), written without looking at the C++ (tracks: list of (id, misses))."""
    out = []
    for track_id, misses in tracks:
        misses = 0 if track_id in matched else misses + 1
        if misses <= max_misses:
            out.append((track_id, misses))
    out += [(next_id + k, 0) for k in range(unmatched)]
    return "".join(f"{i}:{m} " for i, m in out) + "\n"

assert predict_update([(1, 0), (2, 2), (3, 2), (4, 0)], {1, 4}, 1, 2) == WANT_OUT

def check_cpp():
    version_text = run(["g++", "--version"])[1]
    version = run(["g++", "-dumpfullversion"])[1].strip()
    gnu = "Free Software Foundation" in version_text                       # real GCC, hence libstdc++
    reference = gnu and version.startswith("11.") and platform.machine() == "x86_64"

    # 1. the two programs exactly as printed on the page (the buggy one never erases, so it has no UB)
    warnings = gxx("buggy", CPP_BUGGY)
    assert exe("buggy") == (0, BUGGY_OUT, "")
    if reference:
        assert warnings == ""
    assert gxx("fixed", CPP_FIXED) == "" or not reference
    assert exe("fixed") == (0, WANT_OUT, "")
    if gnu:
        gxx("fixed_dbg", CPP_FIXED, "-D_GLIBCXX_DEBUG")
        assert exe("fixed_dbg") == (0, WANT_OUT, "")

    # 2. only the copy fixed (the follow-up question): the erase loop now runs and is UB
    auto_ref = fix_ref(CPP_BUGGY)
    gxx("auto_ref", auto_ref)
    rc, out, _ = exe("auto_ref")
    print(f"Part 1 (C++): g++ {version}; auto&-only version, plain build -> returncode {rc}, stdout {out!r}")
    if reference:
        assert (rc, out) == (0, SKIPPED_OUT)
    gxx("auto_ref_o2", auto_ref, "-O2")
    rc, out, _ = exe("auto_ref_o2")
    print(f"  auto&-only version, -O2 build -> returncode {rc}, stdout {out!r}")
    if reference:
        assert (rc, out) == (0, SKIPPED_OUT)
    if gnu:
        gxx("auto_ref_dbg", auto_ref, "-D_GLIBCXX_DEBUG")
        rc, out, err = exe("auto_ref_dbg")
        assert rc != 0 and "attempt to increment a singular iterator" in err, err

    # 3. every fix matters on its own: apply two of the three fixes to the statement's code
    assert exe_after(gxx, "no_ref_fix", fix_static(fix_erase(CPP_BUGGY))) == (0, BUGGY_OUT, "")
    if gnu:
        gxx("no_erase_fix", fix_static(fix_ref(CPP_BUGGY)), "-D_GLIBCXX_DEBUG")
        rc, out, err = exe("no_erase_fix")
        assert rc != 0 and "attempt to increment a singular iterator" in err, err
    no_static_fix = fix_erase(fix_ref(CPP_BUGGY))
    assert exe_after(gxx, "no_static_fix", no_static_fix) == (0, WANT_OUT, "")        # the sample does not show this bug
    assert exe_after(gxx, "two_static", with_main(no_static_fix, CPP_TWO_TRACKERS_STATIC)) == (0, "100 101\n", "")
    assert exe_after(gxx, "two_owned", with_main(CPP_FIXED, CPP_TWO_TRACKERS_OWNED)) == (0, "100 100\n", "")
    assert exe_after(gxx, "all_three", fix_static(fix_erase(fix_ref(CPP_BUGGY)))) == (0, WANT_OUT, "")

    # 4. sanitizer runs
    san = ("-g", "-fsanitize=address,undefined")
    (WORK / "probe.cpp").write_text("int main() { return 0; }\n")
    if run(["g++", *san, "-o", "probe", "probe.cpp"])[0] != 0 or exe("probe")[0] != 0:
        print("Part 1 (C++): sanitizer checks SKIPPED, -fsanitize=address,undefined does not work here")
        return
    assert exe_after(gxx, "fixed_san", CPP_FIXED, *san) == (0, WANT_OUT, "")
    gxx("auto_ref_san", auto_ref, *san)
    rc, out, err = exe("auto_ref_san")
    print(f"  auto&-only version, ASan+UBSan build -> returncode {rc}, stdout {out!r}, {len(err)} bytes on stderr")
    if reference:
        assert (rc, out, err) == (0, SKIPPED_OUT, "")                                 # UB that no sanitizer notices
    gxx("tail_san", with_main(auto_ref, CPP_TAIL_MAIN), *san)                         # the stale track is the last element
    rc, out, err = exe("tail_san")
    assert rc != 0 and "heap-buffer-overflow" in err, err
    gxx("tail_plain", with_main(auto_ref, CPP_TAIL_MAIN))
    rc, out, _ = exe("tail_plain")
    print(f"  auto&-only version, stale track last, plain build -> returncode {rc}, stdout {out!r}")
    if reference:
        assert rc == -11                                                              # SIGSEGV
    assert exe_after(gxx, "tail_fixed", with_main(fix_erase(auto_ref), CPP_TAIL_MAIN), *san) == (0, "1:0 \n", "")

    # 5. three correct ways to erase, on random inputs, under ASan+UBSan (+ debug mode on libstdc++)
    debug = ("-D_GLIBCXX_DEBUG",) if gnu else ()
    driver = with_main(CPP_FIXED, CPP_DRIVER_MAIN)
    gxx("erase_remove", driver, *san, *debug)
    gxx("erase_in_loop", sub(driver, ERASE_REMOVE_IN_FIXED, ERASE_IN_LOOP), *san, *debug)
    names = ["erase_remove", "erase_in_loop"]
    probe20 = "#include <vector>\nint main() { std::vector<int> v{1}; return (int)std::erase_if(v, [](int x) { return x > 5; }); }\n"
    (WORK / "probe20.cpp").write_text(probe20)
    if run(["g++", "-std=c++20", "-o", "probe20", "probe20.cpp"])[0] == 0:
        gxx("erase_if", sub(driver, ERASE_REMOVE_IN_FIXED, ERASE_IF), *san, *debug, std="c++20")
        names.append("erase_if")
    else:
        print("  std::erase_if check SKIPPED, this g++ has no C++20 library support")
    cases = [([(1, 0), (2, 2), (3, 2), (4, 0)], {1, 4}, 1, 2), ([(1, 0), (2, 2)], {1}, 0, 2), ([], set(), 0, 0), ([], set(), 3, 1),
             ([(5, 9), (6, 9), (7, 9)], set(), 0, 2)]
    for _ in range(40):
        n = random.randint(0, 7)
        tracks = [(i + 1, random.randint(0, 4)) for i in range(n)]
        matched = {i + 1 for i in range(n) if random.random() < 0.4}
        cases.append((tracks, matched, random.randint(0, 2), random.randint(0, 3)))
    for tracks, matched, unmatched, max_misses in cases:
        args = [max_misses, unmatched, len(tracks), *[x for t in tracks for x in t], *sorted(matched)]
        want = predict_update(tracks, matched, unmatched, max_misses)
        for name in names:
            assert exe(name, *args) == (0, want, ""), (name, args, exe(name, *args))

    # 6. a reference into a vector does not survive a reallocating push_back
    gxx("dangling_san", CPP_DANGLING, *san)
    rc, out, err = exe("dangling_san")
    assert rc != 0 and "heap-use-after-free" in err, err
    gxx("dangling", CPP_DANGLING)
    rc, out, _ = exe("dangling")
    print(f"  dangling reference, plain build -> returncode {rc}, stdout {out!r}")
    if reference:
        assert (rc, out) == (0, "5\n")
    print(f"Part 1 (C++): ok ({', '.join(names)} agree on {len(cases)} inputs)")

def exe_after(compiler, name, src, *flags):
    compiler(name, src, *flags)
    return exe(name)

if shutil.which("g++"):
    check_cpp()
else:
    print("Part 1 (C++): SKIPPED, g++ not found")

# ------------------------------------------------------------------ Part 2: Python
PY_BUGGY = r'''# bucketing.py -- group variable-length clips into length buckets and pad each bucket.

def make_buckets(clips, edges, stats={}):
    """clips: list of clips, each a list of frame ids.
    edges: ascending bucket sizes, e.g. [4, 8, 16].
    A clip goes into the first bucket whose edge >= its length and is padded with -1 up to
    that edge. Clips longer than the last edge are dropped and counted in stats["dropped"].
    Returns (buckets, stats); buckets[i] holds the padded clips for edges[i]."""
    for clip in clips:
        if len(clip) > edges[-1]:
            clips.remove(clip)
            stats["dropped"] = stats.get("dropped", 0) + 1

    padders = []
    for size in edges:
        padders.append(lambda clip: clip + [-1] * (size - len(clip)))

    buckets = [[]] * len(edges)
    for clip in clips:
        for i, edge in enumerate(edges):
            if len(clip) <= edge:
                buckets[i].append(clip)
                break

    padded = [[pad(clip) for clip in bucket] for pad, bucket in zip(padders, buckets)]
    return padded, stats


clips = [[7] * 3, [1] * 20, [2] * 30, [5] * 6, [9] * 9]
buckets, stats = make_buckets(clips, [4, 8, 16])
print([[len(c) for c in b] for b in buckets], stats, len(clips))

_, stats2 = make_buckets([[3] * 2, [4] * 99], [4, 8, 16])
print(stats2, stats)
'''

PY_FIXED = r'''def make_buckets(clips, edges, stats=None):
    """clips: list of clips, each a list of frame ids.
    edges: ascending bucket sizes, e.g. [4, 8, 16].
    A clip goes into the first bucket whose edge >= its length and is padded with -1 up to
    that edge. Clips longer than the last edge are dropped and counted in stats["dropped"].
    Returns (buckets, stats); buckets[i] holds the padded clips for edges[i].
    `clips` itself is not modified."""
    if stats is None:
        stats = {}                                    # a fresh dict per call
    kept = [clip for clip in clips if len(clip) <= edges[-1]]      # build a new list, do not remove in place
    stats["dropped"] = stats.get("dropped", 0) + len(clips) - len(kept)

    buckets = [[] for _ in edges]                     # one distinct list per bucket
    for clip in kept:
        for i, edge in enumerate(edges):
            if len(clip) <= edge:
                buckets[i].append(clip + [-1] * (edge - len(clip)))   # pad here: no closures needed
                break
    return buckets, stats


clips = [[7] * 3, [1] * 20, [2] * 30, [5] * 6, [9] * 9]
buckets, stats = make_buckets(clips, [4, 8, 16])
print([[len(c) for c in b] for b in buckets], stats, len(clips))

_, stats2 = make_buckets([[3] * 2, [4] * 99], [4, 8, 16])
print(stats2, stats)
'''

PY_BUGGY_OUT = '''[[16, 16, 16], [16, 16, 16], [16, 16, 16]] {'dropped': 1} 4
{'dropped': 2} {'dropped': 2}
'''

PY_WANT_OUT = '''[[4], [8], [16]] {'dropped': 2} 5
{'dropped': 1} {'dropped': 2}
'''

PY_BUGGY_ONE_EDGE = '''[[16, 16, 16]] {'dropped': 1} 4
{'dropped': 2} {'dropped': 2}
'''

PY_WANT_ONE_EDGE = '''[[16, 16, 16]] {'dropped': 2} 5
{'dropped': 1} {'dropped': 2}
'''

def run_source(src):
    """Execute a whole source text in a fresh namespace; returns (what it printed, the namespace)."""
    ns, buf = {}, io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "<snippet>", "exec"), ns)
    return buf.getvalue(), ns

buggy_out, buggy_ns = run_source(PY_BUGGY)
assert buggy_out == PY_BUGGY_OUT
assert [len(c) for c in buggy_ns["clips"]] == [3, 30, 6, 9]                      # the caller's list lost one clip
assert buggy_ns["stats"] is buggy_ns["stats2"]                                   # the shared default dict
fixed_out, fixed_ns = run_source(PY_FIXED)
assert fixed_out == PY_WANT_OUT
assert [len(c) for c in fixed_ns["clips"]] == [3, 20, 30, 6, 9]
assert fixed_ns["make_buckets"].__code__.co_code == make_buckets.__code__.co_code   # the block shown above is the same code
assert fixed_ns["buckets"] == [[[7, 7, 7, -1]], [[5] * 6 + [-1] * 2], [[9] * 9 + [-1] * 7]]

# the four fixes, as edits of the statement's code
def f_remove(src):
    return sub(src, '''    for clip in clips:
        if len(clip) > edges[-1]:
            clips.remove(clip)
            stats["dropped"] = stats.get("dropped", 0) + 1
''', '''    kept = [clip for clip in clips if len(clip) <= edges[-1]]
    stats["dropped"] = stats.get("dropped", 0) + len(clips) - len(kept)
    clips = kept
''')

def f_closure(src):
    return sub(src, "lambda clip: clip +", "lambda clip, size=size: clip +")

def f_alias(src):
    return sub(src, "buckets = [[]] * len(edges)", "buckets = [[] for _ in edges]")

def f_default(src):
    src = sub(src, "def make_buckets(clips, edges, stats={}):", "def make_buckets(clips, edges, stats=None):")
    return sub(src, 'buckets[i] holds the padded clips for edges[i]."""\n',
               'buckets[i] holds the padded clips for edges[i]."""\n    stats = {} if stats is None else stats\n')

FIXES = {"remove": f_remove, "closure": f_closure, "alias": f_alias, "default": f_default}

def apply(names):
    src = PY_BUGGY
    for name in names:
        src = FIXES[name](src)
    return src

all_four_out, all_four_ns = run_source(apply(FIXES))
assert all_four_out == PY_WANT_OUT
all_but_one = {left_out: run_source(apply([n for n in FIXES if n != left_out]))[0] for left_out in FIXES}
assert all(out != PY_WANT_OUT for out in all_but_one.values())
assert all_but_one["remove"] == "[[4], [8], [16]] {'dropped': 1} 4\n{'dropped': 1} {'dropped': 1}\n"
assert all_but_one["closure"] == "[[16], [16], [16]] {'dropped': 2} 5\n{'dropped': 1} {'dropped': 2}\n"
assert all_but_one["alias"] == "[[4, 6, 9], [8, 8, 9], [16, 16, 16]] {'dropped': 2} 5\n{'dropped': 1} {'dropped': 2}\n"
assert all_but_one["default"] == "[[4], [8], [16]] {'dropped': 2} 5\n{'dropped': 3} {'dropped': 3}\n"
# each bug alone is also enough to change the output of the otherwise untouched statement code
only_one = {name: run_source(apply([name]))[0] for name in FIXES}
assert len(set(only_one.values()) | {PY_BUGGY_OUT}) == 5

# with a single bucket, the aliasing and closure bugs cannot be seen; the other two still can
def one_edge(src):
    src = sub(src, "make_buckets(clips, [4, 8, 16])", "make_buckets(clips, [16])")
    return sub(src, "[4] * 99], [4, 8, 16])", "[4] * 99], [16])")

assert run_source(one_edge(PY_BUGGY))[0] == PY_BUGGY_ONE_EDGE != PY_WANT_ONE_EDGE
assert run_source(one_edge(PY_FIXED))[0] == PY_WANT_ONE_EDGE
assert run_source(one_edge(apply(["closure", "alias"])))[0] == PY_BUGGY_ONE_EDGE
assert run_source(one_edge(apply(["remove", "default"])))[0] == PY_WANT_ONE_EDGE

def reference_buckets(clips, edges):
    """Independent restatement of the docstring: one pass, smallest edge that fits."""
    by_edge, dropped = {edge: [] for edge in edges}, 0
    for clip in clips:
        fitting = [edge for edge in edges if edge >= len(clip)]
        if fitting:
            by_edge[min(fitting)].append(list(clip) + [-1] * (min(fitting) - len(clip)))
        else:
            dropped += 1
    return [by_edge[edge] for edge in edges], dropped

for _ in range(300):
    edges = sorted(random.sample(range(1, 20), random.randint(1, 4)))
    clips = [[random.randint(0, 99)] * random.randint(0, 24) for _ in range(random.randint(0, 8))]
    before = [list(c) for c in clips]
    want_buckets, want_dropped = reference_buckets(clips, edges)
    for fn in (make_buckets, all_four_ns["make_buckets"]):
        arg = [list(c) for c in clips]
        got_buckets, got_stats = fn(arg, edges)
        assert got_buckets == want_buckets and got_stats == {"dropped": want_dropped}, (clips, edges)
        assert arg == before                                                     # the caller's list is untouched
    carried = {"dropped": 10}
    assert make_buckets(clips, edges, carried)[1] is carried and carried == {"dropped": 10 + want_dropped}

# facts quoted in the follow-ups
assert [f() for f in [lambda: i for i in range(3)]] == [2, 2, 2]
assert [f() for f in [lambda i=i: i for i in range(3)]] == [0, 1, 2]
xs = [1, 2, 2, 3, 2]
for x in xs:
    if x == 2:
        xs.remove(x)
assert xs == [1, 3, 2]
d = {"a": 1, "b": 2}
try:
    for key in d:
        del d[key]
    raise AssertionError("expected RuntimeError")
except RuntimeError as e:
    assert "changed size during iteration" in str(e)
three = [[]] * 3
three[0].append(1)
assert three == [[1], [1], [1]] and three[0] is three[2]
print(f"Part 2 (Python): ok on Python {platform.python_version()}")

if shutil.which("node"):
    (WORK / "closure.js").write_text(
        "var fs = [];\nfor (var i = 0; i < 3; i++) fs.push(() => i);\nconsole.log(fs.map((f) => f()));\n"
        "var gs = [];\nfor (let j = 0; j < 3; j++) gs.push(() => j);\nconsole.log(gs.map((g) => g()));\n")
    assert run(["node", "closure.js"]) == (0, "[ 3, 3, 3 ]\n[ 0, 1, 2 ]\n", "")
else:
    print("follow-up 3 (JavaScript closures): SKIPPED, node not found")

# ------------------------------------------------------------------ Part 3: Bash
SH_BUGGY = r'''#!/bin/bash
# prune_ckpts.sh -- keep the newest KEEP checkpoints of one run, delete the older ones.
# usage: CKPT_ROOT=/data/ckpts prune_ckpts.sh <run name> [KEEP]
# Set DRY_RUN=1 to only print what would be deleted.

RUN=$1
KEEP=${2:-3}
ROOT=$CKPT_ROOT/$RUN

cd $ROOT
count=$(ls | grep ckpt | wc -l)
echo "found $count checkpoints in $ROOT"

if [ $count -gt $KEEP ]; then
    # newest first, skip the first KEEP, delete the rest
    for f in $(ls -t | grep ckpt | tail -n +$((KEEP + 1))); do
        if [ $DRY_RUN ]; then
            echo "would delete $f"
        else
            echo "deleting $f"
            rm -rf $f
        fi
    done
fi

echo "$(date +%F) kept $KEEP" >> $ROOT/prune.log
echo "done"
'''

SH_FIXED = r'''#!/bin/bash
# prune_ckpts.sh -- keep the newest KEEP checkpoints of one run, delete the older ones (fixed).
# usage: CKPT_ROOT=/data/ckpts prune_ckpts.sh <run name> [KEEP]
# Set DRY_RUN=1 to only print what would be deleted (anything else, or unset, means delete).
set -euo pipefail                       # stop on a failed command or an unset variable

run=${1:?usage: prune_ckpts.sh <run name> [KEEP]}
keep=${2:-3}
dry_run=${DRY_RUN:-0}
root="${CKPT_ROOT:?CKPT_ROOT is not set}/$run"

if ! [[ $keep =~ ^[0-9]+$ ]]; then
    echo "KEEP must be a non-negative integer, got '$keep'" >&2
    exit 2
fi

cd -- "$root"                           # quoted; under set -e a failed cd ends the script here

# Only files named ckpt_*.pt, newest first by mtime, one name per line (names may contain spaces).
mapfile -t ckpts < <(ls -td -- ckpt_*.pt 2>/dev/null || true)
echo "found ${#ckpts[@]} checkpoints in $root"

for f in "${ckpts[@]:keep}"; do         # everything after the first $keep entries
    if [[ $dry_run == 1 ]]; then
        echo "would delete $f"
    else
        echo "deleting $f"
        rm -f -- "$f"                   # plain files only: no -r
    fi
done

if [[ $dry_run != 1 ]]; then
    echo "$(date +%F) kept $keep" >> "$root/prune.log"
fi
echo "done"
'''

BUGGY_CMD1 = '''found 6 checkpoints in /data/ckpts/exp1
deleting ckpt_0300
deleting copy.pt
deleting ckpt_0300.pt
deleting ckpt_0200.pt
deleting ckpt_0100.pt
done
'''

BUGGY_CMD2 = r'''/home/me/prune_ckpts.sh: line 10: cd: too many arguments
found 6 checkpoints in /data/ckpts/exp 2
deleting ckpt_0400.pt
deleting ckpt_0300
deleting copy.pt
deleting ckpt_0300.pt
deleting ckpt_0200.pt
deleting ckpt_0100.pt
/home/me/prune_ckpts.sh: line 26: $ROOT/prune.log: ambiguous redirect
done
'''

BUGGY_CMD3 = '''found 6 checkpoints in /data/ckpts/exp1
would delete ckpt_0300
would delete copy.pt
would delete ckpt_0300.pt
would delete ckpt_0200.pt
would delete ckpt_0100.pt
done
'''

FIXED_CMD1 = '''found 6 checkpoints in /data/ckpts/exp1
deleting ckpt_0300 copy.pt
deleting ckpt_0300.pt
deleting ckpt_0200.pt
deleting ckpt_0100.pt
done
'''

FIXED_CMD2 = '''found 1 checkpoints in /data/ckpts/exp 2
done
'''

T0 = 1793491200                                        # 2026-11-01 00:00 UTC; only the order of the mtimes matters
TREE = [("exp1/ckpt_0100.pt", 600), ("exp1/ckpt_0200.pt", 660), ("exp1/ckpt_0300.pt", 720), ("exp1/ckpt_0300 copy.pt", 750),
        ("exp1/ckpt_0400.pt", 780), ("exp1/ckpt_0500.pt", 840), ("exp1/train.log", 845), ("exp 2/ckpt_0100.pt", 1980)]
ARCHIVES = [(f"ckpt_archive_{c}/old.pt", 10 * k) for k, c in enumerate("abcd", 1)]   # four directories directly under the root, a is the oldest
EXP1 = sorted(rel.split("/")[1] for rel, _ in TREE if rel.startswith("exp1/"))
CKPTS_NEWEST_FIRST = [rel.split("/")[1] for rel, minute in sorted(TREE, key=lambda t: -t[1]) if "exp1/ckpt" in rel]

def make_tree(extra=()):
    data = WORK / "data"
    shutil.rmtree(data, ignore_errors=True)
    for rel, minute in [*TREE, *extra]:
        path = data / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        os.utime(path, (T0 + 60 * minute, T0 + 60 * minute))
    for rel, minute in extra:                          # give the extra directories old, distinct mtimes as well
        os.utime((data / rel).parent, (T0 + 60 * minute, T0 + 60 * minute))
    (WORK / "home").mkdir(exist_ok=True)
    return data

def prune(script, args, cwd="home", extra=(), set_root=True, **env_vars):
    """Run one script on a fresh copy of the tree. Returns (returncode, transcript, {directory: sorted names})."""
    data = make_tree(extra)
    where = WORK / cwd
    assert where.is_dir() and WORK in where.parents    # the buggy script deletes in its current directory
    assert set_root or script == "fixed.sh"            # the buggy script is never run without CKPT_ROOT
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", **env_vars}
    if set_root:
        env["CKPT_ROOT"] = str(data)
    p = subprocess.run(["bash", str(WORK / script), *args], cwd=where, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    shown_as = "./prune_ckpts.sh" if cwd == "home" else "/home/me/prune_ckpts.sh"   # how the page invokes it from there
    text = p.stdout.replace(str(data), "/data/ckpts").replace(str(WORK / script), shown_as)
    left = {d.name: sorted(f.name for f in d.iterdir()) for d in data.iterdir() if d.is_dir()}
    return p.returncode, text, left

def stdout_only(transcript):
    return [line for line in transcript.splitlines() if not line.startswith(("./prune_ckpts.sh: line ", "/home/me/prune_ckpts.sh: line "))]

def check_bash():
    major, minor = map(int, run(["bash", "-c", "echo ${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"])[1].split())
    full = run(["bash", "-c", "echo $BASH_VERSION"])[1].strip()
    if (major, minor) < (4, 4) or " " in str(WORK):     # 4.4: an empty array under set -u is no longer an error
        print(f"Part 3 (Bash): SKIPPED (bash {full}; needs bash >= 4.4 and a temp dir without spaces)")
        return
    (WORK / "buggy.sh").write_text(SH_BUGGY)
    (WORK / "fixed.sh").write_text(SH_FIXED)
    exact = major >= 5                                 # the wording of bash's own error messages was checked on 5.1

    # ---- the statement's script
    rc, text, left = prune("buggy.sh", ["exp1", "2"])                                   # command 1
    assert (rc, text) == (0, BUGGY_CMD1)
    assert left["exp1"] == ["ckpt_0300 copy.pt", "ckpt_0400.pt", "ckpt_0500.pt", "prune.log", "train.log"]
    rc, text, left = prune("buggy.sh", ["exp1", "2"], extra=[("exp1/copy.pt", 100)])    # an unrelated file named copy.pt
    assert (rc, text) == (0, BUGGY_CMD1) and "copy.pt" not in left["exp1"] and "ckpt_0300 copy.pt" in left["exp1"]

    rc, text, left = prune("buggy.sh", ["exp 2", "1"], cwd="data/exp1")                 # command 2
    assert rc == 0 and stdout_only(text) == stdout_only(BUGGY_CMD2), text
    assert "cd:" in text and "prune.log" in text
    if exact:
        assert text == BUGGY_CMD2, text
    assert left["exp1"] == ["ckpt_0300 copy.pt", "ckpt_0500.pt", "train.log"]           # the wrong run was pruned
    assert left["exp 2"] == ["ckpt_0100.pt"]                                            # the requested run is untouched

    rc, text, left = prune("buggy.sh", ["exp1", "2"], DRY_RUN="0")                      # command 3
    assert (rc, text) == (0, BUGGY_CMD3)
    assert left["exp1"] == sorted(EXP1 + ["prune.log"])                                 # nothing deleted, log written anyway
    assert (WORK / "data" / "exp1" / "prune.log").read_text().endswith(" kept 2\n")
    rc, text, left = prune("buggy.sh", ["exp1", "2"], DRY_RUN="1")
    assert (rc, text) == (0, BUGGY_CMD3)

    rc, text, left = prune("buggy.sh", ["exp3", "1"], cwd="data/exp1")                  # misspelt run name
    assert rc == 0 and "No such file or directory" in text and text.endswith("done\n")
    assert left["exp1"] == ["ckpt_0300 copy.pt", "ckpt_0500.pt", "train.log"]

    rc, text, left = prune("buggy.sh", [], extra=ARCHIVES)                              # no arguments at all
    assert rc == 0 and text.startswith("found 4 checkpoints in /data/ckpts/\ndeleting ckpt_archive_a\n"), text
    assert sorted(left) == ["ckpt_archive_b", "ckpt_archive_c", "ckpt_archive_d", "exp 2", "exp1"]
    assert left["ckpt_archive_b"] == ["old.pt"] and left["exp1"] == EXP1
    rc, text, left = prune("buggy.sh", ["", "1"], extra=ARCHIVES)                       # an empty run name
    assert rc == 0 and "deleting ckpt_archive_c\ndeleting ckpt_archive_b\ndeleting ckpt_archive_a\n" in text
    assert sorted(left) == ["ckpt_archive_d", "exp 2", "exp1"]
    # with CKPT_ROOT unset as well, ROOT is "/": only the assignment is run here, never the script
    assert run(["bash", "-c", 'unset CKPT_ROOT; RUN=; ROOT=$CKPT_ROOT/$RUN; echo "$ROOT"']) == (0, "/\n", "")

    rc, text, left = prune("buggy.sh", ["exp1", "abc"])                                 # KEEP is not a number
    assert rc == 0 and "integer expression expected" in text and text.endswith("done\n")
    assert left["exp1"] == sorted(EXP1 + ["prune.log"])

    # ---- the fixed script
    rc, text, left = prune("fixed.sh", ["exp1", "2"])
    assert (rc, text) == (0, FIXED_CMD1)
    assert left["exp1"] == ["ckpt_0400.pt", "ckpt_0500.pt", "prune.log", "train.log"]

    rc, text, left = prune("fixed.sh", ["exp 2", "1"], cwd="data/exp1")
    assert (rc, text) == (0, FIXED_CMD2)
    assert left["exp1"] == EXP1 and left["exp 2"] == ["ckpt_0100.pt", "prune.log"]

    rc, text, left = prune("fixed.sh", ["exp1", "2"], DRY_RUN="0")
    assert (rc, text) == (0, FIXED_CMD1) and left["exp1"] == ["ckpt_0400.pt", "ckpt_0500.pt", "prune.log", "train.log"]
    rc, text, left = prune("fixed.sh", ["exp1", "2"], DRY_RUN="1")
    assert (rc, text) == (0, FIXED_CMD1.replace("deleting", "would delete")) and text.count("would delete") == 4
    assert left["exp1"] == EXP1                                                         # nothing deleted, no log

    rc, text, left = prune("fixed.sh", ["exp3", "1"], cwd="data/exp1")
    assert rc == 1 and "No such file or directory" in text and "done" not in text and left["exp1"] == EXP1
    for args in ([], ["", "1"]):
        rc, text, left = prune("fixed.sh", args, extra=ARCHIVES)
        assert rc == 1 and "usage" in text and "done" not in text
        assert sorted(left) == ["ckpt_archive_a", "ckpt_archive_b", "ckpt_archive_c", "ckpt_archive_d", "exp 2", "exp1"]
        assert left["ckpt_archive_a"] == ["old.pt"] and left["exp1"] == EXP1
    rc, text, left = prune("fixed.sh", ["exp1", "abc"])
    assert rc == 2 and "KEEP must be" in text and left["exp1"] == EXP1
    rc, text, left = prune("fixed.sh", ["exp1", "2"], set_root=False)
    assert rc == 1 and "CKPT_ROOT is not set" in text and left["exp1"] == EXP1
    rc, text, left = prune("fixed.sh", ["empty"], extra=[("empty/train.log", 5)])
    assert (rc, text) == (0, "found 0 checkpoints in /data/ckpts/empty\ndone\n") and left["empty"] == ["prune.log", "train.log"]

    # every KEEP from 0 to 7, compared with "sort by mtime in Python and keep the first KEEP"
    assert CKPTS_NEWEST_FIRST == ["ckpt_0500.pt", "ckpt_0400.pt", "ckpt_0300 copy.pt", "ckpt_0300.pt", "ckpt_0200.pt", "ckpt_0100.pt"]
    for keep in range(8):
        rc, text, left = prune("fixed.sh", ["exp1", str(keep)])
        assert rc == 0 and left["exp1"] == sorted(CKPTS_NEWEST_FIRST[:keep] + ["train.log", "prune.log"]), (keep, left)
        assert [l[len("deleting "):] for l in text.splitlines() if l.startswith("deleting ")] == CKPTS_NEWEST_FIRST[keep:]
    rc, text, left = prune("fixed.sh", ["exp1"])                                        # KEEP defaults to 3
    assert rc == 0 and left["exp1"] == sorted(CKPTS_NEWEST_FIRST[:3] + ["train.log", "prune.log"])
    print(f"Part 3 (Bash): ok on bash {full}")

if shutil.which("bash"):
    check_bash()
else:
    print("Part 3 (Bash): SKIPPED, bash not found")
shutil.rmtree(WORK, ignore_errors=True)
```

</details>

</details>
