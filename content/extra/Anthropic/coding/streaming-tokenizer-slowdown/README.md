# Streaming Tokenizer That Slows Down on Long Documents

## Problem

You have inherited `StreamTokenizer`, the streaming tokenization component of a document-ingestion service. When a user uploads a document, the network thread writes the received text in chunks (`write`); the downstream batching thread takes up to 256 tokens at a time and sends them as one batch for embedding (`read`). The tokenization rule is greedy longest match: at the current position, take the longest vocabulary entry that matches; if none matches, emit one character as `unk_id`. Text can be cut into chunks anywhere, including in the middle of a multi-character token, so the component has to wait whenever "later text could still form a longer token".

This code's output is correct and all existing tests pass. The problem is performance. Below are the code, a repro script and the ticket.

```py
class StreamTokenizer:
    """Greedy longest-match tokenizer for text that arrives in pieces.

    vocab maps token strings (all non-empty) to ids. At each position the
    longest key that matches is taken; if no key matches, one character
    becomes unk_id.

        write(chunk)      add text; a cut may fall anywhere, even inside a token
        close()           no more text will be written
        read(max_tokens)  up to max_tokens ids that can no longer change, in
                          order; fewer (maybe none) if the rest needs more text

    Reading until the stream is closed and empty gives the same ids as
    tokenizing the whole text in one piece, however it was cut into chunks.
    """

    def __init__(self, vocab, unk_id):
        if not vocab:
            raise ValueError("empty vocabulary")
        self.vocab = vocab
        self.unk_id = unk_id
        self.max_len = max(len(key) for key in vocab)
        self.buf = ""            # written but not yet tokenized
        self.closed = False
        self.chars_in = 0        # counters for the metrics endpoint
        self.tokens_out = 0

    def write(self, chunk):
        if self.closed:
            raise ValueError("write after close")
        self.buf += chunk
        self.chars_in += len(chunk)

    def close(self):
        self.closed = True

    def pending(self):
        """Characters written but not yet turned into tokens."""
        return len(self.buf)

    def _ready(self):
        """Is the token at the front of the buffer final?"""
        if not self.buf:
            return False
        # With fewer than max_len characters, a longer key could still be
        # completed by text that has not arrived yet.
        return self.closed or len(self.buf) >= self.max_len

    def _longest_match(self):
        """(token id, length in characters) of the token at the front of the buffer."""
        best = None
        for key in self.vocab:
            if self.buf.startswith(key):
                if best is None or len(key) > len(best):
                    best = key
        if best is None:
            return self.unk_id, 1
        return self.vocab[best], len(best)

    def read(self, max_tokens):
        out = []
        while len(out) < max_tokens and self._ready():
            token_id, n = self._longest_match()
            out.append(token_id)
            self.buf = self.buf[n:]
        self.tokens_out += len(out)
        return out
```

The repro script. In production, the network thread and the batching thread are two threads; the script interleaves them: write 4 KB, take one batch, write another 4 KB, and so on. When the upload ends, it closes the stream and drains it. The network is faster than downstream, just as in production.

```py
import random
import time

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def make_vocab(n_words=120, seed=7):
    """Every letter and the space, plus n_words random words of 2..8 letters. Ids start at 1."""
    rng = random.Random(seed)
    vocab = {ch: i + 1 for i, ch in enumerate(LETTERS + " ")}
    while len(vocab) < len(LETTERS) + 1 + n_words:
        word = "".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 8)))
        if word not in vocab:
            vocab[word] = len(vocab) + 1
    return vocab


def make_text(vocab, n_chars, seed=11):
    """Vocabulary words separated by spaces; about one word in ten is a number (not in the vocabulary)."""
    rng = random.Random(seed)
    words = [w for w in vocab if len(w) > 1]
    parts, size = [], 0
    while size < n_chars:
        w = str(rng.randint(0, 999)) if rng.random() < 0.1 else rng.choice(words)
        parts.append(w)
        size += len(w) + 1
    return " ".join(parts)[:n_chars]


def drive(tok, text, chunk=4096, batch=256):
    """As in production: the network writes a chunk, the batcher takes up to one batch,
    and so on; when the upload ends, close and drain."""
    ids = []
    for i in range(0, len(text), chunk):
        tok.write(text[i:i + chunk])
        ids.extend(tok.read(batch))
    tok.close()
    while True:
        got = tok.read(batch)
        if not got:
            return ids
        ids.extend(got)


def bench(sizes=(100_000, 800_000, 1_600_000, 3_200_000)):
    vocab = make_vocab()
    for n_chars in sizes:
        text = make_text(vocab, n_chars)
        tok = StreamTokenizer(vocab, unk_id=0)
        t0 = time.perf_counter()
        ids = drive(tok, text)
        dt = time.perf_counter() - t0
        print(f"{n_chars:>8} chars {len(ids):>7} tokens {dt:7.2f} s {len(ids) / dt:9.0f} tokens/s")
```

The ticket, verbatim:

> The longer the document, the slower the tokenization. Running `bench()` on the same machine, same kind of text, only the length differs:
>
> ```text
>   100000 chars   40044 tokens    0.78 s     51035 tokens/s
>   800000 chars  320300 tokens    9.13 s     35095 tokens/s
>  1600000 chars  640849 tokens   24.61 s     26039 tokens/s
>  3200000 chars 1282719 tokens   75.13 s     17074 tokens/s
> ```
>
> The downstream embedding service needs at least 200,000 tokens per second, and user uploads are often several MB.

(These timings were measured on a shared machine and only describe the symptom; the numbers on your own machine will differ.)

Behavioral contract (must be preserved by your changes): the vocabulary is a `dict[str, int]` with at least one entry; keys are non-empty and distinct, and the longest key has length $L \le 16$. `chunk` may be the empty string. In `read(max_tokens)`, `max_tokens >= 0`, and an id, once handed out, never changes; the token at a position is final only when "the stream is closed" or "there are at least $L$ untokenized characters". Calling `write` after `close()` raises `ValueError`; an empty vocabulary raises `ValueError`. `pending()` returns the number of characters written but not yet turned into tokens.

```text
vocab = {"t": 1, "h": 2, "e": 3, "r": 4, " ": 5, "th": 6, "the": 7, "there": 8}, unk_id = 0, L = 5

tok.write("the")     tok.pending() -> 3   tok.read(10) -> []         # fewer than 5 chars: "re" may follow and make "there"
tok.write("re is")   tok.read(10) -> [8]                             # "there"; the remaining " is" is under 5 chars, so wait
tok.close()          tok.read(10) -> [5, 0, 0]                       # " "; "i" and "s" are not in the vocabulary, one unk each
                     tok.read(10) -> []

The same text cut differently, reading 1 token at a time:
write("t") read(1) -> []    write("here i") read(1) -> [8]    write("s") read(1) -> []
close()    read(1) -> [5]   read(1) -> [0]   read(1) -> [0]   read(1) -> []
```

### Part 1 — Where does the time go

Don't change the code yet. Explain what you plan to measure, which tools you'll use, and which results would lead you to which conclusions; then find the cause of "the longer the document, the slower each token". If you find more than one thing worth fixing, state the per-token cost of each and how that cost changes with document length.

### Part 2 — Fix it without changing any output

Rewrite `StreamTokenizer` with the same interface so that the total work for one document grows linearly with the text length, and the work per token depends only on the vocabulary, not on how much text has already been written. For any input, any way of cutting it into chunks, and any sequence of `write` / `read` / `close` calls, every list returned by `read` and every value of `pending()` must be exactly the same as in the original.

The machine is shared and timings are very noisy. Use deterministic counts (for example, how many characters were copied, how many vocabulary lookups were made) in a doubling experiment to demonstrate the original's problem and your fix: when the text length doubles, the original's counts should grow roughly 4x and yours roughly 2x.

### Part 3 — Two threads at once, with a bounded backlog

Now the network thread and the batching thread call it truly concurrently. Write a `BoundedStreamTokenizer(vocab, unk_id, max_pending)`, with exactly one writer thread and one reader thread at any time:

```py
class BoundedStreamTokenizer:
    def __init__(self, vocab, unk_id, max_pending): ...
    def write(self, chunk): ...      # blocks while pending() >= max_pending, then appends the whole chunk
    def close(self): ...
    def read(self, max_tokens): ...  # blocks until at least one token is final; returns [] only
                                     # once the stream is closed and every token has been read
    def pending(self): ...
```

- After `write` returns, the backlog (`pending()`) is at most `max_pending - 1 + len(chunk)`, so memory is bounded.
- `read(max_tokens)` requires `max_tokens >= 1`; otherwise it raises `ValueError`.
- If `max_pending` is less than the longest vocabulary key length $L$, the constructor raises `ValueError`. Explain why this lower bound is needed.
- The sequence of ids read must be the same as single-threaded tokenization of the whole text, however the threads interleave.

## Reference solution

<details>
<summary>Show the reference solution</summary>

### Part 1 — How to locate it

**Step 1: turn the symptom into a number you can compare.** In the ticket, tokens/s drops as documents get longer. "Average time per token" is more intuitive: about 19.5 µs at 100 KB and about 58.6 µs at 3.2 MB. If the per-token cost were constant, tokens/s would be flat. It isn't, which means some piece of work done for every token is proportional to the document length (or some quantity tied to it). From 800 KB on, each doubling of length makes the total time grow 2.7 to 3.1x, between linear (2x) and quadratic (4x): this is usually "a linear term with a large constant + a quadratic term", with the quadratic term gradually taking over as the size grows.

**Step 2: profile, but do it once at each of two sizes.** Run the repro script under `cProfile`:

```py
import cProfile, pstats

vocab = make_vocab()
for n in (100_000, 400_000):
    text = make_text(vocab, n)
    pr = cProfile.Profile()
    pr.enable()
    drive(StreamTokenizer(vocab, unk_id=0), text)
    pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(5)
```

One run's output on our machine (only the top three lines kept, and some columns removed; `tottime` is the function's own time, excluding the functions it calls):

```text
100 KB, 40044 tokens
   ncalls  tottime  cumtime  function
    40044    1.286    2.140  _longest_match
  5886468    0.848    0.848  {method 'startswith' of 'str' objects}
      158    0.091    2.248  read

400 KB, 160079 tokens
   160079    5.110    8.409  _longest_match
 23531613    3.275    3.275  {method 'startswith' of 'str' objects}
      627    0.984    9.468  read
```

Looking at a single profile, the top entries are `_longest_match` and `startswith`, which account for over 90%. The `startswith` call count of 5,886,468 is exactly 40,044 × 147: every token tries all 147 vocabulary entries. This is a real cost and worth fixing, but it is constant per token, so it can't explain "the longer, the slower".

Compare the two runs side by side: the token count grew 4x, and `_longest_match`'s own time also grew about 4x (1.286 → 5.110), but `read`'s own time grew about 10.8x (0.091 → 0.984). The part that grows faster than linear is in `read`'s own code, not in the functions it calls.

**Step 3: find the line in `read` that is proportional to the length.** `read` itself only does three things: calls `_ready()`, `out.append`, and `self.buf = self.buf[n:]`. Python strings are immutable, so a slice creates a new string and copies all the remaining characters. So every token taken copies the entire backlog once.

How large is the backlog? Each round the network writes 4096 characters and downstream takes 256 tokens; at about 2.5 characters per token on average, that only removes about 640 characters, so about 84% of each chunk stays in the buffer. By the time the upload ends, the buffer holds nearly the whole document. After that, every token that drains it copies $O(N)$ characters, $O(N^2)$ in total. In unit tests and small documents the backlog is tiny, so it doesn't show.

`cProfile` only accounts per function; a slice is not a function call, so its time is charged to `read`'s `tottime` and it can't point to the line. To see lines you can use the third-party `line_profiler` (`kernprof -l -v`); the more reliable approach is counting, in the next step.

**Step 4: confirm with deterministic counts instead of timings.** Without changing a single line of the original, wrap it in two layers: a `dict` subclass that counts how many times the vocabulary is accessed (each key handed out by iteration, and each `get` / `[]` / `in`, counts once); and a subclass that turns `buf` into a property, adding the length of every new string assigned to `self.buf` to "characters copied". Use the same "write faster than read" rhythm as production, but shrink the chunk to 256 characters and the batch to 16 tokens, and run 4,000 / 8,000 / 16,000 characters (the verification code computes and asserts these numbers):

| Text length | Tokens | Original: characters copied | Original: vocabulary probes per token | Fixed: characters copied | Fixed: vocabulary probes per token |
|---|---|---|---|---|---|
| 4,000 | 1,558 | 2,714,273 | 147.9 | 4,004 | 6.4 |
| 8,000 | 3,148 | 10,769,510 | 147.9 | 8,004 | 6.5 |
| 16,000 | 6,366 | 43,349,443 | 147.9 | 16,009 | 6.5 |

The original's characters copied grow about 4x with each doubling, which is quadratic growth; vocabulary probes per token are a constant 147.9, which is linear growth, but with a large constant.

The conclusion is two hot spots of different natures:

| Location | Cost per token | As documents grow | When it is the main problem |
|---|---|---|---|
| `self.buf = self.buf[n:]` in `read` | Copies the whole backlog, $O(\text{backlog})$ | Grows | Large documents (the ticket's cause) |
| `_longest_match` scanning the whole vocabulary | $\lvert V\rvert$ calls to `startswith` | Constant | Small documents (top of the profile) |
| `self.buf += chunk` in `write` | Copies the backlog once per chunk | Grows | When chunks are tiny; secondary under this workload |

**Common wrong diagnoses:**

- Seeing the top of the profile and going to fix `_longest_match` (for example, switching to a trie). Small documents get about 7x faster, but large documents improve only a little: fixing only that, one measurement on our machine gave about 366,000 tokens/s at 100 KB, dropping to about 47,000 at 1.6 MB (original: 26,000) and about 24,000 at 3.2 MB (original: 17,000). That is still far from 200,000, it still gets slower as documents grow, and the growth law hasn't changed.
- Deciding that `self.buf += chunk` is the culprit. It does copy the whole backlog too, but only once every 4 KB; the slice happens once per token, about 1,600 times as often. Still, it is the same disease as the slice, so handle it in the same fix.
- "CPython optimizes `s += t`, so it's amortized O(1)." That is a CPython implementation detail that only kicks in when the left side is a local variable and the string has no other references. `self.buf` is an attribute, and the instance dict still holds a reference, so `self.buf += chunk` copies in full every time.
- "Python is slow, rewrite it in C++." In C++, `std::string::erase(0, n)` also has to shift all the remaining content, so it is still quadratic; the problem is the algorithm, not the language.
- "Make the batch bigger." The slice happens once per token, and each time copies the whole backlog; the backlog is determined by the speed difference between the network and downstream, not by the batch size. As long as downstream is slower than the network, total copying stays quadratic. (In the repro script, raising the batch until downstream outruns the network does make copying linear, but that amounts to assuming downstream got faster, which is not a fix you can make in production.)
- Suspecting the GC or the memory allocator. The counts already show that the work itself is quadratic.

### Part 2 — The fix

Three ideas.

1. **Move an index, not the text.** Use `self.pos` to record where in `self.buf` the next token starts; taking a token only does `self.pos += n`, with no copying.
2. **Queue new chunks and join them only when needed.** `write` just appends the chunk to a `deque`, $O(1)$. Only when `buf` has fewer than $L$ characters left after `pos` and there are new chunks queued does it join once: `buf[pos:] + "".join(incoming)`. Every written character is joined in only once, the old remainder is under $L$ characters each time, and the number of joins is at most the number of `write` calls, so total copying is $O(N + W L)$, where $W$ is the number of writes and $W \le N$ (empty chunks aren't queued).
3. **Look up by length instead of scanning the vocabulary.** Precompute all key lengths that occur in the vocabulary, longest first, and for each length slice one piece and look it up in the `dict` once. At most $D \le L$ lookups, each hashing at most $L$ characters, so each token is $O(L^2)$, independent of both the backlog and $\lvert V\rvert$.

Why the output is exactly the same: the condition for "is the current token final" hasn't changed (closed, or at least $L$ untokenized characters; queued chunks are joined in first when needed, so the number of characters it sees is exactly `pending()`); and the chosen entry is the same too (two different keys of the same length can't both be prefixes at the same position, so the longest one is unique).

```python
from collections import deque


class StreamTokenizer:
    """Same contract as the given version; total work is linear in the text length."""

    def __init__(self, vocab, unk_id):
        if not vocab:
            raise ValueError("empty vocabulary")
        self.vocab = vocab
        self.unk_id = unk_id
        self.max_len = max(len(key) for key in vocab)
        # Distinct key lengths, longest first: at most max_len dict lookups per token.
        self.lengths = sorted({len(key) for key in vocab}, reverse=True)
        self.buf = ""              # working text; the next token starts at self.pos
        self.pos = 0
        self.incoming = deque()    # chunks written but not yet moved into buf
        self.incoming_len = 0
        self.closed = False
        self.chars_in = 0
        self.tokens_out = 0

    def write(self, chunk):
        if self.closed:
            raise ValueError("write after close")
        if chunk:
            self.incoming.append(chunk)
            self.incoming_len += len(chunk)
        self.chars_in += len(chunk)

    def close(self):
        self.closed = True

    def pending(self):
        return len(self.buf) - self.pos + self.incoming_len

    def _ready(self):
        left = len(self.buf) - self.pos
        if left < self.max_len and self.incoming:
            # Only now copy: fewer than max_len old characters plus the new chunks.
            self.buf = self.buf[self.pos:] + "".join(self.incoming)
            self.pos = 0
            self.incoming.clear()
            self.incoming_len = 0
            left = len(self.buf)
        if left == 0:
            return False
        return self.closed or left >= self.max_len

    def _longest_match(self):
        buf, pos = self.buf, self.pos
        for n in self.lengths:
            if pos + n <= len(buf):
                token_id = self.vocab.get(buf[pos:pos + n])
                if token_id is not None:
                    return token_id, n
        return self.unk_id, 1

    def read(self, max_tokens):
        out = []
        while len(out) < max_tokens and self._ready():
            token_id, n = self._longest_match()
            out.append(token_id)
            self.pos += n
        self.tokens_out += len(out)
        return out
```

Complexity: $O(L^2)$ character operations per token and $O(N + WL)$ total copying, so $O(NL^2)$ overall for a document of $N$ characters, which is linear for a fixed vocabulary. Memory is still $O(\text{backlog})$: when the network is faster than downstream, the backlog still grows to the whole document; that is left to Part 3.

The two "Fixed" columns in the table above are this version's counts: characters copied grow 2x with each doubling (essentially $N$ itself), and about 6.5 vocabulary probes per token. The verification code also fixes each spot alone, confirming that both matter on their own: fixing only the lookup, copying still grows 4x; fixing only the copying, each token still makes 147.9 vocabulary probes. On our machine `bench()` gives about 500,000 tokens per second, essentially flat from 100 KB to 3.2 MB (about 2.6 seconds for 3.2 MB); this is for reference only, and the verification code makes no assertions about timings.

### Part 3 — Thread safety and backpressure

Protect the Part 2 object with one lock, and use a condition variable to let both sides wait:

- The writer thread waits while the backlog is `>= max_pending`; before `write` returns, the backlog is at most `max_pending - 1 + len(chunk)`. This is backpressure: when downstream is slow, the network thread slows down instead of piling up without bound.
- The reader thread waits while "there is no final token and the stream hasn't ended".
- Every state change (write, read, close) calls `notify_all()`. With one writer, one reader and `max_pending >= L`, the two sides never wait at the same time (see the next paragraph), so `notify()` would actually be enough; but both kinds of waiters share one condition variable, and once there are several readers or writers, `notify()` might wake only a thread of the same kind, leaving the other kind asleep forever. Using `notify_all()`, or two condition variables on the same lock, avoids that.

**Why `max_pending >= L`:** suppose `max_pending < L` and the backlog is exactly `max_pending` characters. The writer sees the backlog is full and waits; the reader sees fewer than $L$ characters, so the current token is not final yet (a longer key might still be completed), and it waits too. Each waits for the other: deadlock. With `max_pending >= L`, whenever the writer waits the backlog is at least $L$, so the reader can always take at least one token and then wake the writer; whenever the reader waits the backlog is under $L \le$ `max_pending`, so the writer isn't blocked.

```python
import threading


class BoundedStreamTokenizer:
    """StreamTokenizer for one writer thread and one reader thread, with a bounded backlog."""

    def __init__(self, vocab, unk_id, max_pending):
        self.tok = StreamTokenizer(vocab, unk_id)
        if max_pending < self.tok.max_len:
            raise ValueError("max_pending must be at least the longest vocabulary key")
        self.max_pending = max_pending
        self.peak_pending = 0          # largest backlog seen right after a write
        self.cond = threading.Condition()

    def write(self, chunk):
        with self.cond:
            while self.tok.pending() >= self.max_pending and not self.tok.closed:
                self.cond.wait()
            self.tok.write(chunk)      # raises ValueError after close
            self.peak_pending = max(self.peak_pending, self.tok.pending())
            self.cond.notify_all()

    def close(self):
        with self.cond:
            self.tok.close()
            self.cond.notify_all()

    def pending(self):
        with self.cond:
            return self.tok.pending()

    def read(self, max_tokens):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        with self.cond:
            while True:
                out = self.tok.read(max_tokens)
                if out:
                    self.cond.notify_all()   # the writer may have room again
                    return out
                if self.tok.closed and self.tok.pending() == 0:
                    return []
                self.cond.wait()
```

Tokenization happens inside the lock. With only one reader thread that is fine: the writer only needs to wait when the backlog is full, and at that point the reader is busy working. To have several reader threads tokenize in parallel, you would have to cut the text at "final boundaries" and hand the pieces out, which is much more complicated, and the order of the tokens would have to be guaranteed separately.

### Follow-ups

**1. If the client sends only 16 characters at a time, what is slowest in the original? How does the fixed version do?** In the original, `self.buf += chunk` copies the whole backlog on every write; the smaller the chunks, the more writes, so like the slice it becomes a quadratic main cost. In the fixed version `write` is just `deque.append`, and joining only happens when the buffer is about to run out, with each character joined only once, so it is independent of chunk size and stays linear.

**2. Is it enough to cut off the used prefix just once at the end of each `read` (`self.buf = self.buf[self.pos:]`)?** No. That copies the backlog once per call, about 256 times less than once per token, but the number of calls is still proportional to the document length, so the total is still quadratic, just with a smaller constant. In the verification code, this version's copying still grows about 4x per doubling. Either join only when fewer than $L$ characters remain, as in Part 2, or compact only once "the used prefix exceeds half" (amortized $O(1)$).

**3. If the vocabulary grows to 100,000 entries with a longest entry of 50 characters, does the fixed version still hold up?** Looking up by length takes at most $D$ lookups, each hashing at most 50 characters, so the worst case is about $50^2$ character operations per token: usable, but not ideal. A trie is better: walk down from the current position and stop as soon as you can't continue, looking at no more than $L$ characters per token and remembering the last word end along the way. For how to write the trie, see [Longest-Match Tokenizer](#/anthropic/coding/longest-match-tokenizer); the point of this problem is to find the bottleneck in the streaming wrapper, not the matcher itself.

**4. How do you keep this kind of problem from coming back?** Add a timing-free "growth test" to the test suite: use the counting wrappers to run sizes $n$ and $2n$ and assert that the ratio of the counts is below 2.5. It is stable on any machine and more reliable than a benchmark threshold. In production you can monitor "average processing time per token" bucketed by document size; a quadratic problem shows up as a clearly higher value in the large-document buckets.

**5. What if Part 3 has several writer threads?** For the same document stream, if several writers write concurrently, the order of the chunks is undefined and the text itself is scrambled; no lock can fix that. The right approach is one tokenizer per document, or having a single thread put the chunks in order by sequence number before writing them.

**6. Why didn't the original unit tests catch the problem?** In the tests the text was short, or they wrote a little and read a little, so the backlog always stayed small and the slices copied very little. This quadratic term only appears when "production outpaces consumption". Performance problems often depend on the shape of the load, so a repro should match the production rhythm as closely as possible (here, "write 4 KB, read 256 tokens").

<details>
<summary>Verification code (runnable)</summary>

```python
import random
import threading
import time

GIVEN_SRC = r'''
class StreamTokenizer:
    """Greedy longest-match tokenizer for text that arrives in pieces.

    vocab maps token strings (all non-empty) to ids. At each position the
    longest key that matches is taken; if no key matches, one character
    becomes unk_id.

        write(chunk)      add text; a cut may fall anywhere, even inside a token
        close()           no more text will be written
        read(max_tokens)  up to max_tokens ids that can no longer change, in
                          order; fewer (maybe none) if the rest needs more text

    Reading until the stream is closed and empty gives the same ids as
    tokenizing the whole text in one piece, however it was cut into chunks.
    """

    def __init__(self, vocab, unk_id):
        if not vocab:
            raise ValueError("empty vocabulary")
        self.vocab = vocab
        self.unk_id = unk_id
        self.max_len = max(len(key) for key in vocab)
        self.buf = ""            # written but not yet tokenized
        self.closed = False
        self.chars_in = 0        # counters for the metrics endpoint
        self.tokens_out = 0

    def write(self, chunk):
        if self.closed:
            raise ValueError("write after close")
        self.buf += chunk
        self.chars_in += len(chunk)

    def close(self):
        self.closed = True

    def pending(self):
        """Characters written but not yet turned into tokens."""
        return len(self.buf)

    def _ready(self):
        """Is the token at the front of the buffer final?"""
        if not self.buf:
            return False
        # With fewer than max_len characters, a longer key could still be
        # completed by text that has not arrived yet.
        return self.closed or len(self.buf) >= self.max_len

    def _longest_match(self):
        """(token id, length in characters) of the token at the front of the buffer."""
        best = None
        for key in self.vocab:
            if self.buf.startswith(key):
                if best is None or len(key) > len(best):
                    best = key
        if best is None:
            return self.unk_id, 1
        return self.vocab[best], len(best)

    def read(self, max_tokens):
        out = []
        while len(out) < max_tokens and self._ready():
            token_id, n = self._longest_match()
            out.append(token_id)
            self.buf = self.buf[n:]
        self.tokens_out += len(out)
        return out
'''

HARNESS_SRC = r'''
import random
import time

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def make_vocab(n_words=120, seed=7):
    """Every letter and the space, plus n_words random words of 2..8 letters. Ids start at 1."""
    rng = random.Random(seed)
    vocab = {ch: i + 1 for i, ch in enumerate(LETTERS + " ")}
    while len(vocab) < len(LETTERS) + 1 + n_words:
        word = "".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 8)))
        if word not in vocab:
            vocab[word] = len(vocab) + 1
    return vocab


def make_text(vocab, n_chars, seed=11):
    """Vocabulary words separated by spaces; about one word in ten is a number (not in the vocabulary)."""
    rng = random.Random(seed)
    words = [w for w in vocab if len(w) > 1]
    parts, size = [], 0
    while size < n_chars:
        w = str(rng.randint(0, 999)) if rng.random() < 0.1 else rng.choice(words)
        parts.append(w)
        size += len(w) + 1
    return " ".join(parts)[:n_chars]


def drive(tok, text, chunk=4096, batch=256):
    """As in production: the network writes a chunk, the batcher takes up to one batch,
    and so on; when the upload ends, close and drain."""
    ids = []
    for i in range(0, len(text), chunk):
        tok.write(text[i:i + chunk])
        ids.extend(tok.read(batch))
    tok.close()
    while True:
        got = tok.read(batch)
        if not got:
            return ids
        ids.extend(got)


def bench(sizes=(100_000, 800_000, 1_600_000, 3_200_000)):
    vocab = make_vocab()
    for n_chars in sizes:
        text = make_text(vocab, n_chars)
        tok = StreamTokenizer(vocab, unk_id=0)
        t0 = time.perf_counter()
        ids = drive(tok, text)
        dt = time.perf_counter() - t0
        print(f"{n_chars:>8} chars {len(ids):>7} tokens {dt:7.2f} s {len(ids) / dt:9.0f} tokens/s")
'''

FixedTokenizer = StreamTokenizer          # Part 2, defined above in this page


def load_class(src):
    ns = {}
    exec(src, ns)
    return ns["StreamTokenizer"]


GivenTokenizer = load_class(GIVEN_SRC)
harness = {}
exec(HARNESS_SRC, harness)
make_vocab, make_text, drive = harness["make_vocab"], harness["make_text"], harness["drive"]



# ---------- 1. statement examples, both versions ----------
EX_VOCAB = {"t": 1, "h": 2, "e": 3, "r": 4, " ": 5, "th": 6, "the": 7, "there": 8}


def run_ops(cls, vocab, unk_id, ops):
    tok = cls(vocab, unk_id)
    log = []
    for op, arg in ops:
        if op == "w":
            tok.write(arg)
            log.append(("w", tok.pending()))
        elif op == "r":
            log.append(("r", tok.read(arg), tok.pending()))
        else:
            tok.close()
            log.append(("c", tok.pending()))
    return log


for cls in (GivenTokenizer, FixedTokenizer):
    tok = cls(EX_VOCAB, 0)
    assert tok.max_len == 5
    tok.write("the")
    assert tok.pending() == 3 and tok.read(10) == []
    tok.write("re is")
    assert tok.read(10) == [8] and tok.pending() == 3
    tok.close()
    assert tok.read(10) == [5, 0, 0] and tok.read(10) == []
    try:
        tok.write("x")
        raise AssertionError("write after close must raise")
    except ValueError:
        pass
    try:
        cls({}, 0)
        raise AssertionError("empty vocabulary must raise")
    except ValueError:
        pass

    tok = cls(EX_VOCAB, 0)
    got = []
    for piece in ("t", "here i", "s"):
        tok.write(piece)
        got.append(tok.read(1))
    tok.close()
    for _ in range(4):
        got.append(tok.read(1))
    assert got == [[], [8], [], [5], [0], [0], []], got


# ---------- 2. given == fixed on random inputs, random cuts, random read sizes ----------
def ref_tokenize(text, vocab, unk_id):
    """Independent whole-text greedy longest match."""
    L = max(len(k) for k in vocab)
    ids, i = [], 0
    while i < len(text):
        for n in range(min(L, len(text) - i), 0, -1):
            if text[i:i + n] in vocab:
                ids.append(vocab[text[i:i + n]])
                i += n
                break
        else:
            ids.append(unk_id)
            i += 1
    return ids


rng = random.Random(2024)
for trial in range(1500):
    alphabet = "abc"
    vocab = {}
    for _ in range(rng.randint(1, 10)):
        key = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 5)))
        vocab.setdefault(key, len(vocab) + 1)
    text = "".join(rng.choice("abcd ") for _ in range(rng.randint(0, 50)))
    ops, i = [], 0
    while i < len(text):
        j = min(len(text), i + rng.choice([0, 1, 1, 2, 3, 7]))
        ops.append(("w", text[i:j]))
        i = j
        if rng.random() < 0.6:
            ops.append(("r", rng.randint(0, 4)))
    ops.append(("c", None))
    ops += [("r", rng.randint(1, 4)) for _ in range(len(text) + 1)]
    logs = [run_ops(cls, vocab, -1, ops) for cls in (GivenTokenizer, FixedTokenizer)]
    assert logs[0] == logs[1], (vocab, ops)
    ids = [t for entry in logs[1] if entry[0] == "r" for t in entry[1]]
    assert ids == ref_tokenize(text, vocab, -1)
    assert logs[1][-1] == ("r", [], 0)

# every single cut point, including cuts inside multi-character tokens
vocab = make_vocab(30)
text = make_text(vocab, 300)
whole = ref_tokenize(text, vocab, 0)
for cut in range(len(text) + 1):
    for cls in (GivenTokenizer, FixedTokenizer):
        tok = cls(vocab, 0)
        tok.write(text[:cut])
        ids = tok.read(10 ** 9)
        tok.write(text[cut:])
        tok.close()
        assert ids + tok.read(10 ** 9) == whole


# ---------- 3. deterministic work counters and the doubling law ----------
class CountingVocab(dict):
    """Counts every key handed out by iteration and every lookup."""
    probes = 0

    def __iter__(self):
        for key in dict.__iter__(self):
            self.probes += 1
            yield key

    def get(self, key, default=None):
        self.probes += 1
        return dict.get(self, key, default)

    def __getitem__(self, key):
        self.probes += 1
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        self.probes += 1
        return dict.__contains__(self, key)


def counted(cls):
    """Same class, but every string stored into self.buf adds its length to self.copied."""
    class Counted(cls):
        copied = 0

        @property
        def buf(self):
            return self._buf

        @buf.setter
        def buf(self, value):
            self._buf = value
            self.copied += len(value)
    return Counted


def measure(cls, n_chars, chunk=256, batch=16):
    vocab = CountingVocab(make_vocab())
    text = make_text(vocab, n_chars)
    tok = counted(cls)(vocab, unk_id=0)
    vocab.probes = 0                       # ignore the work done by __init__
    ids = drive(tok, text, chunk=chunk, batch=batch)
    probes = vocab.probes
    assert ids == ref_tokenize(text, dict(vocab), 0)
    return len(ids), tok.copied, probes


SIZES = (4000, 8000, 16000)


def table(cls):
    return [measure(cls, n) for n in SIZES]


def ratios(rows, k):
    return [b[k] / a[k] for a, b in zip(rows, rows[1:])]


given_rows = table(GivenTokenizer)
fixed_rows = table(FixedTokenizer)
print("given:", given_rows)
print("fixed:", fixed_rows)
assert [r[:2] for r in given_rows] == [(1558, 2714273), (3148, 10769510), (6366, 43349443)]
assert [r[:2] for r in fixed_rows] == [(1558, 4004), (3148, 8004), (6366, 16009)]
assert [round(r[2] / r[0], 1) for r in given_rows] == [147.9, 147.9, 147.9]
assert [round(r[2] / r[0], 1) for r in fixed_rows] == [6.4, 6.5, 6.5]
assert all(3.5 < x < 4.5 for x in ratios(given_rows, 1)), ratios(given_rows, 1)   # copies: quadratic
assert all(1.8 < x < 2.2 for x in ratios(fixed_rows, 1)), ratios(fixed_rows, 1)   # copies: linear
assert all(1.8 < x < 2.2 for x in ratios(given_rows, 2))                          # probes: linear, but
assert all(1.8 < x < 2.2 for x in ratios(fixed_rows, 2))                          # 147.9 vs 6.5 per token
# a bigger batch only helps once the reader outruns the writer (128 tokens ~ 320 chars per 256-char chunk)
fast_reader = [measure(GivenTokenizer, n, batch=128) for n in SIZES]
assert all(1.8 < x < 2.2 for x in ratios(fast_reader, 1)), ratios(fast_reader, 1)

# the given version hands out every key for every token (one startswith each),
# plus one vocab[best] for every token that is not unk
vocab = CountingVocab(make_vocab())
assert len(vocab) == 147 and max(map(len, vocab)) == 8
tok = GivenTokenizer(vocab, 0)
tok.write(make_text(vocab, 2000))
tok.close()
vocab.probes = 0
ids = tok.read(10 ** 9)
assert vocab.probes == len(ids) * 147 + sum(1 for t in ids if t != 0)

# numbers quoted from the ticket and the profile excerpt
assert round(0.78 / 40044 * 1e6, 1) == 19.5 and round(75.13 / 1282719 * 1e6, 1) == 58.6
assert round(24.61 / 9.13, 1) == 2.7 and round(75.13 / 24.61, 1) == 3.1
assert 40044 * 147 == 5886468 and 160079 * 147 == 23531613
assert round(5.110 / 1.286, 1) == 4.0 and round(0.984 / 0.091, 1) == 10.8

# the workload: about 2.5 characters per token, so a batch of 256 tokens takes about
# 640 of the 4096 characters written per round, and about 84% of the upload piles up
vocab = make_vocab()
text = make_text(vocab, 100_000)
tok = FixedTokenizer(vocab, 0)
n_read = 0
for i in range(0, len(text), 4096):
    tok.write(text[i:i + 4096])
    n_read += len(tok.read(256))
backlog = tok.pending()
tok.close()
n_tokens = n_read + len(tok.read(10 ** 9))
assert n_tokens == 40044 and round(len(text) / n_tokens, 1) == 2.5
assert round(256 * len(text) / n_tokens) == 639 and round(1 - 639 / 4096, 2) == 0.84
assert 0.80 < backlog / len(text) < 0.88, backlog
assert round(4096 / (len(text) / n_tokens), -2) == 1600     # slices per "+= chunk" copy
assert len(drive(FixedTokenizer(vocab, 0), make_text(vocab, 800_000))) == 320300


# ---------- 4. each hot spot matters on its own ----------
class OnlyLookupFixed(GivenTokenizer):
    """Given version with Part 2's dictionary lookup; the per-token slicing stays."""

    def __init__(self, vocab, unk_id):
        super().__init__(vocab, unk_id)
        self.lengths = sorted({len(key) for key in vocab}, reverse=True)

    def _longest_match(self):
        for n in self.lengths:
            if n <= len(self.buf):
                token_id = self.vocab.get(self.buf[:n])
                if token_id is not None:
                    return token_id, n
        return self.unk_id, 1


class OnlyCopyFixed(FixedTokenizer):
    """Part 2 version with the original scan over the whole vocabulary put back."""

    def _longest_match(self):
        best = None
        for key in self.vocab:
            if self.buf.startswith(key, self.pos):
                if best is None or len(key) > len(best):
                    best = key
        if best is None:
            return self.unk_id, 1
        return self.vocab[best], len(best)


class CompactPerRead(FixedTokenizer):
    """Follow-up 2: join new chunks on every read and cut the used prefix off after every read."""

    def read(self, max_tokens):
        if self.incoming:
            self.buf = self.buf[self.pos:] + "".join(self.incoming)
            self.pos = 0
            self.incoming.clear()
            self.incoming_len = 0
        out = super().read(max_tokens)
        self.buf = self.buf[self.pos:]
        self.pos = 0
        return out


for cls in (OnlyLookupFixed, OnlyCopyFixed, CompactPerRead):
    for trial in range(200):
        vocab = {}
        for _ in range(rng.randint(1, 8)):
            vocab.setdefault("".join(rng.choice("ab") for _ in range(rng.randint(1, 4))), len(vocab) + 1)
        text = "".join(rng.choice("abc") for _ in range(rng.randint(0, 40)))
        ops = [("w", text[i:i + 3]) for i in range(0, len(text), 3)]
        ops = [op for w in ops for op in (w, ("r", rng.randint(0, 3)))] + [("c", None)]
        ops += [("r", 2)] * (len(text) + 1)
        assert run_ops(cls, vocab, 0, ops) == run_ops(GivenTokenizer, vocab, 0, ops)

only_lookup = table(OnlyLookupFixed)
only_copy = table(OnlyCopyFixed)
compact = table(CompactPerRead)
print("only lookup fixed:", only_lookup)
print("only copy fixed:  ", only_copy)
print("compact per read: ", compact)
# fixing only the lookup leaves the quadratic copying
assert [r[1] for r in only_lookup] == [r[1] for r in given_rows]
assert all(3.5 < x < 4.5 for x in ratios(only_lookup, 1))
assert [round(r[2] / r[0], 1) for r in only_lookup] == [6.4, 6.5, 6.5]
# fixing only the copying leaves 147.9 vocabulary probes per token
assert [r[1] for r in only_copy] == [r[1] for r in fixed_rows]
assert [round(r[2] / r[0], 1) for r in only_copy] == [147.9, 147.9, 147.9]
# compacting once per read call is still quadratic, only with a smaller constant
assert all(3.5 < x < 4.5 for x in ratios(compact, 1)), ratios(compact, 1)
assert all(f[1] < c[1] < g[1] for f, c, g in zip(fixed_rows, compact, given_rows))


# ---------- 5. timings: printed only, never asserted (shared, noisy machine) ----------
vocab = make_vocab()
for n_chars in (25_000, 50_000, 100_000):
    text = make_text(vocab, n_chars)
    line = [f"{n_chars:>7} chars"]
    for name, cls in (("given", GivenTokenizer), ("only lookup", OnlyLookupFixed), ("fixed", FixedTokenizer)):
        t0 = time.perf_counter()
        n = len(drive(cls(vocab, 0), text))
        line.append(f"{name} {n / (time.perf_counter() - t0):9.0f} tok/s")
    print("   ".join(line))


# ---------- 6. Part 3: threads and a bounded backlog ----------
try:
    BoundedStreamTokenizer(EX_VOCAB, 0, 4)          # L = 5
    raise AssertionError("max_pending < L must raise")
except ValueError:
    pass
b = BoundedStreamTokenizer(EX_VOCAB, 0, 5)
try:
    b.read(0)
    raise AssertionError("max_tokens < 1 must raise")
except ValueError:
    pass

# a full backlog blocks the writer until the reader takes a token
b = BoundedStreamTokenizer(EX_VOCAB, 0, 5)
b.write("there is")                                  # 8 >= 5: the next write must wait
t = threading.Thread(target=b.write, args=(" the",))
t.start()
t.join(0.2)
assert t.is_alive()
assert b.read(1) == [8] and b.pending() in (3, 7)
t.join(10)
assert not t.is_alive() and b.pending() == 7
b.close()
rest = []
while True:
    out = b.read(5)
    if not out:
        break
    rest += out
assert rest == [5, 0, 0, 5, 7] == ref_tokenize(" is the", EX_VOCAB, 0)
try:
    b.write("x")
    raise AssertionError("write after close must raise")
except ValueError:
    pass

# a reader with nothing final waits; close() wakes it
b = BoundedStreamTokenizer(EX_VOCAB, 0, 5)
res = []
t = threading.Thread(target=lambda: res.append(b.read(3)))
t.start()
t.join(0.2)
assert t.is_alive()
b.write("th")                                        # 2 < L: still not final
t.join(0.2)
assert t.is_alive()
b.close()
t.join(10)
assert not t.is_alive() and res == [[6]]


def run_threads(vocab, text, max_pending, seed):
    r = random.Random(seed)
    pieces, i = [], 0
    while i < len(text):
        j = min(len(text), i + r.randint(0, 12))
        pieces.append(text[i:j])
        i = j
    b = BoundedStreamTokenizer(vocab, 0, max_pending)
    got, errors = [], []

    def writer():
        try:
            for piece in pieces:
                b.write(piece)
            b.close()
        except Exception as e:                       # pragma: no cover
            errors.append(e)

    def reader():
        try:
            rr = random.Random(seed + 1)
            while True:
                out = b.read(rr.randint(1, 7))
                if not out:
                    return
                got.extend(out)
        except Exception as e:                       # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(30)
    assert not any(th.is_alive() for th in threads), "deadlock"
    assert not errors, errors
    assert got == ref_tokenize(text, vocab, 0)
    assert b.peak_pending <= max_pending - 1 + max([len(p) for p in pieces] + [0])
    return b.peak_pending


vocab = make_vocab(40)
for seed in range(40):
    text = make_text(vocab, 600, seed=seed)
    run_threads(vocab, text, max_pending=rng.choice([8, 9, 16, 50]), seed=seed)
print("all checks passed")
```

</details>

</details>
