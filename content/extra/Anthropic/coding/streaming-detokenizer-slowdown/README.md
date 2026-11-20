# Streaming Detokenizer That Stalls on Long Replies

## Problem

You have inherited `StreamDecoder`, the streaming decoding component in a model-serving frontend (a detokenizer: it turns token ids back into text). Every time the model generates a token, the frontend calls `push(token_id)` for that reply and pushes the returned text fragment (delta) to the client over SSE (server-sent events); when the reply ends, it calls `finish()`; the result of `text()` is saved in the conversation history. One frontend process serves all concurrent replies from a single event loop, and all decoding happens on that one thread.

The vocabulary is byte-level: each token id stands for a fixed byte string. A Chinese character (3 bytes in UTF-8) or an emoji (4 bytes) is often split across two tokens, and the model may also output bytes that don't form valid UTF-8. So the last token may stop in the middle of a character, and the decoder then has to wait rather than send half a character to the client.

This code's output is correct and all existing tests pass. The problem is performance. Below are the code, a repro script and the ticket. `ByteTokenizer` is a shared library that other services also use, so you can't change it; `StreamDecoder` is yours.

```py
class ByteTokenizer:
    """Byte-level vocabulary: token id i always stands for the byte string pieces[i].

    A piece may end in the middle of a multi-byte UTF-8 character, and a few
    special ids (end of turn and the like) have empty pieces.
    """

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def token_bytes(self, token_id):
        return self.pieces[token_id]

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")
```

```py
class StreamDecoder:
    """Turns the token ids of one model reply into text deltas for the client.

        push(token_id) -> str  text that became final with this token ("" if none)
        finish() -> str        whatever is left, once the reply is over
        text() -> str          everything returned so far (saved with the conversation)

    The deltas, joined, are exactly tokenizer.decode(all ids). A delta never
    contains half a character: while the decoded text ends in U+FFFD the last
    token may have stopped in the middle of a multi-byte character, so we wait.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids = []
        self.emitted = ""
        self.done = False

    def push(self, token_id):
        if self.done:
            raise ValueError("push after finish")
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        delta = text[len(self.emitted):]
        self.emitted += delta
        return delta

    def finish(self):
        if self.done:
            return ""
        self.done = True
        delta = self.tok.decode(self.ids)[len(self.emitted):]
        self.emitted += delta
        return delta

    def text(self):
        return self.emitted
```

The repro script (a single reply, decoding only, no network):

```py
import random
import time

WORDS = ["the", "model", "stream", "token", "reply", "text", "byte", "fast", "slow", "user", "and", "is"]
HANZI = "你好世界流式分词解码模型输出文本速度用户问题回答"
EMOJI = "😀🚀✨🎉"


def make_tokenizer():
    """All 256 single bytes; English words with and without a leading space; every Chinese
    character and emoji cut into two pieces; one special id (end of turn) with no bytes."""
    pieces = [bytes([b]) for b in range(256)]
    for w in WORDS:
        pieces += [w.encode(), (" " + w).encode()]
    for ch in HANZI + EMOJI:
        b = ch.encode()
        pieces += [b[:2], b[2:]]
    pieces.append(b"")
    return ByteTokenizer(pieces)


def encode(tok, text):
    """Greedy longest match over the UTF-8 bytes (a stand-in for the model's tokenizer)."""
    index = {}
    for i, p in enumerate(tok.pieces):
        if p:
            index.setdefault(p, i)
    longest = max(len(p) for p in index)
    data, ids, i = text.encode(), [], 0
    while i < len(data):
        for n in range(min(longest, len(data) - i), 0, -1):
            if data[i:i + n] in index:
                ids.append(index[data[i:i + n]])
                i += n
                break
    return ids


def make_reply(n_tokens, seed=5):
    """Token ids of a mixed English / Chinese / emoji reply, n_tokens long."""
    rng = random.Random(seed)
    tok = make_tokenizer()
    ids = []
    while len(ids) < n_tokens:
        r = rng.random()
        if r < 0.5:
            s = " " + rng.choice(WORDS)
        elif r < 0.9:
            s = "".join(rng.choice(HANZI) for _ in range(rng.randint(1, 6))) + rng.choice("，。")
        else:
            s = rng.choice(EMOJI)
        ids += encode(tok, s)
    return tok, ids[:n_tokens]


def bench(n_tokens=20_000, every=4_000):
    tok, ids = make_reply(n_tokens)
    dec = StreamDecoder(tok)
    t0 = time.perf_counter()
    for k, token_id in enumerate(ids, 1):
        dec.push(token_id)
        if k % every == 0:
            t1 = time.perf_counter()
            print(f"tokens {k - every:>6} to {k:<6} {every / (t1 - t0):9.0f} tokens/s")
            t0 = t1
    dec.finish()
```

The ticket:

```text
Title: long replies get slower and slower, frontend process CPU pegged
Symptom: at peak, one frontend process serves about 60 replies at once, each model producing about 50 token/s, about 3000 token/s in total.
      All is fine while replies are short; once there are many long replies (several thousand tokens or more), this process hits 100% CPU,
      and the token intervals of every stream grow together. Throughput on the GPU side is normal.
Repro: output of bench()
      tokens      0 to 4000        7084 tokens/s
      tokens   4000 to 8000        2385 tokens/s
      tokens   8000 to 12000       1437 tokens/s
      tokens  12000 to 16000       1023 tokens/s
      tokens  16000 to 20000        795 tokens/s
Expected: the time to decode one token does not depend on how long the reply already is; at least 200,000 tokens per second for a single reply.
```

(These timings were measured on a shared machine and only describe the symptom; the numbers on your own machine will differ.)

Behavioral contract (must be preserved by your changes):

- `pieces` are arbitrary byte strings: they may be empty (special tokens such as end of turn) and may not be valid UTF-8. Token ids are always in range.
- The decoding rule is exactly `tokenizer.decode`: concatenate the bytes and decode them as UTF-8 with `errors="replace"`. Under Python's rules, each maximal subpart that can't be decoded becomes one U+FFFD; for example, a truncated `e4 bd` becomes one U+FFFD, and so does `ff`.
- For any `pieces` and any token sequence, after your changes the return value of every `push` / `finish` call, and the value of `text()` at any moment, must be exactly the same as in the original, including when an empty string is returned.
- Calling `push` after `finish()` raises `ValueError`; a second `finish()` returns `""`.

```text
pieces = [b"h", b"i", b" ", b"\xe4\xbd", b"\xa0", b"\xe5\xa5\xbd", b"\xf0\x9f", b"\x98\x80", b"\xff", b""]
#  id:      0     1     2     3            4        5                6            7            8        9
# "你" = e4 bd a0, "好" = e5 a5 bd, "😀" = f0 9f 98 80; ff never appears in valid UTF-8

push(0) -> "h"         push(1) -> "i"
push(3) -> ""          # e4 bd are the first two bytes of a three-byte character; the decoded text ends in U+FFFD, so wait
push(4) -> "你"
push(6) -> ""          push(7) -> "😀"
push(5) -> "好"
push(8) -> ""          # ff decodes to U+FFFD, so it waits as well (the original can't tell "bad byte" from "half a character")
push(2) -> "\ufffd "   # U+FFFD followed by a space
push(9) -> ""          # empty piece
push(3) -> ""
finish() -> "\ufffd"   # the reply stops in the middle of a character: e4 bd becomes one U+FFFD
text()   -> "hi你😀好\ufffd \ufffd"

Another reply: push(3) -> ""   push(0) -> "\ufffdh"    # half a character directly followed by "h"
```

### Part 1 — Where does the time go

Don't change the code yet. Explain what you plan to measure, which tools you'll use, and which results would lead you to which conclusions; then find the cause of "the longer the reply, the slower each token". If you find more than one thing worth fixing, state the per-token cost of each and how that cost changes with reply length.

### Part 2 — Fix it without changing any return value

Rewrite `StreamDecoder` with the same interface so that the work per token (amortized) depends only on that token's number of bytes, not on how long the reply already is. Every call's return value must be exactly the same as in the original (see the contract above). You may use any public method of `ByteTokenizer`.

The machine is shared and timings are very noisy. Use deterministic counts (for example, how many tokens were read, how many characters were copied) in a doubling experiment to demonstrate the original's problem and your fix: when the reply length doubles, the original's counts should grow roughly 4x and yours roughly 2x.

### Part 3 — Stop strings

The product needs to support stop sequences: the caller supplies several strings, and as soon as any of them appears in the reply, the reply is cut just before it; neither the stop string nor anything after it is sent to the client, and the model is told to stop generating. Write a `StopStreamDecoder`:

```py
class StopStreamDecoder:
    def __init__(self, tokenizer, stops): ...   # stops: a list of non-empty strings (the list may be empty)
    def push(self, token_id): ...                # -> str
    def finish(self): ...                        # -> str
    def text(self): ...                          # everything returned so far
    # attribute stopped: bool, True once a stop string has appeared
```

Let $F$ be the text that the Part 2 decoder has returned so far, i.e. the text that is already final.

- The "first" stop string is the occurrence in $F$ that **ends earliest**; among those ending at the same position, take the longer one (it starts earlier). As soon as it appears, `stopped` becomes `True`, and the text returned so far is exactly the part of $F$ before this occurrence; after that, `push` and `finish` both return only `""`, without raising.
- Before stopping, return as early as possible: after each `push`, the text returned so far equals $F$ minus "the longest suffix that is also a proper prefix of some stop string". On `finish()`, return everything that is left.
- The work per call may depend only on the length of the text added by this call and on the number and length of the stop strings, not on how long the reply already is.
- If a stop string is empty, the constructor raises `ValueError`. When `stops` is an empty list, it behaves like the Part 2 decoder. If it hasn't stopped, calling `push` after `finish()` raises `ValueError`, as in Part 2.

```text
pieces as above, stops = ["i h", "你好"]
push(0) -> "h"
push(1) -> ""      # F = "hi"; "i" could be the start of "i h", so hold it
push(3) -> ""      # half a character; F hasn't changed
push(4) -> "i"     # F = "hi你": "i你" doesn't start any stop string, but "你" starts "你好", so hold it
push(5) -> ""      # F = "hi你好": "你好" appears, stopped = True, and "你好" is not returned
push(2) -> ""      finish() -> ""      text() -> "hi"

stops = ["hi hi", "i h"], pushing 0, 1, 2, 0, 1 in order
returns "", "", "", "h", ""          # after the fourth token F = "hi h"; "i h" ends here, earlier than "hi hi", so cut before it
stopped = True, text() -> "h"

stops = ["好", "你好"]: push(3) -> ""   push(4) -> ""   push(5) -> ""   # both end at the same time; take the longer "你好"
text() -> ""

stops = ["i h"]: push(0) -> "h"   push(1) -> ""   finish() -> "i"   stopped = False
```

## Reference solution

<details>
<summary>Show the reference solution</summary>

### Part 1 — How to locate it

**Step 1: turn the symptom into a number you can compare.** Convert the ticket's tokens/s into the average time per token, then divide by the position of that segment (how many tokens precede the token in the middle of the segment):

| Position (tokens) | Time per token | Divided by middle position |
|---|---|---|
| 0–4000 | 141 µs | 0.0706 µs |
| 4000–8000 | 419 µs | 0.0699 µs |
| 8000–12000 | 696 µs | 0.0696 µs |
| 12000–16000 | 978 µs | 0.0698 µs |
| 16000–20000 | 1258 µs | 0.0699 µs |

The last column is almost constant: the time to process the $k$-th token is proportional to $k$, about "70 ns per preceding token". That means every `push` touches all the previous tokens, and the total work for the whole reply is $O(n^2)$. This is far more specific than "it gets slower and slower", and it already tells us what to look for.

The pegged CPU in the ticket matches too: at 3000 token/s in total, if every stream were in the 0–4000 segment, decoding would take only 42% of one core; once they are all in the 4000–8000 segment it needs 1.26 cores, more than one thread can do. The event loop is single-threaded, so when one `push` stalls, every stream waits. Without a ready-made repro script, the first step would be to confirm with a sampling profiler (for example `py-spy`, which can attach directly to a production process) that the CPU really is spent on decoding, not on networking or serialization.

**Step 2: profile, once at each of two sizes.**

```py
import cProfile, pstats

for n in (2_000, 4_000):
    tok, ids = make_reply(n)
    dec = StreamDecoder(tok)
    pr = cProfile.Profile()
    pr.enable()
    for token_id in ids:
        dec.push(token_id)
    dec.finish()
    pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(5)
```

One run's output on our machine (some columns removed; `tottime` is the function's own time, `cumtime` includes the functions it calls):

```text
2000 tokens
   ncalls  tottime  cumtime  function
  2005001    0.311    0.311  <genexpr>        (the generator expression inside ByteTokenizer.decode)
     2001    0.228    0.539  {method 'join' of 'bytes' objects}
     2001    0.006    0.006  {method 'decode' of 'bytes' objects}
     2000    0.003    0.550  push

4000 tokens
  8010001    1.241    1.241  <genexpr>
     4001    0.900    2.141  {method 'join' of 'bytes' objects}
     4001    0.022    0.022  {method 'decode' of 'bytes' objects}
     4000    0.007    2.175  push
```

Key points:

- `ByteTokenizer.decode` was called only 2001 times (once per `push`, once for `finish`), so the call count is fine. The problem is how expensive each call is.
- The generator expression's call count of 2,005,001 is exactly $\sum_{k=1}^{2000}(k+1) + 2001$: a generator counts one call per value produced, plus one more when it finishes. In other words, the $k$-th `push` walks over all $k$ ids. Doubling the token count makes both its time and `join`'s time grow about 4x.
- The actual UTF-8 decoding (`bytes.decode`) accounts for only 1%. Decoding itself isn't slow; it's being given too much data.
- `push`'s own time is almost zero, so `self.emitted += delta` doesn't show up at all here. Remember this; it matters later.

**Step 3: map it back to the code.** In `push`, `text = self.tok.decode(self.ids)` decodes the whole reply from scratch for every token: produce $k$ pieces, join them into one bytes object, then decode that into a str. Whoever wrote this line did it to handle "half a character" correctly, and it really is simple and reliable, but it costs $O(k)$ per token. Replies in the unit tests are only a few dozen tokens, so it doesn't show.

**Step 4: confirm with deterministic counts instead of timings.** Without changing a single line of the original, wrap it in two layers: a `ByteTokenizer` subclass that counts how many tokens the decoder reads (`decode(ids)` counts `len(ids)`, `token_bytes` counts 1); and a subclass that turns `emitted` into a property, adding the length of each new string assigned to it to "characters copied". Run 1,000 / 2,000 / 4,000 tokens from the same reply generator (the verification code computes and asserts these numbers):

| Tokens | Text length | Original: tokens read | Original: characters copied into `emitted` | Fixed: tokens read | Fixed: copies |
|---|---|---|---|---|---|
| 1,000 | 983 | 501,500 | 245,195 | 1,000 | 0 |
| 2,000 | 1,844 | 2,003,000 | 954,795 | 2,000 | 0 |
| 4,000 | 3,793 | 8,006,000 | 3,907,489 | 4,000 | 0 |

The original's tokens read is exactly $n(n+1)/2 + n$, about 4x per doubling. Characters copied also grow about 4x per doubling: that is the second quadratic cost.

**Step 5: fix one spot and measure again.** Make only the decoding incremental (as in Part 2) but keep `self.emitted += delta`, then profile at 50,000 / 100,000 / 200,000 tokens (the following is one measurement on our machine): `push`'s own time goes 0.164 → 0.481 → 1.549 seconds, about 3x per doubling, while the functions it calls only double; `bench(200_000, 40_000)` drops from 380,000 tokens per second to 83,000. The remaining superlinear part is in `push`'s own code, namely `self.emitted += delta`: Python strings are immutable, so concatenation creates a new string and copies all the text emitted so far. When the reply contains emoji, every character of the whole string takes 4 bytes (CPython stores all characters at the width of the widest one), so copying is even more expensive.

The conclusion is two hot spots:

| Location | Cost per token | As replies grow | When it is the main problem |
|---|---|---|---|
| `self.tok.decode(self.ids)` in `push` | Re-joins and re-decodes all previous tokens, $O(k)$, about 70 ns per old token | Grows | Always; the ticket's cause |
| `self.emitted += delta` (also in `finish`) | Copies all the text returned so far, $O(\text{characters emitted})$, but only a memory copy | Grows | After the first fix, in long replies of around a hundred thousand tokens |

**Common wrong diagnoses:**

- "Just decode the new token's bytes" (`decode([token_id])`). It's faster, but split characters turn into garbage: `"你"` split into the two tokens `e4 bd` and `a0` decodes separately into two U+FFFD. The output changes, which is not acceptable.
- "Use `errors="ignore"`, so it never ends in U+FFFD and never has to wait." That silently drops bad bytes, so the output changes. And it doesn't reduce the amount of data decoded each time either.
- "UTF-8 decoding is slow; switch to a faster library." In the profile, `bytes.decode` is only 1%. The slowness is in how many calls there are and how much data each processes, not in the speed of a single call.
- "It's SSE or the network." The repro script doesn't touch the network at all and still slows down.
- "Decode only the last 32 tokens and take the difference." The idea is right, but the start of the window can land in the middle of a character, giving an extra U+FFFD at the beginning, and it takes care to match the original call for call (see follow-up 2). For a byte-level vocabulary, the approach below is more direct.
- Stopping after fixing only the first spot. Medium-length replies do get better, but replies of around a hundred thousand tokens still get slower and slower, because the second quadratic cost is still there.

### Part 2 — The fix

Three ideas.

1. **Decode only the newly arrived bytes.** Keep a `tail`: the unfinished half character left by the previous token (at most 3 bytes). When a new token arrives, decode the part of `tail + new bytes` that is "already settled", and keep the unfinished end as the new tail. Find the cut like this: from the end, skip back over at most 3 continuation bytes (`80`–`BF`); if the byte you reach is a lead byte, there aren't enough bytes after it yet, and what's there is valid so far, the tail starts at it; otherwise there is no tail. The UTF-8 rules: `C2`–`DF` is followed by 1 continuation byte, `E0`–`EF` by 2, `F0`–`F4` by 3; `C0`, `C1` and `F5`–`FF` are never valid; in addition, the second byte after `E0` must be in `A0`–`BF`, after `ED` in `80`–`9F` (excluding surrogates), after `F0` in `90`–`BF`, and after `F4` in `80`–`8F`.
2. **Copy the original's waiting rule exactly.** The original waits while "the whole text ends in U+FFFD". The cut is either at the end (the last character is complete or already known to be invalid) or on a lead byte, and the decoder always starts a new character at a lead byte, so the decoding of the bytes before the cut doesn't change no matter what follows. Hence the whole text = text already returned + text decoded but not yet returned (`held`) + the decoding of the tail; and an unfinished but valid tail decoded on its own is exactly one U+FFFD. So the original waits if and only if `tail` is non-empty, or the last segment of `held` ends in U+FFFD (the text already returned never ends in U+FFFD itself, or the original wouldn't have returned it). When it doesn't wait, the delta the original returns is exactly `held` joined together.
3. **Don't accumulate a string.** Put the returned fragments in a list and join them when `text()` is called.

Complexity: each `push` processes at most 3 + `len(piece)` bytes, plus a constant number of list operations. Each segment in `held` is joined only once, when it is handed out, so the total work is $O(\text{total bytes})$, amortized $O(\text{this token's bytes})$ per token. `text()` is $O(\text{text length})$; the original only calls it once, when saving the conversation.

```python
def incomplete_tail(data):
    """Index where an unfinished UTF-8 character starts at the end of data, or len(data).

    "Unfinished" means: the bytes from there on are a proper prefix of some valid
    UTF-8 encoding, so the next bytes may still complete it.
    """
    n = len(data)
    for i in range(n - 1, max(n - 4, -1), -1):
        b = data[i]
        if 0x80 <= b <= 0xBF:            # continuation byte: keep looking for the lead byte
            continue
        if 0xC2 <= b <= 0xDF:
            need = 2
        elif 0xE0 <= b <= 0xEF:
            need = 3
        elif 0xF0 <= b <= 0xF4:
            need = 4
        else:                            # ASCII, or a byte that can never start a character
            return n
        if n - i >= need:                # long enough: complete (or invalid), decode it now
            return n
        if n - i >= 2:                   # these lead bytes restrict the second byte
            lo, hi = {0xE0: (0xA0, 0xBF), 0xED: (0x80, 0x9F),
                      0xF0: (0x90, 0xBF), 0xF4: (0x80, 0x8F)}.get(b, (0x80, 0xBF))
            if not lo <= data[i + 1] <= hi:
                return n
        return i
    return n


class StreamDecoder:
    """Same return values as the given version; work per token does not grow with the reply."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.tail = b""      # an unfinished character (at most 3 bytes)
        self.held = []       # decoded but not returned; non-empty only while we are waiting
        self.parts = []      # everything returned so far
        self.done = False

    def push(self, token_id):
        if self.done:
            raise ValueError("push after finish")
        data = self.tail + self.tok.token_bytes(token_id)
        cut = incomplete_tail(data)
        piece = data[:cut].decode("utf-8", errors="replace")
        self.tail = data[cut:]
        if piece:
            self.held.append(piece)
        # The given version waits while decode(all ids) ends in U+FFFD, and that text is
        # text() + "".join(held) + (one U+FFFD if tail is not empty).
        if self.tail or (self.held and self.held[-1].endswith("\ufffd")):
            return ""
        delta = "".join(self.held)
        self.held.clear()
        self.parts.append(delta)
        return delta

    def finish(self):
        if self.done:
            return ""
        self.done = True
        delta = "".join(self.held) + self.tail.decode("utf-8", errors="replace")
        self.held.clear()
        self.tail = b""
        self.parts.append(delta)
        return delta

    def text(self):
        return "".join(self.parts)
```

The two "Fixed" columns in the count table above are this version: each token is read once, and there is no accumulated copying. The verification code also checks:

- On random `pieces` (including empty strings, invalid bytes, the edge cases starting with `E0`/`ED`/`F0`/`F4`, and genuine U+FFFD characters) and random token sequences, the original and the fixed version return the same value on every call and have the same `text()` at every moment; the same holds when real text is cut into tokens at random byte positions (including inside characters);
- For all strings of 0–2 bytes and many strings of 3–4 bytes, the front part cut off by `incomplete_tail` decodes the same no matter what follows; the remaining tail is at most 3 bytes and can be completed into a valid character;
- Fixing each spot alone confirms that both matter on their own: fixing only the decoding, copying still grows about 4x; fixing only the accumulation, tokens read still grow about 4x.

On our machine the fixed version's `bench()` runs at about 610,000 tokens per second, essentially flat from the start of the reply to 200,000 tokens; this is for reference only, and the verification code makes no assertions about timings.

### Part 3 — Stop strings

The most direct approach is to search for stop strings in "all the text so far" on every call. That makes each call $O(\text{reply length})$, bringing back the quadratic cost we just removed.

The right approach is to look only at a small window: `hold` (the part not yet returned that could be the start of some stop string, at most $M-1$ characters, where $M$ is the length of the longest stop string) plus the text newly settled by this call.

- **A stop string not yet found must start inside the window.** `hold` is exactly "the longest suffix of $F$ that is a proper prefix of some stop string". If an occurrence started before the window, then the part of it overlapping the old text would be a longer suffix of the old $F$ that is also a proper prefix of a stop string, a contradiction. By the same argument, the new `hold` also lies inside the window.
- **Find the occurrence that ends earliest.** For each stop string `s`, `window.find(s)` gives its leftmost occurrence, which is also the one that ends earliest; across all stop strings, take the one with the smallest (end position, start position).
- **Work.** Each call searches once per stop string in a window of at most $M-1+\text{new text length}$, then checks at most $M-1$ suffixes, independent of the reply length.

When `stopped` becomes `True`, the caller should tell the model to stop generating.

```python
class StopStreamDecoder:
    """Part 2's StreamDecoder, cut just before the first stop string (which is not returned)."""

    def __init__(self, tokenizer, stops):
        if any(not s for s in stops):
            raise ValueError("empty stop string")
        self.inner = StreamDecoder(tokenizer)
        self.stops = list(stops)
        self.keep = max((len(s) for s in self.stops), default=1) - 1
        self.hold = ""           # final text not returned yet: it may be the start of a stop string
        self.stopped = False
        self.parts = []

    def push(self, token_id):
        if self.stopped:
            return ""
        return self._feed(self.inner.push(token_id), final=False)

    def finish(self):
        if self.stopped:
            return ""
        return self._feed(self.inner.finish(), final=True)

    def text(self):
        return "".join(self.parts)

    def _feed(self, new, final):
        window = self.hold + new
        cut = self._first_stop(window)
        if cut >= 0:
            self.stopped = True
            out, self.hold = window[:cut], ""
        else:
            k = 0 if final else self._open_prefix(window)
            out, self.hold = window[:len(window) - k], window[len(window) - k:]
        self.parts.append(out)
        return out

    def _first_stop(self, window):
        """Start of the stop string that ends first in window (the longer one on a tie), or -1."""
        best = None
        for s in self.stops:
            i = window.find(s)
            if i >= 0 and (best is None or (i + len(s), i) < best):
                best = (i + len(s), i)
        return -1 if best is None else best[1]

    def _open_prefix(self, window):
        """Length of the longest suffix of window that is a proper prefix of some stop string."""
        for k in range(min(self.keep, len(window)), 0, -1):
            end = window[len(window) - k:]
            if any(len(s) > k and s.startswith(end) for s in self.stops):
                return k
        return 0
```

Stop strings are compared against "text that is already final", and the Part 2 decoder never hands out half a character, so a Chinese character split across two tokens is still matched correctly. The verification code compares call for call against a brute-force version that depends only on the definition (checking every end position on the full $F$ each time), and uses counts to confirm that the total length of the windows scanned is at most "text length + $(M-1)\times$ number of calls", growing about 2x when the reply length doubles.

### Follow-ups

**1. Can you just use the standard library?** Yes. `codecs.getincrementaldecoder("utf-8")(errors="replace")` is an incremental UTF-8 decoder; it keeps the unfinished bytes itself, and `getstate()[0]` is those bytes. The waiting rule becomes "the decoder still holds bytes, or `held` ends in U+FFFD":

```python
import codecs


class StdlibStreamDecoder:
    """Part 2 with the standard library's incremental UTF-8 decoder instead of incomplete_tail."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.held = []
        self.parts = []
        self.done = False

    def push(self, token_id):
        if self.done:
            raise ValueError("push after finish")
        piece = self.dec.decode(self.tok.token_bytes(token_id))
        if piece:
            self.held.append(piece)
        if self.dec.getstate()[0] or (self.held and self.held[-1].endswith("\ufffd")):
            return ""
        delta = "".join(self.held)
        self.held.clear()
        self.parts.append(delta)
        return delta

    def finish(self):
        if self.done:
            return ""
        self.done = True
        delta = "".join(self.held) + self.dec.decode(b"", final=True)
        self.held.clear()
        self.parts.append(delta)
        return delta

    def text(self):
        return "".join(self.parts)
```

One detail: when CPython's incremental decoder sees `ED A0`–`ED BF` (surrogates, invalid), it buffers them first, whereas `incomplete_tail` decodes them immediately into two U+FFFD. In both cases the original is waiting (the whole text ends in U+FFFD), so the return values are still the same; the verification code compares all three versions on random inputs containing `ED`. In the interview, saying "the standard library has an incremental decoder" is enough; writing the cut by hand is to show the principle.

**2. What if the tokenizer's decode isn't simple byte concatenation?** For example, SentencePiece turns a leading "▁" into a space and strips the space at the start of a sentence when decoding, and some tokenizers also normalize, so decoding a single token doesn't agree with decoding the whole sequence, and you can't decode token by token. The usual approach is to keep only a short window: remember `prefix_offset` and `read_offset`, decode `ids[prefix_offset:]` and `ids[prefix_offset:read_offset]` each time, and the difference between the two (when it doesn't end in U+FFFD) is the new text; then slide the window forward. The window length is bounded, so the work per token is constant. Incremental detokenization in inference frameworks such as vLLM follows this idea.

**3. What if the server calls `text()` on every token to save progress?** The fixed version's `text()` joins all the fragments every time, $O(n)$, so the quadratic cost is back. The original's `text()` is $O(1)$, but only because it already paid for the copying in `push`. As long as the caller wants a complete copy of the full text on every token (for example, writing the whole thing to a database), the total cost is at least the sum of all prefix lengths, which is quadratic however you implement it. What needs to change is the caller: append each delta to storage, or save at time intervals (say once a second), or save only when the reply ends.

**4. What does the fixed version do when the model outputs a long run of invalid bytes?** The whole text keeps ending in U+FFFD, so the original keeps waiting, and so does the fixed version; `held` grows longer and longer until a valid character appears and everything is handed out together. The time is still linear (each segment is joined only once), but the client sees nothing for a while. A better rule is "wait only while `tail` is non-empty": bad bytes are already known to be U+FFFD, so there's no need to wait. The joined full text is the same, only the split into deltas changes, but that violates the contract that "return values are exactly the same as the original", so confirm with the product side before changing it.

**5. With one event loop serving all streams, what else should you watch after the fix?** The CPU time per token directly determines how many streams one process can handle: the fixed version takes about 1.6 µs per token, so 3000 token/s uses less than 1% of a core. Going further, you could move decoding to the model-serving side (which already has the tokens and bytes), or merge several deltas into one SSE event every 20 ms to cut the fixed overhead per event. Spreading decoding across threads doesn't help much in CPython because of the GIL (global interpreter lock).

<details>
<summary>Verification code (runnable)</summary>

```python
import itertools
import random
import time

TOKENIZER_SRC = r'''
class ByteTokenizer:
    """Byte-level vocabulary: token id i always stands for the byte string pieces[i].

    A piece may end in the middle of a multi-byte UTF-8 character, and a few
    special ids (end of turn and the like) have empty pieces.
    """

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def token_bytes(self, token_id):
        return self.pieces[token_id]

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")
'''

GIVEN_SRC = r'''
class StreamDecoder:
    """Turns the token ids of one model reply into text deltas for the client.

        push(token_id) -> str  text that became final with this token ("" if none)
        finish() -> str        whatever is left, once the reply is over
        text() -> str          everything returned so far (saved with the conversation)

    The deltas, joined, are exactly tokenizer.decode(all ids). A delta never
    contains half a character: while the decoded text ends in U+FFFD the last
    token may have stopped in the middle of a multi-byte character, so we wait.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids = []
        self.emitted = ""
        self.done = False

    def push(self, token_id):
        if self.done:
            raise ValueError("push after finish")
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        delta = text[len(self.emitted):]
        self.emitted += delta
        return delta

    def finish(self):
        if self.done:
            return ""
        self.done = True
        delta = self.tok.decode(self.ids)[len(self.emitted):]
        self.emitted += delta
        return delta

    def text(self):
        return self.emitted
'''

HARNESS_SRC = r'''
import random
import time

WORDS = ["the", "model", "stream", "token", "reply", "text", "byte", "fast", "slow", "user", "and", "is"]
HANZI = "你好世界流式分词解码模型输出文本速度用户问题回答"
EMOJI = "😀🚀✨🎉"


def make_tokenizer():
    """All 256 single bytes; English words with and without a leading space; every Chinese
    character and emoji cut into two pieces; one special id (end of turn) with no bytes."""
    pieces = [bytes([b]) for b in range(256)]
    for w in WORDS:
        pieces += [w.encode(), (" " + w).encode()]
    for ch in HANZI + EMOJI:
        b = ch.encode()
        pieces += [b[:2], b[2:]]
    pieces.append(b"")
    return ByteTokenizer(pieces)


def encode(tok, text):
    """Greedy longest match over the UTF-8 bytes (a stand-in for the model's tokenizer)."""
    index = {}
    for i, p in enumerate(tok.pieces):
        if p:
            index.setdefault(p, i)
    longest = max(len(p) for p in index)
    data, ids, i = text.encode(), [], 0
    while i < len(data):
        for n in range(min(longest, len(data) - i), 0, -1):
            if data[i:i + n] in index:
                ids.append(index[data[i:i + n]])
                i += n
                break
    return ids


def make_reply(n_tokens, seed=5):
    """Token ids of a mixed English / Chinese / emoji reply, n_tokens long."""
    rng = random.Random(seed)
    tok = make_tokenizer()
    ids = []
    while len(ids) < n_tokens:
        r = rng.random()
        if r < 0.5:
            s = " " + rng.choice(WORDS)
        elif r < 0.9:
            s = "".join(rng.choice(HANZI) for _ in range(rng.randint(1, 6))) + rng.choice("，。")
        else:
            s = rng.choice(EMOJI)
        ids += encode(tok, s)
    return tok, ids[:n_tokens]


def bench(n_tokens=20_000, every=4_000):
    tok, ids = make_reply(n_tokens)
    dec = StreamDecoder(tok)
    t0 = time.perf_counter()
    for k, token_id in enumerate(ids, 1):
        dec.push(token_id)
        if k % every == 0:
            t1 = time.perf_counter()
            print(f"tokens {k - every:>6} to {k:<6} {every / (t1 - t0):9.0f} tokens/s")
            t0 = t1
    dec.finish()
'''

FixedDecoder = StreamDecoder          # Part 2, defined above in this page
FixedStop = StopStreamDecoder         # Part 3
StdlibDecoder = StdlibStreamDecoder   # follow-up 1

base = {}
exec(TOKENIZER_SRC, base)
ByteTokenizer = base["ByteTokenizer"]
given_ns = {}
exec(GIVEN_SRC, given_ns)
GivenDecoder = given_ns["StreamDecoder"]
harness = {"ByteTokenizer": ByteTokenizer, "StreamDecoder": GivenDecoder}
exec(HARNESS_SRC, harness)
make_tokenizer, encode, make_reply = harness["make_tokenizer"], harness["encode"], harness["make_reply"]
ALL = (GivenDecoder, FixedDecoder, StdlibDecoder)


def run(cls, tok, ids):
    """Every return value, and text() after every call (including a second finish)."""
    dec = cls(tok)
    log = [(dec.push(i), dec.text()) for i in ids]
    log.append((dec.finish(), dec.text()))
    log.append((dec.finish(), dec.text()))
    return log


# ---------- 1. statement examples, all versions ----------
EX = ByteTokenizer([b"h", b"i", b" ", b"\xe4\xbd", b"\xa0", b"\xe5\xa5\xbd", b"\xf0\x9f", b"\x98\x80", b"\xff", b""])
assert "你".encode() == b"\xe4\xbd\xa0" and "好".encode() == b"\xe5\xa5\xbd"
assert "😀".encode() == b"\xf0\x9f\x98\x80"
EX_IDS = [0, 1, 3, 4, 6, 7, 5, 8, 2, 9, 3]
EX_OUT = ["h", "i", "", "你", "", "😀", "好", "", "\ufffd ", "", ""]
for cls in ALL:
    dec = cls(EX)
    assert [dec.push(i) for i in EX_IDS] == EX_OUT, cls
    assert dec.finish() == "\ufffd"
    assert dec.text() == "hi你😀好\ufffd \ufffd" == EX.decode(EX_IDS)
    assert dec.finish() == ""
    try:
        dec.push(0)
        raise AssertionError("push after finish must raise")
    except ValueError:
        pass
    dec = cls(EX)
    assert [dec.push(3), dec.push(0)] == ["", "\ufffdh"]
# the maximal-subpart rule the statement relies on
assert b"\xe4\xbd".decode("utf-8", "replace") == "\ufffd" and b"\xff".decode("utf-8", "replace") == "\ufffd"

# wrong guesses from Part 1 change the output
assert "".join(EX.decode([i]) for i in EX_IDS) != EX.decode(EX_IDS)
assert EX.decode([3]) + EX.decode([4]) == "\ufffd\ufffd" and EX.decode([3, 4]) == "你"
assert b"".join(EX.pieces[i] for i in EX_IDS).decode("utf-8", errors="ignore") != EX.decode(EX_IDS)


# ---------- 2. same return values on random inputs ----------
PALETTE = [0x00, 0x41, 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf, 0xc0, 0xc1, 0xc2, 0xdf,
           0xe0, 0xe1, 0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf3, 0xf4, 0xf5, 0xff]
CHARS = "a 中é😀\ufffd"


def random_piece(rng):
    r = rng.random()
    if r < 0.1:
        return b""
    if r < 0.5:                       # a slice of real UTF-8, usually cut inside a character
        s = "".join(rng.choice(CHARS) for _ in range(rng.randint(1, 3))).encode()
        a = rng.randrange(len(s))
        return s[a:rng.randint(a + 1, len(s))]
    return bytes(rng.choice(PALETTE) for _ in range(rng.randint(1, 4)))


rng = random.Random(7)
for trial in range(3000):
    tok = ByteTokenizer([random_piece(rng) for _ in range(rng.randint(1, 8))])
    ids = [rng.randrange(len(tok.pieces)) for _ in range(rng.randint(0, 14))]
    logs = [run(cls, tok, ids) for cls in ALL]
    assert logs[0] == logs[1] == logs[2], (tok.pieces, ids)
    assert logs[0][-1][1] == tok.decode(ids)

# real text cut at random byte positions, with empty special tokens mixed in
for trial in range(400):
    text = "".join(rng.choice("ab 你好世界éß😀🚀✨\n") for _ in range(rng.randint(0, 40)))
    data = text.encode()
    k = rng.randint(0, max(0, min(20, len(data) - 1)))
    cuts = sorted(rng.sample(range(1, len(data)), k)) if k else []
    pieces = [data[a:b] for a, b in zip([0] + cuts, cuts + [len(data)])]
    tok = ByteTokenizer(pieces + [b""])
    ids = []
    for i in range(len(pieces)):
        ids += [len(pieces)] * (rng.random() < 0.2) + [i]
    logs = [run(cls, tok, ids) for cls in ALL]
    assert logs[0] == logs[1] == logs[2]
    assert logs[0][-1][1] == text
    assert all(text.startswith(t) for _, t in logs[0])    # never half a character

# a model-like reply
tok, ids = make_reply(600)
logs = [run(cls, tok, ids) for cls in ALL]
assert logs[0] == logs[1] == logs[2] and logs[0][-1][1] == tok.decode(ids)
assert sum(d == "" for d, _ in logs[0][:-2]) > 100          # the reply really splits characters


# ---------- 3. incomplete_tail: the cut never changes the decoded text ----------
def D(b):
    return b.decode("utf-8", errors="replace")


CONTINUATIONS = [b"", b"A", b"\x80", b"\xbf", b"\x80\x80", b"\xa0\x80", b"\x90\x80\x80",
                 b"\x8f\xbf\xbf", b"\xe4\xbd\xa0", b"\xff", b"\xc2"]
_completable = {}


def completable(tail):
    """Can some continuation bytes turn tail into exactly one valid character?"""
    if tail not in _completable:
        ok = False
        for k in range(1, 4):
            for c in itertools.product((0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf), repeat=k):
                try:
                    ok = ok or len((tail + bytes(c)).decode("utf-8")) == 1
                except UnicodeDecodeError:
                    pass
        _completable[tail] = ok
    return _completable[tail]


def check_tail(data):
    cut = incomplete_tail(data)
    tail = data[cut:]
    assert len(tail) <= 3
    assert D(data) == D(data[:cut]) + ("\ufffd" if tail else ""), data
    for c in CONTINUATIONS:
        assert D(data + c) == D(data[:cut]) + D(tail + c), (data, c)
    assert not tail or completable(tail), data


n_checked = 0
for length in (0, 1, 2):
    for t in itertools.product(range(256), repeat=length):
        check_tail(bytes(t))
        n_checked += 1
for t in itertools.product(PALETTE, repeat=3):
    check_tail(bytes(t))
    n_checked += 1
for t in itertools.product([0x41, 0x80, 0x8f, 0x9f, 0xa0, 0xbf, 0xc2, 0xe0, 0xed, 0xef, 0xf0, 0xf4, 0xf5], repeat=4):
    check_tail(bytes(t))
    n_checked += 1
assert n_checked > 100_000


# ---------- 4. deterministic work counters, the doubling law, each hot spot alone ----------
class CountingTokenizer(ByteTokenizer):
    """Counts token ids read: decode(ids) reads len(ids) of them, token_bytes reads one."""
    ids_read = 0

    def token_bytes(self, token_id):
        self.ids_read += 1
        return ByteTokenizer.token_bytes(self, token_id)

    def decode(self, ids):
        self.ids_read += len(ids)
        return ByteTokenizer.decode(self, ids)


def counting_emitted(cls):
    """Same class, but each new string stored in self.emitted adds its length to copied."""
    class Counted(cls):
        copied = 0

        @property
        def emitted(self):
            return self._emitted

        @emitted.setter
        def emitted(self, value):
            if value is not getattr(self, "_emitted", None):
                self.copied += len(value)
            self._emitted = value
    return Counted


class OnlyDecodeFixed(FixedDecoder):
    """Part 2's incremental decoding, but the returned text is still accumulated with +=."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.emitted = ""

    def push(self, token_id):
        delta = super().push(token_id)
        self.emitted += delta
        return delta

    def finish(self):
        delta = super().finish()
        self.emitted += delta
        return delta

    def text(self):
        return self.emitted


class OnlyConcatFixed(GivenDecoder):
    """The given decode-everything push, but the returned text is kept in a list."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.parts, self.n_out = [], 0

    def _emit(self, text):
        delta = text[self.n_out:]
        self.parts.append(delta)
        self.n_out += len(delta)
        return delta

    def push(self, token_id):
        if self.done:
            raise ValueError("push after finish")
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        return self._emit(text)

    def finish(self):
        if self.done:
            return ""
        self.done = True
        return self._emit(self.tok.decode(self.ids))

    def text(self):
        return "".join(self.parts)


# the partial fixes are correct too
rng = random.Random(8)
for trial in range(500):
    tok = ByteTokenizer([random_piece(rng) for _ in range(rng.randint(1, 8))])
    ids = [rng.randrange(len(tok.pieces)) for _ in range(rng.randint(0, 14))]
    want = run(GivenDecoder, tok, ids)
    assert run(OnlyDecodeFixed, tok, ids) == want and run(OnlyConcatFixed, tok, ids) == want


def measure(cls, n):
    _, ids = make_reply(n)
    tok = CountingTokenizer(make_tokenizer().pieces)
    dec = counting_emitted(cls)(tok)
    out = "".join([dec.push(i) for i in ids] + [dec.finish()])
    read, copied = tok.ids_read, dec.copied
    assert out == dec.text() == ByteTokenizer.decode(tok, ids)
    return read, copied, len(out)


SIZES = (1000, 2000, 4000)
given, fixed, only_decode, only_concat = (
    [measure(cls, n) for n in SIZES]
    for cls in (GivenDecoder, FixedDecoder, OnlyDecodeFixed, OnlyConcatFixed))
print("tokens  chars | given: ids read, chars copied | fixed: ids read, chars copied")
for n, g, f in zip(SIZES, given, fixed):
    print(f"{n:6} {g[2]:6} | {g[0]:10,} {g[1]:10,} | {f[0]:7,} {f[1]:3}")

assert [g[2] for g in given] == [983, 1844, 3793]
assert [g[0] for g in given] == [501_500, 2_003_000, 8_006_000]
assert all(g[0] == n * (n + 1) // 2 + n for n, g in zip(SIZES, given))
assert [g[1] for g in given] == [245_195, 954_795, 3_907_489]
assert [f[:2] for f in fixed] == [(1000, 0), (2000, 0), (4000, 0)]


def ratios(rows, j):
    return [b[j] / a[j] for a, b in zip(rows, rows[1:])]


assert all(3.5 < r < 4.5 for r in ratios(given, 0) + ratios(given, 1))
assert all(1.8 < r < 2.2 for r in ratios(fixed, 0))
# fixing only the decode: ids read grow x2, but copies still grow x4 (identical to the given version)
assert [r[0] for r in only_decode] == list(SIZES)
assert [r[1] for r in only_decode] == [g[1] for g in given]
# fixing only the accumulation: no copies, but ids read still grow x4
assert [r[0] for r in only_concat] == [g[0] for g in given] and all(r[1] == 0 for r in only_concat)

# the profile excerpt: generator calls = sum(k + 1 for k in 1..n) + (n + 1)
for n, calls in ((2000, 2_005_001), (4000, 8_010_001)):
    assert sum(k + 1 for k in range(1, n + 1)) + n + 1 == calls

# the ticket arithmetic in Part 1
rates = [7084, 2385, 1437, 1023, 795]
per_token = [1e6 / r for r in rates]
assert [round(p) for p in per_token] == [141, 419, 696, 978, 1258]
per_old = [p / mid for p, mid in zip(per_token, (2000, 6000, 10000, 14000, 18000))]
assert [round(x, 4) for x in per_old] == [0.0706, 0.0699, 0.0696, 0.0698, 0.0699]
assert round(3000 / 7084, 2) == 0.42 and round(3000 / 2385, 2) == 1.26
assert 3000 * 1.6e-6 < 0.01                                   # follow-up 5: under 1% of a core


# ---------- 5. timings: printed only, never asserted ----------
def timed(cls, n, every):
    harness["StreamDecoder"] = cls
    print(cls.__name__ if cls is not GivenDecoder else "given StreamDecoder", n, "tokens")
    harness["bench"](n, every)


timed(GivenDecoder, 4000, 1000)
timed(OnlyDecodeFixed, 100_000, 25_000)
timed(FixedDecoder, 100_000, 25_000)
harness["StreamDecoder"] = GivenDecoder


# ---------- 6. Part 3: stop strings ----------
def stop_ref(F, stops, final):
    """Straight from the definition: the text that must have been returned when the final text is F."""
    for p in range(1, len(F) + 1):
        starts = [p - len(s) for s in stops if len(s) <= p and F[p - len(s):p] == s]
        if starts:
            return F[:min(starts)], True
    if final:
        return F, False
    hold = 0
    for k in range(1, len(F) + 1):
        if any(len(s) > k and s.startswith(F[len(F) - k:]) for s in stops):
            hold = k
    return F[:len(F) - hold], False


try:
    FixedStop(EX, ["a", ""])
    raise AssertionError("empty stop string must raise")
except ValueError:
    pass

d = FixedStop(EX, ["i h", "你好"])
outs, flags = [], []
for i in [0, 1, 3, 4, 5, 2]:
    outs.append(d.push(i))
    flags.append(d.stopped)
assert outs == ["h", "", "", "i", "", ""] and flags == [False] * 4 + [True] * 2
assert d.finish() == "" and d.text() == "hi"
assert d.push(0) == "" and d.finish() == ""                 # stopped: no error, nothing more

d = FixedStop(EX, ["hi hi", "i h"])
assert [d.push(i) for i in [0, 1, 2, 0, 1]] == ["", "", "", "h", ""] and d.stopped and d.text() == "h"

d = FixedStop(EX, ["好", "你好"])
assert [d.push(i) for i in [3, 4, 5]] == ["", "", ""] and d.stopped and d.text() == ""

d = FixedStop(EX, ["i h"])
assert [d.push(0), d.push(1), d.finish()] == ["h", "", "i"] and not d.stopped
try:
    d.push(0)
    raise AssertionError("push after finish must raise when not stopped")
except ValueError:
    pass

d = FixedStop(EX, [])
assert [d.push(i) for i in EX_IDS] + [d.finish()] == EX_OUT + ["\ufffd"]

# the same examples, from the brute-force definition
for stops, seq, want in ((["i h", "你好"], [0, 1, 3, 4, 5, 2], ["h", "", "", "i", "", ""]),
                         (["hi hi", "i h"], [0, 1, 2, 0, 1], ["", "", "", "h", ""]),
                         (["好", "你好"], [3, 4, 5], ["", "", ""])):
    dec, F, prev, got = GivenDecoder(EX), "", "", []
    for i in seq:
        F += dec.push(i)
        cur = stop_ref(F, stops, False)[0]
        got.append(cur[len(prev):])
        prev = cur
    assert got == want

# random replies and stop strings, checked after every call
rng = random.Random(11)
for trial in range(1500):
    tok = ByteTokenizer([random_piece(rng) for _ in range(rng.randint(1, 8))])
    ids = [rng.randrange(len(tok.pieces)) for _ in range(rng.randint(0, 16))]
    full = tok.decode(ids)
    stops = []
    for _ in range(rng.randint(0, 3)):
        if full and rng.random() < 0.6:
            a = rng.randrange(len(full))
            stops.append(full[a:a + rng.randint(1, 4)])
        else:
            stops.append("".join(rng.choice("a 中\ufffd") for _ in range(rng.randint(1, 3))))
    ref_dec, d = GivenDecoder(tok), FixedStop(tok, stops)   # F comes from the given decoder
    F, got = "", ""
    for i in ids:
        F += ref_dec.push(i)
        got += d.push(i)
        assert (got, d.stopped) == stop_ref(F, stops, False), (tok.pieces, ids, stops)
        assert d.text() == got
    F += ref_dec.finish()
    got += d.finish()
    assert (got, d.stopped) == stop_ref(F, stops, True)
    assert got == stop_ref(full, stops, True)[0]


class CountingStop(StopStreamDecoder):
    scanned = 0
    calls = 0

    def _first_stop(self, window):
        self.scanned += len(window)
        self.calls += 1
        return super()._first_stop(window)


def stop_work(n):
    tok, ids = make_reply(n)
    d = CountingStop(tok, [" the end", "回答完毕"])        # never complete, often half-matched
    out = "".join([d.push(i) for i in ids] + [d.finish()])
    assert not d.stopped and out == tok.decode(ids)
    assert d.scanned <= len(out) + d.keep * d.calls
    return d.scanned


work = [stop_work(n) for n in (5000, 10000, 20000)]
assert all(1.8 < b / a < 2.2 for a, b in zip(work, work[1:])), work
print("Part 3 window chars scanned:", work)
print("all checks passed")
```

</details>

</details>
