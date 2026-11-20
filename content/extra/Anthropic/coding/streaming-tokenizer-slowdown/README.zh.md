# 流式分词器在长文档上越跑越慢

## 题目

你接手了一个文档入库服务里的流式分词组件 `StreamTokenizer`。用户上传文档时，网络线程把收到的文本按块写进来（`write`）；下游的批处理线程每次取最多 256 个 token，凑成一批送去做嵌入（`read`）。分词规则是贪心最长匹配（greedy longest match）：在当前位置取词表里能匹配上的最长词条；一个都匹配不上时，把一个字符输出成 `unk_id`。文本可以在任意位置被切成块，包括一个多字符词条的中间，所以组件要在“后面的文本还可能拼出更长的词条”时先等一等。

这段代码的输出是对的，现有测试都能通过。问题是性能。下面是代码、复现脚本和工单。

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

复现脚本。线上网络线程和批处理线程是两个线程；脚本把它们交替执行：写入 4 KB，取一批，再写入 4 KB……上传结束后关闭并取完。网络比下游快，这一点和线上一致。

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

工单原文：

> 文档越长，分词越慢。同一台机器上跑 `bench()`，文本是同一种，只是长度不同：
>
> ```text
>   100000 chars   40044 tokens    0.78 s     51035 tokens/s
>   800000 chars  320300 tokens    9.13 s     35095 tokens/s
>  1600000 chars  640849 tokens   24.61 s     26039 tokens/s
>  3200000 chars 1282719 tokens   75.13 s     17074 tokens/s
> ```
>
> 下游的嵌入服务每秒至少要 20 万个 token，用户上传的文档常有几 MB。

（这些计时是在一台共享机器上测的，只用来描述症状；你自己的机器上数字会不同。）

行为约定（修改后必须保持）：词表是 `dict[str, int]`，至少一条，键非空、互不相同，最长键长 $L \le 16$。`chunk` 可以是空串。`read(max_tokens)` 中 `max_tokens >= 0`，返回的 id 一旦交出就不会再变；某个位置的 token 只有在“流已关闭”或“未分词的字符至少有 $L$ 个”时才算确定。`close()` 之后再 `write` 抛 `ValueError`；空词表抛 `ValueError`。`pending()` 返回已写入、还没变成 token 的字符数。

```text
vocab = {"t": 1, "h": 2, "e": 3, "r": 4, " ": 5, "th": 6, "the": 7, "there": 8}, unk_id = 0, L = 5

tok.write("the")     tok.pending() -> 3   tok.read(10) -> []         # 不足 5 个字符：后面可能来 "re"，拼成 "there"
tok.write("re is")   tok.read(10) -> [8]                             # "there"；剩下 " is" 不足 5 个字符，先不动
tok.close()          tok.read(10) -> [5, 0, 0]                       # " "；"i" 和 "s" 不在词表里，各是一个 unk
                     tok.read(10) -> []

同一段文本换一种切法、每次只取 1 个：
write("t") read(1) -> []    write("here i") read(1) -> [8]    write("s") read(1) -> []
close()    read(1) -> [5]   read(1) -> [0]   read(1) -> [0]   read(1) -> []
```

### Part 1 —— 时间花在哪里

先别改代码。说清楚你打算测什么、用什么工具、看到什么结果会让你得出什么结论，然后找出让“文档越长、每个 token 越慢”的原因。如果你发现不止一处值得修，分别说明每一处每个 token 的代价，以及这个代价随文档长度怎么变。

### Part 2 —— 修好它，不改变任何输出

改写 `StreamTokenizer`，接口不变，使处理一份文档的总工作量随文本长度线性增长，每个 token 的工作量只和词表有关、和已经写入了多少文本无关。对任意输入、任意切块方式、任意 `write` / `read` / `close` 调用序列，每次 `read` 返回的列表和 `pending()` 的值都必须和原版完全相同。

机器是共享的，计时噪声很大。请用确定性的计数（例如复制了多少字符、查了多少次词表）做倍增实验，证明原版的问题和你的修复：文本长度翻倍时，原版的计数大约变成 4 倍，你的版本大约变成 2 倍。

### Part 3 —— 两个线程同时用，并限制积压

现在网络线程和批处理线程真正并发地调用它。写一个 `BoundedStreamTokenizer(vocab, unk_id, max_pending)`，同一时刻只有一个写线程和一个读线程：

```py
class BoundedStreamTokenizer:
    def __init__(self, vocab, unk_id, max_pending): ...
    def write(self, chunk): ...      # blocks while pending() >= max_pending, then appends the whole chunk
    def close(self): ...
    def read(self, max_tokens): ...  # blocks until at least one token is final; returns [] only
                                     # once the stream is closed and every token has been read
    def pending(self): ...
```

- `write` 返回后，积压（`pending()`）不超过 `max_pending - 1 + len(chunk)`，所以内存有上界。
- `read(max_tokens)` 要求 `max_tokens >= 1`，否则抛 `ValueError`。
- `max_pending` 小于词表最长键长 $L$ 时，构造函数抛 `ValueError`。请解释为什么需要这条下限。
- 读到的 id 序列必须和单线程整段分词的结果一样，无论线程怎样交错。

## 参考解答

<details>
<summary>展开参考解答</summary>

### Part 1 —— 怎样定位

**第一步：把症状变成一个能比较的数字。** 工单里 tokens/s 随文档变长而下降。换成“每个 token 的平均耗时”更直观：100 KB 时约 19.5 µs，3.2 MB 时约 58.6 µs。如果每个 token 的代价是常数，tokens/s 应该是平的。它不平，说明某段每个 token 都要做的工作，和文档长度（或者和它相关的某个量）成正比。从 800 KB 起，长度每翻倍，总时间涨 2.7 到 3.1 倍，介于线性（2 倍）和平方（4 倍）之间：这通常是“一个线性的大常数项 + 一个平方项”，平方项随规模变大逐渐占上风。

**第二步：profile，但要在两个规模上各做一次。** 用 `cProfile` 跑复现脚本：

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

我们机器上的一次输出（只保留前三行，列也删减了；`tottime` 是函数自身的时间，不含它调用的函数）：

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

只看一次 profile，第一名是 `_longest_match` 和 `startswith`，占了九成以上。`startswith` 的调用次数 5,886,468 正好等于 40,044 × 147：每个 token 都把词表的 147 条全试一遍。这是真实的开销，值得修，但它对每个 token 是常数，解释不了“越长越慢”。

把两次放在一起比：token 数变成 4 倍，`_longest_match` 的自身时间也是约 4 倍（1.286 → 5.110），而 `read` 的自身时间变成了约 10.8 倍（0.091 → 0.984）。增长快于线性的那部分在 `read` 自己的代码里，不在它调用的函数里。

**第三步：在 `read` 里找和长度成正比的那一行。** `read` 自己只做三件事：调 `_ready()`、`out.append`、`self.buf = self.buf[n:]`。Python 的字符串不可变，切片会新建一个字符串，把剩下的字符全部复制一遍。所以每取一个 token，就把整个积压复制一次。

积压有多大？每轮网络写入 4096 个字符，下游取 256 个 token，平均每个 token 约 2.5 个字符，也就是只取走约 640 个字符，每块大约 84% 留在缓冲区里。上传结束时，缓冲区里差不多是整份文档。之后清空它的每一个 token 都要复制 $O(N)$ 个字符，总共 $O(N^2)$。单元测试和小文档里积压很小，所以看不出来。

`cProfile` 只按函数统计，切片不是函数调用，它的时间算在 `read` 的 `tottime` 里，指不出是哪一行。要看到行，可以用第三方的 `line_profiler`（`kernprof -l -v`）；更稳的办法是下一步的计数。

**第四步：用确定性的计数确认，不靠计时。** 不改原版的任何一行，在外面包两层：一个 `dict` 子类数词表被访问了多少次（遍历出的每个键、每次 `get` / `[]` / `in` 各算一次）；一个子类把 `buf` 换成 property，每次给 `self.buf` 赋新字符串，就把它的长度加进“复制的字符数”。用和线上同样“写得比读得快”的节奏，但把块缩小到 256 个字符、每批 16 个 token，跑 4,000 / 8,000 / 16,000 个字符（验证代码会算出并断言这些数）：

| 文本长度 | token 数 | 原版：复制的字符 | 原版：每 token 查词表 | 修复后：复制的字符 | 修复后：每 token 查词表 |
|---|---|---|---|---|---|
| 4,000 | 1,558 | 2,714,273 | 147.9 | 4,004 | 6.4 |
| 8,000 | 3,148 | 10,769,510 | 147.9 | 8,004 | 6.5 |
| 16,000 | 6,366 | 43,349,443 | 147.9 | 16,009 | 6.5 |

原版复制的字符数每翻倍变成约 4 倍，是平方增长；每个 token 查词表的次数是常数 147.9，是线性增长但常数很大。

结论是两处热点（hot spot），性质不同：

| 位置 | 每个 token 的代价 | 随文档变长 | 什么时候是主要矛盾 |
|---|---|---|---|
| `read` 里的 `self.buf = self.buf[n:]` | 复制整个积压，$O(\text{积压})$ | 变大 | 大文档（工单的原因） |
| `_longest_match` 遍历整个词表 | $\lvert V\rvert$ 次 `startswith` | 不变 | 小文档（profile 第一名） |
| `write` 里的 `self.buf += chunk` | 每块复制一次积压 | 变大 | 块很小时；这个负载下次要 |

**常见的错误判断：**

- 看到 profile 第一名就去改 `_longest_match`（比如换成 trie）。小文档快了约 7 倍，大文档却只好了一点：只修这一处，我们机器上的一次测量是 100 KB 约 36.6 万 token/s，1.6 MB 掉到约 4.7 万（原版 2.6 万），3.2 MB 约 2.4 万（原版 1.7 万）。离 20 万还很远，而且仍是越长越慢，增长规律没变。
- 认定 `self.buf += chunk` 是元凶。它确实也复制整个积压，但每 4 KB 才一次；切片是每个 token 一次，次数多约 1,600 倍。不过它和切片是同一种病，修的时候一起处理。
- “CPython 对 `s += t` 有优化，是摊还 O(1) 的。”这是 CPython 的实现细节，只在左边是局部变量、而且这个字符串没有别的引用时才生效。`self.buf` 是属性，实例字典里还握着一个引用，所以 `self.buf += chunk` 每次都完整复制。
- “Python 慢，换 C++ 写。”C++ 里 `std::string::erase(0, n)` 一样要把剩下的内容整体搬移，还是平方级；问题在算法，不在语言。
- “把 batch 调大。”切片是每个 token 一次，每次复制整个积压；积压由网络和下游的速度差决定，不由每批多少个决定，只要下游比网络慢，复制总量仍是平方级。（在复现脚本里把 batch 调到下游比网络快，复制量确实会变成线性，但那等于假设下游变快了，不是线上能做的修复。）
- 怀疑 GC 或内存分配器。计数已经说明，工作量本身就是平方级的。

### Part 2 —— 修复

思路有三点。

1. **移动下标，不移动文本。** 用 `self.pos` 记下一个 token 从 `self.buf` 的哪里开始，取一个 token 只做 `self.pos += n`，不复制。
2. **新块先排队，需要时才拼。** `write` 只把块追加到一个 `deque` 里，$O(1)$。只有当 `buf` 在 `pos` 之后剩下不足 $L$ 个字符、又有新块在排队时，才拼一次：`buf[pos:] + "".join(incoming)`。每个写入的字符只被拼进来一次，旧的残余每次不足 $L$ 个，拼接次数不超过 `write` 次数，所以总复制量是 $O(N + W L)$，$W$ 是写入次数，$W \le N$（空块不排队）。
3. **按长度查表，不遍历词表。** 预先算出词表里出现过的所有键长，从长到短，每个长度切一段去 `dict` 里查一次。最多 $D \le L$ 次查表，每次哈希不超过 $L$ 个字符，所以每个 token 是 $O(L^2)$，和积压、和 $\lvert V\rvert$ 都无关。

为什么输出完全一样：判断“当前 token 是否确定”的条件没变（已关闭，或未分词字符至少 $L$ 个；需要的时候先把排队的块拼进来，所以看到的字符数就是 `pending()`）；选中的词条也一样（长度相同的两个不同键不可能同时是同一位置的前缀，最长的那个是唯一的）。

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

复杂度：每个 token $O(L^2)$ 次字符操作，复制总量 $O(N + WL)$，一份 $N$ 个字符的文档总共 $O(NL^2)$，对固定词表是线性的。内存仍是 $O(\text{积压})$：网络比下游快时，积压还是会涨到整份文档，这留给 Part 3。

上面表格里“修复后”的两列就是这个版本的计数：复制的字符数每翻倍变成 2 倍（基本就是 $N$ 本身），每个 token 查词表约 6.5 次。验证代码还分别只修一处，确认两处各自都重要：只改查表，复制量仍按 4 倍增长；只改复制，每个 token 仍查 147.9 次词表。我们机器上 `bench()` 的结果是每秒约 50 万 token，从 100 KB 到 3.2 MB 基本不变（3.2 MB 约 2.6 秒）；这只作参考，验证代码不对计时做断言。

### Part 3 —— 线程安全与背压

用一把锁保护 Part 2 的对象，再用一个条件变量（condition variable）让两边等待：

- 写线程在积压 `>= max_pending` 时等待；`write` 返回前积压至多是 `max_pending - 1 + len(chunk)`。这就是背压（backpressure）：下游慢，网络线程就慢下来，不再无限堆积。
- 读线程在“没有确定的 token、流也没结束”时等待。
- 每次状态变化（写入、读走、关闭）都 `notify_all()`。在一写一读、`max_pending >= L` 的设定下，两边不会同时在等（见下一段），所以 `notify()` 其实也够；但两种等待者共用一个条件变量，一旦有多个读线程或写线程，`notify()` 可能只叫醒了同类的线程，另一类就永远睡下去了；用 `notify_all()`，或者用同一把锁上的两个条件变量，都能避免。

**为什么 `max_pending >= L`：** 假设 `max_pending < L`，积压恰好是 `max_pending` 个字符。写线程看到积压已满，等待；读线程看到积压不足 $L$ 个，当前 token 还不确定（后面可能拼出更长的键），也等待。两边互相等，死锁。有了 `max_pending >= L`，写线程等待时积压至少 $L$ 个，读线程一定能取走至少一个 token，然后叫醒写线程；读线程等待时积压不足 $L \le$ `max_pending`，写线程不会被挡住。

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

分词在锁内进行。只有一个读线程时这没有问题：写线程只在积压满时才需要等，而那时读线程正在干活。若要多个读线程并行分词，就得把文本按“确定的边界”切开分给它们，复杂得多，而且 token 的顺序要另外保证。

### 追问

**1. 如果客户端每次只发 16 个字符，原版里哪一处最慢？修复版怎么样？** 原版里 `self.buf += chunk` 每次写入都复制整个积压，块越小次数越多，和切片一样变成平方级的主要成本。修复版里 `write` 只是 `deque.append`，拼接只在缓冲区快用完时发生，每个字符只拼一次，所以和块大小无关，仍是线性的。

**2. 只在每次 `read` 结束时把用掉的前缀切掉一次（`self.buf = self.buf[self.pos:]`），够不够？** 不够。这样每次调用复制一次积压，比每个 token 一次少了约 256 倍，但调用次数仍和文档长度成正比，总量还是平方级，只是常数小了。验证代码里这个版本的复制量每翻倍仍是约 4 倍。要么像 Part 2 那样只在剩余不足 $L$ 时才拼，要么在“已用前缀超过一半”时才压缩一次（摊还 $O(1)$）。

**3. 词表换成 10 万条、最长 50 个字符，修复版还行吗？** 按长度查表最多 $D$ 次，每次哈希最多 50 个字符，每个 token 最坏约 $50^2$ 次字符操作，能用但不理想。更好的是 trie：从当前位置往下走，走不动就停，每个 token 最多看 $L$ 个字符，并在路上记住最后一个词尾。如何写 trie 见 [最长匹配分词器](#/anthropic/coding/longest-match-tokenizer)；这道题的重点是找到流式包装层里的瓶颈，匹配器本身不是重点。

**4. 怎样防止这类问题再回来？** 在测试里加一个不看时间的“增长测试”：用计数包装器跑 $n$ 和 $2n$ 两个规模，断言计数之比小于 2.5。它在任何机器上都稳定，比基准测试的阈值靠谱。线上可以把“每个 token 的平均处理时间”按文档大小分桶做监控，平方问题会表现为大文档桶的值明显更高。

**5. Part 3 里如果有多个写线程呢？** 对同一个文档流，多个写线程同时写，块的先后顺序就不确定，文本本身就乱了，这不是锁能解决的。正确做法是每个文档一个 tokenizer，或者先由一个线程按序号把块排好再写入。

**6. 为什么一开始的单元测试没发现问题？** 测试里文本很短，或者写一点读一点，积压一直很小，切片复制的东西很少。这个平方项只有在“生产比消费快”时才出现。性能问题常常依赖负载形状，所以复现时要尽量用线上的节奏（这里是“写 4 KB，读 256 个 token”）。

<details>
<summary>验证代码（可运行）</summary>

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