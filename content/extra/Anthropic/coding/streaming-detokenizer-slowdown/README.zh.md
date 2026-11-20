# 流式解码在长回复里越来越卡

## 题目

你接手了模型服务前端里的流式解码组件 `StreamDecoder`（detokenizer，把 token id 变回文本）。模型每生成一个 token，前端就调用这条回复的 `push(token_id)`，把返回的文本片段（delta）通过 SSE（server-sent events）推给客户端；回复结束时调用 `finish()`；`text()` 的结果存进对话记录。一个前端进程用一个事件循环（event loop）服务所有并发的回复，所有解码都在这一个线程里做。

词表是字节级的（byte-level）：每个 token id 对应一段固定的字节。一个汉字（UTF-8 里 3 个字节）或一个 emoji（4 个字节）经常被拆在两个 token 里，模型也可能输出拼不成合法 UTF-8 的字节。所以最后一个 token 可能停在半个字符上，这时解码器要先等一等，不能把半个字符发给客户端。

这段代码的输出是对的，现有测试都能通过。问题是性能。下面是代码、复现脚本和工单。`ByteTokenizer` 是公共库，别的服务也在用，不能改；`StreamDecoder` 归你。

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

复现脚本（单路回复，只测解码，不经过网络）：

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

工单：

```text
标题：长回复越往后越卡，前端进程 CPU 打满
现象：高峰时一个前端进程同时服务约 60 路回复，每路模型约 50 token/s，合计约 3000 token/s。
      回复都不长时一切正常；长回复（几千 token 以上）一多，这个进程的 CPU 就到 100%，
      所有流的 token 间隔一起变长。GPU 那边吞吐正常。
复现：bench() 的输出
      tokens      0 to 4000        7084 tokens/s
      tokens   4000 to 8000        2385 tokens/s
      tokens   8000 to 12000       1437 tokens/s
      tokens  12000 to 16000       1023 tokens/s
      tokens  16000 to 20000        795 tokens/s
期望：解码一个 token 的耗时和回复已经多长无关；单路至少每秒 20 万 token。
```

（这些计时是在一台共享机器上测的，只用来描述症状；你自己的机器上数字会不同。）

行为约定（修改后必须保持）：

- `pieces` 是任意字节串：可以为空（结束符之类的特殊 token），可以不是合法 UTF-8。token id 都在合法范围内。
- 解码规则就是 `tokenizer.decode`：把字节拼起来，按 UTF-8 解码，`errors="replace"`。按 Python 的规则，每一段无法解码的最大片段（maximal subpart）变成一个 U+FFFD，例如被截断的 `e4 bd` 变成一个 U+FFFD，`ff` 也变成一个。
- 对任意 `pieces`、任意 token 序列，修改后每一次 `push` / `finish` 的返回值，以及任意时刻 `text()` 的值，都必须和原版完全相同，包括在什么时候返回空串。
- `finish()` 之后再 `push` 抛 `ValueError`；第二次 `finish()` 返回 `""`。

```text
pieces = [b"h", b"i", b" ", b"\xe4\xbd", b"\xa0", b"\xe5\xa5\xbd", b"\xf0\x9f", b"\x98\x80", b"\xff", b""]
#  id:      0     1     2     3            4        5                6            7            8        9
# "你" = e4 bd a0，"好" = e5 a5 bd，"😀" = f0 9f 98 80；ff 不会出现在合法 UTF-8 里

push(0) -> "h"         push(1) -> "i"
push(3) -> ""          # e4 bd 是一个三字节字符的前两个字节，解码结果以 U+FFFD 结尾，先等
push(4) -> "你"
push(6) -> ""          push(7) -> "😀"
push(5) -> "好"
push(8) -> ""          # ff 解码成 U+FFFD，同样先等（原版分不清“坏字节”和“半个字符”）
push(2) -> "\ufffd "   # U+FFFD 后面跟一个空格
push(9) -> ""          # 空 piece
push(3) -> ""
finish() -> "\ufffd"   # 回复停在半个字符上：e4 bd 变成一个 U+FFFD
text()   -> "hi你😀好\ufffd \ufffd"

另一条回复：push(3) -> ""   push(0) -> "\ufffdh"    # 半个字符后面直接来了 "h"
```

### Part 1 —— 时间花在哪里

先别改代码。说清楚你打算测什么、用什么工具、看到什么结果会让你得出什么结论，然后找出让“回复越长、每个 token 越慢”的原因。如果你发现不止一处值得修，分别说明每一处每个 token 的代价，以及这个代价随回复长度怎么变。

### Part 2 —— 修好它，不改变任何返回值

改写 `StreamDecoder`，接口不变，使每个 token 的工作量（摊还意义下）只和这个 token 的字节数有关，和回复已经多长无关。每次调用的返回值都必须和原版完全相同（见上面的约定）。可以使用 `ByteTokenizer` 的任何公开方法。

机器是共享的，计时噪声很大。请用确定性的计数（例如读了多少个 token、复制了多少字符）做倍增实验，证明原版的问题和你的修复：回复长度翻倍时，原版的计数大约变成 4 倍，你的版本大约变成 2 倍。

### Part 3 —— 停止串

产品要支持停止串（stop sequences）：调用方给出几个字符串，回复里一旦出现其中任何一个，就在它前面截断，停止串本身和后面的内容都不发给客户端，并通知模型停止生成。写一个 `StopStreamDecoder`：

```py
class StopStreamDecoder:
    def __init__(self, tokenizer, stops): ...   # stops: a list of non-empty strings (the list may be empty)
    def push(self, token_id): ...                # -> str
    def finish(self): ...                        # -> str
    def text(self): ...                          # everything returned so far
    # attribute stopped: bool, True once a stop string has appeared
```

设 $F$ 是 Part 2 的解码器到目前为止返回的文本，也就是已经确定的文本。

- “第一个”停止串，是在 $F$ 里**结束位置最靠前**的那次出现；结束位置相同的，取更长的那个（它开始得更早）。它一出现，`stopped` 就变成 `True`，已返回的文本恰好是 $F$ 在这次出现之前的部分；之后 `push` 和 `finish` 都只返回 `""`，不报错。
- 还没停止时，尽早返回：每次 `push` 之后，已返回的文本等于 $F$ 去掉“最长的、同时是某个停止串的真前缀（proper prefix）的后缀”。`finish()` 时把剩下的全部返回。
- 每次调用的工作量只能和这次新增的文本长度、停止串的个数和长度有关，不能和回复已经多长有关。
- 停止串里有空串时，构造函数抛 `ValueError`。`stops` 是空列表时，行为和 Part 2 的解码器一样。没有停止时，`finish()` 之后再 `push` 和 Part 2 一样抛 `ValueError`。

```text
pieces 同上，stops = ["i h", "你好"]
push(0) -> "h"
push(1) -> ""      # F = "hi"，"i" 可能是 "i h" 的开头，先留着
push(3) -> ""      # 半个字符，F 没有变
push(4) -> "i"     # F = "hi你"："i你" 不是任何停止串的开头，但 "你" 是 "你好" 的开头，留着
push(5) -> ""      # F = "hi你好"：出现 "你好"，stopped = True，"你好" 不返回
push(2) -> ""      finish() -> ""      text() -> "hi"

stops = ["hi hi", "i h"]，依次 push(0), push(1), push(2), push(0), push(1)
返回 "", "", "", "h", ""          # 第四个 token 后 F = "hi h"，"i h" 在这里结束，比 "hi hi" 早，截在它前面
stopped = True，text() -> "h"

stops = ["好", "你好"]：push(3) -> ""   push(4) -> ""   push(5) -> ""   # 两个同时结束，取更长的 "你好"
text() -> ""

stops = ["i h"]：push(0) -> "h"   push(1) -> ""   finish() -> "i"   stopped = False
```

## 参考解答

<details>
<summary>展开参考解答</summary>

### Part 1 —— 怎样定位

**第一步：把症状变成一个能比较的数字。** 把工单里的 tokens/s 换成每个 token 的平均耗时，再除以这一段所处的位置（这一段中间那个 token 前面有多少个 token）：

| 位置（token） | 每个 token 耗时 | 除以中间位置 |
|---|---|---|
| 0–4000 | 141 µs | 0.0706 µs |
| 4000–8000 | 419 µs | 0.0699 µs |
| 8000–12000 | 696 µs | 0.0696 µs |
| 12000–16000 | 978 µs | 0.0698 µs |
| 16000–20000 | 1258 µs | 0.0699 µs |

最后一列几乎是常数：处理第 $k$ 个 token 的耗时和 $k$ 成正比，大约是“前面每个 token 70 ns”。这说明每次 `push` 都在把之前所有的 token 碰一遍，整条回复的总工作量是 $O(n^2)$。这比“越来越慢”具体得多，已经告诉我们要找的是什么。

工单里的 CPU 打满也对得上：合计 3000 token/s，假设所有流都处在 0–4000 这一段，解码只占一个核的 42%；都到了 4000–8000 这一段，就需要 1.26 个核，一个线程已经做不完。事件循环是单线程的，一个 `push` 卡住，所有流都等。如果没有现成的复现脚本，第一步应该先用采样式的 profiler（例如 `py-spy`，可以直接挂到线上进程）确认 CPU 确实耗在解码上，而不是网络或序列化。

**第二步：profile，在两个规模上各做一次。**

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

我们机器上的一次输出（列删减了；`tottime` 是函数自身的时间，`cumtime` 包含它调用的函数）：

```text
2000 tokens
   ncalls  tottime  cumtime  function
  2005001    0.311    0.311  <genexpr>        (ByteTokenizer.decode 里的生成器表达式)
     2001    0.228    0.539  {method 'join' of 'bytes' objects}
     2001    0.006    0.006  {method 'decode' of 'bytes' objects}
     2000    0.003    0.550  push

4000 tokens
  8010001    1.241    1.241  <genexpr>
     4001    0.900    2.141  {method 'join' of 'bytes' objects}
     4001    0.022    0.022  {method 'decode' of 'bytes' objects}
     4000    0.007    2.175  push
```

几个要点：

- `ByteTokenizer.decode` 只被调用了 2001 次（每个 `push` 一次，`finish` 一次），次数没问题。问题在每次调用有多贵。
- 生成器表达式的调用次数 2,005,001 正好是 $\sum_{k=1}^{2000}(k+1) + 2001$：一个生成器每产出一个值算一次调用，结束时再算一次。也就是第 $k$ 次 `push` 把 $k$ 个 id 全部走了一遍。token 数翻倍，它和 `join` 的时间都变成约 4 倍。
- 真正的 UTF-8 解码（`bytes.decode`）只占 1%。解码本身不慢，是解码的数据太多。
- `push` 自身的时间几乎为零，所以 `self.emitted += delta` 在这里完全看不出来。记住这一点，后面会用到。

**第三步：对上代码。** `push` 里的 `text = self.tok.decode(self.ids)` 每来一个 token 就把整条回复从头解码一遍：生成 $k$ 个 piece、拼成一个 bytes、再解码成 str。写这行代码的人是为了正确处理“半个字符”，这确实简单可靠，但代价是每个 token $O(k)$。单元测试里的回复只有几十个 token，看不出来。

**第四步：用确定性的计数确认，不靠计时。** 不改原版的任何一行，在外面包两层：一个 `ByteTokenizer` 子类数解码器读了多少个 token（`decode(ids)` 算 `len(ids)` 个，`token_bytes` 算 1 个）；一个子类把 `emitted` 换成 property，每次给它赋一个新字符串，就把新字符串的长度加进“复制的字符数”。跑同一个回复生成器的 1,000 / 2,000 / 4,000 个 token（验证代码会算出并断言这些数）：

| token 数 | 文本长度 | 原版：读 token 次数 | 原版：复制进 `emitted` 的字符 | 修复后：读 token 次数 | 修复后：复制 |
|---|---|---|---|---|---|
| 1,000 | 983 | 501,500 | 245,195 | 1,000 | 0 |
| 2,000 | 1,844 | 2,003,000 | 954,795 | 2,000 | 0 |
| 4,000 | 3,793 | 8,006,000 | 3,907,489 | 4,000 | 0 |

原版读 token 的次数正好是 $n(n+1)/2 + n$，每翻倍约 4 倍。复制的字符数也是每翻倍约 4 倍：这是第二处平方开销。

**第五步：修一处，再量一次。** 只把解码改成增量的（Part 2 的做法），但保留 `self.emitted += delta`，再在 5 万 / 10 万 / 20 万 token 上 profile（以下是我们机器上的一次测量）：`push` 自身的时间是 0.164 → 0.481 → 1.549 秒，每翻倍约 3 倍，而它调用的函数都只翻倍；`bench(200_000, 40_000)` 从每秒 38 万 token 掉到 8.3 万。剩下的超线性部分在 `push` 自己的代码里，就是 `self.emitted += delta`：Python 的字符串不可变，拼接会新建一个字符串，把已输出的全部文本复制一遍。回复里有 emoji 时，整个字符串每个字符占 4 个字节（CPython 按最宽的字符统一存储），复制更贵。

结论是两处热点（hot spot）：

| 位置 | 每个 token 的代价 | 随回复变长 | 什么时候是主要矛盾 |
|---|---|---|---|
| `push` 里的 `self.tok.decode(self.ids)` | 把前面所有 token 重新拼接、解码，$O(k)$，每个旧 token 约 70 ns | 变大 | 一直是，工单的原因 |
| `self.emitted += delta`（`finish` 里也有） | 复制已返回的全部文本，$O(\text{已输出字符})$，但只是内存复制 | 变大 | 修完上一处之后，在十万 token 级的长回复里 |

**常见的错误判断：**

- “只解码新 token 的字节就行”（`decode([token_id])`）。快是快了，但被拆开的字符会变成乱码：`"你"` 拆成 `e4 bd` 和 `a0` 两个 token，分别解码得到两个 U+FFFD。输出变了，不能接受。
- “用 `errors="ignore"`，就不会以 U+FFFD 结尾，也就不用等了。”这会把坏字节直接丢掉，输出变了。而且它也没有减少每次解码的数据量。
- “UTF-8 解码慢，换个更快的库。”profile 里 `bytes.decode` 只占 1%。慢在调用了多少次、每次处理多少数据，不在单次速度。
- “是 SSE 或网络慢。”复现脚本根本不经过网络，照样变慢。
- “只解码最近 32 个 token，取差值。”思路是对的，但窗口的起点可能正好落在一个字符的中间，开头会多出 U+FFFD，要小心处理才能和原版逐次相同（见追问 2）。对字节级词表，下面的做法更直接。
- 只修第一处就收工。中等长度的回复确实好了，但十万 token 级的回复仍然越来越慢，第二处平方开销还在。

### Part 2 —— 修复

思路有三点。

1. **只解码新来的字节。** 记一个 `tail`：上一个 token 留下的、还没凑齐的半个字符（最多 3 个字节）。新 token 来了，把 `tail + 新字节` 里“已经能确定”的部分解码掉，剩下没凑齐的尾巴继续留着。切点这样找：从末尾往前跳过最多 3 个续字节（continuation byte，`80`–`BF`），看到的那个字节如果是首字节（lead byte），并且它后面的字节还不够、而且到目前为止是合法的，尾巴就从它开始；否则没有尾巴。UTF-8 的规则：`C2`–`DF` 后面跟 1 个续字节，`E0`–`EF` 跟 2 个，`F0`–`F4` 跟 3 个；`C0`、`C1`、`F5`–`FF` 永远不合法；另外 `E0` 的第二个字节必须在 `A0`–`BF`，`ED` 在 `80`–`9F`（排除代理区，surrogates），`F0` 在 `90`–`BF`，`F4` 在 `80`–`8F`。
2. **照搬原版的等待规则。** 原版在“全部文本以 U+FFFD 结尾”时等待。切点要么在末尾（最后一个字符已经完整或者已经确定非法），要么落在一个首字节上，而解码器遇到首字节一定是从一个新字符开始，所以切点之前的字节无论后面接什么，解码结果都不变。于是全部文本 = 已返回的文本 + 已解码但还没返回的文本（`held`）+ 尾巴的解码结果；而一个没凑齐但合法的尾巴单独解码正好是一个 U+FFFD。所以原版等待，当且仅当 `tail` 非空，或者 `held` 的最后一段以 U+FFFD 结尾（已返回的文本本身不会以 U+FFFD 结尾，否则原版当时就不会返回）。不等待时，原版返回的 delta 正好是 `held` 拼起来。
3. **不累加字符串。** 返回过的片段放进列表，`text()` 时再拼。

复杂度：每个 `push` 处理至多 3 + `len(piece)` 个字节，加上常数次列表操作。`held` 里的每一段只在交出时被拼接一次，所以总工作量是 $O(\text{总字节数})$，每个 token 摊还 $O(\text{这个 token 的字节数})$。`text()` 是 $O(\text{文本长度})$，原版存对话记录时只调用一次。

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

上面计数表里“修复后”的两列就是这个版本：每个 token 只读一次，没有累加复制。验证代码还检查了：

- 在随机的 `pieces`（含空串、非法字节、`E0`/`ED`/`F0`/`F4` 开头的边界情况、真正的 U+FFFD 字符）和随机 token 序列上，原版和修复版每一次调用的返回值、每一时刻的 `text()` 都相同；把真实文本在随机字节位置（包括字符中间）切成 token 也一样；
- 对所有 0–2 字节的串和大量 3–4 字节的串，`incomplete_tail` 切出的前半段，无论后面接什么，解码结果都不变；留下的尾巴最多 3 个字节，并且能被补成一个合法字符；
- 分别只修一处，确认两处各自都重要：只修解码，复制量仍按约 4 倍增长；只修累加，读 token 次数仍按约 4 倍增长。

我们机器上修复版的 `bench()` 每秒约 61 万 token，从回复开头到 20 万 token 基本不变；这只作参考，验证代码不对计时做断言。

### Part 3 —— 停止串

最直接的写法是每次在“到目前为止的全部文本”里找停止串。这样每次调用都是 $O(\text{回复长度})$，刚修掉的平方开销又回来了。

正确的做法是只看一个小窗口：`hold`（还没返回的、可能是某个停止串开头的那段，不超过 $M-1$ 个字符，$M$ 是最长停止串的长度）加上这次新确定的文本。

- **还没找到的停止串，一定开始于窗口之内。** `hold` 正好是“$F$ 的最长的、是某个停止串真前缀的后缀”。如果某次出现开始于窗口之前，那么它和旧文本重叠的部分就是旧 $F$ 的一个更长的、同样是停止串真前缀的后缀，矛盾。同理，新的 `hold` 也一定落在窗口之内。
- **找结束最早的那次。** 对每个停止串 `s`，`window.find(s)` 给出它最左边的出现，也就是它结束最早的出现；在所有停止串里取（结束位置，开始位置）最小的那个。
- **工作量。** 每次调用对每个停止串在不超过 $M-1+\text{新文本长度}$ 的窗口里找一次，再检查至多 $M-1$ 个后缀，和回复长度无关。

`stopped` 变成 `True` 时，调用方应该通知模型停止生成。

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

停止串是按“已经确定的文本”比较的，Part 2 的解码器从不交出半个字符，所以一个被拆在两个 token 里的汉字也能正确匹配。验证代码用一个只依赖定义的暴力版本（每次在完整的 $F$ 上逐个结束位置检查）逐次对比，并用计数确认每次扫描的窗口总长不超过“文本长度 + $(M-1)\times$ 调用次数”，回复长度翻倍时约变成 2 倍。

### 追问

**1. 能不能直接用标准库？** 可以。`codecs.getincrementaldecoder("utf-8")(errors="replace")` 就是增量的 UTF-8 解码器，它自己保存没凑齐的字节，`getstate()[0]` 是这些字节。等待规则换成“解码器里还有字节，或 `held` 以 U+FFFD 结尾”：

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

有一个细节：CPython 的增量解码器遇到 `ED A0`–`ED BF`（代理区，不合法）时会先把它缓存起来，而 `incomplete_tail` 会立刻把它解码成两个 U+FFFD。两种情况下原版都在等（全部文本以 U+FFFD 结尾），所以返回值仍然相同；验证代码在含 `ED` 的随机输入上对比了三个版本。面试时能说出“标准库有增量解码器”就够了，手写切点是为了说明原理。

**2. 如果 tokenizer 的 decode 不是简单的字节拼接呢？** 例如 SentencePiece 会把开头的“▁”变成空格、解码时去掉句首空格，有的还有归一化，单个 token 的解码和整段解码不一致，就不能逐 token 解码。常见做法是只保留一个短窗口：记住 `prefix_offset` 和 `read_offset`，每次解码 `ids[prefix_offset:]` 和 `ids[prefix_offset:read_offset]`，两者的差（不以 U+FFFD 结尾时）就是新文本，然后把窗口往前挪。窗口长度有上界，每个 token 的工作量就是常数。vLLM 等推理框架的增量解码就是这个思路。

**3. 如果服务端每个 token 都调用一次 `text()` 来保存进度呢？** 修复版的 `text()` 每次都要把所有片段拼一遍，$O(n)$，平方开销又回来了。原版的 `text()` 虽然是 $O(1)$，但那是因为它在 `push` 里已经付过复制的钱。只要调用方每个 token 都要拿一份完整的全文（比如整段写进数据库），总代价就至少是所有前缀长度之和，怎么实现都是平方级。要改的是调用方：把每个 delta 追加到存储里，或者按时间间隔（例如每秒一次）保存，或者只在回复结束时保存。

**4. 模型输出一长串非法字节时，修复版会怎样？** 全部文本一直以 U+FFFD 结尾，原版一直等，修复版也一直等，`held` 越来越长，直到出现一个合法字符才一起交出。时间仍是线性的（每段只拼一次），但客户端会有一段时间什么都看不到。更好的规则是“只在 `tail` 非空时等”：坏字节已经确定是 U+FFFD，没必要等。拼起来的全文不变，只是 delta 的切分变了，这违反了“返回值和原版完全相同”的约定，要和产品确认后再改。

**5. 一个事件循环服务所有流，修完以后还要注意什么？** 每个 token 的 CPU 时间直接决定一个进程能撑多少路流：修复版每个 token 约 1.6 µs，3000 token/s 只占不到 1% 的核。进一步可以把解码挪到模型服务那一侧（那边本来就有 token 和字节），或者每隔 20 ms 把多个 delta 合成一个 SSE 事件，减少每个事件的固定开销。用多线程分担解码在 CPython 里帮助不大，因为有 GIL（全局解释器锁）。

<details>
<summary>验证代码（可运行）</summary>

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
