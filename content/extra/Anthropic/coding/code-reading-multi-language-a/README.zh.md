# 读代码：三段代码，三种语言（A 卷：C / JavaScript / SQL）

## 题目

这一轮不写代码，只读代码。面试官在共享文档里依次贴出三段代码，每段换一种语言，每段大约十分钟。
代码不能运行，你要一边读一边讲出来。三段代码来自同一个虚构的视频模型训练项目，互相独立。

每一段都按下面五个问题来答：

1. 这段代码想做什么？
2. 对给出的输入，它实际输出什么？（要具体到每一行、每一个数。）
3. 哪里有错，哪里有风险？
4. 你会怎么改？
5. 如果作者在场，你想问他什么？

答完这五问，面试官一般会顺着你的回答再问几句，比如换一个输入会怎样、换一个平台会怎样。
这些追问和它们的答案都放在参考解答里，先自己把五问答完再看。

运行环境：C 用 gcc，x86-64 Linux，`int` 是 32 位；JavaScript 用 Node.js（16 或更新）；SQL 用 SQLite。
JavaScript 那段里的定时器按延迟从短到长依次触发，没有别的东西会拖慢它们。

### Part 1 —— C：一帧灰度图的统计量

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

### Part 2 —— JavaScript：批量取缩略图，失败重试

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

### Part 3 —— SQL：每周实验报告

表结构和数据：

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

生成周报的查询：

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

## 参考解答

<details>
<summary>展开参考解答</summary>

三段代码各有一类“换一种语言就变样”的语义：C 里是整数类型（`char` 有没有符号、整数除法、溢出）；
JavaScript 里是异步的执行顺序和“假值”；SQL 里是 `NULL` 的三值逻辑。
读的时候先用一句话说清楚意图，再拿样例输入一行一行算，最后才谈风险和改法。
下面所有“观察到的输出”都是在这台机器上实际跑出来的：gcc 11.4.0（x86-64 Linux）、Node v24.22.2、SQLite 3.53.1。

### Part 1

**想做什么。** 对一帧 8 位灰度图算三个数：平均亮度；亮度不低于阈值的像素占百分之几；最亮和最暗像素的差。

**实际输出。** 样例的 6 个像素是 10、20、30、200、250、90，阈值 128。
正确答案应该是 `mean=100`（600 ÷ 6）、`bright=33%`（6 个里有 2 个）、`contrast=240`（250 − 10）。观察到的输出是：

```text
mean=14 bright=0% contrast=146
```

三个数全错。一个一个看：

- `pix` 的类型是 `const char *`。在 x86-64 Linux 上，gcc 的 `char` 是有符号的，范围是 −128 到 127。
  字节 200 和 250 被读成 −56 和 −6。于是总和是 10 + 20 + 30 − 56 − 6 + 90 = 88，88 ÷ 6 取整得 14。
- `pix[i] >= threshold`：读出来最大的值只有 90，没有一个不低于 128，所以 `bright` 是 0。
- `lo` 和 `hi` 也是 `char`：最小 −56，最大 90，差是 146。

**面试官接着问：这段代码在所有平台上输出都一样吗？** 不一样。`char` 有没有符号，C 标准交给各个实现自己定（implementation-defined）。
ARM 上的 Linux 默认是无符号的。这里没有 ARM 机器，用 gcc 的 `-funsigned-char` 选项来模拟，观察到：

```text
mean=100 bright=0% contrast=240
```

`mean` 和 `contrast` 对了，`bright` 还是 0。原因在 `bright / n * 100`：先做整数除法，2 ÷ 6 得 0，再乘 100 还是 0。
这个式子只有在所有像素都亮的时候才得 100，其余情况都是 0。

**面试官接着问：输入换成一帧 8K 图（7680×4320），或者一帧空图（`width * height` 等于 0），会怎样？**
两种输入都会走到未定义行为，见下面问题清单的第 3 条和第 4 条。

**问题清单。**

1. **`char` 的符号。** 同一份源码在两种平台上结果不同。`-Wall -Wextra` 一条警告都没有（加上 `-Wconversion` 也没有）。
   改法：字节数据一律用 `uint8_t`（或 `unsigned char`）。
2. **先除后乘。** `bright / n * 100` 应该写成 `bright * 100 / n`。
3. **`int sum` 会溢出。** 每个像素最大 255，`int` 最大 2147483647，所以超过 8421504 个像素就可能溢出。
   4K 全白帧（3840×2160）的总和是 2115072000，刚好装得下；8K 帧（33177600 个像素）装不下。
   有符号整数溢出在 C 里是未定义行为（undefined behavior，简称 UB）：标准不规定会发生什么。
   观察：给一帧 8K、像素全是 127 的图，UBSan 报 `signed integer overflow`，程序接着打印 `mean=-2`。
   这个 −2 只是这一次看到的结果，不能当成规律。`int n = width * height` 也一样，宽高再大就轮到它溢出。
   改法：个数用 `size_t`，总和用 `uint64_t`。
   顺带一提：在 `char` 有符号的平台上，8K 全白帧反而不溢出，因为每个 255 都被读成了 −1，输出是 `mean=-1 bright=0% contrast=0`。
4. **空帧。** `n` 是 0 时，`sum / n` 是除以零，这也是 UB。在这之前，`pix[0]` 已经读了一个不属于这帧图的字节。
   观察：UBSan 报 `division by zero`。不开 sanitizer 时，同一个程序用 `-O0` 编译会收到 SIGFPE 信号被杀掉，用 `-O2` 编译却正常打印 `mean=0 bright=0% contrast=0` 并返回 0。
   同一个 UB，换个优化级别就是两种表现。改法：开头判空，并在注释里写明空帧返回什么。
5. **小问题。** 结果向下取整，注释里没写；`width` 和 `height` 可以传负数；`threshold` 没有检查范围。

**修好的版本。**

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

观察到的输出（`-fsigned-char` 和 `-funsigned-char` 两种编译方式下相同）：

```text
mean=100 bright=33% contrast=240
```

修好的版本对 8K 全白帧输出 `mean=255 bright=100% contrast=0`，对空帧输出全 0，ASan 和 UBSan 都没有报告。

**想问作者的问题。**

- 调用方手里的缓冲区是什么类型？这里为什么收 `const char *`？（多半是把读文件用的 `char` 缓冲区直接传了进来。）
- 这段代码要在哪些平台上跑？有没有 ARM 的开发板或者手机？
- 最大的帧有多大？会不会一次传进来多帧拼在一起的缓冲区？
- 空帧算不算合法输入？应该返回什么？
- 平均值和百分比要不要小数？向下取整还是四舍五入？
- 有没有答案已知的测试图，比如全黑、全白、一半黑一半白？

### Part 2

**想做什么。** 同时给一组帧 id 取缩略图，失败的重试几次，最后返回缩略图数组。

**实际输出。** 观察到的输出（连跑多次都一样）：

```text
fetched 0 of 3
done: []
frame 3: attempt 0 failed
frame 3: attempt 1 failed
frame 3: attempt 2 failed
300 ms later: [ 'thumb-2', 'thumb-1' ]
```

按时间顺序讲一遍：

1. `ids.forEach(async (id) => {...})`：`forEach` 依次调用三次回调。每个回调都是 `async` 函数，
   执行到第一个 `await` 就把控制权交回去，返回一个 Promise。`forEach` 不看返回值，这三个 Promise 没有人等。
2. 所以 `forEach` 马上返回，这时一个请求都还没完成。打印 `fetched 0 of 3`，返回一个空数组。
3. 调用方的 `.then` 运行，打印 `done: []`。
4. 第 10 ms：2 号帧成功，`thumb-2` 被 `push` 进那个已经返回出去的数组。
5. 第 20 ms：3 号帧第一次失败，打印 `frame 3: attempt 0 failed`。调用方传的是 `retries: 0`，意思是不重试。
   但 `0 || 2` 的结果是 2，因为 0 在 JavaScript 里是假值（falsy）。于是马上发出第二次请求。
6. 第 30 ms：1 号帧成功，`push` 进 `thumb-1`。
7. 第 40 ms 和第 60 ms：3 号帧的两次重试各等 20 ms，都失败，打印 `attempt 1` 和 `attempt 2` 两行。
   循环结束后这个回调什么也没留下，也没有报错。
8. 第 300 ms：调用方再看同一个数组，里面是 `[ 'thumb-2', 'thumb-1' ]`。顺序是完成的先后，不是 `ids` 的顺序；3 号帧没有留下任何痕迹。

（在每一行输出后面加上时间戳跑过一次，看到的时刻是 10、21、30、41、60 ms，和上面的推算一致。）

**问题清单。**

1. **`forEach` 加 `async` 回调，没有人等。** 函数在所有请求完成之前就返回了。
   改法：用 `ids.map(...)` 拿到 Promise 数组，再 `await Promise.all(...)` 或 `Promise.allSettled(...)`。
2. **`opts.retries || 2` 把 0 当成“没传”。** 改法：用 `??`，它只在左边是 `null` 或 `undefined` 时才取默认值。
3. **结果按完成顺序排列。** 调用方没法知道哪张图属于哪一帧。只修第 1 条的话，返回的仍然是 `[ 'thumb-2', 'thumb-1' ]`。
   改法：按 `ids` 的位置放结果，`map` 的结果本来就是这个顺序。
4. **失败被吞掉。** 重试用完之后既不抛错，也不占位。改法：定一个约定。下面用 `null` 占位；另一种约定是整个 `fetchAll` 抛错。
5. **`opts` 没有默认值。** 见下面的第一个追问。
6. **风险。** 重试之间没有等待，对一个正在返回 503 的服务会连打三次；所有请求同时发出，id 很多时没有并发上限。

**面试官接着问：调用方写成 `fetchAll([1, 2, 3])`，不传第二个参数，会怎样？** `opts` 是 `undefined`，读 `opts.retries` 抛 `TypeError`。
异常发生在 `async` 函数里，所以它变成一个被拒绝的 Promise。调用方只写了 `.then`，没有 `.catch`，没有人处理这个拒绝。
观察：Node 打印 `TypeError: Cannot read properties of undefined (reading 'retries')`，进程以退出码 1 结束，`done` 那一行没有打印。

**面试官接着问：把请求改成一个接一个地发（前一个结束再发下一个），和同时发相比有什么不同？**
用 `for (const id of ids)` 循环，在循环体里 `await`。结果数组相同，耗时不同。
观察：三个请求的延迟是 30、10、20 ms，一个接一个发用了约 60 ms（三者之和），同时发用了约 30 ms（三者中最大的）。
一个接一个发的好处是对服务的压力小，顺序天然正确；坏处是慢。

**修好的版本。**

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

观察到的输出：

```text
frame 3: attempt 0 failed
fetched 2 of 3
done: [ 'thumb-1', 'thumb-2', null ]
300 ms later: [ 'thumb-1', 'thumb-2', null ]
```

**想问作者的问题。**

- 调用方需要结果和 `ids` 一一对应吗？失败的帧是占位、抛错，还是可以丢掉？
- `retries: 0` 是不是合法用法？（是的话就不能用 `||`。）
- 一次最多有多少个 id？对方的服务有没有并发或限流的要求？
- 两次重试之间要不要等一会儿？哪些错误值得重试？（503 值得，404 不值得。）
- 有没有测试覆盖“某一帧一直失败”的情况？

### Part 3

**想做什么。** 按模型汇总一周的训练任务：任务数、完成的比例、平均验证损失、每个任务平均用多少 GPU 小时。
跳过已经退役的模型和 test-bot 启动的任务。平均损失最低的排在最前面。

**实际返回。** 零行。

原因是 `retired_models` 里有一行 `NULL`。`x NOT IN ('resnet', NULL)` 等价于 `x != 'resnet' AND x != NULL`。
和 `NULL` 比较，结果既不是真也不是假，而是 `NULL`（未知）。SQL 的逻辑有三个值：真、假、未知，这叫三值逻辑（three-valued logic）。
对 `vit-b`，整个条件是“真 AND 未知”，结果是未知。对 `resnet`，条件是假。`WHERE` 只留下条件为真的行，所以一行都不剩。

**面试官接着问：如果 `retired_models` 表里只有 `'resnet'` 这一行，查询返回什么？** 观察到的结果：

| model | runs | done_pct | avg_loss | gpu_h_per_run |
|---|---|---|---|---|
| vit-h | 1 | 0 | NULL | 20 |
| vit-l | 1 | 100 | 0.31 | 40 |
| vit-b | 3 | 0 | 0.4 | 8 |

查询能返回东西了，但几乎每一列都有问题。

**面试官接着问：4 号和 7 号任务是夜间调度器启动的，`owner` 是 `NULL`，它们出现在周报里了吗？**
没有，见下面问题清单的第 2 条。注释只说跳过 test-bot，所以它们本来应该在。

**问题清单。**

1. **`NOT IN` 遇到 `NULL`。** 子查询里只要有一个 `NULL`，`NOT IN` 就永远不为真。
   改法：用 `NOT EXISTS`。同时应该给 `retired_models.model` 加上 `NOT NULL` 约束。
2. **`owner != 'test-bot'` 把 `NULL` 也过滤掉了。** `NULL != 'test-bot'` 是未知，不是真。
   4 号和 7 号任务因此消失：`vit-b` 实际有 4 个任务，表里是 3；`vit-l` 去掉 test-bot 后应该有 2 个，表里是 1。
   改法：`(owner IS NULL OR owner != 'test-bot')`。
3. **`done_pct` 只会是 0 或 100。** SQLite 里整数除以整数还是整数。`vit-b` 的 2 ÷ 3 得 0，再乘 100 还是 0。
   改法：先乘 `100.0`，让运算变成浮点。
4. **`gpu_h_per_run` 被截断。** 也是整数除法：`vit-b` 的 25 ÷ 3 被截成 8。改法：先乘 `1.0`。
5. **`AVG` 会跳过 `NULL`。** `vit-b` 的 0.4 是两个已经评估过的任务的平均，不是全部任务的平均。
   这个算法本身合理，但读周报的人看不出平均了几个。改法：加一列 `COUNT(val_loss)`，它只数非 `NULL` 的值。
6. **排序。** `vit-h` 一个结果都没有，`avg_loss` 是 `NULL`，却排在“最好”的位置上。SQLite 在升序排序时把 `NULL` 放在最前面。
   改法：`ORDER BY avg_loss IS NULL, avg_loss`，先按“有没有值”排，再按值排。

**修好的版本。**

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

观察到的结果（`retired_models` 里有没有那行 `NULL` 都一样）：

| model | runs | done_pct | evaluated | avg_loss | gpu_h_per_run |
|---|---|---|---|---|---|
| vit-l | 2 | 50.0 | 1 | 0.31 | 24.5 |
| vit-b | 4 | 50.0 | 2 | 0.4 | 6.75 |
| vit-h | 1 | 0.0 | 0 | NULL | 20.0 |

**想问作者的问题。**

- `retired_models.model` 为什么允许 `NULL`？那一行是怎么进去的？
- `owner` 为 `NULL` 的调度器任务要不要算进周报？（注释只说跳过 test-bot。）
- `done_pct` 的分母要不要包括还在跑的任务？
- `avg_loss` 只平均评估过的任务，失败的任务不算。失败多的模型会不会因此显得更好？
- 这条查询最终跑在哪种数据库上？（见下面的追问。）

### 追问

**1. “整数除以整数”在这几种语言里各是什么规则？**
C 的整数除法向零截断：观察到 `-7 / 2` 是 −3，`-7 % 2` 是 −1。
JavaScript 的数字默认是浮点数：`7 / 2` 是 3.5，要整数得自己写 `Math.trunc`。
SQLite 的整数除法也向零截断：`-7 / 2` 是 −3，`2 / 3` 是 0。
Python 的 `//` 是向下取整：`-7 // 2` 是 −4，`-7 % 2` 是 1。
所以把一个公式从一种语言抄到另一种语言时，负数和“先除后乘”是最容易出错的两个地方。

**2. “没有值”在这几种语言里各长什么样？**
C 的 `int` 没有“空”这个状态，只能约定一个特殊值，或者另外传一个标志。
JavaScript 有 `null` 和 `undefined` 两种，还有一批假值：`0`、`""`、`NaN`、`false`。`||` 看的是真假，`??` 只看是不是 `null` 或 `undefined`。
容易记错的是：空数组 `[]`、空对象 `{}`、字符串 `"0"` 都是真值（已验证）。
SQL 的 `NULL` 表示“未知”，和任何东西比较都得未知，连 `NULL = NULL` 也是；判断要用 `IS NULL`；聚合函数会跳过它。

**3. C 那段代码为什么编译器一条警告都不给？应该靠什么发现？**
因为每一行都是合法的 C：`char` 有符号是实现的选择，`int` 相除是定义好的运算。
警告查不出“意图和类型不符”。能发现它的办法有三个：
用含义明确的类型（`uint8_t`）；写带已知答案的测试，并且测试图里要有大于 127 的像素；
用 `-fsanitize=address,undefined` 跑测试。注意 sanitizer 只能报出测试实际走到的输入，样例里的 6 个像素不会触发任何报告。

**4. 修好的 JavaScript 版本还缺什么，才能用在生产环境里？**
缺三样。一是并发上限：现在所有请求同时发出，id 有几千个时会压垮对方或者被限流，应该固定同时最多发几个。
二是重试前的等待，最好是逐次加倍再加一点随机量。
三是区分错误种类：服务器暂时不可用才值得重试，参数错误重试多少次都一样。

**5. 这条 SQL 换一种数据库，会有什么不同？**
这一条没有在本页运行，依据的是各数据库的文档，面试时要说明自己假设的是哪一种。
`NULL` 的排序位置各家不同：SQLite 升序时把 `NULL` 放最前，PostgreSQL 升序时默认放最后。
整数除法也不同：SQLite 和 PostgreSQL 截断，MySQL 的 `/` 返回小数。
`NOT IN` 遇到 `NULL` 的行为是 SQL 标准规定的，各家一致。
所以改法里尽量用不依赖这些差异的写法：显式乘 `100.0`，显式写出 `NULL` 排在哪一头。

<details>
<summary>验证代码（可运行）</summary>

下面的代码把题目和解答里的每一段源码原样放进字符串，写到临时目录，再调用 `gcc`、`node` 和 Python 自带的 `sqlite3` 去跑。
机器上没有 `gcc` 或 `node` 时，对应的部分会打印一行说明然后跳过。
对于未定义行为，只对 sanitizer 的报告做断言；普通编译下的表现只在 gcc 11、x86-64 上核对。

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
