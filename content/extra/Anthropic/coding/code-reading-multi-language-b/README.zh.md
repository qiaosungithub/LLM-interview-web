# 读代码：三段代码，三种语言（B 卷：C++ / Python / Bash）

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

运行环境：C++ 用 g++（C++17，Linux）；Python 用 3.x；Bash 用 4.4 或更新的版本，配 GNU 的 `ls`、`rm` 等命令。

### Part 1 —— C++：多目标跟踪器的每帧更新

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

### Part 2 —— Python：把片段按长度分桶并补齐

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

### Part 3 —— Bash：清理旧的 checkpoint

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

磁盘上的目录是这样的（右边是文件的修改时间）：

```text
/data/ckpts/
├── exp1/
│   ├── ckpt_0100.pt         11-01 10:00
│   ├── ckpt_0200.pt         11-01 11:00
│   ├── ckpt_0300.pt         11-01 12:00
│   ├── ckpt_0300 copy.pt    11-01 12:30      文件名里有一个空格
│   ├── ckpt_0400.pt         11-01 13:00
│   ├── ckpt_0500.pt         11-01 14:00
│   └── train.log            11-01 14:05
└── exp 2/                                    目录名里有一个空格
    └── ckpt_0100.pt         11-02 09:00
```

下面三条命令互不影响，每条都从上面这个目录状态开始。除了命令里写出来的，环境里没有设置 `DRY_RUN`。
脚本本身放在 `/home/me/prune_ckpts.sh`，有执行权限。
说出每条命令打印什么（包括报错）、退出码是多少、磁盘上还剩哪些文件。

```text
命令 1（当前目录是 /home/me）：
    CKPT_ROOT=/data/ckpts ./prune_ckpts.sh exp1 2

命令 2（当前目录是 /data/ckpts/exp1）：
    CKPT_ROOT=/data/ckpts /home/me/prune_ckpts.sh "exp 2" 1

命令 3（当前目录是 /home/me）：
    DRY_RUN=0 CKPT_ROOT=/data/ckpts ./prune_ckpts.sh exp1 2
```

## 参考解答

<details>
<summary>展开参考解答</summary>

三段代码各有一类“换一种语言就变样”的语义：C++ 里是值拷贝和迭代器失效；
Python 里是“名字只是引用”带来的别名和闭包的晚绑定；Bash 里是不加引号的变量会被拆成几个词，以及命令失败后脚本照常往下跑。
读的时候先用一句话说清楚意图，再拿样例输入一行一行算，最后才谈风险和改法。
下面所有“观察到的输出”都是在这台机器上实际跑出来的：g++ 11.4.0（x86-64 Linux，libstdc++）、Python 3.11.18、GNU bash 5.1.16。

### Part 1

**想做什么。** 每来一帧调用一次：匹配上的 track 把 `misses` 清零，没匹配上的加 1；
连续丢失超过 `max_misses` 帧的 track 删掉；每个没匹配上的检测框新开一个 track。

**实际输出。** 按意图，2 号和 3 号的 `misses` 从 2 变成 3，超过上限 2，应该被删掉，正确输出是 `1:0 4:0 100:0`。观察到的输出是：

```text
1:0 2:2 3:2 4:0 100:0
```

除了新开的 100 号，什么都没变。原因在 `for (auto t : tracks)`：`auto t` 是按值拷贝，`t` 是元素的一份副本，
`t.misses = ...` 改的是副本，循环一轮结束副本就销毁了。
`misses` 既然没变，就没有哪个 track 超过上限，第二个循环里的 `erase` 一次也没执行。
所以题目里的程序本身没有未定义行为（undefined behavior，简称 UB），它只是什么都没更新。`-Wall -Wextra` 没有任何警告。

**面试官接着问：把第一个循环改成 `for (auto& t : tracks)`，别的不动，输出是什么？这时程序正确了吗？**
不正确。第一个循环对了，2 号和 3 号的 `misses` 变成 3，第二个循环开始真的删东西，第二个错就露出来了。
`tracks.erase(it)` 会让 `it` 和它后面的所有迭代器失效（iterator invalidation）。接下来的 `++it` 用了一个已经失效的迭代器，这是 UB。
观察到的现象（普通编译、`-O2`、ASan 加 UBSan 三种情况相同）：

```text
1:0 3:3 4:0 100:0
```

3 号没被删。过程是：删掉 2 号后，后面的元素往前挪一格，3 号挪到了 `it` 指的位置；紧接着 `++it` 又把它跳过去了。
没有崩溃，sanitizer 也没有报告，只是结果悄悄错了。这是这一次观察到的结果，标准并不保证。
换成 libstdc++ 的调试模式（编译时加 `-D_GLIBCXX_DEBUG`），程序会中止，并报 `attempt to increment a singular iterator`。

如果被删的恰好是最后一个元素，情况更糟。把输入换成 `{{1, 0}, {2, 2}}`，只匹配上 1 号：
删掉 2 号后 `end()` 往前移了一格，`++it` 越过了 `end()`，`it != tracks.end()` 一直为真，循环一路读到缓冲区外面。
观察：ASan 报 `heap-buffer-overflow`；不开 sanitizer 时程序崩溃（段错误）。

**面试官接着问：同一个进程里有两路视频，各有一个跟踪器，各自调用 `update`，新开的 track 的 id 是多少？**
`static int next_id` 是函数里的静态变量，整个进程只有一份。两个跟踪器共用同一个计数器。
观察：两个空的跟踪器各调用一次 `update`，各新开一个 track，得到的 id 是 100 和 101，而不是各自从 100 开始。

**问题清单。**

1. **`for (auto t : tracks)` 按值拷贝。** 改法：`Track& t`。
2. **在循环里 `erase` 之后继续用旧迭代器。** 改法：用 erase-remove 惯用法（先 `std::remove_if` 把要留的元素挪到前面，再一次 `erase` 掉尾巴）。
   也可以写成 `it = tracks.erase(it)`，并且只在没删的时候 `++it`；C++20 可以直接写 `std::erase_if(tracks, ...)`。这三种写法在下面的验证代码里结果一致。
3. **函数里的 `static` 计数器。** 多个跟踪器共用；测试之间没法重置；两个线程同时调用 `update` 时，`next_id++` 是数据竞争（data race）。
   改法：让跟踪器自己持有这个计数器，由调用方传进来。
4. **效率。** 每个 track 都在 `matched_ids` 上做一次线性查找，T 个 track、M 个匹配就是 O(T·M)。
   几十个 track 时无所谓，上千个就该先把 `matched_ids` 放进 `std::unordered_set`。
   另外，在循环里一个一个 `erase`，每次都要搬动后面的元素，最坏是 O(T²)；erase-remove 只搬一遍，是 O(T)。

**修好的版本。**

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

观察到的输出（普通编译、ASan 加 UBSan、`-D_GLIBCXX_DEBUG` 三种情况相同）：

```text
1:0 4:0 100:0
```

**想问作者的问题。**

- 有没有测试覆盖“连续丢失超过上限，track 被删掉”这种情况？（题目里的代码永远不会删任何 track，说明这条路径从来没被测到。）
- 一个进程里会不会有多个跟踪器？会不会从多个线程调用？
- id 要全局唯一，还是每路视频各自从头编号？
- track 一般有多少个？这决定了线性查找要不要换掉。
- “超过 `max_misses`”是严格大于吗？注释和代码都是大于，确认一下是不是本意。

### Part 2

**想做什么。** 把长短不一的片段按长度放进几个桶：每个片段进第一个装得下它的桶，并用 −1 补到桶的长度；
比最大的桶还长的片段丢掉，并在 `stats["dropped"]` 里计数。

**实际输出。** 5 个片段的长度是 3、20、30、6、9，桶的长度是 4、8、16。
按意图，20 和 30 被丢掉，3 进 4 号桶，6 进 8 号桶，9 进 16 号桶。正确的输出应该是：

```text
[[4], [8], [16]] {'dropped': 2} 5
{'dropped': 1} {'dropped': 2}
```

观察到的输出是：

```text
[[16, 16, 16], [16, 16, 16], [16, 16, 16]] {'dropped': 1} 4
{'dropped': 2} {'dropped': 2}
```

这里面有四个互相独立的错。

1. **一边遍历一边删。** `for clip in clips` 内部靠一个下标往前走。走到下标 1（长度 20）时把它删了，后面的元素往前挪一格；
   下标接着走到 2，那里现在是长度 6 的片段。长度 30 的片段被跳过了，没有被检查。
   所以只数到 1 个 `dropped`，长度 30 的片段还留在列表里。到了分桶那一步，它哪个桶都装不下，循环走完也没有 `break`，就这样没了：既没进桶，也没计数。
   另外，调用方传进来的 `clips` 被改掉了，从 5 个变成 4 个。
2. **`[[]] * len(edges)`。** 这个写法得到的是三个指向同一个列表的引用，不是三个列表。往 `buckets[0]` 里放一个片段，三个桶里都多一个。所以每个桶里都有 3 个片段。
3. **闭包的晚绑定（late binding）。** 三个 `lambda` 记住的是变量 `size` 本身，不是创建时它的值。
   真正调用 `pad` 的时候循环早已结束，`size` 停在最后一个值 16。所以所有片段都被补到 16。
4. **可变的默认参数。** `stats={}` 这个字典在 `def` 执行时只创建一次，以后每次不传 `stats` 的调用用的都是它。
   第二次调用的计数从 1 接着加到 2；第一次调用返回的 `stats` 和它是同一个对象，也跟着变成了 2。这就是第二行输出的由来。

**面试官接着问：第一次调用结束后，调用方手里的 `clips` 还是原来的 5 个片段吗？**
不是。只剩 4 个，长度是 3、30、6、9。原因见上面第 1 条。

**面试官接着问：把两次调用里的 `edges` 都换成只有一个桶的 `[16]`，两行输出各是什么？哪些错看不出来了？** 观察到的输出是：

```text
[[16, 16, 16]] {'dropped': 1} 4
{'dropped': 2} {'dropped': 2}
```

正确的输出是：

```text
[[16, 16, 16]] {'dropped': 2} 5
{'dropped': 1} {'dropped': 2}
```

桶里的内容碰巧对了。只有一个桶时，`[[]] * 1` 没有第二个名字可以共用；`size` 的最后一个值也就是唯一的值。
第 2、3 两个错都看不出来，只剩计数和调用方的列表是错的。所以测试只用一个桶的话，这两个错永远测不到。

**修好的版本。**

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

这段代码在本页运行时打印的就是上面“正确的输出”那两行。

**想问作者的问题。**

- 这个函数可以改调用方传进来的 `clips` 吗？（原来的实现改了，文档里没说。）
- 太长的片段是丢掉、截断，还是切成几段？对视频数据来说，总是丢掉最长的样本会让数据分布变偏。
- `stats` 这个参数，是想让调用方跨多次调用累计吗？如果是，就应该要求调用方明确传进来。
- `edges` 保证是升序、非空的吗？
- 用 −1 补齐，会不会和真实的帧 id 冲突？

### Part 3

**想做什么。** 保留某个 run 最新的 KEEP 个 checkpoint，把更旧的删掉，并在 `prune.log` 里记一行。设了 `DRY_RUN` 就只打印，不删。

**命令 1。** 观察到的输出（退出码 0）：

```text
found 6 checkpoints in /data/ckpts/exp1
deleting ckpt_0300
deleting copy.pt
deleting ckpt_0300.pt
deleting ckpt_0200.pt
deleting ckpt_0100.pt
done
```

`exp1` 里剩下 `ckpt_0300 copy.pt`、`ckpt_0400.pt`、`ckpt_0500.pt`、`train.log`，外加新写的 `prune.log`。要求留 2 个，实际留了 3 个。
原因：`$(ls -t | ...)` 的结果没有加引号，shell 会按空白把它拆成一个个词（word splitting）。
`ckpt_0300 copy.pt` 被拆成 `ckpt_0300` 和 `copy.pt` 两个名字。这两个文件都不存在，而 `rm -rf` 里的 `-f` 让它遇到不存在的文件也不出声。
脚本打印了“deleting”，其实什么都没删到。如果目录里恰好有一个叫 `copy.pt` 的文件，被删的就是它（已验证）。

**命令 2。** 观察到的输出（退出码 0；以 `/home/me/prune_ckpts.sh:` 开头的两行是 bash 打到标准错误上的）：

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

这是最严重的一个错。`cd $ROOT` 展开成 `cd /data/ckpts/exp 2`，`cd` 收到两个参数，失败了。
脚本没有检查 `cd` 成不成功，接着在当前目录，也就是 `exp1` 里干活：
`exp1` 的 checkpoint 被删到只剩最新的 `ckpt_0500.pt`（外加那个带空格的文件），而 `exp 2` 一个文件都没动。
打印出来的却是 `in /data/ckpts/exp 2`。最后写日志那一行同样因为没加引号而失败，脚本仍然打印 `done`，退出码是 0。
调用它的人或者定时任务看不出任何异常。

**命令 3。** 观察到的输出（退出码 0）：

```text
found 6 checkpoints in /data/ckpts/exp1
would delete ckpt_0300
would delete copy.pt
would delete ckpt_0300.pt
would delete ckpt_0200.pt
would delete ckpt_0100.pt
done
```

一个文件都没删。`[ $DRY_RUN ]` 测的是“这个字符串是不是非空”。`0` 是一个非空字符串，所以条件为真。
shell 里没有“0 是假”这回事。写 `DRY_RUN=0` 的人想的是“关掉试运行”，得到的却是试运行。
另外，试运行也照样往 `prune.log` 里写了一行 `kept 2`。

**面试官接着问：run 的名字打错了（比如 `exp3`，这个目录不存在），或者干脆忘了传，会怎样？** 两种都实际跑过。

- run 的名字打错：`cd` 报 `No such file or directory`，之后和命令 2 一样，在当前目录里删。
  观察：在 `exp1` 里执行 `CKPT_ROOT=/data/ckpts /home/me/prune_ckpts.sh exp3 1`，`exp1` 被删到只剩 `ckpt_0300 copy.pt`、`ckpt_0500.pt` 和 `train.log`。退出码还是 0。
- 忘了传名字：`RUN` 是空的，`ROOT` 变成 `/data/ckpts/`，`cd` 成功了，脚本开始清理所有 run 的上一层目录。
  `grep ckpt` 会匹配任何名字里带 `ckpt` 的东西，目录也算。
  观察：在 `/data/ckpts` 下放四个目录 `ckpt_archive_a` 到 `ckpt_archive_d`（a 最旧），不带任何参数执行脚本。
  KEEP 取默认值 3，脚本打印 `found 4 checkpoints in /data/ckpts/`，最旧的 `ckpt_archive_a` 连同里面的文件被 `rm -rf` 整个删掉。退出码 0。
  如果连 `CKPT_ROOT` 也没设，`ROOT` 就是 `/`（单独执行那一行赋值，得到的确实是 `/`）。这种情况下的整个脚本没有运行过，后果是推出来的：它会在根目录里找名字带 `ckpt` 的东西来删。

**问题清单。**

1. **`cd $ROOT` 不加引号，也不检查结果。** 改法：加引号；开头写 `set -euo pipefail`，让任何一条命令失败时脚本立刻停下。
2. **`for f in $(ls ...)` 和 `rm -rf $f` 的拆词。** 改法：把文件名读进数组，用的时候写成 `"$f"`。
3. **`grep ckpt` 太宽，`rm -rf` 太狠。** 改法：只匹配 `ckpt_*.pt`；删文件用 `rm -f --`，不带 `-r`。必填的参数缺了就直接报错退出。
4. **`[ $DRY_RUN ]` 测的是非空。** 改法：明确和 `1` 比较。
5. **写日志的重定向没加引号。** 观察到的报错是 `ambiguous redirect`。
6. **退出码永远是 0。** `KEEP` 传成 `abc` 时，`[` 报 `integer expression expected`，脚本什么都不删，照样打印 `done` 并返回 0。
7. **“最新”是按修改时间算的。** 文件被复制、同步或者从备份恢复之后，修改时间就不代表训练的先后了。这一条要问作者。

**修好的版本。**

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

观察到的输出。命令 1（`exp1` 里剩下 `ckpt_0400.pt`、`ckpt_0500.pt`、`train.log` 和 `prune.log`）：

```text
found 6 checkpoints in /data/ckpts/exp1
deleting ckpt_0300 copy.pt
deleting ckpt_0300.pt
deleting ckpt_0200.pt
deleting ckpt_0100.pt
done
```

命令 2（`exp1` 原封不动，`exp 2` 里多了一个 `prune.log`）：

```text
found 1 checkpoints in /data/ckpts/exp 2
done
```

命令 3 的输出和命令 1 相同，文件真的被删了；`DRY_RUN=1` 时打印四行 `would delete ...`，不删文件，也不写日志。
run 的名字打错时，脚本在 `cd` 那一行停下，退出码 1；忘了传名字时退出码 1；`KEEP` 传成 `abc` 时退出码 2。这三种情况下都没有文件被删。

修好的版本仍然在解析 `ls` 的输出，文件名里有换行符时还是会出错。要彻底解决，就按文件名里的步数排序，或者改用 Python。

**想问作者的问题。**

- checkpoint 是单个文件，还是一个目录？原来用 `rm -rf`，是因为真的有目录形式的 checkpoint，还是随手写的？
- “最新”应该按修改时间，还是按文件名里的步数？
- 有没有必须一直留着的文件，比如 `ckpt_best.pt`？
- 这个脚本由谁运行，在哪个目录下运行？放在定时任务里的话，失败了谁会知道？
- run 的名字里会不会有空格或者别的特殊字符？
- 删之前要不要确认训练进程没有正在写这个目录？

### 追问

**1. “拷贝还是引用”，这三种语言默认各是什么？**
C++ 默认拷贝：`auto t`、按值传参、赋值都会复制一份，想改原对象必须写 `&`。
Python 从不自动拷贝：变量和参数都只是对象的名字，所以才有 `[[]] * 3` 里三个名字指向同一个列表、函数改掉调用方的列表这类事；想要副本得自己写。
Bash 里一切都是字符串，赋值就是复制字符串；它的坑不在拷贝，而在使用变量时会不会被重新拆词。
读代码时碰到“改了一个东西”，先问一句：改的是原件还是副本。

**2. 一边遍历一边删，三种语言里各有什么表现，各自安全的写法是什么？**
Python 的列表不报错，只是悄悄跳过元素：对 `[1, 2, 2, 3, 2]` 边遍历边删 2，得到的是 `[1, 3, 2]`。字典则会直接抛 `RuntimeError`。安全的写法是新建一个列表。
C++ 的 `vector` 是未定义行为，可能跳过元素，也可能越界。安全的写法是 erase-remove，或者用 `erase` 的返回值接着走。
Bash 里先把要删的名字收集进数组，再统一删。

**3. 闭包记住的是变量，还是当时的值？**
Python 记住的是变量：循环里创建的三个 `lambda: i`，调用结果是 `[2, 2, 2]`；写成 `lambda i=i: i`，把当时的值存进默认参数，才得到 `[0, 1, 2]`。
JavaScript 看声明方式：`for (var i ...)` 里的三个闭包共用一个 `i`，结果是 `[ 3, 3, 3 ]`；换成 `let`，每轮循环有自己的 `i`，结果是 `[ 0, 1, 2 ]`。
C++ 的 lambda 必须在方括号里写明：`[=]` 复制一份，`[&]` 记引用。记了引用而变量已经销毁，就是悬空引用。

**4. ASan 为什么没有报出 C++ 那个迭代器错误？还有类似的坑吗？**
ASan 检查的是“这次访问的内存归不归你”。`erase` 之后那块内存仍然属于这个 `vector`，访问它是合法的，错的只是逻辑上那个位置已经换了元素。
所以 sanitizer 没报，不等于没问题；这类错要靠 libstdc++ 的调试模式，或者靠带已知答案的测试。
类似的坑是 `push_back`：容量不够时 `vector` 会换一块更大的内存，之前拿到的引用、指针和迭代器全部失效。
观察：先取 `tracks.front()` 的引用，再 `push_back`，再通过这个引用把 `misses` 改成 0。ASan 报 `heap-use-after-free`；普通编译下程序照常结束，打印出来的 `misses` 还是原来的 5，那次写入丢了。

**5. 这个 Bash 脚本要不要干脆用 Python 重写？**
看它会不会继续长。只有十来行、只删固定格式的文件名，留在 Bash 里可以，前提是做到三件事：`set -euo pipefail`、所有变量加引号、不解析 `ls` 的输出（上面修好的版本只做到了前两件，原因见那里的说明）。
一旦需要按步数排序、保留最好的一个、统计大小、写测试，就该换成 Python：文件名是普通字符串，不会被拆开；数字就是数字；出错会抛异常，而不是默默继续。
另外，ShellCheck 这类静态检查工具能直接指出没加引号的变量。这台机器上没有装它，这里没有运行。

<details>
<summary>验证代码（可运行）</summary>

下面的代码把题目和解答里的每一段源码原样放进字符串，写到临时目录，再调用 `g++`、`bash` 和 Python 去跑。
机器上没有 `g++` 或 `bash` 时，对应的部分会打印一行说明然后跳过。
有 bug 的那个 Bash 脚本会在“当前目录”里删文件，所以它每次都在临时目录里面运行，`CKPT_ROOT` 也总是指向临时目录。
对于未定义行为，只对 sanitizer 和 libstdc++ 调试模式的报告做断言；普通编译下的表现只打印出来，在 g++ 11、x86-64 上才核对。

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
