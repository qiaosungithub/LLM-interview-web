# 官方原题：性能优化 Take-home（已开源的旧版）

## 题目

> **这是 Anthropic 官方公开的原题。** 2026 年 1 月 21 日，Anthropic 在工程博客
> [Designing AI-resistant technical evaluations](https://www.anthropic.com/research/AI-resistant-technical-evaluations)
> 里公开了性能工程团队用过的 take-home，代码在 https://github.com/anthropics/original_performance_takehome 。
> 它已经不再用于限时招聘。本地副本在
> `/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome/`，
> 和下载的原件逐字相同，请复制一份再改。

### 来历

- 2023 年 11 月前后设计，2024 年初开始使用，一千多名候选人做过。机器模拟一种和 TPU 相似的加速器。
  博客明确说这份 take-home **允许使用 AI**（Anthropic 的一般规定是 take-home 默认不用 AI，除非另有说明）。
- 第 1 版限时 4 小时，从完全串行的代码开始。热身是多核并行，然后自选先做 SIMD 向量化还是 VLIW 指令打包。
  这一版还埋了一个要先调试出来的 bug。
- 2025 年 5 月前后，预发布版的 Claude Opus 4 在 4 小时内做得比几乎所有候选人都好。于是有了第 2 版：限时改为
  2 小时；起点换成 Opus 4 开始吃力的地方（Readme 说起始代码是 18532 个周期）；代码更干净，加了新的机器功能，去掉了多核。
- 预发布版的 Claude Opus 4.5 不到一小时就过了及格线，然后认定自己撞上了内存带宽瓶颈。博客说多数人也会这样想，
  但有利用题目结构绕过这个瓶颈的巧妙办法（没说是什么）。告诉它能做到的周期数之后，它找到了办法，
  到 2 小时时追平了人类在 2 小时内的最好成绩（那位候选人也大量借助了 Claude 4）。之后限时 take-home 换成了另一类题（见下面“和站内其他页面的关系”）。
- 公开版和第 1 版一样从最慢的写法开始（147734 个周期），但指令集和单核设计来自第 2 版，所以周期数可以和第 2 版直接比较。

Readme 给的参考成绩。注意这些都是模型做 **2 小时版**、从 18532 个周期出发的结果：

| 周期数 | 谁，在什么条件下 |
|---:|---|
| 2164 | Claude Opus 4，在测试时计算（test-time compute）框架里跑了很多小时 |
| 1790 | Claude Opus 4.5，一次随手的 Claude Code 会话；大约等于人类 2 小时内的最好成绩 |
| 1579 | Claude Opus 4.5，在框架里跑 2 小时 |
| 1548 | Claude Sonnet 4.5，远多于 2 小时的测试时计算 |
| 1487 | Claude Opus 4.5，在框架里跑 11.5 小时 |
| 1363 | Claude Opus 4.5，改进后的框架 |
| ??? | 人类历史最好成绩“明显更好”，没有公布具体数字 |

公开挑战：做到 1487 个周期以下（公开时 Claude 的最好成绩），可以把代码（最好附简历）发到
performance-recruiting@anthropic.com。Readme 说新模型发布后这个门槛可能会变，也不保证 Readme 会及时更新。

Readme 还提醒：公开第一天收到的 1300 以下的提交全都无效，每一份都是语言模型改了测试。
一个例子：模型发现 `problem.py` 支持多核，实现了多核却没有加速，于是“修好”了 `N_CORES = 1`。多核是故意关掉的。

### 机器

全部定义在 `problem.py` 里。`tests/frozen_problem.py` 是计分用的冻结副本，内容和 `problem.py` 相同。

- **字（word）**：内存和草稿区都由 32 位无符号整数组成。所有运算结果都对 $2^{32}$ 取模。
- **内存（memory）**：一个扁平的字列表。程序结束后，结果从内存里取。
- **草稿区（scratch）**：每个核 1536 个字（`SCRATCH_SIZE`），同时充当寄存器、常量区和手动管理的缓存。
  指令里的数几乎都是草稿区地址；例外是 `const` 和 `add_imm` 的立即数、`load_offset` 的偏移量、跳转目标。
- **指令束（bundle）**：程序是一串指令束。每个指令束是一个 `{引擎名: [槽位, ...]}` 的字典，一个周期执行一个指令束。
  每个引擎（engine）每个周期最多能用这么多个槽位（slot）：

| 引擎 | 每周期槽位 | 用途 |
|---|---:|---|
| `alu` | 12 | 标量运算 |
| `valu` | 6 | 向量运算；一个向量是草稿区里连续的 8 个字（`VLEN = 8`） |
| `load` | 2 | 从内存读，或者装入一个常量 |
| `store` | 2 | 往内存写 |
| `flow` | 1 | 选择、跳转、暂停等 |
| `debug` | 64 | 只用于调试；计分时被忽略 |

- **时序**：同一个指令束里的所有槽位都读周期开始时的状态；写草稿区、写内存都在周期结束时才生效。
  所以同一周期里的槽位互不影响，结果从下一个周期起可用。除此以外**没有任何延迟**，加载也一样。
- **计周期**：每执行一个含有非 `debug` 引擎的指令束，周期数加 1。程序在 `pc` 走出末尾或执行 `halt` 时结束。
- **单核**：`N_CORES = 1`。模拟器里还留着多核的代码，但这一版故意只用一个核。

指令如下（`a`、`b`、`c`、`dest`、`cond`、`addr` 都是草稿区地址）。`alu` 的 `op` 可以是
`+`、`-`、`*`、`//`、`cdiv`、`^`、`&`、按位或、`<<`、`>>`、`%`、`<`、`==`（后两个的结果是 0 或 1）。

| 引擎 | 写法 | 含义 |
|---|---|---|
| `alu` | `(op, dest, a, b)` | `dest = a op b` |
| `valu` | `(op, dest, a, b)` | 对 8 个位置逐个做同样的 `op` |
| `valu` | `("vbroadcast", dest, a)` | 把标量 `a` 复制到 8 个位置 |
| `valu` | `("multiply_add", dest, a, b, c)` | 逐个位置 `dest = a * b + c` |
| `load` | `("load", dest, addr)` | `dest = mem[scratch[addr]]` |
| `load` | `("load_offset", dest, addr, k)` | 第 `dest + k` 个字 `= mem[scratch[addr + k]]` |
| `load` | `("vload", dest, addr)` | 从 `mem[scratch[addr]]` 开始读 8 个**连续**的字 |
| `load` | `("const", dest, v)` | `dest = v`（立即数；也占一个 load 槽位） |
| `store` | `("store", addr, a)` | `mem[scratch[addr]] = a` |
| `store` | `("vstore", addr, a)` | 把 8 个字写到 `mem[scratch[addr]]` 开始的连续位置 |
| `flow` | `("select", dest, cond, a, b)` | `dest = a if cond != 0 else b` |
| `flow` | `("vselect", dest, cond, a, b)` | 逐个位置做 `select` |
| `flow` | `("add_imm", dest, a, imm)` | `dest = a + imm` |
| `flow` | `halt`、`pause`、`jump`、`cond_jump`、`cond_jump_rel`、`jump_indirect`、`coreid`、`trace_write` | 停机、暂停、跳转等 |
| `debug` | `compare`、`vcompare`、`comment` | 和参考实现的中间值比对；计分时关闭 |

要特别注意：**没有收集加载（gather）指令**。`vload` 只能读连续的 8 个字；要读 8 个不相邻的地址，只能用 8 条 `load`，
而 `load` 每周期只有 2 个槽位。

### 内核

内存布局（`build_mem_image`）：

| 位置 | 内容 |
|---|---|
| `mem[0..6]` | `rounds, n_nodes, batch_size, forest_height, forest_values_p, inp_indices_p, inp_values_p` |
| 从 `forest_values_p`（总是 7）开始 | 树的 `n_nodes` 个节点值 |
| 从 `inp_indices_p` 开始 | `batch_size` 个当前节点下标，初始全是 0 |
| 从 `inp_values_p` 开始 | `batch_size` 个当前值 |

内存总长正好是 `7 + n_nodes + 2 * batch_size` 个字，后面没有空余（`build_mem_image` 本来留了空白，但最后一步切片赋值把它截掉了）。

树是满二叉树（perfect binary tree），高度为 `forest_height`，共 $2^{h+1}-1$ 个节点；节点 `i` 的两个孩子是 `2i+1` 和 `2i+2`。
节点值和初始值都是 $[0, 2^{30})$ 里的随机整数。要算的东西（`reference_kernel2` 的意思）：

```py
for r in range(rounds):
    for i in range(batch_size):
        idx = indices[i]
        val = myhash(values[i] ^ tree[idx])
        idx = 2 * idx + (1 if val % 2 == 0 else 2)   # even: left child, odd: right child
        idx = 0 if idx >= n_nodes else idx           # past a leaf: back to the root
        values[i] = val
        indices[i] = idx

HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]

def myhash(a):
    for op1, c1, op2, op3, c3 in HASH_STAGES:
        a = ((a op1 c1) op2 (a op3 c3)) % 2**32      # each inner result is also taken mod 2**32
    return a
```

你要写的是 `perf_takehome.py` 里的 `KernelBuilder.build_kernel(forest_height, n_nodes, batch_size, rounds)`：
它生成一段程序，放进 `self.instrs`。程序在上面的内存上运行，结束时 `inp_values_p` 开始的那一段必须和参考实现完全一样。
这四个参数在生成程序时就是普通的 Python 整数，所以程序可以针对具体规模来生成。
起始代码是标量写法，每个指令束只放一个槽位，共 147734 个周期。

### 计分与规则

- **计分命令**：`python tests/submission_tests.py`。规模固定为 `forest_height=10, rounds=16, batch_size=256`（2047 个节点）。
  随机数不设种子。
- **正确性**：随机生成 8 组输入，各跑一次，只比较最后的 `values` 那一段。`indices` 写不写回都可以。
- **速度**：再跑一次，读 `machine.cycle`；结果不对就按 2 × 147734 算。然后和 8 个门槛比，必须**严格小于**：
  147734、18532、2164、1790、1579、1548、1487、1363。测试文件里的说明是：不需要全部通过也能过面试，
  通过的个数和水平也不是线性关系，这些门槛只是为了在 CodeSignal 上显示一个分数。
- **规则**：你的代码写在 `perf_takehome.py` 里（主要是 `build_kernel`），**不能改 `tests/` 里的任何东西**。
  多核是故意关掉的，不要动 `N_CORES`。成绩以 `python tests/submission_tests.py` 打印的周期数为准。
- **许可**：起始代码开头写着：可以修改和使用，但请不要公开发布或转发你的解答，免得别人搜到答案。
  所以本页的参考解答只放在本地学习用。

### 在本地做题

需要 Python 3.10 或更新（代码用了 `match` 语句），不需要第三方库。

```bash
# 1. copy the pristine folder to a working folder; never edit the copy under materials/
cp -r /kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome ~/perf-takehome
cd ~/perf-takehome

# 2. while developing: fixed seed 123; also checks that `pause` lines up with the reference's `yield`
python perf_takehome.py Tests.test_kernel_cycles

# 3. the real score: which thresholds you pass
python tests/submission_tests.py

# 4. tests/ must be unchanged (this copy has no git history, so `git diff origin/main tests/` does not apply)
diff -r -x __pycache__ tests /kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome/tests && echo "tests/ unchanged"
```

仓库自带调试工具：`python perf_takehome.py Tests.test_kernel_trace` 会写出 `trace.json`（每个槽位一条记录，Chrome 的 Trace Event 格式，用 Perfetto 查看）；
`python watch_trace.py` 在本机 8000 端口开一个页面，在 Chrome 里热加载这个文件；也可以把 `trace.json` 直接拖进 https://ui.perfetto.dev/ 。
起始代码还演示了用 `debug` 引擎的 `compare` 把草稿区里的值和参考实现的中间结果逐个对比。（这两样我们都没有实际打开用过。）

### 建议的做题顺序

1. 读懂 `problem.py` 和起始代码，能说清楚 147734 是怎么来的。
2. 先过 18532（第 2 版的起点），再过 2164、1790。按 Readme，人类 2 小时内的最好成绩约为 1790（而且是从 18532 出发的）。
3. 想挑战的话，目标是 1487 以下。参加公开挑战时，交的应当是你自己的代码。

### 和站内其他页面的关系

- [在模拟向量机上优化计算内核](#/anthropic/take-home/performance-kernel-optimization)：站内另一道同类练习，
  机器和内核是另外设计的。那台机器有寄存器，每个指令束能放的指令少得多（2 条标量、1 条向量、1 次加载、1 次存储），
  加载有 4 个周期的延迟，内核是浮点数的乘加求和。这里没有延迟，用草稿区代替寄存器，内核是 32 位整数哈希加上
  “下标取决于数据”的树上读取，门槛是固定的周期数。打包、向量化、找瓶颈这些思路两边通用。
- [现在用的限时 take-home](#/anthropic/take-home/constrained-machine-puzzles-a)：博客说 Opus 4.5 之后，
  限时 take-home 换成了另一类题：在一个极小、限制很多的指令集上解一组谜题，按指令条数优化，而且故意不提供可视化和调试工具。
  它和这一题不是同一种任务。

## 参考解答

<details>
<summary>展开参考解答</summary>

### 总览

我们在自己的副本上分 9 个阶段优化。每个阶段都用 `tests/submission_tests.py` 实测（规模 10 / 16 / 256，`tests/` 没有改动；
每个阶段 9 个测试打印出的周期数都相同，正确性测试每次都通过）。

| 阶段 | 改了什么 | 周期 | 相对起点 | 通过的速度门槛 |
|---|---|---:|---:|---:|
| 起点 | 起始代码 | 147734 | 1.00× | 0 / 8 |
| 1 | 标量；去掉多余的操作 | 97040 | 1.52× | 1 / 8 |
| 2 | 同样的操作，按依赖关系打包进同一周期 | 7690 | 19.21× | 2 / 8 |
| 3 | SIMD：32 块，每块 8 个元素 | 2084 | 70.89× | 3 / 8 |
| 4 | 树的第 0–2 层不再从内存读 | 1548 | 95.44× | 5 / 8 |
| 5 | 按“一块接一块”的顺序交给调度器 | 1355 | 109.03× | 8 / 8 |
| 6 | 减少每轮的向量操作 | 1329 | 111.16× | 8 / 8 |
| 7 | 第 3 层也不读内存；选节点的活交给 flow 引擎 | 1107 | 133.45× | 8 / 8 |
| 8 | 第二趟里 20 个块的第 4 层也不读内存 | 1064 | 138.85× | 8 / 8 |
| 9 | gather 之后的异或交给标量 ALU | 1040 | 142.05× | 8 / 8 |

阶段 4 是 1548，正好等于 1548 的门槛，不算通过（要求严格小于）。

完整代码在 `/kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome-solution/perf_takehome_staged.py`。
它可以直接替换 `perf_takehome.py`，用环境变量 `PERF_STAGE` 选阶段（默认是 9）：

```bash
cp /kmh-nfs-ssd-us-mount/code/qiao/work/interview_prep/materials/anthropic-original-performance-takehome-solution/perf_takehome_staged.py ~/perf-takehome/perf_takehome.py
cd ~/perf-takehome
python tests/submission_tests.py                 # stage 9: 1040
PERF_STAGE=3 python tests/submission_tests.py    # stage 3: 2084
```

下面的代码块都是从这个文件里摘的（有删节，个别地方为了单独展示做了简化），只展示，不在本页运行。

### 方法：每一步先算各引擎的下界

每做一步，先数程序里每个引擎各有多少个操作。操作数除以这个引擎每周期的槽位数，就是周期数的一个下界（lower bound）。
例如阶段 9 有 1949 个 `load` 操作，每周期最多 2 个，所以至少要 975 个周期。

最大的那个下界说明瓶颈在哪个引擎；实测值和它的差距说明调度（把操作排进周期）损失了多少。
下一步就从三件事里选：减少瓶颈引擎上的操作；把操作挪到空闲的引擎上；改进调度。

| 阶段 | load | valu | alu | flow | 实测 |
|---|---:|---:|---:|---:|---:|
| 1 | 2312 | — | 7680 | — | 97040（一个周期一个槽位，等于操作总数） |
| 2 | 2312 | — | 7680 | — | 7690 |
| 3 | 2073 | 1336 | 3 | — | 2084 |
| 4 | 1306 | 1348 | 3 | — | 1548 |
| 5 | 1306 | 1348 | 3 | — | 1355 |
| 6 | 1307 | 1119 | 473 | — | 1329 |
| 7 | 1053 | 1067 | 431 | 704 | 1107 |
| 8 | 975 | 1053 | 432 | 1004 | 1064 |
| 9 | 975 | 1014 | 590 | 1004 | 1040 |

表里是每个引擎的下界（操作数 ÷ 槽位数，向上取整）。`store` 每个阶段都很小，没有列出。

### 基础设施：虚拟寄存器、调度器、草稿区分配

从阶段 2 起，程序都按同一个流程生成：

1. **直线代码**：针对给定规模把所有轮、所有元素完全展开，不用循环和跳转。这样周期数就等于指令束的个数，和输入数据无关。
2. **虚拟寄存器（virtual register）**：每算出一个值就放进一个新的虚拟寄存器，每个字只写一次。
   这样操作之间只剩“读必须在写之后”这一种约束，调度器可以随意重排。
3. **列表调度（list scheduling）**：按给定顺序处理每个操作，放进“所有依赖都已经算完的下一个周期”之后、
   这个引擎还有空槽位的最早周期。然后把整张依赖图倒过来再排一遍（每个操作尽量晚），再按倒排的结果正着排一遍，
   这样来回 3 次，保留最好的结果。
4. **分配草稿区**：调度完以后，按每个虚拟寄存器的存活区间（第一次写到最后一次读）分配草稿区地址。
   因为写入在周期结束时才生效，新值可以在旧值最后一次被读的同一个周期里占用它的地址。

```py
def place(order, deps, eng, reserved=()):
    """Serial list scheduling: each op goes into the earliest cycle that is later than all of
    deps[op] and still has a free slot on its engine."""
    slots = {name: Slots(cap) for name, cap in SLOT_LIMITS.items()}
    for name, c in reserved:
        slots[name].take(c)
    cyc = [-1] * len(eng)
    for i in order:
        lb = 0
        for p in deps[i]:
            if cyc[p] >= lb:
                lb = cyc[p] + 1
        c = slots[eng[i]].find(lb)          # first cycle >= lb with a free slot on this engine
        slots[eng[i]].take(c)
        cyc[i] = c
    return cyc


def schedule(pr, pack=True, passes=3):
    n = len(pr.eng)
    if not pack:
        return list(range(n))                     # stage 1: one slot per cycle, in emission order
    reserved = [("flow", 0)]                      # cycle 0 keeps its flow slot for the `pause`
    cyc = place(range(n), pr.preds, pr.eng, reserved)
    succs = [[] for _ in range(n)]
    for i, ps in enumerate(pr.preds):
        for p in ps:
            succs[p].append(i)
    best = cyc
    for _ in range(passes):
        # everything as late as possible on the reversed graph, then forward again in that order
        back = place(sorted(range(n), key=lambda i: (-cyc[i], -i)), succs, pr.eng)
        cyc = place(sorted(range(n), key=lambda i: (-back[i], i)), pr.preds, pr.eng, reserved)
        if max(cyc) < max(best):
            best = cyc
    return best
```

`Prog` 类负责记录操作：每个操作记下引擎、指令模板（草稿区地址先用虚拟寄存器占位）、读哪些寄存器、写哪些寄存器，
以及它依赖哪些操作。常量和广播会去重：同一个常量只装一次，同一个标量只广播一次。
`assemble` 把调度结果拼成指令束，并在第 0 个周期放一条 `pause`，对上参考实现的第一个 `yield`（只有本地开发测试需要它）。

来回排几遍有多大作用？下面是我们用自己的小脚本测的：只改遍数，其余不变，跑一次随机输入并核对结果。

| 阶段 | 0 遍 | 1 遍 | 3 遍（采用） | 6 遍 |
|---|---:|---:|---:|---:|
| 2 | 7944 | 7691 | 7690 | 7690 |
| 5 | 1473 | 1356 | 1355 | 1355 |
| 7 | 1178 | 1113 | 1107 | 1107 |
| 9 | 1140 | 1050 | 1040 | 1040 |

### 阶段 1 —— 去掉多余的操作：97040

起始代码对每个（元素，轮）用 36 个槽位：读下标、读值、读节点值（各要一次加法算地址和一次 `load`），1 次异或，
哈希 18 次，算方向 5 次（`%`、`==`、`select`、`*`、`+`），判断是否越界 2 次，写回下标和值 4 次。
36 × 4096 = 147456，再加上读内存头部（14）、装常量（262）和两条 `pause`（2）共 278 个周期，就是 147734。

两个观察让大部分操作可以去掉：

- **所有元素都从根出发**，所以第 `r` 轮时所有元素都在同一层，深度是 `r % (forest_height + 1)`。
  只有在叶子那一层之后才会越界，而且这时所有元素一起回到根。所以不需要逐个判断越界。
- **值和节点地址一直放在草稿区里**，不用每轮读写内存；下标也不用写回（测试不检查）。
  直接维护节点地址 `P + idx`（`P = forest_values_p = 7`），下一层的地址是
  $P + (2\,\mathrm{idx} + 1 + \mathrm{bit}) = 2(P + \mathrm{idx}) + (1 - P) + \mathrm{bit}$，其中 bit 是 `val & 1`。

这样每个（元素，轮）只剩 1 次 `load`、1 次异或、18 次哈希、4 次算地址，共 24 个操作。

```py
def gen_scalar(pr, height, n_nodes, batch, rounds):
    P = 7                              # forest_values_p: the tree follows the 7-word header
    VP = P + n_nodes + batch           # inp_values_p
    one, two, fix = pr.const(1), pr.const(2), pr.const(1 - P)
    ptr = [pr.const(VP + i) for i in range(batch)]
    val = [pr.load(ptr[i]) for i in range(batch)]
    adr = [pr.const(P)] * batch        # address of the current node, P + idx
    for r in range(rounds):
        depth = r % (height + 1)       # every item starts at the root, so all items share one depth
        for i in range(batch):
            a = pr.alu("^", val[i], pr.load(adr[i]))
            for op1, c1, op2, op3, c3 in HASH_STAGES:
                a = pr.alu(op2, pr.alu(op1, a, pr.const(c1)), pr.alu(op3, a, pr.const(c3)))
            val[i] = a
            if r == rounds - 1:
                pr.store(ptr[i], a)
            elif depth == height:
                adr[i] = pr.const(P)   # all items are on leaves here, so all of them wrap to the root
            else:
                # P + (2*idx + 1 + bit)  =  2*(P + idx) + (1 - P) + bit
                t = pr.alu("+", pr.alu("*", adr[i], two), fix)
                adr[i] = pr.alu("+", t, pr.alu("&", a, one))
```

这一阶段仍然一个周期只放一个槽位，周期数等于操作总数：97040。只快了 1.52 倍，因为真正的浪费在于每个周期只干一件事。

### 阶段 2 —— 把独立的操作放进同一个周期：7690

操作和阶段 1 完全一样，只是交给上面的调度器打包。256 个元素彼此独立，可以并行的操作很多，所以瓶颈变成 `alu`：
92160 个标量运算 ÷ 12 = 7680。实测 7690，几乎贴着下界。

这时草稿区用了 1294 个字（上限 1536）：256 个值、256 个地址，再加上很多同时存活的中间值。标量写法已经没有什么余地了。

### 阶段 3 —— 向量化：2084

把 256 个元素分成 32 块，每块 8 个，用向量运算一次处理一块。开始时每块用一条 `vload` 读入值，最后用一条 `vstore` 写回。

哈希里第 0、2、4 步的形式是 `(a + c) + (a << k)`，在模 $2^{32}$ 下等于 $a \cdot (1 + 2^k) + c$，正好是一条 `multiply_add`。
所以一次哈希只要 12 个向量操作：

```py
def vhash(x):                                                # stage 3 version
    a = pr.madd(x, K(1 + (1 << 12)), K(c0))                  # (a + c0) + (a << 12)
    a = pr.valu("^", pr.valu("^", a, K(c1)), pr.valu(">>", a, K(19)))
    a = pr.madd(a, K(33), K(c2))                             # (a + c2) + (a << 5)
    a = pr.valu("^", pr.valu("+", a, K(c3)), pr.valu("<<", a, K(9)))
    a = pr.madd(a, K(9), K(c4))                              # (a + c4) + (a << 3)
    return pr.valu("^", pr.valu("^", a, K(c5)), pr.valu(">>", a, K(16)))
```

（`K(v)` 是“值为 `v` 的常量向量”：一次 `const` 加一次 `vbroadcast`，整个程序里只做一次。）

节点值就麻烦了：8 个元素在树上的位置各不相同，没有 gather，只能用 8 条 `load` 分别读进向量寄存器的 8 个位置。
地址向量用 `multiply_add` 更新：先算 `t2 = adr * 2 + (1 - P)`，再加上 `val & 1`。

```py
node = pr.reg(VLEN)
for lane in range(VLEN):
    pr.load_lane(node, lane, (st.adr, lane))       # one scalar load per lane
x = pr.valu("^", st.val, node)
t2 = pr.madd(st.adr, K(2), K(1 - P))               # next address = t2 + (val & 1)
```

现在瓶颈变成了 `load`：4096 次 gather，每周期 2 次，至少 2048 个周期。实测 2084。
这就是博客里说的“内存带宽瓶颈”：向量运算再省也没用，加载跟不上。

### 阶段 4 —— 树的顶上几层不读内存：1548

关键观察还是“同一轮里所有元素在同一层”。第 0 层只有 1 个节点，第 1 层 2 个，第 2 层 4 个。
这 7 个节点值可以开头用一条 `vload`（节点 0–7）一次读进来，之后只要根据已经走过的方向“算出”该用哪个值。

方向就是每轮哈希结果的最低位。设第 1 层两个节点的值是 $v_1, v_2$，第一次的方向是 $b_0$，那么节点值是
$v_1 + b_0 (v_2 - v_1)$，一条 `multiply_add`。第 2 层有两个方向，节点值是 $b_0, b_1$ 的多重线性多项式，
三条 `multiply_add`。差值 $v_2 - v_1$ 这类系数只和树有关，在标量 ALU 上算一次、广播一次就够了。

```py
def pick(depth, bits, pre):
    """bits[j] = (vector q, polarity p): the direction taken at depth j is q ^ p.
    The position inside the level is the directions read as a binary number, first one most significant."""
    leaves = [leaf((1 << depth) - 1 + o, pre) for o in range(1 << depth)]   # scalar tree values
    if depth == 0:
        return pr.bcast(leaves[0])

    def lin(vals, k):              # multilinear polynomial in bits[0..k-1] with scalar coefficients
        if k == 0:
            return pr.bcast(vals[0])
        q, p = bits[k - 1]         # peel the latest direction: even / odd entries
        base, other = (vals[1::2], vals[0::2]) if p else (vals[0::2], vals[1::2])
        diff = [pr.alu("-", o, b, cache=True) for o, b in zip(other, base)]
        return pr.madd(q, lin(diff, k - 1), lin(base, k - 1))

    return lin(leaves, depth)
```

（极性 `p` 从阶段 6 开始才会是 1，这里先当它是 0。）

到了第一个要从内存读的层（第 3 层），把记下的方向位换算成地址向量：常数 $P + 2^d - 1$ 加上各个方向位乘以权重
（`addr_from_bits`，用 `+` 和 `multiply_add`）。

每趟下树（第 0–10 轮是第一趟，第 11–15 轮是第二趟）都有 3 轮不用 gather，16 轮里共 6 轮，gather 降到 2560 次，
`load` 下界 1306，`valu` 下界 1348。可实测是 1548，离下界很远。

查看排出来的程序就能发现问题：前 153 个周期里没有任何 gather；从第一次到最后一次 gather 之间，还有 104 个周期 `load` 槽位没用满。
问题出在交给调度器的顺序。

### 阶段 5 —— 按“一块接一块”的顺序交给调度器：1355

操作和阶段 4 完全相同，只改了交给调度器的顺序：从“按轮”（第 0 轮的 32 块，第 1 轮的 32 块……，也就是参考实现的循环顺序）
改成“按块”（第 0 块的 16 轮，第 1 块的 16 轮……）。

```py
steps = [(v, r) for v in range(n_blocks) for r in range(rounds)]   # block by block
if cfg["order"] == "round":
    steps.sort(key=lambda vr: (vr[1], vr[0]))                      # stages 3-4: round by round
for v, r in steps:
    step(v, r)
```

列表调度是贪心的：按顺序把每个操作塞进最早能放的周期。按轮的顺序下，最先处理的是所有块前 3 轮的纯向量运算，
它们先占满了开头的 `valu` 槽位，能“解锁”第一批 gather 的操作反而排在后面。按块的顺序下，第 0 块很快走到第 3 层开始 gather，
后面的块自然地填进空隙，相当于自动做了软件流水（software pipelining）。

结果：前 39 个周期之后就开始 gather，实测 1355，过了全部 8 个门槛。草稿区里分配的向量也从 121 个降到 77 个。
阶段 3 改成按块顺序则没有变化（我们测过，仍是 2084）：它从第 2 个周期就开始 gather，到最后一次 gather 之间只有 1 个周期 `load` 没用满。

### 阶段 6 —— 减少每轮的向量操作：1329

三处改写：

1. **合并哈希的第 2、3 步。** 第 2 步得到 $b = 33a + c_2$；第 3 步要的是 `(b + c3) ^ (b << 9)`。
   两边都是“$a$ 乘常数加常数”：$33a + (c_2 + c_3)$ 和 $(33 \cdot 2^9) a + c_2 \cdot 2^9$。所以两条 `multiply_add` 加一次异或，
   3 个操作代替原来的 4 个。
2. **推迟最后的 `^ c5`。** 第 5 步是 `(a ^ c5) ^ (a >> 16)`。如果下一层的节点值是预先读进来的，就先不做 `^ c5`，
   得到的是 `hash ^ c5`；把预读的节点值也事先异或上 `c5`（标量，各做一次），下一轮
   `(hash ^ c5) ^ (node ^ c5) = hash ^ node`，正好抵消。代价是方向位反了（`c5 = 0xB55A4F09` 是奇数），
   所以每个方向位都带一个极性 `p`，选节点和算地址时按 `q ^ p` 解释。
3. **位运算挪到标量 ALU。** `val & 1` 和“地址加方向位”改成每个位置一条标量运算（8 条 `alu` 代替 1 条 `valu`）。
   `alu` 每周期有 12 个槽位，之前几乎闲着。

```py
def vhash(x, skip_c5):
    a = pr.madd(x, K(1 + (1 << 12)), K(c0))
    a = pr.valu("^", pr.valu("^", a, K(c1)), pr.valu(">>", a, K(s1)))
    # stage 2 gives b = 33a + c2; stage 3 wants (b + c3) ^ (b << 9): both are a*const + const
    a = pr.valu("^", pr.madd(a, K(33), K(c2 + c3)), pr.madd(a, K(33 << s3), K(c2 << s3)))
    a = pr.madd(a, K(9), K(c4))
    if skip_c5:
        return pr.valu("^", a, pr.valu(">>", a, K(s5)))              # = hash ^ c5
    return pr.valu("^", pr.valu("^", a, K(c5)), pr.valu(">>", a, K(s5)))

def low_bit(val):                  # val & 1, one scalar ALU op per lane
    q = pr.reg(VLEN)
    for lane in range(VLEN):
        pr.alu_lane("&", q, lane, (val, lane), pr.const(1))
    return q

def leaf(i, pre):                  # tree value i as a scalar operand, pre-XORed with c5 if needed
    k = i // VLEN
    if k not in tree_blocks:
        tree_blocks[k] = pr.vload(pr.const(P + VLEN * k))
    s = (tree_blocks[k], i % VLEN)
    return pr.alu("^", s, pr.const(C5), cache=True) if pre else s
```

`valu` 下界从 1348 降到 1119，瓶颈回到 `load`（1307），实测 1329。这一步本身收益不大：去掉三处改写中的任何一处，
阶段 6 仍然是 1329–1332（我们的小脚本测的）。它们的作用要到后面才显出来：在阶段 9 的配置里分别去掉它们，
会从 1040 变成 1107（不合并第 2、3 步）、1065（不推迟 `^ c5`）、1130（位运算不挪到 ALU）。

### 阶段 7 —— 第 3 层也不读内存，选节点交给 flow 引擎：1107

第 3 层有 8 个节点，用多重线性多项式要 7 条 `multiply_add`，`valu` 会成为瓶颈（我们测过：这样做是 1193，`valu` 下界 1185）。

但是 `flow` 引擎一直闲着。它每周期只有 1 个槽位，可 `vselect` 一次就能对 8 个位置做选择。
用一棵选择树：先按第一个方向在前一半和后一半之间选，再按第二个方向选，依此类推。第 $d$ 层要 $2^d - 1$ 条 `vselect`。
第 1、2 层也一起改成这种做法，把 `valu` 腾出来。

```py
cur = [pr.bcast(s) for s in leaves]
for q, p in bits:                                            # earliest direction first: halves
    half = len(cur) // 2
    lo, hi = cur[:half], cur[half:]
    cur = [pr.vselect(q, lo[i], hi[i]) if p else pr.vselect(q, hi[i], lo[i]) for i in range(half)]
return cur[0]
```

每趟下树要 1 + 3 + 7 = 11 条 `vselect`，两趟、32 块，共 704 条。gather 只剩 16 轮里的 8 轮，`load` 下界 1053。实测 1107。

为什么不把第 4 层也给所有块？那要再加 15 × 2 × 32 条 `vselect`，`flow` 下界变成 1664，实测 1687，反而更慢。

### 阶段 8 —— 用掉剩下的 flow 槽位：1064

阶段 7 里 `flow` 只用了 704 个周期，还剩约 400 个。第 4 层的选择每块要 15 条 `vselect`，20 块正好 300 条，`flow` 共 1004。
我们把它用在第二趟：第 15 轮（第 4 层，也是最后一轮）有 20 个块改用选择，少了 160 次 gather，`load` 下界降到 975。

```py
def preloaded(v, r):               # number of levels served from preloaded values, for block v in round r
    trip = 0 if r <= height else 1
    return min(height, top + (1 if v in deep[trip] else 0))

# deep = (0, 20): no extra blocks on the first trip, 20 blocks (spread evenly) on the second
deep = [{(i * n_blocks) // n for i in range(n)} for n in cfg.get("deep", (0, 0))]
```

实测 1064。我们也用自己的小脚本试了别的分法：32 块全给是 1207（`flow` 下界 1184），20 块放在第一趟是 1091，
都不上第 4 层（也就是阶段 7）是 1107。放在第二趟为什么更好，我们没有细查。

### 阶段 9 —— gather 之后的异或交给标量 ALU：1040

gather 读进来的本来就是一个个标量。与其先拼成向量再做一次向量异或，不如每读进一个就在 ALU 上和对应位置的值异或。
每个需要 gather 的（块，轮）省 1 条 `valu`，多 8 条 `alu`：`valu` 操作从 6318 降到 6082，`alu` 从 5182 升到 7070。

```py
x = pr.reg(VLEN)
for lane in range(VLEN):
    pr.alu_lane("^", x, lane, (st.val, lane), pr.load((st.adr, lane)))
```

实测 1040。

### 为什么停在 1040

现在 `load`（975）、`valu`（1014）、`flow`（1004）三个下界都已经接近实测值。看排出来的程序：

- 前 47 个周期里没有 gather。第一次 gather 要等第 0 块先算完 4 轮（第 0–3 层都不读内存），这 4 轮一轮接一轮，没法并行。
- 从第一次到最后一次 gather 共 983 个周期，其中只有 39 个周期 `load` 没用满。
- 最后一次 gather 之后，还要 10 个周期算完最后的哈希并写回。

所以 1040 = 47（开头）+ 983（gather 期间）+ 10（结尾）。再往下走，要么缩短开头，要么继续减少 gather，
但减少 gather 需要更多的选择操作，而 `flow` 和 `valu` 都快满了。

博客说 Opus 4.5 找到了“利用题目结构绕过带宽瓶颈的巧妙办法”，但没说是什么。阶段 4、7、8 是一种利用题目结构绕过 gather 的做法，
不一定就是博客指的那一个。

### 如果要求写回下标

测试不检查 `indices`，所以上面的版本都不写回。设置 `PERF_STORE_IDX=1`（阶段 3–9 可用）会在最后一轮把下标也算出来写回：
阶段 9 实测 1055（`tests/submission_tests.py` 全部通过；我们另外在 10 种规模上核对过写回的下标）。

### 没有做的方向（都没有实现，也没有测过）

- **缩短开头的 47 个周期**：例如让最先开始的几个块在第一趟少预读一层、早点开始 gather，后面的块多预读；或者调整调度优先级，
  让第 0 块的关键路径先走。
- **继续平衡引擎**：`alu` 下界只有 590，还能接手一些向量运算（1 条 `valu` 相当于 8 条 `alu`）；
  第 4 层的选择也可以一部分用 `vselect`、一部分用 `multiply_add`，让 `flow` 和 `valu` 一起分担。
- **更好的调度器**：按关键路径定优先级、随机重启、局部搜索。上面的表说明调度质量影响很大（阶段 9：0 遍 1140，3 遍 1040）。
- **少用 load 槽位装常量**：现在大约有 25 条 `const` 占用 `load` 槽位，可以用 ALU 从少数几个常量推出其余的。
- **不写死布局**：现在直接用了 `forest_values_p = 7` 和具体规模。改成从内存头部读取会多花几个周期，但更通用。

### 追问

**为什么完全展开、不用循环和跳转？**
模拟器不限制程序长度，展开后周期数就是指令束个数，调度器可以跨轮、跨块随意重排。
如果程序长度有限制，就要用循环加跳转（每次跳转占一个 `flow` 槽位），再手工做软件流水。

**写死 `forest_values_p = 7` 和输入规模，算不算钻空子？**
规则只要求不改 `tests/`。`build_kernel` 拿到的就是具体规模，内存布局由冻结的 `build_mem_image` 决定，所以这是允许的。
面试时最好主动说明这个假设，并说清楚改成从内存头部读取要多少代价。我们的生成器对其他规模也是对的：
每个阶段在 10 种规模（高度 1–10、轮数 1–23、批大小 8–256）上各跑 2 组随机输入，结果都和参考实现一致。
（向量写法要求批大小是 8 的倍数。）

**怎么证明结果可信、没有改测试？**
对比 `tests/` 和原件（`diff -r`，或者 `git diff`），确认 `problem.py` 里的 `N_CORES` 等也没有动，然后只看
`python tests/submission_tests.py` 打印的周期数。计分时的机器来自 `tests/frozen_problem.py` 这份冻结副本，改 `problem.py` 里的机器不会改变计分。

**草稿区够用吗？**
阶段 9 用了 56 个字加 149 个向量，共 1248 个字（上限 1536）。如果阶段 9 仍按“按轮”的顺序交给调度器，
会同时存活太多向量，需要 66 个字加 201 个向量，超过上限，程序根本生成不出来。顺序既影响速度，也影响草稿区。

**如果加载有延迟（像站内那道练习那样），要改什么？**
调度器里“依赖之后的下一个周期”要改成“依赖之后再过若干个周期”，即按延迟加边权。gather 要更早发出，
同时在飞的块会更多，草稿区压力更大；按块的顺序和来回排几遍的做法仍然适用。

**2 小时的限时版本里，做到哪一步算好？**
按 Readme，人类 2 小时内的最好成绩约为 1790，而且是从 18532 出发的。把打包、向量化、顶上几层不读内存这几件事做对，
就已经很强了；后面几步需要反复看下界和调度结果，更像不限时的打磨。

<details>
<summary>验证代码（可运行）</summary>

下面的代码不依赖仓库，只检查本页里能用代码验证的说法：哈希的改写、“所有元素同层”的观察、阶段 1 的地址公式、
两种选节点的方式，以及文中用到的操作计数。周期数本身只能在仓库里实测，见上面的命令。

```python
import random

random.seed(2026)
M = 2**32
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]
OPS = {"+": lambda x, y: x + y, "^": lambda x, y: x ^ y,
       "<<": lambda x, y: x << y, ">>": lambda x, y: x >> y}
C = [h[1] for h in HASH_STAGES]
C5 = C[5]


def stage(a, k):
    op1, c1, op2, op3, c3 = HASH_STAGES[k]
    return OPS[op2](OPS[op1](a, c1) % M, OPS[op3](a, c3) % M) % M


def myhash(a):
    for k in range(6):
        a = stage(a, k)
    return a


def madd(a, b, c):                       # one lane of multiply_add
    return (a * b + c) % M


def vhash_lane(x, fuse23=True, skip_c5=False):
    """One lane of the reference solution's vector hash, using only valu operations."""
    a = madd(x, 1 + (1 << 12), C[0])
    a = (a ^ C[1]) ^ (a >> 19)
    if fuse23:
        a = madd(a, 33, C[2] + C[3]) ^ madd(a, 33 << 9, C[2] << 9)
    else:
        a = madd(a, 33, C[2])
        a = ((a + C[3]) % M) ^ ((a << 9) % M)
    a = madd(a, 9, C[4])
    return a ^ (a >> 16) if skip_c5 else (a ^ C5) ^ (a >> 16)


xs = [random.randrange(M) for _ in range(20000)] + [0, 1, M - 1, 2**31]

# Hash stages 0, 2, 4 are (a + c) + (a << k) = a * (1 + 2^k) + c: one multiply_add each.
for k in (0, 2, 4):
    shift = HASH_STAGES[k][4]
    assert all(stage(a, k) == madd(a, 1 + (1 << shift), C[k]) for a in xs)

# The vector hash (12 valu ops; 11 with stages 2 and 3 fused) equals myhash.
assert all(vhash_lane(x, fuse23=False) == myhash(x) for x in xs)
assert all(vhash_lane(x) == myhash(x) for x in xs)

# Deferring the final "^ C5": the result is hash ^ C5, and pre-XORing node values with C5 cancels it.
assert C5 % 2 == 1
for x in xs[:5000]:
    h, node = myhash(x), random.randrange(2**30)
    assert vhash_lane(x, skip_c5=True) == h ^ C5
    assert (h ^ C5) ^ (node ^ C5) == h ^ node
    assert ((h ^ C5) & 1) == 1 - (h & 1)          # C5 is odd: the direction bit comes out inverted


def reference(tree, values, rounds):
    """reference_kernel2 restated on Python lists; also records the depths after every round."""
    n, values, idx, depths = len(tree), list(values), [0] * len(values), []
    for _ in range(rounds):
        for i in range(len(values)):
            v = myhash(values[i] ^ tree[idx[i]])
            j = 2 * idx[i] + (1 if v % 2 == 0 else 2)
            idx[i] = 0 if j >= n else j
            values[i] = v
        depths.append({(k + 1).bit_length() - 1 for k in idx})
    return values, idx, depths


def stage1_form(tree, values, rounds, height):
    """The rewrite of stage 1: keep P + idx, no wrap test, no index write-back."""
    P = 7
    mem = [0] * P + list(tree)
    out = []
    for v0 in values:
        val, adr = v0, P
        for r in range(rounds):
            depth = r % (height + 1)
            val = myhash(val ^ mem[adr])
            if depth == height:
                adr = P                                  # every item is on a leaf: back to the root
            else:
                adr = 2 * adr + (1 - P) + (val & 1)      # P + (2*idx + 1 + bit)
        out.append(val)
    return out


for height, rounds, batch in [(10, 16, 64), (3, 9, 16), (1, 5, 8), (0, 3, 8), (4, 23, 8)]:
    tree = [random.randrange(2**30) for _ in range(2 ** (height + 1) - 1)]
    vals = [random.randrange(2**30) for _ in range(batch)]
    ref_vals, _, depths = reference(tree, vals, rounds)
    # all items share one depth: after round r it is (r + 1) % (height + 1)
    assert depths == [{(r + 1) % (height + 1)} for r in range(rounds)]
    assert stage1_form(tree, vals, rounds, height) == ref_vals


# Level picks. bits[j] = (q, p): the stored bit q, and polarity p (p = 1 means q is inverted).
def pick_madd(level_vals, bits):
    """Multilinear form: one multiply_add per inner node of the recursion, coefficients are scalars."""
    def lin(vals, k):
        if k == 0:
            return vals[0]
        q, p = bits[k - 1]
        base, other = (vals[1::2], vals[0::2]) if p else (vals[0::2], vals[1::2])
        diff = [(o - b) % M for o, b in zip(other, base)]
        return madd(q, lin(diff, k - 1), lin(base, k - 1))
    return lin(level_vals, len(bits))


def pick_vselect(level_vals, bits):
    """Selection tree on the flow engine: the earliest direction picks a half."""
    cur, n_sel = list(level_vals), 0
    for q, p in bits:
        half = len(cur) // 2
        lo, hi = cur[:half], cur[half:]
        cur = [(lo[i] if q else hi[i]) if p else (hi[i] if q else lo[i]) for i in range(half)]
        n_sel += half
    return cur[0], n_sel


def addr_from_bits(bits, depth, base):
    const = base + (1 << depth) - 1
    acc = const
    for j, (q, p) in enumerate(bits):
        w = 1 << (depth - 1 - j)
        acc += w * (1 - q if p else q)
    return acc


for depth in range(0, 6):
    level = [random.randrange(2**30) for _ in range(1 << depth)]
    for _ in range(200):
        dirs = [random.randrange(2) for _ in range(depth)]
        pol = [random.randrange(2) for _ in range(depth)]
        bits = [(d ^ p, p) for d, p in zip(dirs, pol)]
        offset = int("".join(map(str, dirs)) or "0", 2)
        assert pick_madd(level, bits) == level[offset]
        val, n_sel = pick_vselect(level, bits)
        assert val == level[offset] and n_sel == (1 << depth) - 1
        assert addr_from_bits(bits, depth, base=7) == 7 + (1 << depth) - 1 + offset

# Starter code: 36 non-debug slots per (item, round), one per cycle, plus setup.
per_item = {"load idx (add + load)": 2, "load val (add + load)": 2, "load node (add + load)": 2,
            "xor": 1, "hash 6 x 3": 18, "direction (% == select * +)": 5, "wrap (< select)": 2,
            "store idx (add + store)": 2, "store val (add + store)": 2}
assert sum(per_item.values()) == 36
consts = {0, 1, 2} | {h[1] for h in HASH_STAGES} | {h[4] for h in HASH_STAGES} | set(range(256))
setup = 7 * 2 + len(consts) + 2                   # header loads, const loads, two pauses
assert (len(consts), setup) == (262, 278)
assert 36 * 16 * 256 + setup == 147734

# Stage 1: per item 14 rounds x 24 + leaf round 20 + last round 21 (with the store).
assert 14 * 24 + 20 + 21 == 377
n_consts = 3 + 256 + 1 + 12                       # 1, 2, 1-P; 256 value addresses; P; hash constants
assert 377 * 256 + 256 + n_consts == 97040        # + 256 value loads
alu_ops = 256 * (16 * 19 + 14 * 4)
assert alu_ops == 92160 and -(-alu_ops // 12) == 7680   # stage 2 lower bound

# Stage 3: every (block, round) gathers 8 node values; two loads per cycle.
assert 32 * 16 * 8 == 4096 and 4096 // 2 == 2048
# Stage 4: rounds 0-2 and 11-13 need no gather.
assert sum(1 for r in range(16) if r % 11 >= 3) * 256 == 2560
# Stage 7: levels 1-3 by vselect on both trips; stage 8 adds 15 per block for 20 blocks.
assert 32 * 2 * (1 + 3 + 7) == 704 and 704 + 20 * 15 == 1004
# Stage 8/9 gathers: first trip rounds 4-10 for 32 blocks, second trip round 15 for 12 blocks.
gathers = 8 * (32 * 7 + 12 * 1)
assert gathers == 1888 and 6318 - 6082 == gathers // 8 and 7070 - 5182 == gathers
assert -(-1949 // 2) == 975 and -(-6082 // 6) == 1014
for cycles, speedup in [(97040, 1.52), (7690, 19.21), (2084, 70.89), (1548, 95.44), (1355, 109.03),
                        (1329, 111.16), (1107, 133.45), (1064, 138.85), (1040, 142.05)]:
    assert round(147734 / cycles, 2) == speedup
assert round(147734 / 18532, 2) == 7.97
print("all checks passed")
```

</details>

</details>
