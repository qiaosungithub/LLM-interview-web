# Anthropic：能不能用 AI，在什么环境里面试

**一句话：默认都不能用 AI，除非那一轮明确告诉你可以。** 写代码的轮次都在能运行代码的环境里进行，面试官看着你写、看着你跑；但除了在线测评，多数时候不给现成的测试。

下面每一条都标了出处：**官方**指 Anthropic 自己公开写的（原文于 2026-11-17 重新打开核对过），**候选人反馈**指收集到的面经里多人提到的说法，没有官方文字可以对照。

## 各环节能不能用 AI

| 环节 | 能不能用 AI | 出处 |
| --- | --- | --- |
| 准备面试 | 可以 | 官方 |
| 写简历、答申请问题 | 先自己写初稿，再用 Claude 润色；个别职位页面要求整个申请过程都不用 AI，以你投的那个职位页面为准 | 官方 |
| 在线测评（CodeSignal） | 不可以 | 官方默认规则 |
| 电话面、现场面的编程轮 | 不可以；查文档、搜索是允许的 | 官方 |
| “AI 辅助编程”轮（评审 PR 并改进） | 可以，用环境里提供的 Claude Code | 候选人反馈 |
| 系统设计、设计文档评审 | 不可以 | 官方默认规则 |
| Take-home | 默认不可以；性能工程方向那一份明确可以 | 官方 |
| 文化面、用人经理面、项目深挖 | 谈话轮，不涉及工具 | — |

官方原文（[给候选人的 AI 使用说明](https://www.anthropic.com/candidate-ai-guidance)）：

- 实时面试：“This is all you–no AI assistance unless we indicate otherwise.”
- Take-home：“Complete these without Claude unless we indicate otherwise.” 以及 “We'll be clear when AI is allowed (example: "You may use Claude for this coding challenge").”
- 明确列为不允许的做法：“Having Claude write your code when we haven't explicitly allowed it”。
- 准备阶段：“Use Claude to research Anthropic, practice your answers, and prepare questions for us.”

两个例外：

- **性能工程 take-home 可以用 AI。** 官方工程博客写道，候选人说明要求 take-home 默认不用 AI，而 “For this take-home, we explicitly indicate otherwise”；理由是这类耗时长的问题 AI 很难独立做完，所以候选人可以像平时工作那样用 AI，同时仍要展示自己的本事。（[工程博客](https://www.anthropic.com/engineering/AI-resistant-technical-evaluations)）
- **“AI 辅助编程”轮可以用 Claude Code。** 这是候选人反馈，没有官方文字：从 2026 年年中开始，部分电话面和现场面在一个类似 VS Code 加 GitHub 的浏览器环境里进行，里面可以用 Claude Code，任务是在陌生仓库里评审几个开着的 PR，再挑一个改进。

本站每道题旁边的标签就是按这张表标的：绝大多数是“默认禁用 AI”，只有 [借助 AI 助手做 PR 评审](#/anthropic/coding/agentic-pr-review) 和 [性能工程 take-home](#/anthropic/take-home/performance-kernel-optimization) 标的是“可以用 AI”。

## 面试官是跑代码，还是只看代码

**写代码的轮次，代码都是要运行的。** 没有哪一轮是在白板或纯文本里写、只让面试官读的。

| 环节 | 在哪里写 | 代码运行吗 | 给不给测试 |
| --- | --- | --- | --- |
| 在线测评 | CodeSignal，90 分钟，一道分四关的题 | 运行 | **给**：平台自带测试，一关的测试全过才解锁下一关；部分分也算分 |
| 电话面、现场面的编程轮 | Google Meet 共享屏幕 + CodeSignal 或 Colab，通常 55 分钟 | 运行 | **通常不给**：编辑器里可能只有一个空的 `main()`，读文件的辅助函数、测试数据都要自己写 |
| “AI 辅助编程”轮 | 类似 VS Code + GitHub 的浏览器环境，带终端 | 运行 | 仓库里自带什么就是什么 |
| 调用模型的轮次（搭智能体循环、写提示词） | notebook，里面给好调用模型的函数 | 运行 | 不给 |
| 系统设计 | 共享的 Google Doc，或你自己共享的白板（有人用 Excalidraw、Figma） | 不写可运行的代码 | — |
| 设计文档评审 | 读一份写好的设计文档，口头讨论 | 不写代码，也不用画图 | — |
| Take-home | 自己的机器 | 运行 | 性能工程那份自带模拟器和计分测试；研究方向的要自己设计实验 |

其中“Google Meet”“CodeSignal 或 Colab”“可以查资料”是官方说法（[招聘页](https://www.anthropic.com/careers)）：

- “All interviews are conducted over Google Meet”
- “For technical roles, we use live coding tools like Colab and CodeSignal.”
- “You can look things up—just be comfortable with basic syntax and standard libraries so it doesn't eat up your time.”

其余细节（55 分钟、共享屏幕、不给测试、空的 `main()`、设计轮用什么工具）都来自候选人反馈。在线测评的标准版不需要开摄像头；Fellows 项目的测评有摄像头监考，并且不能上网搜索。

语言方面，Python 最保险。有候选人事先和招聘联系人约好后用了 Java 或 TypeScript，也有轮次只能用 Python。

## 这对练习意味着什么

- **在线测评题（题目以“在线测评”开头）**：按关卡顺序做，上一关没写对不要看下一关；掐 90 分钟。真实环境会替你跑测试，所以练的时候把题面里的例子当测试自己跑一遍。
- **电话面、现场面的编程题**：真实环境不给测试，所以练习时**自己写几条用例并运行**本身就是要练的一部分。本站把答案存成 `answer.py`，可以直接在本机运行。
- **两道可以用 AI 的题**：就按可以用 AI 的方式练，重点是你能不能为助手写出的每一行负责。
- 其余所有题：练的时候把 AI 助手和自动补全关掉。

## 没有核实到的

- “AI 辅助编程”轮的具体规则只有候选人转述，没有官方页面。
- 在线测评里“能不能上网搜索”，只有 Fellows 那一版有明确反馈（不能）；标准版没有找到明确说法。
- 各团队的 take-home 内容并不统一，是否允许 AI 以你收到的作业说明为准。
