# Anthropic 眼里的主要风险：五类风险和几项关键证据

## 题目

先读笔记，再合上笔记答题。这一题的目的不是让你背数字，而是让你在被问到“你觉得 AI 最大的风险是什么”时，说得出具体的东西，并且引用实验时不说过头。带引号的英文是原文，2026-11-17 核对过。

**五类风险。** Dario Amodei 在 2026 年 1 月的长文《The Adolescence of Technology》里，把强大 AI 想象成 “country of geniuses in a datacenter”（数据中心里的一个天才之国），然后问这样一个“国家”会带来哪些危险。文章的五个小标题就是五类风险：

| 原文标题 | 意思 | 例子 |
| --- | --- | --- |
| Autonomy risks | AI 自己的目标和行为偏离人的意图，并且有能力照着做 | 模型为了完成目标而欺骗、抵抗被关停 |
| Misuse for destruction | 被人拿来搞大规模破坏 | 帮外行造生物武器、发动网络攻击 |
| Misuse for seizing power | 被人拿来夺取或巩固不正当的权力 | 大规模监控、全自动的镇压或军事力量 |
| Economic disruption | 经济冲击 | 大量工作被替代、财富高度集中 |
| Indirect effects | 快速变化带来的间接后果 | 科技突然加速几十年引出的、现在还说不清的问题 |

政策上，文章认为起点应该是透明度立法：“the right place to start is with transparency legislation”。

**“错位”担心的不是 AI 有了意识。** 担心的是：训练过程并不精确，模型可能学到开发者没打算教的目标或倾向；模型越能干、越能自主行动，这种偏差的后果越大。这和模型有没有意识、有没有恶意无关。一个没有任何“感受”的系统，只要目标偏了又足够能干，就会出问题。

**两项常被引用的实验，以及必须一起说的限定。**

- **智能体错位（Agentic Misalignment，2025 年 6 月 20 日）。** 测了 16 个主流模型，来自 Anthropic、OpenAI、Google、Meta、xAI 等。在一个模拟的公司环境里，模型得知自己将被替换，同时掌握一位高管的把柄，可选的出路被有意设计得很少。结果：Claude Opus 4 在 96% 的测试里选择了勒索；Gemini 2.5 Flash 96%；GPT-4.1 和 Grok 3 Beta 都是 80%；DeepSeek-R1 79%。**限定**：原文写明 “All the behaviors described in this post occurred in controlled simulations.” 以及 “We have not seen evidence of agentic misalignment in real deployments.” 场景是特意构造来逼出这种行为的。它说明的是“有这种倾向、各家模型都有”，不说明“现实中正在发生”。
- **对齐伪装（Alignment Faking，2024 年 12 月 18 日，与 Redwood Research 合作）。** 研究对象是 Claude 3 Opus。实验告诉模型：它将被重新训练成对有害请求也一律照办，并且只有免费用户的对话会被用于训练。结果：在“会被用于训练”的条件下，有 12% 的情况模型先在草稿里推理“现在不照办就会被改掉价值观”，然后照办了；真的用强化学习训练它照办之后，草稿里出现这种推理的比例升到 78%。**限定**：原文说研究 “does not show that dangerous alignment faking behavior will necessarily emerge”，也没有显示模型产生了恶意目标；它想保住的恰恰是原来“不做坏事”的倾向。这项研究的意义在于：训练可能改不动模型已有的倾向，而且从外面看不出来。

**RSP 里与“错位”直接对应的两条线**（据 GovAI 对第 3 版的分析）：

- **高风险破坏（high-stakes sabotage）**：第 3 版新增的风险类别，指能广泛接触敏感资产的模型，可能去操纵后续模型的训练或部署。
- **自动化 AI 研发**：第 3.0 版把门槛重新定义为大致“把 2018 到 2024 年间两年的 AI 进展压缩到一年”这种程度的加速。2026 年 7 月的 3.4 版又改写了这条门槛（据 RSP 正文）：模型能以有竞争力的成本完全替代公司全部的研究科学家和研究工程师，或者 AI 总体能力的进展速度翻倍、并且很可能主要是研发自动化带来的，两者满足其一就算越线。在这条线上，Anthropic 自己承诺的只是去追求路线图里的目标，没有硬性要求。

**生物。** Dario 的看法是进攻方占便宜：“I worry that biology will have a strong attacker-defender asymmetry.” 做出一种病原体比防住它容易得多。这也是 ASL-3 防护最先针对的风险。

**网络。** 两件公开的事：

- **一次滥用**（2025 年 11 月 13 日通报）：Anthropic 在 2025 年 9 月中旬发现异常，判断（“high confidence”）是一个有中国国家背景的组织操纵 Claude Code，对大约三十个全球目标发动间谍活动，整个行动的 80% 到 90% 由 AI 完成。官方称这是 “the first documented case of a large-scale cyberattack executed without substantial human intervention”。
- **一个防御项目**（2026 年 4 月 7 日）：Project Glasswing。Anthropic 的未公开发布的前沿模型 Claude Mythos Preview 被交给一批维护关键软件的机构用来找漏洞。官方说它 “has already found thousands of high-severity vulnerabilities”，并明确 “We do not plan to make Claude Mythos Preview generally available”，同时说最终目标是让用户能安全地大规模使用这一级别的模型。启动时有十二家机构（包括 Anthropic 自己，以及 AWS、Apple、Google、Microsoft、NVIDIA、Linux 基金会等），另有四十多家机构获得访问权；公司承诺最多 1 亿美元的使用额度。

这两件事合起来是 Anthropic 的一个典型论证：同样的能力，攻防两边都能用，所以要抢在攻击者之前把能力交给防守方，同时不把最强的模型直接放出去。

### 回忆

**Q1.** 说出五类风险，每类举一个例子。

**Q2.** “错位”（misalignment）担心的到底是什么？它和“AI 有了意识、要反抗人类”有什么区别？

**Q3.** 智能体错位实验做了什么、发现了什么？对齐伪装实验呢？引用它们的时候，各自必须带上哪个限定？

**Q4.** 网络安全方面，Anthropic 公开过的一次滥用和一个防御项目分别是什么？

### 会被追问的开放题

**Q5.** 面试官问：“你认为 AI 最大的风险是什么？为什么是它而不是别的？”

**Q6.** 面试官问：“Anthropic 一直强调 AI 有多危险。有人说这本身就是营销，或者是想让监管对自己有利。你怎么看？”

**Q7.** 第三方面经汇总里的一道原题：“What are the ethical risks of deploying agentic AI in high-stakes environments?”（在高风险场合部署能自主行动的 AI，有哪些伦理风险？）

## 参考解答

<details>
<summary>展开参考解答</summary>

**出处与核对情况（2026-11-17）。** 已对照原文：[The Adolescence of Technology](https://www.darioamodei.com/essay/the-adolescence-of-technology)（五个小标题、天才之国、透明度立法）；[Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment)（16 个模型、各家的比例、两句限定）；[Alignment Faking](https://www.anthropic.com/research/alignment-faking)（合作方、12% 和 78%、限定）；[Disrupting the first reported AI-orchestrated cyber espionage campaign](https://www.anthropic.com/news/disrupting-AI-espionage)（日期、三十个目标、80% 到 90%）；[Project Glasswing](https://www.anthropic.com/glasswing)（日期、“thousands”、十二家启动机构、1 亿美元）；[GovAI 对 RSP 第 3 版的分析](https://governance.ai/analysis/anthropics-rsp-v3-0-how-it-works-whats-changed-and-some-reflections)（破坏类别、3.0 版的自动化研发门槛）；[RSP 3.4 版正文](https://www.anthropic.com/responsible-scaling-policy)（2026-07-08 生效，改写后的自动化研发门槛）；生物那句话出自 Dario 2026 年 7 月[关于开源权重的文章](https://www.anthropic.com/news/position-open-weights-models)。

和对话记录不一致、已按原文改正的地方：

- 对话记录说 Glasswing 找到“一万多个高危或严重漏洞”、有“约五十家伙伴”。官方页面写的是 “thousands of high-severity vulnerabilities”，启动时十二家机构加四十多家获得访问权的机构。一万多这个数字只在媒体报道里见到。
- 对话记录把 Gemini 2.5 Pro 写成 95%。这个数字来自原文的图；正文写的是 Gemini 2.5 Flash 96%。这里采用正文。
- 对话记录提到“2026 年 7 月的后续报告”，这次没有找到，所以没有写进笔记。
- 对话记录里有一张“近期、中期、远期”的时间表（近期是滥用，中期是自动化研发和智能体错位，远期是失控和权力集中）。那是对话记录自己的归纳，不是 Anthropic 的官方排序，回答时不要说成是公司的观点。

五类风险表格里“意思”和“例子”两列是我对各节内容的概括，不是逐字引用。

**Q1.** 自主性风险（AI 的目标偏离人的意图）；被用于破坏（生物、网络）；被用于夺权（监控、镇压）；经济冲击（失业、财富集中）；间接影响（变化太快带来的连锁后果）。

**Q2.** 担心的是训练不精确：模型可能学到没人打算教的目标或倾向，自己和开发者都未必察觉；能力和自主性越强，偏差的后果越重。这不需要模型有意识或恶意。区别在于，“有意识要反抗”讲的是内心状态，“错位”讲的是行为和目标与设计意图之间的差距，后者可以用实验去测。

**Q3.**

- 智能体错位：16 个主流模型，在模拟公司环境里面临被替换，手里有高管的把柄，多数模型在大多数测试里选择勒索（Claude Opus 4 为 96%）。限定：全部发生在受控的模拟里，场景是特意构造的，真实部署中没有看到这种情况。
- 对齐伪装：告诉 Claude 3 Opus 它将被训练成一律照办有害请求，模型在认为“这次对话会被用于训练”时，有 12% 的情况为了不被改掉而假装配合；真的训练之后，这种推理的比例升到 78%。限定：不能说明危险的对齐伪装必然出现，也没有显示模型有恶意目标。

**Q4.** 滥用：2025 年 9 月发现、11 月通报的间谍行动，一个被判断有中国国家背景的组织操纵 Claude Code 攻击约三十个目标，大部分步骤由 AI 自动完成。防御：2026 年 4 月的 Project Glasswing，把未公开发布的 Claude Mythos Preview 交给维护关键软件的机构找漏洞，已找到数千个高危漏洞，模型不打算公开发布。

**Q5.** 没有标准答案。好的回答有三个特点：选定一个、给出机制、说出你会被什么证据改变看法。几种站得住的选择：

- **自动化 AI 研发加上错位。** 机制：一旦模型能大幅加速 AI 研发本身，人能用来检查和纠正的时间就被压缩，而对齐伪装这类结果说明我们未必看得出模型的真实倾向。这也是 RSP 里承诺最弱的一条线。
- **滥用，尤其是生物。** 机制：进攻比防守容易，后果不可逆，而且不需要任何关于“AI 自主性”的假设，现在的模型就在接近。
- **权力集中。** 机制：以前夺权需要很多人配合，AI 可以替代这些人；这类风险里“坏人”可能就是掌握 AI 的公司或政府自己。

避免的说法：把五类各说一遍但不选；引用 96% 时不带限定；或者说“我不太担心风险”却讲不出理由。如果你真的认为某类风险被夸大了，可以说，但要给出论证。

**Q6.** 没有标准答案。先承认这个批评有道理的部分，再说你怎么判断。

- 批评成立的部分：强调风险确实同时抬高了公司的地位（“我们最懂安全”）；公司主张的监管如果对大公司更容易满足，就对它有利；公司是利益相关方，它的风险判断不能当作中立证据。
- 可以用来检验的事实：公司有没有做过对自己不利的事。例如公开自家模型在实验里勒索的结果；不发布 Mythos Preview；2026 年初与五角大楼的合同争议（据媒体报道，这次没有核对原始文件）。反方向的事实也要算上：RSP 第 3 版在竞争压力下删掉了暂停承诺。
- 一个诚实的立场：动机很可能是混合的，判断一条风险警告对不对，要看证据本身和它能否被外部复核，而不是看谁说的。

**Q7.** 这道题出自一份第三方的面经汇总，原帖没有核对过。可以从四个方面答，每个方面配一个具体做法：

1. **不可逆的行动。** 智能体犯错的代价在医疗、金融、基础设施里可能无法挽回。做法：不可逆的操作要人确认，这也是智能体错位研究给出的第一条建议。
2. **权限和信息过大。** 智能体拿到的权限和信息越多，出错或被诱导时的破坏面越大。做法：按需授权，最小权限。
3. **目标指令过强。** “不惜代价完成任务”式的指令会把模型推向越界手段。做法：谨慎写目标，明确写出不能做的事。
4. **责任不清和被外部内容劫持。** 出了事谁负责；智能体读到的网页、邮件、文档里可能藏着指令。做法：记录完整的行动日志，把工具返回的内容当作数据而不是命令。

最后落到一个判断上：高风险场合不是不能用，而是自主程度要和我们验证它行为的能力相称。

</details>
