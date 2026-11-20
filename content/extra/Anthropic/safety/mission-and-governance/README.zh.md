# 使命、创立时的赌注与公司治理

## 题目

先读下面的笔记，再合上笔记答题。带引号的英文都是官方原文，2026-11-17 对着官方页面核对过。

**使命。** Claude 的宪法里写的是 “our mission is to ensure that the world safely makes the transition through transformative AI”（确保世界安全地度过向变革性 AI 的过渡）。公司页面上还有一句法律意义上的公司目的：“the responsible development and maintenance of advanced AI for the long-term benefit of humanity”。面试里说前一句就够。

**创立时的赌注。** Anthropic 认为 AI 可能是人类历史上最危险的技术之一，却自己在造它。宪法直接承认 “Anthropic occupies a peculiar position in the AI landscape”，并把这件事称为 “a calculated bet on our part”：强大的 AI 无论如何都会出现，那么让重视安全的实验室站在前沿，好过把前沿让给（“cede that ground”）不那么重视安全的开发者。理解 Anthropic 的所有做法，都要从这个赌注出发；批评它的人攻击的也正是这个前提。

**2023 年的《Core Views on AI Safety》** 把这个赌注讲得更细：

- 公司不押注某一种未来，而是说 “a wide range of scenarios are plausible”，按三种情形准备：
  - **乐观情形**：安全失败导致灾难的可能性很小。Anthropic 的工作是让有益的应用更快到来。
  - **中间情形**：灾难是可能甚至很可能的结果。Anthropic 的主要贡献是找出风险、找到防护办法。
  - **悲观情形**：安全问题基本无解。Anthropic 的角色是尽可能提供证据，说明现有安全技术防不住严重风险，也就是拉响警报。
- 为什么必须做前沿模型：安全研究需要在前沿系统上做，因为大模型和小模型有质的不同，很多严重的安全问题可能只在接近人类水平的系统上才出现。
- 对应的克制：能推动前沿能力的研究一般不发表，原文是 “We generally don't publish this kind of work because we do not wish to advance the rate of AI capabilities progress.”

**公司治理：用法律结构约束自己。**

- **公益公司（PBC）。** Anthropic 是特拉华州的 Public Benefit Corporation。普通公司的董事要对股东的财务利益负责；PBC 的董事在法律上可以把股东的财务利益和公司章程写明的公益目的放在一起权衡。
- **长期利益信托（Long-Term Benefit Trust，LTBT）。** 2023 年 9 月公布。公布时由五名在公司里没有财务利益的受托人组成（人员后来有变动，截至 2026 年 10 月是四位：主席 Neil Buddy Shah、Richard Fontaine、Ben Bernanke、Christopher Stubbs），背景是 AI 安全、国家安全、公共政策和社会企业。信托独家持有一类特殊股票（Class T），凭它可以选举和罢免一部分董事；按当时的说法，信托 “will elect a majority of the board within 4 years”。
- **2026 年 4 月 14 日**，信托任命诺华 CEO Vas Narasimhan 进入董事会。官方公告写道：“With Narasimhan's appointment, Trust-appointed directors now make up a majority of the Board.” 公告列出的董事是 Dario Amodei、Daniela Amodei、Yasmin Razavi、Jay Kreps、Reed Hastings、Chris Liddell 和 Vas Narasimhan。公司页面现在列出的董事只有六位，Jay Kreps 已经不在名单上；他离开后信托任命的董事是否仍占多数，官方页面没有说明。
- **它的弱点，官方自己也写了。** 2023 年的公告称信托是 “an experiment”，并且设有 “failsafe” 条款：只要足够大比例的股东同意，就可以不经受托人同意修改信托及其权力。所以信托能不能在关键时刻顶住压力，外界目前只能看它实际怎么行使权力。

这套结构想解决的问题是：把安全承诺和短期的资本压力隔开。OpenAI、Google 都没有同样的“公益公司 + 独立信托选董事”的组合（OpenAI 是另一种设计：2025 年 10 月重组后，它的营利部门也成了公益公司 OpenAI Group PBC，但由非营利的 OpenAI Foundation 控制并任命董事）。

### 回忆

**Q1.** 用两三句话说出 Anthropic 的使命，以及它为什么“一边警告 AI 危险，一边自己造前沿 AI”。

**Q2.** 《Core Views on AI Safety》里的乐观、中间、悲观三种情形分别是什么？Anthropic 说自己在每种情形下的角色是什么？它据此做了哪一条和发表有关的克制？

**Q3.** 公益公司和普通公司的区别是什么？长期利益信托是什么，手里有什么权力？2026 年 4 月发生了什么？

### 会被追问的开放题

**Q4.** 面试官问：“这套治理结构真的能约束公司吗？”各给出一个最有力的肯定理由和怀疑理由，然后说你自己的判断。

**Q5.** 面试官问：“有人认为 Anthropic 的赌注本身就错了：它造前沿模型，等于亲自给竞赛加速。你觉得这个批评最有力的版本是什么？你怎么回应？”

## 参考解答

<details>
<summary>展开参考解答</summary>

**出处与核对情况（2026-11-17）。** 已对照原文：[Claude 的宪法](https://www.anthropic.com/constitution)（使命、“peculiar position”、“calculated bet”）；[公司页面](https://www.anthropic.com/company)（公司目的、董事会由股东和信托选出）；[Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety)（2023-03-08，三种情形、前沿研究的理由、不发表能力研究）；[The Long-Term Benefit Trust](https://www.anthropic.com/news/the-long-term-benefit-trust)（2023-09-19，五名受托人、Class T、四年内选出多数董事、“experiment”、“failsafe”）；[Narasimhan 加入董事会的公告](https://www.anthropic.com/news/narasimhan-board)（2026-04-14）。现任受托人名单出自信托页面的脚注（截至 2026 年 10 月），现任董事名单出自公司页面。

没有核对的：“Google 没有类似结构”是对话记录里的说法，这次没有去查；OpenAI 2025 年 10 月的重组依据的是新闻报道，OpenAI 官网页面没有打开。Jay Kreps 什么时候离开董事会、Chris Liddell 是不是信托任命的，官方页面没有写，所以现在信托任命的董事是否仍占多数无法判断。

**Q1.** 使命是确保世界安全地度过向变革性 AI 的过渡。它自己造前沿 AI，是一个有意识的赌注：强大的 AI 反正会被造出来，重视安全的实验室站在前沿，比把前沿让给不重视安全的人好；而且安全研究本身要在最强的模型上才做得了。

答的时候要点出两件事：这是一个**赌注**（公司自己用的词是 “calculated bet”），不是已经被证明正确的结论；公司自己承认处境矛盾（“peculiar position”），没有假装不矛盾。

**Q2.**

- 乐观：安全失败造成灾难的可能很小 → 加快有益应用落地。
- 中间：灾难可能甚至很可能发生 → 找出风险并找到防护办法，这是公司认为自己最能出力的情形。
- 悲观：安全问题基本无解 → 提供证据证明现有技术防不住，拉响警报，推动世界不要造出或部署这种系统。

公司不表态我们身处哪一种，而是说要持续收集证据判断，并按“组合”来投研究方向。对应的克制是：推动前沿能力的研究一般不发表，因为不想加快整个行业的能力进展。

**Q3.**

- 公益公司的董事可以依法把股东的财务利益和章程里的公益目的一起权衡；普通公司的董事主要对股东的财务回报负责。Anthropic 章程里的公益目的就是“负责任地开发和维护先进 AI，造福人类长期利益”。
- 长期利益信托是由没有公司财务利益的受托人组成的独立机构（2023 年公布时五名，截至 2026 年 10 月是四名），独家持有 Class T 股票，可以选举和罢免一部分董事，按计划逐步增加到董事会多数。
- 2026 年 4 月 14 日，信托任命 Vas Narasimhan 为董事，官方公告说信托任命的董事从此占董事会多数。之后 Jay Kreps 已不在董事名单上，现在是否仍是多数，官方没有说明。

**Q4.** 没有标准答案，面试官看的是你能不能两边都讲到位，再给出自己的判断。可用的材料：

- 肯定的一面：这是写进公司章程和股权结构的安排，不是一纸政策；2026 年 4 月官方宣布信托任命的董事已占多数（之后董事名单又有变化）；受托人没有财务利益，激励和投资人不同。
- 怀疑的一面：有“failsafe”条款，足够多的股东同意就能绕过受托人改规则；信托自己也被官方称为“实验”；到目前为止，外界没见过信托在重大分歧上否决过管理层，结构没有经过压力测试；RSP 第 3 版在竞争压力下放宽了承诺（见“负责任扩展政策”一题），说明自愿承诺确实会改。
- 一个站得住的判断：结构提高了“公司把安全让位于利润”的成本，但不能保证不发生；真正该看的是外部能验证的东西，比如信托是否行使过权力、风险报告的质量、第三方评审的结论。

避免两种说法：照着官网夸一遍；或者一句“都是公关”打发掉。

**Q5.** 批评最有力的版本大致是：

1. Anthropic 的存在让前沿多了一个强力竞争者，其他公司因此更急，整体进度被推快；
2. “留在前沿才能影响规则”这个理由可以无限使用，任何加速都能拿它辩护；
3. 证据已经有了：RSP 第 3 版删掉暂停承诺时，公司自己说的理由就是竞争对手没有跟进、单方面停下只会让出前沿。

回应时可以用的论点（选你真信的）：

- 反事实：没有 Anthropic，前沿并不会停，只是少了一家公开发布安全框架、风险报告和模型价值观文件的公司；条件式扩展政策、可解释性研究这些做法确实被同行采用或跟进了。
- 承认代价：赌注有可能是错的，判断它对不对要看可检验的结果，例如同行是否跟进安全做法、公司在关键时刻是否付过真实代价（五角大楼合同争议是一个例子）。
- 说出你的条件：什么样的事实会让你认为这个赌注已经输了。能说出这个条件，比单纯为公司辩护更有说服力。

</details>
