# 用人经理面：跨领域评估多任务模型，不加数据怎么改进模型

## 题目

用人经理（hiring manager）轮里的两道开放题，口头回答，每题大约 10 分钟，面试官会顺着你的回答往下追问。每组第一问是面经里记下的原题，后面的追问是按这类面试常见的追法补写的；每一问都写全了背景，可以单独练。

### 第一题：跨领域评估一个多任务系统

**Q1.** 面试官说：“我们有一个多任务系统：同一个模型要处理好几个领域的请求，比如日常问答、写代码、医疗咨询、看图回答问题。你会怎么评估它在各个领域上的表现？”请像在面试里那样，用几分钟讲出你的思路。

**Q2.** 背景：一个模型同时服务日常问答、代码、医疗三个领域，团队在比较新旧两个版本。(a) 在同一个固定测试集上：日常问答 8000 题，准确率从 80% 到 82%；代码 1500 题，从 60% 到 61%；医疗 500 题，从 70% 到 58%。把所有题放在一起算，新版本的总准确率高了 1.15 个点。面试官问：“能上线吗？”(b) 另一组数据来自线上日志：上个月跑旧版本，简单请求 1000 条答对 900 条，困难请求 100 条答对 50 条；这个月跑新版本，简单请求 300 条答对 276 条，困难请求 700 条答对 385 条。总准确率从 86.4% 掉到了 66.1%。新版本变差了吗？(c) 如果各领域的指标根本不是一种东西（比如分割用 mIoU，检测用 mAP，视频描述用 CIDEr），你怎么把它们合成一个数？该不该合成？

**Q3.** 背景：同一个多任务模型，医疗领域的测试集只有 200 道题。旧版本答对 80%，新版本答对 82%；逐题对比，有 14 道题新版本对、旧版本错，有 10 道题旧版本对、新版本错。(a) 这 2 个点的提升可信吗？算一下误差有多大。(b) 想可靠地看出 2 个点的差别，大约要多少道题？(c) 一共有 10 个领域，每个领域都按 5% 的显著性水平检验“有没有变差”，会出什么问题？(d) 团队想加一条上线门槛：“每个领域都不能比旧版本差 3 个点以上”。在只有 200 道题的领域上，这条门槛该怎么落地？

**Q4.** 背景：同一个多任务模型，测试集里有一部分题来自公开基准（benchmark）。面试官追问：“这个模型以后还会被用在我们现在没测过的领域。另外，公开基准的题可能已经混进了训练数据。你怎么知道它是真的会，而不是背过题？上线前你会设哪些检查？”

### 第二题：不加数据，怎么把模型做得更好

**Q5.** 面试官说：“假设你拿不到任何新数据，怎么把一个模型做得更好？”请像在面试里那样，用几分钟讲出你的思路。

**Q6.** 背景：一个对视频帧做目标分类的模型，验证集错误率 12%，不能再收集新数据。你从验证集的错误里随机抽了 100 个，逐个看过后归类：30 个是标注本身错了（模型其实答对了），25 个是小目标，20 个是运动模糊，15 个是外观相近的类别混淆，10 个是其他。(a) 每一类最多能把错误率降多少？(b) 如果把那 30 个错标纠正过来，模型真实的错误率大约是多少？(c) “小目标占 25%”这个比例有多准？(d) 除了按错误类型归类，还有哪类问题值得最先查？

**Q7.** 背景：同一个视频帧分类模型，不能加数据；标注已经清理过，错误分析也做过了。(a) 接下来还能做什么？按你会尝试的顺序排，并说出每一项在什么情况下有用、怎么知道它有没有用。(b) 有人提议把 5 个用不同随机种子训练的模型平均起来。设单个模型的预测误差方差是 $\sigma^2$，任意两个模型的误差相关系数是 $0.6$。平均之后的误差方差是多少？模型无限多时呢？这说明集成（ensemble）什么时候值得做？

**Q8.** 背景：一个模型，不能加数据，你在验证集上调它。一周里你试了 20 个改动，每个改动训练一次、在验证集上评一次，最好的那个比基线高 0.3 个点。已知同一配置只换随机种子，验证分数的标准差大约是 0.2 个点。面试官问：“这个提升是真的吗？你怎么确认？”

## 参考解答

<details>
<summary>展开参考解答</summary>

用人经理这一轮看的是判断力，不是你记得多少技巧。具体是三件事：先问清楚目标和约束，再按收益和成本排出先后，每一步都说得出“怎么知道它有没有用”。证据只有一份：流程页记录的一场机器学习软件工程师（ML software engineer）面试，团队做 Gemini 应用，走的是 Google 的招聘渠道；用人经理问了“怎么跨领域评估一个多任务系统”和“没有更多数据时怎么改进模型”。两题原文就这么一句，追问和参考答案都是我们补写的。

准备方法：每题准备一个两分钟的骨架（先问什么，分几步，每步一个具体例子），再备几个带数字的小例子，比如“总平均掩盖了某个领域的退步”“200 道题的误差有多大”“试 20 次挑出来的最好结果里有多少是噪声”。例子尽量用自己做过的项目，比如视频生成、视觉模型。完整的评估系统设计见 [大模型评估系统](#/deepmind/system-design/llm-evaluation-system)，这一页只练口头回答和取舍。

### 第一题：跨领域评估一个多任务系统

**Q1.** 强的回答先反问一句：**这次评估是为哪个决定服务的？** 常见的有三种。

- 上线决定：新版本能不能替换旧版本。要的是可靠的新旧对比，以及“哪个领域都不能明显退步”的底线。
- 研发方向：下一季度该补哪方面的能力。要的是细的能力诊断，按领域、难度、输入类型切片。
- 排查问题：某个领域投诉变多。要的是定位，看错误样本，而不是一个总分。

决定不同，测试集怎么建、看什么指标、要多少样本都不同。问完之后，按下面的顺序讲。

1. **每个领域先单独定义“好”。** 问答看准确率或人工评分，代码看单元测试通过率，医疗还要看有害回答的比例，看图问答要分图像类型统计。各领域的指标往往不可比，不要急着平均。换成自己熟悉的场景也一样：一个视频生成模型同时做文生视频、图生视频和视频编辑，三个任务各有指标（画质、与文本是否一致、编辑是否保住了原视频的内容），再加上共同的人工偏好评分和安全指标。
2. **测试集怎么来。** 每个领域按真实流量抽样，再单独补难例和长尾切片（长输入、少见语言、低清图像）。标注要有质量控制，难题请两个人独立标。
3. **怎么汇总。** 既报每个领域，也报总数，并说清总数怎么算：按流量加权，还是各领域等权。只看一个平均数会掩盖退步（见 Q2）。
4. **误差有多大。** 每个领域多少道题，置信区间多宽，新旧版本要在同一批题上配对比较（见 Q3）。
5. **没见过的领域和背题。** 留出领域，防止测试题混进训练数据（见 Q4）。
6. **上线门槛。** 每个领域退步不能超过多少，安全类指标一票否决；离线通过后，再小流量 A/B 测试，加人工抽检。

两分钟的口头版本：“先确认评估是为了什么决定。然后每个领域单独定指标、建测试集；汇总时同时给分领域结果和加权方式写明的总数；每个数都带误差范围，新旧版本配对比较；单独留出没参与开发的领域，并检查测试题有没有泄漏；最后按领域设退步门槛，安全指标一票否决，再上小流量实验。”

容易丢分的地方：

- 一上来就说“算准确率，然后取平均”，没问评估要支持什么决定。
- 只报一个总分，说不出它怎么加权、掩盖了什么。
- 不提每个领域有多少题、误差多大。
- 把测试集当成调参工具反复用，最后报的还是它。

**Q2.** (a) 两种平均给出相反的结论。

- 按题数加权（micro average，等于“随机抽一道题答对的概率”）：旧版本 $(8000 \times 80 + 1500 \times 60 + 500 \times 70)/10000 = 76.5\%$，新版本 $77.65\%$，高 $1.15$ 个点。
- 各领域等权（macro average）：旧版本 $(80 + 60 + 70)/3 = 70.0\%$，新版本 $(82 + 61 + 58)/3 = 67.0\%$，低 $3$ 个点。

医疗掉了 12 个点，被 8000 题的日常问答盖住了。这 12 个点不是噪声：500 道题、准确率 70% 左右时，单个版本的标准误（standard error）约 $\sqrt{0.7 \times 0.3/500} = 2.0$ 个点，两个版本之差的标准误约 $3.0$ 个点，12 个点是它的 4 倍。

所以回答是“不能只凭总数上线”。按题数加权反映的是当前流量下用户的平均体验，等权保护的是小领域；医疗是高风险领域，即使流量小也要单独设门槛。强的回答会说：报每个领域的结果和两种平均，标出退步的领域，用分领域的门槛做决定，而不是看一个数；同时去查医疗为什么退步（训练数据比例变了？某类问题全错？）。

(b) 新版本在两类请求上都更好：简单请求 $90\% \to 92\%$，困难请求 $50\% \to 55\%$。总数变差，是因为流量构成变了：困难请求的比例从 $100/1100 \approx 9\%$ 涨到了 $700/1000 = 70\%$。两个月比的是两群不同的请求。这就是辛普森悖论（Simpson's paradox）。

纠正的办法是在同一个分布上比。按固定的 50/50 构成重算，旧版本 $70.0\%$，新版本 $73.5\%$；按这个月的构成（30% 简单、70% 困难）重算，旧版本 $62.0\%$，新版本 $66.1\%$。更好的办法是让两个版本处理同一批请求：离线回放同一批日志，或者在同一时间段做随机分流的 A/B 测试。要记住的一点是：流量构成一变，总指标就会动，和模型好坏无关。

(c) 几种做法，按推荐程度排：

- **不合成。** 每个领域设门槛，再看有没有哪个领域被牺牲。上线决定通常这样做最稳。
- **相对基线归一化，再取几何平均。** 例如 mIoU $45.0 \to 46.8$、mAP $38.0 \to 37.2$、CIDEr $110 \to 121$，比值分别是 $1.040$、$0.979$、$1.100$，几何平均 $1.038$。但 mAP 掉了 $2.1\%$，只看合成数就看不到。
- **为什么不用比值的算术平均：** 它的结果取决于拿谁当基准。两个系统在两个指标上的分数是 $X = (10, 20)$、$Y = (20, 10)$：以 $X$ 为基准，$Y$ 的比值是 $(2, 0.5)$，算术平均 $1.25$；以 $Y$ 为基准，$X$ 的比值也是 $(2, 0.5)$，同样 $1.25$。两个系统都“比对方好 25%”，自相矛盾。几何平均两边都是 $1$。
- **按重要性加权。** 可以，但权重是产品决定，要明说是谁定的、为什么。

容易丢分的地方：坚持某一种平均“才是对的”，却不和决定挂钩；没注意到小领域的大退步；把两个月的日志直接比，当成同一群请求。

**Q3.** (a) 先看单个版本：准确率 80%、200 道题，标准误 $\sqrt{0.8 \times 0.2/200} = 2.8$ 个点，95% 置信区间大约 $\pm 5.5$ 个点。

两个版本之差，如果当成两组独立样本，标准误约 $\sqrt{(0.8 \times 0.2 + 0.82 \times 0.18)/200} = 3.9$ 个点，2 个点只有它的一半。但这里是同一批题，应该配对比较：两个版本都对或都错的题不提供差别的信息，只有那 24 道不一致的题有用。差值 $(14 - 10)/200 = 2$ 个点，配对的标准误 $\sqrt{14 + 10 - (14 - 10)^2/200}\,/\,200 \approx 2.4$ 个点，比值约 $0.8$。精确的麦克尼马尔检验（McNemar's test）：零假设下 14 服从 $\mathrm{Binomial}(24, 1/2)$，双侧 $p \approx 0.54$。结论是看不出差别。配对把误差从 3.9 降到了 2.4，但还远远不够。

(b) 设两个版本不一致的题约占 12%（新对旧错 7%，旧对新错 5%，差 2 个点）。双侧 5% 的检验要有 80% 的把握看出这个差别，配对时大约要

$$n \approx \frac{(1.96 + 0.84)^2 \times (0.12 - 0.02^2)}{0.02^2} \approx 2350$$

道题；如果两个版本在不同的题上评，每个版本约要 6000 道。做不到这么多时：把相近的领域合起来看；换更灵敏的指标（逐题打分，或正确答案的对数概率，而不是对错）；每道题采样多次；或者承认这个测试集只能看出更大的差别。

(c) 多重比较。即使 10 个领域都没变，每个领域有 5% 的误报概率，至少一个领域误报“变差”的概率是 $1 - 0.95^{10} \approx 40\%$（假设各领域独立）。办法：每个领域用更严的水平（Bonferroni，$5\%/10 = 0.5\%$），或者用 Holm 方法、控制错误发现率（false discovery rate）；事先定好哪些领域是门槛；被标出来的领域在新数据上复核一次。

(d) 门槛要和样本量一起定。200 道题、不一致率 12% 时，即使两个版本完全一样，差值的标准误也有 $\sqrt{24}/200 \approx 2.4$ 个点，单侧 95% 置信下界大约是 $-4.0$ 个点。如果门槛写成“下界不能低于 $-3$”，完全一样的版本也过不了；如果写成“点估计不能低于 $-3$”，真实退步 3 到 5 个点的版本又常常能混过去。要让完全一样的两个版本的下界落在 $-3$ 以内，至少要约 360 道题。

所以落地时：高风险的小领域（医疗）补题到够用，或者加人工复核；低风险的小领域放宽门槛，或和相近领域合并；把“这个测试集能看出多大的退步”写进门槛说明里。

容易丢分的地方：把 2 个点的差当真；手里有配对数据却按独立样本算；不提多重比较；门槛只有一个数字，没有对应的样本量。

**Q4.** 分两件事回答：没见过的领域，和背题。

**没见过的领域。** 留出领域（leave-one-domain-out）：把一部分领域完全排除在训练和调参之外，只在最后评一次，看能力能不能迁移。生产模型不可能每个领域重训一遍，那就至少保留几个“从没拿来调过参”的领域和切片当最终检查。还可以看模型在新领域上是“会说不知道”还是“自信地答错”，这时校准比准确率更重要。视频生成里的对应做法：留出训练集里很少出现的运动类型（流体、快速运镜）、画风、画面比例和时长，单独评。

**背题（数据污染，contamination）。**

- 按时间切：只用训练数据截止之后才出现的题。
- 查重叠：在训练数据里搜测试题的 n-gram 或向量近邻；图像和视频用感知哈希、近重复帧匹配。
- 改写测试题：换说法、换数字、重新渲染图片。真会的模型分数基本不变，背过题的会明显下降。
- 私有测试集：不公开，题里放金丝雀字符串（canary string），以后在训练数据里搜到它就说明泄漏了；持续比较公开基准和私有集上的分数差，公开分一路涨、私有分不动，就是在背题。
- 视频生成的例子：公开基准的提示词列表可能出现在训练视频的描述里，要先查重再评。

还有一种污染来自自己：同一个开发集反复评、反复调，最后也会过拟合到它上面。所以要留一个很少动的测试集。

**上线前的检查。** 每个领域按 Q3 的方法定退步门槛；安全类指标（有害回答、隐私泄漏）一票否决；延迟和成本不超预算；然后小流量 A/B 测试，带护栏指标和人工抽检，事先写好回滚条件。

容易丢分的地方：只报公开基准；把“分数高”等同于“有这个能力”；对没见过的领域没有任何计划；只防训练数据泄漏，没想到自己反复调参也是一种泄漏。

### 第二题：不加数据，怎么把模型做得更好

**Q5.** 强的回答先问清楚“不加数据”具体固定了什么，因为答案取决于它：

- 新标注不能有，那没标注的输入有没有？现有的标注能不能返工、清理？
- 算力预算多少？模型结构能不能改？能不能用外面的预训练权重？
- 推理有没有延迟和成本限制？这决定能不能做集成和测试时增强。
- 指标是什么？是准确率这类不带阈值的，还是“召回率 95% 时的精确率”这类带阈值的？
- 现在是欠拟合还是过拟合？训练误差和验证误差各是多少？

然后按“预期收益高、成本低的先做”排一个梯子，每一步都说清什么时候有用、怎么知道它有没有用：

1. **看错误，查管线。** 抽错误样本归类，查数据和管线的 bug（见 Q6）。最便宜，常常收益最大。
2. **用好现有数据。** 纠正错标，去掉重复和坏样本，对少见类别重新加权或重采样，针对错误类型做数据增强（data augmentation），由易到难的课程学习（curriculum learning）。
3. **训练配方和正则化。** 先按训练和验证误差判断方向：过拟合就加增强、权重衰减、dropout、标签平滑、mixup；欠拟合就加大模型、训练更久、调学习率。另外，学习率调度和权重的指数滑动平均（EMA）几乎总值得调。
4. **迁移。** 允许的话，换一个更好的预训练权重做初始化，或者和相关任务一起训练。
5. **有未标注的输入时。** 自训练（self-training，用高置信度的伪标签）、知识蒸馏（knowledge distillation，大模型教小模型）、在这些输入上做自监督预训练。
6. **推理时。** 集成多个模型、测试时增强（test-time augmentation，比如翻转、多裁剪、多帧平均）。代价是推理成本翻倍。
7. **指标带阈值时。** 先校准（例如温度缩放，temperature scaling），再按业务要求选工作点。指标的提升来自选对工作点；校准本身不改变分数排序，AUC、固定召回率下的精确率这类只看排序的指标不会因它变化，它的用处是让“概率超过 0.9 才处理”这类按概率写的阈值选得准。

判断每一步有没有用，靠的是同一套东西：固定的验证集、多个随机种子给出的噪声水平、按错误类型切片的结果，以及一个只在最后用一次的测试集。

如果你的背景是生成模型，可以用自己的例子：视频生成模型不加数据，可以重新给训练视频写更细的描述（re-captioning），过滤低质量片段，按分辨率和时长分桶采样，调噪声调度和训练时的时间步采样，调无分类器引导（classifier-free guidance）的强度，用 EMA 权重，调采样步数或做步数蒸馏。注意，生成模型上的数据增强会“漏”到生成结果里（比如训练时做了翻转，模型就会生成翻转后的画面），常见的处理有两种：StyleGAN2-ADA 只对判别器的输入做增强，并控制增强的概率，不让它漏到生成结果里；EDM 把增强参数作为条件输入模型，生成时把条件设为“不增强”。

两分钟的口头版本：“先问清楚约束：有没有未标注数据，能不能用预训练权重，推理预算多少，指标是不是带阈值。然后第一步一定是看错误、查管线 bug；第二步把现有数据用好；第三步根据欠拟合还是过拟合调配方；有未标注数据就做自训练或蒸馏；推理预算允许就做集成；指标带阈值就做校准。每一步都在固定的验证集上、用多个种子判断有没有用，测试集只在最后看一次。”

容易丢分的地方：

- 一口气列十几种技巧，不排序，也不说为什么先做这个。
- 不问约束，直接说“加大模型”，没先看是欠拟合还是过拟合。
- 说不出怎么判断某个改动有没有用。
- 在测试集上挑方法。

**Q6.** (a) 每一类的上限等于“总错误率 × 这一类的占比”：

- 标注错了：$12\% \times 0.30 = 3.6$ 个点。这部分不用改模型，纠正验证集的标注后，测出来的错误率就会降。
- 小目标：$3.0$ 个点；运动模糊：$2.4$ 个点；相近类别：$1.8$ 个点；其他：$1.2$ 个点。

这是上限：就算小目标的问题全部解决，错误率也最多降 3 个点；实际只能解决一部分。这个算法的用处是决定先做什么，以及排除那些“即使做成了也没多少收益”的方向。

(b) 那 30 个其实是模型答对了，所以真实错误率大约是 $12\% \times 0.7 = 8.4\%$。先把验证集的标注修好，再比较模型；不然后面所有的对比都带着这 3.6 个点的噪声。还要查训练集里有没有同样比例的错标：验证集错 30%，训练集多半也不干净，修训练集的标注本身就可能提升模型。

(c) 100 个样本里有 25 个，比例的 95% 置信区间约为 $25\% \pm 1.96\sqrt{0.25 \times 0.75/100} = 25\% \pm 8.5\%$，对应的上限大约在 2.0 到 4.0 个点之间。用来粗排先后够了，但 30 个和 25 个这样的差别分不出来。想要更准，就多看一些样本，重点看排在前面的几类。

(d) 最先查数据和管线的 bug，它们往往比任何模型改动都值钱：

- 训练和推理的预处理不一致：缩放方法、RGB 和 BGR 的通道顺序、归一化常数、抽帧的帧率、训练用的视频解码器和线上用的不一样。
- 数据泄漏：同一段视频的相邻帧同时出现在训练集和验证集里，验证分数虚高。应该按视频划分，而不是按帧。
- 标签映射错了，类别编号对不上；验证时误开了数据增强。

检查的办法：用训练时的评估代码和线上的推理代码跑同一批输入，比较输出是否一致；用一个很小的子集训练，确认模型能把它拟合到接近 100%。

找到主要的错误类型之后，针对性的改法：小目标用更高的输入分辨率、多尺度训练、裁剪放大；运动模糊加模糊增强，或者用相邻帧的时序信息；相近类别先看混淆矩阵，确认两类的标注标准是否一致，必要时合并类别或改成层级标签。

容易丢分的地方：不看错误就开始改模型；把 25% 当成精确值；不先修验证集的标注就开始比较模型。

**Q7.** (a) 一个合理的顺序：

1. **先判断方向。** 比如训练误差 2%、验证误差 12% 是过拟合，应该加增强和正则化；训练误差也是 11% 就是欠拟合，应该加容量、训练更久、调学习率。判断错了方向，后面的努力都白费。
2. **针对错误类型的数据增强。** 运动模糊多就加模糊增强，小目标多就加尺度抖动和裁剪。怎么知道有用：对应那一类的错误率下降，总的验证分数涨幅超过随机种子带来的噪声。
3. **训练配方。** 学习率预热加余弦衰减、训练更久、权重衰减、标签平滑、EMA。怎么知道有用：看学习曲线，用多个种子比较。
4. **类别不平衡时重新加权或重采样。** 怎么知道有用：看每个类别的召回率，而不只是总准确率。
5. **更好的初始化。** 允许的话，换在视频上预训练过的骨干网络，而不是只在图片上预训练的。
6. **有未标注的视频帧时，做伪标签和师生训练。** 只用高置信度的伪标签，并且盯着验证集，因为模型会强化自己原来的错误。
7. **集成和测试时增强。** 几乎总能涨一点，但推理成本成倍增加；最后可以把集成蒸馏回一个模型。
8. **指标带阈值时，做校准和阈值选择。**

每一项都要在同一个验证集、同样的几个种子下和基线比，一次只改一样东西。

(b) $M$ 个模型平均，每个的误差方差是 $\sigma^2$，两两相关系数是 $\rho$：

$$\mathrm{Var}\Big(\frac1M \sum_{i=1}^M e_i\Big) = \frac{1}{M^2}\Big[M\sigma^2 + M(M-1)\rho\sigma^2\Big] = \sigma^2\Big(\rho + \frac{1-\rho}{M}\Big).$$

$\rho = 0.6$、$M = 5$：$0.6 + 0.4/5 = 0.68$，即 $0.68\,\sigma^2$，标准差降到原来的 $\sqrt{0.68} \approx 0.825$ 倍。模型无限多时只能降到 $0.6\,\sigma^2$。也就是说，大部分收益来自前几个模型，下限由相关系数决定：模型越像，集成越没用。$\rho = 0.9$ 时，5 个模型平均也只能降到 $0.92\,\sigma^2$。只换随机种子的模型往往很相关，换不同结构、不同数据顺序、不同增强能降低相关性。还要注意：集成减少的是方差那部分误差，所有模型共有的偏差（比如都看不清小目标）不会因为平均而消失。

值不值得做，要和成本一起算：5 倍的推理成本换这点方差的下降，线上通常不划算；但可以先做集成，再把它蒸馏到单个模型。

容易丢分的地方：只列方法不排序；不说怎么判断有用；把集成当成免费的；不知道集成的收益受相关性限制。

**Q8.** 现在还不能信。理由有两层。

第一，单次运行的噪声。改动和基线各跑一次，两者之差的标准差是 $0.2 \times \sqrt2 \approx 0.28$ 个点，0.3 只有大约一个标准差。

第二，挑选带来的偏差（winner's curse）。就算 20 个改动全都没用，每个改动的验证分数也会因为随机性上下浮动，20 个里最好的那个平均会高出约 $1.87$ 个标准差，即 $1.87 \times 0.2 \approx 0.37$ 个点。也就是说，0.3 的提升正好是“什么都没改”时最可能看到的结果。

怎么确认：

1. 用新的随机种子，把选中的改动和基线各重跑几次，比如各 5 次。两组均值之差的标准误约 $0.2 \times \sqrt{2/5} \approx 0.13$ 个点；如果差距仍然在 0.3 左右，那就是 $2.4$ 个标准误，比较可信。
2. 在同一批验证样本上逐题配对比较，看提升是集中在某几类还是普遍存在；只集中在一小撮样本上的提升要警惕。
3. 最后在一个没参与过任何挑选的测试集上评一次，并且只评这一次。
4. 汇报时说清楚一共试了多少个改动。

容易丢分的地方：相信单次运行；不知道自己实验的噪声有多大；在验证集上挑出最好的，又拿同一个验证分数当最终结果；用测试集来挑方法。

<details>
<summary>验证代码（可运行）</summary>

```python
import math
from fractions import Fraction as F

import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)

# ---- Q2, example 1: one fixed test set; micro and macro averages point in opposite directions ----
# domain: (examples, old accuracy %, new accuracy %)
fixed_set = {"qa": (8000, 80, 82), "code": (1500, 60, 61), "medical": (500, 70, 58)}
total = sum(n for n, _, _ in fixed_set.values())
micro_old = F(sum(n * old for n, old, _ in fixed_set.values()), total)
micro_new = F(sum(n * new for n, _, new in fixed_set.values()), total)
macro_old = F(sum(old for _, old, _ in fixed_set.values()), len(fixed_set))
macro_new = F(sum(new for _, _, new in fixed_set.values()), len(fixed_set))
assert (micro_old, micro_new) == (F(765, 10), F(7765, 100)) and micro_new - micro_old == F(115, 100)
assert (macro_old, macro_new) == (70, 67) and macro_new - macro_old == -3
assert fixed_set["medical"][2] - fixed_set["medical"][1] == -12
se_med = math.sqrt(0.70 * 0.30 / 500)                        # 500 medical items: is the 12-point drop noise?
se_med_diff = math.sqrt(0.70 * 0.30 / 500 + 0.58 * 0.42 / 500)
assert round(se_med * 100, 1) == 2.0 and round(se_med_diff * 100, 1) == 3.0 and round(0.12 / se_med_diff) == 4

# ---- Q2, example 2: better in every domain, worse overall, because the traffic mix changed ----
old_logs = {"easy": (900, 1000), "hard": (50, 100)}          # (correct, requests), last month's traffic
new_logs = {"easy": (276, 300), "hard": (385, 700)}          # this month's traffic
rate = lambda logs, d: F(*logs[d])
overall = lambda logs: F(sum(c for c, _ in logs.values()), sum(n for _, n in logs.values()))
assert rate(old_logs, "easy") == F(90, 100) and rate(new_logs, "easy") == F(92, 100)
assert rate(old_logs, "hard") == F(50, 100) and rate(new_logs, "hard") == F(55, 100)
assert all(rate(new_logs, d) > rate(old_logs, d) for d in old_logs)          # new wins in both domains
assert round(float(overall(old_logs)) * 100, 1) == 86.4 and round(float(overall(new_logs)) * 100, 1) == 66.1
assert overall(new_logs) < overall(old_logs)                                 # ... and loses overall
assert round(100 / 1100 * 100) == 9 and 700 / 1000 == 0.7                    # share of hard traffic: 9% -> 70%
fixed_mix = lambda logs: (rate(logs, "easy") + rate(logs, "hard")) / 2       # re-weight both to a 50/50 mix
assert fixed_mix(old_logs) == F(70, 100) and fixed_mix(new_logs) == F(735, 1000)
this_mix = lambda logs: F(3, 10) * rate(logs, "easy") + F(7, 10) * rate(logs, "hard")      # this month's 30/70 mix
assert this_mix(old_logs) == F(62, 100) and this_mix(new_logs) == F(661, 1000)

# ---- Q2: averaging ratios to a baseline ----
x, y = (10, 20), (20, 10)                                    # two systems, two metrics on different scales
arith = lambda a, b: sum(F(ai, bi) for ai, bi in zip(a, b)) / len(a)         # mean of a's scores relative to b
geo = lambda a, b: math.prod(ai / bi for ai, bi in zip(a, b)) ** (1 / len(a))
assert arith(y, x) == F(5, 4) and arith(x, y) == F(5, 4)     # each one looks 25% better than the other
assert math.isclose(geo(y, x), 1) and math.isclose(geo(x, y), 1)
ratios = [46.8 / 45.0, 37.2 / 38.0, 121 / 110]               # mIoU, mAP, CIDEr: new / old
assert [round(r, 3) for r in ratios] == [1.04, 0.979, 1.1]
assert round(math.prod(ratios) ** (1 / 3), 3) == 1.038 and round((1 - ratios[1]) * 100, 1) == 2.1

# ---- Q3: 200 examples in one domain ----
n = 200
se_single = math.sqrt(0.8 * 0.2 / n)
assert round(se_single * 100, 1) == 2.8 and round(1.96 * se_single * 100, 1) == 5.5
se_unpaired = math.sqrt(0.80 * 0.20 / n + 0.82 * 0.18 / n)
assert round(se_unpaired * 100, 1) == 3.9 and round(0.02 / se_unpaired, 1) == 0.5
b, c = 14, 10                                                # new right / old wrong, old right / new wrong
assert (b - c) / n == 0.02
se_paired = math.sqrt(b + c - (b - c) ** 2 / n) / n
assert round(b + c - (b - c) ** 2 / n, 2) == 23.92 and round(se_paired * 100, 1) == 2.4
tail = sum(math.comb(b + c, k) for k in range(b, b + c + 1)) / 2 ** (b + c)  # exact McNemar: Binomial(24, 1/2)
assert round(min(1.0, 2 * tail), 2) == 0.54
z_alpha, z_beta = 1.959964, 0.841621                         # two-sided 5%, power 80%
p_b, p_c = 0.07, 0.05
n_paired = (z_alpha + z_beta) ** 2 * (p_b + p_c - (p_b - p_c) ** 2) / (p_b - p_c) ** 2
n_unpaired = (z_alpha + z_beta) ** 2 * (0.80 * 0.20 + 0.82 * 0.18) / 0.02 ** 2
assert 2300 < n_paired < 2400 and 5900 < n_unpaired < 6100
assert round(n_paired, -1) == 2350 and round(n_unpaired, -2) == 6000
assert round(0.02 / se_paired, 1) == 0.8
counts = rng.multinomial(round(n_paired), [p_b, p_c, 1 - p_b - p_c], size=20000)   # simulate the paired test
bb, cc = counts[:, 0], counts[:, 1]
z = (bb - cc) / np.sqrt(bb + cc - (bb - cc) ** 2 / round(n_paired))
assert abs((z > z_alpha).mean() - 0.80) < 0.02               # power is about 80% at that sample size
lower_bound = 0 - 1.645 * math.sqrt(12 + 12) / n             # identical models: observed difference 0
assert round(lower_bound * 100, 1) == -4.0 and lower_bound < -0.03           # fails a "-3 points" gate
n_gate = math.ceil((1.645 * math.sqrt(0.12) / 0.03) ** 2)    # items needed before identical models pass that gate
assert n_gate == 361
assert round(1 - 0.95 ** 10, 2) == 0.40                      # ten domains, each tested at 5%

# ---- Q5: temperature scaling keeps the ranking, so ranking-based metrics do not move ----
scores = rng.normal(size=5000)
labels = rng.random(5000) < 1 / (1 + np.exp(-(2 * scores - 2)))
def precision_at_recall(prob, rec=0.95):
    order = np.argsort(-prob)
    tp = np.cumsum(labels[order])
    k = int(np.searchsorted(tp, rec * labels.sum())) + 1
    return tp[k - 1] / k
prob_raw, prob_cal = 1 / (1 + np.exp(-5 * scores)), 1 / (1 + np.exp(-5 * scores / 2.5))   # temperature 2.5
assert precision_at_recall(prob_raw) == precision_at_recall(prob_cal)        # same ranking, same metric
assert (prob_raw > 0.9).sum() != (prob_cal > 0.9).sum()                      # a threshold written as a probability does move

# ---- Q6: error analysis on 100 sampled errors ----
error_rate = 12.0                                            # % on the validation set
buckets = {"label wrong": 30, "small objects": 25, "motion blur": 20, "similar classes": 15, "other": 10}
assert sum(buckets.values()) == 100
ceiling = {k: error_rate * v / 100 for k, v in buckets.items()}
assert math.isclose(ceiling["small objects"], 3.0) and math.isclose(ceiling["motion blur"], 2.4)
assert math.isclose(ceiling["similar classes"], 1.8) and math.isclose(ceiling["label wrong"], 3.6)
assert math.isclose(ceiling["other"], 1.2)
assert math.isclose(error_rate - ceiling["label wrong"], 8.4)
half_width = 1.96 * math.sqrt(0.25 * 0.75 / 100)
assert round(half_width * 100, 1) == 8.5
assert round(error_rate * (0.25 - half_width), 1) == 2.0 and round(error_rate * (0.25 + half_width), 1) == 4.0

# ---- Q7: averaging M models whose errors have variance 1 and pairwise correlation rho ----
ens_var = lambda rho, m: rho + (1 - rho) / m
assert math.isclose(ens_var(0.6, 5), 0.68) and math.isclose(ens_var(0.9, 5), 0.92)
assert ens_var(0.6, 10 ** 9) > 0.6 and math.isclose(ens_var(0.6, 10 ** 9), 0.6, abs_tol=1e-8)
assert round(math.sqrt(0.68), 3) == 0.825
shared, own = rng.normal(size=(200000, 1)), rng.normal(size=(200000, 5))
errors = math.sqrt(0.6) * shared + math.sqrt(0.4) * own      # unit variance, correlation 0.6
assert abs(np.corrcoef(errors[:, 0], errors[:, 1])[0, 1] - 0.6) < 0.01
assert abs(errors.mean(axis=1).var() - 0.68) < 0.01

# ---- Q8: the best of 20 changes that do nothing ----
sigma = 0.2
grid = np.linspace(-8, 8, 160001)
pdf = np.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi)
cdf = np.cumsum(pdf) * (grid[1] - grid[0])
expected_max = float(np.sum(grid * 20 * cdf ** 19 * pdf) * (grid[1] - grid[0]))   # E[max of 20 standard normals]
assert round(expected_max, 2) == 1.87 and round(expected_max * sigma, 2) == 0.37
best_of_20 = rng.normal(0, sigma, size=(100000, 20)).max(axis=1)
assert abs(best_of_20.mean() - 0.37) < 0.01
assert round(sigma * math.sqrt(2), 2) == 0.28
se_five = sigma * math.sqrt(2 / 5)
assert round(se_five, 2) == 0.13 and round(0.3 / se_five, 1) == 2.4

print("all checks passed")
```

</details>

</details>
