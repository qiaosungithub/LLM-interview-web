# 经典强化学习：MDP、时序差分、老虎机与离策略评估

## 题目

下面 13 个问题按 DeepMind 早期“小测验”的方式来：每问很短，口头回答，要用的式子写在纸上，答案是确定的。每一问都把要用到的条件写全了，可以单独做；出现数字的地方，请把数算出来。

### MDP 与动态规划

**Q1.** 一个两状态的马尔可夫决策过程（Markov decision process，MDP），折扣因子（discount factor）$\gamma = 0.8$。状态 A：动作 `stay` 得奖励 $+1$，留在 A；动作 `go` 得奖励 $-1$，以概率 $3/4$ 转到 B，以概率 $1/4$ 留在 A。状态 B：动作 `stay` 得奖励 $+5$，以概率 $3/4$ 留在 B，以概率 $1/4$ 回到 A；动作 `cash` 得奖励 $+7$，必定回到 A。策略（policy）$\pi$ 在 A 选 `go`，在 B 选 `cash`。写出 $V^\pi(A)$、$V^\pi(B)$ 满足的贝尔曼期望方程（Bellman expectation equation），解出这两个值，并说明为什么解是唯一的。

**Q2.** 两状态 MDP，折扣因子 $\gamma = 0.8$。状态 A：`stay` 得 $+1$，留在 A；`go` 得 $-1$，以概率 $3/4$ 到 B，以概率 $1/4$ 留在 A。状态 B：`stay` 得 $+5$，以概率 $3/4$ 留在 B，以概率 $1/4$ 回到 A；`cash` 得 $+7$，必定回到 A。策略 $\pi$（A 选 `go`，B 选 `cash`）的价值已经算出：$V^\pi(A) = 10$，$V^\pi(B) = 15$。(a) 算出四个动作价值 $Q^\pi(s,a)$，做一步策略改进（policy improvement），得到新策略 $\pi'$。(b) 求 $V^{\pi'}$。(c) 用贝尔曼最优方程（Bellman optimality equation）判断 $\pi'$ 是不是最优策略。(d) 如果每一步都只挑即时奖励最大的动作（A 选 `stay`，B 选 `cash`），价值是多少？

**Q3.** 设 MDP 的奖励有界，$\lvert r(s,a)\rvert \le R_{\max}$，折扣因子 $\gamma < 1$。贝尔曼最优算子定义为 $(TV)(s) = \max_a \big[r(s,a) + \gamma \sum_{s'} p(s' \mid s,a)\,V(s')\big]$。(a) 证明 $T$ 在无穷范数下是 $\gamma$-压缩（contraction）：$\lVert TV - TU\rVert_\infty \le \gamma\,\lVert V - U\rVert_\infty$。(b) 由此说明值迭代（value iteration）$V_{k+1} = TV_k$ 为什么收敛，并给出 $\lVert V_k - V^*\rVert_\infty$ 的上界。(c) 取 $\gamma = 0.8$、$R_{\max} = 7$、$V_0 = 0$：按这个上界，迭代多少轮能保证误差不超过 $0.01$？(d) 具体例子：两状态 MDP，$\gamma = 0.8$；状态 A：`stay` 得 $+1$，留在 A；`go` 得 $-1$，以概率 $3/4$ 到 B，以概率 $1/4$ 留在 A；状态 B：`stay` 得 $+5$，以概率 $3/4$ 留在 B，以概率 $1/4$ 回到 A；`cash` 得 $+7$，必定回到 A。它的最优价值是 $V^*(A) = 13$、$V^*(B) = 19$。从 $V_0 = 0$ 出发手算 $V_1$、$V_2$、$V_3$，看误差每轮缩小多少；实际要多少轮，误差才不超过 $0.01$？

**Q4.** 折扣因子 $\gamma = 0.99$。常听到的说法是“有效视野（effective horizon）大约是 $1/(1-\gamma) = 100$ 步”。(a) 给出两种能让这句话变精确的解释。(b) 前 100 步的奖励占全部折扣权重 $\sum_{t \ge 0} \gamma^t$ 的多少？(c) 要覆盖 99% 的权重，需要看多少步？(d) 每步奖励都在 $[0,1]$ 内时，折扣回报（return）的取值范围是什么？把轨迹在第 1000 步处截断，回报最多少算多少？

### 时序差分学习

**Q5.** 一个表格型（tabular）控制问题，折扣因子 $\gamma = 0.8$，步长 $\alpha = 0.5$。智能体按 $\varepsilon$-贪心（$\varepsilon$-greedy）行动，$\varepsilon = 0.2$：以概率 $1-\varepsilon$ 选当前 $Q$ 值最大的动作，以概率 $\varepsilon$ 在两个动作里均匀随机选一个。当前的 $Q$ 表是 $Q(A,\text{stay}) = 6$，$Q(A,\text{go}) = 8$，$Q(B,\text{stay}) = 12$，$Q(B,\text{cash}) = 10$。智能体在 A 执行 `go`，得到奖励 $-1$，到了 B；下一步碰巧是探索，在 B 选了 `cash`。(a) 分别写出 SARSA、Q-learning、Expected SARSA 的更新目标，并算出更新后的 $Q(A,\text{go})$。(b) 三者的差别只在哪一项？(c) 如果 $\varepsilon$ 一直固定在 $0.2$，Q-learning 和 SARSA 各自收敛到什么？各需要什么条件？

**Q6.** 不折扣（$\gamma = 1$），只有 A、B 两个非终止状态，没有动作可选。收集到 10 条完整的轨迹，按“状态, 奖励, 状态, 奖励”的顺序记录，记录结束即终止：1 条 `A, 0, B, 0`；1 条 `A, 0, B, 1`；6 条 `B, 1`；2 条 `B, 0`。把这批数据反复回放，直到价值估计不再变化（批量更新，batch updating）。(a) 蒙特卡洛（Monte Carlo，MC）方法给出的 $V(A)$、$V(B)$ 是多少？(b) 时序差分（temporal difference）方法 TD(0) 给出的是多少？(c) 两个 $V(A)$ 为什么不一样？各自在什么意义下是“最优”的？什么时候该信哪一个？

**Q7.** 一条链：$s_1 \to s_2 \to \dots \to s_{10} \to$ 终止，没有动作可选。每走一步得到的奖励独立地以概率 $1/2$ 取 $+1$、以概率 $1/2$ 取 $-1$，不折扣（$\gamma = 1$），所以每个状态的真实价值都是 $0$。当前估计是 $V(s_1) = 0$，$V(s_2) = 0.5$。(a) 用来更新 $V(s_1)$ 的蒙特卡洛目标和 TD(0) 目标各是什么？分别求它们的期望、偏差（bias）、方差，以及相对真值 $0$ 的均方误差。(b) 第一步实际得到的奖励是 $+1$，步长 $\alpha = 0.1$，做一次 TD(0) 更新后 $V(s_1)$ 变成多少？(c) 概括这两种目标在偏差和方差上的取舍。TD 目标有偏差，会不会让表格型 TD(0) 收敛到错误的值？

### 多臂老虎机

**Q8.** 多臂老虎机（multi-armed bandit）问题：两个伯努利臂，成功概率 $\mu_1 = 0.6$、$\mu_2 = 0.4$，算法事先不知道。用固定 $\varepsilon = 0.1$ 的 $\varepsilon$-贪心：每一步以概率 $1-\varepsilon$ 拉样本均值最高的臂，以概率 $\varepsilon$ 在两个臂里均匀随机拉一个。(a) 写出遗憾（regret）的定义，并用各臂被拉的次数把它表示出来。(b) 证明 $T$ 步之后的期望遗憾至少是 $0.01\,T$；$T = 10^4$ 时至少是多少？(c) 怎么改 $\varepsilon$，才能让遗憾只按 $\log T$ 增长？需要什么前提？

**Q9.** 一个多臂老虎机问题，奖励都在 $[0,1]$ 内。UCB1 算法每一步拉指标 $\bar x_i + \sqrt{2\ln n / n_i}$ 最大的臂，其中 $n$ 是到目前为止拉过的总次数，$n_i$ 是臂 $i$ 被拉的次数，$\bar x_i$ 是它的样本均值。现在 $n = 100$：臂 1 拉了 90 次，均值 $0.60$；臂 2 拉了 10 次，均值 $0.40$。(a) 算出两个臂的指标。下一步拉哪个臂？(b) 用霍夫丁不等式（Hoeffding's inequality）解释，加上去的那一项为什么取 $\sqrt{2\ln n / n_i}$。(c) UCB1 的期望遗憾（regret，即与一直拉最好的臂相比少拿的期望奖励）随总步数 $T$ 以什么速度增长？写出你记得的上界，并在两个臂的真实均值为 $0.6$ 和 $0.4$、$T = 10^4$ 时把它算出来。(d) 这个增长速度还能再改进吗？

**Q10.** 两个伯努利臂，成功概率未知，先验都取 $\mathrm{Beta}(1,1)$（即均匀分布）。到现在为止：臂 1 拉了 3 次，成功 2 次；臂 2 拉了 1 次，成功 0 次。(a) 写出两个臂的后验分布和后验均值。(b) 汤普森采样（Thompson sampling）下一步怎么选臂？它选臂 2 的概率是多少？只按后验均值贪心的话呢？(c) 假如这一步拉了臂 2，而且成功了，它的后验变成什么？(d) 用一两句话说明，汤普森采样为什么能兼顾探索和利用。

### 离策略评估、函数近似与奖励塑形

**Q11.** 只有一步的问题：动作 L 得奖励 $1$，动作 R 得奖励 $0$。数据由行为策略（behaviour policy）$\mu$ 产生，它以概率 $1/2$ 选 L、以概率 $1/2$ 选 R；我们想评估的是目标策略（target policy）$\pi$，它以概率 $0.9$ 选 L、以概率 $0.1$ 选 R。从 $\mu$ 采到 $n$ 个独立样本 $(a_i, r_i)$，重要性权重为 $\rho_i = \pi(a_i)/\mu(a_i)$。普通重要性采样（ordinary importance sampling）估计是 $\hat V_{\text{OIS}} = \frac1n \sum_i \rho_i r_i$，加权重要性采样（weighted importance sampling）估计是 $\hat V_{\text{WIS}} = \sum_i \rho_i r_i \big/ \sum_i \rho_i$。(a) 真值 $v_\pi$ 是多少？两个动作的权重各是多少？(b) $n = 1$ 时，两种估计的期望、方差、均方误差各是多少？直接用 $\pi$ 采一个样本，方差又是多少？(c) $n = 2$ 时，求加权估计的期望和均方误差，并与普通估计比较。(d) 概括两种估计在偏差、方差上的一般性质，以及重要性采样成立需要的条件。

**Q12.** (a) “致命三要素”（deadly triad）指哪三样东西同时出现？它的意思是“一定发散”吗？(b) 例子：两个状态共用一个参数 $w$ 做线性近似，状态 1 的估计值是 $w$，状态 2 的估计值是 $2w$。所有奖励都是 $0$，所以真实价值都是 $0$。只反复用“状态 1 → 状态 2，奖励 $0$”这一条转移做半梯度（semi-gradient）TD(0) 更新，$\gamma = 0.9$，$\alpha = 0.1$，$w_0 = 1$。写出更新式，算出 10 次更新后的 $w$。$\gamma$ 取什么值时 $w$ 会发散？(c) 改成同策略采样：每个回合先从状态 1 走到状态 2，再从状态 2 走到终止态（奖励 $0$，终止态价值为 $0$），两条转移依次各更新一次。10 个回合后 $w$ 是多少？(d) 三要素里去掉任何一个会怎样？各举一个不发散的例子。

**Q13.** 两状态 MDP，折扣因子 $\gamma = 0.8$。状态 A：`stay` 得 $+1$，留在 A；`go` 得 $-1$，以概率 $3/4$ 到 B，以概率 $1/4$ 留在 A。状态 B：`stay` 得 $+5$，以概率 $3/4$ 留在 B，以概率 $1/4$ 回到 A；`cash` 得 $+7$，必定回到 A。已知最优策略是“A 选 `go`，B 选 `stay`”，价值为 $(V(A), V(B)) = (13, 19)$，最优动作价值为 $Q^*(A,\text{stay}) = 11.4$，$Q^*(A,\text{go}) = 13$，$Q^*(B,\text{stay}) = 19$，$Q^*(B,\text{cash}) = 17.4$；另一个策略“A 选 `go`，B 选 `cash`”的价值是 $(10, 15)$。为了让智能体更快学会去 B，有人提出两种奖励塑形（reward shaping）方案。方案一：每次从 A 转移到 B，额外加 $4$，其他转移不变。方案二：用势函数（potential function）$\Phi(A) = 0$、$\Phi(B) = 5$，每次从 $s$ 转移到 $s'$ 额外加 $F(s,s') = \gamma\,\Phi(s') - \Phi(s)$。(a) 写出方案二下 A→A、A→B、B→A、B→B 四种转移的附加奖励。(b) 证明方案二不改变任何两个策略之间的优劣，并写出新的最优动作价值。(c) 方案一下，最优策略变成了什么？按原来的奖励算，它比真正的最优策略差多少？(d) 方案一错在哪里？

## 参考解答

<details>
<summary>展开参考解答</summary>

这一轮考的是本科强化学习课的基本功：定义能不能说准，定理的条件记不记得，给一组小数字能不能当场算对。面经只说测验里有强化学习，没有留下具体题目，这一页的题是按那个难度补写的。大模型后训练里的强化学习（策略梯度、PPO、GRPO、DPO）在另一页 [大模型强化学习](#/deepmind/quiz/rl-for-llms)，这里不重复。

准备方法有三条。第一，把贝尔曼方程、三种 TD 目标、UCB 指标、重要性权重这几个式子练到能默写。第二，每个结论配一个两三个状态的小例子，亲手算一遍；面试官最常做的就是给几个数让你现场算。第三，说定理时带上条件：是表格型还是带函数近似，数据是不是当前策略采的，步长满足什么要求，奖励是否有界。追问大多落在条件上。

### MDP 与动态规划

**Q1.** 对策略 $\pi$，价值等于“这一步的奖励，加上折扣后的下一状态价值的期望”：

$$V^\pi(A) = -1 + 0.8\left[\tfrac14 V^\pi(A) + \tfrac34 V^\pi(B)\right], \qquad V^\pi(B) = 7 + 0.8\,V^\pi(A).$$

第一式整理成 $0.8\,V^\pi(A) = -1 + 0.6\,V^\pi(B)$。把第二式代进去：$0.8\,V^\pi(A) = -1 + 4.2 + 0.48\,V^\pi(A)$，即 $0.32\,V^\pi(A) = 3.2$。所以

$$V^\pi(A) = 10, \qquad V^\pi(B) = 7 + 0.8 \times 10 = 15.$$

解为什么唯一：把方程写成矩阵形式 $V = r_\pi + \gamma P_\pi V$，即 $(I - \gamma P_\pi)V = r_\pi$。$P_\pi$ 每行是一个概率分布，特征值的模都不超过 $1$，所以 $\gamma P_\pi$ 的特征值的模不超过 $\gamma < 1$，$I - \gamma P_\pi$ 可逆。另一种说法：算子 $T^\pi V = r_\pi + \gamma P_\pi V$ 在无穷范数下是 $\gamma$-压缩，不动点唯一。

用到的条件是 $\gamma < 1$ 和奖励有界。如果 $\gamma = 1$，就要求任务分回合、并且这个策略一定会走到终止状态，否则价值可能没有定义。

**Q2.** (a) 动作价值的定义是 $Q^\pi(s,a) = r(s,a) + \gamma \sum_{s'} p(s' \mid s,a)\,V^\pi(s')$。代入 $V^\pi = (10, 15)$：

- $Q^\pi(A,\text{stay}) = 1 + 0.8 \times 10 = 9$；
- $Q^\pi(A,\text{go}) = -1 + 0.8\,(0.25 \times 10 + 0.75 \times 15) = -1 + 0.8 \times 13.75 = 10$；
- $Q^\pi(B,\text{stay}) = 5 + 0.8\,(0.75 \times 15 + 0.25 \times 10) = 5 + 0.8 \times 13.75 = 16$；
- $Q^\pi(B,\text{cash}) = 7 + 0.8 \times 10 = 15$。

每个状态取 $Q^\pi$ 最大的动作：A 仍选 `go`（$10 > 9$），B 改选 `stay`（$16 > 15$）。所以 $\pi'$ 是“A 选 `go`，B 选 `stay`”。

(b) 对 $\pi'$ 列方程：$V(A) = -1 + 0.8\,[0.25\,V(A) + 0.75\,V(B)]$，$V(B) = 5 + 0.8\,[0.75\,V(B) + 0.25\,V(A)]$。第二式化成 $0.4\,V(B) = 5 + 0.2\,V(A)$，即 $V(B) = 12.5 + 0.5\,V(A)$；第一式是 $0.8\,V(A) = -1 + 0.6\,V(B)$。代入得 $0.8\,V(A) = 6.5 + 0.3\,V(A)$，所以 $V^{\pi'}(A) = 13$，$V^{\pi'}(B) = 19$。两个状态都比原来的 $(10, 15)$ 高。

(c) 用 $V^{\pi'} = (13, 19)$ 再算一遍四个动作的备份值：A 的 `stay` 是 $1 + 0.8 \times 13 = 11.4$，`go` 是 $-1 + 0.8\,(0.25 \times 13 + 0.75 \times 19) = 13$；B 的 `stay` 是 $5 + 0.8 \times 17.5 = 19$，`cash` 是 $7 + 0.8 \times 13 = 17.4$。每个状态的最大值正好是 $13$ 和 $19$，也就是 $V^{\pi'}$ 本身，所以 $V^{\pi'}$ 满足贝尔曼最优方程 $V(s) = \max_a [r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) V(s')]$。这个方程在 $\gamma < 1$ 时只有一个解，因此 $V^{\pi'} = V^*$，$\pi'$ 是最优策略。策略迭代在这里一步就结束了。

(d) A 选 `stay` 就永远留在 A：$V(A) = 1/(1 - 0.8) = 5$，$V(B) = 7 + 0.8 \times 5 = 11$。它每一步都拿了当时最大的奖励，长期却只有最优值 $(13, 19)$ 的一小半：`go` 眼前亏 $1$，换来的是进入每步 $+5$ 的状态 B。

这里用到的是策略改进定理。条件：状态和动作有限，$\gamma < 1$，$Q^\pi$ 是精确值。结论：对 $Q^\pi$ 取贪心得到的 $\pi'$ 在每个状态都不比 $\pi$ 差；如果没有哪个状态严格变好，$\pi$ 就已经满足最优方程。$Q^\pi$ 只是近似值时，这个保证不再成立。

**Q3.** (a) 先证一个小事实：对任意两组数 $f(a)$、$g(a)$，有 $\lvert \max_a f(a) - \max_a g(a) \rvert \le \max_a \lvert f(a) - g(a) \rvert$。设 $a^\star$ 让 $f$ 取到最大，则 $\max f - \max g \le f(a^\star) - g(a^\star) \le \max_a \lvert f(a) - g(a)\rvert$；交换 $f$、$g$ 得另一个方向。

对任意 $V$、$U$ 和状态 $s$，令 $f(a) = r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) V(s')$，$g(a)$ 是把 $V$ 换成 $U$ 的同一个式子。于是

$$\lvert (TV)(s) - (TU)(s) \rvert \le \max_a\ \gamma \Big\lvert \sum_{s'} p(s' \mid s,a)\,\big(V(s') - U(s')\big) \Big\rvert \le \gamma\,\lVert V - U \rVert_\infty.$$

最后一步用到 $p(\cdot \mid s,a)$ 是概率分布：加权平均的绝对值不超过最大的绝对值。对 $s$ 取最大，就得到 $\lVert TV - TU\rVert_\infty \le \gamma \lVert V - U \rVert_\infty$。

(b) 有界函数配上无穷范数是完备空间，$\gamma < 1$ 时 $T$ 是压缩映射。由巴拿赫不动点定理（Banach fixed-point theorem），$T$ 有唯一不动点，它就是 $V^*$；并且从任意有界的 $V_0$ 出发，

$$\lVert V_k - V^* \rVert_\infty \le \gamma^k\,\lVert V_0 - V^* \rVert_\infty.$$

条件是 $\gamma < 1$ 和奖励有界（这样 $\lVert V^* \rVert_\infty \le R_{\max}/(1-\gamma)$）。要注意两点：压缩是对无穷范数说的，对 2-范数不一定成立；在 $T$ 后面再接一步函数近似（投影），合起来也不一定是压缩，这正是 Q12 里发散的来源。

(c) $V_0 = 0$ 时 $\lVert V_0 - V^*\rVert_\infty \le 7/(1 - 0.8) = 35$。要 $35 \times 0.8^k \le 0.01$，即 $k \ge \ln 3500 / \ln 1.25 = 8.161/0.2231 = 36.57$，所以 **37 轮**。一般地 $k \ge \ln\!\big(R_{\max}/((1-\gamma)\epsilon)\big) / \ln(1/\gamma)$，大约是 $\frac{1}{1-\gamma} \ln \frac{R_{\max}}{(1-\gamma)\epsilon}$：轮数与 $1/(1-\gamma)$ 成正比。同样的精度，$\gamma = 0.99$ 时要 1111 轮。

(d) 逐轮算（每个状态对两个动作取最大）：

- $V_1 = (\max(1, -1),\ \max(5, 7)) = (1, 7)$；
- $V_2$：A 的 `stay` 是 $1 + 0.8 \times 1 = 1.8$，`go` 是 $-1 + 0.8\,(0.25 \times 1 + 0.75 \times 7) = 3.4$；B 的 `stay` 是 $5 + 0.8\,(0.75 \times 7 + 0.25 \times 1) = 9.4$，`cash` 是 $7 + 0.8 \times 1 = 7.8$。所以 $V_2 = (3.4, 9.4)$；
- $V_3$：A 的 `go` 是 $-1 + 0.8\,(0.25 \times 3.4 + 0.75 \times 9.4) = 5.32$，B 的 `stay` 是 $5 + 0.8 \times 7.9 = 11.32$，都大于另一个动作（$3.72$ 和 $9.72$）。所以 $V_3 = (5.32, 11.32)$。

与 $V^* = (13, 19)$ 的距离依次是 $19$（$V_0$）、$12$、$9.6$、$7.68$。第一轮之后，每轮正好乘 $0.8$。原因是 $V_1 = V^* - 12 \cdot \mathbf 1$，两个分量差得一样多；而对任意常数 $c$ 有 $T(V - c\,\mathbf 1) = TV - \gamma c\,\mathbf 1$（每个备份值都少了 $\gamma c$）。所以 $V_k = V^* - 12 \times 0.8^{k-1}\,\mathbf 1$。要 $12 \times 0.8^{k-1} \le 0.01$，即 $k - 1 \ge \ln 1200/\ln 1.25 = 31.77$，所以实际需要 **33 轮**。

上界给的 37 轮偏保守，有两个原因：它把初始距离估成了 $35$（实际是 $19$，换成 $19$ 上界给出 34 轮）；第一轮实际把误差缩到了 $12/19 \approx 0.63$ 倍，比 $0.8$ 更快。实际使用时不知道 $V^*$，常用的停止条件是相邻两轮的变化量：$\lVert V_{k+1} - V^* \rVert_\infty \le \frac{\gamma}{1-\gamma} \lVert V_{k+1} - V_k \rVert_\infty$。

**Q4.** (a) 两种解释。

- 权重之和：$\sum_{t \ge 0} \gamma^t = 1/(1-\gamma) = 100$。每步奖励恒为 $1$ 时折扣回报就是 $100$，相当于不折扣地数了 100 步。
- 随机终止：把折扣看成“每走一步之后，以概率 $1-\gamma$ 结束”。第 $t$ 步（从 $0$ 数起）还没结束的概率是 $\gamma^t$，所以折扣回报的期望，等于这个随机终止过程里不折扣回报的期望；总步数服从几何分布，期望是 $1/(1-\gamma) = 100$ 步。

(b) 前 100 步的权重占比是 $\sum_{t=0}^{99} \gamma^t \big/ \sum_{t \ge 0} \gamma^t = 1 - \gamma^{100} = 1 - 0.366 = 0.634$。这个数接近 $1 - 1/e$，因为 $0.99^{100} \approx e^{-1}$。也就是说，“100 步”之外还有约 37% 的权重，有效视野不是一个截断点。

(c) 要 $1 - \gamma^n \ge 0.99$，即 $\gamma^n \le 0.01$，$n \ge \ln 100 / \ln(1/0.99) = 4.605/0.01005 = 458.2$，所以要 **459 步**，约为有效视野的 4.6 倍。

(d) 回报在 $[0,\ 1/(1-\gamma)] = [0, 100]$ 内。在第 $H$ 步截断，丢掉的部分最多是 $\sum_{t \ge H} \gamma^t = \gamma^H/(1-\gamma)$；$H = 1000$ 时是 $100 \times 0.99^{1000} \approx 0.0043$。

### 时序差分学习

**Q5.** (a) 三种方法的更新都是 $Q(A,\text{go}) \leftarrow Q(A,\text{go}) + \alpha\,[\text{目标} - Q(A,\text{go})]$，目标分别是：

- SARSA：用下一步实际选的动作。目标 $= r + \gamma\,Q(B,\text{cash}) = -1 + 0.8 \times 10 = 7$，更新后 $8 + 0.5 \times (7 - 8) = 7.5$。
- Q-learning：用下一状态的最大值，不管实际选了什么。目标 $= r + \gamma \max_a Q(B,a) = -1 + 0.8 \times 12 = 8.6$，更新后 $8 + 0.5 \times 0.6 = 8.3$。
- Expected SARSA：对下一步的动作按策略求期望。在 B，贪心动作是 `stay`（$12 > 10$），被选中的概率是 $1 - \varepsilon + \varepsilon/2 = 0.9$，`cash` 是 $0.1$；期望值 $0.9 \times 12 + 0.1 \times 10 = 11.8$。目标 $= -1 + 0.8 \times 11.8 = 8.44$，更新后 $8 + 0.5 \times 0.44 = 8.22$。

(b) 差别只在自举（bootstrapping）的那一项，也就是用什么来代表“下一状态之后的价值”：SARSA 用实际采到的 $Q(s',a')$，Q-learning 用 $\max_{a'} Q(s',a')$，Expected SARSA 用 $\sum_{a'} \pi(a' \mid s')\,Q(s',a')$。Expected SARSA 的目标与 SARSA 的目标期望相同，只是去掉了“下一个动作是随机抽的”带来的方差。

(c) Q-learning 收敛到最优动作价值 $Q^*$，与行为策略怎么探索无关。条件：表格型；每个状态-动作对都被访问无穷多次；每个状态-动作对的步长满足 $\sum_t \alpha_t = \infty$、$\sum_t \alpha_t^2 < \infty$；奖励有界；$\gamma < 1$（Watkins 与 Dayan，1992）。固定步长 $\alpha = 0.5$ 不满足第三条，只会在 $Q^*$ 附近抖动。

SARSA 学的是它正在执行的策略的价值。$\varepsilon$ 固定时，在同样的步长和访问条件下，它不收敛到 $Q^*$，而是收敛到“$\varepsilon$-贪心算子”的不动点：

$$Q(s,a) = r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) \Big[(1-\varepsilon) \max_{a'} Q(s',a') + \frac{\varepsilon}{\lvert\mathcal A\rvert} \sum_{a'} Q(s',a')\Big].$$

这个算子同样是 $\gamma$-压缩，不动点唯一；它是所有“保留 $\varepsilon$ 探索”的策略里最好的那个的价值。要让 SARSA 收敛到 $Q^*$，$\varepsilon$ 必须逐渐减到 $0$，同时仍然把每个动作试无穷多次（这组条件叫 GLIE，greedy in the limit with infinite exploration；Singh 等，2000）。

一个算得出来的例子：取两状态 MDP（$\gamma = 0.8$；A：`stay` 得 $+1$ 留在 A，`go` 得 $-1$、以 $3/4$ 到 B；B：`stay` 得 $+5$、以 $3/4$ 留在 B，`cash` 得 $+7$ 回到 A），它的 $Q^*$ 是 $(11.4,\ 13;\ 19,\ 17.4)$，两个状态里最好和次好动作都差 $1.6$。设不动点是 $Q^* - c$，那么每个状态在 $\varepsilon$-贪心下的价值是 $V^*(s) - c - \tfrac{\varepsilon}{2} \times 1.6 = V^*(s) - c - 0.16$，代回去得 $c = 0.8\,(c + 0.16)$，$c = 0.64$。所以 $\varepsilon = 0.2$ 的 SARSA 学到的每个 $Q$ 值都比 $Q^*$ 低 $0.64$，即 $(10.76,\ 12.36;\ 18.36,\ 16.76)$。这个例子里贪心动作没变；在“悬崖行走”那样探索代价很大的任务里，两者的贪心策略也会不同。

**Q6.** (a) 蒙特卡洛用每个状态之后实际拿到的回报的平均值。B 在 10 条轨迹里各出现一次，之后的回报是 $0, 1$，六个 $1$，两个 $0$，共 7 个 $1$：$V(B) = 7/10 = 0.7$。A 出现两次，之后的回报是 $0 + 0 = 0$ 和 $0 + 1 = 1$：$V(A) = 0.5$。

(b) TD(0) 的不动点要求每个状态上的平均 TD 误差为零。B 之后直接终止，目标就是奖励本身，$V(B) = 0.7$，与蒙特卡洛相同。A 的两条转移都是“奖励 $0$，到 B”，所以 $V(A) = 0 + V(B) = 0.7$。

(c) 两种方法回答的是两个不同的问题。

- 批量蒙特卡洛让训练数据上的回报平方误差最小：A 之后见过的回报只有 $0$ 和 $1$ 各一次，平均就是 $0.5$。
- 批量 TD(0) 给出的是“最大似然马尔可夫模型”下的精确价值，叫确定性等价估计（certainty-equivalence estimate）：从数据估出 A 一定到 B、奖励 $0$，B 的期望奖励是 $0.7$，然后在这个模型里解方程。

如果过程确实是马尔可夫的，TD 的答案更可信：它用了全部 10 次经过 B 的数据来估计 A 之后会发生什么，而蒙特卡洛只用了从 A 出发的 2 条。如果状态不是马尔可夫的（例如从 A 过来的 B 和直接开始的 B 其实不一样），TD 的模型假设就错了，蒙特卡洛不依赖这个假设。这个结论的前提是：表格型，同一批数据反复回放，步长足够小。

**Q7.** (a) 记十步的奖励为 $r_1, \dots, r_{10}$，每个的均值是 $0$、方差是 $1$。

- 蒙特卡洛目标是整条回报 $G = r_1 + \dots + r_{10}$。期望 $0$，偏差 $0$；方差是十个独立项之和，等于 $10$；均方误差 $10$。
- TD(0) 目标是 $r_1 + V(s_2) = r_1 + 0.5$。期望 $0.5$，偏差 $+0.5$（全部来自 $V(s_2)$ 的误差）；方差只有一步奖励的方差 $1$；均方误差 $1 + 0.5^2 = 1.25$。

(b) TD 误差 $\delta = r_1 + V(s_2) - V(s_1) = 1 + 0.5 - 0 = 1.5$，$V(s_1) \leftarrow 0 + 0.1 \times 1.5 = 0.15$。

(c) 蒙特卡洛目标无偏，但方差随剩余步数增长，因为它把后面所有的随机性都加了进来。TD 目标只含一步的随机性，方差小；代价是只要下一状态的估计不准，它就有偏差。这个例子里 TD 目标的均方误差是 $1.25$，蒙特卡洛是 $10$，这就是 TD 通常学得更快的原因。$n$ 步回报和 TD($\lambda$) 在两者之间调节。

偏差不会让表格型 TD(0) 收敛到错的地方。这里的偏差是“当前目标”的偏差，它来自 $V(s_2)$ 还不准；随着 $V(s_2)$ 被更新到真值，偏差也跟着消失。对固定策略、表格型、每个状态被访问无穷多次、步长满足 $\sum_t \alpha_t = \infty$、$\sum_t \alpha_t^2 < \infty$ 的情形，TD(0) 以概率 1 收敛到真实价值（Jaakkola、Jordan 与 Singh，1994；Tsitsiklis，1994）。换成函数近似就不同了：线性 TD 的不动点一般不等于最小二乘意义下的最佳近似。

### 多臂老虎机

**Q8.** (a) 遗憾是“一直拉最好的臂能拿到的期望奖励”减去“算法实际拿到的期望奖励”：

$$R_T = T\mu^* - \mathbb E\Big[\sum_{t=1}^T r_t\Big] = \sum_i \Delta_i\ \mathbb E[N_i(T)],$$

其中 $\mu^* = \max_i \mu_i$，$\Delta_i = \mu^* - \mu_i$ 是臂 $i$ 的间隔（gap），$N_i(T)$ 是它在前 $T$ 步被拉的次数。这里只有臂 2 有间隔，$\Delta_2 = 0.2$。

(b) 每一步都有 $\varepsilon$ 的概率做均匀探索，探索时拉到臂 2 的概率是 $1/2$。所以不管估计得多准，每一步拉臂 2 的概率至少是 $\varepsilon/2 = 0.05$，$\mathbb E[N_2(T)] \ge 0.05\,T$，

$$R_T \ge 0.2 \times 0.05 \times T = 0.01\,T.$$

$T = 10^4$ 时至少是 $100$。遗憾随 $T$ 线性增长。这个下界也接近实际值：估计稳定之后，贪心的那些步都拉臂 1，每步的期望遗憾正好是 $0.01$。模拟结果（两个估计相等时拉臂 1）是 $T = 10^4$ 时约 $102$，$T = 2 \times 10^4$ 时约 $202$；相等时随机选，会再多 5 左右。

(c) 让 $\varepsilon$ 随时间衰减，$\varepsilon_t \propto 1/t$。这样探索的总次数是 $\sum_t \varepsilon_t \approx c \ln T$，遗憾可以做到 $O(\log T)$。Auer、Cesa-Bianchi 与 Fischer（2002）给的形式是 $\varepsilon_t = \min(1,\ cK/(d^2 t))$，$K$ 是臂数；前提是 $0 < d \le \min_i \Delta_i$ 并且常数 $c$ 足够大，这时第 $t$ 步拉到次优臂的概率是 $O(1/t)$。代价是要事先知道间隔的一个下界：$c/d^2$ 取小了，探索不够，算法可能长期停在错的臂上，遗憾就不再是对数级的，在实际的步数内可能接近线性。不想依赖间隔，就用 UCB 或汤普森采样。

**Q9.** (a) $\ln 100 = 4.605$。

- 臂 1：$0.60 + \sqrt{2 \times 4.605/90} = 0.60 + 0.320 = 0.920$；
- 臂 2：$0.40 + \sqrt{2 \times 4.605/10} = 0.40 + 0.960 = 1.360$。

下一步拉**臂 2**。它的样本均值低，但只拉过 10 次，不确定性大得多。

(b) 奖励在 $[0,1]$ 内、各次独立时，霍夫丁不等式给出两个方向的界：$P(\bar x_i \le \mu_i - a) \le \exp(-2 n_i a^2)$，$P(\bar x_i \ge \mu_i + a) \le \exp(-2 n_i a^2)$。取 $a = \sqrt{2 \ln n / n_i}$，右边都是 $\exp(-4 \ln n) = n^{-4}$。第一个式子说，指标 $\bar x_i + a$ 低于真值 $\mu_i$ 的概率不超过 $n^{-4}$，所以指标是 $\mu_i$ 的一个置信上界（upper confidence bound）；用在最优臂上，就是最优臂几乎不会被低估。第二个式子说，次优臂的样本均值几乎不会被高估超过 $a$。$n^{-4}$ 衰减得足够快，对所有时刻和所有可能的 $n_i$ 求和仍然有限，这样“置信区间失效”造成的总损失是个常数。

两个界都成立时，次优臂 $i$ 的指标不超过 $\mu_i + 2a$，最优臂的指标不低于 $\mu^*$。所以一旦 $2\sqrt{2\ln n/n_i} < \Delta_i$，即 $n_i > 8 \ln n/\Delta_i^2$，次优臂的指标就一定低于最优臂的指标。所以每个次优臂大约只会被拉 $8 \ln n / \Delta_i^2$ 次。这就是“面对不确定时保持乐观”的做法：没试够的臂给高分，试够了还不行的臂自然被放下。

(c) 按 $\ln T$ 增长。Auer、Cesa-Bianchi 与 Fischer（2002）的定理 1：奖励在 $[0,1]$ 内，每个臂的各次奖励独立同分布，则对任意步数 $n$，

$$R_n \le 8 \sum_{i:\,\Delta_i > 0} \frac{\ln n}{\Delta_i} + \Big(1 + \frac{\pi^2}{3}\Big) \sum_j \Delta_j.$$

代入 $\Delta = 0.2$、$n = 10^4$：$8 \times 9.210/0.2 + 4.290 \times 0.2 = 368.4 + 0.9 = 369.3$。相当于臂 2 最多被拉约 $1842 + 4.3$ 次。这个界的常数很松：模拟里 UCB1 在 $10^4$ 步时的遗憾只有 60 出头。但量级是对的：把 $T$ 从 $10^4$ 加倍到 $2 \times 10^4$，UCB1 的遗憾只多了十来点，而固定 $\varepsilon = 0.1$ 的 $\varepsilon$-贪心会多 100。

(d) 量级不能再改进，常数可以。Lai 与 Robbins（1985）的下界：对伯努利臂，任何“一致好”的算法（在每个问题实例上遗憾都是 $o(T^a)$，对所有 $a > 0$）都满足

$$\liminf_{T \to \infty} \frac{\mathbb E[N_i(T)]}{\ln T} \ge \frac{1}{\mathrm{KL}(\mu_i \,\Vert\, \mu^*)}.$$

这里 $\mathrm{KL}(0.4 \,\Vert\, 0.6) = 0.4 \ln\frac{0.4}{0.6} + 0.6 \ln\frac{0.6}{0.4} = 0.2 \ln 1.5 = 0.0811$，所以遗憾渐近地至少是 $\frac{0.2}{0.0811} \ln T \approx 2.47 \ln T$。作为参照，$T = 10^4$ 时 $2.47 \ln T \approx 22.7$；这是渐近结论，不能当成有限步数下的严格下界。UCB1 的界里系数是 $8/\Delta = 40$，离 $2.47$ 很远；KL-UCB 和汤普森采样能达到这个最优常数。

**Q10.** (a) Beta 分布是伯努利似然的共轭先验：$\mathrm{Beta}(a,b)$ 每见到一次成功 $a$ 加 1，每见到一次失败 $b$ 加 1，均值是 $a/(a+b)$。

- 臂 1：2 次成功、1 次失败，后验 $\mathrm{Beta}(3,2)$，均值 $3/5$；
- 臂 2：0 次成功、1 次失败，后验 $\mathrm{Beta}(1,2)$，均值 $1/3$。

(b) 汤普森采样从每个臂的后验里各抽一个样本 $\theta_1$、$\theta_2$，拉样本值大的那个臂。所以它选臂 2 的概率就是后验下 $\theta_2 > \theta_1$ 的概率。$\mathrm{Beta}(3,2)$ 的密度是 $12x^2(1-x)$；$\mathrm{Beta}(1,2)$ 的密度是 $2(1-y)$，所以 $P(\theta_2 > x) = (1-x)^2$。于是

$$P(\theta_2 > \theta_1) = \int_0^1 12x^2(1-x)\,(1-x)^2\,dx = 12 \int_0^1 x^2 (1-x)^3\,dx = 12 \cdot \frac{2!\,3!}{6!} = \frac{1}{5}.$$

汤普森采样以 $1/5$ 的概率拉臂 2。按后验均值贪心则一定拉臂 1（$3/5 > 1/3$），选臂 2 的概率是 $0$。

(c) 臂 2 多了一次成功，后验变成 $\mathrm{Beta}(2,2)$，均值 $1/2$。

(d) 汤普森采样选每个臂的概率，等于“这个臂是最优臂”的后验概率（概率匹配）。数据少的臂后验宽，抽到大值的机会多，于是得到探索；数据多了后验变窄，明显差的臂几乎不会再被选中，探索自动收掉。它不需要调探索参数。对伯努利臂，它的遗憾渐近地达到 Lai–Robbins 下界（Kaufmann、Korda 与 Munos，2012）。

### 离策略评估、函数近似与奖励塑形

**Q11.** (a) $v_\pi = 0.9 \times 1 + 0.1 \times 0 = 0.9$。权重 $\rho(L) = 0.9/0.5 = 1.8$，$\rho(R) = 0.1/0.5 = 0.2$。

(b) $n = 1$。

- 普通估计是 $\rho r$：抽到 L 时为 $1.8$，抽到 R 时为 $0$，各占一半。期望 $0.9$，无偏；方差 $\tfrac12 \times 1.8^2 - 0.9^2 = 0.81$；均方误差 $0.81$。
- 加权估计是 $\rho r/\rho = r$，权重被约掉了：以概率 $1/2$ 取 $1$，以概率 $1/2$ 取 $0$。期望 $0.5$，偏差 $-0.4$；方差 $0.25$；均方误差 $0.25 + 0.16 = 0.41$。单个样本时，加权估计的期望其实是行为策略的价值 $v_\mu$。
- 直接用 $\pi$ 采一个样本：奖励是成功概率 $0.9$ 的伯努利变量，方差 $0.9 \times 0.1 = 0.09$。普通重要性采样的方差是它的 9 倍。

(c) $n = 2$。四种等可能的情况：

| 两个动作 | 普通估计 | 加权估计 |
| --- | --- | --- |
| L, L | $(1.8 + 1.8)/2 = 1.8$ | $3.6/3.6 = 1$ |
| L, R 或 R, L | $1.8/2 = 0.9$ | $1.8/2.0 = 0.9$ |
| R, R | $0$ | $0/0.4 = 0$ |

加权估计：期望 $(1 + 0.9 + 0.9 + 0)/4 = 0.7$，偏差 $-0.2$；均方误差 $(0.1^2 + 0 + 0 + 0.9^2)/4 = 0.205$。普通估计：期望 $0.9$，方差（也就是均方误差）$(0.9^2 + 0 + 0 + 0.9^2)/4 = 0.405$。两个样本时加权估计的均方误差已经只有普通估计的一半。再往后差距更大：$n = 10$ 时加权估计的期望是 $0.882$、均方误差 $0.0068$，普通估计的方差是 $0.081$。

(d) 一般性质。

- 普通重要性采样无偏。它的方差由权重决定，权重无界时方差可以非常大甚至无穷。
- 加权重要性采样有偏，偏差随 $n$ 增大趋于 $0$，是一致估计。它是观测到的回报的加权平均，不会超出回报的取值范围，所以回报有界时方差有界，实际中通常更稳。
- 两者都需要覆盖条件（coverage）：凡是 $\pi(a \mid s) > 0$ 的地方都要有 $\mu(a \mid s) > 0$，否则目标策略会做的事在数据里从未出现，无从估计。还需要知道（或能估计）行为策略的概率。

多步问题里权重是每一步比值的连乘，方差随步数指数增长，这一点在 [大模型强化学习](#/deepmind/quiz/rl-for-llms) 那一页的最后一问里有推导。

**Q12.** (a) 三样东西是：函数近似（function approximation，不同状态共用参数）、自举（用自己的估计当目标，如 TD 和 Q-learning）、离策略训练（off-policy，更新所用的状态分布不是目标策略自己产生的分布）。三者同时出现时，学习**可能**不稳定甚至发散，不是一定发散；带经验回放和目标网络的 DQN 三样都有，实践中常常能训练好。

(b) 状态 1 的估计 $w$ 对参数的梯度是 $1$。半梯度 TD(0) 只对当前状态的估计求导，把目标当常数：

$$w \leftarrow w + \alpha\,\big[r + \gamma \cdot 2w - w\big] \cdot 1 = w\,\big[1 + \alpha(2\gamma - 1)\big].$$

代入 $\gamma = 0.9$、$\alpha = 0.1$：每次乘 $1.08$。10 次之后 $w = 1.08^{10} = 2.159$，并且一直增长下去，而正确答案是 $w = 0$。只要 $2\gamma - 1 > 0$，即 $\gamma > 1/2$，对任何 $\alpha > 0$ 都发散（$\gamma = 1/2$ 时 $w$ 不动，$\gamma < 1/2$ 时收敛到 $0$）。直观上：把状态 1 的估计往 $2\gamma w$ 拉高时，因为共用参数，状态 2 的估计也跟着涨了两倍的量，目标永远追不上。

(c) 状态 2 的估计 $2w$ 对参数的梯度是 $2$，它的转移通向终止态，目标是 $0$：

$$w \leftarrow w + \alpha\,[0 + 0 - 2w] \cdot 2 = w\,(1 - 4\alpha) = 0.6\,w.$$

一个回合两次更新合起来乘 $1.08 \times 0.6 = 0.648$。10 个回合后 $w = 0.648^{10} = 0.013$，收敛到正确的 $0$。同策略采样时，状态 2 自己的更新会把被抬高的估计拉回来；(b) 里发散，是因为状态 2 的估计只被用作目标，从来没有被自己的转移纠正过。

(d) 去掉任何一个，另外两个可以共存：

- 不用函数近似：表格型 Q-learning 既离策略又自举，在 Q5 所说的条件下收敛到 $Q^*$。
- 不离策略：线性函数近似加同策略 TD(0)，在状态按策略的平稳分布采样、特征线性无关、步长满足 $\sum_t \alpha_t = \infty$、$\sum_t \alpha_t^2 < \infty$ 的条件下收敛，极限点的误差不超过最佳线性近似误差的一个常数倍，这个倍数在 $\gamma \to 1$ 时变大（Tsitsiklis 与 Van Roy，1997）。
- 不自举：用蒙特卡洛回报做回归目标（离策略时加重要性权重），就是普通的监督学习，随机梯度下降在通常条件下收敛，代价是方差大。

实际做法通常是保留三者但削弱其中一个：目标网络减弱自举，多步回报减少对自举的依赖，让回放数据的分布靠近当前策略以减轻离策略的程度。

**Q13.** (a) 代入 $F(s,s') = 0.8\,\Phi(s') - \Phi(s)$：

- A→A：$0.8 \times 0 - 0 = 0$；
- A→B：$0.8 \times 5 - 0 = +4$；
- B→A：$0.8 \times 0 - 5 = -5$；
- B→B：$0.8 \times 5 - 5 = -1$。

进入 B 的奖励与方案一相同，但离开 B 要把势“还回去”。

(b) 沿任意一条从 $s_0$ 出发的轨迹，附加奖励的折扣和是一个逐项相消的和：

$$\sum_{t \ge 0} \gamma^t \big[\gamma \Phi(s_{t+1}) - \Phi(s_t)\big] = -\Phi(s_0) + \lim_{T \to \infty} \gamma^T \Phi(s_T) = -\Phi(s_0).$$

所以对任何策略 $\pi$ 和任何 $(s,a)$，新的动作价值是 $Q'^\pi(s,a) = Q^\pi(s,a) - \Phi(s)$。减掉的量只与当前状态有关，与动作和策略都无关，因此同一状态下各动作的排序不变，任何两个策略的优劣也不变，最优策略还是原来的。新的最优动作价值：

- A：`stay` $11.4 - 0 = 11.4$，`go` $13 - 0 = 13$；
- B：`stay` $19 - 5 = 14$，`cash` $17.4 - 5 = 12.4$。

这是 Ng、Harada 与 Russell（1999）的结论。条件：$\Phi$ 只依赖状态并且有界；$F$ 里的 $\gamma$ 就是 MDP 的折扣因子；不折扣的回合制任务还要求终止状态的势为 $0$。他们还证明了反方向：如果附加奖励不是这种形式，总能找到一个 MDP，让最优策略被它改掉。

(c) 方案一只改了 A 选 `go` 的期望奖励：$-1 + \tfrac34 \times 4 = 2$。在新奖励下比较 A 选 `go` 时 B 的两种选法（记 $V_A$、$V_B$）。两种情况都有 $V_A = 2 + 0.8\,(0.25 V_A + 0.75 V_B)$，即 $0.8\,V_A = 2 + 0.6\,V_B$。

- B 选 `stay`：$V_B = 5 + 0.8\,(0.75 V_B + 0.25 V_A)$，即 $V_B = 12.5 + 0.5\,V_A$。解得 $V_A = 19$，$V_B = 22$。
- B 选 `cash`：$V_B = 7 + 0.8\,V_A$。解得 $0.32\,V_A = 6.2$，$V_A = 19.375$，$V_B = 22.5$。

`cash` 的方案在两个状态都更高。再核对它满足最优方程：以 $(19.375, 22.5)$ 计算，B 的 `stay` 是 $5 + 0.8\,(0.75 \times 22.5 + 0.25 \times 19.375) = 22.375 < 22.5$，A 的 `stay` 是 $1 + 0.8 \times 19.375 = 16.5 < 19.375$。所以方案一下的最优策略变成了“A 选 `go`，B 选 `cash`”。

按原来的奖励，这个策略的价值是 $(10, 15)$，真正的最优是 $(13, 19)$：在 A 少 $3$，在 B 少 $4$。

(d) 方案一每次进入 B 都发奖，离开 B 却不扣，所以“进 B、马上回 A、再进 B”这个循环可以反复领奖。智能体学到的是刷这个循环，而不是留在 B。基于势函数的方案里，绕一圈回到原状态，附加奖励按折扣相消，刷不出好处。要加塑形奖励，就写成 $\gamma\Phi(s') - \Phi(s)$ 的形式；常见的建议是让 $\Phi$ 接近 $V^*$ 的一个粗略估计。

<details>
<summary>验证代码（可运行）</summary>

```python
import itertools
import math
from fractions import Fraction as F

import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)

# ---- the two-state MDP of Q1, Q2, Q3, Q5 and Q13 ----
# states: 0 = A, 1 = B.   actions in A: 0 = stay, 1 = go.   actions in B: 0 = stay, 1 = cash.
GAMMA = 0.8
P = np.array([[[1.0, 0.0], [0.25, 0.75]],        # P[s, a, s']
              [[0.25, 0.75], [1.0, 0.0]]])
R = np.array([[1.0, -1.0], [5.0, 7.0]])          # R[s, a]
POLICIES = list(itertools.product((0, 1), repeat=2))


def solve_policy(policy):
    """V^pi from the linear system (I - gamma * P_pi) V = r_pi."""
    p_pi = np.array([P[s, policy[s]] for s in range(2)])
    r_pi = np.array([R[s, policy[s]] for s in range(2)])
    return np.linalg.solve(np.eye(2) - GAMMA * p_pi, r_pi)


def iterate_policy(policy, sweeps=300):
    """V^pi by applying the Bellman expectation backup again and again."""
    v = np.zeros(2)
    for _ in range(sweeps):
        v = np.array([R[s, policy[s]] + GAMMA * P[s, policy[s]] @ v for s in range(2)])
    return v


def rollout_value(policy, start, episodes=20000, horizon=80):
    """V^pi(start) by simulating the chain; 0.8**80 is about 2e-8, so cutting there costs nothing."""
    s = np.full(episodes, start)
    ret = np.zeros(episodes)
    discount = 1.0
    for _ in range(horizon):
        a = np.array(policy)[s]
        ret += discount * R[s, a]
        s = (rng.random(episodes) < P[s, a, 1]).astype(int)
        discount *= GAMMA
    return ret.mean()


def bellman_opt(v):
    return (R + GAMMA * P @ v).max(axis=1)


# ---- Q1: V^pi for pi = (go, cash) ----
PI = (1, 1)
assert np.allclose(solve_policy(PI), [10, 15])
assert np.allclose(iterate_policy(PI), [10, 15])
assert abs(rollout_value(PI, 0) - 10) < 0.15 and abs(rollout_value(PI, 1) - 15) < 0.15

# ---- Q2: one step of policy improvement, the optimal policy, the myopic policy ----
q_pi = R + GAMMA * P @ solve_policy(PI)
assert np.allclose(q_pi, [[9, 10], [16, 15]])
improved = tuple(int(a) for a in q_pi.argmax(axis=1))
assert improved == (1, 0)                                    # A: go, B: stay
values = {pol: solve_policy(pol) for pol in POLICIES}
assert np.allclose(values[(0, 0)], [5, 15]) and np.allclose(values[(0, 1)], [5, 11])
assert np.allclose(values[(1, 0)], [13, 19]) and np.allclose(values[(1, 1)], [10, 15])
for pol in POLICIES:                                         # (go, stay) beats every other policy in both states
    assert np.all(values[(1, 0)] >= values[pol] - 1e-12)
v_star = np.zeros(2)
for _ in range(300):                                         # second route: value iteration
    v_star = bellman_opt(v_star)
assert np.allclose(v_star, [13, 19])
q_star = R + GAMMA * P @ v_star
assert np.allclose(q_star, [[11.4, 13], [19, 17.4]])

# ---- Q3: contraction, and how many sweeps value iteration needs ----
for _ in range(2000):
    v, u = rng.normal(0, 10, 2), rng.normal(0, 10, 2)
    assert np.abs(bellman_opt(v) - bellman_opt(u)).max() <= GAMMA * np.abs(v - u).max() + 1e-12
bound_sweeps = math.ceil(math.log(35 / 0.01) / math.log(1 / GAMMA))
assert bound_sweeps == 37 and 35 * GAMMA ** 37 <= 0.01 < 35 * GAMMA ** 36
v, errors = np.zeros(2), [19.0]
while errors[-1] > 0.01:
    v = bellman_opt(v)
    errors.append(np.abs(v - [13, 19]).max())
    if len(errors) == 2:
        assert np.allclose(v, [1, 7])
    if len(errors) == 3:
        assert np.allclose(v, [3.4, 9.4])
assert len(errors) - 1 == 33 <= bound_sweeps                 # 33 sweeps are really needed
assert np.allclose(errors[:4], [19, 12, 9.6, 7.68])
assert all(math.isclose(errors[k + 1], GAMMA * errors[k]) for k in range(1, 33))
assert math.ceil(math.log(19 / 0.01) / math.log(1 / GAMMA)) == 34             # the bound with the true distance 19
v_prev = np.zeros(2)
for _ in range(40):                                          # stopping rule: error <= gamma / (1 - gamma) * last change
    v_next = bellman_opt(v_prev)
    change = np.abs(v_next - v_prev).max()
    assert np.abs(v_next - [13, 19]).max() <= GAMMA / (1 - GAMMA) * change + 1e-9
    v_prev = v_next
assert math.ceil(math.log(7 / (0.01 * 0.01)) / math.log(1 / 0.99)) == 1111     # same accuracy with gamma = 0.99

# ---- Q4: effective horizon for gamma = 0.99 ----
weights = 0.99 ** np.arange(200000)
assert math.isclose(weights.sum(), 100, rel_tol=1e-9)
assert math.isclose(weights[:100].sum() / weights.sum(), 1 - 0.99 ** 100) and round(1 - 0.99 ** 100, 3) == 0.634
share = np.cumsum(weights) / weights.sum()
assert int(np.argmax(share >= 0.99)) + 1 == 459 == math.ceil(math.log(100) / math.log(1 / 0.99))
assert round(0.99 ** 100, 2) == 0.37 and 4e-5 < 0.99 ** 1000 < 5e-5
assert round(0.99 ** 1000 / (1 - 0.99), 4) == 0.0043         # most a return can lose when cut at 1000 steps
assert math.isclose(weights[1000:].sum(), 0.99 ** 1000 / (1 - 0.99), rel_tol=1e-6)

# ---- Q5: one update of SARSA, Q-learning and Expected SARSA; what each converges to ----
q_table = {("A", "stay"): 6.0, ("A", "go"): 8.0, ("B", "stay"): 12.0, ("B", "cash"): 10.0}
alpha, eps, reward = 0.5, 0.2, -1.0
target_sarsa = reward + GAMMA * q_table[("B", "cash")]
target_q = reward + GAMMA * max(q_table[("B", "stay")], q_table[("B", "cash")])
prob_greedy = 1 - eps + eps / 2
target_exp = reward + GAMMA * (prob_greedy * 12.0 + (1 - prob_greedy) * 10.0)
assert math.isclose(target_sarsa, 7.0) and math.isclose(target_q, 8.6) and math.isclose(target_exp, 8.44)
new = [q_table[("A", "go")] + alpha * (t - q_table[("A", "go")]) for t in (target_sarsa, target_q, target_exp)]
assert np.allclose(new, [7.5, 8.3, 8.22])

q_eps = np.zeros((2, 2))                                     # fixed point of the eps-greedy backup
for _ in range(400):
    v_eps = (1 - eps) * q_eps.max(axis=1) + eps * q_eps.mean(axis=1)
    q_eps = R + GAMMA * P @ v_eps
assert np.allclose(q_eps, q_star - 0.64) and np.allclose(q_eps, [[10.76, 12.36], [18.36, 16.76]])


def run_td(kind, steps=100000):
    """Tabular SARSA or Q-learning on the MDP, eps-greedy behaviour, step size n(s, a) ** -0.7."""
    q, visits = np.zeros((2, 2)), np.zeros((2, 2))
    u = rng.random((steps + 1, 3))

    def act(state, row):
        return int(q[state].argmax()) if row[0] >= eps else int(row[1] < 0.5)

    s = 0
    a = act(s, u[0])
    for t in range(steps):
        s2 = int(u[t, 2] < P[s, a, 1])
        a2 = act(s2, u[t + 1])
        bootstrap = q[s2, a2] if kind == "sarsa" else q[s2].max()
        visits[s, a] += 1
        q[s, a] += (R[s, a] + GAMMA * bootstrap - q[s, a]) / visits[s, a] ** 0.7
        s, a = s2, a2
    return q


q_learned, sarsa_learned = run_td("q"), run_td("sarsa")
assert np.abs(q_learned - q_star).max() < 0.25               # Q-learning finds Q* although it explores
assert np.abs(sarsa_learned - q_eps).max() < 0.25            # SARSA finds the eps-greedy fixed point, 0.64 lower
assert np.abs(sarsa_learned - q_star).max() > 0.39

# ---- Q6: batch Monte Carlo and batch TD(0) on the ten episodes ----
episodes = ([[("A", 0, "B"), ("B", 0, None)], [("A", 0, "B"), ("B", 1, None)]]
            + [[("B", 1, None)]] * 6 + [[("B", 0, None)]] * 2)
returns = {"A": [], "B": []}
for ep in episodes:
    g = 0
    for state, r, _ in reversed(ep):
        g += r
        returns[state].append(g)
mc = {s: F(sum(v), len(v)) for s, v in returns.items()}
assert mc == {"A": F(1, 2), "B": F(7, 10)}
td = {"A": 0.0, "B": 0.0}
for _ in range(5000):                                        # sweep the batch until nothing changes
    delta = {"A": 0.0, "B": 0.0}
    for ep in episodes:
        for state, r, nxt in ep:
            delta[state] += r + (td[nxt] if nxt else 0.0) - td[state]
    for state in td:
        td[state] += 0.02 * delta[state]
assert math.isclose(td["A"], 0.7, abs_tol=1e-9) and math.isclose(td["B"], 0.7, abs_tol=1e-9)

# ---- Q7: Monte Carlo target against TD(0) target on the 10-step chain ----
mc_targets = np.array([sum(seq) for seq in itertools.product((-1, 1), repeat=10)], dtype=float)
assert mc_targets.mean() == 0 and mc_targets.var() == 10     # exact: all 1024 reward sequences
td_targets = np.array([-1 + 0.5, 1 + 0.5])
assert td_targets.mean() == 0.5 and td_targets.var() == 1 and (td_targets ** 2).mean() == 1.25
sim = rng.choice([-1, 1], size=(200000, 10))
assert abs(sim.sum(axis=1).var() - 10) < 0.3 and abs((sim[:, 0] + 0.5).var() - 1) < 0.01
assert math.isclose(0 + 0.1 * (1 + 0.5 - 0), 0.15)

# ---- Q8 and Q9: eps-greedy and UCB1 on Bernoulli arms 0.6 and 0.4 ----
MU, GAP = np.array([0.6, 0.4]), 0.2


def bad_pulls(algo, horizon, checkpoints, runs=400, eps=0.1):
    """Mean number of pulls of the worse arm at each checkpoint, over independent runs (vectorised)."""
    rows = np.arange(runs)
    pulls, wins = np.zeros((runs, 2)), np.zeros((runs, 2))
    out = {}
    for t in range(1, horizon + 1):
        if algo == "ucb1" and t <= 2:
            arm = np.full(runs, t - 1)                       # pull each arm once to start
        elif algo == "ucb1":
            arm = (wins / pulls + np.sqrt(2 * math.log(t - 1) / pulls)).argmax(axis=1)
        else:
            arm = (wins / np.maximum(pulls, 1)).argmax(axis=1)
            arm = np.where(rng.random(runs) < eps, rng.integers(0, 2, runs), arm)
        pulls[rows, arm] += 1
        wins[rows, arm] += rng.random(runs) < MU[arm]
        if t in checkpoints:
            out[t] = pulls[:, 1].mean()
    return out


eg = bad_pulls("eps", 20000, (10000, 20000))
ucb = bad_pulls("ucb1", 20000, (10000, 20000))
regret_eg = {t: GAP * n for t, n in eg.items()}
regret_ucb = {t: GAP * n for t, n in ucb.items()}
assert 100 <= regret_eg[10000] < 106 and 200 <= regret_eg[20000] < 206        # at least 0.01 * T, and close to it
assert abs((regret_eg[20000] - regret_eg[10000]) - 100) < 3                   # linear: +100 per 10,000 steps
assert 55 < regret_ucb[10000] < 70 and 66 < regret_ucb[20000] < 82
assert 7 < regret_ucb[20000] - regret_ucb[10000] < 17                         # logarithmic: about +12
ucb_bound = 8 * math.log(10000) / GAP + (1 + math.pi ** 2 / 3) * GAP
assert round(ucb_bound, 1) == 369.3 and regret_ucb[10000] < ucb_bound
assert round(8 * math.log(10000) / GAP ** 2) == 1842         # the bound on pulls of the worse arm, without the constant
assert ucb[10000] < 8 * math.log(10000) / GAP ** 2 + 1 + math.pi ** 2 / 3

index_1 = 0.60 + math.sqrt(2 * math.log(100) / 90)           # Q9: the two indices after 100 pulls
index_2 = 0.40 + math.sqrt(2 * math.log(100) / 10)
assert round(index_1, 3) == 0.920 and round(index_2, 3) == 1.360 and index_2 > index_1
assert math.isclose(math.exp(-2 * 10 * (2 * math.log(100) / 10)), 100 ** -4)  # Hoeffding: exp(-2 n a^2) = t^-4
kl = 0.4 * math.log(0.4 / 0.6) + 0.6 * math.log(0.6 / 0.4)
assert round(kl, 4) == 0.0811 and round(GAP / kl, 2) == 2.47 and round(GAP / kl * math.log(10000), 1) == 22.7

# ---- Q10: Thompson sampling with Beta posteriors ----
# P(theta_2 > theta_1) = integral of 12 x^2 (1-x) * (1-x)^2; expand (1-x)^3 and integrate term by term
prob_pick_2 = sum(12 * F((-1) ** k * math.comb(3, k), k + 3) for k in range(4))
assert prob_pick_2 == F(1, 5)
theta_1, theta_2 = rng.beta(3, 2, 400000), rng.beta(1, 2, 400000)
assert abs(theta_1.mean() - 3 / 5) < 0.003 and abs(theta_2.mean() - 1 / 3) < 0.003
assert abs((theta_2 > theta_1).mean() - 0.2) < 0.005
assert abs(rng.beta(2, 2, 400000).mean() - 0.5) < 0.003

# ---- Q11: ordinary and weighted importance sampling ----
RHO, REW = {"L": F(9, 5), "R": F(1, 5)}, {"L": 1, "R": 0}


def exact_is(n):
    """Exact mean and variance of both estimators: enumerate all 2^n equally likely action sequences."""
    ordinary, weighted = [], []
    for seq in itertools.product("LR", repeat=n):
        num = sum(RHO[a] * REW[a] for a in seq)
        ordinary.append(num / n)
        weighted.append(num / sum(RHO[a] for a in seq))
    mean = lambda xs: sum(xs) / len(xs)
    var = lambda xs: mean([x * x for x in xs]) - mean(xs) ** 2
    mse = lambda xs: mean([(x - F(9, 10)) ** 2 for x in xs])
    return mean(ordinary), var(ordinary), mean(weighted), mse(weighted)


assert exact_is(1) == (F(9, 10), F(81, 100), F(1, 2), F(41, 100))
assert exact_is(2) == (F(9, 10), F(81, 200), F(7, 10), F(41, 200))
assert F(1, 4) + (F(1, 2) - F(9, 10)) ** 2 == F(41, 100)     # n = 1, weighted: variance 1/4 plus squared bias 0.16
o_mean, o_var, w_mean, w_mse = exact_is(10)
assert o_mean == F(9, 10) and o_var == F(81, 1000)
assert round(float(w_mean), 3) == 0.882 and round(float(w_mse), 4) == 0.0068
assert math.isclose(0.9 * 0.1, 0.09) and math.isclose(0.81 / 0.09, 9)
lefts = rng.random((100000, 10)) < 0.5                       # simulation with n = 10
ois = 1.8 * lefts.sum(axis=1) / 10
wis = 1.8 * lefts.sum(axis=1) / (1.8 * lefts.sum(axis=1) + 0.2 * (10 - lefts.sum(axis=1)))
assert abs(ois.mean() - 0.9) < 0.01 and abs(ois.var() - 0.081) < 0.003
assert abs(wis.mean() - 0.882) < 0.003 and abs(((wis - 0.9) ** 2).mean() - 0.0068) < 0.0008

# ---- Q12: the w -> 2w example ----
w = 1.0
for _ in range(10):                                          # only the transition state 1 -> state 2 is updated
    w += 0.1 * (0 + 0.9 * 2 * w - w) * 1
assert math.isclose(w, 1.08 ** 10) and round(w, 3) == 2.159
w = 1.0
for _ in range(10):                                          # on-policy: state 2 (then terminal) is updated too
    w += 0.1 * (0 + 0.9 * 2 * w - w) * 1
    w += 0.1 * (0 + 0 - 2 * w) * 2
assert math.isclose(w, 0.648 ** 10) and round(w, 3) == 0.013
for gamma_test, grows in ((0.4, False), (0.6, True)):        # divergence exactly when gamma > 1/2
    assert (1 + 0.1 * (2 * gamma_test - 1) > 1) == grows

# ---- Q13: potential-based shaping against a plain bonus ----
def optimal_q(reward3):
    """Q* by value iteration when the reward may depend on (s, a, s')."""
    q = np.zeros((2, 2))
    for _ in range(400):
        q = (P * (reward3 + GAMMA * q.max(axis=1)[None, None, :])).sum(axis=2)
    return q


base = np.repeat(R[:, :, None], 2, axis=2)
phi = np.array([0.0, 5.0])
shaping = GAMMA * phi[None, None, :] - phi[:, None, None]    # F(s, a, s') = gamma * phi(s') - phi(s)
assert np.allclose(shaping[:, 0, :], [[0, 4], [-5, -1]])
q_shaped = optimal_q(base + shaping)
assert np.allclose(optimal_q(base), q_star)
assert np.allclose(q_shaped, q_star - phi[:, None]) and np.allclose(q_shaped, [[11.4, 13], [14, 12.4]])
assert (q_shaped.argmax(axis=1) == q_star.argmax(axis=1)).all()

bonus = np.zeros((2, 2, 2))
bonus[0, :, 1] = 4.0                                         # scheme 1: +4 whenever A -> B, nothing else
q_bonus = optimal_q(base + bonus)
assert tuple(int(a) for a in q_bonus.argmax(axis=1)) == (1, 1)                # the best policy is now (go, cash)
assert np.allclose(q_bonus.max(axis=1), [19.375, 22.5])
assert np.allclose(q_bonus[1], [22.375, 22.5])
assert np.allclose(values[(1, 1)], [10, 15]) and np.allclose(values[(1, 0)], [13, 19])   # true values of the two

print("all checks passed")
```

</details>

</details>
