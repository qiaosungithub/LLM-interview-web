# Classic Reinforcement Learning: MDPs, TD Learning, Bandits and Off-Policy Evaluation

## Problem

The 13 questions below follow the style of DeepMind's early "quizzes": each question is short and answered out loud, with any formulas you need written on paper, and the answers are definite. Every question states all the conditions it needs, so each can be done on its own; wherever numbers appear, work out the numbers.

### MDPs and dynamic programming

**Q1.** A two-state Markov decision process (MDP) with discount factor $\gamma = 0.8$. State A: action `stay` gives reward $+1$ and stays in A; action `go` gives reward $-1$, moves to B with probability $3/4$, and stays in A with probability $1/4$. State B: action `stay` gives reward $+5$, stays in B with probability $3/4$, and returns to A with probability $1/4$; action `cash` gives reward $+7$ and always returns to A. Policy $\pi$ chooses `go` in A and `cash` in B. Write down the Bellman expectation equations satisfied by $V^\pi(A)$ and $V^\pi(B)$, solve for the two values, and explain why the solution is unique.

**Q2.** A two-state MDP with discount factor $\gamma = 0.8$. State A: `stay` gives $+1$ and stays in A; `go` gives $-1$, moves to B with probability $3/4$, and stays in A with probability $1/4$. State B: `stay` gives $+5$, stays in B with probability $3/4$, and returns to A with probability $1/4$; `cash` gives $+7$ and always returns to A. The value of policy $\pi$ (`go` in A, `cash` in B) has already been computed: $V^\pi(A) = 10$, $V^\pi(B) = 15$. (a) Compute the four action values $Q^\pi(s,a)$ and do one step of policy improvement to get a new policy $\pi'$. (b) Find $V^{\pi'}$. (c) Use the Bellman optimality equation to decide whether $\pi'$ is an optimal policy. (d) If at every step you only pick the action with the largest immediate reward (`stay` in A, `cash` in B), what is the value?

**Q3.** Suppose the MDP has bounded rewards, $\lvert r(s,a)\rvert \le R_{\max}$, and discount factor $\gamma < 1$. The Bellman optimality operator is defined as $(TV)(s) = \max_a \big[r(s,a) + \gamma \sum_{s'} p(s' \mid s,a)\,V(s')\big]$. (a) Prove that $T$ is a $\gamma$-contraction in the infinity norm: $\lVert TV - TU\rVert_\infty \le \gamma\,\lVert V - U\rVert_\infty$. (b) Use this to explain why value iteration $V_{k+1} = TV_k$ converges, and give an upper bound on $\lVert V_k - V^*\rVert_\infty$. (c) Take $\gamma = 0.8$, $R_{\max} = 7$, $V_0 = 0$: according to this bound, how many iterations guarantee an error of at most $0.01$? (d) A concrete example: a two-state MDP with $\gamma = 0.8$; state A: `stay` gives $+1$ and stays in A; `go` gives $-1$, moves to B with probability $3/4$, and stays in A with probability $1/4$; state B: `stay` gives $+5$, stays in B with probability $3/4$, and returns to A with probability $1/4$; `cash` gives $+7$ and always returns to A. Its optimal values are $V^*(A) = 13$, $V^*(B) = 19$. Starting from $V_0 = 0$, compute $V_1$, $V_2$, $V_3$ by hand and see how much the error shrinks each iteration; how many iterations are actually needed before the error is at most $0.01$?

**Q4.** Discount factor $\gamma = 0.99$. A common saying is "the effective horizon is about $1/(1-\gamma) = 100$ steps". (a) Give two interpretations that make this statement precise. (b) What fraction of the total discount weight $\sum_{t \ge 0} \gamma^t$ do the rewards of the first 100 steps account for? (c) How many steps do you need to look at to cover 99% of the weight? (d) When every per-step reward is in $[0,1]$, what is the range of the discounted return? If the trajectory is truncated at step 1000, at most how much of the return is missed?

### Temporal-difference learning

**Q5.** A tabular control problem with discount factor $\gamma = 0.8$ and step size $\alpha = 0.5$. The agent acts $\varepsilon$-greedily with $\varepsilon = 0.2$: with probability $1-\varepsilon$ it picks the action with the largest current $Q$ value, and with probability $\varepsilon$ it picks one of the two actions uniformly at random. The current $Q$ table is $Q(A,\text{stay}) = 6$, $Q(A,\text{go}) = 8$, $Q(B,\text{stay}) = 12$, $Q(B,\text{cash}) = 10$. The agent executes `go` in A, receives reward $-1$, and arrives in B; the next step happens to be exploration, and it picks `cash` in B. (a) Write down the update targets of SARSA, Q-learning and Expected SARSA, and compute the updated $Q(A,\text{go})$ for each. (b) In which single term do the three differ? (c) If $\varepsilon$ stays fixed at $0.2$, what do Q-learning and SARSA each converge to? What conditions does each need?

**Q6.** No discounting ($\gamma = 1$), only two non-terminal states A and B, and no actions to choose. Ten complete trajectories were collected, recorded as "state, reward, state, reward", with the end of the record meaning termination: 1 of `A, 0, B, 0`; 1 of `A, 0, B, 1`; 6 of `B, 1`; 2 of `B, 0`. This batch of data is replayed over and over until the value estimates stop changing (batch updating). (a) What $V(A)$, $V(B)$ does the Monte Carlo (MC) method give? (b) What does the temporal-difference method TD(0) give? (c) Why do the two $V(A)$ values differ? In what sense is each one "optimal"? When should you trust which?

**Q7.** A chain: $s_1 \to s_2 \to \dots \to s_{10} \to$ terminal, with no actions to choose. The reward at each step is independently $+1$ with probability $1/2$ and $-1$ with probability $1/2$, with no discounting ($\gamma = 1$), so every state's true value is $0$. The current estimates are $V(s_1) = 0$, $V(s_2) = 0.5$. (a) What are the Monte Carlo target and the TD(0) target used to update $V(s_1)$? For each, find its expectation, bias, variance, and mean squared error relative to the true value $0$. (b) The reward actually received at the first step is $+1$ and the step size is $\alpha = 0.1$; after one TD(0) update, what does $V(s_1)$ become? (c) Summarize the bias–variance trade-off between these two targets. The TD target is biased — does that make tabular TD(0) converge to the wrong value?

### Multi-armed bandits

**Q8.** A multi-armed bandit problem: two Bernoulli arms with success probabilities $\mu_1 = 0.6$ and $\mu_2 = 0.4$, unknown to the algorithm in advance. Use $\varepsilon$-greedy with fixed $\varepsilon = 0.1$: at each step, with probability $1-\varepsilon$ pull the arm with the highest sample mean, and with probability $\varepsilon$ pull one of the two arms uniformly at random. (a) Write down the definition of regret, and express it in terms of the number of times each arm is pulled. (b) Prove that the expected regret after $T$ steps is at least $0.01\,T$; what is the minimum at $T = 10^4$? (c) How should you change $\varepsilon$ so that regret grows only as $\log T$? What assumption does that need?

**Q9.** A multi-armed bandit problem with all rewards in $[0,1]$. At each step the UCB1 algorithm pulls the arm with the largest index $\bar x_i + \sqrt{2\ln n / n_i}$, where $n$ is the total number of pulls so far, $n_i$ is the number of times arm $i$ has been pulled, and $\bar x_i$ is its sample mean. Now $n = 100$: arm 1 has been pulled 90 times with mean $0.60$; arm 2 has been pulled 10 times with mean $0.40$. (a) Compute the indices of the two arms. Which arm is pulled next? (b) Use Hoeffding's inequality to explain why the added term is $\sqrt{2\ln n / n_i}$. (c) At what rate does UCB1's expected regret (the expected reward lost compared with always pulling the best arm) grow with the total number of steps $T$? Write down the upper bound you remember, and evaluate it when the two arms' true means are $0.6$ and $0.4$ and $T = 10^4$. (d) Can this growth rate be improved?

**Q10.** Two Bernoulli arms with unknown success probabilities, both with prior $\mathrm{Beta}(1,1)$ (i.e. uniform). So far: arm 1 has been pulled 3 times with 2 successes; arm 2 has been pulled once with 0 successes. (a) Write down the posterior distribution and posterior mean of each arm. (b) How does Thompson sampling choose the next arm? What is the probability that it chooses arm 2? What about being greedy with respect to the posterior mean? (c) Suppose arm 2 is pulled at this step and succeeds; what does its posterior become? (d) In a sentence or two, explain why Thompson sampling balances exploration and exploitation.

### Off-policy evaluation, function approximation and reward shaping

**Q11.** A one-step problem: action L gives reward $1$, action R gives reward $0$. The data are generated by a behaviour policy $\mu$ that chooses L with probability $1/2$ and R with probability $1/2$; the policy we want to evaluate is a target policy $\pi$ that chooses L with probability $0.9$ and R with probability $0.1$. We draw $n$ independent samples $(a_i, r_i)$ from $\mu$, with importance weights $\rho_i = \pi(a_i)/\mu(a_i)$. The ordinary importance sampling estimate is $\hat V_{\text{OIS}} = \frac1n \sum_i \rho_i r_i$, and the weighted importance sampling estimate is $\hat V_{\text{WIS}} = \sum_i \rho_i r_i \big/ \sum_i \rho_i$. (a) What is the true value $v_\pi$? What are the weights of the two actions? (b) For $n = 1$, what are the expectation, variance and mean squared error of each estimate? What is the variance if you draw one sample directly from $\pi$? (c) For $n = 2$, find the expectation and mean squared error of the weighted estimate, and compare with the ordinary estimate. (d) Summarize the general bias and variance properties of the two estimators, and the conditions importance sampling needs to be valid.

**Q12.** (a) Which three things appearing together make up the "deadly triad"? Does it mean "guaranteed to diverge"? (b) An example: two states share one parameter $w$ in a linear approximation; state 1's estimate is $w$ and state 2's estimate is $2w$. All rewards are $0$, so all true values are $0$. Repeatedly apply semi-gradient TD(0) updates using only the single transition "state 1 → state 2, reward $0$", with $\gamma = 0.9$, $\alpha = 0.1$, $w_0 = 1$. Write down the update and compute $w$ after 10 updates. For what values of $\gamma$ does $w$ diverge? (c) Switch to on-policy sampling: each episode first goes from state 1 to state 2, then from state 2 to the terminal state (reward $0$, terminal value $0$), and each of the two transitions is updated once, in order. What is $w$ after 10 episodes? (d) What happens if you remove any one of the three? Give a non-divergent example for each.

**Q13.** A two-state MDP with discount factor $\gamma = 0.8$. State A: `stay` gives $+1$ and stays in A; `go` gives $-1$, moves to B with probability $3/4$, and stays in A with probability $1/4$. State B: `stay` gives $+5$, stays in B with probability $3/4$, and returns to A with probability $1/4$; `cash` gives $+7$ and always returns to A. The optimal policy is known to be "`go` in A, `stay` in B", with values $(V(A), V(B)) = (13, 19)$ and optimal action values $Q^*(A,\text{stay}) = 11.4$, $Q^*(A,\text{go}) = 13$, $Q^*(B,\text{stay}) = 19$, $Q^*(B,\text{cash}) = 17.4$; another policy, "`go` in A, `cash` in B", has value $(10, 15)$. To make the agent learn to go to B faster, someone proposes two reward shaping schemes. Scheme 1: add an extra $4$ every time the agent transitions from A to B, leaving other transitions unchanged. Scheme 2: use the potential function $\Phi(A) = 0$, $\Phi(B) = 5$, and add $F(s,s') = \gamma\,\Phi(s') - \Phi(s)$ on every transition from $s$ to $s'$. (a) Write down the extra reward under scheme 2 for the four transitions A→A, A→B, B→A, B→B. (b) Prove that scheme 2 does not change which of any two policies is better, and write down the new optimal action values. (c) Under scheme 1, what does the optimal policy become? Measured with the original rewards, how much worse is it than the true optimal policy? (d) What is wrong with scheme 1?

## Reference solution

<details>
<summary>Show the reference solution</summary>

This round tests the fundamentals of an undergraduate reinforcement learning course: whether you can state definitions precisely, whether you remember the conditions of the theorems, and whether you can get a small set of numbers right on the spot. The interview reports only say that the quiz included reinforcement learning, without recording specific questions; the questions on this page were written to that level of difficulty. Reinforcement learning for LLM post-training (policy gradients, PPO, GRPO, DPO) is on a separate page, [RL for LLMs](#/deepmind/quiz/rl-for-llms), and is not repeated here.

Three things to prepare. First, drill the Bellman equations, the three TD targets, the UCB index and the importance weight until you can write them from memory. Second, pair every result with a small example of two or three states and work it through by hand; the most common thing an interviewer does is give you a few numbers and have you compute on the spot. Third, state theorems with their conditions: tabular or with function approximation, whether the data come from the current policy, what the step sizes must satisfy, whether rewards are bounded. Most follow-up questions land on the conditions.

### MDPs and dynamic programming

**Q1.** For policy $\pi$, the value equals "the reward at this step, plus the discounted expected value of the next state":

$$V^\pi(A) = -1 + 0.8\left[\tfrac14 V^\pi(A) + \tfrac34 V^\pi(B)\right], \qquad V^\pi(B) = 7 + 0.8\,V^\pi(A).$$

The first equation rearranges to $0.8\,V^\pi(A) = -1 + 0.6\,V^\pi(B)$. Substituting the second: $0.8\,V^\pi(A) = -1 + 4.2 + 0.48\,V^\pi(A)$, i.e. $0.32\,V^\pi(A) = 3.2$. So

$$V^\pi(A) = 10, \qquad V^\pi(B) = 7 + 0.8 \times 10 = 15.$$

Why the solution is unique: write the equations in matrix form $V = r_\pi + \gamma P_\pi V$, i.e. $(I - \gamma P_\pi)V = r_\pi$. Each row of $P_\pi$ is a probability distribution, so all its eigenvalues have modulus at most $1$; hence the eigenvalues of $\gamma P_\pi$ have modulus at most $\gamma < 1$, and $I - \gamma P_\pi$ is invertible. Another way to say it: the operator $T^\pi V = r_\pi + \gamma P_\pi V$ is a $\gamma$-contraction in the infinity norm, so its fixed point is unique.

The conditions used are $\gamma < 1$ and bounded rewards. If $\gamma = 1$, the task has to be episodic and this policy has to reach a terminal state with certainty; otherwise the value may be undefined.

**Q2.** (a) The action value is defined as $Q^\pi(s,a) = r(s,a) + \gamma \sum_{s'} p(s' \mid s,a)\,V^\pi(s')$. Plugging in $V^\pi = (10, 15)$:

- $Q^\pi(A,\text{stay}) = 1 + 0.8 \times 10 = 9$;
- $Q^\pi(A,\text{go}) = -1 + 0.8\,(0.25 \times 10 + 0.75 \times 15) = -1 + 0.8 \times 13.75 = 10$;
- $Q^\pi(B,\text{stay}) = 5 + 0.8\,(0.75 \times 15 + 0.25 \times 10) = 5 + 0.8 \times 13.75 = 16$;
- $Q^\pi(B,\text{cash}) = 7 + 0.8 \times 10 = 15$.

In each state take the action with the largest $Q^\pi$: A still chooses `go` ($10 > 9$), and B switches to `stay` ($16 > 15$). So $\pi'$ is "`go` in A, `stay` in B".

(b) Set up the equations for $\pi'$: $V(A) = -1 + 0.8\,[0.25\,V(A) + 0.75\,V(B)]$, $V(B) = 5 + 0.8\,[0.75\,V(B) + 0.25\,V(A)]$. The second simplifies to $0.4\,V(B) = 5 + 0.2\,V(A)$, i.e. $V(B) = 12.5 + 0.5\,V(A)$; the first is $0.8\,V(A) = -1 + 0.6\,V(B)$. Substituting gives $0.8\,V(A) = 6.5 + 0.3\,V(A)$, so $V^{\pi'}(A) = 13$, $V^{\pi'}(B) = 19$. Both states are higher than the original $(10, 15)$.

(c) Recompute the backed-up values of the four actions with $V^{\pi'} = (13, 19)$: in A, `stay` is $1 + 0.8 \times 13 = 11.4$ and `go` is $-1 + 0.8\,(0.25 \times 13 + 0.75 \times 19) = 13$; in B, `stay` is $5 + 0.8 \times 17.5 = 19$ and `cash` is $7 + 0.8 \times 13 = 17.4$. The maximum in each state is exactly $13$ and $19$, i.e. $V^{\pi'}$ itself, so $V^{\pi'}$ satisfies the Bellman optimality equation $V(s) = \max_a [r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) V(s')]$. This equation has a single solution when $\gamma < 1$, so $V^{\pi'} = V^*$ and $\pi'$ is optimal. Policy iteration finishes in one step here.

(d) Choosing `stay` in A means staying in A forever: $V(A) = 1/(1 - 0.8) = 5$, $V(B) = 7 + 0.8 \times 5 = 11$. It takes the largest available reward at every step, yet in the long run gets well under half of the optimal values $(13, 19)$: `go` loses $1$ right now, but what it buys is entry into state B, worth $+5$ per step.

What is used here is the policy improvement theorem. Conditions: finitely many states and actions, $\gamma < 1$, and $Q^\pi$ exact. Conclusion: the $\pi'$ obtained by acting greedily with respect to $Q^\pi$ is no worse than $\pi$ in every state; if no state strictly improves, $\pi$ already satisfies the optimality equation. When $Q^\pi$ is only approximate, this guarantee no longer holds.

**Q3.** (a) First prove a small fact: for any two sets of numbers $f(a)$, $g(a)$, $\lvert \max_a f(a) - \max_a g(a) \rvert \le \max_a \lvert f(a) - g(a) \rvert$. Let $a^\star$ maximize $f$; then $\max f - \max g \le f(a^\star) - g(a^\star) \le \max_a \lvert f(a) - g(a)\rvert$; swapping $f$ and $g$ gives the other direction.

For any $V$, $U$ and state $s$, let $f(a) = r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) V(s')$, and let $g(a)$ be the same expression with $V$ replaced by $U$. Then

$$\lvert (TV)(s) - (TU)(s) \rvert \le \max_a\ \gamma \Big\lvert \sum_{s'} p(s' \mid s,a)\,\big(V(s') - U(s')\big) \Big\rvert \le \gamma\,\lVert V - U \rVert_\infty.$$

The last step uses that $p(\cdot \mid s,a)$ is a probability distribution: the absolute value of a weighted average is at most the largest absolute value. Taking the maximum over $s$ gives $\lVert TV - TU\rVert_\infty \le \gamma \lVert V - U \rVert_\infty$.

(b) Bounded functions with the infinity norm form a complete space, and when $\gamma < 1$, $T$ is a contraction mapping. By the Banach fixed-point theorem, $T$ has a unique fixed point, which is $V^*$; and starting from any bounded $V_0$,

$$\lVert V_k - V^* \rVert_\infty \le \gamma^k\,\lVert V_0 - V^* \rVert_\infty.$$

The conditions are $\gamma < 1$ and bounded rewards (so that $\lVert V^* \rVert_\infty \le R_{\max}/(1-\gamma)$). Note two things: the contraction is with respect to the infinity norm and does not necessarily hold for the 2-norm; and if you follow $T$ with a function approximation step (a projection), the combination is not necessarily a contraction either, which is exactly where the divergence in Q12 comes from.

(c) With $V_0 = 0$, $\lVert V_0 - V^*\rVert_\infty \le 7/(1 - 0.8) = 35$. We need $35 \times 0.8^k \le 0.01$, i.e. $k \ge \ln 3500 / \ln 1.25 = 8.161/0.2231 = 36.57$, so **37 iterations**. In general $k \ge \ln\!\big(R_{\max}/((1-\gamma)\epsilon)\big) / \ln(1/\gamma)$, roughly $\frac{1}{1-\gamma} \ln \frac{R_{\max}}{(1-\gamma)\epsilon}$: the number of iterations is proportional to $1/(1-\gamma)$. For the same accuracy, $\gamma = 0.99$ needs 1111 iterations.

(d) Iteration by iteration (in each state, take the max over the two actions):

- $V_1 = (\max(1, -1),\ \max(5, 7)) = (1, 7)$;
- $V_2$: in A, `stay` is $1 + 0.8 \times 1 = 1.8$ and `go` is $-1 + 0.8\,(0.25 \times 1 + 0.75 \times 7) = 3.4$; in B, `stay` is $5 + 0.8\,(0.75 \times 7 + 0.25 \times 1) = 9.4$ and `cash` is $7 + 0.8 \times 1 = 7.8$. So $V_2 = (3.4, 9.4)$;
- $V_3$: in A, `go` is $-1 + 0.8\,(0.25 \times 3.4 + 0.75 \times 9.4) = 5.32$; in B, `stay` is $5 + 0.8 \times 7.9 = 11.32$; both exceed the other action ($3.72$ and $9.72$). So $V_3 = (5.32, 11.32)$.

The distances to $V^* = (13, 19)$ are $19$ ($V_0$), $12$, $9.6$, $7.68$. After the first iteration, each iteration multiplies the error by exactly $0.8$. The reason is that $V_1 = V^* - 12 \cdot \mathbf 1$: both components are off by the same amount; and for any constant $c$, $T(V - c\,\mathbf 1) = TV - \gamma c\,\mathbf 1$ (every backed-up value drops by $\gamma c$). So $V_k = V^* - 12 \times 0.8^{k-1}\,\mathbf 1$. We need $12 \times 0.8^{k-1} \le 0.01$, i.e. $k - 1 \ge \ln 1200/\ln 1.25 = 31.77$, so in practice **33 iterations** are needed.

The bound's 37 iterations is conservative for two reasons: it estimates the initial distance as $35$ (it is actually $19$; with $19$ the bound gives 34 iterations); and the first iteration actually shrinks the error by a factor of $12/19 \approx 0.63$, faster than $0.8$. In practice you don't know $V^*$, and the usual stopping rule uses the change between consecutive iterations: $\lVert V_{k+1} - V^* \rVert_\infty \le \frac{\gamma}{1-\gamma} \lVert V_{k+1} - V_k \rVert_\infty$.

**Q4.** (a) Two interpretations.

- Total weight: $\sum_{t \ge 0} \gamma^t = 1/(1-\gamma) = 100$. When every per-step reward is $1$, the discounted return is exactly $100$, as if you counted 100 steps without discounting.
- Random termination: view discounting as "after each step, stop with probability $1-\gamma$". The probability of not having stopped by step $t$ (counting from $0$) is $\gamma^t$, so the expected discounted return equals the expected undiscounted return of this randomly terminated process; the total number of steps is geometrically distributed with mean $1/(1-\gamma) = 100$ steps.

(b) The share of weight in the first 100 steps is $\sum_{t=0}^{99} \gamma^t \big/ \sum_{t \ge 0} \gamma^t = 1 - \gamma^{100} = 1 - 0.366 = 0.634$. This is close to $1 - 1/e$, because $0.99^{100} \approx e^{-1}$. In other words, about 37% of the weight lies beyond "100 steps"; the effective horizon is not a cutoff.

(c) We need $1 - \gamma^n \ge 0.99$, i.e. $\gamma^n \le 0.01$, $n \ge \ln 100 / \ln(1/0.99) = 4.605/0.01005 = 458.2$, so **459 steps**, about 4.6 times the effective horizon.

(d) The return lies in $[0,\ 1/(1-\gamma)] = [0, 100]$. Truncating at step $H$ loses at most $\sum_{t \ge H} \gamma^t = \gamma^H/(1-\gamma)$; for $H = 1000$ that is $100 \times 0.99^{1000} \approx 0.0043$.

### Temporal-difference learning

**Q5.** (a) All three methods update $Q(A,\text{go}) \leftarrow Q(A,\text{go}) + \alpha\,[\text{target} - Q(A,\text{go})]$, with targets:

- SARSA: uses the action actually chosen at the next step. Target $= r + \gamma\,Q(B,\text{cash}) = -1 + 0.8 \times 10 = 7$; after the update, $8 + 0.5 \times (7 - 8) = 7.5$.
- Q-learning: uses the maximum at the next state, regardless of what was actually chosen. Target $= r + \gamma \max_a Q(B,a) = -1 + 0.8 \times 12 = 8.6$; after the update, $8 + 0.5 \times 0.6 = 8.3$.
- Expected SARSA: takes the expectation over the next action under the policy. In B the greedy action is `stay` ($12 > 10$), chosen with probability $1 - \varepsilon + \varepsilon/2 = 0.9$, and `cash` with $0.1$; the expected value is $0.9 \times 12 + 0.1 \times 10 = 11.8$. Target $= -1 + 0.8 \times 11.8 = 8.44$; after the update, $8 + 0.5 \times 0.44 = 8.22$.

(b) They differ only in the bootstrapping term, i.e. what stands in for "the value after the next state": SARSA uses the sampled $Q(s',a')$, Q-learning uses $\max_{a'} Q(s',a')$, and Expected SARSA uses $\sum_{a'} \pi(a' \mid s')\,Q(s',a')$. The Expected SARSA target has the same expectation as the SARSA target; it just removes the variance that comes from "the next action is sampled at random".

(c) Q-learning converges to the optimal action values $Q^*$, regardless of how the behaviour policy explores. Conditions: tabular; every state–action pair visited infinitely often; the step sizes for each state–action pair satisfy $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$; bounded rewards; $\gamma < 1$ (Watkins and Dayan, 1992). A fixed step size $\alpha = 0.5$ fails the third condition and only jitters around $Q^*$.

SARSA learns the value of the policy it is executing. With $\varepsilon$ fixed, under the same step-size and visitation conditions, it does not converge to $Q^*$, but to the fixed point of the "$\varepsilon$-greedy operator":

$$Q(s,a) = r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) \Big[(1-\varepsilon) \max_{a'} Q(s',a') + \frac{\varepsilon}{\lvert\mathcal A\rvert} \sum_{a'} Q(s',a')\Big].$$

This operator is also a $\gamma$-contraction with a unique fixed point; it is the value of the best policy among all policies that "keep $\varepsilon$ exploration". For SARSA to converge to $Q^*$, $\varepsilon$ must gradually decrease to $0$ while every action is still tried infinitely often (these conditions are called GLIE, greedy in the limit with infinite exploration; Singh et al., 2000).

An example you can compute: take the two-state MDP ($\gamma = 0.8$; A: `stay` gives $+1$ and stays in A, `go` gives $-1$ and moves to B with $3/4$; B: `stay` gives $+5$ and stays in B with $3/4$, `cash` gives $+7$ and returns to A). Its $Q^*$ is $(11.4,\ 13;\ 19,\ 17.4)$, and in both states the best and second-best actions differ by $1.6$. Suppose the fixed point is $Q^* - c$; then each state's value under $\varepsilon$-greedy is $V^*(s) - c - \tfrac{\varepsilon}{2} \times 1.6 = V^*(s) - c - 0.16$, and substituting back gives $c = 0.8\,(c + 0.16)$, $c = 0.64$. So every $Q$ value SARSA learns with $\varepsilon = 0.2$ is $0.64$ below $Q^*$, i.e. $(10.76,\ 12.36;\ 18.36,\ 16.76)$. In this example the greedy actions don't change; in tasks where exploration is very costly, like "cliff walking", the two greedy policies differ too.

**Q6.** (a) Monte Carlo uses the average of the returns actually obtained after each state. B appears once in each of the 10 trajectories, and the returns after it are $0, 1$, six $1$s and two $0$s, for a total of seven $1$s: $V(B) = 7/10 = 0.7$. A appears twice, with returns after it of $0 + 0 = 0$ and $0 + 1 = 1$: $V(A) = 0.5$.

(b) The TD(0) fixed point requires the average TD error at each state to be zero. B terminates immediately, so its target is just the reward, $V(B) = 0.7$, the same as Monte Carlo. Both of A's transitions are "reward $0$, go to B", so $V(A) = 0 + V(B) = 0.7$.

(c) The two methods answer two different questions.

- Batch Monte Carlo minimizes the squared error of the returns on the training data: the only returns seen after A are one $0$ and one $1$, whose average is $0.5$.
- Batch TD(0) gives the exact values under the "maximum-likelihood Markov model", called the certainty-equivalence estimate: from the data, estimate that A always goes to B with reward $0$ and that B's expected reward is $0.7$, then solve the equations in this model.

If the process really is Markov, TD's answer is more trustworthy: it uses all 10 visits to B to estimate what happens after A, while Monte Carlo uses only the 2 trajectories that start from A. If the state is not Markov (for example, B reached from A is actually different from B at the start of an episode), TD's model assumption is wrong, and Monte Carlo doesn't rely on that assumption. This conclusion assumes: tabular, the same batch of data replayed repeatedly, and a small enough step size.

**Q7.** (a) Denote the ten rewards by $r_1, \dots, r_{10}$, each with mean $0$ and variance $1$.

- The Monte Carlo target is the full return $G = r_1 + \dots + r_{10}$. Expectation $0$, bias $0$; the variance is the sum of ten independent terms, $10$; mean squared error $10$.
- The TD(0) target is $r_1 + V(s_2) = r_1 + 0.5$. Expectation $0.5$, bias $+0.5$ (all of it from the error in $V(s_2)$); the variance is just the variance of one reward, $1$; mean squared error $1 + 0.5^2 = 1.25$.

(b) The TD error is $\delta = r_1 + V(s_2) - V(s_1) = 1 + 0.5 - 0 = 1.5$, and $V(s_1) \leftarrow 0 + 0.1 \times 1.5 = 0.15$.

(c) The Monte Carlo target is unbiased, but its variance grows with the number of remaining steps, because it adds in all the randomness that follows. The TD target contains only one step of randomness, so its variance is small; the price is that it is biased whenever the next state's estimate is inaccurate. In this example the TD target's mean squared error is $1.25$ versus $10$ for Monte Carlo, which is why TD usually learns faster. $n$-step returns and TD($\lambda$) interpolate between the two.

The bias does not make tabular TD(0) converge to the wrong place. The bias here is a bias of "the current target", coming from $V(s_2)$ not yet being accurate; as $V(s_2)$ is updated toward its true value, the bias disappears with it. For a fixed policy, tabular, every state visited infinitely often, and step sizes satisfying $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$, TD(0) converges with probability 1 to the true values (Jaakkola, Jordan and Singh, 1994; Tsitsiklis, 1994). With function approximation it's different: the fixed point of linear TD generally isn't the best approximation in the least-squares sense.

### Multi-armed bandits

**Q8.** (a) Regret is "the expected reward you would get by always pulling the best arm" minus "the expected reward the algorithm actually gets":

$$R_T = T\mu^* - \mathbb E\Big[\sum_{t=1}^T r_t\Big] = \sum_i \Delta_i\ \mathbb E[N_i(T)],$$

where $\mu^* = \max_i \mu_i$, $\Delta_i = \mu^* - \mu_i$ is arm $i$'s gap, and $N_i(T)$ is the number of times it is pulled in the first $T$ steps. Here only arm 2 has a gap, $\Delta_2 = 0.2$.

(b) At every step there is probability $\varepsilon$ of exploring uniformly, and when exploring, arm 2 is pulled with probability $1/2$. So no matter how accurate the estimates are, the probability of pulling arm 2 at each step is at least $\varepsilon/2 = 0.05$, $\mathbb E[N_2(T)] \ge 0.05\,T$, and

$$R_T \ge 0.2 \times 0.05 \times T = 0.01\,T.$$

At $T = 10^4$ it is at least $100$. Regret grows linearly in $T$. This lower bound is also close to the actual value: once the estimates settle, the greedy steps all pull arm 1, and the expected regret per step is exactly $0.01$. Simulation (pulling arm 1 when the two estimates are equal) gives about $102$ at $T = 10^4$ and about $202$ at $T = 2 \times 10^4$; choosing randomly on ties adds about 5 more.

(c) Let $\varepsilon$ decay over time, $\varepsilon_t \propto 1/t$. Then the total number of exploration steps is $\sum_t \varepsilon_t \approx c \ln T$, and regret can be made $O(\log T)$. The form given by Auer, Cesa-Bianchi and Fischer (2002) is $\varepsilon_t = \min(1,\ cK/(d^2 t))$, where $K$ is the number of arms; the assumption is $0 < d \le \min_i \Delta_i$ and a large enough constant $c$, in which case the probability of pulling a suboptimal arm at step $t$ is $O(1/t)$. The cost is that you need to know a lower bound on the gap in advance: if $c/d^2$ is too small, there isn't enough exploration, the algorithm may stay on the wrong arm for a long time, regret is no longer logarithmic, and over a practical number of steps it can be close to linear. If you don't want to depend on the gap, use UCB or Thompson sampling.

**Q9.** (a) $\ln 100 = 4.605$.

- Arm 1: $0.60 + \sqrt{2 \times 4.605/90} = 0.60 + 0.320 = 0.920$;
- Arm 2: $0.40 + \sqrt{2 \times 4.605/10} = 0.40 + 0.960 = 1.360$.

The next pull is **arm 2**. Its sample mean is low, but it has only been pulled 10 times, so its uncertainty is much larger.

(b) With rewards in $[0,1]$ and independent pulls, Hoeffding's inequality gives bounds in both directions: $P(\bar x_i \le \mu_i - a) \le \exp(-2 n_i a^2)$ and $P(\bar x_i \ge \mu_i + a) \le \exp(-2 n_i a^2)$. Taking $a = \sqrt{2 \ln n / n_i}$, both right-hand sides are $\exp(-4 \ln n) = n^{-4}$. The first says the probability that the index $\bar x_i + a$ is below the true value $\mu_i$ is at most $n^{-4}$, so the index is an upper confidence bound on $\mu_i$; applied to the optimal arm, it means the optimal arm is almost never underestimated. The second says a suboptimal arm's sample mean is almost never overestimated by more than $a$. $n^{-4}$ decays fast enough that the sum over all times and all possible $n_i$ is still finite, so the total loss from "the confidence interval failing" is a constant.

When both bounds hold, a suboptimal arm $i$'s index is at most $\mu_i + 2a$, and the optimal arm's index is at least $\mu^*$. So as soon as $2\sqrt{2\ln n/n_i} < \Delta_i$, i.e. $n_i > 8 \ln n/\Delta_i^2$, the suboptimal arm's index is guaranteed to be below the optimal arm's. So each suboptimal arm is pulled only about $8 \ln n / \Delta_i^2$ times. This is "optimism in the face of uncertainty": arms that haven't been tried enough get a high score, and arms that have been tried enough and still fall short are naturally dropped.

(c) It grows as $\ln T$. Theorem 1 of Auer, Cesa-Bianchi and Fischer (2002): if rewards are in $[0,1]$ and each arm's rewards are i.i.d., then for any number of steps $n$,

$$R_n \le 8 \sum_{i:\,\Delta_i > 0} \frac{\ln n}{\Delta_i} + \Big(1 + \frac{\pi^2}{3}\Big) \sum_j \Delta_j.$$

Plugging in $\Delta = 0.2$, $n = 10^4$: $8 \times 9.210/0.2 + 4.290 \times 0.2 = 368.4 + 0.9 = 369.3$. That corresponds to arm 2 being pulled at most about $1842 + 4.3$ times. The constants in this bound are loose: in simulation, UCB1's regret at $10^4$ steps is only a little over 60. But the order is right: doubling $T$ from $10^4$ to $2 \times 10^4$ adds only about ten points to UCB1's regret, while $\varepsilon$-greedy with fixed $\varepsilon = 0.1$ adds 100.

(d) The order can't be improved, but the constant can. The Lai and Robbins (1985) lower bound: for Bernoulli arms, any "uniformly good" algorithm (regret $o(T^a)$ on every problem instance, for all $a > 0$) satisfies

$$\liminf_{T \to \infty} \frac{\mathbb E[N_i(T)]}{\ln T} \ge \frac{1}{\mathrm{KL}(\mu_i \,\Vert\, \mu^*)}.$$

Here $\mathrm{KL}(0.4 \,\Vert\, 0.6) = 0.4 \ln\frac{0.4}{0.6} + 0.6 \ln\frac{0.6}{0.4} = 0.2 \ln 1.5 = 0.0811$, so regret is asymptotically at least $\frac{0.2}{0.0811} \ln T \approx 2.47 \ln T$. For reference, at $T = 10^4$, $2.47 \ln T \approx 22.7$; this is an asymptotic result and can't be treated as a strict lower bound at a finite number of steps. The coefficient in UCB1's bound is $8/\Delta = 40$, far from $2.47$; KL-UCB and Thompson sampling achieve this optimal constant.

**Q10.** (a) The Beta distribution is the conjugate prior for a Bernoulli likelihood: for $\mathrm{Beta}(a,b)$, each observed success adds 1 to $a$, each observed failure adds 1 to $b$, and the mean is $a/(a+b)$.

- Arm 1: 2 successes, 1 failure, posterior $\mathrm{Beta}(3,2)$, mean $3/5$;
- Arm 2: 0 successes, 1 failure, posterior $\mathrm{Beta}(1,2)$, mean $1/3$.

(b) Thompson sampling draws one sample $\theta_1$, $\theta_2$ from each arm's posterior and pulls the arm with the larger sample. So the probability it chooses arm 2 is the posterior probability that $\theta_2 > \theta_1$. The density of $\mathrm{Beta}(3,2)$ is $12x^2(1-x)$; the density of $\mathrm{Beta}(1,2)$ is $2(1-y)$, so $P(\theta_2 > x) = (1-x)^2$. Hence

$$P(\theta_2 > \theta_1) = \int_0^1 12x^2(1-x)\,(1-x)^2\,dx = 12 \int_0^1 x^2 (1-x)^3\,dx = 12 \cdot \frac{2!\,3!}{6!} = \frac{1}{5}.$$

Thompson sampling pulls arm 2 with probability $1/5$. Being greedy with respect to the posterior mean always pulls arm 1 ($3/5 > 1/3$), so the probability of choosing arm 2 is $0$.

(c) Arm 2 gains one success, and its posterior becomes $\mathrm{Beta}(2,2)$, with mean $1/2$.

(d) The probability that Thompson sampling picks each arm equals the posterior probability that "this arm is the optimal arm" (probability matching). An arm with little data has a wide posterior and a good chance of drawing a large value, which gives exploration; as data accumulates, the posterior narrows, clearly worse arms are almost never chosen again, and exploration winds down automatically. It needs no exploration parameter to tune. For Bernoulli arms, its regret asymptotically achieves the Lai–Robbins lower bound (Kaufmann, Korda and Munos, 2012).

### Off-policy evaluation, function approximation and reward shaping

**Q11.** (a) $v_\pi = 0.9 \times 1 + 0.1 \times 0 = 0.9$. The weights are $\rho(L) = 0.9/0.5 = 1.8$ and $\rho(R) = 0.1/0.5 = 0.2$.

(b) $n = 1$.

- The ordinary estimate is $\rho r$: $1.8$ when L is drawn and $0$ when R is drawn, each half the time. Expectation $0.9$, unbiased; variance $\tfrac12 \times 1.8^2 - 0.9^2 = 0.81$; mean squared error $0.81$.
- The weighted estimate is $\rho r/\rho = r$; the weight cancels: $1$ with probability $1/2$ and $0$ with probability $1/2$. Expectation $0.5$, bias $-0.4$; variance $0.25$; mean squared error $0.25 + 0.16 = 0.41$. With a single sample, the weighted estimate's expectation is actually the behaviour policy's value $v_\mu$.
- Drawing one sample directly from $\pi$: the reward is a Bernoulli variable with success probability $0.9$, variance $0.9 \times 0.1 = 0.09$. Ordinary importance sampling has 9 times that variance.

(c) $n = 2$. Four equally likely cases:

| The two actions | Ordinary estimate | Weighted estimate |
| --- | --- | --- |
| L, L | $(1.8 + 1.8)/2 = 1.8$ | $3.6/3.6 = 1$ |
| L, R or R, L | $1.8/2 = 0.9$ | $1.8/2.0 = 0.9$ |
| R, R | $0$ | $0/0.4 = 0$ |

Weighted estimate: expectation $(1 + 0.9 + 0.9 + 0)/4 = 0.7$, bias $-0.2$; mean squared error $(0.1^2 + 0 + 0 + 0.9^2)/4 = 0.205$. Ordinary estimate: expectation $0.9$, variance (which is also the mean squared error) $(0.9^2 + 0 + 0 + 0.9^2)/4 = 0.405$. With two samples, the weighted estimate's mean squared error is already only half that of the ordinary estimate. The gap widens further on: at $n = 10$ the weighted estimate has expectation $0.882$ and mean squared error $0.0068$, while the ordinary estimate has variance $0.081$.

(d) General properties.

- Ordinary importance sampling is unbiased. Its variance is driven by the weights; when the weights are unbounded, the variance can be very large or even infinite.
- Weighted importance sampling is biased; the bias goes to $0$ as $n$ grows, so it is consistent. It is a weighted average of observed returns and never leaves the range of the returns, so with bounded returns its variance is bounded, and in practice it is usually more stable.
- Both need coverage: wherever $\pi(a \mid s) > 0$ we need $\mu(a \mid s) > 0$; otherwise things the target policy would do never appear in the data and can't be estimated. You also need to know (or be able to estimate) the behaviour policy's probabilities.

In multi-step problems the weight is the product of per-step ratios, and the variance grows exponentially with the number of steps; this is derived in the last question on the [RL for LLMs](#/deepmind/quiz/rl-for-llms) page.

**Q12.** (a) The three things are: function approximation (different states share parameters), bootstrapping (using your own estimates as targets, as in TD and Q-learning), and off-policy training (the state distribution used for updates is not the one the target policy itself generates). When all three appear together, learning **can** be unstable or even diverge — not that it necessarily diverges; DQN with experience replay and a target network has all three, and in practice it often trains fine.

(b) The gradient of state 1's estimate $w$ with respect to the parameter is $1$. Semi-gradient TD(0) differentiates only the current state's estimate and treats the target as a constant:

$$w \leftarrow w + \alpha\,\big[r + \gamma \cdot 2w - w\big] \cdot 1 = w\,\big[1 + \alpha(2\gamma - 1)\big].$$

Plugging in $\gamma = 0.9$, $\alpha = 0.1$: each update multiplies by $1.08$. After 10 updates $w = 1.08^{10} = 2.159$, and it keeps growing, whereas the correct answer is $w = 0$. Whenever $2\gamma - 1 > 0$, i.e. $\gamma > 1/2$, it diverges for any $\alpha > 0$ ($w$ stays put at $\gamma = 1/2$ and converges to $0$ for $\gamma < 1/2$). Intuitively: when state 1's estimate is pulled up toward $2\gamma w$, the shared parameter makes state 2's estimate rise by twice as much, so the target can never be caught.

(c) The gradient of state 2's estimate $2w$ with respect to the parameter is $2$; its transition leads to the terminal state, so its target is $0$:

$$w \leftarrow w + \alpha\,[0 + 0 - 2w] \cdot 2 = w\,(1 - 4\alpha) = 0.6\,w.$$

The two updates in one episode together multiply by $1.08 \times 0.6 = 0.648$. After 10 episodes $w = 0.648^{10} = 0.013$, converging to the correct $0$. With on-policy sampling, state 2's own update pulls its inflated estimate back down; in (b) it diverges because state 2's estimate is only ever used as a target and is never corrected by its own transition.

(d) Remove any one, and the other two can coexist:

- No function approximation: tabular Q-learning is both off-policy and bootstrapping, and under the conditions given in Q5 it converges to $Q^*$.
- Not off-policy: linear function approximation with on-policy TD(0) converges when states are sampled from the policy's stationary distribution, the features are linearly independent, and the step sizes satisfy $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$; the error at the limit point is at most a constant multiple of the best linear approximation error, and that multiple grows as $\gamma \to 1$ (Tsitsiklis and Van Roy, 1997).
- No bootstrapping: using Monte Carlo returns as regression targets (with importance weights when off-policy) is ordinary supervised learning, and stochastic gradient descent converges under the usual conditions, at the cost of high variance.

In practice you usually keep all three but weaken one of them: target networks weaken bootstrapping, multi-step returns reduce reliance on bootstrapping, and keeping the replay data distribution close to the current policy reduces how off-policy the training is.

**Q13.** (a) Plugging into $F(s,s') = 0.8\,\Phi(s') - \Phi(s)$:

- A→A: $0.8 \times 0 - 0 = 0$;
- A→B: $0.8 \times 5 - 0 = +4$;
- B→A: $0.8 \times 0 - 5 = -5$;
- B→B: $0.8 \times 5 - 5 = -1$.

The reward for entering B is the same as in scheme 1, but leaving B "pays the potential back".

(b) Along any trajectory starting from $s_0$, the discounted sum of the extra rewards telescopes:

$$\sum_{t \ge 0} \gamma^t \big[\gamma \Phi(s_{t+1}) - \Phi(s_t)\big] = -\Phi(s_0) + \lim_{T \to \infty} \gamma^T \Phi(s_T) = -\Phi(s_0).$$

So for any policy $\pi$ and any $(s,a)$, the new action value is $Q'^\pi(s,a) = Q^\pi(s,a) - \Phi(s)$. The amount subtracted depends only on the current state, not on the action or the policy, so the ranking of actions within a state doesn't change, which of any two policies is better doesn't change either, and the optimal policy stays the same. The new optimal action values:

- A: `stay` $11.4 - 0 = 11.4$, `go` $13 - 0 = 13$;
- B: `stay` $19 - 5 = 14$, `cash` $17.4 - 5 = 12.4$.

This is the result of Ng, Harada and Russell (1999). Conditions: $\Phi$ depends only on the state and is bounded; the $\gamma$ in $F$ is the MDP's discount factor; undiscounted episodic tasks also require the terminal state's potential to be $0$. They also proved the converse: if the extra reward is not of this form, there is always some MDP in which it changes the optimal policy.

(c) Scheme 1 only changes the expected reward of choosing `go` in A: $-1 + \tfrac34 \times 4 = 2$. Under the new rewards, compare B's two choices when A chooses `go` (write $V_A$, $V_B$). In both cases $V_A = 2 + 0.8\,(0.25 V_A + 0.75 V_B)$, i.e. $0.8\,V_A = 2 + 0.6\,V_B$.

- B chooses `stay`: $V_B = 5 + 0.8\,(0.75 V_B + 0.25 V_A)$, i.e. $V_B = 12.5 + 0.5\,V_A$. Solving gives $V_A = 19$, $V_B = 22$.
- B chooses `cash`: $V_B = 7 + 0.8\,V_A$. Solving gives $0.32\,V_A = 6.2$, $V_A = 19.375$, $V_B = 22.5$.

The `cash` option is higher in both states. Check that it satisfies the optimality equation: with $(19.375, 22.5)$, B's `stay` is $5 + 0.8\,(0.75 \times 22.5 + 0.25 \times 19.375) = 22.375 < 22.5$, and A's `stay` is $1 + 0.8 \times 19.375 = 16.5 < 19.375$. So under scheme 1 the optimal policy becomes "`go` in A, `cash` in B".

Under the original rewards, this policy's value is $(10, 15)$, while the true optimum is $(13, 19)$: $3$ less in A and $4$ less in B.

(d) Scheme 1 pays a reward every time you enter B but deducts nothing when you leave, so the loop "enter B, immediately return to A, enter B again" can collect the reward over and over. What the agent learns is to farm this loop, not to stay in B. In the potential-based scheme, going around a loop and back to the original state makes the extra rewards cancel under discounting, so there is nothing to farm. If you add shaping rewards, write them in the form $\gamma\Phi(s') - \Phi(s)$; the common advice is to make $\Phi$ close to a rough estimate of $V^*$.

<details>
<summary>Checks (runnable)</summary>

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
