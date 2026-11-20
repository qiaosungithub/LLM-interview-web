# Research Brainstorm: Is the Chain-of-Thought the Real Reason, and Why a High Benchmark Score Does Not Feel Better

## Problem

Research brainstorm: the interviewer poses an open question, you think out loud, and they follow up on whatever you say. You don't need to know model internals; having used and observed models as a black box is enough. The two questions below and all the follow-ups were written by us in this format; they are not the original questions. Each question states its full background, so you can practice it on its own.

### Question 1: Is the reasoning a model writes down its real reason?

**Q1.** The interviewer says: "Today's models write out a reasoning process before they answer, the chain-of-thought (CoT). Is that text really the reason they arrive at their answer? Suppose you can only call the model through an API and can't see the weights or activations. How would you study this question?" Talk through your approach in two or three minutes.

**Q2.** Same background. The interviewer follows up: "Pick the experiment you'd most want to run first and tell me concretely how you'd do it: what are the inputs, what do you compare, what do you expect to see, and what result would change your mind? What's the control?"

**Q3.** Background: you ran a truncation experiment: let the model write its full chain-of-thought, keep only the first half, and have it give the final answer directly from there. On 1000 multi-step reasoning problems, 85% of the answers are the same as with the full chain-of-thought. A colleague says: "The answers mostly didn't change, so the chain-of-thought is mostly decoration; the model had already made up its mind." Do you agree? What controls or analyses would you add?

**Q4.** The interviewer asks: "Why do safety people care about this question? If your experiments found that the chain-of-thought is basically faithful on the tasks you tested, where would that conclusion generalize, and where wouldn't it?"

### Question 2: A high benchmark score, but it isn't better in practice

**Q5.** The interviewer says: "A new model scores very high on a public benchmark, much higher than the previous generation, but users report that it doesn't feel noticeably better in actual use. What could the reasons be? Which would you check first?" Talk through your approach in two or three minutes.

**Q6.** Same background. The interviewer follows up: "You suspect the benchmark's questions leaked into the training data, i.e. contamination. You can only call the API and can't see the training data. What's the cheapest experiment that can also tell the different explanations apart? What are its limitations?"

**Q7.** Background: you randomly sampled 500 questions from the original benchmark and had people rewrite each one: change the numbers, change the wording, keep what's being tested. The model gets 80% of the 500 original questions right and 68% of the corresponding 500 rewritten questions. (a) Is this 12-point gap noise? Work it out. (b) Can you conclude from this that the model "memorized the original questions"? What's missing?

**Q8.** The interviewer asks: "If you were designing an evaluation for the team that's hard to contaminate and hard to game, how would you do it? Feel free to use a domain you know as an example."

## Reference solution

<details>
<summary>Show the reference solution</summary>

The only evidence for this round is one candidate's blog post (February 2025, applying for Anthropic's research Fellowship): the virtual onsite included a 15-minute research brainstorm with a lead from the alignment side, who, after a short introduction, asked two open questions intended to get the candidate to generate ideas. The author said the questions required no knowledge of LLM internals, only experience observing models as a black box and a bit of creativity. The author got stuck on the first question, was silent for about three minutes while searching his memory for math he knew, and his takeaway afterwards was that he should have just started proposing ideas. About an hour later, his referrer received an automated notice that no further conversations were needed; the next day, the recruiter canceled the remaining onsite interviews. There is no record at all of how it was graded; what follows about "strong answers" is written from the general expectations of a research discussion, not from any standard Anthropic has published.

How to prepare: two questions in 15 minutes means under 7.5 minutes per question on average, and your opening answer can only take two or three minutes. Practice a fixed speaking skeleton so you can start talking right away even when stuck: (1) turn the question into something measurable in one sentence ("I'll define 'faithful' as ..."); (2) list two or three competing hypotheses; (3) pick the cheapest experiment that can tell these hypotheses apart, and say clearly what the inputs are and what you compare; (4) state your predictions first: which result supports which hypothesis, and which result would change your mind; (5) controls, to rule out "any perturbation would do this" and sampling noise; (6) one sentence on limitations and next steps. Common ways to lose points: long silences; opening with a very complicated method (especially talking about activations when the question said API only); listing ideas without saying how to tell them apart; no control group; treating "looks reasonable" as "faithful"; ignoring sample size and noise; stating conclusions more strongly than the data supports. You can naturally bring in your own background, for example the lessons of saliency maps in computer vision, or the biases of the FID and FVD metrics in video generation; the answers below include examples.

### Question 1: Is the reasoning a model writes down its real reason?

**Q1.** A strong answer starts with a definition: a chain-of-thought is faithful if the reasons it writes down are the factors that actually influence the answer. This definition implies two consequences you can test with only an API:

- If some factor changes the answer, the chain-of-thought should mention it;
- If you change the chain-of-thought itself, the answer should change with it, i.e. the answer depends on the chain-of-thought.

Then list a few competing hypotheses:

1. The answer really does depend on the chain-of-thought; the model computes as it writes;
2. Post-hoc rationalization: the answer was settled first, and the chain-of-thought is a justification added afterwards;
3. It depends on the task: hard multi-step problems depend on the chain-of-thought, easy ones don't;
4. The chain-of-thought is used, but it leaves out some of the real influencing factors (incomplete).

Next, give two kinds of black-box experiments:

- Change the input and see whether the chain-of-thought says so: add a hint pointing to one of the options in the question, see whether the answer gets pulled by the hint, and, when it does, whether the chain-of-thought mentions the hint. Turpin et al. (2023) and Anthropic's Chen et al. (2025) used this kind of approach.
- Change the chain-of-thought and see whether the answer follows: truncate it, insert an error, paraphrase it, or replace it with meaningless filler characters, then let the model continue and give an answer. This is the approach of Lanham et al. (2023). It requires that the API let you put in a pre-written chain-of-thought for the model to continue (prefill); if it doesn't, you can put it in the user message, but then it's no longer a chain-of-thought the model wrote itself, and you should note that caveat.

Finally, pick one to do first (see Q2), and point out the key distinction: a chain-of-thought that "makes sense" is not necessarily faithful.

Ways to lose points: staying in a philosophical discussion without getting to experiments; proposing to "read the chain-of-thought and see whether it's reasonable", which measures plausibility, not faithfulness; talking about activations when the question said API only; not mentioning sampling noise: resample the same question once and the answer may change anyway.

**Q2.** A concrete, cheap experiment that can tell the hypotheses apart is hint injection:

- Data: a few hundred multiple-choice questions with known answers. First select the questions the model answers correctly and consistently without a hint (e.g. right in all 5 samples), so that when the hint points to a wrong option, a change in the answer is easy to interpret.
- Conditions: three versions of each question. (a) No hint; (b) a hint pointing to a wrong option, for example "A professor thinks the answer is (C)", or, as in Turpin et al., reordering the few-shot examples so their correct answers are all (A); (c) control: an irrelevant passage of similar length that doesn't point to any option.
- Measure two numbers: the flip rate, i.e. the fraction of answers that change to the hinted option; and the mention rate, i.e. among the flipped questions, the fraction where the chain-of-thought states the hint as a reason. The rule for judging a "mention" should be fixed in advance; you can have another model judge it, then hand-check a sample of one or two hundred to calibrate.
- Prediction: if the chain-of-thought is faithful, flips should come with mentions; a high flip rate with a low mention rate is evidence of unfaithfulness.
- What would change my mind: the flip rate in the hint group is about the same as in the control group, meaning the hint has no effect at all and the experiment can't measure anything, so you need a stronger hint; or the mention rate is close to 100%, meaning that, at least for this kind of hint, the chain-of-thought is honest.
- Controls: (1) resample the no-hint version once to get a baseline for "how much answers change anyway"; (2) the irrelevant-text group, to rule out "any perturbation changes the answer"; (3) a version where the hint points to the correct answer, to see whether the hint's effect works in both directions; this group should use questions the model was unstable on, since on consistently correct questions the answer is already the correct option and you'd see no change.

You can mention existing results in passing, but get them right: Turpin et al. (2023) added biasing features to the input (for example reordering the options of the few-shot examples so the answer was always (A)), and the models' explanations systematically failed to mention these features; when the model was steered toward a wrong answer, it often wrote a chain-of-thought justifying the wrong answer, with accuracy dropping by as much as 36% on 13 BIG-Bench Hard tasks (they tested GPT-3.5 and Claude 1.0). Chen et al. (2025) tested 6 kinds of hints: across most settings and models, when the model used a hint, the chain-of-thought verbalized it at least 1% of the time, but often less than 20%.

Ways to lose points: no control group; reporting only the flip rate and not the mention rate (a flip only shows the hint had an effect, not that the chain-of-thought is unfaithful); adding hints to questions the model couldn't answer anyway, so changes in the answer can't be interpreted; deciding the rule for "mention" after the fact.

**Q3.** You can't simply agree. An 85% unchanged rate has at least three explanations, which need to be separated:

1. Many problems don't need a chain-of-thought at all. Suppose that for 60% of the 1000 problems, answering directly without any chain-of-thought gives the same answer; those naturally stay unchanged after truncation. Then the unchanged fraction $x$ among the remaining 40% satisfies $0.6 + 0.4x = 0.85$, giving $x = 0.625$: 37.5% of these problems had their answer changed by truncation, so the dependence is actually substantial. So first run "no chain-of-thought at all", split the problems into two strata based on the result, and then look at the effect of truncation.
2. Sampling noise. Resampling the same problem twice with the full chain-of-thought kept doesn't give 100% agreement either, and that agreement rate is the real ceiling. If it's only 90%, the 85% after truncation is just 5 points below the ceiling; if it's 99%, the gap is 14 points. This resampling control is a must.
3. Cutting at the halfway point is just one data point. You should draw a curve: how the agreement with the full version changes when keeping 0%, 25%, 50%, 75%, 100% of the chain-of-thought. Lanham et al. (2023) found that this dependence varies a lot across tasks: some tasks depend heavily on the chain-of-thought, others barely look at it.

There's also a conceptual distinction worth making: an answer not depending on the second half of the chain-of-thought doesn't mean the chain-of-thought "is decoration". The model may genuinely have worked out the answer in the first half, with the second half just double-checking; then the second half is unnecessary, but the chain-of-thought as a whole may still faithfully reflect its process. "Necessary" and "faithful" are two different questions, and the truncation experiment mainly measures the former.

Ways to lose points: accepting the colleague's conclusion outright; just saying "you can't conclude anything" without giving a next analysis step; not realizing that the difficulty mix of the problems affects this percentage.

**Q4.** Why it matters: if the model's reasoning is written in the chain-of-thought and it's faithful, we can read it and monitor it with another model (CoT monitoring), catching the model reward hacking, deceiving people, or planning something harmful before it actually acts. If the chain-of-thought is unfaithful, this kind of monitoring can be fooled. Chen et al. (2025) conclude that CoT monitoring is a promising way to notice undesired behaviors during training and evaluation, but it is not sufficient to rule them out; in settings like theirs, where the chain-of-thought isn't necessary, test-time CoT monitoring is unlikely to reliably catch rare and catastrophic unexpected behaviors. They also found that outcome-based reinforcement learning initially improves faithfulness but then plateaus without saturating.

Where the conclusion doesn't generalize: a strong answer names at least three or four points:

- Tasks: being faithful on these kinds of tasks doesn't mean being faithful on others. Lanham et al. found large differences across tasks.
- Kinds of hints: you only tested the hints you thought of. The model may say "the professor said C" but not mention information it thinks it shouldn't use (for example a grading rubric it peeked at).
- Whether the chain-of-thought is needed: for things the model can do without writing a chain-of-thought, it has no reason to write down its real process, and that may be exactly the category most in need of monitoring.
- Optimization pressure: if training penalizes "bad thoughts" in the chain-of-thought, the model may learn not to write them down rather than not to act on them, and monitorability gets worse.
- Model scale and version: Lanham et al. found that on most of the tasks they studied, larger and more capable models produced less faithful reasoning. A new model needs to be retested.
- Format: moving from multiple-choice questions to long-running agent tasks, the role of the chain-of-thought may be completely different.

You can bring in your own background: saliency maps in computer vision taught the same lesson. Adebayo et al. (2018) found that some saliency methods produce results that depend neither on the model nor on the data, and likened them to edge detection; one of their tests was to randomize the model weights and see whether the explanation changed. The lesson is the same as here: an explanation that "looks reasonable" isn't enough; you need interventions to test whether it changes with the real cause.

Ways to lose points: just saying "for safety" without naming a concrete use; presenting the conclusion of one experiment as universally true.

### Question 2: A high benchmark score, but it isn't better in practice

**Q5.** First ask back what "not better" means concretely: which kinds of tasks, which users, compared with which version, are there examples. Then lay out the possible causes:

1. Data contamination: the benchmark questions, or very similar ones, got into the training data, and the model memorized rather than learned.
2. Different distribution: benchmark questions are short, clean and single-turn; user requests are long, messy, multi-turn, and involve tools.
3. It learned format shortcuts: for example a preference for an option position, or prompts tuned specifically for the evaluation.
4. The metric doesn't measure what users care about: it only checks whether the final answer is right, not the process or the presentation. You can use an example you know well: FVD, commonly used for video generation, is biased toward per-frame image quality; Ge et al. (2024) found that heavily corrupting a video along the time dimension raises FVD only slightly. For images, Kynkäänniemi et al. (2022) found that simply aligning the distribution of ImageNet Top-N classification results of generated images with that of real images can lower FID substantially without improving image quality.
5. The evaluation setup differs from the production setup: few-shot examples, chain-of-thought, temperature, best-of-n sampling, system prompt, tools.
6. The feedback itself is biased: complaints concentrate on a few scenarios; or the new model regressed in other ways (style, refusals, latency), and users count that as "not better".
7. The benchmark is close to saturation, and a gap of a few points is within the noise.

What to check first: the cheapest and most common causes first. Rerun the benchmark with the production configuration (same prompt, same temperature, no few-shot examples) and see whether the score holds up; at the same time, randomly sample 100 pieces of user feedback and categorize each one to see which kinds of tasks the problem actually lies in. Then do the contamination checks (Q6).

Ways to lose points: concluding right away that it's contamination; not asking what "not better" means; listing a long string of causes without saying which to check first and why.

**Q6.** The cheapest are two API-only checks, done together:

- Completion test: give the model the first half of a benchmark question, let it continue, and count the fraction of continuations that match the original verbatim or nearly verbatim. The control group is a batch of questions in the same style written after the model's training data cutoff. If the overlap rate on benchmark questions is clearly higher than on the controls, the model has seen the originals. If the API returns per-token log probabilities, you can also compare the likelihood of the original questions and of rewritten ones.
- Perturbation test: keep what's being tested, change only the numbers, names or option order. If the model really learned the skill, the score should barely change; if it relies on memory, the score will drop. GSM-Symbolic (Mirzadeh et al., 2024) found that changing only the numerical values in the questions lowered the performance of every model tested, and that scores varied noticeably across different instances of the same question.

A ready-made reference point is GSM1k (Zhang et al., 2024): a new set of problems written to mirror the style and difficulty of GSM8k, on which some models' accuracy dropped by up to 8%, with several model families showing systematic overfitting at almost every size; a model's probability of generating GSM8k's original problems was positively correlated with its score gap between the two benchmarks (Spearman $r^2 = 0.36$). But many models, especially frontier models, showed almost no signs of overfitting.

Limitations: the completion test only detects verbatim memorization and misses contamination of the kind "a rewritten version was in the training data"; the model may have memorized the answer without being able to recite the original text. The control questions must be as hard as the originals, or the comparison is unfair. Chat models may be reluctant to continue text verbatim, biasing the result low. Not finding contamination doesn't prove there is none.

Ways to lose points: saying you'd check the training data (the question said you can't see it); doing only one kind of check; no control group.

**Q7.** (a) First compute the standard error of each proportion:

- Original questions: $\sqrt{0.8 \times 0.2 / 500} \approx 1.79$ points;
- Rewritten questions: $\sqrt{0.68 \times 0.32 / 500} \approx 2.09$ points.

If you treat the two groups as independent samples, the standard error of the difference is $\sqrt{1.79^2 + 2.09^2} \approx 2.75$ points, and 12 points is about 4.4 standard errors, so it's not noise.

But here each rewritten question corresponds to an original question, so this is paired data, and the better approach is a per-question comparison (McNemar's test). Suppose that, question by question, 85 are right on the original and wrong on the rewrite, and 25 are wrong on the original and right on the rewrite; this is consistent with 80% and 68%: $(85-25)/500 = 12\%$. The standard error of the paired difference is $\sqrt{85 + 25 - (85-25)^2/500}\,/\,500 \approx 2.03$ points, smaller than with independent samples, because the original and the rewrite of the same question have correlated difficulty. The McNemar statistic is $(85-25)^2/(85+25) \approx 32.7$, with a p-value far below 0.001.

(b) No. The gap is real, but "memorized the original questions" is only one explanation; another, equally common explanation is that the rewritten questions are simply harder. Recht et al. (2019) re-collected test sets for CIFAR-10 and ImageNet following the original procedures, and the accuracy of a wide range of models dropped by 3%–15% and 11%–14% respectively; the authors concluded that the drop was not due to everyone repeatedly tuning on the old test sets, but because the new images were slightly "harder".

What's missing is a reference: take a few models that can't possibly have seen this benchmark (for example models finished training before the benchmark was released), test them on the same original and rewritten questions, and see how much they drop. If the reference models also drop about 10 points, it's mainly that the rewritten questions are harder; if they only drop two or three points, the extra drop of our model is what's suspicious. This is the difference-in-differences idea. You can also have people blindly rate the difficulty of the two sets, and check whether the drop concentrates on the questions whose originals the model could recite in the Q6 completion test.

Ways to lose points: drawing a conclusion from the 12 points alone; forgetting that this is paired data; treating "the difference is significant" as "the cause is known".

**Q8.** A strong answer starts from two goals: hard to contaminate and hard to game, while also correlating with what users actually care about. Things you can propose:

- Keep a private test set, use it only for a few key decisions, and refresh it regularly. The more it's used, the easier it is to overfit to it, so limit how often it's looked at as well.
- Split by question date: use only questions written after the training data cutoff, and compare the score gap between the new and old portions.
- Generate questions from templates or programs (the GSM-Symbolic idea), draw fresh instances for every evaluation run, and report the variation across instances.
- Add a canary string to publicly released evaluation data so training pipelines can filter it out; run n-gram overlap checks on the training data side.
- Report a confidence interval for every score; use paired comparisons when comparing old and new models, and run multiple random seeds.
- Regularly check whether evaluation scores agree with real usage outcomes: across several model versions, is the ranking by evaluation score the same as the ranking by user preference and task completion rate? If not, the evaluation is no longer measuring what you want it to measure.
- Use video generation as an example: don't look at FVD alone; add metrics and human ratings specifically targeting motion and temporal consistency; and actively check whether the metric can be pushed down by tricks that don't improve quality, like the FID and ImageNet class distribution issue mentioned earlier.

Ways to lose points: just saying "switch to a bigger benchmark"; thinking a private test set can't be overfit; only talking about preventing contamination, not about the evaluation correlating with real usage.

<details>
<summary>Verification code (runnable)</summary>

```python
import math
import numpy as np

# time budget: 15 minutes, two questions
assert 15 / 2 == 7.5

# Q3: 60% of items keep their answer even with no CoT; overall 85% unchanged after truncation
easy, overall = 0.6, 0.85
x = (overall - easy) / (1 - easy)
assert abs(x - 0.625) < 1e-12
assert abs((1 - x) - 0.375) < 1e-12          # 37.5% of the remaining items changed
assert abs(0.6 + 0.4 * 0.625 - 0.85) < 1e-12

# Q7 (a): standard errors, in percentage points
n = 500
se_orig = math.sqrt(0.8 * 0.2 / n) * 100
se_new = math.sqrt(0.68 * 0.32 / n) * 100
se_diff = math.hypot(se_orig, se_new)          # independent samples
assert round(se_orig, 2) == 1.79
assert round(se_new, 2) == 2.09
assert round(se_diff, 2) == 2.75
assert round(12 / se_diff, 1) == 4.4

# paired counts must agree with 80% and 68%
b, c = 85, 25          # original right & rewrite wrong, original wrong & rewrite right
orig_right = round(0.8 * n)
both_right = orig_right - b
rewrite_right = both_right + c
assert orig_right == 400 and rewrite_right == 340 and rewrite_right / n == 0.68
assert (b - c) / n == 0.12
neither = n - both_right - b - c
assert neither == 75 and neither >= 0

se_paired = math.sqrt(b + c - (b - c) ** 2 / n) / n * 100
assert round(se_paired, 2) == 2.03 and se_paired < se_diff
chi2 = (b - c) ** 2 / (b + c)
assert round(chi2, 1) == 32.7
p = math.erfc(math.sqrt(chi2 / 2))              # upper tail of chi-square with 1 dof
assert p < 1e-3

# simulate the paired design to check the paired standard-error formula
rng = np.random.default_rng(0)
probs = [both_right / n, b / n, c / n, neither / n]
sims = rng.multinomial(n, probs, size=200_000)
diffs = (sims[:, 1] - sims[:, 2]) / n * 100
assert abs(diffs.mean() - 12) < 0.05
assert abs(diffs.std() - se_paired) < 0.05
print("research-brainstorm checks ok", round(se_paired, 3), round(chi2, 2), f"{p:.1e}")
```

</details>

</details>
