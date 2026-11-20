# Hiring-Manager Round: Evaluating a Multi-Task Model Across Domains, and Improving a Model Without New Data

## Problem

Two open-ended questions from the hiring-manager round, answered out loud, about 10 minutes each; the interviewer follows up based on what you say. The first question in each group is the one recorded in the interview report; the follow-ups were written by us, modeled on the way this kind of interview usually probes. Each question carries its full context, so you can practice any of them on its own.

### Question 1: Evaluating a multi-task system across domains

**Q1.** The interviewer says: "We have a multi-task system: one model handles requests from several domains, such as general Q&A, coding, medical questions, and visual question answering. How would you evaluate how well it performs in each domain?" Talk through your approach for a few minutes, as you would in the interview.

**Q2.** Context: one model serves three domains, general Q&A, code, and medical, and the team is comparing an old and a new version. (a) On the same fixed test set: general Q&A has 8,000 items and accuracy goes from 80% to 82%; code has 1,500 items, from 60% to 61%; medical has 500 items, from 70% to 58%. Pooling all items together, the new version's overall accuracy is 1.15 points higher. The interviewer asks: "Can we ship it?" (b) A second set of numbers comes from production logs: last month, running the old version, 900 of 1,000 easy requests and 50 of 100 hard requests were answered correctly; this month, running the new version, 276 of 300 easy requests and 385 of 700 hard requests were answered correctly. Overall accuracy fell from 86.4% to 66.1%. Did the new version get worse? (c) If the per-domain metrics are not the same kind of thing at all (say mIoU for segmentation, mAP for detection, CIDEr for video captioning), how would you combine them into one number? Should you combine them at all?

**Q3.** Context: the same multi-task model, and the medical test set has only 200 items. The old version gets 80% right and the new version 82%; comparing item by item, there are 14 items the new version gets right and the old one gets wrong, and 10 items the old version gets right and the new one gets wrong. (a) Is this 2-point improvement credible? Work out how large the error is. (b) Roughly how many items would you need to reliably detect a 2-point difference? (c) There are 10 domains in total, and each one is tested for "did it get worse" at a 5% significance level. What goes wrong? (d) The team wants to add a launch gate: "No domain may be more than 3 points worse than the old version." How should this gate be implemented for a domain with only 200 items?

**Q4.** Context: the same multi-task model; part of the test set comes from public benchmarks. The interviewer follows up: "This model will later be used in domains we haven't tested yet. Also, public benchmark items may have leaked into the training data. How do you know it has really learned the skill rather than memorized the answers? What checks would you put in place before launch?"

### Question 2: Making a model better without new data

**Q5.** The interviewer says: "Suppose you can't get any new data. How would you make a model better?" Talk through your approach for a few minutes, as you would in the interview.

**Q6.** Context: a model that classifies objects in video frames has a 12% error rate on the validation set, and you can't collect any new data. You randomly sample 100 of the validation errors, look at each one, and categorize them: 30 are label errors (the model was actually right), 25 are small objects, 20 are motion blur, 15 are confusions between similar-looking classes, and 10 are other. (a) At most how much can each category reduce the error rate? (b) If those 30 mislabels were corrected, roughly what is the model's true error rate? (c) How accurate is the figure "small objects make up 25%"? (d) Besides categorizing by error type, what kind of problem is worth checking first?

**Q7.** Context: the same video-frame classification model, no new data allowed; the labels have already been cleaned and the error analysis has been done. (a) What else can you do next? Order the options in the sequence you would try them, and for each say when it helps and how you would know whether it helped. (b) Someone proposes averaging 5 models trained with different random seeds. Let the variance of a single model's prediction error be $\sigma^2$ and the correlation coefficient between any two models' errors be $0.6$. What is the error variance after averaging? What about with infinitely many models? What does this say about when an ensemble is worth it?

**Q8.** Context: a model, no new data allowed, and you are tuning it on the validation set. Over a week you try 20 changes; each change is trained once and evaluated once on the validation set, and the best one beats the baseline by 0.3 points. You know that for the same configuration with only the random seed changed, the standard deviation of the validation score is about 0.2 points. The interviewer asks: "Is this improvement real? How would you confirm it?"

## Reference solution

<details>
<summary>Show the reference solution</summary>

The hiring-manager round tests judgment, not how many tricks you remember. Concretely, three things: first pin down the goal and the constraints, then order the steps by benefit and cost, and for every step be able to say "how would I know whether it helped." The evidence is a single report: one ML software engineer interview recorded on the process page, for a team working on Gemini applications, through Google's hiring pipeline; the hiring manager asked "how would you evaluate a multi-task system across domains" and "how would you improve a model when there is no more data." The original questions are just those two sentences; the follow-ups and reference answers were written by us.

How to prepare: for each question, prepare a two-minute skeleton (what you ask first, how many steps, one concrete example per step), plus a few small examples with numbers, such as "the overall average hid a regression in one domain," "how large the error is on 200 items," and "how much of the best result out of 20 tries is noise." Use projects you have worked on yourself where possible, such as video generation or vision models. For the full design of an evaluation system, see [LLM evaluation system](#/deepmind/system-design/llm-evaluation-system); this page only practices the spoken answers and trade-offs.

### Question 1: Evaluating a multi-task system across domains

**Q1.** A strong answer starts with a question back: **which decision is this evaluation meant to inform?** There are three common ones.

- A launch decision: can the new version replace the old one? What you need is a reliable old-vs-new comparison, plus a floor of "no domain may regress noticeably."
- Research direction: which capabilities should we shore up next quarter? What you need is fine-grained capability diagnosis, sliced by domain, difficulty, and input type.
- Debugging: complaints are up in one domain. What you need is localization, looking at the failing examples, not an overall score.

Different decisions call for different test sets, different metrics, and different sample sizes. Once you have asked, walk through the following in order.

1. **Define "good" separately for each domain first.** Q&A uses accuracy or human ratings; code uses unit-test pass rate; medical also needs the rate of harmful answers; visual QA should be broken down by image type. Metrics across domains are often not comparable, so don't rush to average them. The same holds in a setting you know well: a video generation model that does text-to-video, image-to-video, and video editing has its own metric for each task (visual quality, whether it matches the text, whether an edit preserves the original video's content), plus a shared human preference score and safety metrics.
2. **Where the test sets come from.** Sample each domain from real traffic, then add hard examples and long-tail slices separately (long inputs, rare languages, low-resolution images). Labeling needs quality control; have two people label hard items independently.
3. **How to aggregate.** Report both each domain and the total, and say how the total is computed: weighted by traffic, or with domains weighted equally. Looking at a single average hides regressions (see Q2).
4. **How large the error is.** How many items each domain has, how wide the confidence interval is, and the old and new versions should be compared pairwise on the same items (see Q3).
5. **Unseen domains and memorization.** Hold out domains, and keep test items from leaking into the training data (see Q4).
6. **Launch gate.** A maximum allowed regression per domain, with veto power for safety metrics; after passing offline, run a small-traffic A/B test plus manual spot checks.

The two-minute spoken version: "First confirm what decision the evaluation is for. Then define metrics and build a test set for each domain separately; when aggregating, give both per-domain results and a total with its weighting spelled out; attach an error bar to every number and compare old and new versions pairwise; hold out domains that played no part in development and check for test-set leakage; finally set a regression gate per domain, give safety metrics a veto, and then run a small-traffic experiment."

Common ways to lose points:

- Opening with "compute accuracy, then take the average," without asking what decision the evaluation needs to support.
- Reporting only one overall score, and being unable to say how it is weighted or what it hides.
- Not mentioning how many items each domain has or how large the error is.
- Using the test set repeatedly as a tuning tool, and then still reporting it at the end.

**Q2.** (a) The two averages lead to opposite conclusions.

- Weighted by item count (micro average, equal to "the probability of getting a randomly drawn item right"): old $(8000 \times 80 + 1500 \times 60 + 500 \times 70)/10000 = 76.5\%$, new $77.65\%$, up $1.15$ points.
- Domains weighted equally (macro average): old $(80 + 60 + 70)/3 = 70.0\%$, new $(82 + 61 + 58)/3 = 67.0\%$, down $3$ points.

Medical dropped 12 points, masked by the 8,000 general Q&A items. Those 12 points are not noise: with 500 items and accuracy around 70%, the standard error of a single version is about $\sqrt{0.7 \times 0.3/500} = 2.0$ points, the standard error of the difference between two versions is about $3.0$ points, and 12 points is 4 times that.

So the answer is "not on the overall number alone." Weighting by item count reflects the average user experience under current traffic; equal weighting protects small domains; medical is a high-risk domain and needs its own gate even with little traffic. A strong answer says: report each domain's result and both averages, flag the regressing domain, decide with per-domain gates rather than one number, and at the same time find out why medical regressed (did the training-data mix change? is one type of question failing entirely?).

(b) The new version is better on both kinds of request: easy requests $90\% \to 92\%$, hard requests $50\% \to 55\%$. The total got worse because the traffic mix changed: the share of hard requests went from $100/1100 \approx 9\%$ to $700/1000 = 70\%$. The two months are comparing two different populations of requests. This is Simpson's paradox.

The fix is to compare on the same distribution. Recomputing with a fixed 50/50 mix, old is $70.0\%$ and new is $73.5\%$; recomputing with this month's mix (30% easy, 70% hard), old is $62.0\%$ and new is $66.1\%$. Better still is to have both versions handle the same batch of requests: replay the same logs offline, or run a randomized A/B split over the same time window. The point to remember: when the traffic mix changes, the overall metric moves, regardless of whether the model is better or worse.

(c) A few approaches, in order of preference:

- **Don't combine them.** Set a gate per domain, then check whether any domain is being sacrificed. For a launch decision this is usually the safest.
- **Normalize relative to a baseline, then take the geometric mean.** For example, mIoU $45.0 \to 46.8$, mAP $38.0 \to 37.2$, CIDEr $110 \to 121$ give ratios of $1.040$, $0.979$, $1.100$, with a geometric mean of $1.038$. But mAP dropped $2.1\%$, which you can't see from the combined number alone.
- **Why not the arithmetic mean of ratios:** its result depends on which system you take as the reference. Say two systems score $X = (10, 20)$ and $Y = (20, 10)$ on two metrics. With $X$ as the reference, $Y$'s ratios are $(2, 0.5)$, arithmetic mean $1.25$; with $Y$ as the reference, $X$'s ratios are also $(2, 0.5)$, again $1.25$. Each system is "25% better than the other," which is a contradiction. The geometric mean is $1$ both ways.
- **Weight by importance.** That's fine, but the weights are a product decision; say explicitly who set them and why.

Common ways to lose points: insisting that one particular average "is the right one" without tying it to the decision; not noticing a large regression in a small domain; directly comparing two months of logs as if they were the same population of requests.

**Q3.** (a) Start with a single version: 80% accuracy on 200 items gives a standard error of $\sqrt{0.8 \times 0.2/200} = 2.8$ points, and a 95% confidence interval of roughly $\pm 5.5$ points.

For the difference between the two versions, treating them as two independent samples gives a standard error of about $\sqrt{(0.8 \times 0.2 + 0.82 \times 0.18)/200} = 3.9$ points, and 2 points is only half of that. But here the items are the same, so you should do a paired comparison: items both versions get right or both get wrong carry no information about the difference; only the 24 discordant items matter. The difference is $(14 - 10)/200 = 2$ points, the paired standard error is $\sqrt{14 + 10 - (14 - 10)^2/200}\,/\,200 \approx 2.4$ points, and the ratio is about $0.8$. The exact McNemar's test: under the null hypothesis 14 follows $\mathrm{Binomial}(24, 1/2)$, two-sided $p \approx 0.54$. Conclusion: no detectable difference. Pairing brought the error down from 3.9 to 2.4, but that is still nowhere near enough.

(b) Suppose about 12% of items are discordant between the two versions (new right / old wrong 7%, old right / new wrong 5%, a 2-point difference). For a two-sided 5% test to have 80% power to detect this difference, a paired comparison needs about

$$n \approx \frac{(1.96 + 0.84)^2 \times (0.12 - 0.02^2)}{0.02^2} \approx 2350$$

items; if the two versions were evaluated on different items, each version would need about 6,000. When you can't get that many: pool similar domains; switch to a more sensitive metric (a per-item score, or the log-probability of the correct answer, instead of right/wrong); sample each item several times; or accept that this test set can only detect larger differences.

(c) Multiple comparisons. Even if none of the 10 domains changed, each has a 5% false-alarm probability, so the probability that at least one domain falsely flags "worse" is $1 - 0.95^{10} \approx 40\%$ (assuming domains are independent). Remedies: use a stricter level per domain (Bonferroni, $5\%/10 = 0.5\%$), or use the Holm method or control the false discovery rate; decide in advance which domains are gates; re-check any flagged domain on fresh data.

(d) The gate has to be set together with the sample size. With 200 items and a 12% discordance rate, even if the two versions are identical, the standard error of the difference is $\sqrt{24}/200 \approx 2.4$ points, and the one-sided 95% lower confidence bound is about $-4.0$ points. If the gate is written as "the lower bound may not fall below $-3$," even identical versions fail; if it is written as "the point estimate may not fall below $-3$," versions with a true regression of 3 to 5 points will often slip through. For the lower bound of two identical versions to land within $-3$, you need at least about 360 items.

So in practice: for high-risk small domains (medical), add items until there are enough, or add human review; for low-risk small domains, loosen the gate or merge with a similar domain; and write "how large a regression this test set can detect" into the gate's documentation.

Common ways to lose points: taking a 2-point difference at face value; having paired data and computing as if the samples were independent; not mentioning multiple comparisons; a gate that is only a number, with no corresponding sample size.

**Q4.** Answer it as two separate issues: unseen domains, and memorization.

**Unseen domains.** Hold out domains (leave-one-domain-out): exclude some domains completely from training and tuning, evaluate on them only once at the end, and see whether the capability transfers. A production model can't be retrained once per domain, so at minimum keep a few domains and slices that have "never been used for tuning" as the final check. You can also look at whether the model "says it doesn't know" or "answers wrong with confidence" in a new domain; here calibration matters more than accuracy. The equivalent in video generation: hold out motion types that rarely appear in training (fluids, fast camera moves), art styles, aspect ratios, and durations, and evaluate them separately.

**Memorization (contamination).**

- Split by time: use only items that appeared after the training-data cutoff.
- Check for overlap: search the training data for the test items' n-grams or vector nearest neighbors; for images and video, use perceptual hashing and near-duplicate frame matching.
- Rewrite the test items: rephrase, change the numbers, re-render the images. A model that has really learned the skill scores about the same; one that memorized drops noticeably.
- A private test set: don't publish it, and put a canary string in the items, so that finding it later in training data shows a leak; keep comparing the score gap between public benchmarks and the private set — if the public score keeps climbing while the private one doesn't move, that is memorization.
- A video generation example: the prompt lists of public benchmarks may appear in the captions of training videos, so deduplicate before evaluating.

Another kind of contamination comes from yourself: evaluating and tuning on the same dev set over and over eventually overfits to it too. So keep a test set that you rarely touch.

**Checks before launch.** Set a regression gate per domain using the method in Q3; safety metrics (harmful answers, privacy leaks) get a veto; latency and cost stay within budget; then a small-traffic A/B test with guardrail metrics and manual spot checks, with rollback conditions written down in advance.

Common ways to lose points: reporting only public benchmarks; equating "high score" with "has the capability"; having no plan at all for unseen domains; guarding only against training-data leakage, without realizing that your own repeated tuning is also a kind of leakage.

### Question 2: Making a model better without new data

**Q5.** A strong answer first asks what exactly "no new data" fixes, because the answer depends on it:

- No new labels — but are there unlabeled inputs? Can the existing labels be reworked and cleaned?
- What is the compute budget? Can the model architecture change? Can external pretrained weights be used?
- Are there latency and cost limits at inference? That decides whether ensembling and test-time augmentation are options.
- What is the metric? A threshold-free one like accuracy, or a thresholded one like "precision at 95% recall"?
- Is the model currently underfitting or overfitting? What are the training error and the validation error?

Then lay out a ladder ordered by "high expected benefit, low cost first," and for every step say when it helps and how you would know whether it helped:

1. **Look at the errors, check the pipeline.** Sample error cases and categorize them, check for data and pipeline bugs (see Q6). Cheapest, and often the biggest win.
2. **Make better use of existing data.** Fix mislabels, remove duplicates and bad samples, reweight or resample rare classes, apply data augmentation targeted at the error types, and use curriculum learning (easy to hard).
3. **Training recipe and regularization.** First decide the direction from the training and validation errors: if overfitting, add augmentation, weight decay, dropout, label smoothing, mixup; if underfitting, use a bigger model, train longer, tune the learning rate. Separately, the learning-rate schedule and an exponential moving average (EMA) of the weights are almost always worth tuning.
4. **Transfer.** If allowed, initialize from better pretrained weights, or train jointly with related tasks.
5. **When there are unlabeled inputs.** Self-training (using high-confidence pseudo-labels), knowledge distillation (a large model teaching a small one), and self-supervised pretraining on those inputs.
6. **At inference time.** Ensembling multiple models, test-time augmentation (e.g. flips, multiple crops, averaging over several frames). The cost is multiplied inference cost.
7. **When the metric is thresholded.** Calibrate first (e.g. temperature scaling), then choose the operating point according to the business requirement. The metric gain comes from choosing the right operating point; calibration by itself does not change the ranking of scores, so ranking-only metrics such as AUC or precision at a fixed recall do not move because of it. Its use is to make thresholds expressed as probabilities, such as "act only when the probability exceeds 0.9," land in the right place.

Judging whether each step helped relies on the same set of things: a fixed validation set, the noise level given by multiple random seeds, results sliced by error type, and a test set used only once at the very end.

If your background is in generative models, you can use your own examples: for a video generation model with no new data, you can re-caption the training videos with more detailed descriptions, filter out low-quality clips, bucket samples by resolution and duration, tune the noise schedule and the timestep sampling during training, tune the strength of classifier-free guidance, use EMA weights, and tune the number of sampling steps or do step distillation. Note that data augmentation on generative models "leaks" into the generated outputs (e.g. if you flip during training, the model will generate flipped images). There are two common fixes: StyleGAN2-ADA augments only the discriminator's inputs and controls the augmentation probability so it doesn't leak into the outputs; EDM feeds the augmentation parameters into the model as a condition and sets the condition to "no augmentation" at generation time.

The two-minute spoken version: "First pin down the constraints: is there unlabeled data, can we use pretrained weights, what is the inference budget, is the metric thresholded. Then step one is always to look at the errors and check for pipeline bugs; step two is to make good use of existing data; step three is to adjust the recipe depending on whether we are underfitting or overfitting; if there is unlabeled data, do self-training or distillation; if the inference budget allows, ensemble; if the metric is thresholded, calibrate. Judge every step on a fixed validation set with multiple seeds, and look at the test set only once at the end."

Common ways to lose points:

- Rattling off a dozen techniques with no ordering and no reason for doing one first.
- Not asking about constraints and going straight to "use a bigger model," without first checking whether you are underfitting or overfitting.
- Being unable to say how you would judge whether a change helped.
- Choosing methods on the test set.

**Q6.** (a) The ceiling for each category equals "total error rate × that category's share":

- Label errors: $12\% \times 0.30 = 3.6$ points. This needs no model change; once the validation labels are corrected, the measured error rate drops.
- Small objects: $3.0$ points; motion blur: $2.4$ points; similar classes: $1.8$ points; other: $1.2$ points.

These are ceilings: even if the small-object problem were solved completely, the error rate would drop by at most 3 points; in practice you will only solve part of it. The use of this calculation is to decide what to do first, and to rule out directions that "wouldn't gain much even if they worked."

(b) Those 30 were actually cases where the model was right, so the true error rate is about $12\% \times 0.7 = 8.4\%$. Fix the validation labels first, then compare models; otherwise every later comparison carries these 3.6 points of noise. Also check whether the training set has the same proportion of mislabels: if 30% of validation errors are label errors, the training set is probably not clean either, and fixing the training labels may by itself improve the model.

(c) 25 out of 100 samples: the 95% confidence interval for the proportion is about $25\% \pm 1.96\sqrt{0.25 \times 0.75/100} = 25\% \pm 8.5\%$, which puts the corresponding ceiling somewhere between roughly 2.0 and 4.0 points. That is good enough for a rough ordering, but it can't tell 30 apart from 25. For more precision, look at more samples, focusing on the top few categories.

(d) Check for data and pipeline bugs first; they are often worth more than any model change:

- Preprocessing mismatch between training and inference: resizing method, RGB vs BGR channel order, normalization constants, frame sampling rate, a different video decoder in training than in production.
- Data leakage: adjacent frames of the same video appear in both the training and validation sets, inflating the validation score. Split by video, not by frame.
- A wrong label mapping, so class indices don't line up; augmentation accidentally left on during validation.

How to check: run the same batch of inputs through the training-time evaluation code and the production inference code and compare whether the outputs match; train on a tiny subset and confirm the model can fit it to close to 100%.

Once you have found the main error types, targeted fixes: for small objects, higher input resolution, multi-scale training, crop-and-zoom; for motion blur, add blur augmentation, or use temporal information from adjacent frames; for similar classes, look at the confusion matrix first, confirm whether the labeling criteria for the two classes are consistent, and if needed merge the classes or switch to hierarchical labels.

Common ways to lose points: changing the model without looking at the errors; treating 25% as an exact value; starting to compare models before fixing the validation labels.

**Q7.** (a) A reasonable order:

1. **Decide the direction first.** For example, 2% training error and 12% validation error is overfitting, so add augmentation and regularization; if the training error is also 11%, it is underfitting, so add capacity, train longer, and tune the learning rate. Get the direction wrong and all the work after it is wasted.
2. **Data augmentation targeted at the error types.** Lots of motion blur: add blur augmentation; lots of small objects: add scale jitter and cropping. How you know it helped: the error rate for that category goes down, and the overall validation score rises by more than the noise from random seeds.
3. **Training recipe.** Learning-rate warmup plus cosine decay, longer training, weight decay, label smoothing, EMA. How you know it helped: look at the learning curves and compare across multiple seeds.
4. **Reweighting or resampling when classes are imbalanced.** How you know it helped: look at per-class recall, not just overall accuracy.
5. **Better initialization.** If allowed, switch to a backbone pretrained on video rather than one pretrained only on images.
6. **When there are unlabeled video frames, pseudo-labeling and teacher-student training.** Use only high-confidence pseudo-labels, and keep an eye on the validation set, because the model will reinforce its own existing mistakes.
7. **Ensembling and test-time augmentation.** Almost always gains a little, but multiplies inference cost; at the end you can distill the ensemble back into a single model.
8. **When the metric is thresholded, calibration and threshold selection.**

Each item should be compared against the baseline on the same validation set with the same few seeds, changing one thing at a time.

(b) Average $M$ models, each with error variance $\sigma^2$ and pairwise correlation coefficient $\rho$:

$$\mathrm{Var}\Big(\frac1M \sum_{i=1}^M e_i\Big) = \frac{1}{M^2}\Big[M\sigma^2 + M(M-1)\rho\sigma^2\Big] = \sigma^2\Big(\rho + \frac{1-\rho}{M}\Big).$$

With $\rho = 0.6$ and $M = 5$: $0.6 + 0.4/5 = 0.68$, i.e. $0.68\,\sigma^2$, and the standard deviation drops to $\sqrt{0.68} \approx 0.825$ times the original. With infinitely many models it only goes down to $0.6\,\sigma^2$. In other words, most of the gain comes from the first few models, and the floor is set by the correlation: the more alike the models, the less the ensemble helps. At $\rho = 0.9$, averaging 5 models only gets you to $0.92\,\sigma^2$. Models that differ only in random seed are often highly correlated; different architectures, different data orderings, and different augmentations lower the correlation. Also note: an ensemble reduces the variance part of the error; a bias shared by all models (e.g. they all struggle with small objects) does not go away by averaging.

Whether it is worth it has to be weighed against the cost: 5× the inference cost for this much variance reduction is usually not worth it in production, but you can build the ensemble first and then distill it into a single model.

Common ways to lose points: listing methods without ordering them; not saying how you would judge whether something helped; treating the ensemble as free; not knowing that the ensemble's gain is limited by correlation.

**Q8.** It can't be trusted yet, for two reasons.

First, single-run noise. With one run each of the change and the baseline, the standard deviation of their difference is $0.2 \times \sqrt2 \approx 0.28$ points, so 0.3 is only about one standard deviation.

Second, selection bias (the winner's curse). Even if all 20 changes did nothing, each one's validation score would fluctuate randomly, and the best of 20 would on average come out about $1.87$ standard deviations high, i.e. $1.87 \times 0.2 \approx 0.37$ points. In other words, a 0.3 improvement is exactly the result you would most likely see if "nothing changed."

How to confirm:

1. With new random seeds, rerun the chosen change and the baseline several times each, say 5 times each. The standard error of the difference in means is about $0.2 \times \sqrt{2/5} \approx 0.13$ points; if the gap is still around 0.3, that is $2.4$ standard errors, which is fairly credible.
2. Do an item-by-item paired comparison on the same validation samples and see whether the gain is concentrated in a few categories or spread across the board; be wary of a gain concentrated in a small handful of samples.
3. Finally, evaluate once on a test set that played no part in any selection, and evaluate on it only that once.
4. When reporting, say how many changes you tried in total.

Common ways to lose points: trusting a single run; not knowing how noisy your own experiments are; picking the best on the validation set and then using that same validation score as the final result; using the test set to choose methods.

<details>
<summary>Verification code (runnable)</summary>

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
