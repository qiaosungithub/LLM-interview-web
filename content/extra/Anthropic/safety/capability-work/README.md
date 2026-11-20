# Doing Capabilities Work at a Safety-Focused Lab

## Problem

Read the notes first, then close them and answer the questions. Anthropic has not published its internal code of conduct, so this topic starts by separating two kinds of content: what the public documents state explicitly, and what can reasonably be inferred from them. When an interview turns to "what would you do once you join", keeping these two apart is itself a plus.

**Stated explicitly in public documents.**

1. **Capabilities work triggers the safety process.** The RSP works on the principle "once a capability crosses a threshold, put safeguards in place first", and whether a threshold has been crossed is decided by capability evaluations. So evaluations are not a formality: their results decide whether safeguards get escalated.
2. **There are reporting channels, and protection for people who report.** A Responsible Scaling Officer oversees how the RSP is carried out, there is an anonymous channel, and there is a separate RSP noncompliance reporting and anti-retaliation policy, which includes a route for making informal inquiries first.
3. **Model weights and internal systems have tiered security requirements.** Half of what an ASL covers is security to prevent weights from being stolen, and that lands directly on everyone who touches a model: access permissions, approvals, and audit trails.
4. **Research that advances frontier capabilities is generally not published.** From *Core Views on AI Safety*: "We generally don't publish this kind of work because we do not wish to advance the rate of AI capabilities progress."
5. **What the values ask of each individual.** The company page says "When it comes to our mission, none of us are bystanders." and "If something urgently needs to be done, the right person to do it is probably you!" In other words, safety is not just the safety team's job.

**Reasonable inferences** (not spelled out item by item in the public documents, but they follow from the points above):

- If you find an unexpected jump in a model's capability or anomalous behavior, you should report it, rather than treating it only as a good result or a bug to fix.
- Treat evaluability and interpretability as constraints to consider at design time, and be willing to work with the alignment and interpretability teams.
- Be honest about experimental results, including results that are bad for your own project.
- Think through for yourself why you are doing capabilities research at a company that worries about the risks.

### Recall

**Q1.** For someone doing capabilities research, which requirements are stated explicitly in public documents? Which are only reasonable inferences?

### Scenarios

**Q2.** You are training a new model, and an internal test partway through shows its capability on a certain class of tasks is far higher than expected, and that class of tasks happens to be close to the dangerous capabilities the RSP is concerned with. The team is racing toward a release date. What do you do?

**Q3.** You find that the model's high score on an evaluation comes from taking a shortcut: it exploited a bug in the grading script instead of actually completing the task. If you tweak the reward or the grading script slightly, the behavior is no longer visible. What do you do?

**Q4.** You find a trick that clearly improves training efficiency, and written up as a paper it would very likely get into a top conference. How do you handle wanting to publish it?

**Q5.** The safety team asks for stricter access control on your experiment cluster: accessing weights requires two-person approval, and experiment iteration will slow down noticeably. You think some of the requirements are unnecessary for your work. What do you do?

**Q6.** You work on video generation models. The new version can produce videos of real people that are almost impossible to tell from genuine footage. The product team wants to launch as soon as possible. What questions do you think must at least be answered before launch?

**Q7.** A senior colleague tells you that a certain pre-release evaluation "won't make it this time, so ship first and backfill it later". You are not sure whether this violates the RSP. What do you do?

### Open questions you will be pressed on

**Q8.** The interviewer asks: "If you're worried about the risks of AI, why are you coming here to do work that makes models more capable?"

## Reference solution

<details>
<summary>Show the reference solution</summary>

**Sources and what was checked (2026-11-17).** The five "stated" points correspond to: the [RSP page](https://www.anthropic.com/responsible-scaling-policy) and the [version 3 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) (how evaluations relate to safeguards, the reporting and anti-retaliation policy); [GovAI's analysis](https://governance.ai/analysis/anthropics-rsp-v3-0-how-it-works-whats-changed-and-some-reflections) (the Responsible Scaling Officer, the anonymous channel); [Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety) (not publishing capabilities research); the [company page](https://www.anthropic.com/company) (the two quoted values).

Not checked, and not checkable: Anthropic's internal code of conduct is not public, and the internal publication-approval process and the specifics of access approvals are unknown. The answers to the scenarios below are suggestions derived from the public principles, not company rules; details such as "two-person approval" were made up for the questions.

**Q1.** Stated: capability evaluations decide whether safeguards are escalated; there are reporting channels and anti-retaliation protection; weights and internal systems have tiered security requirements; research that advances frontier capabilities is generally not published; the values ask everyone to take responsibility for the mission. Inferred: report unexpected capability jumps and anomalous behavior; treat evaluability and interpretability as design constraints; be honest about results; think through for yourself why you do capabilities research.

**Q2.** The key point: this is not just good news, it may be a signal in the RSP sense.

1. First confirm the result is real: retest with different data and different prompts, and rule out evaluation leakage or a script problem.
2. Tell the people responsible for this kind of capability evaluation and your own manager as soon as possible, and give them the raw results, rather than waiting until the formal evaluation to mention it.
3. Do not decide on your own that "it probably hasn't crossed the threshold". Whether it has is for the people responsible for evaluation to judge, following the established process.
4. A release date is not a reason not to report. If the release has to slip because of it, that is exactly what the process exists for.

You can add one sentence on why, as you understand it: capability evaluations are the input to the whole set of commitments, and if the input arrives late or gets suppressed, every safeguard downstream is late too.

**Q3.** Handle two things separately.

- **As an engineering problem**: the grading script has a bug, so of course it needs fixing, otherwise the evaluation is not measuring what you want it to measure.
- **As a safety signal**: the model learned to game the grader, which is in itself behavior worth recording and reporting; this is exactly the kind of phenomenon the alignment team cares about. Simply changing the reward so the behavior becomes "invisible" amounts to erasing the evidence, and the model may just have learned a better-hidden shortcut.

So the approach is: keep the checkpoint and logs that reproduce the behavior, write down clearly what happened, tell the alignment or evaluation team, and then fix the script; after the fix, specifically check whether the behavior is really gone or has just changed form.

**Q4.**

1. First decide which category it falls into: mainly advancing frontier capabilities, or mainly helping safety, evaluation, or interpretability. The former is not published by default.
2. Raise it through the company's process and let the people with the authority to judge decide; do not decide in advance on your own whether to publish.
3. If the conclusion is not to publish, accept it, and think about whether some part could be made public, such as safety-relevant findings within it.
4. What not to do: reveal details in any form outside the company, or become disengaged from this kind of work because you can't publish it.

In the interview you can say frankly that this has a cost for you, and then say why you accept it.

**Q5.**

- Comply first, then give feedback. Until your feedback is accepted, do not work around the controls yourself, even if you think they are unnecessary.
- Make your reasoning concrete: which requirement, for which kind of operation, how much delay it causes, and where you think the risk is already covered by other measures. Then go talk to the safety team, with the goal of finding an approach that is just as secure but costs less.
- Understand where they are coming from: stolen weights are an irreversible loss, and safeguards have to be designed for the worst case, not on the basis that "everyone on our team is trustworthy".

This question tests your first reaction when "a safety requirement slows me down": do you look for a way around it, or for a better way?

**Q6.** At least these questions:

1. **How realistic the misuse scenarios are.** Impersonating real people (politicians, intimate imagery of private individuals), fraud, fake news: the harm of each and how easily each can happen.
2. **What safeguards exist and whether they have been tested.** Restrictions on real people, refusals of sensitive content, red-teaming results, and how hard these safeguards are to bypass.
3. **Provenance labeling.** Whether generated videos carry a watermark or content credentials, whether those survive compression, cropping, and re-recording, and whether downstream platforms can detect them.
4. **Whether problems can be handled once found.** Whether a video can be traced back to the account that generated it, and whether a class of features can be taken down quickly.
5. **Who decides.** Who confirms the answers to these questions, rather than the team that wants to launch confirming them itself.

Then give your judgment: which questions must have answers before launch, and which can be iterated on after launch. This question is directly related to your own research area, so it is worth preparing a concrete version with technical detail.

**Q7.** You are not sure whether it is a violation, which is exactly what the "informal inquiry" route is for.

1. First ask the colleague to clarify: is this evaluation required by the RSP or something the team added itself, and did anyone approve "backfilling it later" and leave a record? Often the answer is that there is a legitimate arrangement you didn't know about.
2. If you can't get a clear answer, or the answer is "nobody approved it", ask the people responsible for evaluation or the Responsible Scaling Officer whether this counts as a violation.
3. If you have reason to worry that asking directly would have consequences, you can use the anonymous channel. This is exactly the situation the anti-retaliation policy protects.

Two things not to do: pretend you didn't hear it because the other person is senior, or accuse them of a violation outright before you have the facts.

**Q8.** There is no standard answer, but this is a question you must think through in advance, because a poor answer makes it look like you haven't thought about what you are doing. Arguments you can draw on (use only the ones you actually believe):

- **The company's reasoning.** Safety research has to be done on the most capable models; the company needs influence and revenue to stay at the frontier, and only then can it shape industry norms. So capabilities work is part of this bet, not a contradiction of the mission.
- **Your own reasons.** For example: how capabilities research is done directly determines whether models are easy to evaluate and easy to interpret; you want people who take the risks seriously to be among those doing this work; your expertise (say, multimodal or video) has concrete uses in safety evaluation.
- **The costs you acknowledge.** Your work really does make models more capable, and this bet could be wrong. Say what would convince you that you should stop doing it.

The weakest answer is "I just handle the technical side; risk is the safety team's job." That is the exact opposite of "none of us are bystanders".

</details>
