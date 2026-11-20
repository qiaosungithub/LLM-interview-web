# The Responsible Scaling Policy

## Problem

Read the notes first, then close them and answer the questions. You will almost certainly be asked what you think of the RSP in an interview, especially the changes in version 3.

**In one sentence.** The RSP (Responsible Scaling Policy) is Anthropic's public policy for managing catastrophic risk. At its core is a conditional: **once a model's capabilities cross a certain line, it must first be paired with the corresponding level of safeguards.**

**Safety levels (ASL).** Safeguards are divided into several AI Safety Levels, modeled on the biosafety levels used for handling dangerous pathogens. The higher the level, the stricter the required safeguards, which come in two kinds: deployment safeguards that prevent the model from being misused, and security safeguards that prevent the model weights from being stolen. In May 2025, Anthropic activated ASL-3 safeguards for the relevant models; the official wording is “We activated ASL-3 safeguards for relevant models in May 2025”, targeting chemical, biological, radiological and nuclear weapons risks. At the time, Anthropic described this as a precautionary, provisional measure: it had not determined whether Claude Opus 4 had actually crossed the capability threshold that requires ASL-3, only that it could no longer clearly rule this out.

**Versions.** Version 1.0, September 19, 2023; version 2.0, October 2024; version 3.0, February 24, 2026; the current version is 3.4, effective July 8, 2026.

**The structure of version 3.** Anthropic says there are now "two sets of measures": those it will carry out no matter what, and those it thinks the whole industry should adopt. Add the two new kinds of public documents and you can remember it as four parts:

| Part | Content | Hard commitment? |
| --- | --- | --- |
| Unilateral commitments | Measures Anthropic carries out regardless of what other companies do. The clearest one is maintaining at least ASL-3 safeguards for chemical and biological risks | Yes |
| Industry-wide recommendations | A "which capabilities call for which safeguards" mapping table, including higher-level security standards | No. The company says it will work to advance them, but does not commit to them unilaterally and unconditionally |
| Frontier Safety Roadmap | Public goals in four areas: security, alignment, misuse prevention, and policy | No. In Anthropic's own words: “Rather than being hard commitments, these are public goals that we will openly grade our progress towards.” |
| Risk Reports | Published every 3 to 6 months, assessing the risk of deployed models (overall risk and marginal risk) and explaining why development and deployment continue | Publishing the reports is itself a commitment; in certain cases an external third party must review them |

**Internal oversight arrangements** (as summarized by GovAI from the policy text): a Responsible Scaling Officer oversees implementation; employees can report possible violations through an anonymous channel; a third party annually checks compliance with the procedural commitments; unredacted Risk Reports are shared internally (version 3.0 required making them available to all employees with regular access; version 3.4 in July 2026 changed this to sharing with at least 200 employees). The RSP page also links a separate "RSP Noncompliance Reporting and Anti-Retaliation Policy"; a March 2026 update expanded the reporting channels and added a way for employees to ask informally whether something counts as a violation.

**The biggest controversy: version 3 removed the "pause commitment".**

- The old commitment was: if it cannot ensure safeguards are in place, it will not train or deploy models that could cause catastrophic harm. In other words, it would stop if necessary.
- Version 3 removed this. According to TIME, Anthropic said it would no longer abide by its commitment to pause scaling or delay deploying new models. GovAI's summary: the new policy starts from a lower level of safeguards and adopts stricter measures only if other companies do so too, or if Anthropic is clearly in the lead; it also commits to follow suit if a competitor's safeguards are more effective at a similar cost. A Risk Report can say that the industry's overall risk is unacceptable and still allow deployment.
- **Anthropic's two reasons.** First, the collective action problem: if it stops and others don't, it simply cedes the frontier, and with it the ability to do safety research and pursue its public-benefit mission. Second, the old policy created a perverse incentive: declaring that a model had reached a higher capability threshold would trigger expensive safeguards or even a pause, which made people reluctant to declare it.
- **The assessment by the research organization GovAI** (the article is signed by two researchers and notes that it represents the authors' own views, not the organization's). Their biggest concern is precisely the removal of the pause commitment, because it makes Anthropic more likely to deploy a model with unacceptable risk. Other concerns: some people will stop trusting the company's commitments as a result; other companies may loosen their own policies in turn; since the problem is one of collective action, Anthropic should push harder for regulation; transparency still depends on the company's self-reporting, which amounts to grading its own exam. But they also point out that Anthropic has not lowered any existing safeguard. Their initial reaction was fairly negative, and after a closer reading their overall view became more positive, on the grounds that being honest about the constraints is better than keeping commitments it won't actually honor; whether the change is good or bad in the end depends on implementation and on how other companies respond.
- The later version 3.1 added a sentence saying the company will “remain free to take measures such as pausing the development of our AI systems”. It keeps the freedom to pause, but no longer commits to pausing.

### Recall

**Q1.** State the RSP's core mechanism in one sentence. What is an ASL? When was ASL-3 activated, and for what kind of risk?

**Q2.** What are the four parts of version 3? Which are hard commitments and which are not?

**Q3.** What did version 3 remove? What are the two reasons Anthropic gave?

**Q4.** Name three of GovAI's concerns and its final overall judgment.

### Open-ended follow-ups you will be pressed on

**Q5.** The interviewer asks: "What do you think about us removing the pause commitment in version 3 of the RSP?" Give your own position, and explain why, holding that position, you still want to join.

**Q6.** Suppose you have joined and work on model training. Before a release, you discover that a capability evaluation was only partly run and was recorded as passed. Under the RSP's arrangements, what avenues do you have? In what order would you use them?

## Reference solution

<details>
<summary>Show the reference solution</summary>

**Sources and verification status (2026-11-17).** Checked against the originals: [the RSP page](https://www.anthropic.com/responsible-scaling-policy) (versions and dates, Risk Reports, the noncompliance reporting and anti-retaliation policy, the version 3.1 sentence about pausing); [the official announcement of version 3](https://www.anthropic.com/news/responsible-scaling-policy-v3) (the two sets of measures, the roadmap not being a hard commitment, Risk Reports every 3 to 6 months, external review, ASL-3 activated in May 2025); [GovAI's analysis](https://governance.ai/analysis/anthropics-rsp-v3-0-how-it-works-whats-changed-and-some-reflections) (the four-part structure, ASL-3 remaining a unilateral commitment while higher security levels are now only recommendations, the removal of the pause commitment, Anthropic's reasons, the six concerns, the overall judgment, the internal oversight arrangements).

That ASL-3 was a precautionary measure comes from [the announcement activating ASL-3](https://www.anthropic.com/news/activating-asl3-protections) (2025-05-22); the scope of internal sharing of unredacted Risk Reports follows the version 3.4 (2026-07-08) update notes on the RSP page.

Not directly verified: the original TIME article would not open, so only reports paraphrasing it were seen; the full PDF text of RSP version 3.4 was not read clause by clause, and the items "Responsible Scaling Officer, anonymous channel, annual third-party check" rely on GovAI's summary of the version 3.0 policy text.

**Q1.** Once capabilities cross a certain line, the corresponding level of safeguards must be in place before training or deployment can continue. ASL is the level of safeguards, modeled on biosafety levels, stricter as it goes higher, covering deployment safeguards against misuse and security safeguards against weight theft. ASL-3 was activated for the relevant models in May 2025, for chemical, biological, radiological and nuclear weapons risks; it was precautionary at the time: it had not been determined that the model had actually crossed the line, only that this could no longer be ruled out.

**Q2.**

1. Unilateral commitments: carried out whether or not others do, for example maintaining at least ASL-3 for chemical and biological risks. **Hard commitment.**
2. Industry-wide recommendations: a mapping table from capabilities to safeguards. **Not a hard commitment**; the company only says it will work to advance them.
3. Frontier Safety Roadmap: public goals, with the company publicly grading itself. **Not a hard commitment.**
4. Risk Reports: published every 3 to 6 months, assessing overall and marginal risk; reviewed by an external third party in certain cases.

**Q3.** It removed the commitment to "pause training or delay deployment if safeguards can't keep up". The two reasons: the collective action problem (stopping unilaterally only cedes the frontier and loses the ability to do safety research); the perverse incentive (declaring that a model has reached a higher capability threshold triggers expensive safeguards or a pause, so nobody wants to declare it).

**Q4.** Concerns (any three will do): it is more likely to deploy a model with unacceptable risk; outside trust in the company's commitments is damaged; other companies may loosen their policies in turn; it should push harder for regulation; it could have made more unconditional unilateral commitments; transparency depends on self-reporting. Overall judgment: initially fairly negative, more positive after a closer reading, holding that being honest about the constraints is better than keeping commitments it can't meet, but the outcome depends on implementation and on how peers respond. It also confirms that Anthropic has not lowered any existing safeguard.

**Q5.** There is no standard answer. The interviewer wants to see three things: that you know the facts, that you can lay out the strongest arguments on both sides, and that you have your own judgment. A structure you can use:

1. **State the facts first, without hedging.** "Version 3 turned 'pause if we can't do it' from a commitment into an option the company reserves, while keeping the existing ASL-3 safeguards and adding the roadmap and periodic Risk Reports."
2. **Make clear which part of the reasoning you accept.** For example: one company stopping on its own really doesn't reduce the world's total risk; a commitment left on paper that everyone knows won't be honored hurts credibility more than removing it.
3. **Make clear which part you're uneasy about.** For example: a commitment is most valuable exactly when honoring it is most costly, and this one was changed just as the cost was rising; the constraints now rest mainly on self-reporting; "waiting until others do it too" may mean waiting forever.
4. **Your conditions.** What you would look at to judge whether this change was good or bad: whether the Risk Reports state conclusions unfavorable to the company; whether the external review is genuinely independent; whether the company is at the same time pushing hard for binding regulation.
5. **Why you still want to join.** For example: you think these issues have a better chance of being pushed forward from inside the company than from outside; the company has openly acknowledged this tension instead of hiding it. Only say what you actually believe.

What not to do: recite the company's defense; or criticize only, to look critical, without being able to say why you still want to join.

**Q6.** The avenues are: go directly to the owner of this evaluation and to your own manager; go to the Responsible Scaling Officer; report through the anonymous channel; the "RSP Noncompliance Reporting and Anti-Retaliation Policy" protects people who raise concerns and allows an informal inquiry first.

As for order, the reasonable approach is to establish the facts first (was it really not completed, is there a substitute evaluation you don't know about) and raise it directly with the evaluation owner; if you get no explanation, or are told to drop it, escalate to the Responsible Scaling Officer or the anonymous channel. The key point is that **you can't look the other way because there's a release to ship**, and you also shouldn't quietly finish running the evaluation yourself and call it done: whether an evaluation is valid has to be confirmed by the person responsible, and a record has to be left.

When answering this kind of question, also show that you understand why it matters: in the RSP framework, capability evaluations are the input that decides whether to escalate safeguards; if an evaluation is skimped, every commitment downstream comes to nothing.

</details>
