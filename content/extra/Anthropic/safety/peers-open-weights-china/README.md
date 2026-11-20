# Peers, Open Weights and Chinese Labs

## Problem

Read the notes first, then close them and answer the questions. Every position in this topic is contested. In the interview you don't need to defend the company, and you don't need to disagree with it to look independent; what you need is to know what the company actually said and then give your own view. Quoted English is the original wording, checked on 2026-11-17.

**On peers: a race to the top on safety.** This is one of the company's seven values. The original: “We work to inspire a ‘race to the top’ dynamic where AI developers must compete to develop the most safe and secure AI systems.” The idea: Anthropic builds a safety practice first and publishes it, so that other companies find it awkward not to follow in front of the public, customers and regulators.

- Evidence usually cited: after Anthropic published its RSP in 2023, OpenAI came out with its Preparedness Framework and Google DeepMind with its Frontier Safety Framework, both with the same structure of "add safeguards once capabilities cross a threshold". In August 2025, Anthropic and OpenAI also ran their own internal misalignment evaluations on each other's models and each published the results.
- Reasons for doubt: peers may have followed because of regulatory and public pressure, not necessarily because Anthropic led them; and in 2026 version 3 of the RSP dropped the pause commitment precisely on the grounds that "others did not follow", which suggests the race to the top did not work on at least the hardest item.

**"We Must Pace the Frontier" (Dario Amodei, September 2026).** Argues that the whole frontier should slow down: “We must slow the pace at which we improve the capabilities of AI models.” But it also makes clear that “pacing does not mean halting model training or technical progress.” Concrete proposals include: third-party evaluators embedded inside the companies with “ongoing, employee-like access”; some kind of "speed limit" on the pace of recursive self-improvement; common safety standards adopted by frontier companies in democracies; and continued export controls on advanced chips.

**On open weights.** On July 27, 2026, Dario published a post explaining the position. The key points:

1. **No call for a ban.** “Anthropic has never advocated for a ban on open-weights models.”
2. **Open models without dangerous capabilities are a good thing.** “Open-weights models that don't have dangerous capabilities are a public good.”
3. **Regulate capability, not openness.** “All sufficiently capable models, open and closed, should go through mandatory safety testing.”
4. **But once you reach dangerous capabilities, open weights are harder to govern.** Three reasons:
   - **One fewer layer of protection.** A closed model has two layers: the model itself is trained to refuse dangerous requests, and the server side can also monitor usage and ban accounts. Once the weights are public, the second layer is gone, and the first can be fine-tuned away. The original: “it is very difficult to apply guardrails to them or monitor their usage”.
   - **Irreversible.** “once weights are released they cannot be withdrawn”. A closed model can be taken down or patched when a problem is found.
   - **In biology, offense has the advantage.** “I worry that biology will have a strong attacker-defender asymmetry.” So in his view the argument that "open source helps defenders too" does not apply to biology.

The post was a response to a July 24, 2026 open letter in support of open weights. Dario says he agrees with much of the letter (“I agree with much of it”); what he disagrees with is the claim that open source necessarily helps safety and favors defenders. According to media reports, Anthropic did not sign that letter.

**On Chinese labs.**

- **The distillation disclosure** (February 23, 2026). Anthropic disclosed that three labs, DeepSeek, Moonshot and MiniMax, used roughly 24,000 fraudulent accounts to carry out more than 16 million exchanges with Claude in order to train their own models. The company's reasoning is not only commercial: “Illicitly distilled models lack necessary safeguards, creating significant national security risks.” The September 2026 threat intelligence report added that since February it had detected and blocked distillation attacks from seven more Chinese labs.
- **Policy positions** (from the open-weights post above): “We should crack down on industrial-scale distillation operations.” and “We should not sell powerful chips or chipmaking equipment to China”.

**The criticism you can't avoid: conflict of interest.** Anthropic sells access to closed models. Treating open weights more strictly, restricting chip exports to China, cracking down on distillation: every one of these also serves its commercial interests. That doesn't make these positions wrong, but it does mean the company is not a neutral referee.

### Recall

**Q1.** What does "race to the top" mean? Give one piece of evidence that it works and one reason to doubt it.

**Q2.** Summarize Anthropic's position on open weights in four sentences.

**Q3.** Why does Anthropic think open weights are harder to govern than closed models once dangerous capabilities are involved? Give three reasons.

**Q4.** What did the February 2026 distillation disclosure say? What does "We Must Pace the Frontier" argue for, and what does it not argue for?

### Open questions you will be pushed on

**Q5.** The interviewer asks: "We sell closed models, and our positions on open weights and export controls happen to benefit us. How do you think about that conflict of interest?"

**Q6.** The interviewer asks: "You come from a research community used to publishing papers and open-sourcing code. Here, a lot of capabilities-related work doesn't get published. How do you feel about that?"

**Q7.** The interviewer asks: "What do you think of our position on Chinese AI labs?"

## Reference solution

<details>
<summary>Show the reference solution</summary>

**Sources and verification status (2026-11-17).** Checked against the originals: the [company page](https://www.anthropic.com/company) (the race-to-the-top wording); the [position on open-weights models](https://www.anthropic.com/news/position-open-weights-models) (2026-07-27, all the open-weights and export-control quotes in this topic); the [distillation disclosure](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks) (2026-02-23, three labs, about 24,000 accounts, more than 16 million exchanges); [We Must Pace the Frontier](https://darioamodei.com/post/we-must-pace-the-frontier) (the quotes and the four proposals); the [report on the Anthropic–OpenAI cross-evaluation](https://alignment.anthropic.com/2025/openai-findings/) (only the title and abstract were checked; the publication date 2025-08-27 and which models each side tested come from search results).

Places where the conversation transcript disagrees with the sources, or that could not be verified:

- The transcript says the pacing post was from "October 2026". It was actually September 2026, titled "We Must Pace the Frontier".
- The transcript describes other companies' reactions to the post. Different sources contradict each other, so the notes leave this out.
- "Anthropic did not sign the open letter" comes from media reports; Dario's post itself does not say so.
- The transcript mentions "seven more labs disclosed in September 2026". Anthropic's September 2026 [threat intelligence report](https://www.anthropic.com/threat-intelligence-report-september-2026) does say this (“additional distillation attacks against Claude from seven labs based in China”); the notes only give the number.
- That the OpenAI and Google DeepMind frameworks "followed the RSP": the chronology is accurate, but the causal link is Anthropic's own claim.
- "Two layers of protection" is a summary of the post's argument, not its wording.

**Q1.** It means: build a safety practice yourself first and publish it, so that peers have to follow, steering the competition from "who is faster" toward "who is safer". For: after the RSP, both OpenAI and Google DeepMind published frameworks with a similar structure; in 2025 the two companies tested each other's models. Against: following may have come from regulatory and public pressure; nobody followed on the hardest item (pause if you can't meet the bar), and Anthropic itself dropped it in version 3.

**Q2.** It does not call for banning open weights; open models without dangerous capabilities are a public good; sufficiently capable models, open or closed, should undergo mandatory safety testing; once you reach dangerous capabilities, open weights are harder to govern than closed models, so testing first matters even more.

**Q3.** One fewer layer of protection (no server-side monitoring or account bans, and the model's built-in refusals can be fine-tuned away); irreversible (once the weights are out they can't be pulled back, while a closed model can be taken down or patched); in biology offense has the advantage, so open source does little to help defenders.

**Q4.** The distillation disclosure: three labs, DeepSeek, Moonshot and MiniMax, used about 24,000 fraudulent accounts to carry out more than 16 million exchanges with Claude to train their own models; Anthropic argues that models distilled this way lack the corresponding safeguards, which makes it a security problem and not only a commercial one. "We Must Pace the Frontier" argues that the whole frontier should slow the pace of capability gains, through embedded third-party evaluation, a speed limit on recursive self-improvement, common standards for frontier companies in democracies, and chip export controls; it does not argue for stopping training or halting technical progress.

**Q5.** There is no standard answer. A structure that holds up:

1. **Acknowledge the conflict directly.** These positions really do benefit the company, so the company's statements can't be treated as neutral evidence.
2. **Separate the argument from the motive.** Whether a position is right depends on the argument itself. Go through it point by point: "weights can't be recalled once released" is a fact; "closed models have an extra layer of protection" is also a fact, though how reliable that layer is can be questioned; "offense has the advantage in biology" is an empirical judgment that can be argued.
3. **Point to practices that reduce the conflict of interest.** For example, Dario's post requires closed models to undergo the same mandatory testing; and an independent third party, not the company itself, could decide what counts as a "dangerous capability".
4. **State your own reservations.** For example: who sets the threshold and where; if the answer is "the closed-model companies", the conflict of interest is not resolved. Open source has real value for safety research, for academia and for preventing concentration of power, and that has to be counted too.

Don't say "I fully agree with the company's position", and don't just say "it's all commercial".

**Q6.** There is no standard answer, but be honest, because you will face this every day once you join.

- First get the facts right: what Anthropic doesn't publish is work that advances frontier capabilities, because it doesn't want to speed up capability progress across the industry; it publishes a great deal of research on safety, interpretability and evaluations.
- Then give your real view. Possible angles: which part of the rationale you agree with; what you would give up (academic visibility, peer feedback, reuse by the open-source community); and how you plan to handle that cost, for example by putting your energy into publishable safety-related directions, or by accepting that your impact will show up in other ways.
- If you have reservations, say so, and explain under what conditions you could accept it. What the interviewer fears more is finding out after you join that you can't accept it.

**Q7.** There is no standard answer. It helps to discuss three things separately, because they are different in kind:

- **Distillation.** Scraping outputs at scale through fraudulent accounts violates the terms of service; that part is not very controversial. What is controversial is framing it as a national security issue, and naming only Chinese companies (public information isn't enough to tell whether others do the same).
- **Export controls.** This is a geopolitical position. The argument for it is "we don't want unaccountable actors to get the most powerful capabilities first"; arguments against include that it may accelerate the other side's domestic development and reduce channels of communication on safety, while global risks ultimately require all the major players to abide by some set of rules.
- **People.** A judgment about a company's or a government's policy is not a judgment about the researchers there. You can say directly that you know and respect peers there, or that international safety cooperation is a direction you think is undervalued.

Give your real view. What the interviewer wants to see is whether you can lay out the facts and both sides of a question where positions conflict, without performing. If this position makes you uncomfortable, saying so is better than glossing over it.

</details>
