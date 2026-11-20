# Mission, the Founding Bet and Governance

## Problem

Read the notes below first, then close them and answer the questions. All quoted English is the official wording, checked against the official pages on 2026-11-17.

**Mission.** Claude's constitution says “our mission is to ensure that the world safely makes the transition through transformative AI”. The company page also carries a purpose in the legal sense: “the responsible development and maintenance of advanced AI for the long-term benefit of humanity”. In an interview, the first sentence is enough.

**The founding bet.** Anthropic believes AI may be one of the most dangerous technologies in human history, yet it builds AI itself. The constitution admits this directly — “Anthropic occupies a peculiar position in the AI landscape” — and calls it “a calculated bet on our part”: powerful AI is coming regardless, so it is better to have safety-focused labs at the frontier than to “cede that ground” to developers less focused on safety. Everything Anthropic does has to be understood starting from this bet, and it is exactly this premise that its critics attack.

**The 2023 “Core Views on AI Safety”** spells the bet out in more detail:

- The company does not bet on any one future. It says “a wide range of scenarios are plausible” and prepares for three:
  - **Optimistic scenario**: there is very little chance that safety failures lead to catastrophe. Anthropic's job is to bring beneficial applications sooner.
  - **Intermediate scenario**: catastrophe is a possible or even likely outcome. Anthropic's main contribution is to identify the risks and find ways to guard against them.
  - **Pessimistic scenario**: the safety problem is essentially unsolvable. Anthropic's role is to provide as much evidence as possible that current safety techniques cannot prevent serious risks — in other words, to sound the alarm.
- Why it has to build frontier models: safety research has to be done on frontier systems, because large models are qualitatively different from small ones, and many serious safety problems may only show up in systems close to human level.
- The matching restraint: research that would push frontier capabilities is generally not published. In the original: “We generally don't publish this kind of work because we do not wish to advance the rate of AI capabilities progress.”

**Governance: binding itself with legal structure.**

- **Public Benefit Corporation (PBC).** Anthropic is a Delaware Public Benefit Corporation. Directors of an ordinary corporation answer for the shareholders' financial interests; directors of a PBC are legally allowed to weigh the shareholders' financial interests together with the public-benefit purpose written into the company's charter.
- **Long-Term Benefit Trust (LTBT).** Announced in September 2023. At announcement it consisted of five trustees with no financial interest in the company (the membership has changed since; as of October 2026 there are four: chair Neil Buddy Shah, Richard Fontaine, Ben Bernanke and Christopher Stubbs), with backgrounds in AI safety, national security, public policy and social enterprise. The Trust exclusively holds a special class of stock (Class T), which lets it elect and remove some of the directors; as stated at the time, the Trust “will elect a majority of the board within 4 years”.
- **On April 14, 2026**, the Trust appointed Novartis CEO Vas Narasimhan to the board. The official announcement says: “With Narasimhan's appointment, Trust-appointed directors now make up a majority of the Board.” The directors it lists are Dario Amodei, Daniela Amodei, Yasmin Razavi, Jay Kreps, Reed Hastings, Chris Liddell and Vas Narasimhan. The company page now lists only six directors; Jay Kreps is no longer on the list. The official pages do not say whether Trust-appointed directors still hold a majority after his departure.
- **Its weaknesses, which the company itself wrote down.** The 2023 announcement calls the Trust “an experiment” and includes a “failsafe” provision: with the agreement of a large enough supermajority of stockholders, the Trust and its powers can be amended without the trustees' consent. So whether the Trust can hold up under pressure at a critical moment is something outsiders can only judge from how it actually exercises its powers.

The problem this structure is meant to solve: insulating safety commitments from short-term capital pressure. Neither OpenAI nor Google has the same combination of "public benefit corporation + an independent trust that elects directors" (OpenAI has a different design: after its October 2025 restructuring, its for-profit arm also became a public benefit corporation, OpenAI Group PBC, but it is controlled by the nonprofit OpenAI Foundation, which appoints its directors).

### Recall

**Q1.** In two or three sentences, state Anthropic's mission and why it "warns about the dangers of AI while building frontier AI itself."

**Q2.** What are the optimistic, intermediate and pessimistic scenarios in "Core Views on AI Safety"? What does Anthropic say its role is in each? What publication-related restraint does it adopt as a result?

**Q3.** How does a public benefit corporation differ from an ordinary corporation? What is the Long-Term Benefit Trust, and what powers does it hold? What happened in April 2026?

### Open-ended follow-ups you will be pressed on

**Q4.** The interviewer asks: "Can this governance structure really constrain the company?" Give the strongest argument for and the strongest argument for skepticism, then give your own judgment.

**Q5.** The interviewer asks: "Some people think Anthropic's bet is wrong from the start: by building frontier models, it is accelerating the race itself. What is the strongest version of this criticism? How do you respond?"

## Reference solution

<details>
<summary>Show the reference solution</summary>

**Sources and verification status (2026-11-17).** Checked against the originals: [Claude's constitution](https://www.anthropic.com/constitution) (mission, “peculiar position”, “calculated bet”); [the company page](https://www.anthropic.com/company) (company purpose, board elected by stockholders and the Trust); [Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety) (2023-03-08; the three scenarios, the case for frontier research, not publishing capabilities research); [The Long-Term Benefit Trust](https://www.anthropic.com/news/the-long-term-benefit-trust) (2023-09-19; five trustees, Class T, electing a board majority within four years, “experiment”, “failsafe”); [the announcement of Narasimhan joining the board](https://www.anthropic.com/news/narasimhan-board) (2026-04-14). The current list of trustees comes from a footnote on the Trust page (as of October 2026); the current list of directors comes from the company page.

Not verified: "Google has no similar structure" is a claim from the conversation transcript and was not checked this time; OpenAI's October 2025 restructuring is based on news reports, and OpenAI's own pages were not opened. The official pages do not say when Jay Kreps left the board or whether Chris Liddell was appointed by the Trust, so it is impossible to tell whether Trust-appointed directors still hold a majority.

**Q1.** The mission is to ensure that the world safely makes the transition through transformative AI. Building frontier AI itself is a deliberate bet: powerful AI will be built anyway, and it is better to have safety-focused labs at the frontier than to cede it to people who do not prioritize safety; on top of that, safety research itself can only be done on the most capable models.

When you answer, make two points: this is a **bet** (the company's own word is “calculated bet”), not a conclusion that has been proven right; and the company admits its position is contradictory (“peculiar position”) rather than pretending it isn't.

**Q2.**

- Optimistic: very little chance that safety failures cause catastrophe → speed up the arrival of beneficial applications.
- Intermediate: catastrophe is possible or even likely → identify the risks and find ways to guard against them; this is the scenario where the company thinks it can contribute most.
- Pessimistic: the safety problem is essentially unsolvable → provide evidence that current techniques cannot prevent the risks, sound the alarm, and push the world not to build or deploy such systems.

The company does not commit to which scenario we are in; it says it will keep gathering evidence to judge, and it invests in research directions as a "portfolio". The matching restraint: research that pushes frontier capabilities is generally not published, because it does not want to speed up capability progress across the industry.

**Q3.**

- Directors of a public benefit corporation may legally weigh the shareholders' financial interests together with the public-benefit purpose in the charter; directors of an ordinary corporation answer mainly for the shareholders' financial returns. The public-benefit purpose in Anthropic's charter is "the responsible development and maintenance of advanced AI for the long-term benefit of humanity".
- The Long-Term Benefit Trust is an independent body of trustees with no financial interest in the company (five when announced in 2023, four as of October 2026). It exclusively holds Class T stock, can elect and remove some of the directors, and is scheduled to grow its share to a board majority over time.
- On April 14, 2026, the Trust appointed Vas Narasimhan as a director, and the official announcement said Trust-appointed directors now make up a majority of the board. Jay Kreps has since dropped off the list of directors, and the company has not said whether that is still a majority.

**Q4.** There is no standard answer; the interviewer is checking whether you can argue both sides properly and then give your own judgment. Material you can use:

- For: this arrangement is written into the company's charter and share structure, not just a policy document; in April 2026 the company announced that Trust-appointed directors held a majority (the board list has changed since); the trustees have no financial interest, so their incentives differ from investors'.
- Skeptical: there is the "failsafe" provision, so enough stockholders can bypass the trustees and change the rules; the company itself calls the Trust an "experiment"; so far, nobody outside has seen the Trust overrule management on a major disagreement, so the structure has not been stress-tested; version 3 of the RSP loosened commitments under competitive pressure (see the "Responsible Scaling Policy" question), which shows that voluntary commitments really do get changed.
- A defensible judgment: the structure raises the cost of "the company putting safety behind profit", but cannot guarantee it won't happen; what you should really watch is what outsiders can verify, such as whether the Trust has ever exercised its powers, the quality of the risk reports, and the conclusions of third-party reviews.

Avoid two kinds of answers: repeating the company website's praise; or brushing it off with "it's all PR."

**Q5.** The strongest version of the criticism is roughly:

1. Anthropic's existence adds one more strong competitor at the frontier, which makes the other companies more hurried and pushes overall progress faster;
2. The argument "you have to stay at the frontier to influence the rules" can be used without limit; any acceleration can be justified with it;
3. The evidence is already there: when RSP version 3 dropped the pause commitment, the company's own stated reason was that competitors had not followed suit and stopping unilaterally would only hand over the frontier.

Arguments you can use in response (pick the ones you actually believe):

- The counterfactual: without Anthropic, the frontier would not stop; there would just be one fewer company publicly releasing safety frameworks, risk reports and documents on model values. Practices such as conditional scaling policies and interpretability research really have been adopted or followed by peers.
- Acknowledge the cost: the bet could be wrong, and whether it is right has to be judged by testable outcomes, for example whether peers follow the safety practices, and whether the company has paid a real price at critical moments (the Pentagon contract dispute is one example).
- State your conditions: what facts would make you conclude the bet has been lost. Being able to name that condition is more convincing than simply defending the company.

</details>
