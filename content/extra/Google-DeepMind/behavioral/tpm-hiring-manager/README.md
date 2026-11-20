# TPM Hiring-Manager Round: Technical Depth and Stakeholder Management

## Problem

For the Technical Program Manager (TPM) process at Google DeepMind, there is currently a one-sentence record: after the recruiter screen comes a hiring-manager round that tests technical depth and stakeholder management. Nobody wrote down what was actually asked. The two groups of questions below were written to fit that description, and every scenario is set in "a TPM working alongside ML research teams and engineering teams". How long this round lasts is not recorded either; practice it as 45 minutes, answering each question out loud in 2–3 minutes.

### Technical depth

- Pick an ML system you have program-managed (a training pipeline, an evaluation platform, model serving — any of them). Draw its structure on the whiteboard: where the data comes from, what steps it goes through, and who receives the result. Then point to where the real technical risk sits, and explain how you knew at the time that it was there.
- A multi-team model launch has an evaluation gate among its launch criteria: the new model has to beat the previous version on a set of evals before it can ship. The gate has now failed three times in a row, and each time the team said "give us two more weeks". You take over the program today. What do you do in the first two weeks? How do you tell whether this is a model problem, an eval problem, or a planning problem?
- The research team tells you: "This experiment needs another three to six weeks; it depends on the results." The product side's launch date is fixed. Tell me about a time you fit this kind of unpredictable research work into a fixed-date plan: how you scheduled it, what slack you left, and what "at that point we switch plans" conditions you set in advance.
- You are not the author of the system. Tell me about a time you found that what an engineer or researcher told you didn't hold up — an overly optimistic estimate, a missed dependency, or a metric computed differently from how everyone assumed. What made you suspicious, how did you verify it, and how did you raise it?
- Halfway through training a large model, someone discovers that part of an eval set has leaked into the training data (contamination). The planned launch date is four weeks away. Tell me how you would organize the technical assessment: which questions need answering, who answers each one, and which decisions should not be yours to make.
- Before launch, the model's inference cost is over budget. The engineering team proposes three paths: quantization, distillation into a smaller model, or capping output length. Choosing among them is not your call, but you have to put it in the plan. What do you need to understand, at minimum, about each path to estimate its impact on the date, the compute, and the evaluation gate? What questions would you ask the engineering team?

### Stakeholder management

- The research team wants a pool of accelerators (GPUs or TPUs) to run a set of large experiments, and the product team wants the same machines for pre-launch evaluation and fine-tuning. Both sides say theirs is more urgent, and you have no authority to allocate compute. How do you drive a decision?
- One researcher thinks the model is ready to ship; an engineering lead thinks it is not — his concerns are serving stability and whether you can roll back if something goes wrong. Both are more senior than you. How do you handle the disagreement?
- You have confirmed the program will be three weeks late, and just last week your manager reported "on track" to leadership. When do you say it, to whom first, and how? Give me the first three sentences of your message, word for word.
- A director asks to add one more feature to an already-agreed scope, with no change to the date. You judge that it can't be done. Tell me about a time you said "no" to someone much more senior than you; if you have never done that, tell me how you would say it this time.
- The eval team supports five programs at once, and yours is fourth in line. You need them to add a new set of evals within two weeks. You have no management authority over them, and this work isn't in their goals for the quarter. What do you do?
- The colleague responsible for the pre-launch safety review raises an issue that would delay the launch if it were handled properly. The product lead thinks it is an overreaction and wants you to "find a way to get it through". What do you do?

## Reference solution

<details>
<summary>Show the reference solution</summary>

**How much evidence there is.** The only basis for this page is one sentence on the [process page](#/deepmind/process): the TPM process "reportedly is a short recruiter screen, then a hiring-manager round on technical depth and stakeholder management", followed by "evidence for program roles is thin". Not a single actual question was recorded. So all twelve questions above were written by us, and the real interview may not ask them. How long this round lasts and whether there are other rounds afterwards is not recorded either, and we don't guess here. The only things we know for sure are the two areas being tested, so the preparation below is organized around them.

**How to use this page.** For each question, first answer it out loud yourself (2–3 minutes), then read the matching section. The people, projects and numbers in the sample answers are made up and are only there to show the structure; in the interview you must replace them with your own real experiences. On hypothetical questions, do not invent how this company works internally — you don't know how compute is allocated here or who approves launches — saying "I would first find out how that gets decided here" is enough.

**How this page divides the work with two others.** The [team-lead rounds](#/deepmind/behavioral/team-lead-rounds) ask about research experience and research taste, and the [PM case round](#/deepmind/behavioral/pm-ai-product-sense) asks about product judgment. This page only practices the two things specific to TPMs, and its questions don't overlap with those two pages.

### Technical depth

**What is being tested.** The hiring manager doesn't expect a TPM to write code, or to make judgment calls in place of researchers. They want to know four things:

1. Can you explain the system "one layer below the slides": how data flows, which step is slow, which step tends to break, which part belongs to which team.
2. Can you point to where the risk really is and give a basis for it, rather than reciting every possible risk.
3. Can you turn technical uncertainty into things in a plan: buffers, checkpoints, fallbacks, decision criteria agreed in advance.
4. Do you know where your own understanding ends. At that boundary, will you say "I don't know this part; I'd go ask so-and-so".

**Answer skeleton (about 2 minutes).**

1. One sentence: what the system does, who uses it, roughly how big it is (only state the scale you actually know).
2. Three to five boxes: input → processing → output, marking the handoffs between teams.
3. Where the risk is, and what your basis is. The four common bases: it slipped here before; there is no fallback here; this step waits on another team; the measurement itself is unstable here.
4. What you did to shrink the risk or surface it earlier.
5. The outcome, plus one sentence on "one thing I would change now".

**Hints, question by question.**

- **Draw the structure, point to the risk.** Use the skeleton above. The usual follow-up is "Why not that other spot?", so prepare the comparison: why the other spot is less dangerous.
- **The evaluation gate failed three times.** Don't push yet. Lay out the record of all three attempts: what changed each time and how far short of the bar it was. Then ask one question: "If you evaluate the same model twice, how much does the score vary?" If the noise is larger than the gap to the bar, the problem is the eval; if it falls short by a steady margin every time, the problem is the model; if several things were changed at once each time and nobody can say which one helped, the problem is the plan. Then find out who has the authority to change the bar (not you), agree with everyone on a decision date, and on the options available on that day: launch with reduced scope, delay, or stop. The two answers that lose the most points are "I'd push the team to pass as fast as possible" and "I'd suggest lowering the bar".
- **Fitting research estimates into a fixed date.** Track "research milestones" (we have figured something out) and "delivery milestones" (we have built something) separately. Schedule work on the critical path at the pessimistic end of the estimate. Time-box experiments and write the decision criteria in advance, e.g. "if the metric hasn't reached X by week four, switch to the old approach we know works". Meanwhile, keep the fallback moving forward cheaply. When reporting up, give a range and a confidence level, not just a single date.
- **Finding a claim that doesn't hold up.** The reason for suspicion should be concrete: how long the last three similar tasks actually took; who is missing from the dependency list; what the metric's numerator and denominator are and which dataset it uses. Ways to verify: ask the person to walk you through it; ask a second person to look; run a small check. How to raise it: privately first, with data, framed as a question rather than an accusation.
- **Data contamination.** There are three kinds of questions to answer. Scope: which data, how much of it, which checkpoints are affected. Impact: can the eval results still be trusted — scores on a contaminated eval cannot be used to pass the gate. Remediation: retraining from a clean checkpoint, filtering the data and continuing training, switching to a new eval set — how much compute and how many weeks each takes. Who answers: the data and training owners investigate the scope; the eval owner and the research lead judge whether the evals are still valid; the product lead and the program's decision-maker decide the trade-off on the date. Your job is: establish the facts within a few days, write the options and their costs on one page, make sure someone makes the decision, and record it. What you should not do is decide that "a little bit leaking in doesn't matter".
- **Three paths to cut cost.** You don't need to know how to do them, but you need to know what each one changes. Ask the same four questions about each path: will the model's outputs change (if so, it has to pass the evaluation gate again); does it require retraining (if so, count it in weeks, and it needs compute); can it be reverted afterwards; who signs off on it. The general picture is this: quantization lowers the numerical precision of the weights; the common approach (post-training quantization) needs no retraining and is quick, but quality may drop, so it needs re-evaluation. Distillation trains a small model to imitate the large one; it needs a new training run and a full round of evaluation, so it takes the longest. Capping output length barely touches the model, but it changes product behavior and needs the product lead's agreement. Whether that holds for this particular model is for the engineering team to answer. Then put the three paths' timelines, compute, whether they need re-evaluation, and whether they can be reverted into one table and hand it to whoever has the authority to choose. Two answers lose points: picking one yourself; or only asking "how many weeks?" without asking whether the evals need to be redone.

**The most common ways to lose points.**

- Talking only about process ("I ran a weekly sync and maintained a risk register") without being able to give a single concrete detail about the system.
- Lots of jargon, but no answer when pushed one layer deeper; making things up where you don't know. Saying directly "I'm not clear on that layer; so-and-so owned it at the time, and what I did was…" actually earns points.
- A long list of risks, unranked, with no idea which one actually happened.
- Every delay was someone else's fault.
- Making technical decisions in place of the experts; or the opposite, describing yourself as someone who only relays messages and writes meeting notes.
- Disclosing a former employer's confidential details. Scale and structure are enough; don't cite internal codenames or unpublished numbers.

**Sample answer (question 1, about 90 seconds; the content is made up).**

> "I ran the launch program for a text-to-video model that involved three teams: training, evaluation, and inference serving. The structure was: the training team delivered a candidate checkpoint every two weeks; the eval team ran two kinds of evals on it, automatic metrics and human side-by-side comparisons; once it passed the bar, the serving team did acceleration and deployment.
>
> On the surface the biggest risk was training, because it was the most expensive. But I went through the records of the previous two versions, and both delays came from human evaluation: one round took five working days; and if you evaluated the same model twice, the win rate could differ by three percentage points, while the bar only required beating the previous version by two points. In other words, the noise in the eval was larger than the gap we were trying to measure, so passing or failing was partly luck.
>
> I did three things. First, I asked the eval lead to increase the sample size for each round and report confidence intervals instead of a single number. Second, I moved human evaluation from the last step to running a small-sample version on every candidate first, so problems surfaced two weeks earlier. Third, I agreed in advance with the three leads: by week six, if the interval still straddled the bar, we would launch with the reduced-scope plan.
>
> In the end we were a week later than the original plan, but we never again found out right before launch that we wouldn't pass. Looking back, I should have asked 'how much does this eval vary if we run it again?' at kickoff, not after the second delay."

Where this answer scores: the risk it identifies runs against intuition and is backed by records; all three actions target that risk; it doesn't decide for the eval team what the sample size should be; and it ends with an honest retrospective.

### Stakeholder management

**What is being tested.**

1. When the decision isn't yours, can you get it made: who decides, by what criteria, by when.
2. Do you deliver bad news early, and with options.
3. Facing senior people, can you disagree with facts and alternatives; and once they have decided, do you follow through.
4. Do you represent each side's position fairly. Hearing your description, both the researcher and the engineering lead should say "yes, that's exactly what I mean".
5. Do you avoid surprising people: no escalating behind the people involved, no agreeing to their face and then not doing it.

**Answer skeleton.** For hypothetical questions, use these five steps:

1. First get the facts, and what each side really needs: what they need, by what date, and the cost of not getting it.
2. Find out who has the authority to decide and by what criteria. If there are no established criteria, first push to get them set.
3. Write out the options, expressing each option's cost in the same units (weeks, compute, level of risk).
4. Have the two parties talk directly first. If they can't agree, escalate together, with a description of the facts both sides accept.
5. After the decision: record it, communicate it, follow up. What you fix afterwards is the rule, not blame for a person.

For experience questions, use this order: situation (two sentences) → my judgment at the time → what I did → outcome (with numbers if possible) → which practice I changed afterwards.

**Hints, question by question.**

- **Compute conflict.** Don't open with "split it fifty-fifty". Check the facts first: each side's deadline, the cost of being a week late, whether the work can be split up or run at different times. Then find out who allocates compute here and what the organization's current priorities are. Then propose a concrete schedule and ask both sides to amend it. If they can't agree, write a one-pager and go with both leads to whoever has allocation authority. Afterwards, propose a standing rule so the same conflict doesn't need escalating next time.
- **The researcher and the engineering lead disagree.** They are really answering two different questions: "Is the model good enough?" and "If something goes wrong, can we handle it?" Write out the launch criteria one by one, and ask each of them: "What evidence would change your mind?" Then look for a middle path, e.g. launch to a small audience first and rehearse the rollback beforehand. In the end the person with launch authority decides; you don't vote.
- **Reporting a delay upward.** Say it the same day you confirm the delay is real, without waiting until you have a fix. Tell your direct manager first: he reported "on track" just last week, and he can't hear it from someone else. The message covers, in order: what changed; the new date and your confidence in it; the reason in one sentence; the options, and the decision you need from him.
- **Saying "no" to a director.** Don't say a bare "no". Explain that scope, date, quality and staffing can't all stay fixed, give two or three options with their costs, recommend one, and ask them to choose. If they still decide to force it in, do it, while writing down the risks and agreeing on checkpoints.
- **Asking the eval team for help.** First find out what the three programs ahead of yours are and why they rank higher. Shrink the request to the minimum; do the preparation you can do yourself (data, spec) up front; think about what you can offer them, e.g. lending a person, or building the new evals so other programs can use them too. Discuss the trade-offs openly with their lead, without going around him. If it really can't be fit in, go together with him to the person who owns both sets of priorities.
- **The safety review raises an issue.** The review is not an obstacle to get around, and a TPM has no authority to waive it. First make the concern concrete: what could happen, how likely it is, how severe the consequences are, and what evidence would put it to rest. Get the product lead and the reviewer together to talk, and look for mitigations that can be finished by the original date. If a delay really is needed, say so early, the way the "reporting a delay upward" question describes. "I'll find a way to get it through" is the worst-scoring answer to this question.

**The most common ways to lose points.**

- Describing yourself as the referee ("in the end I decided…"), or as the messenger ("I reported the problem to my boss"). A TPM's contribution sits between the two: getting the person who should decide to make the decision, with clear options in hand.
- All process, no outcome; or an outcome that is just "everyone was happy".
- Speaking badly of other teams or former colleagues.
- Holding bad news until you have a fix.
- Saying "no" to a manager without alternatives; or agreeing to their face and then not doing it.
- Inventing this company's internal processes in hypothetical questions.

**Sample answer (question 3, about 90 seconds; the content is made up).**

> "I'd say it the same day I confirmed the delay was real, and tell my direct manager first. He reported 'on track' upward just last week, and he can't hear it from somewhere else.
>
> The first three sentences of the message: 'The video model launch date needs to move from March 10 to around March 31, and I'm 80% confident it won't slip further. The reason is that the evaluation gate failed for the third time, and finding the problem plus re-evaluating takes three weeks. I've prepared two options and would like to go through them with you this afternoon: one is a full launch on the new date, the other is shipping the parts that don't depend on the new model on March 10.'
>
> Then I'd attach a one-page note: the results of the three eval rounds, how the new date was estimated, and the two things that could push it later still. The message doesn't say whose fault it was.
>
> Once before, I waited four days because I wanted to have the options worked out before reporting. My manager ended up hearing it first in another meeting, which put him in a very awkward position. Since then my practice has been: send the three sentences the same day, and follow up with the options the next day."

Where this answer scores: it says first who gets told and why him; the first three sentences contain the new date, the confidence level, the reason, and what you need from the other person; and it volunteers a time you got it wrong and what you changed afterwards.

### Questions worth asking the hiring manager

- Does this role mainly support research programs, product launches, or the handoff between the two?
- Who decides that a model is "ready to ship"? What does the TPM do in that decision?
- When teams compete for compute or eval resources, how does that get resolved today?
- What separates the best people in this role from the average ones?
- What is the one thing you'd most want me to accomplish in my first three months?

These questions also fill in the gaps on this page: the "I'd first find out" parts of the hypothetical questions above are exactly what you ask here.

### Prep outline

Copy the outline below into your own notes and fill in each item with real experiences.

```text
Two systems (prepare one diagram each)
  System A: one-sentence purpose / 3–5 boxes / team handoffs / the riskiest spot and why / what I did / outcome
  System B: same as above

Four experiences (2 minutes each)
  1. Fitting unpredictable research work into a fixed date
  2. Finding a problem with a technical claim
  3. Reporting bad news upward
  4. Saying "no" to someone more senior
  Each one: situation in two sentences → my judgment → what I did → outcome (numbers) → what I changed afterwards

Things I can't say
  A former employer's confidential information: list it first, and work out how to replace it with scale and structure
```

</details>
