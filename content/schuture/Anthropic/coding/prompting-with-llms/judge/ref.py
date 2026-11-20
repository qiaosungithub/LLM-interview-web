"""Helpers for the ticket-triage judge: the statement's data, an independent parse_reply, score and exact sign test,
random replies, and two scripted stand-ins that never call a model:

- ScriptedComplete replaces complete() for one classify() call (Part 1);
- FakeClassify replaces classify() for evaluate() and compare() (Part 2).

In 自定义运行, for example:
    from ref import ScriptedComplete
    ticket = "How do I invite a teammate to my workspace?"
    examples = [{"text": "How do I change my workspace name?", "label": "how_to", "reason": "Asks how."}]
    complete = ScriptedComplete(ticket, examples, ["no tags here", "<label>how_to</label><reason>Asks how.</reason>"])
    print(classify(ticket, examples))
    complete.finish()
"""
import copy
import html
import json
import math
from fractions import Fraction

from judgelib import Mismatch

# The statement's data, verbatim.
LABELS = ["billing", "bug", "how_to", "access"]

TICKETS = [
    {"id": "t01", "text": "You charged me twice for my March subscription, can you refund the extra payment?", "label": "billing"},
    {"id": "t02", "text": "My invoice shows a $49 charge but my plan should only cost $29 a month.", "label": "billing"},
    {"id": "t03", "text": "I cancelled my subscription last week but was still billed today.", "label": "billing"},
    {"id": "t04", "text": "Can I get a receipt for tax purposes for my last three payments?", "label": "billing"},
    {"id": "t05", "text": "The annual plan discount wasn't applied at checkout, I paid full price.", "label": "billing"},
    {"id": "t06", "text": "I want to downgrade my plan and get a prorated refund for the unused days.", "label": "billing"},
    {"id": "t07", "text": "My credit card was charged in a currency I did not select at signup.", "label": "billing"},
    {"id": "t08", "text": "The export as CSV option is grayed out and I can't click it, how do I enable it?", "label": "bug"},
    {"id": "t09", "text": "Every time I upload a PNG larger than 2MB the page just goes blank.", "label": "bug"},
    {"id": "t10", "text": "Search results are missing items that were there yesterday, nothing was deleted.", "label": "bug"},
    {"id": "t11", "text": "The dashboard chart shows negative numbers for a metric that can never be negative.", "label": "bug"},
    {"id": "t12", "text": "Dark mode resets to light mode every time I refresh the page.", "label": "bug"},
    {"id": "t13", "text": "Saving a draft sometimes duplicates it into two identical drafts.", "label": "bug"},
    {"id": "t14", "text": "The mobile app crashes immediately after I tap the notifications bell.", "label": "bug"},
    {"id": "t15", "text": "How do I invite a teammate to my workspace?", "label": "how_to"},
    {"id": "t16", "text": "Is there a way to schedule a report to send every Monday morning?", "label": "how_to"},
    {"id": "t17", "text": "Where do I change the default currency shown on invoices?", "label": "how_to"},
    {"id": "t18", "text": "How can I export my project data as a CSV file?", "label": "how_to"},
    {"id": "t19", "text": "What's the difference between the viewer and editor roles?", "label": "how_to"},
    {"id": "t20", "text": "How do I set up a keyboard shortcut for creating a new task?", "label": "how_to"},
    {"id": "t21", "text": "Can you point me to where I turn off email notifications for comments?", "label": "how_to"},
    {"id": "t22", "text": "I can't log in because it says my payment failed and my account is suspended until I update my card.", "label": "access"},
    {"id": "t23", "text": "My account got locked after too many failed login attempts, please unlock it.", "label": "access"},
    {"id": "t24", "text": "Two-factor authentication won't accept the code from my authenticator app.", "label": "access"},
    {"id": "t25", "text": "I'm logged out every few minutes even though I checked 'stay signed in'.", "label": "access"},
    {"id": "t26", "text": "I forgot my password and the reset email never arrives.", "label": "access"},
    {"id": "t27", "text": "Single sign-on redirects me back to the login page without any error.", "label": "access"},
    {"id": "t28", "text": "My teammate left the company and I need to be added as the new account owner.", "label": "access"},
]

# Worked examples for the tests, none of them one of TICKETS. The first four are the reference's EXEMPLAR_A..D.
EXAMPLES = [
    {"text": "I was charged for the Pro plan but I'm still on the Free plan.", "label": "billing",
     "reason": "A charge does not match the plan the sender is actually on."},
    {"text": "How do I change my workspace name?", "label": "how_to",
     "reason": "Asks how to do something the product already supports."},
    {"text": "The 'Share' button is disabled and I can't invite anyone to the document.", "label": "bug",
     "reason": "A control that should work is disabled; that is broken behavior, not a request for instructions."},
    {"text": "I can't sign in because the system says my trial expired and my workspace is frozen.", "label": "access",
     "reason": "The action needed right now is regaining entry to the account, even though the cause is "
               "billing-related."},
    {"text": "The app shows error 500 whenever I open the calendar view.", "label": "bug",
     "reason": "A feature that should work fails with an error."},
    {"text": "I changed phones and the login code never reaches my new number.", "label": "access",
     "reason": "The sender cannot sign in without the verification code."},
    {"text": "Please refund the seat I removed from my team plan last month.", "label": "billing",
     "reason": "Asks for a refund on the subscription."},
    {"text": "Where do I turn on two-step verification for my team?", "label": "how_to",
     "reason": "Asks where to find a setting the product already has."},
]


def _brief(x, limit=300):
    try:
        text = repr(x)
    except Exception:       # noqa: BLE001
        text = f"<{type(x).__name__}>"
    return text if len(text) <= limit else text[:limit] + " …"


# ---------------------------------------------------------------- Part 1

def _first_tag(text, name):
    """The text between the first <name> and the first </name> after it, or None."""
    start, end = f"<{name}>", f"</{name}>"
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(end, i + len(start))
    return None if j < 0 else text[i + len(start):j]


def parse_reply(reply):
    """The statement's parse_reply, written with str.find. (Test replies never put whitespace around a reason.)"""
    content = _first_tag(reply, "label")
    if content is None or content.strip().lower() not in LABELS:
        return None
    reason = _first_tag(reply, "reason")
    return {"label": content.strip().lower(), "reason": "" if reason is None else reason.strip()}


def classify_outcome(replies):
    """(what classify returns, how many times it calls complete) when the model's replies are `replies`."""
    first = parse_reply(replies[0])
    if first is not None:
        return {"label": first["label"], "reason": first["reason"], "retried": False}, 1
    second = parse_reply(replies[1])
    if second is None:
        return {"label": None, "reason": "", "retried": True}, 2
    return {"label": second["label"], "reason": second["reason"], "retried": True}, 2


class ScriptedComplete:
    """A fake complete() for one classify(ticket, examples, temperature) call, with the statement's signature.

    It returns `replies` in order. If replies[0] parses, classify must call it exactly once; otherwise exactly twice,
    the second time showing replies[0]. On the first call the ticket text and every example's text and reason must be
    shown, and `temperature` must be classify's. "Shown" means inside system or prompt (the statement lets either hold
    any part), as is or quoted with repr(), json.dumps() or html.escape(). The first problem raises Mismatch, and
    finish() raises it again.
    """

    def __init__(self, ticket, examples, replies, temperature=0.0):
        self.ticket = ticket
        self.examples = copy.deepcopy(examples)
        self.replies = list(replies)
        self.temperature = temperature
        self.expected_calls = classify_outcome(self.replies)[1]
        self.calls = []
        self.error = None

    def __call__(self, system, prompt, temperature=0.0):
        self.calls.append((system, prompt, temperature))
        k = len(self.calls)
        if self.error is None:
            problem = self._check(k, system, prompt, temperature)
            if problem:
                self.error = Mismatch(f"第 {k} 次调用 complete 时：{problem}", show=False)
        if self.error is not None:
            raise self.error
        return self.replies[k - 1]

    def finish(self):
        """After classify returned: no problem was seen, and complete was called as often as it should be."""
        if self.error is not None:
            raise self.error
        if not self.calls:
            raise Mismatch("classify 没有调用 complete 就返回了", show=False)
        if len(self.calls) < self.expected_calls:
            raise Mismatch(f"第一次回复 {self.replies[0]!r} 无法被 parse_reply 解析，classify 应当再调用一次 complete "
                           f"重试，实际只调用了 1 次", show=False)

    def _check(self, k, system, prompt, temperature):
        if k > self.expected_calls:
            if self.expected_calls == 1:
                return (f"第一次回复 {self.replies[0]!r} 已经能被 parse_reply 解析（缺少 <reason> 也算解析成功），"
                        f"classify 不应再调用 complete")
            return "第一次回复无法解析时只能重试一次：complete 最多调用 2 次，两次都解析失败就返回 label 为 None"
        if not isinstance(system, str) or not isinstance(prompt, str):
            return f"system 和 prompt 都应当是 str，实际是 {type(system).__name__} 和 {type(prompt).__name__}"
        shown = system + "\n" + prompt
        if k == 1:
            if not _shown(self.ticket, shown):
                return f"system 或 prompt 里应当包含工单原文 {self.ticket!r}"
            for ex in self.examples:
                if not _shown(ex["text"], shown):
                    return (f"传给 classify 的示例 {ex['text']!r} 没有出现在 system 或 prompt 里："
                            f"examples 参数里的每个示例都应当展示给模型")
                if not _shown(ex["reason"], shown):
                    return (f"示例 {ex['text']!r} 的 reason {ex['reason']!r} 没有出现在 system 或 prompt 里："
                            f"示例应当用同样的标签格式（<label> 和 <reason>）展示给模型")
            if temperature != self.temperature:
                return f"temperature 应当是传给 classify 的 {self.temperature!r}，实际是 {temperature!r}"
        if k == 2 and not _shown(self.replies[0], shown):
            return f"重试时的 system 或 prompt 应当包含第一次没能解析的回复原文 {self.replies[0]!r}"
        return None


def _shown(text, where):
    """`text` appears in `where` as is, or quoted the way code commonly embeds a string in a prompt."""
    return any(form in where for form in (text, repr(text), json.dumps(text), json.dumps(text, ensure_ascii=False),
                                          html.escape(text), html.escape(text, quote=False)))


# ---------------------------------------------------------------- Part 2

def score(tickets, predictions):
    """The statement's score; every test has at least one ticket of each label."""
    got = [predictions.get(t["id"]) for t in tickets]
    true = [t["label"] for t in tickets]
    n = len(tickets)
    per_label = {}
    for lab in LABELS:
        tp = sum(1 for g, y in zip(got, true) if g == lab and y == lab)
        predicted = got.count(lab)
        per_label[lab] = {"precision": tp / predicted if predicted else 0.0, "recall": tp / true.count(lab)}
    return {"accuracy": sum(1 for g, y in zip(got, true) if g == y) / n,
            "parse_failure_rate": got.count(None) / n,
            "per_label": per_label}


def evaluate_result(tickets, outcomes):
    """What evaluate returns when classify gives outcomes[ticket id] = (label, retried)."""
    predictions = {t["id"]: outcomes[t["id"]][0] for t in tickets}
    result = score(tickets, predictions)
    result["predictions"] = predictions
    result["retry_count"] = sum(1 for t in tickets if outcomes[t["id"]][1])
    return result


def changed(tickets, outcomes_a, outcomes_b):
    return sorted(t["id"] for t in tickets if outcomes_a[t["id"]][0] != outcomes_b[t["id"]][0])


class FakeClassify:
    """A fake classify() for evaluate() and compare(), with the statement's signature.

    `runs` lists (examples, outcomes): called with those examples, it returns outcomes[ticket id] = (label, retried)
    for the ticket whose text it is given, as {"label", "reason", "retried"}. Every call must pass one ticket's text,
    one run's examples and `temperature`; each ticket must be classified exactly once per run. The first problem
    raises Mismatch, and finish() raises it again.
    """

    def __init__(self, tickets, runs, temperature=0.0):
        self.ids = {t["text"]: t["id"] for t in tickets}
        self.runs = [(copy.deepcopy(ex), dict(out)) for ex, out in runs]
        for _, out in self.runs:
            assert all(retried for label, retried in out.values() if label is None)
        self.temperature = temperature
        self.seen = [set() for _ in self.runs]
        self.error = None

    def __call__(self, ticket_text, examples, temperature=0.0):
        if self.error is None:
            problem, found = self._check(ticket_text, examples, temperature)
            if problem:
                self.error = Mismatch(f"调用 classify 时：{problem}", show=False)
        if self.error is not None:
            raise self.error
        k, tid = found
        self.seen[k].add(tid)
        label, retried = self.runs[k][1][tid]
        return {"label": label, "reason": "" if label is None else f"scripted reason for {label}", "retried": retried}

    def finish(self):
        """After evaluate or compare returned: no problem was seen, and every ticket was classified in every run."""
        if self.error is not None:
            raise self.error
        for k, (examples, out) in enumerate(self.runs):
            missing = [tid for tid in out if tid not in self.seen[k]]
            if missing:
                which = "" if len(self.runs) == 1 else f"用第 {k + 1} 组示例时，"
                raise Mismatch(f"{which}工单 {missing} 没有被 classify：每条工单都应当用 classify 分类一次", show=False)

    def _check(self, ticket_text, examples, temperature):
        if isinstance(ticket_text, dict):
            return "第一个参数应当是 ticket['text']，实际传了整个工单字典", None
        if not isinstance(ticket_text, str) or ticket_text not in self.ids:
            return f"第一个参数 {_brief(ticket_text)} 不是 tickets 里任何一条工单的 text", None
        k = next((i for i, (ex, _) in enumerate(self.runs) if ex == examples), None)
        if k is None:
            return f"examples 应当原样传入，实际收到 {_brief(examples)}", None
        if temperature != self.temperature:
            return f"temperature 应当原样传入（{self.temperature!r}），实际是 {temperature!r}", None
        tid = self.ids[ticket_text]
        if tid in self.seen[k]:
            return f"工单 {tid} 被 classify 了不止一次：每条工单应当恰好分类一次", None
        return None, (k, tid)


# ---------------------------------------------------------------- Part 3

def sign_test_p(a_only, b_only):
    """Exact two-sided sign test at p = 0.5: min(1, 2 * P(X <= min(a, b))), X ~ Binomial(a + b, 0.5)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = Fraction(sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)), 2 ** n)
    return float(min(Fraction(1), 2 * tail))


def paired(tickets, predictions_a, predictions_b):
    a_only = sum(1 for t in tickets if predictions_a[t["id"]] == t["label"] != predictions_b[t["id"]])
    b_only = sum(1 for t in tickets if predictions_b[t["id"]] == t["label"] != predictions_a[t["id"]])
    return {"a_only": a_only, "b_only": b_only, "p_value": sign_test_p(a_only, b_only)}


# ---------------------------------------------------------------- random replies

PREFIXES = ["", "Sure! ", "Here is my answer:\n", "Okay.\n\n", "Based on the ticket, ", "Label below. "]
SUFFIXES = ["", " Thanks!", "\nHope this helps.", "\n", " Let me know if you need more."]
INVALID = ["spam", "how to", "how-to", "billing.", "", "bugs", "access denied", "refund", "label", "bug/billing"]
REASONS = ["The sender was charged twice.", "A control that should work is disabled.",
           "Asks how to use an existing feature.", "The sender cannot log in.",
           "Two problems:\nthe page crashes and the draft is lost.", "Locked out\nafter a password reset.", "ok"]
NO_TAG = ["label: bug\nreason: it crashes", "This is a billing question.", "**access**", "Category = how_to",
          "I am not sure which category fits."]


def _label_tag(r, valid):
    if valid:
        content = r.choice(LABELS)
        content = r.choice([content, content.upper(), content.title(),
                            "".join(c.upper() if r.random() < 0.5 else c for c in content)])
    else:
        content = r.choice(INVALID)
    pads = ["", "", " ", "  ", "\n", " \n", "\t"]
    return f"<label>{r.choice(pads)}{content}{r.choice(pads)}</label>"


def random_reply(r, p_valid=0.6):
    """A reply with text around the tags, odd case and spacing, maybe a second label or reason tag, maybe no tag."""
    if r.random() < 0.15:
        return r.choice(NO_TAG)
    parts = [_label_tag(r, r.random() < p_valid)]
    if r.random() < 0.7:
        reason = f"<reason>{r.choice(REASONS)}</reason>"
        if r.random() < 0.2:
            parts.insert(0, reason)
        else:
            parts.append(reason)
    if r.random() < 0.2:
        parts.append(_label_tag(r, r.random() < 0.5))
    if r.random() < 0.15:
        parts.append(f"<reason>{r.choice(REASONS)}</reason>")
    return r.choice(PREFIXES) + r.choice(["", "\n", " ", "\n\n"]).join(parts) + r.choice(SUFFIXES)
