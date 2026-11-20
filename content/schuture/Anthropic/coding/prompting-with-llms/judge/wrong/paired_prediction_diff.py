# expect: WA
# Wrong: treats every ticket where the predictions differ as discordant and credits b whenever a is wrong, even when b is wrong too.
import re

from scipy.stats import binomtest


def complete(system: str, prompt: str, temperature: float = 0.0) -> str:
    """Provided by the notebook; the judge replaces it with a scripted stand-in."""
    raise NotImplementedError("complete() is provided by the notebook")


LABELS = ["billing", "bug", "how_to", "access"]

SYSTEM_PROMPT = (
    "You triage incoming support tickets into exactly one of four categories.\n\n"
    "billing: about a charge, invoice, payment method, refund, or subscription price.\n"
    "bug: the product is not behaving as documented -- an error, a crash, or a control that stops working.\n"
    "how_to: the sender wants to know how to do something the product already supports.\n"
    "access: the sender cannot get into their account -- login, password, a verification code, or account ownership.\n\n"
    "When a ticket touches more than one category, label it by the action support must take right now, "
    "not by any secondary cause mentioned in passing.\n\n"
    "Reply with exactly one <label>...</label> tag containing one of billing, bug, how_to, access, "
    "followed by one <reason>...</reason> tag with a one-sentence justification. Do not use any other "
    "tag and do not repeat the ticket text."
)


def render_example(ex: dict) -> str:
    return f"Ticket: {ex['text']}\n<label>{ex['label']}</label>\n<reason>{ex['reason']}</reason>"


def build_user_prompt(ticket_text: str, examples: list[dict]) -> str:
    blocks = [render_example(ex) for ex in examples] + [f"Ticket: {ticket_text}"]
    return "\n\n".join(blocks)


LABEL_TAG = re.compile(r"<label>\s*(.*?)\s*</label>", re.S)
REASON_TAG = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.S)


def parse_reply(reply: str) -> dict | None:
    m = LABEL_TAG.search(reply)                 # NOTE: search, not match -- tolerates text around the tags
    if not m:
        return None
    label = m.group(1).strip().lower()
    if label not in LABELS:                     # NOTE: validated against the label set, not just "looks like a tag"
        return None
    r = REASON_TAG.search(reply)
    return {"label": label, "reason": r.group(1).strip() if r else ""}


def _parse_complaint(reply: str) -> str:
    return ("Your reply could not be parsed: no valid <label> tag was found, or its content was not "
            "one of " + ", ".join(LABELS) + ".")


def classify(ticket_text: str, examples: list[dict], temperature: float = 0.0) -> dict:
    prompt = build_user_prompt(ticket_text, examples)
    reply = complete(SYSTEM_PROMPT, prompt, temperature)
    parsed = parse_reply(reply)
    retried = False
    if parsed is None:                           # NOTE: exactly one retry, with the failure fed back verbatim
        retried = True
        retry_prompt = (f"{prompt}\n\nYour previous reply was:\n{reply}\n\n{_parse_complaint(reply)} "
                         "Reply again, using exactly the tag format above and nothing else.")
        parsed = parse_reply(complete(SYSTEM_PROMPT, retry_prompt, temperature))
    if parsed is None:
        return {"label": None, "reason": "", "retried": retried}
    return {"label": parsed["label"], "reason": parsed["reason"], "retried": retried}


def score(tickets: list[dict], predictions: dict) -> dict:
    n = len(tickets)
    correct = sum(1 for t in tickets if predictions.get(t["id"]) == t["label"])
    per_label = {}
    for lab in LABELS:
        tp = sum(1 for t in tickets if t["label"] == lab and predictions.get(t["id"]) == lab)
        pred_n = sum(1 for t in tickets if predictions.get(t["id"]) == lab)
        true_n = sum(1 for t in tickets if t["label"] == lab)
        per_label[lab] = {"precision": tp / pred_n if pred_n else 0.0,   # NOTE: 0.0, not a division error
                           "recall": tp / true_n if true_n else 0.0}
    return {
        "accuracy": correct / n,
        "parse_failure_rate": sum(1 for t in tickets if predictions.get(t["id"]) is None) / n,
        "per_label": per_label,
    }


def evaluate(tickets: list[dict], examples: list[dict], temperature: float = 0.0) -> dict:
    predictions, retry_count = {}, 0
    for t in tickets:
        out = classify(t["text"], examples, temperature)
        predictions[t["id"]] = out["label"]
        retry_count += out["retried"]
    result = score(tickets, predictions)
    result["predictions"], result["retry_count"] = predictions, retry_count
    return result


def compare(tickets: list[dict], examples_a: list[dict], examples_b: list[dict],
            temperature: float = 0.0) -> dict:
    a = evaluate(tickets, examples_a, temperature)
    b = evaluate(tickets, examples_b, temperature)
    changed = sorted(t["id"] for t in tickets if a["predictions"][t["id"]] != b["predictions"][t["id"]])
    return {"a": a, "b": b, "changed": changed}


EXEMPLAR_A = {"text": "I was charged for the Pro plan but I'm still on the Free plan.",
              "label": "billing", "reason": "A charge does not match the plan the sender is actually on."}
EXEMPLAR_B = {"text": "How do I change my workspace name?",
              "label": "how_to", "reason": "Asks how to do something the product already supports."}
EXEMPLARS_V1 = [EXEMPLAR_A, EXEMPLAR_B]


EXEMPLAR_C = {"text": "The 'Share' button is disabled and I can't invite anyone to the document.",
              "label": "bug",
              "reason": "A control that should work is disabled; that is broken behavior, not a request for instructions."}
EXEMPLAR_D = {"text": "I can't sign in because the system says my trial expired and my workspace is frozen.",
              "label": "access",
              "reason": "The action needed right now is regaining entry to the account, even though the cause is billing-related."}
EXEMPLARS_V2 = EXEMPLARS_V1 + [EXEMPLAR_C, EXEMPLAR_D]


def paired_improvement(tickets: list[dict], result_a: dict, result_b: dict) -> dict:
    a_only = b_only = 0
    for t in tickets:
        if result_a["predictions"][t["id"]] != result_b["predictions"][t["id"]]:
            if result_a["predictions"][t["id"]] == t["label"]:
                a_only += 1
            else:
                b_only += 1
    n = a_only + b_only
    p_value = 1.0 if n == 0 else binomtest(min(a_only, b_only), n, 0.5, alternative="two-sided").pvalue
    return {"a_only": a_only, "b_only": b_only, "p_value": p_value}
