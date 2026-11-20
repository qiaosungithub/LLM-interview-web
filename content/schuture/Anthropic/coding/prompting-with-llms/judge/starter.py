# ---- the notebook ----
# `complete` sends one turn to the language model. The judge replaces it with a scripted fake that plays back fixed
# replies and checks each call, so call it by name from classify. In 自定义运行 you can do the same:
#     from ref import ScriptedComplete
#     ticket = "How do I invite a teammate to my workspace?"
#     examples = [{"text": "How do I change my workspace name?", "label": "how_to", "reason": "Asks how."}]
#     complete = ScriptedComplete(ticket, examples, ["no tags here", "<label>how_to</label><reason>Asks how.</reason>"])
#     print(classify(ticket, examples))
#     complete.finish()
def complete(system: str, prompt: str, temperature: float = 0.0) -> str:
    """Sends `prompt` under the `system` instructions and returns the model's text reply."""
    raise NotImplementedError("complete() is provided by the notebook")


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


# ---- Part 1 ----
SYSTEM_PROMPT = ""  # your system prompt; its wording is not judged


def parse_reply(reply: str) -> dict | None:
    """{"label": ..., "reason": ...} from the first <label> and <reason> tags, or None; see the statement."""
    raise NotImplementedError


def classify(ticket_text: str, examples: list[dict], temperature: float = 0.0) -> dict:
    """Calls complete() (and retries once if the reply does not parse); returns {"label", "reason", "retried"}."""
    raise NotImplementedError


# ---- Part 2 ----
# evaluate and compare are tested with a scripted fake classify (ref.FakeClassify), so call classify by name.
def score(tickets: list[dict], predictions: dict[str, str | None]) -> dict:
    """{"accuracy", "parse_failure_rate", "per_label"} computed from predictions alone."""
    raise NotImplementedError


def evaluate(tickets: list[dict], examples: list[dict], temperature: float = 0.0) -> dict:
    """score(...) of classify's predictions, plus "predictions" and "retry_count"."""
    raise NotImplementedError


def compare(tickets: list[dict], examples_a: list[dict], examples_b: list[dict],
            temperature: float = 0.0) -> dict:
    """{"a": evaluate with examples_a, "b": evaluate with examples_b, "changed": sorted ids whose labels differ}."""
    raise NotImplementedError


# ---- Part 3 ----
def paired_improvement(tickets: list[dict], result_a: dict, result_b: dict) -> dict:
    """{"a_only", "b_only", "p_value"}: the discordant counts and the exact two-sided sign test p-value."""
    raise NotImplementedError
