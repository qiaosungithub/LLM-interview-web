"""Tests for the ticket-triage problem. The system prompt, the examples' wording and the written parts are not judged.

classify() runs against ref.ScriptedComplete, a fake complete() that plays back fixed replies and checks each call;
evaluate() and compare() run against ref.FakeClassify, a fake classify() that returns fixed predictions. Each failure
shows a replay that can be pasted into 自定义运行."""
import copy
import math
import numbers

from judgelib import calling, case, expect_valid, rng, show_call
import ref

LABELS = ref.LABELS
T = {t["id"]: t for t in ref.TICKETS}
EX = ref.EXAMPLES

TRACE_TICKET = "Can I get a receipt for tax purposes for my last three payments?"
TRACE_PROSE = "This is a billing question -- they want a receipt for past payments."
TRACE_TAGGED = "<label>billing</label>\n<reason>Requests a receipt for past payments.</reason>"


def keys_of(x):
    return sorted(map(str, x)) if isinstance(x, dict) else type(x).__name__


def num_ok(x, want, rtol=1e-9):
    return (isinstance(x, numbers.Real) and not isinstance(x, bool)
            and math.isclose(x, want, rel_tol=rtol, abs_tol=1e-12))


def with_fake(sol, name, fake, fn):
    """Runs fn() with sol.<name> replaced by `fake`, then fake.finish(); puts the answer's own function back."""
    saved = getattr(sol, name, None)
    setattr(sol, name, fake)
    try:
        got = fn()
        fake.finish()
    finally:
        setattr(sol, name, saved)
    return got


# ---------------------------------------------------------------- Part 1 helpers

def parse_problem(got, want, why):
    if want is None:
        why = why or "：第一个完整的 <label>...</label> 标签不存在，或其内容去空白、不分大小写后不是四个标签之一"
        return None if got is None else f"应当返回 None{why}"
    why = why or "（label 取第一个 <label> 标签的内容去空白、转小写，reason 取第一个 <reason> 标签的内容，没有就是 ''）"
    if not isinstance(got, dict):
        return f"应当返回 dict，实际是 {type(got).__name__}{why}"
    if set(got) != {"label", "reason"}:
        return f"返回的 dict 应当恰好有 'label' 和 'reason' 两个键，实际是 {keys_of(got)}"
    if got["label"] != want["label"]:
        return f"label 应当是 {want['label']!r}{why}"
    if got["reason"] != want["reason"]:
        return f"reason 应当是 {want['reason']!r}{why}"
    return None


def check_parse(sol, reply, want=..., why=""):
    if want is ...:
        want = ref.parse_reply(reply)
    call = calling(show_call("parse_reply", reply))
    got = sol.parse_reply(reply)
    expect_valid(got, lambda g: parse_problem(g, want, why), call=call, want=want)


def classify_problem(got, want):
    if not isinstance(got, dict):
        return f"classify 应当返回 dict，实际是 {type(got).__name__}"
    if set(got) != {"label", "reason", "retried"}:
        return f"返回的 dict 应当恰好有 'label'、'reason'、'retried' 三个键，实际是 {keys_of(got)}"
    if got["label"] != want["label"]:
        if want["label"] is None:
            return "两次回复都无法解析，label 应当是 None"
        return f"label 应当是 {want['label']!r}（{'第二次' if want['retried'] else '第一次'}回复解析出的标签）"
    if got["reason"] != want["reason"]:
        if want["label"] is None:
            return "label 为 None 时 reason 应当是空串 ''"
        return f"reason 应当是 {want['reason']!r}（解析成功的那次回复里第一个 <reason> 的内容，没有就是 ''）"
    if got["retried"] != want["retried"]:
        if want["retried"]:
            return "retried 应当是 True：第一次回复无法解析，classify 调用了第二次 complete"
        return "retried 应当是 False：第一次回复就解析成功，没有第二次调用"
    return None


def run_classify(sol, ticket, examples, replies, temperature=None, want=None):
    """classify(ticket, examples[, temperature]) against a fake complete() that replies with `replies` in order."""
    outcome, n = ref.classify_outcome(replies)
    replies = list(replies[:n])
    want = outcome if want is None else want
    t_kw = "" if temperature is None else f", temperature={temperature!r}"
    t_arg = "" if temperature is None else f", {temperature!r}"
    call = calling("\n".join([
        "from ref import ScriptedComplete",
        f"ticket = {ticket!r}",
        f"examples = {examples!r}",
        f"complete = ScriptedComplete(ticket, examples, {replies!r}{t_kw})",
        f"print(classify(ticket, examples{t_arg}))",
        "complete.finish()",
    ]))
    fake = ref.ScriptedComplete(ticket, examples, replies, 0.0 if temperature is None else temperature)
    args = (ticket, copy.deepcopy(examples)) + (() if temperature is None else (temperature,))
    got = with_fake(sol, "complete", fake, lambda: sol.classify(*args))
    expect_valid(got, lambda g: classify_problem(g, want), call=call, want=want)


# ---------------------------------------------------------------- Part 2 helpers

def result_problem(got, want, skip=(), hint="", where=""):
    """A score() or evaluate() result: the dicts exactly, the numbers within 1e-9."""
    if not isinstance(got, dict):
        return f"{where}应当是 dict，实际是 {type(got).__name__}"
    if set(got) != set(want):
        return f"{where}的键应当是 {sorted(want)}，实际是 {keys_of(got)}"
    if "predictions" in want and got["predictions"] != want["predictions"]:
        return (f"{where}predictions 应当是 ticket id -> classify 返回的 label 的 dict："
                f"期望 {want['predictions']!r}，得到 {got['predictions']!r}")
    for k in ("accuracy", "parse_failure_rate"):
        if k not in skip and not num_ok(got[k], want[k]):
            return f"{where}{k} 不对：期望 {want[k]!r}，得到 {got[k]!r}{hint}"
    per = got["per_label"]
    if not isinstance(per, dict) or set(per) != set(LABELS):
        return f"{where}per_label 的键应当正好是四个标签 {LABELS}，实际是 {keys_of(per)}"
    for lab in LABELS:
        v = per[lab]
        if not isinstance(v, dict) or set(v) != {"precision", "recall"}:
            return f"{where}per_label[{lab!r}] 应当是只有 'precision' 和 'recall' 两个键的 dict，实际是 {v!r}"
        for m in ("precision", "recall"):
            w = want["per_label"][lab][m]
            if not num_ok(v[m], w):
                return f"{where}per_label[{lab!r}][{m!r}] 不对：期望 {w!r}，得到 {v[m]!r}{hint}"
    if "retry_count" in want and got["retry_count"] != want["retry_count"]:
        return (f"{where}retry_count 应当是 classify 返回 retried 为 True 的工单数 {want['retry_count']}，"
                f"得到 {got['retry_count']!r}{hint}")
    return None


def check_score(sol, tickets, predictions, want=None, skip=(), hint=""):
    want = ref.score(tickets, predictions) if want is None else want
    call = calling(show_call("score", tickets, predictions))
    got = sol.score(copy.deepcopy(tickets), dict(predictions))
    expect_valid(got, lambda g: result_problem(g, want, skip, hint), call=call, want=want)


def tickets_src(tickets):
    """Source for `tickets`: TICKETS, or an index expression into it, or a literal; and whether TICKETS is used."""
    if all(t in ref.TICKETS for t in tickets):
        idx = tuple(ref.TICKETS.index(t) for t in tickets)
        if idx == tuple(range(len(ref.TICKETS))):
            return "TICKETS", True
        return f"[TICKETS[i] for i in {idx!r}]", True
    return repr(tickets), False


def run_evaluate(sol, tickets, examples, outcomes, temperature=None, hint=""):
    """evaluate(tickets, examples[, temperature]) against a fake classify() that returns outcomes[id]."""
    src, uses = tickets_src(tickets)
    t_kw = "" if temperature is None else f", temperature={temperature!r}"
    t_arg = "" if temperature is None else f", {temperature!r}"
    call = calling("\n".join([
        f"from ref import {'TICKETS, ' if uses else ''}FakeClassify",
        f"tickets = {src}",
        f"examples = {examples!r}",
        f"classify = FakeClassify(tickets, [(examples, {outcomes!r})]{t_kw})",
        f"print(evaluate(tickets, examples{t_arg}))",
        "classify.finish()",
    ]))
    fake = ref.FakeClassify(tickets, [(examples, outcomes)], 0.0 if temperature is None else temperature)
    want = ref.evaluate_result(tickets, outcomes)
    args = (copy.deepcopy(tickets), copy.deepcopy(examples)) + (() if temperature is None else (temperature,))
    got = with_fake(sol, "classify", fake, lambda: sol.evaluate(*args))
    expect_valid(got, lambda g: result_problem(g, want, (), hint), call=call, want=want)


def compare_problem(got, want, hint):
    if not isinstance(got, dict):
        return f"compare 应当返回 dict，实际是 {type(got).__name__}"
    if set(got) != {"a", "b", "changed"}:
        return f"返回的 dict 应当恰好有 'a'、'b'、'changed' 三个键，实际是 {keys_of(got)}"
    for side in ("a", "b"):
        problem = result_problem(got[side], want[side], where=f"{side}（用 examples_{side} 的 evaluate 结果）")
        if problem:
            return problem
    ch = got["changed"]
    if not isinstance(ch, (list, tuple)):
        return f"changed 应当是 list，实际是 {type(ch).__name__}"
    if list(ch) != want["changed"]:
        try:
            same_set = sorted(ch) == want["changed"]
        except TypeError:
            same_set = False
        if same_set:
            return "changed 应当按 id 排序（sorted）"
        return f"changed 应当是两边预测的 label 不同的工单 id：期望 {want['changed']!r}，得到 {list(ch)!r}{hint}"
    return None


def run_compare(sol, tickets, examples_a, examples_b, out_a, out_b, temperature=None, hint=""):
    """compare(tickets, examples_a, examples_b[, temperature]) against a fake classify()."""
    src, uses = tickets_src(tickets)
    t_kw = "" if temperature is None else f", temperature={temperature!r}"
    t_arg = "" if temperature is None else f", {temperature!r}"
    call = calling("\n".join([
        f"from ref import {'TICKETS, ' if uses else ''}FakeClassify",
        f"tickets = {src}",
        f"examples_a = {examples_a!r}",
        f"examples_b = {examples_b!r}",
        f"outcomes_a = {out_a!r}",
        f"outcomes_b = {out_b!r}",
        f"classify = FakeClassify(tickets, [(examples_a, outcomes_a), (examples_b, outcomes_b)]{t_kw})",
        f"print(compare(tickets, examples_a, examples_b{t_arg}))",
        "classify.finish()",
    ]))
    fake = ref.FakeClassify(tickets, [(examples_a, out_a), (examples_b, out_b)],
                            0.0 if temperature is None else temperature)
    want = {"a": ref.evaluate_result(tickets, out_a), "b": ref.evaluate_result(tickets, out_b),
            "changed": ref.changed(tickets, out_a, out_b)}
    args = (copy.deepcopy(tickets), copy.deepcopy(examples_a), copy.deepcopy(examples_b)) + \
        (() if temperature is None else (temperature,))
    got = with_fake(sol, "classify", fake, lambda: sol.compare(*args))
    expect_valid(got, lambda g: compare_problem(g, want, hint), call=call, want=want)


def labeled(labels, prefix):
    return [{"id": f"{prefix}{i + 1}", "label": lab} for i, lab in enumerate(labels)]


def random_tickets(r, lo, hi):
    """Some of TICKETS, at least one of each label, in random order."""
    chosen = [r.choice([t for t in ref.TICKETS if t["label"] == lab]) for lab in LABELS]
    chosen += r.sample([t for t in ref.TICKETS if t not in chosen], r.randint(lo, hi) - len(chosen))
    r.shuffle(chosen)
    return chosen


def random_outcomes(r, tickets):
    out = {}
    for t in tickets:
        if r.random() < 0.15:
            out[t["id"]] = (None, True)
        else:
            label = t["label"] if r.random() < 0.65 else r.choice(LABELS)
            out[t["id"]] = (label, r.random() < 0.25)
    return out


# ---------------------------------------------------------------- Part 3 helpers

def paired_problem(got, want, hint):
    if not isinstance(got, dict):
        return f"应当返回 dict，实际是 {type(got).__name__}"
    if set(got) != {"a_only", "b_only", "p_value"}:
        return f"返回的 dict 应当恰好有 'a_only'、'b_only'、'p_value' 三个键，实际是 {keys_of(got)}"
    if got["a_only"] != want["a_only"] or got["b_only"] != want["b_only"]:
        return (f"a_only / b_only 应当是 {want['a_only']} / {want['b_only']}（只有一边预测对真实标签的工单数），"
                f"得到 {got['a_only']!r} / {got['b_only']!r}{hint}")
    if not num_ok(got["p_value"], want["p_value"], 1e-7):
        return (f"p_value 不对：a_only={want['a_only']}、b_only={want['b_only']} 的精确双侧符号检验 p 值是 "
                f"{want['p_value']!r}，得到 {got['p_value']!r}{hint}")
    return None


def check_paired(sol, true, preds_a, preds_b, want=None, hint=""):
    tickets = labeled(true, "y")
    result_a = {"predictions": {t["id"]: p for t, p in zip(tickets, preds_a)}}
    result_b = {"predictions": {t["id"]: p for t, p in zip(tickets, preds_b)}}
    want = ref.paired(tickets, result_a["predictions"], result_b["predictions"]) if want is None else want
    call = calling(show_call("paired_improvement", tickets, result_a, result_b))
    got = sol.paired_improvement(copy.deepcopy(tickets), copy.deepcopy(result_a), copy.deepcopy(result_b))
    expect_valid(got, lambda g: paired_problem(g, want, hint), call=call, want=want)


def other(label, r=None):
    """A label other than `label` (the next one, or a random one)."""
    rest = [lab for lab in LABELS if lab != label]
    return r.choice(rest + [None]) if r else rest[0]


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目轨迹：解析两条回复")
def p1_parse_trace(sol):
    check_parse(sol, TRACE_PROSE, None, "：回复里没有 <label> 标签")
    check_parse(sol, TRACE_TAGGED, {"label": "billing", "reason": "Requests a receipt for past payments."})


@case("Part 1", sample=True, title="题目轨迹：解析失败后重试一次")
def p1_classify_trace(sol):
    run_classify(sol, TRACE_TICKET, EX[:2], [TRACE_PROSE, TRACE_TAGGED],
                 want={"label": "billing", "reason": "Requests a receipt for past payments.", "retried": True})


@case("Part 1", title="忽略标签前后和之间的文字")
def p1_parse_around(sol):
    why = "：标签前后、标签之间的文字都应当忽略"
    check_parse(sol, "Sure! <label>bug</label> <reason>ok</reason> Thanks!", {"label": "bug", "reason": "ok"}, why)
    check_parse(sol, "Category:\n<label>access</label>\nWhy:\n<reason>The sender cannot log in.</reason>\n"
                     "Let me know if you need more.", {"label": "access", "reason": "The sender cannot log in."}, why)
    check_parse(sol, "I think <label>how_to</label> fits, because <reason>It asks how to use a feature.</reason>",
                {"label": "how_to", "reason": "It asks how to use a feature."}, why)
    check_parse(sol, "<reason>Asks for a refund.</reason> so: <label>billing</label>",
                {"label": "billing", "reason": "Asks for a refund."}, "：<reason> 出现在 <label> 前面也照样取")


@case("Part 1", title="去掉空白、不分大小写")
def p1_parse_case_space(sol):
    why = "：<label> 的内容去掉首尾空白、不区分大小写地和 LABELS 比较，返回小写的标签"
    check_parse(sol, "<label>Bug</label><reason>A crash.</reason>", {"label": "bug", "reason": "A crash."}, why)
    check_parse(sol, "<label> BILLING </label> <reason>Charged twice.</reason>",
                {"label": "billing", "reason": "Charged twice."}, why)
    check_parse(sol, "<label>\tHow_To  </label><reason>Asks how.</reason>", {"label": "how_to", "reason": "Asks how."},
                why)
    check_parse(sol, "<label>aCCess</label> <reason>Locked out.</reason>", {"label": "access", "reason": "Locked out."},
                why)


@case("Part 1", title="没有 <reason> 时 reason 为空串")
def p1_parse_no_reason(sol):
    why = "：没有完整的 <reason>...</reason> 标签时 reason 是空串 ''"
    check_parse(sol, "<label>bug</label>", {"label": "bug", "reason": ""}, why)
    check_parse(sol, "<label>access</label>\nThe user cannot log in.", {"label": "access", "reason": ""}, why)
    check_parse(sol, "<label>how_to</label>\n<reason>unfinished", {"label": "how_to", "reason": ""}, why)


@case("Part 1", title="标签内容可以跨行")
def p1_parse_multiline(sol):
    why = "：标签内容里可以有换行"
    check_parse(sol, "<label>\nbilling\n</label>\n<reason>Two charges for one month.\n"
                     "The sender wants the extra one refunded.</reason>",
                {"label": "billing", "reason": "Two charges for one month.\nThe sender wants the extra one refunded."},
                why)
    check_parse(sol, "<label>\nbug\n</label><reason>ok</reason>", {"label": "bug", "reason": "ok"}, why)
    check_parse(sol, "<label>access</label>\n<reason>Locked out\nafter a password reset.</reason>",
                {"label": "access", "reason": "Locked out\nafter a password reset."}, why)


@case("Part 1", title="内容不是四个标签之一：None")
def p1_parse_invalid(sol):
    why = "：<label> 的内容去空白、转小写后必须是四个标签之一"
    for reply in ["<label>spam</label>", "<label>refund</label><reason>Wants money back.</reason>",
                  "<label></label><reason>Unclear.</reason>", "<label>how to</label><reason>Asks how.</reason>",
                  "<label>billing.</label>", "<label>bug report</label>"]:
        check_parse(sol, reply, None, why)
    why = "：回复里没有完整的 <label>...</label> 标签"
    for reply in ["no tags in this reply at all", "label: bug\nreason: it crashes", "<reason>Clearly a bug.</reason>",
                  "<label>bug", "bug</label>"]:
        check_parse(sol, reply, None, why)


@case("Part 1", title="只看第一个 <label> 和 <reason>")
def p1_parse_first_tag(sol):
    check_parse(sol, "<label>bug</label><label>billing</label>", {"label": "bug", "reason": ""},
                "：只看第一个 <label> 标签")
    check_parse(sol, "<label>access</label><reason>Locked out.</reason>\n<label>billing</label>"
                     "<reason>Card declined.</reason>", {"label": "access", "reason": "Locked out."},
                "：只看第一个 <label> 和第一个 <reason>")
    why = "：第一个 <label> 的内容不是四个标签之一就返回 None，不再看后面的 <label>"
    check_parse(sol, "<label>spam</label> sorry, I meant <label>bug</label><reason>A crash.</reason>", None, why)
    check_parse(sol, "<label>how to</label><label>how_to</label>", None, why)
    check_parse(sol, "<label>bug</label><reason>A crash.</reason><reason>Something else.</reason>",
                {"label": "bug", "reason": "A crash."}, "：reason 取第一个 <reason> 的内容")


@case("Part 1", title="随机回复")
def p1_parse_random(sol):
    r = rng()
    for _ in range(80):
        check_parse(sol, ref.random_reply(r))


@case("Part 1", title="第一次就能解析：不重试")
def p1_classify_no_retry(sol):
    run_classify(sol, T["t09"]["text"], EX[:2],
                 ["<label>bug</label>\n<reason>Uploading a large PNG blanks the page.</reason>"])
    run_classify(sol, T["t23"]["text"], EX[:2], ["Sure. <label>Access</label>"])


@case("Part 1", title="解析失败：重试一次")
def p1_classify_retry(sol):
    run_classify(sol, T["t01"]["text"], EX[:2],
                 ["label: billing\nreason: duplicate charge", "<label>billing</label><reason>Duplicate charge.</reason>"])
    run_classify(sol, T["t15"]["text"], EX[:2],
                 ["<label>question</label><reason>Asks how to invite someone.</reason>", "<label>how_to</label>"])


@case("Part 1", title="两次都失败：label 为 None")
def p1_classify_both_fail(sol):
    run_classify(sol, T["t12"]["text"], EX[:2],
                 ["I am not sure.", "<label>unknown</label><reason>Could be several things.</reason>"])
    run_classify(sol, T["t26"]["text"], EX[:2],
                 ["<label>password</label><reason>Reset email missing.</reason>", "Still not sure, sorry."])


@case("Part 1", title="重试的 prompt 含失败的回复")
def p1_classify_retry_prompt(sol):
    run_classify(sol, T["t27"]["text"], EX[:2],
                 ["The ticket is about logging in, so access.", "<label>access</label><reason>SSO loop.</reason>"])
    run_classify(sol, T["t08"]["text"], EX[:2],
                 ["<label>Bug report</label>\n<reason>The export option is grayed out.</reason>",
                  "<label>bug</label><reason>The export option is grayed out.</reason>"])


@case("Part 1", title="传入的示例要展示给模型")
def p1_classify_examples(sol):
    run_classify(sol, T["t10"]["text"], EX[2:5], ["<label>bug</label><reason>Items vanished from search.</reason>"])
    run_classify(sol, T["t28"]["text"], [EX[5]], ["<label>access</label><reason>Account ownership.</reason>"])
    run_classify(sol, T["t17"]["text"], EX[4:8],
                 ["no tags", "<label>how_to</label><reason>Asks where a setting is.</reason>"])


@case("Part 1", title="temperature 传给 complete")
def p1_classify_temperature(sol):
    run_classify(sol, T["t05"]["text"], EX[:2], ["<label>billing</label><reason>Discount missing.</reason>"],
                 temperature=0.7)
    run_classify(sol, T["t20"]["text"], EX[:4],
                 ["Here you go: how_to", "<label>how_to</label><reason>Asks how to set a shortcut.</reason>"],
                 temperature=1.0)


@case("Part 1", title="随机回复序列")
def p1_classify_random(sol):
    r = rng()
    for _ in range(25):
        ticket = r.choice(ref.TICKETS)["text"]
        examples = r.sample(EX, r.randint(1, 3))
        run_classify(sol, ticket, examples, [ref.random_reply(r), ref.random_reply(r)],
                     temperature=r.choice([None, 0.0, 0.5, 1.0]))


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目示例：score")
def p2_score_example(sol):
    tickets = [{"id": "x1", "label": "billing"}, {"id": "x2", "label": "billing"},
               {"id": "x3", "label": "bug"}, {"id": "x4", "label": "bug"},
               {"id": "x5", "label": "how_to"}, {"id": "x6", "label": "access"}]
    predictions = {"x1": "billing", "x2": "access", "x3": "bug", "x4": None, "x5": "how_to", "x6": "access"}
    want = {
        "accuracy": 4 / 6,
        "parse_failure_rate": 1 / 6,
        "per_label": {
            "billing": {"precision": 1.0, "recall": 0.5},
            "bug": {"precision": 1.0, "recall": 0.5},
            "how_to": {"precision": 1.0, "recall": 1.0},
            "access": {"precision": 0.5, "recall": 1.0},
        },
    }
    check_score(sol, tickets, predictions, want)


@case("Part 2", title="没被预测过的标签：precision 0.0")
def p2_score_never_predicted(sol):
    hint = "：从没被预测过的标签 precision 是 0.0"
    tickets = labeled(["billing", "bug", "how_to", "access", "bug"], "x")
    check_score(sol, tickets, {"x1": "billing", "x2": "bug", "x3": "bug", "x4": "billing", "x5": "bug"}, hint=hint)
    check_score(sol, tickets, {t["id"]: None for t in tickets}, hint=hint + "；None 不算预测了任何标签")


@case("Part 2", title="缺少的预测算错")
def p2_score_missing(sol):
    tickets = labeled(["billing", "billing", "bug", "how_to", "access", "access"], "m")
    check_score(sol, tickets, {"m1": "billing", "m3": "bug", "m4": "billing", "m6": "access"},
                skip=("parse_failure_rate",), hint="：predictions 里没有的工单算预测错误，不从分母里去掉")


@case("Part 2", title="随机预测")
def p2_score_random(sol):
    r = rng()
    for _ in range(15):
        labels = LABELS + [r.choice(LABELS) for _ in range(r.randint(0, 8))]
        r.shuffle(labels)
        tickets = labeled(labels, "r")
        predictions = {t["id"]: t["label"] if r.random() < 0.5 else r.choice(LABELS + [None]) for t in tickets}
        check_score(sol, tickets, predictions)


@case("Part 2", title="evaluate：逐条分类并汇总")
def p2_evaluate_basic(sol):
    tickets = [T[i] for i in ("t01", "t02", "t08", "t09", "t15", "t16", "t22", "t23")]
    outcomes = {"t01": ("billing", False), "t02": ("billing", False), "t08": ("how_to", False),
                "t09": ("bug", False), "t15": ("how_to", False), "t16": ("how_to", False),
                "t22": ("billing", False), "t23": (None, True)}
    run_evaluate(sol, tickets, EX[:2], outcomes)


@case("Part 2", title="retry_count 数重试过的工单")
def p2_evaluate_retry_count(sol):
    tickets = [T[i] for i in ("t03", "t04", "t10", "t11", "t17", "t18", "t24", "t25")]
    outcomes = {"t03": ("billing", True), "t04": ("billing", False), "t10": ("bug", True), "t11": ("how_to", True),
                "t17": ("how_to", False), "t18": (None, True), "t24": ("access", False), "t25": ("access", False)}
    run_evaluate(sol, tickets, EX[:2], outcomes,
                 hint="：retry_count 数的是 retried 为 True 的工单，重试后解析成功的也算")


@case("Part 2", title="examples、temperature 原样传入")
def p2_evaluate_args(sol):
    tickets = [T[i] for i in ("t05", "t12", "t19", "t26")]
    outcomes = {"t05": ("billing", False), "t12": ("bug", True), "t19": ("how_to", False), "t26": ("access", False)}
    run_evaluate(sol, tickets, EX[2:5], outcomes, temperature=0.7)
    tickets = [T[i] for i in ("t27", "t06", "t20", "t13")]
    outcomes = {"t27": (None, True), "t06": ("billing", False), "t20": ("bug", False), "t13": ("bug", False)}
    run_evaluate(sol, tickets, EX[5:8], outcomes, temperature=1.0)


@case("Part 2", title="evaluate：随机")
def p2_evaluate_random(sol):
    r = rng()
    for _ in range(6):
        tickets = random_tickets(r, 4, 10)
        run_evaluate(sol, tickets, r.sample(EX, r.randint(1, 4)), random_outcomes(r, tickets),
                     temperature=r.choice([None, 0.0, 0.3, 1.0]))


@case("Part 2", title="compare：两组示例")
def p2_compare_basic(sol):
    out_a = {t["id"]: (t["label"], t["id"] == "t03") for t in ref.TICKETS}
    out_b = dict(out_a)
    out_a["t08"], out_a["t22"] = ("how_to", False), ("billing", False)
    run_compare(sol, ref.TICKETS, EX[:2], EX[:4], out_a, out_b)


SHUFFLED_IDS = [
    {"id": "q9", "text": "My card was charged twice for the same order.", "label": "billing"},
    {"id": "q10", "text": "The upload button does nothing when I click it.", "label": "bug"},
    {"id": "b2", "text": "How do I add a second admin to my team?", "label": "how_to"},
    {"id": "q1", "text": "I cannot log in since I changed my email address.", "label": "access"},
    {"id": "a7", "text": "The invoice PDF shows last year's prices.", "label": "billing"},
    {"id": "m3", "text": "Where can I see who edited a document?", "label": "how_to"},
]

BOTH_WRONG = [
    {"id": "w1", "text": "I was billed after my free trial ended without a reminder.", "label": "billing"},
    {"id": "w2", "text": "The calendar shows my meetings on the wrong day.", "label": "bug"},
    {"id": "w3", "text": "How can I change the language of the interface?", "label": "how_to"},
    {"id": "w4", "text": "My security key is no longer accepted at sign in.", "label": "access"},
    {"id": "w5", "text": "Notifications arrive twice for every comment.", "label": "bug"},
    {"id": "w6", "text": "Is there a way to archive old projects?", "label": "how_to"},
]


@case("Part 2", title="changed 按 id 排序")
def p2_compare_sorted(sol):
    out_a = {"q9": ("billing", False), "q10": ("bug", False), "b2": ("how_to", False), "q1": ("billing", False),
             "a7": ("billing", False), "m3": ("how_to", False)}
    out_b = {"q9": ("billing", False), "q10": ("how_to", False), "b2": ("how_to", False), "q1": ("access", True),
             "a7": (None, True), "m3": ("how_to", False)}
    run_compare(sol, SHUFFLED_IDS, EX[:2], EX[2:4], out_a, out_b)


@case("Part 2", title="两边都错但标签不同也算 changed")
def p2_compare_both_wrong(sol):
    out_a = {"w1": ("billing", False), "w2": ("how_to", False), "w3": ("how_to", False), "w4": (None, True),
             "w5": (None, True), "w6": ("how_to", False)}
    out_b = {"w1": ("access", False), "w2": ("access", True), "w3": ("how_to", False), "w4": ("billing", True),
             "w5": (None, True), "w6": ("how_to", False)}
    run_compare(sol, BOTH_WRONG, EX[:1], EX[:3], out_a, out_b,
                hint="：changed 比较的是两边预测的 label 本身，两边都错但标签不同（含 None 对标签）也算")


@case("Part 2", title="compare：随机")
def p2_compare_random(sol):
    r = rng()
    for _ in range(4):
        tickets = random_tickets(r, 4, 9)
        examples_a = r.sample(EX, r.randint(1, 3))
        examples_b = r.sample([e for e in EX if e not in examples_a], r.randint(1, 3))
        run_compare(sol, tickets, examples_a, examples_b, random_outcomes(r, tickets), random_outcomes(r, tickets),
                    temperature=r.choice([None, 0.0, 0.5]))


# ---------------------------------------------------------------- Part 3
# cpu=3: the statement allows scipy, and `from scipy.stats import binomtest` inside paired_improvement costs about
# 1 s of CPU in whichever Part 3 case calls it first.

@case("Part 3", cpu=3, sample=True, title="题目示例：paired_improvement")
def p3_example(sol):
    check_paired(sol, ["bug", "access", "billing"], ["how_to", "billing", "billing"], ["bug", "access", "billing"],
                 want={"a_only": 0, "b_only": 2, "p_value": 0.5})


@case("Part 3", cpu=3, title="没有分歧：p_value 为 1.0")
def p3_no_discordant(sol):
    hint = "：a_only + b_only == 0 时 p_value 是 1.0"
    check_paired(sol, ["bug", "access", "billing"], ["bug", "access", "billing"], ["bug", "access", "billing"],
                 hint=hint)
    check_paired(sol, ["bug", "how_to", "access", "billing"], ["bug", "bug", "billing", "billing"],
                 ["bug", "bug", "billing", "billing"], hint=hint)
    check_paired(sol, ["bug", "access"], [None, "access"], [None, "access"], hint=hint)


@case("Part 3", cpu=3, title="分歧对半：p_value 为 1.0")
def p3_even_split(sol):
    hint = "：p 值是概率，不能超过 1"
    check_paired(sol, ["bug", "access"], ["bug", "billing"], ["how_to", "access"], hint=hint)
    check_paired(sol, ["bug", "access", "billing", "how_to", "bug"], ["bug", "access", None, "bug", "bug"],
                 ["how_to", None, "billing", "how_to", "bug"], hint=hint)
    true = ["billing", "bug", "how_to", "access", "bug", "billing", "access"]
    check_paired(sol, true, ["billing", "bug", "how_to", "billing", "how_to", None, "access"],
                 ["access", "how_to", None, "access", "bug", "billing", "access"], hint=hint)


@case("Part 3", cpu=3, title="两边都错不算分歧")
def p3_both_wrong(sol):
    hint = "：只有一边预测对真实标签的工单才算分歧；两边都错（标签不同、或其中一边是 None）不算"
    check_paired(sol, ["bug", "access", "billing", "how_to", "bug"], ["how_to", "billing", "billing", None, "bug"],
                 ["access", "how_to", "access", "billing", "bug"], hint=hint)
    check_paired(sol, ["access", "bug", "bug", "how_to"], [None, "bug", "bug", "billing"],
                 ["access", None, "how_to", "bug"], hint=hint)


@case("Part 3", cpu=3, title="不同分歧数的 p_value")
def p3_values(sol):
    r = rng()
    for a_only, b_only in [(0, 5), (0, 6), (1, 9), (7, 2), (4, 1), (0, 1), (11, 3), (2, 12)]:
        true, pa, pb = [], [], []
        for k in range(a_only + b_only + r.randint(0, 2)):
            lab = r.choice(LABELS)
            if k < a_only:
                row = (lab, lab, other(lab, r))
            elif k < a_only + b_only:
                row = (lab, other(lab, r), lab)
            else:
                wrong = r.choice([lab, other(lab)])
                row = (lab, wrong, wrong)
            true.append(row[0])
            pa.append(row[1])
            pb.append(row[2])
        order = list(range(len(true)))
        r.shuffle(order)
        check_paired(sol, [true[i] for i in order], [pa[i] for i in order], [pb[i] for i in order])


@case("Part 3", cpu=3, title="随机预测")
def p3_random(sol):
    r = rng()
    for _ in range(12):
        true = [r.choice(LABELS) for _ in range(r.randint(1, 14))]
        pa = [lab if r.random() < 0.6 else other(lab, r) for lab in true]
        pb = [lab if r.random() < 0.6 else other(lab, r) for lab in true]
        check_paired(sol, true, pa, pb)
