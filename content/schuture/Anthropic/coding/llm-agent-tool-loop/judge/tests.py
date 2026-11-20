"""Tests for the agent loop. `client` is replaced by ref.ScriptedClient, a fake model that plays back fixed replies
and, on every create() call, checks the conversation against the statement's loop. Each failure shows a replay
that can be pasted into 自定义运行."""
from judgelib import calling, case, expect, expect_raises, rng, show_call
import ref

AAPL_Q = "What was AAPL's closing price on 2026-01-07?"
MSFT_Q = "What was MSFT's closing price on 2026-01-09?"
GOOG_Q = "What was GOOG's closing price on 2026-01-05?"
GAIN_Q = "Between 2026-01-05 and 2026-01-09, did AAPL or MSFT gain more, in percentage terms?"
WEEKEND_Q = "What was GOOG's closing price on 2026-01-10?"
HIGH_Q = "Which of AAPL, MSFT and GOOG had the highest closing price on 2026-01-08?"

REF_TOOLS = ("QUOTE_TOOL_ALT", "QUOTES_TOOL", "NEWS_TOOL")
BATCH_SYSTEM = "Before answering, request every price the question needs in a single turn."
FINAL_MSG = "返回值应当是最后一次回复（stop_reason == 'end_turn'）里那个 text 块的文字"


def q(ticker, date):
    return ("get_quote", {"ticker": ticker, "date": date})


def qs(*pairs):
    return ("get_quotes", {"lookups": [{"ticker": t, "date": d} for t, d in pairs]})


def turns_src(turns):
    if len(turns) >= 3 and not isinstance(turns[0], str) and all(t == turns[0] for t in turns):
        return f"[{turns[0]!r}] * {len(turns)}"
    return "[\n" + "".join(f"    {t!r},\n" for t in turns) + "]"


def ask(sol, question, turns):
    """answer_question(question) against a scripted model whose last reply is the final answer."""
    calling("\n".join(["from ref import ScriptedClient", f"q = {question!r}",
                       f"client = ScriptedClient(q, {turns_src(turns)})", "print(repr(answer_question(q)))"]))
    client = ref.ScriptedClient(question, turns)
    sol.client = client
    got = sol.answer_question(question)
    client.finish()
    expect(got, turns[-1], msg=FINAL_MSG)


def agent(sol, question, turns, tools=("GET_QUOTE_TOOL",), system="SYSTEM_PROMPT", max_calls=None):
    """run_agent(question, tools, system[, max_calls]) against a scripted model. `tools` names the tool dicts
    (GET_QUOTE_TOOL is the answer's own, the others come from ref); system "SYSTEM_PROMPT" is the answer's own."""
    imports = ["ScriptedClient"] + [t for t in tools if t in REF_TOOLS]
    tool_objs = [getattr(ref, t) if t in REF_TOOLS else getattr(sol, t) for t in tools]
    sys_obj, sys_src = (sol.SYSTEM_PROMPT, "SYSTEM_PROMPT") if system == "SYSTEM_PROMPT" else (system, repr(system))
    cap = "" if max_calls is None else f", max_calls={max_calls}"
    calling("\n".join([f"from ref import {', '.join(imports)}", f"q = {question!r}",
                       f"tools, system = [{', '.join(tools)}], {sys_src}",
                       f"client = ScriptedClient(q, {turns_src(turns)}, tools=tools, system=system)",
                       f"print(repr(run_agent(q, tools, system{cap})))"]))
    client = ref.ScriptedClient(question, turns, tools=tool_objs, system=sys_obj)
    sol.client = client
    if max_calls is None:
        got = sol.run_agent(question, list(tool_objs), sys_obj)
    else:
        got = sol.run_agent(question, list(tool_objs), sys_obj, max_calls=max_calls)
    client.finish()
    if isinstance(turns[-1], str):
        expect(got, turns[-1], msg=FINAL_MSG)
    else:
        n = len(turns)
        expect(got, f"gave up after {n} calls",
               msg=f"前 {n} 次回复都是 tool_use，max_calls 是 {n}：应当返回 'gave up after {n} calls'")


def random_script(r, rounds, max_blocks, batched=False):
    return [ref.random_turn(r, max_blocks, batched) for _ in range(rounds)] + [ref.random_answer(r)]


# ---------------------------------------------------------------- Part 1

@case("Part 1", sample=True, title="题目轨迹：AAPL 2026-01-07")
def p1_example(sol):
    ask(sol, AAPL_Q, [[q("AAPL", "2026-01-07")], "AAPL closed at $179.00 on 2026-01-07."])


@case("Part 1", sample=True, title="MSFT 2026-01-09 -> 420.00")
def p1_msft(sol):
    ask(sol, MSFT_Q, [[q("MSFT", "2026-01-09")], "MSFT closed at $420.00 on 2026-01-09."])


@case("Part 1", sample=True, title="GOOG 2026-01-05 -> 140.00")
def p1_goog(sol):
    ask(sol, GOOG_Q, [[q("GOOG", "2026-01-05")], "GOOG closed at $140.00 on 2026-01-05."])


@case("Part 1", title="其他单次调用的问题，id 不是 call_N")
def p1_other_question(sol):
    ask(sol, "What was GOOG's closing price on 2026-01-12?",
        [[q("GOOG", "2026-01-12") + ("toolu_01A09q90qw90lq917835lq9",)], "GOOG closed at $146.50 on 2026-01-12."])
    ask(sol, "And MSFT on 2026-01-06?", [[q("MSFT", "2026-01-06") + ("x",)], "$406.00"])


@case("Part 1", title="工具失败：is_error=True")
def p1_tool_error(sol):
    ask(sol, "What was TSLA's closing price on 2026-01-07?",
        [[q("TSLA", "2026-01-07")], "I could not look up TSLA: unknown ticker: TSLA."])
    ask(sol, "What was AAPL's closing price on 2026-01-11?",
        [[q("AAPL", "2026-01-11")], "There is no trading data for 2026-01-11."])


@case("Part 1", title="模型不调用工具直接回答")
def p1_direct_answer(sol):
    ask(sol, "Which tickers can you look up?", ["AAPL, MSFT and GOOG."])


@case("Part 1", title="一次回复里有两个 tool_use 块")
def p1_two_blocks(sol):
    ask(sol, "(two independent lookups)",
        [[q("AAPL", "2026-01-07") + ("a1",), q("MSFT", "2026-01-09") + ("a2",)], "done"])


@case("Part 1", title="相同的请求也按 id 各回答一次")
def p1_same_request(sol):
    ask(sol, "What did AAPL and GOOG close at?",
        [[q("AAPL", "2026-01-07") + ("x1",), q("AAPL", "2026-01-07") + ("x2",), q("GOOG", "2026-01-10") + ("x3",)],
         "AAPL closed at $179.00; GOOG has no data for 2026-01-10."])


@case("Part 1", title="要一直循环到 end_turn")
def p1_several_rounds(sol):
    ask(sol, "Did AAPL close higher on 2026-01-12 than on 2026-01-05?",
        [[q("AAPL", "2026-01-05")], [q("AAPL", "2026-01-12")], "Yes: $188.00 against $180.00."])
    ask(sol, GAIN_Q, [[q("AAPL", "2026-01-05"), q("AAPL", "2026-01-09")], [q("MSFT", "2026-01-10")],
                      [q("MSFT", "2026-01-09")], [q("MSFT", "2026-01-05"), q("GOOG", "2026-01-05")],
                      [q("NVDA", "2026-01-05")], "AAPL gained more."])


@case("Part 1", title="随机对话 150 段")
def p1_random(sol):
    r = rng("p1")
    for _ in range(150):
        ask(sol, f"Question #{r.randrange(10 ** 4)}?", random_script(r, r.randint(0, 4), 3))


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目轨迹：GOOG 2026-01-10 没有数据")
def p2_example(sol):
    agent(sol, WEEKEND_Q, [[q("GOOG", "2026-01-10")],
                           "I could not get GOOG's price on 2026-01-10: no trading data for 2026-01-10."])


@case("Part 2", sample=True, title="AAPL 和 MSFT 谁涨得多：5 次调用")
def p2_gain(sol):
    agent(sol, GAIN_Q, [[q("AAPL", "2026-01-05")], [q("AAPL", "2026-01-09")], [q("MSFT", "2026-01-05")],
                        [q("MSFT", "2026-01-09")], "AAPL gained more: about 5.6% against MSFT's 2.4%."])


@case("Part 2", sample=True, title="2026-01-08 谁最高：4 次调用")
def p2_highest(sol):
    agent(sol, HIGH_Q, [[q("AAPL", "2026-01-08")], [q("MSFT", "2026-01-08")], [q("GOOG", "2026-01-08")],
                        "MSFT had the highest close on 2026-01-08: $415.00."])


@case("Part 2", title="tools 和 system 原样传给 create")
def p2_params(sol):
    agent(sol, HIGH_Q, [[q("AAPL", "2026-01-08"), q("MSFT", "2026-01-08")], [q("GOOG", "2026-01-08")], "MSFT."],
          tools=("QUOTE_TOOL_ALT", "NEWS_TOOL"), system="Answer in one word. Look up every price you use.")
    agent(sol, MSFT_Q, [[q("MSFT", "2026-01-09")], "420.00"],
          tools=("QUOTE_TOOL_ALT",), system="You answer questions about closing prices.")


@case("Part 2", title="同一轮里有成功也有失败")
def p2_mixed_results(sol):
    agent(sol, "How did AAPL, NVDA and GOOG do on 2026-01-09 and 2026-01-11?",
          [[q("AAPL", "2026-01-09"), q("AAPL", "2026-01-11"), q("NVDA", "2026-01-09"), q("GOOG", "2026-01-09")],
           [q("GOOG", "2026-01-11"), q("GOOG", "2026-01-12")],
           "AAPL $190.00 and GOOG $145.00 on 2026-01-09; no data on 2026-01-11; NVDA is not covered."])


@case("Part 2", title="max_calls 次后放弃，不再调用")
def p2_cap(sol):
    agent(sol, "(never resolves)", [[q("AAPL", "2026-01-07")]] * 3, max_calls=3)
    agent(sol, "(never resolves)", [[q("MSFT", "2026-01-06"), q("GOOG", "2026-01-13")]] * 5, max_calls=5)


@case("Part 2", title="第 max_calls 次正好 end_turn")
def p2_cap_boundary(sol):
    agent(sol, GAIN_Q, [[q("AAPL", "2026-01-05"), q("AAPL", "2026-01-09")],
                        [q("MSFT", "2026-01-05"), q("MSFT", "2026-01-09")], "AAPL gained more."], max_calls=3)
    agent(sol, HIGH_Q, [[q("AAPL", "2026-01-08")], [q("MSFT", "2026-01-08")], "MSFT."], max_calls=4)


@case("Part 2", title="max_calls=1")
def p2_cap_one(sol):
    agent(sol, "Which tickers can you look up?", ["AAPL, MSFT and GOOG."], max_calls=1)
    agent(sol, AAPL_Q, [[q("AAPL", "2026-01-07")]], max_calls=1)


@case("Part 2", title="默认 max_calls=8")
def p2_cap_default(sol):
    agent(sol, "(never resolves)", [[q("GOOG", "2026-01-06")]] * 8)
    agent(sol, "(seven rounds)", [[q("AAPL", d)] for d in ref.DATES + ["2026-01-10"]] + ["Seven lookups."])


@case("Part 2", title="上限更大时")
def p2_cap_large(sol):
    agent(sol, "(never resolves)", [[q("AAPL", "2026-01-05"), q("AAPL", "2026-01-04")]] * 20, max_calls=20,
          tools=("QUOTE_TOOL_ALT",), system="Keep looking things up.")
    turns = [[q(t, d)] for t in ref.TICKERS for d in ref.DATES[:5]] + ["Fifteen prices looked up."]
    agent(sol, "(fifteen rounds)", turns, max_calls=25)


@case("Part 2", title="随机对话 150 段")
def p2_random(sol):
    r = rng("p2")
    setups = [(("GET_QUOTE_TOOL",), "SYSTEM_PROMPT"), (("QUOTE_TOOL_ALT",), "You are a careful assistant."),
              (("QUOTE_TOOL_ALT", "NEWS_TOOL"), "Use the tools for every number.")]
    for _ in range(150):
        tools, system = r.choice(setups)
        max_calls = None if r.random() < 0.3 else r.randint(1, 6)
        cap = 8 if max_calls is None else max_calls
        question = f"Question #{r.randrange(10 ** 4)}?"
        if r.random() < 0.3:
            turns = [ref.random_turn(r, 2) for _ in range(cap)]
        else:
            turns = random_script(r, r.randint(0, cap - 1), 2)
        agent(sol, question, turns, tools=tools, system=system, max_calls=max_calls)


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目轨迹：一次 get_quotes 查四个价格")
def p3_example(sol):
    agent(sol, GAIN_Q, [[qs(("AAPL", "2026-01-05"), ("AAPL", "2026-01-09"), ("MSFT", "2026-01-05"),
                            ("MSFT", "2026-01-09"))],
                        "AAPL gained more in percentage terms between 2026-01-05 and 2026-01-09."],
          tools=("GET_QUOTE_TOOL", "QUOTES_TOOL"), system=BATCH_SYSTEM)


def lookups(*pairs):
    return [{"ticker": t, "date": d} for t, d in pairs]


def check_quotes(sol, pairs):
    arg = lookups(*pairs)
    want = ref.quotes(arg)
    call = calling(show_call("get_quotes", arg))
    expect(sol.get_quotes(lookups(*pairs)), want, call=call)


def check_fails(sol, pairs):
    """get_quotes must fail with get_quote's message for the first invalid pair. The statement does not name the
    exception type (only run_tool's str(e) reaches the model), so any exception with that message is accepted."""
    arg = lookups(*pairs)
    call = calling(show_call("get_quotes", arg))
    bad = next(p for p in pairs if ref.tool_output("get_quote", {"ticker": p[0], "date": p[1]})[1])
    want = ref.tool_output("get_quote", {"ticker": bad[0], "date": bad[1]})[0]
    e = expect_raises(Exception, sol.get_quotes, lookups(*pairs), call=call,
                      msg=f"lookups 里有无效组合 {bad}，应当像 get_quote 一样抛出异常（{want!r}），不返回任何结果")
    expect(str(e), want, call=call, msg=f"应当在第一个无效组合 {bad} 处失败，报错和 get_quote{bad} 相同")


@case("Part 3", sample=True, title="get_quotes 题目例子")
def p3_get_quotes_example(sol):
    check_quotes(sol, [("AAPL", "2026-01-05"), ("AAPL", "2026-01-09"), ("MSFT", "2026-01-05"), ("MSFT", "2026-01-09")])


@case("Part 3", title="按 lookups 的顺序，含重复和空列表")
def p3_order(sol):
    check_quotes(sol, [("GOOG", "2026-01-06")])
    check_quotes(sol, [("MSFT", "2026-01-12"), ("AAPL", "2026-01-05"), ("MSFT", "2026-01-12"), ("GOOG", "2026-01-08")])
    check_quotes(sol, [])
    pairs = [(t, d) for t in ref.TICKERS for d in ref.DATES]
    rng("p3order").shuffle(pairs)
    check_quotes(sol, pairs)


@case("Part 3", title="有一个无效组合就整体失败")
def p3_all_or_nothing(sol):
    check_fails(sol, [("AAPL", "2026-01-05"), ("AAPL", "2026-01-10")])
    check_fails(sol, [("TSLA", "2026-01-05"), ("MSFT", "2026-01-05"), ("GOOG", "2026-01-05")])
    check_fails(sol, [("MSFT", "2026-01-05"), ("GOOG", "2026-01-13"), ("MSFT", "2026-01-09")])
    check_fails(sol, [("GOOG", "2026-01-11")])


@case("Part 3", title="报的是第一个无效组合")
def p3_first_bad_pair(sol):
    check_fails(sol, [("AAPL", "2026-01-05"), ("AAPL", "2026-01-11"), ("TSLA", "2026-01-05")])
    check_fails(sol, [("NVDA", "2026-01-10"), ("AAPL", "2026-01-10")])
    check_fails(sol, [("MSFT", "2026-01-04"), ("GOOG", "2026-01-13"), ("aapl", "2026-01-05")])
    check_fails(sol, [("GOOG", "2026-01-05"), ("TSLA", "2026-01-05"), ("NVDA", "2026-01-05")])


@case("Part 3", title="run_tool 能运行 get_quotes")
def p3_run_tool(sol):
    for pairs in ([("AAPL", "2026-01-05"), ("AAPL", "2026-01-09"), ("MSFT", "2026-01-05"), ("MSFT", "2026-01-09")],
                  [("GOOG", "2026-01-06")], [], [("AAPL", "2026-01-05"), ("AAPL", "2026-01-10")],
                  [("MSFT", "2026-01-07"), ("TSLA", "2026-01-07")]):
        arg = {"lookups": lookups(*pairs)}
        want = ref.tool_output("get_quotes", arg)
        call = calling(show_call("run_tool", "get_quotes", arg))
        expect(sol.run_tool("get_quotes", {"lookups": lookups(*pairs)}), want, call=call)


@case("Part 3", title="get_quotes 失败也传回给模型")
def p3_agent_error(sol):
    agent(sol, "What did AAPL and GOOG close at on 2026-01-09 and 2026-01-10?",
          [[qs(("AAPL", "2026-01-09"), ("AAPL", "2026-01-10"), ("GOOG", "2026-01-09"))],
           [qs(("AAPL", "2026-01-09"), ("GOOG", "2026-01-09")), q("GOOG", "2026-01-10")],
           "AAPL $190.00 and GOOG $145.00 on 2026-01-09; 2026-01-10 had no trading."],
          tools=("GET_QUOTE_TOOL", "QUOTES_TOOL"), system=BATCH_SYSTEM)


@case("Part 3", title="随机对话 100 段（含 get_quotes）")
def p3_random(sol):
    r = rng("p3")
    for _ in range(100):
        agent(sol, f"Question #{r.randrange(10 ** 4)}?", random_script(r, r.randint(0, 3), 3, batched=True),
              tools=("QUOTE_TOOL_ALT", "QUOTES_TOOL"), system=BATCH_SYSTEM)
