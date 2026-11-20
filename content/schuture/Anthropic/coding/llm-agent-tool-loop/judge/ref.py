"""Helpers for the agent-loop judge: the statement's price table, the tool outputs a correct run_tool gives, and
ScriptedClient, a fake model that plays back fixed replies and checks every conversation it is sent.

In 自定义运行:  from ref import ScriptedClient;  client = ScriptedClient(question, turns);  answer_question(question)
"""
import copy

from judgelib import Mismatch

PRICES = {
    "AAPL": {"2026-01-05": 180.00, "2026-01-06": 181.50, "2026-01-07": 179.00,
             "2026-01-08": 184.00, "2026-01-09": 190.00, "2026-01-12": 188.00},
    "MSFT": {"2026-01-05": 410.00, "2026-01-06": 406.00, "2026-01-07": 412.00,
             "2026-01-08": 415.00, "2026-01-09": 420.00, "2026-01-12": 418.00},
    "GOOG": {"2026-01-05": 140.00, "2026-01-06": 141.40, "2026-01-07": 139.00,
             "2026-01-08": 142.50, "2026-01-09": 145.00, "2026-01-12": 146.50},
}
TICKERS = list(PRICES)
DATES = list(PRICES["AAPL"])
BAD_TICKERS = ["TSLA", "NVDA", "aapl"]
BAD_DATES = ["2026-01-10", "2026-01-11", "2026-01-04", "2026-01-13"]


def quote(ticker, date):
    """The statement's get_quote."""
    if ticker not in PRICES:
        raise ValueError(f"unknown ticker: {ticker}")
    if date not in PRICES[ticker]:
        raise ValueError(f"no trading data for {date}")
    return PRICES[ticker][date]


def quotes(lookups):
    """The statement's get_quotes: in order, failing at the first invalid pair."""
    return [quote(p["ticker"], p["date"]) for p in lookups]


def tool_output(name, tool_input):
    """What run_tool(name, tool_input) returns for the statement's tools: (text, is_error)."""
    fn = {"get_quote": quote, "get_quotes": quotes}[name]
    try:
        return str(fn(**tool_input)), False
    except ValueError as e:
        return str(e), True


_QUOTE_SCHEMA = {
    "type": "object",
    "properties": {"ticker": {"type": "string"}, "date": {"type": "string"}},
    "required": ["ticker", "date"],
}
# Tool definitions owned by the judge, so a run_agent that ignores its `tools` argument is noticed.
QUOTE_TOOL_ALT = {
    "name": "get_quote",
    "description": "Closing price of one ticker on one trading day (YYYY-MM-DD).",
    "input_schema": _QUOTE_SCHEMA,
}
QUOTES_TOOL = {
    "name": "get_quotes",
    "description": "Closing prices for several (ticker, date) pairs in one call.",
    "input_schema": {
        "type": "object",
        "properties": {"lookups": {"type": "array", "items": _QUOTE_SCHEMA}},
        "required": ["lookups"],
    },
}
NEWS_TOOL = {
    "name": "get_news",
    "description": "Headlines about one ticker on one day.",
    "input_schema": _QUOTE_SCHEMA,
}


class Response:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason

    def __repr__(self):
        return f"Response(content={self.content!r}, stop_reason={self.stop_reason!r})"


def _brief(x, limit=300):
    try:
        text = repr(x)
    except Exception:       # noqa: BLE001
        text = f"<{type(x).__name__}>"
    return text if len(text) <= limit else text[:limit] + " …"


def _names(tools):
    try:
        return [t.get("name") if isinstance(t, dict) else t for t in tools]
    except TypeError:
        return tools


def _is_user_msg(m):
    return isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), (list, tuple))


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else x


def _is_question(m, question):
    """The first message: the question as one text block, or as a plain string (the API accepts both)."""
    if not isinstance(m, dict) or m.get("role") != "user" or set(m) != {"role", "content"}:
        return False
    c = m["content"]
    return c == question or _as_list(c) == [{"type": "text", "text": question}]


def _result_text(c):
    """A tool_result's content as text: a string, or a list of text blocks (the API accepts both)."""
    if isinstance(c, (list, tuple)) and c and all(isinstance(b, dict) and b.get("type") == "text"
                                                  and isinstance(b.get("text"), str) and set(b) == {"type", "text"}
                                                  for b in c):
        return "".join(b["text"] for b in c)
    return c


class ScriptedClient:
    """A fake model for one question; create() has the statement's signature.

    `turns` lists its replies in order. A list of tool calls, each (name, input) or (name, input, id), is a
    reply with stop_reason "tool_use" (ids default to call_1, call_2, ...); a str is the final answer, a single
    text block with stop_reason "end_turn". A script that ends with tool calls is for a max_calls test: its
    length is the max_calls under test.

    Each create() call checks the conversation against the statement's loop: the question; then, for every
    earlier tool_use reply, that reply's content unchanged as an assistant message, followed by one user message
    with exactly one tool_result per tool_use block (any order), carrying run_tool's text and is_error.
    `tools` and `system`, when given, must be passed to create() unchanged. Forms the API accepts are fine: the
    question as a plain string, a tool_result's content as a list of text blocks, tuples for lists, extra
    keyword arguments. The first problem raises Mismatch and is raised again by finish().
    """

    def __init__(self, question, turns, tools=None, system=None):
        self.question = question
        self.tools = copy.deepcopy(tools)
        self.system = system
        self.calls = 0
        self.error = None
        self.replies = []
        n = 0
        for turn in turns:
            if isinstance(turn, str):
                self.replies.append(([{"type": "text", "text": turn}], "end_turn"))
                continue
            blocks = []
            for call in turn:
                if len(call) > 2:
                    block_id = call[2]
                else:
                    n += 1
                    block_id = f"call_{n}"
                blocks.append({"type": "tool_use", "id": block_id, "name": call[0], "input": copy.deepcopy(call[1])})
            self.replies.append((blocks, "tool_use"))

    def create(self, model=None, system=None, messages=None, tools=None, max_tokens=None, **kwargs):
        # Extra keyword arguments (temperature, metadata, ...) and the max_tokens value do not matter here.
        self.calls += 1
        if self.error is None:
            problem = self._check(self.calls, system, messages, tools)
            if problem:
                self.error = Mismatch(f"第 {self.calls} 次调用 client.create 时：{problem}", show=False)
        if self.error is not None:
            raise self.error
        blocks, stop = self.replies[self.calls - 1]
        return Response(copy.deepcopy(blocks), stop)

    def finish(self):
        """After the function returned: no problem was seen, and every reply was used."""
        if self.error is not None:
            raise self.error
        n = len(self.replies)
        if self.calls >= n:
            return
        if self.replies[-1][1] == "end_turn":
            if self.calls == 0:
                raise Mismatch("函数没有调用 client.create 就返回了", show=False)
            raise Mismatch(f"第 {self.calls} 次回复的 stop_reason 是 'tool_use'，对话还没结束（最终回答在第 {n} 次回复里），"
                           f"函数却只调用了 {self.calls} 次 create 就返回了：要一直循环到 stop_reason == 'end_turn'", show=False)
        raise Mismatch(f"模型一直没有 end_turn，max_calls 是 {n}：应当正好调用 {n} 次 create 再放弃，实际只调用了 "
                       f"{self.calls} 次", show=False)

    # ------------------------------------------------------------ checks

    def _check(self, k, system, messages, tools):
        n = len(self.replies)
        if k > n:
            if self.replies[-1][1] == "end_turn":
                return f"模型在第 {n} 次回复里已经给出最终回答（stop_reason == 'end_turn'），不应再调用 create"
            return (f"前 {n} 次回复都是 tool_use，而这个测试的 max_calls 是 {n}：此时应当停下来返回 "
                    f"'gave up after {n} calls'，不能再调用 create")
        if self.system is not None and system != self.system:
            return f"system 应当原样使用传给 run_agent 的 system 参数 {_brief(self.system)}，实际传的是 {_brief(system)}"
        if self.tools is not None and _as_list(tools) != self.tools:
            if _names(tools) == _names(self.tools):
                return (f"tools 应当原样使用传给 run_agent 的 tools 参数；实际传的工具名相同（{_names(tools)}），"
                        f"但内容不同（例如 description）")
            return f"tools 应当原样使用传给 run_agent 的 tools 参数（工具名 {_names(self.tools)}），实际传的是 {_brief(tools)}"
        if not isinstance(messages, (list, tuple)):
            return f"messages 应当是列表，实际是 {type(messages).__name__}"
        msgs = list(messages)
        first = {"role": "user", "content": [{"type": "text", "text": self.question}]}
        if not msgs:
            return f"messages 是空的；第一条应当是 {first!r}"
        if not _is_question(msgs[0], self.question):
            return f"messages[0] 应当是 {first!r}，实际是 {_brief(msgs[0])}"
        for j in range(1, k):           # the j-th reply was a tool_use turn
            blocks = self.replies[j - 1][0]
            a, u = 2 * j - 1, 2 * j
            if len(msgs) <= a:
                return (f"messages 只有 {len(msgs)} 条：第 {j} 次回复是 tool_use，之后应当先追加 "
                        f"{{'role': 'assistant', 'content': response.content}}，再追加一条装着 tool_result 的 user 消息")
            m = msgs[a]
            if not isinstance(m, dict) or m.get("role") != "assistant":
                return (f"messages[{a}] 应当是原样放回第 {j} 次回复的 assistant 消息 "
                        f"{{'role': 'assistant', 'content': response.content}}，实际是 {_brief(m)}")
            if _as_list(m.get("content")) != blocks:
                return (f"messages[{a}] 的 content 应当原样等于第 {j} 次回复的 response.content {_brief(blocks)}，"
                        f"实际是 {_brief(m.get('content'))}")
            if len(msgs) <= u:
                return f"messages 少了第 {j} 次回复对应的 tool_result 消息：它应当是 messages[{u}]，一条 role 为 'user' 的消息"
            m = msgs[u]
            if not _is_user_msg(m):
                return f"messages[{u}] 应当是装着第 {j} 次回复全部 tool_result 的 user 消息，实际是 {_brief(m)}"
            problem = self._check_results(j, u, list(m["content"]), blocks, msgs[u + 1:])
            if problem:
                return problem
        want_len = 2 * k - 1
        if len(msgs) > want_len:
            return (f"messages 应当正好有 {want_len} 条（问题，然后每次 tool_use 回复各一条 assistant 消息和一条 "
                    f"tool_result 消息），实际有 {len(msgs)} 条，多出的第一条是 messages[{want_len}] = "
                    f"{_brief(msgs[want_len])}")
        return None

    def _check_results(self, j, u, results, blocks, later):
        uses = {b["id"]: b for b in blocks}
        seen = set()
        for r in results:
            if not isinstance(r, dict) or r.get("type") != "tool_result":
                return f"messages[{u}] 里只应有 tool_result 块，却出现了 {_brief(r)}"
            rid = r.get("tool_use_id")
            if rid not in uses:
                return (f"messages[{u}] 里有一个 tool_result 的 tool_use_id 是 {rid!r}，它不是第 {j} 次回复里任何 "
                        f"tool_use 块的 id（这一轮的 id：{list(uses)}）")
            if rid in seen:
                return f"tool_use 块 {rid!r} 有不止一个 tool_result，应当正好一个"
            seen.add(rid)
            b = uses[rid]
            text, is_error = tool_output(b["name"], b["input"])
            if _result_text(r.get("content")) != text:
                return (f"tool_use 块 {rid!r}（{b['name']}，input={b['input']!r}）的 tool_result 的 content 应当是 "
                        f"run_tool 返回的文字 {text!r}，实际是 {_brief(r.get('content'))}")
            got = r.get("is_error", False)
            if got != is_error:
                return (f"tool_use 块 {rid!r}（{b['name']}，input={b['input']!r}）的 tool_result 的 is_error 应当是 "
                        f"{is_error}（run_tool 返回 {(text, is_error)!r}），实际是 {_brief(got)}")
        missing = [i for i in uses if i not in seen]
        if not missing:
            return None
        i = missing[0]
        if any(_is_user_msg(m) and any(isinstance(r, dict) and r.get("tool_use_id") == i for r in m["content"])
               for m in later):
            return (f"第 {j} 次回复的 tool_result 分散在了多条 user 消息里：这一轮所有 tool_use 块的 tool_result "
                    f"应当合并进同一条 user 消息 messages[{u}]")
        return (f"第 {j} 次回复里的 tool_use 块 {i!r}（{uses[i]['name']}，input={uses[i]['input']!r}）没有对应的 "
                f"tool_result：一次回复里的每个 tool_use 块都要回答")


# ---------------------------------------------------------------- random scripts

def random_pair(r, bad=0.2):
    """A {"ticker", "date"} input; with probability `bad` one the table does not have."""
    x = r.random()
    if x < bad / 2:
        return {"ticker": r.choice(BAD_TICKERS), "date": r.choice(DATES + BAD_DATES)}
    if x < bad:
        return {"ticker": r.choice(TICKERS), "date": r.choice(BAD_DATES)}
    return {"ticker": r.choice(TICKERS), "date": r.choice(DATES)}


def random_turn(r, max_blocks, batched=False):
    """One tool_use reply: 1..max_blocks get_quote calls (and get_quotes calls if batched); sometimes own ids."""
    calls = []
    for _ in range(r.randint(1, max_blocks)):
        if batched and r.random() < 0.5:
            call = ("get_quotes", {"lookups": [random_pair(r, 0.1) for _ in range(r.randint(0, 4))]})
        else:
            call = ("get_quote", random_pair(r))
        if r.random() < 0.2:
            call += (f"toolu_{r.randrange(16 ** 6):06x}",)
        calls.append(call)
    if len(calls) > 1 and r.random() < 0.2:
        calls.append(calls[0][:2])          # the same request again, under a new id
    return calls


def random_answer(r):
    t, d = r.choice(TICKERS), r.choice(DATES)
    return r.choice([f"{t} closed at ${PRICES[t][d]:.2f} on {d}.", f"No data for {t} on {r.choice(BAD_DATES)}.",
                     f"{t} had the highest close.", "Done."])
