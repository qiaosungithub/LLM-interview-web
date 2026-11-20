# expect: WA
# Wrong for Part 2: still sends the global [GET_QUOTE_TOOL] and SYSTEM_PROMPT instead of the arguments.
PRICES = {
    "AAPL": {"2026-01-05": 180.00, "2026-01-06": 181.50, "2026-01-07": 179.00,
             "2026-01-08": 184.00, "2026-01-09": 190.00, "2026-01-12": 188.00},
    "MSFT": {"2026-01-05": 410.00, "2026-01-06": 406.00, "2026-01-07": 412.00,
             "2026-01-08": 415.00, "2026-01-09": 420.00, "2026-01-12": 418.00},
    "GOOG": {"2026-01-05": 140.00, "2026-01-06": 141.40, "2026-01-07": 139.00,
             "2026-01-08": 142.50, "2026-01-09": 145.00, "2026-01-12": 146.50},
}


def get_quote(ticker: str, date: str) -> float:
    if ticker not in PRICES:
        raise ValueError(f"unknown ticker: {ticker}")
    if date not in PRICES[ticker]:
        raise ValueError(f"no trading data for {date}")
    return PRICES[ticker][date]


GET_QUOTE_TOOL = {
    "name": "get_quote",
    "description": "Look up a single stock's closing price on one trading day.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker, e.g. 'AAPL'."},
            "date": {"type": "string", "description": "Trading day, formatted YYYY-MM-DD."},
        },
        "required": ["ticker", "date"],
    },
}

MODEL_NAME = "agent-loop-demo"
MAX_TOKENS = 1024
SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer only using the get_quote tool; "
    "never state a price you did not just look up."
)

TOOL_FUNCTIONS = {"get_quote": get_quote}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    try:
        return str(TOOL_FUNCTIONS[name](**tool_input)), False
    except Exception as e:
        return str(e), True


def _answered(block: dict) -> dict:
    text, is_error = run_tool(block["name"], block["input"])
    return {"type": "tool_result", "tool_use_id": block["id"], "content": text, "is_error": is_error}


def answer_question(question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    while True:
        response = client.create(model=MODEL_NAME, system=SYSTEM_PROMPT, messages=messages,
                                  tools=[GET_QUOTE_TOOL], max_tokens=MAX_TOKENS)
        if response.stop_reason == "end_turn":                                # NOTE: loop until end_turn
            return response.content[0]["text"]
        messages.append({"role": "assistant", "content": response.content})   # NOTE: keep the model's own turn
        tool_uses = [b for b in response.content if b["type"] == "tool_use"]  # NOTE: every block, not just one
        messages.append({"role": "user", "content": [_answered(b) for b in tool_uses]})


def run_agent(question: str, tools: list[dict], system: str, max_calls: int = 8) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    for _ in range(max_calls):                            # NOTE: a hard cap -- see Follow-ups
        response = client.create(model=MODEL_NAME, system=SYSTEM_PROMPT, messages=messages,
                                  tools=[GET_QUOTE_TOOL], max_tokens=MAX_TOKENS)
        if response.stop_reason == "end_turn":
            return response.content[0]["text"]
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b["type"] == "tool_use"]
        messages.append({"role": "user", "content": [_answered(b) for b in tool_uses]})
    return f"gave up after {max_calls} calls"


def get_quotes(lookups: list[dict]) -> list[float]:
    return [get_quote(pair["ticker"], pair["date"]) for pair in lookups]   # NOTE: fails on the first bad pair


TOOL_FUNCTIONS["get_quotes"] = get_quotes

GET_QUOTES_TOOL = {
    "name": "get_quotes",
    "description": (
        "Look up closing prices for several (ticker, date) pairs in a single call. "
        "Prefer this over repeated get_quote calls whenever more than one price is needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "lookups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}, "date": {"type": "string"}},
                    "required": ["ticker", "date"],
                },
            },
        },
        "required": ["lookups"],
    },
}

SYSTEM_PROMPT_BATCHED = SYSTEM_PROMPT + (
    " When a question needs more than one price, request every price you already know you will "
    "need in a single turn: one call to get_quotes if it is offered, or one get_quote call per "
    "price, in parallel, in the same turn, if it is not."
)
TOOLS_BATCHED = [GET_QUOTE_TOOL, GET_QUOTES_TOOL]
