# ---- the notebook ----
# `client` is the model client. The judge replaces it with a scripted fake model that checks every conversation
# sent to client.create; in 自定义运行 you can do the same:
#     from ref import ScriptedClient
#     client = ScriptedClient("What was AAPL's closing price on 2026-01-07?",
#                             [[("get_quote", {"ticker": "AAPL", "date": "2026-01-07"})], "AAPL closed at $179.00."])
client = None

PRICES = {
    "AAPL": {"2026-01-05": 180.00, "2026-01-06": 181.50, "2026-01-07": 179.00,
             "2026-01-08": 184.00, "2026-01-09": 190.00, "2026-01-12": 188.00},
    "MSFT": {"2026-01-05": 410.00, "2026-01-06": 406.00, "2026-01-07": 412.00,
             "2026-01-08": 415.00, "2026-01-09": 420.00, "2026-01-12": 418.00},
    "GOOG": {"2026-01-05": 140.00, "2026-01-06": 141.40, "2026-01-07": 139.00,
             "2026-01-08": 142.50, "2026-01-09": 145.00, "2026-01-12": 146.50},
}


def get_quote(ticker: str, date: str) -> float:
    """The closing price from the table above; fails for a ticker or date the table does not have."""
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


def run_tool(name: str, input: dict) -> tuple[str, bool]:
    """Runs the tool registered under `name` with keyword arguments `input`: (str(result), False), or (str(e), True)."""
    try:
        return str(TOOL_FUNCTIONS[name](**input)), False
    except Exception as e:
        return str(e), True


# ---- Part 1: the first version of the agent; it has exactly four bugs ----
def answer_question(question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    response = client.create(model=MODEL_NAME, system=SYSTEM_PROMPT, messages=messages,
                              tools=[GET_QUOTE_TOOL], max_tokens=MAX_TOKENS)

    if response.stop_reason == "tool_use":
        tool_use = next(b for b in response.content if b["type"] == "tool_use")
        text, is_error = run_tool(tool_use["name"], tool_use["input"])
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use["name"],
                         "content": text, "is_error": is_error}],
        })
        response = client.create(model=MODEL_NAME, system=SYSTEM_PROMPT, messages=messages,
                                  tools=[GET_QUOTE_TOOL], max_tokens=MAX_TOKENS)

    return response.content[0]["text"]


# ---- Part 2 ----
def run_agent(question: str, tools: list[dict], system: str, max_calls: int = 8) -> str:
    """The loop with the tools, system prompt and call cap as parameters; f"gave up after {max_calls} calls" at the cap."""
    raise NotImplementedError


# ---- Part 3 ----
def get_quotes(lookups: list[dict]) -> list[float]:
    """One price per {"ticker", "date"} pair, in order; fails at the first invalid pair the way get_quote does."""
    raise NotImplementedError
