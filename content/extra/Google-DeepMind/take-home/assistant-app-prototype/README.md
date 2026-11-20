# Take-home: a working assistant app prototype in a few hours (streaming, one tool, failure handling)

## Problem

This is a take-home assignment: in a few hours, build a working chat assistant backend and hand in a code repository.
It doesn't test algorithms; it tests whether you can finish something small but complete in limited time: a sensible
scope, failure cases thought through, trustworthy tests, and something others can run as soon as they get it.

**About the source.** There is only one lead: a public code repository online that describes itself as a "Gemini app
prototype for DeepMind interview demo". From this we infer that at least one interview asked the candidate to build a
small runnable app, but that is only an inference. The original prompt, the time limit, and the grading criteria were
not recorded. The requirements below are ones we wrote, following the most common form of this kind of assignment.

**What to build.** A minimal chat assistant service:

- It has conversation state: across multiple turns in the same conversation, the model can see what came before.
- Answers are returned by streaming: whenever the model generates a piece of text, the client receives it right away,
  without waiting for the whole answer to be written.
- The model can call one tool: a calculator.
- It handles failures: the model timing out, the model producing a malformed tool call, and the tool itself failing.
- It has tests, and a README that explains how to run it, why it is designed this way, and what was cut.

A web UI is not required. If you have time you can add the simplest possible page, but it earns no points.

**Timing and rules (assumptions set for this practice).**

- 4 hours in total; hand in a git repository.
- The reviewer starts from a clean checkout on their own machine. Starting the service must take one command and running
  the tests must take one command, and neither may **need an API key or network access**.
- Connecting a real model (for example Gemini via the Google Gen AI SDK) is a bonus. The API key may only be read from an
  environment variable and must not be committed to the repository.
- Any language and framework; the fewer dependencies the better. The reference solution uses only the Python 3.11
  standard library.
- No AI coding assistants. DeepMind's official interview guide states that unless you are told otherwise, you should not
  use any AI tools in interview tasks. Your app calling Gemini is part of the assignment itself; whether you may use AI
  to help write the code is up to the recruiter's instructions.

**Interface contract.** Request bodies and ordinary responses are JSON.

| Request | On success | Notes |
| --- | --- | --- |
| `POST /v1/conversations` | `201`, `{"id": "<32 lowercase hex characters>"}` | Create a conversation. The request body may be empty; its contents are ignored |
| `POST /v1/conversations/{id}/messages`, body `{"text": "..."}` | `200`, `Content-Type: text/event-stream` | Send a user message; the turn is streamed back |
| `GET /v1/conversations/{id}` | `200`, `{"id": "...", "messages": [...]}` | The conversation's committed history; the entry format is up to you |

If a request is bad, return `{"error": "<a one-sentence explanation>"}` with the status codes below. In none of these
cases may the model be called.

- `404`: the path is wrong, or the conversation does not exist.
- `400`: the request body is not a JSON object; or `text` is not a string, is empty after stripping leading and trailing
  whitespace, or is longer than 4000 characters.
- `409`: this conversation's previous turn has not finished. Only one turn per conversation can run at a time; different
  conversations don't affect each other.

The streaming response uses the SSE (Server-Sent Events) format: each event is two lines, `event: <name>` and
`data: <JSON>`, followed by a blank line. When the turn ends, the server closes the connection. There are five events:

| Event | data | When it is sent |
| --- | --- | --- |
| `text` | `{"delta": "..."}` | Forwarded immediately whenever the model produces a piece of text |
| `tool_call` | `{"id", "name", "arguments"}` | The model requests a tool call. `arguments` is the raw JSON text the model produced, as a string, not yet parsed |
| `tool_result` | `{"id", "name", "ok", "content"}` | The tool has finished. When `ok` is false, `content` is an error explanation for the model |
| `error` | `{"code", "message"}` | The turn failed. The values of `code` are listed in Part 3 |
| `done` | `{"ok": true}` or `{"ok": false}` | Exactly one per turn, and always the last |

An example. The user sends `{"text": "What is (2+3)*4?"}`; the model first requests the calculator, and after getting the
result outputs its text in two pieces. The response body is:

```text
event: tool_call
data: {"id": "c1", "name": "calculator", "arguments": "{\"expression\": \"(2+3)*4\"}"}

event: tool_result
data: {"id": "c1", "name": "calculator", "ok": true, "content": {"value": 20}}

event: text
data: {"delta": "(2+3)*4 = "}

event: text
data: {"delta": "20."}

event: done
data: {"ok": true}
```

The whitespace inside the JSON doesn't have to match this.

### Part 1 — The conversation loop and the calculator

**The calculator.** The tool is named `calculator`, its arguments are `{"expression": <string>}`, and it returns
`{"value": <number>}`.

- Supports `+ - * / // % **`, unary plus and minus, and parentheses. Numbers can be integers or decimals, including
  forms like `1e3`. Arithmetic follows Python's rules: integer operations give integers, and `/` gives a float.
- Everything else is rejected: variable names, function calls, attributes, strings, `True` and `False`, complex numbers,
  comparisons, and so on. **You may not use `eval`.**
- Guard against blow-ups: reject an expression longer than 200 characters; reject a `**` whose exponent has absolute
  value greater than 64; reject if any step produces an integer with absolute value greater than $10^{100}$, or a
  float that is not finite.
- Division by zero (including `%` and `//`), or a complex result (such as `(-8) ** 0.5`), is also rejected.
- On rejection, give a one-sentence explanation, such as `division by zero`, and hand it back to the model as an
  `ok: false` tool result.

**The conversation loop.** The flow of one turn:

1. Give the model the conversation history plus this user message, and forward the model's output to the client as it
   is produced.
2. If the model requested tools (it may request several at once), execute them in order, send a `tool_result` event for
   each result, then call the model again with these results, and go back to step 1.
3. When the model no longer requests tools, the turn ends successfully.

Two rules:

- **At most 4 tool rounds per turn.** The model requesting tools, executing them, and handing the results back to the
  model counts as one round, no matter how many tools were requested in it. So a turn calls the model at most 5 times.
  If the 5th model call still requests tools: send its `tool_call` events as usual, but don't execute them, and the turn
  fails with `too_many_tool_rounds`.
- **Commit only on success.** Only when a turn ends successfully is all of its content (the user message, the model
  output, the tool results) written into the conversation history. If a turn fails, the conversation history stays
  unchanged, as if the turn never happened, and the model in later turns doesn't see it either.

Model calls must sit behind an interface. Tests use a fake model and no network. How you define the interface is up to
you; explain it in the README.

### Part 2 — The streaming HTTP interface

Implement the interface contract above. Three more points:

- Text must really arrive incrementally: the client can read the earlier `text` events before the model has finished
  generating. Write a test that proves this; the test may not rely on `sleep` or timing.
- A second concurrent request on the same conversation returns `409`; write a test for this.
- The service listens only on `127.0.0.1`.

### Part 3 — Failure handling and tests

When a turn fails, first send an `error` event, then `done {"ok": false}`. `code` is one of the four values in the table
below; `message` is a one-sentence, human-readable explanation, with content of your choosing.

| Situation | Handling |
| --- | --- |
| The model times out, and this model call has not produced any output yet (neither `text` nor `tool_call`) | Retry once. If the retry also times out, `model_timeout` |
| The model times out, and this call has already sent output to the client | Don't retry, or the client would see duplicated text. `model_timeout` |
| Any other failure of the model call | `model_error`, no retry |
| More than 4 tool rounds | `too_many_tool_rounds` |
| An exception the code didn't anticipate | `internal_error`. `message` must not reveal the exception's contents; details go only to the log |

The following situations do **not** count as a failed turn. Write the problem up as an `ok: false` tool result and hand
it back to the model, which decides what to do; the turn continues:

- The model requested a tool that doesn't exist;
- `arguments` is not valid JSON, or not a JSON object;
- Arguments are missing, extra, or of the wrong type (`expression` must be a string);
- The tool reports an error, such as division by zero;
- The tool raises an unexpected exception. In that case the result says `the tool failed`, without revealing the
  exception's contents, and the details go to the log.

An example. The user sends `{"text": "hi"}`, and the model times out after outputting `Let me `:

```text
event: text
data: {"delta": "Let me "}

event: error
data: {"code": "model_timeout", "message": "no response within 30 s"}

event: done
data: {"ok": false}
```

After this turn, the conversation history is still empty.

The tests must at least cover: a text-only turn; one complete tool call; each of the failures above; a failed turn not
entering the history; 400, 404, and 409 at the HTTP layer; and the stream being incremental.

### Part 4 — README

Explain clearly: how to start it, how to run the tests, how to connect a real model; the main design choices and the
reasons for them; what was cut, why it was cut, and how it would be added later.

**What the reviewer looks at.** Starting from a clean checkout, whether it runs with one command; whether the scope is
sensible, since finishing something small and complete beats half of something big; whether the model sits behind an
interface so it can be tested with a fake model; whether the failure cases above are handled and tested; whether the
tests are reliable, without `sleep` and without the network; and whether the code and README are clear.

This problem is only about "building a working prototype in a few hours". For the problems that come with scale, such
as tens of millions of users, resuming after a disconnect, context truncation, and safety filtering, see the system
design problem [Design the Gemini App](#/deepmind/system-design/gemini-app).

## Reference solution

<details>
<summary>Show the reference solution</summary>

There is no hard algorithm in this problem. The points come from three things: keeping the scope under control,
thinking the failure cases through, and trustworthy tests. The reference solution's application code is about 400
lines and its tests about 200 lines, using only the standard library. The reviewer runs `python app.py` to start the
service and `python -m unittest` to run the tests, neither of which uses the network.

The repository has four files: `assistant.py` (the conversation logic and the HTTP layer), `app.py` (the entry point,
only a dozen or so lines), `gemini_client.py` (the adapter for the real model), and `test_assistant.py` (the tests).
This page runs them in order in one namespace, so the line at the top of each file that imports from the others is
omitted, as the code comments note.

### How to split the four hours

| Time | What to do | What you have at the end of this stretch |
| --- | --- | --- |
| 0:00–0:20 | Read the prompt. Settle three things first: the HTTP and event formats, the model interface, the format of history entries. Write the "how to run" section of the README | The interfaces are settled, so no rework later |
| 0:20–1:20 | Part 1: the fake model, the calculator, the conversation loop, adding unit tests as you go | A full turn runs without HTTP |
| 1:20–2:00 | Part 2: HTTP and SSE. Look at a stream once by hand with curl | `python app.py` works |
| 2:00–3:00 | Part 3: add the failure cases one by one, writing the test first for each | All tests pass |
| 3:00–3:30 | Connect the real model. If you have a key, run it once by hand; if not, write the adapter and state in the README that it "has not been run" | The adapter |
| 3:30–4:00 | Finish the README. Clone the repository into a fresh directory and run it from scratch following the README | Ready to submit |

Two disciplines. First, get the simplest turn working end to end before adding features on top. Second, add no
features in the last half hour.

### Part 1

The approach, in five decisions:

1. **The model interface has a single method.** `stream(history, tools)` yields three kinds of things in sequence:
   `TextDelta` (a piece of text), `ToolCall` (one complete tool call, with the arguments kept as raw JSON text), and
   `ModelDone` (the end, carrying a `meta` that the app doesn't try to understand). On failure it raises `ModelTimeout`
   or `ModelError`. All the details of the real model SDK are locked inside the adapter. `ToolCall` doesn't parse the
   arguments for the app, because "the arguments are not valid JSON" is exactly one of the failures to handle, and
   parsing and validation should happen in the app, in one place.
2. **The fake model plays from a script.** Each time `ScriptedModel` is called, it plays the next item in its script.
   An item can contain events, an exception to raise, or a `Gate`. It also stores a copy of the history it receives on
   each call, so tests can check "what the model saw". A `Gate` makes the script stop partway and wait for the test to
   release it. Part 2 uses it to prove the stream is incremental.
3. **A turn is a generator.** `Assistant.run_turn` yields events `{"event": ..., "data": ...}` one by one and knows
   nothing about HTTP. So the tests of the conversation logic don't need to start a server, and the HTTP layer is only
   responsible for writing events out as SSE.
4. **Accumulate first, commit on success.** The turn's entries first go into a local variable `pending`. When the
   model's final reply no longer requests tools, they are written into the history with a single `extend`. Any failure
   just discards `pending`. With this one rule, "a failed turn doesn't enter the history" and "the history never holds
   a tool call without its result" both hold automatically, and a `GET` never reads half a turn.
5. **The calculator walks the syntax tree (AST) instead of using `eval`.** After `ast.parse`, it accepts only numeric
   constants, unary plus and minus, and seven binary operators, and rejects every other node. The size of the result is
   checked at every step, so no intermediate result exceeds $10^{100}$. Raising such a number to a power with exponent
   at most 64 is cheap, so expressions like `9 ** 9 ** 9` can't hang the process (its exponent $9^9$ exceeds 64 and is
   rejected outright).

```python
# assistant.py -- everything except the entry point; standard library only (Python 3.11)
import ast
import copy
import json
import logging
import math
import operator
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator, Protocol

log = logging.getLogger("assistant")
log.addHandler(logging.NullHandler())   # quiet unless the app configures logging (main() does)


# ---- the seam: everything the app knows about a model ------------------------------------------
@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON text, exactly as the model produced it; the app parses and validates it


@dataclass(frozen=True)
class ModelDone:
    meta: dict = field(default_factory=dict)    # opaque to the app; the client may use it next time (an interaction id)


class ModelError(Exception):
    """A model call failed. `code` is what the client of our API gets to see."""
    code = "model_error"


class ModelTimeout(ModelError):
    code = "model_timeout"


class ModelClient(Protocol):
    def stream(self, history: list[dict], tools: list[dict]) -> Iterator[TextDelta | ToolCall | ModelDone]:
        """Stream one model response to `history`. Raise ModelTimeout / ModelError on failure."""


# ---- a deterministic fake for tests and for running without a key -----------------------------
@dataclass
class Gate:
    """In a script: wait here until the test opens the gate (lets a test look at a half-finished stream)."""
    event: threading.Event = field(default_factory=threading.Event)
    opened: list = field(default_factory=list)  # True if the test opened it, False if we gave up waiting


class ScriptedModel:
    """Each stream() call plays the next entry of the script: a list of events, Gates and exceptions to raise."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []                           # deep copies of the history of every call

    def stream(self, history, tools):
        self.seen.append(copy.deepcopy(history))
        if not self.script:
            raise ModelError("scripted model has no more responses")
        for item in self.script.pop(0):
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, Gate):
                item.opened.append(item.event.wait(timeout=30))
                continue
            yield item


# ---- tools --------------------------------------------------------------------------------------
class ToolError(Exception):
    """A failure the model should be told about (bad input, division by zero, ...)."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict        # a small JSON-schema subset: object with typed properties
    fn: Callable[..., object]

    def declaration(self):
        return {"type": "function", "name": self.name, "description": self.description, "parameters": self.parameters}


_TYPES = {"string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,)}


def check_arguments(schema: dict, args) -> str | None:
    """Return a message saying what is wrong with `args`, or None if they fit the schema."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            return f"missing required argument {name!r}"
    for name, value in args.items():
        if name not in props:
            return f"unexpected argument {name!r}"
        want = props[name]["type"]
        if (isinstance(value, bool) and want != "boolean") or not isinstance(value, _TYPES[want]):
            return f"argument {name!r} must be a {want}"
    return None


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPRESSION = 200
MAX_EXPONENT = 64
MAX_INT = 10 ** 100


def _checked(value):
    if isinstance(value, complex):
        raise ToolError("result is not a real number")
    if (isinstance(value, float) and not math.isfinite(value)) or (isinstance(value, int) and abs(value) > MAX_INT):
        raise ToolError("result is too large")
    return value


def calculate(expression: str) -> dict:
    """Arithmetic on numbers with + - * / // % ** and parentheses. Never calls eval()."""
    if len(expression) > MAX_EXPRESSION:
        raise ToolError(f"expression is longer than {MAX_EXPRESSION} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError("not a valid arithmetic expression") from None

    def ev(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return _checked(node.value)         # a literal can already be too large: 1e999 parses as inf
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _checked(_UNARY[type(node.op)](ev(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ToolError(f"exponent larger than {MAX_EXPONENT}")
            try:
                return _checked(_BINARY[type(node.op)](left, right))
            except ZeroDivisionError:
                raise ToolError("division by zero") from None
            except OverflowError:
                raise ToolError("result is too large") from None
        raise ToolError(f"unsupported syntax: {type(node).__name__}")

    return {"value": ev(tree.body)}


CALCULATOR = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression. Supports + - * / // % ** and parentheses.",
    parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    fn=calculate,
)


# ---- conversations ----------------------------------------------------------------------------
class Conversation:
    def __init__(self):
        self.id = uuid.uuid4().hex
        self.history: list[dict] = []    # committed turns only
        self.busy = threading.Lock()     # one turn at a time per conversation


class Store:
    """In memory. A real service would put this behind the same two methods and use a database."""

    def __init__(self):
        self._items: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def create(self) -> Conversation:
        conv = Conversation()
        with self._lock:
            self._items[conv.id] = conv
        return conv

    def get(self, conv_id: str) -> Conversation | None:
        with self._lock:
            return self._items.get(conv_id)


def event(kind: str, /, **data) -> dict:      # positional-only: a tool call carries a field called "name"
    return {"event": kind, "data": data}


class Assistant:
    def __init__(self, model: ModelClient, tools=(CALCULATOR,), max_tool_rounds=4, retries_on_timeout=1):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.declarations = [t.declaration() for t in tools]
        self.max_tool_rounds = max_tool_rounds
        self.retries_on_timeout = retries_on_timeout

    def run_turn(self, conv: Conversation, user_text: str) -> Iterator[dict]:
        """Yield the events of one turn; the last event is always `done`.
        The turn is added to conv.history only if it finishes. The caller holds conv.busy."""
        pending = [{"type": "user", "text": user_text}]
        try:
            for round_no in range(self.max_tool_rounds + 1):
                entry = yield from self._call_model(conv.history + pending)
                pending.append(entry)
                if not entry["tool_calls"]:
                    conv.history.extend(pending)
                    yield event("done", ok=True)
                    return
                if round_no == self.max_tool_rounds:
                    break
                for call in entry["tool_calls"]:
                    result = self._run_tool(ToolCall(**call))
                    pending.append(result)
                    yield event("tool_result", **{k: v for k, v in result.items() if k != "type"})
            yield event("error", code="too_many_tool_rounds",
                        message=f"the model asked for tools more than {self.max_tool_rounds} times in one turn")
        except ModelError as exc:
            yield event("error", code=exc.code, message=str(exc) or exc.code)
        except Exception:
            log.exception("turn failed")
            yield event("error", code="internal_error", message="internal error")
        yield event("done", ok=False)

    def _call_model(self, history):
        """One model response, streamed out as events. Retry a timeout only if nothing was shown yet."""
        for attempt in range(self.retries_on_timeout + 1):
            text, calls, meta, shown = [], [], {}, False
            try:
                for item in self.model.stream(history, self.declarations):
                    if isinstance(item, TextDelta):
                        shown = True
                        text.append(item.text)
                        yield event("text", delta=item.text)
                    elif isinstance(item, ToolCall):
                        shown = True
                        calls.append(asdict(item))
                        yield event("tool_call", **asdict(item))
                    elif isinstance(item, ModelDone):
                        meta = item.meta
                    else:                        # fail loudly: silently dropping output would hide a broken client
                        raise TypeError(f"model client produced {type(item).__name__}")
                break
            except ModelTimeout:
                if shown or attempt == self.retries_on_timeout:
                    raise
                log.warning("model timed out before any output; retrying")
        return {"type": "model", "text": "".join(text), "tool_calls": calls, "meta": meta}

    def _run_tool(self, call: ToolCall) -> dict:
        def result(ok, content):
            return {"type": "tool_result", "id": call.id, "name": call.name, "ok": ok, "content": content}

        tool = self.tools.get(call.name)
        if tool is None:
            return result(False, f"unknown tool {call.name!r}")
        try:
            args = json.loads(call.arguments)
        except json.JSONDecodeError:
            return result(False, "arguments are not valid JSON")
        problem = check_arguments(tool.parameters, args)
        if problem:
            return result(False, problem)
        try:
            return result(True, tool.fn(**args))
        except ToolError as exc:
            return result(False, str(exc))
        except Exception:
            log.exception("tool %s crashed", call.name)
            return result(False, "the tool failed")
```

### Part 2

The approach:

- **Why SSE.** You POST a message, and the response body is that turn's event stream. It is one-way, runs over plain
  HTTP, can be watched with curl, and browsers have a ready-made parser. WebSocket is two-way, which isn't needed here,
  and it adds a layer of connection state to manage.
- **Why `ThreadingHTTPServer`.** It is in the standard library, with one thread per connection. A streaming response
  doesn't end until the model is done talking, and it occupies its thread the whole time, so a single-threaded server
  won't do. A few dozen concurrent connections are enough at the prototype stage; the README states this limit.
- **One lock per conversation.** Each conversation has a `Lock`, taken with `acquire(blocking=False)`; if it can't be
  taken, reply `409` instead of queueing. Queueing would leave the client hanging on a request with no visible progress;
  replying `409` lets the client decide what to do.
- **Read the whole request body before replying with an error.** HTTP/1.1 connections can be reused. If you reply 400
  or 404 without reading the whole body, the remaining bytes are taken as the start of the next request. So the first
  thing `do_POST` does is read the entire request body, capped at 64 KB; if the length is invalid, it replies with an
  error and closes the connection.
- **How the stream ends.** The SSE response doesn't send `Content-Length`; it sends `Connection: close`, and the
  connection closing means the turn is over. Each event is `flush`ed as soon as it is written. If a write finds the
  client has already disconnected, the turn's generator is closed and the turn is not committed.

```python
# assistant.py, continued: the HTTP layer
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 64 * 1024
MAX_TEXT = 4000
CONV_PATH = re.compile(r"/v1/conversations/([0-9a-f]{32})")
MESSAGES_PATH = re.compile(r"/v1/conversations/([0-9a-f]{32})/messages")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):            # keep test output quiet; a real service logs structured lines
        log.debug("%s " + fmt, self.address_string(), *args)

    def send_json(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def read_body(self):
        """Read the whole body first, so that an early error reply never leaves unread bytes on the connection."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self.close_connection = True
            self.send_json(413 if length > MAX_BODY else 400, {"error": "bad or too large Content-Length"})
            return None
        return self.rfile.read(length)

    def do_GET(self):
        m = CONV_PATH.fullmatch(self.path)
        conv = m and self.server.store.get(m.group(1))
        if not conv:
            return self.send_json(404, {"error": "not found"})
        history = list(conv.history)              # a turn is added with one extend() at its end: no half turns here
        messages = [{k: v for k, v in e.items() if k != "meta"} for e in history]
        self.send_json(200, {"id": conv.id, "messages": messages})

    def do_POST(self):
        raw = self.read_body()
        if raw is None:
            return
        if self.path == "/v1/conversations":     # no options yet; the body is ignored
            return self.send_json(201, {"id": self.server.store.create().id})
        m = MESSAGES_PATH.fullmatch(self.path)
        conv = m and self.server.store.get(m.group(1))
        if not conv:
            return self.send_json(404, {"error": "not found"})
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            return self.send_json(400, {"error": f"body must be {{\"text\": ...}}, 1 to {MAX_TEXT} characters"})
        if not conv.busy.acquire(blocking=False):
            return self.send_json(409, {"error": "a turn is already running in this conversation"})
        try:
            self.stream_turn(conv, text)
        finally:
            conv.busy.release()

    def stream_turn(self, conv, text):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")   # no Content-Length: the end of the stream is the end of the body
        self.end_headers()
        self.close_connection = True
        turn = self.server.assistant.run_turn(conv, text)
        try:
            for ev in turn:
                self.wfile.write(f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log.info("client went away during a turn; the turn is dropped")
        finally:
            turn.close()                          # an unfinished turn is never committed


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, assistant, store, host="127.0.0.1", port=0):
        super().__init__((host, port), Handler)
        self.assistant, self.store = assistant, store


def start_server(assistant, store=None):
    """Start on 127.0.0.1 with an ephemeral port, in a background thread. Returns the server."""
    server = AppServer(assistant, store or Store())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
```

The entry point lives separately in `app.py`. If `main()` were written in `assistant.py`, then when started with
`python assistant.py` that file runs as `__main__`; the adapter's `from assistant import ...` would import a second
copy of it, the two `TextDelta`s would be two different classes, and every `isinstance` check would miss. That is why
`_call_model` raises an error as soon as it meets something it doesn't recognize (which becomes `internal_error`),
instead of silently dropping the model's output.

```python
# app.py -- the one command: `python app.py`. In the repo it begins with `from assistant import *`;
# keeping the entry point out of assistant.py means gemini_client.py and the tests import the same module.
class EchoModel:
    """What `python app.py` talks to when no real model is configured."""

    def stream(self, history, tools):
        yield TextDelta(f"(fake model) you said: {history[-1]['text']}")
        yield ModelDone()


def main():
    import os
    logging.basicConfig(level=logging.INFO)
    model = EchoModel()
    if os.environ.get("MODEL_BACKEND") == "gemini":
        from gemini_client import GeminiClient    # the adapter shown further down; needs `pip install google-genai`
        model = GeminiClient(model_name=os.environ["GEMINI_MODEL"])
    server = AppServer(Assistant(model), Store(), port=int(os.environ.get("PORT", "8000")))
    print(f"listening on http://127.0.0.1:{server.server_address[1]}")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

An actual run on a local machine (with `PORT=0` so the system picks the port; below, the port is rewritten as 8000 and
the conversation id is truncated):

```text
$ python app.py
listening on http://127.0.0.1:8000
$ curl -s -X POST http://127.0.0.1:8000/v1/conversations
{"id": "0e9aaf3ff594..."}
$ curl -s -N -X POST -d '{"text": "hello there"}' http://127.0.0.1:8000/v1/conversations/0e9aaf3ff594.../messages
event: text
data: {"delta": "(fake model) you said: hello there"}

event: done
data: {"ok": true}
```

### Part 3

The approach: one test per failure, with the test name stating the behavior it guarantees. A few points worth
mentioning:

- **How to prove the stream is incremental without relying on timing.** The fake model outputs `first ` and then stops
  at a `Gate`. The test first reads the `first ` event from the HTTP response; at that moment the model is still stuck,
  so the event can only have been sent by the server as it was produced. Then, while the model is stuck, the test sends
  another request and confirms it gets `409`, and only then releases the gate. The `Gate` waits at most 30 seconds,
  which is just insurance against the test hanging; the test asserts `gate.opened == [True]`, i.e. it really was the
  test that released it, not a timeout.
- **Why a timeout is retried only when there has been no output.** Text already sent to the client can't be taken
  back. A retry makes the model say everything again from the start, and the client sees the beginning twice. So
  "has this call output anything yet" has to be tracked separately (`shown` in the code).
- **Log assertions.** The two tests for a crashing tool and for a code error use `assertLogs` to check that the details
  went to the log, and also assert that nothing returned to the client contains the exception information.

```python
# test_assistant.py -- in the repo this file starts with `from assistant import *`; run with `python -m unittest`
import http.client
import sys
import unittest


def calc_call(expression, call_id="c1"):
    return ToolCall(id=call_id, name="calculator", arguments=json.dumps({"expression": expression}))


def run(assistant, conv, text):
    events = list(assistant.run_turn(conv, text))
    assert [e["event"] for e in events].count("done") == 1 and events[-1]["event"] == "done"
    return events


def names(events):
    return [e["event"] for e in events]


class CalculatorTest(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculate("(2 + 3) * 4")["value"], 20)
        self.assertEqual(calculate("7 // 2 + 7 % 2 - -1")["value"], 5)
        self.assertEqual(calculate("2 ** 10 / 4")["value"], 256.0)

    def test_rejected_inputs(self):
        for expr, message in [("1 / 0", "division by zero"), ("2 +", "not a valid"),
                              ("__import__('os').system('true')", "unsupported syntax"), ("x + 1", "unsupported syntax"),
                              ("True + 1", "unsupported syntax"), ("2 ** 65", "exponent"), ("9 ** 64 * 9 ** 64", "too large"),
                              ("(-8) ** 0.5", "not a real number"), ("1e308 * 10", "too large"), ("1" * 201, "longer than")]:
            with self.subTest(expr=expr[:30]):
                with self.assertRaisesRegex(ToolError, message):
                    calculate(expr)


class TurnTest(unittest.TestCase):
    def setUp(self):
        self.conv = Conversation()

    def test_plain_answer_is_streamed_and_committed(self):
        model = ScriptedModel([[TextDelta("Hel"), TextDelta("lo"), ModelDone()]])
        events = run(Assistant(model), self.conv, "hi")
        self.assertEqual(names(events), ["text", "text", "done"])
        self.assertEqual(self.conv.history, [{"type": "user", "text": "hi"},
                                             {"type": "model", "text": "Hello", "tool_calls": [], "meta": {}}])

    def test_tool_call_round_trip(self):
        model = ScriptedModel([[calc_call("(2 + 3) * 4"), ModelDone()], [TextDelta("It is 20."), ModelDone()]])
        events = run(Assistant(model), self.conv, "what is (2+3)*4?")
        self.assertEqual(names(events), ["tool_call", "tool_result", "text", "done"])
        self.assertEqual(events[1]["data"], {"id": "c1", "name": "calculator", "ok": True, "content": {"value": 20}})
        self.assertEqual(model.seen[1][-1]["content"], {"value": 20})          # the model saw the result
        self.assertEqual([e["type"] for e in self.conv.history], ["user", "model", "tool_result", "model"])

    def test_bad_tool_calls_go_back_to_the_model(self):
        bad = [ToolCall("a", "calculator", "{not json"), ToolCall("b", "weather", "{}"),
               ToolCall("c", "calculator", '{"expression": 5}'), ToolCall("d", "calculator", "{}"),
               ToolCall("e", "calculator", '{"expression": "1", "x": 1}'), ToolCall("f", "calculator", "[1]"),
               calc_call("1/0", "g")]
        model = ScriptedModel([bad + [ModelDone()], [TextDelta("Sorry, I could not compute that."), ModelDone()]])
        events = run(Assistant(model), self.conv, "go")
        results = [e["data"] for e in events if e["event"] == "tool_result"]
        self.assertEqual([r["ok"] for r in results], [False] * 7)
        self.assertEqual([r["content"] for r in results], [
            "arguments are not valid JSON", "unknown tool 'weather'", "argument 'expression' must be a string",
            "missing required argument 'expression'", "unexpected argument 'x'", "arguments must be a JSON object",
            "division by zero"])
        self.assertEqual(events[-1]["data"], {"ok": True})                     # the turn still finishes
        self.assertEqual(len(model.seen[1]), 1 + 1 + 7)                        # user, model, seven results

    def test_crashing_tool_is_reported_without_internals(self):
        def broken(expression):
            raise KeyError("secret internal detail")
        tool = Tool("calculator", "broken", CALCULATOR.parameters, broken)
        model = ScriptedModel([[calc_call("1"), ModelDone()], [TextDelta("ok"), ModelDone()]])
        with self.assertLogs("assistant", "ERROR"):                            # the details go to the log only
            events = run(Assistant(model, tools=[tool]), self.conv, "go")
        self.assertEqual(events[1]["data"]["content"], "the tool failed")
        self.assertNotIn("secret", json.dumps(events))

    def test_timeout_before_output_is_retried_once(self):
        model = ScriptedModel([[ModelTimeout("slow")], [TextDelta("fine"), ModelDone()]])
        with self.assertLogs("assistant", "WARNING"):
            events = run(Assistant(model), self.conv, "hi")
        self.assertEqual(names(events), ["text", "done"])
        self.assertEqual(len(model.seen), 2)

    def test_timeout_twice_gives_up(self):
        model = ScriptedModel([[ModelTimeout("slow")], [ModelTimeout("slow")], [TextDelta("never")]])
        with self.assertLogs("assistant", "WARNING"):
            events = run(Assistant(model), self.conv, "hi")
        self.assertEqual(names(events), ["error", "done"])
        self.assertEqual(events[0]["data"]["code"], "model_timeout")
        self.assertEqual(self.conv.history, [])

    def test_timeout_after_output_is_not_retried(self):
        model = ScriptedModel([[TextDelta("par"), ModelTimeout("slow")], [TextDelta("again")]])
        events = run(Assistant(model), self.conv, "hi")
        self.assertEqual(names(events), ["text", "error", "done"])
        self.assertEqual(events[-1]["data"], {"ok": False})
        self.assertEqual(len(model.seen), 1)                                   # no duplicated text on screen
        self.assertEqual(self.conv.history, [])

    def test_failed_turn_leaves_no_dangling_tool_call(self):
        model = ScriptedModel([[calc_call("1+1"), ModelDone()], [ModelError("upstream 500")],
                               [TextDelta("hello again"), ModelDone()]])
        assistant = Assistant(model, retries_on_timeout=0)
        first = run(assistant, self.conv, "one")
        self.assertEqual(first[-2]["data"], {"code": "model_error", "message": "upstream 500"})
        self.assertEqual(self.conv.history, [])
        run(assistant, self.conv, "two")
        self.assertEqual(model.seen[2], [{"type": "user", "text": "two"}])     # the failed turn is gone

    def test_tool_loop_is_bounded(self):
        model = ScriptedModel([[calc_call("1"), ModelDone()]] * 3)
        events = run(Assistant(model, max_tool_rounds=2), self.conv, "loop")
        self.assertEqual(names(events), ["tool_call", "tool_result"] * 2 + ["tool_call", "error", "done"])
        self.assertEqual(events[-2]["data"]["code"], "too_many_tool_rounds")
        self.assertEqual(len(model.seen), 3)
        self.assertEqual(self.conv.history, [])

    def test_bug_in_model_client_becomes_internal_error(self):
        model = ScriptedModel([[RuntimeError("bug")]])
        with self.assertLogs("assistant", "ERROR"):
            events = run(Assistant(model), self.conv, "hi")
        self.assertEqual(events[0]["data"], {"code": "internal_error", "message": "internal error"})


def read_sse(resp):
    """Yield (event, data) pairs from a text/event-stream response, as they arrive."""
    name = None
    while True:
        line = resp.readline()
        if not line:
            return
        line = line.decode().rstrip("\n")
        if line.startswith("event: "):
            name = line[len("event: "):]
        elif line.startswith("data: "):
            yield name, json.loads(line[len("data: "):])


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.model = ScriptedModel([])
        self.server = start_server(Assistant(self.model))
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        payload = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
        return conn, conn.getresponse()

    def new_conversation(self):
        conn, resp = self.request("POST", "/v1/conversations")
        self.assertEqual(resp.status, 201)
        conv_id = json.loads(resp.read())["id"]
        conn.close()
        return conv_id

    def test_turn_over_http(self):
        self.model.script += [[calc_call("6*7"), ModelDone()], [TextDelta("42"), ModelDone()]]
        conv_id = self.new_conversation()
        conn, resp = self.request("POST", f"/v1/conversations/{conv_id}/messages", {"text": "6 times 7?"})
        self.assertEqual((resp.status, resp.getheader("Content-Type")), (200, "text/event-stream"))
        events = list(read_sse(resp))
        conn.close()
        self.assertEqual([e for e, _ in events], ["tool_call", "tool_result", "text", "done"])
        conn, resp = self.request("GET", f"/v1/conversations/{conv_id}")
        messages = json.loads(resp.read())["messages"]
        conn.close()
        self.assertEqual([m["type"] for m in messages], ["user", "model", "tool_result", "model"])
        self.assertNotIn("meta", messages[1])

    def test_client_errors(self):
        conv_id = self.new_conversation()
        missing = "0" * 32
        for method, path, kwargs, status in [
                ("GET", f"/v1/conversations/{missing}", {}, 404),
                ("POST", f"/v1/conversations/{missing}/messages", {"body": {"text": "hi"}}, 404),
                ("POST", "/v1/elsewhere", {"body": {}}, 404),
                ("POST", f"/v1/conversations/{conv_id}/messages", {"raw": b"{oops"}, 400),
                ("POST", f"/v1/conversations/{conv_id}/messages", {"body": ["hi"]}, 400),
                ("POST", f"/v1/conversations/{conv_id}/messages", {"body": {"text": "  "}}, 400),
                ("POST", f"/v1/conversations/{conv_id}/messages", {"body": {"text": "x" * 4001}}, 400)]:
            with self.subTest(path=path, status=status):
                conn, resp = self.request(method, path, **kwargs)
                self.assertEqual(resp.status, status)
                self.assertIn("error", json.loads(resp.read()))
                conn.close()
        self.assertEqual(self.model.seen, [])                                  # none of these reached the model

    def test_streams_before_the_model_finishes_and_rejects_a_second_turn(self):
        gate = Gate()
        self.model.script += [[TextDelta("first "), gate, TextDelta("second"), ModelDone()]]
        conv_id = self.new_conversation()
        conn, resp = self.request("POST", f"/v1/conversations/{conv_id}/messages", {"text": "hi"})
        stream = read_sse(resp)
        self.assertEqual(next(stream), ("text", {"delta": "first "}))          # arrived while the model is blocked
        conn2, resp2 = self.request("POST", f"/v1/conversations/{conv_id}/messages", {"text": "again"})
        self.assertEqual(resp2.status, 409)
        conn2.close()
        gate.event.set()
        self.assertEqual([e for e, _ in stream], ["text", "done"])
        conn.close()
        self.assertEqual(gate.opened, [True])                                  # the test, not a timeout, opened it
        self.assertEqual(len(self.model.seen), 1)                              # the rejected turn never reached it


def run_tests(verbosity=1):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (CalculatorTest, TurnTest, HttpTest))
    return unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout).run(suite)


result = run_tests()
assert result.wasSuccessful() and result.testsRun == 15
```

### Connecting a real model

Below is the adapter for Gemini, implementing the same `stream(history, tools)` interface. It uses the Interactions API
of the Google Gen AI SDK. It is written following the official docs as of 2026-11-17
(`ai.google.dev/gemini-api/docs/...` now redirects to `aistudio.google.com/docs/...`):

- Streaming events and the streaming loop with tools: https://aistudio.google.com/docs/streaming
- Function calling, and how to hand tool results back to the model: https://aistudio.google.com/docs/function-calling
- API reference: https://aistudio.google.com/api-reference/interactions-api
- Also read: https://aistudio.google.com/docs/libraries , https://aistudio.google.com/docs/interactions-overview ,
  https://aistudio.google.com/docs/troubleshooting

This code **was not run on this page** and **has never connected to the real service**: there is no API key here, and
the SDK is not installed. We did only two checks: feeding it a fake event stream built to the documented format, and
checking the requests it sends and the events it yields; and, with a fake SDK stand-in also built to the documented
format, starting the service with `MODEL_BACKEND=gemini python app.py` and running one turn with a calculator call
through curl, to confirm the wiring works. Whether the real SDK's objects exactly match these stand-ins has not been
verified. The model name is read from the environment variable `GEMINI_MODEL`, not hard-coded.

```py
# gemini_client.py -- the real ModelClient. Not run on this page (no API key here).
# Written against the official docs as served on 2026-11-17:
#   https://aistudio.google.com/docs/streaming            stream events; the tool-call loop
#   https://aistudio.google.com/docs/function-calling     function_result input; tools resent each call
#   https://aistudio.google.com/api-reference/interactions-api
# Setup: pip install "google-genai>=2.3.0"; export GEMINI_API_KEY=...; export GEMINI_MODEL=<a current model name>
import json
import logging

from google import genai

from assistant import ModelDone, ModelError, ModelTimeout, TextDelta, ToolCall

log = logging.getLogger("assistant.gemini")
SYSTEM = "You are a concise assistant. Use the calculator tool for any arithmetic."


class GeminiClient:
    """Interactions API in stateful mode: the server keeps the conversation; we send only what is new."""

    def __init__(self, model_name, client=None):
        self.model_name = model_name                    # from GEMINI_MODEL; not hard-coded
        self.client = client or genai.Client()          # reads GEMINI_API_KEY from the environment

    def stream(self, history, tools):
        # Each model entry this client produced keeps its interaction id in meta; continue from the last one.
        marked = [i for i, e in enumerate(history) if e.get("meta", {}).get("interaction_id")]
        request = dict(model=self.model_name, system_instruction=SYSTEM, tools=tools, stream=True,
                       input=self._input(history[marked[-1] + 1:] if marked else history))
        if marked:
            request["previous_interaction_id"] = history[marked[-1]]["meta"]["interaction_id"]
        interaction_id, calls = None, []
        try:
            for ev in self.client.interactions.create(**request):
                kind = ev.event_type
                if kind == "interaction.created":
                    interaction_id = ev.interaction.id
                elif kind == "step.start" and ev.step.type == "function_call":
                    calls.append((ev.step, []))
                elif kind == "step.delta" and ev.delta.type == "text":
                    yield TextDelta(ev.delta.text)
                elif kind == "step.delta" and ev.delta.type in ("arguments_delta", "arguments") and calls:
                    # The streaming guide uses arguments_delta / .arguments; the function-calling page's Python
                    # sample uses arguments / .partial_arguments. Accept both until one is confirmed.
                    calls[-1][1].append(getattr(ev.delta, "arguments", None) or getattr(ev.delta, "partial_arguments", ""))
                elif kind == "error":
                    raise (ModelTimeout if "timeout" in (ev.error.code or "") else ModelError)(ev.error.message)
                else:
                    log.debug("skipping event %s", kind)  # the docs say new event types may be added
        except ModelError:
            raise
        except Exception as exc:                        # the pages I read name no SDK exception classes
            log.warning("Gemini call failed: %r", exc)
            raise ModelError("the model service failed") from exc
        for step, parts in calls:
            yield ToolCall(step.id, step.name, "".join(parts) or json.dumps(step.arguments or {}))
        yield ModelDone(meta={"interaction_id": interaction_id})

    @staticmethod
    def _input(entries):
        if [e["type"] for e in entries] == ["user"]:
            return entries[0]["text"]
        if entries and all(e["type"] == "tool_result" for e in entries):
            return [{"type": "function_result", "name": e["name"], "call_id": e["id"], "is_error": not e["ok"],
                     "result": [{"type": "text", "text": json.dumps(e["content"])}]} for e in entries]
        raise ModelError("this conversation was not started with the Gemini client")
```

Some notes:

- **Stateful mode.** The docs recommend the Interactions API for new projects. It can keep the conversation on the
  server: a later call passes `previous_interaction_id` and only needs to send the new content. The adapter puts each
  interaction id into `ModelDone.meta`, the app stores it in the history as is, and on the next call the adapter
  continues from the last entry that has an id. Because a failed turn is never committed, the next turn continues from
  the last successful interaction, and the failed branch on the server is left behind. The "commit only on success"
  rule fits stateful mode exactly. The cost is that the conversation content is stored on the provider's side.
- **Resend `tools` and `system_instruction` every time.** The docs say settings like these apply to a single
  interaction only and are not carried over.
- **Two places where the docs are inconsistent.** First, in the streaming guide the type of an argument fragment is
  `arguments_delta` with field `.arguments`; the Python streaming example on the function-calling page uses
  `arguments` and `.partial_arguments`, and finishes up only at `interaction.completed`. The adapter accepts both,
  emits the tool call only after the stream ends, and assumes an argument fragment belongs to the most recently started
  function call (which is how the streaming guide's example sends them, one at a time). Second, for a tool result's
  `result`, the function-calling page writes a list of content blocks, while the streaming page wraps it in
  `{"content": [...]}`; the API reference says a list, an object, or a string all work, and here we use a list. The API
  reference also has an `is_error` field, used here to tell the model the tool failed.
- **Timeouts.** None of the pages I read give a timeout parameter or name the exception classes the SDK raises. The
  troubleshooting page says the SDK retries transient errors by itself, up to four times. So the adapter maps the
  documented timeout error event (the example's code is `gateway_timeout`) to `ModelTimeout`, and wraps every other
  exception as `ModelError`. To set a deadline on each call, you would first need to look it up in the SDK's reference
  docs. Since the SDK already retries, when connecting to Gemini you can set the app-level `retries_on_timeout` to 0 so
  the retry counts don't multiply.

### Part 4

Below is a sample README. The actual submission is written in English, with the same structure.

**How to run.** `python app.py` starts the service, on port 8000 by default, which can be changed with the environment
variable `PORT`. No dependencies need to be installed and no key is needed; by default it connects to a fake model that
echoes the user's message back. `python -m unittest` runs all 15 tests, with no network and no reliance on `sleep`. To
connect Gemini: `pip install "google-genai>=2.3.0"`, set `GEMINI_API_KEY`, `GEMINI_MODEL`, and `MODEL_BACKEND=gemini`,
then run `python app.py`.

**Interface.** Three HTTP endpoints and five events (copied from the two tables in the prompt).

**Design choices.**

1. The model sits behind the single method `stream(history, tools)`. Tests use a fake model that plays from a script,
   and switching to a real model only adds an adapter.
2. A turn is a generator that yields events, independent of HTTP. The conversation logic can be tested without starting
   a server; switching to WebSocket later only changes the transport layer.
3. A turn is written into the history only if it succeeds. A failed turn leaves no half-finished content in the
   history, and the model in the next turn doesn't see it. The cost is that on failure, the partial answer already shown
   to the user isn't saved.
4. A bad tool call is handed back to the model instead of failing the turn. The model can usually fix the arguments
   itself and try again; failing outright would leave the user no choice but to ask again.
5. A model timeout is retried once, and only if there has been no output yet, so the client never sees duplicated text.
6. Only one turn runs per conversation at a time; a conflict replies `409` instead of queueing.
7. Standard library only, so the reviewer doesn't need to install anything. The cost is handling some HTTP details
   yourself, such as reading the whole request body before replying with an error.

**What was cut.**

- Persistence. Conversations live in memory and are gone on restart. `Store` has only two methods, `create` and `get`,
  so switching to a database changes only this one place.
- Authentication and rate limiting. There is no notion of a user; anyone who knows a conversation id can read and write
  it. That is why the service listens only on `127.0.0.1`.
- History truncation. The history grows without bound, and a long conversation will sooner or later exceed the model's
  context. With Gemini's stateful mode the history is on the server; when sending the history yourself, you need to
  truncate or summarize it to a token budget.
- A timeout for individual tools, and an overall deadline for a turn. The calculator has an input limit and can't run
  for long; with slow tools such as network lookups, both need to be added.
- Resuming after a disconnect. If the client disconnects, the turn is lost.
- A web UI. Demonstrated with curl.
- The real model. The adapter follows the official docs and has not been run with a real key (if it has been run with a
  key, say here what was run and what the result was).

**Known limitations.** Each connection takes a thread, so the concurrency limit depends on the number of threads. When
deploying multiple processes, the in-memory conversations and locks can't be shared.

### Follow-ups

**What happens if the client disconnects halfway through the stream?** On its next write the server may find the
connection broken, and then it closes the turn's generator and the turn is not committed. But this isn't guaranteed:
right after a disconnect, writes often still succeed into the operating system's send buffer, and it takes a few more
writes before an error is reported. So if only a few events remain, the turn may be committed as usual, and the client
can see it with `GET` after reconnecting. Reliable resumption requires giving each turn an id, storing the events, and
having the client reconnect with the number of the last event it received so the server can continue from there. For
that design, see the streaming and reconnection part of [Design the Gemini App](#/deepmind/system-design/gemini-app).

**If not `eval`, what about `eval` with the builtins removed?** No. Even with the builtins removed, you can still walk
from any object's attributes all the way to every subclass of `object` and get hold of dangerous classes. Besides,
pure arithmetic can cause trouble too: `9 ** 9 ** 9` has about 370 million decimal digits, just storing it takes more
than a hundred MB, and computing it takes far longer. Walking the syntax tree with a whitelist, plus a size limit at
every step, solves both problems at once. This page's verification code calls `eval` only on expressions it builds
itself, to check the calculator's answers against.

**After a timeout, if half the output has already been sent and you still want to recover, what do you do?** There are
two routes. One is to have the client cooperate: send a "this turn is restarting" event, the client clears what it has
already displayed, the server retries the whole turn, and the user sees the answer replaced. The other is to use the
text already generated as a prefix and have the model continue writing; this depends on whether the model and API
support it, and the interface is more complex. The prototype picks the simplest: no retry, report the error, and let
the user ask again. Also watch out for retries stacking up: the SDK itself retries up to four times, the app retries
once more, and in the worst case one request becomes ten calls.

**To deploy multiple instances, what has to change?** The in-memory `Store` is replaced with a shared database. The
per-conversation lock also has to work across instances: either route requests to a fixed instance by conversation id,
or use a distributed lock with an expiry, otherwise a crashed instance would make the conversations it held reply
`409` forever. The meaning of `409` can stay the same.

**How do you know these tests really catch bugs rather than passing by chance?** The verification code does a small
piece of mutation testing: it deliberately writes four buggy versions of `Assistant`, namely "retry even after output",
"a bad tool call crashes the turn outright", "a failed turn is also written into the history", and "no limit on tool
rounds". Each one makes at least one test fail. On top of that, the streaming test doesn't use `sleep`, so the results
don't depend on how fast the machine is.

**How do you choose between stateful mode and managing the history yourself?** Stateful mode sends only the new content
each time, so requests are small, and the model's own intermediate state (such as the signatures of its thinking) also
stays on the server and isn't lost. The downside is that the conversation is stored with the provider, retention and
deletion follow the provider's rules, and you can't migrate directly to another model provider. When you manage the
history yourself (stateless mode), the docs require sending the model's earlier output back exactly as it was,
signatures included, so history entries have to store the raw content the model returned, not just the text; the
benefit is that truncation, summarization, and migration are all under your control. For the prototype, stateful mode
is the least effort, and the README states this choice.

<details>
<summary>Checks (runnable)</summary>

```python
import io
import random

# ---- the two transcripts in the statement, byte for byte, over real HTTP ------------------------
EXAMPLE_OK = '''event: tool_call
data: {"id": "c1", "name": "calculator", "arguments": "{\\"expression\\": \\"(2+3)*4\\"}"}

event: tool_result
data: {"id": "c1", "name": "calculator", "ok": true, "content": {"value": 20}}

event: text
data: {"delta": "(2+3)*4 = "}

event: text
data: {"delta": "20."}

event: done
data: {"ok": true}

'''
EXAMPLE_TIMEOUT = '''event: text
data: {"delta": "Let me "}

event: error
data: {"code": "model_timeout", "message": "no response within 30 s"}

event: done
data: {"ok": false}

'''


def raw_turn(script, text):
    """Run one turn through a real server; return the raw response body, the status, and the committed history."""
    server = start_server(Assistant(ScriptedModel(script)))
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("POST", "/v1/conversations")
        conv_id = json.loads(conn.getresponse().read())["id"]
        assert len(conv_id) == 32 and conv_id == conv_id.lower() and int(conv_id, 16) >= 0
        conn.request("POST", f"/v1/conversations/{conv_id}/messages", body=json.dumps({"text": text}))
        resp = conn.getresponse()
        body = resp.read().decode()
        conn.close()
        return resp.status, body, server.store.get(conv_id).history
    finally:
        server.shutdown()
        server.server_close()


status, body, history = raw_turn([[calc_call("(2+3)*4"), ModelDone()],
                                  [TextDelta("(2+3)*4 = "), TextDelta("20."), ModelDone()]], "What is (2+3)*4?")
assert status == 200 and body == EXAMPLE_OK, body
assert [e["type"] for e in history] == ["user", "model", "tool_result", "model"]
status, body, history = raw_turn([[TextDelta("Let me "), ModelTimeout("no response within 30 s")]], "hi")
assert status == 200 and body == EXAMPLE_TIMEOUT and history == []

# ---- the calculator against Python's own arithmetic on random expressions (eval only on strings we built) ----
rng = random.Random(0)


def random_expression(depth=0):
    if depth > 3 or rng.random() < 0.3:
        return str(rng.randint(0, 9)) if rng.random() < 0.8 else f"{rng.randint(0, 9)}.5"
    op = rng.choice(["+", "-", "*", "/", "//", "%", "**"])
    right = str(rng.randint(0, 3)) if op == "**" else random_expression(depth + 1)
    left = random_expression(depth + 1)
    return f"-({left})" if rng.random() < 0.1 else f"({left}) {op} ({right})"


agree = errors = 0
for _ in range(3000):
    expr = random_expression()
    try:
        want = eval(expr, {"__builtins__": {}})
    except ZeroDivisionError:
        want = ZeroDivisionError
    except OverflowError:
        want = OverflowError
    try:
        got = calculate(expr)["value"]
    except ToolError as exc:
        got = exc
    if want is ZeroDivisionError:
        assert isinstance(got, ToolError) and str(got) == "division by zero", expr
        errors += 1
    elif isinstance(want, complex) or (isinstance(want, float) and not math.isfinite(want)) \
            or want is OverflowError or (isinstance(want, int) and abs(want) > MAX_INT):
        assert isinstance(got, ToolError), expr
        errors += 1
    else:
        assert not isinstance(got, ToolError) and type(got) is type(want) and got == want, (expr, got, want)
        agree += 1
print(f"calculator vs Python on 3000 random expressions: {agree} same value, {errors} rejected (zero division, complex, too large)")
assert agree > 2000 and errors > 100

# ---- the tests have teeth: each of these deliberately broken assistants fails at least one test ----
GoodAssistant = Assistant


class RetriesAfterOutput(GoodAssistant):          # retries a timeout even after text was shown
    def _call_model(self, history):
        try:
            return (yield from super()._call_model(history))
        except ModelTimeout:
            return (yield from super()._call_model(history))


class CrashesOnBadToolCall(GoodAssistant):        # raises instead of telling the model what was wrong
    def _run_tool(self, call):
        result = super()._run_tool(call)
        if not result["ok"]:
            raise ToolError(result["content"])
        return result


class CommitsFailedTurns(GoodAssistant):          # keeps half a turn in the history after a failure
    def run_turn(self, conv, user_text):
        events = list(super().run_turn(conv, user_text))
        if not events[-1]["data"]["ok"]:
            conv.history.append({"type": "user", "text": user_text})
        yield from events


class UnboundedToolLoop(GoodAssistant):           # ignores the tool-round limit
    def __init__(self, model, **kwargs):
        kwargs["max_tool_rounds"] = 100
        super().__init__(model, **kwargs)


def failing_tests(mutant):
    global Assistant
    Assistant = mutant
    try:
        result = unittest.TextTestRunner(stream=io.StringIO()).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(TurnTest))
    finally:
        Assistant = GoodAssistant
    return sorted(test.id().split(".")[-1] for test, _ in result.failures + result.errors)


assert failing_tests(GoodAssistant) == []
assert "test_timeout_after_output_is_not_retried" in failing_tests(RetriesAfterOutput)
assert "test_bad_tool_calls_go_back_to_the_model" in failing_tests(CrashesOnBadToolCall)
assert {"test_failed_turn_leaves_no_dangling_tool_call", "test_timeout_after_output_is_not_retried",
        "test_tool_loop_is_bounded"} <= set(failing_tests(CommitsFailedTurns))
assert "test_tool_loop_is_bounded" in failing_tests(UnboundedToolLoop)

# ---- small facts quoted in the text --------------------------------------------------------------
assert CALCULATOR.declaration() == {
    "type": "function", "name": "calculator",
    "description": "Evaluate an arithmetic expression. Supports + - * / // % ** and parentheses.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}
assert (MAX_BODY, MAX_TEXT, MAX_EXPRESSION, MAX_EXPONENT, MAX_INT) == (65536, 4000, 200, 64, 10 ** 100)
assert calculate("10 ** 64 * 10 ** 36")["value"] == 10 ** 100
try:
    calculate("10 ** 64 * 10 ** 37")
    raise AssertionError("should have failed")
except ToolError as exc:
    assert str(exc) == "result is too large"
assert calculate("1e3 + +(-2)")["value"] == 998.0 and calculate("7 / 2")["value"] == 3.5
assert calculate("1+" * 99 + "11")["value"] == 110                      # exactly 200 characters is allowed
for expr, message in [("1j", "unsupported"), ("'a'", "unsupported"), ("abs(-1)", "unsupported"),
                      ("(1).real", "unsupported"), ("[1]", "unsupported"), ("1 < 2", "unsupported"),
                      ("9 ** 9 ** 9", "exponent larger than 64"), ("((9 ** 64) ** 64) ** 64", "result is too large"),
                      ("5 % 0", "division by zero"), ("0.0 ** -1", "division by zero"), ("1e999", "result is too large")]:
    try:
        calculate(expr)
        raise AssertionError(expr)
    except ToolError as exc:
        assert message in str(exc), (expr, exc)
# the size quoted for 9 ** 9 ** 9 in the follow-up questions (computed from logarithms, never evaluated)
digits, megabytes = 9 ** 9 * math.log10(9), 9 ** 9 * math.log2(9) / 8 / 1e6
print(f"9 ** 9 ** 9: about {digits:.4g} decimal digits, {megabytes:.4g} MB as a binary integer")
assert 3.6e8 < digits < 3.8e8 and 100 < megabytes < 200
# a model client that yields something unexpected fails the turn loudly instead of losing output
events = list(Assistant(ScriptedModel([[TextDelta("a"), "b", ModelDone()]])).run_turn(conv := Conversation(), "x"))
assert [e["event"] for e in events] == ["text", "error", "done"] and events[1]["data"]["code"] == "internal_error"
assert conv.history == []
# default limit: four tool rounds, so a fifth request for a tool is announced but not run
events = list(Assistant(model := ScriptedModel([[calc_call("1"), ModelDone()]] * 5)).run_turn(Conversation(), "x"))
assert [e["event"] for e in events] == ["tool_call", "tool_result"] * 4 + ["tool_call", "error", "done"]
assert len(model.seen) == 5 and events[-2]["data"]["code"] == "too_many_tool_rounds"
assert Assistant(ScriptedModel([])).max_tool_rounds == 4 and Assistant(ScriptedModel([])).retries_on_timeout == 1
print("all checks passed")
```

</details>

</details>
