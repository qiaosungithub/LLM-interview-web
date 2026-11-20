# 几小时做一个能跑的助手应用原型：流式对话、一个工具、失败处理

## 题目

这是一道 take-home（带回家做的作业）：几个小时内做出一个能跑的聊天助手后端，交一个代码仓库。考的不是算法，
而是能不能在有限时间里做完一个小而完整的东西：范围定得合理，失败情况想清楚，测试可信，别人拿到就能跑。

**关于来源。** 能找到的线索只有一条：网上有一个公开的代码仓库，自我介绍是“为 DeepMind 面试演示做的 Gemini app 原型”
（Gemini app prototype for DeepMind interview demo）。由此推测，至少有一次面试要求候选人做一个小的可运行应用，但这只是推测。题目原文、
时长、评分方式都没有记录。下面的要求是我们按这类作业最常见的形式补写的。

**要做什么。** 一个最小的聊天助手服务：

- 有会话（conversation）状态：同一个会话里的多轮对话，模型能看到之前的内容。
- 回答以流式（streaming）方式返回：模型每生成一段文字，客户端马上就能收到，不用等整段写完。
- 模型可以调用一个工具：计算器。
- 能处理失败：模型超时，模型给出格式错误的工具调用，工具本身出错。
- 有测试；有一份 README，写清楚怎么跑、为什么这样设计、砍掉了什么。

不要求网页界面。时间够可以加一个最简单的页面，但它不算分。

**计时和规则（为练习定的假设）。**

- 总共 4 小时，交一个 git 仓库。
- 评审会在自己的机器上，从一份干净的代码检出（clean checkout）开始。要求一条命令启动服务，一条命令跑测试，两者都
  **不需要 API key，也不联网**。
- 接上真实模型（比如用 Google Gen AI SDK 接 Gemini）是加分项。API key 只能从环境变量读，不能提交进仓库。
- 语言和框架不限，依赖越少越好。参考解答只用 Python 3.11 的标准库。
- 不用 AI 编程助手。DeepMind 的官方面试指南写明：除非另有通知，面试任务中不要使用任何 AI 工具。你的应用调用
  Gemini 是作业本身的内容；用 AI 帮你写代码，则以招聘方的通知为准。

**接口约定。** 请求体和普通响应都是 JSON。

| 请求 | 成功时 | 说明 |
| --- | --- | --- |
| `POST /v1/conversations` | `201`，`{"id": "<32 位小写十六进制>"}` | 新建会话。请求体可以为空，内容被忽略 |
| `POST /v1/conversations/{id}/messages`，请求体 `{"text": "..."}` | `200`，`Content-Type: text/event-stream` | 发一条用户消息，流式返回这一轮（turn） |
| `GET /v1/conversations/{id}` | `200`，`{"id": "...", "messages": [...]}` | 这个会话已经提交的历史；条目格式自定 |

请求有问题时，返回 `{"error": "<一句说明>"}` 和下面的状态码。这几种情况都不能调用模型。

- `404`：路径不对，或者会话不存在。
- `400`：请求体不是 JSON 对象；或者 `text` 不是字符串、去掉首尾空白后为空、长度超过 4000 个字符。
- `409`：这个会话的上一轮还没结束。同一个会话同一时间只能有一轮在跑，不同会话互不影响。

流式响应用 SSE（Server-Sent Events）格式：每个事件是两行，`event: <名字>` 和 `data: <JSON>`，后面跟一个空行。
这一轮结束后，服务端关闭连接。事件有五种：

| 事件 | data | 什么时候发 |
| --- | --- | --- |
| `text` | `{"delta": "..."}` | 模型每产生一段文字，马上转发 |
| `tool_call` | `{"id", "name", "arguments"}` | 模型请求调用工具。`arguments` 是模型给出的原始 JSON 文本，是字符串，还没解析 |
| `tool_result` | `{"id", "name", "ok", "content"}` | 工具执行完。`ok` 为 false 时，`content` 是给模型看的错误说明 |
| `error` | `{"code", "message"}` | 这一轮失败了。`code` 的取值见 Part 3 |
| `done` | `{"ok": true}` 或 `{"ok": false}` | 每一轮恰好一个，而且总是最后一个 |

例子。用户发 `{"text": "What is (2+3)*4?"}`，模型先请求计算器，拿到结果后分两段输出文字。响应体是：

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

JSON 里的空格不必和这里一样。

### Part 1 —— 对话循环和计算器

**计算器。** 工具名 `calculator`，参数是 `{"expression": <字符串>}`，返回 `{"value": <数>}`。

- 支持 `+ - * / // % **`、一元正负号和括号。数字可以是整数或小数，包括 `1e3` 这种写法。运算规则和 Python 相同：
  整数运算得整数，`/` 得小数。
- 其他写法一律拒绝：变量名、函数调用、属性、字符串、`True` 和 `False`、复数、比较等。**不能用 `eval`。**
- 防止算爆：表达式超过 200 个字符，拒绝；`**` 的指数绝对值超过 64，拒绝；任何一步的结果是绝对值超过 $10^{100}$ 的整数，
  或者不是有限的小数，拒绝。
- 除以零（包括 `%` 和 `//`），或者结果是复数（比如 `(-8) ** 0.5`），也拒绝。
- 拒绝时给一句说明，比如 `division by zero`，作为 `ok: false` 的工具结果交回模型。

**对话循环。** 一轮的流程：

1. 把会话历史加上这条用户消息交给模型，模型的输出边产生边转发给客户端。
2. 如果模型请求了工具（一次可以请求多个），按顺序执行，每个结果发一个 `tool_result` 事件，然后带上这些结果再调用一次
   模型，回到第 1 步。
3. 模型不再请求工具，这一轮就成功结束。

两条规则：

- **一轮最多 4 个工具回合（tool round）。** 模型请求工具、执行、把结果交回模型，算一个回合，不管这一次请求了几个工具。
  所以一轮最多调用模型 5 次。如果第 5 次调用模型还要工具：照常发出它的 `tool_call` 事件，但不执行，这一轮以
  `too_many_tool_rounds` 失败。
- **成功才提交。** 一轮成功结束时，才把这一轮的全部内容（用户消息、模型输出、工具结果）写进会话历史。一轮失败了，
  会话历史保持不变，就像这一轮没有发生过，后面几轮的模型也看不到它。

模型调用必须放在一个接口后面。测试用假模型（fake），不联网。接口怎么定由你决定，在 README 里说明。

### Part 2 —— 流式 HTTP 接口

按上面的接口约定实现。另外三点：

- 文字要真的是增量到达：模型还没生成完，客户端就能读到前面的 `text` 事件。写一个测试证明这一点，测试不能靠 `sleep`
  或计时。
- 同一个会话的第二个并发请求返回 `409`，并为此写一个测试。
- 服务只监听 `127.0.0.1`。

### Part 3 —— 失败处理和测试

一轮失败时，先发一个 `error` 事件，再发 `done {"ok": false}`。`code` 取下表四个值之一；`message` 是给人看的一句话，
内容自定。

| 情况 | 处理 |
| --- | --- |
| 模型超时，而这一次模型调用还没有任何输出（既没有 `text`，也没有 `tool_call`） | 重试一次。重试还超时，`model_timeout` |
| 模型超时，而这一次调用已经有输出发给了客户端 | 不重试，否则客户端会看到重复的文字。`model_timeout` |
| 模型调用的其他失败 | `model_error`，不重试 |
| 工具回合超过 4 个 | `too_many_tool_rounds` |
| 代码里没有料到的异常 | `internal_error`。`message` 不能带出异常内容，细节只写日志 |

下面这些情况**不算**这一轮失败。把问题写成 `ok: false` 的工具结果交回模型，由模型决定怎么办，这一轮继续：

- 模型请求了不存在的工具；
- `arguments` 不是合法的 JSON，或者不是 JSON 对象；
- 参数缺了、多了，或者类型不对（`expression` 必须是字符串）；
- 工具报错，比如除以零；
- 工具抛出意料之外的异常。这时结果写 `the tool failed`，不带出异常内容，细节写日志。

例子。用户发 `{"text": "hi"}`，模型输出 `Let me ` 之后超时：

```text
event: text
data: {"delta": "Let me "}

event: error
data: {"code": "model_timeout", "message": "no response within 30 s"}

event: done
data: {"ok": false}
```

这一轮之后，会话历史仍然是空的。

测试至少要覆盖：只有文字的一轮；一次完整的工具调用；上面每一种失败；失败的一轮不进历史；HTTP 层的 400、404、409；
流是增量的。

### Part 4 —— README

写清楚：怎么启动，怎么跑测试，怎么接真实模型；几个主要的设计选择和理由；砍掉了什么，为什么砍，以后怎么补。

**评审看什么。** 从干净检出开始，一条命令能不能跑起来；范围是否合理，做完一个小而完整的东西，好过做了一半的大东西；
模型是不是放在接口后面，能用假模型测试；上面那些失败情况有没有处理，有没有测试；测试是否可靠，不靠 `sleep`，不联网；
代码和 README 是否清楚。

这道题只管“几小时做一个能跑的原型”。规模化之后的问题，比如几千万用户、断线续传、上下文截断、安全过滤，见系统设计题
[设计 Gemini App](#/deepmind/system-design/gemini-app)。

## 参考解答

<details>
<summary>展开参考解答</summary>

这道题没有难的算法。分数落在三件事上：范围控制住，失败情况想清楚，测试可信。参考解答的应用代码约 400 行，测试约
200 行，只用标准库。评审运行 `python app.py` 就能起服务，运行 `python -m unittest` 就能跑测试，都不联网。

仓库里四个文件：`assistant.py`（对话逻辑和 HTTP 层）、`app.py`（入口，只有十几行）、`gemini_client.py`（真实模型的
适配器）、`test_assistant.py`（测试）。本页把它们按顺序放在一个命名空间里运行，所以各文件开头的 `import` 互相引用
那一行省掉了，代码注释里写明了。

### 四个小时怎么分

| 时间 | 做什么 | 这一段结束时手里有什么 |
| --- | --- | --- |
| 0:00–0:20 | 读题。先定三样东西：HTTP 和事件格式，模型接口，历史条目的格式。写好 README 里“怎么跑”那一节 | 接口定了，后面不返工 |
| 0:20–1:20 | Part 1：假模型、计算器、对话循环，边写边加单元测试 | 不用 HTTP 也能跑完一轮 |
| 1:20–2:00 | Part 2：HTTP 和 SSE。用 curl 手动看一次流 | `python app.py` 能用 |
| 2:00–3:00 | Part 3：失败情况一个一个加，每加一个先写测试 | 测试全部通过 |
| 3:00–3:30 | 接真实模型。有 key 就手动跑一次；没有 key 就写好适配器，README 里写明“没有跑过” | 适配器 |
| 3:30–4:00 | 写完 README。把仓库克隆到一个新目录，照着 README 从头跑一遍 | 可以提交 |

两条纪律。第一，先让最简单的一轮从头到尾跑通，再往上加功能。第二，最后半小时不加功能。

### Part 1

思路，五个决定：

1. **模型接口只有一个方法。** `stream(history, tools)` 依次产出三种东西：`TextDelta`（一段文字）、`ToolCall`（一个完整的
   工具调用，参数保留原始 JSON 文本）、`ModelDone`（结束，带一个应用不去理解的 `meta`）。失败就抛 `ModelTimeout` 或
   `ModelError`。真实模型 SDK 的细节全部关在适配器里。`ToolCall` 不替应用解析参数，因为“参数不是合法 JSON”正是要处理的
   失败之一，解析和校验应该在应用里、在一个地方做。
2. **假模型照脚本出牌。** `ScriptedModel` 每被调用一次，就播放脚本里的下一项。一项里可以有事件、要抛出的异常，或者一个
   `Gate`。它还把每次收到的历史复制一份存下来，测试可以检查“模型看到了什么”。`Gate` 让脚本停在半路，等测试放行。
   Part 2 用它证明流是增量的。
3. **一轮就是一个生成器（generator）。** `Assistant.run_turn` 逐个产出事件 `{"event": ..., "data": ...}`，完全不知道 HTTP
   的存在。所以对话逻辑的测试不用起服务器，HTTP 层只负责把事件写成 SSE。
4. **先攒着，成功了再提交。** 这一轮的条目先放在局部变量 `pending` 里。模型的最后一次回复不再请求工具时，用一次 `extend`
   写进历史。任何失败都只是丢掉 `pending`。有了这一条，“失败的一轮不进历史”“历史里不会留下没有结果的工具调用”都
   自动成立，`GET` 也不会读到半轮。
5. **计算器走语法树（AST），不用 `eval`。** `ast.parse` 之后只认数字常量、一元正负号和七种二元运算，其余节点一律拒绝。
   每一步都检查结果大小，所以任何中间结果都不超过 $10^{100}$。在这样的数上做一次指数不超过 64 的乘方，代价很小，
   `9 ** 9 ** 9` 这类表达式卡不住进程（它的指数 $9^9$ 超过 64，直接被拒绝）。

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

思路：

- **为什么用 SSE。** POST 一条消息，响应体就是这一轮的事件流。它是单向的，跑在普通 HTTP 上，curl 就能看，浏览器也有现成的
  解析。WebSocket 是双向的，这里用不上，还要多管一层连接状态。
- **为什么用 `ThreadingHTTPServer`。** 它在标准库里，每个连接一个线程。一个流式响应要等模型说完才结束，这段时间一直占着它的
  线程，所以不能用单线程的服务器。原型阶段几十个并发够用，README 里写明这个上限。
- **一个会话一把锁。** 每个会话有一个 `Lock`，用 `acquire(blocking=False)`，拿不到就回 `409`，不排队等。排队会让客户端挂着
  一个看不到任何进度的请求；回 `409`，由客户端自己决定怎么办。
- **先读完请求体，再回错误。** HTTP/1.1 的连接可以复用。如果没读完请求体就回了 400 或 404，剩下的字节会被当成下一个请求的
  开头。所以 `do_POST` 第一步就读完整个请求体，上限 64 KB；长度不合法就回错误并关闭连接。
- **流怎么结束。** SSE 响应不写 `Content-Length`，而是发 `Connection: close`，连接关闭就表示这一轮结束。每个事件写完立刻
  `flush`。写的时候发现客户端已经断开，就关掉这一轮的生成器，这一轮不提交。

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

入口单独放在 `app.py`。如果把 `main()` 写在 `assistant.py` 里，用 `python assistant.py` 启动时，这个文件是以 `__main__` 的
身份运行的；适配器里的 `from assistant import ...` 会把它再导入一份，两份 `TextDelta` 是两个不同的类，`isinstance` 判断
全部落空。所以 `_call_model` 遇到认不出的东西会直接报错（变成 `internal_error`），而不是悄悄丢掉模型的输出。

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

在本机实际跑了一遍（端口用 `PORT=0` 让系统分配，下面把端口改写成了 8000，会话 id 截短了）：

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

思路：每一种失败对应一个测试，测试名就是要保证的行为。几个值得说的地方：

- **怎么证明流是增量的，又不靠计时。** 假模型输出 `first ` 之后停在 `Gate` 上。测试先从 HTTP 响应里读到 `first ` 这个事件，
  这时模型还卡着，所以这个事件只可能是服务端边产生边发出来的。然后测试在模型卡着的时候再发一个请求，确认得到 `409`，
  最后才放行。`Gate` 最多等 30 秒，这只是防止测试挂死的保险；测试断言 `gate.opened == [True]`，即确实是测试放行的，
  不是等超时。
- **为什么超时只在没有输出时重试。** 已经发给客户端的文字收不回来。重试会让模型从头再说一遍，客户端看到的就是两遍开头。
  所以“这一次调用有没有输出过东西”要单独记下来（代码里的 `shown`）。
- **日志断言。** 工具崩溃和代码出错这两个测试用 `assertLogs` 检查细节进了日志，同时断言返回给客户端的内容里没有异常信息。

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

### 接真实模型

下面是接 Gemini 的适配器，实现同一个 `stream(history, tools)` 接口。它用 Google Gen AI SDK 的 Interactions API。写法照
2026-11-17 的官方文档（`ai.google.dev/gemini-api/docs/...` 现在会跳转到 `aistudio.google.com/docs/...`）：

- 流式事件和带工具的流式循环：https://aistudio.google.com/docs/streaming
- 函数调用，以及怎样把工具结果交回模型：https://aistudio.google.com/docs/function-calling
- 接口参考：https://aistudio.google.com/api-reference/interactions-api
- 另外看过：https://aistudio.google.com/docs/libraries 、https://aistudio.google.com/docs/interactions-overview 、
  https://aistudio.google.com/docs/troubleshooting

这段代码**没有在本页运行**，也**没有连过真实服务**：这里没有 API key，也没有安装 SDK。我们做过的检查只有两项：用按文档
格式伪造的事件流喂给它，核对它发出的请求和产出的事件；再用一个同样按文档格式伪造的 SDK 替身，以
`MODEL_BACKEND=gemini python app.py` 起服务，用 curl 跑通一轮带计算器调用的对话，确认接线没问题。真实 SDK 的对象和
这些替身是否完全一致，没有核实。模型名从环境变量 `GEMINI_MODEL` 读，不写死。

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

几点说明：

- **有状态模式。** 文档推荐新项目用 Interactions API。它可以在服务端保存对话：后一次调用带上 `previous_interaction_id`，
  只需发送新的内容。适配器把每次的 interaction id 放进 `ModelDone.meta`，应用原样存进历史，下一次调用时适配器从最后一个
  带 id 的条目接着发。因为失败的一轮从不提交，下一轮会从最后一次成功的 interaction 接着走，服务端那条失败的分支就被
  丢下了。“成功才提交”的规则和有状态模式正好配合。代价是对话内容存在服务方那边。
- **每次都重发 `tools` 和 `system_instruction`。** 文档说这类设置只对一次 interaction 有效，不会沿用。
- **文档前后不一致的两处。** 第一，流式指南里参数片段的类型是 `arguments_delta`，字段是 `.arguments`；函数调用页的 Python
  流式示例用的是 `arguments` 和 `.partial_arguments`，并在 `interaction.completed` 时才收尾。适配器两种都接受，等流结束后
  再发出工具调用，并假设参数片段属于最近开始的那个函数调用（流式指南的示例就是这样一个一个来的）。第二，工具结果的
  `result`，函数调用页写成内容块列表，流式页包了一层 `{"content": [...]}`；接口参考说列表、对象、字符串都可以，这里用
  列表。接口参考里还有一个 `is_error` 字段，这里用来告诉模型工具失败了。
- **超时。** 我看过的这几页都没有给出超时参数，也没有写 SDK 抛出的异常类名。故障排查页说 SDK 会自己重试临时性错误，最多
  四次。所以适配器把文档里的超时错误事件（示例的 code 是 `gateway_timeout`）映射成 `ModelTimeout`，其余异常一律包成
  `ModelError`。要给每次调用设截止时间，需要先在 SDK 的参考文档里查清楚。SDK 已经在重试，接 Gemini 时可以把应用层的
  `retries_on_timeout` 设为 0，免得重试次数相乘。

### Part 4

下面是 README 的范例。实际提交用英文写，结构相同。

**怎么跑。** `python app.py` 启动服务，默认端口 8000，可以用环境变量 `PORT` 改。不需要安装任何依赖，也不需要 key，默认接
一个把用户消息原样回显的假模型。`python -m unittest` 跑全部 15 个测试，不联网，也不靠 `sleep`。接 Gemini：
`pip install "google-genai>=2.3.0"`，设置 `GEMINI_API_KEY`、`GEMINI_MODEL` 和 `MODEL_BACKEND=gemini`，再运行
`python app.py`。

**接口。** 三个 HTTP 接口和五种事件（照抄题目里的两张表）。

**设计选择。**

1. 模型放在 `stream(history, tools)` 这一个方法后面。测试用照脚本出牌的假模型，换真实模型只加一个适配器。
2. 一轮是一个产出事件的生成器，和 HTTP 无关。对话逻辑不起服务器就能测；以后换 WebSocket，只改传输层。
3. 一轮成功了才写进历史。失败的一轮不会在历史里留下半截内容，下一轮的模型看不到它。代价是失败时，已经显示给用户的
   半段回答不会保存。
4. 坏的工具调用交回模型，而不是让这一轮失败。模型通常能自己改正参数再试一次；直接失败的话，用户只能重问。
5. 模型超时只在还没有输出时重试一次，避免客户端看到重复的文字。
6. 同一个会话同一时间只跑一轮，冲突回 `409`，不排队。
7. 只用标准库，评审不用装任何东西。代价是 HTTP 的一些细节要自己处理，比如先读完请求体再回错误。

**砍掉的东西。**

- 持久化。会话存在内存里，重启就没了。`Store` 只有 `create` 和 `get` 两个方法，换成数据库只改这一处。
- 鉴权和限流。没有用户的概念，知道会话 id 的人就能读写。所以服务只监听 `127.0.0.1`。
- 历史截断。历史无限增长，长会话迟早超出模型的上下文。用 Gemini 的有状态模式时历史在服务端；自己发历史时，需要按
  token 预算截断或者摘要。
- 单个工具的超时，以及一轮的总截止时间。计算器有输入上限，跑不久；接网络查询这类慢工具时，两者都要加。
- 断线续传。客户端断开，这一轮就丢了。
- 网页界面。用 curl 演示。
- 真实模型。适配器照官方文档写，没有用真的 key 跑过（有 key 跑过的话，这里写跑了什么、结果如何）。

**已知限制。** 每个连接占一个线程，并发上限取决于线程数。部署多个进程时，内存里的会话和锁不能共享。

### 追问

**客户端在流到一半时断开，会怎样？** 服务端下一次写入时可能发现连接断了，于是关掉这一轮的生成器，这一轮不提交。但这不是
一定的：刚断开时，写入常常还能成功进入操作系统的发送缓冲区，要再写几次才会报错。所以如果剩下的事件很少，这一轮可能照常
提交，客户端重连后可以用 `GET` 看到它。要做到可靠的续传，需要给每一轮一个 id，把事件存下来，客户端带着最后收到的事件
编号重连，从那里接着发。这部分设计见 [设计 Gemini App](#/deepmind/system-design/gemini-app) 的流式与重连部分。

**不用 `eval`，那给 `eval` 去掉内置函数（builtins）行不行？** 不行。去掉内置函数之后，仍然可以从任意对象的属性一路找到
`object` 的全部子类，拿到危险的类。另外，纯算术也能出事：`9 ** 9 ** 9` 的结果有大约 3.7 亿位十进制数字，光存下来就要一百多 MB，计算更要很久。白名单式地遍历语法树，再加上
每一步的大小限制，两个问题一起解决。本页的验证代码只在自己拼出来的表达式上调用 `eval`，用来和计算器对答案。

**超时之后，如果已经输出了一半，还想恢复，怎么办？** 有两条路。一是让客户端配合：发一个“这一轮重新开始”的事件，客户端
清掉已经显示的部分，服务端整轮重试，用户会看到回答被替换。二是把已经生成的文字作为开头，让模型接着写；这依赖模型和 API
是否支持，接口也更复杂。原型里选最简单的：不重试，报错，让用户重问。另外要注意重试会层层叠加：SDK 自己最多重试四次，
应用再重试一次，最坏情况下一次请求会变成十次调用。

**要部署多个实例，哪些地方要改？** 内存里的 `Store` 换成共享的数据库。每个会话的锁也要跨实例：要么按会话 id 把请求路由到
固定的实例，要么用带过期时间的分布式锁，否则一个崩溃的实例会让它持有的会话永远回 `409`。`409` 的语义可以保持不变。

**怎么知道这些测试真的能抓住错误，而不是碰巧通过？** 验证代码里做了一个小的变异测试（mutation testing）：故意写了四个
有错的 `Assistant`，分别是“输出之后还重试”“坏的工具调用直接让这一轮崩掉”“失败的一轮也写进历史”“不限制工具回合”。
每一个都至少让一个测试失败。再加上流式测试不用 `sleep`，结果不受机器快慢影响。

**有状态模式和自己管历史，怎么选？** 有状态模式每次只发新内容，请求小，模型自己的中间状态（比如思考过程的签名）也留在
服务端，不会丢。缺点是对话存在服务方，保存期限和删除要看对方的规则，换一家模型也没法直接迁移。自己管历史（无状态模式）
时，文档要求把模型之前的输出原样发回去，包括签名，所以历史条目要保存模型返回的原始内容，而不只是文字；好处是截断、
摘要、迁移都由自己控制。原型里用有状态模式最省事，README 里写明这个选择。

<details>
<summary>验证代码（可运行）</summary>

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
