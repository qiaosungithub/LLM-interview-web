import time

from judgelib import calling, case, expect, expect_true, rng
import ref

SHOW = 24           # a replay lists at most this many messages

SCRIPT = [
    ("Maya", "Hello everyone", 540),
    ("Theo", "/focus 20", 545),
    ("Priya", "Hey Theo, quick question", 550),
    ("Noah", "/cheer @Priya", 552),
    ("Sana", "/cheer @Priya", 560),
    ("Leo", "/cheer @Theo", 561),
    ("Maya", "/poll Lunch spot? | Tacos | Pasta | Salad", 600),
    ("Theo", "/vote B", 605),
    ("Priya", "/vote B", 606),
    ("Noah", "/vote c", 607),
    ("Sana", "/vote Z", 608),
    ("Leo", "/focus abc", 609),
    ("Maya", "/cheer bob", 610),
    ("Theo", "/poll Coffee? | OnlyOneOption", 611),
    ("Priya", "/vote B", 612),
]

EXPECTED_LOG = [
    "Maya: Hello everyone",
    "Theo: /focus 20",
    "FocusBot: Theo is heads-down for 20 min, back at 09:25.",
    "Priya: Hey Theo, quick question",
    "FocusBot: Theo is heads-down until 09:25, try again later.",
    "Noah: /cheer @Priya",
    "CheerBot: Noah cheers for @Priya! @Priya's cheer count is now 1.",
    "Sana: /cheer @Priya",
    "CheerBot: Sana cheers for @Priya! @Priya's cheer count is now 2.",
    "Leo: /cheer @Theo",
    "FocusBot: Theo is heads-down until 09:25, try again later.",
    "CheerBot: Leo cheers for @Theo! @Theo's cheer count is now 1.",
    "Maya: /poll Lunch spot? | Tacos | Pasta | Salad",
    "PollBot: Maya started a poll -- Lunch spot? [A) Tacos, B) Pasta, C) Salad]",
    "Theo: /vote B",
    "PollBot: Theo voted B (Pasta). Tally -- A: 0, B: 1, C: 0",
    "Priya: /vote B",
    "PollBot: Priya voted B (Pasta). Tally -- A: 0, B: 2, C: 0",
    "Noah: /vote c",
    "PollBot: Noah voted C (Salad). Tally -- A: 0, B: 2, C: 1",
    "Sana: /vote Z",
    "Leo: /focus abc",
    "Maya: /cheer bob",
    "Theo: /poll Coffee? | OnlyOneOption",
    "Priya: /vote B",
    "PollBot: Priya voted B (Pasta). Tally -- A: 0, B: 3, C: 1",
]

DECIDED = "CheerBot: Maya's poll reached 3 votes! Maya's cheer count is now 1."

CLOSED = "投票已经结束：之后的 /vote 要和没有进行中的投票时一样被忽略，奖励那一行也只在结束的那条消息里出现一次"


# ---------------------------------------------------------------- how calls are shown

def _lit(x):
    """A Python literal for x, with strings in double quotes as in the statement."""
    if isinstance(x, str):
        return f'"{x}"' if '"' not in x and "\\" not in x and x.isprintable() else repr(x)
    if isinstance(x, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(v)}" for k, v in x.items()) + "}"
    return repr(x)


def _need(sol, names):
    for n in names:
        expect_true(callable(getattr(sol, n, None)), f"没有找到 {n}（类名要和题目一致）")


def _build(sol, setup, names):
    """Runs the setup code (exactly what the user is shown) with the answer's classes."""
    _need(sol, names)
    ns = {n: getattr(sol, n) for n in names}
    calling("\n".join(setup) + "   # <- 构造时出错")
    exec("\n".join(setup), ns)
    return ns


def _explain(got, want):
    """A sentence on how the appended lines differ."""
    try:
        if sorted(map(str, got)) == sorted(want):
            return "这条消息追加的行都对，但顺序不对"
        missing = [w for w in want if w not in got]
        extra = [g for g in got if g not in want]
    except TypeError:
        return "这条消息追加的行不对"
    if missing and not extra:
        return f"少了 {len(missing)} 行，例如 {missing[0]!r}"
    if extra and not missing:
        return f"多了 {len(extra)} 行，例如 {extra[0]!r}"
    return "这条消息追加的行不对"


def _with(msg, hint):
    return f"{msg}。提示：{hint}" if hint else msg


# ---------------------------------------------------------------- rooms

P2_NAMES = ["Clock", "ChatRoom", "FocusBot", "CheerBot", "PollBot"]
P3_NAMES = ["Clock", "ChatRoom", "FocusBot", "EventBus", "DecisivePollBot", "RewardedCheerBot"]


class Room:
    """One room a test builds: the setup code (shown to the user, and run exactly as shown) and the model of the
    statement it is checked against. Part 2 registers FocusBot, CheerBot, PollBot; Part 3 registers FocusBot,
    DecisivePollBot, RewardedCheerBot."""

    def __init__(self, part, name="room", sfx="", counts=None, focus=None):
        self.part, self.name = part, name
        self.counts, self.focus = dict(counts or {}), dict(focus or {})
        clock, bus, c, f = "clock" + sfx, "bus" + sfx, _lit(self.counts), _lit(self.focus)
        if part == 2:
            self.names = P2_NAMES
            self.code = [f"{clock} = Clock()", f"{name} = ChatRoom({clock})",
                         f"{name}.register(FocusBot({clock}, {f}))",
                         f"{name}.register(CheerBot({c}))",
                         f"{name}.register(PollBot({{}}))"]
        else:
            self.names = P3_NAMES
            self.code = [f"{bus} = EventBus()", f"{clock} = Clock()", f"{name} = ChatRoom({clock})",
                         f"{name}.register(FocusBot({clock}, {f}))",
                         f"{name}.register(DecisivePollBot({{}}, {bus}))",
                         f"{name}.register(RewardedCheerBot({c}, {bus}))"]

    def model(self):
        return ref.Model(decisive=self.part == 3, cheer_counts=self.counts, focus_until=self.focus)


def _replay(setup, done, n_before, mark="# <- 这一步"):
    """The setup code and the sends so far (the last one marked), then a print of what the last send appended."""
    calls = [f"{room}.send({_lit(s)}, {_lit(t)}, {now})" for room, s, t, now in done]
    skipped = max(0, len(calls) - SHOW)
    lines = list(setup)
    if skipped:
        lines.append(f"# …（省略了前面 {skipped} 次 send，所以不能直接粘贴重现）")
    calls = calls[skipped:]
    calls[-1] = f"log = {calls[-1]}   {mark}"
    return "\n".join(lines + calls + [f"print(log[{n_before}:])   # 这条消息追加的行（在它之前日志有 {n_before} 行）"])


def _log_ok(got, before, want):
    try:
        return isinstance(got, (list, tuple)) and list(got) == before + want
    except Exception:
        return False


def _check_log(got, before, want, call, hint=None):
    """send returns the whole log so far: the lines before this message unchanged, then exactly `want`."""
    expect_true(isinstance(got, (list, tuple)),
                f"send 应当返回日志（到目前为止的所有行，list[str]），得到的是 {type(got).__name__}", call=call, got=got)
    got, n = list(got), len(before)
    if got[:n] != before:
        why = ("send 应当返回到目前为止的整个日志，而不只是这条消息追加的行" if n and got == want
               else "send 返回的日志里，这条消息之前就有的行变了")
        expect(got, before + want, call=call, msg=why)
    new = got[n:]
    if new != want:
        expect(new, want, call=call, msg=_with(_explain(new, want), hint))


def _steps(rooms, steps):
    """Normalizes steps to (room, sender, text, now, hint). A step is (sender, text, now[, hint]) for one room, or
    (room, sender, text, now[, hint]) when the case has several."""
    out = []
    for step in steps:
        if isinstance(step[2], int):
            sender, text, now, *h = step
            room = rooms[0].name
        else:
            room, sender, text, now, *h = step
        out.append((room, sender, text, now, h[0] if h else None))
    return out


def play(sol, rooms, steps, wants=None, hint=None):
    """Builds the rooms, sends every message to the answer and checks the log each send returns right away.
    Expected lines come from `wants` (one list per step, hard-coded) or from the model."""
    setup = [line for r in rooms for line in r.code]
    ns = _build(sol, setup, sorted({n for r in rooms for n in r.names}))
    models = {r.name: r.model() for r in rooms}
    logs = {r.name: [] for r in rooms}
    done = []
    for k, (room, sender, text, now, step_hint) in enumerate(_steps(rooms, steps)):
        done.append((room, sender, text, now))
        want = list(wants[k]) if wants is not None else models[room].send(sender, text, now)
        before = logs[room]
        call = calling(_replay(setup, done, len(before)))
        try:
            got = ns[room].send(sender, text, now)
        except Exception:
            calling(_replay(setup, done, len(before), mark="# <- 这一步抛出异常"))
            raise
        _check_log(got, before, want, call, step_hint or hint)
        logs[room] = before + want
    return ns


def _per_step(script, log):
    """Splits a hard-coded log into the lines each message of the script appended."""
    starts, i = [], 0
    for sender, text, _ in script:
        i = log.index(f"{sender}: {text}", i)
        starts.append(i)
        i += 1
    starts.append(len(log))
    return [log[a:b] for a, b in zip(starts, starts[1:])]


# ---------------------------------------------------------------- random sequences against the model

def _outcome(sol, room, steps):
    """Replays steps on a fresh answer room and a fresh model; None if every returned log agrees."""
    ns = {n: getattr(sol, n) for n in room.names}
    exec("\n".join(room.code), ns)
    model, log = room.model(), []
    for k, (sender, text, now) in enumerate(steps):
        want = model.send(sender, text, now)
        try:
            got = ns[room.name].send(sender, text, now)
        except Exception as e:
            return {"k": k, "exc": e, "before": log, "want": want}
        if not _log_ok(got, log, want):
            return {"k": k, "exc": None, "before": log, "want": want, "got": got}
        log = log + want
    return None


def _shrink(sol, room, steps, bad, budget=0.6):
    """Drops messages the failure does not need, so the replay shown is short."""
    steps = steps[: bad["k"] + 1]
    stop = time.process_time() + budget
    size = max(1, len(steps) // 2)
    while time.process_time() < stop:
        i, progress = 0, False
        while i < len(steps) and time.process_time() < stop:
            cand = steps[:i] + steps[i + size:]
            out = _outcome(sol, room, cand) if cand else None
            if out is not None and type(out["exc"]) is type(bad["exc"]):
                steps, bad, progress = cand[: out["k"] + 1], out, True
            else:
                i += size
        if size > 1:
            size //= 2
        elif not progress:
            break
    return steps, bad


def random_check(sol, part, sequences, length, kinds, names=ref.NAMES):
    r = rng()
    room = Room(part)
    _build(sol, room.code, room.names)
    note = "随机消息序列（已自动删去与出错无关的消息）"
    for _ in range(sequences):
        steps = ref.random_script(r, length, kinds, names)
        bad = _outcome(sol, room, steps)
        if bad is None:
            continue
        steps, bad = _shrink(sol, room, steps, bad)
        play(sol, [room], steps, hint=note)
        # the short replay passed on its own: the answer depends on rooms built before this one
        k = bad["k"]
        call = _replay(room.code, [(room.name, *s) for s in steps[: k + 1]], len(bad["before"]))
        if bad["exc"] is not None:
            calling(call)
            raise bad["exc"]
        _check_log(bad["got"], bad["before"], bad["want"], call,
                   "随机消息序列；单独重放这段消息时结果又对了，说明之前建的聊天室影响了这一个")


# ---------------------------------------------------------------- one bot without a room

DELIVER = ["def deliver(bot, sender, text):   # what ChatRoom does with each bot",
           "    return bot.handle(sender, text) if bot.can_handle(sender, text) else []"]
BOTS = {   # kind -> (classes needed, setup code, prefix of its lines, the model's arguments)
    "CheerBot": (["CheerBot"], lambda init: [f"bot = CheerBot({_lit(init)})"], "CheerBot: ", {}),
    "FocusBot": (["Clock", "FocusBot"], lambda init: ["clock = Clock()", f"bot = FocusBot(clock, {_lit(init)})"],
                 "FocusBot: ", {}),
    "PollBot": (["PollBot"], lambda init: ["bot = PollBot({})"], "PollBot: ", {}),
    "DecisivePollBot": (["EventBus", "DecisivePollBot"], lambda init: ["bus = EventBus()", "bot = DecisivePollBot({}, bus)"],
                        "PollBot: ", {"decisive": True}),
    "RewardedCheerBot": (["EventBus", "RewardedCheerBot"],
                         lambda init: ["bus = EventBus()", f"bot = RewardedCheerBot({_lit(init)}, bus)"], "CheerBot: ", {}),
}


def play_bot(sol, kind, steps, init=None):
    """Drives one bot the way ChatRoom would (can_handle, then handle), with no room; FocusBot's clock is set by
    the test. Expected lines: the statement's lines for that bot only."""
    names, setup_fn, prefix, margs = BOTS[kind]
    init = dict(init or {})
    setup = setup_fn(init)
    ns = _build(sol, setup, names)
    if kind == "FocusBot":
        model = ref.Model(focus_until=init)
    elif "Cheer" in kind:
        model = ref.Model(cheer_counts=init)
    else:
        model = ref.Model(**margs)
    bot, clocked, shown = ns["bot"], kind == "FocusBot", []
    for sender, text, now, *h in steps:
        want = [line for line in model.send(sender, text, now) if line.startswith(prefix)]
        head = f"clock.now = {now}; " if clocked else ""
        shown.append(f"{head}deliver(bot, {_lit(sender)}, {_lit(text)})")
        skipped = max(0, len(shown) - SHOW)
        lines = DELIVER + setup + ([f"# …（省略了前面 {skipped} 步，所以不能直接粘贴重现）"] if skipped else [])
        last = f"{head}print(deliver(bot, {_lit(sender)}, {_lit(text)}))   # <- 这一步"
        call = calling("\n".join(lines + shown[skipped:-1] + [last]))
        if clocked:
            ns["clock"].now = now
        got = bot.handle(sender, text) if bot.can_handle(sender, text) else []
        expect_true(isinstance(got, (list, tuple)), f"handle 应当返回 list[str]，得到的是 {type(got).__name__}",
                    call=call, got=got)
        if list(got) != want:
            expect(list(got), want, call=call, msg=_with(_explain(list(got), want), h[0] if h else None))


# ---------------------------------------------------------------- Part 2

@case("Part 2", sample=True, title="题目脚本 SCRIPT → EXPECTED_LOG")
def p2_example(sol):
    play(sol, [Room(2)], SCRIPT, wants=_per_step(SCRIPT, EXPECTED_LOG))


@case("Part 2", title="两个聊天室互不影响")
def p2_two_rooms(sol):
    hint = "每个新建的聊天室都从构造时给的状态开始，机器人只用自己拿到的那份状态（不能留在模块级变量或类属性里）"
    rooms = [Room(2, "room1", "1"), Room(2, "room2", "2")]
    steps = [(name, s, t, now, hint) for s, t, now in SCRIPT for name in ("room1", "room2")]
    play(sol, rooms, steps)


@case("Part 2", title="命令要在开头，命令词后跟空格")
def p2_command_start(sol):
    play(sol, [Room(2)], [
        ("Theo", "/focus 30", 600),
        ("Noah", "/cheer\t@Priya", 601),
        ("Noah", "/Cheer @Priya", 602),
        ("Noah", " /cheer @Priya", 603),
        ("Noah", "/cheer@Priya", 604),
        ("Noah", "/cheers @Priya", 605),
        ("Noah", "hi /cheer @Priya", 606),
        ("Noah", "/cheer", 607),
        ("Maya", "/poll\tLunch? | Tacos | Pasta", 608),
        ("Maya", "/poll", 609),
        ("Maya", "/POLL Lunch? | Tacos | Pasta", 610),
        ("Priya", "/vote A", 611),
        ("Leo", "/focus\t10", 612),
        ("Leo", "/focus", 613),
        ("Sana", "/cheer\t@Theo", 614, "不是命令的消息照样参与 FocusBot 的提醒检查"),
        ("Noah", "/cheer @Priya", 615),
    ])


@case("Part 2", title="参数首尾的空白被忽略")
def p2_whitespace(sol):
    play(sol, [Room(2)], [
        ("Noah", "/cheer   @Priya  ", 600),
        ("Sana", "/cheer @Priya\t", 601),
        ("Maya", "/poll   Lunch?  |  Tacos|Pasta  |   Salad ", 602),
        ("Theo", "/vote   b  ", 603),
        ("Leo", "/focus  15 ", 604),
        ("Kofi", "/cheer @Kofi Mensah ", 605),
    ])


@case("Part 2", title="/cheer 的合法与不合法形式")
def p2_cheer_rules(sol):
    play(sol, [Room(2)], [
        ("Iris", "/cheer Kofi", 10),
        ("Iris", "/cheer @", 11),
        ("Iris", "/cheer @Kofi", 12),
        ("Noah", "/cheer @Kofi", 13),
        ("Kofi", "/cheer @Iris", 14),
        ("Iris", "/cheer  @Kofi", 15),
        ("Iris", "/cheer @@x", 16),
        ("Iris", "/cheer @x", 17),
        ("Iris", "/cheer @kofi", 18),
        ("Iris", "/cheer @Kofi!", 19),
        ("Iris", "/cheer @@x", 20),
    ])


@case("Part 2", title="/focus 只认十进制数字且 ≥ 1")
def p2_focus_args(sol):
    hint = "<n> 必须是一个或多个十进制数字，拼出的数 ≥ 1"
    play(sol, [Room(2)], [
        ("Leo", "/focus 0", 600, hint),
        ("Leo", "/focus 00", 601, hint),
        ("Leo", "/focus -3", 602, hint),
        ("Leo", "/focus abc", 603, hint),
        ("Leo", "/focus 5 extra", 604, hint),
        ("Leo", "/focus +5", 605, hint),
        ("Leo", "/focus 1.5", 606, hint),
        ("Leo", "/focus 2_0", 607, hint),
        ("Leo", "/focus ", 608, hint),
        ("Leo", "/focus 007", 609, hint),
        ("Maya", "/focus 1", 610),
        ("Noah", "hi Leo and Maya", 611, "只有 now 严格早于时段结束的时刻才提醒"),
    ])


@case("Part 2", title="非 ASCII 的十进制数字")
def p2_focus_unicode_digits(sol):
    hint = "要和旧代码一致：它用 str.isdecimal() 判断，再用 int() 取值"
    play(sol, [Room(2)], [
        ("Leo", "/focus ²", 600, hint),
        ("Leo", "/focus ٣", 601, hint),
        ("Maya", "/focus ١٠", 602, hint),
        ("Noah", "Leo? Maya?", 603),
    ])


@case("Part 2", title="跨过午夜、跨天的钟点")
def p2_focus_midnight(sol):
    play(sol, [Room(2)], [
        ("Kofi", "/focus 20", 1430),
        ("Iris", "Kofi, lunch?", 1435),
        ("Ana", "/focus 10", 1450),
        ("Ana", "/focus 1440", 1455),
        ("Leo", "/focus 60", 2820),
        ("Iris", "Ana and Leo and Kofi", 2830),
        ("Maya", "/focus 3000", 20000),
        ("Iris", "Maya?", 22999),
    ])


@case("Part 2", title="时段结束的那一分钟不再提醒")
def p2_reminder_boundary(sol):
    hint = "只有 now 严格早于时段结束时刻才提醒"
    play(sol, [Room(2)], [
        ("Theo", "/focus 20", 545),
        ("Priya", "Theo?", 564, hint),
        ("Priya", "Theo?", 565, hint),
        ("Priya", "Theo?", 566, hint),
        ("Theo", "/focus 1", 600),
        ("Priya", "Theo?", 601, hint),
    ])


@case("Part 2", title="用户名作为子串出现就提醒")
def p2_reminder_substring(sol):
    play(sol, [Room(2)], [
        ("Theo", "/focus 30", 600),
        ("TheoBot", "/focus 30", 601),
        ("Pri", "/focus 30", 602),
        ("Noah", "TheoBot is up", 603),
        ("Noah", "Priya?", 604),
        ("Noah", "theo", 605),
        ("Noah", "/cheer @Theo", 606, "提醒排在这条消息其他机器人的行之前"),
        ("Maya", "/poll Who? | Theo | Pri", 607, "提醒排在这条消息其他机器人的行之前"),
        ("Noah", "/vote b", 608),
    ])


@case("Part 2", title="多人提醒按第一次专注的先后")
def p2_reminder_order(sol):
    hint = "多个用户的提醒按他们第一次开启专注时段的先后排列"
    play(sol, [Room(2)], [
        ("Zed", "/focus 10", 100),
        ("Amy", "/focus 60", 105),
        ("Noah", "Amy and Zed", 106, hint),
        ("Zed", "/focus 120", 107),
        ("Noah", "Amy and Zed", 108, hint),
        ("Bo", "/focus 5", 400),
        ("Cy", "/focus 50", 410),
        ("Bo", "/focus 30", 420),
        ("Noah", "Cy, Bo", 421, hint),
    ])


@case("Part 2", title="提醒在本条 /focus 生效前确定")
def p2_reminder_before_focus(sol):
    hint = "提醒在同一条消息里的 /focus 生效之前就已确定"
    play(sol, [Room(2)], [
        ("us", "/focus 20", 100, hint),
        ("us", "/focus 5", 110, hint),
        ("Noah", "hi us", 116),
        ("focus", "/focus 30", 117, hint),
        ("Maya", "/focus 10", 118),
    ])


@case("Part 2", title="/poll 的切分与合法性")
def p2_poll_rules(sol):
    play(sol, [Room(2)], [
        ("Maya", "/poll Lunch? | Tacos", 600),
        ("Maya", "/poll Lunch?", 601),
        ("Maya", "/poll Lunch?|Tacos|Pasta", 602),
        ("Theo", "/vote a", 603),
        ("Leo", "/poll Coffee? | | ", 604, "空字符串也是一个选项"),
        ("Theo", "/vote B", 605),
        ("Sana", "/poll  | Yes | No | Maybe", 606, "问题可以是空字符串"),
        ("Noah", "/vote c", 607),
        ("Noah", "/poll Pick | a | b | c | d | e | f | g", 608),
        ("Priya", "/vote G", 609),
        ("Priya", "/vote H", 610),
        ("Kofi", "/poll Tea? | Green | Black", 611, "合法的 /poll 会替换掉进行中的投票"),
        ("Kofi", "/vote a", 612),
        ("Kofi", "/poll Again? | x", 613),
        ("Kofi", "/vote b", 614, "不合法的 /poll 不影响进行中的投票"),
    ])


@case("Part 2", title="/vote 的匹配与票数")
def p2_vote_rules(sol):
    play(sol, [Room(2)], [
        ("Theo", "/vote A", 600),
        ("Maya", "/poll Lunch spot? | Tacos | Pasta | Salad", 601),
        ("Theo", "/vote AB", 602),
        ("Theo", "/vote ", 603),
        ("Theo", "/vote d", 604),
        ("Theo", "/vote c", 605),
        ("Theo", "/vote C", 606),
        ("Priya", "/vote b", 607),
        ("Noah", "/vote  a ", 608),
        ("Leo", "/vote b", 609),
        ("Sana", "/vote B", 610),
        ("Sana", "/vote B", 611, "Part 2 的 PollBot 没有“3 票结束”的规则"),
    ])


@case("Part 2", title="send 把 clock.now 设成消息的 now")
def p2_clock(sol):
    room = Room(2)
    ns = _build(sol, room.code, room.names)
    sends = []
    for sender, text, now in [("Maya", "Hello everyone", 540), ("Theo", "hi", 541), ("Maya", "hi", 1980)]:
        sends.append(f"room.send({_lit(sender)}, {_lit(text)}, {now})")
        calling("\n".join(room.code + sends[:-1] + [sends[-1] + "   # <- 这一步抛出异常"]))
        ns["room"].send(sender, text, now)
        code = "\n".join(room.code + sends + ["print(clock.now)"])
        got = getattr(ns["clock"], "now", None)
        expect(got, now, call=code, msg="ChatRoom.send 应当把 clock.now 设成这条消息的 now")


@case("Part 2", title="不经过 ChatRoom 单独用 CheerBot")
def p2_unit_cheer(sol):
    play_bot(sol, "CheerBot", [
        ("Iris", "/cheer Kofi", 0),
        ("Iris", "/cheer @Kofi", 1),
        ("Iris", "hi @Kofi", 2),
        ("Kofi", "/cheer @Iris", 3),
        ("Noah", "/cheer  @Kofi ", 4),
        ("Noah", "/focus 10", 5),
        ("Noah", "/vote A", 6),
    ])


@case("Part 2", title="不经过 ChatRoom 单独用 FocusBot")
def p2_unit_focus(sol):
    play_bot(sol, "FocusBot", [
        ("Kofi", "/focus soon", 1430),
        ("Kofi", "/focus 20", 1430),
        ("Iris", "Kofi, lunch?", 1435),
        ("Iris", "/cheer @Kofi", 1440, "FocusBot 检查每一条消息，包括其他机器人的命令"),
        ("Iris", "Kofi?", 1450),
        ("Iris", "/focus 0", 1451),
        ("Iris", "/focus 5", 1452),
        ("Kofi", "Iris?", 1456),
    ])


@case("Part 2", title="不经过 ChatRoom 单独用 PollBot")
def p2_unit_poll(sol):
    play_bot(sol, "PollBot", [
        ("Theo", "/vote A", 0),
        ("Maya", "/poll Lunch? | Tacos", 1),
        ("Maya", "/poll Lunch? | Tacos | Pasta", 2),
        ("Theo", "/vote b", 3),
        ("Theo", "/vote z", 4),
        ("Theo", "hello", 5),
        ("Theo", "/cheer @Maya", 6),
        ("Priya", "/vote B", 7),
    ])


@case("Part 2", title="构造时传入的已有状态")
def p2_initial_state(sol):
    hint = "构造函数收到的 dict 就是机器人的状态，里面已有的内容要算数"
    play_bot(sol, "CheerBot", [
        ("Iris", "/cheer @Kofi", 0, hint),
        ("Iris", "/cheer @Ana", 1),
    ], init={"Kofi": 2})
    play_bot(sol, "FocusBot", [
        ("Priya", "Theo?", 590, hint),
        ("Priya", "Theo?", 600),
    ], init={"Theo": 600})


@case("Part 2", title="随机消息与 handle_message 比较")
def p2_random(sol):
    random_check(sol, 2, 40, 60, ref.MIXED)


@case("Part 2", title="随机消息：大量专注与提及")
def p2_random_mentions(sol):
    random_check(sol, 2, 40, 60, ref.MENTIONS, names=["Theo", "TheoBot", "Pri", "Priya", "us", "B", "cheer", "o"])


# ---------------------------------------------------------------- Part 3

@case("Part 3", sample=True, title="题目脚本：最后一票后加一行")
def p3_example(sol):
    play(sol, [Room(3)], SCRIPT, wants=_per_step(SCRIPT, EXPECTED_LOG + [DECIDED]))


@case("Part 3", title="两个聊天室各用各的总线")
def p3_two_rooms(sol):
    hint = "每个聊天室的机器人只经自己的总线通信，两个 EventBus 的订阅互不相干"
    rooms = [Room(3, "room1", "1"), Room(3, "room2", "2")]
    play(sol, rooms, [(name, s, t, now, hint) for s, t, now in SCRIPT for name in ("room1", "room2")])


@case("Part 3", title="EventBus：按订阅顺序调用对应类型")
def p3_eventbus(sol):
    _need(sol, ["Event", "EventBus"])
    code = ["bus = EventBus()", "seen = []"]
    buses, seen = {"bus": sol.EventBus()}, []

    def subscribe(bus, label, etype):
        code.append(f'{bus}.subscribe("{etype}", lambda e: seen.append(("{label}", e)))')
        calling("\n".join(code) + "   # <- 这一步")
        buses[bus].subscribe(etype, lambda e: seen.append((label, e)))

    def publish(bus, etype, data, want, msg):
        code.append(f"seen.clear(); {bus}.publish(Event({_lit(etype)}, {_lit(data)}))")
        show = "\n".join(code + ["print([label for label, e in seen])"])
        calling(show)
        seen.clear()
        event = sol.Event(etype, dict(data))
        buses[bus].publish(event)
        expect([label for label, _ in seen], want, call=show, msg=msg)
        expect_true(all(e is event for _, e in seen), "处理函数收到的应当是发布的那个 Event 对象", call=show)

    subscribe("bus", "h1", "vote")
    subscribe("bus", "h2", "vote")
    subscribe("bus", "h3", "cheer")
    publish("bus", "vote", {"n": 1}, ["h1", "h2"], "要按订阅顺序调用这个类型的每个处理函数，别的类型的不调用")
    publish("bus", "other", {}, [], "没有处理函数订阅的类型，发布后什么都不调用")
    publish("bus", "cheer", {}, ["h3"], "要按订阅顺序调用这个类型的每个处理函数，别的类型的不调用")
    subscribe("bus", "h4", "vote")
    publish("bus", "vote", {"n": 2}, ["h1", "h2", "h4"], "后订阅的处理函数排在后面，之前的订阅仍然有效")
    code.append("bus2 = EventBus()")
    buses["bus2"] = sol.EventBus()
    publish("bus2", "vote", {}, [], "两个 EventBus 各有各的订阅，在 bus2 上发布不应调用 bus 的处理函数")
    subscribe("bus2", "h5", "vote")
    publish("bus", "vote", {"n": 3}, ["h1", "h2", "h4"], "两个 EventBus 各有各的订阅，订阅到 bus2 的处理函数不属于 bus")


@case("Part 3", title="结束后的 /vote 当作没有投票")
def p3_closed_poll(sol):
    play(sol, [Room(3)], SCRIPT + [
        ("Noah", "/vote A", 615, CLOSED),
        ("Sana", "/vote b", 616, CLOSED),
        ("Leo", "hello", 617, "奖励那一行只在投票结束的那条消息里出现一次"),
        ("Maya", "/poll Coffee? | OnlyOneOption", 618),
        ("Theo", "/vote A", 619, CLOSED),
        ("Theo", "/poll Dinner? | Soup | Rice", 620),
        ("Noah", "/vote a", 621, "新的投票票数从 0 开始"),
        ("Sana", "/vote b", 622),
    ])


@case("Part 3", title="奖励和欢呼共用同一份计数")
def p3_reward_shared_count(sol):
    hint = "奖励加在 CheerBot 的同一份欢呼计数上（按用户名，不带 @）"
    play(sol, [Room(3)], [
        ("Noah", "/cheer @Maya", 10),
        ("Sana", "/cheer @Maya", 11),
        ("Maya", "/poll Lunch? | Tacos | Pasta", 12),
        ("Theo", "/vote a", 13),
        ("Priya", "/vote A", 14),
        ("Leo", "/vote a", 15, hint),
        ("Kofi", "/cheer @Maya", 16, hint),
    ])


@case("Part 3", title="奖励给当前这次投票的发起人")
def p3_current_author(sol):
    play(sol, [Room(3)], [
        ("Maya", "/poll Lunch? | Tacos | Pasta", 10),
        ("Theo", "/vote a", 11),
        ("Priya", "/vote a", 12),
        ("Theo", "/poll Coffee? | Latte | Mocha", 13),
        ("Noah", "/vote a", 14, "新的投票票数从 0 开始"),
        ("Sana", "/vote a", 15, "新的投票票数从 0 开始"),
        ("Leo", "/vote a", 16, "奖励给当前这次投票的发起人"),
        ("Maya", "/vote b", 17, CLOSED),
    ])


@case("Part 3", title="提醒、投票行、奖励行的顺序")
def p3_line_order(sol):
    play(sol, [Room(3)], [
        ("B", "/focus 30", 100),
        ("Maya", "/poll Pick one | Alpha | Beta", 101),
        ("Theo", "/vote B", 102),
        ("Priya", "/vote B", 103),
        ("Noah", "/vote B", 104, "顺序是 FocusBot 的提醒、PollBot 的投票行、CheerBot 的奖励行"),
        ("Leo", "/vote B", 105, CLOSED),
    ])


@case("Part 3", title="多次投票各自结束")
def p3_multiple_polls(sol):
    play(sol, [Room(3)], [
        ("Maya", "/poll One? | a | b", 10),
        ("Theo", "/vote a", 11),
        ("Priya", "/vote a", 12),
        ("Noah", "/vote a", 13),
        ("Maya", "/poll Two? | a | b | c", 20),
        ("Theo", "/vote a", 21),
        ("Priya", "/vote b", 22),
        ("Noah", "/vote a", 23),
        ("Sana", "/vote b", 24),
        ("Leo", "/vote c", 25),
        ("Kofi", "/vote b", 26),
        ("Theo", "/poll Three? | x | y", 30),
        ("Maya", "/vote y", 31),
        ("Priya", "/vote y", 32),
        ("Noah", "/vote x", 33),
        ("Sana", "/vote y", 34),
        ("Leo", "/vote x", 35, CLOSED),
    ])


@case("Part 3", title="单独用 DecisivePollBot")
def p3_unit_poll(sol):
    play_bot(sol, "DecisivePollBot", [
        ("Maya", "/poll Lunch? | Tacos | Pasta", 0),
        ("Theo", "/vote a", 1),
        ("Priya", "/vote a", 2),
        ("Leo", "/vote A", 3),
        ("Noah", "/vote b", 4, CLOSED),
        ("Maya", "/poll Again? | x | y", 5),
        ("Noah", "/vote b", 6),
    ])


@case("Part 3", title="单独用 RewardedCheerBot")
def p3_unit_cheer(sol):
    play_bot(sol, "RewardedCheerBot", [
        ("Iris", "/cheer Kofi", 0),
        ("Iris", "/cheer @Kofi", 1),
        ("Iris", "/vote A", 2),
        ("Noah", "hello", 3),
        ("Noah", "/cheer @Kofi", 4),
    ])


@case("Part 3", title="随机消息与新规则比较")
def p3_random(sol):
    random_check(sol, 3, 40, 80, ref.VOTES)
