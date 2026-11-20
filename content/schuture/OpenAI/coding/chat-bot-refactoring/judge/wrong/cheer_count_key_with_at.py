# expect: WA
# CheerBot keys counts by the whole target with its @, so counts given to the constructor and the poll reward miss them
from abc import ABC, abstractmethod


def after(prefix, text):
    return text[len(prefix):].strip() if text.startswith(prefix) else None


def hhmm(m):
    return f"{m // 60 % 24:02d}:{m % 60:02d}"


class Clock:
    def __init__(self, now=0):
        self.now = now


class ChatBot(ABC):
    @abstractmethod
    def can_handle(self, sender, text): ...

    @abstractmethod
    def handle(self, sender, text): ...


class ChatRoom:
    def __init__(self, clock):
        self.clock, self.bots, self.log = clock, [], []

    def register(self, bot):
        self.bots.append(bot)

    def send(self, sender, text, now):
        self.clock.now = now
        self.log.append(f"{sender}: {text}")
        for bot in self.bots:  # NOTE: registration order = order of the bots' lines
            if bot.can_handle(sender, text):
                self.log.extend(bot.handle(sender, text))
        return self.log


class CheerBot(ChatBot):
    def __init__(self, cheer_counts):
        self.cheer_counts = cheer_counts

    def _target(self, text):
        arg = after("/cheer ", text) or ""
        return arg if arg.startswith("@") and len(arg) > 1 else None

    def can_handle(self, sender, text):
        return bool(self._target(text))

    def handle(self, sender, text):
        target = self._target(text)
        n = self.cheer_counts[target] = self.cheer_counts.get(target, 0) + 1
        return [f"CheerBot: {sender} cheers for {target}! {target}'s cheer count is now {n}."]


class FocusBot(ChatBot):
    def __init__(self, clock, focus_until):
        self.clock, self.focus_until = clock, focus_until

    def _minutes(self, text):
        arg = after("/focus ", text) or ""
        return int(arg) if arg.isdecimal() and int(arg) >= 1 else None

    def _mentioned(self, text):  # NOTE: dict order = order of each user's FIRST session
        return [u for u, end in self.focus_until.items() if u in text and self.clock.now < end]

    def can_handle(self, sender, text):
        return bool(self._mentioned(text) or self._minutes(text))

    def handle(self, sender, text):
        lines = [f"FocusBot: {u} is heads-down until {hhmm(self.focus_until[u])}, try again later."
                 for u in self._mentioned(text)]  # NOTE: before this message's own /focus is recorded
        if minutes := self._minutes(text):
            end = self.focus_until[sender] = self.clock.now + minutes
            lines.append(f"FocusBot: {sender} is heads-down for {minutes} min, back at {hhmm(end)}.")
        return lines


class PollBot(ChatBot):
    def __init__(self, poll_state):
        self.poll_state = poll_state  # {} while no poll is in progress

    def _poll(self, text):
        pieces = [p.strip() for p in (after("/poll ", text) or "").split("|")]
        return pieces if len(pieces) >= 3 else None

    def _letter(self, text):
        letter = (after("/vote ", text) or "").upper()
        return letter if self.poll_state and letter in self.poll_state["votes"] else None

    def can_handle(self, sender, text):
        return bool(self._poll(text) or self._letter(text))

    def handle(self, sender, text):
        state, pieces = self.poll_state, self._poll(text)
        if pieces:
            question, *options = pieces
            votes = {chr(65 + i): 0 for i in range(len(options))}
            state.clear()
            state.update(options=options, author=sender, votes=votes)
            listed = ", ".join(f"{k}) {o}" for k, o in zip(votes, options))
            return [f"PollBot: {sender} started a poll -- {question} [{listed}]"]
        letter = self._letter(text)
        state["votes"][letter] += 1
        tally = ", ".join(f"{k}: {v}" for k, v in state["votes"].items())
        return [f"PollBot: {sender} voted {letter} ({state['options'][ord(letter) - 65]}). Tally -- {tally}"]


class Event:
    def __init__(self, type, data):
        self.type, self.data = type, data


class EventBus:
    def __init__(self):
        self.handlers = {}

    def subscribe(self, event_type, handler):
        self.handlers.setdefault(event_type, []).append(handler)

    def publish(self, event):  # NOTE: synchronous and depth-first
        for handler in list(self.handlers.get(event.type, [])):
            handler(event)


class DecisivePollBot(PollBot):
    def __init__(self, poll_state, bus):
        super().__init__(poll_state)
        self.bus = bus

    def handle(self, sender, text):
        letter = self._letter(text)  # None unless a valid /vote
        lines = super().handle(sender, text)
        if letter and self.poll_state["votes"][letter] >= 3:
            author = self.poll_state["author"]
            self.poll_state.clear()  # NOTE: close before publishing, so no handler sees it open
            self.bus.publish(Event("poll_decided", {"author": author}))
        return lines


class RewardedCheerBot(CheerBot):
    def __init__(self, cheer_counts, bus):
        super().__init__(cheer_counts)
        self.pending = []
        bus.subscribe("poll_decided", self._on_poll_decided)

    def _on_poll_decided(self, event):
        author = event.data["author"]
        n = self.cheer_counts[author] = self.cheer_counts.get(author, 0) + 1
        self.pending.append(f"CheerBot: {author}'s poll reached 3 votes! {author}'s cheer count is now {n}.")

    def can_handle(self, sender, text):
        return super().can_handle(sender, text) or bool(self.pending)

    def handle(self, sender, text):
        lines = super().handle(sender, text) if self._target(text) else []
        lines, self.pending = lines + self.pending, []
        return lines
