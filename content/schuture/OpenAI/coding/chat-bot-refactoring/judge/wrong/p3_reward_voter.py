# expect: WA
# the reward goes to the member who cast the deciding vote instead of the poll's author.
from abc import ABC, abstractmethod


def _clock_text(t):
    return "%02d:%02d" % ((t // 60) % 24, t % 60)


def _argument(text, word):
    """The stripped argument of `word ` at the very start of text, or None."""
    head = word + " "
    if text.startswith(head):
        return text[len(head):].strip()
    return None


class ChatBot(ABC):
    @abstractmethod
    def can_handle(self, sender: str, text: str) -> bool: ...

    @abstractmethod
    def handle(self, sender: str, text: str) -> list[str]: ...


class Clock:
    def __init__(self, now: int = 0):
        self.now = now


class ChatRoom:
    def __init__(self, clock: Clock):
        self._clock = clock
        self._bots = []
        self._log = []

    def register(self, bot) -> None:
        self._bots.append(bot)

    def send(self, sender, text, now):
        self._clock.now = now
        self._log.append(f"{sender}: {text}")
        for bot in self._bots:
            if bot.can_handle(sender, text):
                self._log += list(bot.handle(sender, text))
        return list(self._log)


class CheerBot(ChatBot):
    def __init__(self, cheer_counts):
        self._counts = cheer_counts

    @staticmethod
    def _target(text):
        arg = _argument(text, "/cheer")
        return arg if arg is not None and len(arg) >= 2 and arg[0] == "@" else None

    def can_handle(self, sender, text):
        return self._target(text) is not None

    def handle(self, sender, text):
        target = self._target(text)
        if target is None:
            return []
        user = target[1:]
        self._counts[user] = self._counts.get(user, 0) + 1
        return [f"CheerBot: {sender} cheers for {target}! {target}'s cheer count is now {self._counts[user]}."]


class FocusBot(ChatBot):
    def __init__(self, clock, focus_until):
        self._clock = clock
        self._until = focus_until

    def can_handle(self, sender, text):
        return True

    def handle(self, sender, text):
        now = self._clock.now
        lines = [f"FocusBot: {user} is heads-down until {_clock_text(end)}, try again later."
                 for user, end in self._until.items() if user in text and now < end]
        arg = _argument(text, "/focus")
        if arg is not None and arg.isdecimal() and int(arg) >= 1:
            minutes = int(arg)
            self._until[sender] = now + minutes
            lines.append(f"FocusBot: {sender} is heads-down for {minutes} min, back at {_clock_text(now + minutes)}.")
        return lines


class PollBot(ChatBot):
    def __init__(self, poll_state):
        self._poll = poll_state

    def can_handle(self, sender, text):
        return text.startswith("/poll ") or text.startswith("/vote ")

    def _start(self, sender, text):
        pieces = [p.strip() for p in text[len("/poll "):].split("|")]
        question, options = pieces[0], pieces[1:]
        if len(options) < 2:
            return []
        labels = [chr(ord("A") + i) for i in range(len(options))]
        self._poll.clear()
        self._poll.update(question=question, options=options, votes=dict.fromkeys(labels, 0), author=sender)
        listing = ", ".join(f"{lab}) {opt}" for lab, opt in zip(labels, options))
        return [f"PollBot: {sender} started a poll -- {question} [{listing}]"]

    def _vote(self, sender, text):
        letter = text[len("/vote "):].strip().upper()
        if not self._poll or letter not in self._poll["votes"]:
            return []
        votes = self._poll["votes"]
        votes[letter] += 1
        option = self._poll["options"][ord(letter) - ord("A")]
        tally = ", ".join(f"{k}: {n}" for k, n in votes.items())
        return [f"PollBot: {sender} voted {letter} ({option}). Tally -- {tally}"]

    def handle(self, sender, text):
        if text.startswith("/poll "):
            return self._start(sender, text)
        if text.startswith("/vote "):
            return self._vote(sender, text)
        return []


class Event:
    def __init__(self, type: str, data: dict):
        self.type = type
        self.data = data


class EventBus:
    def __init__(self):
        self._handlers = {}

    def subscribe(self, event_type, handler):
        self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event):
        for handler in list(self._handlers.get(event.type, [])):
            handler(event)


class DecisivePollBot(PollBot):
    def __init__(self, poll_state, bus):
        super().__init__(poll_state)
        self._bus = bus

    def _vote(self, sender, text):
        lines = super()._vote(sender, text)
        if lines and max(self._poll["votes"].values()) >= 3:
            author = sender
            self._poll.clear()
            self._bus.publish(Event("poll_closed", {"author": author}))
        return lines


class RewardedCheerBot(CheerBot):
    def __init__(self, cheer_counts, bus):
        super().__init__(cheer_counts)
        self._owed = []
        bus.subscribe("poll_closed", self._reward)

    def _reward(self, event):
        author = event.data["author"]
        self._counts[author] = self._counts.get(author, 0) + 1
        self._owed.append(f"CheerBot: {author}'s poll reached 3 votes! {author}'s cheer count is now "
                          f"{self._counts[author]}.")

    def can_handle(self, sender, text):
        return bool(self._owed) or super().can_handle(sender, text)

    def handle(self, sender, text):
        lines = super().handle(sender, text) + self._owed
        self._owed = []
        return lines
