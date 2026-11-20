from abc import ABC

cheer_counts = {}
focus_until = {}
poll_state = {}
messages = []


def format_time(t):
    return f"{(t // 60) % 24:02d}:{t % 60:02d}"


def handle_message(name, msg, now):
    messages.append(name + ": " + msg)

    # FocusBot logic: warn if this message mentions someone who is still heads-down
    for user in focus_until:
        end = focus_until[user]
        if user in msg and now < end:
            messages.append("FocusBot: " + user + " is heads-down until " + format_time(end) + ", try again later.")

    # CheerBot logic
    if msg[:7] == "/cheer ":
        target = msg.split(" ", 1)[1].strip()
        if target[:1] == "@" and len(target) > 1:
            username = target[1:]
            if username not in cheer_counts:
                cheer_counts[username] = 0
            cheer_counts[username] += 1
            messages.append(
                "CheerBot: "
                + name
                + " cheers for "
                + target
                + "! "
                + target
                + "'s cheer count is now "
                + str(cheer_counts[username])
                + "."
            )

    # PollBot logic: start a new poll
    if msg[:6] == "/poll ":
        parts = msg[6:].split("|")
        parts = [p.strip() for p in parts]
        question = parts[0]
        options = parts[1:]
        if len(options) >= 2:
            poll_state.clear()
            poll_state["question"] = question
            poll_state["options"] = options
            votes = {}
            for i in range(len(options)):
                votes[chr(65 + i)] = 0
            poll_state["votes"] = votes
            poll_state["author"] = name
            labeled = []
            for i in range(len(options)):
                labeled.append(chr(65 + i) + ") " + options[i])
            messages.append(f"PollBot: {name} started a poll -- {question} [{', '.join(labeled)}]")

    # PollBot logic: vote on the poll in progress
    if msg[:6] == "/vote ":
        letter = msg.split(" ", 1)[1].strip().upper()
        if poll_state and letter in poll_state["votes"]:
            poll_state["votes"][letter] += 1
            idx = ord(letter) - 65
            option_text = poll_state["options"][idx]
            tally_bits = []
            for k in poll_state["votes"]:
                tally_bits.append(k + ": " + str(poll_state["votes"][k]))
            tally = ", ".join(tally_bits)
            messages.append(f"PollBot: {name} voted {letter} ({option_text}). Tally -- {tally}")

    # FocusBot logic: start a focus session
    if msg[:7] == "/focus ":
        arg = msg.split(" ", 1)[1].strip()
        if arg.isdecimal() and int(arg) >= 1:
            minutes = int(arg)
            end = now + minutes
            focus_until[name] = end
            messages.append(
                "FocusBot: " + name + " is heads-down for " + str(minutes) + " min, back at " + format_time(end) + "."
            )


# ---------------------------------------------------------------- Part 2


class ChatBot(ABC):
    def can_handle(self, sender: str, text: str) -> bool:
        """Whether this bot has anything to say about this message."""
        raise NotImplementedError

    def handle(self, sender: str, text: str) -> list[str]:
        """The lines this bot appends to the log for this message."""
        raise NotImplementedError


class Clock:
    """now: minutes since midnight of the first day, never decreasing. ChatRoom.send sets it; bots only read it."""

    def __init__(self, now: int = 0):
        """Starts the clock at now."""
        raise NotImplementedError


class ChatRoom:
    def __init__(self, clock: Clock):
        """A room whose send advances this clock."""
        raise NotImplementedError

    def register(self, bot: "ChatBot") -> None:
        """Adds a bot that will see the messages sent to this room."""
        raise NotImplementedError

    def send(self, sender: str, text: str, now: int) -> list[str]:
        """Advances clock.now, appends "sender: text", dispatches to every registered bot, returns the log."""
        raise NotImplementedError


class CheerBot(ChatBot):
    def __init__(self, cheer_counts: dict[str, int]):
        """Handles /cheer @<name>; cheer_counts is the running count per name."""
        raise NotImplementedError


class FocusBot(ChatBot):
    def __init__(self, clock: Clock, focus_until: dict[str, int]):
        """Handles /focus <n> and the reminders; focus_until is the end minute of each user's session."""
        raise NotImplementedError


class PollBot(ChatBot):
    def __init__(self, poll_state: dict):
        """Handles /poll and /vote; poll_state is the poll in progress."""
        raise NotImplementedError


# ---------------------------------------------------------------- Part 3


class Event:
    def __init__(self, type: str, data: dict):
        """An event of this type carrying data."""
        raise NotImplementedError


class EventBus:
    def subscribe(self, event_type: str, handler) -> None:
        """handler(event) is called for every published event of this type, in subscription order."""
        raise NotImplementedError

    def publish(self, event: "Event") -> None:
        """Calls the handlers subscribed to this event's type."""
        raise NotImplementedError


class DecisivePollBot(ChatBot):
    def __init__(self, poll_state: dict, bus: EventBus):
        """PollBot with the new rule: a poll ends as soon as one option reaches 3 votes."""
        raise NotImplementedError


class RewardedCheerBot(ChatBot):
    def __init__(self, cheer_counts: dict[str, int], bus: EventBus):
        """CheerBot with the new rule: the author of a poll that reached 3 votes gets one more cheer."""
        raise NotImplementedError
