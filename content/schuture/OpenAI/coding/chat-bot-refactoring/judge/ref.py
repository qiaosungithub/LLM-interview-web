"""Helpers for tests.py: the statement's handle_message (the behaviour Part 2 must keep) run in a fresh namespace for
every room, Part 3's rule written directly on top of it, and random message generators."""

# The statement's legacy code, verbatim.
LEGACY = '''
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
'''


class Model:
    """One room as the statement defines it: handle_message on its own module-level dicts. decisive=True adds
    Part 3's rule (an option reaching 3 votes ends the poll and earns its author a cheer). cheer_counts and
    focus_until are the state the room starts from."""

    def __init__(self, decisive=False, cheer_counts=None, focus_until=None):
        self.ns = {}
        exec(LEGACY, self.ns)
        self.ns["cheer_counts"].update(cheer_counts or {})
        self.ns["focus_until"].update(focus_until or {})
        self.decisive = decisive

    def send(self, sender, text, now):
        """The lines this message appends to the log."""
        ns = self.ns
        before = len(ns["messages"])
        ns["handle_message"](sender, text, now)
        poll, counts = ns["poll_state"], ns["cheer_counts"]
        if self.decisive and poll and max(poll["votes"].values()) >= 3:
            author = poll["author"]
            poll.clear()                    # a later /vote now finds no poll in progress
            counts[author] = counts.get(author, 0) + 1
            ns["messages"].append(f"CheerBot: {author}'s poll reached 3 votes! "
                                  f"{author}'s cheer count is now {counts[author]}.")
        return ns["messages"][before:]


# ---------------------------------------------------------------- random messages

NAMES = ["Theo", "TheoBot", "Priya", "Pri", "Noah", "Sana", "Leo", "Maya", "Kofi", "us", "B", "cheer"]
WORDS = ["hello", "quick", "update", "see", "you", "at", "the", "meeting", "lunch", "@", "|"]
JUNK = ["", "/", "/cheer", "/vote", "/poll", "/focus", "/Cheer @Theo", "/cheer\t@Theo", "hi @Theo", " /cheer @Theo",
        "/cheers @Theo", "/cheer@Theo", "/focus\t5", "/vote\tA", "/poll\tQ? | a | b", "/VOTE A", " /vote A"]
PADS = ["", "", "", " ", "  ", "\t"]


def random_message(r, kinds, names=NAMES):
    kind = r.choice(kinds)
    sender = r.choice(names)
    pad, pad2 = r.choice(PADS), r.choice(PADS)
    if kind == "cheer":
        arg = r.choice(["@" + r.choice(names), "@" + r.choice(names), "@", "@@x", "@Kofi Mensah", r.choice(names), ""])
    elif kind == "focus":
        arg = r.choice(["1", "0", "-3", "abc", "007", "5 extra", "", "+5", "1.5", "2_0",
                        str(r.randint(1, 30)), str(r.randint(1, 30)), str(r.randint(1, 3000))])
    elif kind == "poll":
        opts = [r.choice(WORDS[:9] + ["", "Tacos", "Pasta"]) for _ in range(r.randint(0, 5))]
        arg = r.choice(WORDS[:9] + ["", "Lunch?"]) + "".join(f"{r.choice(PADS)}|{r.choice(PADS)}{o}" for o in opts)
    elif kind == "vote":
        arg = r.choice(["A", "B", "C", "D", "Z", "a", "b", "c", "", "AB", "?"])
    elif kind == "plain":
        return sender, " ".join(r.choice(WORDS + names) for _ in range(r.randint(1, 5)))
    else:
        return sender, r.choice(JUNK)
    return sender, f"/{kind} {pad}{arg}{pad2}"


def random_script(r, n, kinds, names=NAMES):
    """n (sender, text, now) triples with now never decreasing; time steps cross midnight now and then."""
    now, out = r.randint(0, 1500), []
    for _ in range(n):
        now += r.choice([0, 0, 1, 1, 2, 5, 15, 120, 700])
        sender, text = random_message(r, kinds, names)
        out.append((sender, text, now))
    return out


MIXED = ["cheer", "focus", "poll", "vote", "plain", "junk"]
MENTIONS = ["focus"] * 4 + ["plain"] * 4 + ["cheer", "cheer", "vote", "poll", "junk"]
VOTES = MIXED + ["vote"] * 6
