"""Helpers for tests.py: a plain restatement of the run rules, a brute force over every single flip, a plain
parser, and random generators (programs, single-fix instances, program texts)."""
import re

FLIP = {"jump": "next", "next": "jump"}


def simulate(program):
    """("end", acc, None) on normal termination, ("loop", acc, line) when line is about to run a second time,
    ("out", acc, (line, pc)) when the instruction on `line` moves pc outside 0 .. n."""
    n = len(program)
    pc, acc, seen, last = 0, 0, set(), None
    while True:
        if pc == n:
            return "end", acc, None
        if not 0 <= pc < n:
            return "out", acc, (last, pc)
        if pc in seen:
            return "loop", acc, pc
        seen.add(pc)
        op, value = program[pc]
        if op == "plus":
            acc += value
        last = pc
        pc += value if op == "jump" else 1


def fixes(program):
    """Every (line, acc) whose single flip makes the program terminate normally."""
    out = []
    for i, (op, value) in enumerate(program):
        if op in FLIP:
            trial = list(program)
            trial[i] = (FLIP[op], value)
            kind, acc, _ = simulate(trial)
            if kind == "end":
                out.append((i, acc))
    return out


def is_instance(program):
    """True if the program meets Part 2's premise: it loops as given, and exactly one flip terminates."""
    return simulate(program)[0] == "loop" and len(fixes(program)) == 1


def show_program(program):
    return "[" + ", ".join(f"({op!r}, {v})" for op, v in program) + "]"


# ---------------------------------------------------------------- random programs

def random_program(r, n, span=None):
    """Any n-line program; by default jump values reach a little past both ends, so all three outcomes occur."""
    span = n + 2 if span is None else span
    out = []
    for _ in range(n):
        op = r.choice(["plus", "next", "jump", "jump"])
        out.append((op, r.randint(-span, span) if op == "jump" else r.randint(-50, 50)))
    return out


def terminating_program(r, n, span=None):
    """A random n-line program that terminates: a random path of distinct lines from 0 to n, every other line
    filled with junk the run never reaches. Returns (program, path)."""
    program, free = [None] * n, set(range(1, n))
    path, pc = [0], 0
    while pc != n:
        choices = [n]
        if pc + 1 == n or pc + 1 in free:
            choices += [pc + 1] * 3
        if free:
            choices.append(r.choice(sorted(free)))
        target = r.choice(choices)
        if target == pc + 1:
            program[pc] = (r.choice(["plus", "plus", "next"]), r.randint(-50, 50))
        else:
            program[pc] = ("jump", target - pc)
        free.discard(target)
        pc = target
        path.append(pc)
    span = span or max(3, min(30, n))
    for i in range(n):
        if program[i] is None:
            op = r.choice(["plus", "next", "jump"])
            program[i] = (op, r.randint(-span, span) if op == "jump" else r.randint(-50, 50))
    return program, path


def single_fix_instance(r, n, want_op=None, tries=60):
    """A program meeting Part 2's premise, made by flipping one line on a terminating path. want_op is the
    operator the corrupted line now shows ("next": it should have been a jump; "jump": it should have been a
    next). Returns (program, (line, acc)) or None."""
    for _ in range(tries):
        program, path = terminating_program(r, n)
        lines = [i for i in path[:-1] if program[i][0] in FLIP and (want_op is None or FLIP[program[i][0]] == want_op)]
        if not lines:
            continue
        line = r.choice(lines)
        bad = list(program)
        bad[line] = (FLIP[program[line][0]], program[line][1])
        if simulate(bad)[0] != "loop":
            continue
        found = fixes(bad)
        if len(found) == 1:
            return bad, found[0]
    return None


# ---------------------------------------------------------------- parsing

_SIGNED = re.compile(r"[+-][0-9]+")


def parse(text):
    """The parsed program, or ("error", line_number) for the first malformed line."""
    out = []
    for number, line in enumerate(text.split("\n"), start=1):
        code = line.split("#")[0].split()
        if not code:
            continue
        if len(code) != 2 or code[0] not in ("plus", "next", "jump") or not _SIGNED.fullmatch(code[1]):
            return "error", number
        out.append((code[0], int(code[1])))
    return out


BAD_LINES = ["plus 4", "boot +1", "jump +1 +2", "jump", "+3", "next 3.0", "next ++3", "plus +-3", "plus +3.0",
             "jump -2x", "jump + 3", "PLUS +1", "Jump +1", "plus +", "next -", "nextt +1", "plus+1", "plus, +1",
             "plus +1 extra", "jump +1x # comment"]
COMMENTS = ["# note", "#", "#plus +1", "   # indented", "\t# tab", "## twice", "# jump +1 # again"]
PADS = ["", "", " ", "   ", "\t", " \t "]


def random_line(r):
    op = r.choice(["plus", "next", "jump"])
    value = r.choice(["+", "-"]) + str(r.choice([0, 0, 1, 2, 7, 13, 250, r.randint(0, 99999)]))
    line = r.choice(PADS) + op + r.choice([" ", " ", "  ", "\t", " \t"]) + value + r.choice(PADS)
    if r.random() < 0.3:
        line += r.choice(PADS) + r.choice(COMMENTS)
    return line


def random_text(r, n, bad=False):
    """A program text with n instruction lines between blank and comment lines; with bad=True one line is
    replaced by a malformed one. Lines are joined with "\\n" (sometimes "\\r\\n"), with or without a final one."""
    lines = []
    for _ in range(n):
        while r.random() < 0.35:
            lines.append(r.choice(["", "   ", "\t"] + COMMENTS))
        lines.append(random_line(r))
    if bad and lines:
        k = r.randrange(len(lines))
        lines[k] = r.choice(PADS) + r.choice(BAD_LINES) + r.choice(PADS)
    sep = "\r\n" if r.random() < 0.2 else "\n"
    return sep.join(lines) + (sep if r.random() < 0.5 else "")
