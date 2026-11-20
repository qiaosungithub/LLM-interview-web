# The judge runs every answer with the checker from the statement, copied verbatim below (parse, run, the
# five spec functions, PUZZLES). Helpers for tests.py follow it, under "judge helpers".

"""Validity checker for the guarded accumulator machine. This file is all the tooling you get."""

LIMIT = 99            # acc and bak saturate to [-LIMIT, LIMIT]
MAX_LEN = 12          # a program has 1..MAX_LEN instructions
MAX_STEPS = 20_000    # per test; a program that runs longer is invalid
OPS = ("IN", "OUT", "INC", "NEG", "ADD", "SAV", "SWP", "TZ", "TP", "JMP")


def parse(source):
    """Instructions are separated by newlines or ';'. '#' starts a comment.
    A '+' or '-' in front of the mnemonic is the guard. Returns [(guard, op, arg)]."""
    prog = []
    for line in source.replace(";", "\n").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        guard = ""
        if line[0] in "+-":
            guard, line = line[0], line[1:].strip()
        parts = line.split()
        if not parts or parts[0] not in OPS or len(parts) != (2 if parts[0] == "JMP" else 1):
            raise ValueError(f"bad instruction: {line!r}")
        prog.append((guard, parts[0], int(parts[1]) if parts[0] == "JMP" else None))
    if not 1 <= len(prog) <= MAX_LEN:
        raise ValueError(f"a program has 1..{MAX_LEN} instructions")
    if any(op == "JMP" and not 0 <= arg < len(prog) for _, op, arg in prog):
        raise ValueError("jump target out of range")
    return prog


def run(prog, inputs):
    """Outputs of one run, or None if the program is still running after MAX_STEPS steps."""
    acc = bak = 0
    flag = False
    pc = pos = 0
    out = []
    for _ in range(MAX_STEPS):
        guard, op, arg = prog[pc]
        pc = (pc + 1) % len(prog)                 # the program is a ring
        if (guard == "+" and not flag) or (guard == "-" and flag):
            continue                              # skipped: one step, no effect
        if op == "IN":
            if pos == len(inputs):
                return out                        # the only way to halt
            acc = inputs[pos]
            pos += 1
        elif op == "OUT":
            out.append(acc)
        elif op == "INC":
            acc = min(acc + 1, LIMIT)
        elif op == "NEG":
            acc = -acc
        elif op == "ADD":
            acc = max(-LIMIT, min(LIMIT, acc + bak))
        elif op == "SAV":
            bak = acc
        elif op == "SWP":
            acc, bak = bak, acc
        elif op == "TZ":
            flag = acc == 0
        elif op == "TP":
            flag = acc > 0
        elif op == "JMP":
            pc = arg
    return None


# ---- the five puzzles: a reference function (the spec) and the test inputs ----

def after_zero(xs):
    return xs[xs.index(0) + 1:] if 0 in xs else []


def group_sums(xs):
    out, total = [], 0
    for x in xs:
        if x == 0:
            out.append(total)
            total = 0
        else:
            total += x
    return out


def mute(xs):
    out, muted = [], False
    for x in xs:
        if x == 0:
            muted = not muted
        elif not muted:
            out.append(x)
    return out


def count_down(xs):
    return [k for x in xs for k in range(x, 0, -1)]


def count_up(xs):
    return [k for x in xs for k in range(1, x + 1)]


PUZZLES = {
    "p1": (after_zero, [
        [5, 0, 3, 0, -2, 7], [0, 4], [1, 2, 3], [], [0], [0, 0],
        [-6, 9, 0, 0, 0, 5, -99, 99, 0, 1], [7, -7, 0, 12, 0, 0, -3]]),
    "p2": (group_sums, [
        [3, 4, 0, 5, 0], [0], [], [2, 2, 2], [0, 0, 7, 0],
        [-5, 9, -1, 0, 20, 20, 20, 0, 6], [10, -10, 0, 1, 0, 0, -8, -9, 0],
        [4, 0, -4, 0, 4, -4, 0, 15, 3, 0, 0, 1]]),
    "p3": (mute, [
        [3, 0, 5, 0, 7], [], [0], [0, 0], [1, 2, 3],
        [0, 4, 4, 0, 0, 9, 0, 6, 1], [8, 0, 0, 0, 2, 0, 99, 5, 0, 0, 3],
        [5, 5, 0, 1, 0, 0, 0, 2]]),
    "p4": (count_down, [
        [3], [], [0], [1], [2, 0, 4], [0, 0, 1, 5, 0], [20], [7, 1, 0, 2, 2]]),
    "p5": (count_up, [
        [3], [], [0], [1], [2, 0, 4], [0, 0, 1, 5, 0], [20], [7, 1, 0, 2, 2]]),
}


def check(name, source):
    """(valid, instruction count). Valid: on every test the program halts with exactly the right output."""
    prog = parse(source)
    spec, tests = PUZZLES[name]
    return all(run(prog, xs) == spec(xs) for xs in tests), len(prog)


def report(submission):
    """submission: {puzzle name: program text}. Prints one line per puzzle, returns the total score."""
    total = 0
    for name in PUZZLES:
        valid, count = check(name, submission[name]) if name in submission else (False, 0)
        total += count if valid else 20
        print(f"{name}: {'valid' if valid else 'INVALID':7s} {count:2d} instructions")
    print(f"total score: {total}")
    return total


# ---- judge helpers (not part of the statement) ----

def count_steps(prog, inputs):
    """Steps of one run, counted like the statement's example ("共 10 步": the halting IN is a step too),
    or None if the program is still running after MAX_STEPS steps. Only used to report step counts;
    whether an answer is right is always decided by run() above."""
    acc = bak = 0
    flag = False
    pc = pos = 0
    for step in range(1, MAX_STEPS + 1):
        guard, op, arg = prog[pc]
        pc = (pc + 1) % len(prog)
        if (guard == "+" and not flag) or (guard == "-" and flag):
            continue
        if op == "IN":
            if pos == len(inputs):
                return step
            acc = inputs[pos]
            pos += 1
        elif op == "INC":
            acc = min(acc + 1, LIMIT)
        elif op == "NEG":
            acc = -acc
        elif op == "ADD":
            acc = max(-LIMIT, min(LIMIT, acc + bak))
        elif op == "SAV":
            bak = acc
        elif op == "SWP":
            acc, bak = bak, acc
        elif op == "TZ":
            flag = acc == 0
        elif op == "TP":
            flag = acc > 0
        elif op == "JMP":
            pc = arg
    return None


def random_input(rng, name, length):
    """A random input inside the range the statement promises for this puzzle (Part 4 and 5: at most 8 numbers)."""
    if name == "p1":                                    # each in -99..99
        return [0 if rng.random() < 0.3 else rng.randint(-99, 99) for _ in range(length)]
    if name == "p3":                                    # each in 0..99
        return [0 if rng.random() < 0.35 else rng.choice([rng.randint(1, 9), rng.randint(1, 99)]) for _ in range(length)]
    if name in ("p4", "p5"):                            # at most 8 numbers, each in 0..99
        return [0 if rng.random() < 0.15 else rng.randint(1, 99) for _ in range(min(length, 8))]
    xs, total = [], 0                                   # p2: every running sum (and every number) in -99..99
    for _ in range(length):
        x = 0 if rng.random() < 0.3 else rng.randint(max(-99, -99 - total), min(99, 99 - total))
        total = 0 if x == 0 else total + x
        xs.append(x)
    return xs
